# train/ablations/train_ablation.py
"""
消融实验训练脚本
支持不同的模型变体: baseline, postnorm, nope, silu
"""
import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# 项目路径设置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from train.ablations.models import create_model
from cs336_basics.trainer.optimizer import AdamWOptim


def get_logger(log_file: Optional[Path] = None):
    """配置日志"""
    logger = logging.getLogger("ablation_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def get_batch(data: np.memmap, batch_size: int, context_length: int, device: torch.device):
    """从 memmap 数据中随机采样一个 batch"""
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = torch.stack([torch.from_numpy(data[s:s + context_length].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(data[s + 1:s + context_length + 1].astype(np.int64)) for s in starts])
    return x.to(device), y.to(device)


def cosine_lr_schedule(
    it: int,
    max_iters: int,
    warmup_iters: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """余弦退火学习率调度"""
    if it < warmup_iters:
        return lr_max * (it + 1) / warmup_iters
    if it >= max_iters:
        return lr_min
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_min + coeff * (lr_max - lr_min)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, eval_iters, device):
    """估计训练和验证损失"""
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Training")
    
    # 模型类型
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "postnorm", "nope", "silu"],
                        help="Model variant to train")
    
    # 数据路径
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    
    # 模型超参
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # 日志和保存
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志
    logger = get_logger(ckpt_dir / "train.log")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")
    
    # 保存配置
    config = vars(args).copy()
    config["device"] = str(device)
    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    # 加载数据
    logger.info(f"Loading training data from {args.train_data}")
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    logger.info(f"Training data: {len(train_data):,} tokens")
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")
    logger.info(f"Validation data: {len(val_data):,} tokens")
    
    # 创建模型
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    model = model.to(device)
    
    n_params = model.get_num_params(non_embedding=False)
    n_params_nonemb = model.get_num_params(non_embedding=True)
    logger.info(f"Model parameters: {n_params:,} (non-embedding: {n_params_nonemb:,})")
    
    # 优化器
    optimizer = AdamWOptim(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # 训练循环
    logger.info("Starting training...")
    model.train()
    
    tokens_per_iter = args.batch_size * args.context_length
    start_time = time.time()
    
    for iter_num in range(args.max_iters):
        # 学习率调度
        lr = cosine_lr_schedule(
            iter_num, args.max_iters, args.warmup_iters, args.lr, args.lr_min
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # 获取 batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # 前向传播
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # 日志
        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (iter_num + 1) * tokens_per_iter / elapsed if elapsed > 0 else 0
            logger.info(f"iter {iter_num:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tokens/s")
        
        # 评估
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                args.batch_size, args.context_length,
                args.eval_iters, device
            )
            logger.info(f"iter {iter_num:6d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        
        # 保存 checkpoint
        if iter_num > 0 and iter_num % args.checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_{iter_num}.pt"
            torch.save({
                "iter": iter_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")
    
    # 最终评估
    losses = estimate_loss(
        model, train_data, val_data,
        args.batch_size, args.context_length,
        args.eval_iters, device
    )
    logger.info(f"Final | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    
    # 保存最终模型
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "iter": args.max_iters,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
