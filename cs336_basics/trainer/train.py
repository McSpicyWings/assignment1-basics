#!/usr/bin/env python3
"""
Training script for Transformer Language Model.

This script provides a complete training loop with:
- Configurable model and optimizer hyperparameters via command-line arguments
- Memory-efficient loading of large datasets using np.memmap
- Checkpoint serialization and resumption
- Periodic logging of training and validation performance
- Optional Weights & Biases integration

Usage:
    python -m cs336_basics.trainer.train \
        --train_data data/TinyStoriesV2-GPT4-train_encode.npy \
        --val_data data/TinyStoriesV2-GPT4-valid_encode.npy \
        --vocab_size 10000 \
        --d_model 512 \
        --num_layers 6 \
        --num_heads 8 \
        --context_length 256 \
        --batch_size 32 \
        --max_iters 10000 \
        --lr 1e-3 \
        --checkpoint_dir checkpoints/
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


from cs336_basics.layer.transformer import TransformerLM
from cs336_basics.trainer.optimizer import AdamWOptim, lr_cosine_schedule, gradient_clip
from cs336_basics.trainer.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.trainer.util import cross_entorpy, get_batch


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def load_dataset(path: str, dtype=np.uint16) -> np.ndarray:
    """
    Load dataset using memory-mapped mode for efficient loading of large files.
    
    Args:
        path: Path to the .npy file containing tokenized data
        dtype: Data type of the array (default: np.uint16 for token IDs)
        
    Returns:
        Memory-mapped numpy array
    """
    if path.endswith('.npy'):
        # Use mmap_mode='r' for read-only memory-mapped access
        data = np.load(path, mmap_mode='r')
    else:
        # For raw binary files, use np.memmap directly
        data = np.memmap(path, dtype=dtype, mode='r')
    
    return data


def estimate_loss(
    model: torch.nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    context_length: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict:
    """
    Estimate loss on training and validation sets.
    
    Args:
        model: The language model
        train_data: Training dataset
        val_data: Validation dataset
        context_length: Context length for batches
        batch_size: Batch size for evaluation
        eval_iters: Number of iterations to average over
        device: Device to run on
        
    Returns:
        Dictionary with 'train' and 'val' loss estimates
    """
    out = {}
    model.eval()
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            with torch.no_grad():
                logits = model(x)
                # Reshape for cross entropy: (batch * seq, vocab) and (batch * seq,)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)
                loss = cross_entorpy(logits_flat, targets_flat)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    
    model.train()
    return out


def train(args):
    """Main training function."""
    # Create checkpoint directory first (before setting up logging)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logger = setup_logging(
        log_file=os.path.join(args.checkpoint_dir, "train.log") if args.checkpoint_dir else None
    )
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load datasets with memory mapping
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_dataset(args.train_data)
    logger.info(f"Training data shape: {train_data.shape}, dtype: {train_data.dtype}")
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_data = load_dataset(args.val_data)
    logger.info(f"Validation data shape: {val_data.shape}, dtype: {val_data.dtype}")
    
    # Verify data looks correct
    max_token = max(train_data.max(), val_data.max())
    logger.info(f"Max token ID in data: {max_token}")
    if max_token >= args.vocab_size:
        logger.warning(f"Max token ID ({max_token}) >= vocab_size ({args.vocab_size})!")
    
    # Calculate d_ff if not provided
    if args.d_ff is None:
        d_ff = ((args.d_model * 8 // 3) + 63) // 64 * 64
    else:
        d_ff = args.d_ff
    logger.info(f"Using d_ff: {d_ff}")
    
    # Create model
    logger.info("Creating model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    model = model.to(device)
    
    n_params = model.get_num_params()
    logger.info(f"Model parameters (non-embedding): {n_params:,}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = AdamWOptim(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )
    
    # Resume from checkpoint if provided
    start_iter = 0
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"Resumed from iteration {start_iter}")
    
    # Initialize Weights & Biases if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            wandb.watch(model, log="gradients", log_freq=args.log_interval)
        except ImportError:
            logger.warning("wandb not installed, disabling logging to W&B")
            args.wandb = False
    
    # Save model config
    if args.checkpoint_dir:
        config_path = os.path.join(args.checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_config": model.get_config(),
                "training_args": vars(args),
            }, f, indent=2)
        logger.info(f"Saved config to {config_path}")
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for iter_num in range(start_iter, args.max_iters):
        # Get learning rate from schedule
        lr = lr_cosine_schedule(
            t=iter_num,
            lr_max=args.lr,
            lr_min=args.lr_min,
            warmup_iter=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        
        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)
        loss = cross_entorpy(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clip(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (iter_num - start_iter + 1) * args.batch_size * args.context_length / elapsed
            logger.info(
                f"iter {iter_num:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:.0f} tokens/s"
            )
            
            if args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "iter": iter_num,
                })
        
        # Evaluation
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                args.context_length, args.batch_size,
                args.eval_iters, device,
            )
            logger.info(
                f"iter {iter_num:6d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )
            
            if args.wandb:
                wandb.log({
                    "eval/train_loss": losses['train'],
                    "eval/val_loss": losses['val'],
                    "eval/train_ppl": math.exp(losses['train']),
                    "eval/val_ppl": math.exp(losses['val']),
                    "iter": iter_num,
                })
            
            # Save best model
            if losses['val'] < best_val_loss and args.checkpoint_dir:
                best_val_loss = losses['val']
                best_path = os.path.join(args.checkpoint_dir, "best.pt")
                save_checkpoint(model, optimizer, iter_num, best_path)
                logger.info(f"Saved best model to {best_path}")
        
        # Save checkpoint
        if args.checkpoint_dir and iter_num > 0 and iter_num % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{iter_num:06d}.pt")
            save_checkpoint(model, optimizer, iter_num, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")
    
    # Final evaluation
    losses = estimate_loss(
        model, train_data, val_data,
        args.context_length, args.batch_size,
        args.eval_iters, device,
    )
    logger.info(f"Final | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    logger.info(f"Final | train ppl {math.exp(losses['train']):.2f} | val ppl {math.exp(losses['val']):.2f}")
    
    # Save final checkpoint
    if args.checkpoint_dir:
        final_path = os.path.join(args.checkpoint_dir, "final.pt")
        save_checkpoint(model, optimizer, args.max_iters, final_path)
        logger.info(f"Saved final model to {final_path}")
    
    if args.wandb:
        wandb.finish()
    
    logger.info("Training complete!")
    return model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (.npy file)")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation data (.npy file)")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model embedding dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None,
                        help="FFN hidden dimension (default: 8/3 * d_model rounded to 64)")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Optimizer arguments
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Maximum learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-4,
                        help="Minimum learning rate (after decay)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="AdamW beta2")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay coefficient")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="AdamW epsilon")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold (0 to disable)")
    parser.add_argument("--warmup_iters", type=int, default=100,
                        help="Number of warmup iterations")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_iters", type=int, default=10000,
                        help="Maximum number of training iterations")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=10,
                        help="How often to log training loss")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="How often to evaluate on validation set")
    parser.add_argument("--eval_iters", type=int, default=100,
                        help="Number of iterations for loss estimation")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="How often to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Weights & Biases arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-lm",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
