#!/usr/bin/env python3
# train/ablations/run_ablations.py
"""
运行所有消融实验
用法:
    python -m train.ablations.run_ablations
    python -m train.ablations.run_ablations --models baseline postnorm
    python -m train.ablations.run_ablations --seeds 42 43 44
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_EXE = sys.executable

# 数据路径
TRAIN_DATA = PROJECT_ROOT / "data/TinyStoriesV2-GPT4-train_encode.npy"
VAL_DATA = PROJECT_ROOT / "data/TinyStoriesV2-GPT4-valid_encode.npy"

# 输出目录
ABLATIONS_DIR = PROJECT_ROOT / "data/ablations"

# 统一的训练超参数（确保公平对比）
BASE_ARGS = {
    "vocab_size": 10000,
    "d_model": 512,
    "d_ff": 1344,  # SwiGLU 的 d_ff
    "num_layers": 4,
    "num_heads": 16,
    "context_length": 256,
    "rope_theta": 10000.0,
    "batch_size": 32,
    "lr": 1e-3,
    "lr_min": 1e-4,
    "max_iters": 10000,
    "warmup_iters": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_interval": 500,
    "eval_iters": 100,
    "log_interval": 10,
    "checkpoint_interval": 2000,
    "device": "auto",
}

# 消融模型类型
MODEL_TYPES = ["baseline", "postnorm", "nope", "silu"]


def build_cmd(model_type: str, run_dir: Path, seed: int) -> list[str]:
    """构建训练命令"""
    cmd = [PYTHON_EXE, "-m", "train.ablations.train_ablation"]
    cmd += ["--model_type", model_type]
    cmd += ["--train_data", str(TRAIN_DATA)]
    cmd += ["--val_data", str(VAL_DATA)]
    cmd += ["--checkpoint_dir", str(run_dir)]
    cmd += ["--seed", str(seed)]
    
    for k, v in BASE_ARGS.items():
        if k not in ["seed"]:  # seed 已单独添加
            cmd += [f"--{k}", str(v)]
    
    return cmd


def run_experiment(model_type: str, seed: int, stamp: str, dry_run: bool = False):
    """运行单个实验"""
    run_name = f"{stamp}_{model_type}_seed{seed}"
    run_dir = ABLATIONS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = build_cmd(model_type, run_dir, seed)
    
    # 保存实验元信息
    meta = {
        "run_name": run_name,
        "model_type": model_type,
        "seed": seed,
        "cmd": cmd,
        "base_args": BASE_ARGS,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"{'='*60}")
    print(" ".join(cmd))
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return True
    
    # 运行训练（同时输出到终端和文件）
    with (run_dir / "stdout.txt").open("w", encoding="utf-8") as f_out:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,  # 确保在项目根目录运行
        )
        for line in p.stdout:
            print(line, end="")
            f_out.write(line)
            f_out.flush()
        p.wait()
    
    if p.returncode != 0:
        print(f"[FAILED] {run_name}")
        return False
    else:
        print(f"[OK] {run_name}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=MODEL_TYPES,
        choices=MODEL_TYPES,
        help=f"Model types to run (default: all {MODEL_TYPES})",
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds (default: [42])",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print commands, don't run",
    )
    args = parser.parse_args()
    
    ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir: {ABLATIONS_DIR}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {len(args.models) * len(args.seeds)}")
    
    results = []
    for model_type in args.models:
        for seed in args.seeds:
            success = run_experiment(model_type, seed, stamp, args.dry_run)
            results.append({
                "model_type": model_type,
                "seed": seed,
                "success": success,
            })
    
    # 保存汇总
    summary = {
        "stamp": stamp,
        "base_args": BASE_ARGS,
        "results": results,
    }
    (ABLATIONS_DIR / f"{stamp}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: {ABLATIONS_DIR}")
    print(f"{'='*60}")
    
    # 打印结果汇总
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['model_type']} (seed={r['seed']})")


if __name__ == "__main__":
    main()
