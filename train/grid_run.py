# grid_run.py
import itertools
import json
import subprocess
from pathlib import Path
from datetime import datetime
import os
import sys

# ===== 项目根目录（grid_run.py 应该放在项目根目录下运行）=====
PROJECT_ROOT = Path(__file__).parent.parent  # 假设 grid_run.py 在 train/ 下


# 使用当前 Python 解释器（确保使用虚拟环境中的 Python）
PYTHON_EXE = sys.executable

# ===== 训练脚本和数据路径 =====
TRAIN_DATA = PROJECT_ROOT / "data/TinyStoriesV2-GPT4-train_encode.npy"
VAL_DATA   = PROJECT_ROOT / "data/TinyStoriesV2-GPT4-valid_encode.npy"

# ===== 固定推荐超参=====
BASE_ARGS = {
    "vocab_size": 10000,
    "d_model": 512,
    "d_ff": 1344,              
    "num_layers": 4,
    "num_heads": 16,
    "context_length": 256,     
    "rope_theta": 10000.0,
    "max_iters": 10000,        
    "warmup_iters": 100,
    "lr_min": 1e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_interval": 500,
    "eval_iters": 100,
    "log_interval": 10,
    "checkpoint_interval": 2000,
    "seed": 42,
    "device": "auto",
}

# =====  batch / lr =====
BATCH_SIZES = [8, 16, 32, 64]
LRS = [3e-4, 1e-3, 3e-3]

RUNS_DIR = PROJECT_ROOT / "runs_grid"

def build_cmd(run_dir: Path, batch_size: int, lr: float) -> list[str]:
    # 使用当前 Python 解释器 + -m 模块方式运行训练脚本
    cmd = [PYTHON_EXE, "-m", "cs336_basics.trainer.train"]
    cmd += ["--train_data", str(TRAIN_DATA), "--val_data", str(VAL_DATA)]

    # 固定参数
    for k, v in BASE_ARGS.items():
        cmd += [f"--{k}", str(v)]

    # 变量参数
    cmd += ["--batch_size", str(batch_size)]
    cmd += ["--lr", str(lr)]

    # 每个 run 一个独立 checkpoint_dir（你的 train.py 会写 train.log / config.json / ckpt）
    cmd += ["--checkpoint_dir", str(run_dir)]

    return cmd

def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 切换到项目根目录运行（确保 python -m 能找到模块）
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {os.getcwd()}")

    all_runs = []
    for batch_size, lr in itertools.product(BATCH_SIZES, LRS):
        run_name = f"{stamp}_bs{batch_size}_lr{lr:g}"
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_cmd(run_dir, batch_size, lr)

        # 保存本次运行的命令与超参，方便复现实验
        meta = {
            "run_name": run_name,
            "batch_size": batch_size,
            "lr": lr,
            "cmd": [str(c) for c in cmd],  # 转换为字符串
            "base_args": BASE_ARGS,
            "train_data": str(TRAIN_DATA),
            "val_data": str(VAL_DATA),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        all_runs.append(meta)

        print("\n=== Running:", run_name, "===")
        print(" ".join(cmd))

        # 直接运行训练；stdout/stderr 也写入文件，便于排查
        with (run_dir / "stdout.txt").open("w", encoding="utf-8") as f_out:
            p = subprocess.run(cmd, stdout=f_out, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(f"[FAILED] {run_name} (see {run_dir/'stdout.txt'})")
        else:
            print(f"[OK] {run_name}")

    (RUNS_DIR / f"{stamp}_index.json").write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    print("\nAll runs done. Logs in:", RUNS_DIR)

if __name__ == "__main__":
    main()
