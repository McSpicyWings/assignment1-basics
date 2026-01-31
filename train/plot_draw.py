# plot_runs.py
import re
from pathlib import Path
import matplotlib.pyplot as plt

# Use absolute path relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs_grid"
OUT_DIR = PROJECT_ROOT / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你的 train.log 里常见两类行（来自 train.py 的 logger.info）:contentReference[oaicite:6]{index=6}
RE_TRAIN = re.compile(r"iter\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|\s+lr\s+([0-9.eE+-]+)")
RE_EVAL  = re.compile(r"iter\s+(\d+)\s+\|\s+train loss\s+([0-9.]+)\s+\|\s+val loss\s+([0-9.]+)")

def parse_log(log_path: Path):
    train_iters, train_loss = [], []
    eval_iters, eval_train, eval_val = [], [], []

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_TRAIN.search(line)
        if m:
            it = int(m.group(1))
            loss = float(m.group(2))
            train_iters.append(it)
            train_loss.append(loss)
            continue

        m = RE_EVAL.search(line)
        if m:
            it = int(m.group(1))
            tr = float(m.group(2))
            va = float(m.group(3))
            eval_iters.append(it)
            eval_train.append(tr)
            eval_val.append(va)

    return (train_iters, train_loss), (eval_iters, eval_train, eval_val)

def run_label(run_dir: Path) -> str:
    # 目录名是 grid_run.py 里生成的：..._bs{batch}_lr{lr}
    return run_dir.name

def plot_train():
    plt.figure()
    for run_dir in sorted(RUNS_DIR.iterdir()):
        log_path = run_dir / "train.log"
        if not log_path.exists():
            continue
        (ti, tl), _ = parse_log(log_path)
        if len(ti) == 0:
            continue
        plt.plot(ti, tl, label=run_label(run_dir))
    plt.xlabel("iteration")
    plt.ylabel("train loss")
    plt.title("Training loss (by run)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    out = OUT_DIR / "train_loss.png"
    plt.savefig(out, dpi=200)
    print("saved:", out)

def plot_val():
    plt.figure()
    for run_dir in sorted(RUNS_DIR.iterdir()):
        log_path = run_dir / "train.log"
        if not log_path.exists():
            continue
        _, (ei, etr, ev) = parse_log(log_path)
        if len(ei) == 0:
            continue
        plt.plot(ei, ev, label=run_label(run_dir))
    plt.xlabel("iteration")
    plt.ylabel("val loss")
    plt.title("Validation loss (by run)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    out = OUT_DIR / "val_loss.png"
    plt.savefig(out, dpi=200)
    print("saved:", out)

def main():
    plot_train()
    plot_val()

if __name__ == "__main__":
    main()
