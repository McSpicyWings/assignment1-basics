#!/usr/bin/env python3
# train/ablations/plot_ablations.py
"""
绘制消融实验对比图
用法:
    python -m train.ablations.plot_ablations
    python -m train.ablations.plot_ablations --runs_dir /path/to/ablations
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ABLATIONS_DIR = PROJECT_ROOT / "data/ablations"
PLOTS_DIR = PROJECT_ROOT / "data/ablation_plots"

# 日志解析正则
RE_TRAIN = re.compile(r"iter\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|\s+lr\s+([0-9.eE+-]+)")
RE_EVAL = re.compile(r"iter\s+(\d+)\s+\|\s+train loss\s+([0-9.]+)\s+\|\s+val loss\s+([0-9.]+)")

# 模型类型对应的显示名称和颜色
MODEL_STYLES = {
    "baseline": {"label": "Baseline (Pre-norm + RoPE + SwiGLU)", "color": "#1f77b4", "linestyle": "-"},
    "postnorm": {"label": "Post-norm", "color": "#ff7f0e", "linestyle": "--"},
    "nope": {"label": "NoPE (No Position Encoding)", "color": "#2ca02c", "linestyle": "-."},
    "silu": {"label": "SiLU FFN (No Gating)", "color": "#d62728", "linestyle": ":"},
}


def parse_log(log_path: Path):
    """解析训练日志"""
    train_iters, train_loss = [], []
    eval_iters, eval_train, eval_val = [], [], []
    
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_TRAIN.search(line)
        if m:
            train_iters.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            continue
        
        m = RE_EVAL.search(line)
        if m:
            eval_iters.append(int(m.group(1)))
            eval_train.append(float(m.group(2)))
            eval_val.append(float(m.group(3)))
    
    return {
        "train": (train_iters, train_loss),
        "eval": (eval_iters, eval_train, eval_val),
    }


def get_model_type(run_dir: Path) -> str:
    """从运行目录名提取模型类型"""
    name = run_dir.name
    for model_type in MODEL_STYLES.keys():
        if f"_{model_type}_" in name:
            return model_type
    return None


def collect_runs(runs_dir: Path) -> dict:
    """收集所有运行的数据，按模型类型分组"""
    data = defaultdict(list)
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        log_path = run_dir / "train.log"
        if not log_path.exists():
            continue
        
        model_type = get_model_type(run_dir)
        if model_type is None:
            continue
        
        parsed = parse_log(log_path)
        if len(parsed["train"][0]) == 0:
            continue
        
        data[model_type].append({
            "run_dir": run_dir,
            "data": parsed,
        })
    
    return data


def plot_train_loss_comparison(data: dict, output_path: Path, title: str = "Training Loss Comparison"):
    """绘制训练损失对比图"""
    plt.figure(figsize=(10, 6))
    
    for model_type, runs in data.items():
        if model_type not in MODEL_STYLES:
            continue
        
        style = MODEL_STYLES[model_type]
        
        # 如果有多个 seed，计算均值和标准差
        if len(runs) > 1:
            # 找到所有运行共有的迭代点
            all_iters = [set(r["data"]["train"][0]) for r in runs]
            common_iters = sorted(set.intersection(*all_iters))
            
            if len(common_iters) == 0:
                continue
            
            # 收集每个迭代点的损失值
            losses_by_iter = defaultdict(list)
            for run in runs:
                iters, losses = run["data"]["train"]
                iter_to_loss = dict(zip(iters, losses))
                for it in common_iters:
                    if it in iter_to_loss:
                        losses_by_iter[it].append(iter_to_loss[it])
            
            iters = list(losses_by_iter.keys())
            mean_losses = [np.mean(losses_by_iter[it]) for it in iters]
            std_losses = [np.std(losses_by_iter[it]) for it in iters]
            
            plt.plot(iters, mean_losses, label=style["label"], color=style["color"], 
                     linestyle=style["linestyle"], linewidth=2)
            plt.fill_between(iters, 
                           np.array(mean_losses) - np.array(std_losses),
                           np.array(mean_losses) + np.array(std_losses),
                           alpha=0.2, color=style["color"])
        else:
            # 单个 seed
            iters, losses = runs[0]["data"]["train"]
            plt.plot(iters, losses, label=style["label"], color=style["color"],
                     linestyle=style["linestyle"], linewidth=2)
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def plot_val_loss_comparison(data: dict, output_path: Path, title: str = "Validation Loss Comparison"):
    """绘制验证损失对比图"""
    plt.figure(figsize=(10, 6))
    
    for model_type, runs in data.items():
        if model_type not in MODEL_STYLES:
            continue
        
        style = MODEL_STYLES[model_type]
        
        if len(runs) > 1:
            # 多个 seed：计算均值和标准差
            all_iters = [set(r["data"]["eval"][0]) for r in runs]
            common_iters = sorted(set.intersection(*all_iters))
            
            if len(common_iters) == 0:
                continue
            
            losses_by_iter = defaultdict(list)
            for run in runs:
                iters, _, val_losses = run["data"]["eval"]
                iter_to_loss = dict(zip(iters, val_losses))
                for it in common_iters:
                    if it in iter_to_loss:
                        losses_by_iter[it].append(iter_to_loss[it])
            
            iters = list(losses_by_iter.keys())
            mean_losses = [np.mean(losses_by_iter[it]) for it in iters]
            std_losses = [np.std(losses_by_iter[it]) for it in iters]
            
            plt.plot(iters, mean_losses, label=style["label"], color=style["color"],
                     linestyle=style["linestyle"], linewidth=2, marker='o', markersize=4)
            plt.fill_between(iters,
                           np.array(mean_losses) - np.array(std_losses),
                           np.array(mean_losses) + np.array(std_losses),
                           alpha=0.2, color=style["color"])
        else:
            iters, _, val_losses = runs[0]["data"]["eval"]
            plt.plot(iters, val_losses, label=style["label"], color=style["color"],
                     linestyle=style["linestyle"], linewidth=2, marker='o', markersize=4)
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def plot_ablation_pair(data: dict, baseline_type: str, ablation_type: str, 
                        output_dir: Path, ablation_name: str):
    """绘制特定消融对比（baseline vs 变体）"""
    if baseline_type not in data or ablation_type not in data:
        print(f"Skipping {ablation_name}: missing data for {baseline_type} or {ablation_type}")
        return
    
    pair_data = {
        baseline_type: data[baseline_type],
        ablation_type: data[ablation_type],
    }
    
    # 训练损失对比
    plot_train_loss_comparison(
        pair_data,
        output_dir / f"ablation_{ablation_name}_train.png",
        f"Ablation: {ablation_name} - Training Loss"
    )
    
    # 验证损失对比
    plot_val_loss_comparison(
        pair_data,
        output_dir / f"ablation_{ablation_name}_val.png",
        f"Ablation: {ablation_name} - Validation Loss"
    )


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--runs_dir", type=Path, default=ABLATIONS_DIR,
                        help="Directory containing ablation runs")
    parser.add_argument("--output_dir", type=Path, default=PLOTS_DIR,
                        help="Directory to save plots")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting runs from: {args.runs_dir}")
    data = collect_runs(args.runs_dir)
    
    if not data:
        print("No valid runs found!")
        return
    
    print(f"Found models: {list(data.keys())}")
    for model_type, runs in data.items():
        print(f"  {model_type}: {len(runs)} run(s)")
    
    # 1. 所有模型对比图
    print("\nGenerating overall comparison plots...")
    plot_train_loss_comparison(data, args.output_dir / "all_models_train.png",
                                "All Models - Training Loss")
    plot_val_loss_comparison(data, args.output_dir / "all_models_val.png",
                              "All Models - Validation Loss")
    
    # 2. 各消融对比图
    print("\nGenerating individual ablation plots...")
    
    # Ablation 1: Pre-norm vs Post-norm
    plot_ablation_pair(data, "baseline", "postnorm", args.output_dir, "prenorm_vs_postnorm")
    
    # Ablation 2: RoPE vs NoPE
    plot_ablation_pair(data, "baseline", "nope", args.output_dir, "rope_vs_nope")
    
    # Ablation 3: SwiGLU vs SiLU
    plot_ablation_pair(data, "baseline", "silu", args.output_dir, "swiglu_vs_silu")
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
