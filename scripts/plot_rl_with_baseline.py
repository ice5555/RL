from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ast

LOG_DIR   = Path(".cache/rl_logs")
TRAIN_CSV = LOG_DIR / "train_log.csv"   # per-episode
EVAL_CSV  = LOG_DIR / "eval_log.csv"    # periodic eval during training
BL_FIXED  = LOG_DIR / "eval_baseline_fixed.txt"
BL_RANDOM = LOG_DIR / "eval_baseline_random.txt"

def load_baseline(p: Path):
    if p.exists():
        try:
            return ast.literal_eval(p.read_text())
        except Exception:
            return None
    return None

def add_baseline_lines(metric_name, fixed, random_):
    if fixed and f"avg_{metric_name}" in fixed:
        plt.axhline(fixed[f"avg_{metric_name}"], linestyle="--", label=f"fixed avg {metric_name}")
    if random_ and f"avg_{metric_name}" in random_:
        plt.axhline(random_[f"avg_{metric_name}"], linestyle=":", label=f"random avg {metric_name}")

def plot_metric(df, col, title, ylabel, out_path, window=20, with_baseline=True, fixed=None, random_=None):
    plt.figure()
    df[col].plot(alpha=0.3, label=col)  # per-episode curve
    if len(df) >= window:
        df[col].rolling(window).mean().plot(label=f"{col} (rolling {window})")
    if with_baseline:
        # 这三个支持 baseline；其它就不用加
        add_baseline_lines(col if col in ("reward","em","f1") else "reward", fixed, random_)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_eval_metric(eval_df, metric, title, ylabel, out_path, fixed=None, random_=None):
    """eval_log.csv: 横轴是 timesteps，更适合论文主图"""
    if eval_df is None or eval_df.empty or metric not in eval_df.columns:
        return
    plt.figure()
    eval_df.plot(x="timesteps", y=metric, legend=False)
    add_baseline_lines(metric, fixed, random_)
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def make_grid_overview(train_df, fixed, random_, out_path):
    """可选：合成 2x2 网格（Reward/EM/F1/Latency），用于报告鸟瞰"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # Reward
    ax = axes[0,0]
    train_df["reward"].plot(ax=ax, alpha=0.3)
    if len(train_df) >= 20:
        train_df["reward"].rolling(20).mean().plot(ax=ax)
    if fixed and "avg_reward" in fixed:  ax.axhline(fixed["avg_reward"], linestyle="--", label="fixed")
    if random_ and "avg_reward" in random_: ax.axhline(random_["avg_reward"], linestyle=":", label="random")
    ax.set_title("Reward"); ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.legend()

    # EM
    ax = axes[0,1]
    train_df["em"].plot(ax=ax, alpha=0.3)
    if len(train_df) >= 20:
        train_df["em"].rolling(20).mean().plot(ax=ax)
    if fixed and "avg_em" in fixed:      ax.axhline(fixed["avg_em"], linestyle="--", label="fixed")
    if random_ and "avg_em" in random_:  ax.axhline(random_["avg_em"], linestyle=":", label="random")
    ax.set_title("EM"); ax.set_xlabel("Episode"); ax.set_ylabel("EM"); ax.legend()

    # F1
    ax = axes[1,0]
    train_df["f1"].plot(ax=ax, alpha=0.3)
    if len(train_df) >= 20:
        train_df["f1"].rolling(20).mean().plot(ax=ax)
    if fixed and "avg_f1" in fixed:      ax.axhline(fixed["avg_f1"], linestyle="--", label="fixed")
    if random_ and "avg_f1" in random_:  ax.axhline(random_["avg_f1"], linestyle=":", label="random")
    ax.set_title("F1"); ax.set_xlabel("Episode"); ax.set_ylabel("F1"); ax.legend()

    # Latency
    ax = axes[1,1]
    train_df["latency_sec"].plot(ax=ax, alpha=0.3)
    if len(train_df) >= 20:
        train_df["latency_sec"].rolling(20).mean().plot(ax=ax)
    ax.set_title("Latency (s)"); ax.set_xlabel("Episode"); ax.set_ylabel("Latency")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main(make_grid=False):
    assert TRAIN_CSV.exists(), f"train log not found: {TRAIN_CSV}"
    train_df = pd.read_csv(TRAIN_CSV)
    eval_df  = pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else None
    fixed    = load_baseline(BL_FIXED)
    random_  = load_baseline(BL_RANDOM)

    outdir = Path("figs"); outdir.mkdir(parents=True, exist_ok=True)

    # 训练（Episode 轴）
    plot_metric(train_df, "reward",      "Episode Reward (with baselines)", "Reward",       outdir / "reward_with_baseline.png", fixed=fixed, random_=random_)
    plot_metric(train_df, "em",          "Exact Match (EM)",                "EM",           outdir / "em_with_baseline.png",     fixed=fixed, random_=random_)
    plot_metric(train_df, "f1",          "F1 Score",                        "F1",           outdir / "f1_with_baseline.png",     fixed=fixed, random_=random_)
    plot_metric(train_df, "latency_sec", "Latency per Episode (s)",         "Latency (s)",  outdir / "latency.png",              with_baseline=False)

    # 评估（Timesteps 轴）
    if eval_df is not None:
        plot_eval_metric(eval_df, "avg_reward", "Eval: Reward vs Timesteps", "Avg Reward", outdir / "eval_reward.png", fixed=fixed, random_=random_)
        plot_eval_metric(eval_df, "avg_em",     "Eval: EM vs Timesteps",     "Avg EM",     outdir / "eval_em.png",     fixed=fixed, random_=random_)
        plot_eval_metric(eval_df, "avg_f1",     "Eval: F1 vs Timesteps",     "Avg F1",     outdir / "eval_f1.png",     fixed=fixed, random_=random_)
        plot_eval_metric(eval_df, "avg_latency","Eval: Latency vs Timesteps","Avg Latency (s)", outdir / "eval_latency.png")
    # 动作分布
    plt.figure()
    train_df["top_k"].value_counts().sort_index().plot(kind="bar")
    plt.title("Action Distribution: top_k")
    plt.xlabel("top_k"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(outdir / "topk_hist.png", dpi=150); plt.close()

    plt.figure()
    train_df["temperature"].value_counts().sort_index().plot(kind="bar")
    plt.title("Action Distribution: temperature")
    plt.xlabel("temperature"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(outdir / "temp_hist.png", dpi=150); plt.close()

    if make_grid:
        make_grid_overview(train_df, fixed, random_, outdir / "grid_overview.png")

    print(f"[ok] saved plots to {outdir.resolve()}")

if __name__ == "__main__":
    # 默认不做网格；需要就传 True
    main(make_grid=True)