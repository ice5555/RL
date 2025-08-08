# scripts/plot_rl_logs.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def maybe_pct(s):
    # 某些字段可能缺省，安全转换
    try:
        return s.astype(float)
    except Exception:
        return pd.Series([0.0] * len(s))

def plot_curve(series, title, ylabel, out_path):
    plt.figure()
    series.plot()
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(csv_path=".cache/rl_logs/train_log.csv", out_dir="figs"):
    csv_path = Path(csv_path)
    out_dir = ensure_dir(Path(out_dir))

    if not csv_path.exists():
        print(f"[warn] CSV not found: {csv_path}. Train more steps first.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[warn] CSV is empty. Train more steps first.")
        return

    # 保障必需列存在
    for col in ["reward","em","f1","latency_sec","top_k","temperature","rerank_on","timesteps","global_step"]:
        if col not in df.columns:
            df[col] = 0.0

    # 1) Reward / EM / F1 / Latency
    plot_curve(df["reward"],         "Episode Reward",         "Reward",       out_dir / "reward.png")
    plot_curve(df["em"],             "Exact Match (EM)",       "EM",           out_dir / "em.png")
    plot_curve(df["f1"],             "F1 Score",               "F1",           out_dir / "f1.png")
    plot_curve(df["latency_sec"],    "Latency per Episode(s)", "Latency (s)",  out_dir / "latency.png")

    # 2) 动作分布（累计直方图）
    plt.figure()
    df["top_k"].value_counts().sort_index().plot(kind="bar")
    plt.title("Action Distribution: top_k")
    plt.xlabel("top_k")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "topk_hist.png", dpi=150)
    plt.close()

    plt.figure()
    df["temperature"].value_counts().sort_index().plot(kind="bar")
    plt.title("Action Distribution: temperature")
    plt.xlabel("temperature")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "temp_hist.png", dpi=150)
    plt.close()

    print(f"[ok] Saved plots to {out_dir.resolve()}")

if __name__ == "__main__":
    main()