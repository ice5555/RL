# utils/plot_rl_logs.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main(csv_path=".cache/rl_logs/train_log.csv", out_dir="figs"):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Empty CSV. Train more steps first.")
        return

    # 1) 奖励曲线（每行=1 episode）
    plt.figure()
    df["reward"].plot()
    plt.title("Episode Reward over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(out_dir / "reward.png", dpi=150)
    plt.close()

    # 2) EM / F1 曲线
    plt.figure()
    df["em"].plot()
    plt.title("EM over Time")
    plt.xlabel("Episode")
    plt.ylabel("EM")
    plt.tight_layout()
    plt.savefig(out_dir / "em.png", dpi=150)
    plt.close()

    plt.figure()
    df["f1"].plot()
    plt.title("F1 over Time")
    plt.xlabel("Episode")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(out_dir / "f1.png", dpi=150)
    plt.close()

    # 3) 延迟曲线
    plt.figure()
    df["latency_sec"].plot()
    plt.title("Latency (sec) over Time")
    plt.xlabel("Episode")
    plt.ylabel("Latency (sec)")
    plt.tight_layout()
    plt.savefig(out_dir / "latency.png", dpi=150)
    plt.close()

    # 4) 动作分布（累计直方图）
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

    print(f"Saved plots to {out_dir.resolve()}")

if __name__ == "__main__":
    main()