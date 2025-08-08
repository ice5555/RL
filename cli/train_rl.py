# cli/train_rl.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 静音并行 tokenizer 警告

import yaml
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd

from envs.rag_env import RAGEnv
from cli.qa_system import QASystem
from retrieval.bm25 import BM25Retriever
from models.generator import TextGenerator
from utils.prompt import PromptBuilder
from data.hotpot import prepare_hotpot_data
from trainer.rl_trainer import RLTrainer
from retrieval.rerank_ce import CrossEncoderReranker

def run_baseline(env: RAGEnv, mode: str, episodes: int, fixed=None, log_dir: Path | None = None):
    """
    Baseline rollout，不依赖 SB3。
    mode: 'fixed' | 'random'
    - fixed: 用固定动作 (top_k, temperature, rerank_on)
    - random: 每个 episode 随机动作
    """
    rng = random.Random(42)
    rewards, ems, f1s, lats = [], [], [], []
    actions = []

    # 解析固定动作
    if mode == "fixed":
        fixed = fixed or {"top_k": 20, "temperature": 0.7, "rerank_on": 0}
        triple = (fixed["top_k"], fixed["temperature"], fixed["rerank_on"])
        # 如果环境提供映射函数，用它；否则直接找 index
        if hasattr(env, "action_id_for"):
            try:
                fixed_id = env.action_id_for(*triple)
            except Exception:
                fixed_id = 0
        else:
            try:
                fixed_id = env.actions.index(triple)
            except Exception:
                fixed_id = 0

    for _ in range(episodes):
        # 每个 episode 走一次：reset -> step
        env.reset()
        if mode == "random":
            action_id = env.action_space.sample()
        else:
            action_id = fixed_id

        _, r, terminated, truncated, info = env.step(action_id)
        # 单步 episode，直接拿 info
        rewards.append(float(r))
        ems.append(float(info.get("em", 0.0)))
        f1s.append(float(info.get("f1", 0.0)))
        lats.append(float(info.get("latency_sec", 0.0)))
        actions.append(action_id)

    metrics = {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_em": float(np.mean(ems)) if ems else 0.0,
        "avg_f1": float(np.mean(f1s)) if f1s else 0.0,
        "avg_latency": float(np.mean(lats)) if lats else 0.0,
        "episodes": int(episodes),
        "mode": mode,
    }
    if mode == "fixed":
        metrics["fixed_action"] = str(triple)

    # 可选：写个简易 baseline 日志
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"eval_baseline_{mode}.txt").write_text(str(metrics), encoding="utf-8")

    print(f"[Baseline-{mode}] {metrics}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--baseline_only", action="store_true", help="只跑 baseline，不训练 PPO")
    ap.add_argument("--baseline_mode", choices=["fixed", "random"], default=None,
                    help="覆盖配置里的 baseline.mode")
    ap.add_argument("--episodes", type=int, default=None,
                    help="覆盖 baseline 评测轮数")
    args = ap.parse_args()

    log_dir = Path(".cache/rl_logs")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1) 数据准备（parquet）
    corpus_p, eval_p = prepare_hotpot_data(
        cfg["paths"]["dataset_json"],
        cfg["data"]["cache_dir"],
        sample=cfg["data"]["sample"]
    )
    df_c = pd.read_parquet(corpus_p)
    df_e = pd.read_parquet(eval_p)

    # 2) 组装 QA 系统
    retriever = BM25Retriever(df_c)
    generator = TextGenerator(
        model_id=cfg["model"]["name"],
        device=cfg["model"]["device"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
    )

    reranker = CrossEncoderReranker()  # 默认 ms-marco-MiniLM
    qa_system = QASystem(retriever, generator, PromptBuilder(), reranker=reranker)
    
    # 3) 环境
    env = RAGEnv(qa_system, df_e)
    eval_env = RAGEnv(qa_system, df_e) 

    # 4) Baseline 对照
    bl_cfg = cfg.get("baseline", {
        "mode": "fixed",
        "episodes": min(50, len(df_e)),
        "fixed": {"top_k": 20, "temperature": 0.7, "rerank_on": 0},
    })
    if args.baseline_mode is not None:
        bl_cfg["mode"] = args.baseline_mode
    if args.episodes is not None:
        bl_cfg["episodes"] = args.episodes

    if bl_cfg["mode"] in ("fixed", "random"):
        print(f"[Baseline] mode={bl_cfg['mode']} episodes={bl_cfg['episodes']}")
        run_baseline(env, bl_cfg["mode"], bl_cfg["episodes"], fixed=bl_cfg.get("fixed"), log_dir=log_dir)

    if args.baseline_only:
        return

    # 5) PPO 训练
    ppo_kwargs = cfg.get("ppo", {})
    eval_cfg = cfg.get("eval", {"freq": 200, "episodes": min(50, len(df_e))})
    trainer = RLTrainer(
        env,
        log_dir=str(log_dir),
        total_timesteps=cfg["train"]["total_timesteps"],
        ppo_kwargs=ppo_kwargs,
        eval_env=eval_env,                         # ✅
        eval_n_episodes=eval_cfg.get("episodes", 50),
        eval_freq=eval_cfg.get("freq", 200),
    )
    print(f"[Info] Starting PPO with n_steps={ppo_kwargs.get('n_steps', 2048)} "
          f"batch_size={ppo_kwargs.get('batch_size', 64)} | Eval freq={eval_cfg.get('freq', 200)} eps={eval_cfg.get('episodes', 50)}")    
    trainer.train()

    # 6) 评估
    metrics3 = trainer.evaluate(n_episodes=3)
    print("Eval(3eps):", metrics3)
    metrics_full = trainer.evaluate(n_episodes=min(50, len(df_e)))
    print("Eval(full):", metrics_full)
    print(f"[Logs] CSV written to: {log_dir / 'train_log.csv'}")
    print(f"[Logs] Eval  CSV: {log_dir / 'eval_log.csv'}")

if __name__ == "__main__":
    main()