# trainer/callbacks.py
from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import csv, time
import numpy as np

class CSVLoggerCallback(BaseCallback):
    """
    逐步从 env infos 里抓取指标写入 CSV。
    适配 SB3 的 vec env；我们的 episode 是单步，所以每步都算一个样本。
    """
    def __init__(self, log_path: str | Path):
        super().__init__()
        self.log_path = Path(log_path)
        self._fh = None
        self._writer = None
        self._t0 = time.time()

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "global_step","timesteps","walltime_s",
                "reward","em","f1","latency_sec",
                "top_k","temperature","rerank_on"
            ]
        )
        self._writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        timesteps = int(self.num_timesteps)
        wall = time.time() - self._t0
        for info in infos:
            if not info or "em" not in info:  # 可能有空 dict
                continue
            row = {
                "global_step": timesteps,
                "timesteps": timesteps,
                "walltime_s": f"{wall:.3f}",
                "reward":     float(info.get("reward", 0.0) if "reward" in info else self.locals.get("rewards",[0])[0]),
                "em":         float(info.get("em", 0.0)),
                "f1":         float(info.get("f1", 0.0)),
                "latency_sec":float(info.get("latency_sec", 0.0)),
                "top_k":      int(info.get("top_k", -1)),
                "temperature":float(info.get("temperature", -1)),
                "rerank_on":  int(info.get("rerank_on", -1)),
            }
            self._writer.writerow(row)
        return True

    def _on_training_end(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None


class EvalCSVCallback(BaseCallback):
    """
    每隔 eval_freq steps，用当前策略在 eval_env 上评估 n_episodes 次，
    将 avg_reward / avg_em / avg_f1 / avg_latency 追加到 CSV。
    """
    def __init__(self, eval_env, log_path: str | Path, n_episodes: int = 50, eval_freq: int = 500):
        super().__init__()
        self.eval_env = eval_env
        self.log_path = Path(log_path)
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self._fh = None
        self._writer = None
        self._t0 = time.time()
        self._last_eval_step = 0

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.log_path.exists()
        self._fh = self.log_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=["timesteps", "walltime_s", "avg_reward", "avg_em", "avg_f1", "avg_latency"]
        )
        if is_new:
            self._writer.writeheader()

    def _rollout_eval(self):
        rewards, ems, f1s, lats = [], [], [], []
        for _ in range(self.n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_r = 0.0
            last_info = None
            while not done:
                # 用当前策略（确定性）选动作
                action, _state = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = self.eval_env.step(action)
                ep_r += float(r)
                last_info = info
                done = bool(terminated or truncated)
            rewards.append(ep_r)
            if last_info:
                ems.append(float(last_info.get("em", 0.0)))
                f1s.append(float(last_info.get("f1", 0.0)))
                lats.append(float(last_info.get("latency_sec", 0.0)))
        def avg(x): return float(np.mean(x)) if x else 0.0
        return {
            "avg_reward": avg(rewards),
            "avg_em":     avg(ems),
            "avg_f1":     avg(f1s),
            "avg_latency":avg(lats),
        }

    def _on_step(self) -> bool:
        now = int(self.num_timesteps)
        if now - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = now
            metrics = self._rollout_eval()
            wall = time.time() - self._t0
            row = {"timesteps": now, "walltime_s": f"{wall:.3f}", **metrics}
            self._writer.writerow(row)
            # 也顺手在控制台打一个简短行
            print(f"[Eval@{now}] {metrics}")
        return True

    def _on_training_end(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None           