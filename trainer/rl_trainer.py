# trainer/rl_trainer.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from trainer.callbacks import CSVLoggerCallback, EvalCSVCallback
import numpy as np
from pathlib import Path
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  
class RLTrainer:
    def __init__(self, env, log_dir=".cache/rl_logs", total_timesteps=1000, ppo_kwargs=None,
                  eval_env=None, eval_n_episodes: int = 50, eval_freq: int = 500):
        self.env = DummyVecEnv([lambda: env])
        self.log_dir = Path(log_dir)           
        self.log_dir.mkdir(parents=True, exist_ok=True)        
        self.total_timesteps = total_timesteps
        self.eval_env = eval_env           # ✅ 评估环境（非 vec）
        self.eval_n_episodes = int(eval_n_episodes)
        self.eval_freq = int(eval_freq)

        ppo_kwargs = ppo_kwargs or {}   # ✅ 新增        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            n_steps=int(ppo_kwargs.get("n_steps", 2048)),
            batch_size=int(ppo_kwargs.get("batch_size", 64)),
            gamma=float(ppo_kwargs.get("gamma", 0.99)),
            learning_rate=float(ppo_kwargs.get("learning_rate", 3e-4)),
        )
        print(f"[PPO] n_steps={self.model.n_steps}, batch_size={self.model.batch_size}")  # ✅ 确认
    def train(self):
        callbacks = [
            CheckpointCallback(save_freq=500, save_path=str(self.log_dir), name_prefix="ppo_rag"),
            CSVLoggerCallback(self.log_dir / "train_log.csv"),
        ]
        if self.eval_env is not None:
            callbacks.append(EvalCSVCallback(
                eval_env=self.eval_env,
                log_path=self.log_dir / "eval_log.csv",
                n_episodes=self.eval_n_episodes,
                eval_freq=self.eval_freq
            ))
        self.model.learn(total_timesteps=self.total_timesteps, callback=CallbackList(callbacks))
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path):
        self.model = PPO.load(path, env=self.env)

    def evaluate(self, n_episodes=50):
        """简单评估：用当前策略 roll n_episodes 次，返回平均 reward/EM/F1/latency。"""
        total_r, ems, f1s, lats = [], [], [], []
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            ep_r = 0.0
            last_info = None
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, dones, infos = self.env.step(action)
                ep_r += float(r[0])
                last_info = infos[0]
                done = bool(dones[0])
            total_r.append(ep_r)
            if last_info:
                ems.append(float(last_info.get("em", 0.0)))
                f1s.append(float(last_info.get("f1", 0.0)))
                lats.append(float(last_info.get("latency_sec", 0.0)))
        return {
            "avg_reward": float(np.mean(total_r)),
            "avg_em":     float(np.mean(ems)) if ems else 0.0,
            "avg_f1":     float(np.mean(f1s)) if f1s else 0.0,
            "avg_latency":float(np.mean(lats)) if lats else 0.0,
        }