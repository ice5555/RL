# envs/rag_env.py
from __future__ import annotations
import time, re
import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from pathlib import Path

from evaluation.evaluator import exact_match, f1

def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())

class RAGEnv(Env):
    """
    单步 Episode：一次问答即结束。
    动作：离散 id → (top_k, temperature, rerank_on) 三元组
    观测：简化的 16 维特征向量（可逐步扩展）
    奖励：0.7*EM + 0.3*F1 - 0.05*latency_sec，裁剪到 [-1,1]
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        qa_system,                 # 你的 QASystem 实例
        eval_df,     # 含 question/answer/gold_docs
        topk_choices=(5, 10, 20, 50),
        temp_choices=(0.0, 0.3, 0.7, 1.0),
        rerank_choices=(0, 1),
        obs_dim=16,
        reward_coef=(0.7, 0.3),
        latency_coef=0.05,
        seed: int | None = 42,
    ):
        super().__init__()
        self.qa = qa_system
        if isinstance(eval_df, (str, Path)):
            self.df = pd.read_parquet(eval_df).reset_index(drop=True)
        else:
            self.df = eval_df.reset_index(drop=True)
        self.n = len(self.df)

        # 动作空间：笛卡尔积
        self.topk_choices  = list(topk_choices)
        self.temp_choices  = list(temp_choices)
        self.rerank_choices= list(rerank_choices)
        self.actions = [(tk, tp, rr)
                        for tk in self.topk_choices
                        for tp in self.temp_choices
                        for rr in self.rerank_choices]
        self.action_space = spaces.Discrete(len(self.actions))

        # 观测空间
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.alpha, self.beta = reward_coef
        self.latency_coef = latency_coef

        self._rng = np.random.RandomState(seed)
        self._idx = 0
        self._last_action = (self.topk_choices[0], self.temp_choices[0], self.rerank_choices[0])

    def _make_obs(self, question: str) -> np.ndarray:
        # 非常简化的 16 维：长度、上一次动作、若干占位
        q_len = len(_tok(question))
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = min(q_len / 64.0, 1.0)
        # 上一步动作 one-hot 的前三维（粗略编码）
        tk_i = self.topk_choices.index(self._last_action[0]) / max(1, len(self.topk_choices)-1)
        tp_i = self.temp_choices.index(self._last_action[1]) / max(1, len(self.temp_choices)-1)
        rr_i = float(self._last_action[2])
        obs[1:4] = [tk_i, tp_i, rr_i]
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        self._idx = self._rng.randint(0, self.n)
        q = self.df.loc[self._idx, "question"]
        return self._make_obs(q), {}

    def step(self, action_id: int):
        assert self.action_space.contains(action_id), "invalid action id"
        top_k, temperature, rerank_on = self.actions[action_id]
        self._last_action = (top_k, temperature, rerank_on)

        row = self.df.loc[self._idx]
        q, gold = row["question"], row["answer"]

        # 应用动作：目前没有 reranker，就先忽略 rerank_on
        t0 = time.perf_counter()
        pred = self.qa.answer(q, top_k=top_k, temperature=temperature, rerank_on=rerank_on)
        t1 = time.perf_counter()

        # 质量奖励
        em = exact_match(pred, gold)
        f1_score = f1(pred, gold)
        quality = self.alpha * em + self.beta * f1_score

        # 延迟惩罚（秒）
        latency = t1 - t0
        reward = quality - self.latency_coef * latency
        reward = float(np.clip(reward, -1.0, 1.0))

        obs = self._make_obs(q)
        terminated = True     # 单步 episode
        truncated  = False
        info = {
            "pred": pred, "gold": gold,
            "em": em, "f1": f1_score,
            "top_k": top_k, "temperature": temperature, "rerank_on": rerank_on,
            "latency_sec": latency,
            'reward': reward,
        }
        return obs, reward, terminated, truncated, info
    
    def action_id_for(self, top_k: int, temperature: float, rerank_on: int) -> int:
        """把 (top_k, temperature, rerank_on) 映射到 action_id"""
        try:
            return self.actions.index((top_k, temperature, rerank_on))
        except ValueError:
            raise ValueError(f"Invalid action triple: {(top_k, temperature, rerank_on)}; "
                             f"valid choices: top_k={self.topk_choices}, temp={self.temp_choices}, rerank={self.rerank_choices}")    