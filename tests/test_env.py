import pandas as pd
import numpy as np
from envs.rag_env import RAGEnv

class FakeQA:
    """最小伪 QA：直接返回 gold，当作上限测试；也保留 retriever 属性占位。"""
    def __init__(self, eval_df: pd.DataFrame):
        self.eval_df = eval_df

    def answer(self, query: str, top_k: int = 5, **gen_kwargs) -> str:
        # 在 eval_df 里查同问句的 gold
        row = self.eval_df[self.eval_df["question"] == query].iloc[0]
        return row["answer"]

def _prep_eval_df(tmp_path, sample=10):
    # 复用你已存在的 prepare_hotpot_data
    from data.hotpot import prepare_hotpot_data
    corpus_p, eval_p = prepare_hotpot_data("data/hotpot_train_v1.1.json", str(tmp_path), sample=sample)
    return pd.read_parquet(eval_p)

def test_env_step(tmp_path):
    eval_df = _prep_eval_df(tmp_path, sample=12)
    qa = FakeQA(eval_df)
    env = RAGEnv(qa, eval_df, topk_choices=(5,10), temp_choices=(0.0,0.7), rerank_choices=(0,1), obs_dim=16)

    obs, info = env.reset()
    assert obs.shape == (16,)
    assert (obs >= -1e9).all() and (obs <= 1e9).all()

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (16,)
    assert -1.0 <= reward <= 1.0
    assert terminated and not truncated
    for k in ["pred", "gold", "em", "f1", "top_k", "temperature", "latency_sec"]:
        assert k in info
    # 因为 FakeQA 返回 gold，em 应该是 1
    assert info["em"] == 1.0