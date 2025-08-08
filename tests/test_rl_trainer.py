# tests/test_rl_trainer.py
import pytest
from envs.rag_env import RAGEnv
from cli.qa_system import QASystem
from retrieval.bm25 import BM25Retriever
from models.generator import TextGenerator
from utils.prompt import PromptBuilder
import pandas as pd
from trainer.rl_trainer import RLTrainer
from data.hotpot import prepare_hotpot_data

@pytest.mark.slow
def test_rl_smoke(tmp_path):
    corpus_p, qa_p = prepare_hotpot_data("data/hotpot_train_v1.1.json", str(tmp_path), sample=5)
    df = pd.read_parquet(corpus_p)
    retriever = BM25Retriever(df)
    generator = TextGenerator(model_id="sshleifer/tiny-gpt2", device="mps", max_new_tokens=16)
    prompt_builder = PromptBuilder()
    qa_system = QASystem(retriever, generator, prompt_builder)
    env = RAGEnv(qa_system, qa_p)

    trainer = RLTrainer(env, total_timesteps=10)
    trainer.train()
    avg_r = trainer.evaluate(n_episodes=1)
    assert isinstance(avg_r, float)