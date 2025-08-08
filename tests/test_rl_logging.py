# tests/test_rl_logging.py
from data.hotpot import prepare_hotpot_data
from retrieval.bm25 import BM25Retriever
from models.generator import TextGenerator
from utils.prompt import PromptBuilder
from cli.qa_system import QASystem
from envs.rag_env import RAGEnv
from trainer.rl_trainer import RLTrainer
import pandas as pd
from pathlib import Path

def test_rl_logging(tmp_path):
    corpus_p, eval_p = prepare_hotpot_data("data/hotpot_train_v1.1.json", str(tmp_path), sample=5)
    df_c, df_e = pd.read_parquet(corpus_p), pd.read_parquet(eval_p)
    qa = QASystem(BM25Retriever(df_c), TextGenerator("sshleifer/tiny-gpt2","mps",16), PromptBuilder())
    env = RAGEnv(qa, df_e)
    log_dir = tmp_path / "logs"
    trainer = RLTrainer(env, log_dir=str(log_dir), total_timesteps=10)
    trainer.train()
    assert (log_dir / "train_log.csv").exists()