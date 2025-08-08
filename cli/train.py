#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import pandas as pd

from data.hotpot import prepare_hotpot_data
from retrieval.bm25 import BM25Retriever
from models.generator import TextGenerator
from utils.prompt import PromptBuilder
from evaluation.evaluator import Evaluator

# 允许无 omegaconf 运行
def load_cfg(path: str | None):
    default = {
        "paths": {"work_dir": ".cache", "dataset_json": "data/hotpot_train_v1.1.json"},
        "data":  {"sample": 50, "cache_dir": ".cache/prep"},
        "retrieval": {"top_k": 5},
        "model": {"name": "sshleifer/tiny-gpt2", "device": "cpu", "max_new_tokens": 64},
    }
    if not path:
        return default
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore
    except Exception:
        print("[warn] omegaconf not installed or config invalid, using defaults.")
        return default

# 轻量版 QASystem（沿用你前面的实现）
class QASystem:
    def __init__(self, retriever, generator, prompt_builder: PromptBuilder):
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder

    def answer(self, query: str, top_k: int = 5, **gen_kwargs) -> str:
        result = self.retriever.retrieve(query, top_k=top_k)
        # 用检索器内部缓存的 texts/ids 来取上下文
        contexts = []
        for rid in result["ids"]:
            idx = self.retriever.ids.index(rid)
            contexts.append(self.retriever.texts[idx])
        prompt = self.prompt_builder.build(query, contexts)
        return self.generator.generate(prompt, **gen_kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/hotpotqa_sft.yaml")
    ap.add_argument("--sample", type=int, default=None, help="override sample size")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.sample is not None:
        cfg["data"]["sample"] = args.sample

    work = Path(cfg["paths"]["work_dir"]); work.mkdir(parents=True, exist_ok=True)

    # 1) 准备数据（parquet）
    corpus_p, eval_p = prepare_hotpot_data(
        cfg["paths"]["dataset_json"],
        cfg["data"]["cache_dir"],
        sample=cfg["data"]["sample"],
    )
    corpus_df = pd.read_parquet(corpus_p)
    eval_df   = pd.read_parquet(eval_p)

    # 2) 组装 QA
    retr = BM25Retriever(corpus_df)
    gen  = TextGenerator(
        model_id=cfg["model"]["name"],
        device=cfg["model"]["device"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
    )
    from utils.prompt import PromptBuilder
    pb = PromptBuilder(system_prompt="You answer briefly and factually.", max_context=3)
    qa = QASystem(retr, gen, pb)

    # 3) 简单 smoke test
    demo_q = "Who wrote Pride and Prejudice?"
    print("DEMO Q:", demo_q)
    print("DEMO A:", qa.answer(demo_q, top_k=cfg["retrieval"]["top_k"])[:200], "...\n")

    # 4) 评估
    ev = Evaluator(qa, eval_df)
    r1 = ev.eval_retrieval(k_list=(1,3,5))
    r2 = ev.eval_generation(top_k=cfg["retrieval"]["top_k"])
    print("Retrieval:", r1)
    print("Generation:", r2)

if __name__ == "__main__":
    main()