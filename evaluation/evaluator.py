import pandas as pd
from typing import Dict, Iterable
import re

def _normalize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())

def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())

def f1(pred: str, gold: str) -> float:
    p_tokens = _normalize(pred)
    g_tokens = _normalize(gold)
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = min(common.get(t, 0) + 1, g_tokens.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall    = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

class Evaluator:
    def __init__(self, qa_system, eval_df: pd.DataFrame):
        self.qa = qa_system
        self.df = eval_df.reset_index(drop=True)

    def eval_retrieval(self, k_list: Iterable[int] = (1, 3, 5)) -> Dict[str, float]:
        res = {}
        for k in k_list:
            hits = 0
            for q, gold_ids in zip(self.df["question"], self.df["gold_docs"]):
                ids = self.qa.retriever.retrieve(q, top_k=k)["ids"]
                if any(g in ids for g in gold_ids):
                    hits += 1
            res[f"Recall@{k}"] = hits / len(self.df)
        return res

    def eval_generation(self, top_k: int = 3) -> Dict[str, float]:
        em_sum, f1_sum = 0.0, 0.0
        for q, gold in zip(self.df["question"], self.df["answer"]):
            pred = self.qa.answer(q, top_k=top_k)
            em_sum += exact_match(pred, gold)
            f1_sum += f1(pred, gold)
        n = len(self.df)
        return {"EM": em_sum / n, "F1": f1_sum / n}