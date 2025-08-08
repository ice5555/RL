from retrieval.bm25 import BM25Retriever
from utils.prompt import PromptBuilder
from typing import Optional

class QASystem:
    def __init__(self, retriever, generator, prompt_builder: Optional[PromptBuilder]=None, reranker=None):
        self.retriever = retriever
        self.generator = generator
        self.prompt = prompt_builder or PromptBuilder()
        self.reranker = reranker

    def answer(self, query: str, top_k: int = 5, temperature: float = 0.7, rerank_on: int | bool = 0) -> str:
        # 1) 先用 BM25 拿多一点候选
        cand = self.retriever.retrieve(query, top_k=max(top_k * 5, 20))
        row_ids = cand["ids"] if isinstance(cand, dict) else cand

        # 2) 可选精排
        if self.reranker and int(rerank_on) == 1:
            docs = self.retriever.get_content(row_ids, field="text")
            row_ids = self.reranker.rerank(query, docs, row_ids, top_k=top_k)
        else:
            row_ids = row_ids[:top_k]

        # 3) 拼上下文 + 生成
        contexts = self.retriever.get_content(row_ids, field="text")
        prompt = self.prompt.build(query, contexts)
        return self.generator.generate(prompt, temperature=temperature)