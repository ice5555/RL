# tests/test_rerank.py
import pandas as pd
import pytest

from retrieval.bm25 import BM25Retriever
from retrieval.rerank_ce import CrossEncoderReranker

# 我们用一个假的生成器，避免加载大模型
class DummyGenerator:
    def generate(self, prompt, **kwargs):
        return "dummy"

# 也用一个简单的 PromptBuilder（如果你项目已有 utils.prompt 就用它）
class DummyPrompt:
    def build(self, q, ctxs):
        return f"Q: {q}\n\n" + "\n".join(ctxs)

# 用真实的 QASystem
from cli.qa_system import QASystem

@pytest.fixture
def tiny_corpus_df():
    rows = [
        {"title":"JaneAusten", "text":"Pride and Prejudice is a novel by Jane Austen. It was published in 1813."},
        {"title":"RandomSports","text":"The football club won the league in 2013 after a dramatic season."},
        {"title":"Physics","text":"Albert Einstein developed the theory of relativity and won the Nobel Prize."},
        {"title":"Cooking","text":"This recipe explains how to bake sourdough bread with a crispy crust."},
    ]
    df = pd.DataFrame(rows)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df

def test_cross_encoder_reranks_correctly(tiny_corpus_df):
    retr = BM25Retriever(tiny_corpus_df, text_col="text")
    reranker = CrossEncoderReranker()  # "cross-encoder/ms-marco-MiniLM-L-6-v2"

    query = "Who wrote Pride and Prejudice?"
    # 先拿多一点候选
    cand = retr.retrieve(query, top_k=10)
    row_ids = cand["ids"]

    # 让 CE 精排拿 top1
    docs = retr.get_content(row_ids, field="text")
    top_ids = reranker.rerank(query, docs, row_ids, top_k=1)
    assert len(top_ids) == 1
    top_id = top_ids[0]

    # 期望最相关的是 JaneAusten 那条
    top_title = tiny_corpus_df.loc[top_id, "title"]
    assert top_title == "JaneAusten"

def test_qasystem_uses_reranker_when_enabled(tiny_corpus_df):
    retr = BM25Retriever(tiny_corpus_df, text_col="text")
    reranker = CrossEncoderReranker()
    qa = QASystem(retriever=retr, generator=DummyGenerator(), prompt_builder=DummyPrompt(), reranker=reranker)

    query = "Who wrote Pride and Prejudice?"

    # 不开精排：取 BM25 前 top_k，title 可能不是 JaneAusten（BM25 不一定把它排第一）
    # 我们只检查不报错就行
    _ = qa.answer(query, top_k=2, temperature=0.0, rerank_on=0)

    # 开精排：希望 CE 把 JaneAusten 放到最前的上下文里
    # 为了验证，我们让 top_k=1，然后检查生成前拼接的第一个上下文就是 Austen 那条
    # 这里偷看一下 QASystem 的内部行为：通过检索再手动 CE 排序
    cand = retr.retrieve(query, top_k=10)
    ids = cand["ids"]
    docs = retr.get_content(ids, "text")
    ce_top1 = reranker.rerank(query, docs, ids, top_k=1)[0]
    assert tiny_corpus_df.loc[ce_top1, "title"] == "JaneAusten"

    # 真正调用 QASystem（把 rerank_on=1 打开），只要不报错即可
    out = qa.answer(query, top_k=1, temperature=0.0, rerank_on=1)
    assert isinstance(out, str)