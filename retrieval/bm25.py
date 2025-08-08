# retrieval/bm25.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus_df: pd.DataFrame, text_col: str = "text"):
        """
        兼容两种情况：
        - 有 'row_id' 列：用它当外部ID，并把 DataFrame 的 index 设为 row_id（保留列）
        - 没有 'row_id' 列：用行号当 ID
        """
        df = corpus_df.copy()
        self.text_col = text_col

        if "row_id" in df.columns:
            # 把 index 设为 row_id，保留列，便于 .loc 访问
            if df.index.name != "row_id":
                df = df.set_index("row_id", drop=False)
            self.ids = df["row_id"].tolist()
            self._use_loc = True   # 通过 .loc 以 row_id 取行
        else:
            # 没有 row_id 列就用自然行号
            df = df.reset_index(drop=True)
            self.ids = list(range(len(df)))
            self._use_loc = False  # 用 .iloc 以位置取行

        self.corpus = df
        # 构建 BM25 需要的分词文本序列（与 self.ids 顺序一一对应）
        self.texts = df[self.text_col].astype(str).tolist()
        self.tokenized = [re.findall(r"\w+", t.lower()) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, top_k: int = 5):
        q_tok = re.findall(r"\w+", str(query).lower())
        scores = self.bm25.get_scores(q_tok)               # ndarray, 与 self.texts 对齐
        order = np.argsort(scores)[::-1][:top_k]           # 位置索引
        ids = [self.ids[i] for i in order]                 # 外部 ID
        return {"ids": ids, "scores": scores[order].tolist()}

    def get_content(self, ids, field: str = None):
        """
        根据外部 ID 取指定字段（默认 self.text_col）。
        支持单个 id 或列表。
        """
        field = field or self.text_col
        if isinstance(ids, (int, np.integer)):
            ids = [int(ids)]
        out = []
        if self._use_loc:
            # index 是 row_id
            for rid in ids:
                out.append(self.corpus.loc[rid, field])
        else:
            # index 是 0..N-1，ids 就是位置
            for pos in ids:
                out.append(self.corpus.iloc[int(pos)][field])
        return out