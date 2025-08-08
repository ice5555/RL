import pandas as pd
from data.hotpot import prepare_hotpot_data
from retrieval.bm25 import BM25Retriever

def test_bm25(tmp_path):
    corpus_p, _ = prepare_hotpot_data(
        'data/hotpot_train_v1.1.json', str(tmp_path), sample=20
    )

    df = pd.read_parquet(corpus_p)
    retr = BM25Retriever(df)
    out = retr.retrieve('Pride and Prejudice', top_k=10)
    assert 'ids' in out and len(out['ids']) == 10