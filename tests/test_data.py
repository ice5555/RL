import pytest
from data.hotpot import prepare_hotpot_data

def test_prepare(tmp_path):
    corpus_p, eval_p = prepare_hotpot_data(
        'data/hotpot_train_v1.1.json', str(tmp_path), sample=10
    )
    assert corpus_p.exists() and eval_p.exists()