from pathlib import Path
import json, pandas as pd, re

def prepare_hotpot_data(
    json_path: str,
    base_dir: str,
    sample: int = None
) -> tuple[Path, Path]:
    """
    从 JSON 构建 corpus.parquet 和 eval.parquet，返回两者路径。
    如果 sample 有值，则只用前 sample 条数据。
    """
    base = Path(base_dir)
    base.mkdir(exist_ok=True, parents=True)
    corpus_p = base / "corpus.parquet"
    eval_p   = base / "eval.parquet"

    data = json.load(open(json_path, 'r', encoding='utf-8'))
    if sample:
        data = data[:sample]

    # 构建段落语料
    docs = []
    for ex in data:
        for title, sents in ex['context']:
            docs.append({'title': title, 'text': ' '.join(sents).strip()})
    df = pd.DataFrame(docs).drop_duplicates('title').reset_index(drop=True)
    df['row_id'] = df.index
    df.to_parquet(corpus_p, index=False)

    # 构建评估集
    title2id = {r.title: r.row_id for r in df.itertuples()}
    rows = []
    for ex in data:
        gold = sorted({title2id[t] for t,_ in ex['supporting_facts'] if t in title2id})
        rows.append({
            'question': ex['question'],
            'answer':   ex['answer'],
            'gold_docs': gold
        })
    pd.DataFrame(rows).to_parquet(eval_p, index=False)

    return corpus_p, eval_p