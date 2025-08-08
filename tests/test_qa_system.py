import pandas as pd
from data.hotpot import prepare_hotpot_data
from retrieval.bm25 import BM25Retriever
from models.generator import TextGenerator
from utils.prompt import PromptBuilder
from cli.qa_system import QASystem

def test_qa(tmp_path):
    corpus_p, _ = prepare_hotpot_data('data/hotpot_train_v1.1.json', str(tmp_path), sample=20)
    df = pd.read_parquet(corpus_p)
    retriever = BM25Retriever(df)
    generator = TextGenerator(model_id="sshleifer/tiny-gpt2", device="cpu")
    prompt_builder = PromptBuilder()
    qa = QASystem(retriever, generator, prompt_builder)
    answer = qa.answer("Who wrote Pride and Prejudice?", top_k=3)
    print(answer)
    assert isinstance(answer, str) and len(answer) > 0

if __name__=='__main__':
    test_qa()