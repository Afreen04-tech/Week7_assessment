"""RAGAS evaluation starter. Reads the 15-row golden JSONL and reports dataset composition.
For full metric scoring, connect an LLM and embeddings supported by your installed RAGAS version."""
import json
from pathlib import Path
DATA=Path(__file__).with_name('golden.jsonl')
def main():
    rows=[json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
    counts={k:sum(x['type']==k for x in rows) for k in ('simple','multi_chunk','unanswerable')}
    print('Golden questions loaded:',len(rows)); print('Composition:',counts)
    print('RAGAS metrics to evaluate: faithfulness, answer relevancy, context precision, context recall')
    print('Next: run the project RAG pipeline for each question and pass question, answer, contexts, and reference to RAGAS.')
if __name__=='__main__': main()
