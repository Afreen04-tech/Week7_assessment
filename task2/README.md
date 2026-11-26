# Task 2 – Advanced Retrieval

Run from this folder: `python -m pip install -r requirements.txt`, then `python evaluate_retrieval.py`. First run downloads embedding/cross-encoder models. Run `python rerank.py` to see top-50 to top-5 reranking. `retrieval.py` includes BM25, dense retrieval, RRF and metadata filters; `query_rewriting.py` merges three alternate phrasings using RRF. The included evaluation uses 12 queries; adjust expected chunk IDs if you edit documents. The small demo reports actual local metrics, not preset example scores.
