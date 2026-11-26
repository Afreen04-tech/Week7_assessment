"""Cross-encoder reranking. Retrieves top 50, then reranks to top 5."""
from sentence_transformers import CrossEncoder
from retrieval import HybridRetriever

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
def rerank(query, candidates, top_k=5):
    if not candidates: return []
    model=CrossEncoder(MODEL)
    pairs=[(query,item['text']) for item in candidates]
    scores=model.predict(pairs)
    ranked=sorted(zip(scores,candidates),key=lambda x:x[0],reverse=True)[:top_k]
    return [{**item,"rerank_score":float(score)} for score,item in ranked]

if __name__ == '__main__':
    r=HybridRetriever(); q='How many casual leaves per year?'
    before=r.hybrid_search(q,50); after=rerank(q,before,5)
    print('Before reranking top 5:', [x['id'] for x in before[:5]])
    print('After reranking top 5:', [x['id'] for x in after])
