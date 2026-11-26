"""BM25, dense-vector retrieval and Reciprocal Rank Fusion for policy documents."""
from pathlib import Path
import re, time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

DOCS = Path(__file__).resolve().parent / "documents"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks():
    chunks=[]
    for path in sorted(DOCS.glob("*.txt")):
        text=path.read_text(encoding="utf-8")
        for i, part in enumerate(re.split(r"(?<=[.!?])\s+", text)):
            if part.strip():
                chunks.append({"id": f"{path.name}:{i}", "text": part.strip(),
                               "source": path.name, "department": path.stem.split('_')[0],
                               "year": 2026})
    return chunks

def tokenize(text): return re.findall(r"[a-z0-9]+", text.lower())

class HybridRetriever:
    def __init__(self):
        self.chunks=load_chunks()
        self.bm25=BM25Okapi([tokenize(c['text']) for c in self.chunks])
        self.model=SentenceTransformer(MODEL)
        self.vectors=self.model.encode([c['text'] for c in self.chunks], normalize_embeddings=True)
    def bm25_search(self, query, k=50):
        scores=self.bm25.get_scores(tokenize(query))
        return [self.chunks[i] for i in np.argsort(scores)[::-1][:k]]
    def vector_search(self, query, k=50):
        q=self.model.encode([query], normalize_embeddings=True)[0]
        scores=self.vectors @ q
        return [self.chunks[i] for i in np.argsort(scores)[::-1][:k]]
    @staticmethod
    def rrf(*ranked_lists, k=60):
        scores={}
        lookup={}
        for results in ranked_lists:
            for rank, item in enumerate(results, 1):
                key=item['id']; lookup[key]=item
                scores[key]=scores.get(key,0.0)+1/(k+rank)
        return [lookup[key] for key in sorted(scores,key=scores.get,reverse=True)]
    def hybrid_search(self, query, k=5):
        return self.rrf(self.bm25_search(query,50),self.vector_search(query,50))[:k]
    def filtered_search(self, query, department=None, year=None, k=5):
        candidates=self.hybrid_search(query, max(k, len(self.chunks)))
        return [c for c in candidates if (department is None or c['department']==department) and (year is None or c['year']==year)][:k]

if __name__ == '__main__':
    r=HybridRetriever()
    q='How many casual leaves can an employee take?'
    for name, results in [('BM25',r.bm25_search(q,3)),('Vector',r.vector_search(q,3)),('Hybrid',r.hybrid_search(q,3))]:
        print(name, [(x['id'],x['text']) for x in results])
