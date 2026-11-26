"""Simple query rewriting baseline; merges three rewrites with RRF."""
from retrieval import HybridRetriever

def rewrite_query(question):
    return [question, f"company policy about {question}", f"employee rules: {question}", f"What does the policy state regarding {question}"]

def rewritten_search(retriever, question, top_k=5):
    ranked=[retriever.hybrid_search(q,50) for q in rewrite_query(question)]
    return retriever.rrf(*ranked)[:top_k]

if __name__ == '__main__':
    r=HybridRetriever(); q='How many casual leaves per year?'
    print([x['id'] for x in rewritten_search(r,q)])
