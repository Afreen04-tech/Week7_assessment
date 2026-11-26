import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "vector_store"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_TEXT = "I don't have information on that."
SCORE_THRESHOLD = 0.28

load_dotenv(BASE_DIR / ".env")


def load_store():
    chunks_path = STORE_DIR / "chunks.json"
    embeddings_path = STORE_DIR / "embeddings.npy"

    if not chunks_path.exists() or not embeddings_path.exists():
        raise RuntimeError("Vector store not found. Run: python index.py")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    return chunks, embeddings


def fit_context(chunks, budget):
    """Fit retrieved chunks into an approximate word-based context budget."""
    selected = []
    used_words = 0

    for chunk in chunks:
        text = chunk["text"]
        word_count = len(text.split())

        if used_words + word_count <= budget:
            selected.append(chunk)
            used_words += word_count
        elif not selected and budget > 0:
            # Keep a shortened first chunk if it alone exceeds the budget.
            shortened = dict(chunk)
            shortened["text"] = " ".join(text.split()[:budget])
            selected.append(shortened)
            break
        else:
            break

    return selected


class PolicyRAG:
    def __init__(self):
        self.chunks, self.embeddings = load_store()
        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, question, k=4):
        query_embedding = self.model.encode(
            [question], normalize_embeddings=True
        )[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:k]

        results = []
        for index in ranked_indices:
            item = dict(self.chunks[int(index)])
            item["score"] = float(scores[index])
            results.append(item)

        return results

    def answer(self, question, k=4, context_budget=500):
        retrieved = self.retrieve(question, k=k)

        if not retrieved or retrieved[0]["score"] < SCORE_THRESHOLD:
            return {
                "answer": FALLBACK_TEXT,
                "sources": [],
                "retrieved_count": len(retrieved),
                "top_score": retrieved[0]["score"] if retrieved else 0.0
            }

        context_chunks = fit_context(retrieved, context_budget)
        context = "\\n\\n".join(
            f'[Source: {item["metadata"]["source"]}, Page {item["metadata"]["page"]}] '
            f'{item["text"]}'
            for item in context_chunks
        )

        prompt = f"""Answer ONLY using the context below.
If the answer is not present in the context, say exactly:
{FALLBACK_TEXT}

Context:
{context}

Question: {question}

Answer:"""

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a company policy Q&A assistant. Answer only from supplied context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            answer_text = response.choices[0].message.content.strip()
        else:
            # Offline demo mode: return the retrieved policy text without an LLM.
            # Set OPENAI_API_KEY in .env to enable LLM generation.
            answer_text = context_chunks[0]["text"]

        sources = [
            {
                "source": item["metadata"]["source"],
                "page": item["metadata"]["page"],
                "score": round(item["score"], 4)
            }
            for item in context_chunks
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_count": len(retrieved),
            "top_score": round(retrieved[0]["score"], 4)
        }


if __name__ == "__main__":
    rag = PolicyRAG()
    questions = [
        "How many casual leaves per year?",
        "How many sick leaves are allowed?",
        "What are the standard working hours?",
        "How many public holidays are there?",
        "What is the daily meal expense limit during approved business travel?",
        "How long is the probation period?",
        "Who won the world cup?",
        "What is the company's share price?",
        "Who is the CEO?",
        "What is the office address?"
    ]

    for question in questions:
        result = rag.answer(question, k=4)
        print("\\nQ:", question)
        print("Answer:", result["answer"])
        print("Sources:", result["sources"])
        print("Top score:", result["top_score"])

    print("\\nContext budget tests")
    for k in [2, 5, 10]:
        retrieved = rag.retrieve("What is the leave policy?", k=k)
        fitted = fit_context(retrieved, budget=100)
        total_words = sum(len(item["text"].split()) for item in fitted)
        print(f"k={k}: retrieved={len(retrieved)}, context_words={total_words}, budget=100")
