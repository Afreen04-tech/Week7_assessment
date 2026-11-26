# Task 1 – RAG Pipeline with Fallback

## 1. Open terminal in this folder

```powershell
cd path\\to\\week7_assessment\\task1
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Build the index

```powershell
python index.py
```

Run it a second time. It should report that there are no new chunks to add.

## 3. Run the question-answering pipeline

```powershell
python rag.py
```

The project includes 8 sample company policy documents in `documents/`.

## 4. Optional: enable OpenAI generation

Copy `.env.example` to `.env`, add your own API key, and run `python rag.py` again.
Without an API key, the script runs in offline demo mode and returns retrieved context text;
that mode demonstrates retrieval and fallback but is not an LLM-generated answer.

## Notes
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2`.
- Similarity is cosine similarity because vectors are normalized.
- `fit_context` uses an approximate word budget, not an exact tokenizer token count.
- The score threshold is a starting value and should be calibrated using your own test set.
