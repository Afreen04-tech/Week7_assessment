import json
import re
import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"
STORE_DIR = BASE_DIR / "vector_store"
STORE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_chunks(text, filename, chunk_size=450):
    """Split a document into small overlapping word chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - 60)

    for start in range(0, len(words), step):
        part = words[start:start + chunk_size]
        if not part:
            continue

        chunk_text = " ".join(part)
        stable_id = hashlib.sha256(
            f"{filename}:{start}:{chunk_text}".encode("utf-8")
        ).hexdigest()[:16]

        chunks.append({
            "id": stable_id,
            "text": chunk_text,
            "metadata": {
                "source": filename,
                "page": 1,
                "chunk_start": start
            }
        })

    return chunks


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    files = sorted(DOCS_DIR.glob("*.txt"))
    print(f"Loading documents... {len(files)} found")

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        print(f"Parsing: {file_path.name}")
        all_chunks.extend(make_chunks(text, file_path.name))

    print(f"Total chunks created: {len(all_chunks)}")

    existing_file = STORE_DIR / "chunks.json"
    existing_ids = set()

    if existing_file.exists():
        old_chunks = json.loads(existing_file.read_text(encoding="utf-8"))
        existing_ids = {item["id"] for item in old_chunks}
    else:
        old_chunks = []

    new_chunks = [c for c in all_chunks if c["id"] not in existing_ids]

    if new_chunks:
        print("Generating embeddings for new chunks...")
        new_embeddings = model.encode(
            [c["text"] for c in new_chunks],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        old_embeddings_path = STORE_DIR / "embeddings.npy"

        if old_embeddings_path.exists() and old_chunks:
            old_embeddings = np.load(old_embeddings_path)
            combined_embeddings = np.vstack([old_embeddings, new_embeddings])
        else:
            combined_embeddings = np.asarray(new_embeddings)

        combined_chunks = old_chunks + new_chunks
        np.save(old_embeddings_path, combined_embeddings)
        existing_file.write_text(
            json.dumps(combined_chunks, indent=2),
            encoding="utf-8"
        )
    else:
        combined_chunks = old_chunks
        print("No new chunks to add.")

    # Remove deleted/changed source chunks by rebuilding when IDs differ.
    current_ids = {c["id"] for c in all_chunks}
    stored_ids = {c["id"] for c in combined_chunks}
    if current_ids != stored_ids:
        print("Document set changed. Rebuilding the vector store...")
        embeddings = model.encode(
            [c["text"] for c in all_chunks],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        np.save(STORE_DIR / "embeddings.npy", np.asarray(embeddings))
        existing_file.write_text(
            json.dumps(all_chunks, indent=2),
            encoding="utf-8"
        )
        combined_chunks = all_chunks

    print(f"Documents stored: {len(combined_chunks)}")
    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()
