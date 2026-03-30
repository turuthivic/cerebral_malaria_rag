"""
Phase 2: Embed document chunks and store in ChromaDB.

Usage:
    python -m src.ingest

Input:
    data/processed/chunks.json — chunked documents from processing phase

Output:
    data/vectorstore/ — ChromaDB persistent directory
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = Path("data/processed/chunks.json")
VECTORSTORE_DIR = Path("data/vectorstore")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "cerebral_malaria"
BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    """Load processed chunks."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("No chunks.json found. Run src.process first.")
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def main():
    print("=" * 60)
    print("Embedding & ChromaDB Ingestion")
    print("=" * 60)

    # Load chunks
    print("\n[1/3] Loading chunks...")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks to embed")

    # Load embedding model
    print(f"\n[2/3] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Create ChromaDB collection
    print(f"\n[3/3] Embedding and storing in ChromaDB...")
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Process in batches
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="  Batches"):
        batch = chunks[i : i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [c["metadata"] for c in batch]

        # BGE models benefit from "Represent this sentence:" prefix for documents
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"\n  Collection '{COLLECTION_NAME}': {collection.count()} vectors")
    print(f"  Stored at: {VECTORSTORE_DIR}/")

    # Quick sanity check
    print("\n  Sanity check — querying: 'cerebral malaria diagnosis'")
    query_embedding = model.encode(
        ["cerebral malaria diagnosis"], normalize_embeddings=True
    ).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    for j, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"\n  Result {j + 1} [{meta['source']}]: {meta['title'][:70]}")
        print(f"    {doc[:150]}...")

    print(f"\n{'=' * 60}")
    print("Done. Vector store ready for RAG queries.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
