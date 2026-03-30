"""
Phase 2: Extract, clean, and chunk documents for RAG ingestion.

Usage:
    python -m src.process

Input:
    data/raw/*.txt — raw documents from collection phase

Output:
    data/processed/chunks.json — chunked documents with metadata
"""

import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_raw_documents() -> list[dict]:
    """Load all raw text documents and their metadata."""
    metadata_path = RAW_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("No metadata.json found. Run src.collect first.")

    metadata = json.loads(metadata_path.read_text())
    meta_by_filename = {m["filename"]: m for m in metadata}

    documents = []
    for txt_file in sorted(RAW_DIR.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8")
        meta = meta_by_filename.get(txt_file.name, {})

        # Split header from body
        parts = content.split("\n---\n", 1)
        body = parts[1].strip() if len(parts) > 1 else content.strip()

        if len(body) < 50:
            continue

        documents.append({
            "text": body,
            "metadata": {
                "source": meta.get("source", "unknown"),
                "title": meta.get("title", txt_file.stem),
                "year": meta.get("year", ""),
                "journal": meta.get("journal", ""),
                "pmid": meta.get("pmid", ""),
                "pmcid": meta.get("pmcid", ""),
                "url": meta.get("url", ""),
                "filename": txt_file.name,
            },
        })

    return documents


def clean_text(text: str) -> str:
    """Clean document text for better chunking and retrieval."""
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common boilerplate patterns
    text = re.sub(r"Copyright ©.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"All rights reserved\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", "", text)  # URLs add noise for retrieval

    # Remove figure/table references that are just noise without the actual figures
    text = re.sub(r"\(Fig(?:ure)?\.?\s*\d+[A-Za-z]?\)", "", text)
    text = re.sub(r"\(Table\s*\d+\)", "", text)

    return text.strip()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        cleaned = clean_text(doc["text"])
        chunks = splitter.split_text(cleaned)

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue

            all_chunks.append({
                "text": chunk.strip(),
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                },
            })

    return all_chunks


def main():
    print("=" * 60)
    print("Document Processing & Chunking")
    print("=" * 60)

    print("\n[1/3] Loading raw documents...")
    documents = load_raw_documents()
    print(f"  Loaded {len(documents)} documents")

    total_chars = sum(len(d["text"]) for d in documents)
    print(f"  Total text: {total_chars:,} characters")

    print("\n[2/3] Cleaning and chunking...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Avg chunk size: {sum(len(c['text']) for c in chunks) // len(chunks)} chars")

    print("\n[3/3] Saving chunks...")
    output_path = PROCESSED_DIR / "chunks.json"
    output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved to {output_path}")

    # Summary by source
    sources = {}
    for c in chunks:
        s = c["metadata"]["source"]
        sources[s] = sources.get(s, 0) + 1
    print("\n  Chunks by source:")
    for s, count in sorted(sources.items()):
        print(f"    {s}: {count}")

    print(f"\n{'=' * 60}")
    print(f"Done. {len(chunks)} chunks ready for embedding.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
