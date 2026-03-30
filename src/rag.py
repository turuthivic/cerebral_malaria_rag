"""
Phase 3: RAG query pipeline — retrieve relevant chunks and generate answers.

Usage:
    python -m src.rag "What causes cerebral malaria?"

Uses:
    - ChromaDB vector store (from src.ingest)
    - BGE-small embedding model (for query embedding)
    - Qwen2.5-7B via Ollama (for answer generation)
"""

import sys
from pathlib import Path

import chromadb
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

VECTORSTORE_DIR = Path("data/vectorstore")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "qwen2.5:7b"
COLLECTION_NAME = "cerebral_malaria"
TOP_K = 5

SYSTEM_PROMPT = """You are a medical expert specializing in cerebral malaria. Answer the question based ONLY on the provided context. If the context does not contain enough information to answer, say so clearly.

Rules:
- Be accurate and cite which source supports each claim
- Use clear, professional medical language
- If multiple sources agree, synthesize them into a coherent answer
- Do not make claims beyond what the context supports"""

PROMPT_TEMPLATE = """{system}

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("Connecting to vector store...")
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"  {self.collection.count()} vectors loaded")

        print(f"Connecting to Ollama ({LLM_MODEL})...")
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Find the most relevant chunks for a query."""
        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        chunks = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "year": meta.get("year", ""),
                "similarity": 1 - distance,  # cosine distance → similarity
            })
        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[{chunk['source'].upper()}]"
            if chunk["title"]:
                source_info += f" {chunk['title'][:80]}"
            if chunk["year"]:
                source_info += f" ({chunk['year']})"
            parts.append(f"Source {i} {source_info}:\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        """Full RAG pipeline: retrieve → format → generate."""
        chunks = self.retrieve(question, top_k)

        context = self.format_context(chunks)
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
        )

        answer = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "title": c["title"][:100],
                    "source": c["source"],
                    "year": c["year"],
                    "similarity": round(c["similarity"], 3),
                }
                for c in chunks
            ],
        }


def main():
    if len(sys.argv) < 2:
        question = "What causes cerebral malaria?"
    else:
        question = " ".join(sys.argv[1:])

    rag = RAGPipeline()

    print(f"\nQuestion: {question}")
    print("-" * 60)

    result = rag.query(question)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n{'=' * 60}")
    print("Sources:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  {i}. [{src['source']}] {src['title']} ({src['year']}) — similarity: {src['similarity']}")


if __name__ == "__main__":
    main()
