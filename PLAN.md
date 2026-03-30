# Cerebral Malaria RAG System — Build Plan

## Goal
Build a question-answering system that answers cerebral malaria questions grounded in real medical literature. Local-first, then deployable to a remote GPU (HuggingFace Spaces) for global access.

---

## Architecture Overview

```
[Documents] → [Chunking] → [Embedding Model] → [Vector DB (ChromaDB)]
                                                        ↓
[User Question] → [Embedding] → [Similarity Search] → [Top-K Chunks]
                                                        ↓
                                              [LLM (Qwen2.5-7B)] → [Answer]
```

## Technology Choices

| Component        | Choice                        | Why                                                  |
|-----------------|-------------------------------|------------------------------------------------------|
| LLM             | Qwen2.5-7B-Instruct          | Strong multilingual, good at instruction following, fits in 16GB RAM with quantization |
| Embedding Model | BAAI/bge-small-en-v1.5        | Top-tier retrieval quality, small (130MB), fast       |
| Vector DB       | ChromaDB                      | Simple, file-based, no server needed                  |
| Local runner    | Ollama                        | One-command model serving, quantized models           |
| Framework       | LangChain                     | Mature RAG abstractions, easy swap between local/API  |
| Deployment      | HuggingFace Spaces (Gradio)   | Free GPU tier, public URL, simple UI                  |

---

## Phase 1: Document Collection

### Sources
1. **PubMed/PMC** — open-access cerebral malaria research papers
2. **WHO** — cerebral malaria treatment guidelines
3. **CDC** — cerebral malaria fact sheets and clinical guidance
4. **Textbook chapters** — open-access parasitology/tropical medicine content
5. **Review articles** — systematic reviews on cerebral malaria pathogenesis, diagnosis, treatment

### Method
- Use `pymed` (PubMed API wrapper) to search and download abstracts + full-text links
- Use `requests` + `BeautifulSoup` to scrape open-access HTML content
- Use `PyMuPDF` to extract text from downloaded PDFs
- Save all raw documents to `data/raw/`
- Save processed text to `data/processed/`

### Search Terms
- "cerebral malaria"
- "cerebral malaria pathogenesis"
- "cerebral malaria treatment"
- "cerebral malaria diagnosis"
- "cerebral malaria children"
- "plasmodium falciparum brain"
- "severe malaria neurological"

### Target
~100-200 documents (papers, guidelines, reviews). Quality over quantity.

---

## Phase 2: Document Processing

1. **Extract text** from PDFs and HTML into plain text files
2. **Clean text** — remove references sections, figure captions, boilerplate
3. **Chunk documents** using RecursiveCharacterTextSplitter
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   - Preserve metadata (source, title, year)
4. **Generate embeddings** using `bge-small-en-v1.5`
5. **Store in ChromaDB** with metadata for filtering

### Output
- `data/processed/` — cleaned text files
- `data/vectorstore/` — ChromaDB persistent directory

---

## Phase 3: RAG Pipeline (Local)

1. **Install Ollama** and pull `qwen2.5:7b`
2. **Build retrieval chain:**
   - User question → embed with BGE → search ChromaDB → top 5 chunks
   - Construct prompt: system instruction + retrieved context + question
   - Send to Qwen2.5-7B via Ollama
   - Return grounded answer with source citations
3. **Test with sample questions:**
   - "What causes cerebral malaria?"
   - "How is cerebral malaria diagnosed?"
   - "What is the mortality rate of cerebral malaria in children?"
   - "What are the neurological sequelae of cerebral malaria?"

### Files
- `src/collect.py` — document collection scripts
- `src/process.py` — text extraction and chunking
- `src/ingest.py` — embedding and vector store creation
- `src/rag.py` — RAG query pipeline
- `src/app.py` — Gradio UI for local testing

---

## Phase 4: Evaluation

1. Create 20-30 Q&A pairs manually from known sources
2. Run each question through the pipeline
3. Score on:
   - **Relevance** — does the answer address the question?
   - **Groundedness** — is the answer supported by retrieved chunks?
   - **Completeness** — does it cover key points?
4. Iterate on chunk size, top-K, prompt template as needed

### Files
- `eval/questions.json` — test Q&A pairs
- `eval/evaluate.py` — evaluation script

---

## Phase 5: Remote Deployment (HuggingFace Spaces)

### Option A: HuggingFace Spaces with free GPU
1. Create a HuggingFace Space with Gradio SDK
2. Upload vector store (ChromaDB files) to the Space
3. Use `transformers` + `bitsandbytes` to load Qwen2.5-7B in 4-bit quantization
4. Run the same RAG pipeline on the Space's T4 GPU (free tier)

### Option B: HuggingFace Inference Endpoints
1. Deploy Qwen2.5-7B as an Inference Endpoint
2. Space calls the endpoint instead of running the model locally
3. More reliable but may cost after free credits

### Deployment Files
- `app.py` — Gradio app (entrypoint for HF Spaces)
- `requirements.txt` — Python dependencies
- `README.md` — HuggingFace Space card

---

## Phase 6: Stretch Goals (Optional)
- Add chat memory (multi-turn conversations)
- Add source highlighting in the UI
- Add document upload (users add their own papers)
- Fine-tune Qwen on cerebral malaria Q&A pairs (Option 2 from earlier)

---

## Directory Structure

```
cerebral_malaria_rag/
├── PLAN.md                  # This file
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── data/
│   ├── raw/                 # Original downloaded documents
│   └── processed/           # Cleaned text files
│   └── vectorstore/         # ChromaDB files
├── src/
│   ├── collect.py           # Document collection from PubMed, WHO, CDC
│   ├── process.py           # Text extraction and chunking
│   ├── ingest.py            # Embedding + ChromaDB ingestion
│   ├── rag.py               # RAG query pipeline
│   └── app.py               # Gradio UI
├── eval/
│   ├── questions.json       # Test Q&A pairs
│   └── evaluate.py          # Evaluation script
└── deploy/
    ├── app.py               # HuggingFace Spaces entrypoint
    ├── requirements.txt     # Deployment dependencies
    └── README.md            # HF Space card
```

---

## Execution Order

| Step | Phase | What to Do | Done |
|------|-------|-----------|------|
| 1 | Setup | Install dependencies, create project structure | [x] |
| 2 | Phase 1 | Collect documents from PubMed, WHO, CDC | [x] |
| 3 | Phase 2 | Extract, clean, and chunk documents | [ ] |
| 4 | Phase 2 | Generate embeddings and store in ChromaDB | [ ] |
| 5 | Phase 3 | Install Ollama + pull Qwen2.5-7B | [ ] |
| 6 | Phase 3 | Build RAG pipeline and test locally | [ ] |
| 7 | Phase 3 | Build Gradio UI for local use | [ ] |
| 8 | Phase 4 | Create test Q&A pairs and evaluate | [ ] |
| 9 | Phase 4 | Iterate and improve retrieval/generation | [ ] |
| 10 | Phase 5 | Deploy to HuggingFace Spaces | [ ] |

---

## Progress Log

### Step 1 — Setup (2026-03-30)
- Created project structure: `data/{raw,processed,vectorstore}`, `src/`, `eval/`, `deploy/`
- Python 3.14 venv created at `.venv/`
- Installed all dependencies: langchain, chromadb, gradio, sentence-transformers, pymed, beautifulsoup4, PyMuPDF, langchain-ollama, tqdm
- Created `.env`, `.env.example`, `.gitignore`, `requirements.txt`
- Note: `lxml` not available for Python 3.14 — used `xml.etree.ElementTree` for XML parsing instead

### Step 2 — Document Collection (2026-03-30)
- Collected **119 documents** total:
  - PubMed abstracts: 102 (across 5 search queries, 115 unique PMIDs, 102 had abstracts)
  - PMC full-text open-access articles: 14 (ranging from 7k-42k chars each)
  - CDC clinical pages: 2 (clinical guidance + diagnosis/testing)
  - WHO fact sheet: 1
- Search queries used: cerebral malaria pathogenesis, treatment, diagnosis, children mortality, neurological sequelae
- Issues encountered:
  - NCBI E-utilities had intermittent SSL timeouts — resolved with retry logic (5 retries, exponential backoff)
  - WHO guidelines PDF URL returned 404 (stale link) — got fact sheet instead
  - 2 CDC URLs returned 404 (site restructured) — got 2 valid pages
  - lxml unavailable on Python 3.14 — switched to stdlib `xml.etree.ElementTree`
- All documents saved to `data/raw/` with `metadata.json` index
- Collection script: `src/collect.py` (uses requests session with retry adapter)
