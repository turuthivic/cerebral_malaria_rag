# Cerebral Malaria RAG

A question-answering system for cerebral malaria, grounded in real medical literature using Retrieval-Augmented Generation (RAG).

## Architecture

```
[Documents] → [Chunking] → [Embedding Model] → [Vector DB (ChromaDB)]
                                                        ↓
[User Question] → [Embedding] → [Similarity Search] → [Top-K Chunks]
                                                        ↓
                                              [LLM (Qwen2.5-7B)] → [Answer]
```

## Stack

| Component        | Choice                    |
|-----------------|---------------------------|
| LLM             | Qwen2.5-7B-Instruct       |
| Embedding Model | BAAI/bge-small-en-v1.5     |
| Vector DB       | ChromaDB 1.5.5             |
| Local Runner    | Ollama                     |
| Framework       | LangChain                  |
| UI              | Gradio 6.10                |
| Deployment      | HuggingFace Spaces         |

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Install Ollama (macOS)
brew install ollama
ollama serve          # start server
ollama pull qwen2.5:7b  # download model
```

## Usage

### Step 1: Collect documents
```bash
python -m src.collect
```
Fetches cerebral malaria literature from PubMed, PMC, WHO, and CDC. Saves to `data/raw/`.

### Step 2: Process and chunk documents
```bash
python -m src.process
```
Cleans text and splits into 1000-char chunks with 200-char overlap. Saves to `data/processed/`.

### Step 3: Build vector store
```bash
python -m src.ingest
```
Embeds chunks with BGE-small (384 dimensions) and stores in ChromaDB at `data/vectorstore/`.

### Step 4: Query (CLI)
```bash
python -m src.rag "What causes cerebral malaria?"
```

### Step 5: Launch UI
```bash
python -m src.app
```
Opens a Gradio interface at `http://localhost:7860`.

### Evaluate
```bash
python -m eval.evaluate
```
Runs 20 test questions and scores relevance, groundedness, and topic coverage.

## Project Structure

```
cerebral_malaria_rag/
├── README.md
├── PLAN.md               # Detailed build plan and progress log
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/              # Downloaded documents (175 files)
│   ├── processed/        # Cleaned and chunked text (730 chunks)
│   └── vectorstore/      # ChromaDB embeddings (730 vectors)
├── src/
│   ├── collect.py        # Document collection from PubMed, PMC, WHO, CDC
│   ├── process.py        # Text extraction and chunking
│   ├── ingest.py         # Embedding + ChromaDB ingestion
│   ├── rag.py            # RAG query pipeline
│   └── app.py            # Gradio UI
├── eval/
│   ├── questions.json    # 20 test Q&A pairs
│   └── evaluate.py       # Automated evaluation script
└── deploy/
    ├── app.py            # HuggingFace Spaces entrypoint
    └── requirements.txt  # Deployment dependencies
```

## Data Sources

| Source | Count | Type |
|--------|-------|------|
| PubMed | 158 | Abstracts |
| PMC | 14 | Full-text open-access papers |
| CDC | 2 | Clinical guidance pages |
| WHO | 1 | Malaria fact sheet |
| **Total** | **175** | |

## Evaluation Results

Tested against 20 questions across 6 categories (pathogenesis, diagnosis, treatment, epidemiology, sequelae, definition):

| Metric | Score |
|--------|-------|
| Relevance | 70% |
| Groundedness | 87% |
| Topic Coverage | 65% |
| Avg Response Time | 13.2s |

Key behaviors:
- Off-topic questions are rejected (similarity threshold filtering)
- Answers cite sources by number
- Strongest on pathogenesis (75% coverage) and treatment (70% coverage)

## Progress

- [x] Step 1: Project setup and dependencies
- [x] Step 2: Document collection (175 documents)
- [x] Step 3: Text extraction and chunking (730 chunks)
- [x] Step 4: Embedding and vector store (730 vectors, 384 dims)
- [x] Step 5: Ollama + Qwen2.5-7B setup
- [x] Step 6: RAG pipeline with source citations
- [x] Step 7: Gradio web UI
- [x] Step 8: Evaluation (20 Q&A pairs, automated scoring)
- [x] Step 9: Iteration (threshold filtering, expanded corpus, improved prompt)
- [ ] Step 10: HuggingFace Spaces deployment

## Deployment

Deployable to HuggingFace Spaces with free T4 GPU using 4-bit quantized Qwen2.5-7B via `bitsandbytes`.

## License

Research use only.
