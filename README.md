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
| Vector DB       | ChromaDB                   |
| Local Runner    | Ollama                     |
| Framework       | LangChain                  |
| UI              | Gradio                     |
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
Cleans text and splits into chunks. Saves to `data/processed/`.

### Step 3: Build vector store
```bash
python -m src.ingest
```
Embeds chunks with BGE-small and stores in ChromaDB at `data/vectorstore/`.

### Step 4: Query
```bash
python -m src.rag "What causes cerebral malaria?"
```

### Step 5: Launch UI
```bash
python -m src.app
```
Opens a Gradio interface at `http://localhost:7860`.

## Project Structure

```
cerebral_malaria_rag/
├── README.md
├── PLAN.md               # Detailed build plan and progress log
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/              # Downloaded documents (119 files)
│   ├── processed/        # Cleaned and chunked text
│   └── vectorstore/      # ChromaDB embeddings
├── src/
│   ├── collect.py        # Document collection
│   ├── process.py        # Text extraction and chunking
│   ├── ingest.py         # Embedding and vector store creation
│   ├── rag.py            # RAG query pipeline
│   └── app.py            # Gradio UI
├── eval/
│   ├── questions.json    # Test Q&A pairs
│   └── evaluate.py       # Evaluation script
└── deploy/
    ├── app.py            # HuggingFace Spaces entrypoint
    └── requirements.txt  # Deployment dependencies
```

## Data Sources

| Source | Count | Type |
|--------|-------|------|
| PubMed | 102 | Abstracts |
| PMC | 14 | Full-text open-access papers |
| CDC | 2 | Clinical guidance pages |
| WHO | 1 | Malaria fact sheet |
| **Total** | **119** | |

## Progress

- [x] Step 1: Project setup and dependencies
- [x] Step 2: Document collection (119 documents)
- [ ] Step 3: Text extraction and chunking
- [ ] Step 4: Embedding and vector store
- [ ] Step 5: Ollama + Qwen2.5-7B setup
- [ ] Step 6: RAG pipeline
- [ ] Step 7: Gradio UI
- [ ] Step 8: Evaluation
- [ ] Step 9: Iteration
- [ ] Step 10: HuggingFace Spaces deployment

## Deployment

Deployable to HuggingFace Spaces with free T4 GPU using 4-bit quantized Qwen2.5-7B via `bitsandbytes`.

## License

Research use only.
