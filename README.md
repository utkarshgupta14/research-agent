# ResearchPilot (MVP 0)

ResearchPilot is an agentic research assistant designed to answer technical questions over a curated collection of research papers.

## Project Architecture

- **Ingestion**: PDF parsing via PyMuPDF, document section detection, and metadata extraction.
- **Retrieval**: Separate paper-level and evidence-level FAISS vector indexes with `Retriever` abstraction interface.
- **Agent**: LangGraph stateful agent orchestrating `search_papers`, `retrieve_evidence`, and `get_paper` tools.
- **Interface**: CLI (`scripts/query.py`) and Streamlit web UI (`frontend/app.py`).

## Quick Start

### 1. Environment Setup
```bash
conda activate research-pilot
pip install -e .
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API key:
```bash
cp .env.example .env
```

### 3. Ingestion & Search
Add your PDF papers to `data/papers/`, then run:
```bash
python scripts/ingest.py
```

Query via CLI:
```bash
python scripts/query.py "What are the key approaches to domain generalization?"
```

Run Streamlit app:
```bash
streamlit run frontend/app.py
```
