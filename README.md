# research-ingestion

**Topics & Documentation Agent + Research Ingestion Engine**

Fetches papers from arXiv by domain/keyword, enriches them with Claude-generated summaries, keywords, and cross-domain applications.

## Components
- `topics_agent.py` — Claude-powered agent that suggests search queries for a given Kaggle problem
- `arxiv_client.py` — Async arXiv API client
- `paper_processor.py` — Enriches raw papers via Claude Code CLI

## Usage
```bash
pip install -e .
```

Ingestion is driven by `TopicsAgent.run_ingestion_job(job)`.

## Part of
[joseph-isaac-kaggle](https://github.com/joseph-isaac-kaggle) — multi-agent Kaggle RAG system
