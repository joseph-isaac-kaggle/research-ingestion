"""Claude Code-powered Topics & Documentation Agent for driving ingestion jobs."""
from __future__ import annotations

import asyncio
import json
import logging

from src.claude_code import call_claude
from src.models import IngestionJob, Paper
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.paper_processor import PaperProcessor

logger = logging.getLogger(__name__)

_SUGGEST_TOPICS_SYSTEM = "You are an expert ML research assistant specializing in Kaggle competitions."

_SUGGEST_TOPICS_PROMPT = """\
A data scientist is working on the following problem:
{problem_description}

Generate a list of 6-10 arXiv search queries that would surface the most relevant academic papers for this problem. Focus on:
- Core ML/statistical techniques that apply directly
- Feature engineering and preprocessing methods
- Model architectures or ensembling strategies
- Any domain-specific methods (e.g. time-series, NLP, computer vision) relevant to this task

Return ONLY a JSON array of query strings — no explanation, no markdown fences.
Example format: ["gradient boosting tabular data", "feature selection neural networks"]
"""


class TopicsAgent:
    """Agent that orchestrates topic discovery and paper ingestion jobs."""

    def __init__(self, arxiv_client: ArxivClient, processor: PaperProcessor) -> None:
        self._arxiv = arxiv_client
        self._processor = processor

    async def suggest_topics_for_kaggle(self, problem_description: str) -> list[str]:
        prompt = _SUGGEST_TOPICS_PROMPT.format(problem_description=problem_description)
        try:
            raw_text = await call_claude(prompt, system=_SUGGEST_TOPICS_SYSTEM)

            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                raw_text = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            queries: list[str] = json.loads(raw_text)
            return [str(q) for q in queries]
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse topic suggestions JSON: %s", exc)
            return []
        except Exception as exc:
            logger.error("Error during topic suggestion: %s", exc)
            return []

    async def run_ingestion_job(self, job: IngestionJob) -> list[Paper]:
        job.status = "running"

        async def _search_and_enrich(keyword: str, domain: str) -> list[Paper]:
            try:
                max_per_combo = max(1, job.max_papers // max(1, len(job.keywords) * len(job.domains)))
                raw_papers = await self._arxiv.search(
                    query=keyword,
                    categories=[domain] if domain else [],
                    max_results=max_per_combo,
                )
                raw_papers = [p.model_copy(update={"domain": domain}) for p in raw_papers]
                return await self._processor.process_batch(raw_papers)
            except Exception as exc:
                logger.error("Error in search/enrich for keyword=%r domain=%r: %s", keyword, domain, exc)
                return []

        tasks = [
            _search_and_enrich(keyword, domain)
            for keyword in job.keywords
            for domain in (job.domains if job.domains else [""])
        ]

        batch_results: list[list[Paper]] = await asyncio.gather(*tasks)

        all_papers: list[Paper] = []
        seen_ids: set[str] = set()
        for batch in batch_results:
            for paper in batch:
                if paper.arxiv_id not in seen_ids:
                    seen_ids.add(paper.arxiv_id)
                    all_papers.append(paper)

        job.papers_processed = len(all_papers)
        job.status = "completed"
        logger.info("Ingestion job %s completed: %d unique papers.", job.job_id, job.papers_processed)
        return all_papers
