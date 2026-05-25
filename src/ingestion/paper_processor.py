"""Enriches Paper objects using Claude Code CLI."""
from __future__ import annotations

import asyncio
import json
import logging

from src.claude_code import call_claude
from src.models import Paper

logger = logging.getLogger(__name__)

_ENRICHMENT_SYSTEM = "You are a research analyst helping data scientists discover relevant ML papers."

_ENRICHMENT_PROMPT = """\
Given the paper title and abstract below, return a JSON object with exactly these keys:
- "summary": A 2-3 sentence plain-language summary of the paper.
- "keywords": A list of 5-8 lowercase keyword strings (techniques, concepts, datasets).
- "what_it_solves": One sentence describing the core problem this paper solves.
- "cross_domain_applications": One sentence explaining how this paper's ideas could help in a Kaggle competition.

Respond with ONLY the raw JSON object — no markdown fences, no explanation.

Title: {title}

Abstract:
{abstract}
"""


class PaperProcessor:
    """Enriches raw Paper objects with summaries and keywords via Claude Code CLI."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(5)

    async def process_paper(self, paper: Paper) -> Paper:
        prompt = _ENRICHMENT_PROMPT.format(title=paper.title, abstract=paper.abstract)
        try:
            raw_text = await call_claude(prompt, system=_ENRICHMENT_SYSTEM)

            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                raw_text = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            data: dict = json.loads(raw_text)
            paper = paper.model_copy(
                update={
                    "summary": str(data.get("summary", "")),
                    "keywords": [str(k) for k in data.get("keywords", [])],
                    "what_it_solves": str(data.get("what_it_solves", "")),
                    "cross_domain_applications": str(data.get("cross_domain_applications", "")),
                }
            )
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON enrichment for paper %s: %s", paper.arxiv_id, exc)
        except Exception as exc:
            logger.error("Error enriching paper %s: %s", paper.arxiv_id, exc)

        return paper

    async def process_batch(self, papers: list[Paper]) -> list[Paper]:
        async def _bounded(paper: Paper) -> Paper:
            async with self._semaphore:
                return await self.process_paper(paper)

        return list(await asyncio.gather(*(_bounded(p) for p in papers)))
