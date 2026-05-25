"""Async client wrapping the arxiv Python package."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import arxiv
from tenacity import retry, stop_after_attempt, wait_fixed

from src.models import Paper

# arXiv ToS: be polite — max 1 concurrent request, 3 s between calls.
# https://info.arxiv.org/help/api/tou.html
_ARXIV_DELAY_SECONDS = 3.0
_ARXIV_MAX_CONCURRENT = 1


class ArxivClient:
    """Async wrapper around the arxiv package for searching and fetching papers.

    Enforces arXiv's rate-limit policy: one in-flight request at a time with
    a 3-second cooldown between calls, applied globally via a module-level
    semaphore so multiple ArxivClient instances still cooperate.
    """

    # Shared across all instances so concurrent callers don't double-fire.
    _semaphore: asyncio.Semaphore | None = None

    def __init__(self) -> None:
        self._client = arxiv.Client(delay_seconds=_ARXIV_DELAY_SECONDS, num_retries=3)

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(_ARXIV_MAX_CONCURRENT)
        return cls._semaphore

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def _search_sync(self, query: str, categories: list[str], max_results: int) -> list[Paper]:
        """Synchronous search executed in a thread pool."""
        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            full_query = f"({query}) AND ({cat_filter})"
        else:
            full_query = query

        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[Paper] = []
        for result in self._client.results(search):
            papers.append(self._map_result(result))
        return papers

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_by_id_sync(self, arxiv_id: str) -> Paper | None:
        """Synchronous fetch by ID executed in a thread pool."""
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(self._client.results(search))
        if not results:
            return None
        return self._map_result(results[0])

    def _map_result(self, result: arxiv.Result) -> Paper:
        """Map an arxiv.Result to a Paper model."""
        published: datetime | None = None
        if result.published is not None:
            # arxiv returns timezone-aware datetimes; keep as-is
            published = result.published

        return Paper(
            arxiv_id=result.entry_id.split("/abs/")[-1],
            title=result.title,
            authors=[str(a) for a in result.authors],
            abstract=result.summary,
            published=published,
            url=result.entry_id,
            # Downstream processors fill these fields
            summary="",
            keywords=[],
            domain="",
            what_it_solves="",
            cross_domain_applications="",
        )

    async def search(self, query: str, categories: list[str], max_results: int) -> list[Paper]:
        """Search arXiv, serialised through the global semaphore + cooldown."""
        loop = asyncio.get_running_loop()
        async with self._get_semaphore():
            result = await loop.run_in_executor(
                None, self._search_sync, query, categories, max_results
            )
            await asyncio.sleep(_ARXIV_DELAY_SECONDS)
        return result

    async def fetch_by_id(self, arxiv_id: str) -> Paper | None:
        """Fetch a single paper by arXiv ID, serialised through the global semaphore."""
        loop = asyncio.get_running_loop()
        async with self._get_semaphore():
            result = await loop.run_in_executor(None, self._fetch_by_id_sync, arxiv_id)
            await asyncio.sleep(_ARXIV_DELAY_SECONDS)
        return result
