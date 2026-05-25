"""Async client wrapping the arxiv Python package."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import arxiv
from tenacity import retry, stop_after_attempt, wait_fixed

from src.models import Paper


class ArxivClient:
    """Async wrapper around the arxiv package for searching and fetching papers."""

    def __init__(self) -> None:
        self._client = arxiv.Client()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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
        """Search arXiv and return a list of Paper objects.

        Args:
            query: Free-text query string.
            categories: arXiv category codes to filter by (e.g. ["cs.LG", "stat.ML"]).
            max_results: Maximum number of results to return.

        Returns:
            List of Paper objects with basic metadata populated.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query, categories, max_results)

    async def fetch_by_id(self, arxiv_id: str) -> Paper | None:
        """Fetch a single paper by its arXiv ID.

        Args:
            arxiv_id: The arXiv paper ID (e.g. "2301.07041").

        Returns:
            A Paper object if found, otherwise None.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_by_id_sync, arxiv_id)
