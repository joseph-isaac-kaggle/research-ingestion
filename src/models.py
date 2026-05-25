"""Shared data models across all components."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    domain: str = ""
    what_it_solves: str = ""
    cross_domain_applications: str = ""
    published: datetime | None = None
    url: str = ""
    embedding: list[float] = Field(default_factory=list, exclude=True)


class SearchResult(BaseModel):
    paper: Paper
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    relevance_explanation: str = ""


class UserQuery(BaseModel):
    raw_query: str
    domain: str = ""
    reformulated_queries: list[str] = Field(default_factory=list)
    query_type: str = "general"  # brainstorming | debugging | statistics | general


class RecommendationResult(BaseModel):
    query: UserQuery
    results: list[SearchResult]
    strategic_recommendation: str = ""
    suggested_methods: list[str] = Field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionJob(BaseModel):
    job_id: str
    domains: list[str]
    keywords: list[str]
    max_papers: int = 100
    status: str = "pending"  # pending | running | completed | failed
    papers_processed: int = 0
    error: str = ""
