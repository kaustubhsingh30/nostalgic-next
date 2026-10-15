"""
Ranking and Top-N Selection Module for NostalgicNext.

Responsibilities:
- Sorting scored movie candidates in descending order of relevance.
- Deterministic tie-breaking using rating quality and alphabetical order.
- Enforcing top-N result limits.
- Converting results into user-facing formats (DataFrames, JSON serializable lists).
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from .scoring import ScoredMovie

logger = logging.getLogger(__name__)


class RecommendationRanker:
    """Sorts, breaks ties, and selects the top-N recommendations."""

    def __init__(self, secondary_tie_breaker: str = "vote_average"):
        self.secondary_tie_breaker = secondary_tie_breaker

    def rank(self, candidates: List[ScoredMovie], top_n: int = 5) -> List[ScoredMovie]:
        """
        Sort candidate movies deterministically:
        1. similarity_score descending
        2. vote_average descending (quality signal)
        3. vote_count descending (popularity/confidence)
        4. title ascending (guaranteed deterministic stability)
        """
        if not candidates:
            return []

        # Sort with tuple key for deterministic ordering
        ranked = sorted(
            candidates,
            key=lambda c: (
                -c.similarity_score,
                -c.vote_average,
                -c.vote_count,
                c.title
            )
        )

        top_candidates = ranked[:top_n]
        return top_candidates

    def to_dataframe(self, ranked_movies: List[ScoredMovie]) -> pd.DataFrame:
        """Convert a list of ranked ScoredMovie objects into a clean pandas DataFrame."""
        if not ranked_movies:
            return pd.DataFrame(columns=[
                "movie_id", "title", "similarity_score", "release_year",
                "genres", "vote_average", "director", "explanation"
            ])

        records = []
        for m in ranked_movies:
            records.append({
                "movie_id": m.movie_id,
                "title": m.title,
                "similarity_score": round(m.similarity_score, 4),
                "release_year": m.release_year,
                "genres": ", ".join(m.genres) if m.genres else "Unknown",
                "vote_average": m.vote_average,
                "director": m.director if m.director else "Unknown",
                "explanation": m.explanation.summary_text if m.explanation else ""
            })

        return pd.DataFrame(records)
