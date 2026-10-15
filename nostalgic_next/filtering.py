"""
Candidate Filtering Module for NostalgicNext.

Responsibilities:
- Resolving movie queries by Title or ID with fuzzy case-insensitive fallback.
- Strictly excluding the seed movie itself from recommendation candidates.
- Filtering out previously seen or unsuitable movies.
- Applying optional metadata filters (genre constraints, release year range, minimum rating).
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MovieNotFoundError(KeyError):
    """Raised when the requested movie title or ID cannot be identified."""
    pass


@dataclass
class RecommendationFilterCriteria:
    """Optional metadata filtering criteria for candidate movies."""
    genre: Optional[str] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_vote_average: Optional[float] = None
    excluded_ids: Optional[Set[int]] = None


class CandidateFilter:
    """Filters candidate movies prior to scoring and ranking."""

    def __init__(self, catalog_df: pd.DataFrame):
        self.catalog = catalog_df.reset_index(drop=True)
        # Build fast lookup indexes
        self._title_lookup = {
            str(title).strip().lower(): idx for idx, title in enumerate(self.catalog["title"])
        }
        self._id_lookup = {
            int(movie_id): idx for idx, movie_id in enumerate(self.catalog["id"])
        }

    def resolve_movie(self, query: str | int) -> Tuple[int, pd.Series]:
        """
        Locate the movie record given an integer ID or title string.
        Supports exact case-insensitive match and closest fuzzy match.
        Returns: (catalog_row_index, movie_series_row)
        """
        # If integer or numeric string representing movie ID
        if isinstance(query, int) or (isinstance(query, str) and query.strip().isdigit()):
            mid = int(query)
            if mid in self._id_lookup:
                idx = self._id_lookup[mid]
                return idx, self.catalog.iloc[idx]

        # String title search
        title_norm = str(query).strip().lower()
        if not title_norm:
            raise MovieNotFoundError("Empty movie query provided.")

        if title_norm in self._title_lookup:
            idx = self._title_lookup[title_norm]
            return idx, self.catalog.iloc[idx]

        # Fuzzy match fallback
        candidates = list(self._title_lookup.keys())
        matches = difflib.get_close_matches(title_norm, candidates, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            idx = self._title_lookup[best_match]
            logger.info(f"Fuzzy matched '{query}' to '{self.catalog.iloc[idx]['title']}'")
            return idx, self.catalog.iloc[idx]

        raise MovieNotFoundError(
            f"Movie '{query}' not found in catalog of {len(self.catalog)} records."
        )

    def search_titles(self, query: str, limit: int = 10) -> List[Tuple[int, str, int]]:
        """Search titles by substring or fuzzy match. Returns list of (id, title, year)."""
        query_norm = str(query).strip().lower()
        if not query_norm:
            return []

        results = []
        for idx, row in self.catalog.iterrows():
            title_lower = str(row["title"]).lower()
            if query_norm in title_lower:
                results.append((int(row["id"]), str(row["title"]), int(row.get("release_year", 0))))
                if len(results) >= limit:
                    break

        if not results:
            candidates = list(self._title_lookup.keys())
            matches = difflib.get_close_matches(query_norm, candidates, n=limit, cutoff=0.5)
            for m in matches:
                idx = self._title_lookup[m]
                row = self.catalog.iloc[idx]
                results.append((int(row["id"]), str(row["title"]), int(row.get("release_year", 0))))

        return results

    def get_candidate_mask(
        self,
        seed_index: int,
        criteria: Optional[RecommendationFilterCriteria] = None
    ) -> np.ndarray:
        """
        Generate a boolean mask over catalog rows indicating eligible candidates.
        Guarantees that seed_index is excluded.
        """
        mask = np.ones(len(self.catalog), dtype=bool)

        # 1. Strictly exclude the seed movie itself
        mask[seed_index] = False

        if criteria is None:
            return mask

        # 2. Exclude specified IDs (e.g., already watched)
        if criteria.excluded_ids:
            for excl_id in criteria.excluded_ids:
                if excl_id in self._id_lookup:
                    mask[self._id_lookup[excl_id]] = False

        # 3. Optional Genre filter
        if criteria.genre and "genres_list" in self.catalog.columns:
            target_genre = criteria.genre.strip().lower()
            genre_matches = self.catalog["genres_list"].apply(
                lambda glist: any(str(g).lower() == target_genre for g in glist) if isinstance(glist, list) else False
            )
            mask &= genre_matches.to_numpy()

        # 4. Optional Year range
        if "release_year" in self.catalog.columns:
            if criteria.min_year is not None:
                mask &= (self.catalog["release_year"] >= criteria.min_year).to_numpy()
            if criteria.max_year is not None:
                mask &= (self.catalog["release_year"] <= criteria.max_year).to_numpy()

        # 5. Optional Minimum Rating
        if criteria.min_vote_average is not None and "vote_average" in self.catalog.columns:
            mask &= (self.catalog["vote_average"] >= criteria.min_vote_average).to_numpy()

        return mask
