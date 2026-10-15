"""
Core Recommendation Engine for NostalgicNext.

Orchestrates:
- Data loading and cleaning.
- Feature engineering and vectorization.
- Filtering, scoring, and ranking pipeline.
- Preserves and extends the HybridRecommender (content + collaborative) architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .data import DataCleaner, DataCleaningReport, DataLoader
from .features import ContentSoupBuilder, FeatureVectorizer, SoupConfig
from .filtering import CandidateFilter, MovieNotFoundError, RecommendationFilterCriteria
from .ranking import RecommendationRanker
from .scoring import ScoredMovie, SimilarityScorer

logger = logging.getLogger(__name__)


class NostalgicNextEngine:
    """
    High-level content-based movie recommendation engine.
    Orchestrates filtering, scoring, and ranking into a testable, maintainable pipeline.
    """

    def __init__(
        self,
        catalog_df: pd.DataFrame,
        cleaning_report: Optional[DataCleaningReport] = None,
        soup_config: Optional[SoupConfig] = None,
        max_features: int = 15000
    ):
        self.catalog = catalog_df.reset_index(drop=True)
        self.cleaning_report = cleaning_report

        # 1. Feature Engineering
        self.soup_builder = ContentSoupBuilder(soup_config)
        self.soups = self.soup_builder.build_soup(self.catalog)
        self.catalog["soup"] = self.soups

        # 2. Vectorization
        self.vectorizer = FeatureVectorizer(max_features=max_features)
        self.feature_matrix = self.vectorizer.fit_transform(self.soups)

        # 3. Pipeline components
        self.filter = CandidateFilter(self.catalog)
        self.scorer = SimilarityScorer(self.feature_matrix, self.catalog)
        self.ranker = RecommendationRanker()

    @classmethod
    def load_from_tmdb(
        cls,
        data_dir: Path | str = None,
        soup_config: Optional[SoupConfig] = None
    ) -> NostalgicNextEngine:
        """Factory method to ingest and clean the TMDB 5000 dataset."""
        loader = DataLoader(data_dir) if data_dir else DataLoader()
        movies, credits = loader.load_tmdb_dataset()
        cleaner = DataCleaner()
        clean_df, report = cleaner.clean_tmdb_dataset(movies, credits)
        logger.info(f"Loaded TMDB dataset: {len(clean_df)} clean records.")
        return cls(clean_df, cleaning_report=report, soup_config=soup_config)

    @classmethod
    def load_from_toy(
        cls,
        data_dir: Path | str = None
    ) -> NostalgicNextEngine:
        """Factory method to load the toy sample dataset for quick tests."""
        loader = DataLoader(data_dir) if data_dir else DataLoader()
        movies, _ = loader.load_toy_dataset()
        cleaner = DataCleaner()
        clean_df, report = cleaner.clean_toy_dataset(movies)
        return cls(clean_df, cleaning_report=report)

    def recommend(
        self,
        query: Union[str, int],
        top_n: int = 5,
        criteria: Optional[RecommendationFilterCriteria] = None,
        explain: bool = True
    ) -> List[ScoredMovie]:
        """
        Main recommendation entrypoint:
        1. Resolve seed movie
        2. Generate candidate mask (excluding seed movie itself and matching filters)
        3. Score candidates via on-demand cosine similarity
        4. Rank deterministically and select top-N
        """
        seed_idx, seed_movie = self.filter.resolve_movie(query)
        candidate_mask = self.filter.get_candidate_mask(seed_idx, criteria)

        scored = self.scorer.score_single_seed(seed_idx, candidate_mask, explain=explain)
        ranked = self.ranker.rank(scored, top_n=top_n)
        return ranked

    def recommend_dataframe(
        self,
        query: Union[str, int],
        top_n: int = 5,
        criteria: Optional[RecommendationFilterCriteria] = None
    ) -> pd.DataFrame:
        """Recommend movies and return a user-friendly pandas DataFrame."""
        ranked = self.recommend(query, top_n=top_n, criteria=criteria, explain=True)
        return self.ranker.to_dataframe(ranked)

    def recommend_for_profile(
        self,
        queries: List[Union[str, int]],
        weights: Optional[List[float]] = None,
        top_n: int = 5,
        criteria: Optional[RecommendationFilterCriteria] = None
    ) -> List[ScoredMovie]:
        """Generate recommendations based on multiple seed movies (user taste profile)."""
        seed_indices = []
        seed_ids = set()
        for q in queries:
            try:
                idx, row = self.filter.resolve_movie(q)
                seed_indices.append(idx)
                seed_ids.add(int(row["id"]))
            except MovieNotFoundError:
                continue

        if not seed_indices:
            raise MovieNotFoundError("None of the provided profile movies were found.")

        # Build mask excluding all seed movies
        comb_criteria = criteria or RecommendationFilterCriteria()
        if comb_criteria.excluded_ids:
            comb_criteria.excluded_ids.update(seed_ids)
        else:
            comb_criteria.excluded_ids = seed_ids

        # Use first seed as placeholder to initialize mask
        mask = self.filter.get_candidate_mask(seed_indices[0], comb_criteria)
        for s_idx in seed_indices:
            mask[s_idx] = False

        scored = self.scorer.score_profile_seeds(
            seed_indices,
            weights=weights,
            candidate_mask=mask,
            explain=False
        )
        return self.ranker.rank(scored, top_n=top_n)

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search catalog for movie suggestions with basic metadata."""
        matches = self.filter.search_titles(query, limit=limit)
        return [
            {"id": mid, "title": title, "year": year}
            for mid, title, year in matches
        ]

    def get_movie_details(self, query: Union[str, int]) -> Dict[str, Any]:
        """Fetch metadata for a specific movie in the catalog."""
        _, row = self.filter.resolve_movie(query)
        return {
            "movie_id": int(row["id"]),
            "title": str(row["title"]),
            "release_year": int(row.get("release_year", 0)),
            "genres": list(row.get("genres_list", [])),
            "vote_average": float(row.get("vote_average", 0.0)),
            "vote_count": int(row.get("vote_count", 0)),
            "overview": str(row.get("overview", "")),
            "director": str(row.get("director", "")),
            "cast": list(row.get("cast_list", []))
        }

    def get_cleaning_report(self) -> Optional[DataCleaningReport]:
        return self.cleaning_report


# ==============================================================================
# Preserved and Enhanced Hybrid Recommender for Collaborative + Content support
# ==============================================================================

class HybridRecommender:
    """
    Combines content-based similarity with collaborative rating patterns.
    Preserves and refactors the user's original Time Capsule Cinema architecture.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: Optional[pd.DataFrame] = None,
        content_weight: float = 0.5,
        collab_weight: float = 0.5
    ):
        if abs((content_weight + collab_weight) - 1.0) > 1e-6:
            raise ValueError("content_weight + collab_weight must sum to 1.0")

        self.movies = movies_df.copy()
        if "movie_id" not in self.movies.columns and "id" in self.movies.columns:
            self.movies["movie_id"] = self.movies["id"]

        self.ratings = ratings_df.copy() if ratings_df is not None else pd.DataFrame(columns=["user_id", "movie_id", "rating"])
        self.content_weight = content_weight
        self.collab_weight = collab_weight

        # Build content similarity
        self.content_sim = self._build_content_similarity(self.movies)
        self.collab_sim = self._build_collaborative_similarity(self.ratings, self.movies["movie_id"])
        self._rebuild_hybrid()

    @staticmethod
    def _build_content_similarity(movies_df: pd.DataFrame) -> pd.DataFrame:
        """Compute genre/text content similarity."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        genre_col = movies_df["genres"].fillna("").astype(str) if "genres" in movies_df.columns else pd.Series([""] * len(movies_df))
        tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")
        mat = tfidf.fit_transform(genre_col)
        sim = cosine_similarity(mat)
        return pd.DataFrame(sim, index=movies_df["movie_id"], columns=movies_df["movie_id"])

    @staticmethod
    def _build_collaborative_similarity(ratings_df: pd.DataFrame, movie_ids: pd.Series) -> pd.DataFrame:
        """Compute item-based cosine similarity from user ratings."""
        if ratings_df.empty or "user_id" not in ratings_df.columns:
            # If no ratings provided, return zero matrix
            return pd.DataFrame(0.0, index=movie_ids, columns=movie_ids)
        user_item = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
        user_item = user_item.reindex(columns=movie_ids, fill_value=0)
        sim = cosine_similarity(user_item.T)
        return pd.DataFrame(sim, index=movie_ids, columns=movie_ids)

    def _rebuild_hybrid(self) -> None:
        self.hybrid_sim = (
            self.content_weight * self.content_sim +
            self.collab_weight * self.collab_sim
        )

    def similar_movies(self, movie_id: int, top_n: int = 5) -> pd.DataFrame:
        """Return movies most similar to a given movie ID using the hybrid score."""
        if movie_id not in self.hybrid_sim.index:
            raise ValueError(f"movie_id {movie_id} not found")
        scores = self.hybrid_sim.loc[movie_id].drop(movie_id).sort_values(ascending=False)
        top = scores.head(top_n).reset_index()
        top.columns = ["movie_id", "score"]
        # Merge back metadata
        cols = ["movie_id", "title"]
        if "genres" in self.movies.columns:
            cols.append("genres")
        return top.merge(self.movies[cols], on="movie_id")

    def recommend_for_user(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """Recommend unseen movies for a user by aggregating hybrid similarity."""
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"No ratings found for user_id {user_id}")

        seen_ids = set(user_ratings["movie_id"])
        scores = pd.Series(0.0, index=self.movies["movie_id"])

        for _, row in user_ratings.iterrows():
            m_id, r = row["movie_id"], row["rating"]
            if m_id in self.hybrid_sim.columns:
                scores = scores.add(self.hybrid_sim[m_id] * r, fill_value=0.0)

        scores = scores.drop(labels=list(seen_ids), errors="ignore")
        scores = scores.sort_values(ascending=False)

        top = scores.head(top_n).reset_index()
        top.columns = ["movie_id", "score"]
        cols = ["movie_id", "title"]
        if "genres" in self.movies.columns:
            cols.append("genres")
        return top.merge(self.movies[cols], on="movie_id")

    def add_rating(self, user_id: int, movie_id: int, rating: float) -> None:
        """Add a rating dynamically and update similarity matrices."""
        new_row = pd.DataFrame([{"user_id": user_id, "movie_id": movie_id, "rating": rating}])
        self.ratings = pd.concat([self.ratings, new_row], ignore_index=True)
        self.collab_sim = self._build_collaborative_similarity(self.ratings, self.movies["movie_id"])
        self._rebuild_hybrid()
