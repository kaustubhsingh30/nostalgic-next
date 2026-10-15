"""
Similarity Scoring and Explainability Module for NostalgicNext.

Responsibilities:
- Computing pairwise cosine similarity between query vectors and candidate vectors on-demand.
- Operating memory-efficiently without storing full O(N^2) dense matrices.
- Calculating human-interpretable similarity explanations (shared genres, keywords, cast, director).
- Providing clear documentation that similarity scores measure content resemblance rather than calibrated probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)


@dataclass
class RecommendationExplanation:
    """Explains why a candidate movie was recommended relative to the seed movie."""
    shared_genres: List[str]
    shared_keywords: List[str]
    same_director: Optional[str]
    shared_cast: List[str]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shared_genres": self.shared_genres,
            "shared_keywords": self.shared_keywords,
            "same_director": self.same_director,
            "shared_cast": self.shared_cast,
            "summary_text": self.summary_text
        }


@dataclass
class ScoredMovie:
    """Movie candidate with similarity score and explanation."""
    index: int
    movie_id: int
    title: str
    similarity_score: float
    release_year: int
    genres: List[str]
    vote_average: float
    vote_count: int
    overview: str
    director: str
    explanation: Optional[RecommendationExplanation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "similarity_score": round(self.similarity_score, 4),
            "release_year": self.release_year,
            "genres": self.genres,
            "vote_average": round(self.vote_average, 1),
            "vote_count": self.vote_count,
            "overview": self.overview,
            "director": self.director,
            "explanation": self.explanation.to_dict() if self.explanation else None
        }


class SimilarityScorer:
    """
    Computes content similarity scores using normalized linear kernel (cosine similarity)
    on TF-IDF representations.
    
    NOTE ON SCORES:
    Cosine similarity scores range from 0.0 to 1.0 (for non-negative TF-IDF vectors).
    A score of 0.35 indicates moderate-to-high content resemblance in terminology,
    genres, and cast/crew, but is NOT a probability or a percentage accuracy prediction.
    """

    def __init__(self, feature_matrix: csr_matrix, catalog_df: pd.DataFrame):
        self.feature_matrix = feature_matrix
        self.catalog = catalog_df.reset_index(drop=True)

    def score_single_seed(
        self,
        seed_index: int,
        candidate_mask: np.ndarray,
        explain: bool = True
    ) -> List[ScoredMovie]:
        """
        Compute similarity scores from a single seed movie against eligible candidates.
        Efficiently performs sparse-dense dot product without allocating full matrix.
        """
        seed_vec = self.feature_matrix[seed_index]
        # linear_kernel on L2-normalized TF-IDF equals cosine similarity
        sim_scores = linear_kernel(seed_vec, self.feature_matrix).flatten()

        candidate_indices = np.where(candidate_mask)[0]
        scored_candidates: List[ScoredMovie] = []

        seed_row = self.catalog.iloc[seed_index]

        for c_idx in candidate_indices:
            score = float(sim_scores[c_idx])
            cand_row = self.catalog.iloc[c_idx]

            explanation = None
            if explain:
                explanation = self.generate_explanation(seed_row, cand_row)

            scored_candidates.append(
                ScoredMovie(
                    index=c_idx,
                    movie_id=int(cand_row["id"]),
                    title=str(cand_row["title"]),
                    similarity_score=score,
                    release_year=int(cand_row.get("release_year", 0)),
                    genres=list(cand_row.get("genres_list", [])),
                    vote_average=float(cand_row.get("vote_average", 0.0)),
                    vote_count=int(cand_row.get("vote_count", 0)),
                    overview=str(cand_row.get("overview", "")),
                    director=str(cand_row.get("director", "")),
                    explanation=explanation
                )
            )

        return scored_candidates

    def score_profile_seeds(
        self,
        seed_indices: List[int],
        weights: Optional[List[float]],
        candidate_mask: np.ndarray,
        explain: bool = False
    ) -> List[ScoredMovie]:
        """
        Compute aggregated similarity for a user taste profile (multiple liked movies).
        """
        if not seed_indices:
            return []

        w = np.array(weights if weights else [1.0] * len(seed_indices), dtype=float)
        w = w / w.sum()  # Normalize weights

        # Aggregate weighted seed vectors
        profile_vec = self.feature_matrix[seed_indices[0]] * w[0]
        for i in range(1, len(seed_indices)):
            profile_vec = profile_vec + (self.feature_matrix[seed_indices[i]] * w[i])

        sim_scores = linear_kernel(profile_vec, self.feature_matrix).flatten()

        candidate_indices = np.where(candidate_mask)[0]
        scored_candidates: List[ScoredMovie] = []

        for c_idx in candidate_indices:
            score = float(sim_scores[c_idx])
            cand_row = self.catalog.iloc[c_idx]

            scored_candidates.append(
                ScoredMovie(
                    index=c_idx,
                    movie_id=int(cand_row["id"]),
                    title=str(cand_row["title"]),
                    similarity_score=score,
                    release_year=int(cand_row.get("release_year", 0)),
                    genres=list(cand_row.get("genres_list", [])),
                    vote_average=float(cand_row.get("vote_average", 0.0)),
                    vote_count=int(cand_row.get("vote_count", 0)),
                    overview=str(cand_row.get("overview", "")),
                    director=str(cand_row.get("director", "")),
                    explanation=None
                )
            )

        return scored_candidates

    @staticmethod
    def generate_explanation(seed: pd.Series, cand: pd.Series) -> RecommendationExplanation:
        """Compute interpretable intersection of metadata features between two movies."""
        seed_genres = set(seed.get("genres_list", []))
        cand_genres = set(cand.get("genres_list", []))
        shared_genres = sorted(list(seed_genres & cand_genres))

        seed_keywords = set([k.lower() for k in seed.get("keywords_list", [])])
        cand_keywords = set([k.lower() for k in cand.get("keywords_list", [])])
        shared_keywords = sorted(list(seed_keywords & cand_keywords))

        seed_dir = str(seed.get("director", "")).strip()
        cand_dir = str(cand.get("director", "")).strip()
        same_director = seed_dir if (seed_dir and cand_dir and seed_dir.lower() == cand_dir.lower()) else None

        seed_cast = set([c.lower() for c in seed.get("cast_list", [])])
        cand_cast = set([c.lower() for c in cand.get("cast_list", [])])
        shared_cast = sorted(list(seed_cast & cand_cast))

        reasons = []
        if same_director:
            reasons.append(f"Directed by {same_director}")
        if shared_cast:
            reasons.append(f"Starring {' and '.join(shared_cast[:2])}")
        if shared_genres:
            reasons.append(f"Shares genre(s): {', '.join(shared_genres[:3])}")
        if shared_keywords:
            reasons.append(f"Themes: {', '.join(shared_keywords[:3])}")

        summary = "; ".join(reasons) if reasons else "Shares overall plot themes and thematic elements."

        return RecommendationExplanation(
            shared_genres=shared_genres,
            shared_keywords=shared_keywords,
            same_director=same_director,
            shared_cast=shared_cast,
            summary_text=summary
        )
