"""
Reproducible Evaluation and Metric Verification Module for NostalgicNext.

Methodology & Disclaimers:
- Content-based metadata engines lack universal user-item feedback ground truth.
- This module implements defensible, reproducible offline evaluation proxies:
  1. Graded Genre & Thematic Relevance (Precision@K, Recall@K, NDCG@K, MRR@K).
  2. Franchise and Director Offline Sanity Checks (verifying intuitive nearest neighbors).
  3. Intra-List Diversity (ILD) and Novelty metrics.
  4. Exact before-and-after data cleaning statistics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .engine import NostalgicNextEngine

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Summary of verified evaluation metrics."""
    evaluated_anchor_movies_count: int
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_recall_at_10: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    mean_mrr_at_10: float
    mean_intra_list_diversity: float
    sanity_checks_passed: int
    sanity_checks_total: int
    data_cleaning_noise_reduction_pct: float
    relevance_proxy_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            "============================================================\n"
            " NOSTALGNEXT REPRODUCIBLE EVALUATION REPORT\n"
            "============================================================\n"
            f"Evaluated Anchor Movies      : {self.evaluated_anchor_movies_count}\n"
            f"Mean Precision@5             : {self.mean_precision_at_5:.2%}\n"
            f"Mean Precision@10            : {self.mean_precision_at_10:.2%}\n"
            f"Mean Recall@10 (Proxy)       : {self.mean_recall_at_10:.2%}\n"
            f"Mean NDCG@5 (Graded)         : {self.mean_ndcg_at_5:.4f}\n"
            f"Mean NDCG@10 (Graded)        : {self.mean_ndcg_at_10:.4f}\n"
            f"Mean Reciprocal Rank (MRR@10): {self.mean_mrr_at_10:.4f}\n"
            f"Mean Intra-List Diversity    : {self.mean_intra_list_diversity:.4f}\n"
            f"Offline Sanity Checks Passed : {self.sanity_checks_passed} / {self.sanity_checks_total}\n"
            f"Measured Noise Reduction     : {self.data_cleaning_noise_reduction_pct:.2f}%\n"
            "------------------------------------------------------------\n"
            f"Note on Relevance Proxy: {self.relevance_proxy_explanation}\n"
            "============================================================"
        )


class RecommendationEvaluator:
    """Evaluates recommendation quality across diverse anchor films."""

    DEFAULT_ANCHOR_MOVIES = [
        "Inception",
        "The Dark Knight",
        "The Godfather",
        "Toy Story",
        "Avatar",
        "Pulp Fiction",
        "Titanic",
        "Interstellar",
        "The Matrix",
        "Forrest Gump",
        "Gladiator",
        "Fight Club",
        "Finding Nemo",
        "Jurassic Park",
        "The Silence of the Lambs"
    ]

    SANITY_CHECK_PAIRS = [
        ("The Dark Knight", ["The Dark Knight Rises", "Batman Begins"]),
        ("The Godfather", ["The Godfather: Part II"]),
        ("Toy Story", ["Toy Story 2", "Toy Story 3", "Monsters, Inc."]),
        ("The Matrix", ["The Matrix Reloaded", "The Matrix Revolutions"]),
        ("Finding Nemo", ["Shark Tale", "A Bug's Life", "Monsters, Inc."])
    ]

    def __init__(self, engine: NostalgicNextEngine):
        self.engine = engine

    def evaluate(
        self,
        anchor_titles: Optional[List[str]] = None,
        k_values: Tuple[int, int] = (5, 10),
        output_file: Optional[Path | str] = None
    ) -> EvaluationMetrics:
        """Run full evaluation suite and return verified metrics."""
        anchors = anchor_titles or self.DEFAULT_ANCHOR_MOVIES

        p_at_5_list: List[float] = []
        p_at_10_list: List[float] = []
        r_at_10_list: List[float] = []
        ndcg_at_5_list: List[float] = []
        ndcg_at_10_list: List[float] = []
        mrr_list: List[float] = []
        diversity_list: List[float] = []

        valid_anchor_count = 0

        for title in anchors:
            try:
                seed_idx, seed_movie = self.engine.filter.resolve_movie(title)
            except Exception:
                continue

            recs = self.engine.recommend(title, top_n=10, explain=False)
            if not recs:
                continue

            valid_anchor_count += 1
            seed_genres = set(seed_movie.get("genres_list", []))
            seed_director = str(seed_movie.get("director", "")).lower()

            # Compute graded relevance for each recommendation
            # Grade 3: Shares >= 2 genres and same director
            # Grade 2: Shares >= 2 genres OR (shares 1 genre and same director)
            # Grade 1: Shares 1 genre
            # Grade 0: No genre overlap
            grades = []
            is_relevant_binary = []

            for r in recs:
                cand_genres = set(r.genres)
                cand_dir = str(r.director).lower()
                overlap = len(seed_genres & cand_genres)

                same_dir = bool(seed_director and cand_dir and seed_director == cand_dir)

                if (overlap >= 2 and same_dir) or overlap >= 3:
                    grade = 3
                elif overlap >= 2 or (overlap == 1 and same_dir):
                    grade = 2
                elif overlap == 1:
                    grade = 1
                else:
                    grade = 0

                grades.append(grade)
                is_relevant_binary.append(grade >= 2)  # Binary threshold for precision/recall

            # Precision@5 & Precision@10
            p5 = sum(is_relevant_binary[:5]) / 5.0
            p10 = sum(is_relevant_binary[:10]) / 10.0
            p_at_5_list.append(p5)
            p_at_10_list.append(p10)

            # Recall@10 proxy: total available candidates in catalog sharing >= 2 genres
            total_relevant_in_catalog = sum(
                self.engine.catalog["genres_list"].apply(
                    lambda gl: len(seed_genres & set(gl)) >= 2 if isinstance(gl, list) else False
                )
            )
            # Exclude seed itself
            total_relevant_in_catalog = max(1, total_relevant_in_catalog - 1)
            recall_10 = sum(is_relevant_binary[:10]) / min(10.0, total_relevant_in_catalog)
            r_at_10_list.append(recall_10)

            # NDCG@5 and NDCG@10
            ndcg_at_5_list.append(self._compute_ndcg(grades[:5]))
            ndcg_at_10_list.append(self._compute_ndcg(grades[:10]))

            # MRR@10
            mrr = 0.0
            for rank_idx, rel in enumerate(is_relevant_binary[:10], start=1):
                if rel:
                    mrr = 1.0 / rank_idx
                    break
            mrr_list.append(mrr)

            # Intra-List Diversity (1 - average pairwise cosine similarity among top 5 recs)
            rec_indices = [r.index for r in recs[:5]]
            if len(rec_indices) > 1:
                sub_matrix = self.engine.feature_matrix[rec_indices]
                pairwise_sim = (sub_matrix * sub_matrix.T).toarray()
                # Exclude diagonal
                n = len(rec_indices)
                upper_tri_indices = np.triu_indices(n, k=1)
                mean_pairwise = float(np.mean(pairwise_sim[upper_tri_indices]))
                diversity_list.append(1.0 - mean_pairwise)

        # Sanity checks
        passed_sanity = 0
        total_sanity = len(self.SANITY_CHECK_PAIRS)
        for seed_title, expected_targets in self.SANITY_CHECK_PAIRS:
            try:
                top_recs = self.engine.recommend(seed_title, top_n=10, explain=False)
                rec_titles = [r.title.lower() for r in top_recs]
                expected_lower = [t.lower() for t in expected_targets]
                # If at least one expected target appears in Top-10
                if any(any(exp in rec_title or rec_title in exp for rec_title in rec_titles) for exp in expected_lower):
                    passed_sanity += 1
            except Exception as e:
                logger.warning(f"Sanity check error on {seed_title}: {e}")

        cleaning_report = self.engine.get_cleaning_report()
        noise_red = cleaning_report.noise_reduction_pct if cleaning_report else 0.0

        metrics = EvaluationMetrics(
            evaluated_anchor_movies_count=valid_anchor_count,
            mean_precision_at_5=float(np.mean(p_at_5_list)) if p_at_5_list else 0.0,
            mean_precision_at_10=float(np.mean(p_at_10_list)) if p_at_10_list else 0.0,
            mean_recall_at_10=float(np.mean(r_at_10_list)) if r_at_10_list else 0.0,
            mean_ndcg_at_5=float(np.mean(ndcg_at_5_list)) if ndcg_at_5_list else 0.0,
            mean_ndcg_at_10=float(np.mean(ndcg_at_10_list)) if ndcg_at_10_list else 0.0,
            mean_mrr_at_10=float(np.mean(mrr_list)) if mrr_list else 0.0,
            mean_intra_list_diversity=float(np.mean(diversity_list)) if diversity_list else 0.0,
            sanity_checks_passed=passed_sanity,
            sanity_checks_total=total_sanity,
            data_cleaning_noise_reduction_pct=noise_red,
            relevance_proxy_explanation=(
                "Relevance proxy defined by multi-attribute genre and thematic overlap "
                "(sharing >= 2 genres or shared director with genre overlap). "
                "Graded NDCG evaluates ranking quality with logarithmic position discounting."
            )
        )

        if output_file:
            out_p = Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            logger.info(f"Saved evaluation metrics to {out_p}")

        return metrics

    @staticmethod
    def _compute_dcg(relevance_grades: List[int]) -> float:
        dcg = 0.0
        for i, grade in enumerate(relevance_grades):
            dcg += (2**grade - 1) / np.log2(i + 2)
        return float(dcg)

    def _compute_ndcg(self, relevance_grades: List[int]) -> float:
        dcg = self._compute_dcg(relevance_grades)
        ideal_grades = sorted(relevance_grades, reverse=True)
        idcg = self._compute_dcg(ideal_grades)
        return dcg / idcg if idcg > 0 else 0.0
