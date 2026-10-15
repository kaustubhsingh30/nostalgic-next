"""
Tests for Recommendation Evaluation, NDCG Calculation, and Metrics.
"""

import pytest
from nostalgic_next import NostalgicNextEngine, RecommendationEvaluator


def test_ndcg_calculation():
    # If perfect ranking: [3, 2, 1, 0] -> NDCG should be 1.0
    engine = NostalgicNextEngine.load_from_toy()
    evaluator = RecommendationEvaluator(engine)

    perfect_grades = [3, 2, 1, 0]
    ndcg_perfect = evaluator._compute_ndcg(perfect_grades)
    assert pytest.approx(ndcg_perfect, 0.001) == 1.0

    # Reversed ranking: [0, 1, 2, 3] -> NDCG should be < 1.0
    reversed_grades = [0, 1, 2, 3]
    ndcg_reversed = evaluator._compute_ndcg(reversed_grades)
    assert ndcg_reversed < 1.0
    assert ndcg_reversed > 0.0

    # All zeros
    assert evaluator._compute_ndcg([0, 0, 0]) == 0.0


def test_evaluator_runs_on_toy():
    engine = NostalgicNextEngine.load_from_toy()
    evaluator = RecommendationEvaluator(engine)
    # Evaluate on small sample of toy catalog movies
    metrics = evaluator.evaluate(anchor_titles=["The Matrix", "Inception"])
    assert metrics.evaluated_anchor_movies_count > 0
    assert 0.0 <= metrics.mean_ndcg_at_5 <= 1.0
    assert isinstance(metrics.summary(), str)
