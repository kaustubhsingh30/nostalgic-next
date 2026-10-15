"""
NostalgicNext — Content-Based Movie Recommendation System.

Exposes high-level engine, evaluation, data loaders, and hybrid recommender.
"""

from .data import DataLoader, DataCleaner, DataCleaningReport
from .features import ContentSoupBuilder, FeatureVectorizer, SoupConfig
from .filtering import CandidateFilter, MovieNotFoundError, RecommendationFilterCriteria
from .scoring import SimilarityScorer, ScoredMovie, RecommendationExplanation
from .ranking import RecommendationRanker
from .engine import NostalgicNextEngine, HybridRecommender
from .evaluation import RecommendationEvaluator, EvaluationMetrics

__version__ = "1.0.0"
__author__ = "Kaustubh Singh Rathore"

__all__ = [
    "NostalgicNextEngine",
    "HybridRecommender",
    "DataLoader",
    "DataCleaner",
    "DataCleaningReport",
    "ContentSoupBuilder",
    "FeatureVectorizer",
    "SoupConfig",
    "CandidateFilter",
    "MovieNotFoundError",
    "RecommendationFilterCriteria",
    "SimilarityScorer",
    "ScoredMovie",
    "RecommendationExplanation",
    "RecommendationRanker",
    "RecommendationEvaluator",
    "EvaluationMetrics",
]
