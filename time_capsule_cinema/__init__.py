"""Time Capsule Cinema — a hybrid movie recommendation system."""

from .recommender import (
    HybridRecommender,
    build_content_similarity,
    build_collaborative_similarity,
    load_data,
)

__version__ = "1.0.0"
__all__ = [
    "HybridRecommender",
    "build_content_similarity",
    "build_collaborative_similarity",
    "load_data",
]
