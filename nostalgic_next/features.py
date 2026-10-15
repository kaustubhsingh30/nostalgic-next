"""
Feature Engineering and Vectorization Module for NostalgicNext.

Handles:
- Building unified metadata representations ('content soup') from cleaned movie fields.
- Entity token normalization (space collapse to avoid entity token collisions).
- Configurable TF-IDF text vectorization with n-gram support.
- Safe serialization and deserialization of fitted model artifacts without data leakage.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def clean_entity_token(text: str) -> str:
    """Normalize names and entities by removing spaces and punctuation."""
    if not text:
        return ""
    # Remove non-alphanumeric chars and lowercase
    cleaned = re.sub(r"[^\w\s]", "", text)
    return cleaned.replace(" ", "").lower()


def clean_text_field(text: str) -> str:
    """Standardize free-form text (overview, tagline)."""
    if not text or pd.isna(text):
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(text))
    return re.sub(r"\s+", " ", cleaned).strip().lower()


@dataclass
class SoupConfig:
    """Configuration for weights and components in the content soup."""
    include_overview: bool = True
    include_tagline: bool = True
    include_genres: bool = True
    include_keywords: bool = True
    include_cast: bool = True
    include_director: bool = True
    director_weight: int = 2
    cast_limit: int = 3


class ContentSoupBuilder:
    """Constructs a consolidated textual representation ('content soup') per movie."""

    def __init__(self, config: Optional[SoupConfig] = None):
        self.config = config or SoupConfig()

    def build_soup(self, df: pd.DataFrame) -> pd.Series:
        """Generate content soup string for each movie in DataFrame."""
        soups: List[str] = []

        for _, row in df.iterrows():
            parts: List[str] = []

            # 1. Overview
            if self.config.include_overview and "overview" in row:
                overview_clean = clean_text_field(row.get("overview", ""))
                if overview_clean:
                    parts.append(overview_clean)

            # 2. Tagline
            if self.config.include_tagline and "tagline" in row:
                tagline_clean = clean_text_field(row.get("tagline", ""))
                if tagline_clean:
                    parts.append(tagline_clean)

            # 3. Genres
            if self.config.include_genres and "genres_list" in row:
                genres = row.get("genres_list", [])
                if isinstance(genres, list):
                    cleaned_genres = [clean_entity_token(g) for g in genres if g]
                    if cleaned_genres:
                        parts.append(" ".join(cleaned_genres))

            # 4. Keywords
            if self.config.include_keywords and "keywords_list" in row:
                keywords = row.get("keywords_list", [])
                if isinstance(keywords, list):
                    cleaned_keywords = [clean_entity_token(k) for k in keywords if k]
                    if cleaned_keywords:
                        parts.append(" ".join(cleaned_keywords))

            # 5. Top Cast
            if self.config.include_cast and "cast_list" in row:
                cast = row.get("cast_list", [])
                if isinstance(cast, list):
                    cleaned_cast = [clean_entity_token(c) for c in cast[:self.config.cast_limit] if c]
                    if cleaned_cast:
                        parts.append(" ".join(cleaned_cast))

            # 6. Director (boosted by director_weight)
            if self.config.include_director and "director" in row:
                director = clean_entity_token(str(row.get("director", "")))
                if director:
                    parts.append(" ".join([director] * self.config.director_weight))

            # Combine all tokens
            movie_soup = " ".join([p for p in parts if p]).strip()
            # If movie had zero metadata, supply a fallback empty string
            soups.append(movie_soup if movie_soup else "movie")

        return pd.Series(soups, index=df.index, name="soup")


class FeatureVectorizer:
    """
    Fits and applies TF-IDF vectorization over movie metadata.
    Separates fitting from transformation to prevent evaluation leakage.
    """

    def __init__(
        self,
        max_features: int = 15000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        stop_words: str = "english"
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=stop_words
        )
        self.is_fitted = False

    def fit(self, soups: pd.Series) -> FeatureVectorizer:
        """Fit vectorizer vocabulary on content soups."""
        self.vectorizer.fit(soups)
        self.is_fitted = True
        return self

    def transform(self, soups: pd.Series) -> csr_matrix:
        """Transform soups into sparse TF-IDF matrix."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")
        return self.vectorizer.transform(soups)

    def fit_transform(self, soups: pd.Series) -> csr_matrix:
        """Fit and transform in a single pass."""
        matrix = self.vectorizer.fit_transform(soups)
        self.is_fitted = True
        return matrix

    def save_artifacts(self, output_dir: Path | str) -> None:
        """Serialize fitted vectorizer artifact."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, out_path / "tfidf_vectorizer.joblib")
        logger.info(f"Saved vectorizer artifacts to {out_path}")

    @classmethod
    def load_artifacts(cls, artifact_dir: Path | str) -> FeatureVectorizer:
        """Load fitted vectorizer artifact."""
        in_path = Path(artifact_dir) / "tfidf_vectorizer.joblib"
        if not in_path.exists():
            raise FileNotFoundError(f"Vectorizer artifact not found at {in_path}")
        instance = cls()
        instance.vectorizer = joblib.load(in_path)
        instance.is_fitted = True
        return instance
