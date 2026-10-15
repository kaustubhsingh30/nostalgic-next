"""
Tests for Feature Extraction, Content Soup Construction, and TF-IDF Vectorization.
"""

import pandas as pd
import pytest
from pathlib import Path

from nostalgic_next.features import (
    clean_entity_token,
    clean_text_field,
    ContentSoupBuilder,
    FeatureVectorizer,
    SoupConfig
)


def test_clean_entity_token():
    assert clean_entity_token("Christopher Nolan") == "christophernolan"
    assert clean_entity_token("Sci-Fi") == "scifi"
    assert clean_entity_token("") == ""


def test_clean_text_field():
    assert clean_text_field("Hello, World! A quick-test.") == "hello world a quick test"
    assert clean_text_field(None) == ""


def test_content_soup_builder():
    df = pd.DataFrame([{
        "title": "Interstellar",
        "overview": "A team of explorers travel through a wormhole in space.",
        "tagline": "Mankind was born on Earth.",
        "genres_list": ["Adventure", "Drama", "Science Fiction"],
        "keywords_list": ["space", "black hole", "future"],
        "cast_list": ["Matthew McConaughey", "Anne Hathaway"],
        "director": "Christopher Nolan"
    }])
    builder = ContentSoupBuilder(SoupConfig(director_weight=2))
    soups = builder.build_soup(df)
    soup = soups.iloc[0]

    assert "wormhole" in soup
    assert "adventure" in soup
    assert "sciencefiction" in soup
    assert "matthewmcconaughey" in soup
    assert "christophernolan christophernolan" in soup  # Boosted director


def test_vectorizer_fit_and_transform():
    soups = pd.Series([
        "space travel future astronaut galaxy",
        "romantic comedy wedding date love",
        "action hero explosion car chase thriller",
        "space galaxy stars astronaut future"
    ])
    vectorizer = FeatureVectorizer(max_features=50, min_df=1)
    matrix = vectorizer.fit_transform(soups)

    assert matrix.shape[0] == 4
    assert matrix.shape[1] > 0
    assert vectorizer.is_fitted

    # Transform new data without refitting (preventing data leakage)
    new_sample = pd.Series(["future astronaut"])
    new_vec = vectorizer.transform(new_sample)
    assert new_vec.shape == (1, matrix.shape[1])


def test_vectorizer_serialization(tmp_path: Path):
    soups = pd.Series(["sample movie soup text one", "sample movie soup text two"])
    vec = FeatureVectorizer(min_df=1)
    vec.fit(soups)

    vec.save_artifacts(tmp_path)
    loaded = FeatureVectorizer.load_artifacts(tmp_path)

    assert loaded.is_fitted
    vec1 = vec.transform(soups)
    vec2 = loaded.transform(soups)
    assert (vec1 != vec2).nnz == 0
