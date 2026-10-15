"""
Tests for Data Loading, Schema Validation, Deduplication, and Cleaning Reports.
"""

import pandas as pd
import pytest
from pathlib import Path

from nostalgic_next.data import (
    DataLoader,
    DataCleaner,
    DataValidationError,
    safe_json_parse,
    extract_names,
    extract_director
)


def test_safe_json_parse():
    assert safe_json_parse(None) == []
    assert safe_json_parse("") == []
    assert safe_json_parse("[]") == []
    assert safe_json_parse("invalid json") == []
    parsed = safe_json_parse('[{"id": 28, "name": "Action"}]')
    assert len(parsed) == 1
    assert parsed[0]["name"] == "Action"


def test_extract_names():
    raw = '[{"name": "Action"}, {"name": "Sci-Fi"}, {"name": "Thriller"}]'
    names = extract_names(raw)
    assert names == ["Action", "Sci-Fi", "Thriller"]
    limited = extract_names(raw, limit=2)
    assert limited == ["Action", "Sci-Fi"]


def test_extract_director():
    crew = '[{"job": "Producer", "name": "Jane"}, {"job": "Director", "name": "Christopher Nolan"}]'
    assert extract_director(crew) == "Christopher Nolan"
    no_dir = '[{"job": "Producer", "name": "Jane"}]'
    assert extract_director(no_dir) == ""


def test_schema_validation_failure():
    bad_movies = pd.DataFrame([{"wrong_col": 1}])
    bad_credits = pd.DataFrame([{"wrong_col": 1}])
    with pytest.raises(DataValidationError):
        DataLoader._validate_tmdb_schemas(bad_movies, bad_credits)


def test_clean_toy_dataset():
    toy_movies = pd.DataFrame([
        {"movie_id": 1, "title": "Movie A", "genres": "Action Sci-Fi"},
        {"movie_id": 1, "title": "Movie A", "genres": "Action Sci-Fi"}, # Duplicate
        {"movie_id": 2, "title": "Movie B", "genres": "Drama"}
    ])
    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean_toy_dataset(toy_movies)
    assert len(cleaned_df) == 2
    assert report.duplicate_records_removed == 1
    assert report.retained_records_count == 2
    assert "genres_list" in cleaned_df.columns


def test_distinct_movies_with_same_title_not_merged():
    # Defensible strategy test: Batman 1989 vs Batman 1966 should both be kept
    movies = pd.DataFrame([
        {"id": 101, "title": "Batman", "genres": "[]", "overview": "1989 version", "tagline": "", "release_date": "1989-06-23"},
        {"id": 102, "title": "Batman", "genres": "[]", "overview": "1966 version", "tagline": "", "release_date": "1966-07-30"}
    ])
    credits = pd.DataFrame([
        {"movie_id": 101, "title": "Batman", "cast": "[]", "crew": "[]"},
        {"movie_id": 102, "title": "Batman", "cast": "[]", "crew": "[]"}
    ])
    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean_tmdb_dataset(movies, credits)
    assert len(cleaned_df) == 2
    assert report.duplicate_titles_examined == 1
    assert report.duplicate_records_removed == 0
    assert set(cleaned_df["id"]) == {101, 102}


def test_missing_values_imputed_safely():
    movies = pd.DataFrame([
        {"id": 1, "title": "Test Nulls", "genres": "[]", "overview": None, "tagline": None, "release_date": None}
    ])
    credits = pd.DataFrame([
        {"movie_id": 1, "title": "Test Nulls", "cast": "[]", "crew": "[]"}
    ])
    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean_tmdb_dataset(movies, credits)
    assert cleaned_df.iloc[0]["overview"] == ""
    assert cleaned_df.iloc[0]["tagline"] == ""
    assert cleaned_df.iloc[0]["release_year"] == 0
    assert report.missing_overview_imputed == 1
    assert report.missing_tagline_imputed == 1
