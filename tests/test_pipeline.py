"""
Tests for Candidate Filtering, Scoring, Ranking, and Engine Integration.
"""

import pandas as pd
import pytest

from nostalgic_next import (
    NostalgicNextEngine,
    CandidateFilter,
    MovieNotFoundError,
    RecommendationFilterCriteria,
    RecommendationRanker,
    ScoredMovie
)


@pytest.fixture
def sample_catalog():
    return pd.DataFrame([
        {
            "id": 1,
            "title": "The Matrix",
            "release_year": 1999,
            "genres_list": ["Action", "Science Fiction"],
            "keywords_list": ["virtual reality", "future"],
            "cast_list": ["Keanu Reeves", "Laurence Fishburne"],
            "director": "Lana Wachowski",
            "overview": "A computer hacker learns about the true nature of his reality.",
            "tagline": "Welcome to the Real World.",
            "vote_average": 8.2,
            "vote_count": 20000
        },
        {
            "id": 2,
            "title": "Inception",
            "release_year": 2010,
            "genres_list": ["Action", "Science Fiction", "Thriller"],
            "keywords_list": ["dream", "subconscious"],
            "cast_list": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
            "director": "Christopher Nolan",
            "overview": "A thief who steals corporate secrets through dream-sharing technology.",
            "tagline": "Your mind is the scene of the crime.",
            "vote_average": 8.3,
            "vote_count": 28000
        },
        {
            "id": 3,
            "title": "Interstellar",
            "release_year": 2014,
            "genres_list": ["Adventure", "Drama", "Science Fiction"],
            "keywords_list": ["space", "black hole"],
            "cast_list": ["Matthew McConaughey", "Anne Hathaway"],
            "director": "Christopher Nolan",
            "overview": "A team of explorers travel through a wormhole in space.",
            "tagline": "Mankind was born on Earth.",
            "vote_average": 8.3,
            "vote_count": 25000
        },
        {
            "id": 4,
            "title": "The Notebook",
            "release_year": 2004,
            "genres_list": ["Romance", "Drama"],
            "keywords_list": ["love", "memory"],
            "cast_list": ["Ryan Gosling", "Rachel McAdams"],
            "director": "Nick Cassavetes",
            "overview": "An epic love story centered around an older man reading to an older woman.",
            "tagline": "Behind every great love is a great story.",
            "vote_average": 7.7,
            "vote_count": 8000
        },
        {
            "id": 5,
            "title": "La La Land",
            "release_year": 2016,
            "genres_list": ["Comedy", "Drama", "Romance"],
            "keywords_list": ["music", "los angeles"],
            "cast_list": ["Ryan Gosling", "Emma Stone"],
            "director": "Damien Chazelle",
            "overview": "A jazz pianist falls for an aspiring actress in Los Angeles.",
            "tagline": "Here's to the fools who dream.",
            "vote_average": 7.9,
            "vote_count": 12000
        }
    ])


@pytest.fixture
def engine(sample_catalog):
    return NostalgicNextEngine(sample_catalog)


def test_resolve_movie_exact_and_fuzzy(engine):
    idx, movie = engine.filter.resolve_movie("Inception")
    assert movie["id"] == 2

    # Lowercase match
    idx2, movie2 = engine.filter.resolve_movie("the matrix")
    assert movie2["id"] == 1

    # Numeric ID resolution
    idx3, movie3 = engine.filter.resolve_movie(3)
    assert movie3["id"] == 3

    # Fuzzy match
    idx4, movie4 = engine.filter.resolve_movie("inceptn")
    assert movie4["id"] == 2


def test_unknown_movie_raises_error(engine):
    with pytest.raises(MovieNotFoundError):
        engine.filter.resolve_movie("Completely Nonexistent Movie 9999")

    with pytest.raises(MovieNotFoundError):
        engine.filter.resolve_movie("")


def test_recommend_strictly_excludes_seed(engine):
    recs = engine.recommend("Inception", top_n=3)
    rec_ids = [r.movie_id for r in recs]
    assert 2 not in rec_ids  # Inception itself must NOT be in recommendations
    assert len(recs) <= 3


def test_recommend_criteria_filtering(engine):
    # Only Romance
    recs = engine.recommend(
        "Inception",
        top_n=5,
        criteria=RecommendationFilterCriteria(genre="Romance")
    )
    for r in recs:
        assert "Romance" in r.genres


def test_deterministic_ranking_and_tie_breaking():
    # If two items have identical similarity score, ranker breaks tie by vote_average
    ranker = RecommendationRanker()
    c1 = ScoredMovie(
        index=0, movie_id=1, title="A", similarity_score=0.5,
        release_year=2000, genres=["Action"], vote_average=7.0, vote_count=100,
        overview="", director=""
    )
    c2 = ScoredMovie(
        index=1, movie_id=2, title="B", similarity_score=0.5,
        release_year=2000, genres=["Action"], vote_average=8.5, vote_count=200,
        overview="", director=""
    )
    ranked = ranker.rank([c1, c2], top_n=2)
    assert ranked[0].movie_id == 2  # Higher vote_average wins tie


def test_multi_seed_profile_recommendations(engine):
    # Seed profile with Inception and Interstellar
    recs = engine.recommend_for_profile(["Inception", "Interstellar"], top_n=2)
    assert len(recs) == 2
    rec_ids = [r.movie_id for r in recs]
    assert 2 not in rec_ids
    assert 3 not in rec_ids


def test_search_movies(engine):
    results = engine.search_movies("Matrix")
    assert len(results) == 1
    assert results[0]["title"] == "The Matrix"
