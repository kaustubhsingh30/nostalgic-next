import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_capsule_cinema import HybridRecommender, load_data


def test_load_data():
    movies, ratings = load_data()
    assert not movies.empty
    assert not ratings.empty
    assert {"movie_id", "title", "genres"}.issubset(movies.columns)
    assert {"user_id", "movie_id", "rating"}.issubset(ratings.columns)


def test_weights_must_sum_to_one():
    movies, ratings = load_data()
    try:
        HybridRecommender(movies, ratings, content_weight=0.7, collab_weight=0.7)
        assert False, "Expected ValueError for invalid weights"
    except ValueError:
        pass


def test_similar_movies_excludes_self():
    movies, ratings = load_data()
    recommender = HybridRecommender(movies, ratings)
    result = recommender.similar_movies(movie_id=1, top_n=5)
    assert 1 not in result["movie_id"].values
    assert len(result) == 5


def test_recommend_for_user_excludes_seen():
    movies, ratings = load_data()
    recommender = HybridRecommender(movies, ratings)
    seen = set(ratings[ratings.user_id == 1]["movie_id"])
    result = recommender.recommend_for_user(user_id=1, top_n=5)
    assert seen.isdisjoint(set(result["movie_id"]))


def test_add_rating_updates_model():
    movies, ratings = load_data()
    recommender = HybridRecommender(movies, ratings)
    before = recommender.ratings.shape[0]
    recommender.add_rating(user_id=1, movie_id=3, rating=5)
    after = recommender.ratings.shape[0]
    assert after == before + 1


if __name__ == "__main__":
    test_load_data()
    test_weights_must_sum_to_one()
    test_similar_movies_excludes_self()
    test_recommend_for_user_excludes_seen()
    test_add_rating_updates_model()
    print("All tests passed.")
