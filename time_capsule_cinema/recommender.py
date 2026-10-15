"""
Core recommendation engine for Time Capsule Cinema.

Combines:
    * Content-based filtering  -> TF-IDF + cosine similarity on genres
    * Collaborative filtering  -> item-based cosine similarity on user ratings
into a single weighted hybrid similarity score.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the movies and ratings CSVs into DataFrames."""
    movies = pd.read_csv(Path(data_dir) / "movies.csv")
    ratings = pd.read_csv(Path(data_dir) / "ratings.csv")
    return movies, ratings


def build_content_similarity(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Return a movie x movie cosine-similarity matrix based on genre text."""
    tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")
    genre_matrix = tfidf.fit_transform(movies_df["genres"])
    sim = cosine_similarity(genre_matrix)
    return pd.DataFrame(sim, index=movies_df["movie_id"], columns=movies_df["movie_id"])


def build_collaborative_similarity(ratings_df: pd.DataFrame, movie_ids: pd.Index) -> pd.DataFrame:
    """Return a movie x movie cosine-similarity matrix based on rating patterns."""
    user_item = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
    user_item = user_item.reindex(columns=movie_ids, fill_value=0)
    sim = cosine_similarity(user_item.T)
    return pd.DataFrame(sim, index=movie_ids, columns=movie_ids)


class HybridRecommender:
    """
    Combines content-based and collaborative similarity into one hybrid score.

    Parameters
    ----------
    movies_df : DataFrame with columns [movie_id, title, genres]
    ratings_df : DataFrame with columns [user_id, movie_id, rating]
    content_weight, collab_weight : must sum to 1.0
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                 content_weight: float = 0.5, collab_weight: float = 0.5):
        if abs((content_weight + collab_weight) - 1.0) > 1e-6:
            raise ValueError("content_weight + collab_weight must sum to 1.0")

        self.movies = movies_df
        self.ratings = ratings_df
        self.content_weight = content_weight
        self.collab_weight = collab_weight

        self.content_sim = build_content_similarity(movies_df)
        self.collab_sim = build_collaborative_similarity(ratings_df, movies_df["movie_id"])
        self._rebuild_hybrid()

    def _rebuild_hybrid(self) -> None:
        self.hybrid_sim = (
            self.content_weight * self.content_sim +
            self.collab_weight * self.collab_sim
        )

    def similar_movies(self, movie_id: int, top_n: int = 5) -> pd.DataFrame:
        """Movies most similar to a given movie, using the hybrid score."""
        if movie_id not in self.hybrid_sim.index:
            raise ValueError(f"movie_id {movie_id} not found")
        scores = self.hybrid_sim.loc[movie_id].drop(movie_id).sort_values(ascending=False)
        top = scores.head(top_n).reset_index()
        top.columns = ["movie_id", "score"]
        return top.merge(self.movies, on="movie_id")[["movie_id", "title", "genres", "score"]]

    def recommend_for_user(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend unseen movies for a user by aggregating hybrid similarity
        to everything the user has already rated, weighted by their rating.
        """
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"No ratings found for user_id {user_id}")

        seen_ids = set(user_ratings["movie_id"])
        scores = pd.Series(0.0, index=self.movies["movie_id"])

        for _, row in user_ratings.iterrows():
            movie_id, rating = row["movie_id"], row["rating"]
            scores = scores.add(self.hybrid_sim[movie_id] * rating, fill_value=0.0)

        scores = scores.drop(labels=list(seen_ids), errors="ignore")
        scores = scores.sort_values(ascending=False)

        top = scores.head(top_n).reset_index()
        top.columns = ["movie_id", "score"]
        return top.merge(self.movies, on="movie_id")[["movie_id", "title", "genres", "score"]]

    def add_rating(self, user_id: int, movie_id: int, rating: float) -> None:
        """Add a new rating and rebuild the collaborative + hybrid similarity matrices."""
        new_row = pd.DataFrame([{"user_id": user_id, "movie_id": movie_id, "rating": rating}])
        self.ratings = pd.concat([self.ratings, new_row], ignore_index=True)
        self.collab_sim = build_collaborative_similarity(self.ratings, self.movies["movie_id"])
        self._rebuild_hybrid()
