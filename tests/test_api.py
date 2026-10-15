"""
Tests for FastAPI Endpoints and Web Application.
"""

from fastapi.testclient import TestClient
import pytest

from nostalgic_next.api import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "NostalgicNext" in response.text


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dataset_records"] > 0


def test_search_movies_endpoint():
    response = client.get("/api/movies?q=Inception")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert any("Inception" in m["title"] for m in data)


def test_recommend_endpoint_success():
    response = client.get("/api/recommend?query=Inception&top_n=3")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Inception"
    assert "seed_movie" in data
    assert len(data["recommendations"]) == 3
    # Check that seed movie is not in recommendations
    rec_titles = [r["title"] for r in data["recommendations"]]
    assert "Inception" not in rec_titles


def test_recommend_endpoint_not_found():
    response = client.get("/api/recommend?query=NonExistentMovieABCXYZ9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_stats_endpoint():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_catalog_size" in data
    assert data["total_catalog_size"] > 0
