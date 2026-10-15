"""
FastAPI Application and Interactive Web Interface for NostalgicNext.

Provides:
- REST API endpoints for movie search, recommendations, explanations, health, and statistics.
- Embedded responsive Web UI for interactive exploration and live interview demonstrations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .engine import NostalgicNextEngine
from .filtering import MovieNotFoundError, RecommendationFilterCriteria

logger = logging.getLogger(__name__)

# Singleton engine instance
_engine: Optional[NostalgicNextEngine] = None


def get_engine() -> NostalgicNextEngine:
    global _engine
    if _engine is None:
        try:
            logger.info("Initializing NostalgicNextEngine from TMDB dataset...")
            _engine = NostalgicNextEngine.load_from_tmdb()
        except Exception as e:
            logger.warning(f"Failed to load TMDB dataset ({e}), falling back to toy dataset.")
            _engine = NostalgicNextEngine.load_from_toy()
    return _engine


app = FastAPI(
    title="NostalgicNext — Movie Recommendation System API",
    description="Content-based movie recommendation engine utilizing TF-IDF, cosine similarity, and explainable metadata filtering.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Response Models
class HealthResponse(BaseModel):
    status: str
    dataset_records: int
    data_cleaning_noise_reduction_pct: float
    engine_type: str


class MovieSuggestion(BaseModel):
    id: int
    title: str
    year: int


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    similarity_score: float
    release_year: int
    genres: List[str]
    vote_average: float
    vote_count: int
    overview: str
    director: str
    explanation: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    query: str
    seed_movie: Dict[str, Any]
    total_recommendations: int
    recommendations: List[RecommendationItem]


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint providing engine status and catalog size."""
    engine = get_engine()
    report = engine.get_cleaning_report()
    noise_red = report.noise_reduction_pct if report else 0.0
    return HealthResponse(
        status="healthy",
        dataset_records=len(engine.catalog),
        data_cleaning_noise_reduction_pct=round(noise_red, 2),
        engine_type="TF-IDF Content-Based Recommender"
    )


@app.get("/api/movies", response_model=List[MovieSuggestion])
def search_movies(
    q: str = Query(..., min_length=1, description="Movie title search query"),
    limit: int = Query(10, ge=1, le=50)
):
    """Search movies by title prefix or fuzzy match for autocomplete."""
    engine = get_engine()
    results = engine.search_movies(q, limit=limit)
    return results


@app.get("/api/suggested-anchors")
def suggested_anchors():
    """Returns curated seed movies ideal for demonstrations."""
    return [
        {"title": "Inception", "year": 2010, "genre": "Sci-Fi / Thriller"},
        {"title": "The Dark Knight", "year": 2008, "genre": "Action / Crime"},
        {"title": "The Godfather", "year": 1972, "genre": "Drama / Crime"},
        {"title": "Toy Story", "year": 1995, "genre": "Animation / Family"},
        {"title": "Avatar", "year": 2009, "genre": "Action / Sci-Fi"},
        {"title": "Interstellar", "year": 2014, "genre": "Sci-Fi / Adventure"},
        {"title": "Pulp Fiction", "year": 1994, "genre": "Crime / Drama"},
        {"title": "Titanic", "year": 1997, "genre": "Romance / Drama"}
    ]


@app.get("/api/recommend", response_model=RecommendationResponse)
def get_recommendations(
    query: str = Query(..., min_length=1, description="Movie title or ID"),
    top_n: int = Query(5, ge=1, le=20),
    genre: Optional[str] = Query(None, description="Optional genre constraint"),
    min_year: Optional[int] = Query(None, description="Earliest release year")
):
    """Generate ranked recommendations for a given movie title or ID."""
    engine = get_engine()
    criteria = RecommendationFilterCriteria(
        genre=genre if genre and genre.strip() != "All" else None,
        min_year=min_year
    )

    try:
        _, seed_row = engine.filter.resolve_movie(query)
        recs = engine.recommend(query, top_n=top_n, criteria=criteria, explain=True)
    except MovieNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal recommendation error.")

    seed_dict = {
        "movie_id": int(seed_row["id"]),
        "title": str(seed_row["title"]),
        "release_year": int(seed_row.get("release_year", 0)),
        "genres": list(seed_row.get("genres_list", [])),
        "director": str(seed_row.get("director", "")),
        "vote_average": float(seed_row.get("vote_average", 0.0)),
        "overview": str(seed_row.get("overview", ""))
    }

    return RecommendationResponse(
        query=query,
        seed_movie=seed_dict,
        total_recommendations=len(recs),
        recommendations=[RecommendationItem(**r.to_dict()) for r in recs]
    )


@app.get("/api/stats")
def get_stats():
    """Retrieve verified data cleaning and evaluation metrics."""
    engine = get_engine()
    report = engine.get_cleaning_report()
    return {
        "cleaning_report": report.to_dict() if report else None,
        "total_catalog_size": len(engine.catalog)
    }


# ==============================================================================
# Embedded Single-Page Web Interface (Fast, Modern, Zero external Node/NPM deps)
# ==============================================================================

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NostalgicNext — Movie Recommendation System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #131b2e;
      --card-hover: #1a253e;
      --border: #232f48;
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --accent-hover: #4f46e5;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --badge-bg: #1e293b;
      --gold: #f59e0b;
      --green: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.5;
    }
    header {
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 1rem 2rem;
    }
    .header-content {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .logo-badge {
      background: linear-gradient(135deg, #6366f1, #a855f7);
      color: white;
      font-weight: 800;
      font-size: 1.2rem;
      width: 40px;
      height: 40px;
      display: grid;
      place-items: center;
      border-radius: 10px;
      box-shadow: 0 0 20px var(--accent-glow);
    }
    .brand h1 { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
    .brand span { font-size: 0.8rem; color: var(--text-muted); font-weight: 400; margin-left: 0.5rem; }
    .header-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.85rem;
      margin-left: 1.5rem;
      transition: color 0.2s;
    }
    .header-links a:hover { color: var(--accent); }
    .container {
      max-width: 1200px;
      margin: 2rem auto;
      padding: 0 1.5rem;
    }
    .hero {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    .hero h2 {
      font-size: 2.5rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, #ffffff 40%, #a5b4fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .hero p {
      color: var(--text-muted);
      max-width: 650px;
      margin: 0 auto 1.5rem;
      font-size: 1rem;
    }
    .search-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.75rem;
      box-shadow: 0 20px 40px rgba(0,0,0,0.4);
      margin-bottom: 2rem;
    }
    .search-form {
      display: flex;
      gap: 1rem;
      position: relative;
    }
    .input-wrapper {
      flex: 1;
      position: relative;
    }
    input[type="text"] {
      width: 100%;
      background: #0b0f19;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem 1.25rem;
      color: var(--text);
      font-size: 1rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    input[type="text"]:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    .autocomplete-list {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: #101726;
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-top: 6px;
      max-height: 250px;
      overflow-y: auto;
      z-index: 100;
      display: none;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
    .autocomplete-item {
      padding: 0.75rem 1rem;
      cursor: pointer;
      font-size: 0.9rem;
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .autocomplete-item:hover {
      background: var(--card-hover);
      color: var(--accent);
    }
    .autocomplete-item span.year {
      color: var(--text-muted);
      font-size: 0.8rem;
    }
    button.btn-primary {
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 0.85rem 1.75rem;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: background 0.2s, transform 0.1s;
    }
    button.btn-primary:hover {
      background: var(--accent-hover);
      transform: translateY(-1px);
    }
    .controls {
      display: flex;
      gap: 1.5rem;
      margin-top: 1.25rem;
      align-items: center;
      flex-wrap: wrap;
      color: var(--text-muted);
      font-size: 0.85rem;
    }
    select {
      background: #0b0f19;
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      padding: 0.4rem 0.8rem;
      outline: none;
      font-size: 0.85rem;
    }
    .anchors {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
      align-items: center;
    }
    .anchors span.label {
      color: var(--text-muted);
      font-size: 0.8rem;
      font-weight: 500;
    }
    .pill {
      background: var(--badge-bg);
      border: 1px solid var(--border);
      color: #c7d2fe;
      padding: 0.35rem 0.75rem;
      border-radius: 20px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .pill:hover {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    .results-container {
      margin-top: 2rem;
    }
    .seed-header {
      background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.05));
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .seed-header h3 { font-size: 1.2rem; font-weight: 700; color: #fff; }
    .seed-meta { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    .tag {
      background: rgba(255,255,255,0.08);
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      font-size: 0.75rem;
      color: #e2e8f0;
    }
    .movie-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1.5rem;
    }
    .movie-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      transition: transform 0.2s, border-color 0.2s;
    }
    .movie-card:hover {
      transform: translateY(-4px);
      border-color: var(--accent);
    }
    .rank-badge {
      position: absolute;
      top: -10px;
      left: -10px;
      background: linear-gradient(135deg, #6366f1, #a855f7);
      color: white;
      font-weight: 800;
      font-size: 0.8rem;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
    .score-badge {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--green);
      font-weight: 600;
      display: inline-block;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.75rem;
      gap: 0.5rem;
    }
    .movie-title {
      font-size: 1.1rem;
      font-weight: 700;
      line-height: 1.3;
      color: #fff;
    }
    .movie-director {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }
    .movie-genres {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin: 0.75rem 0;
    }
    .movie-overview {
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.5;
      margin-bottom: 1rem;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .explanation-box {
      background: rgba(99, 102, 241, 0.08);
      border-left: 3px solid var(--accent);
      padding: 0.6rem 0.8rem;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #c7d2fe;
      margin-top: 0.5rem;
    }
    .explanation-box strong { color: #fff; }
    .status-box {
      text-align: center;
      padding: 3rem 1rem;
      color: var(--text-muted);
    }
    .spinner {
      border: 3px solid rgba(255,255,255,0.1);
      border-top: 3px solid var(--accent);
      border-radius: 50%;
      width: 36px;
      height: 36px;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 1rem;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .error-banner {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid #ef4444;
      color: #fca5a5;
      padding: 1rem;
      border-radius: 10px;
      margin-bottom: 1.5rem;
      font-size: 0.9rem;
    }
    footer {
      border-top: 1px solid var(--border);
      text-align: center;
      padding: 2rem;
      color: var(--text-muted);
      font-size: 0.8rem;
      margin-top: 4rem;
    }
  </style>
</head>
<body>
  <header>
    <div class="header-content">
      <div class="brand">
        <div class="logo-badge">N</div>
        <div>
          <h1>NostalgicNext</h1>
        </div>
        <span>Content-Based Recommender</span>
      </div>
      <div class="header-links">
        <a href="/docs" target="_blank">Swagger API</a>
        <a href="/api/health" target="_blank">Health Check</a>
        <a href="/api/stats" target="_blank">Data Cleaning Stats</a>
      </div>
    </div>
  </header>

  <div class="container">
    <div class="hero">
      <h2>Discover Your Next Classic</h2>
      <p>Content-based movie recommendation engine powered by TF-IDF vectorization, on-demand cosine similarity, and explainable metadata filtering.</p>
    </div>

    <div class="search-card">
      <form class="search-form" id="search-form">
        <div class="input-wrapper">
          <input type="text" id="movie-input" placeholder="Type a movie title (e.g. Inception, The Dark Knight, Toy Story)..." autocomplete="off" required>
          <div class="autocomplete-list" id="autocomplete-list"></div>
        </div>
        <button type="submit" class="btn-primary">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          Recommend
        </button>
      </form>

      <div class="controls">
        <div>
          <label for="top-n">Results: </label>
          <select id="top-n">
            <option value="5" selected>Top 5</option>
            <option value="10">Top 10</option>
            <option value="15">Top 15</option>
          </select>
        </div>
        <div>
          <label for="genre-filter">Genre Filter: </label>
          <select id="genre-filter">
            <option value="All" selected>All Genres</option>
            <option value="Action">Action</option>
            <option value="Adventure">Adventure</option>
            <option value="Animation">Animation</option>
            <option value="Comedy">Comedy</option>
            <option value="Crime">Crime</option>
            <option value="Drama">Drama</option>
            <option value="Fantasy">Fantasy</option>
            <option value="Horror">Horror</option>
            <option value="Mystery">Mystery</option>
            <option value="Romance">Romance</option>
            <option value="Science Fiction">Science Fiction</option>
            <option value="Thriller">Thriller</option>
          </select>
        </div>
        <div style="margin-left: auto;">
          <span id="catalog-count" style="color: var(--green); font-size: 0.8rem; font-family: monospace;">Catalog: 4,803 films</span>
        </div>
      </div>

      <div class="anchors">
        <span class="label">Try an anchor film:</span>
        <div id="anchor-pills" style="display: inline-flex; gap: 0.4rem; flex-wrap: wrap;"></div>
      </div>
    </div>

    <div id="error-box" class="error-banner" style="display: none;"></div>

    <div id="loading" class="status-box" style="display: none;">
      <div class="spinner"></div>
      <p>Calculating on-demand cosine similarity across TF-IDF space...</p>
    </div>

    <div id="results-area" class="results-container" style="display: none;">
      <div class="seed-header" id="seed-header">
        <div>
          <h3 id="seed-title">Movie Title</h3>
          <p id="seed-director" style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.2rem;"></p>
        </div>
        <div class="seed-meta" id="seed-meta"></div>
      </div>

      <div class="movie-grid" id="movie-grid"></div>
    </div>
  </div>

  <footer>
    <p>NostalgicNext Movie Recommendation Engine • 4,803 TMDB records • TF-IDF + Cosine Similarity</p>
    <p style="margin-top: 0.4rem; color: #64748b;">Notice: Cosine similarity scores measure content feature alignment in TF-IDF space and are not calibrated probability predictions.</p>
  </footer>

  <script>
    const input = document.getElementById('movie-input');
    const autoList = document.getElementById('autocomplete-list');
    const form = document.getElementById('search-form');
    const loading = document.getElementById('loading');
    const resultsArea = document.getElementById('results-area');
    const errorBox = document.getElementById('error-box');
    const movieGrid = document.getElementById('movie-grid');
    const seedTitle = document.getElementById('seed-title');
    const seedDirector = document.getElementById('seed-director');
    const seedMeta = document.getElementById('seed-meta');
    const anchorContainer = document.getElementById('anchor-pills');

    // Populate suggested anchors
    fetch('/api/suggested-anchors')
      .then(res => res.json())
      .then(anchors => {
        anchorContainer.innerHTML = anchors.map(a => 
          `<button class="pill" onclick="selectMovie('${a.title}')">${a.title}</button>`
        ).join('');
      });

    // Autocomplete on typing
    let debounceTimer;
    input.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      const q = input.value.trim();
      if (q.length < 2) {
        autoList.style.display = 'none';
        return;
      }
      debounceTimer = setTimeout(() => {
        fetch(`/api/movies?q=${encodeURIComponent(q)}&limit=6`)
          .then(r => r.json())
          .then(data => {
            if (data.length > 0) {
              autoList.innerHTML = data.map(m => 
                `<div class="autocomplete-item" onclick="selectMovie('${m.title.replace(/'/g, "\\\\'")}')">
                  <span>${m.title}</span>
                  <span class="year">${m.year || ''}</span>
                </div>`
              ).join('');
              autoList.style.display = 'block';
            } else {
              autoList.style.display = 'none';
            }
          })
          .catch(() => { autoList.style.display = 'none'; });
      }, 200);
    });

    document.addEventListener('click', (e) => {
      if (!autoList.contains(e.target) && e.target !== input) {
        autoList.style.display = 'none';
      }
    });

    function selectMovie(title) {
      input.value = title;
      autoList.style.display = 'none';
      runRecommendation(title);
    }

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      autoList.style.display = 'none';
      runRecommendation(input.value.trim());
    });

    function runRecommendation(title) {
      if (!title) return;
      errorBox.style.display = 'none';
      resultsArea.style.display = 'none';
      loading.style.display = 'block';

      const topN = document.getElementById('top-n').value;
      const genre = document.getElementById('genre-filter').value;
      let url = `/api/recommend?query=${encodeURIComponent(title)}&top_n=${topN}`;
      if (genre && genre !== 'All') {
        url += `&genre=${encodeURIComponent(genre)}`;
      }

      fetch(url)
        .then(async res => {
          loading.style.display = 'none';
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to fetch recommendations.');
          }
          return res.json();
        })
        .then(data => {
          renderResults(data);
        })
        .catch(err => {
          loading.style.display = 'none';
          errorBox.textContent = err.message;
          errorBox.style.display = 'block';
        });
    }

    function renderResults(data) {
      const seed = data.seed_movie;
      seedTitle.textContent = `${seed.title} (${seed.release_year || 'N/A'})`;
      seedDirector.textContent = seed.director ? `Directed by ${seed.director}` : '';

      seedMeta.innerHTML = `
        <span class="tag">⭐ ${seed.vote_average ? seed.vote_average.toFixed(1) : 'N/A'}</span>
        ${(seed.genres || []).map(g => `<span class="tag">${g}</span>`).join('')}
      `;

      movieGrid.innerHTML = data.recommendations.map((m, idx) => `
        <div class="movie-card">
          <div class="rank-badge">#${idx + 1}</div>
          <div>
            <div class="card-header">
              <div>
                <div class="movie-title">${m.title}</div>
                <div class="movie-director">${m.release_year ? m.release_year : ''} ${m.director ? '• Directed by ' + m.director : ''}</div>
              </div>
              <span class="score-badge">Similarity: ${m.similarity_score.toFixed(4)}</span>
            </div>
            
            <div class="movie-genres">
              <span class="tag" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24;">⭐ ${m.vote_average.toFixed(1)}</span>
              ${(m.genres || []).map(g => `<span class="tag">${g}</span>`).join('')}
            </div>

            <p class="movie-overview">${m.overview || 'No overview available.'}</p>
          </div>

          ${m.explanation ? `
            <div class="explanation-box">
              <strong>Match explanation:</strong> ${m.explanation.summary_text}
            </div>
          ` : ''}
        </div>
      `).join('');

      resultsArea.style.display = 'block';
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_view():
    """Serves the interactive single-page application UI."""
    return HTMLResponse(content=HTML_UI)
