# 🎬 NostalgicNext — Content-Based Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)](https://github.com/kaustubhsingh30/nostalgic-next)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%2B%20scikit--learn-indigo.svg)](https://fastapi.tiangolo.com/)

## 🌐 Live Demo

**[🚀 Launch NostalgicNext](https://nostalgic-next.onrender.com)** · [GitHub Repository](https://github.com/kaustubhsingh30/nostalgic-next)

Explore content-based movie recommendations, search the movie catalog, and discover why movies are recommended.

> **Note:** The free hosted service may take 50 seconds or more to respond after a period of inactivity.

---

NostalgicNext is a content-based movie recommendation system that processes movie metadata, extracts meaningful features, calculates cosine similarity, and ranks relevant movies with explanations.

The project uses a modular architecture that separates data cleaning, feature engineering, candidate filtering, similarity scoring, and recommendation ranking into independently testable components.

## 🌟 Key Features

- **Movie Metadata Processing:** Processes movie information such as genres, keywords, cast, crew, overviews, and taglines.
- **Data Cleaning:** Parses nested metadata and prepares meaningful text features for recommendations.
- **Duplicate Handling:** Uses movie identifiers and release information to help distinguish duplicate records from different films with similar titles.
- **TF-IDF Feature Engineering:** Converts movie metadata into numerical feature vectors.
- **Cosine Similarity:** Measures content similarity between movies.
- **Explainable Recommendations:** Provides information about shared genres, keywords, directors, and cast members where available.
- **Multiple Favorite Movies:** Supports recommendations based on one movie or multiple favorite movies, where supported by the engine.
- **Deterministic Ranking:** Uses consistent ranking and tie-breaking rules.
- **Interactive Web Interface:** Provides a web-based interface for searching movies and exploring recommendations.
- **REST API:** Exposes recommendation functionality through FastAPI.
- **Automated Testing:** Includes tests for data cleaning, feature engineering, recommendation logic, evaluation, and API behavior.

## 🏗️ System Architecture

```text
                 Raw Movie Metadata
                         |
                         v
              Data Cleaning and Validation
                         |
                         v
                 Feature Engineering
                  TF-IDF Vectorization
                         |
                         v
                  Candidate Filtering
                         |
                         v
                 Cosine Similarity
                         |
                         v
                Recommendation Ranking
                         |
                         v
              Explainable Recommendations
                         |
             +-----------+-----------+
             |                       |
             v                       v
       FastAPI Web UI            Python CLI
       and REST API              and SDK
```

## 📊 Evaluation and Results

The following results were documented during project development. Re-run the evaluation command below to confirm the results for the current version and dataset.

| Metric | Previously reported value |
|---|---:|
| Processed movie catalog | 4,803 records |
| Metadata noise reduction | 97.98% |
| Mean Precision@5 | 72.00% |
| Mean Precision@10 | 65.33% |
| Mean NDCG@5 | 0.9390 |
| Mean NDCG@10 | 0.9104 |
| Mean Reciprocal Rank (MRR@10) | 0.9667 |
| Intra-List Diversity (ILD) | 0.8751 |
| Offline sanity checks | 5/5 |

**Important:** Evaluation results depend on the dataset, relevance definitions, and evaluation procedure. They should not be interpreted as universal recommendation accuracy.

Cosine similarity measures content resemblance. A similarity score is not a calibrated probability that a user will like a movie.

## 📁 Repository Structure

```text
nostalgic-next/
├── app.py
├── main.py
├── download_data.py
├── requirements.txt
├── setup.py
├── pyproject.toml
├── Dockerfile
├── Procfile
├── render.yaml
├── README.md
├── INTERVIEW_GUIDE.md
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── nostalgic_next/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── filtering.py
│   ├── scoring.py
│   ├── ranking.py
│   ├── engine.py
│   ├── evaluation.py
│   └── api.py
├── time_capsule_cinema/
│   ├── __init__.py
│   └── recommender.py
└── tests/
    ├── test_data_cleaning.py
    ├── test_features.py
    ├── test_pipeline.py
    ├── test_evaluation.py
    ├── test_api.py
    └── test_recommender.py
```

## 🚀 Installation and Quickstart

### 1. Prerequisites

- Python 3.10–3.12 recommended
- Git

### 2. Clone the repository

```bash
git clone https://github.com/kaustubhsingh30/nostalgic-next.git
cd nostalgic-next
```

### 3. Create a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Download the dataset

```bash
python download_data.py
```

Dataset downloads require an internet connection and a working source configured by the project.

## 💻 Running the Application

### Option A: Web Interface and REST API

```bash
python app.py
```

Open the local application:

- Web interface: http://localhost:8000
- Interactive API documentation: http://localhost:8000/docs

### Option B: Command-Line Demo

```bash
python main.py
```

### Option C: Python SDK

```python
from nostalgic_next import NostalgicNextEngine

engine = NostalgicNextEngine.load_from_tmdb()

recommendations = engine.recommend("Inception", top_n=5)

for movie in recommendations:
    print(
        f"{movie.title} ({movie.release_year}) "
        f"- Score: {movie.similarity_score:.4f}"
    )

    if movie.explanation:
        print(f"Why: {movie.explanation.summary_text}")
```

## 🧪 Testing

Run the automated tests from the project root:

```bash
python -m pytest tests -v
```

The project test suite previously completed with **32 passing tests**. Run the command above after making changes to confirm the current result.

## 📈 Reproducing Evaluation Metrics

Run the evaluation against the available TMDB dataset:

```bash
python -c "from nostalgic_next.engine import NostalgicNextEngine; from nostalgic_next.evaluation import RecommendationEvaluator; engine = NostalgicNextEngine.load_from_tmdb(); print(RecommendationEvaluator(engine).evaluate().summary())"
```

Use the output from this command when reporting measured evaluation results.

## 🌐 Deployment

### Live application

**[https://nostalgic-next.onrender.com](https://nostalgic-next.onrender.com)**

The application is hosted on Render. Free instances may spin down after inactivity, resulting in a delay on the first request.

### Deploy your own instance on Render

1. Sign in to [Render](https://render.com/).
2. Create a new Web Service and connect this GitHub repository.
3. Configure the build and start commands according to the repository's deployment configuration.
4. Confirm that the application listens on the port provided by Render.
5. Deploy the service and test the public URL.

### Docker

If the repository's Dockerfile is configured correctly for your environment, build and run the container with:

```bash
docker build -t nostalgic-next .
docker run -p 8000:8000 nostalgic-next
```

Check the Dockerfile for the actual internal port before running the container.

## 🎯 Example Recommendations

For example, a user searching for **The Dark Knight** may receive movies with overlapping genres, themes, cast, or directors.

A user searching for **Inception** may receive other science-fiction, thriller, or Christopher Nolan films, depending on the available metadata and ranking scores.

Actual results depend on the dataset and the running application.

## 🛠️ Technology Stack

- **Language:** Python
- **API Framework:** FastAPI
- **Machine Learning:** scikit-learn
- **Feature Engineering:** TF-IDF vectorization
- **Similarity Calculation:** Cosine similarity
- **Testing:** pytest
- **Deployment:** Docker and Render
- **Version Control:** Git and GitHub

## 📄 License and Dataset Attribution

- **Source Code:** MIT License. See [LICENSE](LICENSE).
- **Dataset:** TMDB 5000 Movie Dataset, originally distributed through Kaggle and associated with The Movie Database (TMDb). Review the dataset's original license and attribution requirements before redistributing the data.

---

## 🔗 Project Links

- **Live Demo:** https://nostalgic-next.onrender.com
- **GitHub Repository:** https://github.com/kaustubhsingh30/nostalgic-next
- **API Documentation:** https://nostalgic-next.onrender.com/docs
