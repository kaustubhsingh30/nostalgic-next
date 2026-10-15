"""
NostalgicNext — Movie Recommendation System CLI Demo.

Run:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nostalgic_next import NostalgicNextEngine, RecommendationEvaluator


def main():
    print("=" * 65)
    print(" NOSTALGNEXT — Content-Based Movie Recommendation System")
    print("=" * 65)

    try:
        print("\n[1/4] Ingesting and validating TMDB 5000 dataset...")
        engine = NostalgicNextEngine.load_from_tmdb()
    except Exception as e:
        print(f"Notice: Failed to load full TMDB dataset ({e}).")
        print("Falling back to toy sample dataset...")
        engine = NostalgicNextEngine.load_from_toy()

    report = engine.get_cleaning_report()
    if report:
        print(f"-> Successfully loaded and cleaned {report.retained_records_count:,} records.")
        print(f"-> Metadata noise reduction: {report.noise_reduction_pct:.2f}%")
        print(f"-> Missing overviews imputed: {report.missing_overview_imputed}")
        print(f"-> Missing taglines imputed: {report.missing_tagline_imputed}")

    # Ensure Windows console handles UTF-8 gracefully
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("[2/4] Querying recommendations for anchor movie: 'Inception' (2010)...")
    recs = engine.recommend("Inception", top_n=5, explain=True)
    for i, r in enumerate(recs, start=1):
        genres_str = ", ".join(r.genres[:3]) if r.genres else "N/A"
        print(f"  #{i} {r.title} ({r.release_year})")
        print(f"      Similarity Score: {r.similarity_score:.4f} | Rating: {r.vote_average:.1f}/10")
        print(f"      Genres: {genres_str}")
        if r.explanation:
            print(f"      Match Details: {r.explanation.summary_text}")
        print()

    print("[3/4] Querying recommendations for anchor movie: 'The Dark Knight' (2008)...")
    recs2 = engine.recommend("The Dark Knight", top_n=5, explain=True)
    for i, r in enumerate(recs2, start=1):
        genres_str = ", ".join(r.genres[:3]) if r.genres else "N/A"
        print(f"  #{i} {r.title} ({r.release_year})")
        print(f"      Similarity Score: {r.similarity_score:.4f} | Rating: {r.vote_average:.1f}/10")
        print(f"      Genres: {genres_str}")
        if r.explanation:
            print(f"      Match Details: {r.explanation.summary_text}")
        print()

    print("[4/4] Optional Genre Filter: Recommending 'Avatar' with Animation/Sci-Fi constraint...")
    from nostalgic_next import RecommendationFilterCriteria
    recs3 = engine.recommend("Avatar", top_n=3, criteria=RecommendationFilterCriteria(genre="Science Fiction"))
    for i, r in enumerate(recs3, start=1):
        print(f"  #{i} {r.title} ({r.release_year}) [Score: {r.similarity_score:.4f}] - {', '.join(r.genres[:3])}")

    print("\n" + "=" * 65)
    print(" CLI Demo Complete.")
    print(" To start the interactive web app: python app.py")
    print(" To run automated unit tests:       pytest")
    print("=" * 65)


if __name__ == "__main__":
    main()
