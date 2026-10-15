FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY . .

# Download dataset during build so container starts instantly without latency
RUN python download_data.py

# Pre-cache evaluation results
RUN python -c "from nostalgic_next.engine import NostalgicNextEngine; from nostalgic_next.evaluation import RecommendationEvaluator; engine = NostalgicNextEngine.load_from_tmdb(); RecommendationEvaluator(engine).evaluate(output_file='evaluation_results.json')"

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn nostalgic_next.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
