"""
Production and Local Server Entrypoint for NostalgicNext.

Run with:
    python app.py
or:
    uvicorn nostalgic_next.api:app --host 0.0.0.0 --port 8000
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("=" * 60)
    print(f" NOSTALGNEXT — Movie Recommendation System Server")
    print(f" Starting server on http://{host}:{port}")
    print("=" * 60)
    uvicorn.run("nostalgic_next.api:app", host=host, port=port, reload=False)
