# ---- base
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (sqlite, faiss CPU build deps if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# ---- deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- app + data
# Copy your source and bake in your DB + FAISS index
COPY src /app/src

# (Optional) if your ingest creates files into src/data/... they’re now baked in.
# If you prefer, keep them on a volume instead—this image bakes them in for offline simplicity.

# ---- run
EXPOSE 8000
ENV PYTHONPATH=/app/src \
    APP_MODULE=src.app:app \
    OLLAMA_BASE_URL=http://ollama:11434 \
    USE_OLLAMA=1 \
    OLLAMA_MODEL=llama3.1:8b
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
