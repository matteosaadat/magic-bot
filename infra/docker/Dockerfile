# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

# Avoid interactive prompts & set UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install security updates & runtime deps (curl for healthchecks / debugging)
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

# Copy just requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install deps (no cache to keep image smaller)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY src /app/src

# Tell Python where our src lives
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Drop privileges
USER appuser

# Default command (Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
