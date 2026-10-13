# Dockerfile for SABRE Research Artifact Reproduction

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir hypothesis scikit-learn

# Copy project source and configuration
COPY . /app/

# Default command: run all tests and reproduce experiments
CMD ["python", "-m", "pytest", "-q"]
