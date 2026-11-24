# Container for the ML core: installs the package and runs the test suite by default.
# Build:  docker build -t weather-impact .
# Test:   docker run --rm weather-impact
# Train:  docker run --rm weather-impact python scripts/train_model.py --fast

FROM python:3.12-slim

# XGBoost needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-dev.txt requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements-dev.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY app ./app
COPY .streamlit ./.streamlit
COPY data/sample ./data/sample
RUN pip install -e .

CMD ["pytest"]
