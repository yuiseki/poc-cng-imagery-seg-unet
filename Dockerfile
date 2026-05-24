FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/

# ---

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

# Models: copy trained checkpoints.
# buildings.pt is the sag_yug combined model (Japan's best building segmentation).
# Add roads.pt / parkings.pt / parks.pt here as more models are trained.
COPY tmp/runs/hotosm_buildings_sag_yug/best.pt ./models/buildings.pt

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV MODELS_DIR="/app/models"

EXPOSE 8080

CMD uvicorn imagery_seg.api:app --host 0.0.0.0 --port ${PORT:-8080}
