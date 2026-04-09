# PyTorch official image with CUDA 12.4 + cuDNN 9 (Python 3.11, Ubuntu 22.04)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Git needed for pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torchange + app deps from pyproject.toml (PyTorch already in base image)
RUN pip install --no-cache-dir \
    "torchange[app] @ git+https://github.com/biplovbhandari/pytorch-change-models.git@add-app-and-deployment"

# App code
COPY ./app .

# Env for Vertex AI
ENV AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict \
    AIP_PORT=8080 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MALLOC_ARENA_MAX=2

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
