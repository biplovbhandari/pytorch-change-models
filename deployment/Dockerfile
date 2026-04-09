# CUDA 12.4 + cuDNN 9 runtime, Ubuntu 22.04 (supported tag)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils curl git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Make python3.11 the default "python"
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set the working directory in the container
WORKDIR /app

# 1) CUDA-enabled PyTorch (must come from special index before other deps)
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1+cu124 torchvision==0.19.1+cu124

# 2) torchange + app deps from pyproject.toml (single source of truth)
RUN pip install --no-cache-dir \
    "torchange[app] @ git+https://github.com/biplovbhandari/pytorch-change-models.git@add-app-and-deployment"

# 3) App code
COPY ./app .

# Env for Vertex
ENV AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict \
    AIP_PORT=8080 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MALLOC_ARENA_MAX=2

# Expose the port the app runs on
EXPOSE 8080

CMD ["sh", "-c", "echo '[Container] Starting server...' && python -c 'import main; print(\"[Container] Imports OK\")' && exec uvicorn main:app --host 0.0.0.0 --port 8080"]
