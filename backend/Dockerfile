FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CUDA torch works on GPU when the host exposes one; falls back to CPU otherwise.
RUN --mount=type=cache,target=/root/.cache/pip,id=textt-pip,sharing=locked \
    pip install --upgrade pip \
    && pip install typing-extensions "jinja2>=3.1" \
    && pip install torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless \
    && pip install numpy==1.26.4 opencv-python-headless==4.9.0.80

COPY app/ ./app/
COPY api/ ./api/
COPY main.py download_models.py start.sh ./
RUN chmod +x start.sh && mkdir -p /app/user_data /app/models

ENV TEXT_USE_CUDA=auto
ENV TEXT_MODELS_DIR=/app/models
ENV UPLOAD_DIR=/app/user_data
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
ENTRYPOINT ["./start.sh"]
