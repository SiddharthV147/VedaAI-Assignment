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

COPY backend/requirements.txt backend/download_models.py ./

RUN pip install --upgrade pip \
    && pip install typing-extensions "jinja2>=3.1" \
    && pip install torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless \
    && pip install numpy==1.26.4 opencv-python-headless==4.9.0.80

ARG HUGGINGFACE_API_KEY=
ENV HUGGINGFACE_API_KEY=$HUGGINGFACE_API_KEY
ENV HF_TOKEN=$HUGGINGFACE_API_KEY
ENV HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_API_KEY

RUN python download_models.py -o /app/models

COPY backend/app/ ./app/
COPY backend/api/ ./api/
COPY backend/main.py backend/start.sh ./
RUN chmod +x start.sh && mkdir -p /app/user_data

ENV TEXT_USE_CUDA=cpu
ENV TEXT_MODELS_DIR=/app/models
ENV UPLOAD_DIR=/app/user_data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["./start.sh"]
