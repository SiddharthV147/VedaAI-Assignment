# VedaAI

Upload a question paper and an answer sheet. The backend detects handwritten regions with CRAFT, reads them with TrOCR, and the frontend highlights the mapped answers.

## Local (Docker Compose)

Models stay on the host and are mounted into the backend. GPU is used when the machine has one.

```bash
# first time only, if models/ is empty:
cd backend && python download_models.py -o models && cd ..

docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000/health

Copy `backend/.env.example` to `backend/.env` if you need a Hugging Face token.

## Render

`render.yaml` defines two web services:

| Service | How it runs |
|---|---|
| `vedai-backend` | Docker (`backend/Dockerfile.render`), CPU PyTorch, models downloaded at image build |
| `vedai-frontend` | Node 20, `NEXT_PUBLIC_BACKEND_URL` set from the backend URL |

Render does not provide a GPU. The backend is set to **4c-8g (8 GB RAM)** — TrOCR-large will OOM on smaller instances. The first backend build downloads about 2.2 GB of weights and can take 15–20 minutes.

1. Push this repo to GitHub.
2. In Render: **New → Blueprint** and select the repo.
3. Optionally set `HUGGINGFACE_API_KEY` on `vedai-backend` if the Hub rate-limits the model download.
4. After the first deploy, confirm the frontend env `NEXT_PUBLIC_BACKEND_URL` is the backend `https://….onrender.com` URL, then redeploy the frontend if the first build ran before that URL existed.

Uploads are stored on a 10 GB disk at `/app/user_data`.
