# VedaAI

Upload a question paper and an answer sheet. The backend detects handwritten regions with CRAFT, reads them with TrOCR, and the frontend highlights the mapped answers.

## Deploy on Render

`render.yaml` defines two services:

| Service | How it runs |
|---|---|
| `vedai-backend` | Root `Dockerfile` (CPU PyTorch; models download at image build) |
| `vedai-frontend` | Node 20 (`npm ci && npm run build`) |

Render has no GPU. The backend is **4c-8g (8 GB RAM)** — smaller plans will OOM. The first backend build downloads about 2.2 GB of weights and can take 15–20 minutes.

1. Push this repo to GitHub.
2. In Render: **New → Blueprint** and select the repo.
3. If you already created a Docker service by hand, set **Dockerfile path** to `Dockerfile` and **context** to the repo root. Do not point it at `backend/`.
4. Optionally set `HUGGINGFACE_API_KEY` on `vedai-backend` if the Hub rate-limits the download.
5. After the first deploy, confirm the frontend env `NEXT_PUBLIC_BACKEND_URL` is the backend `https://….onrender.com` URL. Redeploy the frontend if that URL was missing at build time.

Uploads are stored on a 10 GB disk at `/app/user_data`.
