# VedaAI

Upload a question paper and an answer sheet. The backend detects handwritten regions with CRAFT, reads them with TrOCR, and the frontend highlights the mapped answers.

## Deploy on Render

`render.yaml` defines two services:

| Service | How it runs |
|---|---|
| `vedai-backend` | Root `Dockerfile` (CPU PyTorch; models download at image build) |
| `vedai-frontend` | Node 20 (`npm ci && npm run build`) |

Render has no GPU, so the image installs CPU PyTorch and bakes in
`trocr-base-handwritten`, which loads in roughly half the memory of the large
checkpoint and is much faster on CPU. Recognition is a little less accurate.

Two env vars control this, and `TROCR_MODEL` must match the `ARG` in the
`Dockerfile` because the weights are downloaded at build time:

| Variable | Render | Local GPU |
|---|---|---|
| `TROCR_MODEL` | `trocr-base-handwritten` | `trocr-large-handwritten` (default) |
| `TROCR_BATCH_SIZE` | `1` | unset (4 on GPU) |

Give the backend at least 4 GB of RAM. If the process is killed mid-OCR you'll
see the service 503 and restart, and the browser will report it as a CORS
error. The first build downloads the weights and can take 10–20 minutes.

1. Push this repo to GitHub.
2. In Render: **New → Blueprint** and select the repo.
3. If you already created a Docker service by hand, set **Dockerfile path** to `Dockerfile` and **context** to the repo root. Do not point it at `backend/`.
4. Optionally set `HUGGINGFACE_API_KEY` on `vedai-backend` if the Hub rate-limits the download.
5. After the first deploy, confirm the frontend env `NEXT_PUBLIC_BACKEND_URL` is the backend `https://….onrender.com` URL. Redeploy the frontend if that URL was missing at build time.

Uploads are stored on a 10 GB disk at `/app/user_data`.
