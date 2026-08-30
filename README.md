# VedaAI

<video src="https://github.com/SiddharthV147/VedaAI-Assignment/raw/main/video.mp4" controls muted width="100%"></video>

Upload a question paper and a student answer sheet. The backend segments the
handwriting, reads it, maps each answer to its question, and the frontend
highlights the matching region on the PDF.

I tested with the 4 model answer papers from the CBSE website, which cover
clean, cursive, and messy handwriting.

## Pipeline

```
PDF to image conversion          (PyMuPDF, 300 dpi)
              |
Correcting the orientation       (4-way, ink profile)
              |
Binarisation of images           (adaptive + Otsu)
              |
Text detection using CRAFT       (CRAFT + refiner)
              |
Splitting into single lines      (blank-row cuts)
              |
Crop and reading order           (top-left order)
              |
Handwriting recognition          (TrOCR large)
              |
Question paper parsing           (regex sections)
              |
Answer marker detection          (Q-number heuristics)
              |
Mapping and coordinate restore   (original page space)
              |
Persistence and response         (JSON + crops)
```

Detection worked better on images than on PDFs, so every page is rasterised
first.

For cost efficiency, both text detection and text extraction run locally on
the machine rather than through a paid OCR API.

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
