# YOLO Plant Inspector

Phone-friendly **plant disease detection** API and web UI. Point your phone camera at a leaf or plant, capture a photo, and get YOLO object-detection results with bounding boxes and an issue summary.

Built with **FastAPI**, **Ultralytics YOLO**, and a mobile-first HTML camera page — no native app required.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [End-to-end flow](#end-to-end-flow)
- [Components](#components)
- [AI models](#ai-models)
- [Issue detection logic](#issue-detection-logic)
- [Phone camera modes](#phone-camera-modes)
- [PostgreSQL & admin panel](#postgresql--admin-panel)
- [Quick start (local)](#quick-start-local)
- [Install on VM / EC2](#install-on-vm--ec2)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## What it does

1. You open the web UI on your phone (same Wi‑Fi as the laptop running the server).
2. You take a photo of a plant or leaf (or use live camera over HTTPS).
3. The image is sent to `POST /api/v1/detect`.
4. A YOLO model runs inference and returns:
   - bounding boxes + class names + confidence scores
   - `issue_found: true/false` based on disease/pest keyword rules
   - an annotated JPEG (boxes drawn on the image)
   - a human-readable `message`

The default model detects **~116 crop disease classes** (tomato blight, apple scab, etc.) — not generic COCO objects like cars or people.

---

## Architecture

```mermaid
flowchart TB
    subgraph Phone["Phone browser"]
        UI["static/camera.html"]
        Photo["Native camera / gallery"]
        Live["Live video stream (HTTPS only)"]
    end

    subgraph Server["FastAPI server (app/main.py)"]
        API["REST API"]
        Life["lifespan: load YoloDetector once"]
    end

    subgraph Core["Inference pipeline"]
        Loader["model_loader.py\nresolve + download weights"]
        Det["detector.py\nYOLO predict"]
        Reg["models_registry.py\npresets"]
        Cfg["config.py\n.env settings"]
    end

    subgraph External["External"]
        HF["Hugging Face Hub\n(plant weights)"]
        Wts["weights/*.pt\n(local cache)"]
    end

    UI --> Photo
    UI --> Live
    Photo -->|multipart JPEG| API
    Live -->|frame JPEG| API
    API --> Det
    Life --> Det
    Det --> Loader
    Loader --> Wts
    Loader -.->|first run| HF
    Cfg --> Reg
    Reg --> Cfg
    Det -->|JSON + base64 image| UI
```

### Request lifecycle (single detection)

```mermaid
sequenceDiagram
    participant P as Phone UI
    participant F as FastAPI
    participant D as YoloDetector
    participant Y as Ultralytics YOLO

    P->>F: POST /api/v1/detect (image/jpeg)
    F->>F: Validate size + content-type
    F->>D: detect(image_bytes)
    D->>D: PIL decode → RGB numpy array
    D->>Y: model.predict(conf, iou)
    Y-->>D: boxes, classes, annotated frame
    D->>D: Apply issue keyword rules
    D-->>F: detections + annotated image
    F->>F: Encode annotated JPEG as base64
    F-->>P: DetectResponse JSON
    P->>P: Show message + overlay image
```

---

## End-to-end flow

### Startup

1. `uvicorn app.main:app` starts FastAPI.
2. The **lifespan** hook runs once and creates a global `YoloDetector`.
3. `YoloDetector` reads settings from `.env` (via `config.py`).
4. If `MODEL_PRESET=plant_disease`, preset defaults are merged (model path, thresholds, keywords).
5. `model_loader.resolve_yolo_weights()` checks for `weights/plant_disease.pt`.
   - If missing → downloads from Hugging Face and caches locally (~457 MB).
6. Ultralytics loads the `.pt` file into memory. The model stays loaded for all requests.

### Detection request

1. Phone UI captures a JPEG (camera app or live frame).
2. Browser sends `multipart/form-data` with field `file`.
3. FastAPI reads bytes, calls `detector.detect()`.
4. Image is decoded with Pillow, converted to RGB `numpy` array.
5. YOLO runs object detection with `confidence_threshold` and `iou_threshold`.
6. Each box becomes a `Detection` with class name, confidence, and `x1/y1/x2/y2` bbox.
7. Issue rules mark which classes count as problems (`is_issue`).
8. Optional annotated image is drawn by Ultralytics and returned as base64 JPEG.

---

## Components

### Backend (`app/`)

| File | Role |
|------|------|
| **`main.py`** | FastAPI app, routes, CORS, static files, global detector lifecycle |
| **`detector.py`** | YOLO inference wrapper — decode image, predict, issue matching, messages |
| **`model_loader.py`** | Resolves weight paths; downloads from Hugging Face; caches to `weights/` |
| **`models_registry.py`** | Named presets (`plant_disease`, `agrosight`, etc.) with default thresholds |
| **`config.py`** | Pydantic settings from `.env`; merges preset defaults with env overrides |
| **`schemas.py`** | Request/response Pydantic models (`DetectResponse`, `Detection`, etc.) |

#### `main.py` — API layer

- **`GET /`** — serves the mobile camera UI (`static/camera.html`).
- **`POST /api/v1/detect`** — main detection endpoint (multipart image upload).
- **`GET /api/v1/model-info`** — active model, class count, sample labels.
- **`GET /api/v1/presets`** — list available model presets.
- **`GET /health`** — liveness check.
- **`/static/*`** — static assets.

The YOLO model is loaded **once at startup** (not per request) for performance.

#### `detector.py` — inference engine

Responsibilities:

- Load YOLO via `resolve_yolo_weights()`
- Convert raw bytes → RGB numpy array
- Call `model.predict()` with configurable `conf` and `iou`
- Parse Ultralytics result boxes into typed `Detection` objects
- Classify detections as issues using `issue_match_mode`
- Generate user-facing status messages
- Produce annotated overlay image via `results[0].plot()`

#### `model_loader.py` — weight resolution

Ultralytics **does not** support `hf://` URLs directly. This module:

1. Checks if `weights/<preset>.pt` exists locally.
2. If not, downloads from Hugging Face using `huggingface_hub`.
3. Caches the file under `weights/` for offline reuse.
4. Returns an absolute filesystem path for `YOLO(path)`.

#### `config.py` — settings

Environment variables are loaded from `.env`. When `MODEL_PRESET` is set (and not `custom`), defaults from `models_registry.PRESETS` are applied first, then `.env` values override them.

#### `models_registry.py` — presets

Central catalog of model configurations. Each preset defines:

- `yolo_model` — local weight path
- `confidence_threshold` / `iou_threshold`
- `issue_match_mode` — how to flag problems
- `issue_keywords` — substring rules for disease names

---

### Frontend (`static/camera.html`)

Single-page mobile UI with no build step (plain HTML/CSS/JS).

| Feature | Description |
|---------|-------------|
| **Photo mode** | `Take photo & scan` / `Choose from gallery` — works on **HTTP** |
| **Live mode** | Video preview + Scan now + auto-scan — requires **HTTPS** |
| **HTTPS banner** | Shown on HTTP; links to `https://` URL for live camera |
| **Result overlay** | Displays annotated image with bounding boxes |
| **Model line** | Fetches `/api/v1/model-info` to show active preset + class count |

#### Why two camera modes?

| Mode | Protocol | Mechanism | Works on phone? |
|------|----------|-----------|-----------------|
| Photo | `http://` | `<input type="file" capture="environment">` opens native camera | Yes |
| Live | `https://` | `navigator.mediaDevices.getUserMedia()` browser stream | Yes (with self-signed cert) |

Browsers block `getUserMedia` on non-secure origins (anything that is not `localhost` or HTTPS). Photo upload bypasses this by delegating to the OS camera app.

---

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| **`download_model.py`** | Pre-download and cache weights: `python scripts/download_model.py --preset plant_disease` |
| **`run_https.sh`** | Start server with self-signed TLS cert for live camera on phone |

---

### Other files

| File | Purpose |
|------|---------|
| **`requirements.txt`** | Python dependencies |
| **`.env.example`** | Template configuration |
| **`Dockerfile`** | Container image (CPU inference) |
| **`weights/`** | Cached `.pt` model files (gitignored) |
| **`.certs/`** | Self-signed TLS key/cert for HTTPS dev (gitignored) |

---

## AI models

### Default: `plant_disease`

| Property | Value |
|----------|-------|
| Source | [JK-TK/PlantDiseaseDetection](https://huggingface.co/JK-TK/PlantDiseaseDetection) |
| Weights file | `PlantDiseaseDetection.pt` → cached as `weights/plant_disease.pt` |
| Architecture | YOLOv11x |
| Classes | ~116 crop diseases |
| Size | ~457 MB |

Trained on laboratory + field images across major food crops.

### All presets

| Preset | Weights path | Hugging Face source | Best for |
|--------|--------------|---------------------|----------|
| **`plant_disease`** *(default)* | `weights/plant_disease.pt` | JK-TK/PlantDiseaseDetection | General crop disease detection |
| `agrosight` | `weights/agrosight.pt` | Nick-Maximillien/Agrosight-YOLOv11-Crop-Disease | African staple crops, pests, deficiencies |
| `crop_stress` | `weights/crop_stress.pt` | iamnotpalak/yolov8-transfpn-crop-disease-detection | Field crop stress |
| `plant_local` | `weights/plant_disease.pt` | Same as plant_disease | Offline use after download |
| `coco_demo` | `yolov8n.pt` | Ultralytics hub | Smoke tests only — **not for plants** |
| `custom` | Set `YOLO_MODEL` in `.env` | Your own `.pt` file | Custom trained models |

Switch preset in `.env`:

```env
MODEL_PRESET=agrosight
```

Or pre-download:

```bash
python scripts/download_model.py --preset plant_disease
```

---

## Issue detection logic

YOLO returns class names like `Tomato___Late_blight` or `Apple___Apple_scab`. The app decides whether each detection is an **issue** using `ISSUE_MATCH_MODE`:

### `keywords` (default for plant presets)

- A detection is an issue if its class name **contains** any word from `ISSUE_KEYWORDS`.
- Classes containing **`healthy`** are always ignored.
- Example keywords: `blight`, `rust`, `virus`, `pest`, `mold`, `rot`.

```env
ISSUE_MATCH_MODE=keywords
ISSUE_KEYWORDS=blight,rust,scab,mold,rot,spot,virus,wilt,pest,weed,disease
```

### `exact`

- Only classes listed in `ISSUE_CLASSES` (comma-separated, case-insensitive) count as issues.

```env
ISSUE_MATCH_MODE=exact
ISSUE_CLASSES=Tomato___Late_blight,Potato___Early_blight
```

### `all`

- Every detection counts as an issue (rarely useful).

### Response fields

| Field | Meaning |
|-------|---------|
| `issue_found` | `true` if at least one detection matched issue rules |
| `issue_classes_matched` | Human-readable list of matched disease names |
| `detections[].is_issue` | Per-box issue flag |
| `message` | Summary string for the UI |

---

## Phone camera modes

### Option A — Photo scan (HTTP, easiest)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On phone (same Wi‑Fi): `http://192.168.x.x:8000`

- Tap **Take photo & scan** or **Choose from gallery**
- Works without TLS

### Option B — Live camera (HTTPS, recommended for streaming)

```bash
chmod +x scripts/run_https.sh
./scripts/run_https.sh
```

On phone: `https://192.168.x.x:8000`

1. Accept the self-signed certificate warning once (Advanced → Proceed).
2. Allow camera access when prompted.
3. Use **Scan now** or enable **Auto-scan every 2 seconds**.

The UI shows a yellow banner on HTTP with a copyable HTTPS link.

---

## PostgreSQL & admin panel

Every call to `POST /api/v1/detect` is **persisted in PostgreSQL** with:

- detection results (classes, confidence, bounding boxes)
- issue flags and matched disease names
- **YOLO advisory notes & treatment suggestions** per class
- annotated scan image (saved under `uploads/events/`)
- inference time, model preset, client IP

### Admin UI

Open **`http://localhost:8000/admin`** to:

- view stats (total scans, issues found, avg inference ms)
- browse all detection events (filter by issue / clear)
- click an event to see annotated image, advisories, and raw detections

### Database schema

Table `detection_events`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | serial | Primary key (returned as `event_id` in detect API) |
| `created_at` | timestamptz | Scan timestamp |
| `issue_found` | boolean | Whether disease keywords matched |
| `message` | text | Short status from detector |
| `summary_suggestion` | text | Overall actionable advice |
| `detections` | jsonb | Full YOLO box list |
| `advisories` | jsonb | Per-class notes + suggestions |
| `annotated_image_path` | text | Path to JPEG with boxes |
| `model_preset`, `inference_ms`, … | | Model metadata |

Tables are created automatically on startup (`Base.metadata.create_all`).

### Start PostgreSQL

**Option A — Docker Compose (API + Postgres):**

```bash
docker compose up --build
```

**Option B — Postgres only:**

```bash
docker compose up db -d
# set DATABASE_URL in .env, then:
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Option C — local Postgres:**

```bash
createdb plant_inspector
# DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/plant_inspector
```

### Admin API

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/admin/stats` | Aggregate counts |
| `GET /api/v1/admin/events?skip=0&limit=50&issue_only=true` | Paginated event list |
| `GET /api/v1/admin/events/{id}` | Single event detail |

Detect responses now include `event_id`, `summary_suggestion`, and `advisories[]`.

---

## Quick start (local)

### Prerequisites

- Python 3.9+
- PostgreSQL (or Docker for `docker compose up db`)
- Same Wi‑Fi network for phone + server
- ~500 MB disk space for plant model weights

### Install and run

```bash
git clone https://github.com/sidhurana/yolo-fastapi-mvp.git
cd yolo-fastapi-mvp

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
python scripts/setup_db.py         # creates DB + tables (auto on startup too)

docker compose up db -d            # or use local Postgres

python scripts/download_model.py --preset plant_disease

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or one-shot install script (Ubuntu):

```bash
git clone https://github.com/sidhurana/yolo-fastapi-mvp.git
cd yolo-fastapi-mvp
chmod +x scripts/install.sh
./scripts/install.sh
source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Find your server IP:

```bash
# macOS
ipconfig getifaddr en0

# Linux / EC2
curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}'
```

Open on your phone in Safari or Chrome:

- **Camera:** `http://<SERVER_IP>:8000`
- **Admin:** `http://<SERVER_IP>:8000/admin`

HTTPS live camera: `./scripts/run_https.sh`

---

## Install on VM / EC2

Full guide for a **fresh Ubuntu 22.04/24.04** server (AWS EC2, DigitalOcean, GCP, etc.).

### 1. Launch the instance

| Setting | Recommendation |
|---------|----------------|
| OS | Ubuntu 22.04 or 24.04 LTS |
| Instance type | `t3.small` minimum (2 GB RAM); `t3.medium`+ for faster inference |
| Storage | 20 GB+ (model ~500 MB + uploads) |
| Security group | Inbound **22** (SSH), **8000** (HTTP API/UI) |

For HTTPS on port 443 later, also open **443** and use a reverse proxy (see below).

### 2. SSH in and clone

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt-get update && sudo apt-get install -y git
sudo mkdir -p /opt/yolo-fastapi-mvp
sudo chown "$USER:$USER" /opt/yolo-fastapi-mvp
git clone https://github.com/sidhurana/yolo-fastapi-mvp.git /opt/yolo-fastapi-mvp
cd /opt/yolo-fastapi-mvp
```

### 3. Run the install script

```bash
chmod +x scripts/install.sh scripts/run_https.sh scripts/setup_db.py
./scripts/install.sh
```

This installs Python deps, starts Postgres via Docker, creates the database, and downloads the plant YOLO model.

### 4. Start the API

**Manual (foreground):**

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Production (systemd — survives reboot):**

```bash
sudo cp deploy/plant-inspector.service /etc/systemd/system/
# Edit User/WorkingDirectory if not using ubuntu@/opt/yolo-fastapi-mvp
sudo systemctl daemon-reload
sudo systemctl enable plant-inspector
sudo systemctl start plant-inspector
sudo systemctl status plant-inspector
```

Ensure Docker Postgres starts on boot:

```bash
cd /opt/yolo-fastapi-mvp
sudo systemctl enable docker
docker compose up db -d
```

### 5. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","database":"ok"}

curl http://localhost:8000/api/v1/admin/stats
```

From your browser (replace IP):

- Camera: `http://<EC2_PUBLIC_IP>:8000`
- Admin panel: `http://<EC2_PUBLIC_IP>:8000/admin`

### 6. Optional — HTTPS (live phone camera)

Browsers require HTTPS for live camera on phones:

```bash
cd /opt/yolo-fastapi-mvp
./scripts/run_https.sh
# Open https://<EC2_PUBLIC_IP>:8000 on phone (accept cert warning)
```

For production HTTPS, put **nginx** or **Caddy** in front with Let's Encrypt:

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
# Point domain A-record to EC2 IP, then:
sudo certbot --nginx -d yourdomain.com
```

Proxy `yourdomain.com` → `http://127.0.0.1:8000`.

### 7. Docker Compose (all-in-one)

If Docker is installed, run API + Postgres together:

```bash
cd /opt/yolo-fastapi-mvp
docker compose up --build -d
docker compose logs -f api
```

Mount `weights/` and `uploads/` volumes so model and scan images persist (already configured in `docker-compose.yml`).

### EC2 checklist

- [ ] Security group allows **8000** from your IP (or 0.0.0.0/0 for public demo)
- [ ] `./scripts/install.sh` completed without errors
- [ ] `curl localhost:8000/health` returns `database: ok`
- [ ] Model downloaded: `ls -lh weights/plant_disease.pt`
- [ ] Phone can reach `http://<PUBLIC_IP>:8000` on same network or public IP
- [ ] (Optional) systemd service enabled for auto-start

### Environment on EC2

Edit `/opt/yolo-fastapi-mvp/.env`:

```env
MODEL_PRESET=plant_disease
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/plant_inspector
STORE_EVENT_IMAGES=true
# DEVICE=cpu
```

When using `docker compose`, Postgres host is `db` not `localhost`:

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@db:5432/plant_inspector
```

---

## Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PRESET` | `plant_disease` | Model preset name (see [AI models](#ai-models)) |
| `YOLO_MODEL` | *(from preset)* | Override weight path when `MODEL_PRESET=custom` |
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence (0–1) |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `ISSUE_MATCH_MODE` | `keywords` | `keywords` \| `exact` \| `all` |
| `ISSUE_KEYWORDS` | *(from preset)* | Comma-separated substrings for issue matching |
| `ISSUE_CLASSES` | *(empty)* | Comma-separated exact class names (for `exact` mode) |
| `DEVICE` | *(auto)* | `cpu`, `cuda:0`, etc. |
| `RETURN_ANNOTATED_IMAGE` | `true` | Include base64 JPEG with boxes in API response |
| `MAX_UPLOAD_BYTES` | `10485760` | Max upload size (10 MB) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `DATABASE_URL` | `postgresql+psycopg2://...` | PostgreSQL connection string |
| `STORE_EVENT_IMAGES` | `true` | Save annotated JPEGs to disk |
| `UPLOADS_DIR` | `uploads` | Directory for event images |

Preset values are merged first; any variable you set in `.env` overrides the preset.

---

## API reference

Interactive docs: `http://localhost:8000/docs`

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /api/v1/model-info`

Returns active model metadata.

```json
{
  "model": "/path/to/weights/plant_disease.pt",
  "model_preset": "plant_disease",
  "description": "JK-TK PlantDiseaseDetection YOLOv11x (~116 disease classes)",
  "issue_match_mode": "keywords",
  "class_count": 116,
  "sample_classes": ["Apple___Apple_scab", "..."],
  "live_camera_requires_https": true
}
```

### `GET /api/v1/presets`

Lists all registered presets and the active one.

### `POST /api/v1/detect`

**Request:** `multipart/form-data` with field `file` (JPEG or PNG)

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@leaf.jpg"
```

**Response:**

```json
{
  "issue_found": true,
  "issue_classes_matched": ["Tomato — Late blight"],
  "detections": [
    {
      "class_name": "Tomato___Late_blight",
      "confidence": 0.87,
      "bbox": { "x1": 120, "y1": 80, "x2": 340, "y2": 290 },
      "is_issue": true
    }
  ],
  "detection_count": 1,
  "inference_ms": 245.3,
  "model": "weights/plant_disease.pt",
  "model_preset": "plant_disease",
  "annotated_image_base64": "/9j/4AAQ...",
  "message": "Possible plant issue: Tomato — Late blight."
}
```

**Error codes:**

| Code | Cause |
|------|-------|
| `400` | Not an image or empty file |
| `413` | File exceeds `MAX_UPLOAD_BYTES` |
| `503` | Model still loading |

---

## Project structure

```
yolo-fastapi-mvp/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes + app factory
│   ├── detector.py          # YOLO inference + issue rules
│   ├── model_loader.py      # HF download + local weight cache
│   ├── models_registry.py   # Preset definitions
│   ├── config.py            # Environment settings
│   ├── database.py          # PostgreSQL models + session
│   ├── event_service.py     # Persist & query detection events
│   ├── advisory.py          # Disease notes & suggestions
│   ├── serializers.py       # DB → API mapping
│   └── schemas.py           # Pydantic API models
├── static/
│   ├── camera.html          # Mobile web UI
│   └── admin.html           # Admin dashboard
├── scripts/
│   ├── install.sh           # One-shot Ubuntu/EC2 setup
│   ├── setup_db.py          # Create Postgres DB + tables
│   ├── download_model.py    # Pre-download weights
│   └── run_https.sh         # HTTPS dev server for live camera
├── deploy/
│   └── plant-inspector.service  # systemd unit example
├── docker-compose.yml
├── weights/                 # Cached .pt files (gitignored)
├── uploads/events/          # Annotated scan images (gitignored)
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Docker

```bash
docker build -t yolo-plant-inspector .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/weights:/app/weights" \
  -e MODEL_PRESET=plant_disease \
  yolo-plant-inspector
```

Mount `weights/` so the model is not re-downloaded on every container start.

For HTTPS in Docker, terminate TLS with a reverse proxy (nginx, Caddy) or mount certs and pass `--ssl-keyfile` / `--ssl-certfile` to uvicorn.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: hf:/...` | Ultralytics does not support `hf://` URLs | Use local path `weights/plant_disease.pt`; run `download_model.py` |
| Model download 404 | Wrong HF filename | Plant model file is `PlantDiseaseDetection.pt`, not `best.pt` — already handled in `model_loader.py` |
| `database "plant_inspector" does not exist` | DB not created | Run `python scripts/setup_db.py` (auto-created on startup since v0.3) |
| Live camera blocked | HTTP on phone | Use `./scripts/run_https.sh` and open `https://` URL |
| "Take photo" does nothing | In-app browser (Instagram, etc.) | Open in Safari or Chrome |
| No detections | Wrong crop, bad lighting, or confidence too high | Lower `CONFIDENCE_THRESHOLD`; frame the leaf closer |
| Wrong objects detected | Using `coco_demo` preset | Set `MODEL_PRESET=plant_disease` |
| Slow first request | Model downloading at startup | Run `python scripts/download_model.py` first |
| Stale UI after update | Browser cache | Hard refresh; server sends `Cache-Control: no-store` on `/` |

---

## Limitations

- **Not a medical/agronomic diagnosis tool** — results are ML predictions for demo/inspection assistance only.
- **Weed-specific models** are uncommon in public hubs; this repo defaults to **crop disease** detection. For dedicated weed detection, train a custom YOLO model and set `MODEL_PRESET=custom`.
- **First inference is slow** on CPU (~1–5 s depending on hardware). GPU (`DEVICE=cuda:0`) is much faster.
- **Model generalization** depends on training data; field photos of crops not in the training set may produce false negatives.
- **Self-signed HTTPS** triggers browser warnings — acceptable for local dev, not for production.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + Uvicorn |
| ML | Ultralytics YOLO (PyTorch) |
| Image I/O | Pillow, NumPy |
| Config | Pydantic Settings |
| Weights | Hugging Face Hub |
| Frontend | Vanilla HTML/CSS/JS |

---

## License

MIT (or your chosen license). Model weights are subject to their respective Hugging Face repository licenses.
