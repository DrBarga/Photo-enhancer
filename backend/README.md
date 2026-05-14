# AI Light Backend

FastAPI backend for local image analysis and correction:

- prompt-guided correction parameters
- luminance, gradient, contrast and edge maps
- problem heatmap
- gradient, reflection and shadow processors
- quality metrics for each processing block

## Run

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

## Run with ML enabled

After optional dependencies and SAM checkpoint are installed, start the full app with:

```powershell
.\start-ml.ps1
```

This starts the backend with Depth Anything, SAM, CLIP and the lightweight classifier enabled. Inpainting stays disabled until an API key is configured.

Useful product endpoints:

- `POST /api/analyze` builds the initial heatmap, depth map, smart masks and ML understanding before correction.
- `POST /api/process` applies local correction only inside detected problem masks.
- `GET /api/ml/status` shows which providers are active: CV fallback, Depth Anything, SAM, CLIP, inpainting and classifier.

## Optional pretrained ML mode

The app works without heavy models. For product-level experiments, install the optional stack:

```powershell
.\.venv\Scripts\python -m pip install -r backend\requirements-ml.txt
```

Enable providers with environment variables:

```powershell
$env:AI_LIGHT_ML_ENABLED="true"
$env:AI_LIGHT_DEPTH_PROVIDER="depth_anything"
$env:AI_LIGHT_SEGMENTATION_PROVIDER="cv"
$env:AI_LIGHT_CLIP_PROVIDER="auto"
```

For SAM, provide a local checkpoint:

```powershell
$env:AI_LIGHT_SEGMENTATION_PROVIDER="sam"
$env:AI_LIGHT_SAM_CHECKPOINT="D:\models\sam_vit_b.pth"
$env:AI_LIGHT_SAM_MODEL_TYPE="vit_b"
```

For Stability AI inpainting:

```powershell
$env:AI_LIGHT_INPAINT_PROVIDER="stability"
$env:STABILITY_API_KEY="..."
# Optional, if Stability changes the endpoint:
$env:AI_LIGHT_STABILITY_INPAINT_ENDPOINT="https://api.stability.ai/v2beta/stable-image/edit/inpaint"
```

Train the small prompt/problem classifier after collecting examples:

```powershell
.\.venv\Scripts\python backend\app\services\classifier_training.py
$env:AI_LIGHT_CLASSIFIER_PATH="backend\models\problem_classifier.joblib"
```

## Build the local evaluation dataset

The product dataset is reproducible. It downloads free media metadata through Wikimedia Commons and creates labelled local problem examples for gradient, shadow and reflection testing:

```powershell
.\.venv\Scripts\python backend\app\services\dataset_builder.py --target-examples 180 --max-sources 80
.\.venv\Scripts\python backend\app\services\classifier_training.py
```

Generated artifacts:

- `backend/data/training/source` - downloaded source images
- `backend/data/training/images` - labelled training/test examples
- `backend/data/training/manifest.jsonl` - labels for classifier training
- `backend/models/problem_classifier_report.json` - classifier validation report
- `backend/data/history` - saved processing history

## Production storage, history and jobs

The backend now runs with safe local fallbacks, but can be switched to production services with environment variables.

Storage providers:

```powershell
# Local default
$env:AI_LIGHT_STORAGE_PROVIDER="local"

# Supabase Storage
$env:AI_LIGHT_STORAGE_PROVIDER="supabase"
$env:SUPABASE_URL="https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="..."
$env:SUPABASE_STORAGE_BUCKET="photo-enhancer"

# S3 or Cloudflare R2
$env:AI_LIGHT_STORAGE_PROVIDER="r2"
$env:AI_LIGHT_S3_BUCKET="photo-enhancer"
$env:AI_LIGHT_S3_ENDPOINT_URL="https://ACCOUNT_ID.r2.cloudflarestorage.com"
$env:AI_LIGHT_S3_ACCESS_KEY_ID="..."
$env:AI_LIGHT_S3_SECRET_ACCESS_KEY="..."
$env:AI_LIGHT_S3_PUBLIC_BASE_URL="https://cdn.example.com"
```

Postgres/Supabase history and job state:

```powershell
$env:AI_LIGHT_POSTGRES_DSN="postgresql://USER:PASSWORD@HOST:5432/postgres"
```

Create the tables from `backend/sql/001_history_jobs.sql`. When this DSN is present, history and job progress are saved in Postgres instead of local JSON files.

Local job fallback:

```powershell
$env:AI_LIGHT_JOB_DIR="backend/data/jobs"
```

For heavy processing, deploy the backend on a persistent worker host such as Render, Railway, Fly, Modal, RunPod or Hugging Face Spaces. Vercel is fine for the frontend, but serverless functions are not ideal for SAM/Depth/CLIP/upscaling workloads.

## Auto Enhance API

- `POST /api/auto_enhance` - synchronous auto plan + enhancement.
- `POST /api/jobs/auto_enhance` - creates a background job.
- `GET /api/jobs/{job_id}` - polls job progress and result.

Auto Enhance combines global corrections (white balance, CLAHE contrast, exposure, denoise, sharpen, dehaze, JPEG cleanup) with local gradient/reflection/shadow processors and semantic protection for faces/text.
