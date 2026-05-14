$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SamCheckpoint = "D:\models\sam_vit_b_01ec64.pth"

if (!(Test-Path -LiteralPath $Python)) {
    throw "Python virtual environment was not found: $Python"
}

if (!(Test-Path -LiteralPath $SamCheckpoint)) {
    throw "SAM checkpoint was not found: $SamCheckpoint"
}

$env:AI_LIGHT_ML_ENABLED = "true"
$env:AI_LIGHT_DEPTH_PROVIDER = "depth_anything"
$env:AI_LIGHT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
$env:AI_LIGHT_SEGMENTATION_PROVIDER = "sam"
$env:AI_LIGHT_SAM_CHECKPOINT = $SamCheckpoint
$env:AI_LIGHT_SAM_MODEL_TYPE = "vit_b"
$env:AI_LIGHT_CLIP_PROVIDER = "auto"
$env:AI_LIGHT_CLIP_MODEL = "openai/clip-vit-base-patch32"
$env:AI_LIGHT_CLASSIFIER_PATH = "backend\models\problem_classifier.joblib"
$env:AI_LIGHT_INPAINT_PROVIDER = "none"

foreach ($port in @(8000, 5173)) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Start-Sleep -Seconds 1

Start-Process -FilePath $Python `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden

Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden

Write-Host "AI Light ML mode started."
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host "Backend docs: http://127.0.0.1:8000/docs"
Write-Host "ML status: http://127.0.0.1:8000/api/ml/status"
