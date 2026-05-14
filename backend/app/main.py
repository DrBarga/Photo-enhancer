import os
import traceback

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings


app = FastAPI(title=settings.app_name)
app.state.startup_error = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


try:
    from app.api.routes import router

    app.include_router(router, prefix="/api")
    if os.getenv("VERCEL"):
        app.include_router(router)
except Exception as exc:  # noqa: BLE001
    app.state.startup_error = traceback.format_exc(limit=10)

    @app.get("/api/startup-error")
    def startup_error() -> dict[str, str]:
        return {"status": "degraded", "error": str(exc), "traceback": app.state.startup_error}

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    def api_unavailable(path: str) -> None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Backend API started, but the image-processing runtime failed to load. "
                "Open /api/startup-error or Vercel Function logs for the import traceback."
            ),
        )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/api/health",
    }
