# src/app.py
from fastapi import FastAPI
from src.settings import settings

app = FastAPI(title=settings.app_name)          # or title=settings.APP_NAME

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "env": settings.ENV,
        "debug": settings.DEBUG,
        "app": settings.app_name,
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env}

@app.get("/")
def hello():
    return {"message": "hello v1.2"}
