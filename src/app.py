# src/app.py
from fastapi import FastAPI

from src.settings import settings

app = FastAPI(title=settings.ENV)  # or title=settings.ENV


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "env": settings.ENV,
        "debug": settings.DEBUG,
        "app": settings.ENV,
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.ENV}


@app.get("/")
def hello():
    return {"message": "hello v00.02"}
