# ============================================================
# Matteo-bot FastAPI App
# ------------------------------------------------------------
# This app wires everything together:
#   - Multi-bot routing (via bots/<bot>/bot.yaml)
#   - Generic search + generator layers
#   - Support for Ollama, OpenAI, or Echo clients
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from functools import lru_cache
import yaml
import os

# --- Local imports ---
from src.settings import settings
from src.search import Retriever, ContextChunk, Persona
from src.generate import ChatGenerator, Message, ChatResponse, ModelParams
from src.generate.clients.echo_dev_client import EchoDevClient
# ------------------------------------------------------------
# 🔧 Model client selection
# ------------------------------------------------------------
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if USE_OLLAMA:
    from src.generate.clients.ollama_client import OllamaClient
    model_client = OllamaClient(model=os.getenv("OLLAMA_MODEL", "mistral:7b-instruct"))
elif OPENAI_KEY:
    from src.generate.clients.openai_client import OpenAIClient
    model_client = OpenAIClient(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
else:
    model_client = EchoDevClient()

chat_gen = ChatGenerator(model_client=model_client)

# ------------------------------------------------------------
# 🧠 Helpers: load bot config dynamically
# ------------------------------------------------------------
BOTS_DIR = "bots"

@lru_cache(maxsize=16)
def load_bot_config(bot_name: str) -> dict:
    path = os.path.join(BOTS_DIR, bot_name, "bot.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bot config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def resolve_bot_paths(bot_name: str) -> dict:
    cfg = load_bot_config(bot_name)
    paths = cfg.get("paths", {})
    return {
        "raw_dir": paths.get("raw_dir", f"bots/{bot_name}/data/raw"),
        "db_path": paths.get("db_path", f"bots/{bot_name}/data/db/{bot_name}.db"),
        "faiss_path": paths.get("faiss_path", f"bots/{bot_name}/data/index/faiss.index"),
    }

# ------------------------------------------------------------
# 🚀 FastAPI init
# ------------------------------------------------------------
app = FastAPI(title="Matteo-bot API", version="0.2")

# ------------------------------------------------------------
# 📦 Pydantic models
# ------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    bot: Optional[str] = "portfolio"
    message: str
    history: Optional[List[ChatTurn]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    persona: Optional[str] = None

class ChatPayload(BaseModel):
    text: str
    citations: List[str]
    meta: Dict[str, Any]

class RetrieveResponse(BaseModel):
    query: str
    docs: List[Dict[str, Any]]

# ------------------------------------------------------------
# 💬 Main chat route
# ------------------------------------------------------------
@app.post("/chat", response_model=ChatPayload)
def chat(req: ChatRequest):
    bot = req.bot or "portfolio"
    try:
        bot_cfg = load_bot_config(bot)
        paths = resolve_bot_paths(bot)
        s_cfg = bot_cfg.get("search", {})
        g_cfg = bot_cfg.get("generate", {})

        # --- 1) Retriever per bot ---
        retriever = Retriever(
            db_path=paths["db_path"],
            faiss_path=paths["faiss_path"],
            top_k=s_cfg.get("top_k", 6),
        )

        # --- 2) Retrieve context ---
        context_chunks: List[ContextChunk] = retriever.retrieve(req.message)

        # --- 3) Persona ---
        persona_key = req.persona or s_cfg.get("persona_key", "matteo-default")
        persona = retriever.load_persona(persona_key)

        # --- 4) Generator params ---
        temperature = req.temperature or g_cfg.get("temperature", 0.3)
        max_tokens = req.max_tokens or g_cfg.get("max_tokens", 1000)
        model_override = req.model or g_cfg.get("model")

        if model_override and hasattr(model_client, "set_model"):
            model_client.set_model(model_override)

        # --- 5) Build history ---
        history = [Message(**h.model_dump()) for h in (req.history or [])]

        # --- 6) Generate ---
        out: ChatResponse = chat_gen.chat(
            user_message=req.message,
            history=history,
            context=context_chunks,
            persona=persona,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return ChatPayload(
            text=out.text,
            citations=out.citations,
            meta={
                "bot": bot,
                "persona": persona_key,
                "model": getattr(model_client, "model", None),
                "db_path": paths["db_path"],
                "faiss_path": paths["faiss_path"],
                "engine": getattr(model_client, "__class__", type(model_client)).__name__,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# 🔎 Retrieval-only route
# ------------------------------------------------------------
@app.get("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(q: str = Query(..., description="Search query"), bot: str = "portfolio", top_k: int = 6):
    try:
        paths = resolve_bot_paths(bot)
        retriever = Retriever(db_path=paths["db_path"], faiss_path=paths["faiss_path"], top_k=top_k)
        chunks = retriever.retrieve(q)
        docs = [{"id": c.id, "text": c.text, "score": c.score, "source": c.source} for c in chunks]
        return {"query": q, "docs": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# 🤖 Bots discovery
# ------------------------------------------------------------
@app.get("/bots")
def list_bots():
    items = []
    for name in os.listdir(BOTS_DIR):
        path = os.path.join(BOTS_DIR, name, "bot.yaml")
        if os.path.exists(path):
            try:
                cfg = load_bot_config(name)
                items.append({
                    "key": name,
                    "description": cfg.get("description", ""),
                    "paths": cfg.get("paths", {}),
                })
            except Exception:
                continue
    return {"bots": items}

# ------------------------------------------------------------
# 🧭 Health checks
# ------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "env": settings.ENV,
        "debug": settings.DEBUG,
        "app": "Matteo-bot",
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": settings.ENV}

@app.get("/")
def hello():
    return {"message": "Matteo-bot service running."}
