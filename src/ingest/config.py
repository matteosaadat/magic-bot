from pathlib import Path
from pydantic import BaseModel

# Resolve project root assuming this file lives at matteo-bot/src/ingest/config.py
ROOT = Path(__file__).resolve().parents[2]

class Paths(BaseModel):
    # Data artifacts
    db_path: Path = ROOT / "data" / "db" / "portfolio.db"
    faiss_path: Path = ROOT / "data" / "index" / "faiss.index"
    knowledge_dir: Path = ROOT / "knowledge"

    # Local model binaries/weights
    llama_bin: Path = ROOT / "models" / "llama" / "llama-cli"  # will be built in a later step
    llama_model: Path = ROOT / "models" / "llama" / "Llama-3.1-8B-Instruct.Q4_K_M.gguf"

    # Embedding model identifier (downloaded on first use)
    embed_model_name: str = "sentence-transformers/bge-small-en-v1.5"

PATHS = Paths()
