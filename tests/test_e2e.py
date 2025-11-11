# ===============================================
# tests/test_e2e.py
# End-to-end: load bot config → retrieve → generate
# ===============================================
import os, sys, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.search import Retriever
from src.generate import ChatGenerator, Message
from src.generate.clients.echo_dev_client import EchoDevClient

# Choose model client by env
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

def load_bot_cfg(bot="portfolio"):
    path = ROOT / "bots" / bot / "bot.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    bot = os.getenv("BOT", "portfolio")
    q = os.getenv("Q", "What does Matteo-bot do?")
    cfg = load_bot_cfg(bot)
    paths = cfg.get("paths", {})
    db_path = paths.get("db_path", f"bots/{bot}/data/db/{bot}.db")
    faiss_path = paths.get("faiss_path", f"bots/{bot}/data/index/faiss.index")
    s_cfg = cfg.get("search", {})
    g_cfg = cfg.get("generate", {})

    retriever = Retriever(db_path=db_path, faiss_path=faiss_path, top_k=s_cfg.get("top_k", 6))
    chunks = retriever.retrieve(q, alpha=s_cfg.get("alpha", 0.5))
    persona_key = s_cfg.get("persona_key", "matteo-default")
    persona = retriever.load_persona(persona_key)

    gen = ChatGenerator(model_client=model_client)
    out = gen.chat(
        user_message=q,
        history=[],
        context=chunks,
        persona=persona,
        temperature=g_cfg.get("temperature", 0.3),
        max_tokens=g_cfg.get("max_tokens", 800),
    )

    print("\n🧪 E2E RESULT")
    print("Bot:", bot)
    print("Model:", getattr(model_client, "model", "echo-dev"))
    print("Persona:", persona_key)
    print("\nAnswer:\n", out.text)
    print("\nCitations:", out.citations)

if __name__ == "__main__":
    main()
