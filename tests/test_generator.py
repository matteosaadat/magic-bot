# ===============================================
# Simple standalone test for the ChatGenerator
# ===============================================

import os
from src.generate import ChatGenerator, Message
from src.generate.clients.echo_dev_client import EchoDevClient
# If testing Ollama or OpenAI, import the client you want:
# from src.generate.clients.ollama_client import OllamaClient
# from src.generate.clients.openai_client import OpenAIClient
from src.search.types import ContextChunk, Persona

# --- choose model client ---
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

# --- instantiate generator ---
gen = ChatGenerator(model_client=model_client)

# --- dummy data ---
query = "What is Matteo-bot?"
context = [
    ContextChunk(id="A1", text="Matteo-bot is a portfolio chatbot built with FastAPI and Ollama.", score=0.88, source="portfolio.md"),
    ContextChunk(id="B2", text="It uses FAISS and SQLite FTS5 for hybrid retrieval.", score=0.79, source="architecture.md"),
]

persona = Persona(
    key="matteo-default",
    name="Matteo-bot (Default)",
    style="friendly, concise, senior engineer",
    directives="Focus on clarity and reasoning.",
)

# --- run generator ---
out = gen.chat(
    user_message=query,
    history=[],
    context=context,
    persona=persona,
    temperature=0.3,
    max_tokens=300,
)

print("\n🧠 Generated Text:\n")
print(out.text)
print("\n📚 Citations:", out.citations)
print("\n⚙️ Meta:", out.meta)
