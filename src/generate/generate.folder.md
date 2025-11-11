

---

## 🧠 **`src/generate/README.md`**

```markdown
# 🧠 Generator Layer — Developer Guide

**Folder:** `src/generate/`

---

## 🧭 Overview

The **Generator layer** is the final stage in the pipeline.  
It receives:
- the user message  
- chat history  
- retrieved context chunks  
- and (optionally) a persona  

Then it produces a coherent **natural language answer** using an underlying model (Ollama, OpenAI, or a mock Echo client).

The generator is **model-agnostic** and can switch between local and cloud LLMs dynamically.

---

## 📘 Table of Contents
1. [Purpose](#-purpose)
2. [Folder Structure](#-folder-structure)
3. [File-by-File Breakdown](#-file-by-file-breakdown)
4. [How It Works](#-how-it-works)
5. [Testing](#-testing)
6. [Dependencies](#-dependencies)
7. [Developer Notes](#-developer-notes)
8. [Summary](#-summary)

---

## 🎯 Purpose
Unify all model clients under one abstraction (`ModelClient`) so that the app can generate text regardless of backend (Ollama, OpenAI, etc.).  
The generator handles:
- prompt construction (context + persona + user message)
- message history
- streaming or single-shot responses
- meta info (citations, temperature, tokens)

---

## 🗂 Folder Structure

```

src/generate/
│
├── **init**.py
├── generator.py            # ChatGenerator core class
├── base_client.py          # Abstract interface for model clients
├── clients/
│   ├── ollama_client.py    # Local model (via Ollama)
│   ├── openai_client.py    # Remote API (OpenAI-compatible)
│   └── echo_dev_client.py  # Mock client for testing
├── types.py                # Data classes for message and responses
└── persona.yaml            # Optional, local personas or style presets

````

---

## 🧩 File-by-File Breakdown

### `generator.py`
Main logic orchestrating model calls.

**Responsibilities:**
- Build prompts (persona + retrieved context)
- Call chosen model client
- Handle temperature, token limits, etc.
- Wrap result as `ChatResponse`

**Key Class**
```python
ChatGenerator(model_client)
ChatGenerator.chat(user_message, history, context, persona, temperature, max_tokens)
````

---

### `base_client.py`

Defines the minimal API all model clients must implement.

```python
class BaseModelClient(ABC):
    def generate(self, prompt: str, **kwargs) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...
```

---

### `clients/ollama_client.py`

Connects to a **local Ollama** server.
Uses REST API calls to `/api/generate` and `/api/embeddings`.

```bash
OLLAMA_HOST="http://localhost:11434"
OLLAMA_MODEL="mistral:7b-instruct"
```

---

### `clients/openai_client.py`

Wrapper for **OpenAI-compatible APIs**.

Supports:

* `OPENAI_API_KEY`
* `OPENAI_MODEL` (e.g., `gpt-4o-mini`)
* Standard `temperature` and `max_tokens` args

---

### `clients/echo_dev_client.py`

Offline testing model — echoes input in structured form.

Example output:

```
[ECHO RESPONSE]
User question: ...
Context:
[A1] ... [B2] ...
```

Used in tests or when no LLM is configured.

---

### `types.py`

Defines generator-level structures.

```python
@dataclass
class Message:
    role: str
    content: str

@dataclass
class ChatResponse:
    text: str
    citations: List[str]
    meta: Dict[str, Any]
```

---

## ⚙️ How It Works

```mermaid
flowchart LR
    A[User Query + ContextChunks] --> B[ChatGenerator]
    B --> C[Prompt Builder]
    C --> D[Model Client (Ollama/OpenAI/Echo)]
    D --> E[ChatResponse]
```

---

## 🧪 Testing

**Standalone test**

```bash
python -m tests.test_generator
```

You should see an “ECHO RESPONSE” message when using `EchoDevClient`.

**Integration test**
Run:

```bash
python -m tests.test_e2e
```

---

## 🧩 Dependencies

| Package    | Purpose                                |
| ---------- | -------------------------------------- |
| `requests` | API calls to Ollama/OpenAI             |
| `numpy`    | Embeddings handling                    |
| `pydantic` | Type safety for request models         |
| `fastapi`  | Used indirectly for server integration |

---

## 💡 Developer Notes

* Generator is modular — add new clients under `clients/`.
* Persona and context injection happen before model call.
* For Ollama streaming, enable `"stream": true` in requests.
* To debug, print the constructed prompt before sending.

---

## ✅ Summary

| File             | Purpose                             |
| ---------------- | ----------------------------------- |
| `generator.py`   | Builds prompts, calls models        |
| `base_client.py` | Interface for all model clients     |
| `clients/*.py`   | Model implementations               |
| `types.py`       | Shared message/response dataclasses |
| `persona.yaml`   | Optional tone/style presets         |

```
