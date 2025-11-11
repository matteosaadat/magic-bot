Perfect — here’s a **consolidated developer document** for your `src/search` folder.
It explains the purpose of the *search layer*, the structure, each file’s role, and how the components interact.

---

# 📘 **Search Layer — Design & Developer Guide**

**Folder:** `src/search/`

---

## 🧭 **Purpose**

The **Search layer** provides **hybrid retrieval** capabilities for any bot.
It finds the most relevant knowledge chunks (context) from a local knowledge base by combining:

* **Lexical Search (FTS5/BM25)** → matches keywords and text directly.
* **Vector Search (FAISS)** → matches semantic meaning using embeddings.

The retriever merges these results, ranks them, and provides **context chunks** for the generator (LLM) to use when answering a user’s question.

This layer is **shared** by all bots (portfolio, market, etc.) — each bot has its own DB and FAISS index, but the same retrieval logic.

---

## 🗂️ **Folder Structure**

```
src/search/
│
├── __init__.py
├── retriever.py        # Hybrid retriever: lexical + vector search
├── rank.py             # Merge + rank algorithm for combining results
├── types.py            # Dataclasses (ContextChunk, Persona)
└── personas.yaml       # Optional: predefined bot/persona styles
```

---

## 🧩 **File-by-File Overview**

### **1️⃣ retriever.py** — Core Hybrid Retriever

Handles all retrieval logic.

**Responsibilities:**

* Connect to `documents` FTS5 table (SQLite)
* Load FAISS vector index and `ids.npy`
* Query FTS5 (`MATCH` + BM25) and FAISS
* Merge and rank lexical + vector results
* Optionally load personas from `personas.yaml`
* Embed queries using Ollama (default `bge-m3:latest`)

**Key Classes/Functions:**

```python
class Retriever:
    def __init__(self, db_path, faiss_path, top_k=6)

    retrieve(query, alpha=0.5) -> List[ContextChunk]
        # 1) Lexical search with BM25
        # 2) Vector search via FAISS
        # 3) Merge & rank results
        # Returns top ContextChunk list

    load_persona(key) -> Persona
        # Loads persona style/config from personas.yaml

    close()
        # Cleanly closes SQLite connection
```

**Helpers:**

* `_embed_query_ollama(text)` → gets query embedding from Ollama `/api/embeddings`
* `_fts5_safe_query(raw)` → sanitizes input to avoid SQL errors (e.g., hyphens)
* `_load_faiss_ids()` → loads `ids.npy` (FAISS row → documents.id)

**Environment Variables:**

```bash
OLLAMA_HOST="http://localhost:11434"
EMBED_MODEL="bge-m3:latest"
```

---

### **2️⃣ rank.py** — Result Merger and Ranker

Defines how to combine lexical and vector results into a single ranked list.

**Example Logic:**

```python
def merge_and_rank(lex_results, vec_results, alpha=0.5, top_k=6):
    # Normalize both score sets
    # Combine via weighted average:  score = alpha * vec + (1 - alpha) * lex
    # Deduplicate by source/id
    # Return top_k ContextChunks
```

**Purpose:**
Ensures fair blending of lexical (exact match) and semantic (vector) signals.
The `alpha` parameter tunes emphasis:

* `alpha=0` → only lexical
* `alpha=1` → only vector
* `alpha=0.5` → balanced

---

### **3️⃣ types.py** — Shared Data Structures

Defines the objects that flow between retriever and generator.

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ContextChunk:
    id: str
    text: str
    source: str
    score: float
    meta: Dict[str, Any]

@dataclass
class Persona:
    key: str
    name: str
    style: str
    directives: str
    meta: Dict[str, Any]
```

These are lightweight containers passed downstream to the generator (`ChatGenerator`) for context and tone.

---

### **4️⃣ personas.yaml** — Personality / Tone Profiles

A simple YAML file storing optional LLM “personas”.

Example:

```yaml
matteo-default:
  name: Matteo Bot
  style: professional
  directives: |
    You are Matteo-bot, an AI assistant specialized in explaining software architecture and engineering projects clearly.
    Maintain an encouraging and practical tone.

friendly-helper:
  name: Friendly Helper
  style: conversational
  directives: |
    Be upbeat and friendly. Simplify technical details when possible.
```

Each persona can be referenced by key (e.g. `"matteo-default"`) in a bot’s config or retrieval query.

---

## ⚙️ **How Retrieval Works**

```mermaid
flowchart LR
    A[User Query] --> B[Retriever.retrieve()]
    B --> C1[Lexical Search (FTS5 + BM25)]
    B --> C2[Vector Search (FAISS + bge-m3)]
    C1 --> D[merge_and_rank()]
    C2 --> D
    D --> E[Top ContextChunk list]
    E --> F[ChatGenerator.chat()]
```

### Step-by-Step:

1. User sends a query to `/chat`.
2. Retriever loads the correct bot DB + FAISS index.
3. Query is embedded (Ollama `bge-m3` → vector).
4. Performs both:

   * FTS5 search (`MATCH`) for lexical hits.
   * FAISS search for semantic hits.
5. Results merged and sorted by composite score.
6. Returns top N chunks to the generator as context.

---

## 🧪 **Testing & Validation**

### **Unit test:**

* `tests/test_search.py` → standalone test of search logic.
* Builds a temp DB + FAISS index and verifies merged results.

Run:

```bash
python -m tests.test_search
```

### **E2E test:**

* `tests/test_e2e.py` → runs full pipeline (retriever → generator).

Run:

```bash
python -m tests.test_e2e
```

### **Server test:**

Once the app is running:

```bash
curl "http://127.0.0.1:8000/retrieve?q=watchtower&bot=portfolio"
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"bot":"portfolio","message":"Explain Matteo-bot architecture"}'
```

---

## 🔐 **Dependencies**

| Library     | Purpose                | Install                 |
| ----------- | ---------------------- | ----------------------- |
| `faiss-cpu` | Vector search backend  | `pip install faiss-cpu` |
| `sqlite3`   | Local lexical DB       | built-in                |
| `requests`  | Ollama API client      | `pip install requests`  |
| `pyyaml`    | Persona config         | `pip install pyyaml`    |
| `numpy`     | Embeddings, vector ops | `pip install numpy`     |

---

## 📄 **Bot-Specific Data Layout**

Each bot (e.g., `bots/portfolio`) stores its own data:

```
bots/portfolio/
└── data/
    ├── db/
    │   └── portfolio.db       ← SQLite FTS5 with table `documents`
    └── index/
        ├── faiss.index        ← FAISS binary index
        └── ids.npy            ← FAISS row → document.id mapping
```

The retriever doesn’t care which bot — just use the right paths.

---

## 🧠 **Developer Notes**

* Keep `documents` schema consistent: `(id, text, source)`.
* Ensure `ids.npy` aligns exactly with FAISS vector order.
* Embedding model used for FAISS must match `EMBED_MODEL` used at query time.
* For safety, always sanitize text before `MATCH` (done via `_fts5_safe_query`).

---

## ✅ **Summary**

| Component       | Role                             | Output                    |
| --------------- | -------------------------------- | ------------------------- |
| `retriever.py`  | Combines lexical + vector search | `List[ContextChunk]`      |
| `rank.py`       | Normalizes & merges result lists | Ranked hybrid list        |
| `types.py`      | Defines shared structures        | `ContextChunk`, `Persona` |
| `personas.yaml` | Persona definitions              | Style & tone configs      |

**End result:**
Your bots can ask any question → the search layer finds the best context → the generator uses that to produce a grounded, context-aware response.

---

Would you like me to generate a **`README.md`** version of this to drop directly into `src/search/` (with formatting + table of contents)?
