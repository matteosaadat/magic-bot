````

---

## 🤖 **`bots/README.md`**

```markdown
# 🤖 Bots Folder — Developer Guide

**Folder:** `bots/`

---

## 🧭 Overview

Each bot (e.g., **Portfolio Bot**, **Market Bot**) is an independent knowledge domain that uses shared logic from the `ingest`, `search`, and `generate` layers.

Bots contain only **data and configuration** — not code.  
Each bot can run its own ingest process, build its own FAISS/SQLite dataset, and serve unique personalities or styles.

---

## 📘 Table of Contents
1. [Purpose](#-purpose)
2. [Structure](#-structure)
3. [Bot Folder Layout](#-bot-folder-layout)
4. [bot.yaml Configuration](#-botyaml-configuration)
5. [How Bots Use Shared Layers](#-how-bots-use-shared-layers)
6. [Example Commands](#-example-commands)
7. [Developer Notes](#-developer-notes)
8. [Summary](#-summary)

---

## 🎯 Purpose

Organize bot-specific knowledge and configurations to make the system modular.  
You can add or remove bots without changing the shared backend.

---

## 🗂 Structure

````

bots/
│
├── portfolio/
│   ├── bot.yaml              # Configuration for this bot
│   ├── raw/                  # Original source content
│   └── data/
│       ├── db/               # SQLite FTS5 (documents table)
│       └── index/            # FAISS + ids.npy
│
├── market/
│   └── (future bot) ...
│
└── README.md                 # This document

````

---

## 🧩 Bot Folder Layout

| Folder | Purpose |
|---------|----------|
| `raw/` | Raw or Markdown source files |
| `data/db/` | SQLite FTS5 database with documents |
| `data/index/` | FAISS index + ids.npy |
| `bot.yaml` | Bot-level config (paths, search settings, persona) |

---

## ⚙️ bot.yaml Configuration

Defines how the bot connects to its database and model defaults.

**Example:**
```yaml
bot:
  name: Portfolio Bot
  description: Answers questions about Matteo’s projects and architecture.
paths:
  db_path: bots/portfolio/data/db/portfolio.db
  faiss_path: bots/portfolio/data/index/faiss.index
search:
  top_k: 6
  alpha: 0.5
  persona_key: matteo-default
generate:
  temperature: 0.3
  max_tokens: 800
````

---

## 🧠 How Bots Use Shared Layers

```mermaid
flowchart LR
    A[User Query] --> B[Retriever (src/search)]
    B --> C[ContextChunks]
    C --> D[ChatGenerator (src/generate)]
    D --> E[Answer + Citations]
```

* **Ingest** builds the bot’s dataset.
* **Search** retrieves relevant context.
* **Generator** turns that into natural language output.

---

## 🧪 Example Commands

**Rebuild Portfolio DB**

```bash
python src/ingest/run_ingest.py \
  --src bots/portfolio/raw \
  --db bots/portfolio/data/db/portfolio.db \
  --faiss bots/portfolio/data/index/faiss.index
```

**Test Chat**

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"bot":"portfolio","message":"Explain CI/CD setup"}'
```

---

## 💡 Developer Notes

* Use lowercase folder names for bot identifiers.
* Each bot is self-contained and can be deployed separately.
* All bots share `src/search/personas.yaml` for consistent tone options.
* Multiple bots can run on the same FastAPI instance.

---

## ✅ Summary

| Component               | Role                         |
| ----------------------- | ---------------------------- |
| `bot.yaml`              | Configuration for the bot    |
| `data/db`               | FTS5 database for retrieval  |
| `data/index`            | FAISS vector index + ids.npy |
| `raw/`                  | Original data to ingest      |
| `portfolio/`, `market/` | Example bots                 |

---

````

