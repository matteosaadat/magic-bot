Perfect 👍 — here are the **remaining two documentation files** you can drop directly into your project:

---

## ⚙️ **`src/ingest/README.md`**

```markdown
# ⚙️ Ingest Pipeline — Developer Guide

**Folder:** `src/ingest/`

---

## 🧭 Overview

The **Ingest pipeline** is responsible for **building the knowledge base** that powers all bots.  
It processes raw data (Markdown, text, HTML, PDFs, etc.) through multiple stages — cleaning, paraphrasing, deduplication, embedding, and indexing — producing:

- a **SQLite FTS5 database** (`documents` table)  
- a **FAISS vector index** (`faiss.index`)  
- a **row mapping** file (`ids.npy`)

These outputs form the searchable dataset used by the `search` and `generator` layers.

---

## 📘 Table of Contents

1. [Pipeline Stages](#-pipeline-stages)  
2. [Folder Structure](#-folder-structure)  
3. [File-by-File Breakdown](#-file-by-file-breakdown)  
4. [Output Artifacts](#-output-artifacts)  
5. [Usage](#-usage)  
6. [Dependencies](#-dependencies)  
7. [Developer Notes](#-developer-notes)  
8. [Summary](#-summary)

---

## 🔄 Pipeline Stages

```

0_collect_extract → 1_normalize → 2_paraphrase → 3_dedup → 4_embed → 5_index → 6_search

```

| Stage | Purpose |
|--------|----------|
| **0_collect_extract** | Gather raw text or structured data |
| **1_normalize** | Clean text and normalize Unicode/whitespace |
| **2_paraphrase** | Expand and restate data for language variety |
| **3_dedup** | Remove redundant or similar records (SimHash/cosine) |
| **4_embed** | Convert text into vector embeddings |
| **5_index** | Build FAISS + SQLite indexes |
| **6_search** | Optional: test retrieval and accuracy |

Each stage can be run independently or as a full pipeline via `run_ingest.py`.

---

## 🗂 Folder Structure

```

src/ingest/
│
├── **init**.py
├── run_ingest.py         # Orchestrator CLI
├── 0_collect_extract.py
├── 1_normalize.py
├── 2_paraphrase.py
├── 3_dedup.py
├── 4_embed.py
├── 5_index.py
├── 6_search.py            # Retrieval QA tester
├── makefile               # Developer shortcuts
└── ingest.ai              # AI assistant guide (for Cursor IDE)

````

---

## 🧩 File-by-File Breakdown

### `run_ingest.py`
Coordinates the full ingest pipeline.

Example usage:
```bash
python src/ingest/run_ingest.py \
  --src bots/portfolio/raw \
  --db bots/portfolio/data/db/portfolio.db \
  --faiss bots/portfolio/data/index/faiss.index
````

---

### `1_normalize.py`

Cleans raw text:

* removes control characters
* unescapes HTML entities
* normalizes Unicode (NFKC)
* preserves fenced code blocks
* collapses excessive whitespace/newlines

---

### `2_paraphrase.py`

Generates alternate phrasings for data augmentation.
Can use:

* Ollama model (local paraphraser)
* Cached paraphrasing from earlier runs

---

### `3_dedup.py`

Removes duplicate or near-duplicate chunks using:

* **SimHash** for token overlap
* or cosine similarity thresholds

---

### `4_embed.py`

Embeds text into high-dimensional vectors using:

* Ollama API (`bge-m3:latest`)
* FAISS-compatible NumPy arrays

---

### `5_index.py`

Creates:

* `faiss.index` for fast vector retrieval
* `ids.npy` (row → doc.id mapping)
* `portfolio.db` SQLite FTS5 table with (id, text, source)

---

### `6_search.py`

Standalone script for testing retrieval quality before serving the bot.

---

## 🧾 Output Artifacts

| File           | Description                      |
| -------------- | -------------------------------- |
| `portfolio.db` | SQLite FTS5 knowledge store      |
| `faiss.index`  | Vector index of embeddings       |
| `ids.npy`      | FAISS row-to-document ID mapping |

---

## 🧪 Usage

**Full run**

```bash
make ingest
# or
python src/ingest/run_ingest.py --src src/knowledge --db bots/portfolio/data/db/portfolio.db --faiss bots/portfolio/data/index/faiss.index
```

**Stage-only example**

```bash
make ingest ARGS="--paraphrase-only"
```

---

## 🧩 Dependencies

| Library        | Purpose          |
| -------------- | ---------------- |
| `faiss-cpu`    | Vector indexing  |
| `sqlite-utils` | Database access  |
| `numpy`        | Vector storage   |
| `simhash`      | Deduplication    |
| `requests`     | Ollama API calls |
| `tqdm`         | Progress display |

---

## 🧠 Developer Notes

* Always rebuild `ids.npy` alongside `faiss.index`.
* Each chunk must have a unique `id` for mapping consistency.
* The ingest pipeline can run **offline** with local Ollama models.
* `ingest.ai` provides Cursor IDE with step-by-step creation instructions.
* Ensure `bm25(documents)` works → requires `fts5` in SQLite build.

---

## ✅ Summary

| File              | Role                     |
| ----------------- | ------------------------ |
| `run_ingest.py`   | Orchestrates all stages  |
| `1_normalize.py`  | Text normalization       |
| `2_paraphrase.py` | Language augmentation    |
| `3_dedup.py`      | Duplicate filtering      |
| `4_embed.py`      | Embedding computation    |
| `5_index.py`      | FAISS/SQLite index build |
| `6_search.py`     | Test search quality      |
| `makefile`        | Local CLI shortcuts      |
| `ingest.ai`       | AI assistant hints       |

---

