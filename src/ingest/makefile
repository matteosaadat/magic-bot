# =========================
# Ingest & Search Makefile
# =========================
# Usage:
#   make help
#   make venv           # create .venv and install deps
#   make ingest         # run the full ingest pipeline (normalize→paraphrase→dedup→embed→index)
#   make search q='watchtower'
#   make rebuild        # clean + ingest
#   make clean          # delete DB + FAISS index
#   make info           # show env + file locations
#
# Notes:
# - Assumes scripts live in src/ingest and data under src/data.
# - Defaults are easy to tweak via variables below.

# --------- Paths ---------
WORKDIR := src
DB      := $(WORKDIR)/data/db/portfolio.db
INDEX   := $(WORKDIR)/data/index/faiss.index
RAW     := $(WORKDIR)/data/raw
TMP     := $(WORKDIR)/data/tmp
NORM    := $(TMP)/normalized
PARA    := $(TMP)/paraphrased
DEDUP   := $(TMP)/deduped

# --------- Python ---------
# Prefer the project venv if it exists; otherwise fallback to python3
PY := $(shell [ -x .venv/bin/python ] && echo ".venv/bin/python" || echo "python3")

# --------- Defaults ---------
q ?= watchtower

# --------- Help ---------
.PHONY: help
help:
	@echo ""
	@echo "Targets:"
	@echo "  make venv        - Create .venv and install dependencies"
	@echo "  make ingest      - Run full ingest pipeline into $(DB) and $(INDEX)"
	@echo "  make search q=.. - Search the index for a query (default: '$(q)')"
	@echo "  make rebuild     - clean + ingest"
	@echo "  make clean       - Remove DB/index and temp artifacts"
	@echo "  make info        - Show Python/env and important paths"
	@echo ""

# --------- Env setup ---------
.PHONY: venv
venv:
	@echo ">> Creating .venv and installing deps..."
	@test -d .venv || python3 -m venv .venv
	@.venv/bin/pip install -U pip
	@.venv/bin/pip install -r requirements.txt

# --------- Ingest (one-shot) ---------
.PHONY: ingest
ingest:
	@echo ">> Ingest starting (normalize→paraphrase→dedup→embed→index)"
	@$(PY) $(WORKDIR)/ingest/run_ingest.py \
		--src $(RAW) \
		--db $(DB) \
		--faiss $(INDEX)
	@echo ">> Ingest complete."
	@$(MAKE) info

# --------- Ingest (step-by-step) ---------
.PHONY: ingest-steps
ingest-steps:
	@echo ">> 1) Normalize"
	@$(PY) $(WORKDIR)/ingest/1_normalize.py --input $(RAW) --output $(NORM)
	@echo ">> 2) Paraphrase"
	@$(PY) $(WORKDIR)/ingest/2_paraphrase.py --input $(NORM) --output $(PARA)
	@echo ">> 3) Dedup"
	@$(PY) $(WORKDIR)/ingest/3_dedup.py --input $(PARA) --output $(DEDUP)
	@echo ">> 4) Embed + Index"
	@$(PY) $(WORKDIR)/ingest/4_embed.py --input $(DEDUP) --db $(DB) --faiss $(INDEX)
	@echo ">> Done."

# --------- Search ---------
.PHONY: search
search:
	@$(PY) $(WORKDIR)/ingest/6_search.py --db $(DB) --faiss $(INDEX) --query "$(q)"

# --------- Clean / Rebuild ---------
.PHONY: clean
clean:
	@echo ">> Removing DB and FAISS index and temp dirs"
	@rm -f $(DB) $(INDEX)
	@rm -rf $(TMP)

.PHONY: rebuild
rebuild: clean ingest

# --------- Info ---------
.PHONY: info
info:
	@echo ">> Python: $(PY)"
	@$(PY) -c "import sys,platform;print('Python',sys.version);print('Platform',platform.platform())"
	@echo "DB:    $(DB)   -> $$([ -f $(DB) ] && echo exists || echo missing)"
	@echo "Index: $(INDEX) -> $$([ -f $(INDEX) ] && echo exists || echo missing)"
