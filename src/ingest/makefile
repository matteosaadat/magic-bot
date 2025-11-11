# ==============================
# Ingest Makefile (src/ingest/)
# ==============================
SHELL := /bin/bash

# Python
PY ?= python3

# This Makefile's folder (…/src/ingest)
INGEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Repo root assumed at one level up from src/
# Adjust if your layout differs.
ROOT := $(abspath $(INGEST_DIR)/..)

# Inputs / Outputs
RAW   ?= $(ROOT)/knowledge
DB    ?= $(ROOT)/data/db/portfolio.db
INDEX ?= $(ROOT)/data/index/faiss.index

# Optional extra args like:
# make ingest ARGS="--disable-normalize --disable-paraphrase --disable-dedup --disable-embed"
ARGS ?=

.PHONY: ingest collect normalize paraphrase dedup embed info paths

paths:
	@echo "INGEST_DIR: $(INGEST_DIR)"
	@echo "ROOT      : $(ROOT)"
	@echo "RAW       : $(RAW)"
	@echo "DB        : $(DB)"
	@echo "INDEX     : $(INDEX)"

ingest:
	@echo ">> Ingest starting (collect→normalize→paraphrase→dedup→embed)"
	@echo ">> CMD: $(PY) \"$(INGEST_DIR)/run_ingest.py\" --src \"$(RAW)\" --db \"$(DB)\" --faiss \"$(INDEX)\" $(ARGS)"
	@$(PY) "$(INGEST_DIR)/run_ingest.py" \
		--src "$(RAW)" \
		--db "$(DB)" \
		--faiss "$(INDEX)" \
		$(ARGS)
	@echo ">> Ingest complete."
	@$(MAKE) info

collect:
	@$(PY) "$(INGEST_DIR)/0_collect_extract.py" --src "$(RAW)"

normalize:
	@$(PY) "$(INGEST_DIR)/1_normalize.py" --src "$(RAW)"

paraphrase:
	@$(PY) "$(INGEST_DIR)/2_paraphrase.py" --src "$(RAW)"

dedup:
	@$(PY) "$(INGEST_DIR)/3_dedup.py" --src "$(RAW)" --db "$(DB)"

embed:
	@$(PY) "$(INGEST_DIR)/4_embed.py" --db "$(DB)" --faiss "$(INDEX)"

info:
	@echo ">> DB:    $(DB)"
	@echo ">> FAISS: $(INDEX)"
