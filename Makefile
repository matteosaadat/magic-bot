# ===============================================
# Makefile — Developer ergonomics for lint/format

# ===============================================
# Minimal Makefile: build-only, test, and build
# ===============================================
PY := python
PYTHONPATH_ABS := $(shell pwd)

.PHONY: build-only test build

# 1) Build image only (no checks)
build-only:
	docker build -t magic-bot:local .

# 2) Run tests (passes if no tests collected)
test:
	@echo ">> Running pytest"
	@PYTHONPATH=$(PYTHONPATH_ABS) $(PY) -m pytest -q || code=$$?; if [ $$code -eq 5 ]; then echo ">> No tests collected; treating as pass"; exit 0; else exit $$code; fi

# 3) Full build: lint -> test -> docker build (fails fast)
build:
	@echo ">> Linting with Ruff"
	ruff check .
	@echo ">> Tests"
	$(MAKE) test
	@echo ">> Docker build"
	$(MAKE) build-only


# -----------------------------------------------

# 1) See everything (no fixing)
lint:
	ruff check . --statistics

# 2) See only changed files (since last commit)
lint-changed:
	@git diff --name-only HEAD | grep -E '\.py$$' | xargs -r ruff check --statistics

# 3) Manually fix in IDE, then re-check
check:
	ruff check .
	black --check .

# 4) Let Ruff auto-fix ONLY safe things (imports, unused vars, simple transforms)
fix-safe:
	ruff check . --fix

# 5) Let Ruff auto-fix a single rule across the repo (e.g., unused imports F401)
fix-rule:
	@echo "Usage: make fix-rule CODE=F401"
	@test -n "$(CODE)" || (echo "Missing CODE=..." && exit 1)
	ruff check . --select $(CODE) --fix

# 6) Let Black format everything (when you choose)
format:
	black .

# 7) Full belt-and-suspenders after you’re happy
all: fix-safe fmt check

