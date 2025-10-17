# ===============================================
# Makefile — Developer ergonomics for lint/format
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
fmt:
	black .

# 7) Full belt-and-suspenders after you’re happy
all: fix-safe fmt check
