# Use the project's virtualenv Python
PY := .venv/bin/python

.PHONY: run predict install freeze clean

# Launch the Streamlit app
run: install
	$(PY) -m streamlit run app.py

# CLI example; override with: make predict SEASON=2024 WEEKS="3 4 5"
SEASON ?= 2024
WEEKS  ?= 1 2
predict: install
	$(PY) src/predict.py --season $(SEASON) --weeks $(WEEKS)

# Ensure venv exists and deps are installed (idempotent)
install:
	test -x $(PY) || python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# Update requirements.txt from your current env
freeze:
	$(PY) -m pip freeze > requirements.txt

# Clean caches
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + ; \
	rm -rf .pytest_cache
