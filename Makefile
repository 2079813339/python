.PHONY: help venv install install-all lint fmt type test test-ml cov e2e api worker frontend dev up down migrate clean

PY := .venv/Scripts/python.exe
ifeq ($(OS),)
	PY := .venv/bin/python
endif

help:
	@echo "make venv          - create virtualenv"
	@echo "make install       - install core deps (offline-capable)"
	@echo "make install-all   - install all optional extras"
	@echo "make lint          - ruff check"
	@echo "make fmt           - ruff format + fix"
	@echo "make type          - mypy"
	@echo "make test          - unit + integration tests"
	@echo "make test-ml       - ML regression gate tests"
	@echo "make cov           - coverage report"
	@echo "make e2e           - end-to-end synthetic pipeline smoke"
	@echo "make api           - run FastAPI dev server"
	@echo "make frontend      - run frontend dev server"
	@echo "make dev           - full local stack via docker compose"
	@echo "make migrate       - alembic upgrade head"

venv:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip

install:
	$(PY) -m pip install -e ".[dev]"

install-all:
	$(PY) -m pip install -e ".[dev,geo,store,queue,dl,krige,explain,tracking,auth]"

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest tests/unit tests/integration -m "not ml"

test-ml:
	$(PY) -m pytest tests -m "ml or integration"

cov:
	$(PY) -m pytest tests/unit tests/integration -m "not ml" --cov=satemis --cov-report=term-missing --cov-report=html

e2e:
	$(PY) -m pytest tests/e2e -m "e2e"

api:
	$(PY) -m uvicorn satemis.api.main:app --reload --port 8000

worker:
	$(PY) -m celery -A satemis.workers.celery_app worker -l info -Q main,co,o3,inversion

frontend:
	npm --prefix frontend run dev

migrate:
	$(PY) -m alembic -c alembic.ini upgrade head

dev:
	docker compose up --build

down:
	docker compose down -v

clean:
	Remove-Item -Recurse -Force .pytest_cache,.mypy_cache,.ruff_cache,htmlcov -ErrorAction SilentlyContinue
