.PHONY: setup train-vision train-audio train-text train-fusion lint test docker-build docker-up docker-down

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

train-vision:
	python scripts/train_vision.py

train-audio:
	python scripts/train_audio.py

train-text:
	python scripts/train_text.py

train-fusion:
	python scripts/train_fusion.py

lint:
	ruff check src scripts


frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

rag-build:
	python scripts/build_vector_store.py --input data/knowledge_base --output models/text/knowledge_base

test:
	pytest

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down
