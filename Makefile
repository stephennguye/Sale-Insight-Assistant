.PHONY: setup ingest analytics train rag-index serve ui test lint

setup:
	python -m venv venv && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

ingest:
	python -m src.ingest

analytics:
	python -m src.analytics

train:
	python -m src.train_model

rag-index:
	python -m src.rag_index

serve:
	uvicorn src.app:app --reload --port 8000

ui:
	streamlit run ui.py

test:
	pytest -q

lint:
	flake8 src