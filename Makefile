# Convenience targets (Linux/macOS). Windows users: run the python commands shown in README.md.
PY ?= python

install:        ## install runtime + dev dependencies and the airq package
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

data:           ## rebuild the hourly dataset from the raw archive (downloads ~GBs from openAFRICA)
	$(PY) -m airq.pipeline ingest --download

data-sample:    ## offline demo of the data pipeline on the bundled sample months
	$(PY) -m airq.pipeline ingest --raw-dir data/sample_raw --no-network --no-xlsx

train:          ## fit detectors, meta-model and cause model -> models/, reports/metrics.json
	$(PY) -m airq.pipeline train

predict:        ## batch-score the sample file
	$(PY) -m airq.pipeline predict --input data/sample/sample_request.csv --output reports/sample_predictions.csv --explain

test:
	$(PY) -m pytest -q

api:            ## serve the API locally on :8000
	uvicorn api.main:app --reload --port 8000

dashboard:
	AIRQ_API_URL=http://localhost:8000 streamlit run dashboard/app.py

docker:         ## build and start API + dashboard
	docker compose up --build

.PHONY: install data data-sample train predict test api dashboard docker
