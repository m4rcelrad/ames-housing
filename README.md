# Ames Housing - Streamlit + MLflow

This project trains regression models on the Ames Housing dataset and serves predictions through a Streamlit web application.
MLflow is used for experiment tracking, metric comparison, and model run management.

## Features

- Train multiple regression models on Ames Housing
- Track experiments and metrics with MLflow
- Run single and batch predictions in Streamlit
- Run locally or with Docker Compose
- Persist MLflow runs and generated reports across container restarts
- Includes basic smoke tests

## Project Structure

- `src/` - core application logic
- `tests/` - automated tests
- `streamlit_app.py` - Streamlit user interface
- `main.py` - training entry point
- `predict.py` - command-line prediction script
- `docker-compose.yml` - container orchestration
- `requirements.txt` - Python dependencies

## Local Setup

### Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train models locally

```bash
python main.py
```

### Start Streamlit locally

```bash
streamlit run streamlit_app.py
```

## Docker Setup

### 1) Build and start the web application

```bash
docker compose up --build app
```

The app is available at `http://localhost:8501`.

### 2) Start MLflow UI

```bash
docker compose up mlflow-ui
```

MLflow UI is available at `http://localhost:5000`.

### 3) Run training in the dedicated `trainer` service

```bash
docker compose run --rm trainer
```

Alternative (with profile):

```bash
docker compose --profile train up trainer
```

### 4) Run CLI prediction in a container

```bash
docker compose run --rm app python predict.py
```

## Persistent Data

`docker-compose.yml` mounts these volumes:
- `./mlruns -> /app/mlruns`
- `./reports -> /app/reports`

This keeps MLflow runs and report artifacts on the host after container restarts.

## Smoke Tests

```bash
python -m unittest discover -s tests
```

