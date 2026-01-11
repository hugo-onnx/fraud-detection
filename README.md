# 🛡️ MLOps Fraud Detection Service

A production-ready machine learning operations (MLOps) platform for real-time credit card fraud detection, demonstrating end-to-end ML engineering best practices from experimentation to deployment.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.18+-orange.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

## 🎯 Project Overview

This project showcases a complete MLOps pipeline including data preprocessing, hyperparameter optimization, model training, deployment, monitoring, and drift detection. The service exposes a REST API and interactive UI for fraud prediction while maintaining full experiment tracking and model versioning.

**Key Highlights:**
- 🚀 FastAPI inference service with ONNX-optimized models
- 🔬 Full experiment tracking & model registry (MLflow)
- 📈 Automated data drift detection (Evidently)
- 🎛️ Hyperparameter optimization (Optuna)
- 🐳 Fully containerized, reproducible setup
- 🧪 Production-grade logging, monitoring, and fallbacks

## 🌐 Live Demo

You can explore the deployed application here:

- **Interactive UI (Gradio)**  
  👉 https://fraud-detection-api-5gq2.onrender.com/ui

- **API Documentation (Swagger / OpenAPI)**  
  👉 https://fraud-detection-api-5gq2.onrender.com/docs

> ⚠️ **Deployment Notice**  
> This service is deployed on **Render’s free tier**.  
> The first request may take **30–60 seconds** to respond due to cold starts.  
> Subsequent requests will be significantly faster.

## 💡 Problem & Motivation

Credit card fraud detection systems face three key production challenges:
1. **Extreme class imbalance** requiring careful evaluation and monitoring
2. **Model decay over time** due to changing transaction patterns
3. **Operational complexity** when moving from notebooks to production APIs

This project was designed to simulate a **real-world fraud detection system**, focusing not just on model accuracy, but on:
- Reliable deployment
- Continuous monitoring
- Reproducibility
- Operational robustness

The goal is to demonstrate how an ML system behaves *after* deployment — not just how it trains.

## 🏗️ Architecture

The system is designed as a microservices architecture orchestrated by **Docker Compose**, separating the API logic, experiment tracking, and data persistence.

![System Architecture](/images/system-architecture.png)

### Component Breakdown:
* **fraud-api container**: Houses the FastAPI application, Gradio UI, and the **ONNX Runtime** for high-performance inference.
* **mlflow container**: Manages the Model Registry and experiment logs, allowing the API to dynamically pull the "champion" model.
* **postgres container**: Acts as the backend store for MLflow metadata and request logging.
* **Evidently AI**: Integrated within the private API routes to monitor data drift and model health.

## 🧠 Key Design Decisions

- **ONNX for inference**  
  Chosen to decouple training frameworks from production serving and improve latency consistency.

- **MLflow Model Registry with aliases**  
  Enables safe model promotion (champion/challenger) without redeploying the API.

- **Evidently for drift detection**  
  Focused on *data drift* rather than concept drift, as labels may not be immediately available in fraud systems.

- **Parquet for intermediate artifacts**  
  Optimizes I/O performance and schema consistency across training stages.

- **Docker Compose over Kubernetes**  
  Keeps the project locally reproducible while mirroring real microservice separation.

## 🛠️ Tech Stack

| Category | Tools |
| :--- | :--- |
| **ML Frameworks** | scikit-learn, LightGBM, SHAP |
| **MLOps & Tracking** | MLflow, Optuna, ONNX Runtime |
| **Monitoring** | Evidently AI |
| **Backend** | FastAPI, Gradio (UI), Pydantic |
| **Infrastructure** | Docker, Docker Compose, MinIO (S3), PostgreSQL |
| **Data Handling** | Pandas, Parquet |

## 📊 ML Pipeline

The training pipeline is fully automated and designed for reproducibility, moving from raw data in S3-compatible storage to a versioned, production-ready ONNX model.

![Training Pipeline](/images/training-pipeline.png)

### 1. Data Preprocessing
```bash
python src/data_preprocessing.py
```
- Loads credit card transaction data
- Performs train/test split with stratification
- Applies StandardScaler for feature normalization
- Exports preprocessed data in Parquet format
- Generates reference dataset for drift detection

### 2. Hyperparameter Optimization
```bash
python src/hpo.py
```
- Uses Optuna for Bayesian optimization
- Compares multiple algorithms: Logistic Regression, Random Forest, LightGBM
- Optimizes for AUC-ROC score
- Logs all trials to MLflow with parameters and metrics

### 3. Model Training & Registration
```bash
python src/train.py
```
- Trains champion models with optimized hyperparameters
- Converts models to ONNX format for production deployment
- Registers models in MLflow with versioning
- Generates evaluation metrics and visualizations
- Compares model performance across algorithms

### 4. Model Deployment
The API automatically loads the "champion" model from MLflow registry:
```bash
# Assign champion alias to best model
mlflow models set-alias --name fraud_detection_lightgbm --alias champion --version 3
```

## 🚀 Quick Start

> 💡 Tip  
> If you just want to explore the system behavior without running it locally,  
> you can use the live deployment listed in the **Live Demo** section above.

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mlops-fraud-detection-service.git
cd mlops-fraud-detection-service
```

2. **Install dependencies**
```bash
uv sync
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start services**
```bash
docker-compose up -d --build
```

### Access Points
- **Fraud API**: http://localhost:8000
- **Interactive UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001

## 📡 API Usage

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0,
    "V1": -1.359807, "V2": -0.072781, ..., "V28": -0.021053,
    "Amount": 149.62
  }'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.0234,
  "model_version": "fraud_detection_lightgbm_v3",
  "timestamp": "2026-01-08T18:05:01Z"
}
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"Time": 0, "V1": -1.359807, ..., "Amount": 149.62},
      {"Time": 1, "V1": 1.191857, ..., "Amount": 2.69}
    ]
  }'
```

### Drift Detection
```bash
curl -X POST "http://localhost:8000/monitoring/drift" \
  -H "Content-Type: application/json" \
  -d '{"days": 30}'
```

## 🎯 Key Features

### Model Management
- **Automated Model Selection**: Champion model loaded from MLflow registry
- **A/B Testing Ready**: Easy model version switching via aliases
- **Fallback Mode**: Local model serving if MLflow unavailable
- **Version Tracking**: All predictions logged with model metadata

### Monitoring & Observability
- **Data Drift Detection**: Automated comparison against reference data
- **Prediction Logging**: SQLite database for production tracking
- **Health Checks**: Liveness and readiness endpoints
- **Metrics Dashboard**: Visualize model performance over time

### Performance Optimization
- **ONNX Inference**: 3-5x faster than native Python execution
- **Async API**: Non-blocking request handling with FastAPI
- **Background Logging**: Database writes don't block responses
- **Batch Processing**: Efficient vectorized predictions

## 📈 Model Performance

Metrics prioritize recall and AUC due to the asymmetric cost of false negatives in fraud detection.

| Model | AUC-ROC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| LightGBM | 0.98 | 0.92 | 0.89 | 0.90 |
| Random Forest | 0.97 | 0.89 | 0.86 | 0.87 |
| Logistic Regression | 0.95 | 0.84 | 0.81 | 0.82 |

*Performance metrics on test set with class imbalance (0.17% fraud rate)*

## 🧪 Development

### Training New Models
```bash
# Run full pipeline
uv run src/data_preprocessing.py
uv run src/hpo.py
uv run src/train.py
```

### Local API Development
```bash
docker-compose up -d --build
```

### View Logs
```bash
docker-compose logs -f fraud-api
```

### Stop Services
```bash
docker-compose down
```

## 📁 Project Structure

```
mlops-fraud-detection-service/
├── app/
│   ├── api/             # FastAPI routes and endpoints
│   ├── config/          # Configuration management
│   ├── db/              # Database initialization
│   ├── models/          # Pydantic schemas
│   ├── services/        # ML inference and drift detection
│   └── ui/              # Gradio interface
├── src/
│   ├── data_preprocessing.py  # Data pipeline
│   ├── hpo.py                 # Hyperparameter optimization
│   └── train.py               # Model training
├── data/
│   ├── raw/             # Original dataset
│   ├── processed/       # Scaled features and artifacts
│   └── production/      # Reference data
├── models/              # Local model storage (fallback)
├── reports/             # Drift detection reports
├── scripts/             # Utility scripts
├── docker-compose.yml   # Service orchestration
├── Dockerfile           # API container definition
└── pyproject.toml       # Project dependencies
```

## 🎓 Skills Demonstrated

**Machine Learning Engineering:**
- End-to-end ML pipeline design and implementation
- Model selection, training, and hyperparameter optimization
- Model deployment with ONNX for production inference
- Experiment tracking and model versioning

**MLOps & Infrastructure:**
- Docker containerization and orchestration
- Model registry and artifact management (MLflow)
- Data drift monitoring and alerting
- REST API design and async programming

**Software Engineering:**
- Clean architecture with separation of concerns
- Configuration management and environment handling
- Error handling and graceful degradation
- Documentation and code maintainability

**Data Engineering:**
- Data preprocessing and feature engineering
- Efficient data formats (Parquet) for analytics
- Production data logging and monitoring
- Reference data management for drift detection

> ⚠️ Note  
> Some components (e.g., SQLite logging, batch API design) are intentionally simplified to keep the project self-contained, while preserving production-relevant architecture and patterns.