# 🛡️ MLOps Fraud Detection Service

A production-ready machine learning operations (MLOps) platform for real-time credit card fraud detection, demonstrating end-to-end ML engineering best practices from experimentation to deployment.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.18+-orange.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

## 🎯 Project Overview

This project showcases a complete MLOps pipeline including data preprocessing, hyperparameter optimization, model training, deployment, monitoring, and drift detection. The service exposes a REST API and interactive UI for fraud prediction while maintaining full experiment tracking and model versioning.

**Key Highlights:**
- 🚀 **Production-Ready API**: FastAPI service with ONNX-optimized inference
- 📊 **Interactive UI**: Gradio interface for real-time predictions
- 🔬 **Experiment Tracking**: MLflow integration with model registry
- 📈 **Model Monitoring**: Automated data drift detection with Evidently
- 🐳 **Containerized**: Docker Compose orchestration for easy deployment
- ⚡ **Optimized Performance**: ONNX runtime for faster inference
- 🎛️ **Hyperparameter Tuning**: Optuna-based automated optimization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fraud Detection Service                   │
├─────────────────────────────────────────────────────────────┤
│  FastAPI REST API          │  Gradio Web UI                 │
│  • /predict (single)       │  • Interactive prediction      │
│  • /predict/batch          │  • Real-time visualization     │
│  • /monitoring/drift       │                                │
│  • /health                 │                                │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  ONNX Runtime    │          │  SQLite Logger   │
    │  (ML Inference)  │          │  (Predictions)   │
    └──────────────────┘          └──────────────────┘
               │                              │
               ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  MLflow Registry │          │ Evidently Drift  │
    │  (Model Store)   │          │  Detection       │
    └──────────────────┘          └──────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Infrastructure (Docker Compose)      │
    │  • PostgreSQL (MLflow backend)       │
    │  • MinIO (S3-compatible artifacts)   │
    │  • MLflow Server                     │
    └──────────────────────────────────────┘
```

## 🛠️ Tech Stack

**Machine Learning & MLOps:**
- **Frameworks**: scikit-learn, LightGBM, XGBoost
- **Optimization**: Optuna (hyperparameter tuning)
- **Inference**: ONNX Runtime (framework-agnostic deployment)
- **Experiment Tracking**: MLflow (with model registry)
- **Monitoring**: Evidently (drift detection)

**Backend & API:**
- **Web Framework**: FastAPI (async REST API)
- **UI**: Gradio (interactive web interface)
- **Database**: SQLite (production logging), PostgreSQL (MLflow backend)
- **Object Storage**: MinIO (S3-compatible artifact store)

**DevOps & Deployment:**
- **Containerization**: Docker & Docker Compose
- **Dependency Management**: uv (fast Python package manager)
- **Data Processing**: Pandas, Polars
- **Environment**: Python 3.11+

## 📊 ML Pipeline

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

4. **Prepare data**
```bash
# Place creditcard.csv in data/raw/
python src/data_preprocessing.py
```

5. **Start services**
```bash
docker-compose up -d
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
python src/data_preprocessing.py
python src/hpo.py
python src/train.py
```

### Local API Development
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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

## 🔮 Future Enhancements

- [ ] Kubernetes deployment with Helm charts
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Prometheus metrics and Grafana dashboards
- [ ] Feature store integration (Feast)
- [ ] A/B testing framework
- [ ] Automated model retraining pipeline
- [ ] Multi-model ensemble predictions
- [ ] Real-time streaming with Kafka/Kinesis

## 📝 License

This project is created for portfolio demonstration purposes.

## 👤 Contact

**Hugo González**
- GitHub: [@hugoglez](https://github.com/hugoglez)
- LinkedIn: [Hugo González](https://linkedin.com/in/hugoglez)
- Portfolio: [hugoglez.com](https://hugoglez.com)

---

⭐ If you find this project interesting, please consider starring the repository!