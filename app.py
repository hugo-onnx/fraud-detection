import os
import json
import time
import joblib
import logging
import sqlite3
import numpy as np
import pandas as pd
import onnxruntime as rt
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from evidently import Dataset, DataDefinition, Report, BinaryClassification
from evidently.presets import DataDriftPreset, DataSummaryPreset

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud-api")

app = FastAPI(title="Fraud Detection API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "http://mlflow:5000")
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
logger.info(f"MLflow registry URI: {MLFLOW_REGISTRY_URI}")

client = MlflowClient()

# Model names in registry
MODEL_NAMES = {
    "lightgbm": "fraud_detection_lightgbm",
    "randomforest": "fraud_detection_randomforest",
    "logisticregression": "fraud_detection_logisticregression"
}

def find_champion_model():
    for model_key, model_name in MODEL_NAMES.items():
        try:
            model_version = client.get_model_version_by_alias(model_name, "champion")
            logger.info(f"Found champion model: {model_name} (version {model_version.version})")
            return model_name, model_version
        except Exception:
            continue
    
    raise ValueError(
        "No model with 'champion' alias found in registry. "
        "Please run the training pipeline to register and promote a champion model."
    )

def load_model_from_registry():
    try:
        model_name, model_version = find_champion_model()
        model_uri = f"models:/{model_name}@champion"
        
        logger.info(f"Loading champion model: {model_uri}")
        logger.info(f"Model version: {model_version.version}")
        logger.info(f"Description: {model_version.description}")
        
        artifact_path = f"runs:/{model_version.run_id}"
        
        model_type = None
        for key in MODEL_NAMES.keys():
            if key in model_name.lower():
                model_type = key.title().replace("forest", "Forest")
                if "lightgbm" in key:
                    model_type = "LightGBM"
                elif "logisticregression" in key:
                    model_type = "LogisticRegression"
                elif "randomforest" in key:
                    model_type = "RandomForest"
                break
        
        if not model_type:
            raise ValueError(f"Could not determine model type from {model_name}")
        
        onnx_artifact_path = f"{artifact_path}/{model_type}/onnx/{model_type}_best.onnx"
        model_path = mlflow.artifacts.download_artifacts(onnx_artifact_path)
        
        session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        scaler = joblib.load("data/processed/scaler.pkl")
        
        with open("data/processed/feature_columns.json", "r") as f:
            cols = json.load(f)
        
        return session, scaler, cols, model_version.version, model_name, model_version.description
    
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {str(e)}")
        logger.warning("Falling back to local model files")
        onnx_model_path = "models/LightGBM_best.onnx"
        scaler = joblib.load("data/processed/scaler.pkl")
        with open("data/processed/feature_columns.json", "r") as f:
            cols = json.load(f)
        session = rt.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        return session, scaler, cols, "local", "local_fallback", "Local fallback model"

session, scaler, cols, model_version, model_name, model_description = load_model_from_registry()

prob_output = None
for out in session.get_outputs():
    if "prob" in out.name.lower() or "probability" in out.name.lower():
        prob_output = out.name
        break

output_name = prob_output if prob_output else session.get_outputs()[0].name
input_name = session.get_inputs()[0].name

DB_PATH = "data/production/requests.db"
REPORT_PATH = "reports/data_drift_report.html"
REPORT_JSON = "reports/data_drift_report.json"
REFERENCE_CSV = "data/production/creditcard_reference.csv"

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    feature_cols = ", ".join([f'"{c}" REAL' for c in cols])
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {feature_cols},
            fraud_probability REAL,
            timestamp TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

def log_request_to_db(features: dict, proba: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    columns = ", ".join([f'"{k}"' for k in features.keys()] + ["fraud_probability", "timestamp"])
    placeholders = ", ".join(["?"] * (len(features) + 2))
    values = list(features.values()) + [proba, datetime.now(timezone.utc).isoformat()]

    cursor.execute(f"INSERT INTO requests ({columns}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


class Transaction(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name to value mapping")

    @model_validator(mode="after")
    def validate_features(self):
        for k, v in self.features.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Feature '{k}' must be numeric, got {type(v).__name__}")
        return self
    

class DriftReportRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365, description="Number of days to look back")


@app.post("/predict")
def predict(transaction: Transaction, request: Request):
    start_time = time.time()
    try:
        logger.info(f"Prediction request from {request.client.host}")

        input_features = transaction.features

        missing = [c for c in cols if c not in input_features]
        extra = [k for k in input_features if k not in cols]

        if missing:
            msg = f"Missing features: {missing}"
            logger.warning(msg)
            raise HTTPException(status_code=400, detail=msg)
        if extra:
            msg = f"Unexpected features: {extra}"
            logger.warning(msg)
            raise HTTPException(status_code=400, detail=msg)

        X_df = pd.DataFrame([input_features], columns=cols)
        X_scaled = scaler.transform(X_df).astype(np.float32)

        preds_list = session.run([output_name], {input_name: X_scaled})

        pred_dict = preds_list[0][0]

        proba = float(pred_dict[1])

        latency = time.time() - start_time
        logger.info(
            json.dumps({
                "event": "prediction",
                "fraud_probability": proba,
                "latency_sec": round(latency, 4)
            })
        )

        log_request_to_db(input_features, proba)

        return {"fraud_probability": proba}
    
    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e
    
    except Exception as e:
        logger.exception(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/monitoring/drift")
def generate_drift_report(params: DriftReportRequest):
    start_time = time.time()
    
    try:
        logger.info(f"Starting drift report generation for last {params.days} days")
        
        # Check if reference data exists
        if not os.path.exists(REFERENCE_CSV):
            raise HTTPException(
                status_code=404,
                detail=f"Reference data not found at {REFERENCE_CSV}"
            )
        
        # Load reference data
        reference = pd.read_csv(REFERENCE_CSV)
        reference["fraud_probability"] = np.nan
        
        # Load current data from DB
        now_utc = datetime.now(timezone.utc)
        since = (now_utc - timedelta(days=params.days)).isoformat()
        
        query = "SELECT * FROM requests WHERE timestamp >= ?"
        
        try:
            with sqlite3.connect(DB_PATH) as conn:
                current = pd.read_sql_query(query, conn, params=[since])
        except Exception as e:
            logger.error(f"Error reading database: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        
        if current.empty:
            logger.warning("No recent data found in the database")
            raise HTTPException(
                status_code=404,
                detail=f"No data found in the last {params.days} days"
            )
        
        # Drop unnecessary columns
        current = current.drop(columns=[c for c in ["id", "timestamp"] if c in current.columns])
        
        # Ensure same column order (excluding Class)
        feature_cols = [col for col in reference.columns if col not in ["Class"]]
        current = current[feature_cols]
        
        # Data definition
        data_definition = DataDefinition(
            classification=[
                BinaryClassification(
                    target="Class",
                    prediction_probas="fraud_probability",
                )
            ],
            numerical_columns=[
                "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
                "Amount"
            ]
        )
        
        # Build Evidently datasets
        reference_data = Dataset.from_pandas(
            reference,
            data_definition=data_definition
        )
        
        current_data = Dataset.from_pandas(
            current,
            data_definition=data_definition
        )
        
        # Report with summary + drift
        report = Report(
            metrics=[
                DataDriftPreset(drift_share=0.7),
                DataSummaryPreset()
            ],
            include_tests=True
        )
        
        # Run monitoring
        logger.info("Running drift analysis...")
        report_results = report.run(reference_data=reference_data, current_data=current_data)
        
        # Save reports
        report_results.save_html(REPORT_PATH)
        report_results.save_json(REPORT_JSON)
        
        elapsed = time.time() - start_time
        
        logger.info(
            json.dumps({
                "event": "drift_report_generated",
                "days": params.days,
                "current_data_rows": len(current),
                "reference_data_rows": len(reference),
                "duration_sec": round(elapsed, 2)
            })
        )
    
        # Return summary
        return {
            "status": "success",
            "report_url": "/monitoring/drift/report",
            "current_data_rows": len(current),
            "reference_data_rows": len(reference),
            "days_analyzed": params.days,
            "duration_sec": round(elapsed, 2),
            "message": f"Drift report generated successfully. View at /monitoring/drift/report"
        }
    
    except HTTPException as e:
        raise e
    
    except Exception as e:
        logger.exception(f"Error generating drift report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate drift report: {str(e)}"
        )


@app.get("/monitoring/drift/report")
def get_drift_report():
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail="No drift report found. Generate one first using POST /monitoring/drift"
        )
    
    return FileResponse(
        REPORT_PATH,
        media_type="text/html",
        filename="data_drift_report.html",
    )


@app.get("/monitoring/drift/report/json")
def get_drift_report_json():
    if not os.path.exists(REPORT_JSON):
        raise HTTPException(
            status_code=404,
            detail="No JSON drift report found. Generate one first using POST /monitoring/drift"
        )
    
    with open(REPORT_JSON, "r") as f:
        report_data = json.load(f)
    
    return report_data


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_name": model_name,
        "model_version": model_version,
        "model_description": model_description,
    }


@app.get("/model/info")
def get_model_info():
    """Get detailed information about the current champion model"""
    try:
        model_name_clean, model_version_obj = find_champion_model()
        
        return {
            "model_name": model_name_clean,
            "model_version": model_version_obj.version,
            "description": model_version_obj.description,
            "run_id": model_version_obj.run_id,
            "status": model_version_obj.status,
            "tags": model_version_obj.tags,
            "creation_timestamp": model_version_obj.creation_timestamp,
            "last_updated_timestamp": model_version_obj.last_updated_timestamp
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/model/reload")
def reload_model():
    """Reload the champion model from MLflow registry (useful when champion alias is updated)"""
    global session, scaler, cols, model_version, model_name, model_description, output_name, input_name
    
    try:
        logger.info("Reloading champion model from MLflow registry...")
        session, scaler, cols, model_version, model_name, model_description = load_model_from_registry()
        
        prob_output = None
        for out in session.get_outputs():
            if "prob" in out.name.lower() or "probability" in out.name.lower():
                prob_output = out.name
                break
        
        output_name = prob_output if prob_output else session.get_outputs()[0].name
        input_name = session.get_inputs()[0].name
        
        logger.info(f"Model reloaded successfully: {model_name} version {model_version}")
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "model_description": model_description,
            "message": f"Champion model reloaded from MLflow registry"
        }
    
    except Exception as e:
        logger.exception(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )


@app.get("/models/available")
def list_available_models():
    """List all registered models and their versions"""
    try:
        models_info = []
        
        for model_key, model_name_reg in MODEL_NAMES.items():
            try:
                versions = client.search_model_versions(f"name='{model_name_reg}'")
                
                model_data = {
                    "model_name": model_name_reg,
                    "versions": []
                }
                
                for version in versions:
                    version_info = {
                        "version": version.version,
                        "description": version.description,
                        "status": version.status,
                        "run_id": version.run_id,
                        "aliases": version.aliases,
                        "tags": version.tags,
                        "is_champion": "champion" in version.aliases
                    }
                    model_data["versions"].append(version_info)
                
                models_info.append(model_data)
                
            except Exception as e:
                logger.warning(f"Could not fetch info for {model_name_reg}: {e}")
                continue
        
        return {
            "models": models_info,
            "current_champion": {
                "model_name": model_name,
                "version": model_version
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )