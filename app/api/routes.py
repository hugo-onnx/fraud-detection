import os
import time
import json

from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request

from app.db import log_request
from app.config import settings, logger
from app.services.ml_service import model_service
from app.services.drift_service import drift_service
from app.schemas import Transaction, DriftReportRequest, PredictionResponse


router = APIRouter()


# Prediction Endpoint
@router.post("/predict", response_model=PredictionResponse, status_code=200, tags=["Prediction"])
def predict(transaction: Transaction, background_tasks: BackgroundTasks, request: Request):
    """
    Receives a transaction and returns the predicted fraud probability.
    Logs the request asynchronously to the production database.
    """
    start_time = time.time()
    try:
        logger.info(f"Prediction request received from {request.client.host}")

        probability = model_service.predict(transaction.features)
        
        background_tasks.add_task(log_request, transaction.features, probability)
        
        latency = time.time() - start_time
        logger.info(
            json.dumps({
                "event": "prediction",
                "fraud_probability": probability,
                "latency_sec": round(latency, 4)
            })
        )
        
        return {
            "fraud_probability": probability, 
            "model_version": model_service.model_meta.get("version", "local")
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error during inference.")
    

# Monitoring Endpoints
@router.post("/monitoring/drift", tags=["Monitoring"])
def generate_drift_report(params: DriftReportRequest):
    """
    Triggers the generation of a data drift report using Evidently.
    Compares the last N days of production data against reference data.
    """
    try:
        results = drift_service.generate_report(params.days)
        return results
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error generating drift report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate drift report: {str(e)}")
    

@router.get("/monitoring/drift/report", tags=["Monitoring"])
def get_drift_report():
    """Returns the last generated HTML data drift report."""
    if not os.path.exists(settings.REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail="No drift report found. Generate one first using POST /monitoring/drift"
        )
    
    return FileResponse(
        settings.REPORT_PATH,
        media_type="text/html",
        filename="data_drift_report.html",
    )


# Health and Model Endpoints
@router.get("/health", tags=["System"])
def health_check():
    """Simple health check returning current model information."""
    if model_service.session is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed.")
    
    return {
        "status": "ok",
        "model_name": model_service.model_meta.get("name"),
        "model_version": model_service.model_meta.get("version"),
        "model_description": model_service.model_meta.get("description"),
        "model_type": model_service.model_meta.get("type"),
    }


@router.post("/model/reload", tags=["System"])
def reload_model():
    """Triggers a reload of the champion model from MLflow registry."""
    try:
        model_service.load_model() # This method handles the reload logic
        return {
            "status": "success",
            "model_name": model_service.model_meta.get("name"),
            "model_version": model_service.model_meta.get("version"),
            "message": "Champion model reloaded successfully."
        }
    except Exception as e:
        logger.exception(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")