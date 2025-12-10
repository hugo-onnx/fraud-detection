import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import onnxruntime as rt
from mlflow.tracking import MlflowClient
from app.config import settings, logger
from app.db import init_db

class ModelService:
    def __init__(self):
        self.client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # Model State
        self.session = None
        self.scaler = None
        self.feature_columns = []
        self.input_name = None
        self.output_name = None
        
        # Metadata for health checks/logging
        self.model_meta = {
            "name": None,
            "version": None,
            "description": None,
            "type": None
        }

    def load_model(self):
        """
        Main entry point to load the model. 
        Tries MLflow registry first, falls back to local files if that fails.
        """
        try:
            logger.info("Attempting to load champion model from MLflow registry...")
            self._load_from_registry()
            logger.info(f"Successfully loaded {self.model_meta['name']} (v{self.model_meta['version']})")
        except Exception as e:
            logger.error(f"Failed to load from registry: {e}")
            logger.warning("Falling back to local model files...")
            self._load_fallback()
        
        # Initialize DB schema based on loaded model features
        init_db(self.feature_columns)

    def predict(self, features: dict) -> float:
        """
        Runs inference on a single dictionary of features.
        """
        if not self.session:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # 1. Validation
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # 2. Preprocessing
        # Create DataFrame in the exact order required by the model
        X_df = pd.DataFrame([features], columns=self.feature_columns)
        
        # Scale features
        # Note: input needs to be float32 for ONNX Runtime in most cases
        X_scaled = self.scaler.transform(X_df).astype(np.float32)

        # 3. Inference
        preds = self.session.run([self.output_name], {self.input_name: X_scaled})

        # 4. Output Extraction
        # ONNX output is typically a list of dicts or arrays. 
        # For sklearn-onnx classifiers, it's often [label, probabilities_dict]
        # We assume index 1 is the probability map, and we want the probability of "1" (Fraud)
        pred_val = preds[0][0] # Adjust based on specific ONNX export shape if needed
        
        # Handle different return shapes from ONNX (Map vs Array)
        if isinstance(pred_val, (dict, map)):
            # If it returns a dictionary {0: 0.99, 1: 0.01}
            fraud_prob = pred_val.get(1, 0.0)
        elif isinstance(pred_val, (list, np.ndarray)):
            # If it returns an array [0.99, 0.01]
            fraud_prob = pred_val[1]
        else:
            # Fallback if structure is unexpected (e.g. bare float)
            fraud_prob = float(pred_val)

        return float(fraud_prob)

    def _find_champion_model(self):
        """Iterates through known model names to find one tagged 'champion'."""
        for model_key, model_name in settings.MODEL_NAMES.items():
            try:
                # We use alias 'champion' to find the specific version
                version = self.client.get_model_version_by_alias(model_name, "champion")
                return model_name, version, model_key
            except Exception:
                continue
        
        raise ValueError("No model with 'champion' alias found in registry.")

    def _load_from_registry(self):
        """Loads artifacts from MLflow."""
        model_name, model_version, model_key = self._find_champion_model()
        
        # Construct URI
        # Using "models:/" URI is standard for loading, but for artifact downloading
        # we often need the source run_id if we want specific sub-files (like scaler).
        run_id = model_version.run_id
        artifact_uri = f"runs:/{run_id}"

        # Determine Model Type Folder Name (e.g. 'RandomForest', 'LightGBM')
        # This matches the logic you had for directory structure
        if "lightgbm" in model_key:
            model_type_dir = "LightGBM"
        elif "logisticregression" in model_key:
            model_type_dir = "LogisticRegression"
        elif "randomforest" in model_key:
            model_type_dir = "RandomForest"
        else:
            model_type_dir = model_key.title()

        # 1. Download & Load ONNX Model
        onnx_path = f"{model_type_dir}/onnx/{model_type_dir}_best.onnx"
        local_model_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/{onnx_path}")
        self.session = rt.InferenceSession(local_model_path, providers=["CPUExecutionProvider"])
        
        # 2. Download & Load Preprocessors (Scaler/Features)
        # Assuming these are logged in the same run, usually at root or specific folder
        # Adjust paths based on your specific artifact structure
        try:
            local_scaler_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/scaler.pkl")
            self.scaler = joblib.load(local_scaler_path)
            
            local_features_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/feature_columns.json")
            with open(local_features_path, "r") as f:
                self.feature_columns = json.load(f)
        except Exception:
            # If not in registry, try local fallback for preprocessors
            logger.warning("Preprocessors not found in MLflow run. Using local files.")
            self.scaler = joblib.load(settings.SCALER_PATH)
            with open(settings.FEATURES_PATH, "r") as f:
                self.feature_columns = json.load(f)

        # 3. Configure Session I/O
        self._configure_session()

        # 4. Update Metadata
        self.model_meta = {
            "name": model_name,
            "version": model_version.version,
            "description": model_version.description,
            "type": model_key
        }

    def _load_fallback(self):
        """Loads from local disk if MLflow fails."""
        # Hardcoded fallback paths
        local_onnx_path = "models/LightGBM_best.onnx" # Ensure this exists in your project
        
        if not self.scaler:
            self.scaler = joblib.load(settings.SCALER_PATH)
        
        if not self.feature_columns:
            with open(settings.FEATURES_PATH, "r") as f:
                self.feature_columns = json.load(f)
                
        self.session = rt.InferenceSession(local_onnx_path, providers=["CPUExecutionProvider"])
        self._configure_session()
        
        self.model_meta = {
            "name": "Local Fallback",
            "version": "0.0.0",
            "description": "Loaded from local disk due to registry failure",
            "type": "fallback"
        }

    def _configure_session(self):
        """Helper to determine input/output names from the ONNX session."""
        self.input_name = self.session.get_inputs()[0].name
        
        # Logic to find probability output
        prob_output = None
        for out in self.session.get_outputs():
            if "prob" in out.name.lower():
                prob_output = out.name
                break
        self.output_name = prob_output if prob_output else self.session.get_outputs()[0].name


model_service = ModelService()