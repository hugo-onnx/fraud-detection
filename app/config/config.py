import os
import logging
from pathlib import Path
from typing import Optional

def str_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}

class Config:
    def __init__(self) -> None:
        # Base paths
        self.BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent.parent))

        self.MODELS_DIR = self.BASE_DIR / "models"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.REPORTS_DIR = self.BASE_DIR / "reports"

        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # MLflow
        self.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        self.MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", self.MLFLOW_TRACKING_URI)
        self.MLFLOW_ENABLED = str_to_bool(os.getenv("MLFLOW_ENABLED", "true")) and self.MLFLOW_TRACKING_URI is not None

        # Data / DB
        self.DB_PATH = Path(os.getenv("DB_PATH", str(self.DATA_DIR / "requests.db")))
        self.REFERENCE_CSV = self.DATA_DIR / "production" / "creditcard_reference.csv"

        # Model artifacts
        self.SCALER_PATH = self.MODELS_DIR / "scaler.pkl"
        self.FEATURES_PATH = self.MODELS_DIR / "feature_columns.json"

        # Drift reports
        self.REPORT_PATH = self.REPORTS_DIR / "data_drift_report.html"
        self.REPORT_JSON = self.REPORTS_DIR / "data_drift_report.json"

        # Model registry names
        self.MODEL_NAMES = {
            "lightgbm": "fraud_detection_lightgbm",
            "randomforest": "fraud_detection_randomforest",
            "logisticregression": "fraud_detection_logisticregression",
        }

        self._validate_paths()

    def _validate_paths(self) -> None:
        missing = []

        for path in [
            self.SCALER_PATH,
            self.FEATURES_PATH,
            self.REFERENCE_CSV,
        ]:
            if not path.exists():
                missing.append(str(path))

        if missing:
            raise RuntimeError(
                "Missing required files:\n" + "\n".join(missing)
            )
        
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("fraud-api")

settings = Config()