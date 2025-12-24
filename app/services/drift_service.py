import os
import json
import sqlite3
import time
import pandas as pd
import numpy as np

from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
from evidently import Dataset, DataDefinition, Report, BinaryClassification
from evidently.presets import DataDriftPreset, DataSummaryPreset

from app.config.config import settings, logger

class DriftService:
    def __init__(self):
        # Data definition, matching the features used by your model
        self.data_definition = DataDefinition(
            classification=[
                BinaryClassification(
                    target="Class",
                    prediction_probas="fraud_probability", # Column name expected in data for this metric
                )
            ],
            # List of all feature columns (V1-V28, Time, Amount)
            numerical_columns=[
                "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
                "Amount"
            ]
        )

    def _load_reference_data(self) -> pd.DataFrame:
        """Loads the baseline/reference dataset for drift comparison."""
        if not os.path.exists(settings.REFERENCE_CSV):
            raise HTTPException(
                status_code=404,
                detail=f"Reference data not found at {settings.REFERENCE_CSV}. Cannot generate report."
            )
        
        reference = pd.read_csv(settings.REFERENCE_CSV)
        
        # Add a placeholder for fraud_probability (required by BinaryClassification metric)
        # Even though we don't have predictions on ref data, Evidently needs this column.
        reference["fraud_probability"] = np.nan
        
        return reference
    
    def _load_current_data(self, days: int) -> pd.DataFrame:
        """Loads production data from the SQLite DB for the specified lookback period."""
        now_utc = datetime.now(timezone.utc)
        since = (now_utc - timedelta(days=days)).isoformat()
        
        query = "SELECT * FROM requests WHERE timestamp >= ?"
        
        try:
            with sqlite3.connect(settings.DB_PATH) as conn:
                current = pd.read_sql_query(query, conn, params=[since])
        except Exception as e:
            logger.error(f"Error reading database: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error while fetching production data: {str(e)}"
            )
        
        if current.empty:
            logger.warning(f"No recent data found in the database for the last {days} days.")
            raise HTTPException(
                status_code=404,
                detail=f"No prediction data found in the last {days} days."
            )
            
        # Drop metadata columns (id, timestamp)
        current = current.drop(columns=[c for c in ["id", "timestamp"] if c in current.columns], errors='ignore')
        
        return current
    
    def generate_report(self, days: int):
        """Generates the data drift report."""
        start_time = time.time()
        
        reference = self._load_reference_data()
        current = self._load_current_data(days)
        
        logger.info(f"Loaded {len(reference)} reference rows and {len(current)} current rows.")

        # Build Evidently datasets
        reference_data = Dataset.from_pandas(
            reference,
            data_definition=self.data_definition
        )
        
        current_data = Dataset.from_pandas(
            current,
            data_definition=self.data_definition
        )
        
        # Report with summary + drift
        report = Report(
            metrics=[
                # Drift threshold set to 70% of columns showing drift
                DataDriftPreset(drift_share=0.7), 
                DataSummaryPreset()
            ],
            include_tests=True # Optionally include tests for a more comprehensive report
        )
        
        # Run monitoring
        logger.info("Running drift analysis...")
        report_results = report.run(reference_data=reference_data, current_data=current_data)
        
        # Save reports
        os.makedirs(os.path.dirname(settings.REPORT_PATH), exist_ok=True)
        report_results.save_html(settings.REPORT_PATH)
        report_results.save_json(settings.REPORT_JSON)
        
        elapsed = time.time() - start_time
        
        logger.info(
            json.dumps({
                "event": "drift_report_generated",
                "days": days,
                "current_data_rows": len(current),
                "duration_sec": round(elapsed, 2)
            })
        )
    
        # Return summary
        return {
            "status": "success",
            "report_url": "/monitoring/drift/report",
            "current_data_rows": len(current),
            "reference_data_rows": len(reference),
            "days_analyzed": days,
            "duration_sec": round(elapsed, 2),
            "message": "Drift report generated successfully. View at /monitoring/drift/report"
        }
    

drift_service = DriftService()