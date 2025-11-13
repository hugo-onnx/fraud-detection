import os
import sqlite3
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
from evidently import Report
from evidently.presets import DataSummaryPreset
from evidently.metrics import DriftedColumnsCount, ValueDrift

DB_PATH = "data/production/requests.db"
REPORT_PATH = "reports/data_drift_report.html"
REFERENCE_CSV = "data/production/creditcard_reference.csv"

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# Load reference
reference = pd.read_csv(REFERENCE_CSV)

now_utc = datetime.now(timezone.utc)
since = (now_utc - timedelta(days=7)).isoformat()

# Read recent rows via parameterized query and context manager
query = "SELECT * FROM requests WHERE timestamp >= ?"
try:
    with sqlite3.connect(DB_PATH) as conn:
        current = pd.read_sql_query(query, conn, params=[since])
except Exception as e:
    print("Error reading DB:", e)
    sys.exit(1)

if current.empty:
    print("No recent data found in the database.")
    sys.exit(0)

# Drop columns safely
current = current.drop(columns=[c for c in ["id", "timestamp"] if c in current.columns])

# Ensure "Amount" exists and is numeric
if "Amount" not in current.columns or "Amount" not in reference.columns:
    print("Error: 'Amount' column missing from reference or current data.")
    sys.exit(1)

current["Amount"] = pd.to_numeric(current["Amount"], errors="coerce")
reference["Amount"] = pd.to_numeric(reference["Amount"], errors="coerce")

# Build report
report = Report(
    metrics=[
        # 1. Data Quality Checks (production-critical)
        DataSummaryPreset(),

        # 2. Whole-dataset drift
        DriftedColumnsCount(),

        # 3. Column-level drift (signal-rich, low false positives)
        ValueDrift(column="Amount"),
        ValueDrift(column="V4"),
        ValueDrift(column="V14"),
        ValueDrift(column="V3"),
        ValueDrift(column="V12"),

        # 4. Prediction drift
        ValueDrift(column="fraud_probability")
    ]
)
my_eval = report.run(reference_data=reference, current_data=current)
my_eval.save_html(REPORT_PATH)