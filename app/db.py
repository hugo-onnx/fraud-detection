import sqlite3
import os
from datetime import datetime, timezone
from app.config import settings

def init_db(feature_cols: list):
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    with sqlite3.connect(settings.DB_PATH) as conn:
        cursor = conn.cursor()
        cols_sql = ", ".join([f'"{c}" REAL' for c in feature_cols])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols_sql},
                fraud_probability REAL,
                timestamp TEXT
            );
        """)
        conn.commit()

def log_request(features: dict, proba: float):
    with sqlite3.connect(settings.DB_PATH) as conn:
        cursor = conn.cursor()
        columns = ", ".join([f'"{k}"' for k in features.keys()] + ["fraud_probability", "timestamp"])
        placeholders = ", ".join(["?"] * (len(features) + 2))
        values = list(features.values()) + [proba, datetime.now(timezone.utc).isoformat()]
        cursor.execute(f"INSERT INTO requests ({columns}) VALUES ({placeholders})", values)
        conn.commit()