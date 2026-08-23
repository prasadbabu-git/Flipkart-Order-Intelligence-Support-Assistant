from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "return_risk_model.pkl"
THRESHOLD_PATH = ROOT / "results" / "thresholds.json"
DATASET = ROOT / "orders_dataset.csv"

class OrderEngine:
    def __init__(self, model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH, dataset_path=DATASET):
        self.model = joblib.load(model_path)
        self.df = pd.read_csv(dataset_path)
        with open(threshold_path, encoding="utf-8") as f:
            thresholds = json.load(f)
        self.t_rf = float(thresholds.get("t_rf", thresholds.get("random_forest_optimal_threshold", 0.46)))

    def lookup(self, order_id: int) -> dict:
        row = self.df.loc[self.df.order_id == int(order_id)]
        if row.empty:
            raise KeyError(f"Order {order_id} was not found")
        return row.iloc[0].to_dict()

    def risk(self, order: dict) -> dict:
        features = {k: order[k] for k in self.df.columns if k not in {"order_id", "returned"} and k in order}
        x = pd.DataFrame([features])
        probability = float(self.model.predict_proba(x)[0, 1])
        if probability < self.t_rf:
            bucket = "Low"
        elif probability < self.t_rf + 0.15:
            bucket = "Medium"
        else:
            bucket = "High"
        return {"probability": probability, "risk": bucket, "threshold": self.t_rf}

    def order_summary(self, order_id: int) -> dict:
        order = self.lookup(order_id)
        risk = self.risk(order)
        return {"order": order, "risk": risk}
