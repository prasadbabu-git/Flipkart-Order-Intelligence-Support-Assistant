from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "return_risk_model.pkl"
THRESH = ROOT / "results" / "thresholds.json"
DATASET = ROOT / "orders_dataset.csv"

FEATURES = [
    "product_category",
    "price_inr",
    "discount_pct",
    "payment_method",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]


def _risk_buckets(p: float):
    info = json.loads(THRESH.read_text(encoding="utf-8"))
    t = float(info["random_forest_optimal_threshold_t_star_rf"])

    if p < t:
        return "Low"

    if p < t + 0.15:
        return "Medium"

    return "High"


def _reference_values():
    """
    Build neutral/reference values from the training dataset.

    Numeric features use medians.
    Categorical features use the most frequent value.
    """
    df = pd.read_csv(DATASET)

    reference = {}

    numeric = [
        "price_inr",
        "discount_pct",
        "customer_tenure_days",
        "num_previous_orders",
        "num_previous_returns",
        "delivery_distance_km",
        "delivery_days",
        "is_weekend_order",
        "rating_given",
    ]

    categorical = [
        "product_category",
        "payment_method",
    ]

    for col in numeric:
        reference[col] = float(df[col].median())

    for col in categorical:
        reference[col] = df[col].mode(dropna=True).iloc[0]

    return reference


def _local_feature_explanation(model, row: dict, original_probability: float):
    """
    Estimate local feature influence using one-feature-at-a-time
    counterfactual perturbation.

    Positive impact means the current feature value increases the
    predicted return probability relative to its reference value.
    Negative impact means it decreases the probability.
    """
    reference = _reference_values()
    impacts = []

    for feature in FEATURES:
        current_value = row.get(feature)
        reference_value = reference[feature]

        # Skip features that are already at the reference value.
        if pd.isna(current_value) and pd.isna(reference_value):
            continue

        if current_value == reference_value:
            continue

        counterfactual = dict(row)
        counterfactual[feature] = reference_value

        if pd.isna(counterfactual.get("rating_given")):
            counterfactual["rating_given"] = float("nan")

        cf_df = pd.DataFrame([counterfactual])
        cf_probability = float(model.predict_proba(cf_df)[0, 1])

        impact = original_probability - cf_probability

        impacts.append(
            {
                "feature": feature,
                "impact": float(impact),
                "current_value": current_value,
                "reference_value": reference_value,
            }
        )

    impacts.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return impacts[:5]


def _humanize_feature(feature: str, value):
    labels = {
        "product_category": "product category",
        "price_inr": "order price",
        "discount_pct": "discount",
        "payment_method": "payment method",
        "customer_tenure_days": "customer tenure",
        "num_previous_orders": "previous order count",
        "num_previous_returns": "previous return count",
        "delivery_distance_km": "delivery distance",
        "delivery_days": "delivery time",
        "is_weekend_order": "weekend ordering",
        "rating_given": "customer rating",
    }

    name = labels.get(feature, feature)

    if isinstance(value, float):
        if np.isnan(value):
            value_text = "missing"
        else:
            value_text = f"{value:.2f}"
    else:
        value_text = str(value)

    return f"{name} = {value_text}"


def check_return_risk(order_features: dict):
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing {MODEL}")

    model = joblib.load(MODEL)

    row = {
        k: order_features.get(k)
        for k in FEATURES
    }

    if row.get("rating_given") is None:
        row["rating_given"] = float("nan")

    X = pd.DataFrame([row])

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    explanation = _local_feature_explanation(
        model,
        row,
        probability
    )

    factors = []

    for item in explanation:
        impact = item["impact"]

        factors.append(
            {
                "feature": item["feature"],
                "description": _humanize_feature(
                    item["feature"],
                    item["current_value"]
                ),
                "impact": round(impact, 4),
                "direction": (
                    "increases risk"
                    if impact > 0
                    else "reduces risk"
                ),
            }
        )

    return {
        "return_probability": probability,
        "risk_bucket": _risk_buckets(probability),
        "threshold_file": str(THRESH.name),
        "explanation": factors,
    }


def classify_product_image(image_path: str):
    from part2.inference import (
        classify_product_image as _classify
    )

    return _classify(image_path)