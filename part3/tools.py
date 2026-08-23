from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'return_risk_model.pkl'
THRESH=ROOT/'results'/'thresholds.json'

FEATURES=['product_category','price_inr','discount_pct','payment_method','customer_tenure_days',
          'num_previous_orders','num_previous_returns','delivery_distance_km','delivery_days',
          'is_weekend_order','rating_given']

def _risk_buckets(p):
    info=json.loads(THRESH.read_text())
    t=float(info['random_forest_optimal_threshold_t_star_rf'])
    # Assignment-style buckets centered on t*_rf; no fixed 0.3/0.6 thresholds.
    if p < t: return 'Low'
    if p < t + 0.15: return 'Medium'
    return 'High'

def check_return_risk(order_features: dict):
    if not MODEL.exists(): raise FileNotFoundError(f'Missing {MODEL}')
    model=joblib.load(MODEL)
    row={k:order_features.get(k) for k in FEATURES}
    if row.get('rating_given') is None: row['rating_given']=float('nan')
    X=pd.DataFrame([row])
    p=float(model.predict_proba(X)[0,1])
    return {'return_probability':p,'risk_bucket':_risk_buckets(p),'threshold_file':str(THRESH.name)}

def classify_product_image(image_path: str):
    from part2.inference import classify_product_image as _classify
    return _classify(image_path)
