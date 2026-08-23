from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1]
model=ROOT/'models/return_risk_model.pkl'
assert model.exists(), model
obj=joblib.load(model)
assert hasattr(obj, 'predict_proba')
print('Return-risk model: OK')
product=ROOT/'models/product_classifier.pt'
print('Product classifier:', 'PRESENT' if product.exists() else 'PENDING FIRST PART-2 TRAINING RUN')
