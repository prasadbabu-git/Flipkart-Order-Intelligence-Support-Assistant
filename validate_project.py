"""Fast repository validation; does not retrain models."""
from pathlib import Path
import json
import joblib

ROOT = Path(__file__).resolve().parent
checks = []

def check(label, ok, detail=""):
    checks.append((label, bool(ok), detail))

check("orders_dataset.csv", (ROOT/"orders_dataset.csv").exists())
check("return_risk_model.pkl", (ROOT/"models/return_risk_model.pkl").exists())
check("t*_rf thresholds", (ROOT/"results/thresholds.json").exists())
check("Part 2 model", (ROOT/"models/product_classifier.pt").exists())
check("5 sample PNGs", len(list((ROOT/"data/sample_images").glob("*.png"))) >= 5,
      f"found {len(list((ROOT/'data/sample_images').glob('*.png')))}")
check("Part 3 KB >= 12", len(list((ROOT/"part3/kb").glob("*.txt"))) >= 12)
check(">= 8 transcripts", len(list((ROOT/"transcripts").glob("*.json"))) >= 8)

if (ROOT/"models/return_risk_model.pkl").exists():
    try:
        model = joblib.load(ROOT/"models/return_risk_model.pkl")
        check("return-risk model loads", hasattr(model, "predict_proba"))
    except Exception as e:
        check("return-risk model loads", False, repr(e))

if (ROOT/"models/product_classifier.pt").exists():
    try:
        import torch
        ckpt = torch.load(ROOT/"models/product_classifier.pt", map_location="cpu", weights_only=False)
        check("product model loads", "state_dict" in ckpt)
    except Exception as e:
        check("product model loads", False, repr(e))

for label, ok, detail in checks:
    print(f"[{'PASS' if ok else 'PENDING'}] {label}" + (f" — {detail}" if detail else ""))

pending = [x for x in checks if not x[1]]
print(f"\n{len(checks)-len(pending)}/{len(checks)} checks passed")
raise SystemExit(1 if pending else 0)
