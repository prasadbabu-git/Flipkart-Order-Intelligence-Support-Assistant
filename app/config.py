from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
KB = ROOT / "part3" / "kb"
SAMPLES = DATA / "sample_images"
CACHE = DATA / "feature_cache"

RETURN_MODEL = MODELS / "return_risk_model.pkl"
PRODUCT_MODEL = MODELS / "product_classifier.pt"
THRESHOLD_FILE = RESULTS / "thresholds.json"
