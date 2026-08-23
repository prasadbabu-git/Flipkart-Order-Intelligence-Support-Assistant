from pathlib import Path
import json

def test_required_files():
    root = Path(__file__).resolve().parents[1]
    for p in ["README.md", "generate_orders.py", "part1/train_and_evaluate.py", "part2/train_classifier.py", "part3/graph.py", "api/main.py", "app/dashboard.py"]:
        assert (root / p).exists(), p

def test_threshold():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "results/thresholds.json").read_text())
    assert 0 < float(data.get("t_rf", data.get("random_forest_optimal_threshold", 0.46))) < 1
