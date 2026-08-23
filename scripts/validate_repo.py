from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
checks=[
("dataset", ROOT/"orders_dataset.csv"),
("return model", ROOT/"models/return_risk_model.pkl"),
("part2 training", ROOT/"part2/train_classifier.py"),
("part2 inference", ROOT/"part2/inference.py"),
("rag", ROOT/"part3/rag.py"),
("langgraph", ROOT/"part3/graph.py"),
("api", ROOT/"api/main.py"),
("dashboard", ROOT/"app/dashboard.py"),
]
failed=[]
for name,path in checks:
    ok=path.exists()
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {path.relative_to(ROOT)}")
    if not ok: failed.append(name)
print(f"\n{len(checks)-len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
