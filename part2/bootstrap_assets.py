"""Download and verify the official Part 2 assets, then train the model."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fashion_mnist"
DATA.mkdir(parents=True, exist_ok=True)

# torchvision performs the official Fashion-MNIST download and checksum verification.
# The training script likewise downloads and verifies ImageNet ResNet-18 weights.
print("Bootstrapping official Fashion-MNIST + ImageNet ResNet-18 assets...")
subprocess.check_call([sys.executable, str(ROOT / "part2" / "train_classifier.py")])
print("Part 2 artifact created at models/product_classifier.pt")
