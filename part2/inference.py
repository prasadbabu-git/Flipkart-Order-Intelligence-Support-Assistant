from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "product_classifier.pt"
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


class ProductClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Run 'python part2/train_classifier.py' first."
            )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.classes = ckpt.get("classes", CLASSES)
        image_size = int(ckpt.get("image_size", 224))
        mean = ckpt.get("mean", [0.485, 0.456, 0.406])
        std = ckpt.get("std", [0.229, 0.224, 0.225])

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model = model
        self.tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def predict(self, image_path: str):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)
        img = Image.open(path).convert("L")
        x = self.tfm(img).unsqueeze(0)
        with torch.inference_mode():
            p = torch.softmax(self.model(x), dim=1)[0]
        idx = int(p.argmax())
        return {
            "label": self.classes[idx],
            "confidence": float(p[idx]),
            "class_index": idx,
        }


def classify_product_image(image_path: str):
    return ProductClassifier().predict(image_path)
