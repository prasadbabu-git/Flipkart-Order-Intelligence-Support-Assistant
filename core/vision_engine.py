from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from torchvision import models, transforms

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class ProductClassifier:
    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing {self.path}. Run: python part2/train_classifier.py")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASSES))
        state = torch.load(self.path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path: str | Path) -> dict:
        image = Image.open(image_path).convert("L")
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(torch.argmax(probs))
        return {"label": CLASSES[idx], "confidence": float(probs[idx]), "class_probabilities": {c: float(p) for c, p in zip(CLASSES, probs)}}
