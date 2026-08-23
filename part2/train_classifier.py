"""Part 2: Fashion-MNIST transfer learning with pretrained ResNet-18.

The script uses the official Fashion-MNIST train/test splits and ImageNet-pretrained
ResNet-18. It keeps the official test set untouched until final evaluation, uses a
stratified 5,000-image validation split, caches frozen-backbone features, and
fine-tunes layer4 automatically when validation accuracy is below 80%.

First run requires internet access because torchvision downloads Fashion-MNIST and
ResNet-18 ImageNet weights. Subsequent runs are local/offline once those assets are
cached.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import models, transforms
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fashion_mnist"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SAMPLES = ROOT / "data" / "sample_images"
CACHE = ROOT / "data" / "feature_cache"
for p in (DATA, MODELS, RESULTS, SAMPLES, CACHE):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_datasets():
    tfm = make_transform()
    train = FashionMNIST(DATA, train=True, download=True, transform=tfm)
    test = FashionMNIST(DATA, train=False, download=True, transform=tfm)
    return train, test


def stratified_split(targets: np.ndarray, per_class_val: int = 500):
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in range(10):
        ids = np.where(targets == c)[0]
        rng.shuffle(ids)
        val_idx.extend(ids[:per_class_val].tolist())
        train_idx.extend(ids[per_class_val:].tolist())
    return train_idx, val_idx


def build_backbone():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


def make_classifier(feature_dim: int):
    return nn.Linear(feature_dim, 10)


def extract_features(backbone, dataset, indices, device, batch_size, cache_path):
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["features"], data["labels"]

    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    backbone.eval()
    feats, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = backbone(x).cpu()
            feats.append(f)
            labels.append(y.cpu())
    features = torch.cat(feats)
    labels = torch.cat(labels)
    torch.save({"features": features, "labels": labels}, cache_path)
    return features, labels


def train_head(classifier, features, labels, epochs, batch_size, device, lr=1e-3):
    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    classifier = classifier.to(device)
    opt = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        classifier.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(classifier(x), y)
            loss.backward()
            opt.step()
        acc = accuracy_from_features(classifier, features, labels, device)
        print(f"head_epoch={epoch+1}/{epochs} train_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
    classifier.load_state_dict(best_state)
    return classifier, best_acc


def accuracy_from_features(classifier, features, labels, device):
    classifier.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(features), 2048):
            x = features[start:start+2048].to(device)
            y = labels[start:start+2048].to(device)
            correct += int((classifier(x).argmax(1) == y).sum())
    return correct / len(labels)


def evaluate_full(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    cm = np.zeros((10, 10), dtype=np.int64)
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += int((pred == y).sum())
            total += len(y)
            for actual, predicted in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[int(actual), int(predicted)] += 1
    return correct / total, cm


def build_full_model(backbone, classifier):
    # Recreate the normal ResNet architecture and attach the trained head.
    weights = None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    backbone_state = backbone.state_dict()
    model_state = model.state_dict()
    for k in model_state:
        if k != "fc.weight" and k != "fc.bias" and k in backbone_state:
            model_state[k] = backbone_state[k]
    model_state["fc.weight"] = classifier.weight.detach().cpu()
    model_state["fc.bias"] = classifier.bias.detach().cpu()
    model.load_state_dict(model_state)
    return model


def fine_tune(backbone_full, classifier, train_ds, train_idx, val_loader, device, epochs, batch_size):
    model = build_full_model(backbone_full, classifier).to(device)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        acc, _ = evaluate_full(model, val_loader, device)
        print(f"fine_tune_epoch={epoch+1}/{epochs} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


def per_class_metrics(cm):
    out = []
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out.append({"class": name, "precision": float(precision), "recall": float(recall)})
    return out


def export_samples():
    raw = FashionMNIST(DATA, train=False, download=False)
    targets = np.asarray(raw.targets)
    chosen = []
    for c in range(10):
        ids = np.where(targets == c)[0]
        if len(ids):
            chosen.append(int(ids[0]))
    for i, idx in enumerate(chosen[:5], start=1):
        image, label = raw[idx]
        filename = f"{i:02d}_{label}_{CLASSES[label].replace('/', '_').replace(' ', '_')}.png"
        image.save(SAMPLES / filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--fine-tune-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--force-fine-tune", action="store_true")
    args = ap.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_ds, test_ds = load_datasets()
    targets = np.asarray(train_ds.targets)
    train_idx, val_idx = stratified_split(targets, 500)

    if args.force_recompute:
        for p in CACHE.glob("*.pt"):
            p.unlink()

    backbone, feature_dim = build_backbone()
    backbone = backbone.to(device)

    train_cache = CACHE / "resnet18_train_features.pt"
    val_cache = CACHE / "resnet18_val_features.pt"
    train_features, train_labels = extract_features(backbone, train_ds, train_idx, device, args.batch_size, train_cache)
    val_features, val_labels = extract_features(backbone, train_ds, val_idx, device, args.batch_size, val_cache)

    classifier = make_classifier(feature_dim)
    classifier, head_train_acc = train_head(
        classifier, train_features, train_labels, args.epochs, args.batch_size, device
    )
    val_head_acc = accuracy_from_features(classifier, val_features, val_labels, device)
    print(f"feature_head_val_acc={val_head_acc:.4f}")

    # Rebuild frozen backbone in normal ResNet form for optional late-layer fine tuning.
    backbone_full = models.resnet18(weights=None)
    backbone_full.fc = nn.Linear(backbone_full.fc.in_features, 10)
    state = backbone_full.state_dict()
    for name, value in backbone.state_dict().items():
        if name in state:
            state[name] = value.detach().cpu()
    state["fc.weight"] = classifier.weight.detach().cpu()
    state["fc.bias"] = classifier.bias.detach().cpu()
    backbone_full.load_state_dict(state)

    # The head-only model is already a valid transfer-learning model. Fine-tune if required.
    if args.force_fine_tune or val_head_acc < 0.80:
        val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
        model, best_val = fine_tune(
            backbone_full, classifier, train_ds, train_idx, val_loader, device,
            args.fine_tune_epochs, args.batch_size
        )
    else:
        model = build_full_model(backbone, classifier).to(device)
        best_val = val_head_acc

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_acc, cm = evaluate_full(model, test_loader, device)

    artifact = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": CLASSES,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "architecture": "resnet18",
    }
    torch.save(artifact, MODELS / "product_classifier.pt")
    np.savetxt(RESULTS / "fashion_mnist_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    result = {
        "validation_accuracy": float(best_val),
        "head_training_accuracy": float(head_train_acc),
        "test_accuracy": float(test_acc),
        "device": str(device),
        "classes": CLASSES,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics(cm),
        "validation_size": 5000,
        "official_test_size": 10000,
        "architecture": "ResNet-18 ImageNet pretrained, 10-class head",
        "image_size": IMAGE_SIZE,
    }
    (RESULTS / "part2_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    export_samples()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
