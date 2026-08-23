# Part 2 — Product Image Classification

This implementation follows the assignment's required Fashion-MNIST + transfer-learning workflow.

## What it does

- Downloads the official Fashion-MNIST train/test split through `torchvision.datasets.FashionMNIST`.
- Creates a deterministic stratified validation set of 5,000 training images (500 per class).
- Leaves the official 10,000-image test set untouched until final evaluation.
- Converts grayscale 28×28 images to 3 channels, resizes to 224×224, and applies ImageNet normalization.
- Uses ImageNet-pretrained ResNet-18.
- Freezes the backbone and caches frozen-backbone features for the head-training stage.
- Automatically fine-tunes `layer4` + classifier when validation accuracy is below 80%.
- Saves the real trained artifact to `models/product_classifier.pt`.
- Exports five real official test images to `data/sample_images/`.
- Writes the confusion matrix and per-class precision/recall to `results/`.

## Run

From the repository root:

```bash
python part2/train_classifier.py
```

For a fresh feature extraction:

```bash
python part2/train_classifier.py --force-recompute
```

To force fine-tuning even if the head reaches 80%:

```bash
python part2/train_classifier.py --force-fine-tune
```

The first run needs internet access for the official Fashion-MNIST files and ResNet-18 ImageNet weights. After those assets are cached, the saved model and feature caches can be reused offline.
