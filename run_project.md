# Run order

## Part 1
```bash
python generate_orders.py
python part1/train_and_evaluate.py
```
This creates `orders_dataset.csv`, `models/return_risk_model.pkl`, `results/part1_results.json`, threshold sweeps, feature-importance files, and subgroup metrics.

## Part 2
Run in an environment with internet access the first time so torchvision can download Fashion-MNIST and pretrained ResNet-18 weights:
```bash
python part2/train_classifier.py
```
The script exports `models/product_classifier.pt` and 5 real test images in `data/sample_images/`.

## Part 3
Install `requirements.txt`, then build/run the support agent:
```bash
python part3/retrieval_eval.py
python part3/run_demo.py
```
The agent uses a local sentence-transformer + FAISS path when available. This repository also contains a TF-IDF fallback for offline smoke tests; the graded/submission environment should install the listed `sentence-transformers` and `faiss-cpu` dependencies so the source-derived requirement is met.

## Part 2 one-command run

Windows PowerShell: `./run_part2.ps1`

Cross-platform: `python part2/train_classifier.py` followed by `python validate_project.py`.
