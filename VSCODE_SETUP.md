# VS Code / Windows Setup Guide

## 1. Open the project

Extract this ZIP and open the folder `flipkart-order-intelligence` in VS Code.

Open:
Terminal -> New Terminal

Check:
```powershell
pwd
dir
```

## 2. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate again.

In VS Code:
`Ctrl+Shift+P` -> `Python: Select Interpreter` -> choose `.venv\Scripts\python.exe`.

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify:
```powershell
python -c "import numpy, pandas, sklearn, torch, torchvision; print('Environment OK')"
```

Check PyTorch:
```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 4. Part 1 — dataset and return-risk model

The PDF requires the supplied seeded order generator to be used unchanged.

Run:
```powershell
python generate_orders.py
python part1/train_and_evaluate.py
```

This creates/updates:
- `orders_dataset.csv`
- `models/return_risk_model.pkl`
- Part 1 results including threshold/feature analysis

## 5. Part 2 — real Fashion-MNIST + ResNet-18

Run:
```powershell
python part2/train_classifier.py
```

The first run requires internet access to download Fashion-MNIST and the pretrained ResNet-18 weights.

The script performs:
- Fashion-MNIST download
- stratified validation split
- grayscale -> 3 channels
- resize/normalization for ResNet-18
- frozen-backbone feature extraction
- local feature caching
- classifier-head training
- late-layer fine-tuning if needed
- untouched official test evaluation
- confusion matrix/per-class metrics
- five real test PNG exports
- `models/product_classifier.pt`

`data/feature_cache/` is intentionally ignored by Git because it is a generated intermediate cache. The code regenerates it.

## 6. Check artifacts

```powershell
python scripts/healthcheck.py
```

Required final artifacts:
- `models/return_risk_model.pkl`
- `models/product_classifier.pt`
- `data/sample_images/` containing real test PNGs

## 7. Part 3 — RAG / LangGraph

```powershell
python part3/retrieval_eval.py
python part3/test_suite.py
python part3/run_demo.py
```

## 8. Run the API

Open a second VS Code terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

Open:
`http://127.0.0.1:8000/docs`

## 9. Run the dashboard

Open another VS Code terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/dashboard.py
```

Open the URL printed by Streamlit, normally:
`http://localhost:8501`

## 10. Test the project

```powershell
pytest -q
python scripts/validate_repo.py
```

## 11. Git / GitHub

Initialize after the project is working:

```powershell
git init
git add .
git commit -m "build: initialize order intelligence assistant"
git branch -M main
```

Create an empty PUBLIC GitHub repository, then:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

For the required feature-branch workflow:

```powershell
git checkout -b feature/production-platform
```

Make a meaningful change and commit:
```powershell
git add .
git commit -m "feat: improve support platform"
```

Make another meaningful change and commit:
```powershell
git add .
git commit -m "test: add production validation"
```

Push:
```powershell
git push -u origin feature/production-platform
```

Create a Pull Request on GitHub and merge it into `main`.

Then:
```powershell
git checkout main
git pull origin main
```

## 12. Git safety

Do NOT commit:
- `.venv/`
- `__pycache__/`
- `data/feature_cache/`

Do commit the required artifacts:
- `models/return_risk_model.pkl`
- `models/product_classifier.pt`
- `data/sample_images/`

## 13. Final verification

```powershell
git status
python scripts/healthcheck.py
python scripts/validate_repo.py
pytest -q
```

## Final Health Check

Before pushing the project to GitHub, run:

```powershell
python scripts/healthcheck.py
python scripts/validate_repo.py
python -m pytest -q