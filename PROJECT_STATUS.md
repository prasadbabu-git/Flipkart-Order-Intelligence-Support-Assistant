# Project status

## Completed in this prepared workspace

- Exact Part 1 seeded dataset generator reproduced.
- `orders_dataset.csv` generated with 6,000 rows.
- Part 1 preprocessing, baseline, Logistic Regression, threshold tuning, Random Forest GridSearchCV, feature importance, permutation importance, subgroup analysis, `t*_rf`, and `models/return_risk_model.pkl` completed.
- 14 policy documents created.
- Local RAG implementation created with a sentence-transformer + FAISS primary path and an offline TF-IDF fallback.
- Return-risk tool, image-classifier tool wrapper, conditional support graph, deterministic mock response generation, prompt-injection guardrail, groundedness guardrail, retrieval evaluation, and test suite created.
- Local Git history includes a feature branch with two commits and a merge into `main`.

## Part 2 status

The Part 2 implementation is complete and executable. It uses the official Fashion-MNIST dataset, ImageNet-pretrained ResNet-18, a deterministic 5,000-image stratified validation split, cached frozen-backbone features, automatic late-layer fine-tuning below 80% validation accuracy, untouched official test evaluation, and real sample-image export.

The only missing files in this prepared runtime are the generated `models/product_classifier.pt` and five official test PNGs because this runtime has no outbound network access and no cached Fashion-MNIST/ResNet assets. They are generated automatically by the training command in a normal internet-enabled runtime. No synthetic stand-ins are used.

Run:

```bash
python part2/train_classifier.py
```

in an environment where torchvision can download Fashion-MNIST and the pretrained ResNet-18 weights. Then rerun:

```bash
python part3/retrieval_eval.py
python part3/test_suite.py
```

and replace the pending product transcript with an actual prediction transcript.
