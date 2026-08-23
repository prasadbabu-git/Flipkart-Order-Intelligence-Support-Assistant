# Part 1 Results

Dataset: 6,000 rows, 13 columns; overall return rate = 22.7500%; missing rating = 13.0500%.

## Missingness

- COD missing rating rate: 22.8309%
- Non-COD missing rating rate: 6.0589%
- Classification: MAR, because rating missingness is generated from observed payment method.

## Model metrics

### baseline

```json
{
  "accuracy": 0.7725,
  "precision_class1": 0.0,
  "recall_class1": 0.0,
  "f1_class1": 0.0
}
```
### logistic_regression_0.5

```json
{
  "accuracy": 0.5916666666666667,
  "precision_class1": 0.2964352720450281,
  "recall_class1": 0.5787545787545788,
  "f1_class1": 0.3920595533498759,
  "roc_auc": 0.6252632660399652
}
```
### logistic_regression_tuned

```json
{
  "accuracy": 0.5016666666666667,
  "precision_class1": 0.28010825439783493,
  "recall_class1": 0.7582417582417582,
  "f1_class1": 0.4090909090909091,
  "roc_auc": 0.6252632660399652,
  "threshold": 0.44
}
```
### random_forest_0.5

```json
{
  "accuracy": 0.6408333333333334,
  "precision_class1": 0.31880733944954126,
  "recall_class1": 0.5091575091575091,
  "f1_class1": 0.3921015514809591,
  "roc_auc": 0.6142861094317404
}
```
### random_forest_tuned

```json
{
  "accuracy": 0.5783333333333334,
  "precision_class1": 0.2938053097345133,
  "recall_class1": 0.608058608058608,
  "f1_class1": 0.39618138424821003,
  "roc_auc": 0.6142861094317404,
  "threshold": 0.46
}
```

## Threshold interpretation

The Logistic Regression tuned threshold increased recall versus 0.5 by 17.95 percentage points. The Random Forest t*_rf is 0.46. Risk buckets in Part 3 use t*_rf and t*_rf + 0.15 rather than fixed 0.3/0.6 cutoffs.

## Feature importance

Impurity importance top 5:
- cat__payment_method_COD: 0.166461
- num__price_inr: 0.137116
- num__customer_tenure_days: 0.107431
- num__delivery_distance_km: 0.097244
- num__discount_pct: 0.089011

Permutation importance top 5:
- payment_method: 0.108765 ± 0.014651
- product_category: 0.007466 ± 0.005171
- discount_pct: 0.006518 ± 0.004015
- delivery_days: 0.004990 ± 0.005970
- num_previous_returns: 0.003729 ± 0.005124

Interpretation: `delivery_distance_km` appears in the impurity top five but its permutation importance is negative, illustrating why tree split importance can overrate noisy/high-cardinality continuous features. The required feature `payment_method` is dominant in both views; `discount_pct` and `num_previous_returns` also survive the permutation ranking.

## Subgroups

| dimension        | group        |   n |   precision |    recall |
|:-----------------|:-------------|----:|------------:|----------:|
| product_category | Apparel      | 385 |    0.28169  | 0.6       |
| product_category | Beauty       | 116 |    0.431818 | 0.612903  |
| product_category | Electronics  | 261 |    0.293478 | 0.519231  |
| product_category | Footwear     | 217 |    0.336449 | 0.642857  |
| product_category | Home         | 221 |    0.220183 | 0.705882  |
| payment_method   | COD          | 503 |    0.317328 | 0.980645  |
| payment_method   | Prepaid_Card | 283 |    0.130435 | 0.0612245 |
| payment_method   | Prepaid_UPI  | 294 |    0.228571 | 0.166667  |
| payment_method   | Wallet       | 120 |    0.107143 | 0.142857  |

The weakest subgroup by recall is payment_method=Prepaid_Card (recall 0.061). A concrete follow-up is to validate a subgroup-specific threshold for this payment method using cross-validation, rather than using one global threshold blindly.