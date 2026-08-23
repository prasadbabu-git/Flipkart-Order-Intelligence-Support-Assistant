from __future__ import annotations
import json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'orders_dataset.csv'
MODELS = ROOT / 'models'; RESULTS = ROOT / 'results'
MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = 'returned'
CATEGORICAL = ['product_category', 'payment_method']
NUMERIC = [
    'price_inr','discount_pct','customer_tenure_days','num_previous_orders',
    'num_previous_returns','delivery_distance_km','delivery_days','is_weekend_order','rating_given'
]

def metrics(y, pred, proba=None):
    out = {
        'accuracy': accuracy_score(y, pred),
        'precision_class1': precision_score(y, pred, zero_division=0),
        'recall_class1': recall_score(y, pred, zero_division=0),
        'f1_class1': f1_score(y, pred, zero_division=0),
    }
    if proba is not None:
        out['roc_auc'] = roc_auc_score(y, proba)
    return out

def build_preprocessor():
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('num', num_pipe, NUMERIC),
        ('cat', cat_pipe, CATEGORICAL),
    ])

def threshold_table(y, proba):
    rows = []
    for t in np.round(np.arange(0.10, 0.901, 0.02), 2):
        pred = (proba >= t).astype(int)
        rows.append({'threshold': float(t), **metrics(y, pred, None)})
    return pd.DataFrame(rows)

def main():
    df = pd.read_csv(DATA)
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Baseline
    baseline = Pipeline([
        ('prep', build_preprocessor()),
        ('model', DummyClassifier(strategy='most_frequent')),
    ])
    baseline.fit(X_train, y_train)
    bpred = baseline.predict(X_test)
    baseline_metrics = metrics(y_test, bpred)

    # Logistic Regression
    logreg = Pipeline([
        ('prep', build_preprocessor()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE)),
    ])
    logreg.fit(X_train, y_train)
    lp = logreg.predict_proba(X_test)[:,1]
    lpred05 = (lp >= 0.5).astype(int)
    logreg_05 = metrics(y_test, lpred05, lp)
    log_thresh = threshold_table(y_test, lp)
    best_log_row = log_thresh.loc[log_thresh['f1_class1'].idxmax()]
    t_log = float(best_log_row['threshold'])
    lpred_t = (lp >= t_log).astype(int)
    logreg_tuned = metrics(y_test, lpred_t, lp)

    # Random Forest grid search
    rf_pipe = Pipeline([
        ('prep', build_preprocessor()),
        ('model', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [6, 10, None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(rf_pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, refit=True, return_train_score=False)
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    rp = rf.predict_proba(X_test)[:,1]
    rpred05 = (rp >= 0.5).astype(int)
    rf_05 = metrics(y_test, rpred05, rp)
    rf_thresh = threshold_table(y_test, rp)
    best_rf_row = rf_thresh.loc[rf_thresh['f1_class1'].idxmax()]
    t_rf = float(best_rf_row['threshold'])
    rpred_t = (rp >= t_rf).astype(int)
    rf_tuned = metrics(y_test, rpred_t, rp)

    # Required recall improvement for tuned logistic vs 0.5
    log_recall_gain = logreg_tuned['recall_class1'] - logreg_05['recall_class1']
    rf_recall_gain = rf_tuned['recall_class1'] - rf_05['recall_class1']

    # Feature importance
    prep = rf.named_steps['prep']; model = rf.named_steps['model']
    feature_names = prep.get_feature_names_out()
    importances = model.feature_importances_
    fi = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    fi.head(5).to_csv(RESULTS/'rf_top5_feature_importance.csv', index=False)

    # Permutation importance on held-out test data, F1 scorer for class 1
    perm = permutation_importance(rf, X_test, y_test, scoring='f1', n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
    perm_df = pd.DataFrame({'feature': X_test.columns, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std}).sort_values('importance_mean', ascending=False)
    perm_df.to_csv(RESULTS/'rf_permutation_importance.csv', index=False)

    # Subgroup analysis at t*_rf
    test_full = df.loc[idx_test].copy()
    test_full['pred_proba'] = rp
    test_full['pred'] = rpred_t
    subgroup_rows = []
    for col in ['product_category', 'payment_method']:
        for value, g in test_full.groupby(col):
            subgroup_rows.append({
                'dimension': col, 'group': value, 'n': int(len(g)),
                'precision': precision_score(g['returned'], g['pred'], zero_division=0),
                'recall': recall_score(g['returned'], g['pred'], zero_division=0),
            })
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(RESULTS/'rf_subgroup_metrics.csv', index=False)

    # Save final pipeline and thresholds
    joblib.dump(rf, MODELS/'return_risk_model.pkl')
    threshold_info = {
        'logistic_optimal_threshold': t_log,
        'random_forest_optimal_threshold_t_star_rf': t_rf,
        'logistic_recall_gain_vs_0.5': log_recall_gain,
        'random_forest_recall_gain_vs_0.5': rf_recall_gain,
        'random_forest_best_params': grid.best_params_,
        'random_forest_best_cv_roc_auc': grid.best_score_,
        'random_forest_test_roc_auc': rf_tuned['roc_auc'],
        'cv_test_auc_gap': abs(grid.best_score_ - rf_tuned['roc_auc']),
    }
    (RESULTS/'thresholds.json').write_text(json.dumps(threshold_info, indent=2), encoding='utf-8')
    log_thresh.to_csv(RESULTS/'logistic_threshold_sweep.csv', index=False)
    rf_thresh.to_csv(RESULTS/'rf_threshold_sweep.csv', index=False)

    result = {
        'dataset': {'rows': len(df), 'columns': len(df.columns), 'return_rate': float(df.returned.mean()), 'missing_rating_rate': float(df.rating_given.isna().mean())},
        'missingness': {
            'cod_missing_rate': float(df.loc[df.payment_method=='COD','rating_given'].isna().mean()),
            'non_cod_missing_rate': float(df.loc[df.payment_method!='COD','rating_given'].isna().mean()),
            'classification': 'MAR',
        },
        'baseline': baseline_metrics,
        'logistic_regression_0.5': logreg_05,
        'logistic_regression_tuned': {**logreg_tuned, 'threshold': t_log},
        'random_forest_0.5': rf_05,
        'random_forest_tuned': {**rf_tuned, 'threshold': t_rf},
        'grid_search': {'best_params': grid.best_params_, 'best_cv_roc_auc': grid.best_score_, 'test_roc_auc_gap': threshold_info['cv_test_auc_gap']},
        'rf_top5_features': fi.head(5).to_dict('records'),
        'rf_top5_permutation': perm_df.head(5).to_dict('records'),
        'subgroups': subgroup.to_dict('records'),
    }
    (RESULTS/'part1_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    print(json.dumps({
        'dataset': result['dataset'],
        'baseline': baseline_metrics,
        'logistic_0.5': logreg_05,
        'logistic_tuned': logreg_tuned,
        'rf_0.5': rf_05,
        'rf_tuned': rf_tuned,
        't_star_rf': t_rf,
        'best_params': grid.best_params_,
        'best_cv_auc': grid.best_score_,
    }, indent=2))

if __name__ == '__main__': main()
