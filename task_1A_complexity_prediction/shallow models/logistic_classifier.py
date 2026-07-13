"""
Task 1A — Logistic Regression binary classifier (needs simplification: yes/no)

Procedura in due fasi (coerente con BERTino-cls):
  1. selezione di C via 5-fold group CV sul train, massimizzando ROC-AUC
  2. calibrazione della soglia decisionale sul VALIDATION set (massimizza F1 sulla classe positiva), poi applicata invariata al TEST

Uso:
    python logistic_classifier.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────────────
OUT_DIR  = "logistic_cls"
DATA_DIR = "/code/HLTproject_code/data"
SEED     = 42

# ── Data ─────────────────────────────────────────────────────────────────────
print("Loading data from parquet…")
train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
val   = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
test  = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

feat_cols = [c for c in train.columns
             if c not in ("original_sentence_idx", "text", "score", "is_original")]

y_train = train["is_original"].astype(int)
y_val   = val["is_original"].astype(int).values
y_test  = test["is_original"].astype(int)

# ── FASE 1: selezione di C via GroupKFold CV, criterio ROC-AUC ───────────────
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=SEED),
)
param_grid = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

gs = GridSearchCV(
    pipeline, param_grid,
    cv=GroupKFold(n_splits=5),
    scoring="roc_auc",          # threshold-free: seleziona il miglior ranker
    n_jobs=-1,
    verbose=1,
)
gs.fit(train[feat_cols], y_train, groups=train["original_sentence_idx"])

print(f"\nBest params: {gs.best_params_}")
print(f"Best CV ROC-AUC: {gs.best_score_:.4f}")
model = gs.best_estimator_

# ── FASE 2: calibrazione della soglia sul VALIDATION set ─────────────────────
val_probs = model.predict_proba(val[feat_cols])[:, 1]
cands = np.quantile(val_probs, np.linspace(0.1, 0.9, 81))
thr   = max(cands, key=lambda t: f1_score(y_val, (val_probs > t).astype(int)))
print(f"Soglia calibrata su val: {thr:.4f}  (default sarebbe 0.5)")

# ── Valutazione sul TEST con la soglia calibrata ─────────────────────────────
test_probs = model.predict_proba(test[feat_cols])[:, 1]
test_preds = (test_probs > thr).astype(int)

auc = roc_auc_score(y_test, test_probs)
print(f"\nAUC (test): {auc:.4f}\n")
print(classification_report(y_test, test_preds,
                            target_names=["semplice", "complesso"], digits=3))

tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
print(f"Confusion matrix — TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  FP (semplice -> complesso): {fp}")
print(f"  FN (complesso -> semplice): {fn}")
print(f"  {'FP > FN: over-predice complessita' if fp > fn else 'FN > FP: sotto-predice complessita'}")

# ── Save ─────────────────────────────────────────────────────────────────────
report = classification_report(y_test, test_preds,
                               target_names=["semplice", "complesso"],
                               digits=3, output_dict=True)

metrics = {
    "best_params":  gs.best_params_,
    "cv_roc_auc":   round(float(gs.best_score_), 4),
    "threshold":    round(float(thr), 4),
    "auc_test":     round(float(auc), 4),
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "classification_test": {
        "semplice":     {k: round(v, 4) for k, v in report["semplice"].items()},
        "complesso":    {k: round(v, 4) for k, v in report["complesso"].items()},
        "macro avg":    {k: round(v, 4) for k, v in report["macro avg"].items()},
        "weighted avg": {k: round(v, 4) for k, v in report["weighted avg"].items()},
    },
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSalvato in {OUT_DIR}/eval_results.json")