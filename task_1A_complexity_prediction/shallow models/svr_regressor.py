"""
Task 1A — LinearSVR regression (readability score prediction)

Uso:
    python svr_regressor.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from task_1A_complexity_prediction.scripts.utils import vista_binaria

# ── Config ───────────────────────────────────────────────────────────────────
SEED     = 42
OUT_DIR  = "svr_reg"
DATA_DIR = "data"

# ── Data ─────────────────────────────────────────────────────────────────────
print("Loading data from parquet…")
train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
val   = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
test  = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

feat_cols = [c for c in train.columns
             if c not in ("original_sentence_idx", "text", "score", "is_original")]

# ── GridSearchCV su C ─────────────────────────────────────────────────────────
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LinearSVR(epsilon=0.0, loss="squared_epsilon_insensitive",
              dual=False, max_iter=10_000, random_state=SEED),
)
param_grid = {"linearsvr__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

gs = GridSearchCV(
    pipeline, param_grid,
    cv=GroupKFold(n_splits=5),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
gs.fit(
    train[feat_cols], train["score"],
    groups=train["original_sentence_idx"],
)
print(f"Best params: {gs.best_params_}")
model = gs.best_estimator_

val_pred  = model.predict(val[feat_cols])
test_pred = model.predict(test[feat_cols])

# ── Regression metrics (test) ─────────────────────────────────────────────────
metrics = {
    "best_params": gs.best_params_,
    "spearman": spearmanr(test_pred, test["score"]).correlation,
    "mae":  float(np.mean(np.abs(test_pred - test["score"].values))),
    "rmse": float(np.sqrt(np.mean((test_pred - test["score"].values) ** 2))),
}
print("Regression:", metrics)

# ── Binary view — soglia calibrata su val, applicata su test ─────────────────
binary_val  = vista_binaria(val_pred,  val["is_original"].values)
binary_test = vista_binaria(test_pred, test["is_original"].values, thr=binary_val["thr"])
print("Binary (val ):", binary_val)
print("Binary (test):", binary_test)

metrics["binary_test"] = binary_test

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
    json.dump(metrics, f, indent=2)
