"""
Task 1A — BERTino regression (readability score prediction)
         con learning-rate search + early stopping su val

Single GPU:
    python bertino_reg.py

Multi-GPU (es. GPU 1 e 2):
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 bertino_reg.py
"""

import json
import os
import shutil

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.stats import spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from task_1A_complexity_prediction.scripts.utils import vista_binaria

# ── Config ───────────────────────────────────────────────────────────────────
SEED    = 42
MODEL   = "indigo-ai/BERTino"
OUT_DIR = "bert_reg"
LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5]

set_seed(SEED)

local_rank = int(os.environ.get("LOCAL_RANK", -1))
is_main    = local_rank in (-1, 0)

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
if is_main:
    print("Loading data from parquet…")

train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
val   = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
test  = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

if is_main:
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# ── Tokenize ─────────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL)

def make_ds(d: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(
        d[["text", "score"]].rename(columns={"score": "labels"}).reset_index(drop=True)
    )
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

train_ds = make_ds(train)
val_ds   = make_ds(val)
test_ds  = make_ds(test)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    pred  = preds.squeeze(-1).astype("float64")
    lab   = p.label_ids.astype("float64")
    return {
        "spearman": spearmanr(pred, lab).correlation,
        "mae":  float(np.mean(np.abs(pred - lab))),
        "rmse": float(np.sqrt(np.mean((pred - lab) ** 2))),
    }

# ── LR search con early stopping ─────────────────────────────────────────────
best_val_spearman = -float("inf")
best_lr           = None
best_trainer      = None

for lr in LR_GRID:
    if is_main:
        print(f"\n{'='*50}\nLR = {lr:.0e}\n{'='*50}")

    run_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, problem_type="regression"
    )
    run_dir = os.path.join(OUT_DIR, f"lr_{lr:.0e}")

    run_args = TrainingArguments(
        output_dir=run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        save_total_limit=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        learning_rate=lr,
        fp16=True,
        eval_accumulation_steps=8,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        ddp_find_unused_parameters=False,
    )

    run_trainer = Trainer(
        model=run_model,
        args=run_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    run_trainer.train()
    val_metrics      = run_trainer.evaluate()
    val_spearman     = val_metrics["eval_spearman"]

    if is_main:
        print(f"LR={lr:.0e} → val spearman={val_spearman:.4f}")

    if val_spearman > best_val_spearman:
        best_val_spearman = val_spearman
        best_lr           = lr
        best_trainer      = run_trainer

# ── Valutazione finale — predict() deve essere chiamato da tutti i rank ───────
if is_main:
    print(f"\nBest LR: {best_lr:.0e} (val spearman={best_val_spearman:.4f})")

test_out = best_trainer.predict(test_ds)
val_out  = best_trainer.predict(val_ds)

if is_main:
    test_preds = test_out.predictions
    if isinstance(test_preds, tuple):
        test_preds = test_preds[0]
    test_preds = test_preds.squeeze(-1).astype("float64")

    metrics = {
        "best_lr": best_lr,
        "spearman": spearmanr(test_preds, test["score"].values).correlation,
        "mae":  float(np.mean(np.abs(test_preds - test["score"].values))),
        "rmse": float(np.sqrt(np.mean((test_preds - test["score"].values) ** 2))),
    }
    print("Regression (test):", metrics)

    val_preds = val_out.predictions
    if isinstance(val_preds, tuple):
        val_preds = val_preds[0]
    val_preds = val_preds.squeeze(-1).astype("float64")

    binary_val  = vista_binaria(val_preds,  val["is_original"].values)
    binary_test = vista_binaria(test_preds, test["is_original"].values, thr=binary_val["thr"])
    print("Binary (val ):", binary_val)
    print("Binary (test):", binary_test)

    metrics["binary_test"] = binary_test

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # for lr in LR_GRID:
    #     run_dir = os.path.join(OUT_DIR, f"lr_{lr:.0e}")
    #     if os.path.isdir(run_dir):
    #         shutil.rmtree(run_dir)
