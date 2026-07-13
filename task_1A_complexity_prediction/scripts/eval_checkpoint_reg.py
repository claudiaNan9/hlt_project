"""
Carica un checkpoint salvato da bertino.py o bert_italian_xxl.py
e produce le metriche su val e test senza rifare la LR search.

Uso:
    python eval_checkpoint.py

    # multi-GPU
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 eval_checkpoint.py
"""

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.stats import spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from task_1A_complexity_prediction.scripts.utils import vista_binaria

# ── Config — cambia questi due parametri ─────────────────────────────────────
CHECKPOINT = "bert_reg/lr_5e-05/checkpoint-1886"
OUT_DIR    = "bert_reg"
DATA_DIR   = "data"

# ── Data ─────────────────────────────────────────────────────────────────────
local_rank = int(os.environ.get("LOCAL_RANK", -1))
is_main    = local_rank in (-1, 0)

if is_main:
    print(f"Loading checkpoint: {CHECKPOINT}")

val  = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

tok = AutoTokenizer.from_pretrained(CHECKPOINT)

def make_ds(d: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(
        d[["text", "score"]].rename(columns={"score": "labels"}).reset_index(drop=True)
    )
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

val_ds  = make_ds(val)
test_ds = make_ds(test)

# ── Carica modello dal checkpoint ────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_eval_batch_size=16,
    fp16=True,
    eval_accumulation_steps=8,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
)

# ── Predict (tutti i rank) ───────────────────────────────────────────────────
test_out = trainer.predict(test_ds)
val_out  = trainer.predict(val_ds)

if is_main:
    test_preds = test_out.predictions
    if isinstance(test_preds, tuple):
        test_preds = test_preds[0]
    test_preds = test_preds.squeeze(-1).astype("float64")

    val_preds = val_out.predictions
    if isinstance(val_preds, tuple):
        val_preds = val_preds[0]
    val_preds = val_preds.squeeze(-1).astype("float64")

    metrics = {
        "checkpoint": CHECKPOINT,
        "spearman": spearmanr(test_preds, test["score"].values).correlation,
        "mae":  float(np.mean(np.abs(test_preds - test["score"].values))),
        "rmse": float(np.sqrt(np.mean((test_preds - test["score"].values) ** 2))),
    }
    print("Regression (test):", metrics)

    binary_val  = vista_binaria(val_preds,  val["is_original"].values)
    binary_test = vista_binaria(test_preds, test["is_original"].values, thr=binary_val["thr"])
    print("Binary (val ):", binary_val)
    print("Binary (test):", binary_test)

    metrics["binary_test"] = binary_test

    with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
