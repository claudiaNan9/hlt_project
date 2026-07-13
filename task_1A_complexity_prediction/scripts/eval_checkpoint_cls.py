"""
Carica un checkpoint salvato da bertino_cls.py e produce le metriche di classificazione su test senza rifare il training.

Procedura in due fasi (simmetrica alla Logistic Regression):
  1. il modello (learning rate) e' gia' stato selezionato su val durante il training
  2. la soglia decisionale viene calibrata sul VALIDATION set (massimizza F1 sulla classe positiva) 

Uso:
    python eval_checkpoint_cls.py

    # multi-GPU
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 eval_checkpoint_cls.py
"""

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT = "/code/HLTproject_code/task_1A_complexity_prediction/bert_cls/lr_5e-05/checkpoint-2828"
OUT_DIR    = "bert_cls_2"
DATA_DIR   = "/code/HLTproject_code/data"

# ── Setup ─────────────────────────────────────────────────────────────────────
local_rank = int(os.environ.get("LOCAL_RANK", -1))
is_main    = local_rank in (-1, 0)

if is_main:
    print(f"Loading checkpoint: {CHECKPOINT}")

# ── Data ─────────────────────────────────────────────────────────────────────
val  = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

tok = AutoTokenizer.from_pretrained(CHECKPOINT)

def make_ds(d: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(
        d[["text", "is_original"]]
        .rename(columns={"is_original": "labels"})
        .assign(labels=lambda df: df["labels"].astype(int))
        .reset_index(drop=True)
    )
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

val_ds  = make_ds(val)
test_ds = make_ds(test)

# ── Modello ───────────────────────────────────────────────────────────────────
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

# ── Predict ───────────────────────────────────────────────────────────────────
val_out  = trainer.predict(val_ds)
test_out = trainer.predict(test_ds)

if is_main:
    def extract_probs(out):
        logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
        return softmax(logits, axis=-1)[:, 1]        # P(complesso)

    val_probs  = extract_probs(val_out)
    test_probs = extract_probs(test_out)

    y_val  = val["is_original"].astype(int).values
    y_test = test["is_original"].astype(int).values

    # ── FASE 2: calibrazione della soglia sul validation set ─────────────────
    cands = np.quantile(val_probs, np.linspace(0.1, 0.9, 81))
    thr   = max(cands, key=lambda t: f1_score(y_val, (val_probs > t).astype(int)))
    print(f"\nSoglia calibrata su val: {thr:.4f}  (default sarebbe 0.5)")

    # ── Valutazione sul test ─────────────────────────
    test_preds = (test_probs > thr).astype(int)

    auc = roc_auc_score(y_test, test_probs)
    print(f"AUC (test): {auc:.4f}\n")
    print(classification_report(
        y_test, test_preds,
        target_names=["semplice", "complesso"], digits=3,
    ))

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    print(f"Confusion matrix — TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  FP (semplice -> complesso): {fp}")
    print(f"  FN (complesso -> semplice): {fn}")
    print(f"  {'FP > FN: over-predice complessita' if fp > fn else 'FN > FP: sotto-predice complessita'}")

    # ── Save ─────────────────────────────────────────────────────────────────
    report = classification_report(
        y_test, test_preds,
        target_names=["semplice", "complesso"],
        digits=3, output_dict=True,
    )

    metrics = {
        "checkpoint":    CHECKPOINT,
        "threshold":     round(float(thr), 4),
        "auc_test":      round(float(auc), 4),
        "accuracy_test": round(float(report["accuracy"]), 4),
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
    print(f"\nRisultati salvati in {OUT_DIR}/eval_results.json")