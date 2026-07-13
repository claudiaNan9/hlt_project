"""
Task 1A — BERTino classificazione binaria (complesso=1 / semplice=0)
         con learning-rate search + early stopping su val

Single GPU:
    python bertino_cls.py

Multi-GPU (es. GPU 1 e 2):
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 bertino_cls.py
"""

import json
import os

import pandas as pd
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Config ───────────────────────────────────────────────────────────────────
SEED    = 42
MODEL   = "indigo-ai/BERTino"
OUT_DIR = "bert_cls"
LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5]

set_seed(SEED)

local_rank = int(os.environ.get("LOCAL_RANK", -1))
is_main    = local_rank in (-1, 0)

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/code/HLTproject_code/data"
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
        d[["text", "is_original"]]
        .rename(columns={"is_original": "labels"})
        .assign(labels=lambda df: df["labels"].astype(int))
        .reset_index(drop=True)
    )
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

train_ds = make_ds(train)
val_ds   = make_ds(val)
test_ds  = make_ds(test)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs  = softmax(logits, axis=-1)[:, 1]          # P(complesso)
    preds  = logits.argmax(-1)
    labels = p.label_ids

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    return {
        "f1":  f1_score(labels, preds),
        "acc": accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
    }

# ── LR search con early stopping ─────────────────────────────────────────────
best_val_f1  = -float("inf")
best_lr      = None
best_trainer = None

for lr in LR_GRID:
    if is_main:
        print(f"\n{'='*50}\nLR = {lr:.0e}\n{'='*50}")

    run_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2
    )
    run_dir = os.path.join(OUT_DIR, f"lr_{lr:.0e}")

    run_args = TrainingArguments(
        output_dir=run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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
    val_metrics = run_trainer.evaluate()
    val_f1      = val_metrics["eval_f1"]

    if is_main:
        print(f"LR={lr:.0e} → val F1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        best_lr      = lr
        best_trainer = run_trainer

# ── Valutazione finale ────────────────────────────────────────────────────────
if is_main:
    print(f"\nBest LR: {best_lr:.0e} (val F1={best_val_f1:.4f})")

test_out = best_trainer.predict(test_ds)
val_out  = best_trainer.predict(val_ds)

if is_main:
    def extract_logits(out):
        return out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions

    def classifier_metrics(out, y_true):
        logits = extract_logits(out)
        probs  = softmax(logits, axis=-1)[:, 1]   # P(complesso)
        preds  = logits.argmax(-1)                # decisione nativa del classifier
        return {
            "auc":       roc_auc_score(y_true, probs),
            "f1":        f1_score(y_true, preds),
            "precision": precision_score(y_true, preds),
            "recall":    recall_score(y_true, preds),
            "acc":       accuracy_score(y_true, preds),
        }, preds

    y_val  = val["is_original"].astype(int).values
    y_test = test["is_original"].astype(int).values

    cls_val,  _          = classifier_metrics(val_out,  y_val)
    cls_test, test_preds = classifier_metrics(test_out, y_test)
    print("Classifier (val ):", cls_val)
    print("Classifier (test):", cls_test)

    print(classification_report(
        y_test,
        test_preds,
        target_names=["semplice", "complesso"],
        digits=3,
    ))

    metrics = {
        "best_lr":     best_lr,
        "classification_test":    cls_test,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
