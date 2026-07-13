"""
Task 1A — Preparazione dati 

Scarica il dataset, costruisce train/val/test e li salva come parquet.
Split per gruppo (original_sentence_idx).
Proporzioni: ~60% train / ~20% val / ~20% test

Uso:
    python prepare_data.py

Output:
    data/train.parquet
    data/val.parquet
    data/test.parquet
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

# ── Config ───────────────────────────────────────────────────────────────────
DOMAIN  = "wikipedia_profiling"
N_ORIG  = 40_000   # numero di frasi originali da campionare
SEED    = 42
OUT_DIR = "data"

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading full dataset…")
df = load_dataset("mpapucci/impacts", DOMAIN, split="train").to_pandas()

# Campiona N_ORIG original_sentence_idx unici, poi tieni tutte le loro righe

all_orig_ids  = df["original_sentence_idx"].unique()
rng           = np.random.default_rng(SEED)
selected_ids  = set(rng.choice(all_orig_ids, size=min(N_ORIG, len(all_orig_ids)), replace=False))
df            = df[df["original_sentence_idx"].isin(selected_ids)].copy()
print(f"Originals selezionati: {len(selected_ids)} | Righe totali: {len(df)}")

# Tieni solo coppie in cui la semplificazione è effettivamente più leggibile

DELTA = 0.05
df = df[df["original_all"] - df["simplification_all"] > DELTA].copy()
print(f"Dopo filtro DELTA={DELTA}: {len(df)} righe "
      f"({df['original_sentence_idx'].nunique()} originals con almeno 1 semplificazione valida)")

orig_feats = [c for c in df.columns if c.endswith("_original")]
simp_feats = [c for c in df.columns if c.endswith("_simplification")]

def _strip(cols, suf):
    return {c: c[: -len(suf)] for c in cols}

orig = (
    df[["original_sentence_idx", "original_text", "original_all"] + orig_feats]
    .drop_duplicates("original_sentence_idx")
    .rename(columns={"original_text": "text", "original_all": "score",
                     **_strip(orig_feats, "_original")})
)
orig["is_original"] = True
valid_ids = set(df["original_sentence_idx"])
orig = orig[orig["original_sentence_idx"].isin(valid_ids)]

simp = (
    df[["original_sentence_idx", "simplification", "simplification_all"] + simp_feats]
    .rename(columns={"simplification": "text", "simplification_all": "score",
                     **_strip(simp_feats, "_simplification")})
)
simp["is_original"] = False

# ── Cap: K semplificazioni per originale ──

K = 1

simp = simp.sample(frac=1, random_state=SEED).reset_index(drop=True)   # mescola tutto
simp["_rank"] = simp.groupby("original_sentence_idx").cumcount()       # ordine casuale entro gruppo
simp = simp[simp["_rank"] < K].drop(columns="_rank").reset_index(drop=True)
print(f"Semplificazioni dopo cap K={K}: {len(simp)} "
      f"(media {len(simp)/simp['original_sentence_idx'].nunique():.1f} per originale)")

data = pd.concat([orig, simp], ignore_index=True)

feat_cols = sorted(
    set(_strip(orig_feats, "_original").values()) &
    set(_strip(simp_feats, "_simplification").values())
)
data[feat_cols] = data[feat_cols].apply(pd.to_numeric, errors="coerce")
data["score"]   = data["score"].astype("float32")
data = data.dropna(subset=["text", "score"])
data = data[data["text"].str.len() > 0]

# ── Split 60 / 20 / 20 per gruppo ────────────────────────────────────────────

# Passo 1: separa test (20%) da train_val (80%)

tv_idx, te_idx = next(
    GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
    .split(data, groups=data["original_sentence_idx"])
)
train_val, test = data.iloc[tv_idx].copy(), data.iloc[te_idx].copy()

# Passo 2: dal train_val, separa val (25% di train_val ≈ 20% del totale)

tr_idx, va_idx = next(
    GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
    .split(train_val, groups=train_val["original_sentence_idx"])
)
train, val = train_val.iloc[tr_idx].copy(), train_val.iloc[va_idx].copy()

print(f"Totale: {len(data)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
for name, split in [("train/val", (train, val)), ("train/test", (train, test)), ("val/test", (val, test))]:
    overlap = set(split[0]["original_sentence_idx"]) & set(split[1]["original_sentence_idx"])
    print(f"Overlap {name}: {len(overlap)} (deve essere 0)")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
train.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
val.to_parquet(os.path.join(OUT_DIR,   "val.parquet"),   index=False)
test.to_parquet(os.path.join(OUT_DIR,  "test.parquet"),  index=False)
print(f"Salvati in {OUT_DIR}/")
