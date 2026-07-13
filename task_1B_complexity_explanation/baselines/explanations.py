"""
Task 1B — Complexity explanation (italiano)
===========================================

Filo logico unico (dall'alto in basso):

  0. Setup            : spaCy + costanti
  1. Ground truth     : silver-label TIPATE (keep / del / sub) via allineamento
  2. Dati             : coppie (originale, semplificazione) -> eval_set
  3. Risorse          : Vocabolario di Base
  4. Metriche         : tokenwise P/R/F1, Edit Distance, TER  (per tutti i metodi)
  5. Metodi           : random, Vocabolario di Base, (IG ed LLM fatti in notebook separati)
  6. Spiegazione      : il tipo di operazione, livello sintattico/lessicale ricavato da ProfilingUD

MODELLO 
---------------
  * Ogni metodo (sez. 5) restituisce `preds`: una lista di vettori binari,
    preds[i] allineato a eval_set[i]["tokens"]  (1 = token evidenziato).
  * La valutazione avviene con la funzione evaluate(eval_set, preds).

"""

import numpy as np
import pandas as pd
import spacy
from difflib import SequenceMatcher
from sklearn.metrics import precision_recall_fscore_support
from sacrebleu.metrics import TER
import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
import unicodedata
from difflib import SequenceMatcher

# =============================================================================
# 0. SETUP
# =============================================================================
nlp = spacy.load("it_core_news_md") 
SEED = 42

# relazioni di dipendenza che segnalano una clausola subordinata/participiale 
SUB_DEPS    = {"advcl", "acl", "acl:relcl", "ccomp", "xcomp", "csubj"}
PASS_DEPS   = {"aux:pass", "nsubj:pass"}
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


# =============================================================================
# 1. GROUND TRUTH — silver-label tipate (keep / del / sub)
# =============================================================================

def _norm(s):
    """Normalizza per il confronto: minuscolo, senza accenti, senza apostrofi."""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))   # toglie accenti
    s = s.replace("'", "").replace("’", "").replace("`", "")
    return s

def _similar(a, b, thr=0.80):
    """True se due token sono quasi-uguali (variante ortografica, non vera modifica)."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    # uno contiene l'altro (es. 'panamericani' vs 'panamericani' splittato)
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= thr


def silver_labels(original, simplification,
                     mode="both",          # "both" = del+sub ; "sub_only" = solo sostituzioni
                     ignore_expansions=True,
                     guard_false_del=True):
    """
    Etichette silver migliorate da una coppia (originale, semplificata).

    Migliorie rispetto alla versione base:
    - confronto su lemmi normalizzati (niente falsi 'sub' per accenti/apostrofi/maiuscole);
    - dentro i blocchi 'replace', i token che hanno una controparte quasi-identica
      nel testo semplificato restano 'keep' (varianti ortografiche, non semplificazioni);
    - le espansioni (semplificata più lunga che aggiunge materiale) non generano 'sub' spuri;
    - guardia anti falso-'del': se il token compare ancora (normalizzato) nella semplificata,
      probabilmente è un disallineamento del matcher -> 'keep';
    - mode='sub_only' per marcare solo le sostituzioni lessicali (segnale più pulito).

    Ritorna: tokens (list[str]), labels (list[int 0/1]), tipo (list["keep"|"del"|"sub"]).
    """
    doc_o = [t for t in nlp(original) if not t.is_space]
    doc_s = [t for t in nlp(simplification) if not t.is_space]

    tok_o = [t.text for t in doc_o]
    lem_o = [t.lemma_.lower() for t in doc_o]
    lem_s = [t.lemma_.lower() for t in doc_s]
    txt_o = tok_o
    txt_s = [t.text for t in doc_s]

    # insieme normalizzato dei token semplificati
    norm_s_set = {_norm(w) for w in txt_s}

    tipo = ["keep"] * len(doc_o)

    sm = SequenceMatcher(a=lem_o, b=lem_s, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():

        if tag == "delete":
            # cancellazione pura: marca, ma con guardia anti-disallineamento
            for i in range(i1, i2):
                if guard_false_del and _norm(txt_o[i]) in norm_s_set:
                    continue                      # il token c'è ancora altrove -> keep
                tipo[i] = "del"

        elif tag == "replace":
            span_b = txt_s[j1:j2]                  # token semplificati corrispondenti

            # espansione: il blocco semplificato è molto più lungo dell'originale
            # => è materiale aggiunto, non una sostituzione di ciò che c'era
            if ignore_expansions and len(span_b) > 2 * max(1, (i2 - i1)):
                # tratta come keep salvo i token che spariscono del tutto
                for i in range(i1, i2):
                    if _norm(txt_o[i]) not in norm_s_set and not any(
                            _similar(txt_o[i], b) for b in span_b):
                        tipo[i] = "del"
                continue

            for i in range(i1, i2):
                # variante ortografica? il token ha un quasi-gemello nel blocco b -> keep
                if any(_similar(txt_o[i], b) for b in span_b):
                    continue
                # token ancora presente identico altrove nella semplificata -> keep
                if guard_false_del and _norm(txt_o[i]) in norm_s_set:
                    continue
                tipo[i] = "sub"

    # applica il mode
    if mode == "sub_only":
        labels = [int(x == "sub") for x in tipo]
    else:  # both
        labels = [int(x != "keep") for x in tipo]

    return txt_o, labels, tipo

# =============================================================================
# 2. DATI — coppie di valutazione + eval_set
# =============================================================================
def build_pairs(test_parquet="data/test.parquet"):
    """Ricostruisce le coppie (originale, semplificazione) dal test set melted."""
    test = pd.read_parquet(test_parquet)
    orig = (test[test["is_original"]].drop_duplicates("original_sentence_idx")
              .set_index("original_sentence_idx")["text"])
    simp = (test[~test["is_original"]].drop_duplicates("original_sentence_idx")
              .set_index("original_sentence_idx")["text"])
    pairs = (pd.concat([orig.rename("original"), simp.rename("simplification")], axis=1)
               .dropna().reset_index())
    return pairs, test


def build_eval_set(pairs):
    """
    Per ogni coppia salva tutto cio' che serve a metodi e spiegazione:
        tokens, lemmas, pos, dep, silver, tipo, simp_tokens
    """
    eval_set = []
    for _, row in pairs.iterrows():
        doc_o = [t for t in nlp(row["original"]) if not t.is_space]
        tokens, labels, tipo = silver_labels(row["original"], row["simplification"])
        eval_set.append({
            "idx":         row["original_sentence_idx"],
            "tokens":      tokens,
            "lemmas":      [t.lemma_.lower() for t in doc_o],
            "pos":         [t.pos_ for t in doc_o],
            "dep":         [t.dep_ for t in doc_o],
            "silver":      labels,
            "tipo":        tipo,
            "simp_tokens": [t.text for t in nlp(row["simplification"]) if not t.is_space],
        })
    return eval_set


# =============================================================================
# 3. RISORSE — Vocabolario di Base
# =============================================================================
def carica_vdb(path):
    """Carica i lemmi del Nuovo Vocabolario di Base (De Mauro) in un set."""
    vdb = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().split("\t")[0].split(",")[0].lower()   # adatta al formato del file
            if w and not w.startswith("#"):
                vdb.add(w)
    return vdb


# =============================================================================
# 4. METRICHE — tokenwise P/R/F1, Edit Distance, TER  (UNICA per tutti i metodi)
# =============================================================================
## metriche di valutazione (precision, recall, F1, edit distance, TER)
## per predizioni binarie allineate a eval_set[i]["tokens"] con alcuni check aggiuntivi

_ter = TER()

def word_edit_distance(a, b, sub_cost=1.0):
    """Levenshtein a livello di parola"""
    n, m = len(a), len(b)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+sub_cost)
    return dp[n][m]


def check_alignment(eval_set, preds, verbose=True, max_show=10):
    """
    Verifica che, per ogni frase, len(tokens) == len(silver) == len(pred).
    Ritorna la lista degli indici disallineati. NON modifica i dati.
    """
    bad = []
    for i, (ex, pred) in enumerate(zip(eval_set, preds)):
        n_tok = len(ex["tokens"])
        n_sil = len(ex["silver"])
        n_prd = len(pred)
        if not (n_tok == n_sil == n_prd):
            bad.append((i, n_tok, n_sil, n_prd))

    if len(preds) != len(eval_set):
        print(f"!! ATTENZIONE: {len(preds)} predizioni ma {len(eval_set)} frasi nel set")

    if verbose:
        if not bad:
            print(f"OK: tutte {len(eval_set)} le frasi sono allineate "
                  f"(tokens == silver == pred).")
        else:
            print(f"!! DISALLINEAMENTO in {len(bad)}/{len(eval_set)} frasi.")
            print("   idx | len(tokens) len(silver) len(pred)")
            for row in bad[:max_show]:
                i, nt, ns, npd = row
                flag_sil = "" if nt == ns else "  <- silver != tokens"
                flag_prd = "" if nt == npd else "  <- pred != tokens"
                print(f"   {i:4d} |   {nt:4d}      {ns:4d}     {npd:4d}{flag_sil}{flag_prd}")
            if len(bad) > max_show:
                print(f"   ... e altre {len(bad)-max_show} frasi")
            print("\n   -> Finché c'è disallineamento, precision/recall/F1 non sono "
                  "affidabili:\n      si confrontano posizioni di token diverse.")
    return bad


def evaluate(eval_set, preds, strict=True):
    """
    preds[i] = vettore binario allineato a eval_set[i]['tokens'].
    Se strict=True, interrompe quando trova frasi disallineate.
    """
    bad = check_alignment(eval_set, preds)
    if bad and strict:
        raise ValueError(
            f"{len(bad)} frasi disallineate: correggi l'allineamento prima di "
            f"calcolare le metriche, oppure passa strict=False per saltarle.")

    P, R, F1, ED1, ED15, ED2, TERs = ([] for _ in range(7))
    skipped = 0
    for ex, pred in zip(eval_set, preds):
        p = list(pred)
        y = ex["silver"]

        # salta le frasi disallineate (solo se strict=False)
        if len(y) != len(p) or len(p) != len(ex["tokens"]):
            skipped += 1
            continue

        pr, rc, f1, _ = precision_recall_fscore_support(
            y, p, average="binary", pos_label=1, zero_division=0)
        P.append(pr); R.append(rc); F1.append(f1)

        # parte NON evidenziata di d  vs  d'
        kept    = [t for t, pl in zip(ex["tokens"], p) if pl == 0]
        d_prime = ex["simp_tokens"]
        ED1.append( word_edit_distance(kept, d_prime, 1.0))
        ED15.append(word_edit_distance(kept, d_prime, 1.5))
        ED2.append( word_edit_distance(kept, d_prime, 2.0))
        TERs.append(_ter.sentence_score(" ".join(kept), [" ".join(d_prime)]).score)

    if skipped:
        print(f"(saltate {skipped} frasi disallineate nel calcolo)")

    return {
        "precision": float(np.mean(P)),  "recall": float(np.mean(R)),
        "f1": float(np.mean(F1)),
        "ed_sub1":  float(np.mean(ED1)), "ed_sub1.5": float(np.mean(ED15)),
        "ed_sub2":  float(np.mean(ED2)), "ter": float(np.mean(TERs)),
        "n_valutate": len(P),
    }

# =============================================================================
# 5. METODI — ognuno ritorna `preds` (lista di vettori binari)
# =============================================================================
def predict_random(eval_set, rate=0.3, seed=SEED):
    """Floor: evidenzia posizioni a caso (frazione `rate` dei token)."""
    rng = np.random.default_rng(seed)
    preds = []
    for ex in eval_set:
        n = len(ex["tokens"])
        k = int(round(rate * n))
        idx = rng.choice(n, size=min(k, n), replace=False)
        p = [0] * n
        for i in idx:
            p[i] = 1
        preds.append(p)
    return preds


def predict_vdb(eval_set, vdb):
    """Baseline lessicale: evidenzia le parole di CONTENUTO il cui lemma NON e' nel VdB."""
    return [[int(pos in CONTENT_POS and lem not in vdb)
             for lem, pos in zip(ex["lemmas"], ex["pos"])]
            for ex in eval_set]


# =============================================================================
# 6. SPIEGAZIONE — il "perche" (qualitativo, NON valutato dalle metriche)
# =============================================================================
def spiega_token(ex, vdb):
    """
    Spiegazione LOCALE, per ogni token complesso di una frase:
        operazione (del/sub) + motivo sintattico/lessicale dal parse.
    """
    out = []
    for tok, lem, pos, dep, lab, tp in zip(
            ex["tokens"], ex["lemmas"], ex["pos"], ex["dep"], ex["silver"], ex["tipo"]):
        if not lab:
            continue
        if dep in SUB_DEPS:
            motivo = "clausola subordinata/participiale (sintattico)"
        elif dep in PASS_DEPS:
            motivo = "costruzione passiva (sintattico)"
        elif pos in CONTENT_POS and lem not in vdb:
            motivo = "parola non di base (lessicale)"
        else:
            motivo = "altro"
        out.append({"token": tok, "operazione": tp, "motivo": motivo})
    return out


def delta_profiling(test, feat_cols=None):
    """
    Spiegazione GLOBALE per frase: quanto CALA ogni feature ProfilingUD da d a d'.
    Ritorna un DataFrame z-normalizzato (index = original_sentence_idx);
    valori alti = dimensione di complessita' ridotta di piu' nella semplificazione.

    NB sulla direzione: per essere rigorosa, pesa per il segno della correlazione
    della feature col punteggio di complessita' (o col coefficiente del modello 1A),
    cosi' emergono solo le feature mosse verso il "semplice".
    """
    meta = {"original_sentence_idx", "text", "score", "is_original"}
    if feat_cols is None:
        feat_cols = [c for c in test.columns if c not in meta]
    of = (test[test["is_original"]].drop_duplicates("original_sentence_idx")
            .set_index("original_sentence_idx")[feat_cols])
    sf = (test[~test["is_original"]].drop_duplicates("original_sentence_idx")
            .set_index("original_sentence_idx")[feat_cols])
    delta = (of - sf).dropna()
    return (delta - delta.mean()) / delta.std(ddof=0)


# =============================================================================
# USO (esempio end-to-end)
# =============================================================================
if __name__ == "__main__":
    import json, os

    OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(OUT_DIR, exist_ok=True)

    pairs, test = build_pairs("/code/HLTproject_code/data/test.parquet")
    eval_set    = build_eval_set(pairs)
    vdb         = carica_vdb('/code/HLTproject_code/nvdb/nvdb.words.txt')
    print(f"Frasi di valutazione: {len(eval_set)} | VdB: {len(vdb)} lemmi")

    # --- Metodi -> metriche ---
    metrics = {
        "random": evaluate(eval_set, predict_random(eval_set)),
        "vdb":    evaluate(eval_set, predict_vdb(eval_set, vdb)),
    }
    print("RANDOM :", metrics["random"])
    print("VdB    :", metrics["vdb"])

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetriche salvate in: {metrics_path}")

    # --- Spiegazione LOCALE (tutte le frasi) ---
    rows = []
    for ex in eval_set:
        for e in spiega_token(ex, vdb):
            rows.append({"idx": ex["idx"], **e})
    expl_df = pd.DataFrame(rows)
    expl_path = os.path.join(OUT_DIR, "explanations_local.csv")
    expl_df.to_csv(expl_path, index=False, encoding="utf-8")
    print(f"Spiegazioni locali  salvate in: {expl_path}")

    # Stampa esempio (frase 0)
    print("\nSpiegazione token-level (frase 0):")
    for e in spiega_token(eval_set[0], vdb):
        print(" ", e)

    # --- Spiegazione GLOBALE ProfilingUD (tutte le frasi) ---
    dz = delta_profiling(test)
    delta_path = os.path.join(OUT_DIR, "delta_profiling.csv")
    dz.to_csv(delta_path, encoding="utf-8")
    print(f"\nDelta ProfilingUD   salvato in: {delta_path}")

    # Stampa esempio (frase 0)
    print("\nDimensioni ProfilingUD ridotte di piu' (frase 0):")
    print(dz.loc[eval_set[0]["idx"]].sort_values(ascending=False).head(5))