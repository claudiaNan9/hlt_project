import pickle
from parse_responses import extract_annotation   # riusa il parser robusto

# ── CONFIG: cambia i nomi file secondo il run che vuoi valutare ───────────────
EVAL_PKL = "eval_set_venti.pkl"          # frasi (servono i token per la lunghezza)
RESP_PKL = "llm_resp_anita_venti.pkl"    # risposte grezze del modello
OUT_PKL  = "y_pred.pkl"            # vettori binari in uscita

def to_binary(complex_idx, n_tokens):
    """Lista di indici complessi -> vettore [0,1,0,...] lungo n_tokens."""
    labels = [0] * n_tokens
    for i in complex_idx:
        if 0 <= i < n_tokens:          # guardia anti out-of-range
            labels[i] = 1
    return labels

if __name__ == "__main__":
    with open(EVAL_PKL, "rb") as f:
        eval_set = pickle.load(f)
    with open(RESP_PKL, "rb") as f:
        resp = pickle.load(f)

    y_pred = []
    n_failed = 0
    for ex, raw in zip(eval_set, resp):
        n = len(ex["tokens"])
        parsed, status = extract_annotation(raw, n_tokens=n)
        if parsed is None:
            n_failed += 1
            idx = []                   # parsing fallito -> nessun token complesso
        else:
            idx = parsed["complex"]
        y_pred.append(to_binary(idx, n))

    with open(OUT_PKL, "wb") as f:
        pickle.dump(y_pred, f)

    print(f"Generati {len(y_pred)} vettori binari -> {OUT_PKL}")
    print(f"Parsing falliti (vettore tutto 0): {n_failed}")

    # anteprima: frase + vettore per i primi esempi
    print("\n--- anteprima ---")
    for i in range(min(3, len(eval_set))):
        toks = eval_set[i]["tokens"]
        vec  = y_pred[i]
        print(f"\nesempio {i}:")
        for t, b in zip(toks, vec):
            mark = " <<<" if b == 1 else ""
            print(f"  {b}  {t}{mark}")