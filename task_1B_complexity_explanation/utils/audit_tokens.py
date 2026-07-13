## Audit per-parola: mostra QUALI token sono marcati come complessi,
## per capire a occhio se gli indici dell'LLM sono sensati o spostati.

import pickle

def audit(eval_set, preds, only_marked=True, max_show=30):
    """
    Stampa, per ogni frase, i token marcati come complessi (pred==1)
    accanto a quelli del silver (silver==1), per confronto visivo.
    """
    shown = 0
    for i, (ex, pred) in enumerate(zip(eval_set, preds)):
        toks   = ex["tokens"]
        silver = ex["silver"]
        pred   = list(pred)

        pred_words   = [toks[j] for j in range(len(toks)) if j < len(pred)   and pred[j]   == 1]
        silver_words = [toks[j] for j in range(len(toks)) if j < len(silver) and silver[j] == 1]

        if only_marked and not pred_words and not silver_words:
            continue

        print(f"\n── frase {i} ──")
        print("  testo :", " ".join(toks))
        print(f"  LLM   ({len(pred_words):2d}): {pred_words}")
        print(f"  silver({len(silver_words):2d}): {silver_words}")

        # token in comune / solo-LLM / solo-silver, a livello di POSIZIONE
        pset = {j for j in range(len(toks)) if j < len(pred)   and pred[j]   == 1}
        sset = {j for j in range(len(toks)) if j < len(silver) and silver[j] == 1}
        common = pset & sset
        if common:
            print(f"  ✓ concordi: {[toks[j] for j in sorted(common)]}")
        only_llm = pset - sset
        if only_llm:
            print(f"  + solo LLM: {[(j, toks[j]) for j in sorted(only_llm)]}")
        only_sil = sset - pset
        if only_sil:
            print(f"  - solo silver: {[(j, toks[j]) for j in sorted(only_sil)]}")

        shown += 1
        if shown >= max_show:
            print(f"\n... fermato a {max_show} frasi")
            break


if __name__ == "__main__":
    with open("eval_set_venti.pkl", "rb") as f:
        eval_set = pickle.load(f)
    with open("y_pred.pkl", "rb") as f:
        preds = pickle.load(f)
    audit(eval_set, preds)