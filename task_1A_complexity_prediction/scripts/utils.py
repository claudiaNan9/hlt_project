import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def vista_binaria(scores_pred, is_original, thr=None):
    """
    Valuta le predizioni di regressione come classificazione binaria
    (complesso=1 / semplice=0).

    Se thr è None la soglia ottimale viene cercata sui dati passati
    (usare sul validation set). Se thr è fornita viene applicata
    direttamente (usare sul test set con la soglia trovata su val).
    """
    y   = is_original.astype(int)
    auc = roc_auc_score(y, scores_pred)
    if thr is None:
        cand = np.quantile(scores_pred, np.linspace(0.1, 0.9, 81))
        thr  = max(cand, key=lambda t: f1_score(y, (scores_pred > t).astype(int)))
    pred = (scores_pred > thr).astype(int)
    return {
        "auc":       auc,
        "f1":        f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall":    recall_score(y, pred),
        "acc":       accuracy_score(y, pred),
        "thr":       thr,
    }