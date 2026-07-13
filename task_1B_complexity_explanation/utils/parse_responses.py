import json, re, pickle

def extract_annotation(raw, n_tokens=None):
    """
    Estrae indici complessi e motivi da una risposta grezza di ANITA,
    in modo tollerante agli errori di formato (apici singoli, graffe extra,
    testo-delirio dentro le liste).

    Strategia a cascata:
      1) prova un parse JSON pulito (dopo riparazioni leggere);
      2) se fallisce, estrae gli indici e i motivi con regex.

    Ritorna (dict | None, status):
      dict = {"complex": [int, ...], "motivi": {"idx": "motivo", ...}}
    """
    if not raw or not raw.strip():
        return None, "empty"

    # isola dal primo '{' all'ultimo '}'
    start, end = raw.find("{"), raw.rfind("}")
    blob = raw[start:end + 1] if (start != -1 and end > start) else raw

    # ── tentativo 1: JSON pulito con riparazioni leggere ─────────────────────
    parsed = _try_clean_json(blob)
    if parsed is not None:
        result = _validate(parsed, n_tokens)
        if result is not None:
            return result, "ok"

    # ── tentativo 2: estrazione regex (tollerante a tutto) ───────────────────
    complex_idx = _extract_complex_indices(blob, n_tokens)
    motivi      = _extract_motivi(blob, n_tokens)

    if complex_idx or motivi:
        # tieni in complex anche gli indici che compaiono solo nei motivi
        all_idx = sorted(set(complex_idx) | {int(k) for k in motivi})
        if n_tokens is not None:
            all_idx = [i for i in all_idx if 0 <= i < n_tokens]
        return {"complex": all_idx, "motivi": motivi}, "recovered_regex"

    return None, "unparsable"


# ─────────────────────────────────────────────────────────────────────────────
def _try_clean_json(blob):
    for candidate in _json_variants(blob):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None

def _json_variants(blob):
    """Genera versioni progressivamente più 'riparate' del blob."""
    yield blob
    # apici singoli -> doppi solo quando delimitano valori/chiavi
    v = re.sub(r"(?<=[:\{\[,\s])'([^']*?)'(?=[\}\]:,\s])", r'"\1"', blob)
    yield v
    # rimuovi graffe/apici/virgolette spurie in coda
    yield re.sub(r'[}"\'\s]+$', "}", blob)
    yield re.sub(r'[}"\'\s]+$', "}", v)
    # tronca all'ultima graffa bilanciata
    yield _balance(blob)
    yield _balance(v)

def _balance(s):
    depth, last = 0, None
    for i, ch in enumerate(s):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: last = i
    return s[:last + 1] if last is not None else s


# ─────────────────────────────────────────────────────────────────────────────
def _extract_complex_indices(blob, n_tokens):
    """Pesca gli interi dentro la lista 'complex': [...], ignorando testo."""
    m = re.search(r'"?complex"?\s*:\s*\[(.*?)\]', blob, re.DOTALL)
    region = m.group(1) if m else blob
    idx = [int(x) for x in re.findall(r'\b\d+\b', region)]
    if n_tokens is not None:
        idx = [i for i in idx if 0 <= i < n_tokens]
    return sorted(set(idx))

def _extract_motivi(blob, n_tokens):
    """
    Pesca coppie "indice": "motivo" dalla sezione motivi, tollerando
    apici singoli o doppi. Tiene l'ULTIMA occorrenza per indice duplicato.
    """
    m = re.search(r'"?motivi"?\s*:\s*\{(.*)\}', blob, re.DOTALL)
    region = m.group(1) if m else ""
    out = {}
    # "12": "testo"  oppure  '12': 'testo'  (virgolette miste)
    for k, q, v in re.findall(r'["\'](\d+)["\']\s*:\s*(["\'])(.*?)\2', region, re.DOTALL):
        if n_tokens is None or (0 <= int(k) < n_tokens):
            out[k] = v.strip()
    return out


# ─────────────────────────────────────────────────────────────────────────────
def _validate(parsed, n_tokens):
    if not isinstance(parsed, dict):
        return None
    cx = parsed.get("complex", [])
    mt = parsed.get("motivi", {})
    if not isinstance(cx, list):
        return None
    clean = []
    for x in cx:
        if isinstance(x, int): clean.append(x)
        elif isinstance(x, str) and x.strip().lstrip("-").isdigit():
            clean.append(int(x.strip()))
        else:
            return None   # spazzatura nella lista -> forza il fallback regex
    if n_tokens is not None:
        clean = [i for i in clean if 0 <= i < n_tokens]
    mt = mt if isinstance(mt, dict) else {}
    mt = {str(k): str(v) for k, v in mt.items()}
    return {"complex": sorted(set(clean)), "motivi": mt}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("/code/HLTproject_code/task_1B_complexity_explanation/data/eval_set_full.pkl", "rb") as f:
        eval_set = pickle.load(f)
    with open("/code/HLTproject_code/task_1B_complexity_explanation/anita_results/anita_resp_full.pkl", "rb") as f:
        resp = pickle.load(f)

    parsed_all, stats = [], {}
    for ex, raw in zip(eval_set, resp):
        n = len(ex["tokens"])
        parsed, status = extract_annotation(raw, n_tokens=n)
        parsed_all.append(parsed)
        stats[status] = stats.get(status, 0) + 1

    with open("parsed_anita_full.pkl", "wb") as f:
        pickle.dump(parsed_all, f)

    print("Esito parsing:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    n_ok = sum(1 for p in parsed_all if p is not None)
    print(f"\nRecuperati {n_ok}/{len(resp)} record")

    # mostra eventuali falliti per ispezione
    print("\nRecord ancora falliti:")
    for i, (p, raw) in enumerate(zip(parsed_all, resp)):
        if p is None:
            print(f"  [{i}] {raw[:90]!r}")