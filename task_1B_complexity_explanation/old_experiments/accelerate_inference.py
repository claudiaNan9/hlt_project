import os, json, re, pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

MODEL_ID = "swap-uniba/LLaMAntino-2-7b-hf-dolly-ITA"
BS = 16

ALPACA = ("Di seguito è riportata un'istruzione che descrive un'attività, "
          "accompagnata da un input che aggiunge ulteriore informazione. "
          "Scrivi una risposta che completi adeguatamente la richiesta.\n\n"
          "### Istruzione:\n{instruction}\n\n### Input:\n{input}\n\n### Risposta:\n")
ISTRUZIONE = ("Ti viene fornita una frase italiana tokenizzata e numerata. "
              "Individua i token complessi (parola rara o costruzione sintattica difficile) "
              "da rimuovere o sostituire per rendere la frase più semplice, indica l'indice e un motivo breve. "
              'Rispondi solo con un JSON nel formato '
              '{"complex": [indici], "motivi": {"indice": "motivo"}}.')

def _format_indexed(tokens):
    return "\n".join(f"{i}: {t}" for i, t in enumerate(tokens))

accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.to(accelerator.device).eval()

# eval_set salvato a monte (vedi sotto)
with open("eval_set_venti.pkl", "rb") as f:
    eval_set = pickle.load(f)

prompts = [ALPACA.format(instruction=ISTRUZIONE, input=_format_indexed(ex["tokens"]))
           for ex in eval_set]

# ── ogni processo prende una fetta dei prompt ────────────────────────────────
results = []
with accelerator.split_between_processes(list(enumerate(prompts))) as shard:
    for k in range(0, len(shard), BS):
        chunk = shard[k:k+BS]
        idxs  = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(accelerator.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for j, gi in enumerate(idxs):
            gen = tokenizer.decode(out[j, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            results.append((gi, gen))   # tengo l'indice globale per riordinare dopo

# ── raccolgo da tutte le GPU e riordino ──────────────────────────────────────
results = gather_object(results)
if accelerator.is_main_process:
    results.sort(key=lambda x: x[0])              # rimetto nell'ordine originale
    resp = [r[1] for r in results]
    with open("llm_resp.pkl", "wb") as f:
        pickle.dump(resp, f)
    print(f"Salvate {len(resp)} risposte in llm_resp.pkl")