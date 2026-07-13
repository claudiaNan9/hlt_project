"""
Inferenza ANITA multi-GPU per la text simplification (esperimento di gating).

Legge  : gating_data/gating_sample.parquet   (colonna 'input_text')
Scrive : gating_data/gating_with_outputs.parquet   (aggiunge 'anita_output')

Utilizzo:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 run_anita_simplify.py

Logica invariata rispetto allo script 1B:
  - accelerate divide gli input tra i processi (data parallel)
  - ogni processo carica il modello sulla sua GPU
  - i risultati sono raccolti sul processo principale e riordinati per indice globale
Differenze: legge/scrive un DataFrame (allineamento garantito), usa il chat
template di ANITA e i terminators di LLaMA-3 (il modello e' instruction-tuned).
"""

import pathlib

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID       = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
BS             = 8
MAX_INPUT_LEN  = 512
MAX_NEW_TOKENS = 128
DATA_DIR       = pathlib.Path("/code/HLTproject_code/task_2_simplification_generation_gating/gating_data")
INPUT_PARQUET  = DATA_DIR / "gating_sample.parquet"
OUTPUT_PARQUET = DATA_DIR / "gating_with_outputs.parquet"

SYS_MSG  = ("Sei un assistente esperto di lingua italiana. Il tuo compito e' "
            "riscrivere frasi in una forma piu' semplice e chiara, mantenendo "
            "invariato il significato.")
USER_MSG = ("Riscrivi la seguente frase in modo piu' semplice e facile da leggere, "
            "conservando il significato originale. Rispondi SOLO con la frase "
            "semplificata, senza spiegazioni o commenti.\n\nFrase: {text}")

# ── Setup accelerate ──────────────────────────────────────────────────────────
accelerator = Accelerator()
rank    = accelerator.process_index
is_main = accelerator.is_main_process

if is_main:
    print(f"Processi: {accelerator.num_processes}")

# ── Modello + tokenizer ───────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.to(accelerator.device).eval()

# terminators di LLaMA-3 (ferma la generazione al fine-turn)
eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
terminators = [tokenizer.eos_token_id] + ([eot] if eot is not None and eot >= 0 else [])

def build_prompt(text):
    messages = [
        {"role": "system", "content": SYS_MSG},
        {"role": "user",   "content": USER_MSG.format(text=text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ── Dati ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PARQUET)
input_texts = df["input_text"].tolist()
indexed = list(enumerate(input_texts))   # (indice_globale, testo)

if is_main:
    print(f"Input totali: {len(input_texts)}")

# ── Inferenza ─────────────────────────────────────────────────────────────────
results = []   # (indice_globale, output) per questo processo

with accelerator.split_between_processes(indexed) as shard:
    for k in range(0, len(shard), BS):
        chunk = shard[k : k + BS]
        idxs  = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]

        prompts = [build_prompt(t) for t in texts]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_INPUT_LEN,
                        add_special_tokens=False).to(accelerator.device)

        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 eos_token_id=terminators,
                                 pad_token_id=tokenizer.pad_token_id)

        for j, gi in enumerate(idxs):
            gen = tokenizer.decode(out[j, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            results.append((gi, gen.strip()))

        del enc, out

        if is_main and k % (BS * 20) == 0:
            print(f"[proc {rank}] {k + len(chunk)}/{len(shard)} input...")

# ── Raccolta + salvataggio ────────────────────────────────────────────────────
accelerator.wait_for_everyone()
all_results = gather_object(results)

if is_main:
    all_results.sort(key=lambda x: x[0])
    outputs = [text for _, text in all_results]
    assert len(outputs) == len(df), f"Mismatch: {len(outputs)} output vs {len(df)} input"

    df["anita_output"] = outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET)

    print(f"\nSalvati {len(outputs)} output in {OUTPUT_PARQUET}")
    print("\nEsempio:")
    print(f"  Input : {input_texts[0][:100]}")
    print(f"  Output: {outputs[0][:100]}")

import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()