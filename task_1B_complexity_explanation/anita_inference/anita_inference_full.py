import os, json, pickle, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID       = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
BS             = 8
MAX_INPUT_LEN  = 4096
MAX_NEW_TOKENS = 1024
INPUT_PKL      = "/code/HLTproject_code/task_1B_complexity_explanation/data/eval_set_full.pkl"            # <-- il set completo da 7000
CKPT_DIR       = "ckpt_anita"              # cartella dei checkpoint parziali
FINAL_PKL      = "anita_full_responses_second.pkl"
CKPT_EVERY     = 200                       # salva un checkpoint ogni N record (per processo)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT (identico alla versione validata)
# ─────────────────────────────────────────────────────────────────────────────
SYS_BASE = ("Sei un assistente AI per la lingua italiana. Esegui task di annotazione linguistica con precisione. "
            "Rispondi sempre e solo in italiano e con il formato JSON richiesto.")

ISTRUZIONE = (
    "Ti viene fornita una frase italiana, leggila attentamente per comprenderne il significato. Dopo vedrai una versione della stessa frase in cui ogni token è preceduto dal suo indice numerico (es. '0: La', '1: casa'). "
    "Immagina di dover riscrivere la frase in una forma più semplice e più facile da leggere, conservando il significato essenziale.\n\n"
    "Indica tutti i token che andrebbero rimossi o modificati per semplificare la frase, cioè i token che:\n"
    "- andrebbero rimossi perché non aggiungono informazione o costituiscono parti superflue della frase (incisi, apposizioni).\n"
    "- andrebbero sostituiti perché troppo difficili, formali, tecnici o rari (parole che riscriveresti con un termine più comune).\n\n"
    "- fanno parte di strutture sintattiche complesse che potrebbero essere semplificate (subordinate, catene sintattiche profonde con informazioni ridondanti o trascurabili).\n"
    "Non marcare i token che lasceresti invariati: la punteggiatura, il nucleo essenziale della frase (soggetto (nomi propri o entità nominate), verbo e oggetto principali) e le parole già semplici.\n"
    "Ricorda che la frase risultante deve essere leggibile e grammaticalmente corretta senza variazioni di significato o informazioni distorte.\n\n"
    "Per ogni token che modificheresti fornisci:\n"
    "- l'indice numerico esatto del token, così come appare nella frase;\n"
    "- un motivo brevissimo con l'operazione e la ragione.\n\n"
    "Regole sul formato JSON, da rispettare rigorosamente:\n"
    "- usa SOLO virgolette doppie, mai apici singoli;\n"
    "- la lista 'complex' deve contenere SOLO numeri interi separati da virgola, "
    "senza parentesi, note o commenti tra gli indici;\n"
    "- se un token non va marcato, semplicemente NON includerlo: non aggiungere "
    "note per spiegare perché un indice è stato escluso o corretto;\n"
    "- usa una sola volta la chiave 'complex' e una sola volta la chiave 'motivi';\n"
    "- ogni indice in 'motivi' deve essere presente anche in 'complex';\n"
    "- chiudi il JSON con una sola parentesi graffa finale;\n"
    "- non aggiungere spiegazioni, note o altri oggetti dopo il JSON."
)

SYS = SYS_BASE + "\n\n" + ISTRUZIONE

# ─────────────────────────────────────────────────────────────────────────────
# FEWSHOT_EXAMPLES per il task "editing predittivo" (mode=both).
# Motivi LIBERI ed espliciti: insegnano al modello a spiegare con parole proprie
# PERCHÉ quel token va rimosso o sostituito, non a ripetere una formula fissa.
# Indici verificati a mano contro la lista 'tokens'.
# ─────────────────────────────────────────────────────────────────────────────

FEWSHOT_EXAMPLES = [
    {
        # 0:La 1:quiescenza 2:del 3:vulcano 4:perdurò 5:per 6:molti 7:secoli 8:.
        "tokens": ["La", "quiescenza", "del", "vulcano", "perdurò",
                   "per", "molti", "secoli", "."],
        "output": {
            "complex": [1, 4],
            "motivi": {
                "1": "parola rara e letteraria, si può dire 'inattività'",
                "4": "verbo formale poco usato, si può dire 'durò'",
            },
        },
    },
    {
        # 0:Il 1:professor 2:Rossi 3:, 4:noto 5:cardiologo 6:di 7:fama
        # 8:internazionale 9:, 10:deceduto 11:ieri 12:, 13:aveva 14:ottant'anni 15:.
        "tokens": ["Il", "professor", "Rossi", ",", "noto", "cardiologo",
                   "di", "fama", "internazionale", ",", "deceduto",
                   "ieri", ",", "aveva", "ottant'anni", "."],
        "output": {
            "complex": [4, 5, 6, 7, 8, 10],
            "motivi": {
                "4": "inizio di un inciso non essenziale",
                "5": "dettaglio sulla professione, eliminabile",
                "6": "parte dell'inciso accessorio",
                "7": "parte dell'inciso accessorio",
                "8": "parte dell'inciso accessorio",
                "10": "termine formale, meglio 'morto'",
            },
        },
    },
    {
        # 0:Nonostante 1:avesse 2:conseguito 3:la 4:laurea 5:con 6:un
        # 7:cospicuo 8:ritardo 9:, 10:trovò 11:lavoro 12:senza 13:difficoltà 14:.
        "tokens": ["Nonostante", "avesse", "conseguito", "la", "laurea", "con", "un",
                   "cospicuo", "ritardo", ",", "trovò", "lavoro",
                   "senza", "difficoltà", "."],
        "output": {
            "complex": [2, 7],
            "motivi": {
                "2": "verbo ricercato, si può dire 'preso'",
                "7": "aggettivo difficile, si può dire 'grande'",
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _format_indexed(tokens):
    frase    = " ".join(tokens)
    numerata = "\n".join(f"{i}: {t}" for i, t in enumerate(tokens))
    return f"{frase}\n\nToken numerati:\n{numerata}"

def build_messages(tokens):
    msgs = [{"role": "system", "content": SYS}]
    for ex in FEWSHOT_EXAMPLES:
        msgs.append({"role": "user", "content": "Frase:\n" + _format_indexed(ex["tokens"])})
        msgs.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": "Frase:\n" + _format_indexed(tokens)})
    return msgs

def done_indices(ckpt_dir):
    """Legge tutti i checkpoint già salvati e ritorna gli indici globali completati."""
    done = {}
    for path in glob.glob(os.path.join(ckpt_dir, "ckpt_*.pkl")):
        try:
            with open(path, "rb") as f:
                for gi, gen in pickle.load(f):
                    done[gi] = gen
        except Exception:
            pass   # checkpoint corrotto (es. crash a metà write): lo ignora
    return done

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
accelerator = Accelerator()
print(f"[proc {accelerator.process_index}] device = {accelerator.device}")

if accelerator.is_main_process:
    os.makedirs(CKPT_DIR, exist_ok=True)
accelerator.wait_for_everyone()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.to(accelerator.device).eval()

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# ─────────────────────────────────────────────────────────────────────────────
# DATI + RIPRESA: salta gli indici già completati
# ─────────────────────────────────────────────────────────────────────────────
with open(INPUT_PKL, "rb") as f:
    test_set = pickle.load(f)

already = done_indices(CKPT_DIR)
todo = [(i, ex) for i, ex in enumerate(test_set) if i not in already]

if accelerator.is_main_process:
    print(f"Totale record: {len(test_set)} | già fatti: {len(already)} | da fare: {len(todo)}")

# costruisci i prompt solo per i record mancanti
prompt_items = [(i, tokenizer.apply_chat_template(build_messages(ex["tokens"]),
                                                  tokenize=False, add_generation_prompt=True))
                for i, ex in todo]

# check troncamento sul set completo
if accelerator.is_main_process and prompt_items:
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for _, p in prompt_items]
    over = sum(1 for L in lengths if L > MAX_INPUT_LEN)
    print(f"Prompt piu lungo: {max(lengths)} token (limite {MAX_INPUT_LEN}) | troncati: {over}")
    if over:
        print(f"!! {over} prompt verranno troncati: valuta di alzare MAX_INPUT_LEN")

# ─────────────────────────────────────────────────────────────────────────────
# INFERENZA con checkpoint incrementale
# ─────────────────────────────────────────────────────────────────────────────
rank = accelerator.process_index
buffer, processed = [], 0

def flush(buf, tag):
    """Salva un checkpoint parziale per questo processo."""
    if not buf:
        return
    path = os.path.join(CKPT_DIR, f"ckpt_r{rank}_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(buf, f)

with accelerator.split_between_processes(prompt_items) as shard:
    for k in range(0, len(shard), BS):
        chunk = shard[k:k + BS]
        idxs  = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_INPUT_LEN,
                        add_special_tokens=False).to(accelerator.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 eos_token_id=terminators,
                                 pad_token_id=tokenizer.pad_token_id)
        for j, gi in enumerate(idxs):
            gen = tokenizer.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            buffer.append((gi, gen))

        processed += len(chunk)
        # checkpoint periodico
        if processed % CKPT_EVERY < BS:
            flush(buffer, f"{processed:06d}")
            buffer = []
            print(f"[proc {rank}] checkpoint a {processed} record")

# salva l'ultimo buffer residuo
flush(buffer, "final")

accelerator.wait_for_everyone()

# ─────────────────────────────────────────────────────────────────────────────
# MERGE FINALE: unisce tutti i checkpoint nell'ordine originale
# ─────────────────────────────────────────────────────────────────────────────
if accelerator.is_main_process:
    merged = done_indices(CKPT_DIR)          # rilegge TUTTO (vecchi + nuovi)
    resp = [merged.get(i, "") for i in range(len(test_set))]
    missing = sum(1 for r in resp if r == "")
    with open(FINAL_PKL, "wb") as f:
        pickle.dump(resp, f)
    print(f"\nUnite {len(resp)-missing}/{len(test_set)} risposte in {FINAL_PKL}")
    if missing:
        print(f"!! {missing} record mancanti (rilancia lo stesso comando per completarli)")

import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()