# HLTproject — Text Complexity & Simplification

Progetto di Human Language Technologies su complessità linguistica e semplificazione automatica per l'italiano. Il lavoro è diviso in tre task, tutti basati sullo stesso dataset di coppie (frase originale, semplificazione):

- **Task 1A** — Predizione della complessità (classificazione binaria semplice/complesso + regressione su uno score di leggibilità)
- **Task 1B** — Spiegazione della complessità (quali token/costrutti rendono una frase complessa)
- **Task 2** — Generazione di semplificazioni con un LLM, con esperimento di filtering (un classificatore di complessità filtra quali frasi vale la pena semplificare)

## Dataset

I dati provengono dal dataset [`mpapucci/impacts`](https://huggingface.co/datasets/mpapucci/impacts) (dominio `wikipedia_profiling`), che contiene coppie frase originale/semplificazione con punteggi di leggibilità (feature ProfilingUD). `task_1A_complexity_prediction/scripts/prepare_data.py` scarica il dataset, filtra le coppie dove la semplificazione è effettivamente più leggibile, e produce lo split **60/20/20 (train/val/test)** raggruppato per frase originale (`data/*.parquet`), usato da tutti e tre i task.

`nvdb/` è un repository Git annidato (sottomodulo) con il *Nuovo Vocabolario di Base* della lingua italiana (De Mauro), usato come risorsa lessicale nel Task 1B per distinguere parole "di base" da parole rare/complesse.

## Struttura del repository

```
data/                                  train/val/test.parquet (generati, non versionati)
data_analysis/                         analisi esplorativa del dataset
nvdb/                                  Vocabolario di Base (repo annidato)

task_1A_complexity_prediction/
├── scripts/prepare_data.py            download + split del dataset
├── scripts/utils.py, eval_checkpoint_*.py
├── bert/bertino_cls.py                classificatore BERTino (fine-tuning + LR search)
├── bert/bertino_reg.py                regressore BERTino sullo score di leggibilità
├── shallow models/                    baseline: logistic regression, SVR su feature ProfilingUD
├── results/                           metriche 
└── notebooks/predictions.ipynb        analisi dei risultati

task_1B_complexity_explanation/
├── baselines/explanations.py          silver-label (allineamento originale/semplificazione),
│                                       metriche (P/R/F1, edit distance, TER), baseline random + VdB
├── integrated_gradients/              spiegazione via Integrated Gradients sul classificatore 1A
├── anita_inference/, anita_results/   spiegazioni generate da un LLM (ANITA) come metodo aggiuntivo
├── utils/                             parsing/validazione delle risposte del LLM
└── notebooks/                         analisi e confronto tra metodi

task_2_simplification_generation_gating/
├── scripts/run_anita_simplify.py      generazione di semplificazioni con ANITA 
├── gating_data/                       dati e output dell'esperimento di gating
└── notebooks/gating_experiment.ipynb  esperimento di filtraggio delle semplificazioni col predittore
```

## Modelli usati

- **[indigo-ai/BERTino](https://huggingface.co/indigo-ai/BERTino)** — encoder BERT-like per l'italiano, fine-tuned per classificazione/regressione (Task 1A)
- **[swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA](https://huggingface.co/swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA)** — LLM instruction-tuned in italiano, usato per generare semplificazioni (Task 2) e spiegazioni (Task 1B)

