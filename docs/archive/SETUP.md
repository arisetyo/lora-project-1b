# Module Status — Phase 1B

All modules in this directory are Phase 1B implementations. Do not overwrite them with Phase 1A copies.

| File | Origin | Phase 1B changes |
|---|---|---|
| `lora.py` | Copied unchanged from Phase 1A | None — `LoRALinear` and `apply_lora_to_gpt2` are task-agnostic |
| `train.py` | Based on Phase 1A | Removed classification-era accuracy tracking; training loop is now loss-only (correct for causal LM) |
| `evaluate.py` | Rewritten for Phase 1B | Replaced accuracy/example-prediction with `compute_perplexity`, `compute_ood_perplexity` (wikitext-2 forgetting check), and `run_prompt_battery` (greedy + sampled generation) |
| `data.py` | New in Phase 1B | HuggingFace `bigbio/pubmed_qa` loading, TB keyword filtering, local JSON caching, causal-LM tokenisation |
| `ablation.py` | Based on Phase 1A | Updated for causal LM: base params frozen per run, OOD PPL recorded alongside val PPL, CSV gains `ood_ppl` column |
| `config.py` | Based on Phase 1A | Updated model name, W&B project, data path, hyperparameters; dropped classification fields (`NUM_LABELS`, `LABEL_TO_ID`) |
