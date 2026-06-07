# Project Summary

## 1) Architectural Summary

This repository is organized as a small, learning-first machine learning pipeline for LoRA-based adaptation of GPT-2 to tuberculosis (TB) clinical Q&A generation (Phase 1B). It reuses the manual `LoRALinear` implementation from Phase 1A and applies it to a new task type: open-ended causal language modelling over PubMed Q&A pairs.

- Entry points:
  - `run_training.py` runs one end-to-end training experiment (single rank).
  - `src/ablation.py` runs multi-rank experiments (`r = 1, 4, 8, 16, 32`) and writes aggregate results to `outputs/results.csv`.
  - `scripts/build_tb_qa_dataset.py` pulls and filters `bigbio/pubmed_qa` from HuggingFace, writes `data/tb_qa.json`.
- Core architecture (`src/`):
  - `src/lora.py`: LoRA implementation (`LoRALinear`), GPT-2 attention patching (`apply_lora_to_gpt2`), trainable-parameter reporting.
  - `src/data.py`: HuggingFace dataset loading with TB keyword filtering, local JSON caching, reproducible split strategy, tokenization, and `DataLoader` creation.
  - `src/train.py`: optimizer/scheduler setup, epoch training loop (loss only — no accuracy), validation pass, W&B logging, and LoRA-adapter checkpoint saving.
  - `src/evaluate.py`: `compute_perplexity` (in-domain), `compute_ood_perplexity` (wikitext-2 forgetting check), and `run_prompt_battery` (greedy + sampled generation per prompt).
  - `src/config.py`: centralized configuration for data paths, hyperparameters, run modes (full vs smoke), and ablation defaults.
  - `prompts.py`: fixed TB clinical prompt battery used before and after training for qualitative comparison.
- Data and artifacts:
  - Input dataset: `data/tb_qa.json` (TB-filtered PubMed Q&A, cached after first HuggingFace pull).
  - Outputs: LoRA adapter checkpoints in `outputs/checkpoints/`, rank-ablation metrics (including OOD PPL) in `outputs/results.csv`, W&B run metadata in `wandb/`, and qualitative comparison in `docs/base-vs-finetuned.md`.
- Documentation layer:
  - Product and phase planning in `docs/archive/PRD.md`.
  - Operational runbook in `docs/runbook.md`.
  - Dataset gathering walkthrough in `docs/reference/dataset_gathering.md`.

End-to-end flow: TB-filtered PubMed Q&A data is split/tokenized as causal-LM sequences (`labels = input_ids`) → `GPT2LMHeadModel` is loaded and fully frozen → LoRA adapters are injected into `c_attn` attention projections → model is trained with only LoRA parameters updated → in-domain and out-of-domain perplexity are measured → prompt battery is run before and after training for qualitative comparison → checkpoints and experiment results are persisted.

## 2) Technical Decisions Made During The Project

- Chosen paradigm: manual LoRA implementation (not PEFT) for educational transparency and deeper understanding of adapter mechanics.
- Base model: `gpt2` with `GPT2LMHeadModel` for causal language modelling over TB clinical Q&A sequences.
- Task framing: causal LM over `"Question: ...\nAnswer: ..."` strings; loss is cross-entropy over the full sequence (no question-token masking — that is a Phase 2 improvement).
- Parameter strategy: freeze **all** base GPT-2 weights before LoRA injection, then train only LoRA matrices (`lora_A`, `lora_B`) for parameter-efficient adaptation (verified via `print_trainable_params` → ~0.3–0.5% for `r=8`).
- Injection point: apply LoRA to GPT-2 attention `c_attn` projections in each transformer block.
- GPT-2 compatibility choice: reuse EOS token as pad token (`pad_token = eos_token`) to support batch padding.
- Data handling choice: HuggingFace `bigbio/pubmed_qa` with TB keyword filtering; first-run result cached to `data/tb_qa.json`; deterministic shuffling and strict holdout test split.
- Experiment design: include both a single-run workflow and a rank ablation workflow (`r = 1, 4, 8, 16, 32`) to compare adaptation capacity vs parameter cost.
- Default hyperparameters: rank/alpha coupling (`alpha = rank`), `max_length=512` (longer than Phase 1A to fit clinical answers), `batch_size=4`, AdamW optimizer, linear warmup scheduler.
- Evaluation strategy — three layers:
  1. **In-domain perplexity** (TB val set): primary training signal; target ≥15% reduction from baseline.
  2. **Out-of-domain perplexity** (wikitext-2 test set, 50 samples): catastrophic forgetting check. Fine-tuning on a narrow domain can degrade general language ability; a roughly stable OOD PPL confirms the base model's general knowledge is preserved. Computed before and after training and logged to W&B alongside in-domain PPL.
  3. **Prompt battery** (greedy + sampled generation): qualitative check that fine-tuned outputs contain TB-specific terminology (drug names, regimen durations) rather than generic filler.
- Checkpointing policy: save LoRA adapter weights only (A/B matrices), not full model weights, to keep artifacts lightweight and reusable.
- Observability decision: standardize experiment logging in Weights & Biases (`lora-phase-1b` project); ablation CSV includes both `val_ppl` and `ood_ppl` columns.
- Development ergonomics: central config file with full-training and smoke-test modes; smoke scripts in `scripts/smoke_checks/` verify data → LoRA → PPL → mini-train before committing to a full cloud run.
- Tooling choice: `uv` + virtual environment workflow for consistent Python dependency and command execution.

## 3) Concise Project Summary

This project (Phase 1B) is a learning-focused implementation that applies the same manually implemented LoRA adapters from Phase 1A to a new problem: fine-tuning GPT-2 as a causal language model on TB clinical Q&A pairs from PubMed. The codebase provides a complete and reproducible workflow from HuggingFace dataset filtering and caching, through LoRA injection and causal-LM training, to three-layer evaluation (in-domain perplexity, out-of-domain forgetting check, and a qualitative TB prompt battery), rank ablation, and W&B experiment tracking.