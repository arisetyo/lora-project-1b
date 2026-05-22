# Project Summary

## 1) Architectural Summary

This repository is organized as a small, learning-first machine learning pipeline for LoRA-based adaptation of GPT-2 to Indonesian infrastructure sentiment classification.

- Entry points:
  - `run_training.py` runs one end-to-end training experiment.
  - `src/ablation.py` runs multi-rank experiments (`r = 1, 4, 8, 16, 32`) and writes aggregate results to `outputs/results.csv`.
- Core architecture (`src/`):
  - `src/lora.py`: LoRA implementation (`LoRALinear`), GPT-2 attention patching (`apply_lora_to_gpt2`), trainable-parameter reporting.
  - `src/data.py`: local JSON dataset loading, reproducible split strategy, tokenization, and `DataLoader` creation.
  - `src/train.py`: optimizer/scheduler setup, epoch training loop, validation pass, W&B logging, and LoRA-adapter checkpoint saving.
  - `src/evaluate.py`: validation accuracy computation and qualitative example prediction checks.
  - `src/config.py`: centralized configuration for data paths, hyperparameters, run modes (full vs smoke), and ablation defaults.
- Data and artifacts:
  - Input dataset: `data/infra_sentiment.json` (plus tiny smoke dataset).
  - Raw generation sources: `data_source/`.
  - Outputs: LoRA adapter checkpoints in `outputs/checkpoints/`, rank-ablation metrics in `outputs/results.csv`, and W&B run metadata in `wandb/`.
- Documentation layer:
  - Product and phase planning in `docs/archive/PRD.md` and `docs/plan.md`.
  - Operational and interpretation guidance in `docs/checkpoint-inference-guide.md`, `docs/result_interpretation.md`, and other phase docs.

End-to-end flow: local JSON sentiment data is split/tokenized -> GPT-2 sequence-classification model is loaded and frozen -> LoRA adapters are injected into attention projections -> model is trained with only LoRA parameters updated -> accuracy and sample predictions are evaluated -> checkpoints and experiment results are persisted.

## 2) Technical Decisions Made During The Project

- Chosen paradigm: manual LoRA implementation (not PEFT) for educational transparency and deeper understanding of adapter mechanics.
- Base model: `gpt2` with `GPT2ForSequenceClassification` to align with a 3-class sentiment objective.
- Task framing: sequence classification (`positive`, `neutral`, `negative`) rather than causal-LM perplexity optimization.
- Parameter strategy: freeze base GPT-2 weights and train only LoRA matrices (`lora_A`, `lora_B`) for parameter-efficient adaptation.
- Injection point: apply LoRA to GPT-2 attention `c_attn` projections in each transformer block.
- GPT-2 compatibility choice: reuse EOS token as pad token (`pad_token = eos_token`) to support batch padding.
- Data handling choice: local JSON dataset pipeline with deterministic shuffling/splitting and a strict holdout test subset.
- Experiment design: include both a single-run workflow and a rank ablation workflow to compare adaptation capacity vs cost.
- Default hyperparameters: rank/alpha coupling (`alpha = rank`), AdamW optimizer, linear warmup scheduler, small-batch training tuned for local/cloud constraints.
- Evaluation strategy: quantitative metric (validation accuracy) plus qualitative checks (example predictions before/after training).
- Checkpointing policy: save LoRA adapter weights only (A/B matrices), not full model weights, to keep artifacts lightweight and reusable.
- Observability decision: standardize experiment logging in Weights & Biases for reproducibility and run comparison.
- Development ergonomics: central config file with full-training and smoke-test modes to balance correctness checks and runtime speed.
- Tooling choice: `uv` + virtual environment workflow for consistent Python dependency and command execution.

## 3) Concise Project Summary

This project is a phase-based, small-scale ML learning implementation that demonstrates how to adapt GPT-2 to Indonesian infrastructure sentiment classification using manually implemented LoRA adapters. The codebase provides a complete and reproducible workflow from data loading and LoRA injection to training, evaluation, checkpointing, and rank ablation analysis, with clear documentation and W&B tracking to support both practical experimentation and conceptual understanding.