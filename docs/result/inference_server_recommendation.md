# Inference Server Recommendation (Base GPT-2 vs GPT-2 + Best LoRA Adapter)

## Goal

Run a fair, reproducible before-vs-after comparison for scientific reporting:
- Model A: base GPT-2
- Model B: GPT-2 with best-performing LoRA adapter (selected from ablation)

For this project, deploy on Vast.ai (recommended) so server hardware is consistent with training.

## 1) Pick the "best" adapter first

Use `outputs/results.csv` and select the rank checkpoint that satisfies both:
- Lowest in-domain validation perplexity (`val_ppl`)
- No major out-of-domain degradation (`ood_ppl` roughly stable)

Record exact artifact path, for example:
- `outputs/checkpoints/rank_8_epoch3.pt`

## 2) Minimal server design

Use one FastAPI service with two model handles loaded at startup:
- `base`: GPT-2 frozen base model
- `lora_best`: GPT-2 + injected LoRA modules + loaded best adapter state dict

Recommended endpoints:
- `GET /health` -> readiness
- `POST /generate` -> single prompt generation
- `POST /batch_generate` -> list of prompts for systematic evaluation
- `POST /score_perplexity` -> perplexity over provided text list (for controlled test sets)

Request payload fields (fix these across models for fair comparison):
- `prompt`
- `max_new_tokens`
- `do_sample`
- `temperature`
- `top_p`
- `seed`
- `model_id` (`base` or `lora_best`)

## 3) Reproducibility controls

To support publication-quality comparison, keep all controls identical:
- Same prompt set and ordering
- Same decoding config per experiment arm
- Fixed random seed for sampled decoding
- Same batch size and device type
- Same preprocessing/tokenization
- Log metadata for every run:
  - git commit hash
  - checkpoint path
  - model_id
  - decoding parameters
  - timestamp

## 4) Suggested test battery (before/after)

Run three groups and store outputs in CSV/JSONL:

1. In-domain TB prompts
- Use your fixed prompt battery from `prompts.py`
- Collect both greedy and sampled outputs

2. Out-of-domain control prompts
- Add 20-50 general prompts (non-medical)
- Confirm no severe regression in fluency/basic knowledge

3. Perplexity datasets
- In-domain: TB validation split used in training workflow
- Out-of-domain: wikitext-2 sample (same subset for both models)

## 5) What to report in the paper

Report both automatic and human-readable comparisons:
- `val_ppl` before vs after
- `ood_ppl` before vs after
- Relative change percentages
- Side-by-side generations for key TB prompts
- Error analysis categories (hallucination, vagueness, missing key regimen terms)

A compact primary claim structure:
- "LoRA improves TB-domain perplexity by X% while preserving out-of-domain perplexity within Y%."

## 6) Practical deployment notes (Vast.ai)

- Keep one GPU pod running both models in one process to reduce infra variance.
- Warm both models once at startup to avoid first-request latency bias.
- If VRAM is tight, run base and LoRA models sequentially in the same service process and queue requests.

## 7) Optional extension for stronger evidence

If you want stronger scientific rigor:
- Run each sampled test with 3-5 seeds
- Report mean and standard deviation
- Compute confidence intervals for perplexity deltas via bootstrap resampling

This is usually enough to move from "demo evidence" to "paper-ready empirical evidence" for a small-scale learning project.
