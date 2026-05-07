# LoRA Phase 1B (Learning Project)

Manual LoRA fine-tune of GPT-2 on tuberculosis clinical Q&A pairs from PubMed. Reuses the `LoRALinear` implementation from Phase 1A; the only changes are the dataset, evaluation (perplexity instead of accuracy), and prompt battery.

## Current Scope (Phase 1B)
- Reuse manual `LoRALinear` from Phase 1A (no PEFT)
- Apply LoRA to GPT-2 (`c_attn` layers) used as a **causal LM** (not classifier)
- Train on TB-filtered records from `bigbio/pubmed_qa`
- Evaluate via in-domain perplexity, out-of-domain perplexity, and a TB prompt battery
- Rank ablation: `r = 1, 4, 8, 16, 32`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Build the dataset
See [docs/dataset-gathering.md](docs/dataset-gathering.md) for the full walkthrough.

```bash
uv run python scripts/build_tb_qa_dataset.py
```

This produces `data/tb_qa.json` (500–2,000 TB-relevant Q&A records).

## How to Run
Always activate the environment first:

```bash
source .venv/bin/activate
```

Single training run:
```bash
uv run python run_training.py
```

Rank ablation (`r = 1, 4, 8, 16, 32`):
```bash
uv run python -c "from src.ablation import run_ablation; run_ablation()"
```

## Outputs
- Checkpoints: `outputs/checkpoints/` (not tracked in git)
- Ablation CSV: `outputs/results.csv`
- Base vs fine-tuned comparison: `docs/base-vs-finetuned.md`
- W&B project: `lora-phase1b-galenic`

## Documentation
- [PRD](docs/PRD.md) — full requirements and module spec
- [Dataset gathering walkthrough](docs/dataset-gathering.md) — operational guide for assembling `data/tb_qa.json`

## Reference
- Phase 1A code: `../lora-project-1a/`
- Phase 1B learning narrative: `../../RnD/LoRA/lora-phase1b.md` (in the main repo)
