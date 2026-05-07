# SOP: LoRA Phase 1B (Beginner, Chronological, Cost-Aware)

This document is a step-by-step operating procedure for completing Phase 1B from start to finish with minimal wasted GPU cost.

Primary goal: finish one full causal-LM training run and one rank ablation report on the TB Q&A dataset.
Secondary goal: avoid paying for Vast.ai while fixing preventable issues.

---

## How to use this SOP

1. Follow steps in exact order.
2. Do not start a paid GPU until Step 4 is complete.
3. Each step has a "Done checklist". Only move forward when all items pass.

---
---

## Steps

### Step 0 — Preflight mindset (before touching code)

#### Do
- Decide your hard spending cap for this phase (example: USD 5).
- Decide your stop rule (example: if training crashes twice on Vast.ai, stop and debug locally).
- Understand that Phase 1B is a **causal language model** task. There is no accuracy metric — success is measured by perplexity drop and prompt quality. If you are expecting accuracy scores, re-read `docs/PRD.md` Section 3.

#### Done checklist
- [ ] I have a budget cap written down.
- [ ] I have a stop rule written down.
- [ ] I understand success = working pipeline + ≥15% PPL drop + meaningful TB completions, not perfect metrics.

---

### Step 1 — Prepare local repo and environment (free)

Do all of this locally first.

#### Do
1. Open terminal in `lora-project-1b/` root.
2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -U pip uv
uv pip install -r requirements.txt
```

4. Sanity-check imports:

```bash
uv run python -c "import torch, transformers, wandb, datasets; print('imports OK')"
```

5. Confirm the entry-point file is importable without crash:

```bash
uv run python -c "import run_training; print('run_training import OK')"
```

#### Done checklist
- [ ] `.venv` is activated.
- [ ] `uv pip install -r requirements.txt` completes without errors.
- [ ] Import check prints `imports OK`.
- [ ] Entry script import prints `run_training import OK`.

---

### Step 2 — Build the TB Q&A dataset (free)

This step replaces the synthetic data generation from Phase 1A. The dataset is pulled from HuggingFace and filtered locally.

#### Do
1. Create the data directory if missing:

```bash
mkdir -p data
```

2. Run the dataset builder:

```bash
uv run python scripts/build_tb_qa_dataset.py
```

Expected log output (counts may vary slightly):

```
Loading pubmed_qa_labeled_fold0_source...
  → 67 TB-relevant records from primary config
Below threshold (300); pulling fallback pubmed_qa_artificial_source...
  → 1,432 TB-relevant records after fallback merge
  → 1,408 after dedup
Wrote data/tb_qa.json (1,408 records)
```

If the HuggingFace download is slow, it is normal — the artificial config is ~211,000 records. It will cache locally after the first run.

3. Validate the output file:

```bash
uv run python -c "
import json
with open('data/tb_qa.json') as f:
    data = json.load(f)
assert len(data) >= 300, f'Too few records: {len(data)}'
assert all('formatted' in r for r in data), 'missing formatted field'
assert all('Question:' in r['formatted'] and 'Answer:' in r['formatted'] for r in data), 'bad format'
lengths = [len(r['formatted']) for r in data]
print(f'OK — {len(data)} records')
print(f'  char length: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}')
print('Sample:')
print(data[0]['formatted'][:300])
"
```

#### Done checklist
- [ ] `data/tb_qa.json` exists and is non-empty.
- [ ] Validation prints `OK` with record count ≥ 300.
- [ ] Sample record shows a realistic TB question and answer.
- [ ] No records are missing the `formatted` field.

---

### Step 3 — Local smoke tests before renting GPU (free)

This is the most important cost-saving step. Run all four checks here and fix any failures before spending money.

#### 3a — LoRA parameter sanity

```bash
uv run python -c "
from transformers import GPT2LMHeadModel
from src.lora import apply_lora_to_gpt2, print_trainable_params

model = GPT2LMHeadModel.from_pretrained('gpt2')

# GPT-2 base weights start as trainable — freeze them
for p in model.parameters():
    p.requires_grad = False

model = apply_lora_to_gpt2(model, r=8, alpha=8)
print_trainable_params(model)

for name, p in model.named_parameters():
    if 'lora_B' in name:
        assert p.sum() == 0, f'lora_B not zero at init: {name}'
    if 'lora_A' in name or 'lora_B' in name:
        assert p.requires_grad, f'LoRA param not trainable: {name}'
    elif 'lora_A' not in name and 'lora_B' not in name:
        assert not p.requires_grad, f'Base param still trainable: {name}'

print('LoRA sanity passed')
"
```

Expected: trainable % is roughly 0.3–0.5% for `r=8`. Any value in 0.1–2% is acceptable.

#### 3b — DataLoader sanity

```bash
uv run python -c "
from src.data import get_dataloaders
train_l, val_l, test_l = get_dataloaders(batch_size=2)
batch = next(iter(train_l))
print('input_ids shape:', tuple(batch['input_ids'].shape))
print('labels shape:   ', tuple(batch['labels'].shape))
assert batch['input_ids'].shape == batch['labels'].shape, 'input_ids and labels must match for causal LM'
assert tuple(batch['input_ids'].shape)[1] == 512, 'expected max_length=512'
print('DataLoader sanity passed')
"
```

Expected: shape `[2, 512]` for both `input_ids` and `labels` (they must be identical for causal LM).

#### 3c — Baseline perplexity check

Compute baseline PPL before any training. This number must be finite and greater than 1.

```bash
uv run python -c "
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.data import get_dataloaders
from src.evaluate import compute_perplexity

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('device:', device)

train_l, val_l, test_l = get_dataloaders(batch_size=2)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Decode a few val texts and compute PPL
texts = []
for batch in val_l:
    for ids in batch['input_ids']:
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    if len(texts) >= 20:
        break

ppl = compute_perplexity(model, tokenizer, texts[:20], device)
print(f'Baseline val PPL: {ppl:.2f}')
assert 1 < ppl < 1e6, f'PPL out of expected range: {ppl}'
print('Baseline PPL check passed')
"
```

Expected: a finite number, likely between 50 and 500 for an untuned GPT-2 on TB clinical text.

#### 3d — Smoke training run (1 epoch, tiny subset)

```bash
uv run python -c "
import torch
from transformers import GPT2LMHeadModel
from src.lora import apply_lora_to_gpt2
from src.train import train
from src.data import get_dataloaders

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('device:', device)

train_l, val_l, _ = get_dataloaders(batch_size=2)

# Use only first 5 batches for the smoke test
class TinyLoader:
    def __init__(self, loader, n): self.loader = loader; self.n = n
    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n: break
            yield b
    def __len__(self): return self.n

tiny_train = TinyLoader(train_l, 5)
tiny_val   = TinyLoader(val_l, 2)

model = GPT2LMHeadModel.from_pretrained('gpt2')
for p in model.parameters():
    p.requires_grad = False
model = apply_lora_to_gpt2(model, r=4, alpha=4)

train(model=model, train_loader=tiny_train, val_loader=tiny_val,
      epochs=1, lr=2e-4, device=device)

print('Smoke train passed')
"
```

Expected: loss is printed for 1 epoch without errors. The val loss does not need to decrease on 5 batches — just confirm no crash.

#### Done checklist
- [ ] LoRA sanity: trainable % is 0.1–2%; all checks print passed.
- [ ] DataLoader: `input_ids` shape is `[batch, 512]`; `labels == input_ids`.
- [ ] Baseline PPL: a finite number in (1, 1,000,000).
- [ ] Smoke train: 1 epoch completes without runtime error.

If any checklist item fails: stop here and debug locally. Do not rent GPU yet.

---

### Step 4 — Start Vast.ai only after local checks are green

#### Do
1. Create Vast.ai instance:
   - Image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`
   - GPU: RTX 3060 12 GB or equivalent (12 GB VRAM handles `batch_size=4`, `max_length=512`)
   - Persistent storage enabled
   - Port 22 open

2. Connect by SSH. Quick verification after SSH:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

3. Clone repo on instance:

```bash
git clone <your-repo-url> lora-project-1b
cd lora-project-1b
```

4. Create and activate venv on instance:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip install -r requirements.txt
```

5. Build the dataset on the instance (or upload `data/tb_qa.json` directly via `scp` if you already built it locally — faster):

```bash
# Option A — rebuild on instance
uv run python scripts/build_tb_qa_dataset.py

# Option B — upload from local machine (run this locally, not on instance)
# scp data/tb_qa.json user@instance_ip:/workspace/lora-project-1b/data/tb_qa.json
```

#### Optional — Use `tmux` for disconnect-safe long runs

```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s lora_phase1b
```

Inside `tmux`:

```bash
source .venv/bin/activate
uv run python run_training.py
```

Detach: `Ctrl+b`, then `d`. Reattach: `tmux attach -t lora_phase1b`.

#### Done checklist
- [ ] SSH connection works.
- [ ] `nvidia-smi` shows GPU and driver.
- [ ] PyTorch CUDA is available.
- [ ] Repo is present on instance.
- [ ] Dataset file `data/tb_qa.json` exists on instance.
- [ ] `.venv` is active and `requirements.txt` installed.
- [ ] (Optional) `tmux` session created before long runs.

---

### Step 5 — Connect Weights & Biases before training

#### Do
1. Authenticate W&B:

```bash
source .venv/bin/activate
wandb login
```

2. Connectivity check:

```bash
uv run python -c "
import wandb
run = wandb.init(project='lora-tb-phase1b', name='connectivity_test')
run.log({'ping': 1})
run.finish()
print('wandb OK')
"
```

#### Done checklist
- [ ] `wandb login` succeeds.
- [ ] Connectivity test prints `wandb OK`.
- [ ] Test run is visible in W&B project `lora-tb-phase1b`.

---

### Step 6 — Run one full training job (paid GPU)

#### Do
1. Verify training config in `run_training.py`:
   - `RANK = 8`
   - `ALPHA = 8`
   - `EPOCHS = 3`
   - `LR = 2e-4`
   - `BATCH_SIZE = 4`

2. Start training:

```bash
source .venv/bin/activate
uv run python run_training.py
```

3. During the run, watch logs and W&B:
   - `train/loss` and `val/loss` should decrease over epochs.
   - No NaN loss. If loss hits NaN on the first step, lower `LR` to `1e-4`.
   - Baseline PPL will be logged to W&B before training starts (`baseline/val_ppl`).

4. After the run, confirm:
   - Checkpoints exist under `outputs/checkpoints/rank_8_epoch*.pt`.
   - Final PPL is logged (`final/val_ppl`).
   - PPL reduction is printed in the terminal log.

#### Done checklist
- [ ] Training completes all 3 epochs without crash.
- [ ] `train/loss` and `val/loss` are visible in W&B and trending down.
- [ ] At least one checkpoint file exists.
- [ ] Final val PPL is lower than baseline val PPL.
- [ ] (Target) PPL reduction ≥ 15%. If not reached, note it and proceed anyway — ablation may reveal better ranks.

Stop condition: if run crashes twice with the same error, stop the instance and debug locally.

---

### Step 7 — Run rank ablation once single run is healthy (paid GPU)

Only run this after Step 6 completes cleanly.

#### Do
1. Start ablation (runs ranks `1, 4, 8, 16, 32` in sequence):

```bash
source .venv/bin/activate
uv run python -c "from src.ablation import run_ablation; run_ablation(device='cuda')"
```

2. Each rank trains for 3 epochs, then logs val PPL. Expect roughly 10–20 minutes per rank on an RTX 3060 at `max_length=512`, `batch_size=4`.

3. Let all ranks finish. Confirm `outputs/results.csv` exists after completion.

#### Done checklist
- [ ] Ablation completes for all 5 ranks without crash.
- [ ] `outputs/results.csv` has 5 data rows (plus header).
- [ ] CSV columns are `rank, trainable_params, val_ppl`.
- [ ] W&B has 5 runs named `ablation_r1`, `ablation_r4`, `ablation_r8`, `ablation_r16`, `ablation_r32`.

---

### Step 8 — Read and interpret the results

#### Do
1. Open `outputs/results.csv` and check columns:
   - `rank`
   - `trainable_params`
   - `val_ppl`

2. Interpretation rules for causal LM:
   - Lower PPL is better. Baseline (no training) is likely 100–400 on TB clinical text.
   - `r=1` should have the worst PPL (least capacity to learn domain vocabulary).
   - PPL should decrease as rank increases up to a point, then flatten or slightly worsen (overfitting risk at very high ranks with a small dataset).
   - If PPL barely changes across all ranks, the dataset may be too small or the filter too aggressive.

3. In W&B, review:
   - Single-run `train/loss` and `val/loss` curves.
   - Bar chart comparing `val_ppl` across ablation runs.
   - `baseline/val_ppl` vs `final/val_ppl` on the single run.

4. Run the prompt battery on the fine-tuned model and compare against the base GPT-2 response. Save the side-by-side outputs to `docs/base-vs-finetuned.md`.

5. Write a short conclusion (3–5 bullet points):
   - Which rank gave the best PPL?
   - Did diminishing returns appear?
   - How did qualitative completions change?
   - Was the ≥15% PPL drop criterion met?

#### Done checklist
- [ ] CSV format is correct and has 5 rows.
- [ ] I can identify the rank with the lowest val PPL.
- [ ] I can explain whether increasing rank helped.
- [ ] Prompt battery output is saved to `docs/base-vs-finetuned.md`.
- [ ] I have a short written summary of findings.

---

### Step 9 — End session safely and avoid extra cost

#### Do
1. Download or commit all artifacts before destroying the instance:
   - `outputs/results.csv`
   - `outputs/checkpoints/` (at least the best-rank checkpoint)
   - Any W&B screenshots or exported charts

2. Push commits to GitHub:

```bash
git add outputs/results.csv docs/base-vs-finetuned.md
git commit -m "phase1b: ablation results and base-vs-finetuned"
git push
```

3. Stop or destroy Vast.ai instance after backup is confirmed.

#### Done checklist
- [ ] Results CSV and key checkpoints are backed up locally or pushed to git.
- [ ] No critical artifact exists only on the remote instance.
- [ ] Vast.ai instance is stopped when not in use.
- [ ] Instance is destroyed after final backup to eliminate idle charges.

---
---

## Cost Minimization Playbook (Quick Reference)

- Always finish Steps 1–3 locally before starting a paid GPU.
- Build `data/tb_qa.json` locally and `scp` it to the instance — saves HuggingFace download time and cost.
- Run one full training job first, then ablation. Do not ablate a broken pipeline.
- Keep a hard budget cap and stop rule.
- If an error repeats, stop the instance immediately and debug offline.
- End every session by stopping the instance. Destroy after final artifact backup.

---

## Definition of "Phase 1B Complete"

- [ ] TB-filtered dataset built (`data/tb_qa.json`, ≥ 300 records).
- [ ] LoRA sanity checks pass (trainable params 0.1–2%, A and B have gradients, W frozen).
- [ ] One full training run completes with sensible loss curves.
- [ ] Val PPL drops ≥ 15% from baseline on the TB val set.
- [ ] Out-of-domain PPL roughly stable (no catastrophic forgetting).
- [ ] Ablation CSV produced for ranks `1, 4, 8, 16, 32`.
- [ ] Side-by-side base vs fine-tuned outputs documented in `docs/base-vs-finetuned.md`.
- [ ] Short written summary comparing Phase 1A and Phase 1B findings.
