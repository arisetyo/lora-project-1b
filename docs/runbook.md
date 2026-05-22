# SOP: LoRA Phase 1B (Runbook)

Use this file as the single execution checklist.

## 1. Local setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip install -r requirements.txt
uv run python -c "import torch, transformers, wandb, datasets; print('imports OK')"
```

## 2. Build local dataset

Step 2.1 - Create data directory:

```bash
mkdir -p data
```

Step 2.2 - Build dataset JSON:

```bash
uv run python scripts/build_tb_qa_dataset.py
```

Step 2.3 - Validate required fields and minimum size:

```bash
uv run python scripts/smoke_checks/2a_validate_dataset_json.py
```

Step 2.4 - Quick manual sample check (5 rows):

```bash
uv run python scripts/smoke_checks/2b_spot_check_dataset.py
```

Step 2.5 - Verify training loader reads local JSON:

```bash
uv run python -c "from src.data import get_dataloaders; get_dataloaders(batch_size=2)"
```

Expected: logs include Loaded ... records from data/tb_qa.json.

## 3. Run local smoke checks

Run all checks:

```bash
uv run python scripts/smoke_checks/run_all.py
```

Or run one-by-one:

```bash
uv run python scripts/smoke_checks/3a_lora_sanity.py
uv run python scripts/smoke_checks/3b_dataloader_sanity.py
uv run python scripts/smoke_checks/3c_baseline_ppl.py
uv run python scripts/smoke_checks/3d_smoke_train_tiny.py
```

## 4. Provision Vast.ai pod

Create pod with:
- Image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`
- GPU: RTX 3060 12GB (or equivalent)
- Persistent storage enabled
- SSH enabled

After SSH:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## 5. Prepare project on pod

```bash
git clone <your-repo-url> lora-project-1b
cd lora-project-1b
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip install -r requirements.txt
```

Copy dataset from local machine to pod:

```bash
scp data/tb_qa.json <user>@<pod_ip>:/workspace/lora-project-1b/data/tb_qa.json
```

Alternative on pod (if not copying):

```bash
uv run python scripts/build_tb_qa_dataset.py
```

## 6. Connect W&B on pod

```bash
source .venv/bin/activate
wandb login
uv run python -c "import wandb; r=wandb.init(project='lora-tb-phase1b', name='connectivity_test'); r.log({'ping':1}); r.finish(); print('wandb OK')"
```

## 7. Run full training on pod

```bash
source .venv/bin/activate
uv run python run_training.py
```

## 8. Run ablation on pod

```bash
source .venv/bin/activate
uv run python -c "from src.ablation import run_ablation; run_ablation(device='cuda')"
```

## 9. Collect results

Expected artifacts:
- `outputs/results.csv`
- `outputs/checkpoints/`

Optional commit:

```bash
git add outputs/results.csv docs/base-vs-finetuned.md
git commit -m "phase1b: ablation results and base-vs-finetuned"
git push
```

Stop or destroy pod after backup.
