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

Note: smoke check scripts include a project-root import bootstrap, so `src` imports resolve whether you run `run_all.py` or execute each script directly.

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

Step 4.1 - Connect to pod over SSH:

In Vast.ai, open your instance and copy the SSH command shown in the Connect/SSH panel. It is typically in this shape:

```bash
ssh -p <ssh_port> root@<instance_host>
```

If your local key is not loaded, load it first:

```bash
ssh-add ~/.ssh/id_ed25519
```

Step 4.2 - Pod preflight checks after SSH:

```bash
ss -tlnp | grep sshd
python --version
df -h /workspace
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

What these checks confirm:
- `ss -tlnp | grep sshd`: SSH daemon is listening and active inside the pod.
- `python --version`: expected Python is available before creating/using `.venv`.
- `df -h /workspace`: enough free disk under the workspace mount for dataset, env, checkpoints, and logs.
- `nvidia-smi`: GPU is visible and driver stack is healthy.
- `torch.cuda.is_available()`: PyTorch can actually use CUDA from Python.

Suggested minimum free space in `/workspace`: 10 GB.

Step 4.3 - (Recommended for private repos) create a temporary SSH key on the pod for GitHub access:

Security note: do not copy your personal private SSH key to the pod.

Create an ephemeral key pair on the pod:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "vast-temp-$(date +%Y%m%d)" -f ~/.ssh/id_ed25519_github_vast_temp -N ""
cat ~/.ssh/id_ed25519_github_vast_temp.pub
```

Then add the printed public key to GitHub:
- Option A: GitHub user key (Settings -> SSH and GPG keys -> New SSH key)
- Option B: Repo Deploy Key (Repo -> Settings -> Deploy keys) for single-repo scope

Configure SSH on pod to use this key for GitHub:

```bash
cat >> ~/.ssh/config << 'EOF'
Host github.com
	HostName github.com
	User git
	IdentityFile ~/.ssh/id_ed25519_github_vast_temp
	IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh-keyscan github.com >> ~/.ssh/known_hosts
ssh -T git@github.com
```

Expected: authentication succeeds (GitHub may print a non-shell-access message after successful auth).

## 5. Prepare project on pod

```bash
git clone <your-repo-url> lora-project-1b
cd lora-project-1b
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip install -r requirements.txt
```

Step 5.1 - Recommended: copy the already-validated local dataset to pod:

```bash
scp data/tb_qa.json <user>@<pod_ip>:/workspace/lora-project-1b/data/tb_qa.json
```

Why copy instead of rebuilding on pod:
- Faster startup on rented GPU time (no extra dataset fetch/filter run).
- Better reproducibility (you train on the same validated JSON from local smoke checks).
- Avoids surprises from transient dataset/network issues at training time.

Step 5.2 - Alternative: rebuild on pod if needed:

```bash
uv run python scripts/build_tb_qa_dataset.py
```

When rebuilding on pod makes sense:
- You intentionally want a refreshed dataset snapshot.
- You cannot transfer files from local machine.
- You changed dataset-building logic and want pod data regenerated from latest code.

## 6. Connect W&B on pod

```bash
source .venv/bin/activate
wandb login
uv run python -c "import wandb; r=wandb.init(project='lora-phase-1b', name='connectivity_test'); r.log({'ping':1}); r.finish(); print('wandb OK')"
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

## 10. Clean up

Step 10.1 - Revoke temporary pod SSH key from GitHub:
- If added as user key: remove it from GitHub Settings -> SSH and GPG keys
- If added as deploy key: remove it from Repo Settings -> Deploy keys

Step 10.2 - Delete temporary private/public key files on pod:

```bash
rm -f ~/.ssh/id_ed25519_github_vast_temp ~/.ssh/id_ed25519_github_vast_temp.pub
```

Step 10.3 - Remove temporary SSH config stanza (if added):
- Edit `~/.ssh/config` and remove the `Host github.com` block that points to `id_ed25519_github_vast_temp`

Step 10.4 - Stop/destroy pod after backup.
