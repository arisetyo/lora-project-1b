"""
Entry point for a single Phase 1B training run (TB clinical Q&A).
Edit the config block below, then run:

    python run_training.py

For the full rank ablation:

    python -c "from src.ablation import run_ablation; run_ablation(device='cuda')"

Note: lora.py, train.py, and evaluate.py are copied unchanged from Phase 1A.
Only data.py, prompts.py, and this config block differ.
"""

import os
import torch
import wandb
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.data import get_dataloaders
from src.evaluate import compute_perplexity, run_prompt_battery
from src.lora import apply_lora_to_gpt2, print_trainable_params
from src.train import train
from prompts import PROMPTS

# --- Config ---
RANK = 8
ALPHA = 8
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 4          # reduced from Phase 1A — 512-token sequences are heavier
WANDB_PROJECT = "lora-tb-phase1b"
CHECKPOINT_DIR = "outputs/checkpoints"
# --------------

def main():
    device = _get_device()
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    print("\nLoading GPT-2 and applying LoRA...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = apply_lora_to_gpt2(base_model, r=RANK, alpha=ALPHA)
    print_trainable_params(model)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nComputing baseline perplexity...")
    baseline_texts = _loader_to_texts(val_loader, tokenizer, max_examples=100)
    baseline_ppl = compute_perplexity(model, tokenizer, baseline_texts, device)
    print(f"Baseline val PPL: {baseline_ppl:.2f}")

    print("\nRunning baseline prompt battery...")
    baseline_responses = run_prompt_battery(model, tokenizer, PROMPTS, device)
    _print_battery(baseline_responses, label="BEFORE TRAINING")

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"run_r{RANK}",
        config={
            "r": RANK,
            "alpha": ALPHA,
            "lr": LR,
            "epochs": EPOCHS,
            "domain": "tb_clinical_qa",
            "dataset": "bigbio/pubmed_qa",
        },
    )
    run.log({"baseline/val_ppl": baseline_ppl})

    print(f"\nStarting training (r={RANK}, epochs={EPOCHS}, lr={LR})...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        wandb_run=run,
        device=device,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, f"rank_{RANK}"),
    )

    print("\nComputing post-training perplexity...")
    final_ppl = compute_perplexity(model, tokenizer, baseline_texts, device)
    reduction = (1 - final_ppl / baseline_ppl) * 100
    print(f"Final val PPL: {final_ppl:.2f}  (baseline was {baseline_ppl:.2f})")
    print(f"PPL reduction: {reduction:.1f}%")
    run.log({"final/val_ppl": final_ppl, "ppl_reduction_pct": reduction})

    print("\nRunning post-training prompt battery...")
    final_responses = run_prompt_battery(model, tokenizer, PROMPTS, device)
    _print_battery(final_responses, label="AFTER TRAINING")

    run.finish()


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _loader_to_texts(loader, tokenizer, max_examples: int) -> list[str]:
    texts = []
    for batch in loader:
        for ids in batch["input_ids"]:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            if text.strip():
                texts.append(text)
            if len(texts) >= max_examples:
                return texts
    return texts


def _print_battery(results: list[dict], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  PROMPT BATTERY — {label}")
    print(f"{'='*60}")
    for item in results:
        print(f"\nPrompt:   {item['prompt']}")
        print(f"Response: {item['response']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
