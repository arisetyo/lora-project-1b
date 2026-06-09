import csv
import os
from datetime import datetime

import torch
import wandb
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.data import get_dataloaders
from src.evaluate import compute_perplexity, compute_ood_perplexity
from src.lora import apply_lora_to_gpt2, print_trainable_params
from src.train import train

DATASET = "tb_qa.json"


def run_ablation(
    rank_epoch_pairs: list = None,
    lr: float = 2e-4,
    batch_size: int = 4,
    device: str = "cpu",
    wandb_project: str = "lora-phase-1b",
) -> None:
    """
    Trains a fresh LoRA-adapted GPT-2 for each (rank, epochs) pair on TB Q&A data.
    Records trainable parameter count + final validation perplexity per run.
    Results and checkpoints are written to a timestamped output folder.
    """
    if rank_epoch_pairs is None:
        rank_epoch_pairs = [(4, 3), (8, 3), (16, 3)]

    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"outputs-{run_tag}"
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    results_path = os.path.join(output_dir, "results.csv")
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    val_texts = _loader_to_texts(val_loader, tokenizer, max_examples=100)

    rows = []

    for r, epochs in rank_epoch_pairs:
        print(f"\n{'='*50}")
        print(f"Ablation run: r={r}, epochs={epochs}")
        print(f"{'='*50}")

        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        for p in base_model.parameters():
            p.requires_grad = False
        model = apply_lora_to_gpt2(base_model, r=r, alpha=r)
        print_trainable_params(model)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        run = wandb.init(
            project=wandb_project,
            name=f"ablation_r{r}_e{epochs}_{run_tag}",
            config={
                "r": r,
                "alpha": r,
                "lr": lr,
                "epochs": epochs,
                "domain": "tb_clinical_qa",
                "dataset": DATASET,
            },
            reinit="finish_previous",
        )

        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            wandb_run=run,
            device=device,
            checkpoint_path=os.path.join(checkpoints_dir, f"rank_{r}_epoch{epochs}"),
        )

        val_ppl = compute_perplexity(model, tokenizer, val_texts, device)
        ood_ppl = compute_ood_perplexity(model, tokenizer, device)
        print(f"r={r} | epochs={epochs} | trainable={trainable_params:,} | val_ppl={val_ppl:.2f} | ood_ppl={ood_ppl:.2f}")

        run.log({"val/perplexity": val_ppl, "ood/perplexity": ood_ppl})
        run.finish()

        rows.append({
            "rank": r,
            "epochs": epochs,
            "trainable_params": trainable_params,
            "val_ppl": round(val_ppl, 4),
            "ood_ppl": round(ood_ppl, 4),
        })

    _write_csv(rows, results_path)
    print(f"\nAblation complete. Results written to {results_path}")
    _print_summary(rows)


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


def _write_csv(rows: list[dict], results_path: str) -> None:
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "epochs", "trainable_params", "val_ppl", "ood_ppl"])
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    print("\nSummary:")
    print(f"{'Rank':>6} | {'Epochs':>6} | {'Trainable Params':>18} | {'Val PPL':>10} | {'OOD PPL':>10}")
    print("-" * 65)
    for row in rows:
        print(f"{row['rank']:>6} | {row['epochs']:>6} | {row['trainable_params']:>18,} | {row['val_ppl']:>10.4f} | {row['ood_ppl']:>10.4f}")
