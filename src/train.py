import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 2e-4,
    wandb_run=None,
    device: str = "cpu",
    checkpoint_path: str = None,
) -> None:
    """
    Trains a LoRA-patched model. Logs train/val loss to W&B each epoch.
    Optionally saves A and B adapter weights to checkpoint_path.
    """
    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, scheduler, device, training=True)
        val_loss = _run_epoch(model, val_loader, None, None, device, training=False)

        print(f"Epoch {epoch}/{epochs} — train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if wandb_run:
            wandb_run.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "epoch": epoch,
            })

        if checkpoint_path:
            _save_adapter(model, f"{checkpoint_path}_epoch{epoch}.pt")


def _run_epoch(model, loader, optimizer, scheduler, device, training: bool) -> float:
    """Run one full pass over loader, returning avg_loss."""
    model.train() if training else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

    return total_loss / len(loader)


def _save_adapter(model, path: str) -> None:
    """Saves only the LoRA adapter weights (A and B), not W."""
    adapter_state = {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save(adapter_state, path)
    print(f"Adapter saved to {path} ({len(adapter_state)} tensors)")
