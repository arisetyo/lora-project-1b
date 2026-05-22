import torch
from transformers import GPT2LMHeadModel

from src.data import get_dataloaders
from src.lora import apply_lora_to_gpt2
from src.train import train


class TinyLoader:
    def __init__(self, loader, n: int):
        self.loader = loader
        self.n = n

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n:
                break
            yield batch

    def __len__(self):
        return self.n


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    train_l, val_l, _ = get_dataloaders(batch_size=2)

    tiny_train = TinyLoader(train_l, 5)
    tiny_val = TinyLoader(val_l, 2)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    for p in model.parameters():
        p.requires_grad = False
    model = apply_lora_to_gpt2(model, r=4, alpha=4)

    train(
        model=model,
        train_loader=tiny_train,
        val_loader=tiny_val,
        epochs=1,
        lr=2e-4,
        device=device,
    )

    print("Smoke train passed")


if __name__ == "__main__":
    main()
