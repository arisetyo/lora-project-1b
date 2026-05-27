import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataloaders
from src.evaluate import compute_perplexity


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    _, val_l, _ = get_dataloaders(batch_size=2)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    texts = []
    for batch in val_l:
        for ids in batch["input_ids"]:
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        if len(texts) >= 20:
            break

    ppl = compute_perplexity(model, tokenizer, texts[:20], device)
    print(f"Baseline val PPL: {ppl:.2f}")
    assert 1 < ppl < 1e6, f"PPL out of expected range: {ppl}"
    print("Baseline PPL check passed")


if __name__ == "__main__":
    main()
