import sys
from pathlib import Path

from transformers import GPT2LMHeadModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lora import apply_lora_to_gpt2, print_trainable_params


def main() -> None:
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Freeze base GPT-2 weights before LoRA wrapping.
    for p in model.parameters():
        p.requires_grad = False

    model = apply_lora_to_gpt2(model, r=8, alpha=8)
    print_trainable_params(model)

    for name, p in model.named_parameters():
        if "lora_B" in name:
            assert p.sum() == 0, f"lora_B not zero at init: {name}"
        if "lora_A" in name or "lora_B" in name:
            assert p.requires_grad, f"LoRA param not trainable: {name}"
        elif "lora_A" not in name and "lora_B" not in name:
            assert not p.requires_grad, f"Base param still trainable: {name}"

    print("LoRA sanity passed")


if __name__ == "__main__":
    main()
