"""
Replay the TB prompt battery on base GPT-2 vs a trained LoRA adapter,
WITHOUT retraining. Loads a saved adapter checkpoint (lora_A/lora_B only)
and runs the same fixed prompt battery used during training.

Usage (from repo root):

    .venv/bin/python scripts/run_prompt_battery.py \
        --checkpoint outputs/20260609-second-ablation-run_focused-ranks/checkpoints/rank_16_epoch10_epoch10.pt \
        --rank 16 --alpha 16 \
        --out docs/base-vs-finetuned.md

Outputs a side-by-side Markdown file (and prints to stdout). A fixed seed
makes the sampled decodes reproducible.
"""
import argparse
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# project-root bootstrap so `src` and `prompts` import regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import run_prompt_battery  # noqa: E402
from src.lora import apply_lora_to_gpt2  # noqa: E402
from prompts import PROMPTS  # noqa: E402


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_adapted_model(checkpoint: str, rank: int, alpha: float, device: str):
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    for p in base.parameters():
        p.requires_grad = False
    model = apply_lora_to_gpt2(base, r=rank, alpha=alpha)
    state = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # missing = all non-LoRA weights (expected, they keep pretrained values)
    loaded = [k for k in state if k not in unexpected]
    print(f"Loaded {len(loaded)} adapter tensors from {checkpoint}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys in checkpoint: {unexpected[:3]}...")
    return model.to(device)


def _to_markdown(base_res, lora_res, args) -> str:
    lines = [
        "# Base GPT-2 vs. GPT-2 + LoRA — TB Prompt Battery",
        "",
        f"- Adapter checkpoint: `{args.checkpoint}`",
        f"- LoRA config: rank={args.rank}, alpha={args.alpha}",
        f"- Decoding: greedy, and sampled (temperature=0.7, top_p=0.9), max_new_tokens={args.max_new_tokens}",
        f"- Seed: {args.seed}",
        "",
    ]
    for i, (b, l) in enumerate(zip(base_res, lora_res), start=1):
        prompt = b["prompt"].replace("\nAnswer:", " / Answer:")
        lines += [
            f"## Prompt {i}",
            "",
            f"> {prompt}",
            "",
            "| Model | Greedy | Sampled |",
            "| --- | --- | --- |",
            f"| Base GPT-2 | {_cell(b['greedy'])} | {_cell(b['sampled'])} |",
            f"| GPT-2 + LoRA | {_cell(l['greedy'])} | {_cell(l['sampled'])} |",
            "",
        ]
    return "\n".join(lines)


def _cell(text: str) -> str:
    """Make a generation safe for a Markdown table cell."""
    return text.replace("\n", " ").replace("|", "\\|").strip() or "(empty)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to adapter .pt (lora_A/lora_B)")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=100, dest="max_new_tokens")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/base-vs-finetuned.md")
    args = ap.parse_args()

    device = _get_device()
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # --- Base GPT-2 (before adaptation) ---
    torch.manual_seed(args.seed)
    base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    print("\nRunning prompt battery on BASE GPT-2...")
    base_res = run_prompt_battery(base, tokenizer, PROMPTS, device, max_new_tokens=args.max_new_tokens)

    # --- GPT-2 + LoRA adapter (after adaptation) ---
    torch.manual_seed(args.seed)
    lora = _load_adapted_model(args.checkpoint, args.rank, args.alpha, device)
    print("Running prompt battery on GPT-2 + LoRA...")
    lora_res = run_prompt_battery(lora, tokenizer, PROMPTS, device, max_new_tokens=args.max_new_tokens)

    md = _to_markdown(base_res, lora_res, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"\nWrote side-by-side comparison to {args.out}")


if __name__ == "__main__":
    main()
