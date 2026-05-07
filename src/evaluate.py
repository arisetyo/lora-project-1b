import math

import torch
from transformers import GPT2Tokenizer


def compute_perplexity(
    model,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    device: str,
    max_length: int = 512,
) -> float:
    """
    Compute average perplexity over a list of texts.

    Perplexity = exp(average negative log-likelihood per token).
    Lower is better; a perfectly fitted model would have PPL → 1.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            # outputs.loss is mean NLL per token for this sequence
            n_tokens = (attention_mask == 1).sum().item()
            total_nll += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


def run_prompt_battery(
    model,
    tokenizer: GPT2Tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 100,
) -> list[dict]:
    """
    Generate completions for each prompt and return a list of
    {"prompt": ..., "completion": ...} dicts.

    Used to compare base GPT-2 vs fine-tuned output side by side.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            generated = output_ids[0][input_ids.shape[-1]:]
            completion = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({"prompt": prompt, "completion": completion})

    return results
