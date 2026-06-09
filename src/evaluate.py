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
    {"prompt": ..., "greedy": ..., "sampled": ...} dicts.

    Used to compare base GPT-2 vs fine-tuned output side by side.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Greedy decode
            greedy_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            greedy_text = tokenizer.decode(
                greedy_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()

            # Sampled decode
            sampled_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            sampled_text = tokenizer.decode(
                sampled_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()

            results.append({"prompt": prompt, "greedy": greedy_text, "sampled": sampled_text})

    return results


def compute_ood_perplexity(
    model,
    tokenizer: GPT2Tokenizer,
    device: str,
    n_samples: int = 50,
    max_length: int = 512,
) -> float:
    """
    Compute perplexity on wikitext-2 test set as an out-of-domain control.
    A large increase relative to baseline indicates catastrophic forgetting.
    """
    from datasets import load_dataset
    wikitext = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    texts = [t for t in wikitext["text"] if len(t.strip()) > 50][:n_samples]
    return compute_perplexity(model, tokenizer, texts, device, max_length)
