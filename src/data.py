import json
import random
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

TB_KEYWORDS = [
    "tuberculosis", " TB ", "mycobacterium tuberculosis",
    "mdr-tb", "xdr-tb", "isoniazid", "rifampicin",
    "rifampin", "pyrazinamide", "ethambutol", "pulmonary tb",
    "latent tb", "bcg", "tuberculous",
]

DATASET_NAME = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"
FALLBACK_DATASET_CONFIG = "pqa_artificial"
DATA_PATH = "data/tb_qa.json"


class TokenizedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx],
        }


def get_dataloaders(
    data_path: str = DATA_PATH,
    dataset_name: str = DATASET_NAME,
    dataset_config: str = DATASET_CONFIG,
    fallback_dataset_config: str = FALLBACK_DATASET_CONFIG,
    batch_size: int = 4, # batch size for DataLoader
    max_length: int = 512, # maximum token length for truncation / padding
    val_split: float = 0.2, # fraction of remaining data to hold out for validation, eg. 20% of remaining after test split
    test_n: int = 100, # number of test samples to hold out, eg. 100 samples for testing
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads formatted Q&A data from data_path when available.
    Falls back to pulling and filtering Hugging Face PubMed QA if the file
    does not exist. Returns train / val / test DataLoaders.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    texts = _load_local_formatted_texts(data_path)
    if texts:
        print(f"Loaded {len(texts)} records from {data_path}")
    else:
        print(f"{data_path} not found or empty; loading {dataset_name} ({dataset_config})...")
        dataset = load_dataset(dataset_name, dataset_config)
        texts = _extract_and_filter(dataset)

        if len(texts) < 300:
            print(
                f"Only {len(texts)} records from {dataset_config}; "
                f"loading fallback {fallback_dataset_config}..."
            )
            fallback_dataset = load_dataset(dataset_name, fallback_dataset_config)
            texts.extend(_extract_and_filter(fallback_dataset))

        print(f"TB-relevant records found from Hugging Face: {len(texts)}")
        _save_cache(texts, data_path)

    if len(texts) < 300:
        raise ValueError(
            f"Only {len(texts)} TB records found — too few to train meaningfully. "
            "Run scripts/build_tb_qa_dataset.py which includes the fallback dataset merge."
        )

    rng = random.Random(seed)
    shuffled_texts = texts[:]
    rng.shuffle(shuffled_texts)

    test_n = min(test_n, max(1, len(shuffled_texts) // 5)) # divide by 5 to ensure at least 20% of data is left for train/val
    test_texts = shuffled_texts[:test_n]
    remaining = shuffled_texts[test_n:]
    split_idx = int(len(remaining) * (1 - val_split))
    train_texts = remaining[:split_idx]
    val_texts = remaining[split_idx:]

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    train_loader = _make_loader(train_texts, tokenizer, max_length, batch_size, shuffle=True)
    val_loader = _make_loader(val_texts, tokenizer, max_length, batch_size, shuffle=False)
    test_loader = _make_loader(test_texts, tokenizer, max_length, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def _is_tb_relevant(record: dict) -> bool:
    """Returns True if any TB keyword appears in the question or answer."""
    question = record.get("question", "").lower()

    # Supports both old BigBio-style answers and PubMedQA long_answer.
    long_answer = (record.get("long_answer") or "").strip()
    answers = record.get("answers", [])
    answer_text = long_answer or " ".join(a.get("text", "") for a in answers)
    answer_text = answer_text.lower()

    combined = f"{question} {answer_text}"
    return any(kw in combined for kw in TB_KEYWORDS)


def _format_qa(record: dict) -> str:
    """Formats a record as a Question / Answer string for causal LM training."""
    question = record.get("question", "").strip()

    long_answer = (record.get("long_answer") or "").strip()
    answers = record.get("answers", [])
    answer_text = long_answer or " ".join(a.get("text", "") for a in answers).strip()

    return f"Question: {question}\nAnswer: {answer_text}"


def _extract_and_filter(dataset) -> list[str]:
    texts = []
    for split_name in dataset.keys():
        for record in dataset[split_name]:
            if _is_tb_relevant(record):
                formatted = _format_qa(record)
                if len(formatted.split()) > 20:  # skip very short entries
                    texts.append(formatted)
    return texts


def _load_local_formatted_texts(data_path: str) -> list[str]:
    path = Path(data_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    texts = []
    for row in records:
        formatted = (row.get("formatted") or "").strip()
        if not formatted:
            q = (row.get("question") or "").strip()
            a = (row.get("long_answer") or row.get("answer") or "").strip()
            if q and a:
                formatted = f"Question: {q}\nAnswer: {a}"

        if formatted and len(formatted.split()) > 20:
            texts.append(formatted)

    return texts


def _make_loader(texts: list[str], tokenizer, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    ds = TokenizedDataset(encodings)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _save_cache(texts: list[str], data_path: str) -> None:
    """Caches filtered HF dataset to data_path so subsequent runs skip the HF download."""
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"formatted": t} for t in texts]
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Cached {len(records)} records to {data_path}")
