from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

TB_KEYWORDS = [
    "tuberculosis", " TB ", "mycobacterium tuberculosis",
    "mdr-tb", "xdr-tb", "isoniazid", "rifampicin",
    "rifampin", "pyrazinamide", "ethambutol", "pulmonary tb",
    "latent tb", "bcg", "tuberculous",
]

DATASET_NAME = "bigbio/pubmed_qa"
DATASET_CONFIG = "pubmed_qa_labeled_fold0_bigbio_qa"


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
    dataset_name: str = DATASET_NAME,
    dataset_config: str = DATASET_CONFIG,
    batch_size: int = 4,
    max_length: int = 512,
    val_split: float = 0.2,
    test_n: int = 100,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads PubMed QA, filters for TB-relevant records, formats as Q&A pairs,
    and returns train / val / test DataLoaders.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)

    texts = _extract_and_filter(dataset)
    print(f"TB-relevant records found: {len(texts)}")

    if len(texts) < 200:
        raise ValueError(
            f"Only {len(texts)} TB records found — too few to train meaningfully. "
            "Try adding more keywords or using lavita/medical-qa-datasets as fallback."
        )

    test_texts = texts[:test_n]
    remaining = texts[test_n:]
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
    # bigbio format: answers is a list of dicts with "text" key
    answers = record.get("answers", [])
    answer_text = " ".join(a.get("text", "") for a in answers).lower()
    combined = f"{question} {answer_text}"
    return any(kw in combined for kw in TB_KEYWORDS)


def _format_qa(record: dict) -> str:
    """Formats a record as a Question / Answer string for causal LM training."""
    question = record.get("question", "").strip()
    answers = record.get("answers", [])
    answer_text = " ".join(a.get("text", "") for a in answers).strip()
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
