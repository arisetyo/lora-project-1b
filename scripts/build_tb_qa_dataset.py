"""
One-shot dataset builder. Pulls bigbio/pubmed_qa, filters for TB,
formats as Q&A, writes data/tb_qa.json.

Run from the project root:
    uv run python scripts/build_tb_qa_dataset.py
"""

import json
from pathlib import Path

from datasets import load_dataset

TB_KEYWORDS = [
    "tuberculosis",
    " TB ",
    "Mycobacterium",
    "MDR-TB",
    "XDR-TB",
    "isoniazid",
    "rifampicin",
    "pulmonary TB",
    "pyrazinamide",
    "ethambutol",
]

OUTPUT_PATH = Path("data/tb_qa.json")
PRIMARY_CONFIG = "pubmed_qa_labeled_fold0_source"
FALLBACK_CONFIG = "pubmed_qa_artificial_source"
MIN_RECORDS = 300


def is_tb_relevant(question: str, long_answer: str) -> bool:
    text = f"{question} {long_answer}".lower()
    keywords_lower = [kw.lower() for kw in TB_KEYWORDS]
    return any(kw in text for kw in keywords_lower)


def extract_records(dataset_split) -> list[dict]:
    out = []
    for row in dataset_split:
        question = (row.get("QUESTION") or row.get("question") or "").strip()
        long_answer = (row.get("LONG_ANSWER") or row.get("long_answer") or "").strip()
        if not question or not long_answer:
            continue
        if not is_tb_relevant(question, long_answer):
            continue
        out.append({
            "question": question,
            "long_answer": long_answer,
            "formatted": f"Question: {question}\nAnswer: {long_answer}",
        })
    return out


def build():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PRIMARY_CONFIG}...")
    ds = load_dataset("bigbio/pubmed_qa", PRIMARY_CONFIG, trust_remote_code=True)
    records = []
    for split_name in ds:
        records.extend(extract_records(ds[split_name]))
    print(f"  → {len(records)} TB-relevant records from primary config")

    if len(records) < MIN_RECORDS:
        print(f"Below threshold ({MIN_RECORDS}); pulling fallback {FALLBACK_CONFIG}...")
        ds_fb = load_dataset("bigbio/pubmed_qa", FALLBACK_CONFIG, trust_remote_code=True)
        for split_name in ds_fb:
            records.extend(extract_records(ds_fb[split_name]))
        print(f"  → {len(records)} TB-relevant records after fallback merge")

    seen = set()
    deduped = []
    for r in records:
        key = r["formatted"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"  → {len(deduped)} after dedup")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUTPUT_PATH} ({len(deduped)} records)")


if __name__ == "__main__":
    build()
