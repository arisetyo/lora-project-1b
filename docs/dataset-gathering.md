# Dataset Gathering — TB Clinical Q&A (Phase 1B)

This document is the operational walkthrough for assembling the Phase 1B dataset. It assumes you are running locally (or on the same machine that will later push the file to a training instance) with Python 3.10+ and an active virtual environment.

## 1. What we need

A JSON file at `data/tb_qa.json` containing 500–2,000 records, each shaped as:

```json
{
  "question": "What is the standard first-line treatment for drug-sensitive tuberculosis?",
  "long_answer": "The standard first-line regimen for drug-sensitive TB ...",
  "formatted": "Question: ...\nAnswer: ..."
}
```

The `formatted` field is what the trainer actually tokenizes; the other two are kept for traceability and easier inspection.

## 2. Sources

### Primary — `bigbio/pubmed_qa` (HuggingFace)

A PubMed-derived Q&A dataset. Multiple configs exist; the most useful for our purpose are:

- `pubmed_qa_labeled_fold0_source` — ~1,000 expert-labeled Q&A pairs with long-form answers. Smaller but cleaner.
- `pubmed_qa_artificial_source` — ~211,000 automatically generated Q&A pairs from PubMed abstracts. Much larger, slightly noisier, and the right starting point if the labeled fold yields too few TB records after filtering.

### Fallback — `lavita/medical-qa-datasets`

A broader medical Q&A collection. Use this only if both PubMed QA configs yield fewer than 300 TB records after filtering. The format differs — check the schema before reusing the filter.

### Why not scrape PubMed directly?

We could call the NCBI E-utilities API and pull abstracts ourselves, but that adds rate-limiting concerns, requires API key management, and produces raw abstracts rather than Q&A pairs. PubMed QA does that conversion for us. Save the direct-scrape path for Phase 2+ if dataset size becomes a real bottleneck.

## 3. Setup

From the project root with `.venv` activated:

```bash
uv pip install "datasets>=2.18.0"
```

(or `pip install` if you don't use `uv` — the Phase 1A repo uses `uv`, so match that convention)

A HuggingFace account is **not** required for these datasets — they are publicly accessible. If you do want auth (e.g. to avoid rate limits), run `huggingface-cli login` once.

## 4. The filter

A TB-relevant record contains at least one of these tokens (case-insensitive) in either `question` or `long_answer`:

```python
TB_KEYWORDS = [
    "tuberculosis",
    " TB ",            # leading and trailing space avoids matching "TBA", "STB", etc.
    "Mycobacterium",
    "MDR-TB",
    "XDR-TB",
    "isoniazid",
    "rifampicin",
    "pulmonary TB",
    "pyrazinamide",
    "ethambutol",
]
```

The space-padded `" TB "` is intentional. A naive `"TB"` substring match grabs words like "STBT", "PTBM", chemistry terms, and abbreviated patient IDs. The padded form misses TB at the very start or end of a sentence — acceptable, because such records nearly always also contain "tuberculosis" elsewhere in the long answer.

## 5. End-to-end script

Save this as `scripts/build_tb_qa_dataset.py` in the project. Running it once produces `data/tb_qa.json`.

```python
"""
One-shot dataset builder. Pulls bigbio/pubmed_qa, filters for TB,
formats as Q&A, writes data/tb_qa.json.

Run from the project root:
    uv run python scripts/build_tb_qa_dataset.py
"""

import json
import os
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
    ds = load_dataset("bigbio/pubmed_qa", PRIMARY_CONFIG)
    records = []
    for split_name in ds:
        records.extend(extract_records(ds[split_name]))
    print(f"  → {len(records)} TB-relevant records from primary config")

    if len(records) < MIN_RECORDS:
        print(f"Below threshold ({MIN_RECORDS}); pulling fallback {FALLBACK_CONFIG}...")
        ds_fb = load_dataset("bigbio/pubmed_qa", FALLBACK_CONFIG)
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
```

Run it:

```bash
uv run python scripts/build_tb_qa_dataset.py
```

Expected log output:

```
Loading pubmed_qa_labeled_fold0_source...
  → 67 TB-relevant records from primary config
Below threshold (300); pulling fallback pubmed_qa_artificial_source...
  → 1,432 TB-relevant records after fallback merge
  → 1,408 after dedup
Wrote data/tb_qa.json (1,408 records)
```

The exact count varies — anything between roughly 500 and 2,000 is fine.

## 6. Validation

Before pointing the trainer at the file, sanity-check it:

```bash
python -c "
import json
with open('data/tb_qa.json') as f:
    data = json.load(f)
assert len(data) > 0, 'empty dataset'
assert all('formatted' in r for r in data), 'missing formatted field'
assert all('Question:' in r['formatted'] and 'Answer:' in r['formatted'] for r in data), 'bad formatting'
lengths = [len(r['formatted']) for r in data]
print(f'OK — {len(data)} records')
print(f'  formatted-string char length: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}')
print(f'  sample record:')
print(f'    {data[0][\"formatted\"][:300]}...')
"
```

You should see:
- Record count between ~500 and ~2,000
- Median formatted-string length somewhere in the 400–1,500 character range (TB long-answers are typically dense paragraphs)
- The sample record reads like a plausible PubMed abstract excerpt

If the median length is under 100 characters, the loader probably picked up the wrong field — check the dataset schema and adjust the field names in `extract_records`.

## 7. Manual spot-check

Read 5 random records by eye. You're looking for:

- **Real TB content** — drug names, regimen durations, diagnostic terms. If half of the "TB" hits are actually about thyroid-binding globulin or some other unrelated abbreviation, tighten the keyword list (drop the bare `" TB "`, keep only spelled-out terms).
- **Reasonable answer length** — at least a couple of sentences. Single-sentence answers won't give the model enough generative signal.
- **English** — PubMed QA is English-only; if non-English text appears, it's a parsing bug.

```bash
python -c "
import json, random
random.seed(7)
with open('data/tb_qa.json') as f:
    data = json.load(f)
for r in random.sample(data, 5):
    print('---')
    print(r['formatted'][:600])
"
```

## 8. What to do if yield is too low

If even after the fallback you have fewer than 300 records:

1. **Loosen the keyword list.** Add `BCG`, `latent TB`, `extrapulmonary`, `acid-fast bacilli`, `AFB smear`, `directly observed therapy`, `DOTS`.
2. **Drop the substring requirement on `long_answer` and require it on `question`** — flips the false-positive/false-negative tradeoff. Often increases yield because TB-related questions usually mention TB explicitly even when the answer paragraph doesn't.
3. **Switch fallback dataset** to `lavita/medical-qa-datasets`. The schema is different (`input` / `output` instead of `question` / `long_answer`) — adjust `extract_records` accordingly. Document the change in your run log.
4. **As a last resort, augment with a small synthetic batch.** Generate 200–300 TB Q&A pairs using a Claude prompt similar to Phase 1A's synthetic pipeline. Mark these clearly in the JSON (`"source": "synthetic"`) so they can be excluded from the held-out test split — the test set should only contain real PubMed records to keep evaluation honest.

## 9. Train / val / test split

The split is done in `src/data.py` at load time, not at build time. The `tb_qa.json` file is the single source of truth; splitting happens with a fixed seed inside the dataloader so it is reproducible across runs.

Target proportions:
- Train: 80% of the non-test set
- Validation: 20% of the non-test set
- Test: 50–100 records held out **before** the train/val split, never seen during training, used only for the prompt battery and base-vs-fine-tuned comparison

## 10. Caching and re-runs

`data/tb_qa.json` is the cached, processed dataset — commit-worthy if it fits in the repo (it should, at this size — a 2,000-record JSON is well under 10 MB). The HuggingFace `datasets` library will also cache the raw download under `~/.cache/huggingface/datasets/`. You can safely delete that cache; the build script will re-download the next time it runs.

If you change the keyword list, **delete `data/tb_qa.json` and rebuild** — otherwise the trainer keeps using the old filter results.

## 11. Licensing note

`bigbio/pubmed_qa` is derived from PubMed abstracts, which are public-domain in the United States but may carry usage restrictions in other jurisdictions for commercial redistribution. For Phase 1B (a learning artifact, no production use, no model release), we are well within fair use. If outputs of this project are ever published or commercialized, revisit licensing before redistributing the dataset itself.

---

*Once `data/tb_qa.json` is built and validated, the trainer in `run_training.py` can load it directly. No further data prep is needed.*
