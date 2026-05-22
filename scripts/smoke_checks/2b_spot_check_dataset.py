import json
import random
from pathlib import Path

DATA_PATH = Path("data/tb_qa.json")
SAMPLE_N = 5
PREVIEW_CHARS = 600


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Dataset is empty")

    random.seed(7)
    n = min(SAMPLE_N, len(data))
    for row in random.sample(data, n):
        print("---")
        print((row.get("formatted") or "")[:PREVIEW_CHARS])


if __name__ == "__main__":
    main()
