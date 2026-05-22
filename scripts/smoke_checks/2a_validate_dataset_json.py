import json
from pathlib import Path

DATA_PATH = Path("data/tb_qa.json")
MIN_RECORDS = 300


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "dataset JSON must be a list"
    assert len(data) >= MIN_RECORDS, f"Too few records: {len(data)} (expected >= {MIN_RECORDS})"
    assert all("formatted" in r for r in data), "missing formatted field"
    assert all(
        "Question:" in r["formatted"] and "Answer:" in r["formatted"]
        for r in data
    ), "bad format in formatted field"

    print(f"dataset OK: {len(data)}")


if __name__ == "__main__":
    main()
