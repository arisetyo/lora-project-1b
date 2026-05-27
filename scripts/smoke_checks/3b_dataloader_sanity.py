import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataloaders


def main() -> None:
    train_l, val_l, test_l = get_dataloaders(batch_size=2)
    batch = next(iter(train_l))
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("labels shape:   ", tuple(batch["labels"].shape))
    assert batch["input_ids"].shape == batch["labels"].shape, "input_ids and labels must match for causal LM"
    assert tuple(batch["input_ids"].shape)[1] == 512, "expected max_length=512"
    print("DataLoader sanity passed")
    print("splits:", len(train_l), len(val_l), len(test_l))


if __name__ == "__main__":
    main()
