import subprocess
import sys
import os
from pathlib import Path

CHECKS = [
    "scripts/smoke_checks/2a_validate_dataset_json.py",
    "scripts/smoke_checks/2b_spot_check_dataset.py",
    "scripts/smoke_checks/3a_lora_sanity.py",
    "scripts/smoke_checks/3b_dataloader_sanity.py",
    "scripts/smoke_checks/3c_baseline_ppl.py",
    "scripts/smoke_checks/3d_smoke_train_tiny.py",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root)
    )

    for check in CHECKS:
        print("=" * 72)
        print(f"Running {check}")
        print("=" * 72)
        result = subprocess.run([sys.executable, check], check=False, env=env)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print("All smoke checks passed")


if __name__ == "__main__":
    main()
