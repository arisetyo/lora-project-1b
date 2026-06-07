"""Central configuration for model ID and training hyperparameters — Phase 1B."""

# Toggle this flag to switch between quick smoke tests and full training runs
# Set to True for fast local tests that verify code runs without errors (not meaningful training)
USE_SMOKE_CONFIG = False

MODEL_NAME = "gpt2"
WANDB_PROJECT = "lora-phase-1b"
CHECKPOINT_DIR = "outputs/checkpoints"


# Single-run training defaults
TRAIN_RANK = 8
TRAIN_ALPHA = 8
TRAIN_EPOCHS = 3
TRAIN_LR = 2e-4
TRAIN_BATCH_SIZE = 4        # Lower than Phase 1A — 512-token sequences are heavier

# Local smoke tests before renting GPU
# THIS CONFIG IS NOT FOR MEANINGFUL TRAINING — ONLY FOR QUICK SANITY CHECKS TO VERIFY CODE RUNS WITHOUT ERRORS
SMOKE_RANK = 4
SMOKE_ALPHA = 4
SMOKE_EPOCHS = 1
SMOKE_LR = 2e-4
SMOKE_BATCH_SIZE = 2


def get_active_training_config() -> dict:
    """Return the active training settings based on USE_SMOKE_CONFIG."""
    if USE_SMOKE_CONFIG:
        return {
            "rank": SMOKE_RANK,
            "alpha": SMOKE_ALPHA,
            "epochs": SMOKE_EPOCHS,
            "lr": SMOKE_LR,
            "batch_size": SMOKE_BATCH_SIZE,
            "mode": "smoke",
        }

    return {
        "rank": TRAIN_RANK,
        "alpha": TRAIN_ALPHA,
        "epochs": TRAIN_EPOCHS,
        "lr": TRAIN_LR,
        "batch_size": TRAIN_BATCH_SIZE,
        "mode": "full",
    }


# Rank ablation defaults
ABLATION_RANK_VALUES = [1, 4, 8, 16, 32]
ABLATION_EPOCHS = 3
ABLATION_LR = 2e-4
ABLATION_BATCH_SIZE = 4
ABLATION_RESULTS_PATH = "outputs/results.csv"
