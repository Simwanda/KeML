# src/config.py

import os
from pathlib import Path

# Base paths (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
SRC_DIR      = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

# Make sure folders exist
for p in [ARTIFACTS_DIR, OUTPUTS_DIR,
          ARTIFACTS_DIR / "models",
          ARTIFACTS_DIR / "optuna",
          OUTPUTS_DIR / "figures",
          OUTPUTS_DIR / "metrics",
          OUTPUTS_DIR / "predictions"]:
    os.makedirs(p, exist_ok=True)

# File paths
DB_CSV_PATH = DATA_DIR / "rcfst_database.csv"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_SPLITS_CV  = 5

# Default model names used throughout
MODEL_NAMES = ["KNN", "SVR", "DT", "ANN", "RF", "XGB"]
