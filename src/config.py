from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "Datasets"
PROCESSED_DIR = PROJECT_ROOT / "Processed"
RESULTS_DIR = PROJECT_ROOT / "Results"

DATASET_NAMES = [
    "breast_cancer",
    "wine",
    "titanic",
]

SEEDS = [42, 123, 2026]

TEST_SIZE = 0.20
CV_FOLDS = 5
TIME_BUDGET_MINUTES = 10