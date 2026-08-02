from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "home-credit-default-risk"
    / "application_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Processed"
    / "home_credit"
    / "seed_42"
)

TARGET_COLUMN = "TARGET"
RANDOM_SEED = 42
TEST_SIZE = 0.20


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COLUMN],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train_raw.csv"
    test_path = OUTPUT_DIR / "test_raw.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Original shape:", df.shape)
    print("Training shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTraining target proportions:")
    print(train_df[TARGET_COLUMN].value_counts(normalize=True))

    print("\nTest target proportions:")
    print(test_df[TARGET_COLUMN].value_counts(normalize=True))

    print("\nFiles saved to:")
    print(train_path)
    print(test_path)


if __name__ == "__main__":
    main()