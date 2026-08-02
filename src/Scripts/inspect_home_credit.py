from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "home-credit-default-risk"
    / "application_train.csv"
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df["TARGET"].value_counts(dropna=False))
    print("\nTarget proportions:")
    print(df["TARGET"].value_counts(normalize=True, dropna=False))

    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    print("\nNumber of numerical columns:", len(numerical_columns))
    print("Number of categorical columns:", len(categorical_columns))

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nTop 20 columns by missing-value percentage:")
    print(missing.head(20))

    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nMemory usage in MB:")
    print(round(df.memory_usage(deep=True).sum() / 1024**2, 2))


if __name__ == "__main__":
    main()