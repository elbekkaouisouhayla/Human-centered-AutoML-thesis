from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATASETS_DIR, PROCESSED_DIR, SEEDS, TEST_SIZE


TITANIC_FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]


def load_datasets():
    """Load the three raw datasets."""

    return {
        "breast_cancer": pd.read_csv(
            DATASETS_DIR / "breast_cancer.csv"
        ),
        "wine": pd.read_csv(
            DATASETS_DIR / "wine.csv"
        ),
        "titanic": pd.read_csv(
            DATASETS_DIR / "titanic.csv"
        ),
    }


def split_features_and_target(raw_datasets):
    """Separate features X and target y for every dataset."""

    return {
        "breast_cancer": {
            "X": raw_datasets["breast_cancer"].drop(
                columns="target"
            ),
            "y": raw_datasets["breast_cancer"]["target"],
        },
        "wine": {
            "X": raw_datasets["wine"].drop(
                columns="target"
            ),
            "y": raw_datasets["wine"]["target"],
        },
        "titanic": {
            "X": raw_datasets["titanic"][TITANIC_FEATURES].copy(),
            "y": raw_datasets["titanic"]["survived"],
        },
    }


def get_feature_types(datasets):
    """Identify numerical and categorical features."""

    for dataset_name, data in datasets.items():
        X = data["X"]

        if dataset_name == "titanic":
            data["numerical_columns"] = [
                "age",
                "sibsp",
                "parch",
                "fare",
            ]

            data["categorical_columns"] = [
                "pclass",
                "sex",
                "embarked",
            ]

        else:
            data["numerical_columns"] = (
                X.select_dtypes(include=["number"])
                .columns
                .tolist()
            )

            data["categorical_columns"] = (
                X.select_dtypes(exclude=["number"])
                .columns
                .tolist()
            )

    return datasets


def create_feature_pipelines():
    """Create numerical and categorical preprocessing pipelines."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return numeric_pipeline, categorical_pipeline


def create_preprocessors(datasets):
    """Create one ColumnTransformer for each dataset."""

    numeric_pipeline, categorical_pipeline = (
        create_feature_pipelines()
    )

    for data in datasets.values():
        data["preprocessor"] = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    data["numerical_columns"],
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    data["categorical_columns"],
                ),
            ],
            remainder="drop",
        )

    return datasets


def create_train_test_splits(datasets):
    """Create stratified 80/20 splits for every seed."""

    splits = {}

    for dataset_name, data in datasets.items():
        splits[dataset_name] = {}

        for seed in SEEDS:
            X_train, X_test, y_train, y_test = (
                train_test_split(
                    data["X"],
                    data["y"],
                    test_size=TEST_SIZE,
                    random_state=seed,
                    stratify=data["y"],
                )
            )

            splits[dataset_name][seed] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

    return splits


def transform_splits(datasets, splits):
    """
    Fit preprocessing only on training data,
    then transform train and test data.
    """

    processed_splits = {}

    for dataset_name, dataset_splits in splits.items():
        processed_splits[dataset_name] = {}

        for seed, split in dataset_splits.items():
            preprocessor = clone(
                datasets[dataset_name]["preprocessor"]
            )

            X_train_processed = preprocessor.fit_transform(
                split["X_train"]
            )

            X_test_processed = preprocessor.transform(
                split["X_test"]
            )

            feature_names = (
                preprocessor.get_feature_names_out()
            )

            X_train_processed = pd.DataFrame(
                X_train_processed,
                columns=feature_names,
            )

            X_test_processed = pd.DataFrame(
                X_test_processed,
                columns=feature_names,
            )

            processed_splits[dataset_name][seed] = {
                "X_train": X_train_processed,
                "X_test": X_test_processed,
                "y_train": split["y_train"].reset_index(
                    drop=True
                ),
                "y_test": split["y_test"].reset_index(
                    drop=True
                ),
                "preprocessor": preprocessor,
            }

    return processed_splits


def validate_processed_splits(processed_splits):
    """Check that processed datasets are valid."""

    for dataset_name, dataset_splits in (
        processed_splits.items()
    ):
        for seed, split in dataset_splits.items():
            X_train = split["X_train"]
            X_test = split["X_test"]
            y_train = split["y_train"]
            y_test = split["y_test"]

            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

            assert X_train.columns.equals(
                X_test.columns
            )

            assert X_train.isna().sum().sum() == 0
            assert X_test.isna().sum().sum() == 0

            print(
                f"{dataset_name} | seed {seed} | "
                f"train {X_train.shape} | "
                f"test {X_test.shape} | OK"
            )


def save_processed_splits(processed_splits):
    """Save all processed train/test files."""

    for dataset_name, dataset_splits in (
        processed_splits.items()
    ):
        for seed, split in dataset_splits.items():
            output_dir = (
                PROCESSED_DIR
                / dataset_name
                / f"seed_{seed}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            split["X_train"].to_csv(
                output_dir / "X_train.csv",
                index=False,
            )

            split["X_test"].to_csv(
                output_dir / "X_test.csv",
                index=False,
            )

            split["y_train"].to_csv(
                output_dir / "y_train.csv",
                index=False,
            )

            split["y_test"].to_csv(
                output_dir / "y_test.csv",
                index=False,
            )

            print(
                f"Saved: {dataset_name} | seed {seed}"
            )


def preprocess_all_datasets():
    """Run the complete preprocessing workflow."""

    raw_datasets = load_datasets()

    datasets = split_features_and_target(
        raw_datasets
    )

    datasets = get_feature_types(datasets)

    datasets = create_preprocessors(datasets)

    splits = create_train_test_splits(datasets)

    processed_splits = transform_splits(
        datasets,
        splits,
    )

    validate_processed_splits(processed_splits)

    save_processed_splits(processed_splits)

    return processed_splits