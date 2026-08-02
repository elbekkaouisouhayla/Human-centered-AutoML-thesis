from pathlib import Path
from time import perf_counter

import pandas as pd
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "Processed"
    / "home_credit"
    / "seed_42"
    / "train_raw.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "Processed"
    / "home_credit"
    / "seed_42"
    / "test_raw.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "Results"
    / "home_credit"
    / "pycaret"
)

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

SEED = 42
N_FOLDS = 5


# ============================================================
# Helper functions
# ============================================================

def check_files() -> None:
    """Check whether the train and test datasets exist."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found at:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found at:\n{TEST_PATH}"
        )


def get_positive_class_probability(
    predictions_df: pd.DataFrame,
) -> pd.Series | None:
    """Find PyCaret's probability column for class 1."""

    possible_columns = [
        "prediction_score_1",
        "Score_1",
        "score_1",
    ]

    for column in possible_columns:
        if column in predictions_df.columns:
            return predictions_df[column]

    return None


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    print("=" * 70)
    print("HOME CREDIT — PYCARET NATIVE AUTOMATION EXPERIMENT")
    print("=" * 70)

    check_files()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading raw datasets...")

    # reset_index prevents PyCaret duplicate-index errors
    train_df = (
        pd.read_csv(TRAIN_PATH)
        .reset_index(drop=True)
    )

    test_df = (
        pd.read_csv(TEST_PATH)
        .reset_index(drop=True)
    )

    print(f"Training shape: {train_df.shape}")
    print(f"Test shape:     {test_df.shape}")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing "
            "from the training dataset."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing "
            "from the test dataset."
        )

    print("\nTraining target proportions:")
    print(
        train_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nTest target proportions:")
    print(
        test_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
    )

    X_test = (
        test_df
        .drop(columns=[TARGET_COLUMN])
        .reset_index(drop=True)
    )

    y_test = (
        test_df[TARGET_COLUMN]
        .astype(int)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # PyCaret setup
    # --------------------------------------------------------

    experiment = ClassificationExperiment()

    print("\nStarting PyCaret setup...")

    setup_start = perf_counter()

    setup_arguments = {
        "data": train_df,
        "target": TARGET_COLUMN,
        "test_data": test_df,

        # Important fix:
        # PyCaret must not use the pandas index as an identifier.
        "index": False,

        "session_id": SEED,
        "fold_strategy": "stratifiedkfold",
        "fold": N_FOLDS,
        "fold_shuffle": True,

        # Native PyCaret preprocessing
        "preprocess": True,
        "imputation_type": "simple",
        "numeric_imputation": "mean",
        "categorical_imputation": "mode",

        # Keep these disabled to observe the default/basic workflow
        "fix_imbalance": False,
        "normalize": False,
        "remove_outliers": False,
        "feature_selection": False,

        "n_jobs": -1,
        "html": False,
        "verbose": False,
        "system_log": False,
    }

    # Ignore the customer identifier
    if ID_COLUMN in train_df.columns:
        setup_arguments["ignore_features"] = [ID_COLUMN]

    experiment.setup(**setup_arguments)

    setup_runtime = perf_counter() - setup_start

    print(
        f"PyCaret setup completed in "
        f"{setup_runtime:.2f} seconds."
    )

    # --------------------------------------------------------
    # Save setup information
    # --------------------------------------------------------

    setup_summary = pd.DataFrame(
        [
            {
                "framework": "PyCaret",
                "dataset": "Home Credit Default Risk",
                "target": TARGET_COLUMN,
                "ignored_feature": (
                    ID_COLUMN
                    if ID_COLUMN in train_df.columns
                    else "None"
                ),
                "seed": SEED,
                "folds": N_FOLDS,
                "index_handling": "index=False",
                "preprocessing": "PyCaret native preprocessing",
                "numeric_imputation": "Mean",
                "categorical_imputation": "Mode",
                "class_balancing": "Disabled",
                "normalization": "Disabled",
                "outlier_removal": "Disabled",
                "feature_selection": "Disabled",
            }
        ]
    )

    setup_summary.to_csv(
        RESULTS_DIR / "pycaret_setup_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------

    print("\nStarting model comparison...")
    print("The dataset is large, so this may take a long time.")

    search_start = perf_counter()

    best_model = experiment.compare_models(
        sort="F1",
        n_select=1,
        turbo=True,
        errors="ignore",
        verbose=True,
    )

    search_runtime = perf_counter() - search_start

    print(
        f"\nModel comparison completed in "
        f"{search_runtime:.2f} seconds."
    )

    print("\nSelected model:")
    print(best_model)

    # Save the model-comparison table
    leaderboard = experiment.pull()

    leaderboard.to_csv(
        RESULTS_DIR / "pycaret_model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Predict on the held-out test dataset
    # --------------------------------------------------------

    print("\nGenerating held-out test predictions...")

    prediction_start = perf_counter()

    pycaret_predictions = experiment.predict_model(
        estimator=best_model,
        data=X_test,
        raw_score=True,
        verbose=False,
    )

    prediction_runtime = perf_counter() - prediction_start

    if "prediction_label" not in pycaret_predictions.columns:
        raise ValueError(
            "The prediction output does not contain "
            "'prediction_label'.\n"
            f"Available columns:\n"
            f"{list(pycaret_predictions.columns)}"
        )

    predicted_labels = (
        pycaret_predictions["prediction_label"]
        .astype(int)
        .reset_index(drop=True)
    )

    positive_probabilities = get_positive_class_probability(
        pycaret_predictions
    )

    if positive_probabilities is not None:
        positive_probabilities = (
            positive_probabilities
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predicted_labels,
    )

    macro_precision = precision_score(
        y_test,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    positive_class_precision = precision_score(
        y_test,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    positive_class_recall = recall_score(
        y_test,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    positive_class_f1 = f1_score(
        y_test,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    if positive_probabilities is not None:
        roc_auc = roc_auc_score(
            y_test,
            positive_probabilities,
        )
    else:
        roc_auc = None

        print(
            "\nWarning: PyCaret did not return a probability "
            "column for class 1. ROC AUC will remain empty."
        )

    confusion = confusion_matrix(
        y_test,
        predicted_labels,
        labels=[0, 1],
    )

    true_negatives = int(confusion[0, 0])
    false_positives = int(confusion[0, 1])
    false_negatives = int(confusion[1, 0])
    true_positives = int(confusion[1, 1])

    total_runtime = (
        setup_runtime
        + search_runtime
        + prediction_runtime
    )

    # --------------------------------------------------------
    # Create results table
    # --------------------------------------------------------

    results = {
        "framework": "PyCaret",
        "dataset": "Home Credit Default Risk",
        "selected_model": type(best_model).__name__,
        "seed": SEED,
        "folds": N_FOLDS,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "number_of_input_columns": X_test.shape[1],
        "setup_runtime_seconds": setup_runtime,
        "model_search_runtime_seconds": search_runtime,
        "prediction_runtime_seconds": prediction_runtime,
        "total_runtime_seconds": total_runtime,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "positive_class_precision": positive_class_precision,
        "positive_class_recall": positive_class_recall,
        "positive_class_f1": positive_class_f1,
        "roc_auc": roc_auc,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
    }

    results_df = pd.DataFrame([results])

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output_predictions = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": predicted_labels,
        }
    )

    if ID_COLUMN in test_df.columns:
        output_predictions.insert(
            0,
            ID_COLUMN,
            test_df[ID_COLUMN].reset_index(drop=True),
        )

    if positive_probabilities is not None:
        output_predictions["probability_class_1"] = (
            positive_probabilities
        )

    output_predictions.to_csv(
        RESULTS_DIR / "pycaret_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_DIR / "pycaret_results.csv",
        index=False,
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )

    confusion_df.to_csv(
        RESULTS_DIR / "pycaret_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Save selected model
    # --------------------------------------------------------

    model_path = RESULTS_DIR / "pycaret_best_model"

    experiment.save_model(
        best_model,
        str(model_path),
        verbose=False,
    )

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 70)

    print(results_df.T)

    print("\nConfusion matrix:")
    print(confusion_df)

    print("\nFiles saved in:")
    print(RESULTS_DIR)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()