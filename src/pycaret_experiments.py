import time

import pandas as pd

from pycaret.classification import ClassificationExperiment

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    DATASET_NAMES,
    SEEDS,
    CV_FOLDS,
    TIME_BUDGET_MINUTES,
    PROCESSED_DIR,
    RESULTS_DIR,
)


def run_pycaret_experiment(dataset_name, seed):
    """
    Run one PyCaret experiment.
    """

    print(f"\nRunning PyCaret: {dataset_name} | seed {seed}")

    folder = PROCESSED_DIR / dataset_name / f"seed_{seed}"

    # Load processed data
    X_train = pd.read_csv(folder / "X_train.csv")
    X_test = pd.read_csv(folder / "X_test.csv")

    y_train = (
        pd.read_csv(folder / "y_train.csv")
        .squeeze("columns")
        .reset_index(drop=True)
    )

    y_test = (
        pd.read_csv(folder / "y_test.csv")
        .squeeze("columns")
        .reset_index(drop=True)
    )

    # Reset indices (required by PyCaret)
    train_data = X_train.reset_index(drop=True).copy()
    train_data["target"] = y_train

    test_data = X_test.reset_index(drop=True).copy()
    test_data["target"] = y_test

    experiment = ClassificationExperiment()

    total_start = time.perf_counter()

    experiment.setup(
        data=train_data,
        target="target",
        test_data=test_data,

        # IMPORTANT
        index=False,

        preprocess=False,

        fold_strategy="stratifiedkfold",
        fold=CV_FOLDS,
        fold_shuffle=True,

        session_id=seed,

        n_jobs=-1,
        verbose=False,
    )

    experiment.add_metric(
        id="macro_f1",
        name="Macro F1",
        score_func=f1_score,
        greater_is_better=True,
        multiclass=True,
        average="macro",
    )

    search_start = time.perf_counter()

    best_model = experiment.compare_models(
        sort="Macro F1",
        fold=CV_FOLDS,
        budget_time=TIME_BUDGET_MINUTES,
        turbo=True,
        errors="ignore",
        verbose=False,
    )

    search_runtime_seconds = time.perf_counter() - search_start

    leaderboard = experiment.pull().copy()

    predictions = experiment.predict_model(
        best_model,
        data=X_test.reset_index(drop=True),
        verbose=False,
    )

    y_pred = predictions["prediction_label"]

    total_runtime_seconds = time.perf_counter() - total_start

    result = {
        "framework": "PyCaret",
        "dataset": dataset_name,
        "seed": seed,
        "best_model": type(best_model).__name__,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "search_runtime_seconds": search_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard.to_csv(
        RESULTS_DIR
        / f"pycaret_leaderboard_{dataset_name}_seed_{seed}.csv",
        index=False,
    )

    print("Best model:", result["best_model"])
    print("Test Macro F1:", round(result["f1_macro"], 4))
    print(
        "Search runtime:",
        round(search_runtime_seconds, 2),
        "seconds",
    )

    return result


def run_all_pycaret_experiments():
    """
    Run all PyCaret experiments.
    """

    results = []

    for dataset_name in DATASET_NAMES:
        for seed in SEEDS:
            results.append(
                run_pycaret_experiment(
                    dataset_name,
                    seed,
                )
            )

    results = pd.DataFrame(results)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_DIR / "pycaret_results.csv",
        index=False,
    )

    print("\nAll PyCaret experiments completed.")

    return results