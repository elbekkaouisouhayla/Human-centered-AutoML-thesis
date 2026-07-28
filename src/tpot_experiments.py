import time

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from tpot import TPOTClassifier

from src.config import (
    CV_FOLDS,
    DATASET_NAMES,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEEDS,
    TIME_BUDGET_MINUTES,
)


def run_tpot_experiment(dataset_name, seed):
    """
    Run one TPOT classification experiment.

    Uses:
    - externally preprocessed data
    - stratified 5-fold cross-validation
    - 10-minute optimization budget
    - Macro F1 for pipeline selection
    - untouched test data for final evaluation
    """

    print(f"\nRunning TPOT: {dataset_name} | seed {seed}")

    folder = PROCESSED_DIR / dataset_name / f"seed_{seed}"

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

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    model = TPOTClassifier(
        generations=None,
        population_size=100,
        offspring_size=100,
        scoring="f1_macro",
        cv=cross_validation,
        max_time_mins=TIME_BUDGET_MINUTES,
        max_eval_time_mins=5,
        random_state=seed,
        n_jobs=-1,
        verbosity=2,
        disable_update_check=True,
    )

    total_start = time.perf_counter()

    model.fit(X_train, y_train)

    search_runtime_seconds = time.perf_counter() - total_start

    prediction_start = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_runtime_seconds = (
        time.perf_counter() - prediction_start
    )

    total_runtime_seconds = (
        search_runtime_seconds
        + prediction_runtime_seconds
    )

    result = {
        "framework": "TPOT",
        "dataset": dataset_name,
        "seed": seed,
        "best_model": str(model.fitted_pipeline_),
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
        "prediction_runtime_seconds": prediction_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline_path = (
        RESULTS_DIR
        / f"tpot_pipeline_{dataset_name}_seed_{seed}.py"
    )

    model.export(str(pipeline_path))

    result_path = (
        RESULTS_DIR
        / f"tpot_result_{dataset_name}_seed_{seed}.csv"
    )

    pd.DataFrame([result]).to_csv(
        result_path,
        index=False,
    )

    print("Best pipeline:", result["best_model"])
    print("Test Macro F1:", round(result["f1_macro"], 4))
    print(
        "Search runtime:",
        round(search_runtime_seconds, 2),
        "seconds",
    )

    return result


def run_all_tpot_experiments():
    """
    Run TPOT for all datasets and seeds.
    """

    results = []

    for dataset_name in DATASET_NAMES:
        for seed in SEEDS:
            result = run_tpot_experiment(
                dataset_name,
                seed,
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIR / "tpot_results.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved all TPOT results to: {output_path}")

    return results_df