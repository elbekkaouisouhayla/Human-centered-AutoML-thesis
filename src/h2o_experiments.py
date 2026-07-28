from __future__ import annotations

import time

import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    DATASET_NAMES,
    SEEDS,
    CV_FOLDS,
    TIME_BUDGET_MINUTES,
)


def run_h2o_experiment(dataset_name: str, seed: int) -> dict:
    """
    Run one H2O AutoML experiment.
    """

    print(f"\nRunning H2O: {dataset_name} | seed {seed}")

    split_dir = PROCESSED_DIR / dataset_name / f"seed_{seed}"

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(split_dir / "y_test.csv").squeeze("columns")

    target = "target"

    train_df = X_train.copy()
    train_df[target] = y_train

    test_df = X_test.copy()
    test_df[target] = y_test

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    train_h2o[target] = train_h2o[target].asfactor()
    test_h2o[target] = test_h2o[target].asfactor()

    features = list(X_train.columns)

    automl = H2OAutoML(
        max_runtime_secs=TIME_BUDGET_MINUTES * 60,
        nfolds=CV_FOLDS,
        seed=seed,
        verbosity="info",
    )

    start_time = time.perf_counter()

    automl.train(
        x=features,
        y=target,
        training_frame=train_h2o,
    )

    runtime = time.perf_counter() - start_time

    predictions = (
        automl.leader.predict(test_h2o)
        .as_data_frame(use_multi_thread=True)["predict"]
    )

    predictions = pd.to_numeric(predictions)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    leaderboard = automl.leaderboard.as_data_frame(
        use_multi_thread=True
    )

    result_dir = RESULTS_DIR / "h2o" / dataset_name / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(
        result_dir / "leaderboard.csv",
        index=False,
    )

    model_path = h2o.save_model(
        automl.leader,
        path=str(result_dir),
        force=True,
    )

    result = {
        "framework": "H2O AutoML",
        "dataset": dataset_name,
        "seed": seed,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "runtime_seconds": runtime,
        "best_model": automl.leader.model_id,
        "models_evaluated": len(leaderboard),
        "model_path": model_path,
    }

    pd.DataFrame([result]).to_csv(
        result_dir / "result.csv",
        index=False,
    )

    print(f"Best model: {automl.leader.model_id}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")

    return result


def run_all_h2o_experiments() -> pd.DataFrame:
    """
    Run H2O AutoML on every dataset and seed.
    """

    h2o.init(
        nthreads=2,
        max_mem_size="6G",
    )

    all_results = []

    try:
        for dataset in DATASET_NAMES:
            for seed in SEEDS:
                result = run_h2o_experiment(
                    dataset_name=dataset,
                    seed=seed,
                )

                all_results.append(result)

                h2o.remove_all()

        results_df = pd.DataFrame(all_results)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(
            RESULTS_DIR / "h2o_results.csv",
            index=False,
        )

        return results_df

    finally:
        h2o.cluster().shutdown(prompt=False)