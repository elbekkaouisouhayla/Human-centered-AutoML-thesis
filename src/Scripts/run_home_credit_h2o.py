from pathlib import Path
from time import perf_counter
import json

import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "Processed" / "home_credit" / "seed_42" / "train_raw.csv"
TEST_PATH = PROJECT_ROOT / "Processed" / "home_credit" / "seed_42" / "test_raw.csv"
RESULTS_DIR = PROJECT_ROOT / "Results" / "home_credit" / "h2o"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
SEED = 42
N_FOLDS = 5
MAX_RUNTIME_SECONDS = 1800
H2O_MAX_MEMORY = "8G"


def verify_input_files() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found:\n{TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found:\n{TEST_PATH}")


def to_pandas(frame):
    return frame.as_data_frame(use_multi_thread=False)


def main() -> None:
    print("=" * 70)
    print("HOME CREDIT — H2O AUTOML NATIVE AUTOMATION EXPERIMENT")
    print("=" * 70)

    verify_input_files()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting H2O server...")
    h2o.init(nthreads=-1, max_mem_size=H2O_MAX_MEMORY)
    h2o.remove_all()

    try:
        print("\nLoading raw datasets into H2O...")
        train_h2o = h2o.import_file(str(TRAIN_PATH))
        test_h2o = h2o.import_file(str(TEST_PATH))

        print(f"Training shape: ({train_h2o.nrows}, {train_h2o.ncols})")
        print(f"Test shape:     ({test_h2o.nrows}, {test_h2o.ncols})")

        for frame_name, frame in (("training", train_h2o), ("test", test_h2o)):
            if TARGET_COLUMN not in frame.columns:
                raise ValueError(f"{TARGET_COLUMN!r} is missing from the {frame_name} data.")

        train_h2o[TARGET_COLUMN] = train_h2o[TARGET_COLUMN].asfactor()
        test_h2o[TARGET_COLUMN] = test_h2o[TARGET_COLUMN].asfactor()

        excluded_columns = {TARGET_COLUMN, ID_COLUMN}
        feature_columns = [c for c in train_h2o.columns if c not in excluded_columns]
        categorical_columns = [c for c in feature_columns if train_h2o.types[c] == "enum"]
        numerical_columns = [c for c in feature_columns if train_h2o.types[c] != "enum"]

        print(f"\nFeatures used by H2O: {len(feature_columns)}")
        print(f"Categorical features: {len(categorical_columns)}")
        print(f"Numerical features:   {len(numerical_columns)}")

        automl = H2OAutoML(
            max_runtime_secs=MAX_RUNTIME_SECONDS,
            nfolds=N_FOLDS,
            seed=SEED,
            sort_metric="AUC",
            stopping_metric="AUC",
            stopping_rounds=3,
            balance_classes=False,
            keep_cross_validation_predictions=True,
            keep_cross_validation_models=False,
            keep_cross_validation_fold_assignment=False,
            project_name="home_credit_h2o_native",
        )

        print("\nStarting H2O AutoML optimization...")
        print(f"Maximum AutoML runtime: {MAX_RUNTIME_SECONDS / 60:.0f} minutes")
        print(f"Cross-validation folds: {N_FOLDS}")
        print("Optimization metric: AUC")

        training_start = perf_counter()
        automl.train(
            x=feature_columns,
            y=TARGET_COLUMN,
            training_frame=train_h2o,
        )
        training_runtime = perf_counter() - training_start

        print(f"\nH2O AutoML completed in {training_runtime:.2f} seconds.")
        leader = automl.leader
        if leader is None:
            raise RuntimeError("H2O AutoML finished without producing a leader model.")

        print(f"\nLeader model: {leader.model_id}")

        leaderboard_df = to_pandas(automl.leaderboard)
        leaderboard_df.to_csv(RESULTS_DIR / "h2o_leaderboard.csv", index=False)

        print("\nTop leaderboard models:")
        print(leaderboard_df.head(10).to_string(index=False))

        print("\nGenerating held-out test predictions...")
        prediction_start = perf_counter()
        predictions_h2o = leader.predict(test_h2o)
        prediction_runtime = perf_counter() - prediction_start

        predictions_df = to_pandas(predictions_h2o)
        actual_df = to_pandas(test_h2o[TARGET_COLUMN])

        actual_labels = actual_df[TARGET_COLUMN].astype(int).to_numpy()
        predicted_labels = predictions_df["predict"].astype(int).to_numpy()

        if "p1" not in predictions_df.columns:
            raise ValueError("Expected class-1 probabilities in column 'p1'.")

        positive_probabilities = predictions_df["p1"].astype(float).to_numpy()

        accuracy = accuracy_score(actual_labels, predicted_labels)
        macro_precision = precision_score(actual_labels, predicted_labels, average="macro", zero_division=0)
        macro_recall = recall_score(actual_labels, predicted_labels, average="macro", zero_division=0)
        macro_f1 = f1_score(actual_labels, predicted_labels, average="macro", zero_division=0)
        positive_precision = precision_score(actual_labels, predicted_labels, pos_label=1, zero_division=0)
        positive_recall = recall_score(actual_labels, predicted_labels, pos_label=1, zero_division=0)
        positive_f1 = f1_score(actual_labels, predicted_labels, pos_label=1, zero_division=0)
        roc_auc = roc_auc_score(actual_labels, positive_probabilities)

        confusion = confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])
        true_negatives = int(confusion[0, 0])
        false_positives = int(confusion[0, 1])
        false_negatives = int(confusion[1, 0])
        true_positives = int(confusion[1, 1])

        output_predictions = pd.DataFrame(
            {
                "actual": actual_labels,
                "predicted": predicted_labels,
                "probability_class_0": predictions_df["p0"].astype(float),
                "probability_class_1": positive_probabilities,
            }
        )

        if ID_COLUMN in test_h2o.columns:
            test_ids = to_pandas(test_h2o[ID_COLUMN])[ID_COLUMN].reset_index(drop=True)
            output_predictions.insert(0, ID_COLUMN, test_ids)

        output_predictions.to_csv(RESULTS_DIR / "h2o_predictions.csv", index=False)

        confusion_df = pd.DataFrame(
            confusion,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        confusion_df.to_csv(RESULTS_DIR / "h2o_confusion_matrix.csv")

        results_df = pd.DataFrame(
            [
                {
                    "framework": "H2O AutoML",
                    "framework_version": h2o.__version__,
                    "dataset": "Home Credit Default Risk",
                    "seed": SEED,
                    "folds": N_FOLDS,
                    "optimization_metric": "AUC",
                    "maximum_runtime_seconds": MAX_RUNTIME_SECONDS,
                    "train_rows": train_h2o.nrows,
                    "test_rows": test_h2o.nrows,
                    "number_of_features": len(feature_columns),
                    "number_of_categorical_features": len(categorical_columns),
                    "number_of_numerical_features": len(numerical_columns),
                    "leader_model": leader.model_id,
                    "training_runtime_seconds": training_runtime,
                    "prediction_runtime_seconds": prediction_runtime,
                    "total_runtime_seconds": training_runtime + prediction_runtime,
                    "accuracy": accuracy,
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                    "positive_class_precision": positive_precision,
                    "positive_class_recall": positive_recall,
                    "positive_class_f1": positive_f1,
                    "roc_auc": roc_auc,
                    "true_negatives": true_negatives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "true_positives": true_positives,
                }
            ]
        )
        results_df.to_csv(RESULTS_DIR / "h2o_results.csv", index=False)

        setup_summary = {
            "framework": "H2O AutoML",
            "framework_version": h2o.__version__,
            "dataset": "Home Credit Default Risk",
            "train_path": str(TRAIN_PATH),
            "test_path": str(TEST_PATH),
            "target": TARGET_COLUMN,
            "ignored_identifier": ID_COLUMN,
            "seed": SEED,
            "folds": N_FOLDS,
            "optimization_metric": "AUC",
            "runtime_limit_seconds": MAX_RUNTIME_SECONDS,
            "memory_limit": H2O_MAX_MEMORY,
            "native_missing_value_handling": True,
            "native_categorical_handling": True,
            "manual_preprocessing": False,
            "class_balancing": False,
            "leaderboard_basis": "cross-validation",
            "held_out_test_used_for_model_selection": False,
            "train_rows": train_h2o.nrows,
            "test_rows": test_h2o.nrows,
            "feature_count": len(feature_columns),
            "categorical_feature_count": len(categorical_columns),
            "numerical_feature_count": len(numerical_columns),
            "leader_model": leader.model_id,
        }

        with open(RESULTS_DIR / "h2o_setup_summary.json", "w", encoding="utf-8") as output_file:
            json.dump(setup_summary, output_file, indent=4)

        model_path = h2o.save_model(model=leader, path=str(RESULTS_DIR), force=True)
        (RESULTS_DIR / "h2o_leader_model_path.txt").write_text(model_path, encoding="utf-8")

        print("\n" + "=" * 70)
        print("FINAL HELD-OUT TEST RESULTS")
        print("=" * 70)
        print(results_df.T.to_string())
        print("\nConfusion matrix:")
        print(confusion_df.to_string())
        print("\nFiles saved in:")
        print(RESULTS_DIR)
        print("\nH2O experiment completed successfully.")

    finally:
        print("\nShutting down H2O...")
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception as shutdown_error:
            print(f"Warning during H2O shutdown: {shutdown_error}")


if __name__ == "__main__":
    main()