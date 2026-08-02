from pathlib import Path
from time import perf_counter
import pickle
import traceback
import warnings

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tpot import TPOTClassifier


# ============================================================
# CONFIGURATION
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
)

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
    / "tpot"
)

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

SEED = 42
N_FOLDS = 3

MAX_TIME_MINUTES = 60
MAX_EVALUATION_MINUTES = 20
POPULATION_SIZE = 3


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def verify_input_files() -> None:
    """Vérifie que les fichiers train et test existent."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_PATH}"
        )


def get_positive_class_probability(
    model: TPOTClassifier,
    X_test: pd.DataFrame,
):
    """Retourne la probabilité prédite pour la classe 1."""

    try:
        probabilities = model.predict_proba(X_test)

        if (
            probabilities.ndim == 2
            and probabilities.shape[1] >= 2
        ):
            return probabilities[:, 1]

    except Exception as error:
        print("\nWarning: predict_proba failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")

    return None


def save_selected_pipeline(
    selected_pipeline,
) -> str | None:
    """Enregistre le pipeline sélectionné lorsque disponible."""

    if selected_pipeline is None:
        return None

    selected_pipeline_text = str(selected_pipeline)

    (
        RESULTS_DIR
        / "tpot_selected_pipeline.txt"
    ).write_text(
        selected_pipeline_text,
        encoding="utf-8",
    )

    try:
        with open(
            RESULTS_DIR / "tpot_fitted_pipeline.pkl",
            "wb",
        ) as model_file:
            pickle.dump(
                selected_pipeline,
                model_file,
            )

    except Exception as error:
        print("\nWarning: pipeline pickle could not be saved.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")

    return selected_pipeline_text


def save_evaluated_pipelines(
    tpot: TPOTClassifier,
) -> None:
    """Enregistre les pipelines évalués lorsque disponibles."""

    possible_attributes = [
        "evaluated_individuals",
        "evaluated_individuals_",
    ]

    evaluated_individuals = None

    for attribute in possible_attributes:
        value = getattr(
            tpot,
            attribute,
            None,
        )

        if value is not None:
            evaluated_individuals = value
            break

    if evaluated_individuals is None:
        return

    try:
        if hasattr(evaluated_individuals, "to_csv"):
            evaluated_individuals.to_csv(
                RESULTS_DIR
                / "tpot_evaluated_pipelines.csv",
                index=False,
            )
        else:
            pd.DataFrame(
                evaluated_individuals
            ).to_csv(
                RESULTS_DIR
                / "tpot_evaluated_pipelines.csv",
                index=False,
            )

    except Exception as error:
        print(
            "\nWarning: evaluated pipelines "
            "could not be saved."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")


def save_failure_result(
    *,
    stage: str,
    error: Exception,
    runtime_seconds: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    selected_pipeline_text: str | None,
) -> None:
    """Enregistre une preuve complète de l’échec."""

    traceback_text = traceback.format_exc()

    raw_columns_identical = (
        list(X_train.columns)
        == list(X_test.columns)
    )

    failure_df = pd.DataFrame(
        [
            {
                "framework": "TPOT",
                "framework_version": "1.1.0",
                "dataset": "Home Credit Default Risk",
                "status": "Failed",
                "failure_stage": stage,
                "seed": SEED,
                "folds": N_FOLDS,
                "optimization_metric": "ROC AUC",
                "search_space": "linear",
                "population_size": POPULATION_SIZE,
                "initial_population_size": POPULATION_SIZE,
                "maximum_runtime_minutes": (
                    MAX_TIME_MINUTES
                ),
                "maximum_pipeline_evaluation_minutes": (
                    MAX_EVALUATION_MINUTES
                ),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "number_of_raw_features_train": (
                    X_train.shape[1]
                ),
                "number_of_raw_features_test": (
                    X_test.shape[1]
                ),
                "raw_train_test_columns_identical": (
                    raw_columns_identical
                ),
                "number_of_categorical_features": (
                    len(categorical_columns)
                ),
                "number_of_numerical_features": (
                    len(numerical_columns)
                ),
                "runtime_before_failure_seconds": (
                    runtime_seconds
                ),
                "native_preprocessing_requested": True,
                "workers": 1,
                "processes": False,
                "final_pipeline_selected": (
                    selected_pipeline_text is not None
                ),
                "selected_pipeline": (
                    selected_pipeline_text
                    if selected_pipeline_text is not None
                    else ""
                ),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "checkpoint_directory": str(
                    CHECKPOINT_DIR
                ),
            }
        ]
    )

    failure_df.to_csv(
        RESULTS_DIR / "tpot_failure_results.csv",
        index=False,
    )

    report = f"""
HOME CREDIT — TPOT FAILURE REPORT
=================================

Framework
---------
Framework: TPOT
Version: 1.1.0
Dataset: Home Credit Default Risk
Status: Failed
Failure stage: {stage}

Configuration
-------------
Seed: {SEED}
Cross-validation folds: {N_FOLDS}
Optimization metric: ROC AUC
Search space: linear
Population size: {POPULATION_SIZE}
Maximum total runtime: {MAX_TIME_MINUTES} minutes
Maximum evaluation time per pipeline: {MAX_EVALUATION_MINUTES} minutes
Workers: 1
Processes: False
Native preprocessing requested: True

Dataset
-------
Training rows: {len(X_train)}
Test rows: {len(X_test)}
Raw training features: {X_train.shape[1]}
Raw test features: {X_test.shape[1]}
Raw train/test columns identical: {raw_columns_identical}
Categorical features: {len(categorical_columns)}
Numerical features: {len(numerical_columns)}

Failure
-------
Runtime before failure: {runtime_seconds:.2f} seconds
Error type: {type(error).__name__}
Error message: {str(error)}

Selected pipeline
-----------------
{selected_pipeline_text or "No final pipeline was selected."}

Checkpoint directory
--------------------
{CHECKPOINT_DIR}

Complete traceback
------------------
{traceback_text}
""".strip()

    (
        RESULTS_DIR
        / "tpot_failure_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("TPOT EXPERIMENT FAILED")
    print("=" * 70)

    print(f"Failure stage: {stage}")
    print(
        f"Runtime before failure: "
        f"{runtime_seconds:.2f} seconds"
    )
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {error}")

    print("\nFailure evidence saved in:")
    print(RESULTS_DIR)


# ============================================================
# EXPÉRIENCE PRINCIPALE
# ============================================================

def main() -> None:
    print("=" * 70)
    print("HOME CREDIT — TPOT NATIVE AUTOMATION EXPERIMENT")
    print("=" * 70)

    verify_input_files()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Charger les données
    # --------------------------------------------------------

    print("\nLoading raw datasets...")

    train_df = pd.read_csv(
        TRAIN_PATH,
        low_memory=False,
    ).reset_index(drop=True)

    test_df = pd.read_csv(
        TEST_PATH,
        low_memory=False,
    ).reset_index(drop=True)

    print(f"Training shape: {train_df.shape}")
    print(f"Test shape:     {test_df.shape}")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' is missing "
            "from train_raw.csv."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' is missing "
            "from test_raw.csv."
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

    # --------------------------------------------------------
    # Séparer variables et cible
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = (
        train_df[TARGET_COLUMN]
        .astype(int)
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = (
        test_df[TARGET_COLUMN]
        .astype(int)
    )

    test_ids = None

    if ID_COLUMN in X_train.columns:
        X_train = X_train.drop(
            columns=[ID_COLUMN]
        )

    if ID_COLUMN in X_test.columns:
        test_ids = (
            X_test[ID_COLUMN]
            .copy()
            .reset_index(drop=True)
        )

        X_test = X_test.drop(
            columns=[ID_COLUMN]
        )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Assurer le même ordre des colonnes brutes.
    missing_test_columns = [
        column
        for column in X_train.columns
        if column not in X_test.columns
    ]

    extra_test_columns = [
        column
        for column in X_test.columns
        if column not in X_train.columns
    ]

    if missing_test_columns or extra_test_columns:
        raise ValueError(
            "Raw train/test feature columns do not match.\n"
            f"Missing in test: {missing_test_columns}\n"
            f"Extra in test: {extra_test_columns}"
        )

    X_test = X_test.reindex(
        columns=X_train.columns
    )

    raw_columns_identical = (
        list(X_train.columns)
        == list(X_test.columns)
    )

    print(
        "\nRaw train/test columns identical:",
        raw_columns_identical,
    )

    print(
        "Raw training feature count:",
        X_train.shape[1],
    )

    print(
        "Raw test feature count:",
        X_test.shape[1],
    )

    # --------------------------------------------------------
    # Identifier les types de variables
    # --------------------------------------------------------

    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numerical_columns = (
        X_train
        .select_dtypes(
            exclude=["object", "category"]
        )
        .columns
        .tolist()
    )

    print(
        f"\nCategorical features: "
        f"{len(categorical_columns)}"
    )

    print(
        f"Numerical features: "
        f"{len(numerical_columns)}"
    )

    # --------------------------------------------------------
    # Enregistrer la configuration
    # --------------------------------------------------------

    setup_df = pd.DataFrame(
        [
            {
                "framework": "TPOT",
                "framework_version": "1.1.0",
                "dataset": "Home Credit Default Risk",
                "target": TARGET_COLUMN,
                "ignored_identifier": ID_COLUMN,
                "seed": SEED,
                "folds": N_FOLDS,
                "search_space": "linear",
                "optimization_metric": "ROC AUC",
                "native_preprocessing": True,
                "population_size": POPULATION_SIZE,
                "initial_population_size": (
                    POPULATION_SIZE
                ),
                "maximum_runtime_minutes": (
                    MAX_TIME_MINUTES
                ),
                "maximum_evaluation_minutes": (
                    MAX_EVALUATION_MINUTES
                ),
                "workers": 1,
                "processes": False,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "number_of_features": (
                    X_train.shape[1]
                ),
                "number_of_categorical_features": (
                    len(categorical_columns)
                ),
                "number_of_numerical_features": (
                    len(numerical_columns)
                ),
                "raw_train_test_columns_identical": (
                    raw_columns_identical
                ),
            }
        ]
    )

    setup_df.to_csv(
        RESULTS_DIR / "tpot_setup_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Configurer TPOT
    # --------------------------------------------------------

    print("\nCreating TPOT classifier...")

    tpot = TPOTClassifier(
        search_space="linear",

        scorers=["roc_auc"],
        scorers_weights=[1],

        cv=N_FOLDS,

        preprocessing=True,

        categorical_features=categorical_columns,

        population_size=POPULATION_SIZE,
        initial_population_size=POPULATION_SIZE,

        max_time_mins=MAX_TIME_MINUTES,
        max_eval_time_mins=MAX_EVALUATION_MINUTES,

        n_jobs=1,
        processes=False,

        memory="auto",

        periodic_checkpoint_folder=str(
            CHECKPOINT_DIR
        ),

        verbose=5,
        random_state=SEED,
        validation_strategy="none",
    )

    print("\nStarting TPOT optimization...")
    print(
        f"Maximum optimization time: "
        f"{MAX_TIME_MINUTES} minutes"
    )
    print(
        f"Maximum evaluation time per pipeline: "
        f"{MAX_EVALUATION_MINUTES} minutes"
    )
    print(
        f"Population size: "
        f"{POPULATION_SIZE}"
    )
    print(
        f"Cross-validation folds: "
        f"{N_FOLDS}"
    )

    # --------------------------------------------------------
    # Lancer la recherche
    # --------------------------------------------------------

    training_start = perf_counter()

    try:
        tpot.fit(
            X_train,
            y_train,
        )

    except Exception as error:
        training_runtime = (
            perf_counter() - training_start
        )

        save_evaluated_pipelines(tpot)

        selected_pipeline = getattr(
            tpot,
            "fitted_pipeline_",
            None,
        )

        selected_pipeline_text = (
            save_selected_pipeline(
                selected_pipeline
            )
        )

        save_failure_result(
            stage="Model search and training",
            error=error,
            runtime_seconds=training_runtime,
            X_train=X_train,
            X_test=X_test,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            selected_pipeline_text=(
                selected_pipeline_text
            ),
        )

        return

    training_runtime = (
        perf_counter() - training_start
    )

    print(
        f"\nTPOT optimization completed in "
        f"{training_runtime:.2f} seconds."
    )

    save_evaluated_pipelines(tpot)

    # --------------------------------------------------------
    # Récupérer le pipeline sélectionné
    # --------------------------------------------------------

    selected_pipeline = getattr(
        tpot,
        "fitted_pipeline_",
        None,
    )

    if selected_pipeline is None:
        error = RuntimeError(
            "TPOT completed without producing "
            "a fitted pipeline."
        )

        save_failure_result(
            stage="Final pipeline retrieval",
            error=error,
            runtime_seconds=training_runtime,
            X_train=X_train,
            X_test=X_test,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            selected_pipeline_text=None,
        )

        return

    selected_pipeline_text = save_selected_pipeline(
        selected_pipeline
    )

    print("\nSelected pipeline:")
    print(selected_pipeline_text)

    # --------------------------------------------------------
    # Prédire sur le test set
    # --------------------------------------------------------

    print("\nGenerating held-out test predictions...")

    prediction_start = perf_counter()

    try:
        predicted_labels = tpot.predict(
            X_test
        )

        positive_probabilities = (
            get_positive_class_probability(
                tpot,
                X_test,
            )
        )

    except Exception as error:
        total_runtime = (
            perf_counter() - training_start
        )

        save_failure_result(
            stage="Held-out test prediction",
            error=error,
            runtime_seconds=total_runtime,
            X_train=X_train,
            X_test=X_test,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            selected_pipeline_text=(
                selected_pipeline_text
            ),
        )

        return

    prediction_runtime = (
        perf_counter() - prediction_start
    )

    # --------------------------------------------------------
    # Calculer les métriques
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

    positive_precision = precision_score(
        y_test,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    positive_recall = recall_score(
        y_test,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    positive_f1 = f1_score(
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
        training_runtime
        + prediction_runtime
    )

    # --------------------------------------------------------
    # Enregistrer les résultats
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            {
                "framework": "TPOT",
                "framework_version": "1.1.0",
                "dataset": "Home Credit Default Risk",
                "status": "Completed",
                "seed": SEED,
                "folds": N_FOLDS,
                "optimization_metric": "ROC AUC",
                "search_space": "linear",
                "population_size": (
                    POPULATION_SIZE
                ),
                "maximum_runtime_minutes": (
                    MAX_TIME_MINUTES
                ),
                "maximum_pipeline_evaluation_minutes": (
                    MAX_EVALUATION_MINUTES
                ),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "number_of_features": (
                    X_train.shape[1]
                ),
                "number_of_categorical_features": (
                    len(categorical_columns)
                ),
                "number_of_numerical_features": (
                    len(numerical_columns)
                ),
                "training_runtime_seconds": (
                    training_runtime
                ),
                "prediction_runtime_seconds": (
                    prediction_runtime
                ),
                "total_runtime_seconds": (
                    total_runtime
                ),
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "positive_class_precision": (
                    positive_precision
                ),
                "positive_class_recall": (
                    positive_recall
                ),
                "positive_class_f1": (
                    positive_f1
                ),
                "roc_auc": roc_auc,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positives": true_positives,
                "selected_pipeline": (
                    selected_pipeline_text
                ),
            }
        ]
    )

    results_df.to_csv(
        RESULTS_DIR / "tpot_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Enregistrer les prédictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": predicted_labels,
        }
    )

    if test_ids is not None:
        predictions_df.insert(
            0,
            ID_COLUMN,
            test_ids,
        )

    if positive_probabilities is not None:
        predictions_df[
            "probability_class_1"
        ] = positive_probabilities

    predictions_df.to_csv(
        RESULTS_DIR / "tpot_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Enregistrer la matrice de confusion
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        confusion,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )

    confusion_df.to_csv(
        RESULTS_DIR
        / "tpot_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Affichage final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 70)

    print(results_df.T.to_string())

    print("\nConfusion matrix:")
    print(confusion_df.to_string())

    print("\nFiles saved in:")
    print(RESULTS_DIR)

    print(
        "\nTPOT experiment completed successfully."
    )


if __name__ == "__main__":
    main()