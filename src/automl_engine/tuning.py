"""Cross valdiation, model evalutation & tuning utilities"""

from collections.abc import Mapping
import time
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from .config import MAX_CV_FOLDS, RANDOM_STATE



# Cross Validatin Strategy
def create_cv_strategy(
        target: pd.Series,
        task_type: str,
        maximum_folds: int = MAX_CV_FOLDS,
        random_state: int  = RANDOM_STATE
) -> StratifiedKFold | KFold:

    """Create a task-appropriate cross-validation strategy."""

    if target.empty:
        raise ValueError(
            "The target cannot be empty."
        )

    if maximum_folds < 2: 
        raise ValueError(
            "maximum folds should be at least 2."
        )

    if task_type == "classification":
        minimum_class_count = int(
            target.value_counts().min()
        )

        number_of_folds = min(minimum_class_count, maximum_folds)

        if number_of_folds < 2:
            raise ValueError(
                "Cross-validation requires at least "
                "two rows in every target class."
            )

        return StratifiedKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=random_state,
        )

    if task_type == "regression":
        number_of_folds = min(
            maximum_folds, 
            len(target),
        )

        if number_of_folds < 2:
            raise ValueError(
                "Cross-validation requires at least "
                "two target values."
            )

        return KFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=random_state
        )


    raise ValueError(
        f"Unsupported task type: {task_type}"
    )



# Define Metrix configuration
def get_scoring_configuration(
        task_type: str,
        number_of_classes: int | None = None,
) -> dict[str, Any]:
    
    """Return metrics and selection rules for a task"""

    if task_type == "classification":
        if (
            number_of_classes is None
            or number_of_classes < 2
        ):
            raise ValueError(
                "Classification requires at least "
                "two target classes."
            )

        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": ("balanced_accuracy"),
            "precision_macro": (
                "precision_macro"
            ),
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
        }

        if number_of_classes == 2:
            scoring["roc_auc"] = "roc_auc"

        else: 
            scoring["roc_auc"] = (
                "roc_auc_ovr_weighted"
            )

        return {
            "scoring": scoring,
            "primary_metric": "f1_macro",
            "leaderboard_column": "F1 Macro",
            "higher_is_better": True,
        }


    if task_type == "regression":
        return {
            "scoring": {
                "mae": (
                    "neg_mean_absolute_error"
                ),
                "rmse": (
                    "neg_root_mean_squared_error"
                ),
                "r2": "r2",
            },
            "primary_metric": "rmse",
            "leaderboard_column": "RMSE",
            "higher_is_better": False,
        }

    raise ValueError(
        f"Unsupported task type: {task_type}"
    )




def evaluate_baseline_models(
        model_pipelines: Mapping[str, Pipeline],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_strategy: StratifiedKFold | KFold,
        scoring_configuration: Mapping[str, Any],
        task_type: str,
        n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """Evaluate candidate pipelines using cross validation"""

    if not model_pipelines:
        raise ValueError(
            "At least one model pipeline is required"
        )

    successful_results: list[
        dict[str, Any]
    ] = []

    failed_results: list[
        dict[str, str]
    ] = []

    scoring = scoring_configuration["scoring"]


    for model_name, pipeline in model_pipelines.items():
        print(f"Evaluating {model_name}...")

        start_time = time.perf_counter()

        try:
            scores = cross_validate(
                estimator=pipeline,
                X = X_train,
                y = y_train,
                cv = cv_strategy,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=False,
                error_score="raise"
            )


            elapsed_time = (
                time.perf_counter() - start_time
            )

            if task_type == "classification":
                result: dict[str, Any] = {
                    "Model": model_name,

                    "Accuracy": float(
                        scores[
                            "test_accuracy"
                        ].mean()
                    ),

                    "Balanced Accuracy": float(
                        scores[
                            "test_balanced_accuracy"
                        ].mean()
                    ),

                    "Precision Macro": float(
                        scores[
                            "test_precision_macro"
                        ].mean()
                    ),

                    "Recall Macro": float(
                        scores[
                            "test_recall_macro"
                        ].mean()
                    ),

                    "F1 Macro": float(
                        scores[
                            "test_f1_macro"
                        ].mean()
                    ),

                    "F1 Macro Std": float(
                        scores[
                            "test_f1_macro"
                        ].std()
                    ),

                    "Training Time": float(
                        elapsed_time
                    ),
                }

                if "roc_auc" in scoring:
                    result["ROC-AUC"] = float(
                        scores[
                            "test_roc_auc"
                        ].mean()
                    )

            elif task_type == "regression":
                result = {
                    "Model": model_name,

                    "MAE": float(
                        -scores[
                            "test_mae"
                        ].mean()
                    ),

                    "RMSE": float(
                        -scores[
                            "test_rmse"
                        ].mean()
                    ),

                    "R2": float(
                        scores[
                            "test_r2"
                        ].mean()
                    ),

                    "RMSE Std": float(
                        scores[
                            "test_rmse"
                        ].std()
                    ),

                    "Training Time": float(
                        elapsed_time
                    ),
                }

            else:
                raise ValueError(
                    "task_type must be "
                    "'classification' or "
                    "'regression'."
                )


            successful_results.append(result)

        except Exception as error:
            failed_results.append(
                {
                    "Model": model_name,
                    "Error": str(error),
                }
            )

            print(
                f"Failed: {model_name} — "
                f"{error}"
            )


    leaderboard = pd.DataFrame(
        successful_results
    )

    failures = pd.DataFrame(
        failed_results
    )

    if leaderboard.empty:
        raise RuntimeError(
            "All baseline models failed."
        )


    if task_type == "classification":
        leaderboard = (
            leaderboard.sort_values(
                by=["F1 Macro", "F1 Macro Std", "Training Time"],
                ascending=[False, True, True],
            )
        )

    elif task_type == "regression":
        leaderboard = (
            leaderboard.sort_values(
                by=[
                    "RMSE",
                    "RMSE Std",
                    "Training Time",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ],
            )
        )


    leaderboard = (
        leaderboard.reset_index(
            drop=True
        )
    )

    numeric_columns = (
        leaderboard
        .select_dtypes(include="number")
        .columns
    )

    leaderboard[numeric_columns] = leaderboard[numeric_columns].round(4)

    return leaderboard, failures



def select_top_models(
        leaderboard: pd.DataFrame,
        number_of_models: int = 2,
) -> list[str]:

    """Select the strongest non-dummy models"""

    if leaderboard.empty:
        raise ValueError(
            "The leaderboard cannot be empty."
        )

    if number_of_models < 1:
        raise ValueError(
            "number_of_models must be at least 1."
        )

    if "Model" not in leaderboard.columns:
        raise ValueError(
            "The leaderboard must contain "
            "a 'Model' column."
        )


    exluded_models = {
        "Dummy Classifier",
        "Dummy Regressor",
    }

    eligible_models = leaderboard[~leaderboard["Model"].isin(exluded_models)]

    if eligible_models.empty:
        raise RuntimeError(
            "No tunable models are available."
        )

    return (
        eligible_models.head(number_of_models)["Model"].tolist()
    )
    