"""Cross valdiation, model evalutation & tuning utilities"""

from collections.abc import Mapping, Sequence
import time
from typing import Any, cast

import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline

from .config import MAX_CV_FOLDS, RANDOM_STATE, SUPPORTED_TUNING_MODES

from sklearn.base import clone

from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)



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

            "balanced_accuracy": (
                "balanced_accuracy"
            ),

            "precision_macro": make_scorer(
                precision_score,
                average="macro",
                zero_division=0,
            ),

            "recall_macro": make_scorer(
                recall_score,
                average="macro",
                zero_division=0,
            ),

            "f1_macro": make_scorer(
                f1_score,
                average="macro",
                zero_division=0,
            ),
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



def get_tuning_budget(
        tuning_mode: str,
) -> dict[str, int]:

    """Return model_selection and search budgets."""

    budgets = {
        "fast": {
            "number_of_models": 1,
            "n_iter": 8,
        },

        "balanced": {
            "number_of_models": 2,
            "n_iter": 20,
        },

        "thorough": {
            "number_of_models": 3,
            "n_iter": 40
        }
    }

    if tuning_mode not in SUPPORTED_TUNING_MODES:
        raise ValueError(
            f"Unsupported tuning mode: {tuning_mode}."
            f"Choose from: "
            f"{sorted(SUPPORTED_TUNING_MODES)}"
        )

    return budgets[tuning_mode].copy()




# Count possible parameter combinations

def count_parameter_combinations(
        parameter_space: Mapping[
            str,
            Sequence[Any],
        ],
) -> int:

    """Count combinations in a discrete paramter space."""

    if not parameter_space:
        raise ValueError(
            "The paramter space cannot be empty."
        )

    total_combinations = 1

    for parameter_name, values in parameter_space.items():
        if not parameter_name.strip():
            raise ValueError(
                "Parameter names cannot be empty."
            )

        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"Values for '{parameter_name}' "
                "must be a sequence, not a string."
            )

        if not isinstance(values, Sequence):
            raise TypeError(
                f"Values for '{parameter_name}' "
                "must be a discrete sequence."
            )

        if len(values) == 0:
            raise ValueError(
                f"Parameter '{parameter_name}' "
                "has no candidate values."
            )

        total_combinations *= len(values)

    return total_combinations



def tune_selected_models(
    selected_models: Sequence[str],
    model_pipelines: Mapping[str, Pipeline],
    parameter_spaces: Mapping[
        str,
        Mapping[str, Sequence[Any]],
    ],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: StratifiedKFold | KFold,
    scoring_configuration: Mapping[str, Any],
    task_type: str,
    n_iter: int,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,

) -> tuple[
    pd.DataFrame,
    dict[str, Pipeline],
    pd.DataFrame,
]:

    
    """Tune selected models using randomized search."""

    if not selected_models:
        raise ValueError(
            "At least one model must be selected "
            "for tuning."
        )
    

    if n_iter < 1:
        raise ValueError(
            "n_iter must be at least 1."
        )
    

    if task_type not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )
    

    scoring = scoring_configuration[
        "scoring"
    ]

    primary_metric = scoring_configuration[
        "primary_metric"
    ]

    if primary_metric not in scoring:
        raise ValueError(
            "The primary metric must exist in "
            "the scoring configuration."
        )



    successful_results: list[
        dict[str, Any]
    ] = []

    tuned_models: dict[
        str,
        Pipeline,
    ] = {}

    failed_results: list[
        dict[str, str]
    ] = []


    # Unlike baseline evaluation, the tuned models dictionary stores the actual fitted pipelines.



    for model_name in selected_models:
        print(
            f"Tuning {model_name}..."
        )

        start_time = time.perf_counter()

        try:
            if model_name not in model_pipelines:
                raise KeyError(
                    f"No pipeline found for "
                    f"'{model_name}'."
                )

            if model_name not in parameter_spaces:
                raise KeyError(
                    f"No parameter space found for "
                    f"'{model_name}'."
                )

            parameter_space = (
                parameter_spaces[model_name]
            )

            total_combinations = (
                count_parameter_combinations(
                    parameter_space
                )
            )

            actual_iterations = min(
                n_iter,
                total_combinations,
            )


            search = RandomizedSearchCV(
                estimator=clone(
                    model_pipelines[
                        model_name
                    ]
                ),
                param_distributions=dict(
                    parameter_space
                ),
                n_iter=actual_iterations,
                scoring=scoring,
                refit=primary_metric,
                cv=cv_strategy,
                random_state=random_state,
                n_jobs=n_jobs,
                return_train_score=False,
                error_score="raise",
            )

            search.fit(
                X_train,
                y_train,
            )


            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            best_index = int(search.best_index_)

            cv_results = search.cv_results_

            # best_index_ points to the strongest tested parameter combination



            if task_type == "classification":
                result: dict[str, Any] = {
                    "Model": model_name,

                    "Accuracy": float(
                        cv_results[
                            "mean_test_accuracy"
                        ][best_index]
                    ),

                    "Balanced Accuracy": float(
                        cv_results[
                            "mean_test_balanced_accuracy"
                        ][best_index]
                    ),

                    "Precision Macro": float(
                        cv_results[
                            "mean_test_precision_macro"
                        ][best_index]
                    ),

                    "Recall Macro": float(
                        cv_results[
                            "mean_test_recall_macro"
                        ][best_index]
                    ),

                    "F1 Macro": float(
                        cv_results[
                            "mean_test_f1_macro"
                        ][best_index]
                    ),

                    "F1 Macro Std": float(
                        cv_results[
                            "std_test_f1_macro"
                        ][best_index]
                    ),
                }

                if (
                    "mean_test_roc_auc"
                    in cv_results
                ):
                    result["ROC-AUC"] = float(
                        cv_results[
                            "mean_test_roc_auc"
                        ][best_index]
                    )

            else:
                result = {
                    "Model": model_name,

                    "MAE": float(
                        -cv_results[
                            "mean_test_mae"
                        ][best_index]
                    ),

                    "RMSE": float(
                        -cv_results[
                            "mean_test_rmse"
                        ][best_index]
                    ),

                    "R2": float(
                        cv_results[
                            "mean_test_r2"
                        ][best_index]
                    ),

                    "RMSE Std": float(
                        cv_results[
                            "std_test_rmse"
                        ][best_index]
                    ),
                }


            result[
                "Tuning Iterations"
            ] = actual_iterations

            result[
                "Tuning Time"
            ] = float(elapsed_time)

            result[
                "Best Parameters"
            ] = search.best_params_


            successful_results.append(
                result
            )

            tuned_models[model_name] = cast(
                Pipeline,
                search.best_estimator_,
            )


        except Exception as error:
            failed_results.append(
                {
                    "Model": model_name,
                    "Error": str(error),
                }
            )

            print(
                f"Tuning failed: {model_name} — "
                f"{error}"
            )



    tuned_leaderboard = pd.DataFrame(successful_results)

    tuning_failures = pd.DataFrame(failed_results)


    if not tuned_leaderboard.empty:
        if task_type == "classification":
            tuned_leaderboard = (
                tuned_leaderboard.sort_values(
                    by=[
                        "F1 Macro",
                        "F1 Macro Std",
                        "Tuning Time",
                    ],
                    ascending=[
                        False,
                        True,
                        True,
                    ],
                )
            )

        else:
            tuned_leaderboard = (
                tuned_leaderboard.sort_values(
                    by=[
                        "RMSE",
                        "RMSE Std",
                        "Tuning Time",
                    ],
                    ascending=[
                        True,
                        True,
                        True,
                    ],
                )
            )

        tuned_leaderboard = (
            tuned_leaderboard.reset_index(
                drop=True
            )
        )

        numeric_columns = (
            tuned_leaderboard
            .select_dtypes(
                include="number"
            )
            .columns
        )

        tuned_leaderboard[
            numeric_columns
        ] = tuned_leaderboard[
            numeric_columns
        ].round(4)


    return (
        tuned_leaderboard,
        tuned_models,
        tuning_failures,
    )




# Select final model

def select_final_model(
    baseline_leaderboard: pd.DataFrame,
    model_pipelines: Mapping[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring_configuration: Mapping[str, Any],
    tuned_leaderboard: pd.DataFrame | None = None,
    tuned_models: Mapping[
        str,
        Pipeline,
    ] | None = None,
) -> tuple[str, Pipeline, str]:
    
    """Choose and return the final fitted pipeline."""


    if baseline_leaderboard.empty:
        raise ValueError(
            "The baseline leaderboard cannot "
            "be empty."
        )

    leaderboard_column = (
        scoring_configuration[
            "leaderboard_column"
        ]
    )

    higher_is_better = bool(
        scoring_configuration[
            "higher_is_better"
        ]
    )

    required_columns = {
        "Model",
        leaderboard_column,
    }

    if not required_columns.issubset(
        baseline_leaderboard.columns
    ):
        raise ValueError(
            "The baseline leaderboard is "
            "missing required columns."
        )



    baseline_ranked = (
        baseline_leaderboard.sort_values(
            by=leaderboard_column,
            ascending=not higher_is_better,
        )
    )
    
    baseline_row = baseline_ranked.iloc[0]
    
    baseline_model_name = str(
        baseline_row["Model"]
    )
    
    baseline_score = float(
        baseline_row[
            leaderboard_column
        ]
    )  


    tuning_is_available = (
        tuned_leaderboard is not None
        and not tuned_leaderboard.empty
        and tuned_models is not None
    )

    if tuning_is_available:
        if not required_columns.issubset(
            tuned_leaderboard.columns
        ):
            raise ValueError(
                "The tuned leaderboard is "
                "missing required columns."
            )

        tuned_ranked = (
            tuned_leaderboard.sort_values(
                by=leaderboard_column,
                ascending=not higher_is_better,
            )
        )

        tuned_row = tuned_ranked.iloc[0]

        tuned_model_name = str(
            tuned_row["Model"]
        )

        tuned_score = float(
            tuned_row[
                leaderboard_column
            ]
        )

        if higher_is_better:
            tuning_improved_score = (
                tuned_score > baseline_score
            )

        else:
            tuning_improved_score = (
                tuned_score < baseline_score
            )


        if tuning_improved_score:
            if tuned_model_name not in tuned_models:
                raise KeyError(
                    f"No fitted tuned pipeline found "
                    f"for '{tuned_model_name}'."
                )

            return (
                tuned_model_name,
                tuned_models[
                    tuned_model_name
                ],
                "tuned",
            )


    if baseline_model_name not in model_pipelines:
        raise KeyError(
            f"No baseline pipeline found for "
            f"'{baseline_model_name}'."
        )


    final_pipeline = clone(
        model_pipelines[
            baseline_model_name
        ]
    )


    final_pipeline.fit(
        X_train,
        y_train,
    )


    return (
        baseline_model_name,
        final_pipeline,
        "baseline",
    )