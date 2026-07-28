"""Main controller for the tabular AutoML workflow."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .artifacts import (
    create_run_directory,
    save_run_artifacts,
)
from .config import (
    MAX_CV_FOLDS,
    RANDOM_STATE,
    SUPPORTED_TASKS,
    SUPPORTED_TUNING_MODES,
    TEST_SIZE,
)
from .data import (
    clean_dataset,
    detect_feature_types,
    detect_task_type,
    inspect_dataset,
    load_dataset,
    prepare_target,
    separate_features_and_target,
    split_dataset,
)
from .evaluation import (
    EvaluationResult,
    evaluate_final_model,
)
from .models import (
    create_model_pipelines,
    get_candidate_models,
    get_parameter_spaces,
)
from .preprocessing import build_preprocessor
from .tuning import (
    create_cv_strategy,
    evaluate_baseline_models,
    get_scoring_configuration,
    get_tuning_budget,
    select_final_model,
    select_top_models,
    tune_selected_models,
)



@dataclass(frozen=True)
class AutoMLRunResult:
    """Store the important outputs from one AutoML run."""

    run_directory: Path
    task_type: str
    final_model_name: str
    selection_source: str
    metrics: dict[str, float]
    cleaning_actions: tuple[str, ...]
    dataset_summary: dict[str, Any]
    input_schema: dict[str, Any]
    baseline_leaderboard: pd.DataFrame
    tuned_leaderboard: pd.DataFrame
    evaluation_result: EvaluationResult
    model_pipeline: Pipeline
    target_encoder: LabelEncoder | None
    artifacts: dict[str, Path]


def _resolve_task_type(
    requested_task_type: str,
    target: pd.Series,
) -> str:
    """Validate or automatically detect the task type."""

    if requested_task_type not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task type: "
            f"{requested_task_type}. "
            f"Choose from: "
            f"{sorted(SUPPORTED_TASKS)}"
        )

    if requested_task_type == "auto":
        return detect_task_type(target)

    return requested_task_type  



def _filter_candidate_models(
    models: Mapping[str, BaseEstimator],
    include_models: Sequence[str] | None,
) -> dict[str, BaseEstimator]:
    """Optionally keep only user-selected models."""

    if include_models is None:
        return dict(models)

    requested_models = list(
        dict.fromkeys(include_models)
    )

    if not requested_models:
        raise ValueError(
            "include_models cannot be empty."
        )

    missing_models = [
        model_name
        for model_name in requested_models
        if model_name not in models
    ]

    if missing_models:
        raise ValueError(
            "Unknown or unavailable models: "
            f"{missing_models}. "
            f"Available models: "
            f"{sorted(models)}"
        )

    return {
        model_name: models[model_name]
        for model_name in requested_models
    }



def _build_input_schema(
    features: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> dict[str, Any]:
    """Describe the feature structure expected during prediction."""

    return {
        "feature_columns": (
            features.columns.tolist()
        ),
        "numerical_columns": list(
            numerical_columns
        ),
        "categorical_columns": list(
            categorical_columns
        ),
        "feature_dtypes": {
            column: str(data_type)
            for column, data_type
            in features.dtypes.items()
        },
    }


def run_automl(
    csv_path: str | Path,
    target_column: str,
    *,
    task_type: str = "auto",
    tuning_mode: str = "balanced",
    columns_to_drop: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    output_directory: str | Path = "automl_runs",
    run_name: str | None = None,
    test_size: float = TEST_SIZE,
    maximum_cv_folds: int = MAX_CV_FOLDS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,
) -> AutoMLRunResult:
    
    """Run the complete tabular AutoML workflow."""

    # Validate the configuration
    if not target_column.strip():
        raise ValueError(
            "target_column cannot be empty."
        )

    if tuning_mode not in SUPPORTED_TUNING_MODES:
        raise ValueError(
            f"Unsupported tuning mode: "
            f"{tuning_mode}. "
            f"Choose from: "
            f"{sorted(SUPPORTED_TUNING_MODES)}"
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if maximum_cv_folds < 2:
        raise ValueError(
            "maximum_cv_folds must be at least 2."
        )

    if n_jobs == 0:
        raise ValueError(
            "n_jobs cannot be zero."
        )


    # Loading and inspecting the data
    dataframe = load_dataset(csv_path)

    (
        dataset_summary,
        column_report,
    ) = inspect_dataset(
        dataframe=dataframe,
        target_column=target_column,
    )


    # Cleaning the data
    (
        cleaned_dataframe,
        cleaning_actions,
    ) = clean_dataset(
        dataframe=dataframe,
        target_column=target_column,
        columns_to_drop=(
            list(columns_to_drop)
            if columns_to_drop is not None
            else None
        ),
    )

    (
        features,
        raw_target,
    ) = separate_features_and_target(
        dataframe=cleaned_dataframe,
        target_column=target_column,
    )


    # Detect and prepare the target
    resolved_task_type = (
        _resolve_task_type(
            requested_task_type=task_type,
            target=raw_target,
        )
    )

    (
        prepared_target,
        target_encoder,
        target_information,
    ) = prepare_target(
        target=raw_target,
        task_type=resolved_task_type,
    )

    # Detect feature types
    (
        numerical_columns,
        categorical_columns,
    ) = detect_feature_types(
        features
    )

    # Split the data
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        features=features,
        target=prepared_target,
        task_type=resolved_task_type,
        test_size=test_size,
        random_state=random_state,
    )



    # Build preprocessing
    preprocessor = build_preprocessor(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )


    # Build candidate models
    number_of_classes: int | None = None

    if resolved_task_type == "classification":
        number_of_classes = int(
            prepared_target.nunique()
        )

    candidate_models = (
        get_candidate_models(
            task_type=resolved_task_type,
            number_of_classes=(
                number_of_classes
            ),
            number_of_rows=len(X_train),
            random_state=random_state,
        )
    )

    candidate_models = (
        _filter_candidate_models(
            models=candidate_models,
            include_models=include_models,
        )
    )


    # create complete model pipelines
    model_pipelines = (
        create_model_pipelines(
            preprocessor=preprocessor,
            models=candidate_models,
        )
    )


    # create cross validation and scoring
    cv_strategy = create_cv_strategy(
        target=y_train,
        task_type=resolved_task_type,
        maximum_folds=maximum_cv_folds,
        random_state=random_state,
    )

    scoring_configuration = (
        get_scoring_configuration(
            task_type=resolved_task_type,
            number_of_classes=(
                number_of_classes
            ),
        )
    )

    # run baseline config
    (
        baseline_leaderboard,
        baseline_failures,
    ) = evaluate_baseline_models(
        model_pipelines=model_pipelines,
        X_train=X_train,
        y_train=y_train,
        cv_strategy=cv_strategy,
        scoring_configuration=(
            scoring_configuration
        ),
        task_type=resolved_task_type,
        n_jobs=n_jobs,
    )


    # select and tune the top models
    tuning_budget = get_tuning_budget(
        tuning_mode
    )

    selected_models = select_top_models(
        leaderboard=baseline_leaderboard,
        number_of_models=(
            tuning_budget[
                "number_of_models"
            ]
        ),
    )

    parameter_spaces = (
        get_parameter_spaces(
            resolved_task_type
        )
    )

    (
        tuned_leaderboard,
        tuned_models,
        tuning_failures,
    ) = tune_selected_models(
        selected_models=selected_models,
        model_pipelines=model_pipelines,
        parameter_spaces=parameter_spaces,
        X_train=X_train,
        y_train=y_train,
        cv_strategy=cv_strategy,
        scoring_configuration=(
            scoring_configuration
        ),
        task_type=resolved_task_type,
        n_iter=tuning_budget["n_iter"],
        random_state=random_state,
        n_jobs=n_jobs,
    )


    # Select the final model
    (
        final_model_name,
        final_pipeline,
        selection_source,
    ) = select_final_model(
        baseline_leaderboard=(
            baseline_leaderboard
        ),
        model_pipelines=model_pipelines,
        X_train=X_train,
        y_train=y_train,
        scoring_configuration=(
            scoring_configuration
        ),
        tuned_leaderboard=(
            tuned_leaderboard
        ),
        tuned_models=tuned_models,
    )


    # Evaluate the untouched test set
    class_names: list[str] | None = None

    if target_encoder is not None:
        class_names = [
            str(class_name)
            for class_name
            in target_encoder.classes_
        ]

    evaluation_result = (
        evaluate_final_model(
            model_pipeline=final_pipeline,
            X_test=X_test,
            y_test=y_test,
            task_type=resolved_task_type,
            class_names=class_names,
        )
    )

    # create the run directory
    run_directory = create_run_directory(
        base_directory=output_directory,
        run_name=run_name,
    )

    # Build the input schema
    input_schema = _build_input_schema(
        features=features,
        numerical_columns=numerical_columns,
        categorical_columns=(
            categorical_columns
        ),
    )


    # Build run metadata
    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "dataset_path": str(
            Path(csv_path)
        ),
        "target_column": target_column,
        "task_type": resolved_task_type,
        "requested_task_type": task_type,
        "tuning_mode": tuning_mode,
        "final_model": final_model_name,
        "selection_source": (
            selection_source
        ),
        "random_state": random_state,
        "test_size": test_size,
        "maximum_cv_folds": (
            maximum_cv_folds
        ),
        "original_rows": (
            dataset_summary["rows"]
        ),
        "cleaned_rows": len(
            cleaned_dataframe
        ),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "number_of_features": (
            features.shape[1]
        ),
        "number_of_numerical_features": (
            len(numerical_columns)
        ),
        "number_of_categorical_features": (
            len(categorical_columns)
        ),
        "cleaning_actions": (
            cleaning_actions
        ),
        "target_information": (
            target_information
        ),
        "test_metrics": (
            evaluation_result.metrics
        ),
        "selected_models_for_tuning": (
            selected_models
        ),
    }


    # Save all artifacts
    artifacts = save_run_artifacts(
        run_directory=run_directory,
        model_pipeline=final_pipeline,
        evaluation_result=(
            evaluation_result
        ),
        metadata=metadata,
        input_schema=input_schema,
        target_encoder=target_encoder,
        column_report=column_report,
        baseline_leaderboard=(
            baseline_leaderboard
        ),
        tuned_leaderboard=(
            tuned_leaderboard
        ),
        baseline_failures=(
            baseline_failures
        ),
        tuning_failures=(
            tuning_failures
        ),
    )


    return AutoMLRunResult(
        run_directory=run_directory,
        task_type=resolved_task_type,
        final_model_name=final_model_name,
        selection_source=selection_source,
        metrics=evaluation_result.metrics,
        cleaning_actions=tuple(
            cleaning_actions
        ),
        dataset_summary=dict(
            dataset_summary
        ),
        input_schema=input_schema,
        baseline_leaderboard=(
            baseline_leaderboard
        ),
        tuned_leaderboard=(
            tuned_leaderboard
        ),
        evaluation_result=(
            evaluation_result
        ),
        model_pipeline=final_pipeline,
        target_encoder=target_encoder,
        artifacts=artifacts,
    )