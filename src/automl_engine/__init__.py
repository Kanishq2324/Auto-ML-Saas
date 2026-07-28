"""Reusable AutoML engine for tabular datasets."""

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

from .preprocessing import build_preprocessor

from .models import (
    create_model_pipelines,
    get_candidate_models,
    get_parameter_spaces,
)

from .tuning import (
    count_parameter_combinations,
    create_cv_strategy,
    evaluate_baseline_models,
    get_scoring_configuration,
    get_tuning_budget,
    select_final_model,
    select_top_models,
    tune_selected_models,
)

from .evaluation import (
    EvaluationResult,
    evaluate_final_model,
)

from .artifacts import (
    create_run_directory,
    load_joblib_artifact,
    load_model_artifacts,
    save_dataframe_artifact,
    save_joblib_artifact,
    save_json_artifact,
    save_run_artifacts,
)

from .engine import (
    AutoMLRunResult,
    run_automl,
)

from .prediction import (
    load_input_schema,
    predict_csv,
    predict_dataframe,
    predict_from_run,
    prepare_prediction_features,
)


__version__ = "0.1.0"

__all__ = [
    "load_dataset",
    "inspect_dataset",
    "clean_dataset",
    "separate_features_and_target",
    "detect_task_type",
    "prepare_target",
    "detect_feature_types",
    "split_dataset",
    "build_preprocessor"
    "get_candidate_models",
    "create_model_pipelines",
    "get_parameter_spaces",
    "create_cv_strategy",
    "get_scoring_configuration",
    "evaluate_baseline_models",
    "select_top_models",
    "get_tuning_budget",
    "count_parameter_combinations",
    "tune_selected_models",
    "select_final_model",
    "EvaluationResult",
    "evaluate_final_model",
    "create_run_directory",
    "save_json_artifact",
    "save_dataframe_artifact",
    "save_joblib_artifact",
    "load_joblib_artifact",
    "save_run_artifacts",
    "load_model_artifacts",
    "AutoMLRunResult",
    "run_automl",
    "load_input_schema",
    "prepare_prediction_features",
    "predict_dataframe",
    "predict_from_run",
    "predict_csv",
]



"""__init__.py makes autoMl_engine a python package"""