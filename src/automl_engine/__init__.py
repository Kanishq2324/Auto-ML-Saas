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
    create_cv_strategy,
    evaluate_baseline_models,
    get_scoring_configuration,
    select_top_models,
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
]



"""__init__.py makes autoMl_engine a python package"""