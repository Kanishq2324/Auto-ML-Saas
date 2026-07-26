"""Tests for the candidate model registry."""

import pytest
from sklearn.pipeline import Pipeline

from automl_engine.models import (
    create_model_pipelines,
    get_candidate_models,
    get_parameter_spaces,
)
from automl_engine.preprocessing import (
    build_preprocessor,
)


def test_classification_model_registry():
    models = get_candidate_models(
        task_type="classification",
        number_of_classes=2,
        number_of_rows=1000,
    )

    expected_models = {
        "Dummy Classifier",
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "KNN",
        "XGBoost",
    }

    assert set(models) == expected_models


def test_regression_model_registry():
    models = get_candidate_models(
        task_type="regression",
        number_of_rows=1000,
    )

    expected_models = {
        "Dummy Regressor",
        "Ridge Regression",
        "Random Forest",
        "Extra Trees",
        "KNN",
        "XGBoost",
    }

    assert set(models) == expected_models



def test_large_dataset_skips_knn():
    models = get_candidate_models(
        task_type="classification",
        number_of_classes=2,
        number_of_rows=100_000,
    )

    assert "KNN" not in models


def test_create_model_pipelines():
    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=["city"],
    )

    models = get_candidate_models(
        task_type="classification",
        number_of_classes=2,
        number_of_rows=100,
    )

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models=models,
    )

    assert set(pipelines) == set(models)

    for pipeline in pipelines.values():
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps


def test_parameter_spaces_use_pipeline_prefix():
    parameter_spaces = get_parameter_spaces(
        task_type="classification"
    )

    for model_space in parameter_spaces.values():
        assert all(
            parameter_name.startswith("model__")
            for parameter_name in model_space
        )


def test_model_registry_rejects_invalid_task():
    with pytest.raises(
        ValueError,
        match="Unsupported task type",
    ):
        get_candidate_models(
            task_type="clustering",
        )


def test_model_registry_rejects_invalid_task():
    with pytest.raises(
        ValueError,
        match="Unsupported task type",
    ):
        get_candidate_models(
            task_type="clustering",
        )