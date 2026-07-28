"""Tests for final model evaluation utilities."""

import pandas as pd
import pytest
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from automl_engine.evaluation import (
    EvaluationResult,
    evaluate_final_model,
)
from automl_engine.models import (
    create_model_pipelines,
)
from automl_engine.preprocessing import (
    build_preprocessor,
)

def test_evaluate_classification_model():
    features = pd.DataFrame(
        {
            "age": [
                20,
                22,
                24,
                26,
                40,
                42,
                44,
                46,
            ]
        }
    )

    target = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models={
            "Logistic Regression": (
                LogisticRegression(
                    max_iter=1000
                )
            )
        },
    )

    pipeline = pipelines[
        "Logistic Regression"
    ]

    pipeline.fit(
        features,
        target,
    )

    result = evaluate_final_model(
        model_pipeline=pipeline,
        X_test=features,
        y_test=target,
        task_type="classification",
        class_names=[
            "young",
            "older",
        ],
    )

    assert isinstance(
        result,
        EvaluationResult,
    )

    assert result.task_type == "classification"
    assert "F1 Macro" in result.metrics
    assert len(result.predictions) == 8

    assert (
        result.classification_report
        is not None
    )

    assert (
        result.confusion_matrix
        is not None
    )


def test_evaluate_regression_model():
    features = pd.DataFrame(
        {
            "age": [
                20,
                25,
                30,
                35,
                40,
                45,
            ]
        }
    )

    target = pd.Series(
        [
            100.0,
            125.0,
            150.0,
            175.0,
            200.0,
            225.0,
        ]
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models={
            "Ridge Regression": Ridge()
        },
    )

    pipeline = pipelines[
        "Ridge Regression"
    ]

    pipeline.fit(
        features,
        target,
    )

    result = evaluate_final_model(
        model_pipeline=pipeline,
        X_test=features,
        y_test=target,
        task_type="regression",
    )

    assert result.task_type == "regression"
    assert "MAE" in result.metrics
    assert "RMSE" in result.metrics
    assert "R2" in result.metrics

    assert (
        "Residual"
        in result.predictions.columns
    )

    assert (
        result.classification_report
        is None
    )

def test_evaluation_rejects_invalid_task():
    features = pd.DataFrame(
        {
            "age": [20, 30]
        }
    )

    target = pd.Series(
        [0, 1]
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models={
            "Logistic Regression": (
                LogisticRegression()
            )
        },
    )

    with pytest.raises(
        ValueError,
        match="Unsupported task type",
    ):
        evaluate_final_model(
            model_pipeline=pipelines[
                "Logistic Regression"
            ],
            X_test=features,
            y_test=target,
            task_type="clustering",
        )


def test_evaluation_rejects_mismatched_rows():
    features = pd.DataFrame(
        {
            "age": [20, 30, 40]
        }
    )

    target = pd.Series(
        [0, 1]
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models={
            "Logistic Regression": (
                LogisticRegression()
            )
        },
    )

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        evaluate_final_model(
            model_pipeline=pipelines[
                "Logistic Regression"
            ],
            X_test=features,
            y_test=target,
            task_type="classification",
        )