"""Tests for saved-model prediction utilities."""

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from automl_engine.artifacts import (
    save_run_artifacts,
)
from automl_engine.evaluation import (
    EvaluationResult,
)
from automl_engine.models import (
    create_model_pipelines,
)
from automl_engine.prediction import (
    predict_csv,
    predict_dataframe,
    prepare_prediction_features,
)
from automl_engine.preprocessing import (
    build_preprocessor,
)
from sklearn.linear_model import (
    LogisticRegression,
)


def _create_classification_artifacts():
    """Create a fitted model and schema for tests."""

    features = pd.DataFrame(
        {
            "age": [
                20,
                22,
                24,
                42,
                44,
                46,
            ],
            "city": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
            ],
        }
    )

    target_labels = pd.Series(
        [
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "yes",
        ]
    )

    target_encoder = LabelEncoder()

    encoded_target = (
        target_encoder.fit_transform(
            target_labels
        )
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=["city"],
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
        encoded_target,
    )

    input_schema = {
        "feature_columns": [
            "age",
            "city",
        ],
        "numerical_columns": [
            "age",
        ],
        "categorical_columns": [
            "city",
        ],
        "feature_dtypes": {
            "age": "int64",
            "city": "object",
        },
    }

    return (
        pipeline,
        target_encoder,
        input_schema,
    )


def test_prediction_rejects_missing_columns():
    dataframe = pd.DataFrame(
        {
            "age": [25, 45]
        }
    )

    input_schema = {
        "feature_columns": [
            "age",
            "city",
        ],
        "numerical_columns": [
            "age",
        ],
        "categorical_columns": [
            "city",
        ],
        "feature_dtypes": {
            "age": "int64",
            "city": "object",
        },
    }

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        prepare_prediction_features(
            dataframe=dataframe,
            input_schema=input_schema,
        )


def test_prediction_rejects_invalid_numeric_values():
    dataframe = pd.DataFrame(
        {
            "age": [
                "twenty",
                "45",
            ],
            "city": [
                "A",
                "B",
            ],
        }
    )

    input_schema = {
        "feature_columns": [
            "age",
            "city",
        ],
        "numerical_columns": [
            "age",
        ],
        "categorical_columns": [
            "city",
        ],
        "feature_dtypes": {
            "age": "int64",
            "city": "object",
        },
    }

    with pytest.raises(
        ValueError,
        match="non-numeric values",
    ):
        prepare_prediction_features(
            dataframe=dataframe,
            input_schema=input_schema,
        )


def test_predict_dataframe_decodes_labels():
    (
        pipeline,
        target_encoder,
        input_schema,
    ) = _create_classification_artifacts()

    new_data = pd.DataFrame(
        {
            "customer_id": [
                101,
                102,
            ],
            "city": [
                "A",
                "B",
            ],
            "age": [
                21,
                45,
            ],
        }
    )

    predictions = predict_dataframe(
        model_pipeline=pipeline,
        dataframe=new_data,
        input_schema=input_schema,
        target_encoder=target_encoder,
    )

    assert (
        predictions[
            "Prediction"
        ].tolist()
        == ["no", "yes"]
    )

    assert (
        "Probability_no"
        in predictions.columns
    )

    assert (
        "Probability_yes"
        in predictions.columns
    )

    assert (
        "customer_id"
        in predictions.columns
    )


def test_predict_csv_from_saved_run(
    tmp_path,
):
    (
        pipeline,
        target_encoder,
        input_schema,
    ) = _create_classification_artifacts()

    run_directory = tmp_path / "saved_run"

    evaluation_result = EvaluationResult(
        task_type="classification",
        metrics={
            "Accuracy": 1.0,
        },
        predictions=pd.DataFrame(
            {
                "Actual": [0, 1],
                "Predicted": [0, 1],
            }
        ),
    )

    save_run_artifacts(
        run_directory=run_directory,
        model_pipeline=pipeline,
        evaluation_result=(
            evaluation_result
        ),
        metadata={
            "task_type": (
                "classification"
            )
        },
        input_schema=input_schema,
        target_encoder=target_encoder,
    )

    new_data_path = (
        tmp_path / "new_data.csv"
    )

    pd.DataFrame(
        {
            "age": [
                21,
                45,
            ],
            "city": [
                "A",
                "B",
            ],
        }
    ).to_csv(
        new_data_path,
        index=False,
    )

    output_path = (
        tmp_path / "predictions.csv"
    )

    predictions = predict_csv(
        run_directory=run_directory,
        csv_path=new_data_path,
        output_path=output_path,
    )

    assert len(predictions) == 2
    assert output_path.exists()

    assert (
        predictions[
            "Prediction"
        ].tolist()
        == ["no", "yes"]
    )