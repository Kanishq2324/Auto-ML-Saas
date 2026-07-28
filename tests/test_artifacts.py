"""Tests for AutoML artifact utilities."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from automl_engine.artifacts import (
    create_run_directory,
    load_model_artifacts,
    save_json_artifact,
    save_run_artifacts,
)
from automl_engine.evaluation import (
    EvaluationResult,
)


def _create_fitted_pipeline() -> Pipeline:
    """Create a small fitted model for artifact tests."""

    features = pd.DataFrame(
        {
            "age": [
                20.0,
                25.0,
                40.0,
                45.0,
            ]
        }
    )

    target = pd.Series(
        [0, 0, 1, 1]
    )

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(),
            ),
        ]
    )

    pipeline.fit(
        features,
        target,
    )

    return pipeline


def test_create_run_directory(tmp_path):
    run_directory = (
        create_run_directory(
            base_directory=tmp_path,
            run_name="test_run",
        )
    )

    assert run_directory.exists()
    assert run_directory.name == "test_run"

    with pytest.raises(
        FileExistsError
    ):
        create_run_directory(
            base_directory=tmp_path,
            run_name="test_run",
        )


def test_save_json_handles_numpy_values(
    tmp_path,
):
    file_path = (
        tmp_path / "metrics.json"
    )

    save_json_artifact(
        {
            "count": np.int64(10),
            "score": np.float64(0.85),
            "values": np.array(
                [1, 2, 3]
            ),
        },
        file_path,
    )

    with file_path.open(
        encoding="utf-8"
    ) as file:
        saved_data = json.load(file)

    assert saved_data == {
        "count": 10,
        "score": 0.85,
        "values": [1, 2, 3],
    }



def test_save_run_artifacts(tmp_path):
    run_directory = (
        create_run_directory(
            base_directory=tmp_path,
            run_name="classification_run",
        )
    )

    pipeline = (
        _create_fitted_pipeline()
    )

    target_encoder = LabelEncoder()

    target_encoder.fit(
        ["no", "yes"]
    )

    evaluation_result = (
        EvaluationResult(
            task_type="classification",
            metrics={
                "Accuracy": 1.0,
                "F1 Macro": 1.0,
            },
            predictions=pd.DataFrame(
                {
                    "Actual": [0, 1],
                    "Predicted": [0, 1],
                }
            ),
            classification_report=(
                pd.DataFrame(
                    {
                        "precision": [
                            1.0,
                            1.0,
                        ]
                    },
                    index=[
                        "no",
                        "yes",
                    ],
                )
            ),
            confusion_matrix=(
                pd.DataFrame(
                    [
                        [1, 0],
                        [0, 1],
                    ]
                )
            ),
        )
    )

    artifacts = save_run_artifacts(
        run_directory=run_directory,
        model_pipeline=pipeline,
        evaluation_result=(
            evaluation_result
        ),
        metadata={
            "task_type": (
                "classification"
            ),
            "final_model": (
                "Logistic Regression"
            ),
        },
        input_schema={
            "numerical_columns": [
                "age"
            ],
            "categorical_columns": [],
        },
        target_encoder=target_encoder,
        baseline_leaderboard=(
            pd.DataFrame(
                {
                    "Model": [
                        "Logistic Regression"
                    ],
                    "F1 Macro": [1.0],
                }
            )
        ),
    )

    expected_artifacts = {
        "model_pipeline",
        "target_encoder",
        "metadata",
        "input_schema",
        "test_metrics",
        "test_predictions",
        "classification_report",
        "confusion_matrix",
        "baseline_leaderboard",
        "artifact_manifest",
    }

    assert (
        set(artifacts)
        == expected_artifacts
    )

    for artifact_path in (
        artifacts.values()
    ):
        assert artifact_path.exists()



def test_load_model_artifacts(tmp_path):
    pipeline = (
        _create_fitted_pipeline()
    )

    target_encoder = LabelEncoder()

    target_encoder.fit(
        ["no", "yes"]
    )

    evaluation_result = (
        EvaluationResult(
            task_type="classification",
            metrics={
                "Accuracy": 1.0
            },
            predictions=pd.DataFrame(
                {
                    "Actual": [0],
                    "Predicted": [0],
                }
            ),
        )
    )

    save_run_artifacts(
        run_directory=tmp_path,
        model_pipeline=pipeline,
        evaluation_result=(
            evaluation_result
        ),
        metadata={
            "task_type": (
                "classification"
            )
        },
        input_schema={
            "numerical_columns": [
                "age"
            ],
            "categorical_columns": [],
        },
        target_encoder=target_encoder,
    )

    (
        loaded_pipeline,
        loaded_encoder,
    ) = load_model_artifacts(
        tmp_path
    )

    features = pd.DataFrame(
        {
            "age": [22.0, 43.0]
        }
    )

    original_predictions = (
        pipeline.predict(features)
    )

    loaded_predictions = (
        loaded_pipeline.predict(
            features
        )
    )

    assert np.array_equal(
        original_predictions,
        loaded_predictions,
    )

    assert loaded_encoder is not None

    assert (
        loaded_encoder.classes_.tolist()
        == ["no", "yes"]
    )