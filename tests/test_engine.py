"""End-to-end tests for the AutoML controller."""

import pandas as pd
import pytest

from automl_engine.engine import (
    AutoMLRunResult,
    run_automl,
)

def test_run_automl_classification(
    tmp_path,
):
    csv_path = (
        tmp_path / "classification.csv"
    )

    dataframe = pd.DataFrame(
        {
            "age": list(
                range(20, 60)
            ),
            "income": [
                value * 1000
                for value in range(20, 60)
            ],
            "city": (
                ["A", "B"] * 20
            ),
            "target": (
                ["no"] * 20
                + ["yes"] * 20
            ),
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    result = run_automl(
        csv_path=csv_path,
        target_column="target",
        tuning_mode="fast",
        include_models=[
            "Logistic Regression"
        ],
        output_directory=tmp_path,
        run_name="classification_run",
        maximum_cv_folds=3,
    )

    assert isinstance(
        result,
        AutoMLRunResult,
    )

    assert (
        result.task_type
        == "classification"
    )

    assert (
        result.final_model_name
        == "Logistic Regression"
    )

    assert "F1 Macro" in result.metrics
    assert result.target_encoder is not None
    assert result.run_directory.exists()

    assert (
        result.run_directory
        / "model_pipeline.joblib"
    ).exists()


def test_run_automl_regression(
    tmp_path,
):
    csv_path = (
        tmp_path / "regression.csv"
    )

    ages = list(
        range(20, 70)
    )

    dataframe = pd.DataFrame(
        {
            "age": ages,
            "bmi": [
                20 + index * 0.2
                for index in range(
                    len(ages)
                )
            ],
            "region": (
                ["north", "south"]
                * 25
            ),
            "charges": [
                (
                    age * 100
                    + index * 20
                )
                for index, age
                in enumerate(ages)
            ],
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    result = run_automl(
        csv_path=csv_path,
        target_column="charges",
        tuning_mode="fast",
        include_models=[
            "Ridge Regression"
        ],
        output_directory=tmp_path,
        run_name="regression_run",
        maximum_cv_folds=3,
    )

    assert result.task_type == "regression"

    assert (
        result.final_model_name
        == "Ridge Regression"
    )

    assert "RMSE" in result.metrics
    assert result.target_encoder is None
    assert result.run_directory.exists()


def test_run_automl_rejects_unknown_model(
    tmp_path,
):
    csv_path = tmp_path / "data.csv"

    pd.DataFrame(
        {
            "feature": list(
                range(30)
            ),
            "target": [
                "no",
                "yes",
            ] * 15,
        }
    ).to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Unknown or unavailable models",
    ):
        run_automl(
            csv_path=csv_path,
            target_column="target",
            include_models=[
                "Imaginary Model"
            ],
            output_directory=tmp_path,
        )