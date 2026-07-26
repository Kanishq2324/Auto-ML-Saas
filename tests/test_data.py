"""Tests for dataset preparation utilities."""

import pandas as pd

from automl_engine.data import (
    clean_dataset,
    detect_task_type,
    load_dataset,
    prepare_target,
    separate_features_and_target,
)


def test_load_dataset_strips_column_names(tmp_path):
    csv_path = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            " age ": [20, 30],
            " target ": ["yes", "no"],
        }
    ).to_csv(csv_path, index=False)

    dataframe = load_dataset(csv_path)

    assert dataframe.columns.tolist() == [
        "age",
        "target",
    ]


def test_clean_dataset_removes_duplicates_and_constant_columns():
    dataframe = pd.DataFrame(
        {
            "age": [20, 20, 30],
            "constant": [1, 1, 1],
            "target": ["yes", "yes", "no"],
        }
    )

    cleaned_dataframe, actions = clean_dataset(
        dataframe=dataframe,
        target_column="target",
    )

    assert len(cleaned_dataframe) == 2
    assert "constant" not in cleaned_dataframe.columns
    assert len(actions) == 2


def test_detect_task_type():
    classification_target = pd.Series(
        ["yes", "no", "yes"]
    )

    regression_target = pd.Series(
        range(100)
    )

    assert (
        detect_task_type(classification_target)
        == "classification"
    )

    assert (
        detect_task_type(regression_target)
        == "regression"
    )


def test_prepare_target_encodes_classification_labels():
    target = pd.Series(
        ["yes", "no", "yes"],
        name="target",
    )

    (
        prepared_target,
        encoder,
        target_information,
    ) = prepare_target(
        target=target,
        task_type="classification",
    )

    assert encoder is not None
    assert set(prepared_target.unique()) == {0, 1}
    assert target_information["number_of_classes"] == 2


def test_separate_features_and_target():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [1000, 2000],
            "target": [0, 1],
        }
    )

    features, target = separate_features_and_target(
        dataframe=dataframe,
        target_column="target",
    )

    assert "target" not in features.columns
    assert target.name == "target"