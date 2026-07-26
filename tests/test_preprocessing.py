"""Tests for preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from automl_engine.preprocessing import (
    build_preprocessor,
)

def test_preprocessor_handles_mixed_features():
    dataframe = pd.DataFrame(
        {
            "age": [
                20.0,
                np.nan,
                40.0,
            ],
            "income": [
                1000.0,
                2000.0,
                np.nan,
            ],
            "city": [
                "Delhi",
                None,
                "Mumbai",
            ],
        }
    )

    preprocessor = build_preprocessor(
        numerical_columns=[
            "age",
            "income",
        ],
        categorical_columns=[
            "city",
        ],
    )

    transformed_data = (
        preprocessor.fit_transform(
            dataframe
        )
    )

    if hasattr(
        transformed_data,
        "toarray",
    ):
        transformed_data = (
            transformed_data.toarray()
        )

    transformed_data = np.asarray(
        transformed_data
    )

    assert transformed_data.shape[0] == 3
    assert not np.isnan(
        transformed_data
    ).any()


    def test_preprocessor_handles_unknown_categories():
        training_data = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "city": [
                    "Delhi",
                    "Mumbai",
                    "Delhi",
                ],
            }
        )

        test_data = pd.DataFrame(
            {
                "age": [50],
                "city": ["Kolkata"],
            }
        )

        preprocessor = build_preprocessor(
            numerical_columns=["age"],
            categorical_columns=["city"],
        )

        transformed_training_data = (
            preprocessor.fit_transform(
                training_data
            )
        )

        transformed_test_data = (
            preprocessor.transform(
                test_data
            )
        )

        assert (
            transformed_training_data.shape[1]
            == transformed_test_data.shape[1]
        )



def test_preprocessor_rejects_overlapping_columns():
    with pytest.raises(
        ValueError,
        match="both numerical and categorical",
    ):
        build_preprocessor(
            numerical_columns=[
                "age",
                "income",
            ],
            categorical_columns=[
                "age",
                "city",
            ],
        )


def test_preprocessor_rejects_empty_column_lists():
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        build_preprocessor(
            numerical_columns=[],
            categorical_columns=[],
        )


def test_preprocessor_supports_numerical_only_data():
    dataframe = pd.DataFrame(
        {
            "age": [
                20.0,
                np.nan,
                40.0,
            ]
        }
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    transformed_data = (
        preprocessor.fit_transform(
            dataframe
        )
    )

    assert transformed_data.shape == (
        3,
        1,
    )