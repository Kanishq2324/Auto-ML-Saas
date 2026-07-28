"""Prediction utilities for trained AutoML runs."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .artifacts import load_model_artifacts
from .data import load_dataset


def load_input_schema(
    run_directory: str | Path,
) -> dict[str, Any]:
    """Load and validate a saved input schema."""

    schema_path = (
        Path(run_directory)
        / "input_schema.json"
    )

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Input schema not found: "
            f"{schema_path.resolve()}"
        )

    with schema_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        input_schema = json.load(file)

    if not isinstance(input_schema, dict):
        raise TypeError(
            "input_schema.json must contain "
            "a JSON object."
        )

    required_keys = {
        "feature_columns",
        "numerical_columns",
        "categorical_columns",
        "feature_dtypes",
    }

    missing_keys = (
        required_keys
        - set(input_schema)
    )

    if missing_keys:
        raise ValueError(
            "Input schema is missing keys: "
            f"{sorted(missing_keys)}"
        )

    return input_schema



def prepare_prediction_features(
    dataframe: pd.DataFrame,
    input_schema: dict[str, Any],
) -> pd.DataFrame:
    """Validate and prepare features for prediction."""

    if dataframe.empty:
        raise ValueError(
            "Prediction data cannot be empty."
        )

    if dataframe.columns.duplicated().any():
        duplicated_columns = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ].tolist()
        )

        raise ValueError(
            "Duplicate prediction columns found: "
            f"{duplicated_columns}"
        )

    expected_columns = list(
        input_schema["feature_columns"]
    )

    numerical_columns = list(
        input_schema["numerical_columns"]
    )

    categorical_columns = list(
        input_schema["categorical_columns"]
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction data is missing required "
            f"columns: {missing_columns}"
        )


    prepared_features = dataframe.loc[
        :,
        expected_columns,
    ].copy()


    for column in numerical_columns:
        original_values = (
            prepared_features[column]
        )

        converted_values = pd.to_numeric(
            original_values,
            errors="coerce",
        )

        invalid_mask = (
            original_values.notna()
            & converted_values.isna()
        )

        if invalid_mask.any():
            invalid_examples = (
                original_values[
                    invalid_mask
                ]
                .astype(str)
                .unique()
                .tolist()[:5]
            )

            raise ValueError(
                f"Column '{column}' contains "
                "non-numeric values: "
                f"{invalid_examples}"
            )

        prepared_features[column] = (
            converted_values
        )

    for column in categorical_columns:
        prepared_features[column] = (
            prepared_features[column]
            .astype("object")
        )

    return prepared_features



def predict_dataframe(
    model_pipeline: Pipeline,
    dataframe: pd.DataFrame,
    input_schema: dict[str, Any],
    target_encoder: LabelEncoder | None = None,
) -> pd.DataFrame:
    """Generate predictions for a pandas DataFrame."""

    prepared_features = (
        prepare_prediction_features(
            dataframe=dataframe,
            input_schema=input_schema,
        )
    )

    raw_predictions = np.asarray(
        model_pipeline.predict(
            prepared_features
        )
    )

    prediction_output = (
        dataframe.copy()
    )

    if target_encoder is not None:
        encoded_predictions = (
            raw_predictions.astype(int)
        )

        decoded_predictions = (
            target_encoder.inverse_transform(
                encoded_predictions
            )
        )

        prediction_output[
            "Prediction"
        ] = decoded_predictions

    else:
        prediction_output[
            "Prediction"
        ] = raw_predictions



    if hasattr(
        model_pipeline,
        "predict_proba",
    ):
        probabilities = np.asarray(
            model_pipeline.predict_proba(
                prepared_features
            )
        )

        if (
            target_encoder is not None
            and len(target_encoder.classes_)
            == probabilities.shape[1]
        ):
            probability_labels = [
                str(class_name)
                for class_name
                in target_encoder.classes_
            ]

        else:
            model = model_pipeline.named_steps[
                "model"
            ]

            model_classes = getattr(
                model,
                "classes_",
                range(
                    probabilities.shape[1]
                ),
            )

            probability_labels = [
                str(class_name)
                for class_name
                in model_classes
            ]


    for class_index, class_name in enumerate(
            probability_labels
        ):
        prediction_output[
                f"Probability_{class_name}"
            ] = probabilities[
                :,
                class_index,
            ]

    return prediction_output



def predict_from_run(
    run_directory: str | Path,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Load saved artifacts and predict a DataFrame."""

    (
        model_pipeline,
        target_encoder,
    ) = load_model_artifacts(
        run_directory
    )

    input_schema = load_input_schema(
        run_directory
    )

    return predict_dataframe(
        model_pipeline=model_pipeline,
        dataframe=dataframe,
        input_schema=input_schema,
        target_encoder=target_encoder,
    )


def predict_csv(
    run_directory: str | Path,
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate predictions for a new CSV file."""

    dataframe = load_dataset(
        csv_path
    )

    prediction_output = predict_from_run(
        run_directory=run_directory,
        dataframe=dataframe,
    )

    if output_path is not None:
        prediction_path = Path(
            output_path
        )

        prediction_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_output.to_csv(
            prediction_path,
            index=False,
        )

    return prediction_output