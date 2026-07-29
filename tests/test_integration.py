"""End-to-end integration tests for the AutoML package.

These tests verify that the complete workflow works across modules:

1. Load training data from CSV.
2. Train and tune a model.
3. Evaluate the final model.
4. Save model artifacts.
5. Reload the saved model.
6. Predict new unseen data.
7. Confirm saved and in-memory models produce the same predictions.
"""

import json

import numpy as np
import pandas as pd

from automl_engine import (
    predict_csv,
    predict_from_run,
    run_automl,
)
from automl_engine.artifacts import (
    load_model_artifacts,
)


def test_classification_training_to_prediction(
    tmp_path,
):
    """Test classification from CSV training to saved-model prediction."""

    # ---------------------------------------------------------
    # 1. Create a temporary classification training dataset
    # ---------------------------------------------------------

    csv_path = (
        tmp_path / "classification.csv"
    )

    ages = list(
        range(20, 80)
    )

    training_dataframe = pd.DataFrame(
        {
            # Numerical feature
            "age": ages,

            # Another numerical feature
            "income": [
                age * 1000
                for age in ages
            ],

            # Categorical feature
            #
            # There are 60 rows, so repeating three
            # categories 20 times gives exactly 60 values.
            "city": (
                ["A", "B", "C"] * 20
            ),

            # Classification target
            #
            # People below 50 belong to class "no".
            # People aged 50 or above belong to class "yes".
            "target": [
                (
                    "no"
                    if age < 50
                    else "yes"
                )
                for age in ages
            ],
        }
    )

    training_dataframe.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Run the complete AutoML training workflow
    # ---------------------------------------------------------

    result = run_automl(
        csv_path=csv_path,
        target_column="target",

        # Fast mode keeps this integration test quick.
        tuning_mode="fast",

        # Using one model keeps the test predictable
        # and prevents unnecessary training time.
        include_models=[
            "Logistic Regression"
        ],

        # pytest creates tmp_path as a temporary directory.
        output_directory=tmp_path,

        # A fixed run name makes the artifact path predictable.
        run_name="classification_run",

        # Three folds are sufficient for this small test.
        maximum_cv_folds=3,
    )

    # ---------------------------------------------------------
    # 3. Verify the training result
    # ---------------------------------------------------------

    assert (
        result.task_type
        == "classification"
    )

    assert (
        result.final_model_name
        == "Logistic Regression"
    )

    assert "F1 Macro" in result.metrics

    # Classification targets should have a LabelEncoder.
    assert result.target_encoder is not None

    # The run directory should have been created.
    assert result.run_directory.exists()

    # ---------------------------------------------------------
    # 4. Create new unseen prediction data
    # ---------------------------------------------------------

    new_data = pd.DataFrame(
        {
            # This column was not used during training.
            #
            # Prediction code should preserve extra columns
            # in the output but should not send them to the model.
            "customer_id": [
                101,
                102,
            ],

            "age": [
                25,
                70,
            ],

            "income": [
                25_000,
                70_000,
            ],

            "city": [
                "A",

                # "D" was not present in the training data.
                #
                # OneHotEncoder(handle_unknown="ignore")
                # should allow this value without crashing.
                "D",
            ],
        }
    )

    # ---------------------------------------------------------
    # 5. Load the saved run and generate predictions
    # ---------------------------------------------------------

    saved_predictions = (
        predict_from_run(
            run_directory=(
                result.run_directory
            ),
            dataframe=new_data,
        )
    )

    # ---------------------------------------------------------
    # 6. Generate predictions using the in-memory model
    # ---------------------------------------------------------

    # The trained pipeline expects only the original
    # feature columns, not customer_id.
    feature_columns = (
        result.input_schema[
            "feature_columns"
        ]
    )

    direct_encoded_predictions = (
        result.model_pipeline.predict(
            new_data[
                feature_columns
            ]
        )
    )

    # The model internally predicts encoded values such as:
    #
    # 0 -> no
    # 1 -> yes
    #
    # Convert them back into the original labels.
    direct_decoded_predictions = (
        result.target_encoder
        .inverse_transform(
            direct_encoded_predictions
            .astype(int)
        )
    )

    # ---------------------------------------------------------
    # 7. Compare saved-model and in-memory predictions
    # ---------------------------------------------------------

    assert (
        saved_predictions[
            "Prediction"
        ].tolist()
        == direct_decoded_predictions.tolist()
    )

    # Every predicted label must belong to the known classes.
    assert (
        set(
            saved_predictions[
                "Prediction"
            ]
        )
        .issubset(
            {"no", "yes"}
        )
    )

    # Classification predictions should contain probabilities.
    assert (
        "Probability_no"
        in saved_predictions.columns
    )

    assert (
        "Probability_yes"
        in saved_predictions.columns
    )

    # Extra identifier columns should remain in the output.
    assert (
        "customer_id"
        in saved_predictions.columns
    )

    # ---------------------------------------------------------
    # 8. Verify the artifact manifest
    # ---------------------------------------------------------

    manifest_path = (
        result.run_directory
        / "artifact_manifest.json"
    )

    assert manifest_path.exists()

    with manifest_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    assert (
        manifest[
            "model_pipeline"
        ]
        == "model_pipeline.joblib"
    )

    assert (
        manifest[
            "input_schema"
        ]
        == "input_schema.json"
    )

    assert (
        manifest[
            "test_metrics"
        ]
        == "test_metrics.json"
    )

    assert (
        manifest[
            "test_predictions"
        ]
        == "test_predictions.csv"
    )

    # ---------------------------------------------------------
    # 9. Load the model artifacts manually
    # ---------------------------------------------------------

    (
        loaded_pipeline,
        loaded_encoder,
    ) = load_model_artifacts(
        result.run_directory
    )

    loaded_encoded_predictions = (
        loaded_pipeline.predict(
            new_data[
                feature_columns
            ]
        )
    )

    # The model loaded from disk should produce exactly
    # the same encoded predictions as the original model.
    assert np.array_equal(
        direct_encoded_predictions,
        loaded_encoded_predictions,
    )

    assert loaded_encoder is not None

    assert (
        loaded_encoder.classes_.tolist()
        == ["no", "yes"]
    )


def test_regression_training_to_csv_prediction(
    tmp_path,
):
    """Test regression from CSV training to saved CSV prediction."""

    # ---------------------------------------------------------
    # 1. Create a temporary regression training dataset
    # ---------------------------------------------------------

    csv_path = (
        tmp_path / "regression.csv"
    )

    ages = list(
        range(20, 80)
    )

    bmi_values = [
        18.0 + index * 0.25
        for index in range(
            len(ages)
        )
    ]

    training_dataframe = pd.DataFrame(
        {
            # Numerical features
            "age": ages,
            "bmi": bmi_values,

            # Categorical feature
            "region": (
                [
                    "north",
                    "south",
                    "east",
                ]
                * 20
            ),

            # Continuous regression target
            #
            # This formula creates a predictable relationship
            # between age, BMI, and charges.
            "charges": [
                (
                    age * 120
                    + bmi * 75
                )
                for age, bmi
                in zip(
                    ages,
                    bmi_values,
                )
            ],
        }
    )

    training_dataframe.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Run the complete AutoML regression workflow
    # ---------------------------------------------------------

    result = run_automl(
        csv_path=csv_path,
        target_column="charges",
        tuning_mode="fast",

        # Use one lightweight regression model.
        include_models=[
            "Ridge Regression"
        ],

        output_directory=tmp_path,
        run_name="regression_run",
        maximum_cv_folds=3,
    )

    # ---------------------------------------------------------
    # 3. Verify the regression training result
    # ---------------------------------------------------------

    assert (
        result.task_type
        == "regression"
    )

    assert (
        result.final_model_name
        == "Ridge Regression"
    )

    assert "MAE" in result.metrics
    assert "RMSE" in result.metrics
    assert "R2" in result.metrics

    # Regression does not use LabelEncoder.
    assert result.target_encoder is None

    # ---------------------------------------------------------
    # 4. Create new prediction data
    # ---------------------------------------------------------

    new_data = pd.DataFrame(
        {
            # Extra identifier column that was not used
            # during model training.
            "record_id": [
                201,
                202,
            ],

            "age": [
                30,
                65,
            ],

            "bmi": [
                22.0,
                31.0,
            ],

            "region": [
                "north",

                # "west" was unseen during training.
                #
                # The fitted one-hot encoder should safely
                # ignore this unknown category.
                "west",
            ],
        }
    )

    prediction_csv_path = (
        tmp_path
        / "regression_prediction_data.csv"
    )

    output_csv_path = (
        tmp_path
        / "regression_predictions.csv"
    )

    new_data.to_csv(
        prediction_csv_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Predict from the saved run using a CSV file
    # ---------------------------------------------------------

    prediction_output = predict_csv(
        run_directory=(
            result.run_directory
        ),
        csv_path=prediction_csv_path,
        output_path=output_csv_path,
    )

    # ---------------------------------------------------------
    # 6. Generate direct in-memory predictions
    # ---------------------------------------------------------

    feature_columns = (
        result.input_schema[
            "feature_columns"
        ]
    )

    direct_predictions = (
        result.model_pipeline.predict(
            new_data[
                feature_columns
            ]
        )
    )

    # ---------------------------------------------------------
    # 7. Compare saved-model and in-memory predictions
    # ---------------------------------------------------------

    # np.allclose is used for floating-point predictions.
    #
    # Floating-point values may differ by an extremely small
    # amount even when they are effectively equal.
    assert np.allclose(
        prediction_output[
            "Prediction"
        ].to_numpy(),
        direct_predictions,
    )

    # The requested CSV output should have been created.
    assert output_csv_path.exists()

    # Extra columns should be preserved.
    assert (
        "record_id"
        in prediction_output.columns
    )

    # Every row should receive a prediction.
    assert (
        prediction_output[
            "Prediction"
        ]
        .notna()
        .all()
    )

    # Regression predictions should not contain
    # classification probability columns.
    probability_columns = [
        column
        for column
        in prediction_output.columns
        if column.startswith(
            "Probability_"
        )
    ]

    assert probability_columns == []

    # Confirm that the saved predictions CSV contains
    # the same number of rows as the input data.
    saved_output = pd.read_csv(
        output_csv_path
    )

    assert len(saved_output) == len(
        new_data
    )