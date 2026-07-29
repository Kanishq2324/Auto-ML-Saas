"""Tests for cross-validation and model evaluation utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from automl_engine.models import (
    create_model_pipelines,
)
from automl_engine.preprocessing import (
    build_preprocessor,
)

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

from automl_engine.tuning import (
    count_parameter_combinations,
    create_cv_strategy,
    evaluate_baseline_models,
    get_scoring_configuration,
    get_tuning_budget,
    select_final_model,
    select_top_models,
    tune_selected_models,
)


def test_classification_scorers_handle_missing_predictions():
    """Macro scorers should return zero without warnings."""

    configuration = (
        get_scoring_configuration(
            task_type="classification",
            number_of_classes=2,
        )
    )

    scoring = configuration["scoring"]

    actual_values = np.array(
        [0, 0, 1, 1]
    )

    # The model predicts only class 0.
    predicted_values = np.array(
        [0, 0, 0, 0]
    )

    precision_scorer = scoring[
        "precision_macro"
    ]

    recall_scorer = scoring[
        "recall_macro"
    ]

    f1_scorer = scoring[
        "f1_macro"
    ]

    assert (
        precision_scorer._score_func(
            actual_values,
            predicted_values,
            average="macro",
            zero_division=0,
        )
        >= 0
    )

    assert (
        recall_scorer._score_func(
            actual_values,
            predicted_values,
            average="macro",
            zero_division=0,
        )
        >= 0
    )

    assert (
        f1_scorer._score_func(
            actual_values,
            predicted_values,
            average="macro",
            zero_division=0,
        )
        >= 0
    )



def test_classification_cv_uses_smallest_class_count():
    target = pd.Series(
        [0, 0, 0, 1, 1, 1]
    )

    cv_strategy = create_cv_strategy(
        target=target,
        task_type="classification",
        maximum_folds=5,
    )

    assert cv_strategy.n_splits == 3


def test_regression_cv_uses_maximum_folds():
    target = pd.Series(
        range(20)
    )

    cv_strategy = create_cv_strategy(
        target=target,
        task_type="regression",
        maximum_folds=5,
    )

    assert cv_strategy.n_splits == 5


def test_binary_classification_scoring():
    configuration = (
        get_scoring_configuration(
            task_type="classification",
            number_of_classes=2,
        )
    )

    assert (
        configuration["primary_metric"]
        == "f1_macro"
    )

    assert (
        configuration[
            "leaderboard_column"
        ]
        == "F1 Macro"
    )

    assert (
        "roc_auc"
        in configuration["scoring"]
    )


def test_regression_scoring():
    configuration = (
        get_scoring_configuration(
            task_type="regression"
        )
    )

    assert (
        configuration["primary_metric"]
        == "rmse"
    )

    assert (
        configuration[
            "higher_is_better"
        ]
        is False
    )


def test_select_top_models_excludes_dummy():
    leaderboard = pd.DataFrame(
        {
            "Model": [
                "Dummy Classifier",
                "XGBoost",
                "Random Forest",
            ],
            "F1 Macro": [
                0.40,
                0.85,
                0.80,
            ],
        }
    )

    selected_models = select_top_models(
        leaderboard=leaderboard,
        number_of_models=2,
    )

    assert selected_models == [
        "XGBoost",
        "Random Forest",
    ]


def test_evaluate_baseline_classification_models():
    features = pd.DataFrame(
        {
            "age": [
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
                40,
                42,
            ],
            "city": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
        }
    )

    target = pd.Series(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=["city"],
    )

    models = {
        "Dummy Classifier": (
            DummyClassifier(
                strategy="most_frequent"
            )
        ),
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000
            )
        ),
    }

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models=models,
    )

    cv_strategy = create_cv_strategy(
        target=target,
        task_type="classification",
        maximum_folds=3,
    )

    scoring_configuration = (
        get_scoring_configuration(
            task_type="classification",
            number_of_classes=2,
        )
    )

    leaderboard, failures = (
        evaluate_baseline_models(
            model_pipelines=pipelines,
            X_train=features,
            y_train=target,
            cv_strategy=cv_strategy,
            scoring_configuration=(
                scoring_configuration
            ),
            task_type="classification",
        )
    )

    assert len(leaderboard) == 2
    assert "F1 Macro" in leaderboard.columns
    assert failures.empty




def test_get_balanced_tuning_budget():
    budget = get_tuning_budget(
        tuning_mode="balanced"
    )

    assert budget == {
        "number_of_models": 2,
        "n_iter": 20,
    }



def test_count_parameter_combinations():
    parameter_space = {
        "model__max_depth": [
            3,
            5,
            10,
        ],
        "model__n_estimators": [
            100,
            200,
        ],
    }

    number_of_combinations = (
        count_parameter_combinations(
            parameter_space
        )
    )

    assert number_of_combinations == 6


def test_tune_selected_classification_model():
    feature_values, target_values = (
        make_classification(
            n_samples=60,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
    )

    features = pd.DataFrame(
        feature_values,
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ],
    )

    target = pd.Series(
        target_values
    )

    preprocessor = build_preprocessor(
        numerical_columns=features.columns,
        categorical_columns=[],
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000
            )
        )
    }

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models=models,
    )

    parameter_spaces = {
        "Logistic Regression": {
            "model__C": [
                0.1,
                1.0,
            ],
            "model__solver": [
                "lbfgs",
            ],
        }
    }

    cv_strategy = create_cv_strategy(
        target=target,
        task_type="classification",
        maximum_folds=3,
    )

    scoring_configuration = (
        get_scoring_configuration(
            task_type="classification",
            number_of_classes=2,
        )
    )

    (
        tuned_leaderboard,
        tuned_models,
        tuning_failures,
    ) = tune_selected_models(
        selected_models=[
            "Logistic Regression"
        ],
        model_pipelines=pipelines,
        parameter_spaces=parameter_spaces,
        X_train=features,
        y_train=target,
        cv_strategy=cv_strategy,
        scoring_configuration=(
            scoring_configuration
        ),
        task_type="classification",
        n_iter=10,
    )

    assert len(tuned_leaderboard) == 1
    assert "Logistic Regression" in tuned_models
    assert tuning_failures.empty

    assert (
        tuned_leaderboard.loc[
            0,
            "Tuning Iterations",
        ]
        == 2
    )



def test_select_final_model_uses_improved_tuned_model():
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
        [0, 0, 0, 1, 1, 1]
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000
            )
        )
    }

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models=models,
    )

    fitted_tuned_pipeline = clone(
        pipelines[
            "Logistic Regression"
        ]
    )

    fitted_tuned_pipeline.fit(
        features,
        target,
    )

    baseline_leaderboard = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression"
            ],
            "F1 Macro": [0.70],
        }
    )

    tuned_leaderboard = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression"
            ],
            "F1 Macro": [0.80],
        }
    )

    scoring_configuration = (
        get_scoring_configuration(
            task_type="classification",
            number_of_classes=2,
        )
    )

    (
        model_name,
        final_pipeline,
        selection_source,
    ) = select_final_model(
        baseline_leaderboard=(
            baseline_leaderboard
        ),
        model_pipelines=pipelines,
        X_train=features,
        y_train=target,
        scoring_configuration=(
            scoring_configuration
        ),
        tuned_leaderboard=(
            tuned_leaderboard
        ),
        tuned_models={
            "Logistic Regression": (
                fitted_tuned_pipeline
            )
        },
    )

    assert (
        model_name
        == "Logistic Regression"
    )

    assert (
        final_pipeline
        is fitted_tuned_pipeline
    )

    assert selection_source == "tuned"


def test_select_final_model_falls_back_to_baseline():
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
        [0, 0, 0, 1, 1, 1]
    )

    preprocessor = build_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000
            )
        )
    }

    pipelines = create_model_pipelines(
        preprocessor=preprocessor,
        models=models,
    )

    baseline_leaderboard = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression"
            ],
            "F1 Macro": [0.75],
        }
    )

    scoring_configuration = (
        get_scoring_configuration(
            task_type="classification",
            number_of_classes=2,
        )
    )

    (
        model_name,
        final_pipeline,
        selection_source,
    ) = select_final_model(
        baseline_leaderboard=(
            baseline_leaderboard
        ),
        model_pipelines=pipelines,
        X_train=features,
        y_train=target,
        scoring_configuration=(
            scoring_configuration
        ),
        tuned_leaderboard=(
            pd.DataFrame()
        ),
        tuned_models={},
    )

    predictions = final_pipeline.predict(
        features
    )

    assert (
        model_name
        == "Logistic Regression"
    )

    assert selection_source == "baseline"
    assert len(predictions) == len(features)