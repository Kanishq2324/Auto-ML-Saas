"""Tests for cross-validation and model evaluation utilities."""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from automl_engine.models import (
    create_model_pipelines,
)
from automl_engine.preprocessing import (
    build_preprocessor,
)
from automl_engine.tuning import (
    create_cv_strategy,
    evaluate_baseline_models,
    get_scoring_configuration,
    select_top_models,
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