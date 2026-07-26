"""Candidate model registry and model-pipeline utilities."""

from collections.abc import Mapping
from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from .config import RANDOM_STATE



def get_candidate_models(
        task_type: str,
        number_of_classes: int | None = None,
        number_of_rows: int | None = None,
        random_state: int = RANDOM_STATE,
) -> dict[str, BaseEstimator]:

    """Return candidate models for classification & regression"""
    if (
        number_of_rows is not None
        and number_of_rows <= 0
    ): 
        raise ValueError(
            "number_of_rows must be greater than zero."
        )


    if task_type == "classification":
        if(
            number_of_classes is None
            or number_of_classes < 2
        ): 
            raise ValueError(
                "Classification requires at least "
                "two target classes"
            )


        models: dict[str, BaseEstimator] = {
            "Dummy Classifier": DummyClassifier(
                strategy="most_frequent",
            ),

            "Logistic Regression": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1
            ),

            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1
            )
        }

        if (
            number_of_rows is None
            or number_of_rows <= 50000
        ):
            models["KNN"] = KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform"
            )

        if number_of_classes == 2:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
            )
        else:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=number_of_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
            )

        return models


    if task_type == "regression":
        models = {
            "Dummy Regressor": DummyRegressor(
                strategy="mean"
            ),

            "Ridge Regression": Ridge(),

            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=1,
            ),

            "Extra Trees": ExtraTreesRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=1,
            ),

            "XGBoost": XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
            ),
        }

        if (
            number_of_rows is None
            or number_of_rows <= 50_000
        ):
            models["KNN"] = KNeighborsRegressor(
                n_neighbors=5,
                weights="uniform",
            )

        return models

    raise ValueError(
        f"Unsupported task type: {task_type}"
    )




def create_model_pipelines(
        preprocessor: ColumnTransformer,
        models: Mapping[str, BaseEstimator],
) -> dict[str, Pipeline]:

    """Combine every candidate model with prepocessing"""

    if not models:
        raise ValueError(
            "At least one candidate model is required."
        )

    model_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in models.items():
        if not model_name.strip():
            raise ValueError(
                "Model names cannot be empty."
            )

        model_pipelines[model_name] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor)
                ),
                (
                    "model",
                    clone(estimator)
                )
            ]
        )


    return model_pipelines



def get_parameter_spaces(
        task_type: str,
) -> dict[str, dict[str, list[Any]]]:

        if task_type == "classification":
            return {
                "Logistic Regression": {
                    "model__C": [
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                        100.0,
                    ],
                    "model__class_weight": [
                        None,
                        "balanced",
                    ],
                    "model__solver": [
                        "lbfgs",
                        "liblinear",
                    ],
                },

                "KNN": {
                    "model__n_neighbors": list(
                        range(3, 32, 2)
                    ),
                    "model__weights": [
                        "uniform",
                        "distance",
                    ],
                    "model__p": [
                        1,
                        2,
                    ],
                },


                "Random Forest": {
                    "model__n_estimators": [
                        100,
                        200,
                        300,
                        500,
                    ],
                    "model__max_depth": [
                        None,
                        3,
                        5,
                        10,
                        20,
                    ],
                    "model__min_samples_split": [
                        2,
                        5,
                        10,
                    ],
                    "model__min_samples_leaf": [
                        1,
                        2,
                        4,
                    ],
                    "model__max_features": [
                        "sqrt",
                        "log2",
                        None,
                    ],
                    "model__class_weight": [
                        None,
                        "balanced",
                        "balanced_subsample",
                    ],
                },


                "Extra Trees": {
                    "model__n_estimators": [
                        100,
                        200,
                        300,
                        500,
                    ],
                    "model__max_depth": [
                        None,
                        3,
                        5,
                        10,
                        20,
                    ],
                    "model__min_samples_split": [
                        2,
                        5,
                        10,
                    ],
                    "model__min_samples_leaf": [
                        1,
                        2,
                        4,
                    ],
                    "model__max_features": [
                        "sqrt",
                        "log2",
                        None,
                    ],
                    "model__class_weight": [
                        None,
                        "balanced",
                        "balanced_subsample",
                    ],
                },


                "XGBoost": {
                    "model__n_estimators": [
                        100,
                        200,
                        300,
                        500,
                    ],
                    "model__max_depth": [
                        2,
                        3,
                        4,
                        5,
                        6,
                    ],
                    "model__learning_rate": [
                        0.01,
                        0.03,
                        0.05,
                        0.1,
                        0.2,
                    ],
                    "model__subsample": [
                        0.6,
                        0.8,
                        1.0,
                    ],
                    "model__colsample_bytree": [
                        0.6,
                        0.8,
                        1.0,
                    ],
                    "model__min_child_weight": [
                        1,
                        3,
                        5,
                    ],
                    "model__reg_alpha": [
                        0.0,
                        0.01,
                        0.1,
                        1.0,
                    ],
                    "model__reg_lambda": [
                        0.1,
                        1.0,
                        5.0,
                        10.0,
                    ],
                },
            }

        if task_type == "regression":
            return {
                "Ridge Regression": {
                    "model__alpha": [
                        0.001,
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                        100.0,
                    ],
                    "model__solver": [
                        "auto",
                        "lsqr",
                    ],
                },

                "KNN": {
                    "model__n_neighbors": list(
                        range(3, 32, 2)
                    ),
                    "model__weights": [
                        "uniform",
                        "distance",
                    ],
                    "model__p": [
                        1,
                        2,
                    ],
                },


                "Random Forest": {
                    "model__n_estimators": [
                        100,
                        200,
                        300,
                        500,
                    ],
                    "model__max_depth": [
                        None,
                        3,
                        5,
                        10,
                        20,
                    ],
                    "model__min_samples_split": [
                        2,
                        5,
                        10,
                    ],
                    "model__min_samples_leaf": [
                        1,
                        2,
                        4,
                    ],
                    "model__max_features": [
                        "sqrt",
                        "log2",
                        None,
                    ],
                },


                "Extra Trees": {
                    "model__n_estimators": [
                        100,
                        200,
                        300,
                        500,
                    ],
                    "model__max_depth": [
                        None,
                        3,
                        5,
                        10,
                        20,
                    ],
                    "model__min_samples_split": [
                        2,
                        5,
                        10,
                    ],
                    "model__min_samples_leaf": [
                        1,
                        2,
                        4,
                    ],
                    "model__max_features": [
                        "sqrt",
                        "log2",
                        None,
                    ],
                },

                "XGBoost": {
                    "model__n_estimators": [
                        100,
                        200,
                        300,
                        500,
                    ],
                    "model__max_depth": [
                        2,
                        3,
                        4,
                        5,
                        6,
                    ],
                    "model__learning_rate": [
                        0.01,
                        0.03,
                        0.05,
                        0.1,
                        0.2,
                    ],
                    "model__subsample": [
                        0.6,
                        0.8,
                        1.0,
                    ],
                    "model__colsample_bytree": [
                        0.6,
                        0.8,
                        1.0,
                    ],
                    "model__min_child_weight": [
                        1,
                        3,
                        5,
                    ],
                    "model__reg_alpha": [
                        0.0,
                        0.01,
                        0.1,
                        1.0,
                    ],
                    "model__reg_lambda": [
                        0.1,
                        1.0,
                        5.0,
                        10.0,
                    ],
                },
            }


        raise ValueError(
            f"Unsuppored task type: {task_type}"
        )

    