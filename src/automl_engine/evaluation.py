"""Final model evaluation utilities."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report as sklearn_classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline



@dataclass(frozen=True)
class EvaluationResult:

    """Store final model evaluation outputs."""

    task_type: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    classification_report: pd.DataFrame | None = None
    confusion_matrix: pd.DataFrame | None = None




def evaluate_final_model(
        model_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        class_names: Sequence[str] | None = None,
) -> EvaluationResult:

    """Evaluate a fitted pipeline on untouched test data"""

    # Input validation
    if X_test.empty:
        raise ValueError(
            "X_test cannot be empty."
        )

    if y_test.empty:
        raise ValueError(
            "y_test cannot be empty."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test must contain "
            "the same number of rows."
        )

    if task_type not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )


    predicted_values = np.asarray(
        model_pipeline.predict(X_test)
    )

    if task_type == "classification":
        # internal helper function

        return _evaluate_classification_model(
            model_pipeline = model_pipeline,
            X_test = X_test,
            y_test = y_test,
            predicted_values = predicted_values,
            class_names = class_names,
        )

    return _evaluate_regression_model(
        y_test = y_test,
        predicted_values = predicted_values
    )


def _evaluate_classification_model(
        model_pipeline : Pipeline,
        X_test : pd.DataFrame, 
        y_test : pd.Series,
        predicted_values : np.ndarray,
        class_names : Sequence[str] | None,
) -> EvaluationResult:

    """Calculate classification metrics and reports"""

    metrics = {
        "Accuracy": float(
            accuracy_score(
                y_test, predicted_values
            )
        ),

        "Balanced Accuracy": float(
            balanced_accuracy_score(
                y_test, predicted_values
            )
        ),

        "Precision Macro": float(
            precision_score(
                y_test, 
                predicted_values,
                average="macro",
                zero_division=0,
            )
        ),

        "Recall Macro": float(
            recall_score(
                y_test,
                predicted_values,
                average="macro",
                zero_division=0
            )
        ),

        "F1 Macro": float(
            f1_score(
                y_test,
                predicted_values,
                average="macro",
                zero_division=0
            )
        )
    }


    combined_labels = pd.concat(
        [
            y_test.reset_index(drop=True),
            pd.Series(predicted_values),
        ],
        ignore_index=True
    )

    labels = combined_labels.unique().tolist()

    if class_names is not None:
        class_names = list(class_names)

        if len(class_names) != len(labels):
            raise ValueError(
                "class_names must contain one name "
                "for every class."
            )

        display_names = class_names

    else:
        display_names = [
            str(label)
            for label in labels
        ]


    probability: np.ndarray | None = None

    if hasattr(
        model_pipeline,
        "predict_proba",
    ):
        probabilities = np.asarray(
            model_pipeline.predict_proba(X_test)
        )


    # Calculatin ROC AUC
    if probabilities is not None:
        try:
            if probabilities.shape[1] == 2:
                metrics["ROC-AUC"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities[:, 1],
                    )
                )

            elif probabilities.shape[1] > 2:
                metrics["ROC-AUC"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                    )
                )

        except ValueError:
            pass


    prediction_table = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": predicted_values,
        },
        index=X_test.index,
    )

    if probabilities is not None:
        for class_index in range(
            probabilities.shape[1]
        ):
            probability_name = (
                display_names[class_index]
                if class_index < len(display_names)
                else str(class_index)
            )

            prediction_table[
                f"Probability_{probability_name}"
            ] = probabilities[
                :,
                class_index,
            ]


    report_dictionary = (
        sklearn_classification_report(
            y_true=y_test,
            y_pred=predicted_values,
            labels=labels,
            target_names=display_names,
            output_dict=True,
            zero_division=0,
        )
    )

    report_dataframe = (
        pd.DataFrame(
            report_dictionary
        ).transpose()
    )

    matrix_values = (
        sklearn_confusion_matrix(
            y_true=y_test,
            y_pred=predicted_values,
            labels=labels,
        )
    )

    matrix_dataframe = pd.DataFrame(
        matrix_values,
        index=[
            f"Actual_{name}"
            for name in display_names
        ],
        columns=[
            f"Predicted_{name}"
            for name in display_names
        ],
    )
    # The confusion matrix shows exactly which classes were confused with each other.


    rounded_metrics = {
        metric_name: round(
            metric_value,
            4,
        )
        for metric_name, metric_value
        in metrics.items()
    }

    return EvaluationResult(
        task_type="classification",
        metrics=rounded_metrics,
        predictions=prediction_table,
        classification_report=(
            report_dataframe
        ),
        confusion_matrix=(
            matrix_dataframe
        ),
    )


def _evaluate_regression_model(
        y_test: pd.Series,
        predicted_values: np.ndarray,
) -> EvaluationResult:

    """Calculate regression metrics and residuals."""

    mean_squared_error_value = (
        mean_squared_error(
            y_test,
            predicted_values,
        )
    )

    metrics = {
        "MAE": float(
            mean_absolute_error(
                y_test,
                predicted_values,
            )
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error_value
            )
        ),

        "R2": float(
            r2_score(
                y_test,
                predicted_values,
            )
        ),
    }

    prediction_table = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": predicted_values,
        },
        index=y_test.index,
    )


    prediction_table["Residual"] = (
        prediction_table["Actual"] - prediction_table["Predicted"]
    )

    prediction_table["Absolute Error"] = (
        prediction_table["Residual"].abs()
    )


    rounded_metrics = {
        metric_name: round(
            metric_value,
            4,
        )
        for metric_name, metric_value
        in metrics.items()
    }

    return EvaluationResult(
        task_type="regression",
        metrics=rounded_metrics,
        predictions=prediction_table,
    )