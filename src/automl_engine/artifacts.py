"""Saving and loading utilities for AutoML run artifacts."""

from collections.abc import Mapping
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .evaluation import EvaluationResult



# Creating run directories

def create_run_directory(
    base_directory: str | Path = "automl_runs",
    run_name: str | None = None,
) -> Path:
    """Create a unique directory for an AutoML run."""

    base_path = Path(base_directory)

    base_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if run_name is None:
        run_name = datetime.now().strftime(
            "run_%Y%m%d_%H%M%S_%f"
        )

    if (
        not run_name.strip()
        or Path(run_name).name != run_name
        or run_name in {".", ".."}
    ):
        raise ValueError(
            "run_name must be a valid directory name."
        )

    run_directory = base_path / run_name

    run_directory.mkdir(
        parents=False,
        exist_ok=False,
    )

    return run_directory



def _json_default(value: Any) -> Any:
    """Convert common scientific Python values into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (set, tuple)):
        return list(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )



def save_json_artifact(
    data: Any,
    file_path: str | Path,
) -> Path:
    """Save structured information as a JSON file."""

    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=_json_default,
        )

    return output_path
#Returning the path lets the engine record where every artifact was saved.



def save_dataframe_artifact(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Save a pandas DataFrame as a CSV file."""

    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=index,
    )

    return output_path


def save_joblib_artifact(
    value: Any,
    file_path: str | Path,
) -> Path:
    """Serialize a Python object using joblib."""

    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        value,
        output_path,
    )

    return output_path
# joblib preserves the fitted preprocessing and model objects together.



def load_joblib_artifact(
    file_path: str | Path,
) -> Any:
    """Load a joblib artifact from disk."""

    input_path = Path(file_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {input_path.resolve()}"
        )

    return joblib.load(input_path)

# Only load joblib files you trust because serialized Python files can execute code during loading.



# main artifact-saving function
def save_run_artifacts(
    run_directory: str | Path,
    model_pipeline: Pipeline,
    evaluation_result: EvaluationResult,
    metadata: Mapping[str, Any],
    input_schema: Mapping[str, Any],
    target_encoder: LabelEncoder | None = None,
    column_report: pd.DataFrame | None = None,
    baseline_leaderboard: pd.DataFrame | None = None,
    tuned_leaderboard: pd.DataFrame | None = None,
    baseline_failures: pd.DataFrame | None = None,
    tuning_failures: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Save the outputs produced by one AutoML run."""

    output_directory = Path(run_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts: dict[str, Path] = {}

    # Model pipeline includes Preprocessing & final fitted model
    artifacts["model_pipeline"] = (
        save_joblib_artifact(
            model_pipeline,
            output_directory
            / "model_pipeline.joblib",
        )
    )


    if target_encoder is not None:
        artifacts["target_encoder"] = (
            save_joblib_artifact(
                target_encoder,
                output_directory
                / "target_encoder.joblib",
            )
        )

    artifacts["metadata"] = (
        save_json_artifact(
            dict(metadata),
            output_directory
            / "metadata.json",
        )
    )

    artifacts["input_schema"] = (
        save_json_artifact(
            dict(input_schema),
            output_directory
            / "input_schema.json",
        )
    )

    artifacts["test_metrics"] = (
        save_json_artifact(
            evaluation_result.metrics,
            output_directory
            / "test_metrics.json",
        )
    )

    artifacts["test_predictions"] = (
        save_dataframe_artifact(
            evaluation_result.predictions,
            output_directory
            / "test_predictions.csv",
            index=True,
        )
    )


    # Saving classification only repots
    if (
        evaluation_result
        .classification_report
        is not None
    ):
        artifacts[
            "classification_report"
        ] = save_dataframe_artifact(
            evaluation_result
            .classification_report,
            output_directory
            / "classification_report.csv",
            index=True,
        )

    if (
        evaluation_result
        .confusion_matrix
        is not None
    ):
        artifacts[
            "confusion_matrix"
        ] = save_dataframe_artifact(
            evaluation_result
            .confusion_matrix,
            output_directory
            / "confusion_matrix.csv",
            index=True,
        )


    # Saving the dataset report
    if column_report is not None:
        artifacts["column_report"] = (
            save_dataframe_artifact(
                column_report,
                output_directory
                / "column_report.csv",
                index=True,
            )
        )

    if baseline_leaderboard is not None:
        artifacts[
            "baseline_leaderboard"
        ] = save_dataframe_artifact(
            baseline_leaderboard,
            output_directory
            / "baseline_leaderboard.csv",
        )

    if tuned_leaderboard is not None:
        artifacts[
            "tuned_leaderboard"
        ] = save_dataframe_artifact(
            tuned_leaderboard,
            output_directory
            / "tuned_leaderboard.csv",
        )

    # An empty tuned leaderboard can still be useful because 
    # it shows that tuning was attempted but produced no successful results.


    # Saving failure reports
    if (
        baseline_failures is not None
        and not baseline_failures.empty
    ):
        artifacts[
            "baseline_failures"
        ] = save_dataframe_artifact(
            baseline_failures,
            output_directory
            / "baseline_failures.csv",
        )

    if (
        tuning_failures is not None
        and not tuning_failures.empty
    ):
        artifacts[
            "tuning_failures"
        ] = save_dataframe_artifact(
            tuning_failures,
            output_directory
            / "tuning_failures.csv",
        )


    # The manifest acts as an index of every file created during the run.
    manifest = {
        artifact_name: artifact_path.name
        for artifact_name, artifact_path
        in artifacts.items()
    }

    artifacts["artifact_manifest"] = (
        save_json_artifact(
            manifest,
            output_directory
            / "artifact_manifest.json",
        )
    )

    return artifacts
    

def load_model_artifacts(
    run_directory: str | Path,
) -> tuple[
    Pipeline,
    LabelEncoder | None,
]:
    """Load the fitted pipeline and optional target encoder."""

    input_directory = Path(
        run_directory
    )

    model_path = (
        input_directory
        / "model_pipeline.joblib"
    )

    model_pipeline = load_joblib_artifact(
        model_path
    )

    if not isinstance(
        model_pipeline,
        Pipeline,
    ):
        raise TypeError(
            "model_pipeline.joblib does not "
            "contain a scikit-learn Pipeline."
        )

    encoder_path = (
        input_directory
        / "target_encoder.joblib"
    )

    target_encoder: LabelEncoder | None = None

    if encoder_path.exists():
        loaded_encoder = (
            load_joblib_artifact(
                encoder_path
            )
        )

        if not isinstance(
            loaded_encoder,
            LabelEncoder,
        ):
            raise TypeError(
                "target_encoder.joblib does not "
                "contain a LabelEncoder."
            )

        target_encoder = loaded_encoder

    return (
        model_pipeline,
        target_encoder,
    )


    