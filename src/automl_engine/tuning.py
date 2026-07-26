"""Cross valdiation, model evalutation & tuning utilities"""

from collections.abc import Mapping
import time
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from .config import MAX_CV_FOLDS, RANDOM_STATE



# Cross Validatin Strategy
def create_cv_strategy(
        target: pd.Series,
        task_type: str,
        maximum_folds: int = MAX_CV_FOLDS,
        random_state: int  = RANDOM_STATE
) -> StratifiedKFold | KFold:

    """Create a task-appropriate cross-validation strategy."""

    if target.empty:
        raise ValueError(
            "The target cannot be empty."
        )

    if maximum_folds < 2: 
        raise ValueError(
            "maximum folds should be at least 2."
        )

    if task_type == "classification":
        minimum_class_count = int(
            target.value_counts().min()
        )

        number_of_folds = min(minimum_class_count, maximum_folds)

        if number_of_folds < 2:
            raise ValueError(
                "Cross-validation requires at least "
                "two rows in every target class."
            )

        return StratifiedKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=random_state,
        )

    if task_type == "regression":
        number_of_folds = min(
            maximum_folds, 
            len(target),
        )

        if number_of_folds < 2:
            raise ValueError(
                "Cross-validation requires at least "
                "two target values."
            )

        return KFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=random_state
        )


    raise ValueError(
        f"Unsupported task type: {task_type}"
    )