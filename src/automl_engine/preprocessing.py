"""Preprocessing utilities for numerical and categorical features."""

from collections.abc import Sequence
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# One Hot Encoder Helper
def _create_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible one-hot encoder."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )



def build_preprocessor(
        numerical_columns: Sequence[str],
        categorical_columns: Sequence[str]
) -> ColumnTransformer:
    
    """Build prepocessing pipelines for tabular features"""

    numerical_columns = list(numerical_columns)
    categorical_columns = list(categorical_columns)

    overlapping_columns = (
        set(numerical_columns)
        & set(categorical_columns)
    )

    if overlapping_columns:
        raise ValueError(
            "Columns cannot be both numerical "
            "and categorical: "
            f"{sorted(overlapping_columns)}"
        )
    

    if (
        not numerical_columns
        and not categorical_columns
    ): raise ValueError(
        "At least one numerical or "
        "categorical column is required"
    )


    # build the numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
             ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # build the categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                _create_one_hot_encoder(),
            )
        ]
    )


    # Building the transformer registry
    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            )
        )

    # Pipeline name, Pipeline object, Columns receiving that pipeline

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )