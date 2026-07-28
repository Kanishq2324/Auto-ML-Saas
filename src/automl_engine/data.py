""""Dataset loading, inspection, cleaning & preparation utilities"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV dataset"""

    file_path = Path(csv_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path.resolve()}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "AutoML Engine only currently supports CSV file only"
        )


    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError("The dataset is empty.")


    # Remove accidental spaces around column names.
    dataframe.columns = dataframe.columns.str.strip()

    if dataframe.columns.duplicated().any():
        duplicated_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names found: {duplicated_columns}"
        )


    return dataframe


# This function is the entry point that converts a CSV file into a validated pandas DataFrame.




def inspect_dataset(dataframe: pd.DataFrame, target_column: str) -> tuple[dict[str, Any], pd.DataFrame]:

    """Generate dataset-level and column-level quality information"""
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    column_report = pd.DataFrame(
        {
            "Data Type": dataframe.dtypes.astype(str),
            "Missing Count": dataframe.isnull().sum(),
            "Missing Percentage": (
                dataframe.isnull().mean() * 100
            ).round(2),
            "Unique Values": dataframe.nunique(dropna=False),
        }
    )

    column_report = column_report.sort_values(
        by = "Missing Percentage",
        ascending=False,
    )

    constant_columns = [
        column 
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=False) <= 1
    ]

    possible_id_columns = []

    for column in dataframe.columns:
        if column == target_column:
            continue

        unique_ratio = (
            dataframe[column].nunique(dropna=False)/len(dataframe)
        )

        if unique_ratio >= 0.95:
            possible_id_columns.append(column)


    dataset_summary = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "target_column": target_column,
        "target_unique_values": int(
            dataframe[target_column].nunique(dropna=True)
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "total_missing_values": int(
            dataframe.isnull().sum().sum()
        ),
        "constant_columns": constant_columns,
        "possible_id_columns": possible_id_columns,
    }


    return dataset_summary, column_report

# Inspection describes the data but does not modify it.




def clean_dataset(
    dataframe: pd.DataFrame, 
    target_column: str, 
    columns_to_drop: list[str] | None = None,
    remove_duplicates: bool = True,
) -> tuple[pd.DataFrame, list[str]]:\

    """Clean safe, general dataset issues and record every action"""

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    cleaned_dataframe = dataframe.copy()
    cleaning_actions = []

    if remove_duplicates:
        duplicate_count = int(cleaned_dataframe.duplicated().sum())

        if duplicate_count > 0:
            cleaned_dataframe = (
                cleaned_dataframe.drop_duplicates().reset_index(drop = True)
            )

            cleaning_actions.append(
                f"Removed {duplicate_count} duplicate rows."
            )


    missing_target_count = int(cleaned_dataframe[target_column].isnull().sum())

    if missing_target_count > 0:
        cleaned_dataframe = (
            cleaned_dataframe.dropna(subset=[target_column])
            .reset_index(drop = True)
        )

        cleaning_actions.append(
            f"Removed {missing_target_count} rows"
            "with missing target values"
        )


    constant_columns = [
        column
        for column in cleaned_dataframe.columns
        if (
            column != target_column
            and cleaned_dataframe[column]
            .nunique(dropna=False) <= 1
        )
    ]


    if constant_columns:
        cleaned_dataframe = cleaned_dataframe.drop(
            columns=constant_columns
        )

        cleaning_actions.append(
            f"Removed constant columns: {constant_columns}"
        )

    if columns_to_drop:
        valid_columns_to_drop = [
            column
            for column in columns_to_drop
            if (
                column in cleaned_dataframe.columns
                and column != target_column
            )
        ]

        if valid_columns_to_drop:
            cleaned_dataframe = cleaned_dataframe.drop(
                columns=valid_columns_to_drop
            )

            cleaning_actions.append(
                "Removed manually selected columns: "
                f"{valid_columns_to_drop}"
            )

    if cleaned_dataframe.empty:
        raise ValueError(
            "No rows remain after dataset cleaning."
        )

    if not cleaning_actions:
        cleaning_actions.append(
            "No row or column cleaning was required."
        )

    return cleaned_dataframe, cleaning_actions


# Cleaning changes the data, so every change is recorded for transparency.




def separate_features_and_target(
        dataframe: pd.DataFrame,
        target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:

    """Seperate model inputs from the prediction target"""

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column].copy()

    if features.shape[1]== 0:
        raise ValueError(
            "No feature columns remain after removing the target"
        )


    return features, target
   
# X contains inputs and y contains the answer, the model learns to predict.



def detect_task_type(
        target: pd.Series,
        classification_unique_limit: int = 20,
        classification_ratio_limit: float = 0.05,
) -> str:

    """Estimate whether the target represents classification & regression"""

    if target.empty:
        raise ValueError("the target column is empty.")


    unique_count = int(target.nunique(dropna=True))
    unique_ratio = unique_count/len(target)

    is_text = (
        target.dtype == "object"
        or str(target.dtype).startswith("category")
    )

    is_boolean = target.dtype == bool

    if(
        is_text
        or is_boolean
        or unique_count <= classification_unique_limit
        or unique_ratio <= classification_ratio_limit
    ):
        return "classification"

    return "regression"

# Task detection is a heuristic, so users will later be allowed to override it.




def prepare_target(
    target: pd.Series,
    task_type: str,
) -> tuple[pd.Series, LabelEncoder | None, dict[str, Any]]:
    
    """Encode classification targets or validate regression targets."""

    if task_type == "classification":
        target_encoder = LabelEncoder()

        encoded_values = target_encoder.fit_transform(
            target
        )

        prepared_target = pd.Series(
            encoded_values,
            index=target.index,
            name=target.name,
        )

        class_mapping = {
            str(class_name): int(encoded_value)
            for class_name, encoded_value in zip(
                target_encoder.classes_,
                target_encoder.transform(
                    target_encoder.classes_
                ),
            )
        }

        target_information = {
            "number_of_classes": int(
                len(target_encoder.classes_)
            ),
            "classes": target_encoder.classes_.tolist(),
            "class_mapping": class_mapping,
        }

        return (
            prepared_target,
            target_encoder,
            target_information,
        )

    if task_type == "regression":
        prepared_target = pd.to_numeric(
            target,
            errors="coerce",
        )

        invalid_count = int(
            prepared_target.isnull().sum()
        )

        if invalid_count > 0:
            raise ValueError(
                "Regression target contains "
                f"{invalid_count} non-numeric or missing values."
            )

        target_information = {
            "minimum": float(prepared_target.min()),
            "maximum": float(prepared_target.max()),
            "mean": float(prepared_target.mean()),
            "median": float(prepared_target.median()),
        }

        return (
            prepared_target,
            None,
            target_information,
        )

    raise ValueError(
        f"Unsupported task type: {task_type}"
    )

# Classification labels become integer IDs, while regression targets remain continuous values.



def detect_feature_types(
    dataframe: pd.DataFrame,
    categorical_unique_limit: int = 15,
    categorical_ratio_limit: float = 0.05,
) -> tuple[list[str], list[str]]:
    """Detect numerical and categorical feature columns."""

    numerical_columns = []
    categorical_columns = []

    for column in dataframe.columns:
        series = dataframe[column]

        unique_count = int(series.nunique(dropna=True))
        unique_ratio = (unique_count/max(len(series), 1))

        is_text = (
            series.dtype == "object"
            or str(series.dtype).startswith("category")
        )

        is_boolean = series.dtype == bool

        is_low_cardinality_numeric = (
            pd.api.types.is_numeric_dtype(series)
            and unique_count <= categorical_unique_limit
            and unique_ratio <= categorical_ratio_limit
        )

        if(
            is_text
            or is_boolean
            or is_low_cardinality_numeric
        ): 
            categorical_columns.append(column)

        else: numerical_columns.append(column)


    detected_columns = set(numerical_columns + categorical_columns)

    missing_columns = (set(dataframe.columns) - detected_columns)

    if missing_columns:
        raise ValueError(
            "Could not detect feature types for: "
            f"{sorted(missing_columns)}"
        )

    return numerical_columns, categorical_columns

# Stored data types are not always semantic types; numeric codes such as 1 and 2 may represent categories.





### Train_test_split split function
def split_dataset(
        features: pd.DataFrame,
        target: pd.Series,
        task_type: str,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE
        
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:

    """Creating reproducible training & test datasets."""

    stratify_target = None

    if task_type == "classification":
        class_counts = target.value_counts()

        if class_counts.min() < 2:
            raise ValueError(
                "Each target class requires at least two rows."
            )

        stratify_target = target

    elif task_type != "regression":
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )


    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

# The test set remains untouched until final model evaluation.





