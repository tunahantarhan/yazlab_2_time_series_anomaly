from src.feature_filter import drop_non_feature_columns
from src.feature_target_split import split_features_and_target
from src.dataset_split import split_dataset
from src.normalizer import normalize_data


def prepare_dataset(
    df,
    target_column,
    columns_to_drop=None
):
    if columns_to_drop is None:
        columns_to_drop = []

    df = drop_non_feature_columns(
        df,
        columns_to_drop
    )

    X, y = split_features_and_target(
        df,
        target_column
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)

    (
        X_train,
        X_val,
        X_test
    ) = normalize_data(
        X_train,
        X_val,
        X_test
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
