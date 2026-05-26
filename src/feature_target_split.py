def split_features_and_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
