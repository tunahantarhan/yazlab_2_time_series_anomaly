def split_features_and_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
