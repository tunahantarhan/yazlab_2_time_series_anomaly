def drop_non_feature_columns(df, columns_to_drop):
    existing_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    return df.drop(columns=existing_columns)
