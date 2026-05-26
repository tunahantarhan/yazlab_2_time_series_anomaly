from sklearn.preprocessing import StandardScaler


def normalize_data(
    X_train,
    X_val,
    X_test
):
    scaler = StandardScaler()

    X_train_norm = scaler.fit_transform(
        X_train
    )

    X_val_norm = scaler.transform(
        X_val
    )

    X_test_norm = scaler.transform(
        X_test
    )

    return (
        X_train_norm,
        X_val_norm,
        X_test_norm
    )
