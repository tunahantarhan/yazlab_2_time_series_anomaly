from tensorflow.keras.callbacks import EarlyStopping


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=50,
    batch_size=32,
    patience=5
):
    early_stopping = EarlyStopping(
        monitor="val_loss" if X_val is not None and y_val is not None else "loss",
        patience=patience,
        restore_best_weights=True
    )

    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": [early_stopping],
        "verbose": 0
    }

    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)

    history = model.fit(**fit_kwargs)

    return history
