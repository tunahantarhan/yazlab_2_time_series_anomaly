from tensorflow.keras.callbacks import EarlyStopping


def train_model(model, X, y):
    early_stopping = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X,
        y,
        epochs=5,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=0
    )

    return history
