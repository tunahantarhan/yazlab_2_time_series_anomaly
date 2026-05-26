from sklearn.model_selection import KFold


def create_kfold_splits(
    X,
    y,
    n_splits=5,
    shuffle=True,
    random_state=42
):
    kfold = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    return list(
        kfold.split(X, y)
    )
