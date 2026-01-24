import numpy as np
from src.constraints import add_gaussian_noise
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_evaluate(random_state: int = 42) -> float:
    """
    Baseline model under ideal conditions (no constraints yet).
    Returns accuracy on a held-out test set.
    """
    X, y = make_classification(
        n_samples=1500,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        flip_y=0.0,          # no label noise for baseline
        class_sep=2.0,       # makes classes more separable -> higher accuracy
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def train_model(random_state: int = 42):
    """
    Train the baseline model once and return model + test data.
    """
    X, y = make_classification(
        n_samples=1500,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        flip_y=0.0,
        class_sep=2.0,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_with_noise(std: float, seed: int = 0, random_state: int = 42) -> float:
    """
    Evaluate accuracy after applying physics-based sensor noise.
    """
    model, X_test, y_test = train_model(random_state)
    X_noisy = add_gaussian_noise(np.array(X_test), std=std, seed=seed)
    preds = model.predict(X_noisy)
    return accuracy_score(y_test, preds)