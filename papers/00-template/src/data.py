"""Iris data + the shared, deterministic split.

ONE seeded split, imported by both train.py and evaluate.py, so they always
agree on train/test — no leakage. The split uses an *explicit* random_state
(not the global RNG), so it is order-independent across the two scripts.
"""

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_split(seed: int = 42, test_size: float = 0.2):
    """Return (X_train, y_train, X_test, y_test, in_dim, out_dim).

    X is float32, y is long (CrossEntropy needs long labels). Data reports its
    own in_dim/out_dim so train never hardcodes shapes.
    """
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train = torch.tensor(X_tr, dtype=torch.float32)
    X_test = torch.tensor(X_te, dtype=torch.float32)
    y_train = torch.tensor(y_tr, dtype=torch.long)
    y_test = torch.tensor(y_te, dtype=torch.long)

    in_dim = X_train.shape[1]
    out_dim = int(y.max()) + 1
    return X_train, y_train, X_test, y_test, in_dim, out_dim
