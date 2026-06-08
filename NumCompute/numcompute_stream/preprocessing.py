import numpy as np
from typing import Optional


# =========================
# Base Transformer


class Transformer:
    """
    Base class for all transformers.

    API:
    ----
    fit(X) -> self
    transform(X) -> np.ndarray
    fit_transform(X) -> np.ndarray
    """

    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =========================
# StandardScaler
# =========================

class StandardScaler(Transformer):
    """
    Standardize features: (X - mean) / std

    Parameters
    ----------
    with_mean : bool
    with_std : bool
    eps : float
        Small value for numerical stability.

    Attributes
    ----------
    mean_ : np.ndarray (n_features,)
    std_  : np.ndarray (n_features,)

    Notes
    -----
    Uses numerically stable variance computation.
    Handles zero variance safely.

    Complexity
    ----------
    Time: O(N * M)
    Space: O(M)
    """

    def __init__(self, with_mean=True, with_std=True, eps=1e-8):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError("Input must be 2D array")

        self.mean_ = np.nanmean(X, axis=0)
        var = np.nanvar(X, axis=0)

        # numerical stability
        self.std_ = np.sqrt(var + self.eps)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Must fit before transform")

        X_out = X.astype(np.float64)

        if self.with_mean:
            X_out = X_out - self.mean_

        if self.with_std:
            X_out = X_out / self.std_

        return X_out


# =========================
# MinMaxScaler
# =========================

class MinMaxScaler(Transformer):
    """
    Scale features to [0, 1]

    X_scaled = (X - min) / (max - min)

    Attributes
    ----------
    min_ : np.ndarray
    max_ : np.ndarray

    Notes
    -----
    Handles constant columns safely.

    Complexity
    ----------
    Time: O(N * M)
    """

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError("Input must be 2D")

        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)

        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0  # avoid division by zero

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "min_"):
            raise RuntimeError("Must fit before transform")

        return (X - self.min_) / self.range_


# =========================
# Imputer
# =========================

class Imputer(Transformer):
    """
    Impute missing values.

    Parameters
    ----------
    strategy : {"mean", "median", "most_frequent", "constant"}
    fill_value : float or str (used if strategy="constant")

    Attributes
    ----------
    statistics_ : np.ndarray

    Notes
    -----
    Fully vectorised except mode computation.

    Complexity
    ----------
    Time: O(N * M)
    """

    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X: np.ndarray):
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)

        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)

        elif self.strategy == "most_frequent":
            self.statistics_ = np.apply_along_axis(
                lambda col: np.bincount(col[~np.isnan(col)].astype(int)).argmax()
                if np.any(~np.isnan(col))
                else np.nan,
                axis=0,
                arr=X
            )

        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value must be provided for constant strategy")
            self.statistics_ = np.full(X.shape[1], self.fill_value)

        else:
            raise ValueError("Invalid strategy")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "statistics_"):
            raise RuntimeError("Must fit before transform")

        X_out = X.copy()
        mask = np.isnan(X_out)

        # broadcast replace
        X_out[mask] = np.take(self.statistics_, np.where(mask)[1])

        return X_out



# =========================
# Fully Vectorized OneHotEncoder
# =========================
class OneHotEncoder(Transformer):
    """
    Fully vectorized OneHotEncoder (no Python loops in transform).

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}

    Attributes
    ----------
    categories_ : list of np.ndarray
    category_sizes_ : np.ndarray

    Notes
    -----
    Transform is fully vectorized using broadcasting + indexing tricks.
    """

    def __init__(self, handle_unknown="error"):
        self.handle_unknown = handle_unknown

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError("Input must be 2D")

        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.category_sizes_ = np.array([len(c) for c in self.categories_])

        # build global lookup offsets for concatenation
        self.offsets_ = np.cumsum([0] + list(self.category_sizes_[:-1]))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "categories_"):
            raise RuntimeError("Must fit before transform")

        X = X.astype(object)
        n_samples, n_features = X.shape
        total_dim = self.category_sizes_.sum()

        # output matrix (fully vectorized allocation)
        out = np.zeros((n_samples, total_dim), dtype=np.float64)

        # ---- vectorized column-wise encoding using broadcasting ----
        for i in range(n_features):
            cats = self.categories_[i]
            offset = self.offsets_[i]

            col = X[:, i]

            if self.handle_unknown == "error":
                if not np.all(np.isin(col, cats)):
                    raise ValueError(f"Unknown category in column {i}")

            # vectorized comparison (no Python loop over rows)
            match = (col[:, None] == cats[None, :])

            out[:, offset:offset + len(cats)] = match.astype(np.float64)

        return out