import numpy as np
from typing import Optional

def _validate_2d_numeric(X, name="X"):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return X

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
        X = _validate_2d_numeric(X)

        self.n_samples_seen_ = np.zeros(X.shape[1], dtype=float)
        self.mean_ = np.zeros(X.shape[1], dtype=float)
        self._M2 = np.zeros(X.shape[1], dtype=float)

        return self.partial_fit(X)
    
    def partial_fit(self, X: np.ndarray) -> "StandardScaler":
        X = _validate_2d_numeric(X)

        m_per_col  = np.sum(~np.isnan(X), axis=0).astype(float)
        valid_cols = m_per_col > 0

        chunk_mean = np.zeros(X.shape[1])
        chunk_var  = np.zeros(X.shape[1])
        if valid_cols.any():
            chunk_mean[valid_cols] = np.nanmean(X[:, valid_cols], axis=0)
            chunk_var[valid_cols]  = np.nanvar(X[:, valid_cols], axis=0)

        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = np.zeros(X.shape[1])
            self.mean_           = np.zeros(X.shape[1])
            self._M2             = np.zeros(X.shape[1])

        old_n  = self.n_samples_seen_.copy()
        new_n  = old_n + m_per_col
        active = new_n > 0

        delta      = np.where(active, chunk_mean - self.mean_, 0.0)
        safe_n     = np.where(active, new_n, 1.0)
        self.mean_ = np.where(active, (old_n * self.mean_ + m_per_col * chunk_mean) / safe_n, self.mean_)
        self._M2   = np.where(active, self._M2 + chunk_var * m_per_col + delta**2 * old_n * m_per_col / safe_n, self._M2)
        self.n_samples_seen_ = new_n

        safe_seen  = np.where(self.n_samples_seen_ > 0, self.n_samples_seen_, 1.0)
        variance   = np.where(self.n_samples_seen_ > 0, self._M2 / safe_seen, 0.0)
        self.std_  = np.sqrt(variance + self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        
        X = _validate_2d_numeric(X)
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
        X = _validate_2d_numeric(X)

        self.min_ = np.full(X.shape[1], np.inf)
        self.max_ = np.full(X.shape[1], -np.inf)

        return self.partial_fit(X)


    def partial_fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = _validate_2d_numeric(X)

        if not hasattr(self, "min_"):
            self.min_ = np.full(X.shape[1], np.inf)
            self.max_ = np.full(X.shape[1], -np.inf)

        if X.shape[1] != self.min_.shape[0]:
            raise ValueError("Number of features changed between partial_fit calls")

        valid_counts = np.sum(~np.isnan(X), axis=0)
        valid_cols = valid_counts > 0

        chunk_min = np.full(X.shape[1], np.inf)
        chunk_max = np.full(X.shape[1], -np.inf)

        chunk_min[valid_cols] = np.nanmin(X[:, valid_cols], axis=0)
        chunk_max[valid_cols] = np.nanmax(X[:, valid_cols], axis=0)

        self.min_[valid_cols] = np.minimum(self.min_[valid_cols], chunk_min[valid_cols])
        self.max_[valid_cols] = np.maximum(self.max_[valid_cols], chunk_max[valid_cols])

        unseen = ~np.isfinite(self.min_) | ~np.isfinite(self.max_)
        self.min_[unseen] = 0.0
        self.max_[unseen] = 0.0

        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _validate_2d_numeric(X)
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
        X = _validate_2d_numeric(X)
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
    
    def partial_fit(self, X: np.ndarray) -> "Imputer":
        X = _validate_2d_numeric(X)

        if self.strategy == "mean":
            valid_counts = np.sum(~np.isnan(X), axis=0).astype(float)
            valid_cols = valid_counts > 0

            chunk_mean = np.zeros(X.shape[1], dtype=float)
            chunk_mean[valid_cols] = np.nanmean(X[:, valid_cols], axis=0)

            if not hasattr(self, "n_samples_seen_"):
                self.n_samples_seen_ = np.zeros(X.shape[1], dtype=float)
                self.statistics_ = np.zeros(X.shape[1], dtype=float)

            if X.shape[1] != self.statistics_.shape[0]:
                raise ValueError("Number of features changed between partial_fit calls")

            old_n = self.n_samples_seen_
            new_n = old_n + valid_counts

            update_cols = valid_counts > 0

            self.statistics_[update_cols] = (
                old_n[update_cols] * self.statistics_[update_cols]
                + valid_counts[update_cols] * chunk_mean[update_cols]
            ) / new_n[update_cols]

            self.n_samples_seen_ = new_n

        elif self.strategy in {"median", "most_frequent"}:
            if not hasattr(self, "_buffer"):
                self._buffer = X.copy()
            else:
                if X.shape[1] != self._buffer.shape[1]:
                    raise ValueError("Number of features changed between partial_fit calls")
                self._buffer = np.vstack([self._buffer, X])

            if self.strategy == "median":
                self.statistics_ = np.zeros(self._buffer.shape[1], dtype=float)
                valid_counts = np.sum(~np.isnan(self._buffer), axis=0)
                valid_cols = valid_counts > 0
                self.statistics_[valid_cols] = np.nanmedian(self._buffer[:, valid_cols], axis=0)

            else:
                stats = []
                for j in range(self._buffer.shape[1]):
                    col = self._buffer[:, j]
                    col = col[~np.isnan(col)]

                    if col.size == 0:
                        stats.append(0.0)
                    else:
                        values, counts = np.unique(col, return_counts=True)
                        stats.append(values[np.argmax(counts)])

                self.statistics_ = np.asarray(stats, dtype=float)

        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value required for constant strategy")
            self.statistics_ = np.full(X.shape[1], self.fill_value)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

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
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError("Input must be 2D")

        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.category_sizes_ = np.array([len(c) for c in self.categories_])

        # build global lookup offsets for concatenation
        self.offsets_ = np.cumsum([0] + list(self.category_sizes_[:-1]))

        return self
    
    def partial_fit(self, X: np.ndarray) -> "OneHotEncoder":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input must be 2D")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
      
        new_cats = [np.unique(X[:, i]) for i in range(X.shape[1])]
        if not hasattr(self, 'categories_'):
            self.categories_ = new_cats
        else:
            if len(new_cats) != len(self.categories_):
                raise ValueError("Number of features changed between partial_fit calls")
            self.categories_ = [
                np.union1d(old, new) for old, new in zip(self.categories_, new_cats)
            ]
        self.category_sizes_ = np.array([len(c) for c in self.categories_])
        self.offsets_        = np.cumsum([0] + list(self.category_sizes_[:-1]))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        
        if not hasattr(self, "categories_"):          # ← move this FIRST
            raise RuntimeError("Must fit before transform")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input must be 2D")
        if X.shape[1] != len(self.categories_):
            raise ValueError("Number of features does not match fitted data")
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