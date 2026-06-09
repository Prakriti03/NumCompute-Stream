
import numpy as np
from typing import List, Optional, Union

from numcompute_stream.tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: Optional[Union[int, str]] = "sqrt",
        criterion: str = "gini",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")

        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'")

        if max_features not in {None, "sqrt", "log2"} and not isinstance(max_features, int):
            raise ValueError("max_features must be None, 'sqrt', 'log2', or an integer")

        if not isinstance(bootstrap, bool):
            raise ValueError("bootstrap must be True or False")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: List[DecisionTreeClassifier] = []
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self._is_fitted = False

        self._X_seen: Optional[np.ndarray] = None
        self._y_seen: Optional[np.ndarray] = None

    def _validate_input(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y differ in length: {X.shape[0]} vs {y.shape[0]}"
            )

        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        return X, y

    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or len(self.trees_) == 0:
            raise RuntimeError("Forest must be fitted before prediction.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        return X

    def _bootstrap_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: Optional[int],
    ):
        if not self.bootstrap:
            return X, y

        rng = np.random.default_rng(seed)
        indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)

        return X[indices], y[indices]

    def _build_trees(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trees_ = []

        for i in range(self.n_estimators):
            seed = None if self.random_state is None else self.random_state + i

            X_sample, y_sample = self._bootstrap_sample(X, y, seed)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=seed,
            )

            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X, y = self._validate_input(X, y)

        valid_mask = ~np.isnan(y)
        if valid_mask.sum() == 0:
            raise ValueError("y must contain at least one non-NaN label")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y[valid_mask]).astype(int)
        self.n_classes_ = len(self.classes_)

        self._X_seen = X.copy()
        self._y_seen = y.copy()

        self._build_trees(X, y)

        self._is_fitted = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "RandomForestClassifier":
        X, y = self._validate_input(X, y)

        if not self._is_fitted:
            self.n_features_in_ = X.shape[1]

            if classes is not None:
                self.classes_ = np.asarray(classes, dtype=int)
            else:
                valid_mask = ~np.isnan(y)
                if valid_mask.sum() == 0:
                    raise ValueError("y must contain at least one non-NaN label")
                self.classes_ = np.unique(y[valid_mask]).astype(int)

            self.n_classes_ = len(self.classes_)

        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, expected {self.n_features_in_}"
                )

            if classes is not None:
                self.classes_ = np.asarray(classes, dtype=int)
            else:
                valid_mask = ~np.isnan(y)
                new_classes = np.unique(y[valid_mask]).astype(int)
                self.classes_ = np.unique(
                    np.concatenate([self.classes_, new_classes])
                )

            self.n_classes_ = len(self.classes_)

        if self._X_seen is None:
            self._X_seen = X.copy()
            self._y_seen = y.copy()
        else:
            self._X_seen = np.vstack([self._X_seen, X])
            self._y_seen = np.concatenate([self._y_seen, y])

        self._build_trees(self._X_seen, self._y_seen)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_predict_input(X)

        # (n_trees, n_samples) votes
        votes = np.asarray([tree.predict(X) for tree in self.trees_], dtype=int)

        # map each vote to its column index in classes_ (votes outside classes_ are masked out)
        sorter = np.argsort(self.classes_)
        idx = np.searchsorted(self.classes_, votes, sorter=sorter)
        idx = np.clip(idx, 0, len(self.classes_) - 1)
        valid = self.classes_[idx] == votes

        # accumulate per-sample class counts, then argmax (ties -> lowest class index)
        counts = np.zeros((X.shape[0], len(self.classes_)), dtype=int)
        sample_ids = np.broadcast_to(np.arange(X.shape[0]), votes.shape)
        np.add.at(counts, (sample_ids[valid], idx[valid]), 1)

        return self.classes_[np.argmax(counts, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_predict_input(X)

        probabilities = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)

        forest_class_to_index = {
            int(cls): idx for idx, cls in enumerate(self.classes_)
        }

        for tree in self.trees_:
            tree_proba = tree.predict_proba(X)

            for tree_idx, tree_class in enumerate(tree.classes_):
                forest_idx = forest_class_to_index.get(int(tree_class))
                if forest_idx is not None:
                    probabilities[:, forest_idx] += tree_proba[:, tree_idx]

        probabilities /= len(self.trees_)

        row_sums = probabilities.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze() == 0

        if np.any(zero_rows):
            probabilities[zero_rows, :] = 1.0 / len(self.classes_)
            row_sums = probabilities.sum(axis=1, keepdims=True)

        return probabilities / row_sums

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X = self._validate_predict_input(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y differ in length: {X.shape[0]} vs {y.shape[0]}"
            )

        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def get_feature_importances(self) -> np.ndarray:
        if not self._is_fitted or len(self.trees_) == 0:
            raise RuntimeError("Forest must be fitted first.")

        importances = np.zeros(self.n_features_in_, dtype=float)

        for tree in self.trees_:
            counts = np.zeros(self.n_features_in_, dtype=float)

            def walk(node):
                if node is None or node.is_leaf:
                    return
                counts[node.feature] += 1
                walk(node.left)
                walk(node.right)

            walk(tree.tree_)
            importances += counts

        total = importances.sum()

        if total == 0:
            return importances

        return importances / total

    def get_n_trees(self) -> int:
        return len(self.trees_)

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "criterion": self.criterion,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "RandomForestClassifier":
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)

        return self

    def __repr__(self) -> str:
        return (
            "RandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"max_features={self.max_features}, "
            f"criterion='{self.criterion}', "
            f"bootstrap={self.bootstrap})"
        )