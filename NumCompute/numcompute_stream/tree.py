import numpy as np
from typing import Optional, Tuple, Union


EPS = 1e-15
MAX_DEPTH_DEFAULT = 10
MIN_SAMPLES_LEAF_DEFAULT = 1
MIN_SAMPLES_SPLIT_DEFAULT = 2


class TreeNode:
    def __init__(self, depth: int = 0):
        self.depth = depth
        self.is_leaf = True
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.value: Optional[int] = None
        self.samples = 0
        self.impurity = 0.0
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None
        self.class_counts = {}

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"LeafNode(class={self.value}, samples={self.samples})"
        return (
            f"Node(feature={self.feature}, "
            f"threshold={self.threshold:.4f}, samples={self.samples})"
        )


def _gini_impurity(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0

    valid_mask = ~np.isnan(y)
    if valid_mask.sum() == 0:
        return 0.0

    y_valid = y[valid_mask].astype(int)
    counts = np.bincount(y_valid)
    proportions = counts / counts.sum()

    return float(1.0 - np.sum(proportions ** 2))


def _entropy_impurity(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0

    valid_mask = ~np.isnan(y)
    if valid_mask.sum() == 0:
        return 0.0

    y_valid = y[valid_mask].astype(int)
    counts = np.bincount(y_valid)
    probabilities = counts[counts > 0] / counts.sum()

    return float(-np.sum(probabilities * np.log2(probabilities)))


def _information_gain(
    y_parent: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    criterion: str = "gini",
) -> float:
    n_parent = y_parent.size
    if n_parent == 0:
        return 0.0

    n_left = y_left.size
    n_right = y_right.size

    if n_left == 0 or n_right == 0:
        return 0.0

    impurity_fn = _gini_impurity if criterion == "gini" else _entropy_impurity

    parent_impurity = impurity_fn(y_parent)
    left_impurity = impurity_fn(y_left)
    right_impurity = impurity_fn(y_right)

    weighted_child_impurity = (
        (n_left / n_parent) * left_impurity
        + (n_right / n_parent) * right_impurity
    )

    return float(parent_impurity - weighted_child_impurity)


def _resolve_feature_indices(
    n_features: int,
    max_features: Optional[Union[int, str]],
    random_state: Optional[int],
) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    if max_features is None:
        return np.arange(n_features)

    if max_features == "sqrt":
        k = max(1, int(np.sqrt(n_features)))
        return rng.choice(n_features, size=k, replace=False)

    if max_features == "log2":
        k = max(1, int(np.log2(n_features)))
        return rng.choice(n_features, size=k, replace=False)

    if isinstance(max_features, int):
        if max_features < 1 or max_features > n_features:
            raise ValueError("max_features must be between 1 and n_features")
        return rng.choice(n_features, size=max_features, replace=False)

    raise ValueError("max_features must be None, 'sqrt', 'log2', or an integer")


def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
    max_features: Optional[Union[int, str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[Optional[int], Optional[float], float]:
    _, n_features = X.shape

    feature_indices = _resolve_feature_indices(
        n_features=n_features,
        max_features=max_features,
        random_state=random_state,
    )

    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None
    best_gain = 0.0

    for feature in feature_indices:
        col = X[:, feature].astype(float)

        valid_mask = ~np.isnan(col)
        if valid_mask.sum() == 0:
            continue

        col_valid = col[valid_mask]
        unique_vals = np.unique(col_valid)

        if len(unique_vals) < 2:
            continue

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            nan_mask = np.isnan(col)
            left_mask = (col <= threshold) | nan_mask
            right_mask = col > threshold

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            gain = _information_gain(
                y,
                y_left,
                y_right,
                criterion=criterion,
            )

            if gain > best_gain or (
                np.isclose(gain, best_gain)
                and (best_feature is None or feature < best_feature)
            ):
                best_feature = int(feature)
                best_threshold = float(threshold)
                best_gain = float(gain)

    return best_feature, best_threshold, best_gain


class DecisionTreeClassifier:

    def __init__(
        self,
        max_depth: Optional[int] = MAX_DEPTH_DEFAULT,
        min_samples_leaf: int = MIN_SAMPLES_LEAF_DEFAULT,
        min_samples_split: int = MIN_SAMPLES_SPLIT_DEFAULT,
        criterion: str = "gini",
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
    ):
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

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.tree_: Optional[TreeNode] = None
        self.n_classes_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self._is_fitted = False

        self._X_seen: Optional[np.ndarray] = None
        self._y_seen: Optional[np.ndarray] = None

    def _validate_input(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have different lengths: {X.shape[0]} vs {y.shape[0]}"
            )

        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        return X, y

    def _majority_class(self, y: np.ndarray) -> int:
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() == 0:
            return -1

        y_valid = y[valid_mask].astype(int)
        unique_classes, counts = np.unique(y_valid, return_counts=True)

        return int(unique_classes[np.argmax(counts)])

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0,
    ) -> TreeNode:
        n_samples = X.shape[0]

        node = TreeNode(depth=depth)
        node.samples = n_samples

        valid_mask = ~np.isnan(y)
        if valid_mask.sum() == 0:
            node.is_leaf = True
            node.value = -1
            node.impurity = 0.0
            return node

        y_valid = y[valid_mask].astype(int)
        unique_classes, counts = np.unique(y_valid, return_counts=True)

        node.class_counts = {
            int(cls): int(count)
            for cls, count in zip(unique_classes, counts)
        }

        impurity_fn = _gini_impurity if self.criterion == "gini" else _entropy_impurity
        node.impurity = impurity_fn(y)

        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(unique_classes) == 1
        )

        if should_stop:
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node

        feature, threshold, gain = _find_best_split(
            X,
            y,
            criterion=self.criterion,
            max_features=self.max_features,
            random_state=self.random_state,
        )

        if feature is None or threshold is None or gain < EPS:
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node

        col = X[:, feature].astype(float)
        nan_mask = np.isnan(col)
        left_mask = (col <= threshold) | nan_mask
        right_mask = col > threshold

        n_left = int(left_mask.sum())
        n_right = int(right_mask.sum())

        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node

        node.is_leaf = False
        node.feature = int(feature)
        node.threshold = float(threshold)

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X, y = self._validate_input(X, y)

        valid_mask = ~np.isnan(y)
        if valid_mask.sum() == 0:
            raise ValueError("y must contain at least one non-NaN class label")

        self.classes_ = np.unique(y[valid_mask]).astype(int)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.n_features_ = X.shape[1]

        self._X_seen = X.copy()
        self._y_seen = y.copy()

        self.tree_ = self._build_tree(X, y, depth=0)
        self._is_fitted = True

        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "DecisionTreeClassifier":
        X, y = self._validate_input(X, y)

        if not self._is_fitted:
            if classes is not None:
                self.classes_ = np.asarray(classes, dtype=int)
            else:
                valid_mask = ~np.isnan(y)
                if valid_mask.sum() == 0:
                    raise ValueError("y must contain at least one non-NaN class label")
                self.classes_ = np.unique(y[valid_mask]).astype(int)

            self.n_classes_ = len(self.classes_)
            self.n_features_in_ = X.shape[1]
            self.n_features_ = X.shape[1]
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

        self.tree_ = self._build_tree(self._X_seen, self._y_seen, depth=0)
        self._is_fitted = True

        return self

    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.tree_ is None:
            raise RuntimeError(
                "Tree must be fitted before prediction. "
                "Call fit() or partial_fit() first."
            )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        return X

    def _traverse_tree(self, x_sample: np.ndarray) -> TreeNode:
        node = self.tree_

        while not node.is_leaf:
            feature = node.feature
            threshold = node.threshold
            value = x_sample[feature]

            if np.isnan(value):
                node = node.left
            elif value <= threshold:
                node = node.left
            else:
                node = node.right

        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Tree traversal is inherently sample-wise because each row may follow
        # a different path through the learned tree.
        
        X = self._validate_predict_input(X)

        predictions = np.zeros(X.shape[0], dtype=int)

        for i, x_sample in enumerate(X):
            leaf = self._traverse_tree(x_sample)
            predictions[i] = leaf.value

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Probability lookup follows the leaf reached by each sample. The class
        # probability vector is then filled from the leaf's stored class counts.
        
        X = self._validate_predict_input(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        class_to_idx = {
            int(cls): idx
            for idx, cls in enumerate(self.classes_)
        }

        for i, x_sample in enumerate(X):
            leaf = self._traverse_tree(x_sample)
            total = sum(leaf.class_counts.values())

            if total == 0:
                proba[i, :] = 1.0 / n_classes
            else:
                for cls_label, count in leaf.class_counts.items():
                    idx = class_to_idx.get(int(cls_label))
                    if idx is not None:
                        proba[i, idx] = count / total

        return proba

    def get_depth(self) -> int:
        if not self._is_fitted or self.tree_ is None:
            return 0

        def _max_depth(node: TreeNode) -> int:
            if node.is_leaf:
                return 0
            return 1 + max(_max_depth(node.left), _max_depth(node.right))

        return _max_depth(self.tree_)

    def get_n_leaves(self) -> int:
        if not self._is_fitted or self.tree_ is None:
            return 0

        def _count_leaves(node: TreeNode) -> int:
            if node.is_leaf:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)

        return _count_leaves(self.tree_)

    def get_params(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "DecisionTreeClassifier":
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)

        return self

    def __repr__(self) -> str:
        params = (
            f"max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"min_samples_split={self.min_samples_split}, "
            f"criterion='{self.criterion}', "
            f"max_features={self.max_features}"
        )
        return f"DecisionTreeClassifier({params})"