"""
Unit tests for ensemble.py.

Coverage includes:
- RandomForestClassifier fit/predict workflow
- streaming partial_fit accumulation
- probability output shape and normalisation
- bootstrap and reproducibility behaviour
- feature importance output
- invalid parameters and input validation
"""

import numpy as np
import pytest

from numcompute_stream.ensemble import RandomForestClassifier


def test_fit_creates_expected_number_of_trees():
    X = np.array([[0], [1], [2], [3]], float)
    y = np.array([0, 0, 1, 1], float)

    clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    clf.fit(X, y)

    assert clf._is_fitted
    assert len(clf.trees_) == 5
    assert clf.n_classes_ == 2


def test_predict_returns_correct_shape():
    X = np.array([[0], [1], [2], [3]], float)
    y = np.array([0, 0, 1, 1], float)

    clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    clf.fit(X, y)

    pred = clf.predict(X)

    assert pred.shape == (4,)
    assert set(pred).issubset({0, 1})


def test_partial_fit_accumulates_chunks():
    clf = RandomForestClassifier(n_estimators=3, random_state=42)

    clf.partial_fit(np.array([[0], [1]], float), np.array([0, 0], float))
    clf.partial_fit(np.array([[2], [3]], float), np.array([1, 1], float))

    assert clf._is_fitted
    assert clf._X_seen.shape[0] == 4
    assert clf._y_seen.shape[0] == 4


def test_predict_proba_shape_and_sum():
    X = np.array([[0], [1], [2], [3]], float)
    y = np.array([0, 0, 1, 1], float)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_score_returns_valid_accuracy():
    X = np.array([[0], [1], [2], [3]], float)
    y = np.array([0, 0, 1, 1], float)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    score = clf.score(X, y)

    assert 0.0 <= score <= 1.0


def test_bootstrap_false_runs():
    X = np.array([[0], [1], [2], [3]], float)
    y = np.array([0, 0, 1, 1], float)

    clf = RandomForestClassifier(
        n_estimators=3,
        bootstrap=False,
        random_state=42,
    )
    clf.fit(X, y)

    assert len(clf.trees_) == 3


def test_feature_importances_shape():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
    y = np.array([0, 0, 1, 1], float)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    importances = clf.get_feature_importances()

    assert importances.shape == (2,)
    assert np.isclose(importances.sum(), 1.0) or np.isclose(importances.sum(), 0.0)


def test_predict_before_fit_raises_error():
    clf = RandomForestClassifier()

    with pytest.raises(RuntimeError):
        clf.predict(np.array([[0]], float))


def test_invalid_parameters_raise_error():
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0)

    with pytest.raises(ValueError):
        RandomForestClassifier(criterion="invalid")

    with pytest.raises(ValueError):
        RandomForestClassifier(max_features="invalid")


def test_mismatched_lengths_raise_error():
    X = np.array([[0], [1], [2]], float)
    y = np.array([0, 1], float)

    clf = RandomForestClassifier()

    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_partial_fit_wrong_feature_count_raises_error():
    clf = RandomForestClassifier()

    clf.partial_fit(np.array([[0, 0], [1, 1]], float), np.array([0, 1], float))

    with pytest.raises(ValueError):
        clf.partial_fit(np.array([[0, 0, 0]], float), np.array([1], float))


def test_reproducibility_with_random_state():
    np.random.seed(42)

    X = np.random.randn(30, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    clf1 = RandomForestClassifier(n_estimators=5, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=5, random_state=42)

    clf1.fit(X, y)
    clf2.fit(X, y)

    assert np.array_equal(clf1.predict(X), clf2.predict(X))