import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, Imputer
from numcompute_stream.ensemble import RandomForestClassifier


def test_pipeline_partial_fit_predicts_with_streaming_chunks():
    X1 = np.array([[0.0, 0.0], [1.0, np.nan]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0, 2.0], [3.0, 3.0]])
    y2 = np.array([1, 1])

    pipe = Pipeline([
        ("impute", Imputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)),
    ])

    pipe.partial_fit(X1, y1)
    pipe.partial_fit(X2, y2)

    preds = pipe.predict(np.vstack([X1, X2]))

    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})


def test_pipeline_predict_before_fit_raises_error():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=3)),
    ])

    with pytest.raises(RuntimeError):
        pipe.predict(np.array([[1.0, 2.0]]))


def test_pipeline_predict_proba_shape_and_sums():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
    ])

    pipe.partial_fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_score_returns_valid_accuracy():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
    ])

    pipe.partial_fit(X, y)

    score = pipe.score(X, y)

    assert 0.0 <= score <= 1.0


def test_pipeline_get_step_and_indexing():
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=3)

    pipe = Pipeline([
        ("scale", scaler),
        ("model", model),
    ])

    assert pipe.get_step("scale") is scaler
    assert pipe["model"] is model


def test_pipeline_duplicate_step_names_raise_error():
    with pytest.raises(ValueError):
        Pipeline([
            ("scale", StandardScaler()),
            ("scale", StandardScaler()),
        ])


def test_pipeline_invalid_step_format_raises_error():
    with pytest.raises(TypeError):
        Pipeline([
            ("scale", StandardScaler()),
            "not_a_tuple",
        ])


def test_pipeline_partial_fit_rejects_1d_X():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=3)),
    ])

    with pytest.raises(ValueError):
        pipe.partial_fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 1]))