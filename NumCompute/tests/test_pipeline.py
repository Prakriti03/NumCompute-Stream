# tests/test_pipeline.py
import pytest
import numpy as np
from numcompute.pipeline import Transformer, Estimator, Pipeline


def test_transformer_fit_returns_self():
    t = Transformer()
    X = np.array([[1.0, 2.0]])
    assert t.fit(X) is t


def test_transformer_transform_raises():
    with pytest.raises(NotImplementedError):
        Transformer().transform(np.array([[1.0]]))


def test_estimator_fit_returns_self():
    e = Estimator()
    assert e.fit(np.array([[1.0]]), np.array([1.0])) is e


def test_estimator_predict_raises():
    with pytest.raises(NotImplementedError):
        Estimator().predict(np.array([[1.0]]))

def test_pipeline_valid_construction():
    class T(Transformer):
        def transform(self, X): return X
    pipe = Pipeline([("a", T()), ("b", T())])
    assert len(pipe.steps) == 2

def test_pipeline_empty_steps_raises():
    with pytest.raises(ValueError):
        Pipeline([])

def test_pipeline_duplicate_name_raises():
    class T(Transformer):
        def transform(self, X): return X
    with pytest.raises(ValueError, match="Duplicate"):
        Pipeline([("a", T()), ("a", T())])

def test_pipeline_missing_transform_on_nonfinal_raises():
    class NoTransform:
        def fit(self, X): return self
    class T(Transformer):
        def transform(self, X): return X
    with pytest.raises(TypeError):
        Pipeline([("bad", NoTransform()), ("ok", T())])

def test_fit_transform_applies_steps_in_order():
    class Double(Transformer):
        def transform(self, X): return X * 2
    class AddOne(Transformer):
        def transform(self, X): return X + 1
    pipe = Pipeline([("double", Double()), ("addone", AddOne())])
    result = pipe.fit_transform(np.array([[3.0]]))
    np.testing.assert_array_equal(result, [[7.0]])

def test_transform_applies_after_fit():
    class Double(Transformer):
        def transform(self, X): return X * 2
    pipe = Pipeline([("d", Double())])
    pipe.fit(np.array([[1.0]]))
    result = pipe.transform(np.array([[4.0]]))
    np.testing.assert_array_equal(result, [[8.0]])


def test_predict_calls_final_estimator():
    class T(Transformer):
        def transform(self, X): return X * 2
    class FakeModel(Estimator):
        def predict(self, X): return np.array([1.0])
    pipe = Pipeline([("t", T()), ("m", FakeModel())])
    pipe.fit(np.array([[1.0]]), y=np.array([1.0]))
    result = pipe.predict(np.array([[2.0]]))
    np.testing.assert_array_equal(result, [1.0])


def test_predict_raises_if_no_predict_on_final():
    class T(Transformer):
        def transform(self, X): return X
    pipe = Pipeline([("t", T())])
    pipe.fit(np.array([[1.0]]))
    with pytest.raises(TypeError):
        pipe.predict(np.array([[1.0]]))
