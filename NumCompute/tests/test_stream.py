import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, Imputer
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.metrics import StreamingAccuracy


def make_pipeline():
    return Pipeline([
        ("impute", Imputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=3,
            max_depth=3,
            random_state=42,
        )),
    ])


def test_fit_chunk_trains_and_logs_required_fields():
    X = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 0, 1, 1])

    trainer = StreamTrainer(
        make_pipeline(),
        metrics={"accuracy": StreamingAccuracy()},
    )

    log = trainer.fit_chunk(X, y)

    assert trainer._is_fitted
    assert log["chunk"] == 1
    assert log["chunk_size"] == 4
    assert "train_time_sec" in log
    assert "memory_bytes" in log
    assert "chunk_accuracy" in log
    assert "cumulative_accuracy" in log
    assert "metrics" in log
    assert 0.0 <= log["chunk_accuracy"] <= 1.0
    assert 0.0 <= log["cumulative_accuracy"] <= 1.0


def test_fit_stream_splits_dataset_into_chunks():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1, 1])

    trainer = StreamTrainer(make_pipeline())

    logs = trainer.fit_stream(X, y, chunk_size=2)

    assert len(logs) == 3
    assert trainer.chunk_index_ == 3
    assert trainer.total_seen_ == 5
    assert logs[0]["chunk_size"] == 2
    assert logs[-1]["chunk_size"] == 1


def test_score_chunk_requires_fitted_trainer():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 0])

    trainer = StreamTrainer(make_pipeline())

    with pytest.raises(RuntimeError):
        trainer.score_chunk(X, y)


def test_score_chunk_logs_without_training_after_fit():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([[0.5], [2.5]])
    y_test = np.array([0, 1])

    trainer = StreamTrainer(
        make_pipeline(),
        metrics={"accuracy": StreamingAccuracy()},
    )

    trainer.fit_chunk(X_train, y_train)
    log = trainer.score_chunk(X_test, y_test)

    assert log["chunk"] == 2
    assert log["chunk_size"] == 2
    assert "score_time_sec" in log
    assert "chunk_accuracy" in log
    assert "cumulative_accuracy" in log
    assert 0.0 <= log["chunk_accuracy"] <= 1.0


def test_metrics_are_updated_across_chunks():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    metric = StreamingAccuracy()
    trainer = StreamTrainer(
        make_pipeline(),
        metrics={"accuracy": metric},
    )

    trainer.fit_chunk(X1, y1)
    trainer.fit_chunk(X2, y2)

    logs = trainer.get_logs()

    assert len(logs) == 2
    assert "accuracy" in logs[-1]["metrics"]
    assert 0.0 <= logs[-1]["metrics"]["accuracy"] <= 1.0


def test_get_metric_history_returns_values_across_chunks():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    trainer = StreamTrainer(make_pipeline())
    trainer.fit_stream(X, y, chunk_size=2)

    history = trainer.get_metric_history("cumulative_accuracy")

    assert history.shape == (2,)
    assert np.all((history >= 0.0) & (history <= 1.0))


def test_memory_tracking_can_be_disabled():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 0])

    trainer = StreamTrainer(make_pipeline(), track_memory=False)
    log = trainer.fit_chunk(X, y)

    assert log["memory_bytes"] is None


def test_reset_clears_logs_counters_and_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    metric = StreamingAccuracy()
    trainer = StreamTrainer(
        make_pipeline(),
        metrics={"accuracy": metric},
    )

    trainer.fit_chunk(X, y)
    trainer.reset()

    assert trainer.get_logs() == []
    assert trainer.chunk_index_ == 0
    assert trainer.total_seen_ == 0
    assert trainer.total_eval_seen_ == 0
    assert not trainer._is_fitted
    assert metric.result() == 0.0


def test_invalid_pipeline_missing_partial_fit_raises_error():
    class BadPipeline:
        def predict(self, X):
            return np.zeros(X.shape[0])

    with pytest.raises(TypeError):
        StreamTrainer(BadPipeline())


def test_invalid_pipeline_missing_predict_raises_error():
    class BadPipeline:
        def partial_fit(self, X, y):
            return self

    with pytest.raises(TypeError):
        StreamTrainer(BadPipeline())


def test_invalid_chunk_shapes_raise_error():
    trainer = StreamTrainer(make_pipeline())

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([1.0, 2.0]), np.array([0, 1]))

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([[1.0], [2.0]]), np.array([[0], [1]]))

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([[1.0], [2.0]]), np.array([0]))


def test_invalid_chunk_size_in_fit_stream_raises_error():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    trainer = StreamTrainer(make_pipeline())

    with pytest.raises(ValueError):
        trainer.fit_stream(X, y, chunk_size=0)