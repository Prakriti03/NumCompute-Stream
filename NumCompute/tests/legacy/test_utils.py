import numpy as np
from numcompute_stream.utils import (
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
    relu,
    sigmoid,
    softmax,
    logsumexp,
    batch_iter,
    one_hot_encode,
    pairwise_distances
)

# ------------------------------------------------------------
# TEST 1: Distance functions correctness (euclidean + manhattan + cosine)
# ------------------------------------------------------------
def test_distance_functions():
    X = np.array([[0, 0], [1, 1]])
    Y = np.array([[1, 0], [2, 2]])

    # Euclidean
    d_euc = euclidean_distance(X, Y)
    assert d_euc.shape == (2, 2)
    assert np.isclose(d_euc[0, 0], 1.0)

    # Manhattan
    d_man = manhattan_distance(X, Y)
    assert np.isclose(d_man[0, 0], 1.0)
    assert np.isclose(d_man[1, 1], 2.0)

    # Cosine similarity
    sim = cosine_similarity(X, Y)
    assert sim.shape == (2, 2)
    assert np.all(sim <= 1.0) and np.all(sim >= -1.0)


# ------------------------------------------------------------
# TEST 2: Activations + numerical stability
# ------------------------------------------------------------
def test_activations_and_numerics():
    x = np.array([-1.0, 0.0, 1.0])

    # ReLU
    assert np.array_equal(relu(x), np.array([0.0, 0.0, 1.0]))

    # Sigmoid bounds
    s = sigmoid(x)
    assert np.all(s > 0) and np.all(s < 1)

    # Softmax sums to 1
    sm = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.isclose(sm.sum(), 1.0)

    # Logsumexp stability
    lse = logsumexp(np.array([1000, 1000]))
    assert np.isfinite(lse)


# ------------------------------------------------------------
# TEST 3: Utilities (batching, one-hot, pairwise distances)
# ------------------------------------------------------------
def test_utils_batch_and_encoding():
    # one-hot encoding
    labels = np.array([0, 1, 2])
    ohe = one_hot_encode(labels)
    assert ohe.shape == (3, 3)
    assert np.all(ohe.sum(axis=1) == 1)

    # batching
    X = np.arange(10).reshape(5, 2)
    batches = list(batch_iter(X, batch_size=2))
    assert len(batches) == 3  # 2,2,1 batches

    # pairwise distance wrapper
    D = pairwise_distances(np.array([[0.0], [1.0]]), metric="euclidean")
    assert D.shape == (2, 2)
    assert np.allclose(np.diag(D), 0.0)