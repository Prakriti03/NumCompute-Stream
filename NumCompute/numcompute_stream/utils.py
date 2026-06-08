import numpy as np
# -------------------------------------------------------------------------------------------------
# Distance functions
# -------------------------------------------------------------------------------------------------
def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  """
    Compute pairwise Euclidean distances between rows of X and Y.
 
    Parameters
    ----------
    X : np.ndarray, shape (m, d)
    Y : np.ndarray, shape (n, d)
 
    Returns
    -------
    D : np.ndarray, shape (m, n)
        D[i, j] = ||X[i] - Y[j]||_2
 
    Raises
    ------
    ValueError
        If X and Y have different numbers of features (columns).
 
    Complexity
    ----------
    Time : O(m * n * d)
    Space: O(m * n)
  """
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch: X has {X.shape[1]} features")
  XX = np.sum(X ** 2, axis=1, keepdims=True)   # (m, 1)
  YY = np.sum(Y ** 2, axis=1, keepdims=True)   # (n, 1)
  D_sq = XX + YY.T - 2.0 * (X @ Y.T)           # (m, n)
  return np.sqrt(np.maximum(D_sq, 0.0))
def manhattan_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  """
    Compute pairwise Manhattan (L1) distances between rows of X and Y.
 
    Parameters
    ----------
    X : np.ndarray, shape (m, d)
    Y : np.ndarray, shape (n, d)
 
    Returns
    -------
    D : np.ndarray, shape (m, n)
        D[i, j] = sum_k |X[i,k] - Y[j,k]|
 
    Raises
    ------
    ValueError
        If X and Y have different numbers of features.
 
    Complexity
    ----------
    Time : O(m * n * d)
    Space: O(m * n * d)  — intermediate broadcast tensor
  """
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch.")
  return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  """
    Compute pairwise cosine similarities between rows of X and Y.
 
    Parameters
    ----------
    X : np.ndarray, shape (m, d)
    Y : np.ndarray, shape (n, d)
 
    Returns
    -------
    S : np.ndarray, shape (m, n)
        S[i, j] in [-1, 1].  Zero vectors yield similarity 0.0.
 
    Raises
    ------
    ValueError
        If X and Y have different numbers of features.
 
    Complexity
    ----------
    Time : O(m * n * d)
    Space: O(m * n)
  """
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch.")
  X_norm = np.linalg.norm(X, axis=1, keepdims=True)   # (m, 1)
  Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)   # (n, 1)
  # Avoid division by zero for zero vectors
  X_safe = np.where(X_norm == 0, 1.0, X_norm)
  Y_safe = np.where(Y_norm == 0, 1.0, Y_norm)
  X_unit = X / X_safe
  Y_unit = Y / Y_safe
  S = X_unit @ Y_unit.T
  # Zero vectors: set their rows/cols to 0
  zero_rows = (X_norm == 0).ravel()
  zero_cols = (Y_norm == 0).ravel()
  S[zero_rows, :] = 0.0
  S[:, zero_cols] = 0.0
  return np.clip(S, -1.0, 1.0)
def minkowski_distance(X: np.ndarray, Y: np.ndarray, p: float = 2.0) -> np.ndarray:
  """
    Compute pairwise Minkowski distances of order p.
 
    Parameters
    ----------
    X : np.ndarray, shape (m, d)
    Y : np.ndarray, shape (n, d)
    p : float, default 2.0
        Order of the norm. p=1 → Manhattan, p=2 → Euclidean, p=inf → Chebyshev.
 
    Returns
    -------
    D : np.ndarray, shape (m, n)
 
    Raises
    ------
    ValueError
        If p < 1 or feature dimensions differ.
 
    Complexity
    ----------
    Time : O(m * n * d)
    Space: O(m * n * d)
  """
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch.")
  if p < 1:
    raise ValueError(f"p must be >= 1, got {p}.")
  diff = np.abs(X[:, None, :] - Y[None, :, :])  # (m, n, d)
  if np.isinf(p):
    return np.max(diff, axis=2)
  return np.sum(diff ** p, axis=2) ** (1.0 / p)
# -------------------------------------------------------------------------------------------------
# Activation Functions
# -------------------------------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
  """
    Rectified Linear Unit: max(0, x), element-wise.
 
    Parameters
    ----------
    x : np.ndarray, any shape
 
    Returns
    -------
    np.ndarray, same shape as x
 
    Complexity
    ----------
    Time : O(n)
    Space: O(n)
  """
  return np.maximum(0.0, np.asarray(x, dtype=float))
def relu_derivative(x: np.ndarray) -> np.ndarray:
  """
    Derivative of ReLU: 1 where x > 0, else 0.
 
    Parameters
    ----------
    x : np.ndarray, any shape
 
    Returns
    -------
    np.ndarray, same shape as x (dtype float)
  """
  return (np.asarray(x, dtype=float) > 0).astype(float)
def sigmoid(x: np.ndarray) -> np.ndarray:
  """
    Sigmoid function: 1 / (1 + exp(-x)), numerically stable.
 
    Uses the identity:
        sigmoid(x) = exp(x) / (1 + exp(x))   for x >= 0
        sigmoid(x) = 1 / (1 + exp(-x))        for x <  0
    to avoid overflow in exp for large |x|.
 
    Parameters
    ----------
    x : np.ndarray, any shape
 
    Returns
    -------
    np.ndarray, same shape, values in (0, 1)
 
    Complexity
    ----------
    Time : O(n)
    Space: O(n)
  """
  x = np.asarray(x, dtype=float)
  pos_mask = x >= 0
  result = np.empty_like(x)
  # For positive values: 1 / (1 + exp(-x))
  result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
  # For negative values: exp(x) / (1 + exp(x))
  exp_x = np.exp(x[~pos_mask])
  result[~pos_mask] = exp_x / (1.0 + exp_x)
  return result
def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
  """
    Derivative of sigmoid: sigma(x) * (1 - sigma(x)).
 
    Parameters
    ----------
    x : np.ndarray, any shape
 
    Returns
    -------
    np.ndarray, same shape
  """
  s = sigmoid(x)
  return s * (1.0 - s)
def tanh_activation(x: np.ndarray) -> np.ndarray:
  """
    Hyperbolic tangent activation, wraps np.tanh.
 
    Parameters
    ----------
    x : np.ndarray, any shape
 
    Returns
    -------
    np.ndarray, same shape, values in (-1, 1)
  """
  return np.tanh(np.asarray(x, dtype=float))
def tanh_derivative(x: np.ndarray) -> np.ndarray:
  """
    Derivative of tanh: 1 - tanh(x)^2.
 
    Parameters
    ----------
    x : np.ndarray, any shape
 
    Returns
    -------
    np.ndarray, same shape
  """
  return 1.0 - np.tanh(np.asarray(x, dtype=float)) ** 2
def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
  """
    Leaky ReLU: max(alpha * x, x).
 
    Parameters
    ----------
    x     : np.ndarray, any shape
    alpha : float, default 0.01 — slope for negative inputs
 
    Returns
    -------
    np.ndarray, same shape
 
    Raises
    ------
    ValueError
        If alpha < 0.
  """
  if alpha < 0:
    raise ValueError(f"alpha must be non-negative, got {alpha}.")
  x = np.asarray(x, dtype=float)
  return np.where(x >= 0, x, alpha * x)
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  """
    Numerically stable softmax along the specified axis.
 
    Stability: subtracts max(x) along axis before exponentiation to prevent
    overflow (the shift cancels in the ratio).
 
    Parameters
    ----------
    x    : np.ndarray, any shape
    axis : int, default -1 — axis along which to compute softmax
 
    Returns
    -------
    np.ndarray, same shape as x
        Values are non-negative and sum to 1 along `axis`.
 
    Complexity
    ----------
    Time : O(n)
    Space: O(n)
  """
  x = np.asarray(x, dtype=float)
  x_shifted = x - np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(x_shifted)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# -------------------------------------------------------------------------------------------------
# Numerical Helpers
# -------------------------------------------------------------------------------------------------
def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
  """
    Numerically stable log-sum-exp: log(sum(exp(x))) along axis.
 
    Uses the identity:
        log(sum(exp(x_i))) = c + log(sum(exp(x_i - c)))
    where c = max(x_i), to prevent overflow/underflow.
 
    Parameters
    ----------
    x        : np.ndarray, any shape
    axis     : int, default -1
    keepdims : bool, default False
 
    Returns
    -------
    np.ndarray — same shape as x with `axis` reduced (or kept if keepdims=True)
 
    Complexity
    ----------
    Time : O(n)
    Space: O(n)
  """
  x = np.asarray(x, dtype=float)
  c = np.max(x, axis=axis, keepdims=True)
  shifted = x - c
  result = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) + c
  if not keepdims:
    result = np.squeeze(result, axis=axis)
  return result
def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  """
    Numerically stable log-softmax: x - logsumexp(x).
 
    Parameters
    ----------
    x    : np.ndarray, any shape
    axis : int, default -1
 
    Returns
    -------
    np.ndarray, same shape — log-probabilities (all <= 0)
  """
  x = np.asarray(x, dtype=float)
  return x - logsumexp(x, axis=axis, keepdims=True)
def clip_gradient(grad: np.ndarray, max_norm: float) -> np.ndarray:
  """
    Clip gradient by global L2 norm (gradient clipping).
 
    If ||grad||_2 > max_norm, scale grad so ||grad||_2 == max_norm.
 
    Parameters
    ----------
    grad     : np.ndarray, any shape
    max_norm : float — maximum allowed L2 norm
 
    Returns
    -------
    np.ndarray, same shape
 
    Raises
    ------
    ValueError
        If max_norm <= 0.
  """
  if max_norm <= 0:
    raise ValueError(f"max_norm must be positive, got {max_norm}.")
  grad = np.asarray(grad, dtype=float)
  norm = np.linalg.norm(grad)
  if norm > max_norm:
    grad = grad * (max_norm / norm)
  return grad

# -------------------------------------------------------------------------------------
# Batching Utilities 
# -------------------------------------------------------------------------------------
def batch_iter(*arrays: np.ndarray,batch_size: int,shuffle: bool = False,seed: int = None,):
  """
    Yield mini-batches from one or more aligned arrays.
 
    Parameters
    ----------
    *arrays    : one or more np.ndarray, all with the same first dimension
    batch_size : int — number of samples per batch
    shuffle    : bool, default False — shuffle indices each call
    seed       : int or None — random seed for reproducibility
 
    Yields
    ------
    tuple of np.ndarray
        Each element is a slice of the corresponding input array.
        The last batch may be smaller than batch_size.
 
    Raises
    ------
    ValueError
        If arrays have different first dimensions or batch_size < 1.
 
    Examples
    --------
    >>> X = np.arange(10).reshape(5, 2)
    >>> y = np.arange(5)
    >>> for xb, yb in batch_iter(X, y, batch_size=2):
    ...     print(xb.shape, yb.shape)
    (2, 2) (2,)
    (2, 2) (2,)
    (1, 2) (1,)   # <- last batch smaller
  """
  if batch_size < 1:
    raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
  n = arrays[0].shape[0]
  for arr in arrays[1:]:
    if arr.shape[0] != n:
      raise ValueError(f"All arrays must have the same first dimension.")
  rng = np.random.default_rng(seed)
  indices = rng.permutation(n) if shuffle else np.arange(n)
  for start in range(0, n, batch_size):
    idx = indices[start: start + batch_size]
    if len(arrays) == 1:
      yield arrays[0][idx]
    else:
      yield tuple(arr[idx] for arr in arrays)

def one_hot_encode(labels: np.ndarray, num_classes: int = None) -> np.ndarray:
  """
    Convert integer class labels to one-hot encoded matrix.
 
    Parameters
    ----------
    labels      : np.ndarray of int, shape (n,) — class indices, 0-based
    num_classes : int or None — inferred as max(labels)+1 if None
 
    Returns
    -------
    np.ndarray, shape (n, num_classes), dtype float64
 
    Raises
    ------
    ValueError
        If labels contain negative values or exceed num_classes-1.
 
    Complexity
    ----------
    Time : O(n * C)
    Space: O(n * C)
  """
  labels = np.asarray(labels, dtype=int).ravel()
  if labels.min() < 0:
    raise ValueError("labels must be non-negative integers.")
  if num_classes is None:
    num_classes = int(labels.max()) + 1
  if labels.max() >= num_classes:
    raise ValueError(f"label {labels.max()} exceeds num_classes-1 = {num_classes - 1}.")
  n = len(labels)
  ohe = np.zeros((n, num_classes), dtype=float)
  ohe[np.arange(n), labels] = 1.0
  return ohe

def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
  """
    Compute the full (n, n) pairwise distance matrix for a single dataset.
 
    Parameters
    ----------
    X      : np.ndarray, shape (n, d)
    metric : str — one of {"euclidean", "manhattan", "cosine"}
 
    Returns
    -------
    D : np.ndarray, shape (n, n), symmetric with zero diagonal
 
    Raises
    ------
    ValueError
        If metric is not recognised.
  """
  dispatch = {
      "euclidean": euclidean_distance,
      "manhattan": manhattan_distance,
      "cosine":lambda a, b: 1.0 - cosine_similarity(a, b),}
  if metric not in dispatch:
    raise ValueError(f"Unknown metric '{metric}'. Choose from {list(dispatch)}.")
  return dispatch[metric](X, X)
