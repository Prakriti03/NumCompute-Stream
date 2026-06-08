import numpy as np

# --------------------------------------------------------------------------
# Statistical Functions
# --------------------------------------------------------------------------
def mean(x: np.ndarray, axis: int = None, ignore_nan: bool = False) -> np.ndarray:
  """
    Compute the arithmetic mean, optionally ignoring NaNs.
 
    Parameters
    ----------
    x          : np.ndarray, any shape
    axis       : int or None — axis along which to compute mean
    ignore_nan : bool, default False — use nanmean if True
 
    Returns
    -------
    np.ndarray or float
 
    Complexity
    ----------
    Time : O(n)
    Space: O(1) for scalar output, O(n/axis_size) for axis reduction
  """
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanmean(x, axis=axis)
  return np.mean(x, axis=axis)

def median(x: np.ndarray, axis: int = None, ignore_nan: bool = False) -> np.ndarray:
  """
    Compute the median.
 
    Parameters
    ----------
    x          : np.ndarray, any shape
    axis       : int or None
    ignore_nan : bool, default False
 
    Returns
    -------
    np.ndarray or float
 
    Complexity
    ----------
    Time : O(n log n)  — sort-based
    Space: O(n)
  """
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanmedian(x, axis=axis)
  return np.median(x, axis=axis)

def variance(x: np.ndarray,axis: int = None,ddof: int = 0,ignore_nan: bool = False,) -> np.ndarray:
  """
    Compute variance.
 
    Parameters
    ----------
    x          : np.ndarray, any shape
    axis       : int or None
    ddof       : int, default 0 — delta degrees of freedom
                 (ddof=0 → population variance, ddof=1 → sample variance)
    ignore_nan : bool, default False
 
    Returns
    -------
    np.ndarray or float
 
    Raises
    ------
    ValueError
        If ddof >= number of observations along axis.
 
    Complexity
    ----------
    Time : O(n)
    Space: O(1) for scalar output
  """
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanvar(x, axis=axis, ddof=ddof)
  return np.var(x, axis=axis, ddof=ddof)

def std(x: np.ndarray,axis: int = None,ddof: int = 0,ignore_nan: bool = False,) -> np.ndarray:
  """
    Compute standard deviation.
 
    Parameters
    ----------
    x          : np.ndarray, any shape
    axis       : int or None
    ddof       : int, default 0 — delta degrees of freedom
    ignore_nan : bool, default False
 
    Returns
    -------
    np.ndarray or float
 
    Complexity
    ----------
    Time : O(n)
    Space: O(1)
  """
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanstd(x, axis=axis, ddof=ddof)
  return np.std(x, axis=axis, ddof=ddof)

def mode(x: np.ndarray) -> tuple:
  """
    Compute the mode (most frequent value) of a 1-D array.
 
    If multiple values share the maximum frequency, the smallest is returned.
 
    Parameters
    ----------
    x : np.ndarray, 1-D
 
    Returns
    -------
    (mode_value, count) : tuple
        mode_value — the most frequent element
        count      — how many times it appears
 
    Raises
    ------
    ValueError
        If x is empty or not 1-D.
 
    Complexity
    ----------
    Time : O(n)
    Space: O(k) where k = number of unique values
  """
  x = np.asarray(x).ravel()
  if x.size == 0:
    raise ValueError("Cannot compute mode of an empty array.")
  values, counts = np.unique(x, return_counts=True)
  idx = np.argmax(counts)
  return values[idx], counts[idx]

def skewness(x: np.ndarray, axis: int = None, bias: bool = True) -> np.ndarray:
  """
    Compute the (Fisher-Pearson) skewness.
 
    Parameters
    ----------
    x    : np.ndarray
    axis : int or None
    bias : bool, default True
        If False, apply the adjusted Fisher-Pearson correction for finite samples.
 
    Returns
    -------
    np.ndarray or float
 
    Raises
    ------
    ValueError
        If the standard deviation is zero (constant array).
 
    Complexity
    ----------
    Time : O(n)
    Space: O(1)
  """
  x = np.asarray(x, dtype=float)
  mu = np.mean(x, axis=axis, keepdims=True)
  sigma = np.std(x, axis=axis, keepdims=True, ddof=0)
  if np.any(sigma == 0):
    raise ValueError("Standard deviation is 0 — skewness is undefined.")
  m3 = np.mean(((x - mu) / sigma) ** 3, axis=axis)
  if not bias:
    n = x.shape[axis] if axis is not None else x.size
    m3 = m3 * np.sqrt(n * (n - 1)) / (n - 2)
  return m3

def kurtosis(x: np.ndarray, axis: int = None, excess: bool = True) -> np.ndarray:
  """
    Compute kurtosis.
 
    Parameters
    ----------
    x      : np.ndarray
    axis   : int or None
    excess : bool, default True — return excess (Fisher) kurtosis (normal → 0)
 
    Returns
    -------
    np.ndarray or float
 
    Raises
    ------
    ValueError
        If standard deviation is zero.
 
    Complexity
    ----------
    Time : O(n)
    Space: O(1)
  """
  x = np.asarray(x, dtype=float)
  mu = np.mean(x, axis=axis, keepdims=True)
  sigma = np.std(x, axis=axis, keepdims=True, ddof=0)
  if np.any(sigma == 0):
    raise ValueError("Standard deviation is 0 — kurtosis is undefined.")
  m4 = np.mean(((x - mu) / sigma) ** 4, axis=axis)
  return m4 - 3.0 if excess else m4

# --------------------------------------------------------------------------
# Quantiles and Percentile
# --------------------------------------------------------------------------
def quantile(x: np.ndarray,q: float | list,axis: int = None,ignore_nan: bool = False) -> np.ndarray:
  """
    Compute quantile(s) using linear interpolation.
 
    Parameters
    ----------
    x          : np.ndarray
    q          : float in [0, 1] or list of floats — quantile(s) to compute
    axis       : int or None
    ignore_nan : bool, default False
 
    Returns
    -------
    np.ndarray
        If q is scalar and axis is None, returns a scalar.
        Shape follows np.quantile conventions.
 
    Raises
    ------
    ValueError
        If any q value is outside [0, 1].
 
    Complexity
    ----------
    Time : O(n log n) — sort-based
    Space: O(n)
  """
  q_arr = np.asarray(q)
  if np.any(q_arr < 0) or np.any(q_arr > 1):
    raise ValueError(f"All quantile values must be in [0, 1]; got {q}.")
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanquantile(x, q, axis=axis, method="linear")
  return np.quantile(x, q, axis=axis, method="linear")

def percentile(x: np.ndarray,p: float | list,axis: int = None,ignore_nan: bool = False) -> np.ndarray:
  """
    Compute percentile(s) (convenience wrapper around quantile).
 
    Parameters
    ----------
    x          : np.ndarray
    p          : float in [0, 100] or list of floats
    axis       : int or None
    ignore_nan : bool, default False
 
    Returns
    -------
    np.ndarray
 
    Raises
    ------
    ValueError
        If any p value is outside [0, 100].
  """
  p_arr = np.asarray(p)
  if np.any(p_arr < 0) or np.any(p_arr > 100):
    raise ValueError(f"All percentile values must be in [0, 100]; got {p}.")
  return quantile(x, p_arr / 100.0, axis=axis, ignore_nan=ignore_nan)

def iqr(x: np.ndarray, axis: int = None, ignore_nan: bool = False) -> np.ndarray:
  """
    Compute the interquartile range (Q3 - Q1).
 
    Parameters
    ----------
    x          : np.ndarray
    axis       : int or None
    ignore_nan : bool, default False
 
    Returns
    -------
    np.ndarray or float
  """
  q1 = quantile(x, 0.25, axis=axis, ignore_nan=ignore_nan)
  q3 = quantile(x, 0.75, axis=axis, ignore_nan=ignore_nan)
  return q3 - q1

# ----------------------------------------------------
# Histogram
# ----------------------------------------------------
def histogram(x: np.ndarray,bins: int | np.ndarray = 10,range: tuple = None,density: bool = False) -> tuple:
  """
    Compute a histogram of data values.
 
    Parameters
    ----------
    x       : np.ndarray, 1-D (will be flattened)
    bins    : int or 1-D array-like
              If int, creates that many equal-width bins over the data range (or `range`).
              If array-like, these are the bin edges (length = num_bins + 1).
    range   : (float, float) or None — (min, max) of the bins.
              Ignored if bins is array-like.
    density : bool, default False
              If True, normalise counts so the histogram integrates to 1 (i.e., probability density).
 
    Returns
    -------
    counts     : np.ndarray, shape (num_bins,) — bin counts (or density)
    bin_edges  : np.ndarray, shape (num_bins + 1,)
 
    Raises
    ------
    ValueError
        If bins < 1 (when int) or x is empty.
 
    Complexity
    ----------
    Time : O(n log k) where k = number of bins
    Space: O(k)
  """
  x = np.asarray(x, dtype=float).ravel()
  if x.size == 0:
    raise ValueError("Cannot compute histogram of an empty array.")
  if isinstance(bins, int):
    if bins < 1:
      raise ValueError(f"Bins must be >= 1, got {bins}.")
    lo = x.min() if range is None else range[0]
    hi = x.max() if range is None else range[1]
    if lo == hi:
      lo, hi = lo - 0.5, hi + 0.5
    bin_edges = np.linspace(lo, hi, bins + 1)
  else:
    bin_edges = np.asarray(bins, dtype=float)
    if bin_edges.ndim != 1 or len(bin_edges) < 2:
      raise ValueError("Bins array must be 1-D with at least 2 elements.")
  indices = np.searchsorted(bin_edges, x, side="right") - 1
  indices = np.clip(indices, 0, len(bin_edges) - 2)
  counts = np.bincount(indices, minlength=len(bin_edges) - 1).astype(float)
  if density:
    widths = np.diff(bin_edges)
    counts = counts/(counts.sum()*widths)
  return counts, bin_edges

# ----------------------------------------------------
# Covariance and Correlation
# ----------------------------------------------------
def covariance_matrix(X: np.ndarray, ddof: int = 1) -> np.ndarray:
  """
    Compute the (d, d) sample covariance matrix of a dataset.
 
    Parameters
    ----------
    X    : np.ndarray, shape (n, d) — n observations, d features
    ddof : int, default 1 — delta degrees of freedom (1 = unbiased sample cov)
 
    Returns
    -------
    np.ndarray, shape (d, d), symmetric positive semi-definite
 
    Raises
    ------
    ValueError
        If X has fewer rows than ddof+1.
 
    Complexity
    ----------
    Time : O(n * d^2)
    Space: O(d^2)
  """
  X = np.asarray(X, dtype=float)
  if X.ndim == 1:
    X = X[:, None]
  n, d = X.shape
  if n <= ddof:
    raise ValueError(f"Need at least {ddof + 1} observations for ddof={ddof}.")
  X_centred = X - X.mean(axis=0)
  return (X_centred.T @ X_centred) / (n - ddof)

def correlation_matrix(X: np.ndarray) -> np.ndarray:
  """
    Compute the (d, d) Pearson correlation matrix.
 
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
 
    Returns
    -------
    np.ndarray, shape (d, d)
        Values in [-1, 1]; diagonal is always 1.
        Features with zero variance produce NaN in their row/column.
 
    Complexity
    ----------
    Time : O(n * d^2)
    Space: O(d^2)
  """
  cov = covariance_matrix(X, ddof=1)
  d = np.sqrt(np.diag(cov))
  denom = np.outer(d, d)
  with np.errstate(invalid="ignore"):
    corr = np.where(denom == 0, np.nan, cov / denom)
  return corr

# ----------------------------------------------------
# Welford's Online/Streaming algorithm
# ----------------------------------------------------
class WelfordStatistics:
  """
    Incremental (streaming) mean and variance using Welford's algorithm.
 
    Suitable for computing statistics over data that arrives in chunks,
    or when the full dataset does not fit in memory.
 
    Numerical stability: Welford's method avoids catastrophic cancellation
    by accumulating the sum of squared deviations from the running mean,
    rather than tracking sum-of-squares and subtracting later.
 
    Parameters
    ----------
    shape : tuple or None
        Shape of each observation *excluding* the batch dimension.
        Pass None for scalar observations, or a tuple for multi-dimensional.
 
    Examples
    --------
    >>> ws = WelfordStats()
    >>> for val in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
    ...     ws.update(np.array([val]))
    >>> ws.mean(), ws.variance()
    (5.0, 4.571...)   # population variance
 
    >>> # Batch update
    >>> ws2 = WelfordStats()
    >>> ws2.update_batch(np.array([[1, 2], [3, 4], [5, 6]]))
    >>> ws2.mean()
    array([3., 4.])
  """
  def __init__(self, shape: tuple = None):
    self._n = 0
    self._mean = None
    self._M2 = None
    self._shape = shape

  def _init_arrays(self, x: np.ndarray):
    # On 1st update, initialize the internal array
    self._mean = np.zeros_like(x, dtype=float)
    self._M2 = np.zeros_like(x, dtype=float)

  def update(self, x: np.ndarray):
    """
        Incorporate a single observation.
 
        Parameters
        ----------
        x : np.ndarray — one observation (scalar or array)
 
        Returns
        -------
        self
    """
    x = np.asarray(x, dtype=float)
    if self._mean is None:
      self._init_arrays(x)
    self._n += 1
    delta = x - self._mean
    self._mean += delta / self._n
    delta2 = x - self._mean
    self._M2 += delta * delta2
    return self

  def update_batch(self, X: np.ndarray):
    """
        Incorporate a batch of observations efficiently.
 
        Iterates over rows of X, applying the Welford update.
        For large batches this is faster than calling update() in a Python loop
        from the caller's perspective (though internally it still iterates).
        Vectorised batch merging is also supported via merge().
 
        Parameters
        ----------
        X : np.ndarray, shape (n, ...) — n observations
 
        Returns
        -------
        self
    """
    X = np.asarray(X, dtype=float)
    for row in X:
      self.update(row)
    return self

  def mean(self)->np.ndarray:
    """
        Return the current running mean.
 
        Returns
        -------
        np.ndarray or None if no data has been seen yet.
    """
    if self._n==0:
      return None
    return self._mean.copy()

  def variance(self,ddof:int=0)->np.ndarray:
    """
        Return current variance.
 
        Parameters
        ----------
        ddof : int, default 0 — degrees of freedom correction
               (ddof=0 → population, ddof=1 → sample)
 
        Returns
        -------
        np.ndarray or None if fewer than ddof+1 observations seen.
 
        Raises
        ------
        ValueError
            If ddof >= n.
    """
    if self._n==0:
      return None
    if self._n<=ddof:
      raise ValueError(f"Require atleast {ddof+1}observations for ddof={ddof}")
    return self._M2/(self._n-ddof)

  def std(self,ddof:int=0)->np.ndarray:
    """
        Return current standard deviation.
 
        Parameters
        ----------
        ddof : int, default 0
 
        Returns
        -------
        np.ndarray or None
    """
    v=self.variance(ddof=ddof)
    return None if v is None else np.sqrt(v)

  @property
  def n(self)->int: #Number of observations so far
    return self._n

  def reset(self): #Reset all accumulaor to initial state
    self._n=0
    self._mean=None
    self._M2=None
    return self

  def merge(self,other:"WelfordStatistics")->"WelfordStatistics":
    """
        Merge another WelfordStats instance into this one.
 
        Useful for parallel / distributed computation where each worker
        runs a separate WelfordStats and results are combined.
 
        Uses Chan et al.'s parallel algorithm for combining two accumulators.
 
        Parameters
        ----------
        other : WelfordStats
 
        Returns
        -------
        self (updated in-place)
 
        Raises
        ------
        ValueError
            If the two accumulators track different-shaped observations.
    """
    if other._n==0:
      return self
    if self._n==0:
      self._n=other._n
      self._mean=other.mean().copy()
      self._M2=other._M2.copy()
      return self
    if self._mean.shape!=other._mean.shape:
      raise ValueError(f"Shape Mismatched")
    num_a,num_b=self._n,other._n
    num_combined=num_a+num_b
    delta=other._mean-self._mean
    new_mean=self._mean+delta*(num_b/num_combined)
    new_M2=self._M2+other._M2+(delta**2)*(num_a*num_b/num_combined)
    self._n=num_combined
    self._mean=new_mean
    self._M2=new_M2
    return self

  def __repr__(self)->str:
    return f"Welfordstatistics(n={self._n},mean={self._mean},std={self.std()})"

# ----------------------------------------------------
# Summary Utility
# ----------------------------------------------------
def describe(x:np.ndarray, ddof:int=1)->dict:
  """
    Compute a summary of descriptive statistics for a 1-D array.
 
    Parameters
    ----------
    x    : np.ndarray, 1-D
    ddof : int, default 1 — for variance/std
 
    Returns
    -------
    dict with keys:
        n, mean, std, variance, min, q25, median, q75, max, iqr, skewness
 
    Raises
    ------
    ValueError
        If x is empty or not 1-D after flattening.
 
    Complexity
    ----------
    Time : O(n log n)
    Space: O(n)
  """
  x=np.asarray(x,dtype=float).ravel()
  if x.size==0:
    raise ValueError(f"Empty array cannot be described")
  nan_mask=~np.isnan(x)
  x_clean=x[nan_mask]
  if x_clean.size==0:
    raise ValueError(f"All Values are NaN")
  q1,med,q3=np.quantile(x_clean,[0.25,0.50,0.75],method="linear")
  try:
    skew=float(skewness(x_clean))
  except ValueError:
    skew=float("nan")
  try:
    kurt=float(kurtosis(x_clean))
  except ValueError:
    kurt=float("nan")

  return{
      "n": int(x_clean.size),
      "nan_count": int(x.size-x_clean.size),
      "mean": float(np.mean(x_clean)),
      "std": float(np.std(x_clean,ddof=ddof)),
      "variance": float(np.var(x_clean,ddof=ddof)),
      "min": float(np.min(x_clean)),
      "max": float(np.max(x_clean)),
      "q25": float(q1),
      "median": float(med),
      "q75": float(q3),
      "iqr": float(q3-q1),
      "skew": float(skew),
      "kurtosis": float(kurt)
  }
