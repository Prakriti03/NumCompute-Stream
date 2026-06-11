import numpy as np
from typing import Iterator, List, Optional, Union, Callable
import csv as _csv

_MISSING = {"", "NA", "NaN", "nan", None}


def _clean_str_array(arr: np.ndarray) -> np.ndarray:
    """
    Strip whitespace vectorised.

    Parameters
    ----------
    arr : np.ndarray of str
        Raw string array.

    Returns
    -------
    np.ndarray
        Cleaned array.
    """
    return np.char.strip(arr)


def _replace_missing(arr: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """
    Replace missing tokens with NaN (vectorised).

    Parameters
    ----------
    arr : np.ndarray (str)
    fill_value : float

    Returns
    -------
    np.ndarray (float)
    """
    mask = np.isin(arr, list(_MISSING))
    arr = arr.astype(object)
    arr[mask] = fill_value
    return arr


def _infer_dtype(arr: np.ndarray) -> np.ndarray:
    """
    Try casting to float; fallback to object.

    Notes
    -----
    Uses vectorised casting.

    Returns
    -------
    np.ndarray
    """
    try:
        return arr.astype(np.float64)
    except ValueError:
        return arr.astype(object)


class CSVReader:
    """
    High-performance CSV reader with streaming + dtype handling.

    Attributes
    ----------
    header : list[str] or None
        Header row read from the CSV file when has_header=True.
        
    Parameters
    ----------
    filepath : str
        Path to CSV file.
    delimiter : str, default=","
    has_header : bool, default=True
    dtype : Optional[Union[type, List[type]]]
        - None → infer dtype
        - single type → apply to all columns
        - list → per-column types
    missing_values : set
        Tokens treated as missing.
    encoding : str

    Notes
    -----
    - Core numeric conversion is vectorised via NumPy.
    - Chunking avoids loading entire file into memory.

    Complexity
    ----------
    Time: O(N * M)
    Space: O(chunk_size * M)

    Where:
        N = rows, M = columns
    """

    def __init__(
        self,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        dtype: Optional[Union[type, List[type]]] = None,
        missing_values: Optional[set] = None,
        encoding: str = "utf-8",
    ):
        self.filepath = filepath
        self.delimiter = delimiter
        self.has_header = has_header
        self.dtype = dtype
        self.encoding = encoding
        self.missing_values = _MISSING if missing_values is None else missing_values

    def _parse_lines(self, lines: List[str]) -> np.ndarray:
        """
        Convert raw lines → 2D NumPy array (string stage).

        Returns
        -------
        np.ndarray of shape (n_rows, n_cols)
        """
        # CSV parsing itself is line-based, so a small Python loop is acceptable here.
        # Numeric conversion and cleaning are handled by NumPy after parsing.
        split = [line.rstrip("\n").split(self.delimiter) for line in lines]
        arr = np.array(split, dtype=str)
        return _clean_str_array(arr)

    def _apply_dtype(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply dtype logic.

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            If dtype list mismatches column count.
        """
        arr = _replace_missing(arr)

        if self.dtype is None:
            return _infer_dtype(arr)

        # Single dtype
        if isinstance(self.dtype, type):
            return arr.astype(self.dtype)

        # Per-column dtype
        if isinstance(self.dtype, list):
            if len(self.dtype) != arr.shape[1]:
                raise ValueError(
                    f"dtype length {len(self.dtype)} != number of columns {arr.shape[1]}"
                )

            cols = []
            
            # Per-column dtype conversion requires iterating over columns because each
            # column may have a different target dtype. The conversion within each column
            # is still handled by NumPy's vectorised astype().
            for i, dt in enumerate(self.dtype):
                col = arr[:, i]
                cols.append(col.astype(dt))
            return np.column_stack(cols)

        raise TypeError("dtype must be None, type, or list of types")

    def read(self) -> np.ndarray:
        """
        Load entire CSV into memory.

        Returns
        -------
        np.ndarray of shape (n_rows, n_cols)

        Raises
        ------
        FileNotFoundError
        ValueError
        """
        with open(self.filepath, "r", encoding=self.encoding, newline="") as f:
            reader = _csv.reader(f, delimiter=self.delimiter)
            if self.has_header:
                self.header = next(reader)
            else:
                self.header = None
            
            rows = list(reader)

        if not rows:
            raise ValueError("CSV file contains no data rows.")

        # Vectorised string stage
        arr = np.array(rows, dtype=str)
        arr = _clean_str_array(arr)

        # Handle case when only one row exists
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Apply missing value handling + dtype conversion
        return self._apply_dtype(arr)

    def read_chunked(self, chunk_size: int = 1000) -> Iterator[np.ndarray]:
        """
        Stream CSV in chunks.

        Parameters
        ----------
        chunk_size : int

        Yields
        ------
        np.ndarray of shape (chunk_size, n_cols)

        Notes
        -----
        This method is intended for streaming workflows. Each yielded chunk can be
        passed directly into preprocessing.partial_fit(), Pipeline.partial_fit(),
        or StreamTrainer.fit_chunk().

        Raises
        ------
        ValueError
            If chunk_size <= 0
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        with open(self.filepath, "r", encoding=self.encoding) as f:
            if self.has_header:
                self.header = next(f).strip().split(self.delimiter)
            else:
                self.header = None

            buffer = []
            for line in f:
                buffer.append(line)

                if len(buffer) == chunk_size:
                    arr = self._parse_lines(buffer)
                    yield self._apply_dtype(arr)
                    buffer = []

            if buffer:
                arr = self._parse_lines(buffer)
                yield self._apply_dtype(arr)

