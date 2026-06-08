import numpy as np
import pytest

from numcompute.io import CSVReader


def write_csv(tmp_path, content):
    path = tmp_path / "data.csv"
    path.write_text(content)
    return str(path)


# -----------------------------
# 1. Basic correctness
# -----------------------------
def test_basic_csv_load(tmp_path):
    path = write_csv(tmp_path, "a,b,c\n1,2,3\n4,5,6")

    reader = CSVReader(path)
    data = reader.read()

    assert data.shape == (2, 3)
    assert data.dtype != object


# -----------------------------
# 2. Missing values handling
# -----------------------------
def test_missing_values(tmp_path):
    path = write_csv(tmp_path, "1,2\nNA,4\n3,NA")

    reader = CSVReader(path)
    data = reader.read()

    assert np.isnan(data).any()  # ensures NaN handling works


# -----------------------------
# 3. Custom delimiter support
# -----------------------------
def test_custom_delimiter(tmp_path):
    path = write_csv(tmp_path, "1|2|3\n4|5|6")

    reader = CSVReader(path, delimiter="|", has_header=False)
    data = reader.read()

    assert data.shape == (2, 3)


# -----------------------------
# 4. Chunked reading consistency
# -----------------------------
def test_chunked_vs_full(tmp_path):
    path = write_csv(tmp_path, "1,2\n3,4\n5,6\n7,8")

    reader1 = CSVReader(path, has_header=False)
    full = reader1.read()

    reader2 = CSVReader(path, has_header=False)
    chunks = np.vstack(list(reader2.read_chunked(chunk_size=2)))

    np.testing.assert_array_equal(full, chunks)