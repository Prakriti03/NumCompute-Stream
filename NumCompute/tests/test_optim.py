"""
tests/test_optim.py — Unit tests for optim.py
"""

import pytest
import numpy as np
from numcompute.optim import grad, jacobian, _validate_x, _validate_h, _validate_method, _validate_scalar_output


# validator tests 

def test_validate_x_rejects_empty():
    with pytest.raises(ValueError):
        _validate_x(np.array([]))

def test_validate_x_rejects_2d():
    with pytest.raises(ValueError):
        _validate_x(np.array([[1.0, 2.0]]))

def test_validate_x_rejects_nan():
    with pytest.raises(ValueError):
        _validate_x(np.array([1.0, np.nan]))

def test_validate_x_rejects_inf():
    with pytest.raises(ValueError):
        _validate_x(np.array([np.inf, 1.0]))

def test_validate_h_rejects_zero():
    with pytest.raises(ValueError):
        _validate_h(0.0)

def test_validate_h_rejects_negative():
    with pytest.raises(ValueError):
        _validate_h(-1e-5)

def test_validate_method_rejects_invalid():
    with pytest.raises(ValueError):
        _validate_method("backward")

def test_validate_method_case_insensitive():
    assert _validate_method("CENTRAL") == "central"

def test_validate_scalar_output_rejects_array():
    with pytest.raises(ValueError):
        _validate_scalar_output(np.array([1.0, 2.0]))

def test_validate_scalar_output_rejects_nan():
    with pytest.raises(ValueError):
        _validate_scalar_output(np.nan)


# grad tests

def test_grad_quadratic_central():
    f = lambda x: np.sum(x ** 2)
    g = grad(f, np.array([3.0, 4.0]))
    np.testing.assert_allclose(g, [6.0, 8.0], rtol=1e-5)

def test_grad_quadratic_forward():
    f = lambda x: np.sum(x ** 2)
    g = grad(f, np.array([3.0, 4.0]), method="forward")
    np.testing.assert_allclose(g, [6.0, 8.0], rtol=1e-4)

def test_grad_at_zero():
    f = lambda x: np.sum(x ** 2)
    g = grad(f, np.array([0.0, 0.0]))
    np.testing.assert_allclose(g, [0.0, 0.0], atol=1e-8)

def test_grad_non_contiguous_input():
    x = np.array([1.0, 99.0, 2.0, 99.0])[::2]  # non-contiguous stride
    f = lambda x: np.sum(x ** 2)
    np.testing.assert_allclose(grad(f, x), [2.0, 4.0], rtol=1e-5)

def test_grad_rejects_non_callable():
    with pytest.raises(TypeError):
        grad("not a function", np.array([1.0]))

def test_grad_rejects_nan_input():
    with pytest.raises(ValueError):
        grad(lambda x: np.sum(x), np.array([np.nan]))

def test_grad_rejects_array_output():
    with pytest.raises(ValueError):
        grad(lambda x: x * 2, np.array([1.0, 2.0]))

def test_grad_rejects_invalid_method():
    with pytest.raises(ValueError):
        grad(lambda x: np.sum(x), np.array([1.0]), method="bad")


# jacobian tests 

def test_jacobian_shape():
    # F: R^3 -> R^2, J should be (2, 3)
    F = lambda x: np.array([x[0] + x[1], x[1] + x[2]])
    assert jacobian(F, np.array([1.0, 2.0, 3.0])).shape == (2, 3)

def test_jacobian_nonlinear():
    # F(x) = [x0*x1, x0+x1], J at [1,2] = [[2,1],[1,1]]
    F = lambda x: np.array([x[0] * x[1], x[0] + x[1]])
    np.testing.assert_allclose(jacobian(F, np.array([1.0, 2.0])), [[2.0, 1.0], [1.0, 1.0]], rtol=1e-5)

def test_jacobian_forward_method():
    F = lambda x: np.array([x[0] + x[1], x[0] - x[1]])
    np.testing.assert_allclose(jacobian(F, np.array([1.0, 2.0]), method="forward"), [[1, 1], [1, -1]], rtol=1e-4)

def test_jacobian_rejects_nan_output():
    with pytest.raises(ValueError):
        jacobian(lambda x: np.array([np.nan]), np.array([1.0, 2.0]))

def test_jacobian_rejects_2d_output():
    with pytest.raises(ValueError):
        jacobian(lambda x: np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))

def test_jacobian_scalar_output():
    F = lambda x: x[0] ** 2 + x[1] ** 2
    J = jacobian(F, np.array([3.0, 4.0]))
    np.testing.assert_allclose(J, [[6.0, 8.0]], rtol=1e-5)