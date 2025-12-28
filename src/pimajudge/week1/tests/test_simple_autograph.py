import numpy as np
import torch

from pimajudge.week1.protocols import SimpleAutograd
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(part="part-1", exercise="simple-autograph", score=score)


def f(x: float) -> float:
    """Reference function: f(x) = exp(sin(x)) + cos(x²)"""
    f1 = np.exp(np.sin(x))
    f2 = np.cos(x**2)
    return f1 + f2


def df(x: float) -> float:
    """Analytical derivative: df/dx = exp(sin(x)) * cos(x) - sin(x²) * 2x"""
    df1 = np.exp(np.sin(x)) * np.cos(x)
    df2 = -np.sin(x**2) * 2 * x
    return df1 + df2


@test(score="E")
def test_basic_execution(calculate_derivative: SimpleAutograd):
    """Test that function can be called without errors"""
    result = calculate_derivative(1.0)
    assert isinstance(result, float), "Result must be a float"


@test(score="D")
def test_return_type(calculate_derivative: SimpleAutograd):
    """Test that function returns a float, not a Tensor"""
    result = calculate_derivative(1.0)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert not isinstance(result, torch.Tensor), "Should not return a Tensor"


@test(score="C")
def test_derivative_at_zero(calculate_derivative: SimpleAutograd):
    """Test derivative calculation at x=0"""
    result = calculate_derivative(0.0)
    expected = df(0.0)
    # At x=0: df/dx = exp(sin(0)) * cos(0) - sin(0) * 0 = exp(0) * 1 - 0 = 1
    assert isinstance(result, float), "Result must be a float"
    assert not torch.isinf(torch.tensor(result)), "Derivative should be finite"
    assert not torch.isnan(torch.tensor(result)), "Derivative should not be NaN"
    assert np.isclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"


@test(score="B")
def test_derivative_at_various_points(calculate_derivative: SimpleAutograd):
    """Test derivative calculation at various points"""
    test_points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    for x_val in test_points:
        result = calculate_derivative(x_val)
        expected = df(x_val)
        assert isinstance(
            result, float
        ), f"Result must be a float for x={x_val}"
        assert not torch.isinf(
            torch.tensor(result)
        ), f"Derivative should be finite for x={x_val}"
        assert not torch.isnan(
            torch.tensor(result)
        ), f"Derivative should not be NaN for x={x_val}"
        assert np.isclose(
            result, expected, atol=1e-5
        ), f"At x={x_val}: expected {expected}, got {result}"


@test(score="A")
def test_autograd_mechanics(calculate_derivative: SimpleAutograd):
    """Test that autograd is used correctly and produces accurate results"""
    # Test that the function properly uses requires_grad and backward
    x_val = 1.0
    result = calculate_derivative(x_val)
    expected = df(x_val)

    # The result should be a valid derivative value
    assert isinstance(result, float), "Result must be a float"

    # Test accuracy: should match analytical derivative
    assert np.isclose(
        result, expected, atol=1e-5
    ), f"Expected {expected}, got {result}"

    # Test consistency: calling the function multiple times should give same result
    result2 = calculate_derivative(x_val)
    assert result == result2, "Derivative should be consistent for same input"

    # Test that the function handles different input types correctly
    result_int = calculate_derivative(2)
    expected_int = df(2.0)
    assert isinstance(result_int, float), "Should handle integer input and return float"
    assert np.isclose(
        result_int, expected_int, atol=1e-5
    ), f"Expected {expected_int}, got {result_int}"

    # Test at a specific point with known value
    # At x=π/2: sin(π/2)=1, cos(π/2)=0, sin((π/2)²)=sin(π²/4)
    x_special = np.pi / 2
    result_special = calculate_derivative(x_special)
    expected_special = df(x_special)
    assert np.isclose(
        result_special, expected_special, atol=1e-5
    ), f"At x=π/2: expected {expected_special}, got {result_special}"
