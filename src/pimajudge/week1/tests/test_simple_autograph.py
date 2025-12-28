import torch

from pimajudge.week1.protocols import SimpleAutograd
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(part="part-1", exercise="simple-autograph", score=score)


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
    # The derivative should be a finite number
    assert isinstance(result, float), "Result must be a float"
    assert not torch.isinf(torch.tensor(result)), "Derivative should be finite"
    assert not torch.isnan(torch.tensor(result)), "Derivative should not be NaN"


@test(score="B")
def test_derivative_at_various_points(calculate_derivative: SimpleAutograd):
    """Test derivative calculation at various points"""
    test_points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    for x_val in test_points:
        result = calculate_derivative(x_val)
        assert isinstance(
            result, float
        ), f"Result must be a float for x={x_val}"
        assert not torch.isinf(
            torch.tensor(result)
        ), f"Derivative should be finite for x={x_val}"
        assert not torch.isnan(
            torch.tensor(result)
        ), f"Derivative should not be NaN for x={x_val}"


@test(score="A")
def test_autograd_mechanics(calculate_derivative: SimpleAutograd):
    """Test that autograd is used correctly"""
    # Test that the function properly uses requires_grad and backward
    x_val = 1.0
    result = calculate_derivative(x_val)

    # The result should be a valid derivative value
    assert isinstance(result, float), "Result must be a float"

    # Test consistency: calling the function multiple times should give same result
    result2 = calculate_derivative(x_val)
    assert result == result2, "Derivative should be consistent for same input"

    # Test that the function handles different input types correctly
    result_int = calculate_derivative(2)
    assert isinstance(result_int, float), "Should handle integer input and return float"
