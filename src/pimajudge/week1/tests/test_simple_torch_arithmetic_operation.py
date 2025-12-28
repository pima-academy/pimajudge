import torch

from pimajudge.week1.protocols import SimpleArithmetic
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(
        part="part-1",
        exercise="simple-torch-arithmetic-operation",
        score=score,
    )


@test(score="E")
def test_basic_execution(linear_transform: SimpleArithmetic):
    """Test that function can be called without errors"""
    X = torch.randn(2, 3)
    W = torch.randn(3, 4)
    b = torch.randn(4)
    result = linear_transform(X, W, b)
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="D")
def test_correct_shape(linear_transform: SimpleArithmetic):
    """Test that output has correct shape"""
    X = torch.randn(2, 3)
    W = torch.randn(3, 4)
    b = torch.randn(4)
    result = linear_transform(X, W, b)
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"


@test(score="C")
def test_correct_values(linear_transform: SimpleArithmetic):
    """Test that output contains correct values"""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([0.5, 0.5])

    # Expected: X @ W + b
    # [[1*1+2*0, 1*0+2*1], [3*1+4*0, 3*0+4*1]] + [0.5, 0.5]
    # [[1, 2], [3, 4]] + [0.5, 0.5]
    # [[1.5, 2.5], [3.5, 4.5]]
    result = linear_transform(X, W, b)
    expected = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    assert torch.allclose(result, expected, atol=1e-5), (
        f"Expected {expected}, got {result}"
    )


@test(score="B")
def test_various_sizes(linear_transform: SimpleArithmetic):
    """Test with various input sizes"""
    # Test larger batch
    X = torch.randn(32, 10)
    W = torch.randn(10, 5)
    b = torch.randn(5)
    result = linear_transform(X, W, b)
    assert result.shape == (32, 5), "Failed for batch size 32"

    # Test single sample
    X = torch.randn(1, 10)
    W = torch.randn(10, 5)
    b = torch.randn(5)
    result = linear_transform(X, W, b)
    assert result.shape == (1, 5), "Failed for batch size 1"

    # Test different feature dimensions
    X = torch.randn(8, 20)
    W = torch.randn(20, 15)
    b = torch.randn(15)
    result = linear_transform(X, W, b)
    assert result.shape == (8, 15), "Failed for different feature dimensions"


@test(score="A")
def test_edge_cases(linear_transform: SimpleArithmetic):
    """Test edge cases"""
    # Test with zero bias
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.zeros(2)
    result = linear_transform(X, W, b)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected, atol=1e-5), "Failed with zero bias"

    # Test with negative values
    X = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([1.0, 1.0])
    result = linear_transform(X, W, b)
    expected = torch.tensor([[0.0, -1.0], [-2.0, -3.0]])
    assert torch.allclose(result, expected, atol=1e-5), "Failed with negative values"

    # Test broadcasting with different dtypes
    X = torch.randn(2, 3, dtype=torch.float32)
    W = torch.randn(3, 4, dtype=torch.float32)
    b = torch.randn(4, dtype=torch.float32)
    result = linear_transform(X, W, b)
    assert result.shape == (2, 4), "Failed with float32 dtype"
