import torch

from pimajudge.week1.protocols import Reshape
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(part="part-1", exercise="reshape", score=score)


@test(score="E")
def test_basic_execution(reshape: Reshape):
    """Test that function can be called without errors"""
    test_tensor = torch.arange(12).reshape(3, 4)
    result = reshape(test_tensor)
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="D")
def test_correct_shape(reshape: Reshape):
    """Test that output has correct shape (1D)"""
    test_tensor = torch.arange(12).reshape(3, 4)
    result = reshape(test_tensor)
    assert result.shape == (12,), f"Expected shape (12,), got {result.shape}"


@test(score="C")
def test_correct_values(reshape: Reshape):
    """Test that output contains correct values in order"""
    test_tensor = torch.arange(12).reshape(3, 4)
    result = reshape(test_tensor)
    expected = torch.arange(12)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


@test(score="B")
def test_various_shapes(reshape: Reshape):
    """Test with various input shapes"""
    # Test 1D input
    test_tensor = torch.arange(6)
    result = reshape(test_tensor)
    assert result.shape == (6,), "Failed for 1D input"

    # Test 2D input
    test_tensor = torch.arange(12).reshape(3, 4)
    result = reshape(test_tensor)
    assert result.shape == (12,), "Failed for 2D input"

    # Test 3D input
    test_tensor = torch.arange(24).reshape(2, 3, 4)
    result = reshape(test_tensor)
    assert result.shape == (24,), "Failed for 3D input"


@test(score="A")
def test_edge_cases(reshape: Reshape):
    """Test edge cases including non-contiguous tensors"""
    # Test with non-contiguous tensor (transposed)
    test_tensor = torch.arange(12).reshape(3, 4).t()
    result = reshape(test_tensor)
    assert result.shape == (12,), "Failed for non-contiguous tensor"
    assert result.dtype == test_tensor.dtype, "Dtype should be preserved"

    # Test with single element
    test_tensor = torch.tensor([5])
    result = reshape(test_tensor)
    assert result.shape == (1,), "Failed for single element"
    assert result[0] == 5, "Value should be preserved"

    # Test with empty tensor
    test_tensor = torch.tensor([])
    result = reshape(test_tensor)
    assert result.shape == (0,), "Failed for empty tensor"
