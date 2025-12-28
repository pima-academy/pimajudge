import torch

from pimajudge.week1.protocols import Slicing
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(part="part-1", exercise="slicing", score=score)


@test(score="E")
def test_basic_execution(slice: Slicing):
    """Test that function can be called without errors"""

    test_tensor = torch.arange(16).reshape(4, 4)
    result = slice(test_tensor)
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="D")
def test_correct_shape(slice: Slicing):
    """Test that output has correct shape for 4x4 input"""
    test_tensor = torch.arange(16).reshape(4, 4)
    result = slice(test_tensor)
    assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"


@test(score="C")
def test_correct_values(slice: Slicing):
    """Test that output contains correct values"""
    test_tensor = torch.arange(16).reshape(4, 4)
    # Input:
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]
    # Expected output (even rows 0,2; odd cols 1,3):
    # [[ 1,  3],
    #  [ 9, 11]]
    result = slice(test_tensor)
    expected = torch.tensor([[1, 3], [9, 11]])
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


@test(score="B")
def test_various_sizes(slice: Slicing):
    """Test with various input sizes"""
    # Test 6x6
    test_tensor = torch.arange(36).reshape(6, 6)
    result = slice(test_tensor)
    assert result.shape == (3, 3), "Failed for 6x6 input"

    # Test 8x10
    test_tensor = torch.arange(80).reshape(8, 10)
    result = slice(test_tensor)
    assert result.shape == (4, 5), "Failed for 8x10 input"


@test(score="A")
def test_edge_cases(slice: Slicing):
    """Test edge cases"""
    # Test with 2x2 (minimum size)
    test_tensor = torch.arange(4).reshape(2, 2)
    result = slice(test_tensor)
    assert result.shape == (1, 1), "Failed for 2x2 input"
    assert result[0, 0] == 1, "Failed to extract correct value"

    # Test with larger matrix
    test_tensor = torch.arange(100).reshape(10, 10)
    result = slice(test_tensor)
    assert result.shape == (5, 5), "Failed for 10x10 input"
