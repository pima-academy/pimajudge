import torch

from pimajudge.week1.protocols import CreateSimpleTensor
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(part="part-1", exercise="create-simple-tensor", score=score)


@test(score="E")
def test_basic_execution(create_simple_tensor: CreateSimpleTensor):
    """Test that function can be called without errors"""
    result = create_simple_tensor((2, 3), torch.int)
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="D")
def test_correct_shape(create_simple_tensor: CreateSimpleTensor):
    """Test that output has correct shape"""
    result = create_simple_tensor((2, 3), torch.int)
    assert result.shape == (2, 3), f"Expected shape (2, 3), got {result.shape}"


@test(score="D")
def test_correct_dtype(create_simple_tensor: CreateSimpleTensor):
    """Test that output has correct dtype"""
    # Test float32
    result = create_simple_tensor((2, 3), torch.float32)
    assert result.dtype == torch.float32, f"Expected torch.float32, got {result.dtype}"

    # Test int64
    result = create_simple_tensor((2, 3), torch.int64)
    assert result.dtype == torch.int64, f"Expected torch.int64, got {result.dtype}"


@test(score="C")
def test_all_ones(create_simple_tensor: CreateSimpleTensor):
    """Test that all values are 1"""
    result = create_simple_tensor((2, 3), torch.float32)
    assert torch.all(result == 1.0), "All values should be 1"


@test(score="B")
def test_various_shapes(create_simple_tensor: CreateSimpleTensor):
    """Test with various shapes"""
    shapes = [(5,), (3, 4), (2, 3, 4), (1, 1, 1, 1)]
    for shape in shapes:
        result = create_simple_tensor(shape, torch.float32)
        assert result.shape == shape, f"Failed for shape {shape}"
        assert torch.all(result == 1.0), f"Values not all ones for shape {shape}"


@test(score="A")
def test_edge_cases(create_simple_tensor: CreateSimpleTensor):
    """Test edge cases"""
    # Test with different dtypes
    dtypes = [
        torch.float16,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int64,
        torch.bool,
    ]
    for dtype in dtypes:
        result = create_simple_tensor((2, 2), dtype)
        assert result.dtype == dtype, f"Failed for dtype {dtype}"
        if dtype == torch.bool:
            assert torch.all(result == True), "Values not all True for bool dtype"  # noqa: E712
        else:
            assert torch.all(result == 1), f"Values not all ones for dtype {dtype}"
