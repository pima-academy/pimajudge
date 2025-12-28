import torch

from pimajudge.week1.collector.collector import answers
from pimajudge.week1.jury.test_decorator import judge


@judge.test(part="part-1", group="create_simple_tensor", score="F")
def test_basic_execution():
    """Test that function can be called without errors"""
    func = answers["create-simple-tensor"]
    result = func((2, 3), torch.float32)
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@judge.test(part="part-1", group="create_simple_tensor", score="D")
def test_correct_shape():
    """Test that output has correct shape"""
    func = answers["create-simple-tensor"]
    result = func((2, 3), torch.float32)
    assert result.shape == (2, 3), f"Expected shape (2, 3), got {result.shape}"


@judge.test(part="part-1", group="create_simple_tensor", score="D")
def test_correct_dtype():
    """Test that output has correct dtype"""
    func = answers["create-simple-tensor"]

    # Test float32
    result = func((2, 3), torch.float32)
    assert result.dtype == torch.float32, f"Expected torch.float32, got {result.dtype}"

    # Test int32
    result = func((2, 3), torch.int32)
    assert result.dtype == torch.int32, f"Expected torch.int32, got {result.dtype}"


@judge.test(part="part-1", group="create_simple_tensor", score="C")
def test_all_ones():
    """Test that all values are 1"""
    func = answers["create-simple-tensor"]
    result = func((2, 3), torch.float32)
    assert torch.all(result == 1.0), "All values should be 1"


@judge.test(part="part-1", group="create_simple_tensor", score="B")
def test_various_shapes():
    """Test with various shapes"""
    func = answers["create-simple-tensor"]

    shapes = [(5,), (3, 4), (2, 3, 4), (1, 1, 1, 1)]
    for shape in shapes:
        result = func(shape, torch.float32)
        assert result.shape == shape, f"Failed for shape {shape}"
        assert torch.all(result == 1.0), f"Values not all ones for shape {shape}"


@judge.test(part="part-1", group="create_simple_tensor", score="A")
def test_edge_cases():
    """Test edge cases"""
    func = answers["create-simple-tensor"]

    # Test with different dtypes
    dtypes = [
        torch.float16,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int64,
        torch.bool
    ]
    for dtype in dtypes:
        result = func((2, 2), dtype)
        assert result.dtype == dtype, f"Failed for dtype {dtype}"
        if dtype == torch.bool:
            assert torch.all(result == True), "Values not all True for bool dtype"  # noqa: E712
        else:
            assert torch.all(result == 1), f"Values not all ones for dtype {dtype}"
