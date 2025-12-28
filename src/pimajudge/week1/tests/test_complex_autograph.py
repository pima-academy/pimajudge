import torch

from pimajudge.week1.protocols import ComplexAutograd
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(part="part-1", exercise="complex-autograph", score=score)


@test(score="E")
def test_basic_execution(clean_gradient_step: ComplexAutograd):
    """Test that function can be called without errors"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    x = torch.tensor([3.0, 4.0])
    result = clean_gradient_step(w, x)
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="D")
def test_return_gradient(clean_gradient_step: ComplexAutograd):
    """Test that function returns the gradient tensor"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    x = torch.tensor([3.0, 4.0])
    result = clean_gradient_step(w, x)
    assert result is w.grad, "Should return w.grad"
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="C")
def test_gradient_calculation(clean_gradient_step: ComplexAutograd):
    """Test that gradient is calculated correctly"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    x = torch.tensor([3.0, 4.0])

    result = clean_gradient_step(w, x)

    # Loss = (w * x).sum() = w[0]*x[0] + w[1]*x[1]
    # dL/dw[0] = x[0] = 3.0
    # dL/dw[1] = x[1] = 4.0
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(
        result, expected, atol=1e-5
    ), f"Expected {expected}, got {result}"


@test(score="B")
def test_gradient_reset(clean_gradient_step: ComplexAutograd):
    """Test that old gradient is properly reset"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    x = torch.tensor([3.0, 4.0])

    # Create some old gradient
    w.grad = torch.tensor([100.0, 200.0])

    result = clean_gradient_step(w, x)

    # The new gradient should be based on current computation, not accumulated
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(
        result, expected, atol=1e-5
    ), f"Expected {expected}, got {result}"
    assert not torch.allclose(
        result, torch.tensor([100.0, 200.0]), atol=1e-5
    ), "Old gradient should be reset"


@test(score="A")
def test_gradient_accumulation_prevention(clean_gradient_step: ComplexAutograd):
    """Test that gradient accumulation is properly prevented"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    x = torch.tensor([3.0, 4.0])

    # First computation
    result1 = clean_gradient_step(w, x)
    expected1 = torch.tensor([3.0, 4.0])
    assert torch.allclose(result1, expected1, atol=1e-5), "First gradient incorrect"

    # Second computation with same inputs
    result2 = clean_gradient_step(w, x)
    expected2 = torch.tensor([3.0, 4.0])
    assert torch.allclose(result2, expected2, atol=1e-5), "Second gradient incorrect"

    # Gradients should be the same (not accumulated)
    assert torch.allclose(
        result1, result2, atol=1e-5
    ), "Gradients should not accumulate"

    # Test with different x values
    x2 = torch.tensor([5.0, 6.0])
    result3 = clean_gradient_step(w, x2)
    expected3 = torch.tensor([5.0, 6.0])
    assert torch.allclose(
        result3, expected3, atol=1e-5
    ), "Gradient should match new x values"
    assert not torch.allclose(
        result3, result1, atol=1e-5
    ), "Gradient should change with different x"
