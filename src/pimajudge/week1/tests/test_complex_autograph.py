import torch

from pimajudge.week1.protocols import ComplexAutograd
from pimajudge.week1.tests import Score, registry


def test(score: Score):
    return registry.test(part="part-1", exercise="complex-autograph", score=score)


@test(score="E")
def test_basic_execution(clean_gradient_step: ComplexAutograd):
    """Test that function can be called without errors"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    w.grad = torch.rand_like(w)
    x = torch.tensor([3.0, 4.0])
    result = clean_gradient_step(w, x)
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="D")
def test_return_gradient(clean_gradient_step: ComplexAutograd):
    """Test that function returns the gradient tensor"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    w.grad = torch.rand_like(w)
    x = torch.tensor([3.0, 4.0])
    result = clean_gradient_step(w, x)
    # Ensure gradient is not None (lazily propagated after backward)
    assert w.grad is not None, "w.grad should not be None after backward"
    # Result should match w.grad in value
    assert torch.allclose(result, w.grad, atol=1e-5), "Result should match w.grad"
    assert isinstance(result, torch.Tensor), "Result must be a Tensor"


@test(score="C")
def test_gradient_calculation(clean_gradient_step: ComplexAutograd):
    """Test that gradient is calculated correctly"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    w.grad = torch.rand_like(w)
    x = torch.tensor([3.0, 4.0])

    result = clean_gradient_step(w, x)

    # Ensure gradient is not None (lazily propagated after backward)
    assert w.grad is not None, "w.grad should not be None after backward"

    # Loss = (w * x).sum() = w[0]*x[0] + w[1]*x[1]
    # dL/dw[0] = x[0] = 3.0
    # dL/dw[1] = x[1] = 4.0
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(result, expected, atol=1e-5), (
        f"Expected {expected}, got {result}"
    )


@test(score="B")
def test_gradient_reset(clean_gradient_step: ComplexAutograd):
    """Test that old gradient is properly reset"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    w.grad = torch.rand_like(w)
    x = torch.tensor([3.0, 4.0])

    # Create some old gradient
    w.grad = torch.tensor([100.0, 200.0])

    result = clean_gradient_step(w, x)

    # Ensure gradient is not None after backward
    assert w.grad is not None, "w.grad should not be None after backward"

    # The new gradient should be based on current computation, not accumulated
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(result, expected, atol=1e-5), (
        f"Expected {expected}, got {result}"
    )
    assert not torch.allclose(result, torch.tensor([100.0, 200.0]), atol=1e-5), (
        "Old gradient should be reset"
    )


@test(score="A")
def test_gradient_accumulation_prevention(clean_gradient_step: ComplexAutograd):
    """Test that gradient accumulation is properly prevented"""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    w.grad = torch.rand_like(w)
    x = torch.tensor([3.0, 4.0])

    # First computation
    result1 = clean_gradient_step(w, x)
    assert w.grad is not None, "w.grad should not be None after backward"
    expected1 = torch.tensor([3.0, 4.0])
    assert torch.allclose(result1, expected1, atol=1e-5), "First gradient incorrect"

    # Store the first gradient value
    first_grad_value = result1.clone()

    # Second computation with same inputs
    result2 = clean_gradient_step(w, x)
    assert w.grad is not None, "w.grad should not be None after backward"
    expected2 = torch.tensor([3.0, 4.0])
    assert torch.allclose(result2, expected2, atol=1e-5), "Second gradient incorrect"

    # Gradients should be the same (not accumulated)
    # Compare with the stored value, not result1 which is the same object as w.grad
    assert torch.allclose(first_grad_value, result2, atol=1e-5), (
        "Gradients should not accumulate"
    )

    # Test with different x values
    x2 = torch.tensor([5.0, 6.0])
    result3 = clean_gradient_step(w, x2)
    assert w.grad is not None, "w.grad should not be None after backward"
    expected3 = torch.tensor([5.0, 6.0])
    assert torch.allclose(result3, expected3, atol=1e-5), (
        "Gradient should match new x values"
    )
    assert not torch.allclose(result3, first_grad_value, atol=1e-5), (
        "Gradient should change with different x"
    )
