import torch

from pimajudge.week1.signatures.register import register


@register(id="complex-autograph")
def clean_gradient_step(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.randn(1)
