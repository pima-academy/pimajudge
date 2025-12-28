import torch

from pimajudge.week1.signatures.register import register


@register(id="simple-torch-arithmetic-operation")
def linear_transform(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.randn(1)
