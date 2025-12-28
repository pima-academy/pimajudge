import torch

from pimajudge.week1.signatures.register import register


@register(id="reshape")
def safe_flatten(t: torch.Tensor) -> torch.Tensor:
    return torch.randn(1)
