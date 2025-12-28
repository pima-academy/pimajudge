import torch

from pimajudge.week1.signatures.register import register


@register(id="create-simple-tensor")
def create_ones_tensor(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(1)
