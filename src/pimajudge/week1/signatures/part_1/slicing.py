import torch

from pimajudge.week1.signatures.register import register


@register(id="slicing")
def slice_even_rows_odd_cols(t: torch.Tensor) -> torch.Tensor:
    return torch.randn(1)
