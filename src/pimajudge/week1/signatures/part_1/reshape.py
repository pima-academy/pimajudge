import torch

from pimajudge.week1.signatures.register import register


@register(id="reshape")
def safe_flatten(t: torch.Tensor) -> torch.Tensor:
    """
    Yêu cầu:
    Viết hàm nhận vào một Tensor `t` có số chiều bất kỳ.
    Hàm cần trả về một Tensor 1 chiều (vector) chứa tất cả phần tử của `t`.

    Lưu ý quan trọng:
    - Hàm phải hoạt động đúng ngay cả khi `t` không liên tục trong bộ nhớ (non-contiguous).
    - Gợi ý: Hãy nhớ lại sự khác biệt giữa .view() và .reshape().

    Args:
        t: Tensor đầu vào (shape bất kỳ).

    Returns:
        torch.Tensor: Tensor 1 chiều (shape: (N,)).
    """  # noqa: E501
    # TODO: Viết code của bạn ở đây
    return torch.randn(1)
