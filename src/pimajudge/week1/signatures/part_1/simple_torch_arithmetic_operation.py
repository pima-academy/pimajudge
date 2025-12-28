import torch

from pimajudge.week1.signatures.register import register


@register(id="simple-torch-arithmetic-operation")
def linear_transform(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Yêu cầu:
    Tính toán đầu ra của một lớp tuyến tính: Y = X @ W + b.

    Kích thước đầu vào:
    - X: (Batch_Size, In_Features) - ví dụ: (32, 10)
    - W: (In_Features, Out_Features) - ví dụ: (10, 5)
    - b: (Out_Features,) - ví dụ: (5,)

    Gợi ý:
    - Sử dụng phép nhân ma trận cho X và W.
    - Sử dụng broadcasting cho phép cộng với b.

    Args:
        X: Input tensor.
        W: Weight tensor.
        b: Bias tensor.

    Returns:
        torch.Tensor: Kết quả Y có kích thước (Batch_Size, Out_Features).
    """
    # TODO: Viết code của bạn ở đây
    return torch.randn(1)
