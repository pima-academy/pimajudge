import torch

from pimajudge.week1.signatures.register import register


@register(id="create-simple-tensor")
def create_ones_tensor(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """
    Yêu cầu:
    1. Tạo một Tensor có kích thước được quy định bởi tham số `shape`.
    2. Tất cả các giá trị trong Tensor đều là số 1.
    3. Tensor phải có kiểu dữ liệu là `dtype`.
    
    Args:
        shape: Kích thước của Tensor (ví dụ: (2, 3)).
        dtype: Kiểu dữ liệu mong muốn (ví dụ: torch.int32).
        
    Returns:
        torch.Tensor: Tensor kết quả.
    """
    # TODO: Viết code của bạn ở đây
    return torch.randn(1)