import torch

from pimajudge.week1.signatures.register import register


@register(id="complex-autograph")
def clean_gradient_step(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Yêu cầu:
    Giả sử `w` là tham số mô hình (weights) và `x` là dữ liệu đầu vào.
    Hàm số mục tiêu (Loss) là: L = (w * x).sum()
    
    Tham số `w` đầu vào ĐÃ CÓ SẴN giá trị `.grad` từ các bước tính toán trước đó (rác).
    Nhiệm vụ của bạn là tính đạo hàm chính xác của L theo w cho LẦN TÍNH TOÁN NÀY mà thôi.
    
    Các bước cần làm:
    1. Xóa sạch giá trị gradient cũ đang tồn tại trong `w`.
    2. Tính toán hàm L.
    3. Gọi backward để tính đạo hàm mới.
    4. Trả về giá trị đạo hàm mới của w (`w.grad`).
    
    Args:
        w: Tensor tham số (requires_grad=True, đã có sẵn .grad != None).
        x: Tensor dữ liệu.
        
    Returns:
        torch.Tensor: Gradient mới của w.
    """
    return torch.randn(1)
