import torch

from pimajudge.week1.signatures.register import register


@register(id="slicing")
def slice_even_rows_odd_cols(t: torch.Tensor) -> torch.Tensor:
    """
    Yêu cầu:
    Cho một Tensor 2 chiều (ma trận) bất kỳ `t`. Hãy trích xuất một Tensor con thỏa mãn:
    1. Lấy các hàng ở chỉ số chẵn (0, 2, 4,...).
    2. Lấy các cột ở chỉ số lẻ (1, 3, 5,...).
    
    Ví dụ: 
    Input (4x4):
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11],
     [12, 13, 14, 15]]
     
    Output:
    [[ 1,  3],  <- Hàng 0, lấy cột 1 và 3
     [ 9, 11]]  <- Hàng 2, lấy cột 1 và 3
    
    Args:
        t: Tensor đầu vào 2 chiều.
        
    Returns:
        torch.Tensor: Tensor kết quả sau khi cắt.
    """
    # TODO: Viết code của bạn ở đây
    return torch.randn(1)
