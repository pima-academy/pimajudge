from pimajudge.week1.signatures.register import register


@register(id="simple-autograph")
def calculate_derivative_at_point(x_val: float) -> float:
    """
    Yêu cầu:
    Cho hàm số: f(x) dưới dạng đồ thị tính toán

    Hãy thực hiện các bước sau:
    1. Khởi tạo một Tensor vô hướng `x` từ giá trị `x_val`.
       Lưu ý: Để tính đạo hàm, cần thiết lập thuộc tính gì cho x?
    2. Định nghĩa hàm số `f` theo `x` (forward).
    3. Tính đạo hàm của `f` theo `x` (backward).
    4. Trả về giá trị đạo hàm (dạng float, không phải Tensor).

    Args:
        x_val: Giá trị x tại điểm cần tính đạo hàm.

    Returns:
        float: Giá trị đạo hàm dy/dx tại x_val.
    """
    # TODO: Viết code của bạn ở đây
    return 0.0
