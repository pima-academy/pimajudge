from collections import defaultdict
from typing import Callable, Optional

exercise_names: dict[int, str] = {
    1: "Khởi tạo và Kiểu dữ liệu",
    2: "Thao tác Tensor (Slicing)",
    3: "Biến đổi hình dạng (Reshape)",
    4: "Autograd cơ bản",
    5: "Autograd nâng cao (Quản lý Gradient)",
}

answers: defaultdict[int, Optional[Callable]] = defaultdict(lambda: None)


def result():
    pass
