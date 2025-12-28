from collections import defaultdict
from collections.abc import Callable

answers: defaultdict[str, dict | Callable | None] = defaultdict(lambda: None)


def result(part: str):
    print("Hiện tại trình chấm tự động chưa được công bố, hãy chờ thông báo từ đội ngũ giảng dạy.")  # noqa: E501
    pass
