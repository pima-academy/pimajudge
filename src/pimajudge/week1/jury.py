from collections import defaultdict
from typing import Callable, Optional, Union

answers: defaultdict[str, Optional[Union[dict, Callable]]] = defaultdict(lambda: None)


def result(part: str):
    print("Hiện tại trình chấm tự động chưa được công bố, hãy chờ thông báo từ đội ngũ giảng dạy.")
    pass
