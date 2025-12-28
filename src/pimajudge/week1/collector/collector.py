import inspect
from collections.abc import Callable

from pimajudge.week1.jury import answers
from pimajudge.week1.signatures import signatures


def collector(id: str) -> Callable[[Callable], Callable]:
    def wrapper(func: Callable) -> Callable:
        # Check signature
        expected_func = signatures.get(id)
        if expected_func is None:
            raise ValueError(f"Không tìm thấy signature cho id: {id}")

        if not callable(expected_func):
            raise TypeError(f"Signature cho {id} không phải là một hàm")

        expected_sig = inspect.signature(expected_func)
        actual_sig = inspect.signature(func)

        if expected_sig != actual_sig:
            raise TypeError(
                f"Signature không khớp cho {id}!\n"
                f"Mong đợi: {expected_sig}\n"
                f"Nhận được: {actual_sig}"
            )

        answers[id] = func
        print("Bạn đã thành công nộp bài!!!")
        return func

    return wrapper
