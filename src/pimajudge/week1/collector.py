from typing import Callable

from pimajudge.week1.jury import answers


def collector(id: str) -> Callable[[Callable], Callable]:
    def wrapper(func: Callable) -> Callable:
        answers[id] = func
        print("Bạn đã thành công nộp bài!")
        return func

    return wrapper
