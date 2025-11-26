from typing import Callable

from pimajudge.week1.jury import answers, exercise_names


def collector(id: int) -> Callable[[Callable], Callable]:
    def wrapper(func: Callable) -> Callable:
        answers[id] = func
        print(f"Bạn đã thành công nộp bài ``{exercise_names[id]}''!")
        return func

    return wrapper
