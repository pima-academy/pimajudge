import inspect
from collections.abc import Callable

from pimajudge.week1.answers import answers
from pimajudge.week1.protocols import protocols
from pimajudge.week1.types import Exercise


def collector(id: Exercise) -> Callable[[Callable], Callable]:
    def wrapper(func: Callable) -> Callable:
        protocol_cls = protocols.get(id)
        if protocol_cls is None:
            raise ValueError(f"Không tìm thấy protocol cho id: {id}")

        if not isinstance(func, protocol_cls):
            raise TypeError(f"Hàm không tuân thủ Protocol {protocol_cls.__name__}")

        expected_sig = inspect.signature(protocol_cls.__call__)
        expected_params = list(expected_sig.parameters.values())[1:]
        expected_sig = expected_sig.replace(parameters=expected_params)

        actual_sig = inspect.signature(func)

        if expected_sig != actual_sig:
            raise TypeError(
                f"Sai signature cho {id}!\n"
                f"Mong đợi: {expected_sig}\n"
                f"Nhận được: {actual_sig}"
            )

        answers[id] = func
        print(f"✅ Nộp bài thành công: {id}")
        return func

    return wrapper
