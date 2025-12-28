from collections import defaultdict
from collections.abc import Callable

signatures: defaultdict[str, dict | Callable | None] = defaultdict(lambda: None)


def register(id: str) -> Callable[[Callable], Callable]:
    """
    Usage:
    @register(id='...')
    def func(...) -> ...:
        ...
    """
    def wrapper(func: Callable) -> Callable:
        signatures[id] = func
        return func

    return wrapper
