from collections import defaultdict
from typing import Callable, Optional, Union

signatures: defaultdict[str, Optional[Union[dict, Callable]]] = defaultdict(lambda: None)

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
