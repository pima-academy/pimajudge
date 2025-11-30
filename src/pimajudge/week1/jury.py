from collections import defaultdict
from typing import Callable, Optional, Union

answers: defaultdict[str, Optional[Union[dict, Callable]]] = defaultdict(lambda: None)


def result():
    pass
