"""Answers registry for storing student submissions."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from pimajudge.week1.exceptions import FunctionNotFoundError


class AnswersDict(defaultdict):
    """Custom dict that raises an exception when accessing None values."""

    def __init__(self):
        super().__init__(lambda: None)

    def get(self, key: str, default: Any = None) -> Callable:
        """
        Get a function from the registry.

        Args:
            key: The function identifier
            default: Default value (not used, kept for compatibility)

        Returns:
            The registered function

        Raises:
            FunctionNotFoundError: If the function is not registered or is None
        """
        value = super().get(key, None)
        if value is None:
            msg = (
                f"Function '{key}' not found in answers registry. "
                f"Make sure the function is properly registered with @collector."
            )
            raise FunctionNotFoundError(msg)
        return value

    def __getitem__(self, key: str) -> Callable:
        """
        Get a function using bracket notation.

        Args:
            key: The function identifier

        Returns:
            The registered function

        Raises:
            FunctionNotFoundError: If the function is not registered or is None
        """
        value = super().__getitem__(key)
        if value is None:
            msg = (
                f"Function '{key}' not found in answers registry. "
                f"Make sure the function is properly registered with @collector."
            )
            raise FunctionNotFoundError(msg)
        return value


answers: AnswersDict = AnswersDict()
