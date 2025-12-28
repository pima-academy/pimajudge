"""Custom exceptions for the jury module."""


class FunctionNotFoundError(Exception):
    """Raised when a function is not found in the answers registry."""
    pass
