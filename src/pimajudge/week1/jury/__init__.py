import importlib
from pathlib import Path

from pimajudge.week1.jury.answers import answers
from pimajudge.week1.jury.jury import detailed_result, result

# Flag to track if tests have been loaded
_tests_loaded = False


def _load_tests():
    """Lazy load test modules to avoid circular imports."""
    global _tests_loaded
    if _tests_loaded:
        return

    # Get the directory of this __init__.py file
    module_dir = Path(__file__).parent

    # Find all subdirectories that contain test modules (part_1, part_2, etc.)
    test_modules = [
        d.name
        for d in module_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()
    ]

    # Dynamically import all test module directories to register tests
    for module_name in test_modules:
        importlib.import_module(f".{module_name}", package=__package__)

    _tests_loaded = True


# Load tests when result() or detailed_result() is called
_original_result = result
_original_detailed_result = detailed_result


def result(part: str):
    """Evaluate and display results for a given part."""
    _load_tests()
    return _original_result(part)


def detailed_result(part: str, group: str):
    """Get detailed test results for a specific group."""
    _load_tests()
    return _original_detailed_result(part, group)


__all__ = [
    "answers",
    "result",
    "detailed_result",
]
