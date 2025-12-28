import importlib
from pathlib import Path

from pimajudge.week1.tests.registry import Part, Score, TestCase, TestRegistry, registry

# Get the directory of this __init__.py file
_tests_dir = Path(__file__).parent

# Find all subdirectories that are test modules (part_1, part_2, etc.)
_test_modules = [
    d.name
    for d in _tests_dir.iterdir()
    if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()
]

# Dynamically import all test module directories to register tests
for _module_name in _test_modules:
    importlib.import_module(f".{_module_name}", package=__package__)

__all__ = ["registry", "TestRegistry", "Part", "Score", "TestCase"]
