import importlib
from pathlib import Path

from pimajudge.week1.tests.registry import Part, Score, TestCase, TestRegistry, registry

# Get the directory of this __init__.py file
_module_dir = Path(__file__).parent

# Find all .py files in the directory (excluding __init__.py)
_module_files: list[str] = [
    f.stem for f in _module_dir.glob("*.py") if f.name != "__init__.py"
]

# Dynamically import all modules
for _module_name in _module_files:
    importlib.import_module(f".{_module_name}", package=__package__)


__all__ = ["registry", "TestRegistry", "Part", "Score", "TestCase"]
