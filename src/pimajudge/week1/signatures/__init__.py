import importlib
from pathlib import Path

from pimajudge.week1.signatures.register import signatures

# Get the directory of this __init__.py file
_signatures_dir = Path(__file__).parent

# Find all subdirectories that are signature modules (part_1, part_2, etc.)
_signature_modules = [
    d.name
    for d in _signatures_dir.iterdir()
    if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()
]

# Dynamically import all signature module directories to register signatures
for _module_name in _signature_modules:
    importlib.import_module(f".{_module_name}", package=__package__)

__all__ = ["signatures"]
