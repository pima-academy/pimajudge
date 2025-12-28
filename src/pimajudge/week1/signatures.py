from collections import defaultdict
from enum import Enum
from typing import Protocol, runtime_checkable

import torch


# --- 1. Define Protocols for each exercise ---
# We use @runtime_checkable so you can use isinstance(func, Protocol) if needed.
@runtime_checkable
class CreateSimpleTensor(Protocol):
    def __call__(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor: ...


@runtime_checkable
class Slicing(Protocol):
    def __call__(self, t: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class Reshape(Protocol):
    def __call__(self, t: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class SimpleArithmetic(Protocol):
    def __call__(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor: ...


@runtime_checkable
class SimpleAutograd(Protocol):
    def __call__(self, x_val: float) -> float: ...


@runtime_checkable
class ComplexAutograd(Protocol):
    def __call__(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...


# --- 2. Update Enum to store the Protocol classes ---
class Signatures(Enum):
    """Enum containing (Protocol, exercise_id) tuples."""

    CREATE_SIMPLE_TENSOR = (CreateSimpleTensor, "create-simple-tensor")
    SLICING = (Slicing, "slicing")
    RESHAPE = (Reshape, "reshape")
    SIMPLE_TORCH_ARITHMETIC_OPERATION = (
        SimpleArithmetic, "simple-torch-arithmetic-operation"
    )
    SIMPLE_AUTOGRAPH = (SimpleAutograd, "simple-autograph")
    COMPLEX_AUTOGRAPH = (ComplexAutograd, "complex-autograph")


# --- 3. Populate the dictionary ---

# Note: We use Type[Protocol] for the annotation
signatures: defaultdict[str, type | None] = defaultdict(lambda: None)

for member in Signatures:
    proto_cls, exercise_id = member.value
    signatures[exercise_id] = proto_cls

if __name__ == "__main__":
    for exercise_id, proto in signatures.items():
        # Printing the Protocol class name
        print(f"{exercise_id}: {proto.__name__ if proto else None}")
