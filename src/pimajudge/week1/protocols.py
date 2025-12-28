from typing import Protocol, runtime_checkable

import torch

from pimajudge.week1.types import Exercise


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
        self, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor: ...


@runtime_checkable
class SimpleAutograd(Protocol):
    def __call__(self, x_val: float) -> float: ...


@runtime_checkable
class ComplexAutograd(Protocol):
    def __call__(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...


# --- 2. Populate the dictionary directly ---

# Note: We use Type[Protocol] for the annotation
protocols: dict[Exercise, type] = {}

protocols["create-simple-tensor"] = CreateSimpleTensor
protocols["slicing"] = Slicing
protocols["reshape"] = Reshape
protocols["simple-torch-arithmetic-operation"] = SimpleArithmetic
protocols["simple-autograph"] = SimpleAutograd
protocols["complex-autograph"] = ComplexAutograd

# --- 3. Inverse mapping: Protocol class -> exercise ID ---
protocol_to_exercise: dict[type, Exercise] = {
    proto: exercise_id for exercise_id, proto in protocols.items()
}

if __name__ == "__main__":
    for exercise_id, proto in protocols.items():
        # Printing the Protocol class name
        print(f"{exercise_id}: {proto.__name__ if proto else None}")
