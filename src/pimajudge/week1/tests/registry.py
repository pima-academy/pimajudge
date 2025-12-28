import inspect
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from pimajudge.week1.answers import answers
from pimajudge.week1.protocols import protocol_to_exercise
from pimajudge.week1.types import Exercise, Part, Score


@dataclass
class TestResult:
    """Result of a single test execution."""

    passed: bool
    error: str | None = None
    message: str = ""


@dataclass
class TestCase:
    """Represents a single test case."""

    part: Part
    exercise: Exercise
    score: Score
    test_fn: Callable
    name: str
    description: str


class TestRegistry:
    """Registry to store and manage test cases."""

    def __init__(self):
        # Structure: {part: {exercise: [TestCase, ...]}}
        self.tests: dict[str, dict[str, list[TestCase]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Track exercise results for dependency checking
        self.exercise_results: dict[str, str] = {}  # exercise -> score

    def register(self, test_case: TestCase):
        """Register a test case."""
        self.tests[test_case.part][test_case.exercise].append(test_case)

    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        try:
            kwargs = self.solve_args(test_case.test_fn)
            # Call test function (test functions import answers directly)
            test_case.test_fn(**kwargs)
            return TestResult(passed=True, message="Test passed")
        except AssertionError as e:
            return TestResult(passed=False, error=str(e), message="Assertion failed")
        except Exception as e:
            return TestResult(
                passed=False, error=str(e), message=f"Error: {type(e).__name__}"
            )

    def solve_args(self, func: Callable) -> dict[str, Any]:
        """
        Resolve function dependencies from the answers registry.

        Inspects the function signature and resolves any parameters that are
        protocol types by looking up the corresponding function in the answers
        registry using the protocol_to_exercise mapping.

        Args:
            func: The test function to inspect

        Returns:
            Dictionary mapping parameter names to resolved dependencies

        Raises:
            ValueError: If a parameter annotation is not a registered protocol
        """
        sig = inspect.signature(func)
        resolved_args: dict[str, Any] = {}

        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                continue

            if param.annotation in protocol_to_exercise:
                exercise_id = protocol_to_exercise[param.annotation]
                resolved_args[name] = answers[exercise_id]

        return resolved_args

    def test(
        self,
        part: Part,
        exercise: Exercise,
        score: Score,
    ):
        """
        Decorator to register a test.

        Args:
            part: Which part the test belongs to ('part-1' or 'part-2')
            exercise: Test exercise identifier
            score: Score level ('F', 'E', 'D', 'C', 'B', 'A')
            dependencies: List of exercise names this test depends on
        """

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            test_case = TestCase(
                part=part,
                exercise=exercise,
                score=score,
                test_fn=wrapper,
                name=func.__name__,
                description=func.__doc__ or "",
            )

            self.register(test_case)
            return wrapper

        return decorator


# Global instance
registry = TestRegistry()
