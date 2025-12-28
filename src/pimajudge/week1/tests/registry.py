from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Literal

Part = Literal["part-1", "part-2"]
Score = Literal["F", "E", "D", "C", "B", "A"]


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
    group: str
    score: Score
    test_fn: Callable
    name: str
    description: str


class TestRegistry:
    """Registry to store and manage test cases."""

    def __init__(self):
        # Structure: {part: {group: [TestCase, ...]}}
        self.tests: dict[str, dict[str, list[TestCase]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Track group results for dependency checking
        self.group_results: dict[str, str] = {}  # group -> score

    def register(self, test_case: TestCase):
        """Register a test case."""
        self.tests[test_case.part][test_case.group].append(test_case)

    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        try:
            # Call test function (test functions import answers directly)
            test_case.test_fn()
            return TestResult(passed=True, message="Test passed")
        except AssertionError as e:
            return TestResult(passed=False, error=str(e), message="Assertion failed")
        except Exception as e:
            return TestResult(
                passed=False, error=str(e), message=f"Error: {type(e).__name__}"
            )

    def test(
        self,
        part: Part,
        group: str,
        score: Score,
    ):
        """
        Decorator to register a test.

        Args:
            part: Which part the test belongs to ('part-1' or 'part-2')
            group: Test group/question identifier
            score: Score level ('F', 'E', 'D', 'C', 'B', 'A')
            dependencies: List of group names this test depends on
        """

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            test_case = TestCase(
                part=part,
                group=group,
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
