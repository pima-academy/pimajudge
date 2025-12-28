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
    dependencies: list[str]
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

    def check_dependencies(self, test_case: TestCase) -> tuple[bool, str]:
        """Check if all dependencies are satisfied."""
        for dep in test_case.dependencies:
            if dep not in self.group_results:
                return False, f"Dependency '{dep}' has not been evaluated yet"
            if self.group_results[dep] == "F":
                return False, f"Dependency '{dep}' failed (score: F)"
        return True, ""

    def calculate_group_score(self, part: Part, group: str) -> Score:
        """
        Calculate the group score based on test results.

        Score is the highest level where all tests of that level and below pass.
        """
        if group not in self.tests[part]:
            return "F"

        tests = self.tests[part][group]
        score_order: list[Score] = ["F", "E", "D", "C", "B", "A"]

        # Group tests by score level
        tests_by_score: dict[Score, list[TestCase]] = defaultdict(list)
        for test in tests:
            tests_by_score[test.score].append(test)

        # Check dependencies first
        for test in tests:
            deps_ok, msg = self.check_dependencies(test)
            if not deps_ok:
                return "F"  # Dependencies not met

        # Calculate score from bottom up
        achieved_score: Score = "F"
        for score_level in score_order:
            if score_level not in tests_by_score:
                continue  # No tests at this level, continue

            level_tests = tests_by_score[score_level]
            results = [self.run_test(test) for test in level_tests]

            # All tests at this level must pass
            if all(r.passed for r in results):
                achieved_score = score_level
            else:
                break  # Stop at first failure

        return achieved_score

    def evaluate_part(self, part: Part) -> dict[str, Score]:
        """Evaluate all groups in a part and return scores."""
        part_results = {}

        if part not in self.tests:
            return part_results

        # Sort groups by dependencies (topological sort)
        groups = list(self.tests[part].keys())
        evaluated = set()

        while len(evaluated) < len(groups):
            progress = False
            for group in groups:
                if group in evaluated:
                    continue

                # Check if all dependencies are evaluated
                all_tests = self.tests[part][group]
                all_deps = set()
                for test in all_tests:
                    all_deps.update(test.dependencies)

                if all_deps.issubset(evaluated):
                    score = self.calculate_group_score(part, group)
                    part_results[group] = score
                    self.group_results[group] = score
                    evaluated.add(group)
                    progress = True

            if not progress:
                # Circular dependency or missing dependency
                for group in groups:
                    if group not in evaluated:
                        part_results[group] = "F"
                        self.group_results[group] = "F"
                        evaluated.add(group)

        return part_results

    def get_detailed_results(self, part: Part, group: str) -> dict:
        """Get detailed test results for a group."""
        if part not in self.tests or group not in self.tests[part]:
            return {"error": "Group not found"}

        tests = self.tests[part][group]
        score_order: list[Score] = ["F", "E", "D", "C", "B", "A"]

        detailed = {
            "group": group,
            "part": part,
            "final_score": "F",
            "tests_by_score": {},
        }

        for score_level in score_order:
            level_tests = [t for t in tests if t.score == score_level]
            if not level_tests:
                continue

            level_results = []
            for test in level_tests:
                result = self.run_test(test)
                level_results.append(
                    {
                        "name": test.name,
                        "description": test.description,
                        "passed": result.passed,
                        "error": result.error,
                        "message": result.message,
                    }
                )

            detailed["tests_by_score"][score_level] = {
                "passed": all(r["passed"] for r in level_results),
                "tests": level_results,
            }

        # Calculate final score
        detailed["final_score"] = self.calculate_group_score(part, group)

        return detailed


class TestDecorator:
    """Decorator class for registering tests."""

    def __init__(self):
        self.registry = TestRegistry()

    def test(
        self,
        part: Part,
        group: str,
        score: Score,
        dependencies: list[str] | None = None,
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
                dependencies=dependencies or [],
                name=func.__name__,
                description=func.__doc__ or "",
            )

            self.registry.register(test_case)
            return wrapper

        return decorator

    def evaluate(self, part: Part) -> dict[str, Score]:
        """Evaluate all tests for a part."""
        return self.registry.evaluate_part(part)

    def get_details(self, part: Part, group: str) -> dict:
        """Get detailed results for a group."""
        return self.registry.get_detailed_results(part, group)


# Global instance
registry = TestDecorator()
