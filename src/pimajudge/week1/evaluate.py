from collections import defaultdict

from pimajudge.week1.tests import Part, Score, TestCase, TestRegistry
from pimajudge.week1.types import Exercise


def evaluate(registry: TestRegistry, part: Part) -> dict[str, Score]:
    """Evaluate all exercises in a part and return scores."""
    part_results = {}

    if part not in registry.tests:
        return part_results

    # Sort exercises by dependencies (topological sort)
    exercises = list(registry.tests[part].keys())
    evaluated = set()

    while len(evaluated) < len(exercises):
        progress = False
        for exercise in exercises:
            if exercise in evaluated:
                continue

            score = calculate_exercise_score(registry, part, exercise)
            part_results[exercise] = score
            registry.exercise_results[exercise] = score
            evaluated.add(exercise)
            progress = True

        if not progress:
            # Circular dependency or missing dependency
            for exercise in exercises:
                if exercise not in evaluated:
                    part_results[exercise] = "F"
                    registry.exercise_results[exercise] = "F"
                    evaluated.add(exercise)

    return part_results


def get_details(registry: TestRegistry, part: Part, exercise: Exercise):
    """Get detailed test results for a exercise."""
    if part not in registry.tests or exercise not in registry.tests[part]:
        return {"error": "exercise not found"}

    tests = registry.tests[part][exercise]
    score_order: list[Score] = ["F", "E", "D", "C", "B", "A"]

    detailed = {
        "exercise": exercise,
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
            result = registry.run_test(test)
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
    detailed["final_score"] = calculate_exercise_score(registry, part, exercise)

    return detailed


def calculate_exercise_score(
    registry: TestRegistry,
    part: Part,
    exercise: str
) -> Score:
    """
    Calculate the exercise score based on test results.

    Score is the highest level where all tests of that level and below pass.
    """
    if exercise not in registry.tests[part]:
        return "F"

    tests = registry.tests[part][exercise]
    score_order: list[Score] = ["F", "E", "D", "C", "B", "A"]

    # exercise tests by score level
    tests_by_score: dict[Score, list[TestCase]] = defaultdict(list)
    for test in tests:
        tests_by_score[test.score].append(test)

    # Calculate score from bottom up
    achieved_score: Score = "F"
    for score_level in score_order:
        if score_level not in tests_by_score:
            continue  # No tests at this level, continue

        level_tests = tests_by_score[score_level]
        results = [registry.run_test(test) for test in level_tests]

        # All tests at this level must pass
        if all(r.passed for r in results):
            achieved_score = score_level
        else:
            break  # Stop at first failure

    return achieved_score
