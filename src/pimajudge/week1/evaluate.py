from collections import defaultdict

from pimajudge.week1.tests import Part, Score, TestCase, TestRegistry


def evaluate(registry: TestRegistry, part: Part) -> dict[str, Score]:
    """Evaluate all groups in a part and return scores."""
    part_results = {}

    if part not in registry.tests:
        return part_results

    # Sort groups by dependencies (topological sort)
    groups = list(registry.tests[part].keys())
    evaluated = set()

    while len(evaluated) < len(groups):
        progress = False
        for group in groups:
            if group in evaluated:
                continue

            score = calculate_group_score(registry, part, group)
            part_results[group] = score
            registry.group_results[group] = score
            evaluated.add(group)
            progress = True

        if not progress:
            # Circular dependency or missing dependency
            for group in groups:
                if group not in evaluated:
                    part_results[group] = "F"
                    registry.group_results[group] = "F"
                    evaluated.add(group)

    return part_results


def get_details(registry: TestRegistry, part: Part, group: str):
    """Get detailed test results for a group."""
    if part not in registry.tests or group not in registry.tests[part]:
        return {"error": "Group not found"}

    tests = registry.tests[part][group]
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
    detailed["final_score"] = calculate_group_score(registry, part, group)

    return detailed


def calculate_group_score(registry: TestRegistry, part: Part, group: str) -> Score:
    """
    Calculate the group score based on test results.

    Score is the highest level where all tests of that level and below pass.
    """
    if group not in registry.tests[part]:
        return "F"

    tests = registry.tests[part][group]
    score_order: list[Score] = ["F", "E", "D", "C", "B", "A"]

    # Group tests by score level
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
