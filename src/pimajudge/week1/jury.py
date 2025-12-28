from pimajudge.week1.tests import registry


def result(part: str):
    """
    Evaluate and display results for a given part.

    Args:
        part: The part identifier ('part-1' or 'part-2')
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating {part.upper()}")
    print(f"{'=' * 60}\n")

    # Run evaluation
    results = registry.evaluate(part)  # type: ignore

    if not results:
        print(f"No tests found for {part}")
        return

    # Display results
    print(f"{'Group':<30} {'Score':<10}")
    print("-" * 40)

    for group, score in sorted(results.items()):
        print(f"{group:<30} {score:<10}")

    print(f"\n{'=' * 60}")
    print("Evaluation complete")
    print(f"{'=' * 60}\n")


def detailed_result(part: str, group: str):
    """
    Get detailed test results for a specific group.

    Args:
        part: The part identifier ('part-1' or 'part-2')
        group: The group/question identifier
    """
    details = registry.get_details(part, group)  # type: ignore

    if "error" in details:
        print(f"Error: {details['error']}")
        return

    print(f"\n{'=' * 60}")
    print(f"Detailed Results: {group}")
    print(f"Part: {part}")
    print(f"Final Score: {details['final_score']}")
    print(f"{'=' * 60}\n")

    for score_level, level_data in details["tests_by_score"].items():
        print(f"\n{score_level} Level Tests:")
        print(f"  Overall: {'✓ PASSED' if level_data['passed'] else '✗ FAILED'}")

        for test in level_data["tests"]:
            status = "✓" if test["passed"] else "✗"
            print(f"  {status} {test['name']}")
            if test["description"]:
                print(f"     {test['description']}")
            if not test["passed"] and test["error"]:
                print(f"     Error: {test['error']}")
        print()
