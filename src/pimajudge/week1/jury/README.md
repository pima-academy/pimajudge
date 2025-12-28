# Test Decorator System - Usage Guide

## Overview

The test decorator system provides a structured way to create and manage test cases with:
- **Part grouping** (`part-1`, `part-2`)
- **Score levels** (`F`, `E`, `D`, `C`, `B`, `A`)
- **Dependencies** between test groups
- **Automatic test discovery** and registration

## Architecture

```
jury/
├── test_decorator.py      # Core decorator and evaluation engine
├── jury.py                 # Public API (result(), detailed_result())
├── part_1/
│   ├── __init__.py        # Auto-imports all test files
│   ├── test_create_simple_tensor.py
│   └── test_slicing.py
└── part_2/
    └── __init__.py
```

## Writing Tests

### Basic Test

```python
import torch
from pimajudge.week1.collector.collector import answers
from pimajudge.week1.jury.test_decorator import judge

@judge.test(part="part-1", group="my_group", score="F")
def test_basic():
    """Test description"""
    func = answers["my-function-id"]
    result = func(...)
    assert result == expected
```

### Test with Dependencies

```python
@judge.test(
    part="part-1",
    group="advanced_feature",
    score="C",
    dependencies=["basic_feature"]  # Requires basic_feature to pass
)
def test_advanced():
    """Test advanced functionality"""
    func = answers["advanced-function"]
    # ... test code
```

## Score Levels

The system calculates group scores based on a **cumulative passing model**:

- **F**: Default (no tests pass or dependency failed)
- **E-A**: Achieved when all tests at that level AND all lower levels pass

### Example Scoring

Given tests:
- 2 tests at level F
- 2 tests at level D
- 2 tests at level C
- 1 test at level A

**Scenario 1:** All F pass, all D pass, 1/2 C pass
→ **Score: D** (stopped at C where not all passed)

**Scenario 2:** All F pass, 1/2 D pass
→ **Score: F** (stopped at D)

**Scenario 3:** All tests pass
→ **Score: A**

## Using the System

### In Jupyter Notebooks

```python
from pimajudge import week1 as judge

# Run all tests for part-1
judge.result("part-1")

# Get detailed results for a specific group
judge.detailed_result("part-1", "create_simple_tensor")
```

### Example Output

```
============================================================
Evaluating PART-1
============================================================

Group                          Score     
----------------------------------------
create_simple_tensor           A         
slicing                        C         
reshape                        B         

============================================================
Evaluation complete
============================================================
```

### Detailed Output

```
============================================================
Detailed Results: create_simple_tensor
Part: part-1
Final Score: C
============================================================

F Level Tests:
  Overall: ✓ PASSED
  ✓ test_basic_execution
     Test that function can be called without errors

D Level Tests:
  Overall: ✓ PASSED
  ✓ test_correct_shape
     Test that output has correct shape
  ✓ test_correct_dtype
     Test that output has correct dtype

C Level Tests:
  Overall: ✗ FAILED
  ✓ test_all_ones
     Test that all values are 1
  ✗ test_edge_cases
     Test edge cases
     Error: assertion failed
```

## File Organization

### Test File Template

```python
# src/pimajudge/week1/jury/part_1/test_<group_name>.py

import torch
from pimajudge.week1.collector.collector import answers
from pimajudge.week1.jury.test_decorator import judge


@judge.test(part="part-1", group="<group_name>", score="F")
def test_name_f():
    """Description of F-level test"""
    func = answers["<function-id>"]
    # Test code
    assert condition


@judge.test(part="part-1", group="<group_name>", score="D")
def test_name_d():
    """Description of D-level test"""
    func = answers["<function-id>"]
    # Test code
    assert condition


@judge.test(part="part-1", group="<group_name>", score="C")
def test_name_c():
    """Description of C-level test"""
    func = answers["<function-id>"]
    # Test code
    assert condition


# Continue for B, A levels...
```

## Best Practices

### 1. Test Naming
- Use descriptive function names: `test_basic_execution`, `test_correct_shape`
- Add docstrings to explain what's being tested

### 2. Score Level Guidelines
- **F**: Basic execution (function exists, no crashes)
- **E**: Basic correctness (correct output type)
- **D**: Core functionality (correct results for simple cases)
- **C**: Standard functionality (correct results for typical cases)
- **B**: Extended functionality (edge cases, various inputs)
- **A**: Advanced functionality (complex scenarios, optimization)

### 3. Dependencies
- Use dependencies when test group B requires functionality from group A
- Dependencies are checked before running tests
- Circular dependencies are prevented

### 4. Assertions
- Use clear assertion messages
- Test one concept per test function
- Group related tests at the same score level

## API Reference

### `@judge.test()`

Decorator to register a test case.

**Parameters:**
- `part` (Literal["part-1", "part-2"]): Test part identifier
- `group` (str): Test group/question identifier
- `score` (Literal["F", "E", "D", "C", "B", "A"]): Score level
- `dependencies` (list[str] | None): List of group names this test depends on

### `judge.evaluate(part: str) -> dict[str, str]`

Evaluate all groups in a part.

**Returns:** Dictionary mapping group names to scores

### `judge.get_details(part: str, group: str) -> dict`

Get detailed test results for a specific group.

**Returns:** Dictionary with detailed test results

### `result(part: str)`

Public API function to evaluate and display results.

### `detailed_result(part: str, group: str)`

Public API function to display detailed results for a group.

## Adding New Tests

1. Create a new test file in the appropriate `part_X` directory:
   ```bash
   touch src/pimajudge/week1/jury/part_1/test_new_feature.py
   ```

2. Write tests using the `@judge.test()` decorator

3. The file will be automatically imported on next run (no manual registration needed)

## Troubleshooting

### Tests not found
- Ensure test file is in `part_1/` or `part_2/` directory
- Check that `__init__.py` has auto-import code
- Verify imports are correct

### Dependency errors
- Check dependency group names match exactly
- Ensure dependencies are evaluated before dependent groups
- Avoid circular dependencies

### Scoring issues
- Verify all tests at lower levels pass
- Check that test score levels are correctly specified
- Review dependency status
