# Type Checking in pySDC

## Overview

Starting with version 5.6+, pySDC includes type checking with [mypy](https://mypy.readthedocs.io/) in the CI pipeline. This helps catch type-related bugs early and improves code quality.

## Current Status

**Phase 1: Gradual Adoption (Current)**

Type checking is currently **informational only** and does not block builds or PRs. The goal is to:
- Establish a baseline of type checking
- Allow developers to see type issues without breaking existing workflows
- Encourage gradual addition of type hints to the codebase

**Configuration**: See `[tool.mypy]` section in `pyproject.toml`

## Why Type Hints?

Type hints provide several benefits:

1. **Better IDE Support**: Auto-completion, refactoring, and inline error detection
2. **Catch Bugs Early**: Many bugs are type-related and can be caught before runtime
3. **Documentation**: Type hints serve as inline documentation of expected types
4. **Easier Maintenance**: New contributors can understand code faster
5. **Refactoring Safety**: Type checker verifies changes don't break existing code

## Running Type Checking Locally

### Check Specific Files

```bash
mypy pySDC/core/sweeper.py
```

### Check Entire Module

```bash
mypy pySDC/core
```

### Check Multiple Modules

```bash
mypy pySDC/core pySDC/implementations pySDC/helpers
```

## Adding Type Hints to Your Code

### Basic Example

**Before (no type hints):**
```python
def integrate(self):
    L = self.level
    P = L.prob
    me = []
    for m in range(1, self.coll.num_nodes + 1):
        me.append(P.dtype_u(P.init, val=0.0))
    return me
```

**After (with type hints):**
```python
from typing import List

def integrate(self) -> List:
    L = self.level
    P = L.prob
    me = []
    for m in range(1, self.coll.num_nodes + 1):
        me.append(P.dtype_u(P.init, val=0.0))
    return me
```

### Function Parameters

```python
def __init__(self, params: dict, level: 'Level') -> None:
    """
    Initialization routine for the base sweeper

    Args:
        params: parameter dictionary
        level: the level that uses this sweeper
    """
    # implementation
```

### Class Attributes

```python
from typing import Optional
from qmat.qdelta import QDeltaGenerator

class Sweeper:
    coll: CollBase
    genQI: Optional[QDeltaGenerator] = None
    genQE: Optional[QDeltaGenerator] = None
```

### Complex Types

```python
from typing import Union, Optional, List, Dict, Callable

# Union types
def process_value(val: Union[int, float]) -> float:
    return float(val)

# Optional (shorthand for Union[Type, None])
def find_item(name: str) -> Optional[dict]:
    return None  # or return found item

# Collections
def get_nodes(self) -> List[float]:
    return [0.0, 0.5, 1.0]

# Nested types
def get_params(self) -> Dict[str, Union[int, float, str]]:
    return {'num_nodes': 3, 'dt': 0.1, 'method': 'IE'}
```

## Type Checking Configuration

Current mypy configuration in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = false
warn_unused_configs = true
disallow_untyped_defs = false         # Permissive for now
disallow_incomplete_defs = false      # Permissive for now  
check_untyped_defs = false            # Permissive for now
disallow_untyped_calls = false        # Permissive for now
exclude = [
    'pySDC/playgrounds',
    'pySDC/tests',
    'pySDC/projects/deprecated',
]
ignore_missing_imports = true
show_error_codes = true
```

### Configuration Notes

- **Permissive Mode**: Currently allows untyped functions to ease gradual adoption
- **Excludes**: Playgrounds, tests, and deprecated projects don't require type checking
- **ignore_missing_imports**: External libraries without type stubs are ignored
- **show_error_codes**: Makes it easier to selectively ignore specific error types

## Roadmap

### Phase 1 ✅ (Current)
- Add mypy to CI in report-only mode
- Configure basic mypy settings
- Don't block builds on type errors

### Phase 2 🔄 (In Progress)
- Add type hints to `pySDC/core` modules
- Add type hints to commonly used implementations
- Create type hint guide (this document)

### Phase 3 📋 (Future)
- Enable stricter mypy settings:
  - `disallow_untyped_defs = true` for core modules
  - `check_untyped_defs = true`
- Type checking becomes blocking for new code
- Create type stubs for complex data types

## Common Type Checking Errors

### Error: Need type annotation

```python
# Error
data = {}  # error: Need type annotation for "data"

# Fix
data: Dict[str, Any] = {}
```

### Error: Incompatible types

```python
# Error
x: int = "5"  # error: Incompatible types in assignment

# Fix
x: int = 5
# or
x = int("5")
```

### Error: Untyped function

```python
# Warning
def compute():  # note: untyped function
    return 42

# Fix
def compute() -> int:
    return 42
```

## Best Practices

1. **Start with return types**: Easiest to add and most beneficial
2. **Use `Any` sparingly**: Try to be specific when possible
3. **Forward references**: Use string literals for circular dependencies
   ```python
   def get_level(self) -> 'Level':  # 'Level' not yet defined
       return self.__level
   ```
4. **Type aliases**: Define complex types once
   ```python
   NodeArray = List[float]
   ParamsDict = Dict[str, Union[int, float, str]]
   ```
5. **Check incrementally**: Type check files as you work on them

## Resources

- [mypy Documentation](https://mypy.readthedocs.io/)
- [Python Type Hints PEP 484](https://peps.python.org/pep-0484/)
- [Python typing Module](https://docs.python.org/3/library/typing.html)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

## Getting Help

If you encounter type checking issues:

1. Check the error code in the output (e.g., `[var-annotated]`)
2. Search mypy documentation for that error code
3. Use `# type: ignore[error-code]` to temporarily suppress errors
4. Ask in pull request discussions if unsure

---

*Last updated: February 2026*
