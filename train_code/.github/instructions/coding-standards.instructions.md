---
description: "Use when writing or modifying Python code. Enforces SOLID principles, Clean Code, early-return pattern, English-only comments, Poetry environment, and 100% test coverage. Applies to all .py files."
applyTo: "**/*.py"
---

# Python Coding Standards

## SOLID Principles

### Single Responsibility
- Each class/function does ONE thing. If a function name contains "and", split it.
- Config classes hold config. Model classes hold model logic. Training functions train.

### Open/Closed
- Extend via composition or strategy pattern, not by modifying existing classes.
- Use factory functions (`build_loss()`, `build_optimizer()`) for extensibility.

### Liskov Substitution
- Subclasses must be drop-in replacements. Override behavior, not contracts.

### Interface Segregation
- Small, focused protocols/ABCs. Don't force classes to implement methods they don't use.

### Dependency Inversion
- High-level modules depend on abstractions, not concrete implementations.
- Pass config objects, not raw values. Pass interfaces, not implementations.

## Early Return Pattern (NO ELSE)

**NEVER** write `else` after a validation/guard clause. Validate → return/raise → happy path.

```python
# CORRECT
def get_person(face_id: str) -> Person:
    if not face_id:
        raise ValueError("face_id cannot be empty")
    if not face_id.startswith("face_"):
        raise ValueError(f"Invalid face_id format: {face_id}")
    
    person = self._db.query(face_id)
    if person is None:
        raise PersonNotFoundError(f"No person found for {face_id}")
    
    return person

# WRONG — uses else, creates unnecessary nesting
def get_person(face_id: str) -> Person:
    if face_id:
        if face_id.startswith("face_"):
            person = self._db.query(face_id)
            if person is not None:
                return person
            else:
                raise PersonNotFoundError(...)
        else:
            raise ValueError(...)
    else:
        raise ValueError(...)
```

## Language

- All comments in **English**.
- All docstrings in **English**.
- All variable/function/class names in **English**.
- All log messages in **English**.
- All error messages in **English**.
- User-facing strings (patient responses) may be in Portuguese — that's the product language.

## DRY — Don't Repeat Yourself

- If logic appears in 2+ places, extract it.
- Shared training logic → `src/utils/training_strategy.py`.
- Shared checkpoint logic → `src/utils/checkpoint.py`.
- Shared config → `src/config.py` dataclasses.

## Testing

- After ANY code change, update affected tests in `tests/`.
- Run: `poetry run pytest --cov=src --cov-report=term-missing`
- Target: **100% line coverage**. Every branch, every edge case.
- Use `pytest.raises` for error paths. Use `@pytest.fixture` for shared setup.
- Mock external dependencies (Gemma 4 model, filesystem, SQLite).

## Environment

- Always use `.venv` managed by **Poetry**.
- Install: `poetry add <package>` or `poetry add --group dev <package>`.
- Run: `poetry run <command>`.
- Never use `pip install` directly.
- Never use system Python.
