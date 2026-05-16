---
name: doc-sync
description: "Synchronize documentation after code changes. Use when modifying src/ files to ensure docs/architecture.md, docs/training_plan.md, docs/PROJECT.md, docs/KAGGLE_WRITEUP.md, and .instructions/ files stay consistent with the codebase. Triggers on any architecture, training, tool, or routing change."
---

# Documentation Sync Skill

## When to Use

Invoke this skill after ANY code change that affects:
- Model architecture (`src/model.py`, `src/losses.py`)
- Training pipeline (`src/train.py`, `src/train_ddp.py`, `src/train_unsloth.py`)
- Configuration (`src/config.py`)
- Tool definitions (`src/tools/`)
- Task routing (`src/routing.py`)
- Export pipeline (`src/export_litert.py`)
- Data generation (`src/data.py`)
- Inference flow (`src/inference.py`)

## Procedure

### Step 1 — Identify What Changed

Read the modified source files and extract:
- New/removed/renamed classes, functions, or parameters
- Changed hyperparameters (lr, steps, ratios, thresholds)
- New/removed tools or database tables
- Changed model architecture (layers, dimensions, operations)
- Changed routing logic or processing paths

### Step 2 — Map to Documentation Files

| Source Change | Docs to Update |
|---|---|
| `src/model.py` (architecture) | `docs/architecture.md`, `docs/KAGGLE_WRITEUP.md` §TCS, `.instructions/architecture.instructions.md` |
| `src/losses.py` (loss functions) | `docs/architecture.md`, `docs/KAGGLE_WRITEUP.md` §Self-Distillation |
| `src/config.py` (hyperparams) | `docs/training_plan.md`, `docs/KAGGLE_WRITEUP.md` §Training, `.instructions/training.instructions.md` |
| `src/train*.py` (training logic) | `docs/training_plan.md`, `docs/KAGGLE_WRITEUP.md` §Training, `.instructions/training.instructions.md` |
| `src/tools/` (tool definitions) | `docs/PROJECT.md`, `docs/KAGGLE_WRITEUP.md` §Tools, `.instructions/tools-and-data.instructions.md` |
| `src/routing.py` (task router) | `docs/PROJECT.md`, `docs/KAGGLE_WRITEUP.md` §Routing |
| `src/data.py` (data generation) | `docs/KAGGLE_WRITEUP.md` §Synthetic Data, `.instructions/tools-and-data.instructions.md` |
| `src/export_litert.py` (export) | `docs/PROJECT.md`, `docs/KAGGLE_WRITEUP.md` §Technical Summary |
| `src/inference.py` (inference) | `docs/PROJECT.md` |

### Step 3 — Update Each Affected Document

For each document:
1. **Read** the current content of the affected section.
2. **Compare** with the new code to find discrepancies.
3. **Edit** only the sections that need updating — do not rewrite entire files.
4. **Verify** that numbers, names, and descriptions match the code exactly.

### Step 4 — Update Instruction Files

If the code change affects patterns described in `.instructions/`:
1. Read the relevant `.instructions.md` file.
2. Update code examples, parameter values, and architectural descriptions.
3. Ensure the `applyTo` pattern still covers the right files.

### Step 5 — Update This Skill If Needed

If the file organization changes (new source files, renamed directories):
1. Update the mapping table in Step 2.
2. Update the trigger list in "When to Use".
3. Update `.github/copilot-instructions.md` § File Organization.

### Step 6 — Run Tests

After all documentation updates:
```bash
poetry run pytest --cov=src --cov-report=term-missing
```

Verify no tests broke from the code change itself.

## Validation Checklist

- [ ] All hyperparameters in docs match `src/config.py`
- [ ] Tool count in docs matches `src/tools/sqlite_tools.py`
- [ ] Architecture diagram matches `src/model.py`
- [ ] Training phases in docs match phase presets in `src/config.py`
- [ ] File organization in `copilot-instructions.md` matches actual `src/` structure
- [ ] References in `KAGGLE_WRITEUP.md` are still accurate
- [ ] All `.instructions.md` code examples compile
