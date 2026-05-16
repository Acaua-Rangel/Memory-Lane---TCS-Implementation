---
applyTo: '**'
---

# Gemma 4 Memory Companion — Copilot Instructions

## Project Context
This is a hackathon project for the Google DeepMind Gemma 4 Hackathon. We are building "Lane", an Alzheimer memory companion that runs on edge devices using Gemma 4 E2B with Token Compression Sub-network (TCS) + LoRA.

- **Team:** Memory Lane (Acauã Rangel Brazil + Juan Benjamin Suzart), Salvador-BA
- **Deadline:** May 18, 2026
- **Tracks:** Health & Sciences, Cactus ($10K), Unsloth ($10K), LiteRT ($10K)

## Coding Standards (MANDATORY)

### Language & Style
- All comments, docstrings, variable names, log messages, and error messages MUST be in **English**.
- Follow **SOLID principles** and **Clean Code** methodology.
- **DRY** — never duplicate logic. Extract shared behavior into utilities or base classes.
- Type hints on all function signatures.

### Error Handling — Early Return Pattern
- **Never use `else` after a validation check.** Validate first, return/raise immediately, then proceed with the happy path.
- Example:
  ```python
  def process(data: dict) -> Result:
      if not data:
          raise ValueError("Data cannot be empty")
      if "key" not in data:
          raise KeyError("Missing required key")
      # Happy path continues here — no else, no nesting
      return Result(data["key"])
  ```

### Testing
- After ANY code change, update the corresponding unit tests.
- Always run `poetry run pytest --cov=src --cov-report=term-missing` after changes.
- Target: **100% coverage**. Every branch, every edge case.
- Tests live in `tests/` mirroring `src/` structure.

### Environment & Dependencies
- Always use the project's `.venv` managed by **Poetry**.
- Install packages via: `poetry add <package>` (never `pip install` directly).
- Run commands via: `poetry run <command>`.
- Python executable: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux).

### Documentation Cascade
- **Any code change MUST trigger updates to affected markdown files.**
- If architecture changes → update `docs/architecture.md`, `docs/KAGGLE_WRITEUP.md`, `.instructions/`.
- If training changes → update `docs/training_plan.md`, `docs/KAGGLE_WRITEUP.md`.
- If tools change → update `docs/PROJECT.md`, `docs/KAGGLE_WRITEUP.md`.
- If agents/skills change → update this file and the relevant `.agent.md`/`SKILL.md`.

## Architecture Rules
- Gemma 4 base weights are ALWAYS frozen. Only TCS and LoRA adapters are trainable.
- TCS is a strided Conv1D that compresses N tokens into N/r tokens (r ∈ {2, 4, 8}).
- LoRA targets: q_proj, k_proj, v_proj, o_proj in attention layers.
- The model uses Gemma 4's native function calling format for tool use.
- Face recognition: MobileFaceNet (4MB float32 / ~1MB INT8, 128-d) + MediaPipe BlazeFace (<1MB).
- Face embeddings stored/searched via sqlite-vec with cosine similarity threshold 0.85.
- The LLM does NOT process faces — face pipeline is a separate lightweight module.
- Everything must run **fully offline** on a mobile phone.

## Kaggle Constraints (Training Environment)
- **VRAM:** 2× T4 = 32GB total (16GB each). Model must fit in 16GB per GPU.
- **System RAM:** 30GB.
- **Disk:** ~19.5GB available — checkpoint rotation is MANDATORY.
- **Checkpoint policy:** Keep only last 3 checkpoints (`max_checkpoints_to_keep=3`).
- **Phase cleanup:** After each phase, delete intermediate checkpoints. Keep only `final/`.
- **Session limit:** 12 hours.

## File Organization
- `src/config.py` — All dataclass configs (model, training, compression, tools).
- `src/model.py` — Gemma4WithTCS model wrapper.
- `src/losses.py` — Self-distillation, tool calling CE loss.
- `src/data.py` — Dataset loading, synthetic data generation.
- `src/train.py` — Single-GPU training entry point.
- `src/train_ddp.py` — Multi-GPU DDP training entry point.
- `src/train_unsloth.py` — Unsloth-accelerated training.
- `src/inference.py` — Inference with tool execution loop.
- `src/routing.py` — Cactus TaskRouter with 5 processing paths.
- `src/export_litert.py` — LiteRT/TFLite export pipeline.
- `src/tools/` — Tool schemas and SQLite tool implementations (9 tools, 8 tables).
- `src/utils/` — Checkpointing, distributed helpers, metrics.
- `docs/` — All project documentation (architecture, training plan, writeup, PROJECT.md).

## Training Phases
1. **tcs-pretrain**: Train TCS only, self-distillation loss (α=1.0), lr=1e-3, 2000 steps.
2. **tcs-lora**: TCS + LoRA jointly, KL+CE (α=0.5), lr=2e-4, 10000 steps, Unsloth.
3. **tool-calling**: Tool calling fine-tune, CE only, lr=3e-5, 3000 steps, Unsloth.
4. **export**: LiteRT export via ai-edge-torch.

## Key Constraints
- Must fit in 16GB VRAM per GPU (T4).
- Use gradient checkpointing for memory efficiency.
- No external teacher model — self-distillation only (uncompressed path = teacher).
- Mobile deployment: face models must be <5MB total, use ONNX Runtime.
- Prefer `safetensors` for weight serialization.

## Special Technology Tracks
This project targets three Special Technology prizes:
- **Cactus** ($10K): `src/routing.py` — TaskRouter routes between MobileFaceNet, SQLite, rule-based, and Gemma 4 at variable compression levels. This IS the "local-first wearable that routes between models."
- **Unsloth** ($10K): `src/train_unsloth.py` — Unsloth-accelerated LoRA training for Phases 2-3. 2× faster, 60% less VRAM.
- **LiteRT** ($10K): `src/export_litert.py` — Export TCS + merged LoRA to Google AI Edge LiteRT format. Includes deployment manifest.

## Testing
- All core modules must have unit tests.
- Tests must run without GPU (use small mock models).
- Use pytest fixtures for shared test setup.
