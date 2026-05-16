"""
Checkpoint utilities for saving/loading training state.

Disk-aware: keeps only the last N checkpoints (Kaggle has ~19.5GB disk).
At phase end, deletes all intermediate checkpoints and saves only the final
LoRA + TCS weights.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Everything needed to resume training."""
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    best_metric: float
    config: dict[str, Any]


def save_checkpoint(
    state: CheckpointState,
    save_dir: str | Path,
    filename: str | None = None,
    async_save: bool = True,
    max_to_keep: int = 3,
) -> Path:
    """
    Save a training checkpoint and rotate old ones.
    Only saves trainable weights (TCS + LoRA), not the full base model.

    Args:
        max_to_keep: Keep only this many checkpoints. Older ones are deleted.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"step_{state.step:06d}.pt"

    path = save_dir / filename
    payload = {
        "step": state.step,
        "model_state": state.model_state,
        "optimizer_state": state.optimizer_state,
        "scheduler_state": state.scheduler_state,
        "best_metric": state.best_metric,
        "config": state.config,
    }

    if async_save:
        thread = threading.Thread(
            target=_save_worker, args=(payload, path, save_dir, max_to_keep),
            daemon=True,
        )
        thread.start()
        logger.info("Async checkpoint save started → %s", path)
    else:
        _save_worker(payload, path, save_dir, max_to_keep)
        logger.info("Checkpoint saved → %s", path)

    return path


def _save_worker(
    payload: dict, path: Path, save_dir: Path, max_to_keep: int
) -> None:
    """Worker function for saving checkpoint + rotating old ones."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # Atomic on POSIX; on Windows, overwrites target if exists

    # Rotate: keep only last N checkpoints
    _rotate_checkpoints(save_dir, max_to_keep)


def _rotate_checkpoints(checkpoint_dir: Path, max_to_keep: int) -> None:
    """Delete old checkpoints, keeping only the most recent `max_to_keep`."""
    if max_to_keep <= 0:
        return

    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if len(checkpoints) <= max_to_keep:
        return

    to_delete = checkpoints[: len(checkpoints) - max_to_keep]
    for ckpt in to_delete:
        ckpt.unlink(missing_ok=True)
        logger.info("Rotated old checkpoint: %s", ckpt.name)


def cleanup_phase_checkpoints(phase_dir: str | Path) -> None:
    """
    Delete all intermediate checkpoints after a phase finishes.
    Keeps only the 'final/' and 'best/' directories.
    Called at the end of each training phase to free disk for the next phase.
    """
    phase_dir = Path(phase_dir)
    ckpt_dir = phase_dir / "checkpoints"

    if ckpt_dir.exists():
        size_mb = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        shutil.rmtree(ckpt_dir)
        logger.info("Cleaned up checkpoints dir: %s (freed ~%.1f MB)", ckpt_dir, size_mb)

    # Also clean up 'best/' if 'final/' exists (final supersedes best)
    best_dir = phase_dir / "best"
    final_dir = phase_dir / "final"
    if final_dir.exists() and best_dir.exists():
        shutil.rmtree(best_dir)
        logger.info("Cleaned up best/ dir (final/ exists)")


def load_checkpoint(path: str | Path) -> CheckpointState:
    """Load a training checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("Loaded checkpoint from %s (step %d)", path, data["step"])

    return CheckpointState(
        step=data["step"],
        model_state=data["model_state"],
        optimizer_state=data["optimizer_state"],
        scheduler_state=data.get("scheduler_state"),
        best_metric=data.get("best_metric", float("inf")),
        config=data.get("config", {}),
    )


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory by step number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def get_disk_usage_mb(path: str | Path) -> float:
    """Get total disk usage of a directory in MB."""
    path = Path(path)
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
    return checkpoints[-1] if checkpoints else None
