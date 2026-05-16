"""Tests for checkpoint utilities."""

import pytest
import torch
from pathlib import Path

from src.utils.checkpoint import (
    CheckpointState,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    cleanup_phase_checkpoints,
    get_disk_usage_mb,
    _rotate_checkpoints,
)


@pytest.fixture
def sample_state():
    return CheckpointState(
        step=100,
        model_state={"weight": torch.randn(4, 4)},
        optimizer_state={"lr": 1e-3},
        scheduler_state={"last_epoch": 100},
        best_metric=0.5,
        config={"phase": "tcs-pretrain"},
    )


class TestSaveAndLoad:
    def test_save_and_load_sync(self, tmp_path, sample_state):
        path = save_checkpoint(sample_state, tmp_path, filename="step_000100.pt", async_save=False)
        assert path.exists()

        loaded = load_checkpoint(path)
        assert loaded.step == 100
        assert loaded.best_metric == 0.5
        assert loaded.config["phase"] == "tcs-pretrain"
        assert loaded.scheduler_state == {"last_epoch": 100}

    def test_save_async(self, tmp_path, sample_state):
        import time
        path = save_checkpoint(sample_state, tmp_path, filename="step_000100.pt", async_save=True)
        # Wait for async thread to finish
        time.sleep(1)
        assert path.exists()

    def test_save_auto_filename(self, tmp_path, sample_state):
        path = save_checkpoint(sample_state, tmp_path, async_save=False)
        assert "step_000100" in path.name

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_load_without_optional_fields(self, tmp_path):
        """Load checkpoint that lacks scheduler_state and best_metric."""
        payload = {
            "step": 50,
            "model_state": {},
            "optimizer_state": {},
        }
        path = tmp_path / "minimal.pt"
        torch.save(payload, path)
        loaded = load_checkpoint(path)
        assert loaded.step == 50
        assert loaded.scheduler_state is None
        assert loaded.best_metric == float("inf")
        assert loaded.config == {}


class TestRotateCheckpoints:
    def test_keeps_latest_n(self, tmp_path):
        for i in range(5):
            (tmp_path / f"step_{i:06d}.pt").write_text(f"data_{i}")

        _rotate_checkpoints(tmp_path, max_to_keep=3)

        remaining = sorted(tmp_path.glob("step_*.pt"))
        assert len(remaining) == 3
        assert remaining[0].name == "step_000002.pt"

    def test_no_rotation_when_under_limit(self, tmp_path):
        for i in range(2):
            (tmp_path / f"step_{i:06d}.pt").write_text(f"data_{i}")

        _rotate_checkpoints(tmp_path, max_to_keep=3)
        assert len(list(tmp_path.glob("step_*.pt"))) == 2

    def test_max_zero_keeps_all(self, tmp_path):
        for i in range(3):
            (tmp_path / f"step_{i:06d}.pt").write_text(f"data_{i}")

        _rotate_checkpoints(tmp_path, max_to_keep=0)
        assert len(list(tmp_path.glob("step_*.pt"))) == 3


class TestFindLatestCheckpoint:
    def test_finds_latest(self, tmp_path):
        for i in [100, 200, 300]:
            (tmp_path / f"step_{i:06d}.pt").write_text("data")

        latest = find_latest_checkpoint(tmp_path)
        assert latest.name == "step_000300.pt"

    def test_empty_dir(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None

    def test_nonexistent_dir(self, tmp_path):
        assert find_latest_checkpoint(tmp_path / "missing") is None


class TestCleanupPhaseCheckpoints:
    def test_cleanup_checkpoints_dir(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "step_000100.pt").write_text("data")
        (ckpt_dir / "step_000200.pt").write_text("data")

        cleanup_phase_checkpoints(tmp_path)
        assert not ckpt_dir.exists()

    def test_cleanup_best_when_final_exists(self, tmp_path):
        best_dir = tmp_path / "best"
        final_dir = tmp_path / "final"
        best_dir.mkdir()
        final_dir.mkdir()
        (best_dir / "weights.pt").write_text("data")
        (final_dir / "weights.pt").write_text("data")

        cleanup_phase_checkpoints(tmp_path)
        assert not best_dir.exists()
        assert final_dir.exists()

    def test_no_cleanup_when_no_final(self, tmp_path):
        best_dir = tmp_path / "best"
        best_dir.mkdir()
        (best_dir / "weights.pt").write_text("data")

        cleanup_phase_checkpoints(tmp_path)
        assert best_dir.exists()  # best kept because final doesn't exist

    def test_noop_when_empty(self, tmp_path):
        cleanup_phase_checkpoints(tmp_path)  # Should not raise


class TestDiskUsage:
    def test_disk_usage(self, tmp_path):
        (tmp_path / "file.bin").write_bytes(b"\x00" * 1024)
        usage = get_disk_usage_mb(tmp_path)
        assert usage > 0

    def test_nonexistent_path(self, tmp_path):
        assert get_disk_usage_mb(tmp_path / "missing") == 0.0
