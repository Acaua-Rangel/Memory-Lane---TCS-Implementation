"""
Tests for loss functions.
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses import SelfDistillationLoss, ToolCallingLoss, build_loss


class TestSelfDistillationLoss:
    def test_invalid_vocab_chunk_size(self):
        with pytest.raises(ValueError):
            SelfDistillationLoss(vocab_chunk_size=0)

    def test_pure_kl_mode(self):
        """Phase 1: alpha=1.0, pure KL distillation."""
        loss_fn = SelfDistillationLoss(temperature=2.0, alpha=1.0)
        student = torch.randn(2, 32, 100, requires_grad=True)  # (B, M, V) - compressed
        teacher = torch.randn(2, 128, 100)  # (B, T, V) - full
        result = loss_fn(student, teacher)

        assert "loss" in result
        assert "kl_loss" in result
        assert result["loss"].shape == ()
        assert result["loss"].requires_grad

    def test_balanced_mode(self):
        """Phase 2: alpha=0.5, KL + CE."""
        loss_fn = SelfDistillationLoss(temperature=2.0, alpha=0.5)
        student = torch.randn(2, 32, 100, requires_grad=True)
        teacher = torch.randn(2, 128, 100)
        targets = torch.randint(0, 100, (2, 128))

        result = loss_fn(student, teacher, targets)
        assert result["loss"].requires_grad
        assert result["ce_loss"].item() >= 0

    def test_same_length_no_interpolation(self):
        """When student and teacher have same length."""
        loss_fn = SelfDistillationLoss(temperature=2.0, alpha=0.5)
        student = torch.randn(2, 64, 50, requires_grad=True)
        teacher = torch.randn(2, 64, 50)
        targets = torch.randint(0, 50, (2, 64))

        result = loss_fn(student, teacher, targets)
        assert result["loss"].requires_grad

    def test_align_teacher_logits_by_indices(self):
        """Teacher logits are aligned by index sampling when lengths differ."""
        loss_fn = SelfDistillationLoss()
        student = torch.zeros(1, 3, 2)
        teacher = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]]
        )

        aligned = loss_fn._align_teacher_logits(student, teacher)
        expected = teacher[:, [0, 2, 4], :]
        assert torch.equal(aligned, expected)

    def test_chunked_kl_matches_reference(self):
        """Chunked KL matches the reference full-vocabulary KL."""
        loss_fn = SelfDistillationLoss(temperature=1.7, alpha=1.0, vocab_chunk_size=5)
        student = torch.randn(2, 4, 17)
        teacher = torch.randn(2, 4, 17)

        kl_chunked = loss_fn._compute_kl_loss(student, teacher)

        temperature = loss_fn.temperature
        student_log = F.log_softmax(student / temperature, dim=-1)
        teacher_prob = F.softmax(teacher / temperature, dim=-1)
        kl_reference = F.kl_div(
            student_log.reshape(-1, student.size(-1)),
            teacher_prob.reshape(-1, teacher.size(-1)),
            reduction="batchmean",
        ) * (temperature * temperature)

        assert torch.allclose(kl_chunked, kl_reference, atol=1e-5, rtol=1e-4)

    def test_chunked_kl_matches_reference_larger_vocab(self):
        """Chunked log-normalizer path remains numerically close on larger vocab."""
        loss_fn = SelfDistillationLoss(temperature=2.0, alpha=1.0, vocab_chunk_size=7)
        student = torch.randn(1, 3, 53)
        teacher = torch.randn(1, 3, 53)

        kl_chunked = loss_fn._compute_kl_loss(student, teacher)

        temperature = loss_fn.temperature
        student_log = F.log_softmax(student / temperature, dim=-1)
        teacher_prob = F.softmax(teacher / temperature, dim=-1)
        kl_reference = F.kl_div(
            student_log.reshape(-1, student.size(-1)),
            teacher_prob.reshape(-1, teacher.size(-1)),
            reduction="batchmean",
        ) * (temperature * temperature)

        assert torch.allclose(kl_chunked, kl_reference, atol=1e-5, rtol=1e-4)


class TestToolCallingLoss:
    def test_basic(self):
        loss_fn = ToolCallingLoss()
        logits = torch.randn(2, 64, 100, requires_grad=True)
        targets = torch.randint(0, 100, (2, 64))

        result = loss_fn(logits, targets)
        assert "loss" in result
        assert result["loss"].requires_grad

    def test_aligns_targets_to_compressed_logits(self):
        loss_fn = ToolCallingLoss()
        logits = torch.randn(2, 4, 11, requires_grad=True)
        targets = torch.randint(0, 11, (2, 16))

        result = loss_fn(logits, targets)

        indices = torch.linspace(0, targets.size(1) - 1, logits.size(1)).long()
        expected_targets = targets[:, indices]
        expected_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            expected_targets.reshape(-1),
        )

        assert torch.allclose(result["loss"], expected_loss)

    def test_with_ignore_index(self):
        loss_fn = ToolCallingLoss(ignore_index=-100)
        logits = torch.randn(2, 64, 100, requires_grad=True)
        targets = torch.full((2, 64), -100, dtype=torch.long)
        targets[:, 10:20] = torch.randint(0, 100, (2, 10))

        result = loss_fn(logits, targets)
        assert result["loss"].requires_grad


class TestBuildLoss:
    def test_tcs_pretrain(self):
        loss_fn = build_loss("tcs-pretrain")
        assert isinstance(loss_fn, SelfDistillationLoss)
        assert loss_fn.alpha == 1.0

    def test_tcs_lora(self):
        loss_fn = build_loss("tcs-lora", alpha=0.5)
        assert isinstance(loss_fn, SelfDistillationLoss)
        assert loss_fn.alpha == 0.5

    def test_tool_calling(self):
        loss_fn = build_loss("tool-calling")
        assert isinstance(loss_fn, ToolCallingLoss)

    def test_unknown_phase(self):
        with pytest.raises(ValueError):
            build_loss("unknown-phase")


class TestSelfDistillationLossErrors:
    def test_zero_temperature_raises(self):
        loss_fn = SelfDistillationLoss(temperature=0.0, alpha=1.0)
        student = torch.randn(1, 4, 10)
        teacher = torch.randn(1, 4, 10)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            loss_fn._compute_kl_loss(student, teacher)

    def test_negative_temperature_raises(self):
        loss_fn = SelfDistillationLoss(temperature=-1.0, alpha=1.0)
        student = torch.randn(1, 4, 10)
        teacher = torch.randn(1, 4, 10)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            loss_fn._compute_kl_loss(student, teacher)

    def test_shape_mismatch_raises(self):
        loss_fn = SelfDistillationLoss(temperature=2.0, alpha=1.0)
        student = torch.randn(1, 4, 10)
        teacher = torch.randn(1, 4, 20)
        with pytest.raises(ValueError, match="same shape"):
            loss_fn._compute_kl_loss(student, teacher)

    def test_empty_vocab_raises(self):
        loss_fn = SelfDistillationLoss(temperature=2.0, alpha=1.0)
        student = torch.randn(1, 4, 0)
        teacher = torch.randn(1, 4, 0)
        with pytest.raises(ValueError):
            loss_fn._compute_kl_loss(student, teacher)
