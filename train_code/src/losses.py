"""
Loss functions for TCS + LoRA training.

- SelfDistillationLoss: uncompressed path teaches compressed path
- ToolCallingLoss: standard CE on structured tool calling output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _align_sequence_length(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    """Align a sequence tensor on dimension 1 via index sampling."""
    if target_length < 1:
        raise ValueError("target_length must be >= 1")

    current_length = sequence.size(1)
    if current_length == target_length:
        return sequence

    indices = torch.linspace(
        0,
        current_length - 1,
        target_length,
        device=sequence.device,
    ).long()
    return sequence.index_select(1, indices)


class SelfDistillationLoss(nn.Module):
    """
    Self-distillation: the model's own uncompressed path (no TCS, no LoRA)
    acts as teacher for the compressed path (TCS + LoRA).

    L = α · T² · KL(student_soft ‖ teacher_soft) + (1 - α) · CE(student, targets)

    When α = 1.0 (Phase 1 / tcs-pretrain): pure KL distillation.
    When α = 0.5 (Phase 2 / tcs-lora): balanced KL + CE.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        vocab_chunk_size: int = 4096,
    ):
        super().__init__()
        if vocab_chunk_size < 1:
            raise ValueError("vocab_chunk_size must be >= 1")

        self.temperature = temperature
        self.alpha = alpha
        self.vocab_chunk_size = vocab_chunk_size
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def _align_teacher_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Align teacher sequence length to student length by index sampling."""
        student_len = student_logits.size(1)
        return _align_sequence_length(teacher_logits, student_len)

    def _compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL(teacher || student) with chunked vocabulary reduction."""
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

        if student_logits.shape != teacher_logits.shape:
            raise ValueError("student_logits and teacher_logits must have the same shape")

        batch_size, seq_len, vocab_size = student_logits.shape
        kl_sum = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        chunk_size = min(self.vocab_chunk_size, vocab_size)

        student_log_z: torch.Tensor | None = None
        teacher_log_z: torch.Tensor | None = None

        # Compute log-normalizers incrementally to avoid full-vocab temporaries.
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            student_chunk = (student_logits[..., start:end] / self.temperature).float()
            teacher_chunk = (teacher_logits[..., start:end] / self.temperature).float()

            student_chunk_log_z = torch.logsumexp(student_chunk, dim=-1, keepdim=True)
            teacher_chunk_log_z = torch.logsumexp(teacher_chunk, dim=-1, keepdim=True)

            if student_log_z is None:
                student_log_z = student_chunk_log_z
            else:
                student_log_z = torch.logaddexp(student_log_z, student_chunk_log_z)

            if teacher_log_z is None:
                teacher_log_z = teacher_chunk_log_z
            else:
                teacher_log_z = torch.logaddexp(teacher_log_z, teacher_chunk_log_z)

        if student_log_z is None or teacher_log_z is None:
            raise ValueError("Could not compute KL log-normalizers")

        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            student_chunk = (student_logits[..., start:end] / self.temperature).float()
            teacher_chunk = (teacher_logits[..., start:end] / self.temperature).float()

            student_log_prob = student_chunk - student_log_z
            teacher_log_prob = teacher_chunk - teacher_log_z
            teacher_prob = torch.exp(teacher_log_prob)

            kl_sum = kl_sum + (teacher_prob * (teacher_log_prob - student_log_prob)).sum()

        token_count = batch_size * seq_len
        kl_loss = kl_sum / token_count
        return kl_loss * (self.temperature * self.temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            student_logits: (B, M, V) from compressed path. M may differ from T.
            teacher_logits: (B, T, V) from uncompressed path (detached).
            targets: (B, T) ground-truth token ids. Optional when α = 1.0.

        Returns:
            dict with 'loss', 'kl_loss', 'ce_loss'.
        """
        V = student_logits.size(-1)
        M_s = student_logits.size(1)

        teacher_aligned = self._align_teacher_logits(student_logits, teacher_logits)
        kl_loss = self._compute_kl_loss(student_logits, teacher_aligned)

        # Hard-label CE (only when alpha < 1 and targets provided)
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if self.alpha < 1.0 and targets is not None:
            targets_aligned = _align_sequence_length(targets, M_s)

            ce_loss = self.ce(
                student_logits.reshape(-1, V),
                targets_aligned.reshape(-1),
            )

        total = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss

        return {
            "loss": total,
            "kl_loss": kl_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }


class ToolCallingLoss(nn.Module):
    """
    Standard cross-entropy loss for tool calling fine-tuning.

    When the compressed path shortens the sequence, targets are aligned to the
    logits length by index sampling before CE. The model learns to produce
    structured <tool_call>...</tool_call> output on the compressed positions.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, T, V) model output logits.
            targets: (B, T) target token ids (-100 for masked positions).

        Returns:
            dict with 'loss', 'ce_loss'.
        """
        V = logits.size(-1)
        targets_aligned = _align_sequence_length(targets, logits.size(1))
        ce_loss = self.ce(logits.reshape(-1, V), targets_aligned.reshape(-1))
        return {"loss": ce_loss, "ce_loss": ce_loss.detach()}


def build_loss(phase: str, alpha: float = 0.5, temperature: float = 2.0) -> nn.Module:
    """Factory: create the appropriate loss function for a training phase."""
    if phase == "tcs-pretrain":
        return SelfDistillationLoss(temperature=temperature, alpha=1.0)
    elif phase == "tcs-lora":
        return SelfDistillationLoss(temperature=temperature, alpha=alpha)
    elif phase == "tool-calling":
        return ToolCallingLoss()
    else:
        raise ValueError(f"Unknown training phase: {phase}")
