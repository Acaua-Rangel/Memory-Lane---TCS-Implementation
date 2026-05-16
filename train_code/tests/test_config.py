"""
Tests for configuration.
"""

import pytest

from src.config import (
    CompressionConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
    ToolCallingConfig,
    DatabaseConfig,
    PHASE_PRESETS,
    apply_phase_preset,
)


class TestCompressionConfig:
    def test_defaults(self):
        cfg = CompressionConfig()
        assert cfg.ratio == 4
        assert cfg.kernel_size == 7
        assert cfg.enabled is True

    def test_multi_scale(self):
        cfg = CompressionConfig(multi_scale=True)
        assert cfg.ratios == [2, 4, 8]


class TestLoraConfig:
    def test_defaults(self):
        cfg = LoraConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert "q_proj" in cfg.target_modules


class TestTrainingConfig:
    def test_effective_batch_size(self):
        cfg = TrainingConfig(batch_size=4, gradient_accumulation_steps=8)
        assert cfg.effective_batch_size == 32

    def test_phase_output_dir(self):
        cfg = TrainingConfig(phase="tcs-lora", output_dir="out")
        assert cfg.phase_output_dir.parts[-1] == "tcs_lora"
        assert cfg.phase_output_dir.parts[-2] == "out"

    def test_gemini_teacher_defaults(self):
        cfg = TrainingConfig()
        assert cfg.use_gemini_teacher is False
        assert cfg.gemini_teacher_model == "gemma-4-31b-it"
        assert cfg.gemini_api_key is None
        assert cfg.gemini_rpm_limit == 14
        assert cfg.teacher_cache_dir == "data/teacher_cache"
        assert cfg.teacher_loss_weight == 0.3

    def test_ddp_oom_retry_defaults(self):
        cfg = TrainingConfig()
        assert cfg.ddp_auto_batch_retry_on_oom is True
        assert cfg.ddp_oom_retry_max_attempts == 3
        assert cfg.ddp_oom_retry_max_step == 1
        assert cfg.ddp_oom_retry_min_batch_size == 1


class TestPhasePresets:
    def test_tcs_pretrain_defaults(self):
        cfg = TrainingConfig(phase="tcs-pretrain")
        cfg = apply_phase_preset(cfg)
        assert cfg.distill_alpha == 1.0
        assert cfg.max_steps == 2000

    def test_tcs_lora_defaults(self):
        cfg = TrainingConfig(phase="tcs-lora")
        cfg = apply_phase_preset(cfg)
        assert cfg.lr == 2e-4
        assert cfg.distill_alpha == 0.5

    def test_tool_calling_defaults(self):
        cfg = TrainingConfig(phase="tool-calling")
        cfg = apply_phase_preset(cfg)
        assert cfg.lr == 3e-5
        assert cfg.distill_alpha == 0.0

    def test_user_override_preserved(self):
        """User-set values should not be overwritten by presets."""
        cfg = TrainingConfig(phase="tcs-pretrain", max_steps=500)
        # max_steps=500 differs from default 2000
        # Since 500 != dataclass default (2000), the preset shouldn't override it
        # But our implementation checks against dataclass default, so 500 != 2000 → preset won't apply
        cfg = apply_phase_preset(cfg)
        # The preset value for max_steps is 2000, which equals the dataclass default
        # So apply_phase_preset would try to set it. But user already set it to 500.
        # This is a known limitation — the preset applies when the value matches the default.
        # In this case, 500 ≠ 2000 (dataclass default), so preset won't override.
        assert cfg.max_steps == 500


class TestToolCallingConfig:
    def test_system_prompt_contains_tools(self):
        cfg = ToolCallingConfig()
        prompt = cfg.system_prompt
        assert "read_person" in prompt
        assert "write_encounter" in prompt
        assert "get_medication" in prompt
        assert "tool_call" in prompt

    def test_tool_definitions_count(self):
        cfg = ToolCallingConfig()
        assert len(cfg.tools) == 10
