"""
Tests for the Token Compression Sub-network and model wrapper.

All tests use small/mock models to run without GPU.
"""

import math
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.config import CompressionConfig, ModelConfig, LoraConfig
from src.model import (
    CompressionLayer,
    DecompressionLayer,
    MultiScaleCompressionLayer,
    RMSNorm,
    Gemma4WithTCS,
    _load_base_model,
    _apply_lora,
)


# ---------------------------------------------------------------------------
# _load_base_model and _apply_lora
# ---------------------------------------------------------------------------

class TestLoadBaseModel:
    def test_load_base_model_no_quantization(self):
        cfg = ModelConfig(
            base_model_name="test-model",
            torch_dtype="float16",
            load_in_4bit=False,
            load_in_8bit=False,
        )
        mock_model = MagicMock()
        mock_model.config.hidden_size = 64
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"

        with patch("src.model.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load:
            with patch("src.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                model, tokenizer = _load_base_model(cfg)

        assert model is mock_model
        assert tokenizer.pad_token == "</s>"
        # Without quantization, model must load to CPU for DDP compatibility
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["device_map"] == "cpu"
        assert call_kwargs["dtype"] == torch.float16

    def test_load_base_model_4bit(self):
        cfg = ModelConfig(
            base_model_name="test-model",
            load_in_4bit=True,
        )
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"  # Already set

        with patch("src.model.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load:
            with patch("src.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                model, tokenizer = _load_base_model(cfg)

        # Verify quantization config was passed
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["quantization_config"] is not None
        # Without LOCAL_RANK, quantized model uses device_map="auto"
        assert call_kwargs["device_map"] == "auto"

    def test_load_base_model_4bit_ddp(self):
        """With LOCAL_RANK set, quantized model loads on the specific GPU."""
        cfg = ModelConfig(
            base_model_name="test-model",
            load_in_4bit=True,
        )
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch.dict("os.environ", {"LOCAL_RANK": "1"}):
            with patch("src.model.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load:
                with patch("src.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                    _load_base_model(cfg)

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["device_map"] == {"": 1}

    def test_load_base_model_8bit(self):
        cfg = ModelConfig(
            base_model_name="test-model",
            load_in_8bit=True,
        )
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        with patch("src.model.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load:
            with patch("src.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                _load_base_model(cfg)

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["quantization_config"] is not None

    def test_load_base_model_bfloat16(self):
        cfg = ModelConfig(
            base_model_name="test-model",
            torch_dtype="bfloat16",
        )
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"

        with patch("src.model.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load:
            with patch("src.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                _load_base_model(cfg)

        call_kwargs = mock_load.call_args[1]
        import torch as t
        assert call_kwargs["dtype"] == t.bfloat16


class TestApplyLora:
    def test_apply_lora_wraps_model(self):
        mock_model = MagicMock()
        mock_peft_model = MagicMock()

        lora_cfg = LoraConfig(rank=8, alpha=16, dropout=0.05)

        with patch("src.model.get_peft_model", return_value=mock_peft_model) as mock_get_peft:
            result = _apply_lora(mock_model, lora_cfg)

        assert result is mock_peft_model
        mock_get_peft.assert_called_once()
        # Verify the PeftLoraConfig was created with correct params
        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.r == 8
        assert peft_cfg.lora_alpha == 16


# ---------------------------------------------------------------------------
# CompressionLayer
# ---------------------------------------------------------------------------

class TestCompressionLayer:
    def test_output_shape_ratio_4(self):
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        layer = CompressionLayer(d_model=64, comp_cfg=cfg)
        x = torch.randn(2, 128, 64)  # (B, N, D)
        out = layer(x)
        # M = ceil(128/4) = 32
        assert out.shape[0] == 2
        assert out.shape[2] == 64
        assert out.shape[1] == 32 or abs(out.shape[1] - 32) <= 1  # Allow ±1 due to padding

    def test_output_shape_ratio_2(self):
        cfg = CompressionConfig(ratio=2, kernel_size=7)
        layer = CompressionLayer(d_model=64, comp_cfg=cfg)
        x = torch.randn(2, 128, 64)
        out = layer(x)
        expected = 128 // 2
        assert abs(out.shape[1] - expected) <= 1

    def test_output_shape_ratio_8(self):
        cfg = CompressionConfig(ratio=8, kernel_size=8)
        layer = CompressionLayer(d_model=64, comp_cfg=cfg)
        x = torch.randn(2, 128, 64)
        out = layer(x)
        expected = 128 // 8
        assert abs(out.shape[1] - expected) <= 1

    def test_gradient_flow(self):
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        layer = CompressionLayer(d_model=64, comp_cfg=cfg)
        x = torch.randn(2, 64, 64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_batch_sizes(self):
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        layer = CompressionLayer(d_model=64, comp_cfg=cfg)
        for B in [1, 4, 8]:
            x = torch.randn(B, 64, 64)
            out = layer(x)
            assert out.shape[0] == B


# ---------------------------------------------------------------------------
# DecompressionLayer
# ---------------------------------------------------------------------------

class TestDecompressionLayer:
    def test_roundtrip_shape(self):
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        comp = CompressionLayer(d_model=64, comp_cfg=cfg)
        decomp = DecompressionLayer(d_model=64, comp_cfg=cfg)
        x = torch.randn(2, 128, 64)
        compressed = comp(x)
        restored = decomp(compressed, target_len=128)
        assert restored.shape == (2, 128, 64)

    def test_target_len_padding(self):
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        decomp = DecompressionLayer(d_model=64, comp_cfg=cfg)
        x = torch.randn(2, 16, 64)  # Compressed
        restored = decomp(x, target_len=100)
        assert restored.shape == (2, 100, 64)


# ---------------------------------------------------------------------------
# MultiScaleCompressionLayer
# ---------------------------------------------------------------------------

class TestMultiScaleCompression:
    def test_all_ratios(self):
        cfg = CompressionConfig(ratios=[2, 4, 8], kernel_size=8)
        ms = MultiScaleCompressionLayer(d_model=64, max_seq_len=128, comp_cfg=cfg)
        x = torch.randn(2, 128, 64)

        for ratio in [2, 4, 8]:
            out = ms.compress(x, ratio)
            expected = 128 // ratio
            assert abs(out.shape[1] - expected) <= 1
            assert out.shape == (2, out.shape[1], 64)

    def test_pos_embeddings(self):
        cfg = CompressionConfig(ratios=[4], kernel_size=7)
        ms = MultiScaleCompressionLayer(d_model=64, max_seq_len=128, comp_cfg=cfg)
        x = torch.randn(2, 128, 64)
        compressed = ms.compress(x, 4)
        with_pos = ms.add_pos(compressed, 4)
        assert with_pos.shape == compressed.shape

    def test_decompress(self):
        cfg = CompressionConfig(ratios=[4], kernel_size=7)
        ms = MultiScaleCompressionLayer(d_model=64, max_seq_len=128, comp_cfg=cfg)
        x = torch.randn(2, 128, 64)
        compressed = ms.compress(x, 4)
        restored = ms.decompress(compressed, target_len=128, ratio=4)
        assert restored.shape == (2, 128, 64)


# ---------------------------------------------------------------------------
# DecompressionLayer — edge cases
# ---------------------------------------------------------------------------

class TestDecompressionEdgeCases:
    def test_truncation_when_output_longer(self):
        """Test that output is truncated when decompressed is longer than target."""
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        decomp = DecompressionLayer(d_model=64, comp_cfg=cfg)
        x = torch.randn(2, 32, 64)
        # target_len much smaller than what decompression produces
        restored = decomp(x, target_len=10)
        assert restored.shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 100  # Large values
        out = norm(x)
        # RMS norm should bring values to reasonable range
        assert out.abs().mean() < x.abs().mean()


# ---------------------------------------------------------------------------
# Gemma4WithTCS (mocked base model)
# ---------------------------------------------------------------------------

def _make_mock_base_model(d_model=64, vocab_size=100):
    """Create a lightweight mock of a Gemma model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = d_model

    embedding = nn.Embedding(vocab_size, d_model)
    model.get_input_embeddings = MagicMock(return_value=embedding)

    # Prevent MagicMock from auto-creating model.model.language_model
    # which would trigger the multimodal extraction path
    model.model = MagicMock(spec=[])

    # Simulate forward returns
    def mock_forward(**kwargs):
        embeds = kwargs.get("inputs_embeds")
        input_ids = kwargs.get("input_ids")
        if embeds is not None:
            B, T = embeds.shape[0], embeds.shape[1]
        else:
            B, T = input_ids.shape
        result = MagicMock()
        result.logits = torch.randn(B, T, vocab_size)
        result.loss = None
        return result

    model.side_effect = mock_forward
    model.__call__ = mock_forward

    # parameters() for device detection
    model.parameters = MagicMock(side_effect=lambda: iter(embedding.parameters()))

    return model, embedding


def _make_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    return tokenizer


class TestGemma4WithTCS:
    """Test the full Gemma4WithTCS wrapper with mocked base model."""

    @pytest.fixture
    def model_cfg(self):
        return ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=7),
            lora=LoraConfig(enabled=False),  # Disable LoRA for mock tests
            gradient_checkpointing=False,
        )

    def test_init_with_tcs(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper.tcs is not None
        assert wrapper.d_model == 64
        assert hasattr(wrapper, "compressed_pos_emb")
        assert hasattr(wrapper, "decompressor")

    def test_init_hidden_size_from_text_config(self, model_cfg):
        """Gemma 4 multimodal: hidden_size lives under text_config."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        # Remove top-level hidden_size, add under text_config (like Gemma4Config)
        del mock_model.config.hidden_size
        mock_model.config.text_config = MagicMock()
        mock_model.config.text_config.hidden_size = 64

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper.d_model == 64

    def test_init_extracts_language_model(self, model_cfg):
        """Multimodal model: _lm points to model.language_model."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        # Simulate Gemma4ForConditionalGeneration structure
        mock_lm = MagicMock()
        mock_lm.get_input_embeddings = mock_model.get_input_embeddings
        mock_lm.side_effect = mock_model.side_effect
        mock_lm.__call__ = mock_model.__call__
        mock_model.model = MagicMock()
        mock_model.model.language_model = mock_lm

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper._lm is mock_lm

    def test_init_no_language_model_uses_base(self, model_cfg):
        """Non-multimodal model: _lm points to base_model itself."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper._lm is mock_model
        assert wrapper._lm_head is None  # CausalLM, no separate head needed

    def test_compute_logits_with_logits_attr(self, model_cfg):
        """_compute_logits returns .logits when available (CausalLM output)."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        outputs = MagicMock()
        outputs.logits = torch.randn(2, 32, 100)
        result = wrapper._compute_logits(outputs)
        assert torch.equal(result, outputs.logits)

    def test_compute_logits_with_lm_head(self, model_cfg):
        """_compute_logits applies lm_head when output lacks .logits."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        wrapper._lm_head = nn.Linear(64, 100, bias=False)
        outputs = MagicMock(spec=[])  # No .logits attribute
        outputs.last_hidden_state = torch.randn(2, 32, 64)
        result = wrapper._compute_logits(outputs)
        assert result.shape == (2, 32, 100)

    def test_compute_logits_tied_embeddings(self, model_cfg):
        """_compute_logits uses tied embeddings when no lm_head exists."""
        mock_model, embedding = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        wrapper._lm_head = None
        outputs = MagicMock(spec=[])
        outputs.last_hidden_state = torch.randn(2, 32, 64)
        result = wrapper._compute_logits(outputs)
        assert result.shape == (2, 32, 100)  # vocab_size=100

    def test_find_lm_head_on_language_model(self, model_cfg):
        """_find_lm_head finds lm_head on the language model."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        mock_lm = MagicMock()
        mock_lm.get_input_embeddings = mock_model.get_input_embeddings
        mock_lm.lm_head = nn.Linear(64, 100)
        mock_lm.side_effect = mock_model.side_effect
        mock_lm.__call__ = mock_model.__call__
        mock_model.model = MagicMock()
        mock_model.model.language_model = mock_lm

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper._lm_head is mock_lm.lm_head

    def test_init_without_tcs(self):
        cfg = ModelConfig(
            base_model_name="test",
            compression=CompressionConfig(enabled=False),
            lora=LoraConfig(enabled=False),
            gradient_checkpointing=False,
        )
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(cfg)

        assert wrapper.tcs is None

    def test_get_embeddings(self, model_cfg):
        mock_model, embedding = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        ids = torch.randint(0, 100, (2, 16))
        embeds = wrapper._get_embeddings(ids)
        assert embeds.shape == (2, 16, 64)

    def test_forward_compressed(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        ids = torch.randint(0, 100, (2, 64))
        mask = torch.ones(2, 64)
        result = wrapper.forward_compressed(ids, attention_mask=mask)
        assert "logits" in result
        assert "compressed_embeds" in result

    def test_forward_uncompressed(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        ids = torch.randint(0, 100, (2, 64))
        result = wrapper.forward_uncompressed(ids)
        assert "logits" in result

    def test_forward_both(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        ids = torch.randint(0, 100, (2, 64))
        result = wrapper(ids, mode="both")
        assert "student_logits" in result
        assert "teacher_logits" in result
        assert result["teacher_logits"].shape[1] == result["student_logits"].shape[1]

    def test_align_teacher_logits_to_student(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        teacher = torch.randn(1, 9, 4)
        aligned = wrapper._align_teacher_logits_to_student(teacher, student_len=3)
        expected = teacher[:, [0, 4, 8], :]
        assert torch.equal(aligned, expected)

    def test_forward_mode_uncompressed(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        ids = torch.randint(0, 100, (2, 64))
        result = wrapper(ids, mode="uncompressed")
        assert "logits" in result

    def test_forward_no_tcs_routes_to_uncompressed(self):
        cfg = ModelConfig(
            base_model_name="test",
            compression=CompressionConfig(enabled=False),
            lora=LoraConfig(enabled=False),
            gradient_checkpointing=False,
        )
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(cfg)

        ids = torch.randint(0, 100, (2, 32))
        result = wrapper(ids, mode="compressed")  # Should fallback to uncompressed
        assert "logits" in result

    def test_get_trainable_params_phase1(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        params = wrapper.get_trainable_params("tcs-pretrain")
        assert len(params) > 0

    def test_get_trainable_params_phase2(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        params = wrapper.get_trainable_params("tcs-lora")
        assert len(params) > 0

    def test_get_trainable_params_default(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        params = wrapper.get_trainable_params("some-other-phase")
        assert isinstance(params, list)

    def test_save_trainable(self, model_cfg, tmp_path):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        wrapper.save_trainable(str(tmp_path / "saved"))
        assert (tmp_path / "saved" / "tcs_weights.pt").exists()

    def test_load_trainable(self, model_cfg, tmp_path):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        # Save first
        wrapper.save_trainable(str(tmp_path / "weights"))
        # Load back
        wrapper.load_trainable(str(tmp_path / "weights"))

    def test_load_trainable_no_files(self, model_cfg, tmp_path):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        # Load from empty dir should not crash
        wrapper.load_trainable(str(tmp_path))

    def test_forward_compressed_with_multiscale(self):
        cfg = ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=8, multi_scale=True, ratios=[2, 4, 8]),
            lora=LoraConfig(enabled=False),
            gradient_checkpointing=False,
        )
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(cfg)

        ids = torch.randint(0, 100, (2, 64))
        result = wrapper.forward_compressed(ids, compression_ratio=4)
        assert "logits" in result

    def test_log_param_counts(self, model_cfg):
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            # _log_param_counts is called in __init__
            wrapper = Gemma4WithTCS(model_cfg)
        # Just ensure it doesn't crash
        wrapper._log_param_counts()


# ---------------------------------------------------------------------------
# Gemma4WithTCS with LoRA enabled (PeftModel branch coverage)
# ---------------------------------------------------------------------------

def _make_mock_peft_model(d_model=64, vocab_size=100):
    """Create a mock PeftModel to test LoRA-specific code paths."""
    from peft import PeftModel as RealPeftModel

    embedding = nn.Embedding(vocab_size, d_model)

    # Create a mock that isinstance checks see as PeftModel
    model = MagicMock(spec=RealPeftModel)
    model.config = MagicMock()
    model.config.hidden_size = d_model

    model.get_input_embeddings = MagicMock(return_value=embedding)
    base_inner = MagicMock()
    base_inner.get_input_embeddings = MagicMock(return_value=embedding)
    model.get_base_model = MagicMock(return_value=base_inner)

    # Prevent MagicMock from auto-creating model.model.language_model
    model.model = MagicMock(spec=[])

    def mock_forward(**kwargs):
        embeds = kwargs.get("inputs_embeds")
        input_ids = kwargs.get("input_ids")
        if embeds is not None:
            B, T = embeds.shape[0], embeds.shape[1]
        else:
            B, T = input_ids.shape
        result = MagicMock()
        result.logits = torch.randn(B, T, vocab_size)
        result.loss = None
        return result

    base_inner.side_effect = mock_forward
    base_inner.__call__ = mock_forward
    model.side_effect = mock_forward
    model.__call__ = mock_forward

    # For disable_adapter_layers context manager
    model.disable_adapter_layers = MagicMock()
    model.disable_adapter_layers.return_value.__enter__ = MagicMock()
    model.disable_adapter_layers.return_value.__exit__ = MagicMock(return_value=False)

    # LoRA trainable params
    trainable_param = torch.nn.Parameter(torch.randn(4, 4))
    trainable_param.requires_grad = True
    frozen_param = torch.nn.Parameter(torch.randn(4, 4))
    frozen_param.requires_grad = False
    model.parameters = MagicMock(side_effect=lambda: iter([trainable_param, frozen_param]))

    model.save_pretrained = MagicMock()

    # nn.Module internals needed for state_dict() / load_state_dict() traversal
    model._modules = {}
    model._parameters = {}
    model._buffers = {}
    model._load_state_dict_post_hooks = {}
    model._load_state_dict_pre_hooks = {}
    model._state_dict_hooks = {}
    model._state_dict_pre_hooks = {}
    model._non_persistent_buffers_set = set()

    return model, embedding


class TestGemma4WithTCSLoRA:
    """Tests for LoRA-enabled code paths in Gemma4WithTCS."""

    def _make_wrapper_with_lora(self):
        cfg = ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=7),
            lora=LoraConfig(enabled=True, rank=8),
            gradient_checkpointing=False,
        )
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()
        mock_peft, _ = _make_mock_peft_model()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            with patch("src.model._apply_lora", return_value=mock_peft):
                wrapper = Gemma4WithTCS(cfg)

        return wrapper

    def test_init_with_lora(self):
        wrapper = self._make_wrapper_with_lora()
        from peft import PeftModel as RealPeftModel
        assert isinstance(wrapper.base_model, RealPeftModel)

    def test_get_embeddings_peft(self):
        wrapper = self._make_wrapper_with_lora()
        ids = torch.randint(0, 100, (2, 16))
        embeds = wrapper._get_embeddings(ids)
        assert embeds.shape == (2, 16, 64)

    def test_forward_uncompressed_peft(self):
        wrapper = self._make_wrapper_with_lora()
        ids = torch.randint(0, 100, (2, 64))
        result = wrapper.forward_uncompressed(ids)
        assert "logits" in result
        # Should use disable_adapter_layers
        wrapper.base_model.disable_adapter_layers.assert_called()

    def test_save_trainable_with_lora(self, tmp_path):
        wrapper = self._make_wrapper_with_lora()
        wrapper.save_trainable(str(tmp_path / "out"))
        assert (tmp_path / "out" / "tcs_weights.pt").exists()
        wrapper.base_model.save_pretrained.assert_called_once()

    def test_load_trainable_with_lora(self, tmp_path):
        wrapper = self._make_wrapper_with_lora()

        # Manually create tcs_weights.pt and lora_adapters/ dir
        save_dir = tmp_path / "w"
        save_dir.mkdir()
        (save_dir / "lora_adapters").mkdir()
        tcs_state = {k: v for k, v in wrapper.state_dict().items()
                     if "tcs" in k or "compressed_pos_emb" in k or "decompressor" in k}
        torch.save(tcs_state, save_dir / "tcs_weights.pt")

        with patch("src.model.PeftModel.from_pretrained") as mock_peft_load:
            mock_peft_load.return_value = wrapper.base_model
            wrapper.load_trainable(str(save_dir))
            mock_peft_load.assert_called_once()

    def test_get_trainable_params_tcs_lora(self):
        wrapper = self._make_wrapper_with_lora()
        params = wrapper.get_trainable_params("tcs-lora")
        # Should include TCS params + LoRA params (requires_grad=True)
        assert len(params) > 0

    def test_get_trainable_params_tool_calling(self):
        wrapper = self._make_wrapper_with_lora()
        params = wrapper.get_trainable_params("tool-calling")
        assert len(params) > 0

    def test_gradient_checkpointing(self):
        cfg = ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=7),
            lora=LoraConfig(enabled=True, rank=8),
            gradient_checkpointing=True,
        )
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()
        mock_peft, _ = _make_mock_peft_model()
        # Add the missing method to the mock
        mock_peft.gradient_checkpointing_enable = MagicMock()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            with patch("src.model._apply_lora", return_value=mock_peft):
                wrapper = Gemma4WithTCS(cfg)

        mock_peft.gradient_checkpointing_enable.assert_called_once()


# ---------------------------------------------------------------------------
# Decoder-layer bypass for Gemma 4 per-layer input OOM
# ---------------------------------------------------------------------------

class TestDecoderLayerBypass:
    """Tests for direct decoder-layer forward, bypassing Gemma4TextModel.forward()."""

    @pytest.fixture
    def model_cfg(self):
        return ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=7),
            lora=LoraConfig(enabled=False),
            gradient_checkpointing=False,
        )

    def _make_multimodal_wrapper(self, model_cfg, *, with_layers: bool = True):
        """Create a wrapper that simulates Gemma4ForConditionalGeneration."""
        D, V = 64, 100
        mock_model, _ = _make_mock_base_model(d_model=D, vocab_size=V)
        mock_tokenizer = _make_mock_tokenizer()

        # Build a mock Gemma4TextModel with decoder layers + norm
        mock_lm = MagicMock(spec=[])  # spec=[] prevents auto-creating lm_head etc.
        embedding = nn.Embedding(V, D)
        mock_lm.get_input_embeddings = MagicMock(return_value=embedding)

        if with_layers:
            # Real decoder layers (simple linear → identity for testing)
            layer1 = MagicMock()
            layer1.side_effect = lambda h, **kw: (h,)
            layer1.layer_type = "sliding"
            layer2 = MagicMock()
            layer2.side_effect = lambda h, **kw: (h,)
            layer2.layer_type = "global"
            mock_lm.layers = nn.ModuleList([])
            # Use list-like access but actually we need real iteration
            mock_lm.layers = [layer1, layer2]
            mock_lm.norm = MagicMock(side_effect=lambda x: x)
            # Rotary embedding module: returns (cos, sin) tuples
            # Accepts layer_type= kwarg like Gemma4TextRotaryEmbedding
            mock_lm.rotary_emb = MagicMock(
                side_effect=lambda h, pos_ids, layer_type=None: (
                    torch.ones(pos_ids.shape[0], pos_ids.shape[1], D // 2),
                    torch.zeros(pos_ids.shape[0], pos_ids.shape[1], D // 2),
                )
            )
        else:
            # No layers attribute — should fall back to standard forward
            mock_lm.layers = None
            if hasattr(mock_lm, "norm"):
                del mock_lm.norm

        # Wire up multimodal structure
        mock_model.model = MagicMock()
        mock_model.model.language_model = mock_lm

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        return wrapper

    def test_find_decoder_components_found(self, model_cfg):
        """Multimodal model with layers+norm: components found."""
        wrapper = self._make_multimodal_wrapper(model_cfg, with_layers=True)
        assert wrapper._decoder_layers is not None
        assert wrapper._decoder_norm is not None
        assert len(wrapper._decoder_layers) == 2

    def test_find_decoder_components_standard_model(self, model_cfg):
        """Standard CausalLM (non-multimodal): no decoder bypass."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper._decoder_layers is None
        assert wrapper._decoder_norm is None

    def test_embed_scale_set_for_multimodal(self, model_cfg):
        """Embed scale is sqrt(d_model) when decoder layers are found."""
        wrapper = self._make_multimodal_wrapper(model_cfg, with_layers=True)
        expected = 64 ** 0.5  # d_model = 64
        assert abs(wrapper._embed_scale - expected) < 1e-6

    def test_embed_scale_one_for_standard(self, model_cfg):
        """Standard model keeps embed_scale = 1.0."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper._embed_scale == 1.0

    def test_hidden_to_logits_with_lm_head(self, model_cfg):
        """_hidden_to_logits uses lm_head when present."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        wrapper._lm_head = nn.Linear(64, 100, bias=False)
        hidden = torch.randn(2, 16, 64)
        logits = wrapper._hidden_to_logits(hidden)
        assert logits.shape == (2, 16, 100)

    def test_hidden_to_logits_tied_embeddings(self, model_cfg):
        """_hidden_to_logits falls back to tied embedding projection."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        wrapper._lm_head = None
        hidden = torch.randn(2, 16, 64)
        logits = wrapper._hidden_to_logits(hidden)
        assert logits.shape == (2, 16, 100)  # vocab_size=100

    def test_forward_decoder_layers_shape(self, model_cfg):
        """Direct decoder forward produces correct output shape."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        hidden = torch.randn(2, 16, 64)
        result = wrapper._forward_decoder_layers(hidden)
        assert "logits" in result
        assert result["logits"].shape[:2] == (2, 16)
        assert result["loss"] is None

    def test_forward_decoder_layers_calls_each_layer(self, model_cfg):
        """Each decoder layer is called during direct forward."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        hidden = torch.randn(2, 16, 64)
        wrapper._forward_decoder_layers(hidden)
        for layer in wrapper._decoder_layers:
            layer.assert_called_once()

    def test_forward_decoder_layers_passes_position_embeddings(self, model_cfg):
        """Layers receive pre-computed position_embeddings from rotary_emb."""
        wrapper = self._make_multimodal_wrapper(model_cfg)

        received_kwargs = {}
        def capture_kwargs(h, **kw):
            received_kwargs.update(kw)
            return (h,)
        wrapper._decoder_layers[0].side_effect = capture_kwargs

        hidden = torch.randn(2, 16, 64)
        wrapper._forward_decoder_layers(hidden)
        assert "position_embeddings" in received_kwargs
        cos, sin = received_kwargs["position_embeddings"]
        assert cos.shape == (2, 16, 32)  # D // 2 = 32
        assert sin.shape == (2, 16, 32)

    def test_forward_decoder_layers_passes_per_layer_input(self, model_cfg):
        """Layers receive per_layer_input=ones (identity gate)."""
        wrapper = self._make_multimodal_wrapper(model_cfg)

        received_kwargs = {}
        def capture_kwargs(h, **kw):
            received_kwargs.update(kw)
            return (h,)
        wrapper._decoder_layers[0].side_effect = capture_kwargs

        hidden = torch.randn(2, 16, 64)
        wrapper._forward_decoder_layers(hidden)
        assert "per_layer_input" in received_kwargs
        pli = received_kwargs["per_layer_input"]
        assert torch.equal(pli, torch.ones(1))

    def test_rotary_emb_detected(self, model_cfg):
        """Multimodal model: rotary_emb is found and stored."""
        wrapper = self._make_multimodal_wrapper(model_cfg, with_layers=True)
        assert wrapper._rotary_emb is not None

    def test_rotary_emb_none_for_standard(self, model_cfg):
        """Standard CausalLM: no rotary_emb bypass."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        assert wrapper._rotary_emb is None

    def test_rotary_emb_called_per_layer_type(self, model_cfg):
        """Rotary embedding is called once per unique layer_type."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        hidden = torch.randn(2, 16, 64)
        wrapper._forward_decoder_layers(hidden)
        # Two layers with types "sliding" and "global" → 2 calls
        assert wrapper._rotary_emb.call_count == 2
        call_kwargs = [c.kwargs for c in wrapper._rotary_emb.call_args_list]
        layer_types = {kw["layer_type"] for kw in call_kwargs}
        assert layer_types == {"sliding", "global"}

    def test_forward_decoder_layers_applies_norm(self, model_cfg):
        """Decoder norm is applied after all layers."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        hidden = torch.randn(2, 16, 64)
        wrapper._forward_decoder_layers(hidden)
        wrapper._decoder_norm.assert_called_once()

    def test_forward_decoder_layers_scales_embeddings(self, model_cfg):
        """Input embeddings are scaled by sqrt(d_model)."""
        wrapper = self._make_multimodal_wrapper(model_cfg)

        # Capture what the first layer receives
        received = {}
        def capture_input(h, **kw):
            received["hidden"] = h.clone()
            return (h,)
        wrapper._decoder_layers[0].side_effect = capture_input

        hidden = torch.randn(2, 16, 64)
        wrapper._forward_decoder_layers(hidden)
        expected_scaled = hidden * (64 ** 0.5)
        assert torch.allclose(received["hidden"], expected_scaled, atol=1e-5)

    def test_forward_from_embeds_routes_to_decoder_layers(self, model_cfg):
        """_forward_from_embeds uses direct layers when available."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        embeds = torch.randn(2, 16, 64)
        result = wrapper._forward_from_embeds(embeds)
        assert "logits" in result
        # Verify layers were called (not self._lm)
        for layer in wrapper._decoder_layers:
            assert layer.called

    def test_forward_from_embeds_standard_model(self, model_cfg):
        """Standard model goes through self._lm (no decoder bypass)."""
        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        embeds = torch.randn(2, 16, 64)
        result = wrapper._forward_from_embeds(embeds)
        assert "logits" in result

    def test_forward_compressed_end_to_end_multimodal(self, model_cfg):
        """Full compressed forward via decoder layers on multimodal model."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        ids = torch.randint(0, 100, (2, 64))
        mask = torch.ones(2, 64)
        result = wrapper.forward_compressed(ids, attention_mask=mask)
        assert "logits" in result
        assert "compressed_embeds" in result

    def test_layer_returns_tensor_not_tuple(self, model_cfg):
        """Gemma4TextDecoderLayer returns a tensor, not a tuple."""
        wrapper = self._make_multimodal_wrapper(model_cfg)
        # Override layers to return raw tensors (like real Gemma4TextDecoderLayer)
        wrapper._decoder_layers[0].side_effect = lambda h, **kw: h
        wrapper._decoder_layers[1].side_effect = lambda h, **kw: h
        hidden = torch.randn(2, 16, 64)
        result = wrapper._forward_decoder_layers(hidden)
        # Batch dimension must be preserved through all layers
        assert result["logits"].shape[0] == 2

    def test_shared_kv_states_passed_to_layers(self, model_cfg):
        """Layers receive shared_kv_states dict for KV sharing."""
        wrapper = self._make_multimodal_wrapper(model_cfg)

        received_kwargs = {}
        def capture_kwargs(h, **kw):
            received_kwargs.update(kw)
            return (h,)
        wrapper._decoder_layers[0].side_effect = capture_kwargs

        hidden = torch.randn(2, 16, 64)
        wrapper._forward_decoder_layers(hidden)
        assert "shared_kv_states" in received_kwargs
        assert isinstance(received_kwargs["shared_kv_states"], dict)


class TestGemma4WithTCSEdgeCases:
    def test_hidden_size_not_found_raises(self):
        """Model without hidden_size or text_config raises AttributeError."""
        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=[])  # no hidden_size, no text_config
        del mock_model.config.hidden_size
        del mock_model.config.text_config
        mock_model.parameters.return_value = iter([])

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        model_cfg = ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=7),
            lora=LoraConfig(enabled=False),
        )

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            with pytest.raises(AttributeError, match="Cannot determine hidden_size"):
                Gemma4WithTCS(model_cfg)

    def test_find_decoder_no_layers_returns_none(self):
        """When _lm has no .layers or .norm, returns (None, None)."""
        from src.model import Gemma4WithTCS

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 64
        # model.language_model exists but has no .layers
        lm = MagicMock(spec=[])
        del lm.layers
        del lm.norm
        mock_model.model.language_model = lm
        mock_model.config.text_config = MagicMock(hidden_size=64)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        model_cfg = ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=7),
            lora=LoraConfig(enabled=False),
        )

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)
        assert wrapper._decoder_layers is None
        assert wrapper._decoder_norm is None

    def test_layer_type_from_self_attn(self):
        """layer_type fallback: read from self_attn when layer has no layer_type."""
        from tests.test_model import _make_mock_base_model, _make_mock_tokenizer

        model_cfg = ModelConfig(
            base_model_name="test",
            max_seq_len=64,
            compression=CompressionConfig(ratio=4, kernel_size=7),
            lora=LoraConfig(enabled=False),
        )

        mock_model, _ = _make_mock_base_model()
        mock_tokenizer = _make_mock_tokenizer()

        # Build multimodal model with layers that have no layer_type attribute
        # but self_attn has one
        class _FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = MagicMock(layer_type="sliding")
            def forward(self, h, **kw):
                return (h,)

        class _FakeTextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer()])
                self.norm = nn.Identity()
                self.rotary_emb = MagicMock(return_value=(torch.randn(2, 16, 32), torch.randn(2, 16, 32)))
                self._embed = nn.Embedding(100, 64)
            def get_input_embeddings(self):
                return self._embed

        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _FakeTextModel()

        mock_model.model = _Container()
        mock_model.lm_head = nn.Linear(64, 100, bias=False)
        mock_model.config = MagicMock()
        mock_model.config.text_config = MagicMock(hidden_size=64)
        del mock_model.config.hidden_size

        with patch("src.model._load_base_model", return_value=(mock_model, mock_tokenizer)):
            wrapper = Gemma4WithTCS(model_cfg)

        # Forward should work and read layer_type from self_attn
        ids = torch.randint(0, 100, (2, 64))
        result = wrapper.forward_compressed(ids)
        assert "logits" in result
