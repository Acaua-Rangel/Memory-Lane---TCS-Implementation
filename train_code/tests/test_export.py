"""
Tests for LiteRT export module.

Tests exportable wrappers, manifest creation, and CLI parsing.
Actual model export requires ai_edge_torch and is mocked.
"""

import json
import math
import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from src.config import ModelConfig, CompressionConfig, FaceEmbeddingConfig, DatabaseConfig
from src.export_litert import (
    TCSExportWrapper,
    create_deployment_manifest,
    export_face_pipeline,
)


# ---------------------------------------------------------------------------
# TCSExportWrapper
# ---------------------------------------------------------------------------

class TestTCSExportWrapper:
    def test_forward_shape(self):
        from src.model import CompressionLayer
        import math

        cfg = CompressionConfig(ratio=4, kernel_size=7)
        tcs = CompressionLayer(d_model=64, comp_cfg=cfg)
        comp_len = math.ceil(128 / 4) + 1
        pos_emb = torch.nn.Embedding(comp_len, 64)

        wrapper = TCSExportWrapper(tcs, pos_emb)
        x = torch.randn(1, 128, 64)
        out = wrapper(x)
        assert out.shape[0] == 1
        assert out.shape[2] == 64
        assert abs(out.shape[1] - 32) <= 1  # ~128/4

    def test_gradient_flow(self):
        from src.model import CompressionLayer
        import math

        cfg = CompressionConfig(ratio=4, kernel_size=7)
        tcs = CompressionLayer(d_model=64, comp_cfg=cfg)
        comp_len = math.ceil(128 / 4) + 1
        pos_emb = torch.nn.Embedding(comp_len, 64)

        wrapper = TCSExportWrapper(tcs, pos_emb)
        x = torch.randn(1, 128, 64, requires_grad=True)
        out = wrapper(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Deployment Manifest
# ---------------------------------------------------------------------------

class TestDeploymentManifest:
    def test_create_manifest(self, tmp_path):
        model_cfg = ModelConfig()
        face_cfg = FaceEmbeddingConfig()
        db_cfg = DatabaseConfig()

        manifest_path = create_deployment_manifest(
            tmp_path, model_cfg, face_cfg, db_cfg, exported_files={}
        )
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["app_name"] == "Gemma 4 Memory Companion"
        assert "models" in manifest
        assert "llm" in manifest["models"]
        assert "tcs" in manifest["models"]
        assert "face_embedder" in manifest["models"]
        assert "routing" in manifest
        assert manifest["device_requirements"]["offline_capable"] is True

    def test_manifest_compression_ratio(self, tmp_path):
        model_cfg = ModelConfig(compression=CompressionConfig(ratio=8))
        manifest_path = create_deployment_manifest(
            tmp_path, model_cfg, FaceEmbeddingConfig(), DatabaseConfig(), {}
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["models"]["tcs"]["compression_ratio"] == 8

    def test_manifest_uses_explicit_tcs_hidden_size(self, tmp_path):
        manifest_path = create_deployment_manifest(
            tmp_path,
            ModelConfig(),
            FaceEmbeddingConfig(),
            DatabaseConfig(),
            {},
            tcs_hidden_size=1536,
        )

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["models"]["tcs"]["input_shape"] == [1, 512, 1536]


# ---------------------------------------------------------------------------
# Face Pipeline Export
# ---------------------------------------------------------------------------

class TestFacePipelineExport:
    def test_export_missing_models(self, tmp_path):
        face_cfg = FaceEmbeddingConfig(
            embedder_onnx_path=str(tmp_path / "nonexistent.onnx"),
        )
        result = export_face_pipeline(tmp_path / "face_out", face_cfg)
        assert "embedder_onnx" not in result

    def test_export_with_existing_model(self, tmp_path):
        # Create a dummy onnx file
        onnx_file = tmp_path / "mobilefacenet.onnx"
        onnx_file.write_bytes(b"\x00" * 100)

        face_cfg = FaceEmbeddingConfig(embedder_onnx_path=str(onnx_file))
        result = export_face_pipeline(tmp_path / "face_out", face_cfg)
        assert "embedder_onnx" in result
        assert (tmp_path / "face_out" / "mobilefacenet.onnx").exists()


# ---------------------------------------------------------------------------
# TCS TFLite Export (mocked)
# ---------------------------------------------------------------------------

class TestExportTCS:
    def test_export_no_weights(self, tmp_path):
        from src.export_litert import export_tcs_to_tflite
        model_cfg = ModelConfig()
        result = export_tcs_to_tflite(tmp_path, tmp_path / "out", model_cfg)
        assert result is None  # No tcs_weights.pt → returns None

    def test_export_tcs_no_ai_edge_no_onnx(self, tmp_path):
        """When both ai_edge_torch and onnx export fail."""
        from src.export_litert import export_tcs_to_tflite

        # Create fake tcs weights
        weights_path = tmp_path / "tcs_weights.pt"
        from src.model import CompressionLayer
        import math
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        tcs = CompressionLayer(d_model=2048, comp_cfg=cfg)
        comp_len = math.ceil(512 / 4) + 1
        pos_emb = torch.nn.Embedding(comp_len, 2048)
        torch.save({"tcs": tcs.state_dict(), "pos_emb": pos_emb.state_dict()}, weights_path)

        model_cfg = ModelConfig()

        # Mock both export methods to fail
        with patch.dict("sys.modules", {"ai_edge_torch": None}):
            with patch("torch.onnx.export", side_effect=Exception("onnx failed")):
                result = export_tcs_to_tflite(tmp_path, tmp_path / "out", model_cfg)
        assert result is None

    def test_export_tcs_supports_flat_checkpoint_format(self, tmp_path):
        from src.export_litert import export_tcs_to_tflite
        from src.model import CompressionLayer

        cfg = CompressionConfig(ratio=4, kernel_size=7)
        d_model = 1536
        tcs = CompressionLayer(d_model=d_model, comp_cfg=cfg)
        comp_len = math.ceil(512 / 4) + 1
        pos_emb = torch.nn.Embedding(comp_len, d_model)

        flat_state = {}
        for key, value in tcs.state_dict().items():
            flat_state[f"tcs.{key}"] = value
        for key, value in pos_emb.state_dict().items():
            flat_state[f"compressed_pos_emb.{key}"] = value
        torch.save(flat_state, tmp_path / "tcs_weights.pt")

        model_cfg = ModelConfig()
        mock_ai_edge = MagicMock()
        mock_edge_model = MagicMock()
        mock_ai_edge.convert.return_value = mock_edge_model

        def mock_export(path):
            Path(path).write_bytes(b"\x00" * 100)

        mock_edge_model.export.side_effect = mock_export

        import sys
        with patch.dict(sys.modules, {"ai_edge_torch": mock_ai_edge}):
            result = export_tcs_to_tflite(tmp_path, tmp_path / "out", model_cfg)

        assert result is not None
        sample_input = mock_ai_edge.convert.call_args.args[1][0]
        assert sample_input.shape == (1, 512, 1536)


# ---------------------------------------------------------------------------
# Gemma LiteRT Export (mocked)
# ---------------------------------------------------------------------------

class TestExportGemmaLiteRT:
    def test_no_lora_adapters(self, tmp_path):
        from src.export_litert import export_gemma_litert
        model_cfg = ModelConfig()
        result = export_gemma_litert(tmp_path, tmp_path / "out", model_cfg)
        assert result is None  # No lora_adapters/ → returns None

    def test_no_ai_edge_torch(self, tmp_path):
        from src.export_litert import export_gemma_litert
        lora_dir = tmp_path / "lora_adapters"
        lora_dir.mkdir()
        model_cfg = ModelConfig()

        with patch.dict("sys.modules", {"ai_edge_torch": None, "ai_edge_torch.generative": None, "ai_edge_torch.generative.utilities": None, "ai_edge_torch.generative.utilities.converter": None}):
            result = export_gemma_litert(tmp_path, tmp_path / "out", model_cfg)
        assert result is None


# ---------------------------------------------------------------------------
# Full export_all (mocked)
# ---------------------------------------------------------------------------

class TestExportAll:
    def test_export_all_empty(self, tmp_path):
        from src.export_litert import export_all
        results = export_all(tmp_path, tmp_path / "out")
        assert "manifest" in results
        assert "total_size_mb" in results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestExportCLI:
    def test_module_imports(self):
        from src.export_litert import main
        assert callable(main)

    def test_cli_tcs_only(self, tmp_path):
        from src.export_litert import main

        with patch("sys.argv", [
            "export", "--model-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"), "--tcs-only",
        ]):
            with patch("src.export_litert.export_tcs_to_tflite") as mock_export:
                mock_export.return_value = None
                main()
                mock_export.assert_called_once()

    def test_cli_full_export(self, tmp_path):
        from src.export_litert import main

        with patch("sys.argv", [
            "export", "--model-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
        ]):
            with patch("src.export_litert.export_all") as mock_export:
                mock_export.return_value = {"manifest": "path"}
                main()
                mock_export.assert_called_once()


class TestExportTCSWithAiEdge:
    def test_export_tcs_infers_hidden_size_from_nested_checkpoint(self, tmp_path):
        """Export should derive the embedding width from nested TCS checkpoint weights."""
        from src.export_litert import export_tcs_to_tflite
        from src.model import CompressionLayer

        cfg = CompressionConfig(ratio=4, kernel_size=7)
        d_model = 1536
        tcs = CompressionLayer(d_model=d_model, comp_cfg=cfg)
        comp_len = math.ceil(512 / 4) + 1
        pos_emb = torch.nn.Embedding(comp_len, d_model)
        torch.save(
            {"tcs": tcs.state_dict(), "pos_emb": pos_emb.state_dict()},
            tmp_path / "tcs_weights.pt",
        )

        model_cfg = ModelConfig()

        mock_edge = MagicMock()
        mock_edge_model = MagicMock()
        mock_edge.convert.return_value = mock_edge_model

        def mock_export(path):
            Path(path).write_bytes(b"\x00" * 100)

        mock_edge_model.export.side_effect = mock_export

        import sys
        with patch.dict(sys.modules, {"ai_edge_torch": mock_edge}):
            result = export_tcs_to_tflite(tmp_path, tmp_path / "out", model_cfg)

        assert result is not None
        sample_input = mock_edge.convert.call_args.args[1][0]
        assert sample_input.shape == (1, 512, 1536)

    def test_export_tcs_with_ai_edge_mock(self, tmp_path):
        """Test the ai_edge_torch success path with mocked module."""
        from src.export_litert import export_tcs_to_tflite
        from src.model import CompressionLayer

        # Create valid TCS weights
        cfg = CompressionConfig(ratio=4, kernel_size=7)
        tcs = CompressionLayer(d_model=2048, comp_cfg=cfg)
        comp_len = math.ceil(512 / 4) + 1
        pos_emb = torch.nn.Embedding(comp_len, 2048)
        torch.save({"tcs": tcs.state_dict(), "pos_emb": pos_emb.state_dict()},
                    tmp_path / "tcs_weights.pt")

        model_cfg = ModelConfig()

        # Mock ai_edge_torch to simulate successful TFLite export
        mock_edge = MagicMock()
        mock_edge_model = MagicMock()
        mock_edge.convert.return_value = mock_edge_model

        def mock_export(path):
            # Create the tflite file
            Path(path).write_bytes(b"\x00" * 100)

        mock_edge_model.export.side_effect = mock_export

        import sys
        with patch.dict(sys.modules, {"ai_edge_torch": mock_edge}):
            result = export_tcs_to_tflite(tmp_path, tmp_path / "out", model_cfg)

        assert result is not None
        assert result.suffix == ".tflite"

    def test_export_tcs_onnx_fallback(self, tmp_path):
        """Test the ONNX fallback path (when ai_edge_torch not available)."""
        from src.export_litert import export_tcs_to_tflite
        from src.model import CompressionLayer

        cfg = CompressionConfig(ratio=4, kernel_size=7)
        tcs = CompressionLayer(d_model=2048, comp_cfg=cfg)
        comp_len = math.ceil(512 / 4) + 1
        pos_emb = torch.nn.Embedding(comp_len, 2048)
        torch.save({"tcs": tcs.state_dict(), "pos_emb": pos_emb.state_dict()},
                    tmp_path / "tcs_weights.pt")

        model_cfg = ModelConfig()

        # ai_edge_torch not available, but ONNX export succeeds
        with patch.dict("sys.modules", {"ai_edge_torch": None}):
            result = export_tcs_to_tflite(tmp_path, tmp_path / "out", model_cfg)

        # If ONNX is available, result should be an ONNX file path
        if result is not None:
            assert result.suffix == ".onnx"


class TestExportGemmaLiteRTSuccess:
    def test_export_gemma_success(self, tmp_path):
        """Test export_gemma_litert with all dependencies mocked."""
        from src.export_litert import export_gemma_litert

        lora_dir = tmp_path / "lora_adapters"
        lora_dir.mkdir()

        model_cfg = ModelConfig()

        mock_ai_edge = MagicMock()
        mock_converter = MagicMock()

        mock_edge_model = MagicMock()
        mock_ai_edge.convert.return_value = mock_edge_model

        def mock_export(path):
            Path(path).write_bytes(b"\x00" * 100)
        mock_edge_model.export.side_effect = mock_export

        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_base_model
        mock_base_model.eval = MagicMock()

        import sys
        with patch.dict(sys.modules, {
            "ai_edge_torch": mock_ai_edge,
            "ai_edge_torch.generative": MagicMock(),
            "ai_edge_torch.generative.utilities": MagicMock(),
            "ai_edge_torch.generative.utilities.converter": mock_converter,
        }):
            with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()):
                with patch("peft.PeftModel.from_pretrained", return_value=mock_peft_model):
                    result = export_gemma_litert(tmp_path, tmp_path / "out", model_cfg)

        assert result is not None

    def test_export_gemma_conversion_failure(self, tmp_path):
        """Test export_gemma_litert when conversion fails."""
        from src.export_litert import export_gemma_litert

        lora_dir = tmp_path / "lora_adapters"
        lora_dir.mkdir()

        model_cfg = ModelConfig()

        mock_ai_edge = MagicMock()
        mock_ai_edge.convert.side_effect = RuntimeError("Conversion failed")

        mock_peft_model = MagicMock()
        mock_merged = MagicMock()
        mock_merged.eval = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        import sys
        with patch.dict(sys.modules, {
            "ai_edge_torch": mock_ai_edge,
            "ai_edge_torch.generative": MagicMock(),
            "ai_edge_torch.generative.utilities": MagicMock(),
            "ai_edge_torch.generative.utilities.converter": MagicMock(),
        }):
            with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()):
                with patch("peft.PeftModel.from_pretrained", return_value=mock_peft_model):
                    result = export_gemma_litert(tmp_path, tmp_path / "out", model_cfg)

        assert result is None
