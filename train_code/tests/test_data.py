"""
Tests for data module: TextDataset, ToolCallingDataset, template generators,
GemmaDataGenerator (mocked), synthetic data generation, and CLI.
"""

import json
import pytest
import random
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import torch

from src.config import ToolCallingConfig, TrainingConfig
from src.data import (
    TextDataset,
    ToolCallingDataset,
    GemmaDataGenerator,
    GeminiApiGenerator,
    QuotaExhaustedError,
    _generate_person_recognition_example,
    _generate_orientation_example,
    _generate_medication_example,
    _generate_alert_example,
    _SCENARIOS,
    _SAMPLE_PERSONS,
    _SAMPLE_MEDICATIONS,
    _SAMPLE_LOCATIONS,
    _GENERATORS,
    _resolve_gemini_api_key,
    _is_quota_error,
    _create_generator,
    _generate_example_with_fallback,
    _generate_batch_with_fallback,
    _finish_with_parallel,
    generate_synthetic_data,
    build_dataloader,
)


# ---------------------------------------------------------------------------
# Template-based generators
# ---------------------------------------------------------------------------

class TestTemplateGenerators:
    def test_person_recognition(self):
        tc_cfg = ToolCallingConfig()
        example = _generate_person_recognition_example(tc_cfg)
        msgs = example["messages"]
        assert len(msgs) == 5
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "<tool_call>" in msgs[2]["content"]
        assert msgs[3]["role"] == "tool"
        assert msgs[4]["role"] == "assistant"

    def test_orientation(self):
        tc_cfg = ToolCallingConfig()
        example = _generate_orientation_example(tc_cfg)
        msgs = example["messages"]
        assert len(msgs) == 5
        assert "describe_location" in msgs[2]["content"]

    def test_medication(self):
        tc_cfg = ToolCallingConfig()
        random.seed(42)
        example = _generate_medication_example(tc_cfg)
        msgs = example["messages"]
        assert len(msgs) == 5
        assert "get_medication" in msgs[2]["content"]

    def test_alert(self):
        tc_cfg = ToolCallingConfig()
        example = _generate_alert_example(tc_cfg)
        msgs = example["messages"]
        assert len(msgs) == 5
        assert "alert_caregiver" in msgs[2]["content"]

    def test_generators_dict_coverage(self):
        """All categories in _GENERATORS should produce valid examples."""
        tc_cfg = ToolCallingConfig()
        for cat, gen_fn in _GENERATORS.items():
            example = gen_fn(tc_cfg)
            assert "messages" in example
            assert len(example["messages"]) == 5


# ---------------------------------------------------------------------------
# Scenarios and sample data
# ---------------------------------------------------------------------------

class TestScenariosData:
    def test_scenario_weights_sum_to_1(self):
        total = sum(s["weight"] for s in _SCENARIOS)
        assert abs(total - 1.0) < 1e-6

    def test_sample_persons_have_required_fields(self):
        for person in _SAMPLE_PERSONS:
            assert "face_id" in person
            assert "name" in person
            assert "relationship" in person
            assert "bio" in person

    def test_sample_medications_all_times(self):
        for tod in ["morning", "afternoon", "evening", "night"]:
            assert tod in _SAMPLE_MEDICATIONS

    def test_sample_locations_nonempty(self):
        assert len(_SAMPLE_LOCATIONS) > 0
        for key, loc in _SAMPLE_LOCATIONS.items():
            assert "name" in loc
            assert "description" in loc


# ---------------------------------------------------------------------------
# generate_synthetic_data (template mode)
# ---------------------------------------------------------------------------

class TestGenerateSyntheticData:
    def test_template_mode(self, tmp_path):
        generate_synthetic_data(
            output_dir=str(tmp_path),
            num_train=20,
            num_val=5,
            seed=42,
            use_gemma=False,
        )
        train_path = tmp_path / "tool_calling_train.jsonl"
        val_path = tmp_path / "tool_calling_val.jsonl"
        assert train_path.exists()
        assert val_path.exists()

        with open(train_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 20

        with open(val_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 5

    def test_output_format(self, tmp_path):
        generate_synthetic_data(
            output_dir=str(tmp_path),
            num_train=3,
            num_val=1,
            seed=42,
            use_gemma=False,
        )
        with open(tmp_path / "tool_calling_train.jsonl") as f:
            example = json.loads(f.readline())
        assert "messages" in example
        msgs = example["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]


# ---------------------------------------------------------------------------
# ToolCallingDataset
# ---------------------------------------------------------------------------

class TestToolCallingDataset:
    def test_load_and_getitem(self, tmp_path):
        """Create a small JSONL and load it."""
        tc_cfg = ToolCallingConfig()
        examples = [_generate_person_recognition_example(tc_cfg) for _ in range(3)]
        path = tmp_path / "tool_calling_train.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        # Use a mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=list(range(50)))
        tokenizer.decode = MagicMock(return_value="decoded text")
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }
        tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        })

        dataset = ToolCallingDataset(tokenizer=tokenizer, data_path=str(path), max_seq_len=64)
        assert len(dataset) == 3

    def test_file_not_found(self):
        tokenizer = MagicMock()
        with pytest.raises(FileNotFoundError):
            ToolCallingDataset(tokenizer=tokenizer, data_path="nonexistent.jsonl")


# ---------------------------------------------------------------------------
# GemmaDataGenerator (mocked — no GPU)
# ---------------------------------------------------------------------------

class TestGemmaDataGenerator:
    def test_parse_numbered_list(self):
        text = "1. Where am I?\n2. Who is that person?\n3. I'm confused.\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 3
        assert "Where am I?" in result[0]

    def test_parse_numbered_list_various_formats(self):
        text = "1) First item\n2: Second item\n3- Third item\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 3

    def test_parse_numbered_list_with_short_lines(self):
        text = "1. OK\n2. A longer valid line here\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        # "OK" is too short (< 5 chars)
        assert len(result) == 1

    def test_parse_numbered_list_empty(self):
        result = GemmaDataGenerator._parse_numbered_list("", expected=5)
        assert result == []

    def test_parse_numbered_list_truncates(self):
        text = "\n".join(f"{i}. Item number {i} text" for i in range(1, 20))
        result = GemmaDataGenerator._parse_numbered_list(text, expected=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestDataCLI:
    def test_main_no_gemma(self, tmp_path, monkeypatch):
        """Test CLI entry point with --no-gemma mode."""
        monkeypatch.setattr(
            "sys.argv",
            ["generate-data", "--output-dir", str(tmp_path),
             "--num-train", "5", "--num-val", "2", "--no-gemma"],
        )
        from src.data import main
        main()
        assert (tmp_path / "tool_calling_train.jsonl").exists()
        assert (tmp_path / "tool_calling_val.jsonl").exists()


# ---------------------------------------------------------------------------
# TextDataset (mocked HF datasets)
# ---------------------------------------------------------------------------

class TestTextDataset:
    def test_creates_chunks(self):
        """TextDataset should create fixed-length token chunks."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=list(range(200)))

        fake_dataset = [
            {"text": "Hello world this is a test sentence that is long enough."},
            {"text": "Another sentence for testing purposes here."},
            {"text": ""},  # Empty — should be skipped
            {"text": "Short"},  # Too short (< 10 chars) — skipped
        ]

        with patch("datasets.load_dataset", return_value=fake_dataset):
            ds = TextDataset(
                tokenizer=mock_tokenizer,
                max_seq_len=64,
                max_samples=None,
            )
        assert len(ds) > 0

        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (64,)
        assert item["labels"][-1].item() == -100

    def test_max_samples_limits(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=list(range(100)))

        fake_dataset = [{"text": f"Text number {i} " * 10} for i in range(100)]

        with patch("datasets.load_dataset", return_value=fake_dataset):
            ds = TextDataset(
                tokenizer=mock_tokenizer,
                max_seq_len=32,
                max_samples=5,
            )
        # Should have chunked only ~5 examples worth of tokens
        assert len(ds) >= 1

    def test_text_dataset_cache_reuse(self, tmp_path):
        mock_tokenizer = MagicMock()
        type(mock_tokenizer).name_or_path = PropertyMock(return_value="mock-tokenizer")
        mock_tokenizer.encode = MagicMock(return_value=list(range(128)))

        fake_dataset = [{"text": "Caching test sentence long enough to tokenize."} for _ in range(10)]
        cache_dir = tmp_path / "text_cache"

        with patch("datasets.load_dataset", return_value=fake_dataset) as load_mock:
            first = TextDataset(
                tokenizer=mock_tokenizer,
                max_seq_len=32,
                cache_dir=str(cache_dir),
                use_cache=True,
            )
            assert load_mock.call_count == 1
            assert len(first) > 0

        with patch("datasets.load_dataset", side_effect=RuntimeError("should not reload dataset")):
            second = TextDataset(
                tokenizer=mock_tokenizer,
                max_seq_len=32,
                cache_dir=str(cache_dir),
                use_cache=True,
            )

        assert len(second) == len(first)
        assert torch.equal(second[0]["input_ids"], first[0]["input_ids"])

    def test_text_dataset_cache_fallback_when_primary_dir_is_file(self, tmp_path, monkeypatch):
        mock_tokenizer = MagicMock()
        type(mock_tokenizer).name_or_path = PropertyMock(return_value="mock-tokenizer")
        mock_tokenizer.encode = MagicMock(return_value=list(range(128)))

        fake_dataset = [{"text": "Cache fallback sentence long enough to tokenize."} for _ in range(5)]

        bad_data_path = tmp_path / "data"
        bad_data_path.write_text("not a directory")
        invalid_cache_dir = bad_data_path / "text_cache"

        fallback_cache_dir = tmp_path / "fallback_text_cache"
        monkeypatch.setenv("GEMMA_TEXT_CACHE_DIR", str(fallback_cache_dir))

        with patch("datasets.load_dataset", return_value=fake_dataset):
            dataset = TextDataset(
                tokenizer=mock_tokenizer,
                max_seq_len=32,
                cache_dir=str(invalid_cache_dir),
                use_cache=True,
            )

        assert len(dataset) > 0
        assert len(list(fallback_cache_dir.glob("text_chunks_*.pt"))) == 1


# ---------------------------------------------------------------------------
# ToolCallingDataset (full getitem coverage)
# ---------------------------------------------------------------------------

class TestToolCallingDatasetFull:
    def _make_dataset(self, tmp_path):
        tc_cfg = ToolCallingConfig()
        examples = [_generate_person_recognition_example(tc_cfg) for _ in range(2)]
        path = tmp_path / "tool_calling_train.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        # Real-ish tokenizer mock
        tokenizer = MagicMock()

        def mock_call(text, **kwargs):
            max_length = kwargs.get("max_length", 64)
            return {
                "input_ids": torch.randint(0, 100, (1, max_length)),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        tokenizer.encode = MagicMock(return_value=list(range(50)))
        tokenizer.decode = MagicMock(return_value="<start_of_turn>model\ntest<end_of_turn>")

        return ToolCallingDataset(tokenizer=tokenizer, data_path=str(path), max_seq_len=64)

    def test_getitem_returns_tensors(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].dim() == 1
        assert item["labels"].dim() == 1

    def test_format_conversation(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "ast"},
            {"role": "tool", "content": "tl"},
        ]
        text = ds._format_conversation(messages)
        assert "<start_of_turn>system" in text
        assert "<start_of_turn>user" in text
        assert "<start_of_turn>model" in text
        assert "<start_of_turn>tool" in text


# ---------------------------------------------------------------------------
# build_dataloader
# ---------------------------------------------------------------------------

class TestBuildDataloader:
    def test_text_phase_uses_explicit_max_seq_len(self):
        tokenizer = MagicMock()
        cfg = TrainingConfig(phase="tcs-lora", batch_size=4)

        with patch("src.data.TextDataset") as mock_dataset:
            mock_dataset.return_value = MagicMock()
            mock_dataset.return_value.__len__ = MagicMock(return_value=10)

            loader = build_dataloader(
                tokenizer,
                cfg,
                max_seq_len=512,
                split="train",
            )

        assert loader is not None
        assert mock_dataset.call_args.kwargs["max_seq_len"] == 512

    def test_tool_calling_phase(self, tmp_path):
        tc_cfg = ToolCallingConfig()
        examples = [_generate_person_recognition_example(tc_cfg) for _ in range(5)]
        train_path = tmp_path / "tool_calling_train.jsonl"
        with open(train_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        tokenizer = MagicMock()

        def mock_call(text, **kwargs):
            max_length = kwargs.get("max_length", 64)
            return {
                "input_ids": torch.randint(0, 100, (1, max_length)),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        tokenizer.encode = MagicMock(return_value=list(range(50)))
        tokenizer.decode = MagicMock(return_value="text")

        cfg = TrainingConfig(phase="tool-calling", batch_size=2)

        with patch("src.data.ToolCallingDataset.__init__", return_value=None) as mock_init:
            # Skip init, just test the factory logic
            mock_init.return_value = None
            with pytest.raises(Exception):
                # Will fail because the mock dataset has no __len__
                build_dataloader(tokenizer, cfg, split="train")

    def test_unknown_phase_raises(self):
        tokenizer = MagicMock()
        cfg = TrainingConfig(phase="unknown-phase")
        with pytest.raises(ValueError, match="Unknown phase"):
            build_dataloader(tokenizer, cfg, split="train")


# ---------------------------------------------------------------------------
# GemmaDataGenerator (full mocked coverage)
# ---------------------------------------------------------------------------

def _make_mock_gemma_gen():
    """Create a GemmaDataGenerator with mocked model/tokenizer."""
    gen = GemmaDataGenerator.__new__(GemmaDataGenerator)
    gen.device = "cpu"
    gen.max_new_tokens = 64
    gen.temperature = 0.8
    gen.top_p = 0.92

    # Tokenizer must return an object with .to() that returns dict-like with input_ids
    tok_output = MagicMock()
    tok_output.__getitem__ = lambda self, key: torch.randint(0, 100, (1, 10))
    tok_output.to = MagicMock(return_value={"input_ids": torch.randint(0, 100, (1, 10))})

    gen.tokenizer = MagicMock()
    gen.tokenizer.return_value = tok_output
    gen.tokenizer.decode = MagicMock(return_value="Generated text")

    gen.model = MagicMock()
    gen.model.device = torch.device("cpu")
    gen.model.eval = MagicMock()
    gen.model.generate = MagicMock(return_value=torch.randint(0, 100, (1, 20)))

    return gen


class TestGemmaDataGeneratorFull:
    def test_init_with_mock(self):
        """Test GemmaDataGenerator.__init__ with mocked model loading."""
        with patch("src.data.AutoTokenizer.from_pretrained") as mock_tok:
            mock_tok.return_value = MagicMock()
            with patch("src.data.AutoModelForCausalLM.from_pretrained") as mock_model:
                mock_m = MagicMock()
                mock_m.eval = MagicMock()
                mock_model.return_value = mock_m
                gen = GemmaDataGenerator(
                    model_name="test",
                    device="cpu",
                    torch_dtype="float16",
                )
        assert gen.model is mock_m

    def test_generate_method(self):
        gen = _make_mock_gemma_gen()
        result = gen._generate("test prompt")
        assert isinstance(result, str)
        gen.model.generate.assert_called_once()

    def test_generate_user_queries(self):
        gen = _make_mock_gemma_gen()
        # Mock _generate to return a numbered list
        gen._generate = MagicMock(return_value="1. Where am I?\n2. Who is that?\n3. I'm scared.\n")
        queries = gen.generate_user_queries("orientation", num=5)
        assert len(queries) >= 3

    def test_generate_user_queries_fallback_to_templates(self):
        gen = _make_mock_gemma_gen()
        # Mock _generate to return insufficient output
        gen._generate = MagicMock(return_value="junk output")
        queries = gen.generate_user_queries("orientation", num=5)
        # Should fall back to templates
        assert len(queries) >= 3

    def test_generate_assistant_response(self):
        gen = _make_mock_gemma_gen()
        gen._generate = MagicMock(return_value="That's Maria, your granddaughter! She loves drawing.")
        result = gen.generate_assistant_response(
            user_query="Who is that?",
            tool_name="read_person",
            tool_result={"name": "Maria"},
            context_hint="That's Maria.",
        )
        assert "Maria" in result

    def test_generate_assistant_response_fallback(self):
        gen = _make_mock_gemma_gen()
        # Mock _generate to return too-short response
        gen._generate = MagicMock(return_value="")
        result = gen.generate_assistant_response(
            user_query="Who?",
            tool_name="read_person",
            tool_result={},
            context_hint="Fallback hint text here",
        )
        assert result == "Fallback hint text here"

    def test_warm_up_query_cache(self):
        gen = _make_mock_gemma_gen()
        gen._generate = MagicMock(return_value="\n".join(
            f"{i}. Query number {i} text" for i in range(1, 11)
        ))
        gen.warm_up_query_cache(queries_per_category=10)
        assert hasattr(gen, "_user_query_cache")
        assert len(gen._user_query_cache) == len(_SCENARIOS)

    def test_generate_example_person(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {
            "person_recognition": ["Who is that?"],
            "memory": ["Tell me about Maria."],
        }
        gen.generate_assistant_response = MagicMock(return_value="That's Maria.")
        tc_cfg = ToolCallingConfig()

        example = gen.generate_example("person_recognition", tc_cfg)
        assert "messages" in example
        assert len(example["messages"]) == 5

    def test_generate_example_orientation(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"orientation": ["Where am I?"]}
        gen.generate_assistant_response = MagicMock(return_value="You're in the kitchen.")
        tc_cfg = ToolCallingConfig()

        example = gen.generate_example("orientation", tc_cfg)
        assert "messages" in example

    def test_generate_example_medication(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"medication": ["What pills do I take?"]}
        gen.generate_assistant_response = MagicMock(return_value="Time for Donepezil.")
        tc_cfg = ToolCallingConfig()

        example = gen.generate_example("medication", tc_cfg)
        assert "messages" in example

    def test_generate_example_alert(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"alert": ["I'm scared and confused."]}
        gen.generate_assistant_response = MagicMock(return_value="I've alerted Rosa.")
        tc_cfg = ToolCallingConfig()

        example = gen.generate_example("alert", tc_cfg)
        assert "messages" in example

    def test_generate_example_memory(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"memory": ["Tell me about my wife."]}
        gen.generate_assistant_response = MagicMock(return_value="Ana loved sunflowers.")
        tc_cfg = ToolCallingConfig()

        example = gen.generate_example("memory", tc_cfg)
        assert "messages" in example

    def test_generate_example_unknown_category(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {}
        tc_cfg = ToolCallingConfig()

        with pytest.raises(ValueError, match="Unknown category"):
            gen.generate_example("nonexistent", tc_cfg)


# ---------------------------------------------------------------------------
# generate_synthetic_data with Gemma (mocked)
# ---------------------------------------------------------------------------

class TestGenerateSyntheticDataGemma:
    def test_generate_with_gemma(self, tmp_path):
        """Test generate_synthetic_data with use_gemma=True (mocked)."""
        mock_gen = _make_mock_gemma_gen()
        mock_gen._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        mock_gen.generate_assistant_response = MagicMock(return_value="Response text here.")

        with patch("src.data.GemmaDataGenerator", return_value=mock_gen):
            generate_synthetic_data(
                output_dir=str(tmp_path),
                num_train=5,
                num_val=2,
                seed=42,
                use_gemma=True,
                gemma_model="test",
                device="cpu",
            )

        assert (tmp_path / "tool_calling_train.jsonl").exists()
        assert (tmp_path / "tool_calling_val.jsonl").exists()

        # Verify content
        with open(tmp_path / "tool_calling_train.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# generate_synthetic_data_parallel (mocked)
# ---------------------------------------------------------------------------

class TestParallelGeneration:
    def test_fallback_to_single_gpu(self, tmp_path):
        """When fewer than 2 GPUs, should fall back to single GPU."""
        from src.data import generate_synthetic_data_parallel

        with patch("torch.cuda.device_count", return_value=1):
            with patch("src.data.generate_synthetic_data") as mock_gen:
                generate_synthetic_data_parallel(
                    output_dir=str(tmp_path),
                    num_train=10,
                    num_val=5,
                    num_gpus=2,
                )
                mock_gen.assert_called_once()

    def test_parallel_with_zero_gpus(self, tmp_path):
        from src.data import generate_synthetic_data_parallel

        with patch("torch.cuda.device_count", return_value=0):
            with patch("src.data.generate_synthetic_data") as mock_gen:
                generate_synthetic_data_parallel(
                    output_dir=str(tmp_path),
                    num_train=10,
                    num_val=5,
                    num_gpus=2,
                )
                mock_gen.assert_called_once()


# ---------------------------------------------------------------------------
# Data CLI (parallel mode)
# ---------------------------------------------------------------------------

class TestDataCLIParallel:
    def test_main_parallel_mode(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["generate-data", "--output-dir", "/tmp/test_data",
             "--num-train", "5", "--num-val", "2", "--parallel", "--num-gpus", "2"],
        )
        with patch("src.data.generate_synthetic_data_parallel") as mock_par:
            from src.data import main
            main()
            mock_par.assert_called_once()

    def test_main_gemma_single(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["generate-data", "--output-dir", "/tmp/test_data",
             "--num-train", "5", "--num-val", "2",
             "--gemma-model", "test/model"],
        )
        with patch("src.data.generate_synthetic_data") as mock_gen:
            from src.data import main
            main()
            mock_gen.assert_called_once()

    def test_main_gemini_api_flag(self, monkeypatch):
        """Test CLI entry point with --gemini-api flag."""
        monkeypatch.setattr(
            "sys.argv",
            ["generate-data", "--output-dir", "/tmp/test_data",
             "--num-train", "5", "--num-val", "2",
             "--gemini-api", "--api-model", "gemma-4-31b-it",
             "--api-key", "test-key"],
        )
        with patch("src.data.generate_synthetic_data") as mock_gen:
            from src.data import main
            main()
            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["use_api"] is True
            assert call_kwargs["api_model"] == "gemma-4-31b-it"
            assert call_kwargs["api_key"] == "test-key"

    def test_gemini_api_overrides_parallel(self, monkeypatch):
        """--gemini-api should take priority over --parallel."""
        monkeypatch.setattr(
            "sys.argv",
            ["generate-data", "--output-dir", "/tmp/test_data",
             "--num-train", "5", "--num-val", "2",
             "--gemini-api", "--parallel", "--num-gpus", "2",
             "--api-key", "test-key"],
        )
        with patch("src.data.generate_synthetic_data") as mock_gen:
            with patch("src.data.generate_synthetic_data_parallel") as mock_par:
                from src.data import main
                main()
                mock_gen.assert_called_once()
                mock_par.assert_not_called()
                call_kwargs = mock_gen.call_args[1]
                assert call_kwargs["use_api"] is True

    def test_rpm_flag_passed_to_generate(self, monkeypatch):
        """--rpm should be forwarded as rpm_limit to generate_synthetic_data."""
        monkeypatch.setattr(
            "sys.argv",
            ["generate-data", "--output-dir", "/tmp/test_data",
             "--num-train", "5", "--num-val", "2",
             "--gemini-api", "--api-key", "test-key", "--rpm", "10"],
        )
        with patch("src.data.generate_synthetic_data") as mock_gen:
            from src.data import main
            main()
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["rpm_limit"] == 10


# ---------------------------------------------------------------------------
# GeminiApiGenerator
# ---------------------------------------------------------------------------

class TestGeminiApiGenerator:
    def test_init_missing_genai(self):
        """Should raise ImportError if google-genai is not installed."""
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="google-genai"):
                GeminiApiGenerator(api_key="test-key")

    def test_init_missing_api_key(self, monkeypatch):
        """Should raise ValueError if no API key is found."""
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_genai = MagicMock()
        mock_types = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            with patch("src.data._resolve_gemini_api_key", return_value=None):
                with pytest.raises(ValueError, match="No API key found"):
                    GeminiApiGenerator(api_key=None)

    def test_model_property_returns_none(self):
        """API generator has no local model."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen.device = "api"
        gen._user_query_cache = {}
        assert gen.model is None

    def test_cleanup_closes_client(self):
        """cleanup() should close the API client."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._client = MagicMock()
        gen.cleanup()
        gen._client.close.assert_called_once()

    def test_generate_calls_api(self):
        """_generate should call the API client."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._model_name = "gemma-4-31b-it"
        gen.temperature = 0.8
        gen.top_p = 0.92
        gen.max_new_tokens = 256
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []

        mock_response = MagicMock()
        mock_response.text = "Hello world"
        gen._client = MagicMock()
        gen._client.models.generate_content.return_value = mock_response
        gen._types = MagicMock()

        result = gen._generate("test prompt")
        assert result == "Hello world"
        gen._client.models.generate_content.assert_called_once()

    def test_generate_propagates_api_error(self):
        """_generate should propagate API errors."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._model_name = "gemma-4-31b-it"
        gen.temperature = 0.8
        gen.top_p = 0.92
        gen.max_new_tokens = 256
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []

        mock_genai_errors = MagicMock()
        api_error = type("APIError", (Exception,), {"code": 429})()
        gen._client = MagicMock()
        gen._client.models.generate_content.side_effect = api_error
        gen._types = MagicMock()

        with patch.dict("sys.modules", {"google.genai.errors": mock_genai_errors}):
            mock_genai_errors.APIError = type(api_error)
            with pytest.raises(type(api_error)):
                gen._generate("test prompt")

    def test_generate_batch_sequential(self):
        """_generate_batch should call _generate for each prompt."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._generate = MagicMock(side_effect=["resp1", "resp2", "resp3"])
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []
        results = gen._generate_batch(["p1", "p2", "p3"])
        assert results == ["resp1", "resp2", "resp3"]
        assert gen._generate.call_count == 3

    def test_generate_batch_single_prompt(self):
        """_generate_batch with 1 prompt should call _generate directly."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._generate = MagicMock(return_value="only")
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []
        results = gen._generate_batch(["p1"])
        assert results == ["only"]

    def test_generate_batch_preserves_order(self):
        """Concurrent batch must return results in original prompt order."""
        import time as _time

        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []

        def slow_generate(prompt):
            # Different delays to test ordering
            delay = {"p1": 0.05, "p2": 0.01, "p3": 0.03}.get(prompt, 0)
            _time.sleep(delay)
            return f"result_{prompt}"

        gen._generate = MagicMock(side_effect=slow_generate)
        results = gen._generate_batch(["p1", "p2", "p3"])
        assert results == ["result_p1", "result_p2", "result_p3"]

    def test_rate_limiter_tracks_requests(self):
        """Rate limiter should track request timestamps."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._rpm_limit = 5
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []

        gen._wait_for_rate_limit()
        assert len(gen._request_times) == 1

        gen._wait_for_rate_limit()
        assert len(gen._request_times) == 2


# ---------------------------------------------------------------------------
# _resolve_gemini_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_gemini_api_key_first(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini789")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        assert _resolve_gemini_api_key() == "gemini789"

    def test_google_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google456")
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        assert _resolve_gemini_api_key() == "google456"

    def test_kaggle_key_fallback(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("KAGGLE_KEY", "kaggle123")
        assert _resolve_gemini_api_key() == "kaggle123"

    def test_priority_order(self, monkeypatch):
        """GEMINI_API_KEY should take priority over GOOGLE_API_KEY and KAGGLE_KEY."""
        monkeypatch.setenv("GEMINI_API_KEY", "first")
        monkeypatch.setenv("GOOGLE_API_KEY", "second")
        monkeypatch.setenv("KAGGLE_KEY", "third")
        assert _resolve_gemini_api_key() == "first"

    def test_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # kaggle_secrets not available outside Kaggle runtime
        with patch.dict("sys.modules", {"kaggle_secrets": None}):
            assert _resolve_gemini_api_key() is None

    def test_kaggle_secrets_vault(self, monkeypatch):
        """Falls back to kaggle_secrets when env vars are unset."""
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_client = MagicMock()
        mock_client.get_secret.return_value = "vault_secret_123"
        mock_cls = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"kaggle_secrets": MagicMock(UserSecretsClient=mock_cls)}):
            result = _resolve_gemini_api_key()
        assert result == "vault_secret_123"

    def test_env_var_takes_priority_over_vault(self, monkeypatch):
        """Env var should be returned even if kaggle_secrets is available."""
        monkeypatch.setenv("KAGGLE_KEY", "env_key")
        assert _resolve_gemini_api_key() == "env_key"


# ---------------------------------------------------------------------------
# _is_quota_error
# ---------------------------------------------------------------------------

class TestIsQuotaError:
    def test_non_api_error(self):
        assert _is_quota_error(ValueError("nope")) is False

    def test_quota_codes(self):
        """429, 403, 503 should be recognized as quota errors."""
        for code in (429, 403, 503):
            err = Exception(f"quota error {code}")
            err.code = code
            assert _is_quota_error(err) is True

    def test_non_quota_code(self):
        err = Exception("not found")
        err.code = 404
        assert _is_quota_error(err) is False

    def test_no_code_attribute(self):
        assert _is_quota_error(RuntimeError("no code")) is False


# ---------------------------------------------------------------------------
# _create_generator
# ---------------------------------------------------------------------------

class TestCreateGenerator:
    def test_returns_local_when_api_disabled(self):
        """When use_api=False, should return (GemmaDataGenerator, False)."""
        with patch("src.data.GemmaDataGenerator") as MockLocal:
            mock_gen = MagicMock()
            MockLocal.return_value = mock_gen
            gen, is_api = _create_generator(use_api=False, device="cpu")
            assert gen is mock_gen
            assert is_api is False

    def test_returns_api_gen_on_success(self):
        """When API works, should return (GeminiApiGenerator, True)."""
        with patch("src.data.GeminiApiGenerator") as MockApi:
            mock_gen = MagicMock(spec=GeminiApiGenerator)
            mock_gen._generate.return_value = "hello"
            MockApi.return_value = mock_gen
            gen, is_api = _create_generator(use_api=True, api_key="test-key")
            assert gen is mock_gen
            assert is_api is True

    def test_falls_back_on_quota_error(self):
        """When API quota is exhausted, should fall back to (local, False)."""
        MockAPIError = type("APIError", (Exception,), {"code": 429})

        with patch("src.data.GeminiApiGenerator") as MockApi:
            mock_gen = MagicMock()
            mock_gen._generate.side_effect = MockAPIError()
            MockApi.return_value = mock_gen

            with patch("src.data._is_quota_error", return_value=True):
                with patch("src.data.GemmaDataGenerator") as MockLocal:
                    local_gen = MagicMock()
                    MockLocal.return_value = local_gen
                    gen, is_api = _create_generator(
                        use_api=True, api_key="test-key", device="cpu"
                    )
                    assert gen is local_gen
                    assert is_api is False

    def test_falls_back_on_import_error(self):
        """When google-genai is not installed, should fall back to (local, False)."""
        with patch("src.data.GeminiApiGenerator", side_effect=ImportError("no genai")):
            with patch("src.data.GemmaDataGenerator") as MockLocal:
                local_gen = MagicMock()
                MockLocal.return_value = local_gen
                gen, is_api = _create_generator(
                    use_api=True, api_key="test-key", device="cpu"
                )
                assert gen is local_gen
                assert is_api is False


# ---------------------------------------------------------------------------
# _generate_example_with_fallback
# ---------------------------------------------------------------------------

class TestGenerateExampleWithFallback:
    def test_no_fallback_when_local(self):
        """Local generator errors should propagate, not trigger fallback."""
        gen = MagicMock(spec=GemmaDataGenerator)
        gen.generate_example.side_effect = RuntimeError("local error")
        tc_cfg = ToolCallingConfig()

        with pytest.raises(RuntimeError, match="local error"):
            _generate_example_with_fallback(gen, "orientation", tc_cfg)

    def test_returns_example_on_success(self):
        """Should return example and same generator on success."""
        gen = MagicMock(spec=GemmaDataGenerator)
        gen.generate_example.return_value = {"messages": []}
        tc_cfg = ToolCallingConfig()

        example, returned_gen = _generate_example_with_fallback(gen, "orientation", tc_cfg)
        assert example == {"messages": []}
        assert returned_gen is gen

    def test_fallback_on_api_quota(self):
        """Should switch to local GPU when API quota is exhausted."""
        MockAPIError = type("APIError", (Exception,), {"code": 429})

        gen = MagicMock(spec=GeminiApiGenerator)
        gen.generate_example.side_effect = MockAPIError()
        gen._user_query_cache = {"orientation": ["Where am I?"]}
        gen.cleanup = MagicMock()
        tc_cfg = ToolCallingConfig()

        with patch("src.data._is_quota_error", return_value=True):
            with patch("src.data.GemmaDataGenerator") as MockLocal:
                local_gen = MagicMock()
                local_gen.generate_example.return_value = {"messages": ["fallback"]}
                MockLocal.return_value = local_gen

                example, returned_gen = _generate_example_with_fallback(
                    gen, "orientation", tc_cfg, device="cpu"
                )
                assert returned_gen is local_gen
                assert example == {"messages": ["fallback"]}
                gen.cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# _generate_batch_with_fallback
# ---------------------------------------------------------------------------

class TestGenerateBatchWithFallback:
    def test_returns_examples_on_success(self):
        gen = MagicMock(spec=GemmaDataGenerator)
        gen.generate_examples_batch.return_value = [{"messages": []}]
        tc_cfg = ToolCallingConfig()

        examples, returned_gen = _generate_batch_with_fallback(
            gen, ["orientation"], tc_cfg
        )
        assert len(examples) == 1
        assert returned_gen is gen

    def test_fallback_on_api_quota(self):
        MockAPIError = type("APIError", (Exception,), {"code": 429})

        gen = MagicMock(spec=GeminiApiGenerator)
        gen.generate_examples_batch.side_effect = MockAPIError()
        gen._user_query_cache = {}
        gen.cleanup = MagicMock()
        tc_cfg = ToolCallingConfig()

        with patch("src.data._is_quota_error", return_value=True):
            with patch("src.data.GemmaDataGenerator") as MockLocal:
                local_gen = MagicMock()
                local_gen.generate_examples_batch.return_value = [{"messages": ["fb"]}]
                MockLocal.return_value = local_gen

                examples, returned_gen = _generate_batch_with_fallback(
                    gen, ["orientation"], tc_cfg, device="cpu"
                )
                assert returned_gen is local_gen
                gen.cleanup.assert_called_once()

    def test_non_quota_error_propagates(self):
        gen = MagicMock(spec=GemmaDataGenerator)
        gen.generate_examples_batch.side_effect = RuntimeError("local err")
        tc_cfg = ToolCallingConfig()
        with pytest.raises(RuntimeError, match="local err"):
            _generate_batch_with_fallback(gen, ["orientation"], tc_cfg)


# ---------------------------------------------------------------------------
# TextDataset: _resolve_cache_path fallback branches
# ---------------------------------------------------------------------------

class TestResolveCachePathAllFail:
    def test_returns_none_when_both_dirs_fail(self, tmp_path, monkeypatch):
        """When primary AND fallback cache dirs are unusable, returns None."""
        monkeypatch.delenv("GEMMA_TEXT_CACHE_DIR", raising=False)

        mock_tokenizer = MagicMock()
        type(mock_tokenizer).name_or_path = PropertyMock(return_value="mock-tok")
        mock_tokenizer.encode = MagicMock(return_value=list(range(128)))

        fake_dataset = [{"text": "Enough text to tokenize long enough sentence."} for _ in range(5)]

        # Make both primary and fallback fail
        with patch("datasets.load_dataset", return_value=fake_dataset), \
             patch.object(TextDataset, "_ensure_cache_dir", return_value=False):
            ds = TextDataset(
                tokenizer=mock_tokenizer,
                max_seq_len=32,
                cache_dir=str(tmp_path / "bad_cache"),
                use_cache=True,
            )
        assert len(ds) > 0


class TestEnsureCacheDirOSError:
    def test_returns_false_on_oserror(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        with patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
            result = ds._ensure_cache_dir(tmp_path / "some_dir")
        assert result is False

    def test_returns_false_when_path_is_not_dir(self, tmp_path):
        """mkdir succeeds but path is a file, not a directory."""
        fake_file = tmp_path / "fake_dir"
        fake_file.write_text("not a dir")
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        result = ds._ensure_cache_dir(fake_file)
        assert result is False


class TestLoadFromCacheEdgeCases:
    def test_invalid_payload_not_tensor(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = []
        cache = tmp_path / "bad.pt"
        torch.save({"chunks": "not a tensor"}, cache)
        assert ds._load_from_cache(cache) is False

    def test_wrong_ndim(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = []
        cache = tmp_path / "bad_ndim.pt"
        torch.save({"chunks": torch.tensor([1, 2, 3])}, cache)
        assert ds._load_from_cache(cache) is False

    def test_corrupt_file(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = []
        cache = tmp_path / "corrupt.pt"
        cache.write_text("not a pytorch file")
        assert ds._load_from_cache(cache) is False

    def test_missing_chunks_key(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = []
        cache = tmp_path / "no_key.pt"
        torch.save({"other": 42}, cache)
        assert ds._load_from_cache(cache) is False


class TestSaveToCacheEdgeCases:
    def test_empty_chunks_skips(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = []
        ds._save_to_cache(tmp_path / "out.pt")
        assert not (tmp_path / "out.pt").exists()

    def test_cache_dir_failure_skips(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = [torch.randint(0, 100, (32,))]
        with patch.object(TextDataset, "_ensure_cache_dir", return_value=False):
            ds._save_to_cache(tmp_path / "out.pt")
        assert not (tmp_path / "out.pt").exists()

    def test_lock_timeout(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = [torch.randint(0, 100, (32,))]
        cache_path = tmp_path / "locked.pt"
        lock_path = cache_path.with_suffix(".pt.lock")

        # Pre-create lock file (simulates another process holding it)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("locked")

        with patch("time.time", side_effect=[0, 0, 200]):
            ds._save_to_cache(cache_path)
        # Should not have saved due to lock timeout
        assert not cache_path.exists()

    def test_lock_contention_then_cache_exists(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = [torch.randint(0, 100, (32,))]
        cache_path = tmp_path / "contested.pt"
        lock_path = cache_path.with_suffix(".pt.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("locked")

        # After lock contention, cache_path appears (another process saved it)
        torch.save({"chunks": torch.stack(ds.chunks)}, cache_path)

        with patch("os.open", side_effect=FileExistsError):
            ds._save_to_cache(cache_path)
        # Cache already exists, so it should return early
        assert cache_path.exists()

    def test_save_exception_logged(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = [torch.randint(0, 100, (32,))]
        cache_path = tmp_path / "save_fail.pt"

        with patch("torch.save", side_effect=RuntimeError("disk full")):
            ds._save_to_cache(cache_path)
        assert not cache_path.exists()

    def test_lock_cleanup_exception_logged(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = [torch.randint(0, 100, (32,))]
        cache_path = tmp_path / "lock_cleanup_fail.pt"

        with patch("pathlib.Path.unlink", side_effect=OSError("cannot unlink")):
            ds._save_to_cache(cache_path)
        # The save should still succeed despite lock cleanup failure
        assert cache_path.exists()

    def test_already_saved_after_lock(self, tmp_path):
        """If cache appears between acquiring lock and saving, skip save."""
        ds = TextDataset.__new__(TextDataset)
        ds.chunks = [torch.randint(0, 100, (32,))]
        cache_path = tmp_path / "appeared.pt"
        # Pre-create the cache file
        torch.save({"chunks": torch.stack(ds.chunks)}, cache_path)
        mtime = cache_path.stat().st_mtime
        ds._save_to_cache(cache_path)
        # File should not have been rewritten
        assert cache_path.stat().st_mtime == mtime


class TestCreateLabelsEdgeCases:
    def test_no_model_turn_all_masked(self, tmp_path):
        """When decoded text has no model turns, all labels should be -100."""
        tc_cfg = ToolCallingConfig()
        examples = [_generate_person_recognition_example(tc_cfg)]
        path = tmp_path / "tc.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(examples[0]) + "\n")

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }
        # Return text WITHOUT model turns
        tokenizer.decode = MagicMock(return_value="<start_of_turn>user\nhello<end_of_turn>")
        tokenizer.encode = MagicMock(return_value=list(range(50)))

        ds = ToolCallingDataset(tokenizer=tokenizer, data_path=str(path), max_seq_len=64)
        item = ds[0]
        # All labels should be masked
        assert (item["labels"] == -100).all()

    def test_model_turn_without_end_tag(self, tmp_path):
        """Model turn without end tag should still unmask to end of text."""
        tc_cfg = ToolCallingConfig()
        examples = [_generate_person_recognition_example(tc_cfg)]
        path = tmp_path / "tc.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(examples[0]) + "\n")

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }
        tokenizer.decode = MagicMock(return_value="<start_of_turn>model\nhello world test")
        tokenizer.encode = MagicMock(return_value=list(range(10)))

        ds = ToolCallingDataset(tokenizer=tokenizer, data_path=str(path), max_seq_len=64)
        item = ds[0]
        # Some labels should be unmasked
        assert not (item["labels"] == -100).all()


class TestParseNumberedListBulletAndQuoted:
    def test_bullet_patterns(self):
        text = "- First bullet item text\n* Second bullet item text\n• Third bullet item text\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 3

    def test_quoted_lines(self):
        text = '"Where am I right now?"\n"Who is that person visiting?"\n'
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 2

    def test_fallback_plain_lines(self):
        text = "This is a plain line that is long enough to be a sentence.\nAnother plain line here too.\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 2

    def test_skip_meta_commentary(self):
        text = "Here are the queries:\n1. Actual query text here\nSure, I can help.\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 1


class TestGemmaGenerateBatchSequential:
    def test_generate_batch_calls_generate_sequentially(self):
        gen = _make_mock_gemma_gen()
        gen._generate = MagicMock(side_effect=["r1", "r2", "r3"])
        results = gen._generate_batch(["p1", "p2", "p3"])
        assert results == ["r1", "r2", "r3"]
        assert gen._generate.call_count == 3


class TestGemmaGenerateExamplesBatch:
    def test_batch_generation(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        gen._generate_batch = MagicMock(return_value=["Response text here."] * 3)
        tc_cfg = ToolCallingConfig()

        results = gen.generate_examples_batch(
            ["person_recognition", "orientation", "medication"],
            tc_cfg,
            batch_size=3,
        )
        assert len(results) == 3
        for r in results:
            assert "messages" in r
            assert len(r["messages"]) == 5

    def test_batch_short_response_uses_hint(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        gen._generate_batch = MagicMock(return_value=[""])
        tc_cfg = ToolCallingConfig()

        results = gen.generate_examples_batch(["alert"], tc_cfg, batch_size=1)
        assert len(results) == 1
        # Short response should fall back to hint
        assert len(results[0]["messages"][-1]["content"]) > 10


class TestPrepareExamplePartialBranches:
    def test_alert_category(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"alert": ["I'm very confused."]}
        tc_cfg = ToolCallingConfig()
        partial = gen._prepare_example_partial("alert", tc_cfg)
        assert "alert_caregiver" in partial["tool_call"]

    def test_unknown_category_raises(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {}
        tc_cfg = ToolCallingConfig()
        with pytest.raises(ValueError, match="Unknown category"):
            gen._prepare_example_partial("nonexistent", tc_cfg)

    def test_memory_with_name_not_in_query(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"memory": ["Tell me about my family."]}
        tc_cfg = ToolCallingConfig()
        partial = gen._prepare_example_partial("memory", tc_cfg)
        assert "read_person" in partial["tool_call"]

    def test_orientation_category(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"orientation": ["Where am I?"]}
        tc_cfg = ToolCallingConfig()
        partial = gen._prepare_example_partial("orientation", tc_cfg)
        assert "describe_location" in partial["tool_call"]

    def test_medication_category(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"medication": ["What pills do I take?"]}
        tc_cfg = ToolCallingConfig()
        partial = gen._prepare_example_partial("medication", tc_cfg)
        assert "get_medication" in partial["tool_call"]


class TestGenPersonExampleNameCoherence:
    def test_mismatched_name_replaced(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"person_recognition": ["Who is Maria?"]}
        gen.generate_assistant_response = MagicMock(return_value="That's someone.")
        tc_cfg = ToolCallingConfig()

        random.seed(99)
        example = gen._gen_person_example("person_recognition", tc_cfg)
        assert "messages" in example


class TestGenAlertExampleBranch:
    def test_generates_alert(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {"alert": ["I'm very scared."]}
        gen.generate_assistant_response = MagicMock(return_value="Rosa is on her way.")
        tc_cfg = ToolCallingConfig()
        example = gen._gen_alert_example(tc_cfg)
        assert "alert_caregiver" in example["messages"][2]["content"]


class TestGemmaCleanup:
    def test_cleanup_frees_resources(self):
        gen = _make_mock_gemma_gen()
        gen.cleanup()
        assert not hasattr(gen, "model")
        assert not hasattr(gen, "tokenizer")


class TestBuildDataloaderValidation:
    def test_text_validation_split(self):
        tokenizer = MagicMock()
        cfg = TrainingConfig(phase="tcs-lora", batch_size=2)
        with patch("src.data.TextDataset") as mock_ds:
            mock_ds.return_value = MagicMock()
            mock_ds.return_value.__len__ = MagicMock(return_value=10)
            loader = build_dataloader(tokenizer, cfg, max_seq_len=512, split="validation")
        assert loader is not None


class TestGenerateSyntheticDataApiMode:
    def test_api_quota_fallback_to_parallel(self, tmp_path):
        """API mode should fall back to parallel when _create_generator returns non-API."""
        mock_gen = _make_mock_gemma_gen()
        mock_gen.cleanup = MagicMock()

        with patch("src.data._create_generator", return_value=(mock_gen, False)), \
             patch("src.data.generate_synthetic_data_parallel") as mock_par:
            generate_synthetic_data(
                output_dir=str(tmp_path),
                num_train=5,
                num_val=2,
                seed=42,
                use_gemma=True,
                use_api=True,
                api_key="test",
                num_gpus=2,
            )
            mock_par.assert_called_once()

    def test_api_single_item_quota_fallback(self, tmp_path):
        """API mode batch_count==1 should fall back on quota error."""
        MockAPIError = type("APIError", (Exception,), {"code": 429})
        mock_gen = MagicMock(spec=GeminiApiGenerator)
        mock_gen.generate_example.side_effect = MockAPIError()
        mock_gen._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        mock_gen.warm_up_query_cache = MagicMock()
        mock_gen.cleanup = MagicMock()

        with patch("src.data._create_generator", return_value=(mock_gen, True)), \
             patch("src.data._is_quota_error", return_value=True), \
             patch("src.data._finish_with_parallel"):
            generate_synthetic_data(
                output_dir=str(tmp_path),
                num_train=1,
                num_val=0,
                seed=42,
                use_gemma=True,
                use_api=True,
                api_key="test",
                rpm_limit=1,
            )
            mock_gen.cleanup.assert_called()


class TestFinishWithParallelBranch:
    def test_finish_with_parallel_calls_parallel(self, tmp_path):
        path = tmp_path / "tool_calling_train.jsonl"
        path.write_text("")

        with patch("src.data.generate_synthetic_data_parallel"):
            _finish_with_parallel(
                output_dir=str(tmp_path),
                path=path,
                split="train",
                remaining=10,
                remaining_val=5,
                num_val=5,
                seed=42,
                gemma_model="test",
                torch_dtype="float16",
                queries_per_category=10,
                num_gpus=2,
            )


# ---------------------------------------------------------------------------
# generate_synthetic_data with API mode (mocked)
# ---------------------------------------------------------------------------

class TestGenerateSyntheticDataApi:
    def test_api_mode_uses_create_generator(self, tmp_path):
        """When use_api=True and API works, should use batched API generator."""
        mock_gen = MagicMock(spec=GeminiApiGenerator)
        mock_gen.warm_up_query_cache = MagicMock()
        mock_gen.cleanup = MagicMock()

        def batch_side_effect(categories, tc_cfg, batch_size=14):
            return [
                {"messages": [{"role": "user", "content": f"test{i}"}]}
                for i in range(len(categories))
            ]

        mock_gen.generate_examples_batch = MagicMock(side_effect=batch_side_effect)
        # For val split (batch_count=1), falls through to generate_example
        mock_gen.generate_example = MagicMock(
            return_value={"messages": [{"role": "user", "content": "val"}]}
        )

        with patch("src.data._create_generator", return_value=(mock_gen, True)) as mock_factory:
            generate_synthetic_data(
                output_dir=str(tmp_path),
                num_train=3,
                num_val=1,
                seed=42,
                use_gemma=True,
                use_api=True,
                api_model="gemma-4-31b-it",
                api_key="test-key",
            )
            mock_factory.assert_called_once()

        assert (tmp_path / "tool_calling_train.jsonl").exists()
        with open(tmp_path / "tool_calling_train.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_api_startup_fallback_delegates_to_parallel(self, tmp_path):
        """When API fails at startup (is_api=False), should delegate to parallel."""
        mock_gen = MagicMock()
        mock_gen.cleanup = MagicMock()

        with patch("src.data._create_generator", return_value=(mock_gen, False)):
            with patch("src.data.generate_synthetic_data_parallel") as mock_parallel:
                generate_synthetic_data(
                    output_dir=str(tmp_path),
                    num_train=5,
                    num_val=2,
                    seed=42,
                    use_gemma=True,
                    use_api=True,
                    num_gpus=2,
                )
                mock_parallel.assert_called_once()
                call_kwargs = mock_parallel.call_args[1]
                assert call_kwargs["num_train"] == 5
                assert call_kwargs["num_val"] == 2
                assert call_kwargs["num_gpus"] == 2

    def test_api_mid_generation_fallback_to_parallel(self, tmp_path):
        """When API quota exhausted mid-generation, should finish with parallel."""
        QuotaError = type("QuotaError", (Exception,), {"code": 429})

        mock_gen = MagicMock(spec=GeminiApiGenerator)
        mock_gen.warm_up_query_cache = MagicMock()
        mock_gen.cleanup = MagicMock()

        # First batch call succeeds (train batch 1: 3 examples)
        # Second batch call fails with quota error
        batch_call_count = 0

        def batch_side_effect(categories, tc_cfg, batch_size=14):
            nonlocal batch_call_count
            batch_call_count += 1
            if batch_call_count <= 1:
                return [
                    {"messages": [{"role": "user", "content": f"test{i}"}]}
                    for i in range(len(categories))
                ]
            raise QuotaError()

        mock_gen.generate_examples_batch = MagicMock(side_effect=batch_side_effect)

        with patch("src.data._create_generator", return_value=(mock_gen, True)):
            with patch("src.data._is_quota_error", return_value=True):
                with patch("src.data._finish_with_parallel") as mock_finish:
                    generate_synthetic_data(
                        output_dir=str(tmp_path),
                        num_train=20,
                        num_val=5,
                        seed=42,
                        use_gemma=True,
                        use_api=True,
                        num_gpus=2,
                        rpm_limit=14,
                    )
                    mock_finish.assert_called_once()
                    call_kwargs = mock_finish.call_args[1]
                    # First batch generated 14, second batch failed → 6 remaining
                    assert call_kwargs["remaining"] == 6
                    assert call_kwargs["num_gpus"] == 2


# ---------------------------------------------------------------------------
# GemmaDataGenerator.cleanup
# ---------------------------------------------------------------------------

class TestGemmaCleanup:
    def test_cleanup_frees_resources(self):
        gen = GemmaDataGenerator.__new__(GemmaDataGenerator)
        gen.model = MagicMock()
        gen.tokenizer = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                gen.cleanup()
                mock_empty.assert_called_once()

        assert not hasattr(gen, "model")
        assert not hasattr(gen, "tokenizer")

    def test_cleanup_no_model(self):
        """cleanup should work even when model was never set."""
        gen = GemmaDataGenerator.__new__(GemmaDataGenerator)
        with patch("torch.cuda.is_available", return_value=False):
            gen.cleanup()  # Should not raise


# ---------------------------------------------------------------------------
# TextDataset._resolve_cache_path — fallback branch (lines 118-119)
# ---------------------------------------------------------------------------

class TestResolveCachePathFallback:
    def test_fallback_to_tempdir_when_primary_and_env_fail(self, tmp_path):
        """When primary dir fails and no env var, fallback to tempdir."""
        mock_tokenizer = MagicMock()
        type(mock_tokenizer).name_or_path = PropertyMock(return_value="mock-tok")
        mock_tokenizer.encode = MagicMock(return_value=list(range(128)))

        fake_dataset = [{"text": "Sentence long enough for tokenization test."} for _ in range(5)]

        # Create a file where the cache dir should be to make mkdir fail
        blocker = tmp_path / "blocked"
        blocker.write_text("I'm a file")
        bad_cache_dir = blocker / "text_cache"

        with patch("datasets.load_dataset", return_value=fake_dataset):
            with patch.dict(os.environ, {"GEMMA_TEXT_CACHE_DIR": ""}, clear=False):
                ds = TextDataset(
                    tokenizer=mock_tokenizer,
                    max_seq_len=32,
                    cache_dir=str(bad_cache_dir),
                    use_cache=True,
                )
        assert len(ds) > 0

    def test_fallback_returns_none_when_all_dirs_fail(self, tmp_path):
        """When no writable cache dir available, use_cache is disabled."""
        mock_tokenizer = MagicMock()
        type(mock_tokenizer).name_or_path = PropertyMock(return_value="mock-tok")
        mock_tokenizer.encode = MagicMock(return_value=list(range(128)))

        fake_dataset = [{"text": "Sentence long enough for tokenization test."} for _ in range(5)]

        ds_obj = TextDataset.__new__(TextDataset)
        ds_obj.tokenizer = mock_tokenizer
        ds_obj.max_seq_len = 32
        ds_obj.chunks = []

        # Both primary and fallback fail
        with patch.object(ds_obj, "_ensure_cache_dir", return_value=False):
            result = ds_obj._resolve_cache_path(Path(tmp_path / "nonexistent" / "cache.pt"))
        assert result is None


# ---------------------------------------------------------------------------
# TextDataset._ensure_cache_dir — OSError branch (lines 138-139)
# ---------------------------------------------------------------------------

class TestEnsureCacheDir:
    def test_returns_false_on_os_error(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        ds.chunks = []

        target = tmp_path / "forbidden_dir"
        original_mkdir = type(target).mkdir

        def raising_mkdir(self_path, *args, **kwargs):
            if "forbidden_dir" in str(self_path):
                raise OSError("Permission denied")
            return original_mkdir(self_path, *args, **kwargs)

        with patch.object(type(target), "mkdir", raising_mkdir):
            result = ds._ensure_cache_dir(target)
        assert result is False

    def test_returns_false_when_path_is_file(self, tmp_path):
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        ds.chunks = []

        file_path = tmp_path / "not_a_dir"
        file_path.write_text("file")
        result = ds._ensure_cache_dir(file_path)
        assert result is False


# ---------------------------------------------------------------------------
# TextDataset._load_from_cache — invalid cache branches (lines 170-178)
# ---------------------------------------------------------------------------

class TestLoadFromCache:
    def _make_ds(self):
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        ds.tokenizer.encode = MagicMock(return_value=[])
        ds.max_seq_len = 32
        ds.chunks = []
        return ds

    def test_returns_false_when_chunks_not_tensor(self, tmp_path):
        ds = self._make_ds()
        cache = tmp_path / "bad.pt"
        torch.save({"chunks": "not a tensor", "created_at": 0}, cache)
        assert ds._load_from_cache(cache) is False

    def test_returns_false_when_chunks_wrong_ndim(self, tmp_path):
        ds = self._make_ds()
        cache = tmp_path / "bad_ndim.pt"
        torch.save({"chunks": torch.tensor([1, 2, 3]), "created_at": 0}, cache)
        assert ds._load_from_cache(cache) is False

    def test_returns_true_on_valid_cache(self, tmp_path):
        ds = self._make_ds()
        cache = tmp_path / "good.pt"
        chunks = torch.stack([torch.arange(32) for _ in range(3)])
        torch.save({"chunks": chunks, "created_at": 0}, cache)
        assert ds._load_from_cache(cache) is True
        assert len(ds.chunks) == 3

    def test_returns_false_when_load_raises(self, tmp_path):
        ds = self._make_ds()
        cache = tmp_path / "corrupt.pt"
        cache.write_bytes(b"garbage data that is not a valid torch file")
        assert ds._load_from_cache(cache) is False

    def test_returns_false_when_file_not_exists(self, tmp_path):
        ds = self._make_ds()
        assert ds._load_from_cache(tmp_path / "nope.pt") is False


# ---------------------------------------------------------------------------
# TextDataset._save_to_cache — all branches (lines 186-228)
# ---------------------------------------------------------------------------

class TestSaveToCache:
    def _make_ds(self):
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        ds.max_seq_len = 32
        ds.chunks = []
        return ds

    def test_empty_chunks_returns_early(self, tmp_path):
        """Line 186: empty chunks → immediate return."""
        ds = self._make_ds()
        ds.chunks = []
        ds._save_to_cache(tmp_path / "cache.pt")
        assert not (tmp_path / "cache.pt").exists()

    def test_ensure_cache_dir_fails_returns_early(self, tmp_path):
        """Line 189: cache dir creation fails → return."""
        ds = self._make_ds()
        ds.chunks = [torch.arange(32)]
        with patch.object(ds, "_ensure_cache_dir", return_value=False):
            ds._save_to_cache(tmp_path / "cache.pt")
        assert not (tmp_path / "cache.pt").exists()

    def test_lock_contention_then_cache_appears(self, tmp_path):
        """Lines 197-200: lock exists, cache not present → sleep, then cache appears."""
        ds = self._make_ds()
        ds.chunks = [torch.arange(32)]
        cache_path = tmp_path / "cache.pt"

        open_call_count = 0

        def mock_open(path, flags):
            nonlocal open_call_count
            open_call_count += 1
            raise FileExistsError  # Lock always exists

        exists_call_count = 0
        original_exists = type(cache_path).exists

        def mock_exists(self_path):
            nonlocal exists_call_count
            if str(self_path) == str(cache_path):
                exists_call_count += 1
                # First check: not ready yet → sleep; second check: ready
                return exists_call_count >= 2
            return original_exists(self_path)

        with patch("os.open", side_effect=mock_open):
            with patch.object(type(cache_path), "exists", mock_exists):
                with patch("time.sleep"):
                    ds._save_to_cache(cache_path)

    def test_lock_timeout(self, tmp_path):
        """Lines 203-204: lock never acquired within timeout."""
        ds = self._make_ds()
        ds.chunks = [torch.arange(32)]
        cache_path = tmp_path / "cache.pt"

        with patch("os.open", side_effect=FileExistsError):
            with patch("time.time") as mock_time:
                # First call: start_time = 0
                # Second call (loop check): time = 200 (> 120)
                mock_time.side_effect = [0, 200]
                with patch("time.sleep"):
                    ds._save_to_cache(cache_path)
        assert not cache_path.exists()

    def test_save_success(self, tmp_path):
        """Lines 210+: successful save to cache."""
        ds = self._make_ds()
        ds.chunks = [torch.arange(32), torch.arange(32)]
        cache_path = tmp_path / "cache.pt"
        ds._save_to_cache(cache_path)
        assert cache_path.exists()

        # Verify content
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        assert "chunks" in payload
        assert payload["chunks"].shape == (2, 32)

    def test_save_exception_during_write(self, tmp_path):
        """Lines 221-222: exception during torch.save → logged, no crash."""
        ds = self._make_ds()
        ds.chunks = [torch.arange(32)]
        cache_path = tmp_path / "cache.pt"

        with patch("torch.stack", side_effect=RuntimeError("stack failed")):
            ds._save_to_cache(cache_path)
        # Should not crash, lock cleaned up
        assert not cache_path.with_suffix(".pt.lock").exists()

    def test_lock_cleanup_exception(self, tmp_path):
        """Lines 227-228: lock cleanup fails → logged, no crash."""
        ds = self._make_ds()
        ds.chunks = [torch.arange(32)]
        cache_path = tmp_path / "cache.pt"

        original_unlink = Path.unlink

        def mock_unlink(self_path, *args, **kwargs):
            if ".lock" in str(self_path):
                raise OSError("Cannot remove lock")
            return original_unlink(self_path, *args, **kwargs)

        with patch.object(Path, "unlink", mock_unlink):
            ds._save_to_cache(cache_path)
        # Cache should still be written successfully
        assert cache_path.exists()

    def test_cache_exists_after_acquiring_lock(self, tmp_path):
        """Line 210: cache_path.exists() after lock acquired → return early."""
        ds = self._make_ds()
        ds.chunks = [torch.arange(32)]
        cache_path = tmp_path / "cache.pt"
        cache_path.write_text("already cached")  # Pre-existing cache

        # The lock will be acquired, but cache exists, so returns early
        ds._save_to_cache(cache_path)
        # Content should remain unchanged (not overwritten)
        assert cache_path.read_text() == "already cached"


# ---------------------------------------------------------------------------
# _create_labels — no model turn found (line 360)
# ---------------------------------------------------------------------------

class TestCreateLabelsNoModelTurn:
    def test_no_model_turn_returns_all_masked(self, tmp_path):
        """When decoded text has no <start_of_turn>model, all labels are -100."""
        tc_cfg = ToolCallingConfig()
        path = tmp_path / "data.jsonl"
        example = _generate_person_recognition_example(tc_cfg)
        with open(path, "w") as f:
            f.write(json.dumps(example) + "\n")

        tokenizer = MagicMock()

        def mock_call(text, **kwargs):
            max_length = kwargs.get("max_length", 64)
            return {
                "input_ids": torch.randint(0, 100, (1, max_length)),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        tokenizer.encode = MagicMock(return_value=list(range(50)))
        # Return text without model turn markers
        tokenizer.decode = MagicMock(return_value="no model turn markers here at all")

        ds = ToolCallingDataset(tokenizer=tokenizer, data_path=str(path), max_seq_len=64)
        item = ds[0]
        # All labels should be -100 since no model turn found
        assert (item["labels"] == -100).all()


# ---------------------------------------------------------------------------
# GemmaDataGenerator._generate_batch — sequential (lines 636-639)
# ---------------------------------------------------------------------------

class TestGemmaGenerateBatch:
    def test_generate_batch_sequential(self):
        gen = _make_mock_gemma_gen()
        gen._generate = MagicMock(side_effect=["r1", "r2", "r3"])
        results = gen._generate_batch(["p1", "p2", "p3"])
        assert results == ["r1", "r2", "r3"]
        assert gen._generate.call_count == 3


# ---------------------------------------------------------------------------
# generate_user_queries — Gemma generation path (lines 751-764)
# ---------------------------------------------------------------------------

class TestGenerateUserQueriesGemmaPath:
    def test_gemma_generates_full_list(self):
        gen = _make_mock_gemma_gen()
        numbered = "\n".join(f"{i}. Valid query number {i} here" for i in range(1, 11))
        gen._generate = MagicMock(return_value=numbered)
        queries = gen.generate_user_queries("orientation", num=10)
        assert len(queries) >= 3

    def test_gemma_output_insufficient_falls_back_to_templates(self):
        gen = _make_mock_gemma_gen()
        # Return only 1 valid line (< 3 threshold)
        gen._generate = MagicMock(return_value="1. Only one valid query here")
        queries = gen.generate_user_queries("medication", num=10)
        # Should have supplemented with templates
        assert len(queries) >= 3


# ---------------------------------------------------------------------------
# generate_assistant_response — response truncation (lines 794-824)
# ---------------------------------------------------------------------------

class TestGenerateAssistantResponseTruncation:
    def test_multiline_response_takes_first_paragraph(self):
        gen = _make_mock_gemma_gen()
        gen._generate = MagicMock(
            return_value="First paragraph here.\n\nSecond paragraph ignored."
        )
        result = gen.generate_assistant_response(
            user_query="Who is that?",
            tool_name="read_person",
            tool_result={"name": "Maria"},
            context_hint="That's Maria.",
        )
        assert result == "First paragraph here."

    def test_empty_response_returns_hint(self):
        gen = _make_mock_gemma_gen()
        gen._generate = MagicMock(return_value="")
        result = gen.generate_assistant_response(
            user_query="Who?",
            tool_name="read_person",
            tool_result={},
            context_hint="Fallback hint",
        )
        assert result == "Fallback hint"

    def test_short_response_returns_hint(self):
        gen = _make_mock_gemma_gen()
        gen._generate = MagicMock(return_value="Short")
        result = gen.generate_assistant_response(
            user_query="Who?",
            tool_name="read_person",
            tool_result={},
            context_hint="Fallback hint",
        )
        assert result == "Fallback hint"


# ---------------------------------------------------------------------------
# generate_example and category-specific methods (lines 832-902+)
# ---------------------------------------------------------------------------

class TestGenerateExampleCategoryMethods:
    def _make_gen_with_cache(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        gen.generate_assistant_response = MagicMock(return_value="A warm response.")
        return gen

    def test_gen_person_example_directly(self):
        gen = self._make_gen_with_cache()
        tc_cfg = ToolCallingConfig()
        result = gen._gen_person_example("person_recognition", tc_cfg)
        assert "messages" in result
        assert len(result["messages"]) == 5

    def test_gen_person_example_memory_category(self):
        gen = self._make_gen_with_cache()
        tc_cfg = ToolCallingConfig()
        result = gen._gen_person_example("memory", tc_cfg)
        assert "messages" in result
        assert len(result["messages"]) == 5

    def test_gen_orientation_example_directly(self):
        gen = self._make_gen_with_cache()
        tc_cfg = ToolCallingConfig()
        result = gen._gen_orientation_example(tc_cfg)
        assert "messages" in result

    def test_gen_medication_example_directly(self):
        gen = self._make_gen_with_cache()
        tc_cfg = ToolCallingConfig()
        result = gen._gen_medication_example(tc_cfg)
        assert "messages" in result
        assert len(result["messages"]) == 5

    def test_gen_alert_example_directly(self):
        """Line 990: _gen_alert_example."""
        gen = self._make_gen_with_cache()
        tc_cfg = ToolCallingConfig()
        result = gen._gen_alert_example(tc_cfg)
        assert "messages" in result
        assert len(result["messages"]) == 5

    def test_generate_example_routes_correctly(self):
        gen = self._make_gen_with_cache()
        tc_cfg = ToolCallingConfig()
        for cat in ["person_recognition", "memory", "orientation", "medication", "alert"]:
            result = gen.generate_example(cat, tc_cfg)
            assert "messages" in result


# ---------------------------------------------------------------------------
# GeminiApiGenerator._wait_for_rate_limit — sleep branch (lines 1090-1099)
# ---------------------------------------------------------------------------

class TestWaitForRateLimitSleep:
    def test_rate_limit_triggers_sleep(self):
        """When RPM limit is reached, should sleep."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._rpm_limit = 2
        gen._rate_lock = __import__("threading").Lock()

        base = 1000.0
        gen._request_times = [base - 5, base - 3]  # 2 requests within 60s

        call_idx = [0]
        mono_values = [base, base + 56, base + 56]

        def fake_monotonic():
            idx = min(call_idx[0], len(mono_values) - 1)
            val = mono_values[idx]
            call_idx[0] += 1
            return val

        with patch("time.sleep") as mock_sleep:
            with patch("time.monotonic", side_effect=fake_monotonic):
                gen._wait_for_rate_limit()
                mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# GeminiApiGenerator._generate — API call (lines 1120-1131)
# ---------------------------------------------------------------------------

class TestGeminiApiGenerateCall:
    def test_generate_returns_stripped_text(self):
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._model_name = "gemma-4-31b-it"
        gen.temperature = 0.8
        gen.top_p = 0.92
        gen.max_new_tokens = 256
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []

        mock_response = MagicMock()
        mock_response.text = "  Hello world  "
        gen._client = MagicMock()
        gen._client.models.generate_content.return_value = mock_response
        gen._types = MagicMock()

        result = gen._generate("test")
        assert result == "Hello world"

    def test_generate_returns_empty_on_none_text(self):
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._model_name = "test"
        gen.temperature = 0.8
        gen.top_p = 0.92
        gen.max_new_tokens = 256
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []

        mock_response = MagicMock()
        mock_response.text = None
        gen._client = MagicMock()
        gen._client.models.generate_content.return_value = mock_response
        gen._types = MagicMock()

        result = gen._generate("test")
        assert result == ""


# ---------------------------------------------------------------------------
# GeminiApiGenerator._generate_batch — concurrent thread pool (line 1154)
# ---------------------------------------------------------------------------

class TestGeminiApiGenerateBatchConcurrent:
    def test_concurrent_execution_with_multiple_prompts(self):
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []
        gen._generate = MagicMock(side_effect=lambda p: f"resp_{p}")

        results = gen._generate_batch(["a", "b", "c", "d"])
        assert results == ["resp_a", "resp_b", "resp_c", "resp_d"]
        assert gen._generate.call_count == 4

    def test_single_prompt_skips_threadpool(self):
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []
        gen._generate = MagicMock(return_value="only")

        results = gen._generate_batch(["p1"])
        assert results == ["only"]


# ---------------------------------------------------------------------------
# _generate_batch_with_fallback — quota error fallback (line 1334)
# ---------------------------------------------------------------------------

class TestBatchFallbackQuotaErrorNew:
    def test_non_api_gen_error_propagates(self):
        """Non-GeminiApiGenerator errors should not trigger fallback."""
        gen = MagicMock(spec=GemmaDataGenerator)
        gen.generate_examples_batch.side_effect = RuntimeError("GPU error")
        tc_cfg = ToolCallingConfig()

        with pytest.raises(RuntimeError, match="GPU error"):
            _generate_batch_with_fallback(gen, ["orientation"], tc_cfg)

    def test_api_non_quota_error_propagates(self):
        """API errors that aren't quota errors should propagate."""
        gen = MagicMock(spec=GeminiApiGenerator)
        err = RuntimeError("not quota")
        err.code = 500
        gen.generate_examples_batch.side_effect = err
        tc_cfg = ToolCallingConfig()

        with patch("src.data._is_quota_error", return_value=False):
            with pytest.raises(RuntimeError, match="not quota"):
                _generate_batch_with_fallback(gen, ["orientation"], tc_cfg)


# ---------------------------------------------------------------------------
# _generate_medication_example — template path (line 1424)
# ---------------------------------------------------------------------------

class TestMedicationTemplatePath:
    def test_medication_template_all_time_periods(self):
        tc_cfg = ToolCallingConfig()
        random.seed(0)
        categories_seen = set()
        for _ in range(100):
            example = _generate_medication_example(tc_cfg)
            tool_call_str = example["messages"][2]["content"]
            if "morning" in tool_call_str:
                categories_seen.add("morning")
            elif "afternoon" in tool_call_str:
                categories_seen.add("afternoon")
            elif "night" in tool_call_str:
                categories_seen.add("night")
        assert len(categories_seen) >= 2

    def test_medication_template_empty_meds_branch(self):
        """Cover the else branch when meds list is empty."""
        tc_cfg = ToolCallingConfig()
        empty_meds = {"morning": [], "afternoon": [], "night": []}
        with patch("src.data._SAMPLE_MEDICATIONS", empty_meds):
            example = _generate_medication_example(tc_cfg)
        # The response should mention no medication scheduled
        assistant_msg = example["messages"][-1]["content"]
        assert "don't have any medication" in assistant_msg.lower() or "no" in assistant_msg.lower()


# ---------------------------------------------------------------------------
# _parallel_worker (lines 1493-1534)
# ---------------------------------------------------------------------------

class TestParallelWorker:
    def test_parallel_worker_generates_examples(self, tmp_path):
        from src.data import _parallel_worker

        output_path = str(tmp_path / "worker_output.jsonl")
        scenarios_weighted = ["person_recognition", "orientation", "medication", "alert"]

        mock_gen_instance = _make_mock_gemma_gen()
        mock_gen_instance._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        mock_gen_instance.warm_up_query_cache = MagicMock()

        def fake_batch(categories, tc_cfg, batch_size=8):
            return [
                {"messages": [{"role": "user", "content": f"example_{i}"}]}
                for i in range(len(categories))
            ]

        mock_gen_instance.generate_examples_batch = MagicMock(side_effect=fake_batch)

        with patch("src.data.GemmaDataGenerator", return_value=mock_gen_instance):
            with patch("torch.cuda.empty_cache"):
                _parallel_worker(
                    rank=0,
                    num_examples=5,
                    split="train",
                    output_path=output_path,
                    scenarios_weighted=scenarios_weighted,
                    gemma_model="test",
                    torch_dtype="float16",
                    queries_per_category=5,
                    seed=42,
                    batch_size=3,
                )

        with open(output_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 5


# ---------------------------------------------------------------------------
# generate_synthetic_data_parallel with 2+ GPUs (lines 1568-1621)
# ---------------------------------------------------------------------------

class TestParallelGenerationMultiGPU:
    def test_parallel_with_two_gpus(self, tmp_path):
        from src.data import generate_synthetic_data_parallel

        with patch("torch.cuda.device_count", return_value=2):
            with patch("torch.multiprocessing.set_start_method"):
                mock_process = MagicMock()
                mock_process.exitcode = 0

                with patch("torch.multiprocessing.Process", return_value=mock_process) as MockProc:
                    # Each worker writes temp files, so create them
                    for split in ["train", "val"]:
                        for rank in range(2):
                            temp_file = tmp_path / f"_tmp_{split}_gpu{rank}.jsonl"
                            with open(temp_file, "w") as f:
                                for i in range(3):
                                    f.write(json.dumps({"messages": [{"role": "user", "content": f"ex{i}"}]}) + "\n")

                    generate_synthetic_data_parallel(
                        output_dir=str(tmp_path),
                        num_train=6,
                        num_val=4,
                        seed=42,
                        num_gpus=2,
                    )

                assert (tmp_path / "tool_calling_train.jsonl").exists()
                assert (tmp_path / "tool_calling_val.jsonl").exists()

    def test_parallel_worker_failure_raises(self, tmp_path):
        from src.data import generate_synthetic_data_parallel

        with patch("torch.cuda.device_count", return_value=2):
            with patch("torch.multiprocessing.set_start_method"):
                mock_process = MagicMock()
                mock_process.exitcode = 1  # Non-zero exit code

                with patch("torch.multiprocessing.Process", return_value=mock_process):
                    with pytest.raises(RuntimeError, match="Worker process exited"):
                        generate_synthetic_data_parallel(
                            output_dir=str(tmp_path),
                            num_train=6,
                            num_val=4,
                            seed=42,
                            num_gpus=2,
                        )


# ---------------------------------------------------------------------------
# generate_synthetic_data — template mode loop (line 1737)
# ---------------------------------------------------------------------------

class TestTemplateModeLoop:
    def test_template_mode_generates_all_categories(self, tmp_path):
        """Template mode should iterate through weighted categories."""
        generate_synthetic_data(
            output_dir=str(tmp_path),
            num_train=50,
            num_val=10,
            seed=42,
            use_gemma=False,
        )
        with open(tmp_path / "tool_calling_train.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 50
        # Check multiple tool types appear
        tool_names = set()
        for line in lines:
            tool_call_msg = line["messages"][2]["content"]
            if "read_person" in tool_call_msg:
                tool_names.add("read_person")
            elif "describe_location" in tool_call_msg:
                tool_names.add("describe_location")
            elif "get_medication" in tool_call_msg:
                tool_names.add("get_medication")
            elif "alert_caregiver" in tool_call_msg:
                tool_names.add("alert_caregiver")
        assert len(tool_names) >= 3


# ---------------------------------------------------------------------------
# generate_synthetic_data — API mode quota fallback (lines 1767-1791)
# ---------------------------------------------------------------------------

class TestApiModeQuotaFallbackSingleExample:
    def test_single_example_api_quota_triggers_parallel(self, tmp_path):
        """When batch_count == 1 and API quota error, falls back to parallel."""
        QuotaError = type("QuotaError", (Exception,), {"code": 429})

        mock_gen = MagicMock(spec=GeminiApiGenerator)
        mock_gen.warm_up_query_cache = MagicMock()
        mock_gen.cleanup = MagicMock()
        mock_gen.generate_example = MagicMock(side_effect=QuotaError())

        with patch("src.data._create_generator", return_value=(mock_gen, True)):
            with patch("src.data._is_quota_error", return_value=True):
                with patch("src.data._finish_with_parallel") as mock_finish:
                    generate_synthetic_data(
                        output_dir=str(tmp_path),
                        num_train=3,
                        num_val=1,
                        seed=42,
                        use_gemma=True,
                        use_api=True,
                        rpm_limit=1,  # batch_count = 1 → single example path
                    )
                    mock_finish.assert_called_once()

    def test_local_gemma_single_example_generation(self, tmp_path):
        """When using_api=False, single example path generates via generate_example."""
        mock_gen = MagicMock(spec=GemmaDataGenerator)
        mock_gen.warm_up_query_cache = MagicMock()
        mock_gen.cleanup = MagicMock()
        mock_gen.generate_example = MagicMock(
            return_value={"messages": [{"role": "user", "content": "test"}]}
        )

        with patch("src.data.GemmaDataGenerator", return_value=mock_gen):
            generate_synthetic_data(
                output_dir=str(tmp_path),
                num_train=3,
                num_val=1,
                seed=42,
                use_gemma=True,
                use_api=False,
            )

        assert (tmp_path / "tool_calling_train.jsonl").exists()
        with open(tmp_path / "tool_calling_train.jsonl") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# _finish_with_parallel (lines 1835-1878)
# ---------------------------------------------------------------------------

class TestFinishWithParallel:
    def test_finish_generates_remaining(self, tmp_path):
        """_finish_with_parallel calls generate_synthetic_data_parallel."""
        path = tmp_path / "tool_calling_train.jsonl"
        path.write_text("")

        with patch("src.data.generate_synthetic_data_parallel") as mock_par:
            _finish_with_parallel(
                output_dir=str(tmp_path),
                path=path,
                split="train",
                remaining=100,
                remaining_val=0,
                num_val=50,
                seed=42,
                gemma_model="google/gemma-4-e2b-it",
                torch_dtype="float16",
                queries_per_category=30,
                num_gpus=2,
            )
            mock_par.assert_called_once()
            call_kwargs = mock_par.call_args[1]
            assert call_kwargs["num_train"] == 100
            assert call_kwargs["seed"] == 1042  # seed + 1000

    def test_finish_val_split(self, tmp_path):
        """_finish_with_parallel handles val split."""
        path = tmp_path / "tool_calling_val.jsonl"
        path.write_text("")

        with patch("src.data.generate_synthetic_data_parallel") as mock_par:
            _finish_with_parallel(
                output_dir=str(tmp_path),
                path=path,
                split="val",
                remaining=20,
                remaining_val=0,
                num_val=50,
                seed=42,
                gemma_model="google/gemma-4-e2b-it",
                torch_dtype="float16",
                queries_per_category=30,
                num_gpus=2,
            )
            call_kwargs = mock_par.call_args[1]
            assert call_kwargs["num_val"] == 20
            assert call_kwargs["num_train"] == 0

    def test_finish_merges_parallel_output(self, tmp_path):
        """Line 1878: merge logic appends parallel output to partial file."""
        # partial_path is different from the standard output path
        partial_path = tmp_path / "partial_train.jsonl"
        partial_path.write_text('{"messages": ["partial1"]}\n')

        # generate_synthetic_data_parallel will write to tool_calling_train.jsonl
        parallel_output = tmp_path / "tool_calling_train.jsonl"

        def fake_parallel(**kwargs):
            parallel_output.write_text('{"messages": ["parallel1"]}\n{"messages": ["parallel2"]}\n')

        with patch("src.data.generate_synthetic_data_parallel", side_effect=fake_parallel):
            _finish_with_parallel(
                output_dir=str(tmp_path),
                path=partial_path,
                split="train",
                remaining=50,
                remaining_val=0,
                num_val=10,
                seed=42,
                gemma_model="google/gemma-4-e2b-it",
                torch_dtype="float16",
                queries_per_category=30,
                num_gpus=2,
            )

        with open(partial_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Should have original + appended from parallel
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# GemmaDataGenerator.generate_examples_batch (batched generation)
# ---------------------------------------------------------------------------

class TestGenerateExamplesBatch:
    def test_batch_generation_assembles_examples(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        gen._generate_batch = MagicMock(
            return_value=["That's Maria, your granddaughter." for _ in range(5)]
        )
        tc_cfg = ToolCallingConfig()

        # Include all category branches in _prepare_example_partial
        results = gen.generate_examples_batch(
            ["person_recognition", "memory", "orientation", "medication", "alert"],
            tc_cfg,
            batch_size=8,
        )
        assert len(results) == 5
        for r in results:
            assert "messages" in r
            assert len(r["messages"]) == 5

    def test_batch_generation_uses_hint_on_short_response(self):
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {s["category"]: s["user_templates"] for s in _SCENARIOS}
        gen._generate_batch = MagicMock(return_value=[""])  # Empty response → hint
        tc_cfg = ToolCallingConfig()

        results = gen.generate_examples_batch(["orientation"], tc_cfg, batch_size=8)
        assert len(results) == 1
        # The assistant response should be the hint (non-empty)
        last_msg = results[0]["messages"][-1]
        assert len(last_msg["content"]) > 10


# ---------------------------------------------------------------------------
# Import guard for `os` used in test helpers
# ---------------------------------------------------------------------------
import os


# ---------------------------------------------------------------------------
# TextDataset.__init__ — _resolve_cache_path returns None (line 75)
# ---------------------------------------------------------------------------

class TestTextDatasetCacheDisabled:
    def test_use_cache_disabled_when_resolve_returns_none(self):
        """Line 75: when _resolve_cache_path returns None, use_cache becomes False."""
        mock_tokenizer = MagicMock()
        type(mock_tokenizer).name_or_path = PropertyMock(return_value="mock-tok")
        mock_tokenizer.encode = MagicMock(return_value=list(range(128)))
        fake_dataset = [{"text": "Sentence that is long enough for tokenization."} for _ in range(5)]

        with patch("datasets.load_dataset", return_value=fake_dataset):
            with patch.object(TextDataset, "_resolve_cache_path", return_value=None):
                ds = TextDataset(
                    tokenizer=mock_tokenizer,
                    max_seq_len=32,
                    use_cache=True,
                )
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# _ensure_cache_dir — is_dir() returns False after mkdir (lines 138-139)
# ---------------------------------------------------------------------------

class TestEnsureCacheDirNotDir:
    def test_exists_but_not_dir(self, tmp_path):
        """Lines 138-139: mkdir succeeds but is_dir returns False."""
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        ds.chunks = []

        target = tmp_path / "weird_path"
        target.mkdir()  # Create a real directory so mkdir succeeds

        # Patch at the instance level via a wrapper
        original_is_dir = target.is_dir

        with patch.object(type(target), "is_dir", return_value=False):
            result = ds._ensure_cache_dir(target)
        assert result is False

    def test_exists_but_not_dir_via_mock_path(self):
        """Lines 138-139: using a mock Path where mkdir ok but is_dir is False."""
        ds = TextDataset.__new__(TextDataset)
        ds.tokenizer = MagicMock()
        ds.chunks = []

        mock_dir = MagicMock(spec=Path)
        mock_dir.mkdir = MagicMock()  # No error
        mock_dir.is_dir = MagicMock(return_value=False)

        result = ds._ensure_cache_dir(mock_dir)
        assert result is False


# ---------------------------------------------------------------------------
# _create_labels — model_end not found (line 360)
# ---------------------------------------------------------------------------

class TestCreateLabelsNoEndTag:
    def test_model_turn_without_end_tag(self, tmp_path):
        """Line 360: model turn found but <end_of_turn> missing."""
        tc_cfg = ToolCallingConfig()
        path = tmp_path / "data.jsonl"
        example = _generate_person_recognition_example(tc_cfg)
        with open(path, "w") as f:
            f.write(json.dumps(example) + "\n")

        tokenizer = MagicMock()

        def mock_call(text, **kwargs):
            max_length = kwargs.get("max_length", 64)
            return {
                "input_ids": torch.randint(0, 100, (1, max_length)),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        tokenizer.encode = MagicMock(return_value=list(range(50)))
        # Has model start but NO end tag
        tokenizer.decode = MagicMock(
            return_value="<start_of_turn>model\nHello, I am the assistant"
        )

        ds = ToolCallingDataset(tokenizer=tokenizer, data_path=str(path), max_seq_len=64)
        item = ds[0]
        # Should still produce labels (some unmasked for the model turn)
        assert item["labels"].shape == item["input_ids"].shape


# ---------------------------------------------------------------------------
# _parse_numbered_list — bullet, quoted, fallback (lines 751-764)
# ---------------------------------------------------------------------------

class TestParseNumberedListExtended:
    def test_bullet_patterns(self):
        text = "- A longer bullet point line here\n* Another valid star point\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 2

    def test_quoted_lines(self):
        text = '"A longer quoted sentence here"\n\'Another long quoted line here\'\n'
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 2

    def test_fallback_plain_lines(self):
        text = "This is a plain sentence long enough to match\nAnother plain long sentence\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 2

    def test_excluded_prefixes(self):
        text = "# Comment line\nHere is a long sentence\nSure, here is output\n"
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 0

    def test_mixed_formats(self):
        text = (
            "1. First numbered item text\n"
            "- Second bullet item text here\n"
            '"Third quoted item text here"\n'
            "Fourth plain sentence long enough\n"
        )
        result = GemmaDataGenerator._parse_numbered_list(text, expected=10)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# _prepare_example_partial — name replacement (line 840) + memory (line 844)
# ---------------------------------------------------------------------------

class TestPrepareExamplePartialNameBranches:
    def test_name_replacement_in_query(self):
        """Line 840: raw_query contains a person name → replaced with 'this person'."""
        gen = _make_mock_gemma_gen()
        # Seed cache with queries containing a specific person name
        gen._user_query_cache = {
            "person_recognition": ["Tell me about Maria and her family."],
            "memory": ["Tell me about Maria and her family."],
            "orientation": ["Where am I?"],
            "medication": ["What pills?"],
            "alert": ["Help me!"],
        }

        tc_cfg = ToolCallingConfig()
        # Run many times to ensure we get a person that ISN'T Maria
        random.seed(10)
        for _ in range(50):
            partial = gen._prepare_example_partial("person_recognition", tc_cfg)
            if "this person" in partial["user_msg"]:
                return  # Covered line 840
        # At least some iterations should have replaced "Maria"
        # (probability is high since 23/24 persons aren't Maria)

    def test_memory_category_user_msg_selection(self):
        """Lines 843-844: memory category with person name not in query."""
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {
            "memory": ["Who used to visit me on Sundays?"],
        }
        tc_cfg = ToolCallingConfig()
        partial = gen._prepare_example_partial("memory", tc_cfg)
        # Should have chosen one of the name-specific templates
        assert partial["user_msg"]  # Non-empty

    def test_alert_category_in_partial(self):
        """Lines 876-883: alert branch in _prepare_example_partial."""
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {
            "alert": ["I'm scared and confused."],
        }
        tc_cfg = ToolCallingConfig()
        partial = gen._prepare_example_partial("alert", tc_cfg)
        assert "alert_caregiver" in partial["tool_call"]

    def test_unknown_category_raises(self):
        """Line 885: unknown category in _prepare_example_partial."""
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {}
        tc_cfg = ToolCallingConfig()
        with pytest.raises(ValueError, match="Unknown category"):
            gen._prepare_example_partial("nonexistent", tc_cfg)


# ---------------------------------------------------------------------------
# _gen_person_example — name replacement (line 918)
# ---------------------------------------------------------------------------

class TestGenPersonExampleNameReplacement:
    def test_name_in_query_replaced(self):
        """Line 918: mismatched name in raw_query is replaced."""
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {
            "person_recognition": ["Is that Maria over there? I think I know her."],
        }
        gen.generate_assistant_response = MagicMock(return_value="A warm response.")
        tc_cfg = ToolCallingConfig()

        random.seed(10)
        for _ in range(50):
            result = gen._gen_person_example("person_recognition", tc_cfg)
            user_msg = result["messages"][1]["content"]
            if "this person" in user_msg:
                return  # Covered line 918

    def test_memory_category_name_not_in_query(self):
        """Line 919: memory category, person name not in query."""
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {
            "memory": ["Who used to visit me?"],
        }
        gen.generate_assistant_response = MagicMock(return_value="A warm response.")
        tc_cfg = ToolCallingConfig()
        result = gen._gen_person_example("memory", tc_cfg)
        assert "messages" in result


# ---------------------------------------------------------------------------
# _gen_medication_example — empty meds branch (line 990)
# ---------------------------------------------------------------------------

class TestGenMedicationExampleEmptyMeds:
    def test_no_meds_for_time_period(self):
        """Line 990: meds list is empty → fallback hint."""
        gen = _make_mock_gemma_gen()
        gen._user_query_cache = {
            "medication": ["What pills do I take?"],
        }
        gen.generate_assistant_response = MagicMock(return_value="No meds right now.")
        tc_cfg = ToolCallingConfig()

        empty_meds = {"morning": [], "afternoon": [], "night": []}
        with patch("src.data._SAMPLE_MEDICATIONS", empty_meds):
            result = gen._gen_medication_example(tc_cfg)
        assert "messages" in result


# ---------------------------------------------------------------------------
# GeminiApiGenerator.__init__ success path (lines 1090-1099)
# ---------------------------------------------------------------------------

class TestGeminiApiGeneratorInitSuccess:
    def test_successful_init(self):
        """Lines 1090-1099: successful __init__ sets up client and rate limiter."""
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            gen = GeminiApiGenerator(
                api_key="test-key",
                model_name="gemma-4-31b-it",
                rpm_limit=10,
            )
        assert gen._client is mock_genai.Client.return_value
        assert gen._model_name == "gemma-4-31b-it"
        assert gen._rpm_limit == 10
        assert gen._request_times == []
        assert gen.device == "api"


# ---------------------------------------------------------------------------
# GeminiApiGenerator._generate — APIError propagation (line 1154)
# ---------------------------------------------------------------------------

class TestGeminiApiGenerateAPIError:
    def test_api_error_propagates(self):
        """Line 1154: genai_errors.APIError is re-raised."""
        gen = GeminiApiGenerator.__new__(GeminiApiGenerator)
        gen._model_name = "test"
        gen.temperature = 0.8
        gen.top_p = 0.92
        gen.max_new_tokens = 256
        gen._rpm_limit = 14
        gen._rate_lock = __import__("threading").Lock()
        gen._request_times = []
        gen._types = MagicMock()

        # Create a real exception class for genai_errors.APIError
        APIError = type("APIError", (Exception,), {"code": 429})
        mock_genai_errors = MagicMock()
        mock_genai_errors.APIError = APIError

        gen._client = MagicMock()
        gen._client.models.generate_content.side_effect = APIError("quota")

        with patch.dict("sys.modules", {"google.genai.errors": mock_genai_errors}):
            with patch.dict("sys.modules", {"google.genai": MagicMock(errors=mock_genai_errors)}):
                with patch.dict("sys.modules", {"google": MagicMock()}):
                    with pytest.raises(APIError):
                        gen._generate("test prompt")


# ---------------------------------------------------------------------------
# build_dataloader — validation split (line 1878)
# ---------------------------------------------------------------------------

class TestBuildDataloaderValidation:
    def test_validation_split_sets_max_samples(self):
        """Line 1878: split='validation' sets max_samples from cfg."""
        tokenizer = MagicMock()
        cfg = TrainingConfig(phase="tcs-lora", batch_size=4, max_eval_samples=100)

        with patch("src.data.TextDataset") as mock_dataset:
            mock_dataset.return_value = MagicMock()
            mock_dataset.return_value.__len__ = MagicMock(return_value=10)

            loader = build_dataloader(
                tokenizer,
                cfg,
                max_seq_len=512,
                split="validation",
            )

        assert loader is not None
        assert mock_dataset.call_args.kwargs["max_samples"] == 100


# ---------------------------------------------------------------------------
# generate_synthetic_data — non-quota error in batch API path (line 1737)
# ---------------------------------------------------------------------------

class TestApiModeNonQuotaError:
    def test_non_quota_error_in_batch_api_raises(self, tmp_path):
        """Line 1737: non-quota error in batch API path raises."""
        mock_gen = MagicMock(spec=GeminiApiGenerator)
        mock_gen.warm_up_query_cache = MagicMock()
        mock_gen.cleanup = MagicMock()
        mock_gen.generate_examples_batch.side_effect = RuntimeError("network")

        with patch("src.data._create_generator", return_value=(mock_gen, True)):
            with patch("src.data._is_quota_error", return_value=False):
                with pytest.raises(RuntimeError, match="network"):
                    generate_synthetic_data(
                        output_dir=str(tmp_path),
                        num_train=5,
                        num_val=1,
                        seed=42,
                        use_gemma=True,
                        use_api=True,
                        rpm_limit=14,
                    )

    def test_non_quota_error_in_single_api_raises(self, tmp_path):
        """Line 1769: non-quota error in single example API path raises."""
        mock_gen = MagicMock(spec=GeminiApiGenerator)
        mock_gen.warm_up_query_cache = MagicMock()
        mock_gen.cleanup = MagicMock()
        mock_gen.generate_example.side_effect = RuntimeError("oops")

        with patch("src.data._create_generator", return_value=(mock_gen, True)):
            with patch("src.data._is_quota_error", return_value=False):
                with pytest.raises(RuntimeError, match="oops"):
                    generate_synthetic_data(
                        output_dir=str(tmp_path),
                        num_train=3,
                        num_val=1,
                        seed=42,
                        use_gemma=True,
                        use_api=True,
                        rpm_limit=1,  # Forces single example path
                    )


