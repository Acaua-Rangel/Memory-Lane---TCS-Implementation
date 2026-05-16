"""
Tests for inference module.

Tests MemoryCompanion logic and CLI parsing without loading the actual Gemma model.
"""

import pytest
from unittest.mock import MagicMock, patch

import torch

from src.config import ToolCallingConfig, InferenceConfig


# ---------------------------------------------------------------------------
# MemoryCompanion
# ---------------------------------------------------------------------------

class TestMemoryCompanion:
    def _make_companion(self):
        from src.inference import MemoryCompanion

        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=torch.randint(0, 100, (1, 20)))
        model.tokenizer.decode = MagicMock(return_value="That's Maria, your granddaughter!")
        model.tokenizer.eos_token_id = 2
        # Return a fresh iterator each time parameters() is called
        model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(2, 2)]))
        model.eval = MagicMock()

        # Simulate generate: model returns logits that lead to EOS on first token
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 2] = 100.0  # EOS token gets highest logit
        model.return_value = {"logits": logits}

        db = MagicMock()
        return MemoryCompanion(model, db)

    def test_format_messages(self):
        companion = self._make_companion()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "tool result"},
        ]
        text = companion._format_messages(messages)
        assert "<start_of_turn>system" in text
        assert "<start_of_turn>user" in text
        assert "<start_of_turn>model" in text
        assert "<start_of_turn>tool" in text
        # Ends with model turn prompt
        assert text.strip().endswith("<start_of_turn>model")

    def test_respond_no_tool_call(self):
        companion = self._make_companion()
        # Mock parse_tool_call to return None (no tool call)
        with patch("src.inference.parse_tool_call", return_value=None):
            response = companion.respond("Hello")
        assert isinstance(response, str)

    def test_respond_with_tool_call(self):
        companion = self._make_companion()
        call_count = [0]

        def mock_parse(text, tc_cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("read_person", {"face_id": "face_001"})
            return None

        with patch("src.inference.parse_tool_call", side_effect=mock_parse):
            with patch("src.inference.execute_tool_call", return_value={"name": "Maria"}):
                with patch("src.inference.format_tool_response", return_value="<tool_response>...</tool_response>"):
                    response = companion.respond("Who is that?")
        assert isinstance(response, str)

    def test_respond_tool_execution_error(self):
        companion = self._make_companion()
        call_count = [0]

        def mock_parse(text, tc_cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("read_person", {"face_id": "bad"})
            return None

        with patch("src.inference.parse_tool_call", side_effect=mock_parse):
            with patch("src.inference.execute_tool_call", side_effect=ValueError("Tool error")):
                with patch("src.inference.format_tool_response", return_value="<tool_response>error</tool_response>"):
                    response = companion.respond("Who is that?")
        assert isinstance(response, str)

    def test_generate_greedy(self):
        companion = self._make_companion()
        companion.inf_cfg = InferenceConfig(do_sample=False, max_new_tokens=5)

        input_ids = torch.randint(0, 100, (1, 10))
        # Make model return EOS at first step
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 2] = 100.0
        companion.model.return_value = {"logits": logits}

        result = companion._generate(input_ids)
        assert result.shape[1] > input_ids.shape[1]

    def test_generate_sampling(self):
        companion = self._make_companion()
        companion.inf_cfg = InferenceConfig(do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=5)

        input_ids = torch.randint(0, 100, (1, 10))
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 2] = 100.0
        companion.model.return_value = {"logits": logits}

        result = companion._generate(input_ids)
        assert result.shape[1] > input_ids.shape[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestInferenceCLI:
    def test_module_imports(self):
        from src.inference import MemoryCompanion, main
        assert callable(main)

    def test_format_messages_all_roles(self):
        from src.inference import MemoryCompanion
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
        model.eval = MagicMock()
        db = MagicMock()

        companion = MemoryCompanion(model, db)
        text = companion._format_messages([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "ast"},
            {"role": "tool", "content": "tl"},
        ])

        assert "<start_of_turn>system\nsys<end_of_turn>" in text
        assert "<start_of_turn>user\nusr<end_of_turn>" in text
        assert "<start_of_turn>model\nast<end_of_turn>" in text
        assert "<start_of_turn>tool\ntl<end_of_turn>" in text

    def test_main_with_prompt(self):
        """Test CLI main with --prompt mode."""
        from src.inference import main

        with patch("sys.argv", [
            "inference", "--prompt", "Hello", "--model-path", "/nonexistent",
        ]):
            with patch("src.inference.Gemma4WithTCS") as mock_model_cls:
                mock_model = MagicMock()
                mock_model.tokenizer = MagicMock()
                mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
                mock_model.eval = MagicMock()
                mock_model.load_trainable = MagicMock()
                mock_model_cls.return_value = mock_model

                with patch("src.inference.init_database") as mock_db:
                    mock_conn = MagicMock()
                    mock_db.return_value = mock_conn

                    with patch("src.inference.MemoryCompanion") as mock_companion_cls:
                        mock_companion = MagicMock()
                        mock_companion.respond.return_value = "I'm here to help."
                        mock_companion_cls.return_value = mock_companion

                        with patch("builtins.print"):
                            with patch("pathlib.Path.exists", return_value=False):
                                main()

                        mock_companion.respond.assert_called_once_with("Hello")

    def test_main_demo_mode(self):
        """Test CLI main in default demo mode."""
        from src.inference import main

        with patch("sys.argv", ["inference", "--model-path", "/nonexistent"]):
            with patch("src.inference.Gemma4WithTCS") as mock_model_cls:
                mock_model = MagicMock()
                mock_model.tokenizer = MagicMock()
                mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
                mock_model.eval = MagicMock()
                mock_model.load_trainable = MagicMock()
                mock_model_cls.return_value = mock_model

                with patch("src.inference.init_database") as mock_db:
                    mock_db.return_value = MagicMock()

                    with patch("src.inference.MemoryCompanion") as mock_companion_cls:
                        mock_companion = MagicMock()
                        mock_companion.respond.return_value = "Response"
                        mock_companion_cls.return_value = mock_companion

                        with patch("builtins.print"):
                            with patch("pathlib.Path.exists", return_value=False):
                                main()

                        # Demo mode runs 4 queries
                        assert mock_companion.respond.call_count == 4

    def test_main_interactive_mode(self):
        """Test CLI main in interactive mode with immediate quit."""
        from src.inference import main

        with patch("sys.argv", ["inference", "--interactive", "--model-path", "/nonexistent"]):
            with patch("src.inference.Gemma4WithTCS") as mock_model_cls:
                mock_model = MagicMock()
                mock_model.tokenizer = MagicMock()
                mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
                mock_model.eval = MagicMock()
                mock_model_cls.return_value = mock_model

                with patch("src.inference.init_database") as mock_db:
                    mock_db.return_value = MagicMock()

                    with patch("src.inference.MemoryCompanion") as mock_companion_cls:
                        mock_companion = MagicMock()
                        mock_companion_cls.return_value = mock_companion

                        with patch("builtins.input", side_effect=["quit"]):
                            with patch("builtins.print"):
                                with patch("pathlib.Path.exists", return_value=False):
                                    main()

    def test_main_interactive_with_queries(self):
        """Test interactive mode with real queries then quit."""
        from src.inference import main

        with patch("sys.argv", ["inference", "--interactive", "--model-path", "/nonexistent"]):
            with patch("src.inference.Gemma4WithTCS") as mock_model_cls:
                mock_model = MagicMock()
                mock_model.tokenizer = MagicMock()
                mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
                mock_model.eval = MagicMock()
                mock_model_cls.return_value = mock_model

                with patch("src.inference.init_database") as mock_db:
                    mock_db.return_value = MagicMock()

                    with patch("src.inference.MemoryCompanion") as mock_companion_cls:
                        mock_companion = MagicMock()
                        mock_companion.respond.return_value = "Answer"
                        mock_companion_cls.return_value = mock_companion

                        with patch("builtins.input", side_effect=["Hello", "", "exit"]):
                            with patch("builtins.print"):
                                with patch("pathlib.Path.exists", return_value=False):
                                    main()

                        # 1 real query ("Hello"), empty skipped, "exit" quits
                        mock_companion.respond.assert_called_once_with("Hello")


class TestRespondToolRoundExhaustion:
    def test_exhausted_tool_rounds_returns_last_response(self):
        """When max_tool_rounds is exceeded, should return last response text."""
        from src.inference import MemoryCompanion

        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=torch.randint(0, 100, (1, 20)))
        model.tokenizer.decode = MagicMock(return_value="tool call response")
        model.tokenizer.eos_token_id = 2
        model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(2, 2)]))
        model.eval = MagicMock()
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 2] = 100.0
        model.return_value = {"logits": logits}

        db = MagicMock()
        companion = MemoryCompanion(model, db)

        # Always return a tool call
        with patch("src.inference.parse_tool_call", return_value=("read_person", {"face_id": "1"})):
            with patch("src.inference.execute_tool_call", return_value={"name": "Maria"}):
                with patch("src.inference.format_tool_response", return_value="<tool_response>ok</tool_response>"):
                    response = companion.respond("Who?", max_tool_rounds=2)
        assert isinstance(response, str)

    def test_main_interactive_eof(self):
        """Test interactive mode with EOFError."""
        from src.inference import main

        with patch("sys.argv", ["inference", "--interactive", "--model-path", "/nonexistent"]):
            with patch("src.inference.Gemma4WithTCS") as mock_model_cls:
                mock_model = MagicMock()
                mock_model.tokenizer = MagicMock()
                mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
                mock_model.eval = MagicMock()
                mock_model_cls.return_value = mock_model

                with patch("src.inference.init_database") as mock_db:
                    mock_db.return_value = MagicMock()

                    with patch("src.inference.MemoryCompanion") as mock_companion_cls:
                        mock_companion_cls.return_value = MagicMock()

                        with patch("builtins.input", side_effect=EOFError):
                            with patch("builtins.print"):
                                with patch("pathlib.Path.exists", return_value=False):
                                    main()

    def test_main_with_seed_demo(self):
        """Test CLI main with --seed-demo flag."""
        from src.inference import main

        with patch("sys.argv", [
            "inference", "--prompt", "Hi", "--seed-demo",
            "--model-path", "/nonexistent",
        ]):
            with patch("src.inference.Gemma4WithTCS") as mock_model_cls:
                mock_model = MagicMock()
                mock_model.tokenizer = MagicMock()
                mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
                mock_model.eval = MagicMock()
                mock_model_cls.return_value = mock_model

                with patch("src.inference.init_database") as mock_db:
                    mock_db.return_value = MagicMock()

                    with patch("src.inference.seed_demo_data") as mock_seed:
                        with patch("src.inference.MemoryCompanion") as mock_companion_cls:
                            mock_companion = MagicMock()
                            mock_companion.respond.return_value = "Hi!"
                            mock_companion_cls.return_value = mock_companion

                            with patch("builtins.print"):
                                with patch("pathlib.Path.exists", return_value=False):
                                    main()

                        mock_seed.assert_called_once()

    def test_main_loads_weights_when_path_exists(self):
        """Test that main() loads trained weights when model_path exists."""
        from src.inference import main

        with patch("sys.argv", [
            "inference", "--prompt", "Hi",
            "--model-path", "/some/path",
        ]):
            with patch("src.inference.Gemma4WithTCS") as mock_model_cls:
                mock_model = MagicMock()
                mock_model.tokenizer = MagicMock()
                mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.randn(1)]))
                mock_model.eval = MagicMock()
                mock_model.load_trainable = MagicMock()
                mock_model_cls.return_value = mock_model

                with patch("src.inference.init_database") as mock_db:
                    mock_db.return_value = MagicMock()

                    with patch("src.inference.MemoryCompanion") as mock_companion_cls:
                        mock_companion = MagicMock()
                        mock_companion.respond.return_value = "Ok"
                        mock_companion_cls.return_value = mock_companion

                        with patch("builtins.print"):
                            with patch("pathlib.Path.exists", return_value=True):
                                main()

                    mock_model.load_trainable.assert_called_once()
