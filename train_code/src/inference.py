"""
Inference pipeline with tool execution loop.

Generates responses from the compressed model, detects tool calls in the output,
executes them against SQLite, and feeds results back for final response.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.config import (
    ModelConfig,
    CompressionConfig,
    LoraConfig,
    InferenceConfig,
    ToolCallingConfig,
    DatabaseConfig,
)
from src.model import Gemma4WithTCS
from src.tools.sqlite_tools import (
    execute_tool_call,
    format_tool_response,
    init_database,
    parse_tool_call,
    seed_demo_data,
)

logger = logging.getLogger(__name__)


class MemoryCompanion:
    """
    Inference wrapper that handles:
    1. Prompt construction with system prompt + tool definitions
    2. Compressed generation via TCS + LoRA
    3. Tool call detection and execution
    4. Multi-turn tool use (generate → tool → generate → ...)
    """

    def __init__(
        self,
        model: Gemma4WithTCS,
        db_conn: sqlite3.Connection,
        tc_cfg: ToolCallingConfig | None = None,
        inf_cfg: InferenceConfig | None = None,
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        self.db = db_conn
        self.tc_cfg = tc_cfg or ToolCallingConfig()
        self.inf_cfg = inf_cfg or InferenceConfig()
        self.model.eval()

    @torch.no_grad()
    def respond(self, user_message: str, max_tool_rounds: int = 3) -> str:
        """
        Generate a response, executing tool calls as needed.

        Args:
            user_message: The patient's message or camera observation.
            max_tool_rounds: Maximum number of tool call → response rounds.

        Returns:
            Final text response for the patient.
        """
        # Build conversation
        messages = [
            {"role": "system", "content": self.tc_cfg.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_tool_rounds + 1):
            # Format and tokenize
            prompt = self._format_messages(messages)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(next(self.model.parameters()).device)

            # Generate
            output_ids = self._generate(input_ids)
            response_text = self.tokenizer.decode(
                output_ids[0, input_ids.size(1):],
                skip_special_tokens=True,
            )

            # Check for tool call
            tool_call = parse_tool_call(response_text, self.tc_cfg)

            if tool_call is None:
                # No tool call — this is the final response
                return response_text.strip()

            # Execute tool call
            tool_name, tool_args = tool_call
            logger.info("Tool call: %s(%s)", tool_name, tool_args)

            try:
                result = execute_tool_call(self.db, tool_name, tool_args)
            except (ValueError, TypeError) as e:
                logger.warning("Tool execution failed: %s", e)
                result = {"error": str(e)}

            # Add assistant response + tool result to conversation
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "tool",
                "content": format_tool_response(tool_name, result, self.tc_cfg),
            })

        # Exhausted tool rounds — return last response
        return response_text.strip()

    def _generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens using the compressed model."""
        # For models with TCS, we use the compressed forward path
        device = input_ids.device

        # Simple greedy/sampling generation loop
        generated = input_ids
        eos_id = self.tokenizer.eos_token_id

        for _ in range(self.inf_cfg.max_new_tokens):
            outputs = self.model(
                input_ids=generated,
                mode="compressed",
                compression_ratio=self.inf_cfg.compression_ratio,
            )
            logits = outputs["logits"]
            next_logits = logits[:, -1, :]  # Last position

            if self.inf_cfg.do_sample:
                # Temperature + top-p sampling
                next_logits = next_logits / self.inf_cfg.temperature
                probs = torch.softmax(next_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > self.inf_cfg.top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(
                    -1, torch.multinomial(sorted_probs, 1)
                )
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == eos_id:
                break

        return generated

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages into the chat template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
            elif role == "tool":
                parts.append(f"<start_of_turn>tool\n{content}<end_of_turn>")

        # Add start of model turn to prompt generation
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Inference with Memory Companion")
    parser.add_argument("--model", default="google/gemma-4-e2b-it", help="Base model")
    parser.add_argument("--model-path", default="outputs/final", help="Trained weights dir")
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--db-path", default="data/companion.db")
    parser.add_argument("--seed-demo", action="store_true", help="Seed demo data into DB")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()

    # Build model
    comp_cfg = CompressionConfig(ratio=args.compression_ratio)
    lora_cfg = LoraConfig(enabled=True)
    model_cfg = ModelConfig(
        base_model_name=args.model,
        compression=comp_cfg,
        lora=lora_cfg,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    logger.info("Loading model...")
    model = Gemma4WithTCS(model_cfg)

    # Load trained weights
    model_path = Path(args.model_path)
    if model_path.exists():
        model.load_trainable(model_path)
        logger.info("Loaded trained weights from %s", model_path)

    # Database
    db_conn = init_database(args.db_path)
    if args.seed_demo:
        seed_demo_data(db_conn)

    # Inference config
    inf_cfg = InferenceConfig(
        compression_ratio=args.compression_ratio,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    companion = MemoryCompanion(model, db_conn, inf_cfg=inf_cfg)

    if args.prompt:
        # Single prompt mode
        response = companion.respond(args.prompt)
        print(f"\n{'='*60}")
        print(f"Patient: {args.prompt}")
        print(f"Companion: {response}")
        print(f"{'='*60}")

    elif args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Memory Companion — Interactive Mode")
        print("Type 'quit' to exit")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("Patient: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            response = companion.respond(user_input)
            print(f"Companion: {response}\n")

    else:
        # Demo mode — run a few example queries
        demo_queries = [
            "Someone just walked in. Who is that?",
            "Where am I right now?",
            "Do I need to take any medicine now?",
            "I'm very confused and scared.",
        ]
        print("\n" + "=" * 60)
        print("Memory Companion — Demo Mode")
        print("=" * 60)

        for query in demo_queries:
            print(f"\nPatient: {query}")
            response = companion.respond(query)
            print(f"Companion: {response}")
            print("-" * 40)

    db_conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
