"""
Configuration dataclasses for the Gemma 4 Memory Companion project.

All hyperparameters, model settings, and training options are centralized here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Model & Compression
# ---------------------------------------------------------------------------

@dataclass
class CompressionConfig:
    """Token Compression Sub-network settings."""
    enabled: bool = True
    ratio: int = 4
    kernel_size: int = 7
    # Multi-scale ratios (used when multi_scale=True)
    ratios: list[int] = field(default_factory=lambda: [2, 4, 8])
    multi_scale: bool = False


@dataclass
class LoraConfig:
    """LoRA adapter settings (maps to peft.LoraConfig)."""
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"


@dataclass
class ModelConfig:
    """Top-level model configuration."""
    base_model_name: str = "google/gemma-4-e2b-it"
    max_seq_len: int = 512
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    gradient_checkpointing: bool = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    phase: Literal["tcs-pretrain", "tcs-lora", "tool-calling"] = "tcs-pretrain"

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2000
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # Batching
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # DDP OOM recovery (Kaggle / low-VRAM safety)
    ddp_auto_batch_retry_on_oom: bool = True
    ddp_oom_retry_max_attempts: int = 3
    ddp_oom_retry_max_step: int = 1
    ddp_oom_retry_min_batch_size: int = 1

    # Mixed precision
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"

    # Self-distillation
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0

    # Gemini API teacher (optional: uses larger model for teacher signal)
    use_gemini_teacher: bool = False
    gemini_teacher_model: str = "gemma-4-31b-it"
    gemini_api_key: str | None = None
    gemini_rpm_limit: int = 14
    teacher_cache_dir: str = "data/teacher_cache"
    teacher_loss_weight: float = 0.3  # Weight for auxiliary API teacher CE loss

    # Checkpointing
    output_dir: str = "outputs"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    max_checkpoints_to_keep: int = 3  # Disk budget: keep only last N checkpoints
    cleanup_checkpoints_on_finish: bool = True  # Delete checkpoints after saving final

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    max_eval_samples: int = 200

    # Resume
    resume_from: str | None = None
    tcs_checkpoint: str | None = None
    lora_checkpoint: str | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def phase_output_dir(self) -> Path:
        return Path(self.output_dir) / self.phase.replace("-", "_")


# ---------------------------------------------------------------------------
# Tool Calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "read_person",
        "description": "Look up a person by face embedding match ID. Returns their name, relationship to the patient, and stored memories.",
        "parameters": {
            "type": "object",
            "properties": {
                "face_id": {
                    "type": "string",
                    "description": "The face embedding match ID from the recognition system.",
                }
            },
            "required": ["face_id"],
        },
    },
    {
        "name": "write_encounter",
        "description": "Log an encounter with a person. Records who visited, context, and timestamp for future reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "person_name": {
                    "type": "string",
                    "description": "Name of the person encountered.",
                },
                "context": {
                    "type": "string",
                    "description": "Brief description of the encounter.",
                },
            },
            "required": ["person_name", "context"],
        },
    },
    {
        "name": "get_medication",
        "description": "Get the medication schedule for a given time of day.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_of_day": {
                    "type": "string",
                    "enum": ["morning", "afternoon", "evening", "night"],
                    "description": "The time of day to check medication for.",
                }
            },
            "required": ["time_of_day"],
        },
    },
    {
        "name": "describe_location",
        "description": "Identify and describe the current location based on visual features of the scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_features": {
                    "type": "string",
                    "description": "Comma-separated visual features observed (e.g., 'stove, refrigerator, table').",
                }
            },
            "required": ["room_features"],
        },
    },
    {
        "name": "alert_caregiver",
        "description": "Send a silent alert to the registered caregiver when the patient needs help.",
        "parameters": {
            "type": "object",
            "properties": {
                "alert_type": {
                    "type": "string",
                    "enum": ["confusion", "wandering", "medication_missed", "fall_risk"],
                    "description": "The type of alert.",
                },
                "details": {
                    "type": "string",
                    "description": "Additional context about the situation.",
                },
            },
            "required": ["alert_type", "details"],
        },
    },
    {
        "name": "get_agenda",
        "description": "Get the patient's schedule and appointments for a specific day of the week, or all days if not specified.",
        "parameters": {
            "type": "object",
            "properties": {
                "day_of_week": {
                    "type": "string",
                    "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                    "description": "The day to check. Omit to get all days.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "save_preference",
        "description": "Save or update a patient preference, habit, or personal fact. Categories: food, music, hobby, routine, health, social.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Preference category (e.g., 'food', 'music', 'hobby', 'routine').",
                },
                "key": {
                    "type": "string",
                    "description": "The specific preference key (e.g., 'favorite_meal', 'morning_drink').",
                },
                "value": {
                    "type": "string",
                    "description": "The preference value (e.g., 'coffee with milk, no sugar').",
                },
            },
            "required": ["category", "key", "value"],
        },
    },
    {
        "name": "get_preferences",
        "description": "Retrieve patient preferences and personal facts, optionally filtered by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (e.g., 'food', 'music'). Omit to get all.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_routine",
        "description": "Get the patient's daily routine steps for a time of day (morning, afternoon, evening, night).",
        "parameters": {
            "type": "object",
            "properties": {
                "time_of_day": {
                    "type": "string",
                    "enum": ["morning", "afternoon", "evening", "night"],
                    "description": "The time of day to get the routine for.",
                }
            },
            "required": ["time_of_day"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date, time, and day of the week. Use this to know what time of day it is before checking medication or agenda.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


@dataclass
class ToolCallingConfig:
    """Configuration for tool calling training and inference."""
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"
    tools: list[dict] = field(default_factory=lambda: TOOL_DEFINITIONS)

    @property
    def system_prompt(self) -> str:
        tools_json = json.dumps(self.tools, indent=2)
        return (
            "You are a caring memory companion for a person with Alzheimer's disease. "
            "You help them recognize people, remember important information, and stay oriented. "
            "You speak in warm, simple language. Never fabricate information about people or medication.\n\n"
            "You have access to the following tools:\n"
            f"{tools_json}\n\n"
            "When you need information, use a tool by responding with:\n"
            f"{self.tool_call_start}\n"
            '{"name": "<tool_name>", "arguments": {<args>}}\n'
            f"{self.tool_call_end}\n\n"
            "After receiving a tool response, provide a warm, helpful answer to the patient."
        )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@dataclass
class FaceEmbeddingConfig:
    """Face detection & embedding configuration — mobile-optimized."""
    # MobileFaceNet: 1MB model, 128-d embeddings, runs on ARM CPUs in <50ms
    detector: str = "mediapipe"  # Google MediaPipe Face Detection — <1MB, works offline
    embedder: str = "mobilefacenet"  # MobileFaceNet — 1MB, 128-d, ONNX-ready
    embedding_dim: int = 128  # MobileFaceNet output dim (NOT 512 like ArcFace)
    detection_confidence: float = 0.7
    # ONNX Runtime for mobile inference (CPU-only, no CUDA needed)
    onnx_runtime: bool = True
    # Export paths for mobile deployment
    detector_onnx_path: str = "models/face_detector.onnx"
    embedder_onnx_path: str = "models/mobilefacenet.onnx"


@dataclass
class DatabaseConfig:
    """SQLite database configuration."""
    db_path: str = "data/companion.db"
    face_similarity_threshold: float = 0.85
    embedding_dim: int = 128  # MobileFaceNet 128-d (mobile-optimized)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Inference-time configuration."""
    model_path: str = "outputs/final"
    tcs_checkpoint: str = "outputs/final/tcs_weights.pt"
    compression_ratio: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Presets (phase-specific defaults)
# ---------------------------------------------------------------------------

PHASE_PRESETS: dict[str, dict] = {
    "tcs-pretrain": {
        "lr": 1e-3,
        "max_steps": 2000,
        "save_steps": 500,
        "eval_steps": 500,
        "warmup_steps": 100,
        "distill_alpha": 1.0,  # Pure KL, no CE
    },
    "tcs-lora": {
        "lr": 2e-4,
        "max_steps": 10000,
        "save_steps": 1000,
        "eval_steps": 1000,
        "warmup_steps": 200,
        "distill_alpha": 0.5,
    },
    "tool-calling": {
        "lr": 3e-5,
        "max_steps": 3000,  # ~1.5 epochs on 25K examples to avoid memorization
        "save_steps": 500,
        "eval_steps": 500,
        "warmup_steps": 150,
        "distill_alpha": 0.0,  # Pure CE on tool calling output
    },
}


def apply_phase_preset(cfg: TrainingConfig) -> TrainingConfig:
    """Apply phase-specific defaults to a training config (only for unset values)."""
    preset = PHASE_PRESETS.get(cfg.phase, {})
    for key, value in preset.items():
        # Only apply preset if the config still has the dataclass default
        default_field = TrainingConfig.__dataclass_fields__[key]
        if getattr(cfg, key) == default_field.default:
            setattr(cfg, key, value)
    return cfg
