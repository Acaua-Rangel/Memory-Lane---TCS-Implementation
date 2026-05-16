# Gemma 4 Memory Companion

**An AI-powered memory assistant for people with Alzheimer's and dementia, built on Gemma 4 E2B with Token Compression Sub-network (TCS) + LoRA.**

> *Google DeepMind Gemma 4 Hackathon — Health & Sciences / Digital Equity*

---

## The Problem

55 million people worldwide live with dementia. They lose the ability to recognize loved ones, remember where they are, or recall daily routines. Caregivers spend exhausting hours repeating the same information. Current solutions require expensive specialized hardware or constant internet access.

## The Solution

A lightweight AI companion that runs **offline on edge devices** — a phone worn as a pendant or placed nearby — that:

1. **Recognizes faces** via camera → retrieves memories about that person from a local SQLite database
2. **Understands scenes** → orients the patient ("You're in your kitchen. Your bedroom is to the left")
3. **Manages medication** → reminds and visually confirms the right pill
4. **Alerts caregivers** → silently notifies family when the patient is confused or wandering
5. **Speaks gently** → all interactions are voice-based, no screen required

## Technical Innovation

| Technique | Purpose |
|---|---|
| **Token Compression Sub-network (TCS)** | Strided Conv1D compresses token sequences 4× before attention → 16× FLOP reduction |
| **LoRA (Low-Rank Adaptation)** | Adapts attention layers to compressed tokens without destroying Gemma 4's knowledge |
| **Self-Distillation** | Gemma 4 uncompressed path teaches compressed path — no external teacher needed |
| **Native Function Calling** | Model calls SQLite tools to read/write patient data, face embeddings, medication schedules |
| **Face Embeddings + Vector Search** | MobileFaceNet (1MB ONNX, 128-d) + sqlite-vec for mobile face matching |
| **Multi-Scale Compression** | r=8 for fast responses, r=4 for balanced, r=2 for detailed — selected at runtime |

## Architecture

```
Camera Input (face/scene)
        │
        ├──→ MediaPipe Face Detection (<1MB, <10ms on ARM)
        │         │
        │         ▼
        │    MobileFaceNet ONNX (1MB, 128-d, <30ms on ARM)
        │         │
        │         ▼
        │    sqlite-vec cosine search (threshold 0.85)
        │         │
        │         ▼
        │    face_id → person name/relationship
        │
        ▼
┌─────────────────────────┐
│  Gemma 4 E2B (FROZEN)   │
│  Text Embedding          │ ← prompt: "The person in front of you is {name}, {relationship}..."
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  TCS (TRAINABLE ~3M)    │
│  Conv1d stride=4         │
│  N tokens → N/4 tokens   │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Gemma 4 Transformer     │
│  Blocks + LoRA adapters  │
│  (TRAINABLE ~20M, 0.6%) │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Function Calling Head   │
│  → read_person(face_id)  │
│  → write_encounter(...)  │
│  → get_medication(time)  │
│  → describe_scene(img)   │
└─────────────────────────┘
```

## Training Pipeline

**Target hardware:** 2× NVIDIA T4 (Kaggle) — 2×16GB VRAM, 30GB RAM, **19.5GB disk**

| Phase | What | Duration |
|---|---|---|
| 1. TCS Pre-training | Train compression layer only (Gemma 4 frozen) | ~30 min |
| 2. TCS + LoRA | Joint training with self-distillation loss | ~2-4h |
| 3. Tool Calling Fine-tune | Synthetic data for SQLite function calling | ~1-2h |

**Disk strategy:** Keep only last 3 checkpoints; clean up intermediate checkpoints after each phase.

## Quick Start

```bash
# Install
poetry install

# Generate synthetic training data
poetry run generate-data

# Train Phase 1: TCS only
poetry run train --phase tcs-pretrain

# Train Phase 2: TCS + LoRA on both T4s
torchrun --standalone --nproc_per_node=2 -m src.train_ddp --phase tcs-lora

# Train Phase 3: Tool calling fine-tune on both T4s
torchrun --standalone --nproc_per_node=2 -m src.train_ddp --phase tool-calling

# Optional: single-GPU Unsloth 4-bit path
python -m src.train_unsloth --phase tool-calling --load-in-4bit

# Unsloth-accelerated training (single GPU, 2× faster LoRA, 60% less VRAM)
python -m src.train_unsloth --phase tcs-lora --load-in-4bit

# Export to LiteRT for mobile deployment
poetry run export-litert --model-dir outputs/tool_calling/final

# Inference (with intelligent task routing)
poetry run infer --prompt "Who is this person?" --interactive
```

## Special Technology Tracks

This project targets **three Special Technology Track prizes** simultaneously:

| Track | Integration | Key File |
|---|---|---|
| **Cactus** ($10K) | Intelligent task router that routes between MobileFaceNet, SQLite, rule-based logic, and Gemma 4 at variable compression (r=2/4/8) | `src/routing.py` |
| **Unsloth** ($10K) | 2× faster LoRA fine-tuning with 60% less VRAM via Unsloth's fused kernels | `src/train_unsloth.py` |
| **LiteRT** ($10K) | Export TCS + merged LoRA model to Google AI Edge LiteRT format for on-device inference | `src/export_litert.py` |

### Cactus: Intelligent Routing
The `TaskRouter` classifies inputs and routes to the optimal model:
```
Face detected → MobileFaceNet (1MB) → sqlite-vec → LLM with context
Simple greeting → Gemma 4 + TCS r=8 (fastest, 64× FLOP savings)
Medication query → SQLite direct (skip LLM entirely)
Complex memory → Gemma 4 + TCS r=2 (highest quality)
Caregiver alert → Rule-based (instant, no LLM latency)
```

### Unsloth: Faster Training
```bash
# Standard multi-GPU training: ~4h for Phase 2
torchrun --standalone --nproc_per_node=2 -m src.train_ddp --phase tcs-lora

# Unsloth training: ~2h for Phase 2 on a single T4 (2× speedup)
python -m src.train_unsloth --phase tcs-lora --load-in-4bit
```

`src.train_unsloth` is a single-GPU entrypoint. If you want to occupy both Kaggle T4s, launch `src.train_ddp` with `torchrun`.

### LiteRT: Mobile Deployment
```bash
# Export full pipeline for Android/iOS
poetry run export-litert --model-dir outputs/tool_calling/final --output-dir export/

# Generates:
#   export/tcs/tcs_compression.tflite    (~12MB)
#   export/llm/gemma4_memory_companion.tflite  (~2.5GB int8)
#   export/face/mobilefacenet.onnx       (~1MB)
#   export/deployment_manifest.json
```

## Project Structure

```
gemma4-memory-companion/
├── .github/copilot-instructions.md    # Copilot agent instructions
├── .instructions/                     # Topic-specific Copilot skills
├── docs/                              # Architecture & training docs
├── src/
│   ├── config.py                      # All configurations
│   ├── model.py                       # TCS + LoRA wrapper for Gemma 4
│   ├── losses.py                      # Self-distillation + tool calling losses
│   ├── data.py                        # Dataset loading & synthetic generation
│   ├── train.py                       # Single-GPU training
│   ├── train_ddp.py                   # Multi-GPU DDP training
│   ├── train_unsloth.py               # Unsloth-accelerated LoRA training
│   ├── routing.py                     # Cactus task router
│   ├── export_litert.py               # LiteRT mobile export pipeline
│   ├── inference.py                   # Inference with tool execution
│   ├── tools/                         # SQLite tool definitions
│   └── utils/                         # Checkpointing, distributed helpers
├── scripts/                           # Kaggle setup & launch scripts
├── tests/                             # Unit tests
└── data/                              # Generated/cached datasets
```

## License

Apache 2.0 — Following Gemma 4 model license terms.
