# Architecture: Gemma 4 Memory Companion

## Overview

This document describes the technical architecture for compressing Gemma 4 E2B using Token Compression Sub-network (TCS) + LoRA, with native function calling for an Alzheimer memory companion.

## 1. Token Compression Sub-network (TCS)

### Position in the Model

```
Gemma4 Embedding → [TCS inserted here] → Gemma4 Transformer Blocks → LM Head
```

The TCS sits between the embedding layer and the first transformer block. It receives the full token sequence `(B, N, D)` and outputs a compressed sequence `(B, N/r, D)` where `r` is the compression ratio.

### Architecture

```python
TCS = Sequential(
    Conv1d(d_model, d_model, kernel_size=7, stride=4, padding=3),  # compress
    GELU(),
    Conv1d(d_model, d_model, kernel_size=1),  # point-wise refine
    RMSNorm(d_model),
)
```

**Why Conv1D instead of attention-based compression:**
- O(N) complexity vs O(N²) for attention-based methods
- Local n-gram patterns are sufficient for token fusion
- Deterministic compression ratio (predictable memory usage)
- Fully differentiable — gradients flow through compression

### Compression Ratios

| Ratio | Input N=512 | Compressed M | Attention FLOPs | Use Case |
|---|---|---|---|---|
| r=2 | 512 | 256 | 4× reduction | Detailed responses |
| r=4 | 512 | 128 | 16× reduction | Balanced (default) |
| r=8 | 512 | 64 | 64× reduction | Fast/urgent responses |

## 2. LoRA Integration

### Why LoRA is Necessary

The Gemma 4 transformer blocks were pre-trained on full-length token sequences. When we compress tokens 4×, the attention patterns change fundamentally:
- Key-Value distributions shift (fewer, denser tokens)
- Positional relationships change (compressed positions ≠ original positions)
- Self-attention must learn to attend over "super-tokens"

LoRA adapts the attention layers to handle compressed tokens without destroying the original knowledge.

### LoRA Configuration

```python
LoraConfig(
    r=16,                          # Rank — good balance for E2B
    lora_alpha=32,                 # Scaling: alpha/r = 2.0
    target_modules=[
        "q_proj", "k_proj",       # Attention must adapt to compressed tokens
        "v_proj", "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Parameter budget:**
- Gemma 4 E2B: ~2B params (frozen)
- TCS: ~3M params (trainable)
- LoRA: ~20M params (trainable)
- **Total trainable: ~23M (1.1% of model)**

## 3. Function Calling / Tool Use

### Tool Schema

The model is trained to output structured function calls in Gemma 4's native tool calling format:

```json
{
  "tools": [
    {
      "name": "read_person",
      "description": "Look up a person by face embedding ID. Returns name, relationship, memories.",
      "parameters": {
        "face_id": {"type": "string", "description": "The face embedding match ID"}
      }
    },
    {
      "name": "write_encounter",
      "description": "Log an encounter with a person for future reference.",
      "parameters": {
        "person_name": {"type": "string"},
        "context": {"type": "string"},
        "timestamp": {"type": "string"}
      }
    },
    {
      "name": "get_medication",
      "description": "Get medication schedule for current time.",
      "parameters": {
        "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening", "night"]}
      }
    },
    {
      "name": "describe_location",
      "description": "Identify the current location from visual context.",
      "parameters": {
        "room_features": {"type": "string"}
      }
    },
    {
      "name": "alert_caregiver",
      "description": "Send a silent alert to the registered caregiver.",
      "parameters": {
        "alert_type": {"type": "string", "enum": ["confusion", "wandering", "medication_missed", "fall_risk"]},
        "details": {"type": "string"}
      }
    }
  ]
}
```

### SQLite Database Schema

```sql
CREATE TABLE persons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    relationship TEXT,
    bio TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE face_embeddings (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    embedding BLOB NOT NULL,  -- MobileFaceNet 128-d float32 (~512 bytes per face)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    content TEXT NOT NULL,
    memory_type TEXT,  -- 'shared_experience', 'preference', 'routine'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medication_schedule (
    id TEXT PRIMARY KEY,
    medication_name TEXT NOT NULL,
    description TEXT,  -- "small white round pill"
    time_of_day TEXT,
    dosage TEXT,
    notes TEXT
);

CREATE TABLE encounters (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    context TEXT,
    location TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Vector Search for Face Matching

```python
# Using sqlite-vec extension with MobileFaceNet 128-d embeddings
# Threshold: cosine similarity > 0.85 → match (distance < 0.15)
# MobileFaceNet outputs normalized 128-d float32 vectors (~512 bytes each)
SELECT p.name, p.relationship, p.bio,
       vec_distance_cosine(fe.embedding, ?) as distance
FROM face_embeddings fe
JOIN persons p ON fe.person_id = p.id
WHERE distance < 0.15  -- 1 - 0.85 threshold
ORDER BY distance ASC
LIMIT 1;
```

**Why 128-d (MobileFaceNet) instead of 512-d (ArcFace):**
- 4× less storage per face embedding (512 bytes vs 2048 bytes)
- 4× faster cosine distance computation
- MobileFaceNet achieves 99.2% accuracy on LFW — sufficient for ~50 known faces
- Entire model is 1MB vs 250MB for ArcFace-R100

## 4. Training Pipeline

### Phase 1: TCS Pre-training (~30 min on 2× T4)

**Goal:** Teach the compression layer to produce meaningful super-tokens.

- **Trainable:** TCS only (~3M params)
- **Frozen:** All Gemma 4 params
- **Loss:** Self-distillation — KL divergence between compressed and uncompressed paths
- **Implementation detail:** Teacher logits are sequence-aligned to compressed positions via index sampling, and KL (including log-normalizer/logsumexp) is computed in vocabulary chunks to reduce peak VRAM.
- **Memory detail:** In `mode="both"`, teacher logits are aligned to student length before returning from the model, preventing large full-length logits from being materialized in DDP output conversion.
- **Data:** General text (wikitext or similar)

### Phase 2: TCS + LoRA Joint Training (~2-4h on 2× T4)

**Goal:** Adapt attention to work with compressed tokens while maintaining quality.

- **Trainable:** TCS + LoRA adapters (~23M params)
- **Frozen:** Gemma 4 base weights
- **Loss:** α · KL(compressed ‖ uncompressed) + (1-α) · CE(compressed, index-aligned targets)
- **Implementation detail:** Distillation reuses index-based sequence alignment for both teacher logits and hard labels, while chunked KL/logsumexp computation keeps memory stable at 16GB GPU scale.
- **Data:** General text + conversational data

### Phase 3: Tool Calling Fine-tune (~1-2h on 2× T4)

**Goal:** Train the model to correctly invoke SQLite tools in the memory companion context.

- **Trainable:** TCS + LoRA (same params, lower LR)
- **Loss:** CE on structured function-calling tokens after aligning masked labels to compressed positions
- **Implementation detail:** Assistant-only labels remain masked with `-100`, and the remaining targets are mapped to compressed positions by deterministic index sampling before cross-entropy.
- **Data:** Synthetic conversations with tool calls (generated via GPT-4/Claude)

### Memory Budget (2× T4, Kaggle)

**VRAM:** 2× 16GB = 32GB total
**System RAM:** 30GB
**Disk:** ~19.5GB available

| Component | VRAM |
|---|---|
| Gemma 4 E2B (int8/bf16) | ~4GB |
| LoRA adapters | ~0.1GB |
| TCS | ~0.02GB |
| Activations + gradients | ~6GB |
| Optimizer states (AdamW, LoRA only) | ~0.2GB |
| **Total per GPU (DDP)** | **~10GB** |
| **Headroom** | **~6GB** |

**Disk management (19.5GB constraint):**
- Base model cached: ~4GB (int8)
- Dataset cache: ~2GB
- Keep only last 3 checkpoints during training (~150MB each)
- After each phase: delete checkpoints, keep only `final/` (~50MB)
- Total disk per phase: ~6.5GB → fits comfortably

## 5. Inference Flow (Mobile-First, Fully Offline)

### Face Detection & Embedding Pipeline

All face processing runs **on-device** using lightweight mobile models:

| Component | Model | Size | Latency (ARM CPU) | Output |
|---|---|---|---|---|
| Face Detection | **MediaPipe Face Detection** | <1MB | ~10ms | Bounding box |
| Face Embedding | **MobileFaceNet** (ONNX) | ~1MB | ~30ms | 128-d vector |
| Vector Search | **sqlite-vec** | 0 (SQLite ext) | ~1ms | Nearest match |

**Why NOT CLIP/SigLIP/ArcFace:**
- CLIP ViT-B: ~350MB — too large for mobile
- ArcFace ResNet100: ~250MB, 512-d — overkill
- MobileFaceNet: **1MB, 128-d, 99.2% LFW accuracy** — designed for mobile

### Full Inference Pipeline

```
1. Camera captures frame (Android CameraX / iOS AVFoundation)
2. MediaPipe detects face → bounding box + landmarks (<10ms)
3. Crop & align face (affine transform using landmarks)
4. MobileFaceNet ONNX → 128-d embedding (~30ms on ARM)
5. sqlite-vec: cosine search face_embeddings (threshold 0.85)
6. If match: retrieve person info + memories from SQLite
7. Compose prompt: system + person context + user query
8. TCS compresses tokens → Gemma 4 + LoRA generates response
9. If response contains tool_call: execute against SQLite, feed back
10. Final response → on-device TTS → speak to patient
```

**Total face pipeline: <50ms on modern ARM CPU (Snapdragon 8xx / Apple A15+)**

## 6. Multi-Scale Runtime Selection

The same checkpoint supports multiple compression ratios:

| Mode | Ratio | Latency | When |
|---|---|---|---|
| **Urgent** | r=8 | <1s | "Where am I?", "Who is this?" |
| **Normal** | r=4 | ~2s | General conversation, medication reminders |
| **Detailed** | r=2 | ~4s | Telling stories, detailed scene description |

Selection is automatic based on query classification or manual via caregiver app.
