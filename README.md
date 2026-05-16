# Memory Lane — When AI Remembers So They Don't Have To

> **Google DeepMind Gemma 4 Hackathon** · Team Memory Lane  
> Acauã Rangel Brazil & Juan Benjamin Suzart · Salvador, Bahia, Brazil 🇧🇷

---

**Lane** is an AI companion that lives on a patient's phone and never forgets. When a person with Alzheimer's sees a familiar face but can't place it, Lane whispers:

> *"Essa é a Maria, sua neta. Ela tem 8 anos e adora desenhar. Domingo passado vocês pintaram juntos na varanda."*

It runs **100% offline**. No cloud. No subscription. Just a phone.

---

## Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Architecture](#architecture)
- [Technical Innovations](#technical-innovations)
  - [Token Compression Sub-network (TCS)](#1-token-compression-sub-network-tcs)
  - [Self-Distillation](#2-self-distillation-no-external-teacher)
  - [Intelligent Task Routing](#3-intelligent-task-routing-cactus)
  - [Native Tool Calling via SQLite](#4-native-tool-calling-via-sqlite)
  - [Face Recognition Pipeline](#5-face-recognition-pipeline)
- [Repository Structure](#repository-structure)
- [Training Pipeline](#training-pipeline)
- [Mobile App](#mobile-app-lane_app)
- [Quick Start](#quick-start)
- [Hackathon Prize Tracks](#hackathon-prize-tracks)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## The Problem

**55 million people** worldwide live with dementia. Every 3 seconds, someone new is diagnosed.

They lose the ability to:
- Recognize their own children and grandchildren
- Remember where they are or what day it is
- Follow their medication schedule
- Recall the shared memories that define their relationships

**Current solutions fail because they:**
- Require constant internet — unreliable for elderly users
- Are passive apps — the patient must remember to check them
- Don't adapt to daily life in real-time
- Raise serious privacy concerns by storing memories in the cloud

---

## The Solution

Lane routes every interaction through the optimal pipeline — from face recognition to medication reminders — entirely on-device:

| Capability | How it works | Route |
|---|---|---|
| **Face Recognition** | Camera detects face → MobileFaceNet embedding → sqlite-vec cosine search → Lane narrates who it is | MobileFaceNet → SQLite → Gemma 4 r=4 |
| **Active Memory** | Lane automatically records encounters and conversations via tool calling | Gemma 4 r=2 (quality) |
| **Medication** | Tracks schedule, describes each pill visually, confirms intake | SQLite direct (300ms) |
| **Orientation** | "Onde estou?" → Lane describes the room and gives directions | Gemma 4 r=4 (balanced) |
| **Routine & Agenda** | Recalls daily schedule, preferences, habits | Gemma 4 + tool calling |
| **Caregiver Alerts** | Silently notifies family when patient is confused or wandering | Rule-based + push notification |

**Interaction design:** voice-first, wake word "Oi Lane", gentle language, proactive — Lane notices a new face and speaks up without being asked.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PATIENT'S PHONE                          │
│                                                                 │
│  ┌──────────┐   ┌────────────────────────────────────────────┐  │
│  │ Camera   │──▶│ MediaPipe Face Detection  (<1MB, <10ms)    │  │
│  └──────────┘   │              │                             │  │
│                 │              ▼                             │  │
│  ┌──────────┐   │  MobileFaceNet ONNX  (1MB, 128-d, <30ms)  │  │
│  │   Mic    │   │              │                             │  │
│  │  (STT)   │   │              ▼                             │  │
│  └────┬─────┘   │  sqlite-vec cosine search → face_id        │  │
│       │         └───────────────────────────┬────────────────┘  │
│       │                                     │                   │
│       ▼                                     ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  TaskRouter  (Cactus)                    │   │
│  │   Classifies input → routes to optimal pipeline          │   │
│  └──┬──────────┬──────────┬──────────┬──────────┬───────────┘   │
│     │          │          │          │          │               │
│     ▼          ▼          ▼          ▼          ▼               │
│  LLM r=8   LLM r=4   LLM r=2   SQLite    Rule-based           │
│  (fast)  (balanced) (quality)  (direct)   (alerts)             │
│     │          │          │          │          │               │
│     └──────────┴──────────┴──────────┴──────────┘               │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Gemma 4 E2B (frozen) + TCS (3M params) + LoRA (20M)   │   │
│  │   int8 quantized via LiteRT / ONNX Runtime               │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SQLite  (WAL mode, offline, zero-config)                │   │
│  │  persons │ faces │ memories │ medications │ encounters   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│               Speaker ← Google Offline TTS                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Innovations

### 1. Token Compression Sub-network (TCS)

**Original research from Acauã Rangel** — preprint on arXiv for the GPT-2 validation phase.

A trainable Conv1D layer inserted between Gemma 4's embedding layer and its transformer blocks. It compresses N tokens into N/r tokens before self-attention:

```
Input Embeddings  (B, 512, 2048)
        │
        ▼
Conv1d(stride=4) → GELU → Conv1d(1×1) → RMSNorm
        │
        ▼
Compressed        (B, 128, 2048)   ← 4× fewer tokens
        │
        ▼
Gemma 4 Transformer Blocks         ← 16× fewer attention FLOPs
```

| Ratio | Tokens (N=512) | Attention FLOPs | Latency | Use case |
|-------|---------------|-----------------|---------|----------|
| r=2 | 256 | 4× reduction | ~4s | Detailed memory recall |
| r=4 | 128 | **16× reduction** | ~2s | General conversation (default) |
| r=8 | 64 | 64× reduction | <1s | Greetings, urgent alerts |

**Parameter budget:**

| Component | Parameters | Status |
|---|---|---|
| Gemma 4 E2B base | ~2B | Frozen |
| TCS (Conv1D) | ~3M | Trainable |
| LoRA (rank=16 on q/k/v/o) | ~20M | Trainable |
| **Total trainable** | **~23M (1.1%)** | |

The TCS is **model-agnostic** — it can be applied to any transformer, making it a general contribution to efficient LLM inference on edge devices.

### 2. Self-Distillation (No External Teacher)

The uncompressed Gemma 4 path (full tokens, no LoRA) serves as the teacher. No separate larger model is needed:

```
Loss = α · T² · KL(student ‖ teacher)  +  (1-α) · CE(student, labels)
```

| Phase | α | Loss focus |
|---|---|---|
| Phase 1 — TCS pretrain | 1.0 | Pure KL (teach compression) |
| Phase 2 — TCS + LoRA | 0.5 | Balanced KL + CE |
| Phase 3 — Tool calling | 0.0 | Pure CE (structured output) |

### 3. Intelligent Task Routing (Cactus)

A zero-overhead `TaskRouter` classifies every input with regex patterns (bilingual PT+EN) and routes to the cheapest pipeline that meets the latency budget:

```
Face detected     → MobileFaceNet → sqlite-vec → LLM r=4    [500ms]
"Bom dia"         → Gemma 4 TCS r=8  (fastest)              [500ms]
"Que remédio?"    → SQLite direct     (skip LLM entirely)    [300ms]
"Quem é a Maria?" → Gemma 4 TCS r=2  (highest quality)     [2000ms]
"Estou perdido"   → Rule-based + caregiver alert             [100ms]
```

### 4. Native Tool Calling via SQLite

Gemma 4's native function calling format drives all memory operations. Lane **actively writes** to the database — every encounter, every medication confirmation, every conversation fragment is stored for future recall.

**9 tools:**

| Tool | Purpose |
|---|---|
| `read_person(face_id)` | Retrieve name, relationship, bio, memories |
| `write_encounter(person, context)` | Log visit with context and timestamp |
| `get_medication(time_of_day)` | Get schedule + visual pill description |
| `describe_location(features)` | Identify room from visual features |
| `alert_caregiver(type, details)` | Silent push notification to family |
| `get_agenda(day_of_week)` | Retrieve appointments and events |
| `save_preference(category, key, value)` | Store patient habits and preferences |
| `get_preferences(category)` | Recall stored preferences |
| `get_routine(time_of_day)` | Get step-by-step daily routine |

**8 SQLite tables:** `persons`, `face_embeddings`, `memories`, `medication_schedule`, `encounters`, `locations`, `agenda`, `preferences`

### 5. Face Recognition Pipeline

Completely independent from the LLM — runs in <50ms on ARM:

| Component | Model | Size | Latency |
|---|---|---|---|
| Detection | MediaPipe Face Detection | <1MB | ~10ms |
| Embedding | MobileFaceNet ONNX | 1MB | ~30ms |
| Search | sqlite-vec cosine (threshold 0.85) | 0 (SQLite ext) | ~1ms |

**Why MobileFaceNet instead of ArcFace:** 4× smaller (1MB vs 250MB), 4× faster cosine distance, 99.2% accuracy on LFW — sufficient for a database of ~50 known faces.

---

## Repository Structure

```
memory_lane/
│
├── train_code/                    # ML training pipeline (Python)
│   ├── src/
│   │   ├── config.py              # All configurations (model, training, tools)
│   │   ├── model.py               # Gemma 4 + TCS + LoRA wrapper
│   │   ├── losses.py              # Self-distillation + tool calling losses
│   │   ├── data.py                # Dataset + synthetic data generation
│   │   ├── train.py               # Single-GPU training loop
│   │   ├── train_ddp.py           # Multi-GPU DDP training (Accelerate)
│   │   ├── train_unsloth.py       # Unsloth-accelerated 4-bit LoRA
│   │   ├── routing.py             # TaskRouter + RoutedCompanion
│   │   ├── inference.py           # Inference with tool execution loop
│   │   ├── export_litert.py       # LiteRT/TFLite mobile export
│   │   └── tools/
│   │       └── sqlite_tools.py    # 9 SQLite tools + execution engine
│   ├── docs/
│   │   ├── PROJECT.md             # Full project description
│   │   ├── architecture.md        # TCS design + training details
│   │   ├── training_plan.md       # Kaggle hardware constraints
│   │   └── KAGGLE_WRITEUP.md      # Academic writeup
│   ├── tests/                     # Unit tests (no GPU required)
│   ├── scripts/
│   │   └── generate_tool_data.py  # Synthetic data generator
│   └── pyproject.toml
│
└── lane_app/                      # React Native mobile app (Expo 54)
    ├── src/
    │   ├── types/index.ts         # All TypeScript types
    │   ├── config/
    │   │   ├── constants.ts       # Compression ratios, thresholds, tool definitions
    │   │   └── sampleData.ts      # Seed data (persons, medications, locations)
    │   ├── database/
    │   │   ├── LaneDatabase.ts    # Main database class (singleton)
    │   │   ├── schema.ts          # SQL DDL — 8 tables + indexes
    │   │   └── repositories/      # PersonRepository, MedicationRepository, ...
    │   ├── tools/
    │   │   ├── ToolExecutor.ts    # Executes the 9 tools against SQLite
    │   │   └── toolParser.ts      # Parses Gemma 4 <tool_call> JSON output
    │   ├── engine/
    │   │   └── GemmaEngine.ts     # LiteRT bridge + MockEngine for dev
    │   ├── face/
    │   │   ├── FaceDetector.ts    # Vision Camera frame processor hook
    │   │   ├── FaceEmbedder.ts    # MobileFaceNet ONNX (128-d embeddings)
    │   │   ├── FaceIdentifier.ts  # Cosine search against SQLite
    │   │   └── FacePipeline.ts    # Orchestrates detect → embed → identify
    │   ├── routing/
    │   │   ├── TaskRouter.ts      # Port of Python routing.py (bilingual regex)
    │   │   └── patterns.ts        # PT-BR + EN regex patterns
    │   ├── agent/
    │   │   └── LaneAgent.ts       # Main orchestrator (port of RoutedCompanion)
    │   ├── voice/
    │   │   ├── WakeWordDetector.ts  # Always-listening "Oi Lane" detection
    │   │   ├── SpeechToText.ts    # expo-speech-recognition wrapper
    │   │   └── TextToSpeech.ts    # expo-speech wrapper (elderly mode)
    │   ├── alerts/
    │   │   └── CaregiverAlertService.ts  # Offline-first push notifications
    │   ├── store/
    │   │   ├── agentStore.ts      # Zustand — agent state + face embedding cache
    │   │   └── patientStore.ts    # Zustand — patient profile + caregivers
    │   └── hooks/
    │       ├── useAgent.ts        # Main hook — initializes and exposes the agent
    │       ├── useVoice.ts        # STT + TTS + wake word in one hook
    │       └── useAlertSync.ts    # Auto-flush offline alerts on reconnect
    ├── App.tsx                    # Entry point (placeholder for Juan's screens)
    ├── app.json                   # Expo config + permissions + plugins
    └── package.json               # Expo 54 + all dependencies
```

---

## Training Pipeline

**Target hardware:** Kaggle 2× NVIDIA T4 (32GB VRAM total, ~19.5GB disk)

| Phase | What trains | Duration | Backend | Loss |
|---|---|---|---|---|
| **1 — TCS Pretrain** | TCS only (~3M params) | ~30 min | Accelerate DDP | KL (α=1.0) |
| **2 — TCS + LoRA** | TCS + LoRA (~23M params) | ~2–4h | Accelerate DDP or Unsloth | KL + CE (α=0.5) |
| **3 — Tool Calling** | TCS + LoRA (same, lower LR) | ~1–2h | Accelerate DDP or Unsloth | CE (α=0.0) |
| **4 — Export** | — | ~10 min | ai_edge_torch | — |

**Disk strategy:** rolling 3-checkpoint window during training, delete intermediates after each phase — peak disk usage ~7GB (leaves 12GB headroom).

**Validation targets:** KL < 0.5 (Phase 1) · Perplexity < 15.0 (Phase 2) · Tool call accuracy > 90% (Phase 3)

---

## Mobile App (`lane_app/`)

Built with **Expo SDK 54 + React 19 + TypeScript** (New Architecture enabled).

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `expo-sqlite` | ~16.0.10 | On-device SQLite with WAL mode |
| `expo-speech` | ~14.0.8 | Offline TTS (elderly rate: 0.75×) |
| `expo-speech-recognition` | ^3.1.3 | Offline STT (replaces deprecated voice/voice) |
| `expo-camera` | ~17.0.10 | Camera access for face pipeline |
| `expo-notifications` | ~0.32.17 | Caregiver push alerts |
| `expo-network` | ~8.0.8 | Connectivity detection for alert queue |
| `onnxruntime-react-native` | ^1.24.3 | MobileFaceNet ONNX inference |
| `react-native-vision-camera` | ^5.0.9 | High-performance camera frame processor |
| `zustand` | ^5.0.13 | Lightweight global state |

### How Juan uses the agent system

```typescript
import { useAgent, useVoice, useAlertSync } from './src';

function PatientScreen() {
  const { isReady, processText, processFaceFrame } = useAgent();

  const { speak, startListening, startWakeWord } = useVoice(
    async (transcript) => {
      const response = await processText(transcript);
      if (response?.speech) await speak(response.speech);
    }
  );

  // useAlertSync flushes offline alerts automatically on reconnect
  const { pendingCount } = useAlertSync(alertService);

  // isReady is true when database + model + face engine are all loaded
}
```

### Agent initialization flow

```
LaneDatabase.open()          → SQLite schema + seed data
MobileFaceNetEmbedder.load() → ONNX model from assets/
GemmaEngine.load()           → LiteRT model from assets/
LaneAgent constructed        → router + facePipeline + engine + tools + alerts
```

---

## Quick Start

### Training (Python — Kaggle or local GPU)

```bash
# Install dependencies
cd train_code
poetry install

# Generate synthetic training data (~25K examples)
poetry run generate-data

# Phase 1 — TCS only (~30 min, single GPU)
poetry run train --phase tcs-pretrain

# Phase 2 — TCS + LoRA (~2–4h, 2× T4)
torchrun --standalone --nproc_per_node=2 -m src.train_ddp --phase tcs-lora

# Alternative: Unsloth 4-bit single-GPU (2× faster, 60% less VRAM)
python -m src.train_unsloth --phase tcs-lora --load-in-4bit

# Phase 3 — Tool calling (~1–2h)
torchrun --standalone --nproc_per_node=2 -m src.train_ddp --phase tool-calling

# Export to LiteRT for mobile deployment
poetry run export-litert --model-dir outputs/tool_calling/final

# Test inference with intelligent routing
poetry run infer --prompt "Quem é essa pessoa?" --interactive
```

### Mobile App (React Native — Expo)

```bash
cd lane_app

# Install dependencies (already done via expo install)
npm install

# Copy exported models to assets
cp ../train_code/outputs/final/*.tflite assets/models/
cp ../train_code/outputs/final/mobilefacenet.onnx assets/models/

# Start development server
npx expo start --android
# or
npx expo start --ios

# Build for device via EAS
npx eas build --platform android --profile preview
```

> **Development mode:** When `__DEV__ === true`, the app uses `MockGemmaEngine` and `MockFaceEmbedder` — no real models needed to build and test the UI and routing logic.

---

## Hackathon Prize Tracks

### 🌵 Cactus — $10,000
*"Best local-first mobile application that intelligently routes tasks between models"*

Lane is this. Five distinct processing routes, zero cloud dependency:
- **MobileFaceNet** for face embedding
- **Gemma 4 at r=2/4/8** for three quality tiers
- **SQLite direct** for medication and routine queries (no LLM overhead)
- **Rule-based** for instant caregiver alerts (100ms response)

**Key files:** `train_code/src/routing.py` · `lane_app/src/routing/TaskRouter.ts`

### 🦥 Unsloth — $10,000
*"Best fine-tuned Gemma 4 model using Unsloth"*

Phases 2 and 3 use `FastLanguageModel` from Unsloth:
- 2× faster LoRA training via fused CUDA kernels
- 60% less VRAM via optimized gradient checkpointing
- 4-bit training on Kaggle T4 GPUs

**Key file:** `train_code/src/train_unsloth.py`

### 📱 LiteRT — $10,000
*"Most compelling use case using Google AI Edge LiteRT"*

Full export pipeline from trained checkpoint to on-device inference:
- TCS layer → TFLite via `ai_edge_torch`
- LoRA merged into base → int8 quantized LiteRT model
- MobileFaceNet → ONNX Runtime Mobile
- Integration manifest for React Native via `onnxruntime-react-native`

**Key file:** `train_code/src/export_litert.py` · `lane_app/src/engine/GemmaEngine.ts`

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Base LLM | Gemma 4 E2B (2B params) | Native tool calling, excellent multilingual, hackathon |
| Compression | TCS — Conv1D stride=4 | Original research · 4× token reduction · 16× FLOP savings |
| Fine-tuning | LoRA rank=16 on q/k/v/o | Only 0.6% trainable params of Gemma 4 |
| Training | Unsloth + Accelerate DDP | 2× faster on Kaggle T4 · 60% less VRAM |
| Mobile runtime | LiteRT (Google AI Edge) | Google ecosystem · optimized for on-device Gemma |
| Face detection | MediaPipe | Google ecosystem · <1MB · <10ms ARM |
| Face embedding | MobileFaceNet ONNX | 1MB · 128-d · 99.2% LFW accuracy |
| Vector search | sqlite-vec (cosine, threshold 0.85) | On-device · <1ms · zero dependency |
| Database | SQLite + WAL mode | Offline · zero-config · 8 tables |
| Mobile app | React Native (Expo SDK 54) | Cross-platform · New Architecture |
| State | Zustand | Lightweight · React 19 compatible |
| TTS | expo-speech | Google offline voices · elderly rate mode |
| STT | expo-speech-recognition | Offline-capable · on-device recognition |
| Alerts | expo-notifications + offline queue | Works without internet · auto-flush on reconnect |

---

## Team

| Member | Role |
|---|---|
| **Acauã Rangel Brazil** | ML Engineer & Researcher — TCS original research, Gemma 4 training pipeline, task routing, tool calling system, Kaggle infrastructure |
| **Juan Benjamin Suzart** | Mobile Developer — React Native (Expo), UI/UX design (Figma), camera integration, TTS/STT, caregiver app |

**Motivation:**

> *"There's a Brazilian influencer named Renato Cariani whose father had Alzheimer's. He told a story about how his father forgot how to smoke — his body was in withdrawal, craving the nicotine, but his mind couldn't remember what smoking was or how to do it. That image of someone trapped between a body that needs something and a mind that can't remember what it is — that stuck with me. That's who Lane is for."*
>
> — Acauã Rangel Brazil

---

## Limitations (Honest Assessment)

| Limitation | Status |
|---|---|
| Device form factor — phone is not ideal | Future: wearable camera (smart glasses, pendant) |
| Face database cold start — someone must register faces | Caregiver setup screen in the app |
| Offline STT/TTS quality below cloud-based | Working within Google's offline speech ecosystem |
| Clinical validation not yet done | This is a technical prototype |

---

## Future Roadmap

| Timeframe | Goal |
|---|---|
| Post-hackathon | Academic publication on TCS with Gemma 4 benchmarks |
| 3 months | Beta on Play Store · partnership with geriatric clinics in Salvador |
| 6 months | Smart glasses / wearable camera integration |
| 1 year | Clinical pilot study · multi-patient caregiver dashboard |
| Long-term | TCS applied to other edge AI domains (sign language, accessibility) |

---

*Memory Lane · Google DeepMind Gemma 4 Hackathon · Health & Sciences / Digital Equity*
