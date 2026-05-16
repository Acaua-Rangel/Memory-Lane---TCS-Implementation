# Memory Lane: When AI Remembers So They Don't Have To

> **Google DeepMind Gemma 4 Hackathon Submission**
> **Team:** Memory Lane — Acauã Rangel Brazil & Juan Benjamin Suzart
> **Location:** Salvador, Bahia, Brazil

---

## Project Title

**Memory Lane: When AI Remembers So They Don't Have To**

*Subtitle: An offline AI companion for Alzheimer's patients powered by Gemma 4 E2B with Token Compression Sub-network + LoRA*

---

## Elevator Pitch (30 seconds)

Lane is an AI companion that lives on your phone and never forgets. When a person with Alzheimer's sees a familiar face but can't place it, Lane whispers: *"That's Maria, your granddaughter. She's 8 and loves drawing. Last Sunday you two painted together."* It runs 100% offline, recognizes faces, manages medication, and keeps a living memory of everything the patient can't — all on a phone, no cloud needed.

---

## The Problem

**55 million people** worldwide live with dementia. Every 3 seconds, someone new is diagnosed. They lose the ability to:

- Recognize their own children and grandchildren
- Remember where they are or what day it is
- Follow their medication schedule
- Recall shared memories that define their relationships

Caregivers — mostly family members — spend exhausting hours repeating the same information, managing medication schedules, and dealing with the emotional weight of being forgotten by someone they love.

**Current solutions fail because:**
- They require constant internet (unreliable for elderly users)
- They're passive note-taking apps (the patient has to remember to check them)
- They need expensive specialized hardware
- They don't adapt to the patient's daily life in real-time

---

## The Solution: Lane

**Lane** (from *Memory Lane*) is an AI agent that runs entirely on the patient's phone, actively participating in their daily life:

### Core Capabilities

| Feature | How it Works | Model Route |
|---|---|---|
| **Face Recognition** | Camera detects face → MobileFaceNet (1MB) creates 128-d embedding → sqlite-vec matches against known faces → Lane tells the patient who they're looking at | MobileFaceNet → SQLite → Gemma 4 r=4 |
| **Active Memory** | Lane automatically records encounters, conversations, and events via tool calling to SQLite. When asked, it retrieves and narrates memories warmly | Gemma 4 r=2 (quality) |
| **Medication Management** | Tracks schedule, reminds the patient, describes what each pill looks like, confirms they took it | SQLite direct (fast) |
| **Orientation** | "Where am I?" → Lane describes the room, gives directions to bathroom/bedroom/kitchen | Gemma 4 r=4 (balanced) |
| **Daily Routine & Agenda** | Stores and recalls the patient's schedule, preferences, habits, favorite foods, and daily routines | Gemma 4 + tool calling |
| **Caregiver Bridge** | The caregiver can help register people, complete missing stories, and add context Lane can use | Rule-based + SQLite |

### Interaction Design

- **Voice-first**: All interactions via speech — no screen reading required
- **Wake word**: "Oi Lane" to activate, with physical button as fallback
- **Always gentle**: Lane speaks in warm, simple language appropriate for the patient
- **Proactive**: Lane doesn't wait to be asked — it notices a new face and speaks up
- **Google ecosystem**: TTS and STT use Google's offline speech models (keeping the DeepMind ecosystem)

---

## Technical Innovation

### 1. Token Compression Sub-network (TCS) — Original Research

**This is a novel technique from Acauã's original research**, with a preprint being published on arXiv for the GPT-2 validation phase.

The TCS is a trainable Conv1D layer inserted between Gemma 4's embedding layer and its transformer blocks. It compresses N tokens into N/r tokens (r ∈ {2, 4, 8}) before self-attention, achieving:

- **4× token reduction** → **16× FLOP reduction** in self-attention (O(N²) → O(N/4)²)
- **Only ~3M additional parameters** (0.15% of Gemma 4 E2B)
- **Fully differentiable** — trains end-to-end via self-distillation

```
Embedding (B, 512, 2048)
    │
    ▼
Conv1d(2048, 2048, kernel=7, stride=4) → GELU → Conv1d(1×1) → RMSNorm
    │
    ▼
Compressed (B, 128, 2048)  ← 4× fewer tokens
    │
    ▼
Gemma 4 Transformer Blocks (now processing 128 tokens instead of 512)
```

**Research trajectory:**
1. Validated with TinyStories + GPT-2 (preprint on arXiv)
2. Now adapted for Gemma 4 E2B with LoRA (this hackathon)
3. Future: formal publication with full Gemma 4 benchmarks

### 2. Self-Distillation (No External Teacher)

The uncompressed path through Gemma 4 (full tokens, no LoRA) serves as the teacher. The compressed path (TCS + LoRA) is the student. This eliminates the need for a separate larger model as teacher:

```
Loss = α · T² · KL(student ‖ teacher) + (1-α) · CE(student, labels)
```

### 3. Intelligent Task Routing (Cactus Integration)

Lane doesn't use the same model for everything. A lightweight `TaskRouter` classifies each input and routes to the optimal processing path:

```
Face detected  → MobileFaceNet → sqlite-vec → LLM r=4    [500ms budget]
"Bom dia"      → Gemma 4 + TCS r=8 (fastest)              [500ms budget]
"Que remédio?" → SQLite direct lookup (skip LLM)           [300ms budget]
"Quem é Maria?"→ Gemma 4 + TCS r=2 (highest quality)      [2000ms budget]
"Estou perdido"→ Rule-based alert + calming response       [100ms budget]
```

This makes Lane a **true multi-model orchestrator** — the core of the Cactus prize track.

### 4. Native Tool Calling via SQLite

Gemma 4's native function calling format drives all memory operations:

| Tool | Purpose |
|---|---|
| `read_person(face_id)` | Retrieve a person's name, relationship, bio, and recent encounters |
| `write_encounter(person, context)` | Log who visited, what happened, when |
| `get_medication(time_of_day)` | Get medication schedule for morning/afternoon/evening |
| `describe_location(features)` | Identify the current room from visual features |
| `alert_caregiver(type, details)` | Silent alert for confusion/wandering/fall risk |

Lane **actively writes** to the database — it doesn't just read. Every encounter, every conversation fragment, every medication confirmation is recorded for future recall.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PATIENT'S PHONE                          │
│                                                                 │
│  ┌──────────┐   ┌────────────────────────────────────────────┐  │
│  │ Camera   │──→│ MediaPipe Face Detection (<1MB)            │  │
│  └──────────┘   │         │                                  │  │
│                 │         ▼                                  │  │
│  ┌──────────┐   │ MobileFaceNet ONNX (1MB, 128-d)           │  │
│  │ Mic      │   │         │                                  │  │
│  │ (STT)    │   │         ▼                                  │  │
│  └────┬─────┘   │ sqlite-vec cosine search → face_id         │  │
│       │         └──────────────────────────┬─────────────────┘  │
│       │                                    │                    │
│       ▼                                    ▼                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               TaskRouter (Cactus)                        │   │
│  │  Classifies input → routes to optimal model/compression  │   │
│  └─────┬──────────┬──────────┬──────────┬──────────┬────────┘   │
│        │          │          │          │          │             │
│        ▼          ▼          ▼          ▼          ▼             │
│   LLM r=8    LLM r=4    LLM r=2   SQLite     Rule-based       │
│   (fast)    (balanced)  (quality)  (direct)   (alerts)          │
│        │          │          │          │          │             │
│        └──────────┴──────────┴──────────┴──────────┘             │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Gemma 4 E2B (Frozen) + TCS (3M params) + LoRA (20M)   │   │
│  │  Quantized int8 via LiteRT / ONNX Runtime               │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SQLite + sqlite-vec                                     │   │
│  │  persons | memories | medication | encounters | faces    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────┐                                                   │
│  │ Speaker  │  ← Google TTS (offline)                          │
│  │ (TTS)    │                                                   │
│  └──────────┘                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

**Environment:** Kaggle 2× NVIDIA T4 (32GB VRAM total, 30GB RAM, 19.5GB disk)

| Phase | What | Trainable Params | Duration | Backend |
|---|---|---|---|---|
| 1. TCS Pre-train | Compression layer only | ~3M (TCS) | ~30 min | Accelerate DDP |
| 2. TCS + LoRA | Joint self-distillation | ~23M (TCS + LoRA) | ~2-4h | **Accelerate DDP** on 2× T4 or **Unsloth** on 1× T4 |
| 3. Tool Calling | Function calling fine-tune | ~23M | ~1-2h | **Accelerate DDP** on 2× T4 or **Unsloth** on 1× T4 |
| 4. Export | LiteRT / ONNX for mobile | — | ~10 min | ai-edge-torch |

**Synthetic data:** 8,000 training + 500 validation examples across 5 categories (face recognition 40%, orientation 20%, medication 15%, memory sharing 15%, caregiver alerts 10%).

---

## Special Technology Track Prizes

We target **three Special Technology Track prizes** (can be won alongside Main Track):

### Cactus — $10,000 🌵
**"Best local-first mobile or wearable application that intelligently routes tasks between models"**

Lane IS this. It's a mobile companion that routes between:
- MobileFaceNet (face detection + embedding)
- Gemma 4 at three compression levels (r=2, r=4, r=8)
- SQLite direct queries (bypass LLM for speed)
- Rule-based logic (instant caregiver alerts)

**Key file:** `src/routing.py` — `TaskRouter` + `RoutedCompanion`

### Unsloth — $10,000 🦥
**"Best fine-tuned Gemma 4 model created using Unsloth"**

Phases 2 and 3 use Unsloth's `FastLanguageModel` for:
- 2× faster LoRA training via fused CUDA kernels
- 60% less VRAM via optimized gradient checkpointing
- 4-bit training on Kaggle's T4 GPUs

The current `src/train_unsloth.py` entrypoint is single-GPU. Therefore, when we want to occupy both Kaggle T4s, we launch `src.train_ddp.py` with `torchrun` instead.

**Key file:** `src/train_unsloth.py`

### LiteRT — $10,000 📱
**"Most compelling use case built using Google AI Edge's LiteRT"**

Full export pipeline:
- TCS layer → TFLite via `ai_edge_torch`
- LoRA merged into base → quantized LiteRT model
- MobileFaceNet → ONNX Runtime Mobile
- Deployment manifest for the React Native app

Furthermore, the export path now reconstructs the TCS embedding width directly from saved checkpoints and accepts both nested and flat TCS checkpoint layouts, which keeps LiteRT export aligned across standard and Unsloth training outputs.

**Key file:** `src/export_litert.py`

---

## Demo / Video Script (3 minutes)

### Scene 1: The Hook (0:00 - 0:30)
*A doorbell rings. An elderly man is in his living room. A woman walks in. He looks confused.*

**Patient:** "Oi... você é...?"
*Camera on the phone (on a table nearby) detects the face.*
**Lane** (warm voice): "Essa é a Maria, sua neta. Ela tem 8 anos e adora desenhar. Domingo passado vocês pintaram juntos na varanda."
*The man smiles. Maria hugs him.*

### Scene 2: The Reality (0:30 - 0:50)
*Screen fades to black. Text appears, word by word, some words disappearing:*

> "Alzheimer's doesn't just take memories. It takes ~~the people inside them~~ ~~the moments that matter~~ everything."
> "But what if there was someone who never forgets?"

**Lane's voice:** "Eu sou o Lane. Eu lembro por você."

### Scene 3: Features Demo (0:50 - 2:15)
Quick cuts showing:

1. **Medication** — Maria asks "Pai, já tomou os remédios?" Lane answers: "Ainda não. São dois: o comprimido branco redondo do Donepezil e a cápsula azul do ômega-3."

2. **Orientation** — Patient wakes up confused. "Onde estou?" Lane: "Você está no seu quarto, em casa. O banheiro é a segunda porta à esquerda."

3. **Active Memory** — Lane automatically logs: "Maria visited at 3pm. They painted together. Patient was happy." Later: "Conte sobre a Maria" → Lane narrates the stored memories.

4. **Caregiver Setup** — Maria's mom adding family photos and stories via the app, enriching Lane's memory bank.

### Scene 4: The Tech (2:15 - 2:45)
*Screen recording + architecture diagram:*
- TCS compresses tokens 4× → runs on phone
- Intelligent routing: 5 different paths for different tasks
- All offline, all on-device, all private
- Trained with Unsloth on Kaggle, exported via LiteRT

### Scene 5: The Close (2:45 - 3:00)
*Back to the old man and Maria. She's leaving.*

**Patient:** "Lane, quem era essa menina?"
**Lane:** "Maria, sua neta. Ela veio te visitar e trouxe um desenho novo. Vocês se divertiram muito hoje."
*Patient smiles, puts the drawing on the fridge.*

> **Memory Lane: When AI Remembers So They Don't Have To.**

---

## Team: Memory Lane

| Member | Role | Focus |
|---|---|---|
| **Acauã Rangel Brazil** | ML Engineer & Researcher | TCS research (original), model training, agent orchestration, Kaggle pipeline, tool calling system |
| **Juan Benjamin Suzart** | Mobile Developer | React Native (Expo) app, UI/UX design (Figma), TTS/STT integration, camera/face pipeline on device |

**Location:** Salvador, Bahia, Brazil 🇧🇷

---

## Hackathon Category

**Primary:** Health & Sciences
> Lane directly addresses the health challenges of 55M dementia patients worldwide, providing an AI-powered assistive technology.

**Secondary:** Digital Equity
> By running 100% offline on a standard phone, Lane makes AI-powered memory assistance accessible to patients regardless of internet connectivity, location, or economic status — including rural communities in developing countries.

---

## What Makes This Different

| Existing Solutions | Lane |
|---|---|
| Passive note apps (user must remember to check) | **Active agent** that speaks up proactively |
| Cloud-dependent (need internet) | **100% offline** on the patient's phone |
| Generic AI assistants | **Specialized for Alzheimer's** with warm, simple language |
| Static photo albums | **Living memory** that updates automatically via tool calling |
| Require manual logging | **Lane records encounters automatically** |
| One-size-fits-all processing | **Intelligent routing** — fast for greetings, detailed for memories |
| Research prototypes | **Practical mobile app** (React Native + LiteRT) |

**The real innovation:** Lane doesn't just store information — it **actively participates** in the patient's daily life, recording and recalling at the right moment, in the right way.

---

## Academic Contribution

This project has a dual identity:

1. **Practical product:** An assistive technology for Alzheimer's patients
2. **Research contribution:** Novel Token Compression Sub-network technique

**TCS Research Timeline:**
- ✅ Validated on TinyStories + GPT-2 (preprint on arXiv before submission)
- 🔄 Adapted for Gemma 4 E2B with LoRA + self-distillation (this hackathon)
- 📝 Full paper with Gemma 4 benchmarks planned for publication

The TCS technique is **model-agnostic** — it can be applied to any transformer, making it a general contribution to efficient LLM inference on edge devices.

---

## Current Limitations (Honest Assessment)

1. **Device form factor:** Currently runs on a phone, but patients don't want to hold a phone all day. Ideally connects to a wearable camera (smart glasses, pendant) — this is a hardware limitation, not a software one.
2. **Face database cold start:** Someone (caregiver) needs to register faces and stories initially. Lane can't learn faces it's never seen.
3. **Speech quality:** Offline TTS/STT quality is lower than cloud-based. Working within Google's offline speech ecosystem for best results.
4. **Caregiver alerts:** Currently the AI + memory system only. Real-time alerts to caregivers would need network connectivity (future work).
5. **Clinical validation:** Not yet tested with real Alzheimer's patients. This is a prototype demonstrating the technical feasibility.

---

## Future Vision

| Timeframe | Goal |
|---|---|
| **Post-hackathon** | Academic publication on TCS technique (Gemma 4 benchmarks) |
| **3 months** | Beta app on Play Store, partnership with local geriatric clinics in Salvador |
| **6 months** | Integration with smart glasses / wearable cameras |
| **1 year** | Clinical pilot study, caregiver alert system, multi-patient support |
| **Long-term** | Adapt TCS for other edge AI applications (sign language, elder care, accessibility) |

---

## Language Support

Lane inherits Gemma 4's multilingual capabilities. The routing system includes keyword patterns in both **Portuguese (BR)** and **English**, and can be extended to any language Gemma 4 supports. The synthetic training data is generated in English but the model responds in the patient's language.

---

## Personal Motivation

> "There's a Brazilian influencer named Renato Cariani whose father had Alzheimer's. He told a story about how his father forgot how to smoke — his body was in withdrawal, craving the nicotine, but his mind couldn't remember what smoking was or how to do it. That image of someone trapped between a body that needs something and a mind that can't remember what it is — that stuck with me. That's who Lane is for."
>
> — Acauã Rangel Brazil

---

## Tech Stack Summary

| Component | Technology | Why |
|---|---|---|
| Base LLM | Gemma 4 E2B (2B params) | Hackathon requirement, excellent multilingual, native tool calling |
| Compression | TCS (Conv1D stride=4) | Original research — 4× token reduction, 16× FLOP savings |
| Adaptation | LoRA (rank=16, alpha=32) | Efficient fine-tuning, only 0.6% trainable params |
| Training | Unsloth + Accelerate DDP | 2× faster on Kaggle T4s |
| Mobile Runtime | LiteRT (Google AI Edge) | Google ecosystem, optimized for on-device Gemma |
| Face Detection | MediaPipe (<1MB) | Google ecosystem, works offline on ARM |
| Face Embedding | MobileFaceNet ONNX (1MB, 128-d) | Mobile-optimized, <30ms on ARM |
| Vector Search | sqlite-vec | Cosine similarity on 128-d embeddings, <1ms |
| Database | SQLite + WAL mode | Reliable offline storage, zero-config |
| Mobile App | React Native (Expo) | Cross-platform (Android + iOS) |
| Speech | Google Offline TTS/STT | Google ecosystem, works without internet |
| Task Routing | Custom TaskRouter | Cactus integration — multi-model orchestration |

---

## Repository Structure

```
gemma4-memory-companion/
├── src/
│   ├── config.py              # All configurations
│   ├── model.py               # Gemma 4 + TCS + LoRA wrapper
│   ├── losses.py              # Self-distillation + tool calling losses
│   ├── data.py                # Dataset + synthetic data generation
│   ├── train.py               # Single-GPU training
│   ├── train_ddp.py           # Multi-GPU DDP training (Accelerate)
│   ├── train_unsloth.py       # Unsloth-accelerated LoRA training
│   ├── routing.py             # Cactus task router
│   ├── export_litert.py       # LiteRT mobile export pipeline
│   ├── inference.py           # Inference with tool execution loop
│   ├── tools/
│   │   └── sqlite_tools.py    # 5 SQLite tools + execution engine
│   └── utils/
│       ├── checkpoint.py      # Checkpoint rotation + cleanup
│       └── distributed.py     # DDP helpers
├── tests/                     # Unit tests (no GPU required)
├── scripts/
│   ├── kaggle_setup.sh        # Kaggle environment setup
│   └── train_full_pipeline.sh # Full 4-phase training script
├── docs/
│   ├── architecture.md        # Technical architecture details
│   └── training_plan.md       # Training plan with Kaggle constraints
└── export/                    # LiteRT exported models (generated)
```
