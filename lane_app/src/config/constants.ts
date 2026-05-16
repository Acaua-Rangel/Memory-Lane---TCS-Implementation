import type { CompressionRatio, ToolDefinition } from '../types';

// ─── Inferência ────────────────────────────────────────────────
export const COMPRESSION_RATIOS: Record<string, CompressionRatio> = {
  FAST: 8,      // saudação, respostas rápidas — latência 500ms
  BALANCED: 4,  // reconhecimento facial — latência 500ms
  QUALITY: 2,   // memórias, histórias — latência 2000ms
};

export const INFERENCE = {
  MAX_NEW_TOKENS: 256,
  TEMPERATURE: 0.7,
  TOP_P: 0.92,
  FACE_SIMILARITY_THRESHOLD: 0.85, // MobileFaceNet cosine
  LATENCY_BUDGET_MS: {
    face: 500,
    greeting: 500,
    medication: 300,
    memory: 2000,
    alert: 100,
  },
} as const;

// ─── Banco de dados ────────────────────────────────────────────
export const DATABASE = {
  NAME: 'lane.db',
  VERSION: 1,
  EMBEDDING_DIM: 128, // MobileFaceNet
} as const;

// ─── Voz ───────────────────────────────────────────────────────
export const VOICE = {
  WAKE_WORD: 'lane',
  WAKE_WORD_ALTERNATIVES: ['oi lane', 'ei lane', 'hey lane', 'olá lane'],
  TTS_RATE: 0.85,           // mais lento para pacientes com Alzheimer
  TTS_PITCH: 1.0,
  DEFAULT_LANGUAGE: 'pt-BR' as const,
  SILENCE_TIMEOUT_MS: 3000,
  MAX_RECORD_DURATION_MS: 15000,
} as const;

// ─── Alertas ───────────────────────────────────────────────────
export const ALERTS = {
  MAX_RETRY_COUNT: 5,
  RETRY_INTERVAL_MS: 30_000,
  BACKGROUND_TASK_NAME: 'LANE_ALERT_SYNC',
} as const;

// ─── Câmera ────────────────────────────────────────────────────
export const CAMERA = {
  FACE_DETECTION_INTERVAL_MS: 1000,
  MODEL_PATH: 'models/mobilefacenet.onnx',
} as const;

// ─── As 9 ferramentas — espelha src/config.py:ToolCallingConfig ─
export const TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    name: 'read_person',
    description: 'Retrieve biographical information and memories about a person by their face ID.',
    parameters: {
      face_id: { type: 'string', description: 'The unique identifier for the detected face', required: true },
    },
  },
  {
    name: 'write_encounter',
    description: 'Log a new encounter with a person, recording context and location.',
    parameters: {
      person: { type: 'string', description: 'Name or ID of the person encountered', required: true },
      context: { type: 'string', description: 'Description of the encounter context', required: true },
      location: { type: 'string', description: 'Location where encounter occurred', required: false },
    },
  },
  {
    name: 'get_medication',
    description: 'Retrieve medication schedule for a specific time of day.',
    parameters: {
      time_of_day: {
        type: 'string',
        description: 'Time period to get medications for',
        required: true,
        enum: ['morning', 'afternoon', 'evening', 'night'],
      },
    },
  },
  {
    name: 'describe_location',
    description: 'Identify the current room or location based on visible features.',
    parameters: {
      features: {
        type: 'array',
        description: 'List of visible features/objects in the scene (e.g. ["pia", "fogão"])',
        required: true,
      },
    },
  },
  {
    name: 'alert_caregiver',
    description: 'Send a silent alert to registered caregivers.',
    parameters: {
      type: {
        type: 'string',
        description: 'Type of alert',
        required: true,
        enum: ['confused', 'wandering', 'missed_medication', 'fall_detected', 'emergency'],
      },
      details: { type: 'string', description: 'Additional details about the situation', required: true },
    },
  },
  {
    name: 'get_agenda',
    description: 'Get scheduled appointments and events for a specific day of the week.',
    parameters: {
      day_of_week: { type: 'string', description: 'Day of week (0=Sunday through 6=Saturday)', required: true },
    },
  },
  {
    name: 'save_preference',
    description: 'Store a patient preference or habit.',
    parameters: {
      category: { type: 'string', description: 'Category of preference (e.g. "food", "music", "health")', required: true },
      key: { type: 'string', description: 'Preference key', required: true },
      value: { type: 'string', description: 'Preference value', required: true },
    },
  },
  {
    name: 'get_preferences',
    description: 'Retrieve all preferences for a given category.',
    parameters: {
      category: { type: 'string', description: 'Category to retrieve preferences for', required: true },
    },
  },
  {
    name: 'get_routine',
    description: 'Get the daily routine steps for a specific time of day.',
    parameters: {
      time_of_day: {
        type: 'string',
        description: 'Time period to get routine for',
        required: true,
        enum: ['morning', 'afternoon', 'evening', 'night'],
      },
    },
  },
];
