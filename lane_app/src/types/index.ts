// ─────────────────────────────────────────────
// AGENT TYPES
// ─────────────────────────────────────────────

export type RouteType =
  | 'face_pipeline'
  | 'sqlite_direct'
  | 'rule_based'
  | 'llm_fast'
  | 'llm_quality';

export type CompressionRatio = 2 | 4 | 8;

export type AgentState =
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'watching';

export interface RouteDecision {
  route: RouteType;
  compressionRatio?: CompressionRatio;
  confidence: number;
  intent: string;
}

export interface AgentInput {
  text?: string;
  frame?: CameraFrame;
  faceDetected?: boolean;
  timestamp: number;
}

export interface AgentResponse {
  speech: string;
  toolResults?: ToolResult[];
  identifiedPerson?: PersonRecord;
  alertSent?: boolean;
  route: RouteType;
  latencyMs: number;
}

// ─────────────────────────────────────────────
// CAMERA / FACE TYPES
// ─────────────────────────────────────────────

export interface CameraFrame {
  data: Uint8Array;
  width: number;
  height: number;
  timestamp: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceDetection {
  boundingBox: BoundingBox;
  landmarks?: FaceLandmark[];
  probability: number;
}

export interface FaceLandmark {
  x: number;
  y: number;
  name: string;
}

export interface FaceEmbedding {
  values: Float32Array; // 128-dimensional
  faceId?: string;
  confidence: number;
}

export interface PersonMatch {
  personId: string;
  similarity: number;
  person: PersonRecord;
}

export interface IdentifiedPerson {
  person: PersonRecord;
  similarity: number;
  lastEncounter?: EncounterRecord;
}

// ─────────────────────────────────────────────
// DATABASE TYPES — espelha o schema Python
// ─────────────────────────────────────────────

export interface PersonRecord {
  id: string;
  name: string;
  relationship: string; // "filho", "esposa", "médico", etc.
  bio: string;
  phoneNumber?: string;
  isCaregiver: boolean;
  createdAt: number;
  updatedAt: number;
}

export interface FaceEmbeddingRecord {
  id: string;
  personId: string;
  embedding: Float32Array; // 128-d
  createdAt: number;
}

export interface MemoryRecord {
  id: string;
  personId: string;
  title: string;
  description: string;
  date?: string;
  tags?: string[];
  createdAt: number;
}

export interface MedicationRecord {
  id: string;
  name: string;
  dosage: string;
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  description: string;    // cor, formato, tamanho para reconhecimento visual
  instructions: string;
  isActive: boolean;
}

export interface EncounterRecord {
  id: string;
  personId: string;
  context: string;
  location?: string;
  notes?: string;
  timestamp: number;
}

export interface LocationRecord {
  id: string;
  name: string;
  description: string;
  features: string[];   // ["pia", "fogão", "geladeira"] para reconhecimento visual
  navigationHint: string; // "O banheiro fica à esquerda do quarto"
}

export interface AgendaRecord {
  id: string;
  title: string;
  description: string;
  dayOfWeek: number;    // 0=domingo, 6=sábado
  time?: string;
  location?: string;
  isRecurring: boolean;
}

export interface PreferenceRecord {
  id: string;
  category: string;
  key: string;
  value: string;
  updatedAt: number;
}

export interface RoutineStep {
  id: string;
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  order: number;
  title: string;
  description: string;
}

// ─────────────────────────────────────────────
// TOOL TYPES — espelha ToolCallingConfig do Python
// ─────────────────────────────────────────────

export type ToolName =
  | 'read_person'
  | 'write_encounter'
  | 'get_medication'
  | 'describe_location'
  | 'alert_caregiver'
  | 'get_agenda'
  | 'save_preference'
  | 'get_preferences'
  | 'get_routine';

export type AlertType =
  | 'confused'
  | 'wandering'
  | 'missed_medication'
  | 'fall_detected'
  | 'emergency';

export interface ToolDefinition {
  name: ToolName;
  description: string;
  parameters: Record<string, ToolParameterDef>;
}

export interface ToolParameterDef {
  type: 'string' | 'number' | 'boolean' | 'array';
  description: string;
  required?: boolean;
  enum?: string[];
}

export interface ParsedToolCall {
  name: ToolName;
  parameters: Record<string, unknown>;
}

export interface ToolResult {
  toolName: ToolName;
  success: boolean;
  data?: unknown;
  error?: string;
}

// ─────────────────────────────────────────────
// VOICE TYPES
// ─────────────────────────────────────────────

export type TTSLanguage = 'pt-BR' | 'en-US';

export interface TTSOptions {
  language?: TTSLanguage;
  rate?: number;    // 0.5 = mais lento para idosos
  pitch?: number;
  volume?: number;
}

export interface STTResult {
  transcript: string;
  confidence: number;
  isFinal: boolean;
}

// ─────────────────────────────────────────────
// ALERT TYPES
// ─────────────────────────────────────────────

export interface CaregiverContact {
  id: string;
  name: string;
  phone: string;
  pushToken?: string;
  isPrimary: boolean;
}

export interface PendingAlert {
  id: string;
  type: AlertType;
  details: string;
  timestamp: number;
  retryCount: number;
}

// ─────────────────────────────────────────────
// ENGINE TYPES
// ─────────────────────────────────────────────

export interface EngineConfig {
  modelPath: string;
  tcsWeightsPath?: string;
  loraWeightsPath?: string;
  compressionRatio?: CompressionRatio;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
}

export interface TokenizedInput {
  inputIds: Int32Array;
  attentionMask: Int32Array;
}

export interface ToolCallResult {
  rawOutput: string;
  parsedCalls: ParsedToolCall[];
  finalResponse: string;
}

// ─────────────────────────────────────────────
// STORE TYPES
// ─────────────────────────────────────────────

export interface AgentStoreState {
  agentState: AgentState;
  currentRoute: RouteType | null;
  lastResponse: AgentResponse | null;
  faceCache: Map<string, Float32Array>;
  isModelReady: boolean;
  isDatabaseReady: boolean;
}

export interface PatientStoreState {
  patientId: string | null;
  patientName: string | null;
  language: TTSLanguage;
  caregivers: CaregiverContact[];
  pendingAlerts: PendingAlert[];
}
