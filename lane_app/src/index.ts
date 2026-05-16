// Barrel export — use estes imports nas telas do app

// Hooks (interface principal para o colega)
export { useAgent, useVoice, useAlertSync } from './hooks';

// Stores
export { useAgentStore, usePatientStore, selectIsFullyReady } from './store';

// Tipos
export type {
  AgentInput,
  AgentResponse,
  AgentState,
  RouteType,
  PersonRecord,
  MedicationRecord,
  EncounterRecord,
  MemoryRecord,
  LocationRecord,
  AgendaRecord,
  PreferenceRecord,
  RoutineStep,
  IdentifiedPerson,
  CaregiverContact,
  PendingAlert,
  AlertType,
  TTSLanguage,
  CameraFrame,
  FaceDetection,
} from './types';

// Database (acesso direto quando necessário)
export { LaneDatabase } from './database';

// Alertas
export { CaregiverAlertService } from './alerts';
