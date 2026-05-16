import { create } from 'zustand';
import type { AgentState, RouteType, AgentResponse } from '../types';

interface AgentStore {
  // Estado atual do agente
  agentState: AgentState;
  currentRoute: RouteType | null;
  lastResponse: AgentResponse | null;

  // Prontidão dos subsistemas
  isModelReady: boolean;
  isDatabaseReady: boolean;
  isFaceEngineReady: boolean;

  // Cache de face embeddings em memória (evita re-lookup a cada frame)
  faceCache: Map<string, Float32Array>;

  // Latência da última inferência
  lastLatencyMs: number;

  // Ações
  setAgentState: (state: AgentState) => void;
  setCurrentRoute: (route: RouteType | null) => void;
  setLastResponse: (response: AgentResponse) => void;
  setModelReady: (ready: boolean) => void;
  setDatabaseReady: (ready: boolean) => void;
  setFaceEngineReady: (ready: boolean) => void;
  cacheEmbedding: (personId: string, embedding: Float32Array) => void;
  clearEmbeddingCache: () => void;
  setLastLatency: (ms: number) => void;
  reset: () => void;
}

const initialState = {
  agentState: 'idle' as AgentState,
  currentRoute: null,
  lastResponse: null,
  isModelReady: false,
  isDatabaseReady: false,
  isFaceEngineReady: false,
  faceCache: new Map<string, Float32Array>(),
  lastLatencyMs: 0,
};

export const useAgentStore = create<AgentStore>((set, get) => ({
  ...initialState,

  setAgentState: (agentState) => set({ agentState }),

  setCurrentRoute: (currentRoute) => set({ currentRoute }),

  setLastResponse: (lastResponse) => set({
    lastResponse,
    lastLatencyMs: lastResponse.latencyMs,
    currentRoute: lastResponse.route,
  }),

  setModelReady: (isModelReady) => set({ isModelReady }),

  setDatabaseReady: (isDatabaseReady) => set({ isDatabaseReady }),

  setFaceEngineReady: (isFaceEngineReady) => set({ isFaceEngineReady }),

  cacheEmbedding: (personId, embedding) => {
    const faceCache = new Map(get().faceCache);
    faceCache.set(personId, embedding);
    set({ faceCache });
  },

  clearEmbeddingCache: () => set({ faceCache: new Map() }),

  setLastLatency: (lastLatencyMs) => set({ lastLatencyMs }),

  reset: () => set({ ...initialState, faceCache: new Map() }),
}));

// Selector helpers para evitar re-renders desnecessários
export const selectIsFullyReady = (s: AgentStore) =>
  s.isModelReady && s.isDatabaseReady && s.isFaceEngineReady;

export const selectAgentState = (s: AgentStore) => s.agentState;
export const selectLastResponse = (s: AgentStore) => s.lastResponse;
