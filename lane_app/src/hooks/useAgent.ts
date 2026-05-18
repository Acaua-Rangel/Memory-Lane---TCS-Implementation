// Hook principal — dá acesso ao LaneAgent inicializado e ao estado reativo.
// Use este hook em qualquer componente que precise interagir com o agente.

import { useEffect, useRef, useCallback } from 'react';
import { LaneAgent } from '../agent/LaneAgent';
import { LaneDatabase } from '../database/LaneDatabase';
import { ToolExecutor } from '../tools/ToolExecutor';
import { TaskRouter } from '../routing/TaskRouter';
import { FacePipeline } from '../face/FacePipeline';
import { FaceIdentifier } from '../face/FaceIdentifier';
import { MobileFaceNetEmbedder, MockFaceEmbedder } from '../face/FaceEmbedder';
import { CaregiverAlertService } from '../alerts/CaregiverAlertService';
import { TCSOnnxEngine } from '../engine/TCSOnnxEngine';
import type { IGemmaEngine } from '../engine/GemmaEngine';
import { useAgentStore } from '../store/agentStore';
import { usePatientStore } from '../store/patientStore';
import type { AgentInput, AgentResponse, CameraFrame, FaceDetection } from '../types';

const IS_DEV = __DEV__;

export function useAgent() {
  const agentRef = useRef<LaneAgent | null>(null);
  const dbRef = useRef<LaneDatabase | null>(null);
  const alertServiceRef = useRef<CaregiverAlertService | null>(null);
  const engineRef = useRef<TCSOnnxEngine | null>(null);

  const {
    agentState,
    isModelReady,
    isDatabaseReady,
    isFaceEngineReady,
    lastResponse,
    lastLatencyMs,
    setAgentState,
    setModelReady,
    setDatabaseReady,
    setFaceEngineReady,
    setLastResponse,
  } = useAgentStore();

  const { language, elderlyMode, patientName } = usePatientStore();

  // ── Inicialização ──────────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        // 1. Banco de dados
        const db = LaneDatabase.getInstance();
        await db.open();
        await db.seedIfEmpty();
        dbRef.current = db;
        if (mounted) setDatabaseReady(true);

        // 2. Serviço de alertas
        const alertService = new CaregiverAlertService();
        await alertService.initialize();
        alertServiceRef.current = alertService;

        // 3. Pipeline facial
        const embedder = IS_DEV ? new MockFaceEmbedder() : new MobileFaceNetEmbedder();
        if (!IS_DEV) await (embedder as MobileFaceNetEmbedder).loadModel();
        const identifier = new FaceIdentifier(db.faceEmbeddings, db.persons);
        const facePipeline = new FacePipeline(embedder, identifier);
        if (mounted) setFaceEngineReady(true);

        // 4. Engine on-device — carrega tcs_compression.onnx via onnxruntime-react-native.
        // patientName vem do store React e é passado aqui para o engine não
        // precisar importar Zustand diretamente (evita acoplamento e erros de contexto).
        const currentPatientName = usePatientStore.getState().patientName;
        const tcsEngine = new TCSOnnxEngine(db, currentPatientName);
        engineRef.current = tcsEngine;
        const engine: IGemmaEngine = tcsEngine;
        try {
          await engine.loadModel({ modelPath: '', compressionRatio: 4 });
        } catch (engineErr) {
          // Não derruba o app — sqlite_direct / rule_based / face_pipeline continuam funcionando.
          console.warn('[useAgent] TCS ONNX não carregou:', engineErr);
        }
        if (mounted) setModelReady(true);

        // 5. Monta o agente
        const router = new TaskRouter();
        const toolExecutor = new ToolExecutor(db, alertService);
        const agent = new LaneAgent(router, facePipeline, engine, db, toolExecutor, alertService);
        agentRef.current = agent;

        console.log('[useAgent] Lane pronto');
      } catch (err) {
        console.error('[useAgent] Falha na inicialização:', err);
      }
    }

    void init();
    return () => { mounted = false; };
  }, []);

  // Mantém patientName do store sincronizado com o engine (sem recriar o engine).
  useEffect(() => {
    engineRef.current?.setPatientName(patientName);
  }, [patientName]);

  // ── Processar input de texto ───────────────────────────────────────────────
  const processText = useCallback(async (text: string): Promise<AgentResponse | null> => {
    if (!agentRef.current) return null;

    setAgentState('thinking');
    try {
      const input: AgentInput = { text, timestamp: Date.now() };
      const response = await agentRef.current.processInput(input);
      setLastResponse(response);
      return response;
    } finally {
      setAgentState('idle');
    }
  }, [setAgentState, setLastResponse]);

  // ── Processar frame da câmera com face detectada ───────────────────────────
  const processFaceFrame = useCallback(async (
    frame: CameraFrame,
    detection: FaceDetection,
  ): Promise<AgentResponse | null> => {
    if (!agentRef.current) return null;

    setAgentState('thinking');
    try {
      const input: AgentInput = {
        frame,
        faceDetected: true,
        timestamp: Date.now(),
      };
      const response = await agentRef.current.processInput(input);
      setLastResponse(response);
      return response;
    } finally {
      setAgentState('idle');
    }
  }, [setAgentState, setLastResponse]);

  // ── Agendar flush de alertas pendentes ────────────────────────────────────
  const flushAlerts = useCallback(async (): Promise<void> => {
    await alertServiceRef.current?.flushPendingAlerts();
  }, []);

  return {
    // Estado
    agentState,
    isReady: isModelReady && isDatabaseReady && isFaceEngineReady,
    isModelReady,
    isDatabaseReady,
    isFaceEngineReady,
    lastResponse,
    lastLatencyMs,

    // Ações
    processText,
    processFaceFrame,
    flushAlerts,

    // Acesso direto ao banco (para o colega usar nas telas)
    db: dbRef.current,
    alertService: alertServiceRef.current,
  };
}
