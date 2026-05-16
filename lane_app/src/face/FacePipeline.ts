// Orquestra os 3 passos da pipeline de reconhecimento facial:
//   detect → embed → identify
// Espelha src/routing.py:face_pipeline()

import type { IFaceEmbedder } from './FaceEmbedder';
import type { FaceIdentifier } from './FaceIdentifier';
import type { CameraFrame, IdentifiedPerson, FaceDetection } from '../types';
import { CAMERA } from '../config/constants';

export interface FacePipelineResult {
  detected: boolean;
  identified: IdentifiedPerson | null;
  detection: FaceDetection | null;
  processingMs: number;
}

export class FacePipeline {
  private lastFrameTime = 0;
  private frameIntervalMs = CAMERA.FACE_DETECTION_INTERVAL_MS;

  // Cache em memória — evita re-identificar a mesma pessoa a cada frame
  private lastIdentifiedId: string | null = null;
  private lastIdentifiedAt = 0;
  private readonly cacheMs = 5000;

  constructor(
    private embedder: IFaceEmbedder,
    private identifier: FaceIdentifier,
    // detector é opcional — em produção, a detecção ocorre no frame processor da Vision Camera
    private detector?: { detect: (frame: CameraFrame) => Promise<FaceDetection[]> },
  ) {}

  // Chamado quando o frame processor da Vision Camera detecta uma face
  async onFaceDetected(
    frame: CameraFrame,
    detection: FaceDetection,
  ): Promise<FacePipelineResult> {
    const start = Date.now();

    // Rate limiting — não processa todo frame
    if (start - this.lastFrameTime < this.frameIntervalMs) {
      return { detected: true, identified: null, detection, processingMs: 0 };
    }
    this.lastFrameTime = start;

    // Cache — se acabamos de identificar esta pessoa, retorna do cache
    if (this.lastIdentifiedId && start - this.lastIdentifiedAt < this.cacheMs) {
      const person = await this.identifier.findTopMatches(new Float32Array(128), 1);
      if (person.length > 0 && person[0].personId === this.lastIdentifiedId) {
        return {
          detected: true,
          identified: { person: person[0].person, similarity: person[0].similarity },
          detection,
          processingMs: Date.now() - start,
        };
      }
    }

    // Embedding
    const embedding = await this.embedder.embed(frame, detection.boundingBox);

    // Identificação via cosine search no SQLite
    const identified = await this.identifier.identify(embedding.values);

    if (identified) {
      this.lastIdentifiedId = identified.person.id;
      this.lastIdentifiedAt = Date.now();
    }

    return {
      detected: true,
      identified,
      detection,
      processingMs: Date.now() - start,
    };
  }

  // Para uso sem câmera (frame passado diretamente)
  async processFrame(frame: CameraFrame): Promise<FacePipelineResult> {
    const start = Date.now();

    if (!this.detector) {
      throw new Error('FacePipeline: detector não configurado. Use onFaceDetected() com frame processor.');
    }

    const detections = await this.detector.detect(frame);
    if (detections.length === 0) {
      return { detected: false, identified: null, detection: null, processingMs: Date.now() - start };
    }

    // Usa a detecção de maior confiança
    const best = detections.reduce((a, b) => (a.probability > b.probability ? a : b));
    return this.onFaceDetected(frame, best);
  }

  clearCache(): void {
    this.lastIdentifiedId = null;
    this.lastIdentifiedAt = 0;
  }

  setFrameInterval(ms: number): void {
    this.frameIntervalMs = ms;
  }
}
