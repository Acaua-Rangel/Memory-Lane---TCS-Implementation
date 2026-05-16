// Detecção de faces via react-native-vision-camera frame processors.
// MediaPipe Face Detection (<1MB, <10ms em mobile ARM).

import type { CameraFrame, FaceDetection, BoundingBox } from '../types';

export interface IFaceDetector {
  detect(frame: CameraFrame): Promise<FaceDetection[]>;
  isReady(): boolean;
}

// ─── Implementação via Vision Camera frame processor ──────────────────────────
// Para ativar: instale react-native-vision-camera + VisionCamera Face Detection plugin
// Docs: https://github.com/mrousavy/react-native-vision-camera

export class VisionCameraFaceDetector implements IFaceDetector {
  private initialized = false;

  async initialize(): Promise<void> {
    // O frame processor é inicializado automaticamente pela Vision Camera
    // Este método serve como hook para configuração futura (ex: modelo customizado)
    this.initialized = true;
  }

  detect(_frame: CameraFrame): Promise<FaceDetection[]> {
    // A detecção real acontece em um frame processor worklet da Vision Camera.
    // Veja src/face/FacePipeline.ts para o uso integrado com o componente Camera.
    throw new Error(
      'VisionCameraFaceDetector.detect() deve ser chamado dentro de um frame processor worklet. ' +
      'Use FacePipeline.onFrameProcessed() no componente Camera.',
    );
  }

  isReady(): boolean {
    return this.initialized;
  }
}

// ─── Mock para desenvolvimento / testes sem câmera ────────────────────────────
export class MockFaceDetector implements IFaceDetector {
  private mockDetections: FaceDetection[];

  constructor(mockDetections: FaceDetection[] = []) {
    this.mockDetections = mockDetections;
  }

  async detect(_frame: CameraFrame): Promise<FaceDetection[]> {
    return this.mockDetections;
  }

  isReady(): boolean {
    return true;
  }

  setMockDetections(detections: FaceDetection[]): void {
    this.mockDetections = detections;
  }
}

// Converte o resultado do frame processor da Vision Camera para FaceDetection
export function fromVisionCameraResult(result: {
  bounds: { x: number; y: number; width: number; height: number };
  rollAngle?: number;
  pitchAngle?: number;
  yawAngle?: number;
  leftEyeOpenProbability?: number;
  smilingProbability?: number;
}): FaceDetection {
  const box: BoundingBox = {
    x: result.bounds.x,
    y: result.bounds.y,
    width: result.bounds.width,
    height: result.bounds.height,
  };

  return {
    boundingBox: box,
    probability: result.leftEyeOpenProbability ?? 1.0,
  };
}
