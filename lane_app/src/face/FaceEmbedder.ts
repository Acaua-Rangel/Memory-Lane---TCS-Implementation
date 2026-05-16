// MobileFaceNet ONNX — gera embeddings 128-d a partir de imagem de rosto recortado.
// Arquivo do modelo: assets/models/mobilefacenet.onnx (1MB)
// Accuracy: 99.55% no LFW — suficiente para ~50 faces conhecidas

import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import type { FaceEmbedding, BoundingBox, CameraFrame } from '../types';
import { CAMERA } from '../config/constants';

export interface IFaceEmbedder {
  loadModel(modelPath?: string): Promise<void>;
  embed(frame: CameraFrame, boundingBox: BoundingBox): Promise<FaceEmbedding>;
  isReady(): boolean;
}

const MODEL_INPUT_SIZE = 112; // MobileFaceNet espera 112×112

export class MobileFaceNetEmbedder implements IFaceEmbedder {
  private session: InferenceSession | null = null;

  async loadModel(modelPath: string = CAMERA.MODEL_PATH): Promise<void> {
    try {
      this.session = await InferenceSession.create(modelPath);
      console.log('[MobileFaceNetEmbedder] Modelo ONNX carregado');
    } catch (err) {
      console.error('[MobileFaceNetEmbedder] Falha ao carregar modelo:', err);
      throw err;
    }
  }

  async embed(frame: CameraFrame, boundingBox: BoundingBox): Promise<FaceEmbedding> {
    if (!this.session) throw new Error('FaceEmbedder: modelo não carregado');

    const inputTensor = this.preprocessFace(frame, boundingBox);
    const feeds = { input: inputTensor };
    const results = await this.session.run(feeds);

    const outputName = Object.keys(results)[0];
    const embeddingData = results[outputName].data as Float32Array;

    // L2-normalize o embedding (padrão MobileFaceNet)
    const normalized = l2Normalize(embeddingData);

    return {
      values: normalized,
      confidence: 1.0,
    };
  }

  isReady(): boolean {
    return this.session !== null;
  }

  // Recorta e normaliza a face para 112×112×3, range [-1, 1]
  private preprocessFace(frame: CameraFrame, bbox: BoundingBox): Tensor {
    // Recorta a região da face do frame bruto
    const cropped = cropRegion(frame, bbox, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

    // Normaliza para [-1, 1] (padrão MobileFaceNet)
    const normalized = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    for (let i = 0; i < cropped.length; i++) {
      normalized[i] = (cropped[i] / 127.5) - 1.0;
    }

    // Layout: NCHW (1, 3, 112, 112)
    return new Tensor('float32', normalized, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  }
}

// ─── Mock para testes sem câmera ──────────────────────────────────────────────
export class MockFaceEmbedder implements IFaceEmbedder {
  async loadModel(_path?: string): Promise<void> {}

  async embed(_frame: CameraFrame, _bbox: BoundingBox): Promise<FaceEmbedding> {
    // Embedding aleatório normalizado — só para testes
    const values = new Float32Array(128);
    for (let i = 0; i < 128; i++) values[i] = Math.random() - 0.5;
    return { values: l2Normalize(values), confidence: 0.9 };
  }

  isReady(): boolean {
    return true;
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function l2Normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) result[i] = vec[i] / norm;
  return result;
}

// Recorta e redimensiona uma região do frame bruto (RGBA uint8)
// Implementação básica — em produção use react-native-skia ou módulo nativo para performance
function cropRegion(
  frame: CameraFrame,
  bbox: BoundingBox,
  targetW: number,
  targetH: number,
): Float32Array {
  const { data, width, height } = frame;
  const scaleX = width / frame.width;
  const scaleY = height / frame.height;

  const x0 = Math.max(0, Math.floor(bbox.x * scaleX));
  const y0 = Math.max(0, Math.floor(bbox.y * scaleY));
  const x1 = Math.min(width - 1, Math.floor((bbox.x + bbox.width) * scaleX));
  const y1 = Math.min(height - 1, Math.floor((bbox.y + bbox.height) * scaleY));

  const cropW = x1 - x0;
  const cropH = y1 - y0;

  // RGB channels separados (sem alpha) para NCHW
  const rgb = new Float32Array(3 * targetW * targetH);

  for (let ty = 0; ty < targetH; ty++) {
    for (let tx = 0; tx < targetW; tx++) {
      const sx = Math.floor((tx / targetW) * cropW) + x0;
      const sy = Math.floor((ty / targetH) * cropH) + y0;
      const srcIdx = (sy * width + sx) * 4; // RGBA

      rgb[ty * targetW + tx] = data[srcIdx];                          // R
      rgb[targetW * targetH + ty * targetW + tx] = data[srcIdx + 1]; // G
      rgb[2 * targetW * targetH + ty * targetW + tx] = data[srcIdx + 2]; // B
    }
  }

  return rgb;
}
