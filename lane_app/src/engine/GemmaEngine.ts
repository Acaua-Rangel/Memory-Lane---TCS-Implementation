// Bridge para inferência on-device do Gemma 4 + TCS + LoRA via LiteRT/ONNX.
//
// Como integrar o modelo exportado pelo src/export_litert.py:
//   1. Copie os arquivos de outputs/tool_calling/final/ para assets/models/
//   2. Implemente LiteRTBridge usando react-native-litert (quando disponível)
//      ou TFLite via ai_edge_torch bridge nativo.
//   3. Troque MockGemmaEngine por RealGemmaEngine em createGemmaEngine().

import type { EngineConfig, TokenizedInput, ToolCallResult, CompressionRatio } from '../types';
import { TOOL_DEFINITIONS } from '../config/constants';
import { parseAllToolCalls, formatToolResult } from '../tools/toolParser';

export interface IGemmaEngine {
  loadModel(config: EngineConfig): Promise<void>;
  unloadModel(): void;
  generate(input: TokenizedInput, compressionRatio?: CompressionRatio): Promise<string>;
  generateWithTools(text: string, compressionRatio?: CompressionRatio): Promise<ToolCallResult>;
  isReady(): boolean;
  getCompressionRatio(): CompressionRatio;
}

// ─── Tokenizer simples (placeholder — substituir por tokenizer real) ──────────
// O tokenizer real do Gemma 4 deve ser carregado do modelo exportado.
// Para o hackathon, você pode usar o SentencePiece via WASM ou pré-tokenizar no servidor.
export class SimpleTokenizer {
  encode(text: string): Int32Array {
    // Placeholder: cada char vira um token (apenas para testes)
    const arr = new Int32Array(text.length);
    for (let i = 0; i < text.length; i++) {
      arr[i] = text.charCodeAt(i);
    }
    return arr;
  }

  decode(tokens: Int32Array): string {
    return String.fromCharCode(...tokens);
  }
}

// ─── Mock Engine — usado em desenvolvimento / quando modelo não está carregado ─
export class MockGemmaEngine implements IGemmaEngine {
  private ready = false;
  private compressionRatio: CompressionRatio = 4;

  async loadModel(config: EngineConfig): Promise<void> {
    this.compressionRatio = config.compressionRatio ?? 4;
    // Simula tempo de carregamento
    await new Promise((r) => setTimeout(r, 500));
    this.ready = true;
    console.log('[MockGemmaEngine] Modelo carregado (mock)');
  }

  unloadModel(): void {
    this.ready = false;
  }

  async generate(input: TokenizedInput, compressionRatio?: CompressionRatio): Promise<string> {
    if (!this.ready) throw new Error('GemmaEngine: modelo não carregado');
    // Mock: retorna resposta simples para desenvolvimento
    return 'Olá! Estou aqui para ajudar. (resposta mock — modelo não carregado)';
  }

  async generateWithTools(text: string, compressionRatio?: CompressionRatio): Promise<ToolCallResult> {
    if (!this.ready) throw new Error('GemmaEngine: modelo não carregado');

    // Mock: detecta alguns padrões básicos e gera tool call simulada
    let rawOutput = '';

    if (/remédio|medicamento|tomar/i.test(text)) {
      const hour = new Date().getHours();
      const timeOfDay = hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : hour < 21 ? 'evening' : 'night';
      rawOutput = `<tool_call>{"name": "get_medication", "parameters": {"time_of_day": "${timeOfDay}"}}</tool_call>`;
    } else if (/agenda|compromisso|hoje/i.test(text)) {
      const today = new Date().getDay();
      rawOutput = `<tool_call>{"name": "get_agenda", "parameters": {"day_of_week": "${today}"}}</tool_call>`;
    } else if (/rotina|o que fazer/i.test(text)) {
      rawOutput = `<tool_call>{"name": "get_routine", "parameters": {"time_of_day": "morning"}}</tool_call>`;
    }

    const parsedCalls = parseAllToolCalls(rawOutput);
    return { rawOutput, parsedCalls, finalResponse: '' };
  }

  isReady(): boolean {
    return this.ready;
  }

  getCompressionRatio(): CompressionRatio {
    return this.compressionRatio;
  }
}

// ─── Interface para bridge nativo LiteRT ──────────────────────────────────────
// Implemente este módulo nativo quando o react-native-litert estiver disponível
export interface ILiteRTBridge {
  loadModel(modelPath: string): Promise<void>;
  runInference(inputIds: number[], attentionMask: number[]): Promise<number[]>;
  unload(): void;
}

// ─── Engine real — requer módulo nativo LiteRT ────────────────────────────────
export class RealGemmaEngine implements IGemmaEngine {
  private bridge: ILiteRTBridge | null = null;
  private tokenizer = new SimpleTokenizer();
  private config: EngineConfig | null = null;
  private ready = false;

  constructor(private readonly bridgeFactory: () => ILiteRTBridge) {}

  async loadModel(config: EngineConfig): Promise<void> {
    this.config = config;
    this.bridge = this.bridgeFactory();
    await this.bridge.loadModel(config.modelPath);
    this.ready = true;
    console.log('[RealGemmaEngine] Modelo LiteRT carregado:', config.modelPath);
  }

  unloadModel(): void {
    this.bridge?.unload();
    this.bridge = null;
    this.ready = false;
  }

  async generate(input: TokenizedInput, compressionRatio?: CompressionRatio): Promise<string> {
    if (!this.bridge || !this.ready) throw new Error('GemmaEngine não inicializado');

    const outputTokens = await this.bridge.runInference(
      Array.from(input.inputIds),
      Array.from(input.attentionMask),
    );

    return this.tokenizer.decode(new Int32Array(outputTokens));
  }

  async generateWithTools(text: string, compressionRatio?: CompressionRatio): Promise<ToolCallResult> {
    if (!this.bridge || !this.ready) throw new Error('GemmaEngine não inicializado');

    // Constrói o prompt com definições de tools no formato Gemma 4
    const toolsJson = JSON.stringify(TOOL_DEFINITIONS.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    })));

    const systemPrompt = `Você é Lane, um assistente gentil para pacientes com Alzheimer.
Você tem acesso a estas ferramentas: ${toolsJson}
Use as ferramentas quando necessário. Responda sempre em português, com calma e carinho.`;

    const prompt = `${systemPrompt}\n\nUsuário: ${text}\nLane:`;
    const tokens = this.tokenizer.encode(prompt);
    const mask = new Int32Array(tokens.length).fill(1);

    const rawOutput = await this.generate({ inputIds: tokens, attentionMask: mask }, compressionRatio);
    const parsedCalls = parseAllToolCalls(rawOutput);

    // Se houver tool calls, executa e gera resposta final
    // (execução real acontece no LaneAgent após este método retornar)
    return { rawOutput, parsedCalls, finalResponse: rawOutput };
  }

  isReady(): boolean {
    return this.ready;
  }

  getCompressionRatio(): CompressionRatio {
    return this.config?.compressionRatio ?? 4;
  }
}

// ─── Factory ──────────────────────────────────────────────────────────────────
export function createGemmaEngine(useMock = true): IGemmaEngine {
  if (useMock) {
    return new MockGemmaEngine();
  }
  // Quando implementar o bridge nativo:
  // return new RealGemmaEngine(() => NativeModules.LiteRTBridge);
  return new MockGemmaEngine();
}
