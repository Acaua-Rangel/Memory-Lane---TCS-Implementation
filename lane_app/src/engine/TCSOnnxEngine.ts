// Engine on-device para o Lane (Token Compression Sub-network).
//
// Estado da pipeline de export (train_code/src/export_litert.py):
//   ✅ TCS exportada → assets/models/tcs_compression.onnx
//   ❌ Gemma 4 full ainda não exportado (precisa de ai_edge_torch)
//
// O engine faz duas coisas em paralelo:
//   1. Roda o TCS ONNX de verdade via onnxruntime-react-native a cada
//      chamada — comprova latência on-device da contribuição técnica original.
//   2. Constrói a resposta de texto a partir do banco SQLite real do paciente
//      (pessoas, memórias, encontros, agenda, remédios, locais).
//      Não é texto fixo — vem dos dados cadastrados pelo cuidador.
//
// Quando o Gemma 4 estiver exportado para ONNX/LiteRT, encadeie:
//   embedding_layer(tokens) → TCS forward → transformer_blocks → logits
// e troque `composeResponse` por decodificação real de tokens.

import type { InferenceSession, Tensor } from 'onnxruntime-react-native';
import type {
  CompressionRatio,
  EngineConfig,
  TokenizedInput,
  ToolCallResult,
  ParsedToolCall,
} from '../types';
import type { LaneDatabase } from '../database/LaneDatabase';
import { TaskRouter } from '../routing/TaskRouter';
import { PATTERNS } from '../routing/patterns';
import type { IGemmaEngine } from './GemmaEngine';

// onnxruntime-react-native é carregado lazily para não quebrar antes da
// bridge nativa do React Native estar pronta.
type OnnxModule = typeof import('onnxruntime-react-native');
let _onnx: OnnxModule | null = null;
function getOnnx(): OnnxModule {
  if (!_onnx) {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    _onnx = require('onnxruntime-react-native') as OnnxModule;
  }
  return _onnx;
}

// O require retorna um número (asset module ID) registrado pelo Metro.
// onnxruntime-react-native >= 1.14 aceita esse ID diretamente em
// InferenceSession.create() — sem precisar de Asset.fromModule.
// Veja: https://onnxruntime.ai/docs/tutorials/mobile/reactnative.html
// eslint-disable-next-line @typescript-eslint/no-require-imports
const TCS_MODEL_REF: number = require('@assets/models/tcs_compression.onnx') as number;

// Dimensões do checkpoint treinado (o export logou hidden_size=1536).
const TCS_HIDDEN_SIZE = 1536;
// Sequência de prova: mínimo coerente para o stride do kernel (r=4 → 32 tokens).
const TCS_PROBE_SEQ = 32;

type TimeOfDay = 'morning' | 'afternoon' | 'evening' | 'night';

export class TCSOnnxEngine implements IGemmaEngine {
  private session: InferenceSession | null = null;
  private ready = false;
  private compressionRatio: CompressionRatio = 4;
  private lastTcsLatencyMs = 0;

  // patientName é passado pelo useAgent no momento da construção,
  // evitando qualquer import do Zustand store dentro desta classe.
  constructor(
    private readonly db: LaneDatabase,
    private patientName: string | null = null,
  ) {}

  /** Atualiza o nome do paciente sem recriar o engine. */
  setPatientName(name: string | null): void {
    this.patientName = name;
  }

  async loadModel(config: EngineConfig): Promise<void> {
    this.compressionRatio = config.compressionRatio ?? 4;

    // InferenceSession.create aceita o número retornado pelo require().
    // Internamente o ORT RN resolve o asset via React Native's AssetRegistry.
    this.session = await getOnnx().InferenceSession.create(TCS_MODEL_REF as unknown as string, {
      executionProviders: ['cpu'],
    });

    this.ready = true;
    console.log(
      `[TCSOnnxEngine] TCS ONNX carregado (hidden=${TCS_HIDDEN_SIZE}, ratio=${this.compressionRatio})`,
    );
  }

  unloadModel(): void {
    this.session = null;
    this.ready = false;
  }

  isReady(): boolean {
    return this.ready;
  }

  getCompressionRatio(): CompressionRatio {
    return this.compressionRatio;
  }

  async generate(_input: TokenizedInput, ratio?: CompressionRatio): Promise<string> {
    const result = await this.generateWithTools('', ratio);
    return result.finalResponse;
  }

  async generateWithTools(text: string, ratio?: CompressionRatio): Promise<ToolCallResult> {
    if (!this.session) throw new Error('[TCSOnnxEngine] Modelo não inicializado');

    // 1. Forward real do TCS — mede latência on-device do modelo.
    await this.runTcsForward(ratio ?? this.compressionRatio);

    // 2. Resposta aterrada no banco SQLite do paciente.
    const intent = detectIntent(text);
    const { speech, toolCalls } = await this.buildResponse(text, intent);

    return { rawOutput: speech, parsedCalls: toolCalls, finalResponse: speech };
  }

  getLastTcsLatencyMs(): number {
    return this.lastTcsLatencyMs;
  }

  // ── TCS forward ──────────────────────────────────────────────────────────
  private async runTcsForward(ratio: CompressionRatio): Promise<void> {
    if (!this.session) return;

    // Sequência mínima que satisfaz stride = ratio e kernel size 7
    const seqLen = Math.max(TCS_PROBE_SEQ, ratio * 8);
    const n = seqLen * TCS_HIDDEN_SIZE;
    const data = new Float32Array(n);
    // Inicialização senoidal determinística (mais próxima de embeddings reais)
    for (let i = 0; i < n; i++) data[i] = Math.sin(i * 0.01) * 0.02;

    const inputName = this.session.inputNames[0] ?? 'embeddings';
    const Tensor = getOnnx().Tensor;
    const t: Tensor = new Tensor('float32', data, [1, seqLen, TCS_HIDDEN_SIZE]);

    const t0 = Date.now();
    try {
      await this.session.run({ [inputName]: t });
    } catch (err) {
      console.warn('[TCSOnnxEngine] TCS forward falhou:', err);
    } finally {
      this.lastTcsLatencyMs = Date.now() - t0;
      console.log(
        `[TCSOnnxEngine] TCS ${seqLen}→${Math.ceil(seqLen / ratio)} tokens em ${this.lastTcsLatencyMs}ms`,
      );
    }
  }

  // ── Resposta aterrada no banco ────────────────────────────────────────────
  private async buildResponse(
    text: string,
    intent: string,
  ): Promise<{ speech: string; toolCalls: ParsedToolCall[] }> {
    const tod = TaskRouter.timeOfDay();

    switch (intent) {
      case 'greeting':
        return { speech: this.greetSpeech(tod), toolCalls: [] };

      case 'thanks':
        return {
          speech: this.patientName
            ? `De nada, ${this.patientName}. Pode me chamar quando quiser.`
            : 'De nada. Pode me chamar quando quiser.',
          toolCalls: [],
        };

      case 'farewell':
        return { speech: 'Até logo. Vou ficar aqui pertinho.', toolCalls: [] };

      case 'affirmation':
        return { speech: 'Tá bom. Estou aqui se precisar.', toolCalls: [] };

      case 'memory_about_person': {
        const hint = extractPersonHint(text);
        const person = hint ? await this.db.persons.findByName(hint) : null;
        if (person) {
          const [memories, last] = await Promise.all([
            this.db.memories.findByPersonId(person.id, 2),
            this.db.encounters.findLast(person.id),
          ]);
          return {
            speech: buildPersonStory(person.name, person.relationship, person.bio, memories, last),
            toolCalls: [{ name: 'read_person', parameters: { face_id: person.id } } as ParsedToolCall],
          };
        }
        // Nome não reconhecido — lista as pessoas cadastradas
        const all = await this.db.persons.findAll();
        if (all.length === 0) {
          return {
            speech:
              'Ainda não conheço ninguém. Peça para sua cuidadora me apresentar sua família.',
            toolCalls: [],
          };
        }
        const names = all
          .slice(0, 4)
          .map((p) => `${p.name} (${p.relationship})`)
          .join(', ');
        return {
          speech: `Posso te contar sobre: ${names}. Sobre quem você quer saber?`,
          toolCalls: [],
        };
      }

      case 'memory_recall': {
        const recent = await this.db.encounters.findRecent(72);
        if (recent.length === 0) {
          return {
            speech:
              'Não registrei encontros recentes. Quando alguém chegar, me avisa que eu anoto.',
            toolCalls: [],
          };
        }
        const ids = [...new Set(recent.map((e) => e.personId))].slice(0, 3);
        const people = (await Promise.all(ids.map((id) => this.db.persons.findById(id)))).filter(
          (p): p is NonNullable<typeof p> => Boolean(p),
        );
        if (people.length === 0) {
          return { speech: 'Houve algumas visitas, mas ainda não os registrei pelo nome.', toolCalls: [] };
        }
        const nameList = joinList(people.map((p) => p.name));
        return {
          speech: `Nos últimos dias passaram por aqui: ${nameList}. Quer que eu conte mais sobre alguém?`,
          toolCalls: [],
        };
      }

      case 'orientation': {
        const locs = await this.db.locations.findAll();
        if (locs.length === 0) {
          return { speech: 'Você está em casa. Respira fundo — está tudo seguro.', toolCalls: [] };
        }
        const hints = locs
          .slice(0, 3)
          .map((l) => l.navigationHint)
          .filter(Boolean)
          .join(' ');
        return {
          speech: hints ? `Você está em casa. ${hints}` : 'Você está em casa, tudo tranquilo.',
          toolCalls: [],
        };
      }

      default: {
        // Resposta de fallback com contexto real do banco
        const [peopleCount, meds] = await Promise.all([
          this.db.persons.count(),
          this.db.medications.findAll().then((m) => m.length).catch(() => 0),
        ]);
        const namePart = this.patientName ? `, ${this.patientName}` : '';
        const peoplePart =
          peopleCount > 0
            ? ` Conheço ${peopleCount} ${peopleCount === 1 ? 'pessoa' : 'pessoas'} cadastradas`
            : '';
        const medPart =
          meds > 0
            ? ` e ${meds} ${meds === 1 ? 'remédio cadastrado' : 'remédios cadastrados'}`
            : '';
        return {
          speech: `Estou aqui${namePart}.${peoplePart}${medPart}. Pode perguntar sobre remédio, agenda, rotina ou uma pessoa que você quer lembrar.`,
          toolCalls: [],
        };
      }
    }
  }

  private greetSpeech(tod: TimeOfDay): string {
    const period =
      tod === 'morning' ? 'Bom dia' : tod === 'afternoon' ? 'Boa tarde' : 'Boa noite';
    const who = this.patientName ? `, ${this.patientName}` : '';
    return `${period}${who}. Eu sou o Lane, estou aqui com você. Pode me perguntar qualquer coisa.`;
  }
}

// ── Helpers puramente funcionais (fora da classe) ────────────────────────────

function detectIntent(text: string): string {
  const t = text.trim();
  if (!t) return 'greeting';
  if (PATTERNS.GREETING.test(t)) return 'greeting';
  if (/\b(obrigad[ao]|valeu|thanks|thank\s+you)\b/i.test(t)) return 'thanks';
  if (/\b(tchau|até\s+logo|bye|goodbye|adeus)\b/i.test(t)) return 'farewell';
  if (PATTERNS.POSITIVE.test(t)) return 'affirmation';
  if (PATTERNS.MEMORY.test(t)) {
    return /\b(quem\s+[eé]|me\s+conta|conte|fale?\s+sobre|me\s+fala)\b/i.test(t)
      ? 'memory_about_person'
      : 'memory_recall';
  }
  if (PATTERNS.ORIENTATION.test(t)) return 'orientation';
  return 'general';
}

function extractPersonHint(text: string): string | null {
  const m = text.match(
    /(?:sobre|quem\s+[eé]|conte\s+sobre|me\s+fala\s+(?:da|do|de)|fale?\s+(?:da|do|de))\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇ]+)/i,
  );
  return m?.[1] ?? null;
}

function buildPersonStory(
  name: string,
  relationship: string,
  bio: string,
  memories: Array<{ description: string }>,
  last: { timestamp: number } | null,
): string {
  let s = `Essa é ${name}, sua ${relationship}. ${bio}`;
  if (memories.length > 0) s += ` Uma lembrança: ${memories[0].description}`;
  if (last) {
    const days = Math.floor((Date.now() - last.timestamp) / 86_400_000);
    if (days === 0) s += ' Vocês se viram hoje.';
    else if (days === 1) s += ' Vocês se viram ontem.';
    else if (days < 7) s += ` Vocês se viram há ${days} dias.`;
  }
  return s;
}

function joinList(items: string[]): string {
  if (items.length <= 1) return items[0] ?? '';
  return `${items.slice(0, -1).join(', ')} e ${items[items.length - 1]}`;
}
