// Orquestrador central — espelha src/routing.py:RoutedCompanion
//
// Fluxo por rota:
//   face_pipeline  → FacePipeline → ToolExecutor(read_person) → LLM resposta
//   sqlite_direct  → ToolExecutor direto → formata resposta sem LLM
//   rule_based     → lógica instantânea → CaregiverAlertService
//   llm_fast       → GemmaEngine(r=8) → ToolExecutor (se tool call) → resposta
//   llm_quality    → GemmaEngine(r=2) → ToolExecutor (se tool call) → resposta

import type { IGemmaEngine } from '../engine/GemmaEngine';
import type { FacePipeline } from '../face/FacePipeline';
import type { TaskRouter } from '../routing/TaskRouter';
import type { LaneDatabase } from '../database/LaneDatabase';
import type { ToolExecutor } from '../tools/ToolExecutor';
import type { CaregiverAlertService } from '../alerts/CaregiverAlertService';
import type {
  AgentInput,
  AgentResponse,
  RouteDecision,
  CameraFrame,
  FaceDetection,
  IdentifiedPerson,
} from '../types';
import { formatToolResult } from '../tools/toolParser';
import { TaskRouter as TR } from '../routing/TaskRouter';

export class LaneAgent {
  constructor(
    private readonly router: TaskRouter,
    private readonly facePipeline: FacePipeline,
    private readonly engine: IGemmaEngine,
    private readonly db: LaneDatabase,
    private readonly toolExecutor: ToolExecutor,
    private readonly alertService: CaregiverAlertService,
  ) {}

  async processInput(input: AgentInput): Promise<AgentResponse> {
    const start = Date.now();
    const decision = this.router.classify(input);

    console.log('[LaneAgent]', this.router.explain(decision));

    try {
      switch (decision.route) {
        case 'face_pipeline':
          return await this.executeFacePipeline(input, decision, start);

        case 'sqlite_direct':
          return await this.executeSQLiteDirect(input, decision, start);

        case 'rule_based':
          return await this.executeRuleBased(input, decision, start);

        case 'llm_fast':
        case 'llm_quality':
          return await this.executeLLMPath(input, decision, start);

        default:
          return this.fallbackResponse(start);
      }
    } catch (err) {
      console.error('[LaneAgent] Erro no processamento:', err);
      return {
        speech: 'Desculpe, tive um problema. Pode repetir?',
        route: decision.route,
        latencyMs: Date.now() - start,
      };
    }
  }

  // ── Rota 1: Face Pipeline ──────────────────────────────────────────────────
  private async executeFacePipeline(
    input: AgentInput,
    decision: RouteDecision,
    start: number,
  ): Promise<AgentResponse> {
    if (!input.frame || !input.faceDetected) {
      return {
        speech: 'Não consegui ver o rosto claramente. Pode se aproximar um pouco?',
        route: decision.route,
        latencyMs: Date.now() - start,
      };
    }

    // Embedder + cosine search
    const mockDetection: FaceDetection = {
      boundingBox: { x: 0, y: 0, width: input.frame.width, height: input.frame.height },
      probability: 1.0,
    };

    const pipelineResult = await this.facePipeline.onFaceDetected(input.frame, mockDetection);

    if (!pipelineResult.identified) {
      // Rosto desconhecido — alerta silencioso ao cuidador
      await this.alertService.sendAlert('confused', 'Pessoa desconhecida detectada perto do paciente');
      return {
        speech: 'Não reconheci essa pessoa. Chamei o cuidador para confirmar.',
        alertSent: true,
        route: decision.route,
        latencyMs: Date.now() - start,
      };
    }

    const { person, similarity } = pipelineResult.identified;
    const lastEncounter = await this.db.encounters.findLast(person.id);
    const memories = await this.db.memories.findByPersonId(person.id, 2);

    // Registra o encontro
    await this.db.encounters.insert({
      personId: person.id,
      context: `Reconhecido pela câmera com similaridade ${similarity.toFixed(2)}`,
    });

    // Monta resposta baseada no relacionamento
    const speech = this.buildPersonRecognitionResponse(pipelineResult.identified, lastEncounter);

    return {
      speech,
      identifiedPerson: person,
      route: decision.route,
      latencyMs: Date.now() - start,
    };
  }

  // ── Rota 2: SQLite Direto ─────────────────────────────────────────────────
  private async executeSQLiteDirect(
    input: AgentInput,
    decision: RouteDecision,
    start: number,
  ): Promise<AgentResponse> {
    let speech = '';
    const toolResults = [];

    if (decision.intent === 'medication_query') {
      const timeOfDay = TR.timeOfDay();
      const result = await this.toolExecutor.execute({
        name: 'get_medication',
        parameters: { time_of_day: timeOfDay },
      });
      toolResults.push(result);
      speech = this.formatMedicationResponse(result.data as formatMedData);
    } else if (decision.intent === 'agenda_query') {
      const today = new Date().getDay();
      const result = await this.toolExecutor.execute({
        name: 'get_agenda',
        parameters: { day_of_week: String(today) },
      });
      toolResults.push(result);
      speech = this.formatAgendaResponse(result.data as formatAgendaData);
    } else if (decision.intent === 'routine_query') {
      const timeOfDay = TR.timeOfDay();
      const result = await this.toolExecutor.execute({
        name: 'get_routine',
        parameters: { time_of_day: timeOfDay },
      });
      toolResults.push(result);
      speech = this.formatRoutineResponse(result.data as formatRoutineData);
    }

    if (!speech) speech = 'Não encontrei essa informação. Pode me dar mais detalhes?';

    return { speech, toolResults, route: decision.route, latencyMs: Date.now() - start };
  }

  // ── Rota 3: Regra Direta (alertas) ────────────────────────────────────────
  private async executeRuleBased(
    input: AgentInput,
    decision: RouteDecision,
    start: number,
  ): Promise<AgentResponse> {
    const text = input.text?.toLowerCase() ?? '';

    let speech = '';
    let alertSent = false;

    if (/perdido|me\s+perdi|onde\s+estou|where\s+am\s+i/.test(text)) {
      await this.alertService.sendAlert('wandering', `Paciente relatou estar perdido: "${input.text}"`);
      speech = 'Não se preocupe, você está em casa! Já avisei sua família. Um momento.';
      alertSent = true;
    } else if (/ajuda|socorro|help/.test(text)) {
      await this.alertService.sendAlert('emergency', `Paciente pediu ajuda: "${input.text}"`);
      speech = 'Estou chamando ajuda para você agora! Aguarde um momento.';
      alertSent = true;
    } else if (/confus[ao]|confused/.test(text)) {
      await this.alertService.sendAlert('confused', `Paciente relatou confusão: "${input.text}"`);
      speech = 'Tudo bem, vou te ajudar. Avisei sua família também.';
      alertSent = true;
    } else {
      speech = 'Estou aqui. O que você precisa?';
    }

    return { speech, alertSent, route: decision.route, latencyMs: Date.now() - start };
  }

  // ── Rota 4/5: LLM (rápido ou qualidade) ──────────────────────────────────
  private async executeLLMPath(
    input: AgentInput,
    decision: RouteDecision,
    start: number,
  ): Promise<AgentResponse> {
    if (!this.engine.isReady()) {
      return {
        speech: 'Ainda estou carregando. Por favor, aguarde um momento.',
        route: decision.route,
        latencyMs: Date.now() - start,
      };
    }

    const text = input.text ?? 'Hello';
    const ratio = decision.compressionRatio ?? 4;

    const llmResult = await this.engine.generateWithTools(text, ratio);
    const primarySpeech = llmResult.finalResponse.trim();
    let toolResults: Awaited<ReturnType<typeof this.toolExecutor.executeAll>> | undefined;

    // Executa as tools (efeito colateral: registrar encontro, etc.) e, se o
    // engine ainda não tiver fechado a resposta, refina com o contexto.
    if (llmResult.parsedCalls.length > 0) {
      toolResults = await this.toolExecutor.executeAll(llmResult.parsedCalls);

      if (!primarySpeech) {
        const toolContext = toolResults
          .map((r) => formatToolResult(r.toolName, r.data, r.error))
          .join('\n');
        const refined = await this.engine.generateWithTools(
          `${text}\n\nResultados das ferramentas:\n${toolContext}`,
          ratio,
        );
        return {
          speech: refined.finalResponse.trim() || this.intentFallback(decision, text),
          toolResults,
          route: decision.route,
          latencyMs: Date.now() - start,
        };
      }
    }

    return {
      speech: primarySpeech || this.intentFallback(decision, text),
      toolResults,
      route: decision.route,
      latencyMs: Date.now() - start,
    };
  }

  // Fallback usado quando o engine devolve string vazia. Cada intent tem uma
  // frase específica — nunca mais "Entendi. Posso ajudar com mais alguma coisa?".
  private intentFallback(decision: RouteDecision, text: string): string {
    switch (decision.intent) {
      case 'greeting':
        return 'Oi, estou aqui com você.';
      case 'memory_recall':
        return 'Posso te contar sobre as pessoas que registrei. Me diz um nome ou pergunta "quem passou aqui hoje?".';
      case 'location_orientation':
        return 'Você está em casa. Vamos juntos: respira fundo, está tudo seguro.';
      case 'scene_description':
        return 'Estou vendo o ambiente. Pode me perguntar onde está ou quem está aí.';
      case 'general_query':
      default:
        return `Não peguei essa: "${text.trim()}". Tente perguntar sobre remédio, agenda ou alguém que você queira lembrar.`;
    }
  }

  // ── Formatadores de resposta ──────────────────────────────────────────────
  private buildPersonRecognitionResponse(
    identified: IdentifiedPerson,
    lastEncounter: { timestamp: number; context: string } | null,
  ): string {
    const { person } = identified;
    const rel = person.relationship;

    let response = `Essa é ${person.name}, sua ${rel}. `;
    response += person.bio;

    if (lastEncounter) {
      const daysSince = Math.floor((Date.now() - lastEncounter.timestamp) / (1000 * 60 * 60 * 24));
      if (daysSince === 0) {
        response += ' Vocês se viram hoje!';
      } else if (daysSince === 1) {
        response += ' Vocês se viram ontem.';
      } else if (daysSince < 7) {
        response += ` Vocês se viram há ${daysSince} dias.`;
      } else {
        response += ` A última visita foi há ${daysSince} dias.`;
      }
    }

    return response;
  }

  private formatMedicationResponse(data: unknown): string {
    const d = data as { found: boolean; medications?: Array<{ name: string; dosage: string; instructions: string }> };
    if (!d?.found || !d.medications || d.medications.length === 0) {
      return 'Você não tem remédios para tomar agora. Tudo certo!';
    }
    const meds = d.medications.map((m) => `${m.name} — ${m.dosage}. ${m.instructions}`).join('. ');
    return `Seus remédios agora: ${meds}`;
  }

  private formatAgendaResponse(data: unknown): string {
    const d = data as { count: number; dayName: string; events?: Array<{ title: string; time?: string; location?: string }> };
    if (!d || d.count === 0) return 'Você não tem compromissos hoje. Dia livre!';
    const events = (d.events ?? []).map((e) => {
      let text = e.title;
      if (e.time) text += ` às ${e.time}`;
      if (e.location) text += ` em ${e.location}`;
      return text;
    }).join('; ');
    return `Hoje, ${d.dayName}, você tem: ${events}`;
  }

  private formatRoutineResponse(data: unknown): string {
    const d = data as { steps?: Array<{ title: string; description: string }> };
    if (!d?.steps || d.steps.length === 0) return 'Nenhuma rotina configurada para agora.';
    const next = d.steps[0];
    return `Próximo passo: ${next.title}. ${next.description}`;
  }

  private fallbackResponse(start: number): AgentResponse {
    return {
      speech: 'Estou aqui para ajudar. O que você precisa?',
      route: 'llm_fast',
      latencyMs: Date.now() - start,
    };
  }
}

// Tipos internos auxiliares — evitam any
type formatMedData = { found: boolean; medications?: Array<{ name: string; dosage: string; instructions: string }> };
type formatAgendaData = { count: number; dayName: string; events?: Array<{ title: string; time?: string; location?: string }> };
type formatRoutineData = { steps?: Array<{ title: string; description: string }> };
