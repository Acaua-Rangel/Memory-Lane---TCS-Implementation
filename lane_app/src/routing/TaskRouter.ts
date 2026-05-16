// Port TypeScript de src/routing.py:TaskRouter
// Classifica a entrada e decide qual pipeline executar (zero overhead — só regex).

import type { RouteDecision, AgentInput, RouteType } from '../types';
import { PATTERNS, INTENT_LABELS } from './patterns';
import { COMPRESSION_RATIOS } from '../config/constants';

export class TaskRouter {
  // Rota principal — decide o pipeline baseado no input
  classify(input: AgentInput): RouteDecision {
    // Face detectada pela câmera tem prioridade máxima
    if (input.faceDetected) {
      return this.makeFaceRoute();
    }

    if (input.text) {
      return this.classifyText(input.text);
    }

    // Input de frame sem face detectada — orientação
    if (input.frame) {
      return {
        route: 'llm_fast',
        compressionRatio: COMPRESSION_RATIOS.FAST,
        confidence: 0.6,
        intent: 'scene_description',
      };
    }

    return this.defaultRoute();
  }

  private classifyText(text: string): RouteDecision {
    const t = text.trim();

    // 1. Alerta — resposta imediata, regra direta
    if (PATTERNS.ALERT.test(t)) {
      return {
        route: 'rule_based',
        confidence: 0.95,
        intent: INTENT_LABELS.ALERT,
      };
    }

    // 2. Medicamento — SQLite direto, sem LLM
    if (PATTERNS.MEDICATION.test(t)) {
      return {
        route: 'sqlite_direct',
        confidence: 0.95,
        intent: INTENT_LABELS.MEDICATION,
      };
    }

    // 3. Reconhecimento facial explícito ("quem é essa pessoa?")
    if (PATTERNS.FACE.test(t)) {
      return this.makeFaceRoute();
    }

    // 4. Memória/história — LLM com alta qualidade
    if (PATTERNS.MEMORY.test(t)) {
      return {
        route: 'llm_quality',
        compressionRatio: COMPRESSION_RATIOS.QUALITY,
        confidence: 0.85,
        intent: INTENT_LABELS.MEMORY,
      };
    }

    // 5. Orientação/localização — LLM balanceado
    if (PATTERNS.ORIENTATION.test(t)) {
      return {
        route: 'llm_fast',
        compressionRatio: COMPRESSION_RATIOS.BALANCED,
        confidence: 0.8,
        intent: INTENT_LABELS.ORIENTATION,
      };
    }

    // 6. Agenda/compromissos — SQLite direto
    if (PATTERNS.AGENDA.test(t)) {
      return {
        route: 'sqlite_direct',
        confidence: 0.9,
        intent: INTENT_LABELS.AGENDA,
      };
    }

    // 7. Rotina — SQLite direto
    if (PATTERNS.ROUTINE.test(t)) {
      return {
        route: 'sqlite_direct',
        confidence: 0.9,
        intent: INTENT_LABELS.ROUTINE,
      };
    }

    // 8. Saudação — LLM rápido
    if (PATTERNS.GREETING.test(t)) {
      return {
        route: 'llm_fast',
        compressionRatio: COMPRESSION_RATIOS.FAST,
        confidence: 0.9,
        intent: INTENT_LABELS.GREETING,
      };
    }

    // Default: LLM balanceado
    return {
      route: 'llm_fast',
      compressionRatio: COMPRESSION_RATIOS.BALANCED,
      confidence: 0.5,
      intent: 'general_query',
    };
  }

  private makeFaceRoute(): RouteDecision {
    return {
      route: 'face_pipeline',
      compressionRatio: COMPRESSION_RATIOS.BALANCED,
      confidence: 0.99,
      intent: INTENT_LABELS.FACE,
    };
  }

  private defaultRoute(): RouteDecision {
    return {
      route: 'llm_fast',
      compressionRatio: COMPRESSION_RATIOS.FAST,
      confidence: 0.4,
      intent: 'unknown',
    };
  }

  // Converte a hora do dia para o enum de medicamentos/rotina
  static timeOfDay(): 'morning' | 'afternoon' | 'evening' | 'night' {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 17) return 'afternoon';
    if (hour >= 17 && hour < 21) return 'evening';
    return 'night';
  }

  // Explica a decisão de rota (útil para debug/logs)
  explain(decision: RouteDecision): string {
    const ratioStr = decision.compressionRatio ? ` (ratio=${decision.compressionRatio})` : '';
    return `route=${decision.route}${ratioStr} intent=${decision.intent} conf=${decision.confidence.toFixed(2)}`;
  }
}
