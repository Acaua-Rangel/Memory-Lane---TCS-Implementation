// Port TypeScript dos padrões regex de src/routing.py:TaskRouter
// Bilíngue: português (principal) + inglês (fallback)

export const PATTERNS = {
  // Rota 1 — Face pipeline
  FACE: /\b(face|rosto|pessoa|who\s+is|quem\s+[eé]|quem\s+essa|reconhece|este\s+[eé]|essa\s+[eé]|esse\s+homem|essa\s+mulher)\b/i,

  // Rota 2 — SQLite direto (medicamentos — sem LLM)
  MEDICATION: /\b(rem[eé]dio|medicamento|comp[rr]imido|c[aá]psula|dose|tomar|pill|medication|med[s]?|hor[aá]rio.*rem[eé]dio|rem[eé]dio.*hora)\b/i,

  // Rota 3 — Regra direta (alertas — latência 100ms)
  ALERT: /\b(perdido|lost|ajuda|help|socorro|confused|confuso|emerg[eê]ncia|emergency|n[aã]o\s+sei\s+onde|where\s+am\s+i|cadê|onde\s+estou|me\s+perdi)\b/i,

  // Rota 4 — LLM qualidade (memórias — latência 2s, ratio=2)
  MEMORY: /\b(lembr[ao]|lembra|remember|history|hist[oó]ria|conta|me\s+conta|me\s+fala|quem\s+[eé]\s+\w+|who\s+is\s+\w+|conte|fala\s+sobre|fale\s+sobre)\b/i,

  // Rota 5 — Orientação/localização (LLM balanceado, ratio=4)
  ORIENTATION: /\b(onde|where|lugar|place|localiz|estou|ambiente|c[oô]modo|quarto|sala|cozinha|banheiro|bathroom|kitchen|bedroom|living\s+room|aqui\s+[eé]|o\s+que\s+[eé]\s+esse\s+lugar)\b/i,

  // Detectores auxiliares
  GREETING: /^(oi|olá|ola|bom\s+dia|boa\s+tarde|boa\s+noite|hey|hi|hello|lane)\b/i,
  AGENDA: /\b(agenda|compromisso|consulta|appointment|schedule|hoje\s+tenho|o\s+que\s+tenho|o\s+que\s+tem\s+hoje)\b/i,
  ROUTINE: /\b(rotina|routine|o\s+que\s+fa[cç]o|what\s+do\s+i\s+do|pr[oó]xim[oa]\s+passo|next\s+step)\b/i,
  POSITIVE: /\b(sim|yes|ok|certo|tudo\s+bem|obrigad[ao]|entendi|ok\s+entendi)\b/i,
};

// Mapeia padrão para intenção legível
export const INTENT_LABELS: Record<keyof typeof PATTERNS, string> = {
  FACE: 'face_recognition',
  MEDICATION: 'medication_query',
  ALERT: 'emergency_alert',
  MEMORY: 'memory_recall',
  ORIENTATION: 'location_orientation',
  GREETING: 'greeting',
  AGENDA: 'agenda_query',
  ROUTINE: 'routine_query',
  POSITIVE: 'affirmation',
};
