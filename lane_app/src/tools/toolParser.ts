import type { ParsedToolCall, ToolName } from '../types';

// Gemma 4 gera tool calls em dois formatos possíveis:
// 1. JSON puro: {"name": "get_medication", "parameters": {"time_of_day": "morning"}}
// 2. Dentro de bloco <tool_call>...</tool_call>

const TOOL_CALL_TAG_RE = /<tool_call>([\s\S]*?)<\/tool_call>/gi;
const KNOWN_TOOLS = new Set<ToolName>([
  'read_person',
  'write_encounter',
  'get_medication',
  'describe_location',
  'alert_caregiver',
  'get_agenda',
  'save_preference',
  'get_preferences',
  'get_routine',
]);

function tryParseJson(text: string): ParsedToolCall | null {
  try {
    const parsed = JSON.parse(text.trim()) as unknown;
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      'name' in parsed &&
      typeof (parsed as Record<string, unknown>).name === 'string' &&
      KNOWN_TOOLS.has((parsed as Record<string, unknown>).name as ToolName)
    ) {
      const call = parsed as { name: ToolName; parameters?: Record<string, unknown> };
      return { name: call.name, parameters: call.parameters ?? {} };
    }
    return null;
  } catch {
    return null;
  }
}

export function parseToolCall(llmOutput: string): ParsedToolCall | null {
  // Tenta extrair de bloco <tool_call>
  const tagMatches = [...llmOutput.matchAll(TOOL_CALL_TAG_RE)];
  if (tagMatches.length > 0) {
    return tryParseJson(tagMatches[0][1]);
  }

  // Tenta encontrar JSON direto na resposta
  const jsonStart = llmOutput.indexOf('{');
  const jsonEnd = llmOutput.lastIndexOf('}');
  if (jsonStart !== -1 && jsonEnd > jsonStart) {
    return tryParseJson(llmOutput.slice(jsonStart, jsonEnd + 1));
  }

  return null;
}

export function parseAllToolCalls(llmOutput: string): ParsedToolCall[] {
  const calls: ParsedToolCall[] = [];

  // Extrai todos os blocos <tool_call>
  const tagMatches = [...llmOutput.matchAll(TOOL_CALL_TAG_RE)];
  for (const match of tagMatches) {
    const parsed = tryParseJson(match[1]);
    if (parsed) calls.push(parsed);
  }

  // Fallback: tenta o output inteiro como um único JSON
  if (calls.length === 0) {
    const single = parseToolCall(llmOutput);
    if (single) calls.push(single);
  }

  return calls;
}

// Formata o resultado de uma tool para reinjetar no contexto do LLM
export function formatToolResult(toolName: ToolName, data: unknown, error?: string): string {
  if (error) {
    return `<tool_response>\n{"tool": "${toolName}", "error": "${error}"}\n</tool_response>`;
  }
  return `<tool_response>\n${JSON.stringify({ tool: toolName, result: data }, null, 2)}\n</tool_response>`;
}
