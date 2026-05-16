// Ponto de entrada do app — exemplo mínimo para o colega (Juan) implementar as telas.
// Este arquivo demonstra como usar os hooks do sistema de agente.

import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar } from 'react-native';
import { useAgent, useVoice, useAlertSync } from './src/hooks';
import { useAgentStore, selectIsFullyReady } from './src/store';

export default function App() {
  const { isReady, processText, lastResponse, lastLatencyMs, alertService } = useAgent();
  const isFullyReady = useAgentStore(selectIsFullyReady);

  const { speak, startListening, startWakeWord, isListening, isSpeaking } = useVoice(
    async (transcript) => {
      // Quando STT captura fala, envia ao agente e fala a resposta
      const response = await processText(transcript);
      if (response?.speech) {
        await speak(response.speech);
      }
    },
  );

  const { pendingCount } = useAlertSync(alertService);

  // Inicia wake word quando o agente estiver pronto
  useEffect(() => {
    if (isFullyReady) {
      startWakeWord();
    }
  }, [isFullyReady]);

  // ─────────────────────────────────────────────────────────────────
  // O COLEGA (Juan) substituirá este componente pelas telas do Figma.
  // Os hooks acima são a API completa do sistema de agente.
  // ─────────────────────────────────────────────────────────────────
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />

      <View style={styles.statusBar}>
        <View style={[styles.dot, { backgroundColor: isFullyReady ? '#4CAF50' : '#FF9800' }]} />
        <Text style={styles.statusText}>
          {isFullyReady ? 'Lane está pronta' : 'Carregando...'}
        </Text>
        {pendingCount > 0 && (
          <Text style={styles.alertBadge}>{pendingCount} alertas pendentes</Text>
        )}
      </View>

      <Text style={styles.title}>Lane</Text>
      <Text style={styles.subtitle}>Seu companheiro de memória</Text>

      {lastResponse && (
        <View style={styles.responseBox}>
          <Text style={styles.responseText}>{lastResponse.speech}</Text>
          <Text style={styles.metaText}>
            via {lastResponse.route} · {lastLatencyMs}ms
          </Text>
        </View>
      )}

      <TouchableOpacity
        style={[styles.micButton, isListening && styles.micButtonActive]}
        onPress={startListening}
        disabled={!isFullyReady || isListening || isSpeaking}
      >
        <Text style={styles.micIcon}>{isListening ? '🎙️' : '🎤'}</Text>
        <Text style={styles.micText}>
          {isListening ? 'Ouvindo...' : isSpeaking ? 'Falando...' : 'Toque para falar'}
        </Text>
      </TouchableOpacity>

      <Text style={styles.hint}>
        Ou diga "Oi Lane" para ativar por voz
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  statusBar: {
    position: 'absolute',
    top: 56,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusText: { color: '#aaa', fontSize: 13 },
  alertBadge: {
    backgroundColor: '#FF5252',
    color: '#fff',
    fontSize: 11,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  title: {
    fontSize: 72,
    color: '#fff',
    fontWeight: '200',
    letterSpacing: 8,
  },
  subtitle: {
    color: '#666',
    fontSize: 16,
    marginTop: 8,
    marginBottom: 40,
  },
  responseBox: {
    backgroundColor: '#16213e',
    borderRadius: 16,
    padding: 20,
    width: '100%',
    marginBottom: 32,
  },
  responseText: {
    color: '#e0e0e0',
    fontSize: 18,
    lineHeight: 26,
  },
  metaText: {
    color: '#444',
    fontSize: 11,
    marginTop: 8,
  },
  micButton: {
    backgroundColor: '#4A90E2',
    borderRadius: 80,
    width: 160,
    height: 160,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#4A90E2',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 20,
    elevation: 8,
  },
  micButtonActive: {
    backgroundColor: '#E24A4A',
    shadowColor: '#E24A4A',
  },
  micIcon: { fontSize: 48 },
  micText: { color: '#fff', fontSize: 14, marginTop: 8 },
  hint: {
    color: '#444',
    fontSize: 13,
    marginTop: 24,
  },
});
