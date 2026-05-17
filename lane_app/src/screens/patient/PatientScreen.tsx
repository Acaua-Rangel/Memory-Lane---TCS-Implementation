import React, { useEffect } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, SafeAreaView, StatusBar, ScrollView,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../../navigation/types';
import { useAgent } from '../../hooks/useAgent';
import { useVoice } from '../../hooks/useVoice';
import { useAlertSync } from '../../hooks/useAlertSync';
import { useAgentStore, selectIsFullyReady } from '../../store';
import { colors, spacing, radius, typography } from '../../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'Patient'>;

export function PatientScreen({ navigation }: Props) {
  const { isReady, processText, lastResponse, lastLatencyMs, alertService } = useAgent();
  const isFullyReady = useAgentStore(selectIsFullyReady);

  const { speak, startListening, startWakeWord, isListening, isSpeaking } = useVoice(
    async (transcript) => {
      const response = await processText(transcript);
      if (response?.speech) await speak(response.speech);
    },
  );

  const { pendingCount } = useAlertSync(alertService);

  useEffect(() => {
    if (isFullyReady) startWakeWord();
  }, [isFullyReady]);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor={colors.bgDark} />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.statusRow}>
          <View style={[styles.dot, { backgroundColor: isFullyReady ? colors.success : colors.warning }]} />
          <Text style={styles.statusText}>
            {isFullyReady ? 'Lane está pronta' : 'Carregando...'}
          </Text>
          {pendingCount > 0 && (
            <View style={styles.alertBadge}>
              <Text style={styles.alertBadgeText}>{pendingCount}</Text>
            </View>
          )}
        </View>
        <TouchableOpacity onPress={() => navigation.replace('ModeSelector')}>
          <Text style={styles.switchBtn}>⚙️</Text>
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {/* Title */}
        <Text style={styles.title}>Lane</Text>
        <Text style={styles.subtitle}>Seu companheiro de memória</Text>

        {/* Response box */}
        {lastResponse ? (
          <View style={styles.responseBox}>
            <Text style={styles.responseText}>{lastResponse.speech}</Text>
            <Text style={styles.responseMeta}>
              {lastResponse.route} · {lastLatencyMs}ms
            </Text>
          </View>
        ) : (
          <View style={styles.hintBox}>
            <Text style={styles.hintBoxText}>
              💡 Experimente perguntar:{'\n\n'}
              "Quem é essa pessoa?"{'\n'}
              "Que remédio devo tomar agora?"{'\n'}
              "O que tenho hoje na agenda?"{'\n'}
              "Qual é minha rotina da manhã?"
            </Text>
          </View>
        )}

        {/* Mic button */}
        <TouchableOpacity
          style={[
            styles.micBtn,
            isListening && styles.micBtnListening,
            isSpeaking && styles.micBtnSpeaking,
            !isFullyReady && styles.micBtnDisabled,
          ]}
          onPress={startListening}
          disabled={!isFullyReady || isListening || isSpeaking}
          activeOpacity={0.8}
        >
          <Text style={styles.micIcon}>
            {isListening ? '🎙️' : isSpeaking ? '🔊' : '🎤'}
          </Text>
          <Text style={styles.micText}>
            {isListening ? 'Ouvindo...' : isSpeaking ? 'Falando...' : 'Toque para falar'}
          </Text>
        </TouchableOpacity>

        <Text style={styles.wakeHint}>Ou diga "Oi Lane" para ativar por voz</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgDark },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  dot: { width: 10, height: 10, borderRadius: 5 },
  statusText: { ...typography.small, color: colors.textLightMuted },
  alertBadge: {
    backgroundColor: colors.danger,
    borderRadius: radius.full,
    width: 20, height: 20,
    alignItems: 'center', justifyContent: 'center',
  },
  alertBadgeText: { ...typography.caption, color: '#fff', fontWeight: '700' },
  switchBtn: { fontSize: 22 },

  content: {
    alignItems: 'center',
    padding: spacing.xl,
    gap: spacing.xl,
    paddingBottom: spacing.xxl,
  },

  title: { fontSize: 64, color: '#fff', fontWeight: '200', letterSpacing: 10 },
  subtitle: { ...typography.body, color: colors.textLightMuted },

  responseBox: {
    backgroundColor: colors.surfaceDark,
    borderRadius: radius.xl,
    padding: spacing.lg,
    width: '100%',
    gap: spacing.sm,
  },
  responseText: { ...typography.h4, color: '#e0e0e0', lineHeight: 26 },
  responseMeta: { ...typography.caption, color: '#555' },

  hintBox: {
    backgroundColor: colors.surfaceDark,
    borderRadius: radius.xl,
    padding: spacing.lg,
    width: '100%',
    borderWidth: 1,
    borderColor: colors.surfaceDark2,
  },
  hintBoxText: { ...typography.body, color: colors.textLightMuted, lineHeight: 26 },

  micBtn: {
    backgroundColor: colors.primary,
    borderRadius: 90,
    width: 180,
    height: 180,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 24,
    elevation: 10,
    gap: spacing.sm,
  },
  micBtnListening: { backgroundColor: colors.danger, shadowColor: colors.danger },
  micBtnSpeaking: { backgroundColor: colors.success, shadowColor: colors.success },
  micBtnDisabled: { backgroundColor: '#333', shadowOpacity: 0 },
  micIcon: { fontSize: 52 },
  micText: { ...typography.small, color: '#fff' },

  wakeHint: { ...typography.small, color: '#333', textAlign: 'center' },
});
