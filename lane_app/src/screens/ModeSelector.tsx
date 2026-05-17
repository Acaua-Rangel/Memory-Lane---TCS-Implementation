import React from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, SafeAreaView, StatusBar,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';
import { colors, spacing, radius, typography, shadow } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'ModeSelector'>;

export function ModeSelector({ navigation }: Props) {
  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.logo}>🧠</Text>
          <Text style={styles.title}>Memory Lane</Text>
          <Text style={styles.subtitle}>Como você deseja usar o app?</Text>
        </View>

        <View style={styles.cards}>
          <TouchableOpacity
            style={[styles.card, styles.cardPatient]}
            onPress={() => navigation.replace('Patient')}
            activeOpacity={0.85}
          >
            <Text style={styles.cardIcon}>🎤</Text>
            <Text style={styles.cardTitle}>Modo Paciente</Text>
            <Text style={styles.cardDesc}>
              Conversar com a Lane por voz para relembrar pessoas, rotinas e memórias
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.card, styles.cardCaregiver]}
            onPress={() => navigation.replace('Caregiver')}
            activeOpacity={0.85}
          >
            <Text style={styles.cardIcon}>📝</Text>
            <Text style={styles.cardTitle}>Modo Cuidador</Text>
            <Text style={styles.cardDesc}>
              Cadastrar pessoas, memórias, agenda, rotina e medicamentos
            </Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.footer}>
          Toque no modo desejado para continuar
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgLight },
  container: {
    flex: 1,
    padding: spacing.xl,
    justifyContent: 'center',
    gap: spacing.xl,
  },
  header: { alignItems: 'center', gap: spacing.sm },
  logo: { fontSize: 56 },
  title: { ...typography.h1, color: colors.textDark },
  subtitle: { ...typography.body, color: colors.textSecondary, textAlign: 'center' },

  cards: { gap: spacing.md },

  card: {
    borderRadius: radius.xl,
    padding: spacing.xl,
    gap: spacing.sm,
    ...shadow.md,
  },
  cardPatient: { backgroundColor: colors.bgDark },
  cardCaregiver: { backgroundColor: colors.primary },

  cardIcon: { fontSize: 36 },
  cardTitle: { ...typography.h2, color: '#fff' },
  cardDesc: { ...typography.body, color: 'rgba(255,255,255,0.8)', lineHeight: 22 },

  footer: { ...typography.small, color: colors.textMuted, textAlign: 'center' },
});
