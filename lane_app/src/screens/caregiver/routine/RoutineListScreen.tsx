import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet, SafeAreaView,
  StatusBar, Alert, ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RoutineStackParamList } from '../../../navigation/types';
import type { RoutineStep } from '../../../types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Card } from '../../../components/ui/Card';
import { EmptyState } from '../../../components/ui/EmptyState';
import { colors, spacing, radius, typography, TIME_OF_DAY_LABELS } from '../../../theme';

type Props = NativeStackScreenProps<RoutineStackParamList, 'RoutineList'>;

const TIME_ICONS: Record<string, string> = {
  morning: '🌅', afternoon: '☀️', evening: '🌆', night: '🌙',
};
const TIME_COLORS: Record<string, string> = {
  morning: colors.morning, afternoon: colors.afternoon, evening: colors.evening, night: colors.night,
};

export function RoutineListScreen({ navigation }: Props) {
  const { db, isReady } = useDatabase();
  const [steps, setSteps] = useState<RoutineStep[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!db) return;
    setLoading(true);
    const list = await db.routines.findAll();
    setSteps(list);
    setLoading(false);
  }, [db]);

  useEffect(() => { if (isReady) void load(); }, [isReady, load]);
  useEffect(() => {
    return navigation.addListener('focus', () => { if (isReady) void load(); });
  }, [navigation, isReady, load]);

  const handleDelete = (step: RoutineStep) => {
    Alert.alert('Remover passo', `Deseja remover "${step.title}"?`, [
      { text: 'Cancelar', style: 'cancel' },
      {
        text: 'Remover', style: 'destructive',
        onPress: async () => { await db!.routines.delete(step.id); await load(); },
      },
    ]);
  };

  const grouped = steps.reduce<Record<string, RoutineStep[]>>((acc, s) => {
    if (!acc[s.timeOfDay]) acc[s.timeOfDay] = [];
    acc[s.timeOfDay].push(s);
    return acc;
  }, {});

  const timeOrder = ['morning', 'afternoon', 'evening', 'night'];

  if (!isReady || loading) {
    return <SafeAreaView style={styles.safe}><ActivityIndicator style={{ flex: 1 }} color={colors.primary} /></SafeAreaView>;
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <Text style={styles.title}>📋 Rotina</Text>
        <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('RoutineForm', {})}>
          <Text style={styles.addBtnText}>+ Adicionar</Text>
        </TouchableOpacity>
      </View>

      {steps.length === 0 ? (
        <EmptyState
          icon="📋"
          title="Nenhum passo de rotina"
          description="Adicione as atividades diárias do paciente para que a Lane possa guiá-lo."
          actionLabel="+ Adicionar passo"
          onAction={() => navigation.navigate('RoutineForm', {})}
        />
      ) : (
        <FlatList
          data={timeOrder.filter((t) => grouped[t]?.length > 0)}
          keyExtractor={(t) => t}
          contentContainerStyle={styles.list}
          renderItem={({ item: period }) => (
            <View style={styles.section}>
              <View style={[styles.sectionHeader, { backgroundColor: TIME_COLORS[period] + '20' }]}>
                <Text style={styles.sectionIcon}>{TIME_ICONS[period]}</Text>
                <Text style={[styles.sectionTitle, { color: TIME_COLORS[period] }]}>
                  {TIME_OF_DAY_LABELS[period]}
                </Text>
              </View>
              {grouped[period].sort((a, b) => a.order - b.order).map((step) => (
                <Card key={step.id} style={styles.stepCard}>
                  <View style={styles.stepRow}>
                    <View style={[styles.orderBadge, { backgroundColor: TIME_COLORS[period] }]}>
                      <Text style={styles.orderText}>{step.order}</Text>
                    </View>
                    <View style={styles.stepInfo}>
                      <Text style={styles.stepTitle}>{step.title}</Text>
                      <Text style={styles.stepDesc}>{step.description}</Text>
                    </View>
                    <View style={styles.stepActions}>
                      <TouchableOpacity onPress={() => navigation.navigate('RoutineForm', { routineId: step.id })}>
                        <Text style={{ fontSize: 16 }}>✏️</Text>
                      </TouchableOpacity>
                      <TouchableOpacity onPress={() => handleDelete(step)}>
                        <Text style={{ fontSize: 16 }}>🗑️</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                </Card>
              ))}
            </View>
          )}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgLight },
  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    padding: spacing.lg, backgroundColor: colors.surfaceLight,
    borderBottomWidth: 1, borderBottomColor: colors.borderLight,
  },
  title: { ...typography.h2, color: colors.textDark },
  addBtn: { backgroundColor: colors.primary, paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: radius.full },
  addBtnText: { ...typography.small, color: '#fff', fontWeight: '600' },
  list: { padding: spacing.md, gap: spacing.md },
  section: { gap: spacing.sm },
  sectionHeader: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.sm,
    padding: spacing.sm, borderRadius: radius.md,
  },
  sectionIcon: { fontSize: 20 },
  sectionTitle: { ...typography.h4, fontWeight: '700' },
  stepCard: { marginLeft: spacing.sm },
  stepRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  orderBadge: {
    width: 32, height: 32, borderRadius: 16,
    alignItems: 'center', justifyContent: 'center',
  },
  orderText: { ...typography.small, color: '#fff', fontWeight: '700' },
  stepInfo: { flex: 1 },
  stepTitle: { ...typography.bodyBold, color: colors.textDark },
  stepDesc: { ...typography.small, color: colors.textSecondary, marginTop: 2 },
  stepActions: { flexDirection: 'row', gap: spacing.sm },
});
