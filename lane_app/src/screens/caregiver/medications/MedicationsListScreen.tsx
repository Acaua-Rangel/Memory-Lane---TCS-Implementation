import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet, SafeAreaView,
  StatusBar, Alert, ActivityIndicator, Switch,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { MedicationsStackParamList } from '../../../navigation/types';
import type { MedicationRecord } from '../../../types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Card } from '../../../components/ui/Card';
import { EmptyState } from '../../../components/ui/EmptyState';
import { colors, spacing, radius, typography, TIME_OF_DAY_LABELS } from '../../../theme';

type Props = NativeStackScreenProps<MedicationsStackParamList, 'MedicationsList'>;

const TIME_ICONS: Record<string, string> = {
  morning: '🌅', afternoon: '☀️', evening: '🌆', night: '🌙',
};
const TIME_COLORS: Record<string, string> = {
  morning: colors.morning, afternoon: colors.afternoon, evening: colors.evening, night: colors.night,
};

export function MedicationsListScreen({ navigation }: Props) {
  const { db, isReady } = useDatabase();
  const [meds, setMeds] = useState<MedicationRecord[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!db) return;
    setLoading(true);
    const list = await db.medications.findAll(false);
    setMeds(list);
    setLoading(false);
  }, [db]);

  useEffect(() => { if (isReady) void load(); }, [isReady, load]);
  useEffect(() => {
    return navigation.addListener('focus', () => { if (isReady) void load(); });
  }, [navigation, isReady, load]);

  const handleToggle = async (med: MedicationRecord) => {
    await db!.medications.setActive(med.id, !med.isActive);
    await load();
  };

  const handleDelete = (med: MedicationRecord) => {
    Alert.alert('Remover medicamento', `Deseja remover "${med.name}"?`, [
      { text: 'Cancelar', style: 'cancel' },
      {
        text: 'Remover', style: 'destructive',
        onPress: async () => { await db!.medications.delete(med.id); await load(); },
      },
    ]);
  };

  const grouped = meds.reduce<Record<string, MedicationRecord[]>>((acc, m) => {
    if (!acc[m.timeOfDay]) acc[m.timeOfDay] = [];
    acc[m.timeOfDay].push(m);
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
        <Text style={styles.title}>💊 Remédios</Text>
        <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('MedicationForm', {})}>
          <Text style={styles.addBtnText}>+ Adicionar</Text>
        </TouchableOpacity>
      </View>

      {meds.length === 0 ? (
        <EmptyState
          icon="💊"
          title="Nenhum medicamento cadastrado"
          description="Cadastre os remédios com horário e descrição visual para que a Lane possa lembrar o paciente."
          actionLabel="+ Adicionar remédio"
          onAction={() => navigation.navigate('MedicationForm', {})}
        />
      ) : (
        <FlatList
          data={timeOrder.filter((t) => grouped[t]?.length > 0)}
          keyExtractor={(t) => t}
          contentContainerStyle={styles.list}
          renderItem={({ item: period }) => (
            <View style={styles.section}>
              <View style={[styles.sectionHeader, { borderLeftColor: TIME_COLORS[period] }]}>
                <Text style={styles.sectionIcon}>{TIME_ICONS[period]}</Text>
                <Text style={[styles.sectionTitle, { color: TIME_COLORS[period] }]}>
                  {TIME_OF_DAY_LABELS[period]}
                </Text>
              </View>
              {grouped[period].map((med) => (
                <Card key={med.id} style={[styles.medCard, !med.isActive && styles.medCardInactive]}>
                  <View style={styles.medRow}>
                    <View style={styles.medInfo}>
                      <Text style={[styles.medName, !med.isActive && styles.medNameInactive]}>
                        {med.name}
                      </Text>
                      <Text style={styles.medDosage}>{med.dosage}</Text>
                      {med.description ? <Text style={styles.medDesc}>{med.description}</Text> : null}
                      {med.instructions ? <Text style={styles.medInstr}>📋 {med.instructions}</Text> : null}
                    </View>
                    <View style={styles.medActions}>
                      <Switch
                        value={med.isActive}
                        onValueChange={() => handleToggle(med)}
                        trackColor={{ true: colors.success }}
                        thumbColor="#fff"
                      />
                      <TouchableOpacity onPress={() => navigation.navigate('MedicationForm', { medicationId: med.id })}>
                        <Text style={{ fontSize: 16 }}>✏️</Text>
                      </TouchableOpacity>
                      <TouchableOpacity onPress={() => handleDelete(med)}>
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
    borderLeftWidth: 4, paddingLeft: spacing.sm,
  },
  sectionIcon: { fontSize: 18 },
  sectionTitle: { ...typography.h4, fontWeight: '700' },
  medCard: { marginLeft: spacing.sm },
  medCardInactive: { opacity: 0.5 },
  medRow: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' },
  medInfo: { flex: 1, gap: 2 },
  medName: { ...typography.bodyBold, color: colors.textDark },
  medNameInactive: { textDecorationLine: 'line-through', color: colors.textMuted },
  medDosage: { ...typography.small, color: colors.primary, fontWeight: '600' },
  medDesc: { ...typography.small, color: colors.textSecondary, marginTop: 2 },
  medInstr: { ...typography.small, color: colors.textMuted, marginTop: 2 },
  medActions: { alignItems: 'flex-end', gap: spacing.sm },
});
