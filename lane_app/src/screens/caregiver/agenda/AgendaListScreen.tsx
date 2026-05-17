import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet, SafeAreaView,
  StatusBar, Alert, ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { AgendaStackParamList } from '../../../navigation/types';
import type { AgendaRecord } from '../../../types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Card } from '../../../components/ui/Card';
import { EmptyState } from '../../../components/ui/EmptyState';
import { colors, spacing, radius, typography, DAY_NAMES_FULL } from '../../../theme';

type Props = NativeStackScreenProps<AgendaStackParamList, 'AgendaList'>;

const DAY_COLORS = ['#EF4444', '#6B7280', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899'];

export function AgendaListScreen({ navigation }: Props) {
  const { db, isReady } = useDatabase();
  const [events, setEvents] = useState<AgendaRecord[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!db) return;
    setLoading(true);
    const list = await db.agenda.findAll();
    setEvents(list);
    setLoading(false);
  }, [db]);

  useEffect(() => { if (isReady) void load(); }, [isReady, load]);
  useEffect(() => {
    return navigation.addListener('focus', () => { if (isReady) void load(); });
  }, [navigation, isReady, load]);

  const handleDelete = (event: AgendaRecord) => {
    Alert.alert('Remover evento', `Deseja remover "${event.title}"?`, [
      { text: 'Cancelar', style: 'cancel' },
      {
        text: 'Remover', style: 'destructive',
        onPress: async () => { await db!.agenda.delete(event.id); await load(); },
      },
    ]);
  };

  const grouped = events.reduce<Record<number, AgendaRecord[]>>((acc, e) => {
    if (!acc[e.dayOfWeek]) acc[e.dayOfWeek] = [];
    acc[e.dayOfWeek].push(e);
    return acc;
  }, {});

  if (!isReady || loading) {
    return <SafeAreaView style={styles.safe}><ActivityIndicator style={{ flex: 1 }} color={colors.primary} /></SafeAreaView>;
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <Text style={styles.title}>📅 Agenda</Text>
        <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('AgendaForm', {})}>
          <Text style={styles.addBtnText}>+ Adicionar</Text>
        </TouchableOpacity>
      </View>

      {events.length === 0 ? (
        <EmptyState
          icon="📅"
          title="Nenhum evento cadastrado"
          description="Adicione compromissos e atividades semanais para que a Lane lembre o paciente."
          actionLabel="+ Adicionar evento"
          onAction={() => navigation.navigate('AgendaForm', {})}
        />
      ) : (
        <FlatList
          data={[0, 1, 2, 3, 4, 5, 6].filter((d) => grouped[d]?.length > 0)}
          keyExtractor={(day) => String(day)}
          contentContainerStyle={styles.list}
          renderItem={({ item: day }) => (
            <View style={styles.daySection}>
              <View style={[styles.dayHeader, { borderLeftColor: DAY_COLORS[day] }]}>
                <Text style={styles.dayName}>{DAY_NAMES_FULL[day]}</Text>
              </View>
              {grouped[day].map((event) => (
                <Card key={event.id} style={styles.eventCard}>
                  <View style={styles.eventRow}>
                    <View style={styles.eventInfo}>
                      <Text style={styles.eventTitle}>{event.title}</Text>
                      {event.time ? <Text style={styles.eventTime}>⏰ {event.time}</Text> : null}
                      {event.location ? <Text style={styles.eventMeta}>📍 {event.location}</Text> : null}
                      {event.description ? <Text style={styles.eventDesc}>{event.description}</Text> : null}
                    </View>
                    <View style={styles.eventActions}>
                      {event.isRecurring && (
                        <View style={styles.recurringBadge}>
                          <Text style={styles.recurringText}>Semanal</Text>
                        </View>
                      )}
                      <TouchableOpacity onPress={() => navigation.navigate('AgendaForm', { agendaId: event.id })}>
                        <Text style={styles.editIcon}>✏️</Text>
                      </TouchableOpacity>
                      <TouchableOpacity onPress={() => handleDelete(event)}>
                        <Text style={styles.editIcon}>🗑️</Text>
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
  daySection: { gap: spacing.sm },
  dayHeader: { borderLeftWidth: 4, paddingLeft: spacing.sm },
  dayName: { ...typography.h4, color: colors.textDark },
  eventCard: { marginLeft: spacing.sm },
  eventRow: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' },
  eventInfo: { flex: 1, gap: 2 },
  eventTitle: { ...typography.bodyBold, color: colors.textDark },
  eventTime: { ...typography.small, color: colors.primary },
  eventMeta: { ...typography.small, color: colors.textSecondary },
  eventDesc: { ...typography.small, color: colors.textMuted, marginTop: 2 },
  eventActions: { alignItems: 'flex-end', gap: spacing.sm },
  recurringBadge: { backgroundColor: colors.success + '20', borderRadius: radius.full, paddingHorizontal: 8, paddingVertical: 2 },
  recurringText: { ...typography.caption, color: colors.success, fontWeight: '600' },
  editIcon: { fontSize: 16 },
});
