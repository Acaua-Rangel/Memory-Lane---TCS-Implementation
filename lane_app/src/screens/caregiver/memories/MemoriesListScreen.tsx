import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet, SafeAreaView,
  StatusBar, Alert, ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { PeopleStackParamList } from '../../../navigation/types';
import type { MemoryRecord } from '../../../types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Card } from '../../../components/ui/Card';
import { EmptyState } from '../../../components/ui/EmptyState';
import { colors, spacing, radius, typography } from '../../../theme';

type Props = NativeStackScreenProps<PeopleStackParamList, 'MemoriesList'>;

export function MemoriesListScreen({ route, navigation }: Props) {
  const { personId, personName } = route.params;
  const { db, isReady } = useDatabase();
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!db) return;
    setLoading(true);
    const list = await db.memories.findByPersonId(personId, 50);
    setMemories(list);
    setLoading(false);
  }, [db, personId]);

  useEffect(() => { if (isReady) void load(); }, [isReady, load]);
  useEffect(() => {
    return navigation.addListener('focus', () => { if (isReady) void load(); });
  }, [navigation, isReady, load]);

  const handleDelete = (memory: MemoryRecord) => {
    Alert.alert('Remover memória', `Deseja remover "${memory.title}"?`, [
      { text: 'Cancelar', style: 'cancel' },
      {
        text: 'Remover', style: 'destructive',
        onPress: async () => { await db!.memories.delete(memory.id); await load(); },
      },
    ]);
  };

  const formatDate = (ts: number) => new Date(ts).toLocaleDateString('pt-BR');

  if (!isReady || loading) {
    return <SafeAreaView style={styles.safe}><ActivityIndicator style={{ flex: 1 }} color={colors.primary} /></SafeAreaView>;
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.back}>← Voltar</Text>
        </TouchableOpacity>
        <View style={styles.headerCenter}>
          <Text style={styles.title}>Memórias</Text>
          <Text style={styles.subtitle}>{personName}</Text>
        </View>
        <TouchableOpacity
          style={styles.addBtn}
          onPress={() => navigation.navigate('MemoryForm', { personId })}
        >
          <Text style={styles.addBtnText}>+ Nova</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        data={memories}
        keyExtractor={(item) => item.id}
        contentContainerStyle={memories.length === 0 ? { flex: 1 } : styles.list}
        ListEmptyComponent={
          <EmptyState
            icon="💭"
            title="Nenhuma memória registrada"
            description={`Adicione momentos importantes da vida de ${personName} para que a Lane possa relembrar.`}
            actionLabel="+ Adicionar memória"
            onAction={() => navigation.navigate('MemoryForm', { personId })}
          />
        }
        renderItem={({ item }) => (
          <Card style={styles.card}>
            <View style={styles.memoryHeader}>
              <Text style={styles.memoryTitle}>{item.title}</Text>
              <TouchableOpacity onPress={() => handleDelete(item)}>
                <Text style={{ fontSize: 18 }}>🗑️</Text>
              </TouchableOpacity>
            </View>
            <Text style={styles.memoryDesc}>{item.description}</Text>
            <View style={styles.metaRow}>
              {item.date ? <Text style={styles.meta}>📅 {item.date}</Text> : null}
              <Text style={styles.meta}>🕐 {formatDate(item.createdAt)}</Text>
              {item.tags?.map((tag) => (
                <View key={tag} style={styles.tag}>
                  <Text style={styles.tagText}>{tag}</Text>
                </View>
              ))}
            </View>
          </Card>
        )}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgLight },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    padding: spacing.lg, backgroundColor: colors.surfaceLight,
    borderBottomWidth: 1, borderBottomColor: colors.borderLight,
  },
  back: { ...typography.body, color: colors.primary, width: 60 },
  headerCenter: { alignItems: 'center' },
  title: { ...typography.h3, color: colors.textDark },
  subtitle: { ...typography.small, color: colors.textSecondary },
  addBtn: {
    backgroundColor: colors.primary, paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm, borderRadius: radius.full,
  },
  addBtnText: { ...typography.small, color: '#fff', fontWeight: '600' },
  list: { padding: spacing.md, gap: spacing.sm },
  card: { marginHorizontal: spacing.md, marginBottom: spacing.sm },
  memoryHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  memoryTitle: { ...typography.bodyBold, color: colors.textDark, flex: 1, marginRight: spacing.sm },
  memoryDesc: { ...typography.body, color: colors.textSecondary, marginTop: spacing.xs, lineHeight: 22 },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.xs, marginTop: spacing.sm },
  meta: { ...typography.caption, color: colors.textMuted },
  tag: {
    backgroundColor: colors.primary + '15', borderRadius: radius.full,
    paddingHorizontal: 8, paddingVertical: 2,
  },
  tagText: { ...typography.caption, color: colors.primary, fontWeight: '600' },
});
