import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet, SafeAreaView,
  StatusBar, Alert, ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { PeopleStackParamList } from '../../../navigation/types';
import type { PersonRecord } from '../../../types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Card } from '../../../components/ui/Card';
import { EmptyState } from '../../../components/ui/EmptyState';
import { colors, spacing, radius, typography } from '../../../theme';

type Props = NativeStackScreenProps<PeopleStackParamList, 'PeopleList'>;

export function PeopleListScreen({ navigation }: Props) {
  const { db, isReady } = useDatabase();
  const [people, setPeople] = useState<PersonRecord[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!db) return;
    setLoading(true);
    const list = await db.persons.findAll();
    setPeople(list);
    setLoading(false);
  }, [db]);

  useEffect(() => {
    if (isReady) void load();
  }, [isReady, load]);

  useEffect(() => {
    return navigation.addListener('focus', () => { if (isReady) void load(); });
  }, [navigation, isReady, load]);

  const handleDelete = (person: PersonRecord) => {
    Alert.alert(
      'Remover pessoa',
      `Deseja remover ${person.name}? Suas memórias também serão removidas.`,
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Remover', style: 'destructive',
          onPress: async () => {
            await db!.persons.delete(person.id);
            await load();
          },
        },
      ],
    );
  };

  if (!isReady || loading) {
    return (
      <SafeAreaView style={styles.safe}>
        <ActivityIndicator style={{ flex: 1 }} color={colors.primary} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />

      <View style={styles.header}>
        <Text style={styles.title}>👥 Pessoas</Text>
        <TouchableOpacity
          style={styles.addBtn}
          onPress={() => navigation.navigate('PersonForm', {})}
        >
          <Text style={styles.addBtnText}>+ Adicionar</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        data={people}
        keyExtractor={(item) => item.id}
        contentContainerStyle={people.length === 0 ? { flex: 1 } : styles.list}
        ListEmptyComponent={
          <EmptyState
            icon="👤"
            title="Nenhuma pessoa cadastrada"
            description="Cadastre familiares, amigos e cuidadores para que a Lane possa reconhecê-los."
            actionLabel="+ Adicionar pessoa"
            onAction={() => navigation.navigate('PersonForm', {})}
          />
        }
        renderItem={({ item }) => (
          <Card style={styles.card}>
            <TouchableOpacity
              onPress={() => navigation.navigate('PersonForm', { personId: item.id })}
              activeOpacity={0.8}
            >
              <View style={styles.personRow}>
                <View style={styles.avatar}>
                  <Text style={styles.avatarText}>{item.name[0]?.toUpperCase()}</Text>
                </View>
                <View style={styles.personInfo}>
                  <Text style={styles.personName}>{item.name}</Text>
                  <Text style={styles.personRel}>{item.relationship}</Text>
                  {item.isCaregiver && (
                    <View style={styles.caregiverTag}>
                      <Text style={styles.caregiverTagText}>Cuidador</Text>
                    </View>
                  )}
                </View>
                <View style={styles.actions}>
                  <TouchableOpacity
                    style={styles.memoryBtn}
                    onPress={() => navigation.navigate('MemoriesList', {
                      personId: item.id,
                      personName: item.name,
                    })}
                  >
                    <Text style={styles.memoryBtnText}>Memórias</Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => handleDelete(item)}>
                    <Text style={styles.deleteIcon}>🗑️</Text>
                  </TouchableOpacity>
                </View>
              </View>
              {item.bio ? (
                <Text style={styles.bio} numberOfLines={2}>{item.bio}</Text>
              ) : null}
            </TouchableOpacity>
          </Card>
        )}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgLight },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.lg,
    backgroundColor: colors.surfaceLight,
    borderBottomWidth: 1,
    borderBottomColor: colors.borderLight,
  },
  title: { ...typography.h2, color: colors.textDark },
  addBtn: {
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.full,
  },
  addBtnText: { ...typography.small, color: '#fff', fontWeight: '600' },

  list: { padding: spacing.md, gap: spacing.sm },

  card: { marginHorizontal: spacing.md, marginBottom: spacing.sm },
  personRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  avatar: {
    width: 48, height: 48, borderRadius: 24,
    backgroundColor: colors.primary,
    alignItems: 'center', justifyContent: 'center',
  },
  avatarText: { ...typography.h3, color: '#fff' },
  personInfo: { flex: 1, gap: 2 },
  personName: { ...typography.bodyBold, color: colors.textDark },
  personRel: { ...typography.small, color: colors.textSecondary },
  caregiverTag: {
    alignSelf: 'flex-start',
    backgroundColor: colors.success + '20',
    borderRadius: radius.full,
    paddingHorizontal: 8, paddingVertical: 2, marginTop: 2,
  },
  caregiverTagText: { ...typography.caption, color: colors.success, fontWeight: '600' },
  actions: { alignItems: 'flex-end', gap: spacing.sm },
  memoryBtn: {
    backgroundColor: colors.primary + '15',
    paddingHorizontal: 10, paddingVertical: 4,
    borderRadius: radius.full,
  },
  memoryBtnText: { ...typography.caption, color: colors.primary, fontWeight: '600' },
  deleteIcon: { fontSize: 18 },
  bio: { ...typography.small, color: colors.textSecondary, marginTop: spacing.sm, lineHeight: 18 },
});
