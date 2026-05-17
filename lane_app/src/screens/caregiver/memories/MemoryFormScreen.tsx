import React, { useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, StatusBar, ScrollView,
  TouchableOpacity, Alert,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { PeopleStackParamList } from '../../../navigation/types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Input } from '../../../components/ui/Input';
import { Button } from '../../../components/ui/Button';
import { colors, spacing, radius, typography } from '../../../theme';

type Props = NativeStackScreenProps<PeopleStackParamList, 'MemoryForm'>;

const QUICK_TAGS = ['Viagem', 'Família', 'Trabalho', 'Infância', 'Casamento', 'Aniversário', 'Conquista', 'Amizade'];

export function MemoryFormScreen({ route, navigation }: Props) {
  const { personId } = route.params;
  const { db } = useDatabase();

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [date, setDate] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag],
    );
  };

  const validate = () => {
    const e: Record<string, string> = {};
    if (!title.trim()) e.title = 'Título é obrigatório';
    if (!description.trim()) e.description = 'Descrição é obrigatória';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSave = async () => {
    if (!validate() || !db) return;
    setSaving(true);
    try {
      await db.memories.insert({
        personId,
        title: title.trim(),
        description: description.trim(),
        date: date.trim() || undefined,
        tags: selectedTags.length > 0 ? selectedTags : undefined,
      });
      navigation.goBack();
    } catch {
      Alert.alert('Erro', 'Não foi possível salvar. Tente novamente.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.back}>← Voltar</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Nova Memória</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView contentContainerStyle={styles.form} keyboardShouldPersistTaps="handled">
        <Input
          label="Título da memória *"
          value={title}
          onChangeText={setTitle}
          placeholder="Ex: Viagem para o litoral em 1985"
          error={errors.title}
          autoCapitalize="sentences"
        />

        <Input
          label="Descrição *"
          value={description}
          onChangeText={setDescription}
          placeholder="Descreva este momento com detalhes — quem estava presente, o que aconteceu, como foi especial..."
          multiline
          numberOfLines={6}
          error={errors.description}
          hint="Seja detalhado para que a Lane possa recontar esta história"
        />

        <Input
          label="Data (opcional)"
          value={date}
          onChangeText={setDate}
          placeholder="Ex: Verão de 1985 / Janeiro de 2010"
          hint="Pode ser aproximada, como 'anos 80' ou 'quando tinha 30 anos'"
        />

        <View>
          <Text style={styles.tagsLabel}>Categorias (opcional)</Text>
          <View style={styles.tagsRow}>
            {QUICK_TAGS.map((tag) => (
              <TouchableOpacity
                key={tag}
                style={[styles.tag, selectedTags.includes(tag) && styles.tagActive]}
                onPress={() => toggleTag(tag)}
              >
                <Text style={[styles.tagText, selectedTags.includes(tag) && styles.tagTextActive]}>
                  {tag}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <Button label={saving ? 'Salvando...' : 'Salvar Memória'} onPress={handleSave} loading={saving} fullWidth size="lg" style={{ marginTop: spacing.md }} />
      </ScrollView>
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
  back: { ...typography.body, color: colors.primary },
  title: { ...typography.h3, color: colors.textDark },
  form: { padding: spacing.lg, gap: spacing.lg },
  tagsLabel: { ...typography.small, color: colors.textDark, fontWeight: '600', marginBottom: spacing.sm },
  tagsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  tag: {
    borderWidth: 1.5, borderColor: colors.borderLight, borderRadius: radius.full,
    paddingHorizontal: spacing.md, paddingVertical: 6, backgroundColor: colors.surfaceLight,
  },
  tagActive: { borderColor: colors.primary, backgroundColor: colors.primary + '15' },
  tagText: { ...typography.small, color: colors.textSecondary },
  tagTextActive: { color: colors.primary, fontWeight: '600' },
});
