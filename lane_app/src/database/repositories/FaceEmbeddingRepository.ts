import type { SQLiteDatabase } from 'expo-sqlite';
import type { FaceEmbeddingRecord } from '../../types';
import { DATABASE } from '../../config/constants';
import { uuid } from '../utils';

// Cosine similarity em JS — fallback quando sqlite-vec não está disponível no mobile
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function embeddingToBlob(embedding: Float32Array): Uint8Array {
  return new Uint8Array(embedding.buffer);
}

function blobToEmbedding(blob: Uint8Array | ArrayBuffer): Float32Array {
  const buffer = blob instanceof ArrayBuffer ? blob : blob.buffer;
  return new Float32Array(buffer);
}

export class FaceEmbeddingRepository {
  constructor(private db: SQLiteDatabase) {}

  async insert(personId: string, embedding: Float32Array): Promise<FaceEmbeddingRecord> {
    const id = uuid();
    const now = Date.now();
    const blob = embeddingToBlob(embedding);
    await this.db.runAsync(
      'INSERT INTO face_embeddings (id, person_id, embedding, created_at) VALUES (?, ?, ?, ?)',
      [id, personId, blob as unknown as string, now],
    );
    return { id, personId, embedding, createdAt: now };
  }

  async findByPersonId(personId: string): Promise<FaceEmbeddingRecord[]> {
    const rows = await this.db.getAllAsync<RawEmbedding>(
      'SELECT * FROM face_embeddings WHERE person_id = ?',
      [personId],
    );
    return rows.map(toRecord);
  }

  async deleteByPersonId(personId: string): Promise<void> {
    await this.db.runAsync('DELETE FROM face_embeddings WHERE person_id = ?', [personId]);
  }

  // Busca cosine similarity em JS — carrega todos embeddings e compara
  // Para bases grandes (>200 rostos), considerar sqlite-vec via módulo nativo
  async findNearest(
    queryEmbedding: Float32Array,
    threshold: number = DATABASE.EMBEDDING_DIM / 100,
    topK: number = 1,
  ): Promise<Array<{ personId: string; similarity: number }>> {
    const rows = await this.db.getAllAsync<RawEmbedding>(
      'SELECT id, person_id, embedding FROM face_embeddings',
    );

    const results = rows
      .map((row) => {
        const storedEmbedding = blobToEmbedding(row.embedding as unknown as Uint8Array);
        const similarity = cosineSimilarity(queryEmbedding, storedEmbedding);
        return { personId: row.person_id, similarity };
      })
      .filter((r) => r.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);

    return results;
  }

  async count(): Promise<number> {
    const row = await this.db.getFirstAsync<{ count: number }>(
      'SELECT COUNT(*) as count FROM face_embeddings',
    );
    return row?.count ?? 0;
  }
}

interface RawEmbedding {
  id: string;
  person_id: string;
  embedding: unknown;
  created_at: number;
}

function toRecord(raw: RawEmbedding): FaceEmbeddingRecord {
  return {
    id: raw.id,
    personId: raw.person_id,
    embedding: blobToEmbedding(raw.embedding as Uint8Array),
    createdAt: raw.created_at,
  };
}
