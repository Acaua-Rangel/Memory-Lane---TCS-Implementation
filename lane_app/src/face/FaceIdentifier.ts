// Identifica uma pessoa a partir do embedding 128-d via busca cosine no SQLite.

import type { FaceEmbeddingRepository } from '../database/repositories/FaceEmbeddingRepository';
import type { PersonRepository } from '../database/repositories/PersonRepository';
import type { PersonMatch, IdentifiedPerson } from '../types';
import { INFERENCE } from '../config/constants';

export class FaceIdentifier {
  private readonly threshold: number;

  constructor(
    private faceEmbeddings: FaceEmbeddingRepository,
    private persons: PersonRepository,
    threshold: number = INFERENCE.FACE_SIMILARITY_THRESHOLD,
  ) {
    this.threshold = threshold;
  }

  async identify(embedding: Float32Array): Promise<IdentifiedPerson | null> {
    const matches = await this.faceEmbeddings.findNearest(embedding, this.threshold, 1);
    if (matches.length === 0) return null;

    const best = matches[0];
    const person = await this.persons.findById(best.personId);
    if (!person) return null;

    return {
      person,
      similarity: best.similarity,
    };
  }

  async registerFace(personId: string, embedding: Float32Array): Promise<void> {
    await this.faceEmbeddings.insert(personId, embedding);
    console.log(`[FaceIdentifier] Rosto registrado para pessoa ${personId}`);
  }

  async findTopMatches(embedding: Float32Array, topK = 3): Promise<PersonMatch[]> {
    const matches = await this.faceEmbeddings.findNearest(embedding, 0.5, topK);

    const results: PersonMatch[] = [];
    for (const m of matches) {
      const person = await this.persons.findById(m.personId);
      if (person) results.push({ personId: m.personId, similarity: m.similarity, person });
    }

    return results;
  }

  async isKnownFace(embedding: Float32Array): Promise<boolean> {
    const matches = await this.faceEmbeddings.findNearest(embedding, this.threshold, 1);
    return matches.length > 0;
  }
}
