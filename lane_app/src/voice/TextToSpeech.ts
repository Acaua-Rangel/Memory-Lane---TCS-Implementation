import * as Speech from 'expo-speech';
import type { TTSOptions, TTSLanguage } from '../types';
import { VOICE } from '../config/constants';

export class TextToSpeech {
  private language: TTSLanguage;
  private rate: number;
  private pitch: number;
  private isSpeaking = false;

  constructor(language: TTSLanguage = VOICE.DEFAULT_LANGUAGE) {
    this.language = language;
    this.rate = VOICE.TTS_RATE;
    this.pitch = VOICE.TTS_PITCH;
  }

  async speak(text: string, options?: TTSOptions): Promise<void> {
    // Para qualquer fala anterior antes de começar nova
    if (this.isSpeaking) {
      await this.stop();
    }

    this.isSpeaking = true;

    return new Promise((resolve) => {
      Speech.speak(text, {
        language: options?.language ?? this.language,
        rate: options?.rate ?? this.rate,
        pitch: options?.pitch ?? this.pitch,
        volume: options?.volume ?? 1.0,
        onStart: () => {
          this.isSpeaking = true;
        },
        onDone: () => {
          this.isSpeaking = false;
          resolve();
        },
        onStopped: () => {
          this.isSpeaking = false;
          resolve();
        },
        onError: (error) => {
          console.error('[TTS] Erro:', error);
          this.isSpeaking = false;
          resolve(); // Não rejeita — melhor silêncio do que crash
        },
      });
    });
  }

  async stop(): Promise<void> {
    Speech.stop();
    this.isSpeaking = false;
  }

  getIsSpeaking(): boolean {
    return this.isSpeaking;
  }

  setLanguage(lang: TTSLanguage): void {
    this.language = lang;
  }

  // Rate mais lento para pacientes com Alzheimer
  setElderlyMode(enabled: boolean): void {
    this.rate = enabled ? 0.75 : VOICE.TTS_RATE;
    this.pitch = enabled ? 0.95 : VOICE.TTS_PITCH;
  }

  async getAvailableVoices(): Promise<Speech.Voice[]> {
    return Speech.getAvailableVoicesAsync();
  }
}
