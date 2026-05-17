// STT via expo-speech-recognition — reconhecimento offline no Android/iOS.
// Substitui o deprecado @react-native-voice/voice.

import { ExpoSpeechRecognitionModule } from 'expo-speech-recognition';
import type { ExpoSpeechRecognitionResultEvent, ExpoSpeechRecognitionErrorEvent } from 'expo-speech-recognition';
import type { STTResult } from '../types';
import { VOICE } from '../config/constants';

export type STTCallback = (result: STTResult) => void;
export type STTErrorCallback = (error: string) => void;

export class SpeechToText {
  private onResult: STTCallback | null = null;
  private onError: STTErrorCallback | null = null;
  private isListening = false;
  private language: string;
  private subscriptions: { remove: () => void }[] = [];

  constructor(language: string = VOICE.DEFAULT_LANGUAGE) {
    this.language = language;
  }

  private setupListeners(onResult: STTCallback, onError?: STTErrorCallback): void {
    this.clearListeners();

    const resultSub = ExpoSpeechRecognitionModule.addListener('result', (event: ExpoSpeechRecognitionResultEvent) => {
      const transcript = event.results?.[0]?.transcript ?? '';
      const isFinal = event.isFinal ?? false;

      onResult({ transcript, confidence: 1.0, isFinal });

      if (isFinal) {
        this.isListening = false;
      }
    });

    const errorSub = ExpoSpeechRecognitionModule.addListener('error', (event: ExpoSpeechRecognitionErrorEvent) => {
      const error = event.error ?? 'STT error';
      console.error('[STT] Erro:', error);
      this.isListening = false;
      onError?.(error);
    });

    const endSub = ExpoSpeechRecognitionModule.addListener('end', () => {
      this.isListening = false;
    });

    this.subscriptions = [resultSub, errorSub, endSub];
  }

  private clearListeners(): void {
    this.subscriptions.forEach((s) => s.remove());
    this.subscriptions = [];
  }

  async requestPermission(): Promise<boolean> {
    const result = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
    return result.granted;
  }

  async startListening(
    onResult: STTCallback,
    onError?: STTErrorCallback,
  ): Promise<void> {
    if (this.isListening) return;

    const granted = await this.requestPermission();
    if (!granted) {
      onError?.('Permissão de microfone negada');
      return;
    }

    this.onResult = onResult;
    this.onError = onError ?? null;
    this.isListening = true;

    this.setupListeners(onResult, onError);

    ExpoSpeechRecognitionModule.start({
      lang: this.language,
      interimResults: true,
      maxAlternatives: 1,
      continuous: false,
      requiresOnDeviceRecognition: false, // true = offline puro; false = fallback para cloud
      addsPunctuation: false,
    });
  }

  async stopListening(): Promise<void> {
    if (!this.isListening) return;
    ExpoSpeechRecognitionModule.stop();
    this.isListening = false;
  }

  async cancelListening(): Promise<void> {
    ExpoSpeechRecognitionModule.abort();
    this.isListening = false;
  }

  getIsListening(): boolean {
    return this.isListening;
  }

  setLanguage(lang: string): void {
    this.language = lang;
  }

  async listenOnce(timeoutMs = VOICE.MAX_RECORD_DURATION_MS): Promise<string> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        void this.stopListening();
        reject(new Error('STT timeout'));
      }, timeoutMs);

      void this.startListening(
        (result) => {
          if (result.isFinal) {
            clearTimeout(timer);
            resolve(result.transcript);
          }
        },
        (error) => {
          clearTimeout(timer);
          reject(new Error(error));
        },
      );
    });
  }

  destroy(): void {
    ExpoSpeechRecognitionModule.abort();
    this.clearListeners();
    this.isListening = false;
  }
}
