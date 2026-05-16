// Hook de voz — combina WakeWord + STT + TTS em um único ponto de entrada.

import { useEffect, useRef, useCallback, useState } from 'react';
import { TextToSpeech } from '../voice/TextToSpeech';
import { SpeechToText } from '../voice/SpeechToText';
import { WakeWordDetector } from '../voice/WakeWordDetector';
import { useAgentStore } from '../store/agentStore';
import { usePatientStore } from '../store/patientStore';

export function useVoice(onTranscript?: (text: string) => void) {
  const ttsRef = useRef<TextToSpeech | null>(null);
  const sttRef = useRef<SpeechToText | null>(null);
  const wakeWordRef = useRef<WakeWordDetector | null>(null);

  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [wakeWordActive, setWakeWordActive] = useState(false);

  const { setAgentState } = useAgentStore();
  const { language, elderlyMode } = usePatientStore();

  // ── Setup ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    ttsRef.current = new TextToSpeech(language);
    sttRef.current = new SpeechToText(language);
    wakeWordRef.current = new WakeWordDetector();

    if (elderlyMode) {
      ttsRef.current.setElderlyMode(true);
    }

    return () => {
      ttsRef.current?.stop();
      sttRef.current?.destroy();
      wakeWordRef.current?.destroy();
    };
  }, [language, elderlyMode]);

  // ── TTS ────────────────────────────────────────────────────────────────────
  const speak = useCallback(async (text: string): Promise<void> => {
    if (!ttsRef.current || !text.trim()) return;
    setIsSpeaking(true);
    setAgentState('speaking');
    try {
      await ttsRef.current.speak(text);
    } finally {
      setIsSpeaking(false);
      setAgentState('idle');
    }
  }, [setAgentState]);

  const stopSpeaking = useCallback((): void => {
    ttsRef.current?.stop();
    setIsSpeaking(false);
  }, []);

  // ── STT ────────────────────────────────────────────────────────────────────
  const startListening = useCallback((): void => {
    if (!sttRef.current || isListening) return;
    setIsListening(true);
    setAgentState('listening');

    sttRef.current.startListening(
      (result) => {
        if (result.isFinal && result.transcript.trim()) {
          setIsListening(false);
          setAgentState('thinking');
          onTranscript?.(result.transcript);
        }
      },
      (error) => {
        console.warn('[useVoice] STT error:', error);
        setIsListening(false);
        setAgentState('idle');
      },
    );
  }, [isListening, onTranscript, setAgentState]);

  const stopListening = useCallback(async (): Promise<void> => {
    await sttRef.current?.stopListening();
    setIsListening(false);
  }, []);

  // ── Wake Word ──────────────────────────────────────────────────────────────
  const startWakeWord = useCallback((): void => {
    if (!wakeWordRef.current || wakeWordActive) return;
    setWakeWordActive(true);
    wakeWordRef.current.start(() => {
      // Wake word detectada — começa a escutar a pergunta completa
      startListening();
    });
  }, [wakeWordActive, startListening]);

  const stopWakeWord = useCallback((): void => {
    wakeWordRef.current?.stop();
    setWakeWordActive(false);
  }, []);

  return {
    // Estado
    isListening,
    isSpeaking,
    wakeWordActive,

    // Ações
    speak,
    stopSpeaking,
    startListening,
    stopListening,
    startWakeWord,
    stopWakeWord,
  };
}
