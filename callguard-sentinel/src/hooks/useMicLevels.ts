import { useCallback, useEffect, useRef, useState } from "react";

// Captures microphone levels in real time using Web Audio API
export function useMicLevels(enabled: boolean = true, sampleIntervalMs: number = 100) {
  const [levels, setLevels] = useState<number[]>([]);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);

  const stop = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (analyserRef.current) analyserRef.current.disconnect();
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    try {
      // Request mic
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      mediaStreamRef.current = stream;

      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256; // low-latency
      analyser.smoothingTimeConstant = 0.6;
      source.connect(analyser);
      analyserRef.current = analyser;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const tick = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteTimeDomainData(dataArray);
        // Compute RMS as level 0..1
        let sumSq = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const v = (dataArray[i] - 128) / 128; // -1..1
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / dataArray.length); // 0..1
        const level = Math.min(1, Math.max(0, rms * 2));
        setLevels((prev) => [...prev.slice(-63), level]);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      // ignore permission errors; no levels
    }
  }, []);

  useEffect(() => {
    if (enabled) start();
    return () => stop();
  }, [enabled, start, stop]);

  return { levels, start, stop };
}


