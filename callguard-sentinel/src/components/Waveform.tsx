import React, { useEffect, useMemo, useRef } from "react";

interface WaveformProps {
  values: number[]; // 0..1
  barCount?: number;
  height?: number;
  className?: string;
}

export const Waveform: React.FC<WaveformProps> = ({ values, barCount = 32, height = 64, className }) => {
  const normalized = useMemo(() => {
    if (!values || values.length === 0) return new Array(barCount).fill(0.05);
    const take = values.slice(-barCount);
    const pad = Array.from({ length: Math.max(0, barCount - take.length) }, () => 0.05);
    const arr = [...pad, ...take];
    return arr.map((v) => Math.max(0.03, Math.min(1, v)));
  }, [values, barCount]);

  return (
    <div className={className} style={{ height }}>
      <div className="flex items-end gap-[3px] w-full h-full">
        {normalized.map((v, idx) => (
          <div
            key={idx}
            className="flex-1 bg-primary/70 rounded-sm"
            style={{ height: `${v * 100}%` }}
          />
        ))}
      </div>
    </div>
  );
};


