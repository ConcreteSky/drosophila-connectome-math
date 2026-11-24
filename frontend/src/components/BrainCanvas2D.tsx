import React, { useEffect, useMemo, useRef, useState } from "react";
import { LandmarkNeuron, LesionState } from "../api/client";

const REGION_COLORS: Record<string, string> = {
  Sensory: "#60a5fa",
  "Antennal Lobe": "#2dd4bf",
  "Mushroom Body": "#a78bfa",
  "Central Complex": "#f59e0b",
  "Lateral Horn": "#64748b",
  "Motor/Output": "#34d399",
};

interface BrainCanvas2DProps {
  landmarks: LandmarkNeuron[];
  regionNames: string[];
  spikeEvents: [number, number][] | null;
  regionActivity: Record<string, number[]> | null;
  lesions: LesionState;
  pulse: "reward" | "punish" | null;
}

export const BrainCanvas2D: React.FC<BrainCanvas2DProps> = ({
  landmarks,
  regionNames,
  spikeEvents,
  regionActivity,
  lesions,
  pulse,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [tMs, setTMs] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);

  // Pre-index spike events into 60 ms buckets for O(1) 60 FPS lookup
  const spikesByMs = useMemo(() => {
    const buckets: number[][] = Array.from({ length: 60 }, () => []);
    if (spikeEvents) {
      for (const [lmIdx, t] of spikeEvents) {
        if (t >= 0 && t < 60) buckets[t].push(lmIdx);
      }
    }
    return buckets;
  }, [spikeEvents]);

  // Reset and auto-play whenever a new trial arrives
  useEffect(() => {
    if (spikeEvents && spikeEvents.length > 0) {
      setTMs(0);
      setPlaying(true);
    }
  }, [spikeEvents]);

  // 60 FPS requestAnimationFrame playback loop
  useEffect(() => {
    if (!playing) return;
    let rafId = 0;
    let lastTs = performance.now();
    const stepMs = 35; // ~17ms/frame -> ~2s smooth playback for 60ms trial

    const tick = (now: number) => {
      if (now - lastTs >= stepMs) {
        lastTs = now;
        setTMs((prev) => {
          if (prev >= 59) {
            setPlaying(false);
            return 59;
          }
          return prev + 1;
        });
      }
      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [playing]);

  // Render Canvas frame
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.clientWidth || 640;
    const h = canvas.clientHeight || 400;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const padX = 28;
    const padY = 24;
    const drawW = w - padX * 2;
    const drawH = h - padY * 2;

    const activeSet = new Set<number>(spikesByMs[tMs] || []);
    const prevSet = new Set<number>(tMs > 0 ? spikesByMs[tMs - 1] || [] : []);

    const isLesionedRegion = (r: string) =>
      (r === "Mushroom Body" && lesions.mushroom_body) ||
      (r === "Central Complex" && lesions.central_complex);

    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      if (activeSet.has(i)) continue;
      const px = padX + p.x * drawW;
      const py = padY + p.y * drawH;
      const lesioned = isLesionedRegion(p.region);

      ctx.globalAlpha = lesioned ? 0.08 : prevSet.has(i) ? 0.5 : 0.24;
      ctx.fillStyle = lesioned ? "#ef4444" : REGION_COLORS[p.region] || "#64748b";
      ctx.beginPath();
      ctx.arc(px, py, prevSet.has(i) ? 2.2 : 1.3, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1.0;
    ctx.fillStyle = "#38bdf8";
    for (const idx of activeSet) {
      const p = landmarks[idx];
      if (!p) continue;
      const px = padX + p.x * drawW;
      const py = padY + p.y * drawH;
      ctx.beginPath();
      ctx.arc(px, py, 3.0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Dopamine reward/punish pulse centered on Mushroom Body
    if (pulse) {
      const mbX = padX + 0.48 * drawW;
      const mbY = padY + 0.34 * drawH;
      ctx.strokeStyle = pulse === "reward" ? "#22c55e" : "#ef4444";
      ctx.fillStyle = pulse === "reward" ? "rgba(34, 197, 94, 0.14)" : "rgba(239, 68, 68, 0.14)";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(mbX, mbY, 54, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();
  }, [landmarks, spikesByMs, tMs, lesions, pulse]);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">2D Dorsal Fly Brain · 3,000 Landmarks</h2>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
          t = {tMs} ms / 59 ms
        </span>
      </div>

      <div className="canvas-wrap">
        {/* Anatomical dorsal outline SVG under Canvas */}
        <svg viewBox="0 0 640 400" preserveAspectRatio="none" aria-hidden="true">
          <ellipse cx="320" cy="200" rx="220" ry="145" fill="none" stroke="#222832" strokeWidth="1.5" />
          <ellipse cx="105" cy="200" rx="52" ry="95" fill="none" stroke="#1e242e" strokeDasharray="4 4" />
          <ellipse cx="535" cy="200" rx="52" ry="95" fill="none" stroke="#1e242e" strokeDasharray="4 4" />
          <circle
            cx="307"
            cy="136"
            r="46"
            fill={lesions.mushroom_body ? "rgba(239,68,68,0.08)" : "none"}
            stroke={lesions.mushroom_body ? "#ef4444" : "#28303d"}
            strokeDasharray={lesions.mushroom_body ? "3 3" : undefined}
          />
          <circle
            cx="320"
            cy="216"
            r="34"
            fill={lesions.central_complex ? "rgba(239,68,68,0.08)" : "none"}
            stroke={lesions.central_complex ? "#ef4444" : "#28303d"}
            strokeDasharray={lesions.central_complex ? "3 3" : undefined}
          />
        </svg>

        <canvas ref={canvasRef} />

        {pulse && (
          <div className={`pulse-banner ${pulse}`}>
            {pulse === "reward" ? "DOPAMINE +1 REWARD" : "PUNISHMENT -1"}
          </div>
        )}
      </div>

      <div className="timeline-row">
        <button onClick={() => setPlaying((p) => !p)} disabled={!spikeEvents}>
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={59}
          value={tMs}
          aria-label="Simulation timeline ms"
          onChange={(e) => {
            setPlaying(false);
            setTMs(Number(e.target.value));
          }}
        />
      </div>

      <div className="legend-row">
        {regionNames.map((r) => (
          <span key={r} className="legend-item">
            <span className="dot" style={{ background: REGION_COLORS[r] || "#64748b" }} />
            <span>
              {r}
              {regionActivity?.[r] ? ` (${Math.round(regionActivity[r][tMs] || 0)})` : ""}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
};
