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
  spikeEvents: [number, number][] | null;
  lesions: LesionState;
  pulse: "reward" | "punish" | null;
}

export const BrainCanvas2D: React.FC<BrainCanvas2DProps> = ({
  landmarks,
  spikeEvents,
  lesions,
  pulse,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [tMs, setTMs] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);

  const spikesByMs = useMemo(() => {
    const buckets: number[][] = Array.from({ length: 60 }, () => []);
    if (spikeEvents) {
      for (const [lmIdx, t] of spikeEvents) {
        if (t >= 0 && t < 60) buckets[t].push(lmIdx);
      }
    }
    return buckets;
  }, [spikeEvents]);

  useEffect(() => {
    if (spikeEvents && spikeEvents.length > 0) {
      setTMs(0);
      setPlaying(true);
    }
  }, [spikeEvents]);

  useEffect(() => {
    if (!playing) return;
    let rafId = 0;
    let lastTs = performance.now();
    const tick = (now: number) => {
      if (now - lastTs >= 35) {
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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.clientWidth || 640;
    const h = canvas.clientHeight || 390;
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

      ctx.globalAlpha = lesioned ? 0.07 : prevSet.has(i) ? 0.5 : 0.24;
      ctx.fillStyle = lesioned ? "#ef4444" : REGION_COLORS[p.region] || "#64748b";
      ctx.beginPath();
      ctx.arc(px, py, prevSet.has(i) ? 2.1 : 1.3, 0, Math.PI * 2);
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
        <h2 className="panel-title">2D Fly Brain Activity (Live Spike Flow)</h2>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
          {tMs} / 59 ms
        </span>
      </div>

      <div className="canvas-wrap">
        <svg viewBox="0 0 640 390" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563" />
            </marker>
          </defs>

          <ellipse cx="320" cy="195" rx="225" ry="140" fill="none" stroke="#222832" strokeWidth="1.5" />
          <circle
            cx="307"
            cy="134"
            r="46"
            fill={lesions.mushroom_body ? "rgba(239,68,68,0.08)" : "none"}
            stroke={lesions.mushroom_body ? "#ef4444" : "#2f3744"}
            strokeDasharray={lesions.mushroom_body ? "3 3" : undefined}
          />
          <circle
            cx="320"
            cy="212"
            r="34"
            fill={lesions.central_complex ? "rgba(239,68,68,0.08)" : "none"}
            stroke={lesions.central_complex ? "#ef4444" : "#2f3744"}
            strokeDasharray={lesions.central_complex ? "3 3" : undefined}
          />

          {/* Directional signal flow arrows */}
          <line x1="150" y1="190" x2="255" y2="145" stroke="#3b4351" strokeWidth="1.2" markerEnd="url(#arrow)" />
          <line x1="358" y1="145" x2="475" y2="195" stroke="#3b4351" strokeWidth="1.2" markerEnd="url(#arrow)" />

          {/* Plain-English region labels */}
          <text x="108" y="88" fill="#8b949e" fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Inputs (A &amp; B enter)
          </text>
          <text x="307" y="78" fill={lesions.mushroom_body ? "#fca5a5" : "#c4b5fd"} fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Memory &amp; Learning Center{lesions.mushroom_body ? " (OFF)" : ""}
          </text>
          <text x="320" y="264" fill={lesions.central_complex ? "#fca5a5" : "#fcd34d"} fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Signal Routing Hub{lesions.central_complex ? " (OFF)" : ""}
          </text>
          <text x="530" y="88" fill="#6ee7b7" fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Answer Vote
          </text>
        </svg>

        <canvas ref={canvasRef} />

        {pulse && (
          <div className={`pulse-banner ${pulse}`}>
            {pulse === "reward" ? "DOPAMINE +1 REWARD" : "PUNISHMENT -1"}
          </div>
        )}

        <div className="canvas-legend">
          <span><span className="dot" style={{ background: "#38bdf8" }} /> Firing neuron</span>
          <span><span className="dot" style={{ background: "#22c55e" }} /> Reward (+1)</span>
          <span><span className="dot" style={{ background: "#ef4444" }} /> Punish (-1) / Off</span>
        </div>
      </div>

      <div className="timeline-row">
        <button onClick={() => setPlaying((p) => !p)} disabled={!spikeEvents}>
          {playing ? "Pause" : "Play"}
        </button>
        <button
          onClick={() => {
            setTMs(0);
            setPlaying(true);
          }}
          disabled={!spikeEvents}
        >
          Replay
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
        <span style={{ fontSize: 11, color: "var(--text-dim)" }}>60 ms of brain activity</span>
      </div>
    </div>
  );
};
