import React, { useEffect, useMemo, useRef, useState } from "react";
import { LandmarkNeuron, LesionState } from "../api/client";

const REGION_COLORS: Record<string, string> = {
  Sensory: "#38bdf8",
  "Antennal Lobe": "#2dd4bf",
  "Mushroom Body": "#a78bfa",
  "Central Complex": "#f59e0b",
  "Lateral Horn": "#64748b",
  "Motor/Output": "#10b981",
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
    const h = canvas.clientHeight || 410;
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

    const activeNow = spikesByMs[tMs] || [];
    const activeSet = new Set<number>(activeNow);
    const trail1 = new Set<number>(tMs >= 1 ? spikesByMs[tMs - 1] || [] : []);
    const trail2 = new Set<number>(tMs >= 2 ? spikesByMs[tMs - 2] || [] : []);

    const isLesionedRegion = (r: string) =>
      (r === "Mushroom Body" && lesions.mushroom_body) ||
      (r === "Central Complex" && lesions.central_complex);

    // 1. Resting & decaying trail neurons
    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      if (activeSet.has(i)) continue;
      const px = padX + p.x * drawW;
      const py = padY + p.y * drawH;
      const lesioned = isLesionedRegion(p.region);

      if (lesioned) {
        ctx.globalAlpha = 0.06;
        ctx.fillStyle = "#64748b";
        ctx.beginPath();
        ctx.arc(px, py, 1.2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }

      if (trail1.has(i)) {
        ctx.globalAlpha = 0.65;
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(px, py, 2.2, 0, Math.PI * 2);
        ctx.fill();
      } else if (trail2.has(i)) {
        ctx.globalAlpha = 0.35;
        ctx.fillStyle = "#0284c7";
        ctx.beginPath();
        ctx.arc(px, py, 1.8, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.globalAlpha = 0.22;
        ctx.fillStyle = REGION_COLORS[p.region] || "#64748b";
        ctx.beginPath();
        ctx.arc(px, py, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 2. Synaptic pathway flashes between consecutive firing neurons (t-1 -> t)
    const prevArr = tMs >= 1 ? spikesByMs[tMs - 1] || [] : [];
    if (activeNow.length > 0 && prevArr.length > 0) {
      ctx.strokeStyle = "rgba(56, 189, 248, 0.28)";
      ctx.lineWidth = 0.9;
      const maxLinks = Math.min(36, activeNow.length, prevArr.length);
      for (let k = 0; k < maxLinks; k++) {
        const src = landmarks[prevArr[(k * 7) % prevArr.length]];
        const dst = landmarks[activeNow[(k * 5) % activeNow.length]];
        if (!src || !dst) continue;
        ctx.beginPath();
        ctx.moveTo(padX + src.x * drawW, padY + src.y * drawH);
        ctx.lineTo(padX + dst.x * drawW, padY + dst.y * drawH);
        ctx.stroke();
      }
    }

    // 3. Glowing active spikes
    for (const idx of activeNow) {
      const p = landmarks[idx];
      if (!p) continue;
      const px = padX + p.x * drawW;
      const py = padY + p.y * drawH;

      ctx.globalAlpha = 0.28;
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(px, py, 5.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = 1.0;
      ctx.fillStyle = "#e0f2fe";
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // 4. High-impact Reward (gold/emerald ripple) or Punishment (crimson shockwave) on Memory Centers
    if (pulse) {
      const mbX = padX + 0.48 * drawW;
      const mbY = padY + 0.34 * drawH;
      if (pulse === "reward") {
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(mbX, mbY, 64, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = "#10b981";
        ctx.fillStyle = "rgba(16, 185, 129, 0.14)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(mbX, mbY, 44, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.strokeStyle = "#f43f5e";
        ctx.fillStyle = "rgba(244, 63, 94, 0.16)";
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(mbX, mbY, 56, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    ctx.restore();
  }, [landmarks, spikesByMs, tMs, lesions, pulse]);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">2D Anatomical Drosophila Brain · Real-Time Spike Flow</h2>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
          {tMs} / 59 ms
        </span>
      </div>

      <div className="canvas-wrap">
        <svg viewBox="0 0 640 410" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <pattern id="lesion-hatch" width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
              <line x1="0" y1="0" x2="0" y2="8" stroke="#f43f5e" strokeWidth="1.5" strokeOpacity="0.45" />
            </pattern>
            <marker id="flow-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#475569" />
            </marker>
          </defs>

          {/* Left & Right Optic Lobes (Lamina / Medulla / Lobula wings) */}
          <path
            d="M 145,115 C 65,110 42,165 44,205 C 46,255 72,300 145,290 C 165,255 165,150 145,115 Z"
            fill="rgba(56, 189, 248, 0.03)"
            stroke="#1e293b"
            strokeWidth="1.4"
          />
          <path
            d="M 495,115 C 575,110 598,165 596,205 C 594,255 568,300 495,290 C 475,255 475,150 495,115 Z"
            fill="rgba(16, 185, 129, 0.03)"
            stroke="#1e293b"
            strokeWidth="1.4"
          />

          {/* Central Brain Protocerebrum Silhouette */}
          <path
            d="M 145,115 C 185,55 455,55 495,115 C 515,175 515,245 495,290 C 435,345 205,345 145,290 C 125,245 125,175 145,115 Z"
            fill="rgba(15, 23, 42, 0.45)"
            stroke="#243247"
            strokeWidth="1.5"
          />

          {/* Bilateral Antennal Lobes (Paired Anterior Glomeruli) */}
          <ellipse cx="220" cy="256" rx="28" ry="22" fill="rgba(45, 212, 191, 0.05)" stroke="#234e52" strokeWidth="1.2" />
          <ellipse cx="420" cy="256" rx="28" ry="22" fill="rgba(45, 212, 191, 0.05)" stroke="#234e52" strokeWidth="1.2" />

          {/* Bilateral Mushroom Bodies (Iconic L-shaped Calyx + Alpha/Beta Lobes) */}
          <path
            d="M 245,95 C 218,95 210,122 228,142 L 245,165 L 288,165 C 296,165 296,150 284,150 L 258,150 L 258,112 C 258,100 253,95 245,95 Z"
            fill={lesions.mushroom_body ? "url(#lesion-hatch)" : "rgba(167, 139, 250, 0.08)"}
            stroke={lesions.mushroom_body ? "#f43f5e" : "#6d28d9"}
            strokeWidth="1.4"
          />
          <path
            d="M 395,95 C 422,95 430,122 412,142 L 395,165 L 352,165 C 344,165 344,150 356,150 L 382,150 L 382,112 C 382,100 387,95 395,95 Z"
            fill={lesions.mushroom_body ? "url(#lesion-hatch)" : "rgba(167, 139, 250, 0.08)"}
            stroke={lesions.mushroom_body ? "#f43f5e" : "#6d28d9"}
            strokeWidth="1.4"
          />

          {/* Central Complex (Fan-Shaped Body Arch + Ellipsoid Body Ring) */}
          <path
            d="M 278,195 C 292,172 348,172 362,195 C 350,210 290,210 278,195 Z"
            fill={lesions.central_complex ? "url(#lesion-hatch)" : "rgba(245, 158, 11, 0.08)"}
            stroke={lesions.central_complex ? "#f43f5e" : "#b45309"}
            strokeWidth="1.3"
          />
          <ellipse
            cx="320"
            cy="226"
            rx="22"
            ry="14"
            fill={lesions.central_complex ? "url(#lesion-hatch)" : "rgba(245, 158, 11, 0.08)"}
            stroke={lesions.central_complex ? "#f43f5e" : "#b45309"}
            strokeWidth="1.3"
          />

          {/* Flow Arrows */}
          <line x1="125" y1="195" x2="225" y2="142" stroke="#334155" strokeWidth="1.2" markerEnd="url(#flow-arrow)" />
          <line x1="398" y1="142" x2="505" y2="195" stroke="#334155" strokeWidth="1.2" markerEnd="url(#flow-arrow)" />

          {/* Plain-English Anatomical Labels */}
          <text x="94" y="88" fill="#7dd3fc" fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Signal Entry (A &amp; B)
          </text>
          <text x="320" y="68" fill={lesions.mushroom_body ? "#fda4af" : "#c4b5fd"} fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Memory &amp; Learning Center{lesions.mushroom_body ? " (OFF)" : ""}
          </text>
          <text x="320" y="264" fill={lesions.central_complex ? "#fda4af" : "#fcd34d"} fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Signal Routing Hub{lesions.central_complex ? " (OFF)" : ""}
          </text>
          <text x="546" y="88" fill="#6ee7b7" fontSize="11" fontFamily="sans-serif" textAnchor="middle">
            Answer Decision
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
          <span><span className="dot" style={{ background: "#10b981" }} /> Reward (+1)</span>
          <span><span className="dot" style={{ background: "#f43f5e" }} /> Punish (-1) / Off</span>
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
        <span style={{ fontSize: 11, color: "var(--text-dim)" }}>60 milliseconds of brain activity</span>
      </div>
    </div>
  );
};
