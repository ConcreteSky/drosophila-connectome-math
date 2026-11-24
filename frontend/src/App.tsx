import React, { useEffect, useState } from "react";
import {
  apiClient,
  ConnectomeSummary,
  ExperimentsResponse,
  HealthResponse,
  LandmarkNeuron,
  LesionState,
  SolveResponse,
  TaskName,
} from "./api/client";
import { BenchmarkPanel } from "./components/BenchmarkPanel";
import { BrainCanvas2D } from "./components/BrainCanvas2D";
import { LesionPanel } from "./components/LesionPanel";
import { QuizRewardPanel, TASK_LIMITS } from "./components/QuizRewardPanel";

type ActiveTab = "play" | "lesion" | "science";

export const App: React.FC = () => {
  const [tab, setTab] = useState<ActiveTab>("play");
  const [guideOpen, setGuideOpen] = useState<boolean>(true);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [landmarks, setLandmarks] = useState<LandmarkNeuron[]>([]);
  const [summary, setSummary] = useState<ConnectomeSummary | null>(null);
  const [experiments, setExperiments] = useState<ExperimentsResponse | null>(null);

  const [task, setTask] = useState<TaskName>("add_3bit");
  const [a, setA] = useState<number>(3);
  const [b, setB] = useState<number>(5);
  const [busy, setBusy] = useState<boolean>(false);

  const [solveRes, setSolveRes] = useState<SolveResponse | null>(null);
  const [lastIntactRes, setLastIntactRes] = useState<SolveResponse | null>(null);
  const [lesions, setLesions] = useState<LesionState>({
    mushroom_body: false,
    central_complex: false,
    hub_top_5pct: false,
    disabled_neurons: 0,
  });
  const [pulse, setPulse] = useState<"reward" | "punish" | null>(null);

  useEffect(() => {
    Promise.all([
      apiClient.health(),
      apiClient.layout(),
      apiClient.summary(),
      apiClient.experiments(),
    ])
      .then(([h, l, s, e]) => {
        setHealth(h);
        setLandmarks(l.landmarks);
        setSummary(s);
        setExperiments(e);
      })
      .catch((err) => console.error("Init error:", err));
  }, []);

  const triggerPulse = (kind: "reward" | "punish") => {
    setPulse(kind);
    window.setTimeout(() => setPulse(null), 900);
  };

  const handleRun = async (overrideTask?: TaskName, overrideA?: number, overrideB?: number) => {
    const t = overrideTask ?? task;
    const va = overrideA ?? a;
    const vb = overrideB ?? b;
    setBusy(true);
    try {
      const res = await apiClient.solve(t, va, vb);
      setSolveRes(res);
      if (!lesions.mushroom_body && !lesions.central_complex && !lesions.hub_top_5pct) {
        setLastIntactRes(res);
      }
    } finally {
      setBusy(false);
    }
  };

  const opSymbol = TASK_LIMITS[task].op;
  const isAnyLesioned = lesions.mushroom_body || lesions.central_complex || lesions.hub_top_5pct;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <h1>Drosophila Fly Brain · Math Trainer</h1>
          <p>
            Watch a real fruit-fly brain ({summary ? summary.total_neurons.toLocaleString() : "138,639"} neurons) learn math via Reward (+1) &amp; Punishment (-1)
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <span className="pill">
            Engine: <strong>{health?.gpu_name || "Loading..."}</strong>
          </span>
          <span className="pill">
            Synapses: <strong>{summary ? summary.total_synapses.toLocaleString() : "54,492,922"}</strong>
          </span>
        </div>
      </header>

      {/* Collapsible 3-Step Onboarding Strip */}
      <div className="onboarding-strip">
        <div className="onboarding-header" onClick={() => setGuideOpen((o) => !o)}>
          <strong style={{ fontSize: 12 }}>How This Works (3-Step Guide)</strong>
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
            {guideOpen ? "Hide ▲" : "Show ▼"}
          </span>
        </div>
        {guideOpen && (
          <div className="onboarding-steps">
            <div>
              <strong>1. Give it a problem:</strong> Type A and B (or click an example) and hit{" "}
              <em>Ask the Fly</em>.
            </div>
            <div>
              <strong>2. Watch it think:</strong> Electrical spikes travel from{" "}
              <em>Inputs (A, B)</em> → <em>Memory Center</em> → <em>Answer Neurons</em>.
            </div>
            <div>
              <strong>3. Train it:</strong> Click <em>Reward (+1)</em> when right or{" "}
              <em>Punish (-1)</em> when wrong to reshape brain connections.
            </div>
          </div>
        )}
      </div>

      {/* Live Equation Banner */}
      <div className="equation-banner" aria-live="polite">
        <div className="equation-text">
          {a} {opSymbol} {b} = {busy ? "..." : solveRes ? solveRes.predicted_answer : "?"}
        </div>
        <div>
          {busy ? (
            <span className="badge-wait">Signals traveling through brain...</span>
          ) : solveRes ? (
            solveRes.is_correct ? (
              <span className="badge-ok">Correct ✓</span>
            ) : (
              <span className="badge-wrong">Wrong ✗ (Expected {solveRes.true_answer})</span>
            )
          ) : (
            <span style={{ color: "var(--text-dim)", fontSize: 12 }}>
              Press Enter or click Ask the Fly
            </span>
          )}
        </div>
      </div>

      {/* 3-Tab Navigation */}
      <nav className="nav-tabs" aria-label="Workspace tabs">
        <button
          className={`nav-tab ${tab === "play" ? "active" : ""}`}
          onClick={() => setTab("play")}
        >
          1. Play &amp; Train
        </button>
        <button
          className={`nav-tab ${tab === "lesion" ? "active" : ""}`}
          onClick={() => setTab("lesion")}
        >
          2. Brain Damage Lab {isAnyLesioned ? "(Active)" : ""}
        </button>
        <button
          className={`nav-tab ${tab === "science" ? "active" : ""}`}
          onClick={() => setTab("science")}
        >
          3. Science &amp; Benchmarks
        </button>
      </nav>

      {tab === "play" && (
        <div className="grid-main">
          <BrainCanvas2D
            landmarks={landmarks}
            spikeEvents={solveRes?.spike_events || null}
            lesions={lesions}
            pulse={pulse}
          />
          <QuizRewardPanel
            task={task}
            a={a}
            b={b}
            busy={busy}
            solveRes={solveRes}
            onTaskChange={setTask}
            onAChange={setA}
            onBChange={setB}
            onRun={handleRun}
            onFeedbackSuccess={(fb) => triggerPulse(fb.dopamine_pulse)}
          />
        </div>
      )}

      {tab === "lesion" && (
        <div className="grid-main">
          <BrainCanvas2D
            landmarks={landmarks}
            spikeEvents={solveRes?.spike_events || null}
            lesions={lesions}
            pulse={pulse}
          />
          <LesionPanel
            lesions={lesions}
            lastIntactResult={lastIntactRes}
            currentResult={solveRes}
            onChange={setLesions}
            onRerunSame={() => handleRun()}
            onReset={() => setSolveRes(null)}
          />
        </div>
      )}

      {tab === "science" && <BenchmarkPanel summary={summary} experiments={experiments} />}
    </div>
  );
};
