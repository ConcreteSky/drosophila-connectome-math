import React, { useEffect, useState } from "react";
import {
  apiClient,
  ConnectomeSummary,
  ExperimentsResponse,
  HealthResponse,
  LandmarkNeuron,
  LesionState,
  SolveResponse,
} from "./api/client";
import { BenchmarkPanel } from "./components/BenchmarkPanel";
import { BrainCanvas2D } from "./components/BrainCanvas2D";
import { LesionPanel } from "./components/LesionPanel";
import { QuizRewardPanel } from "./components/QuizRewardPanel";

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [landmarks, setLandmarks] = useState<LandmarkNeuron[]>([]);
  const [regionNames, setRegionNames] = useState<string[]>([
    "Sensory",
    "Antennal Lobe",
    "Mushroom Body",
    "Central Complex",
    "Lateral Horn",
    "Motor/Output",
  ]);
  const [summary, setSummary] = useState<ConnectomeSummary | null>(null);
  const [experiments, setExperiments] = useState<ExperimentsResponse | null>(null);

  const [solveRes, setSolveRes] = useState<SolveResponse | null>(null);
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
        setRegionNames(l.region_names);
        setSummary(s);
        setExperiments(e);
      })
      .catch((err) => console.error("Init error:", err));
  }, []);

  const triggerPulse = (kind: "reward" | "punish") => {
    setPulse(kind);
    window.setTimeout(() => setPulse(null), 900);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <h1>Drosophila Connectome · Reward/Punishment Math Lab</h1>
          <p>FlyWire v783 Whole-Brain LIF Spiking Simulator (3-Factor R-STDP &amp; REINFORCE)</p>
        </div>
        <div className="status-pills">
          <span className="pill">
            Device: <strong>{health?.gpu_name || "Loading..."}</strong>
          </span>
          <span className="pill">
            Mem:{" "}
            <strong>
              {health ? `${health.gpu_mem_allocated_mb} / ${health.gpu_mem_total_mb} MB` : "-"}
            </strong>
          </span>
          <span className="pill">
            Neurons:{" "}
            <strong>{summary ? summary.total_neurons.toLocaleString() : "-"}</strong>
          </span>
          <span className="pill">
            Synapses:{" "}
            <strong>{summary ? summary.total_synapses.toLocaleString() : "-"}</strong>
          </span>
        </div>
      </header>

      <div className="grid-main">
        <div>
          <BrainCanvas2D
            landmarks={landmarks}
            regionNames={regionNames}
            spikeEvents={solveRes?.spike_events || null}
            regionActivity={solveRes?.region_activity || null}
            lesions={lesions}
            pulse={pulse}
          />
          <BenchmarkPanel summary={summary} experiments={experiments} />
        </div>

        <div>
          <QuizRewardPanel onSolved={setSolveRes} onPulse={triggerPulse} />
          <LesionPanel
            lesions={lesions}
            onChange={setLesions}
            onReset={() => setSolveRes(null)}
          />
        </div>
      </div>
    </div>
  );
};
