import React, { useState } from "react";
import { apiClient, LesionState, SolveResponse } from "../api/client";

interface LesionPanelProps {
  lesions: LesionState;
  lastIntactResult: SolveResponse | null;
  currentResult: SolveResponse | null;
  onChange: (next: LesionState) => void;
  onRerunSame: () => void;
  onReset: () => void;
}

export const LesionPanel: React.FC<LesionPanelProps> = ({
  lesions,
  lastIntactResult,
  currentResult,
  onChange,
  onRerunSame,
  onReset,
}) => {
  const [mode, setMode] = useState<"checkpoint" | "fresh" | "shuffled">("checkpoint");
  const [busy, setBusy] = useState<boolean>(false);

  const toggle = async (key: "mushroom_body" | "central_complex" | "hub_top_5pct") => {
    setBusy(true);
    try {
      const nextReq = {
        mushroom_body: key === "mushroom_body" ? !lesions.mushroom_body : lesions.mushroom_body,
        central_complex:
          key === "central_complex" ? !lesions.central_complex : lesions.central_complex,
        hub_top_5pct: key === "hub_top_5pct" ? !lesions.hub_top_5pct : lesions.hub_top_5pct,
      };
      const res = await apiClient.lesion(nextReq);
      onChange(res);
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    setBusy(true);
    try {
      await apiClient.reset(mode);
      onChange({ mushroom_body: false, central_complex: false, hub_top_5pct: false, disabled_neurons: 0 });
      onReset();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Turn Off Brain Regions (Brain Damage Lab)</h2>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
          Silenced: {(lesions.disabled_neurons ?? 0).toLocaleString()} neurons
        </span>
      </div>

      <p style={{ marginTop: 0, color: "var(--text-dim)", fontSize: 12 }}>
        Disable specific parts of the fly brain to see how its math answer and spike routing change.
      </p>

      <div className="lesion-list">
        <button
          className={`lesion-btn ${lesions.mushroom_body ? "lesioned" : ""}`}
          disabled={busy}
          title="Mushroom Body: the fly's associative memory & learning center"
          onClick={() => toggle("mushroom_body")}
        >
          <span>Turn off Memory &amp; Learning Center (Mushroom Body)</span>
          <span>{lesions.mushroom_body ? "OFF" : "ON"}</span>
        </button>

        <button
          className={`lesion-btn ${lesions.central_complex ? "lesioned" : ""}`}
          disabled={busy}
          title="Central Complex: central signal integration & routing hub"
          onClick={() => toggle("central_complex")}
        >
          <span>Turn off Signal Routing Hub (Central Complex)</span>
          <span>{lesions.central_complex ? "OFF" : "ON"}</span>
        </button>

        <button
          className={`lesion-btn ${lesions.hub_top_5pct ? "lesioned" : ""}`}
          disabled={busy}
          title="Top 5% highest-degree hub neurons across the connectome"
          onClick={() => toggle("hub_top_5pct")}
        >
          <span>Turn off Top 5% Hub Neurons</span>
          <span>{lesions.hub_top_5pct ? "OFF" : "ON"}</span>
        </button>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
        <button className="primary" disabled={busy} onClick={onRerunSame}>
          Re-run Same Problem
        </button>
        <select
          aria-label="Reset mode"
          value={mode}
          onChange={(e) => setMode(e.target.value as "checkpoint" | "fresh" | "shuffled")}
        >
          <option value="checkpoint">Trained Brain</option>
          <option value="fresh">Untrained Brain</option>
          <option value="shuffled">Randomly Wired Control</option>
        </select>
        <button disabled={busy} onClick={handleReset}>
          Reset Brain
        </button>
      </div>

      {(lastIntactResult || currentResult) && (
        <div
          style={{
            marginTop: 12,
            padding: "8px 10px",
            background: "var(--bg)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            fontFamily: "var(--mono)",
            fontSize: 11,
          }}
        >
          <div>
            Before (Intact):{" "}
            {lastIntactResult
              ? `Answer ${lastIntactResult.predicted_answer} (${lastIntactResult.is_correct ? "Correct" : "Wrong"}) · ${lastIntactResult.total_spikes} spikes`
              : "Run a problem first"}
          </div>
          <div style={{ marginTop: 4 }}>
            After (Current):{" "}
            {currentResult
              ? `Answer ${currentResult.predicted_answer} (${currentResult.is_correct ? "Correct" : "Wrong"}) · ${currentResult.total_spikes} spikes`
              : "-"}
          </div>
        </div>
      )}
    </div>
  );
};
