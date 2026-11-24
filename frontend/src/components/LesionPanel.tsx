import React, { useState } from "react";
import { apiClient, LesionState } from "../api/client";

interface LesionPanelProps {
  lesions: LesionState;
  onChange: (next: LesionState) => void;
  onReset: () => void;
}

export const LesionPanel: React.FC<LesionPanelProps> = ({ lesions, onChange, onReset }) => {
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
        <h2 className="panel-title">Lesion &amp; Checkpoint Controls</h2>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
          Disabled: {(lesions.disabled_neurons ?? 0).toLocaleString()}
        </span>
      </div>

      <div className="lesion-list">
        <button
          className={`lesion-btn ${lesions.mushroom_body ? "lesioned" : ""}`}
          disabled={busy}
          onClick={() => toggle("mushroom_body")}
        >
          <span>Mushroom Body</span>
          <span>{lesions.mushroom_body ? "OFF" : "ON"}</span>
        </button>
        <button
          className={`lesion-btn ${lesions.central_complex ? "lesioned" : ""}`}
          disabled={busy}
          onClick={() => toggle("central_complex")}
        >
          <span>Central Complex</span>
          <span>{lesions.central_complex ? "OFF" : "ON"}</span>
        </button>
        <button
          className={`lesion-btn ${lesions.hub_top_5pct ? "lesioned" : ""}`}
          disabled={busy}
          onClick={() => toggle("hub_top_5pct")}
        >
          <span>Top 5% Hubs</span>
          <span>{lesions.hub_top_5pct ? "OFF" : "ON"}</span>
        </button>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
        <select
          aria-label="Reset mode"
          value={mode}
          onChange={(e) => setMode(e.target.value as "checkpoint" | "fresh" | "shuffled")}
        >
          <option value="checkpoint">Checkpoint</option>
          <option value="fresh">Fresh</option>
          <option value="shuffled">Shuffled Control</option>
        </select>
        <button disabled={busy} onClick={handleReset}>
          Reset Brain
        </button>
      </div>
    </div>
  );
};
