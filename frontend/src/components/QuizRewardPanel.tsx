import React, { useState } from "react";
import { apiClient, FeedbackResponse, SolveResponse, TaskName } from "../api/client";

export const TASK_LIMITS: Record<TaskName, { max: number; label: string; op: string }> = {
  add_3bit: { max: 100, label: "Addition (A + B)", op: "+" },
  sub_3bit: { max: 100, label: "Subtraction (A - B)", op: "-" },
  mod_add: { max: 6, label: "Modular Math ((A + B) mod 7)", op: "+ (mod 7)" },
  delayed_xor: { max: 1, label: "Logic Gate (A XOR B)", op: "XOR" },
};

const EXAMPLES: { label: string; task: TaskName; a: number; b: number }[] = [
  { label: "42 + 37", task: "add_3bit", a: 42, b: 37 },
  { label: "85 - 29", task: "sub_3bit", a: 85, b: 29 },
  { label: "(4 + 5) mod 7", task: "mod_add", a: 4, b: 5 },
  { label: "1 XOR 0", task: "delayed_xor", a: 1, b: 0 },
];

interface QuizRewardPanelProps {
  task: TaskName;
  a: number;
  b: number;
  busy: boolean;
  solveRes: SolveResponse | null;
  onTaskChange: (t: TaskName) => void;
  onAChange: (val: number) => void;
  onBChange: (val: number) => void;
  onRun: (overrideTask?: TaskName, overrideA?: number, overrideB?: number) => Promise<void> | void;
  onFeedbackSuccess: (fb: FeedbackResponse) => void;
}

export const QuizRewardPanel: React.FC<QuizRewardPanelProps> = ({
  task,
  a,
  b,
  busy,
  solveRes,
  onTaskChange,
  onAChange,
  onBChange,
  onRun,
  onFeedbackSuccess,
}) => {
  const [totalShift, setTotalShift] = useState<number>(0);
  const [lastDelta, setLastDelta] = useState<number | null>(null);
  const [trainLoading, setTrainLoading] = useState<boolean>(false);
  const [accHistory, setAccHistory] = useState<number[]>([]);

  const maxVal = TASK_LIMITS[task].max;
  const clampVal = (v: number, mx: number) => Math.max(0, Math.min(mx, Math.trunc(Number.isNaN(v) ? 0 : v)));

  const handleTaskSelect = (next: TaskName) => {
    const mx = TASK_LIMITS[next].max;
    onTaskChange(next);
    onAChange(clampVal(a, mx));
    onBChange(clampVal(b, mx));
  };

  const handleFeedback = async (reward: 1 | -1) => {
    const fb = await apiClient.feedback(reward);
    setLastDelta(fb.weight_delta_norm);
    setTotalShift(fb.total_weight_shift ?? totalShift + fb.weight_delta_norm);
    onFeedbackSuccess(fb);
    // Immediately re-test the same problem so user sees the brain's new answer
    await onRun(task, a, b);
  };

  const handleAutoTrain = async () => {
    setTrainLoading(true);
    try {
      const res = await apiClient.trainStep(task, 10);
      setAccHistory((prev) => [...prev, res.accuracy]);
      if (res.total_weight_shift !== undefined) {
        setTotalShift(res.total_weight_shift);
      }
      await onRun(task, a, b);
    } finally {
      setTrainLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !busy) {
      e.preventDefault();
      onRun();
    }
  };

  const topCandidates = solveRes
    ? solveRes.output_spike_counts
        .map((spikes, idx) => ({
          label: solveRes.answer_labels?.[idx] ?? String(idx),
          spikes,
        }))
        .sort((x, y) => y.spikes - x.spikes)
        .slice(0, 4)
    : [];
  const maxVotes = Math.max(1, ...topCandidates.map((c) => c.spikes));

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Ask &amp; Train the Fly Brain</h2>
        <span
          className="weight-shift-pill"
          title="Cumulative magnitude of synaptic weight changes from Reward/Punish updates"
        >
          Total Weight Shift: +{totalShift.toFixed(3)}
        </span>
      </div>

      <div className="chips-row" aria-label="Example problems">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            className="chip-btn"
            disabled={busy}
            onClick={() => {
              onTaskChange(ex.task);
              onAChange(ex.a);
              onBChange(ex.b);
              onRun(ex.task, ex.a, ex.b);
            }}
          >
            {ex.label}
          </button>
        ))}
      </div>

      <div className="controls-row" onKeyDown={handleKeyDown}>
        <div className="field">
          <label htmlFor="task-select">Operation</label>
          <select
            id="task-select"
            value={task}
            onChange={(e) => handleTaskSelect(e.target.value as TaskName)}
          >
            <option value="add_3bit">Addition (A + B)</option>
            <option value="sub_3bit">Subtraction (A - B)</option>
            <option value="mod_add">Modular Math ((A + B) mod 7)</option>
            <option value="delayed_xor">Logic Gate (A XOR B)</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="input-a">A (0–{maxVal})</label>
          <input
            id="input-a"
            type="text"
            inputMode="numeric"
            value={a}
            onChange={(e) => {
              const digits = e.target.value.replace(/\D/g, "");
              onAChange(digits === "" ? 0 : clampVal(Number(digits), maxVal));
            }}
          />
        </div>

        <div className="field">
          <label htmlFor="input-b">B (0–{maxVal})</label>
          <input
            id="input-b"
            type="text"
            inputMode="numeric"
            value={b}
            onChange={(e) => {
              const digits = e.target.value.replace(/\D/g, "");
              onBChange(digits === "" ? 0 : clampVal(Number(digits), maxVal));
            }}
          />
        </div>

        <button className="primary" type="button" onClick={() => onRun()} disabled={busy}>
          Ask the Fly
        </button>
        <button type="button" onClick={handleAutoTrain} disabled={busy || trainLoading}>
          {trainLoading ? "Training..." : "Auto-Train (10×)"}
        </button>
      </div>

      {solveRes && (
        <>
          <div style={{ marginTop: 10 }}>
            <div className="panel-title" style={{ marginBottom: 6 }}>
              Answer Decision Votes (Top 4 Candidates)
            </div>
            {topCandidates.map((cand, idx) => (
              <div key={cand.label} className="vote-row">
                <span>{cand.label}</span>
                <div className="vote-track">
                  <div
                    className="vote-fill"
                    style={{
                      width: `${Math.max(4, (cand.spikes / maxVotes) * 100)}%`,
                      background: idx === 0 ? "var(--spike)" : "#334155",
                    }}
                  />
                </div>
                <span style={{ textAlign: "right", color: "var(--text-dim)" }}>
                  {cand.spikes.toFixed(1)} spk
                </span>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 14, alignItems: "center", flexWrap: "wrap" }}>
            <button
              className="btn-reward"
              type="button"
              disabled={busy}
              title="Reinforce the active brain pathway (+1 dopamine) and re-test"
              onClick={() => handleFeedback(1)}
            >
              Reward (+1)
            </button>
            <button
              className="btn-punish"
              type="button"
              disabled={busy}
              title="Suppress the wrong answer pathway (-1 penalty) and immediately re-test"
              onClick={() => handleFeedback(-1)}
            >
              Punish (-1)
            </button>
            {lastDelta !== null && (
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
                Last step ΔW: {lastDelta.toFixed(4)}
              </span>
            )}
          </div>
        </>
      )}

      {accHistory.length > 0 && (
        <div style={{ marginTop: 12, fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-dim)" }}>
          Accuracy: {Math.round(accHistory[0] * 100)}%
          {accHistory.length > 1 ? ` → ${Math.round(accHistory[accHistory.length - 1] * 100)}%` : ""} (
          {accHistory.length * 10} training trials)
        </div>
      )}
    </div>
  );
};
