import React, { useRef, useState } from "react";
import { apiClient, FeedbackResponse, SolveResponse, TaskName } from "../api/client";

interface QuizRewardPanelProps {
  onSolved: (res: SolveResponse) => void;
  onPulse: (kind: "reward" | "punish") => void;
}

export const QuizRewardPanel: React.FC<QuizRewardPanelProps> = ({ onSolved, onPulse }) => {
  const [task, setTask] = useState<TaskName>("delayed_xor");
  const [a, setA] = useState<number>(1);
  const [b, setB] = useState<number>(0);
  const [busy, setBusy] = useState<boolean>(false);
  const [solveRes, setSolveRes] = useState<SolveResponse | null>(null);
  const [lastFb, setLastFb] = useState<FeedbackResponse | null>(null);
  const [training, setTraining] = useState<boolean>(false);
  const [accHistory, setAccHistory] = useState<number[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const maxA = task === "delayed_xor" ? 3 : task === "add_3bit" ? 7 : 6;
  const maxB = task === "delayed_xor" ? 1 : task === "add_3bit" ? 7 : 6;

  const handleRandom = () => {
    setA(Math.floor(Math.random() * (maxA + 1)));
    setB(Math.floor(Math.random() * (maxB + 1)));
  };

  const handleRun = async () => {
    setBusy(true);
    try {
      const res = await apiClient.solve(task, a, b);
      setSolveRes(res);
      setLastFb(null);
      onSolved(res);
    } finally {
      setBusy(false);
    }
  };

  const handleFeedback = async (reward: 1 | -1) => {
    setBusy(true);
    try {
      const fb = await apiClient.feedback(reward);
      setLastFb(fb);
      onPulse(fb.dopamine_pulse);
    } finally {
      setBusy(false);
    }
  };

  const handleToggleAutoTrain = () => {
    if (training && wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setTraining(false);
      return;
    }
    const ws = new WebSocket(apiClient.wsTrainUrl());
    wsRef.current = ws;
    setTraining(true);
    setAccHistory([]);
    ws.onopen = () => {
      ws.send(JSON.stringify({ task, episodes: 20, batch_size: 4, plasticity: "rstdp" }));
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      setAccHistory((prev) => [...prev, msg.test_acc]);
    };
    ws.onclose = () => {
      setTraining(false);
      wsRef.current = null;
    };
    ws.onerror = () => {
      setTraining(false);
      wsRef.current = null;
    };
  };

  const maxPoolCount = Math.max(1, ...(solveRes?.output_spike_counts || [1]));

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Quiz &amp; Dopamine Training</h2>
        {lastFb && (
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
            ΔW={lastFb.weight_delta_norm.toFixed(4)} · updates={lastFb.total_updates}
          </span>
        )}
      </div>

      <div className="controls-row">
        <div className="field">
          <label htmlFor="task-select">Task</label>
          <select
            id="task-select"
            value={task}
            onChange={(e) => setTask(e.target.value as TaskName)}
          >
            <option value="delayed_xor">Delayed XOR</option>
            <option value="add_3bit">3-Bit Add</option>
            <option value="mod_add">Mod Add (mod 7)</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="input-a">A</label>
          <input
            id="input-a"
            type="number"
            min={0}
            max={maxA}
            value={a}
            onChange={(e) => setA(Number(e.target.value))}
          />
        </div>

        <div className="field">
          <label htmlFor="input-b">B</label>
          <input
            id="input-b"
            type="number"
            min={0}
            max={maxB}
            value={b}
            onChange={(e) => setB(Number(e.target.value))}
          />
        </div>

        <button onClick={handleRandom} disabled={busy}>
          Random
        </button>
        <button className="primary" onClick={handleRun} disabled={busy}>
          Run Brain
        </button>
        <button onClick={handleToggleAutoTrain} disabled={busy}>
          {training ? "Stop Train" : "Auto-Train"}
        </button>
      </div>

      {solveRes && (
        <>
          <div className="result-grid">
            <div className="metric-box">
              <div className="metric-label">Guess</div>
              <div
                className="metric-value"
                style={{ color: solveRes.is_correct ? "var(--reward)" : "var(--punish)" }}
              >
                {solveRes.predicted_answer}
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">True</div>
              <div className="metric-value">{solveRes.true_answer}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Spikes</div>
              <div className="metric-value">{solveRes.total_spikes.toLocaleString()}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Latency</div>
              <div className="metric-value">{solveRes.latency_ms} ms</div>
            </div>
          </div>

          {/* Output pool spike-count bar chart */}
          <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 42, margin: "8px 0" }}>
            {solveRes.output_spike_counts.map((cnt, k) => (
              <div key={k} style={{ flex: 1, textAlign: "center" }}>
                <div
                  style={{
                    height: `${Math.max(3, (cnt / maxPoolCount) * 28)}px`,
                    background: k === solveRes.predicted_answer ? "var(--spike)" : "#30363d",
                    borderRadius: 2,
                  }}
                />
                <div style={{ fontSize: 9, fontFamily: "var(--mono)", color: "var(--text-dim)" }}>
                  {k}
                </div>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn-reward" disabled={busy} onClick={() => handleFeedback(1)}>
              Reward (+1)
            </button>
            <button className="btn-punish" disabled={busy} onClick={() => handleFeedback(-1)}>
              Punish (-1)
            </button>
          </div>
        </>
      )}

      {accHistory.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
            Live Auto-Train Test Acc: {(accHistory[accHistory.length - 1] * 100).toFixed(1)}% (ep{" "}
            {accHistory.length})
          </div>
        </div>
      )}
    </div>
  );
};
