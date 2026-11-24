import React from "react";
import { ConnectomeSummary, ExperimentsResponse } from "../api/client";

interface BenchmarkPanelProps {
  summary: ConnectomeSummary | null;
  experiments: ExperimentsResponse | null;
}

export const BenchmarkPanel: React.FC<BenchmarkPanelProps> = ({ summary, experiments }) => {
  const hwRuns = experiments?.hardware_benchmark?.runs || [];
  const modelRows = experiments?.summary?.by_task_model || [];
  const lesionRows = experiments?.summary?.lesions || [];

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Scientific Controls &amp; Connectome Telemetry</h2>
      </div>

      {hwRuns.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <table className="table-clean">
            <thead>
              <tr>
                <th>Device</th>
                <th>Batch</th>
                <th>Trial (ms)</th>
                <th>Spikes / sec</th>
                <th>Mem (MB)</th>
              </tr>
            </thead>
            <tbody>
              {hwRuns.map((r, i) => (
                <tr key={i}>
                  <td style={{ color: r.device === "cuda" ? "#86efac" : undefined }}>{r.device_name}</td>
                  <td>{r.batch_size}</td>
                  <td>{r.trial_latency_ms}</td>
                  <td>{r.spikes_per_sec.toLocaleString()}</td>
                  <td>{r.gpu_mem_peak_mb}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {modelRows.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <table className="table-clean">
            <thead>
              <tr>
                <th>Task</th>
                <th>Model</th>
                <th>Plasticity</th>
                <th>Test Acc</th>
                <th>Mean Spikes</th>
              </tr>
            </thead>
            <tbody>
              {modelRows.map((m, i) => (
                <tr key={i}>
                  <td>{m.task}</td>
                  <td>{m.model_type}</td>
                  <td>{m.plasticity}</td>
                  <td>{(m.test_acc * 100).toFixed(1)}%</td>
                  <td>{Math.round(m.mean_spikes).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {lesionRows.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <table className="table-clean">
            <thead>
              <tr>
                <th>Task</th>
                <th>Lesion Condition</th>
                <th>Test Acc</th>
                <th>Mean Spikes</th>
              </tr>
            </thead>
            <tbody>
              {lesionRows.map((l, i) => (
                <tr key={i}>
                  <td>{l.task}</td>
                  <td>{l.lesion}</td>
                  <td>{(l.test_acc * 100).toFixed(1)}%</td>
                  <td>{Math.round(l.mean_spikes).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {summary && (
        <table className="table-clean">
          <thead>
            <tr>
              <th>Brain Region</th>
              <th>Neuron Count</th>
            </tr>
          </thead>
          <tbody>
            {summary.regions.map((r) => (
              <tr key={r.region}>
                <td>{r.region}</td>
                <td>{r.count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};
