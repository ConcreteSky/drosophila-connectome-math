import "@testing-library/jest-dom";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";

const mockFetch = vi.fn();
globalThis.fetch = mockFetch as unknown as typeof fetch;

beforeEach(() => {
  mockFetch.mockReset();
  mockFetch.mockImplementation((url: string) => {
    if (url.includes("/api/health")) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            status: "ok",
            device: "serverless_cpu",
            gpu_name: "serverless_cpu",
            gpu_mem_allocated_mb: 0,
            gpu_mem_total_mb: 0,
            checkpoint_loaded: true,
          })
        )
      );
    }
    if (url.includes("/api/brain/layout")) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            region_names: ["Sensory", "Antennal Lobe", "Mushroom Body", "Central Complex", "Lateral Horn", "Motor/Output"],
            landmarks: [{ id: 0, neuron_idx: 1, x: 0.2, y: 0.5, region: "Sensory", sign: 1 }],
          })
        )
      );
    }
    if (url.includes("/api/connectome/summary")) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            total_neurons: 138639,
            excitatory_neurons: 100000,
            inhibitory_neurons: 38639,
            num_edges: 15091983,
            total_synapses: 54492922,
            regions: [{ region: "Sensory", count: 102444 }],
            top_hubs: [],
          })
        )
      );
    }
    if (url.includes("/api/experiments")) {
      return Promise.resolve(
        new Response(JSON.stringify({ hardware_benchmark: null, summary: null, results: [] }))
      );
    }
    if (url.includes("/api/brain/solve")) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            predicted_answer: 8,
            true_answer: 8,
            is_correct: true,
            answer_labels: Array.from({ length: 15 }, (_, i) => String(i)),
            output_spike_counts: Array.from({ length: 15 }, (_, i) => (i === 8 ? 42 : 5)),
            total_spikes: 1200,
            latency_ms: 45.0,
            region_activity: { Sensory: Array(60).fill(10) },
            spike_events: [[0, 5]],
          })
        )
      );
    }
    if (url.includes("/api/brain/feedback")) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            reward_applied: 1,
            weight_delta_norm: 0.1234,
            total_weight_shift: 0.1234,
            dopamine_pulse: "reward",
            total_updates: 1,
          })
        )
      );
    }
    return Promise.resolve(new Response(JSON.stringify({})));
  });
});

describe("Drosophila Fly Brain Plain-English UI", () => {
  it("clamps keyboard inputs per task, submits on Enter, and applies reward", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/serverless_cpu/i)).toBeInTheDocument();
    });

    const inputA = screen.getByLabelText(/A \(0–100\)/i) as HTMLInputElement;
    // Out-of-range input gets clamped to 100
    fireEvent.change(inputA, { target: { value: "999" } });
    expect(inputA.value).toBe("100");

    // Switching to XOR clamps A and B to 0-1
    const select = screen.getByLabelText("Operation") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "delayed_xor" } });
    expect((screen.getByLabelText(/A \(0–1\)/i) as HTMLInputElement).value).toBe("1");

    // Switching to mod_add caps at 0-6
    fireEvent.change(select, { target: { value: "mod_add" } });
    fireEvent.change(screen.getByLabelText(/A \(0–6\)/i), { target: { value: "10" } });
    expect((screen.getByLabelText(/A \(0–6\)/i) as HTMLInputElement).value).toBe("6");

    // Enter key submits solve
    fireEvent.keyDown(screen.getByLabelText(/A \(0–6\)/i), { key: "Enter", code: "Enter" });
    await waitFor(() => {
      expect(screen.getByText(/Correct ✓/i)).toBeInTheDocument();
    });

    // Reward (+1) triggers dopamine pulse
    fireEvent.click(screen.getByRole("button", { name: "Reward (+1)" }));
    await waitFor(() => {
      expect(screen.getByText(/DOPAMINE \+1 REWARD/i)).toBeInTheDocument();
    });

    // 3-tab navigation works
    fireEvent.click(screen.getByText(/2\. Brain Damage Lab/i));
    expect(screen.getByText(/Turn Off Brain Regions/i)).toBeInTheDocument();
    fireEvent.click(screen.getByText(/3\. Science & Benchmarks/i));
    expect(screen.getByText(/Scientific Controls & Connectome Telemetry/i)).toBeInTheDocument();
  });
});
