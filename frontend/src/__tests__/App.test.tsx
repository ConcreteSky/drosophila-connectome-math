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
            device: "local_gpu",
            gpu_name: "local_gpu",
            gpu_mem_allocated_mb: 330,
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
            predicted_answer: 1,
            true_answer: 1,
            is_correct: true,
            output_spike_counts: [1.0, 4.5],
            total_spikes: 1200,
            latency_ms: 125.0,
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
            dopamine_pulse: "reward",
            total_updates: 1,
          })
        )
      );
    }
    return Promise.resolve(new Response(JSON.stringify({})));
  });
});

describe("Drosophila Connectome App", () => {
  it("renders header, solves problem, and applies reward", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/local_gpu/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Run Brain"));
    await waitFor(() => {
      expect(screen.getByText("Reward (+1)")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Reward (+1)"));
    await waitFor(() => {
      expect(screen.getByText(/DOPAMINE \+1 REWARD/i)).toBeInTheDocument();
    });
  });
});
