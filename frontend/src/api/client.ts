export type TaskName = "add_3bit" | "sub_3bit" | "mod_add" | "delayed_xor";

export interface HealthResponse {
  status: string;
  device: string;
  gpu_name: string;
  gpu_mem_allocated_mb: number;
  gpu_mem_total_mb: number;
  checkpoint_loaded: boolean;
}

export interface LandmarkNeuron {
  id: number;
  neuron_idx: number;
  x: number;
  y: number;
  region: string;
  sign: number;
}

export interface LayoutResponse {
  region_names: string[];
  landmarks: LandmarkNeuron[];
}

export interface SolveResponse {
  predicted_answer: number;
  true_answer: number;
  is_correct: boolean;
  answer_labels?: string[];
  output_spike_counts: number[];
  total_spikes: number;
  latency_ms: number;
  region_activity: Record<string, number[]>;
  spike_events: [number, number][];
}

export interface FeedbackResponse {
  reward_applied: number;
  weight_delta_norm: number;
  total_weight_shift?: number;
  dopamine_pulse: "reward" | "punish";
  total_updates: number;
}

export interface TrainStepResponse {
  task: TaskName;
  episodes: number;
  accuracy: number;
  total_weight_shift?: number;
}

export interface LesionState {
  mushroom_body: boolean;
  central_complex: boolean;
  hub_top_5pct: boolean;
  disabled_neurons?: number;
}

export interface ConnectomeSummary {
  total_neurons: number;
  excitatory_neurons: number;
  inhibitory_neurons: number;
  num_edges: number;
  total_synapses: number;
  regions: { region: string; count: number }[];
  top_hubs: { neuron_idx: number; out_degree: number; total_synapses: number }[];
}

export interface ExperimentsResponse {
  hardware_benchmark: {
    num_neurons: number;
    num_edges: number;
    trial_ms: number;
    runs: {
      device: string;
      device_name: string;
      batch_size: number;
      trial_latency_ms: number;
      ms_per_episode: number;
      total_spikes: number;
      spikes_per_sec: number;
      gpu_mem_peak_mb: number;
    }[];
  } | null;
  summary: {
    by_task_model: {
      task: string;
      model_type: string;
      plasticity: string;
      test_acc: number;
      mean_spikes: number;
    }[];
    lesions: {
      task: string;
      lesion: string;
      test_acc: number;
      mean_spikes: number;
    }[];
  } | null;
  results: Record<string, unknown>[];
}

async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
  }
  return resp.json() as Promise<T>;
}

export const apiClient = {
  health: () => apiJson<HealthResponse>("/api/health"),
  layout: () => apiJson<LayoutResponse>("/api/brain/layout"),
  solve: (task: TaskName, a: number, b: number) =>
    apiJson<SolveResponse>("/api/brain/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, a, b }),
    }),
  feedback: (reward: 1 | -1) =>
    apiJson<FeedbackResponse>("/api/brain/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reward }),
    }),
  trainStep: (task: TaskName, episodes = 10) =>
    apiJson<TrainStepResponse>("/api/brain/train-step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, episodes }),
    }),
  lesion: (req: { mushroom_body: boolean; central_complex: boolean; hub_top_5pct: boolean }) =>
    apiJson<LesionState>("/api/brain/lesion", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  reset: (mode: "checkpoint" | "fresh" | "shuffled") =>
    apiJson<{ status: string; mode: string }>("/api/brain/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    }),
  summary: () => apiJson<ConnectomeSummary>("/api/connectome/summary"),
  experiments: () => apiJson<ExperimentsResponse>("/api/experiments"),
};
