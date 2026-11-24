# Drosophila Connectome Reward/Punishment Math Trainer

**Live Application:** [https://drosophila-connectome-math.vercel.app](https://drosophila-connectome-math.vercel.app)

## Project Overview
A scientific computing and reinforcement-learning application that loads the real **2024 FlyWire v783 *Drosophila melanogaster* whole-brain connectome** (`138,639` proofread neurons, `15,091,983` directed edges, `54,492,922` synapses), simulates it as a batched **Leaky Integrate-and-Fire (LIF)** spiking neural network on GPU/CPU, and trains it to solve delayed working-memory and arithmetic tasks via **dopamine reward (`+1`) and punishment (`-1`)**.

## Features
- **FlyWire v783 Connectome Loader:** Ingests `Completeness_783.csv`, `Connectivity_783.parquet`, and FlyWire annotations into DuckDB + Parquet, compiling a sparse PyTorch CSR graph respecting Dale's Law.
- **Batched Sparse LIF Simulator:** Supports `(B, N)` vectorized membrane integration, exponential synaptic current decay, refractory periods, and region/hub lesioning on `local_gpu` (CUDA) and `local_cpu`.
- **Dual Plasticity Engines:** Three-factor dopamine-modulated **R-STDP** (decaying eligibility traces with strict Dale's Law clamping) and **REINFORCE** policy gradients.
- **Scientific Controls & Ablations:** Degree-preserving configuration-model shuffled brain, matched-budget PyTorch GRU baseline, and targeted lesions (`Mushroom Body`, `Central Complex`, `Top 5% Hubs`) with post-lesion recovery tracking.
- **Interactive 2D Dorsal Brain Visualizer:** Obsidian-indigo HTML5 Canvas + anatomical SVG neuropil interface (optic lobes, antennal lobes, Mushroom Bodies, Central Complex) playing back 60 ms spike cascades at 60 FPS with live `+1` / `-1` dopamine ripples, `0–100` keyboard inputs, and a 3-tab workflow (*Play & Train*, *Brain Damage Lab*, *Science & Benchmarks*).

## Technology Stack
- **Frontend:** React, TypeScript, Vite, HTML5 Canvas, SVG, Vitest, Testing Library
- **Backend (Serverless & Local):** Python 3.12, FastAPI, Pydantic v2, NumPy, SciPy, DuckDB
- **Training & Research:** PyTorch CUDA (`torch.sparse` CSR), Polars, PyArrow, Colab Pro (`notebooks/train_colab.ipynb`)
- **Deployment:** Vercel (same-origin static frontend + Python serverless API)

## System Architecture
```text
2D Brain UI (React + Canvas/SVG)
   │
   ├── POST /api/brain/solve & /feedback
   │        │
   │        ▼
   │   Sparse LIF Simulator + 3-Factor Eligibility Traces
   │        │
   │        ├── Dale's Law Clamped Weight Update (E >= 0, I <= 0)
   │        └── Active Region / Hub Lesion Mask
   │
   └── GET /api/connectome/summary & /experiments (DuckDB + Artifacts)
```

## API Endpoints
| Route | Method | Purpose |
| --- | --- | --- |
| `/api/health` | `GET` | Service health, execution target (`local_cpu`, `local_gpu`, `serverless_cpu`), and memory status |
| `/api/brain/layout` | `GET` | 3,000 dorsal 2D landmark neuron coordinates across 6 anatomical regions |
| `/api/brain/solve` | `POST` | Run a 60 ms LIF trial for `add_3bit` (`0–100`), `sub_3bit` (`0–100`), `mod_add` (`0–6`), or `delayed_xor` (`0–1`) |
| `/api/brain/feedback` | `POST` | Apply manual `+1` (dopamine reward) or `-1` (punishment) to the cached trial |
| `/api/brain/train-step` | `POST` | Run a batch of automated RL training episodes (serverless HTTP) |
| `/api/ws/train` | `WS` | Stream per-episode training metrics in real time (local `flybrain_rl.api` server) |
| `/api/brain/lesion` | `POST` | Toggle `Mushroom Body`, `Central Complex`, and `Top 5% Hubs` lesions |
| `/api/brain/reset` | `POST` | Reload `checkpoint`, `fresh`, or `shuffled` connectome weights |
| `/api/connectome/summary` | `GET` | DuckDB neuron, E/I, synapse, and hub statistics |
| `/api/experiments` | `GET` | Hardware throughput benchmarks and experiment results |

## Reproducibility & Research Structure
```powershell
# 1. Setup Python 3.12 environment
python -m venv .venv
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -e ".[dev]"

# 2. Download & ingest FlyWire v783 connectome
.\.venv\Scripts\python scripts/download_connectome.py

# 3. Run local hardware benchmark & fast smoke experiment
.\.venv\Scripts\python scripts/benchmark_hardware.py
.\.venv\Scripts\python scripts/run_experiments.py --smoke-test
```
For full multi-seed sweeps (`batch_size=256–1024`) on `cloud_gpu`, run [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) and inspect [`artifacts/experiments/`](artifacts/experiments/results.csv).

## Scientific Data Sources & Attribution
- **FlyWire v783 Connectome & Computational Model:** Shiu, P. K., Sterne, G. R., Spiller, N., et al. (2024). *A Drosophila computational brain model reveals sensorimotor processing*. **Nature** (`philshiu/Drosophila_brain_model`).
- **FlyWire Annotations:** `flyconnectome/flywire_annotations` (`Supplemental_file1_neuron_annotations.tsv`).

## Known Limitations
- **Simplified LIF Dynamics:** Uses single-compartment point-neuron LIF equations with fixed time constants rather than multi-compartment Hodgkin–Huxley biophysics.
- **Population Rate Coding:** Discrete operands are encoded via rate-coded sensory pools and decoded via firing-rate argmax over a 60 ms window.
- **Task Scope:** Evaluated on discrete logic and modular arithmetic (`delayed_xor`, `add_3bit`, `mod_add`).
- **Serverless Inference:** The Vercel deployment (`api/index.py`) runs lightweight NumPy/SciPy sparse CSR inference on the 3,000-landmark subnetwork to stay within serverless bundle limits, while full 138,639-neuron GPU training runs via `flybrain_rl` locally or on `cloud_gpu`.

## Documentation & License
- [Biology & Mathematical Formulation](docs/BIOLOGY_AND_MATH.md)
- [Hardware Benchmarks & Controls](docs/BENCHMARKS.md)
- [Data Sources & Schema](docs/DATA_SOURCES.md)
- [MIT License](LICENSE)
