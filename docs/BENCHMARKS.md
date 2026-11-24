# Hardware Benchmarks & Scientific Controls

## 1. `local_cpu` vs. `local_gpu` Throughput (`artifacts/hardware_benchmark.json`)
Measured on the full `138,639`-neuron, `15,091,983`-edge FlyWire v783 connectome (`T = 60 ms` trial):

| Device | Batch Size | Trial Latency (ms) | Spikes / sec | Peak GPU Mem (MB) |
| --- | ---: | ---: | ---: | ---: |
| `local_cpu` | 1 | 1572.10 | 13,630.1 | 0.00 |
| `local_cpu` | 16 | 5417.78 | 63,282.0 | 0.00 |
| `local_cpu` | 64 | 19596.30 | 69,982.2 | 0.00 |
| `local_gpu` | 1 | 129.40 | 165,568.4 | 330.08 |
| `local_gpu` | 16 | 11886.21 | 28,838.8 | 904.25 |
| `local_gpu` | 64 | 59356.44 | 23,100.0 | 2745.96 |

## 2. Local Smoke-Test Results (`scripts/run_experiments.py --smoke-test`)
Run on `local_gpu` (`delayed_xor`, 10 episodes, `batch_size=1`, seed `0`):
- `real_fly` (`rstdp`): `70.0%` held-out test accuracy (`955.6` mean spikes/trial)
- `real_fly` (`reinforce`): `60.0%` held-out test accuracy (`1030.5` mean spikes/trial)
- `shuffled_brain` (`rstdp`): `70.0%` held-out test accuracy (`958.6` mean spikes/trial)
- `pytorch_gru` (`reinforce`): `40.0%` held-out test accuracy

Full multi-seed sweeps across `delayed_xor`, `add_3bit`, and `mod_add` with `batch_size=256` are configured in `notebooks/train_colab.ipynb` for `cloud_gpu`.
