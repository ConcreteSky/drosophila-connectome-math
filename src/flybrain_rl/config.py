"""Centralized configuration dataclasses for `flybrain_rl`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"

REGION_NAMES: tuple[str, ...] = (
    "Sensory",
    "Antennal Lobe",
    "Mushroom Body",
    "Central Complex",
    "Lateral Horn",
    "Motor/Output",
)

# Canonical dorsal 2D region centers [x, y] in [0, 1]^2 for faithful anatomical projection
REGION_CENTERS_2D: dict[str, tuple[float, float]] = {
    "Sensory": (0.16, 0.50),        # Lateral optic lobes & peripheral sensory
    "Antennal Lobe": (0.34, 0.64),  # Anterior-medial antennal lobes
    "Mushroom Body": (0.48, 0.34),  # Dorsal calyces & lobes
    "Central Complex": (0.50, 0.54),# Midline central complex
    "Lateral Horn": (0.64, 0.38),   # Lateral protocerebrum
    "Motor/Output": (0.82, 0.52),   # Descending / motor output tracts
}


@dataclass(frozen=True)
class LIFConfig:
    """Biological Leaky Integrate-and-Fire neuron parameters."""

    dt_ms: float = 1.0
    trial_ms: int = 60
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_thresh_mv: float = -45.0
    refrac_steps: int = 2
    r_mem: float = 1.0
    weight_scale: float = 2.0


@dataclass(frozen=True)
class TrainingConfig:
    """Plasticity and reinforcement learning hyperparameters."""

    batch_size: int = 32
    lr_rstdp: float = 0.35
    conn_lr_rstdp: float = 0.002
    lambda_e: float = 0.95
    a_plus: float = 1.0
    a_minus: float = 0.5
    w_max: float = 5.0
    lr_reinforce: float = 0.03
    temperature: float = 0.5
    reward_baseline_alpha: float = 0.05
