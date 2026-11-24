"""Escalating math & working-memory tasks with zero-leakage train/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from flybrain_rl.encoding import PopulationEncoderDecoder, TrialSchedule

TaskName = Literal["delayed_xor", "add_3bit", "mod_add"]


@dataclass
class MathProblem:
    task: TaskName
    a: int
    b: int
    target: int
    num_classes: int
    label: str
    a_vec: list[float]
    b_vec: list[float]


class MathTaskEnv:
    """Deterministic zero-leakage train/test split environment for Drosophila math RL."""

    def __init__(
        self,
        encoder: PopulationEncoderDecoder,
        task: TaskName = "delayed_xor",
        split_seed: int = 42,
    ) -> None:
        self.encoder = encoder
        self.device = encoder.device
        self.task = task
        self.num_classes = self.num_classes_for(task)
        self.schedule = self.schedule_for(task)

        all_pairs, n_train = self._all_pairs_and_split(task)
        rng = torch.Generator(device="cpu").manual_seed(split_seed)
        perm = torch.randperm(len(all_pairs), generator=rng).tolist()
        self.train_pairs = [all_pairs[i] for i in perm[:n_train]]
        self.test_pairs = [all_pairs[i] for i in perm[n_train:]]

    @staticmethod
    def num_classes_for(task: TaskName) -> int:
        if task == "delayed_xor":
            return 2
        if task == "add_3bit":
            return 15
        if task == "mod_add":
            return 7
        raise ValueError(f"Unknown task: {task}")

    @staticmethod
    def schedule_for(task: TaskName) -> TrialSchedule:
        if task == "delayed_xor":
            # A in [0, 15], silent delay [15, 30], B in [30, 45], readout [45, 60]
            return TrialSchedule(total_ms=60, a_start=0, a_end=15, b_start=30, b_end=45, readout_start=45, readout_end=60)
        # add_3bit & mod_add: operands [0, 30], readout [30, 60]
        return TrialSchedule(total_ms=60, a_start=0, a_end=30, b_start=0, b_end=30, readout_start=30, readout_end=60)

    @staticmethod
    def _all_pairs_and_split(task: TaskName) -> tuple[list[tuple[int, int]], int]:
        if task == "delayed_xor":
            # 3-bit delayed parity: a in 0..3 (2 bits), b in 0..1 (1 bit) -> 8 pairs (6 train / 2 test)
            return [(a, b) for a in range(4) for b in range(2)], 6
        if task == "add_3bit":
            # 64 pairs: 48 train / 16 unseen test
            return [(a, b) for a in range(8) for b in range(8)], 48
        if task == "mod_add":
            # a + b mod 7 -> 49 pairs: 37 train / 12 unseen test
            return [(a, b) for a in range(7) for b in range(7)], 37
        raise ValueError(f"Unknown task: {task}")

    @staticmethod
    def make_problem(
        task: TaskName,
        a: int,
        b: int,
        num_a_channels: int = 7,
        num_b_channels: int = 7,
    ) -> MathProblem:
        a_vec = [0.0] * num_a_channels
        b_vec = [0.0] * num_b_channels

        if task == "delayed_xor":
            av, bv = int(a) & 3, int(b) & 1
            a0, a1 = av & 1, (av >> 1) & 1
            target = a0 ^ a1 ^ bv
            a_vec[0], a_vec[1] = float(a0), float(1 - a0)
            a_vec[2], a_vec[3] = float(a1), float(1 - a1)
            b_vec[0], b_vec[1] = float(bv), float(1 - bv)
            return MathProblem(task, av, bv, target, 2, f"({a1},{a0}) XOR {bv}", a_vec, b_vec)

        if task == "add_3bit":
            av, bv = max(0, min(7, int(a))), max(0, min(7, int(b)))
            target = av + bv
            for bit in range(3):
                ba, bb = (av >> bit) & 1, (bv >> bit) & 1
                a_vec[2 * bit], a_vec[2 * bit + 1] = float(ba), float(1 - ba)
                b_vec[2 * bit], b_vec[2 * bit + 1] = float(bb), float(1 - bb)
            return MathProblem(task, av, bv, target, 15, f"{av} + {bv}", a_vec, b_vec)

        # mod_add (mod 7)
        av, bv = int(a) % 7, int(b) % 7
        target = (av + bv) % 7
        a_vec[av] = 1.0
        b_vec[bv] = 1.0
        return MathProblem(task, av, bv, target, 7, f"({av} + {bv}) mod 7", a_vec, b_vec)

    def sample_batch(
        self,
        split: Literal["train", "test"] = "train",
        batch_size: int = 32,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[list[MathProblem], torch.Tensor, torch.Tensor, torch.Tensor]:
        pool = self.train_pairs if split == "train" else self.test_pairs
        idx = torch.randint(0, len(pool), (batch_size,), generator=rng).tolist()
        problems = [
            self.make_problem(
                self.task,
                pool[i][0],
                pool[i][1],
                num_a_channels=self.encoder.num_a_channels,
                num_b_channels=self.encoder.num_b_channels,
            )
            for i in idx
        ]
        a_vecs = torch.tensor([p.a_vec for p in problems], dtype=torch.float32)
        b_vecs = torch.tensor([p.b_vec for p in problems], dtype=torch.float32)
        targets = torch.tensor([p.target for p in problems], dtype=torch.long, device=self.device)
        return problems, a_vecs, b_vecs, targets

    @staticmethod
    def compute_reward(pred: torch.Tensor | int, target: torch.Tensor | int) -> torch.Tensor | float:
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            return torch.where(
                pred == target,
                torch.ones_like(pred, dtype=torch.float32),
                -torch.ones_like(pred, dtype=torch.float32),
            )
        return 1.0 if int(pred) == int(target) else -1.0
