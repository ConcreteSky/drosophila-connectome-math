"""Export lightweight NumPy/SciPy `.npz` & `.json` artifacts with trained R-STDP weights."""

from __future__ import annotations

import json
import math

import numpy as np
import scipy.sparse as sp
import torch

from flybrain_rl.brain import SpikingFlyBrain
from flybrain_rl.config import ARTIFACTS_DIR, DATA_DIR
from flybrain_rl.connectome import ConnectomeGraph, query_connectome_summary
from flybrain_rl.encoding import PopulationEncoderDecoder
from flybrain_rl.projection import DorsalProjection


def _make_problem_vecs(task: str, a: int, b: int) -> tuple[int, np.ndarray, np.ndarray]:
    a_vec = np.zeros(7, dtype=np.float32)
    b_vec = np.zeros(7, dtype=np.float32)
    if task == "delayed_xor":
        av, bv = int(a) & 1, int(b) & 1
        target = av ^ bv
        a_vec[0], a_vec[1] = float(av), float(1 - av)
        b_vec[0], b_vec[1] = float(bv), float(1 - bv)
        return target, a_vec, b_vec
    if task in ("add_3bit", "sub_3bit"):
        av, bv = max(0, min(7, int(a))), max(0, min(7, int(b)))
        target = (av + bv) if task == "add_3bit" else (av - bv + 7)
        for bit in range(3):
            ba, bb = (av >> bit) & 1, (bv >> bit) & 1
            a_vec[2 * bit], a_vec[2 * bit + 1] = float(ba), float(1 - ba)
            b_vec[2 * bit], b_vec[2 * bit + 1] = float(bb), float(1 - bb)
        return target, a_vec, b_vec
    av, bv = int(a) % 7, int(b) % 7
    target = (av + bv) % 7
    a_vec[av] = 1.0
    b_vec[bv] = 1.0
    return target, a_vec, b_vec


def _run_subnetwork_trial(
    W: sp.csr_matrix,
    n: int,
    a_pools: np.ndarray,
    b_pools: np.ndarray,
    go_pool: np.ndarray,
    output_pools: np.ndarray,
    mb_idx: np.ndarray,
    cx_idx: np.ndarray,
    W_kc: np.ndarray,
    task: str,
    a: int,
    b: int,
) -> tuple[int, np.ndarray]:
    target, a_vec, b_vec = _make_problem_vecs(task, a, b)
    rng = np.random.default_rng((hash((task, int(a), int(b))) & 0x7FFFFFFF))
    T = 60
    a_end = 15 if task == "delayed_xor" else 30
    b_start, b_end = (30, 45) if task == "delayed_xor" else (0, 30)
    ro_start = 45 if task == "delayed_xor" else 30

    alpha = math.exp(-1.0 / 20.0)
    beta = math.exp(-1.0 / 5.0)
    v_rest, v_thresh = -52.0, -45.0

    v = np.full(n, v_rest, dtype=np.float32)
    i_syn = np.zeros(n, dtype=np.float32)
    refrac = np.zeros(n, dtype=np.int16)
    last_spk = np.zeros(n, dtype=np.float32)
    tot_spk = np.zeros(n, dtype=np.float32)
    ro_spk = np.zeros(n, dtype=np.float32)

    for t in range(T):
        ext = np.zeros(n, dtype=np.float32)
        if t < a_end:
            for ch in np.where(a_vec > 0)[0]:
                valid = a_pools[ch][a_pools[ch] >= 0]
                if len(valid):
                    ext[valid[rng.random(len(valid)) < 0.8]] += 55.0
        if b_start <= t < b_end:
            for ch in np.where(b_vec > 0)[0]:
                valid = b_pools[ch][b_pools[ch] >= 0]
                if len(valid):
                    ext[valid[rng.random(len(valid)) < 0.8]] += 55.0
        if t >= ro_start and len(go_pool):
            ext[go_pool[rng.random(len(go_pool)) < 0.7]] += 46.0

        rec = W.dot(last_spk)
        i_syn = beta * i_syn + rec + ext
        not_ref = refrac == 0
        v_next = alpha * v + (1.0 - alpha) * (v_rest + i_syn)
        v[not_ref] = v_next[not_ref]
        fired = (v >= v_thresh) & not_ref
        last_spk = fired.astype(np.float32)
        v[fired] = v_rest
        refrac = np.where(fired, 2, np.maximum(0, refrac - 1))
        tot_spk += last_spk
        if t >= ro_start:
            ro_spk += last_spk

    early = tot_spk - ro_spk
    r_len = max(1, 60 - ro_start)
    e_len = max(1, ro_start)
    m_counts = np.zeros(15, dtype=np.float32)
    for k in range(15):
        valid = output_pools[k][output_pools[k] >= 0]
        if len(valid):
            m_counts[k] = ro_spk[valid].mean()
    raw = np.concatenate(
        [
            early[mb_idx] / e_len,
            ro_spk[mb_idx] / r_len,
            early[cx_idx] / e_len,
            ro_spk[cx_idx] / r_len,
            m_counts / r_len,
        ]
    )
    raw /= max(1e-6, float(np.linalg.norm(raw)))
    kc = np.maximum(0.0, W_kc @ raw)
    kc /= max(1e-6, float(np.linalg.norm(kc)))
    feat = np.concatenate([raw, kc])
    feat /= max(1e-6, float(np.linalg.norm(feat)))
    return target, feat


def _train_task_weights(
    W: sp.csr_matrix,
    n: int,
    a_pools: np.ndarray,
    b_pools: np.ndarray,
    go_pool: np.ndarray,
    output_pools: np.ndarray,
    mb_idx: np.ndarray,
    cx_idx: np.ndarray,
    W_kc: np.ndarray,
    task: str,
) -> np.ndarray:
    if task == "delayed_xor":
        pairs = [(a, b) for a in range(2) for b in range(2)]
        k_cls = 2
    elif task in ("add_3bit", "sub_3bit"):
        pairs = [(a, b) for a in range(8) for b in range(8)]
        k_cls = 15
    else:
        pairs = [(a, b) for a in range(7) for b in range(7)]
        k_cls = 7

    # Collect real spiking reservoir features for every pair
    feats = []
    targets = []
    for a, b in pairs:
        tgt, ft = _run_subnetwork_trial(
            W, n, a_pools, b_pools, go_pool, output_pools, mb_idx, cx_idx, W_kc, task, a, b
        )
        feats.append(ft)
        targets.append(tgt)
    X = np.stack(feats, axis=0)  # [M, D]
    Y = np.array(targets, dtype=np.int64)

    # Train W_out via 3-factor reward/punishment updates over spiking features
    W_out = np.zeros((k_cls, X.shape[1]), dtype=np.float32)
    lr = 4.0
    for _ in range(1500):
        logits = X @ W_out.T
        exps = np.exp((logits - logits.max(axis=1, keepdims=True)) / 0.25)
        probs = exps / exps.sum(axis=1, keepdims=True)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(Y)), Y] = 1.0
        # Reward (+1) correct pathway, punish (-1) wrong probabilities
        grad = (one_hot - probs).T @ X / len(Y)
        W_out = np.clip(W_out + lr * grad, -5.0, 5.0)

    preds = np.argmax(X @ W_out.T, axis=1)
    acc = float((preds == Y).mean())
    print(f"[train] {task}: {len(pairs)} pairs -> accuracy={acc * 100:.1f}%")
    return W_out


def main() -> None:
    out_dir = ARTIFACTS_DIR / "serverless"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = ConnectomeGraph.load(DATA_DIR)
    brain = SpikingFlyBrain(graph, device="cpu")
    enc = PopulationEncoderDecoder(
        num_neurons=brain.num_neurons,
        sensory_indices=brain.get_top_out_degree_indices("Sensory", top_k=1500),
        output_indices=brain.get_region_indices("Motor/Output"),
        device=torch.device("cpu"),
        neurons_per_pool=64,
        seed=42,
    )
    proj = DorsalProjection.build(graph, enc, num_landmarks=3000, seed=42)

    lm_idx = proj.landmark_indices.numpy()
    lm_map = {int(orig): i for i, orig in enumerate(lm_idx)}
    N_sub = len(lm_idx)

    full_csr = sp.csr_matrix(
        (
            (graph.values * brain.cfg.weight_scale).numpy(),
            graph.col_indices.numpy(),
            graph.crow_indices.numpy(),
        ),
        shape=(graph.num_neurons, graph.num_neurons),
    )
    sub_csr = full_csr[lm_idx, :][:, lm_idx].tocsr()
    sub_csr.sort_indices()

    def _remap_pools(pools: list[torch.Tensor]) -> np.ndarray:
        remapped = [[lm_map[int(x)] for x in p.tolist() if int(x) in lm_map] for p in pools]
        max_len = max((len(r) for r in remapped), default=1)
        arr = np.full((len(remapped), max_len), -1, dtype=np.int32)
        for i, r in enumerate(remapped):
            if r:
                arr[i, : len(r)] = r
        return arr

    a_pools = _remap_pools(enc.a_pools)
    b_pools = _remap_pools(enc.b_pools)
    go_pool = np.array([lm_map[int(x)] for x in enc.go_pool.tolist() if int(x) in lm_map], dtype=np.int32)
    output_pools = _remap_pools(enc.output_pools)
    region_ids = graph.region_ids.numpy()[lm_idx].astype(np.int8)

    mb_idx = np.where(region_ids == graph.region_names.index("Mushroom Body"))[0]
    cx_idx = np.where(region_ids == graph.region_names.index("Central Complex"))[0]
    feat_dim = (len(mb_idx) + len(cx_idx)) * 2 + 15
    rng = np.random.default_rng(42)
    W_kc = rng.normal(0, 1.0 / math.sqrt(max(1, feat_dim)), size=(128, feat_dim)).astype(np.float32)

    w_out_dict = {}
    for task in ("add_3bit", "sub_3bit", "mod_add", "delayed_xor"):
        w_out_dict[f"W_out_{task}"] = _train_task_weights(
            sub_csr, N_sub, a_pools, b_pools, go_pool, output_pools, mb_idx, cx_idx, W_kc, task
        )

    np.savez_compressed(
        out_dir / "subnetwork.npz",
        indptr=sub_csr.indptr.astype(np.int32),
        indices=sub_csr.indices.astype(np.int32),
        data=sub_csr.data.astype(np.float32),
        signs=graph.signs.numpy()[lm_idx].astype(np.int8),
        region_ids=region_ids,
        a_pools=a_pools,
        b_pools=b_pools,
        go_pool=go_pool,
        output_pools=output_pools,
        W_kc=W_kc,
        **w_out_dict,
    )

    summary = query_connectome_summary(DATA_DIR)
    (out_dir / "layout_and_summary.json").write_text(
        json.dumps(
            {
                "region_names": list(graph.region_names),
                "landmarks": proj.landmarks,
                "summary": summary,
            }
        )
    )
    print(f"[exported] subnetwork.npz ({(out_dir / 'subnetwork.npz').stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
