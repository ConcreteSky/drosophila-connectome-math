"""Export lightweight NumPy `.npz` & `.json` artifacts with 0-100 bit-coded R-STDP weights.

Trains strictly on a sampled training subset and evaluates on unseen (A, B) pairs
so reported generalization is never memorized.
"""

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
    a_vec = np.zeros(14, dtype=np.float32)
    b_vec = np.zeros(14, dtype=np.float32)
    if task == "delayed_xor":
        av, bv = int(a) & 1, int(b) & 1
        a_vec[0], a_vec[1] = float(av), float(1 - av)
        b_vec[0], b_vec[1] = float(bv), float(1 - bv)
        return av ^ bv, a_vec, b_vec
    if task in ("add_3bit", "sub_3bit"):
        av = max(0, min(100, int(a)))
        bv = max(0, min(100, int(b)))
        target = (av + bv) if task == "add_3bit" else (av - bv)
        for bit in range(7):
            ba = (av >> bit) & 1
            bb = (bv >> bit) & 1
            a_vec[2 * bit], a_vec[2 * bit + 1] = float(ba), float(1 - ba)
            b_vec[2 * bit], b_vec[2 * bit + 1] = float(bb), float(1 - bb)
        return target, a_vec, b_vec
    av, bv = int(a) % 7, int(b) % 7
    a_vec[av] = 1.0
    b_vec[bv] = 1.0
    return (av + bv) % 7, a_vec, b_vec


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
    task_ids = {"delayed_xor": 1, "add_3bit": 2, "sub_3bit": 3, "mod_add": 4}
    rng = np.random.default_rng(task_ids.get(task, 0) * 100000 + int(a) * 131 + int(b))
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
    m_counts = np.zeros(output_pools.shape[0], dtype=np.float32)
    for k in range(output_pools.shape[0]):
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


def _train_bitwise_or_class(
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
    rng = np.random.default_rng(123)
    if task == "delayed_xor":
        train_pairs = [(0, 0), (0, 1), (1, 0)]
        test_pairs = [(1, 1)]
        out_dim = 2
    elif task == "mod_add":
        all_p = [(a, b) for a in range(7) for b in range(7)]
        perm = rng.permutation(len(all_p))
        train_pairs = [all_p[i] for i in perm[:37]]
        test_pairs = [all_p[i] for i in perm[37:]]
        out_dim = 7
    else:
        # 0-100: sample 360 training pairs & 120 completely unseen test pairs
        seen = set()
        train_pairs, test_pairs = [], []
        while len(train_pairs) < 360:
            p = (int(rng.integers(0, 101)), int(rng.integers(0, 101)))
            if p not in seen:
                seen.add(p)
                train_pairs.append(p)
        while len(test_pairs) < 120:
            p = (int(rng.integers(0, 101)), int(rng.integers(0, 101)))
            if p not in seen:
                seen.add(p)
                test_pairs.append(p)
        # 9 output bits for add_3bit/sub_3bit: 8 magnitude bits + 1 sign bit
        out_dim = 9

    def _collect(pairs):
        feats, tgts = [], []
        for a, b in pairs:
            tgt, ft = _run_subnetwork_trial(
                W, n, a_pools, b_pools, go_pool, output_pools, mb_idx, cx_idx, W_kc, task, a, b
            )
            feats.append(ft)
            tgts.append(tgt)
        return np.stack(feats, axis=0), np.array(tgts, dtype=np.int64)

    X_tr, Y_tr = _collect(train_pairs)
    X_te, Y_te = _collect(test_pairs)

    W_out = np.zeros((out_dim, X_tr.shape[1]), dtype=np.float32)

    if task in ("add_3bit", "sub_3bit"):
        # Bit-wise binary targets: 8 magnitude bits (0..200) + sign bit
        def _to_bits(y: np.ndarray) -> np.ndarray:
            bits = np.zeros((len(y), 9), dtype=np.float32)
            mag = np.abs(y)
            for k in range(8):
                bits[:, k] = (mag >> k) & 1
            bits[:, 8] = (y < 0).astype(np.float32)
            return bits

        B_tr = _to_bits(Y_tr)
        lr = 3.0
        for _ in range(1200):
            logits = X_tr @ W_out.T
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
            grad = (B_tr - probs).T @ X_tr / len(Y_tr)
            W_out = np.clip(W_out + lr * grad, -5.0, 5.0)

        def _decode_int(X_in: np.ndarray) -> np.ndarray:
            probs = 1.0 / (1.0 + np.exp(-np.clip(X_in @ W_out.T, -15.0, 15.0)))
            pred_bits = (probs >= 0.5).astype(np.int64)
            mag = np.zeros(len(X_in), dtype=np.int64)
            for k in range(8):
                mag += pred_bits[:, k] << k
            return np.where(pred_bits[:, 8] == 1, -mag, mag)

        tr_acc = float((_decode_int(X_tr) == Y_tr).mean())
        te_acc = float((_decode_int(X_te) == Y_te).mean())
    else:
        lr = 4.0
        for _ in range(1200):
            logits = X_tr @ W_out.T
            exps = np.exp((logits - logits.max(axis=1, keepdims=True)) / 0.25)
            probs = exps / exps.sum(axis=1, keepdims=True)
            oh = np.zeros_like(probs)
            oh[np.arange(len(Y_tr)), Y_tr] = 1.0
            W_out = np.clip(W_out + lr * ((oh - probs).T @ X_tr / len(Y_tr)), -5.0, 5.0)

        tr_acc = float((np.argmax(X_tr @ W_out.T, axis=1) == Y_tr).mean())
        te_acc = float((np.argmax(X_te @ W_out.T, axis=1) == Y_te).mean())

    print(f"[train] {task}: train={tr_acc * 100:.1f}% ({len(Y_tr)} pairs) | unseen_test={te_acc * 100:.1f}% ({len(Y_te)} pairs)")
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
        num_a_channels=14,
        num_b_channels=14,
        neurons_per_pool=32,
        max_classes=18,
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
    feat_dim = (len(mb_idx) + len(cx_idx)) * 2 + output_pools.shape[0]
    rng = np.random.default_rng(42)
    W_kc = rng.normal(0, 1.0 / math.sqrt(max(1, feat_dim)), size=(128, feat_dim)).astype(np.float32)

    w_out_dict = {}
    for task in ("add_3bit", "sub_3bit", "mod_add", "delayed_xor"):
        w_out_dict[f"W_out_{task}"] = _train_bitwise_or_class(
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
