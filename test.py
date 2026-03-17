#!/usr/bin/env python3
"""
================================================================
TRC-SD Pipeline — Synthetic Validation Test
================================================================
This script validates all pipeline phases using synthetically
generated fingerprint images (concentric arc patterns).
Run this to confirm the pipeline works before using real FVC2002 data.
================================================================
"""

import sys
import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# ── patch CONFIG paths before importing pipeline ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath("./trcsd_pipeline.py")))
import trcsd_pipeline as P

# ── Setup ─────────────────────────────────────────────────────────────────────
P.CFG["output_dir"]  = "./outputs/trcsd"
P.CFG["vis_dir"]     = "./outputs/trcsd/sample_vis"
os.makedirs(P.CFG["output_dir"], exist_ok=True)
os.makedirs(P.CFG["vis_dir"],    exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("TEST")

PASS = "✓ PASS"
FAIL = "✗ FAIL"

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC FINGERPRINT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_fingerprint(H=300, W=300, n_arcs=12,
                               noise=25, seed=42) -> np.ndarray:
    """
    Generate a synthetic fingerprint as a noisy concentric-arc pattern.
    Returns (H, W) uint8 grayscale image.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2

    # Concentric arcs (like loops / whorls)
    Y, X    = np.mgrid[0:H, 0:W].astype(float)
    radius  = np.sqrt((Y - cy)**2 + (X - cx)**2)
    spacing = min(H, W) / (2 * n_arcs)

    img = np.sin(radius / spacing * np.pi) * 0.5 + 0.5

    # Add ridge thinning variation
    angle   = np.arctan2(Y - cy, X - cx)
    img    += 0.05 * np.sin(8 * angle)

    # Gaussian envelope (fingerprint is not uniform)
    sigma_env = min(H, W) * 0.35
    envelope  = np.exp(-((Y - cy)**2 + (X - cx)**2) / (2 * sigma_env**2))
    img      *= envelope

    # Normalise + Gaussian noise + quantise
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def make_dataset(tmp_dir: str, n_subjects=6, n_impressions=5, seed=0) -> dict:
    """
    Write synthetic fingerprint images to disk in FVC2002-like structure.
    Returns dataset dict compatible with P.load_dataset().
    """
    root = Path(tmp_dir) / "DB1_B"
    root.mkdir(parents=True, exist_ok=True)
    rng  = np.random.default_rng(seed)

    for s in range(1, n_subjects + 1):
        base_seed = s * 100
        for imp in range(1, n_impressions + 1):
            # Each impression: same base pattern + slight deformation + noise
            img  = make_synthetic_fingerprint(seed=base_seed)
            # Elastic-like distortion: random affine (scale+rotation)
            angle  = rng.uniform(-8, 8)
            scale  = rng.uniform(0.97, 1.03)
            M      = cv2.getRotationMatrix2D((150, 150), angle, scale)
            img    = cv2.warpAffine(img, M, (300, 300))
            # Add impression-specific noise
            noise_layer = rng.normal(0, 12, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise_layer, 0, 255).astype(np.uint8)
            cv2.imwrite(str(root / f"{s}_{imp}.bmp"), img)

    # Return dataset dict
    return {"DB1_B": {
        s: [str(root / f"{s}_{imp}.bmp") for imp in range(1, n_impressions + 1)]
        for s in range(1, n_subjects + 1)
    }}


# ─────────────────────────────────────────────────────────────────────────────
# TEST FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def test_phase1_preprocessing(img):
    log.info("  [Test] Phase 1.1 – Preprocessing & Minutiae Extraction …")
    enhanced = P.enhance_with_gabor(img)
    binary   = P.binarize(enhanced)
    skeleton = P.compute_skeleton(binary)
    minutiae, types = P.extract_minutiae(skeleton)

    assert enhanced.dtype == np.uint8,             "Enhanced must be uint8"
    assert binary.dtype   == bool,                 "Binary must be bool"
    assert skeleton.dtype == np.uint8,             "Skeleton must be uint8"
    assert minutiae.ndim  == 2 and minutiae.shape[1] == 2, "Minutiae must be (N,2)"
    assert len(minutiae)  == len(types),           "Minutiae/types count mismatch"
    assert set(types).issubset({"E", "B"}),        "Types must be E or B"
    assert len(minutiae)  > 0,                     "Should find at least 1 minutia"

    log.info(f"    Found {len(minutiae)} minutiae  "
             f"(endings={np.sum(types=='E')}, bifurcations={np.sum(types=='B')})")
    return PASS, {"minutiae": minutiae, "types": types,
                  "binary": binary, "skeleton": skeleton,
                  "gray": img, "enhanced": enhanced}


def test_phase1_delaunay(minutiae):
    log.info("  [Test] Phase 1.2 – Delaunay Graph Construction …")
    edges, adjacency = P.build_delaunay_graph(minutiae)

    assert edges.ndim == 2 and edges.shape[1] == 2, "Edges must be (E,2)"
    assert len(edges) > 0,                           "Graph must have edges"
    for node in adjacency:
        assert isinstance(adjacency[node], set),     "Adjacency values must be sets"

    log.info(f"    Graph: {len(minutiae)} nodes, {len(edges)} edges")
    return PASS, {"edges": edges, "adjacency": adjacency}


def test_phase2_ridge_weights(edges, minutiae, binary):
    log.info("  [Test] Phase 2 – Ridge-Count Edge Weighting …")
    weights, weight_dict = P.compute_ridge_weights(edges, minutiae, binary)

    assert len(weights) == len(edges),   "Weight count must equal edge count"
    assert np.all(weights >= 1),         "All weights must be ≥ 1"
    assert len(weight_dict) == len(edges), "Weight dict size mismatch"

    log.info(f"    Weights: min={weights.min()}, max={weights.max()}, "
             f"mean={weights.mean():.1f}")
    return PASS, {"weights": weights, "weight_dict": weight_dict}


def test_phase3_spectral(minutiae, adjacency, weight_dict):
    log.info("  [Test] Phase 3 – Spectral Descriptors …")
    size      = P.CFG["descriptor_size"]
    k         = P.CFG["k_hop"]

    # Phase 3.1 – subgraph
    node = 0
    sub_nodes, sub_edges = P.extract_k_hop_subgraph(node, adjacency, k)
    assert len(sub_nodes) >= 1,          "Subgraph must contain at least the centre"
    assert node in sub_nodes,            "Centre node must be in subgraph"

    # Phase 3.2 – spectral descriptor
    desc = P.compute_spectral_descriptor(sub_nodes, sub_edges, weight_dict, size)
    assert desc.shape == (size,),        f"Descriptor shape mismatch: {desc.shape}"
    assert not np.any(np.isnan(desc)),   "Descriptor contains NaN"
    assert not np.any(np.isinf(desc)),   "Descriptor contains Inf"

    # All-node descriptors
    node_descs = P.compute_all_node_descriptors(minutiae, adjacency, weight_dict)
    assert node_descs.shape == (len(minutiae), size), "node_descs shape mismatch"

    # Global descriptor
    global_desc = P.fingerprint_global_descriptor(node_descs)
    assert global_desc.shape == (2 * size,), "Global descriptor shape mismatch"

    log.info(f"    {k}-hop subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} edges")
    log.info(f"    Descriptor dim: {size}  |  global dim: {2*size}")
    log.info(f"    Node descs shape: {node_descs.shape}")
    return PASS, {"node_descs": node_descs, "global_desc": global_desc}


def test_phase4_matching(global_descs, labels):
    log.info("  [Test] Phase 4 – KD-Tree Indexing & Fast Matching …")

    t0      = time.time()
    kdtree, lbs = P.build_kd_index(global_descs, labels)
    t_build = time.time() - t0

    # Query each descriptor against the tree
    t0 = time.time()
    for desc, label in zip(global_descs, labels):
        results = P.query_kd_index(desc, kdtree, lbs, k=3)
        assert len(results) > 0, "KD-tree returned no results"
        # Top-1 should be itself (distance ≈ 0)
        assert results[0][0] < 1e-3, f"Self-match distance too large: {results[0][0]:.4f}"
    t_query = time.time() - t0

    log.info(f"    Build time : {t_build*1000:.1f} ms")
    log.info(f"    Query time : {t_query*1000:.1f} ms  ({len(global_descs)} queries)")
    log.info(f"    Per-query  : {t_query*1000/len(global_descs):.2f} ms")
    return PASS


def test_evaluation(dataset: dict):
    log.info("  [Test] Evaluation – FAR / FRR / EER …")

    # Process all images
    by_subject = {}
    for subj, paths in dataset["DB1_B"].items():
        descs = []
        for path in paths:
            res = P.process_fingerprint(path)
            if res:
                descs.append(res["global_desc"])
        if descs:
            by_subject[subj] = descs

    if len(by_subject) < 2:
        log.warning("    Not enough subjects for evaluation, skipping.")
        return PASS, {}

    metrics = P.evaluate_verification(by_subject)

    assert 0 <= metrics["eer"] <= 1,   "EER must be in [0,1]"
    assert 0 <= metrics["auc"] <= 1,   "AUC must be in [0,1]"
    assert metrics["n_genuine_pairs"]  > 0, "Must have genuine pairs"
    assert metrics["n_impostor_pairs"] > 0, "Must have impostor pairs"

    log.info(f"    EER  : {metrics['eer']:.2%}")
    log.info(f"    AUC  : {metrics['auc']:.4f}")
    log.info(f"    Genuine  pairs : {metrics['n_genuine_pairs']}")
    log.info(f"    Impostor pairs : {metrics['n_impostor_pairs']}")
    return PASS, metrics, by_subject


def test_visualisations(result: dict, metrics: dict):
    log.info("  [Test] Visualisations – saving all 4 phase images …")
    try:
        P.visualise_phase1_1(result, tag="synth_test")
        P.visualise_phase1_2(result, tag="synth_test")
        P.visualise_phase2(result,   tag="synth_test")
        P.visualise_phase3(result,   tag="synth_test")
        if metrics:
            P.visualise_phase4(metrics, "SyntheticDB", tag="synth_test")
    except Exception as e:
        return f"{FAIL}: {e}"

    vis_files = list(Path(P.CFG["vis_dir"]).glob("synth_test_*.png"))
    assert len(vis_files) >= 4, f"Expected ≥4 vis files, got {len(vis_files)}"
    log.info(f"    Saved {len(vis_files)} visualisation images")
    return PASS


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("\n" + "="*60)
    print("  TRC-SD Synthetic Validation Test")
    print("="*60)

    tmp = "./outputs/trcsd/synthetic_dataset"
    log.info("\n[Setup]  Generating synthetic dataset …")
    dataset = make_dataset(tmp, n_subjects=6, n_impressions=5)
    log.info(f"  Created {sum(len(v) for v in dataset['DB1_B'].values())} synthetic images")

    results_table = []

    # ── Phase 1.1 ──────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    log.info("\n[Phase 1.1]  Preprocessing & Minutiae Extraction")
    img = cv2.imread(dataset["DB1_B"][1][0], cv2.IMREAD_GRAYSCALE)
    status_11, prep_data = test_phase1_preprocessing(img)
    results_table.append(("Phase 1.1 – Preprocessing & Minutiae", status_11))

    # ── Phase 1.2 ──────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    log.info("\n[Phase 1.2]  Delaunay Graph Construction")
    status_12, graph_data = test_phase1_delaunay(prep_data["minutiae"])
    results_table.append(("Phase 1.2 – Delaunay Triangulation", status_12))

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    log.info("\n[Phase 2]   Ridge-Count Edge Weighting")
    status_2, ridge_data = test_phase2_ridge_weights(
        graph_data["edges"], prep_data["minutiae"], prep_data["binary"])
    results_table.append(("Phase 2   – Ridge-Count Weighting", status_2))

    # ── Phase 3 ───────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    log.info("\n[Phase 3]   Spectral Descriptors")
    status_3, spec_data = test_phase3_spectral(
        prep_data["minutiae"], graph_data["adjacency"], ridge_data["weight_dict"])
    results_table.append(("Phase 3   – Spectral Descriptors", status_3))

    # ── Phase 4 ───────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    log.info("\n[Phase 4]   KD-Tree Indexing & Matching")

    # Build multiple global descs to test matching properly
    all_descs, all_labels = [], []
    for subj, paths in dataset["DB1_B"].items():
        for path in paths:
            res = P.process_fingerprint(path)
            if res:
                all_descs.append(res["global_desc"])
                all_labels.append(f"{subj}_{Path(path).stem}")

    status_4 = test_phase4_matching(all_descs, all_labels)
    results_table.append(("Phase 4   – KD-Tree Matching", status_4))

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    log.info("\n[Evaluation]  FAR / FRR / EER")
    eval_out     = test_evaluation(dataset)
    status_eval  = eval_out[0]
    metrics      = eval_out[1] if len(eval_out) > 1 else {}
    by_subject   = eval_out[2] if len(eval_out) > 2 else {}
    results_table.append(("Evaluation – FAR/FRR/EER", status_eval))

    # ── Visualisations ────────────────────────────────────────────────────
    print("\n" + "-"*60)
    log.info("\n[Vis]       Saving intermediate step images")
    # Build a full result dict for vis
    full_result = {
        "prep"        : prep_data,
        "edges"       : graph_data["edges"],
        "adjacency"   : graph_data["adjacency"],
        "weights"     : ridge_data["weights"],
        "weight_dict" : ridge_data["weight_dict"],
        "node_descs"  : spec_data["node_descs"],
        "global_desc" : spec_data["global_desc"],
    }
    status_vis = test_visualisations(full_result, metrics)
    results_table.append(("Visualisations", status_vis))

    # ── Summary ───────────────────────────────────────────────────────────
    t_total = time.time() - t0
    n_pass = sum(1 for _, s in results_table if PASS in str(s))
    n_fail = len(results_table) - n_pass

    print("\n" + "="*60)
    print("  VALIDATION TEST RESULTS")
    print("="*60)
    for name, status in results_table:
        mark = "✓" if PASS in str(status) else "✗"
        print(f"  {mark}  {name:<45}  {status}")
    print("-"*60)
    print(f"  Passed: {n_pass} / {len(results_table)}   "
          f"Failed: {n_fail}   "
          f"Time: {t_total:.1f}s")
    print("="*60 + "\n")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
