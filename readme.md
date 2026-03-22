# TRC-SD: Topological Ridge-Count Graphs with Spectral Descriptors

A fingerprint recognition pipeline developed for **DSM 410 – Computer Vision** at IIT Indore.
Evaluated on the FVC2002/FVC2004 benchmark datasets.

**Authors:** Bharath AS & Yash Vinayvanshi (IIT Indore)

Dataset: http://bias.csr.unibo.it/fvc2004/databases.asp

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Requirements & Installation](#requirements--installation)
4. [How to Run](#how-to-run)
5. [Pipeline Architecture](#pipeline-architecture)
6. [File Descriptions](#file-descriptions)
7. [Function Reference — pipeline.py](#function-reference--pipelinepy)
8. [Function Reference — test.py](#function-reference--testpy)
9. [Outputs](#outputs)
10. [Configuration Reference](#configuration-reference)
11. [Evaluation Metrics](#evaluation-metrics)

---

## Overview

TRC-SD is a graph-based fingerprint recognition system.
Each fingerprint is modeled as a **weighted graph** where:

- **Nodes** = minutiae points (ridge endings and bifurcations)
- **Edges** = Delaunay triangulation connections between minutiae
- **Edge weights** = number of ridges crossed when scanning between two minutiae

Each node is described by a **42-dimensional spectral descriptor** derived from the eigenvalue spectrum of the local k-hop subgraph Laplacian.
A **global ~450-dimensional descriptor** is then assembled from the node descriptors plus orientation field, ridge frequency, spatial density, and triplet features, and compressed to 64D via PCA.

Matching uses a combined score: 60% global descriptor cosine similarity + 40% local descriptor matching (Lowe ratio test).

---

## Directory Structure

```
fingerprint/
│
├── pipeline.py              # Main pipeline — all phases, matching, evaluation
├── test.py                  # Synthetic validation test suite
├── requirements.txt         # Python package dependencies
├── readme.md                # This file
│
├── Datasets/
│   └── FVC2004/             # (or FVC2002/) — fingerprint benchmark datasets
│       ├── DB1_B/           # Database 1 — 10 subjects × 8 impressions = 80 images
│       │   ├── 101_1.tif    # Naming: {subject}_{impression}.tif
│       │   ├── 101_2.tif
│       │   └── ...
│       ├── DB2_B/
│       ├── DB3_B/
│       └── DB4_B/
│
├── outputs/
│   └── trcsd/
│       ├── pipeline.log     # Full run log with per-DB metrics
│       └── sample_vis/      # Visualisation images saved during a run
│           ├── DB1_B_phase1_1_preprocessing.png
│           ├── DB1_B_phase1_2_delaunay.png
│           ├── DB1_B_phase2_ridge_weights.png
│           ├── DB1_B_phase3_spectral.png
│           └── DB*_B_phase4_evaluation.png
│
└── debug/                   # Manual debug images (intermediate steps)
    ├── 1_original.png
    ├── 2_roi_mask.png
    ├── 2_skeleton.png
    ├── 3_delaunay.png
    └── 4_delaunay.png
```

---

## Requirements & Installation

**Python:** 3.10+

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24
opencv-python>=4.8
matplotlib>=3.7
networkx>=3.1
scipy>=1.10
scikit-learn>=1.3
scikit-image>=0.21
tqdm>=4.65
```

> `scikit-learn` is optional — the pipeline will skip PCA dimensionality reduction if it is not installed, but this will noticeably hurt matching performance.

---

## How to Run

### 1. Run the full pipeline on FVC2002/FVC2004 data

```bash
python pipeline.py
```

Before running, update the `dataset_dir` in `CFG` at the top of `pipeline.py` to point to your dataset folder:

```python
CFG = {
    "dataset_dir": "./Datasets/FVC2004",   # ← change this
    ...
}
```

The dataset folder must contain `DB1_B`, `DB2_B`, `DB3_B`, `DB4_B` subdirectories, each holding images named `{subject}_{impression}.bmp` (or `.tif`).

**Expected output:**
- Per-database metrics printed to stdout and saved to `outputs/trcsd/pipeline.log`
- Phase visualisations saved to `outputs/trcsd/sample_vis/`

### 2. Run the synthetic validation test

```bash
python test.py
```

This generates synthetic concentric-arc fingerprint images and runs every pipeline phase on them.
Use this to confirm the environment is set up correctly before running on real data.

**Expected output:**
```
  ✓  Phase 1.1 – Preprocessing & Minutiae       ✓ PASS
  ✓  Phase 1.2 – Delaunay Triangulation         ✓ PASS
  ✓  Phase 2   – Ridge-Count Weighting          ✓ PASS
  ✓  Phase 3   – Spectral Descriptors           ✓ PASS
  ✓  Phase 4   – KD-Tree Matching               ✓ PASS
  ✓  Evaluation – FAR/FRR/EER                   ✓ PASS
  ✓  Visualisations                             ✓ PASS
```

---

## Pipeline Architecture

```
Input Image (.bmp / .tif)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  PHASE 1.1 – PREPROCESSING & MINUTIAE           │
│  • Foreground mask (local variance)             │
│  • Centroid-center alignment                    │
│  • Gabor filter bank enhancement (8 directions) │
│  • CLAHE + adaptive binarization                │
│  • Skeletonization (Zhang-Suen)                 │
│  • Crossing Number (CN) minutiae extraction     │
│  • Minutiae orientation estimation              │
│  • Rotation normalization (canonical frame)     │
└────────────────────┬────────────────────────────┘
                     │  minutiae (N × 2), orientations (N,)
                     ▼
┌─────────────────────────────────────────────────┐
│  PHASE 1.2 – DELAUNAY GRAPH CONSTRUCTION        │
│  • Delaunay triangulation of minutiae           │
│  • Prune edges longer than max_edge_px=150 px   │
└────────────────────┬────────────────────────────┘
                     │  edges (E × 2), adjacency dict
                     ▼
┌─────────────────────────────────────────────────┐
│  PHASE 2 – RIDGE-COUNT EDGE WEIGHTING           │
│  • Bresenham line scan between minutiae pairs   │
│  • Count 0→1 transitions in smoothed binary     │
└────────────────────┬────────────────────────────┘
                     │  weights (E,), weight_dict
                     ▼
┌─────────────────────────────────────────────────┐
│  PHASE 3 – SPECTRAL DESCRIPTORS                 │
│  3.1  k-hop BFS subgraph (k=3) per node         │
│  3.2  Normalized Laplacian eigenvalues → 32D    │
│  3.3  + ridge-count histogram 8D                │
│       + orientation (sin/cos) 2D                │
│       → 42D node descriptor per minutia         │
│                                                 │
│  GLOBAL DESCRIPTOR (~450D → 64D via PCA):       │
│  • Global Laplacian spectrum (50D)              │
│  • Ridge histogram (13D)                        │
│  • Degree distribution stats (8D)               │
│  • Node descriptor percentiles (126D)           │
│  • Orientation field × 0.3 weight (~192D)       │
│  • Spatial density map (48D)                    │
│  • Ridge frequency features (~64D)              │
│  • Orientation histogram (24D)                  │
│  • Triplet descriptor (28D)                     │
│  • Edge orientation histogram (18D)             │
│  ─────────────────────────────────────          │
│  PCA: ~450D → 64D (sklearn, per database)       │
└────────────────────┬────────────────────────────┘
                     │  global_desc (64D), node_descs (N×42)
                     ▼
┌─────────────────────────────────────────────────┐
│  PHASE 4 – KD-TREE MATCHING & EVALUATION        │
│  • KD-tree on L2-normalized gallery descriptors │
│  • Pair score = 0.6 × global² + 0.4 × local    │
│    (global: cosine²; local: Lowe ratio test)    │
│  • Verification: FAR/FRR sweep → EER, AUC       │
│  • Identification: Rank-1, Rank-5               │
└─────────────────────────────────────────────────┘
```

---

## File Descriptions

### `pipeline.py`
The complete TRC-SD pipeline (~2500 lines).
Contains every phase from raw image loading to final metric computation and visualisation.
Entry point: `main()`.

### `test.py`
A self-contained validation suite that:
1. Generates synthetic concentric-arc fingerprint images (no real dataset needed).
2. Runs each pipeline phase individually with assertion-based tests.
3. Reports pass/fail for all 7 test cases.

Use this to verify the environment before running on real data.

### `requirements.txt`
Python package dependencies with minimum version pins.

---

## Function Reference — pipeline.py

### Phase 0 – Data Loading

| Function | Description |
|---|---|
| `load_dataset(dataset_dir)` | Scans a directory for `DB*_B` subdirectories. Parses FVC-style filenames (`{subject}_{impression}.ext`) into a dict `{db_name: {subject_id: [paths]}}`. |

---

### Phase 1.1 – Preprocessing & Minutiae

| Function | Description |
|---|---|
| `compute_foreground_mask(gray)` | Computes a binary foreground mask using local variance (Otsu on variance map). Morphological close+open to fill holes and remove noise. |
| `center_on_foreground(gray, mask)` | Translates the image so the foreground centroid aligns with the image centre. Corrects placement shifts between impressions (~10-40 px). |
| `enhance_with_gabor(gray)` | Applies a bank of 8-orientation Gabor filters (freq=0.08). Inverts input, applies CLAHE, takes signed max response to preserve thin ridge-furrow structure. |
| `binarize(enhanced, mask)` | Adaptive thresholding (blockSize=25, C=8) on the Gabor-enhanced image. Applies foreground mask to suppress background. Returns a bool array. |
| `compute_skeleton(binary)` | Converts the binary ridge image to a 1-pixel-wide skeleton using `skimage.morphology.skeletonize`. Returns uint8. |
| `_crossing_number(neighborhood)` | Computes the Crossing Number (CN) for a skeleton pixel from its 8-connected neighbourhood. CN=1 → ridge ending; CN=3 → bifurcation. Internal helper. |
| `_get_8_neighbors_clockwise(skeleton, y, x)` | Returns the 8 clockwise neighbours of a skeleton pixel. Internal helper. |
| `_nms_minutiae(coords, min_dist)` | Non-maximum suppression: removes minutiae closer than `min_dist` pixels to each other, keeping one per cluster. |
| `extract_minutiae(skeleton)` | Finds all ridge endings (CN=1) and bifurcations (CN=3) in the skeleton. Applies border margin exclusion, NMS, and caps at `max_minutiae=120`. Returns `(coords (N,2), types (N,) with 'E'/'B')`. |
| `compute_minutiae_orientations(skeleton, minutiae)` | Estimates ridge orientation at each minutia by sampling skeleton branches and computing the double-angle circular mean. Returns (N,) angles in radians. |
| `compute_reference_angle(gray, mask)` | Computes the dominant ridge orientation from the foreground using Sobel gradients and circular mean. Used as the reference for rotation normalisation. |
| `normalize_fingerprint_orientation(gray, mask, skeleton, minutiae, orientations)` | Rotates the image, mask, skeleton, minutiae, and orientations by the negative reference angle to bring the fingerprint to a canonical orientation frame. Improves cross-impression consistency. |
| `preprocess_fingerprint(img_path)` | End-to-end Phase 1.1 runner for one image. Calls all above functions in sequence. Returns a dict with keys `gray, enhanced, binary, skeleton, minutiae, types, orientations, mask`, or `None` if too few minutiae are found. |

---

### Phase 1.2 – Delaunay Graph

| Function | Description |
|---|---|
| `build_delaunay_graph(minutiae)` | Computes Delaunay triangulation of minutiae coordinates. Prunes edges longer than `max_edge_px=150` px. Returns `(edges (E,2), adjacency dict)`. Falls back to all-pairs if fewer than 4 minutiae. |

---

### Phase 2 – Ridge-Count Edge Weighting

| Function | Description |
|---|---|
| `_bresenham_line(y0, x0, y1, x1)` | Returns all integer pixel coordinates on the line between two points using Bresenham's algorithm. Internal helper. |
| `count_ridges_on_line(binary, p1, p2)` | Counts the number of 0→1 transitions (ridges crossed) along the Bresenham line between two points in the binary image. Clamps result to [1, 30]. |
| `compute_ridge_weights(edges, minutiae, binary)` | Calls `count_ridges_on_line` for every edge. Returns `(weights (E,), weight_dict {(i,j): w})`. |

---

### Phase 3 – Spectral Descriptors

| Function | Description |
|---|---|
| `extract_k_hop_subgraph(center, adjacency, k)` | BFS from `center` node up to depth `k=3`. Returns `(sub_nodes list, sub_edges list)`. |
| `compute_spectral_descriptor(sub_nodes, sub_edges, weight_dict, size)` | Builds the weighted normalized Laplacian `L_sym = I − D^{-½}AD^{-½}` for the subgraph. Computes eigenvalues and interpolates to a fixed-length `size=32` vector. |
| `compute_ridge_count_node_descriptor(node, adjacency, weight_dict)` | Builds an 8-bin histogram of ridge counts from the node to its neighbours (bins: 1,2,3,4,5,6,7,8+). |
| `compute_all_node_descriptors(minutiae, adjacency, weight_dict, orientations)` | For each minutia: concatenates 32D spectral + 8D ridge-count + 2D orientation (sin/cos relative to mean). Returns `(N, 42)` matrix. |
| `compute_orientation_field(gray, mask)` | Divides the foreground into 16×16 px blocks, computes coherent ridge orientation per block via Sobel gradients. Returns a flattened sin/cos orientation vector. |
| `compute_minutiae_spatial_density(minutiae, H, W)` | Divides the image into a grid of cells, counts minutiae per cell, and L1-normalises. Returns a spatial density histogram (48D). |
| `compute_ridge_frequency_features(gray, mask)` | Estimates local ridge frequency by measuring ridge spacing in blocks. Returns a flattened frequency feature vector (~64D). |
| `compute_orientation_histogram(gray, mask)` | Computes a global histogram of dominant ridge orientations (24 bins over 0–π). Returns a 24D normalised histogram. |
| `compute_minutiae_triplet_descriptor(minutiae, orientations)` | Randomly samples triplets of minutiae and computes relative angle/distance ratios for rotation and scale invariance. Returns a 28D descriptor. |
| `compute_edge_orientation_histogram(edges, minutiae)` | Histograms the angle of each Delaunay edge relative to the horizontal. Returns an 18D normalised histogram. |
| `match_local_descriptors_score(descs1, descs2, ratio_thresh)` | Mutual nearest-neighbour matching of two sets of 42D node descriptors via KD-tree. Lowe ratio test at `ratio_thresh=0.85`. Returns a match score in [0, 1]. |
| `compute_global_laplacian_spectrum(edges, n_nodes, weight_dict)` | Builds the full-graph weighted Laplacian and returns its 50 smallest eigenvalues (padded or truncated). |
| `fingerprint_global_descriptor(node_descs, edges, n_nodes, weight_dict, weights, adjacency, gray, mask, minutiae, H, W, orientations)` | Assembles the ~450D global fingerprint descriptor from all sub-descriptors. Down-weights the orientation field component by 0.3. Returns a float64 vector. |

---

### Full-Image Processing

| Function | Description |
|---|---|
| `process_fingerprint(img_path)` | Runs all phases (1.1 → 1.2 → 2 → 3) on a single image. Applies median blur to binary before ridge counting. Returns a result dict or `None` on failure. |

---

### Phase 4 – KD-Tree Matching

| Function | Description |
|---|---|
| `build_kd_index(global_descs, labels)` | L2-normalises the descriptor matrix row-wise and builds a `scipy.spatial.KDTree`. Returns `(kdtree, labels, None, None)`. |
| `query_kd_index(query_desc, kdtree, labels, k)` | Queries the KD-tree for the k nearest gallery descriptors. Returns sorted `(distance, label)` tuples. |
| `cosine_similarity(a, b)` | Cosine similarity between two vectors. Kept for API compatibility; prefer `fingerprint_pair_score` for evaluation. |
| `fingerprint_pair_score(result_a, result_b, alpha)` | Combined match score: `alpha × max(0, cosine)² + (1−alpha) × local_score`. Default alpha=0.6. Squaring sharpens genuine/impostor contrast. |

---

### Evaluation

| Function | Description |
|---|---|
| `evaluate_verification(results_by_subject)` | Computes genuine scores (all same-subject impression pairs) and impostor scores (ALL cross-subject pairs). Sweeps threshold to compute FAR, FRR, EER, AUC, FNMR@0.1%FAR. |
| `evaluate_identification(gallery_labels, gallery_descs, query_labels, query_descs)` | KD-tree based 1:N identification. Computes Rank-1 and Rank-5 accuracy. |
| `evaluate_identification_pairwise(gallery_results, gallery_labels, query_results, query_labels)` | Identification using full `fingerprint_pair_score` for all gallery-query pairs (more accurate). Computes Rank-1 and Rank-5. |

---

### Visualisation

| Function | Description |
|---|---|
| `_save(fig, name)` | Saves a matplotlib figure to `vis_dir/name.png` and closes it. |
| `visualise_phase1_1(result, tag)` | 5-panel figure: raw image, foreground mask, Gabor-enhanced, binary, skeleton with minutiae overlaid. |
| `visualise_phase1_2(result, tag)` | Delaunay graph overlaid on skeleton. Nodes coloured by type (endings=blue, bifurcations=red). |
| `visualise_phase2(result, tag)` | Delaunay graph with edge colours encoding ridge-count weight (colormap) + ridge-count histogram. |
| `visualise_phase3(result, tag, node_example)` | 4-panel figure: node descriptor heatmap, global descriptor bar chart, k-hop subgraph for an example node, eigenvalue spectrum. |
| `visualise_phase4(metrics, db_name, tag)` | 3-panel figure: ROC curve with AUC, FAR/FRR vs threshold with EER marked, score distribution histogram. |

---

### Utilities

| Function | Description |
|---|---|
| `_progress(iterable, desc, total)` | Uses `tqdm` if available, else prints a simple ASCII progress bar. |
| `main()` | Entry point. Loads dataset, processes all images, applies per-database PCA, runs verification and identification evaluation, prints a summary table. |

---

## Function Reference — test.py

| Function | Description |
|---|---|
| `make_synthetic_fingerprint(H, W, n_arcs, noise, seed)` | Generates a synthetic fingerprint as a noisy concentric-arc grayscale image. |
| `make_dataset(tmp_dir, n_subjects, n_impressions, seed)` | Creates a dataset of synthetic fingerprints on disk in FVC-style naming. Applies random affine transforms per impression to simulate real variability. Returns a dataset dict. |
| `test_phase1_preprocessing(img)` | Calls enhancement, binarization, skeletonization, and minutiae extraction. Asserts output shapes and types. |
| `test_phase1_delaunay(minutiae)` | Calls `build_delaunay_graph` and asserts edge and adjacency structure. |
| `test_phase2_ridge_weights(edges, minutiae, binary)` | Calls `compute_ridge_weights` and asserts weight count and minimum values ≥ 1. |
| `test_phase3_spectral(minutiae, adjacency, weight_dict)` | Tests subgraph extraction, spectral descriptor computation, all-node descriptors, and global descriptor. Asserts shapes and absence of NaN/Inf. |
| `test_phase4_matching(global_descs, labels)` | Builds a KD-tree index and queries every descriptor. Asserts self-match distance ≈ 0. Reports build and per-query timing. |
| `test_evaluation(dataset)` | Processes all synthetic images and runs verification evaluation. Asserts EER ∈ [0,1] and non-zero pair counts. |
| `test_visualisations(result, metrics)` | Calls all 5 visualisation functions and asserts that ≥ 4 PNG files were written to `vis_dir`. |
| `main()` | Runs all 7 tests in order. Prints a summary table and exits with code 1 if any test fails. |

---

## Outputs

| File / Folder | Contents |
|---|---|
| `outputs/trcsd/pipeline.log` | Full run log with per-DB metrics (EER, AUC, Rank-1, Rank-5), PCA dimensionality info, timing, and skipped image counts. |
| `outputs/trcsd/sample_vis/DB*_B_phase1_1_preprocessing.png` | 5-panel: raw → mask → Gabor → binary → skeleton+minutiae. |
| `outputs/trcsd/sample_vis/DB*_B_phase1_2_delaunay.png` | Delaunay graph overlaid on the skeleton. |
| `outputs/trcsd/sample_vis/DB*_B_phase2_ridge_weights.png` | Edge-coloured Delaunay graph + ridge-count histogram. |
| `outputs/trcsd/sample_vis/DB*_B_phase3_spectral.png` | Descriptor heatmap, global vector, k-hop subgraph, eigenvalue spectrum. |
| `outputs/trcsd/sample_vis/DB*_B_phase4_evaluation.png` | ROC curve, FAR/FRR vs threshold, score distribution. |

---

## Configuration Reference

All parameters live in the `CFG` dict at the top of `pipeline.py`:

| Key | Default | Description |
|---|---|---|
| `dataset_dir` | `"./datasets/PAMI Lab"` | Root folder containing `DB*_B` subdirectories |
| `output_dir` | `"./outputs/trcsd"` | Directory for log file |
| `vis_dir` | `"./outputs/trcsd/sample_vis"` | Directory for visualisation PNG files |
| `gabor_orientations` | `8` | Number of Gabor filter orientations |
| `gabor_frequency` | `0.08` | Ridge frequency in cycles/pixel (~12 px ridge spacing) |
| `gabor_sigma_x` | `3.0` | Gabor sigma along ridge direction |
| `gabor_sigma_y` | `6.0` | Gabor sigma perpendicular to ridge |
| `border_margin` | `35` | Pixels from image border excluded from minutiae |
| `min_minutiae_dist` | `12` | Minimum pixel distance between two minutiae |
| `min_minutiae` | `8` | Skip image if fewer minutiae found after filtering |
| `max_minutiae` | `120` | Cap on minutiae count (takes strongest 120) |
| `max_edge_px` | `150` | Prune Delaunay edges longer than this (pixels) |
| `k_hop` | `3` | BFS radius for local subgraph extraction |
| `descriptor_size` | `32` | Fixed-length spectral eigenvalue vector size per node |
| `k_nn` | `5` | KD-tree top-k candidates for identification |
| `n_thresh` | `300` | Number of FAR/FRR threshold steps for EER sweep |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **EER** | Equal Error Rate — threshold where FAR = FRR. Lower is better. Target: < 10%. |
| **AUC** | Area Under the ROC Curve. Higher is better. Target: > 0.85. |
| **FNMR@0.1%FAR** | False Non-Match Rate at 0.1% False Accept Rate. Lower is better. |
| **Rank-1** | Fraction of queries where the top-1 gallery match is the correct subject. Target: > 80%. |
| **Rank-5** | Fraction of queries where the correct subject appears in the top-5 gallery matches. |

**Genuine pairs:** All pairwise combinations of impressions from the same subject.
**Impostor pairs:** All pairwise combinations across different subjects — C(N_subjects, 2) × impressions².
