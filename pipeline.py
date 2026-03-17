#!/usr/bin/env python3
"""
================================================================
TRC-SD: Topological Ridge-Count Graphs with Spectral Descriptors
Fingerprint Recognition Pipeline — DSM 410 Computer Vision
================================================================
Authors  : Bharath AS & Yash Vinayvanshi (IIT Indore)
Course   : DSM 410 – Computer Vision

Pipeline Phases:
    Phase 1.1  →  Preprocessing & Minutiae Extraction
                  (Gabor enhancement → Binarise → Skeletonise → CN method)
    Phase 1.2  →  Delaunay Triangulation (Graph Construction)
    Phase 2    →  Ridge-Count Edge Weighting
                  (Bresenham line scan on binary image)
    Phase 3.1  →  Local k-Hop Subgraph Extraction (BFS)
    Phase 3.2  →  Spectral Descriptors (Laplacian eigenvalues, fixed-length)
    Phase 4    →  KD-Tree Indexing & Fast 1:N Matching
    Evaluation →  FAR / FRR / EER & Identification Rank-1

Requirements:
    pip install numpy opencv-python scipy scikit-image matplotlib tqdm
================================================================
"""

# ─────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import warnings
from pathlib import Path
from collections import deque, defaultdict
from itertools import combinations

import numpy as np
import cv2
from scipy.spatial import Delaunay, KDTree
from scipy.linalg import eigvalsh
from skimage.morphology import skeletonize as ski_skeletonize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# optional tqdm — falls back to a plain iterator if not installed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None, **kwargs):          # noqa: E301
        label = f"[{desc}] " if desc else ""
        items = list(iterable)
        n = len(items)
        for i, item in enumerate(items, 1):
            pct = 100 * i // n
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  {label}|{bar}| {i}/{n}", end="", flush=True)
            yield item
        print()


# ─────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────
CFG = {
    # Paths  ──────────────────────────────────────────────────
    "dataset_dir"    : "./datasets/PAMI Lab",
    "output_dir"     : "./outputs/trcsd",
    "vis_dir"        : "./outputs/trcsd/sample_vis",

    # Phase 1.1 – Preprocessing / Minutiae ───────────────────
    "gabor_orientations": 8,        # number of Gabor filter orientations
    "gabor_frequency"   : 0.12,     # ridge frequency (cycles/pixel)
    "gabor_sigma_x"     : 4.0,      # Gabor sigma along ridge direction
    "gabor_sigma_y"     : 8.0,      # Gabor sigma perpendicular to ridge
    "border_margin"     : 20,       # pixels from image border to exclude
    "min_minutiae_dist" : 12,       # min pixel distance between two minutiae
    "min_minutiae"      : 8,        # skip image if fewer minutiae found
    "max_minutiae"      : 120,      # cap; take strongest 120 if more found

    # Phase 1.2 – Delaunay ───────────────────────────────────
    "max_edge_px"       : 150,      # prune Delaunay edges longer than this

    # Phase 3 – Spectral ─────────────────────────────────────
    "k_hop"             : 3,        # k-hop neighbourhood radius
    "descriptor_size"   : 20,       # fixed eigenvalue vector length

    # Phase 4 – Matching ─────────────────────────────────────
    "k_nn"              : 5,        # KD-tree top-k candidates

    # Evaluation ─────────────────────────────────────────────
    "n_thresh"          : 300,      # number of FAR/FRR threshold steps
}


# ─────────────────────────────────────────────────────────────
# 2.  LOGGING
# ─────────────────────────────────────────────────────────────
os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs(CFG["vis_dir"],    exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{CFG['output_dir']}/pipeline.log", mode="w"),
    ],
)
log = logging.getLogger("TRC-SD")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 – DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_dir: str) -> dict:
    """
    Scan FVC2002 DB*_B folders and collect image paths.

    Input
    -----
    dataset_dir : str   Root folder containing DB1_B, DB2_B, DB3_B, DB4_B.

    Output
    ------
    dataset : dict
        {
          "DB1_B": {
              1: ["path/1_1.bmp", "path/1_2.bmp", …],   # subject 1
              2: […],
              …
          },
          "DB2_B": {…},
          …
        }
    """
    dataset = {}
    root = Path(dataset_dir)

    if not root.exists():
        log.error(f"Dataset root not found: {root.resolve()}")
        sys.exit(1)

    db_folders = sorted([d for d in root.iterdir() if d.is_dir() and "DB" in d.name])
    if not db_folders:
        log.error(f"No DB*_B folders found inside {root.resolve()}")
        sys.exit(1)

    for db_dir in db_folders:
        db_name = db_dir.name
        exts    = ["*.bmp", "*.BMP", "*.tif", "*.TIF", "*.png", "*.PNG"]
        images  = []
        for ext in exts:
            images.extend(db_dir.glob(ext))

        subject_map = defaultdict(list)
        for img_path in sorted(images):
            # FVC2002 naming: {subject}_{impression}.ext  e.g. 1_1.bmp
            stem = img_path.stem
            parts = stem.split("_")
            if len(parts) >= 2:
                try:
                    subject_id = int(parts[0])
                    subject_map[subject_id].append(str(img_path))
                except ValueError:
                    pass

        if subject_map:
            dataset[db_name] = dict(subject_map)
            n_imgs = sum(len(v) for v in subject_map.values())
            log.info(f"  {db_name}: {len(subject_map)} subjects, {n_imgs} images")
        else:
            log.warning(f"  {db_name}: no parseable images found")

    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1.1 – PREPROCESSING & MINUTIAE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def enhance_with_gabor(gray: np.ndarray) -> np.ndarray:
    """
    Enhance fingerprint ridges with a bank of Gabor filters.

    Input
    -----
    gray : (H, W) uint8   Raw grayscale fingerprint image.

    Output
    ------
    enhanced : (H, W) uint8   Ridge-enhanced image (higher contrast ridges).
    """
    n_orient = CFG["gabor_orientations"]
    ksize    = 31                          # kernel size (odd)
    freq     = CFG["gabor_frequency"]
    sx       = CFG["gabor_sigma_x"]
    sy       = CFG["gabor_sigma_y"]

    # Normalise input
    norm = cv2.normalize(gray.astype(np.float32), None, 0, 1,
                         cv2.NORM_MINMAX)

    response = np.zeros_like(norm)
    for i in range(n_orient):
        theta = (i / n_orient) * np.pi
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sx, theta,
            1.0 / freq, sy / sx, 0, ktype=cv2.CV_32F
        )
        kernel /= (kernel.sum() + 1e-6)
        filt = cv2.filter2D(norm, cv2.CV_32F, kernel)
        response = np.maximum(response, np.abs(filt))

    enhanced = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)


def binarize(enhanced: np.ndarray) -> np.ndarray:
    """
    Binarize the enhanced image using adaptive thresholding.

    Input
    -----
    enhanced : (H, W) uint8   Gabor-enhanced fingerprint.

    Output
    ------
    binary : (H, W) bool   True = ridge pixel.
    """
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological clean-up: close small holes, remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    return binary.astype(bool)


def compute_skeleton(binary: np.ndarray) -> np.ndarray:
    """
    Produce a 1-pixel-wide skeleton of the ridge pattern.

    Input
    -----
    binary : (H, W) bool   Binarized fingerprint.

    Output
    ------
    skeleton : (H, W) uint8   255 = skeleton pixel, 0 = background.
    """
    skeleton = ski_skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)


def _crossing_number(neighborhood: list) -> int:
    """
    Compute the crossing number (CN) for an 8-connected neighbourhood.

    neighborhood is the 8 binary pixel values in clockwise order starting
    from the top neighbour.

    CN == 1 → ridge ending
    CN == 3 → bifurcation
    """
    cn = 0
    n = len(neighborhood)
    for i in range(n):
        cn += abs(int(neighborhood[i]) - int(neighborhood[(i + 1) % n]))
    return cn // 2


def _get_8_neighbors_clockwise(skeleton: np.ndarray, y: int, x: int) -> list:
    """Return binary values of 8 neighbours in clockwise order."""
    h, w = skeleton.shape
    # Clockwise: top, top-right, right, bottom-right, bottom, bottom-left, left, top-left
    dy = [-1, -1,  0,  1, 1,  1,  0, -1]
    dx = [ 0,  1,  1,  1, 0, -1, -1, -1]
    vals = []
    for ddy, ddx in zip(dy, dx):
        ny, nx = y + ddy, x + ddx
        if 0 <= ny < h and 0 <= nx < w:
            vals.append(1 if skeleton[ny, nx] > 0 else 0)
        else:
            vals.append(0)
    return vals


def _nms_minutiae(coords: list, min_dist: int) -> list:
    """
    Non-maximum suppression: remove minutiae closer than min_dist to another.

    Input
    -----
    coords   : list of (y, x) tuples
    min_dist : int   minimum allowed inter-minutia distance in pixels

    Output
    ------
    kept : list of (y, x) kept after suppression
    """
    if not coords:
        return []
    pts   = np.array(coords, dtype=np.float32)
    kept  = []
    used  = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        kept.append(tuple(pts[i].astype(int)))
        dists = np.linalg.norm(pts - pts[i], axis=1)
        too_close = (dists < min_dist) & (~used)
        used[too_close] = True
    return kept


def extract_minutiae(skeleton: np.ndarray) -> tuple:
    """
    Extract ridge endings and bifurcations using the Crossing Number (CN) method.

    Input
    -----
    skeleton : (H, W) uint8   1-pixel-wide skeleton (255 = ridge, 0 = bg).

    Output
    ------
    minutiae       : (N, 2) int ndarray   [y, x] coordinates.
    minutiae_types : (N,) str ndarray     'E' (ending) or 'B' (bifurcation).
    """
    h, w     = skeleton.shape
    margin   = CFG["border_margin"]
    min_dist = CFG["min_minutiae_dist"]

    endings       = []
    bifurcations  = []

    ys, xs = np.where(skeleton > 0)
    for y, x in zip(ys, xs):
        # Skip border pixels
        if y < margin or y >= h - margin or x < margin or x >= w - margin:
            continue
        nbr = _get_8_neighbors_clockwise(skeleton, y, x)
        cn  = _crossing_number(nbr)
        if cn == 1:
            endings.append((y, x))
        elif cn == 3:
            bifurcations.append((y, x))

    # NMS to remove duplicate / too-close detections
    endings      = _nms_minutiae(endings,      min_dist)
    bifurcations = _nms_minutiae(bifurcations, min_dist)

    all_coords = endings + bifurcations
    all_types  = ["E"] * len(endings) + ["B"] * len(bifurcations)

    # Cap at max_minutiae
    mx = CFG["max_minutiae"]
    if len(all_coords) > mx:
        all_coords = all_coords[:mx]
        all_types  = all_types[:mx]

    if not all_coords:
        return np.empty((0, 2), dtype=int), np.array([], dtype=str)

    return np.array(all_coords, dtype=int), np.array(all_types)


def preprocess_fingerprint(img_path: str) -> dict:
    """
    Full preprocessing pipeline for one fingerprint image.

    Input
    -----
    img_path : str   Path to fingerprint image (BMP/TIF/PNG).

    Output
    ------
    result : dict with keys:
        "gray"      – (H, W) uint8  raw grayscale
        "enhanced"  – (H, W) uint8  Gabor-enhanced
        "binary"    – (H, W) bool   binarized ridges
        "skeleton"  – (H, W) uint8  1-px skeleton
        "minutiae"  – (N, 2) int    minutiae [y, x]
        "types"     – (N,)  str     minutiae types ('E'/'B')
    Returns None if image cannot be read or too few minutiae.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.warning(f"Cannot read image: {img_path}")
        return None

    enhanced = enhance_with_gabor(img)
    binary   = binarize(enhanced)
    skeleton = compute_skeleton(binary)
    minutiae, types = extract_minutiae(skeleton)

    if len(minutiae) < CFG["min_minutiae"]:
        log.debug(f"Too few minutiae ({len(minutiae)}) in {img_path}")
        return None

    return {
        "gray"     : img,
        "enhanced" : enhanced,
        "binary"   : binary,
        "skeleton" : skeleton,
        "minutiae" : minutiae,
        "types"    : types,
        "path"     : img_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1.2 – DELAUNAY TRIANGULATION  (GRAPH CONSTRUCTION)
# ═══════════════════════════════════════════════════════════════════════════════

def build_delaunay_graph(minutiae: np.ndarray) -> tuple:
    """
    Build a sparse graph over minutiae points via Delaunay triangulation.
    Long edges (> max_edge_px) are pruned to keep the graph local.

    Input
    -----
    minutiae : (N, 2) int   [y, x] coordinates of minutiae.

    Output
    ------
    edges       : (E, 2) int   pairs of node indices (undirected).
    adjacency   : dict {node_id: set(neighbour_ids)}
    """
    n = len(minutiae)
    if n < 4:
        # Fallback: fully connect the few points
        edges = np.array(list(combinations(range(n), 2)), dtype=int)
        adjacency = defaultdict(set)
        for i, j in edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        return edges, dict(adjacency)

    # Delaunay expects (x, y) — swap columns
    pts = minutiae[:, ::-1].astype(float)   # (N, 2) → (x, y)
    tri = Delaunay(pts)

    max_len    = CFG["max_edge_px"]
    edge_set   = set()
    for simplex in tri.simplices:
        for a, b in [(0,1),(1,2),(0,2)]:
            i, j = simplex[a], simplex[b]
            if i > j:
                i, j = j, i
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist <= max_len:
                edge_set.add((i, j))

    edges = np.array(sorted(edge_set), dtype=int) if edge_set else np.empty((0,2), dtype=int)

    adjacency = defaultdict(set)
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    return edges, dict(adjacency)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – RIDGE-COUNT EDGE WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════════

def _bresenham_line(y0: int, x0: int, y1: int, x1: int) -> list:
    """
    Enumerate all pixel positions along the Bresenham line from (y0,x0) to (y1,x1).

    Output
    ------
    points : list of (y, x) tuples
    """
    points = []
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy     =  1 if y0 < y1 else -1
    sx     =  1 if x0 < x1 else -1
    err    = dx - dy
    cy, cx = y0, x0
    while True:
        points.append((cy, cx))
        if cy == y1 and cx == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx  += sx
        if e2 < dx:
            err += dx
            cy  += sy
    return points


def count_ridges_on_line(binary: np.ndarray, p1: tuple, p2: tuple) -> int:
    """
    Count the number of friction ridges crossed by the line segment p1→p2.
    Uses 0→1 transitions on the (non-thinned) binary image.

    Input
    -----
    binary : (H, W) bool / uint8   Ridge pixels = True / 255.
    p1, p2 : (y, x) int tuples     Start and end coordinates.

    Output
    ------
    ridge_count : int  (≥ 1 to avoid zero-weight edges)
    """
    pts = _bresenham_line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
    h, w = binary.shape

    # Skip a small border around each endpoint to avoid counting their own ridge
    skip = max(2, len(pts) // 20)
    mid_pts = pts[skip: len(pts) - skip]
    if not mid_pts:
        return 1

    prev = 0
    transitions = 0
    for y, x in mid_pts:
        if 0 <= y < h and 0 <= x < w:
            curr = 1 if binary[y, x] else 0
        else:
            curr = 0
        if prev == 0 and curr == 1:
            transitions += 1
        prev = curr

    return max(1, transitions)


def compute_ridge_weights(edges: np.ndarray,
                          minutiae: np.ndarray,
                          binary: np.ndarray) -> tuple:
    """
    Assign ridge-count weights to all Delaunay edges.

    Input
    -----
    edges    : (E, 2) int   edge list.
    minutiae : (N, 2) int   [y, x] minutiae coordinates.
    binary   : (H, W) bool  binarized fingerprint.

    Output
    ------
    weights      : (E,) int   ridge count per edge.
    weight_dict  : dict {(i, j): weight}  (i < j always)
    """
    weights     = np.zeros(len(edges), dtype=int)
    weight_dict = {}
    for k, (i, j) in enumerate(edges):
        p1 = tuple(minutiae[i])
        p2 = tuple(minutiae[j])
        rc = count_ridges_on_line(binary, p1, p2)
        weights[k] = rc
        key = (min(i, j), max(i, j))
        weight_dict[key] = rc
    return weights, weight_dict


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3.1 – LOCAL k-HOP SUBGRAPHS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_k_hop_subgraph(center: int, adjacency: dict, k: int = 3) -> tuple:
    """
    Extract the k-hop neighbourhood subgraph around a centre node using BFS.

    Input
    -----
    center    : int   Centre node index.
    adjacency : dict  {node: set(neighbours)} for the full graph.
    k         : int   Number of hops.

    Output
    ------
    sub_nodes : list int   All node indices in the subgraph (including centre).
    sub_edges : list (i,j) All edges whose both endpoints are in sub_nodes.
    """
    visited = {center}
    frontier = {center}
    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            for nb in adjacency.get(node, set()):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier

    sub_nodes = sorted(visited)
    node_set  = set(sub_nodes)
    sub_edges = []
    seen      = set()
    for node in sub_nodes:
        for nb in adjacency.get(node, set()):
            if nb in node_set:
                key = (min(node, nb), max(node, nb))
                if key not in seen:
                    seen.add(key)
                    sub_edges.append(key)

    return sub_nodes, sub_edges


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3.2 – SPECTRAL DESCRIPTORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spectral_descriptor(sub_nodes: list,
                                sub_edges: list,
                                weight_dict: dict,
                                size: int) -> np.ndarray:
    """
    Compute a fixed-length spectral descriptor from the Laplacian eigenvalues
    of a subgraph.

    Steps:
        1. Build weighted adjacency matrix A using ridge counts.
        2. Compute Laplacian L = D − A.
        3. Compute eigenvalues of L (sorted ascending).
        4. Pad / truncate to `size` dimensions.

    Input
    -----
    sub_nodes   : list int     Nodes in the subgraph.
    sub_edges   : list (i,j)   Edges in the subgraph.
    weight_dict : dict         {(i,j): ridge_count}
    size        : int          Target descriptor dimensionality.

    Output
    ------
    descriptor : (size,) float64   Sorted eigenvalue vector (invariant to
                                   rotation, translation, non-linear distortion).
    """
    n = len(sub_nodes)
    if n < 2:
        return np.zeros(size)

    # Local re-indexing
    idx = {node: i for i, node in enumerate(sub_nodes)}

    A = np.zeros((n, n), dtype=float)
    for (i, j) in sub_edges:
        li, lj = idx.get(i, -1), idx.get(j, -1)
        if li == -1 or lj == -1:
            continue
        w = float(weight_dict.get((min(i,j), max(i,j)), 1))
        A[li, lj] += w
        A[lj, li] += w

    D = np.diag(A.sum(axis=1))
    L = D - A                       # Graph Laplacian

    try:
        eigs = np.sort(np.abs(eigvalsh(L)))   # real, non-negative for PSD L
    except Exception:
        eigs = np.zeros(n)

    # Pad to `size` (zero-pad) or truncate
    if len(eigs) >= size:
        descriptor = eigs[:size]
    else:
        descriptor = np.pad(eigs, (0, size - len(eigs)))

    return descriptor.astype(np.float64)


def compute_all_node_descriptors(minutiae: np.ndarray,
                                 adjacency: dict,
                                 weight_dict: dict) -> np.ndarray:
    """
    Compute spectral descriptors for every minutia node in the fingerprint.

    Input
    -----
    minutiae    : (N, 2) int
    adjacency   : dict {node: set(neighbours)}
    weight_dict : dict {(i,j): ridge_count}

    Output
    ------
    node_descs : (N, descriptor_size) float64
        Row i is the spectral descriptor of the k-hop subgraph around node i.
    """
    n    = len(minutiae)
    k    = CFG["k_hop"]
    size = CFG["descriptor_size"]
    descs = np.zeros((n, size), dtype=np.float64)

    for i in range(n):
        sub_nodes, sub_edges = extract_k_hop_subgraph(i, adjacency, k)
        descs[i] = compute_spectral_descriptor(sub_nodes, sub_edges,
                                               weight_dict, size)
    return descs


def fingerprint_global_descriptor(node_descs: np.ndarray) -> np.ndarray:
    """
    Aggregate per-node spectral descriptors into a single fingerprint vector.

    Representation: concatenation of [mean, std] across all node descriptors.
    This produces a (2 * descriptor_size,) vector used for KD-tree indexing.

    Input
    -----
    node_descs : (N, D) float64

    Output
    ------
    global_vec : (2D,) float64
    """
    if len(node_descs) == 0:
        return np.zeros(2 * CFG["descriptor_size"])
    mu  = node_descs.mean(axis=0)
    sig = node_descs.std(axis=0)
    return np.concatenate([mu, sig])


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE  –  process one fingerprint image end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def process_fingerprint(img_path: str) -> dict | None:
    """
    Run all pipeline phases on a single fingerprint image.

    Input
    -----
    img_path : str   Path to fingerprint image.

    Output
    ------
    result : dict with keys:
        "path"         – original image path
        "prep"         – preprocessing dict (gray/enhanced/binary/skeleton/…)
        "edges"        – (E, 2) Delaunay edges
        "adjacency"    – graph adjacency dict
        "weights"      – (E,) ridge-count weights
        "weight_dict"  – {(i,j): weight}
        "node_descs"   – (N, D) spectral node descriptors
        "global_desc"  – (2D,) fingerprint global descriptor
    Returns None if fingerprint cannot be processed.
    """
    prep = preprocess_fingerprint(img_path)
    if prep is None:
        return None

    minutiae = prep["minutiae"]
    binary   = prep["binary"]

    # Phase 1.2
    edges, adjacency = build_delaunay_graph(minutiae)
    if len(edges) == 0:
        return None

    # Phase 2
    weights, weight_dict = compute_ridge_weights(edges, minutiae, binary)

    # Phase 3
    node_descs  = compute_all_node_descriptors(minutiae, adjacency, weight_dict)
    global_desc = fingerprint_global_descriptor(node_descs)

    return {
        "path"        : img_path,
        "prep"        : prep,
        "edges"       : edges,
        "adjacency"   : adjacency,
        "weights"     : weights,
        "weight_dict" : weight_dict,
        "node_descs"  : node_descs,
        "global_desc" : global_desc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 – KD-TREE INDEXING & FAST MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def build_kd_index(global_descs: list, labels: list) -> tuple:
    """
    Build a KD-tree index for fast 1:N fingerprint identification.

    Input
    -----
    global_descs : list of (2D,) vectors   One per fingerprint in the gallery.
    labels       : list of str             Fingerprint IDs (subject_impression).

    Output
    ------
    kdtree : scipy.spatial.KDTree
    labels : list (same order as tree leaves)
    """
    matrix = np.vstack(global_descs)
    # L2 normalise each row so Euclidean distance ≈ angular distance
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix = matrix / norms
    kdtree = KDTree(matrix)
    return kdtree, labels


def query_kd_index(query_desc: np.ndarray,
                   kdtree: KDTree,
                   labels: list,
                   k: int = None) -> list:
    """
    Query KD-tree for nearest fingerprints to a given descriptor.

    Input
    -----
    query_desc : (2D,) float64   Query fingerprint's global descriptor.
    kdtree     : KDTree          Pre-built index.
    labels     : list            Gallery labels matching tree order.
    k          : int             Number of nearest neighbours.

    Output
    ------
    results : list of (distance, label) tuples, sorted ascending by distance.
    """
    k   = k or CFG["k_nn"]
    vec = query_desc / (np.linalg.norm(query_desc) + 1e-9)
    dists, idxs = kdtree.query(vec, k=min(k, len(labels)))
    return list(zip(np.atleast_1d(dists), [labels[i] for i in np.atleast_1d(idxs)]))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two descriptor vectors (range 0–1).

    Input
    -----
    a, b : (D,) float64

    Output
    ------
    sim : float   1 = identical, 0 = orthogonal.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION – FAR / FRR / EER
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_verification(results_by_subject: dict) -> dict:
    """
    Compute verification performance (FAR, FRR, EER).

    Genuine pairs  : same subject, different impressions.
    Impostor pairs : different subjects (first impression of each).

    Input
    -----
    results_by_subject : dict
        {subject_id: [global_desc_0, global_desc_1, …]}  one desc per impression.

    Output
    ------
    metrics : dict
        {
          "genuine_scores"  : list[float],
          "impostor_scores" : list[float],
          "eer"             : float,
          "eer_threshold"   : float,
          "fnmr_at_fmr0001" : float,   # FRR at FAR=0.1 %
          "auc"             : float,
        }
    """
    genuine_scores  = []
    impostor_scores = []

    subjects = sorted(results_by_subject.keys())

    # Genuine pairs (same subject, all impression combinations)
    for subj in subjects:
        descs = results_by_subject[subj]
        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                score = cosine_similarity(descs[i], descs[j])
                genuine_scores.append(score)

    # Impostor pairs (first impression of each pair of subjects)
    for s1, s2 in combinations(subjects, 2):
        d1 = results_by_subject[s1][0]
        d2 = results_by_subject[s2][0]
        score = cosine_similarity(d1, d2)
        impostor_scores.append(score)

    if not genuine_scores or not impostor_scores:
        log.warning("Insufficient data for evaluation.")
        return {}

    gen  = np.array(genuine_scores)
    imp  = np.array(impostor_scores)
    thresholds = np.linspace(0, 1, CFG["n_thresh"])

    far_list, frr_list = [], []
    for t in thresholds:
        fa  = np.sum(imp >= t) / len(imp)   # impostors accepted (≥ threshold)
        fr  = np.sum(gen < t)  / len(gen)   # genuines rejected (< threshold)
        far_list.append(fa)
        frr_list.append(fr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # EER: crossing point of FAR and FRR curves
    diff    = np.abs(far_arr - frr_arr)
    eer_idx = int(np.argmin(diff))
    eer     = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    eer_thr = float(thresholds[eer_idx])

    # FNMR @ FMR=0.001 (FAR ≈ 0.1 %)
    idx_001 = np.where(far_arr <= 0.001)[0]
    fnmr_at_001 = float(frr_arr[idx_001[-1]]) if len(idx_001) else 1.0

    # AUC (using trapezoidal rule on ROC curve)
    order = np.argsort(far_arr)
    auc   = float(np.trapezoid(1 - frr_arr[order], far_arr[order]))

    return {
        "genuine_scores"   : gen.tolist(),
        "impostor_scores"  : imp.tolist(),
        "n_genuine_pairs"  : len(gen),
        "n_impostor_pairs" : len(imp),
        "eer"              : round(eer,       4),
        "eer_threshold"    : round(eer_thr,   4),
        "fnmr_at_fmr0001"  : round(fnmr_at_001, 4),
        "auc"              : round(auc,       4),
        "far_curve"        : far_arr,
        "frr_curve"        : frr_arr,
        "thresholds"       : thresholds,
    }


def evaluate_identification(gallery_labels: list,
                            gallery_descs:  list,
                            query_labels:  list,
                            query_descs:   list,
                            rank: int = 1) -> float:
    """
    Compute Rank-k identification accuracy.

    Input
    -----
    gallery_labels / descs : list  – gallery set (first impression per subject)
    query_labels   / descs : list  – query set   (remaining impressions)
    rank                   : int   – Rank-k (default 1)

    Output
    ------
    rank_k_acc : float  (0–1)
    """
    if not gallery_descs or not query_descs:
        return 0.0

    tree, labels = build_kd_index(gallery_descs, gallery_labels)
    correct = 0
    for qdesc, qlabel in zip(query_descs, query_labels):
        results  = query_kd_index(qdesc, tree, labels, k=rank)
        top_k    = [lbl for _, lbl in results]
        q_subj   = qlabel.split("_")[0]
        g_subjts = [lbl.split("_")[0] for lbl in top_k]
        if q_subj in g_subjts:
            correct += 1

    return correct / len(query_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION – save intermediate images for one sample
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    path = Path(CFG["vis_dir"]) / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    Saved: {path}")


def visualise_phase1_1(result: dict, tag: str = "sample") -> None:
    """
    Save Phase 1.1 visualisation: gray → enhanced → binary → skeleton → minutiae.

    Input
    -----
    result : dict from preprocess_fingerprint()
    tag    : str filename prefix
    """
    prep = result["prep"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle("Phase 1.1 – Preprocessing & Minutiae Extraction", fontsize=13, y=1.01)

    axes[0].imshow(prep["gray"],     cmap="gray"); axes[0].set_title("1. Raw Grayscale")
    axes[1].imshow(prep["enhanced"], cmap="gray"); axes[1].set_title("2. Gabor Enhanced")
    axes[2].imshow(prep["binary"],   cmap="gray"); axes[2].set_title("3. Binarised Ridges")
    axes[3].imshow(prep["skeleton"], cmap="gray"); axes[3].set_title("4. Skeleton")

    # Overlay minutiae on skeleton
    axes[4].imshow(prep["gray"], cmap="gray", alpha=0.6)
    minutiae = prep["minutiae"]
    types    = prep["types"]
    endings  = minutiae[types == "E"]
    bifurcs  = minutiae[types == "B"]
    if len(endings):
        axes[4].scatter(endings[:, 1], endings[:, 0],
                        s=25, c="lime",  marker="o", label="Ending")
    if len(bifurcs):
        axes[4].scatter(bifurcs[:, 1], bifurcs[:, 0],
                        s=25, c="red",   marker="^", label="Bifurcation")
    axes[4].legend(fontsize=7, loc="upper right")
    axes[4].set_title(f"5. Minutiae (N={len(minutiae)})")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, f"{tag}_phase1_1_preprocessing.png")


def visualise_phase1_2(result: dict, tag: str = "sample") -> None:
    """
    Save Phase 1.2 visualisation: minutiae + Delaunay graph.
    """
    prep     = result["prep"]
    minutiae = prep["minutiae"]
    edges    = result["edges"]
    weights  = result["weights"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 1.2 – Delaunay Triangulation", fontsize=13, y=1.01)

    # Left: raw minutiae scatter
    axes[0].imshow(prep["gray"], cmap="gray", alpha=0.7)
    axes[0].scatter(minutiae[:, 1], minutiae[:, 0],
                    s=30, c="cyan", edgecolors="white", linewidths=0.5)
    axes[0].set_title(f"Minutiae Points  (N={len(minutiae)})")

    # Right: Delaunay graph with edge weight colouring
    axes[1].imshow(prep["gray"], cmap="gray", alpha=0.7)
    if len(edges):
        w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        cmap   = plt.cm.plasma
        for (i, j), wn in zip(edges, w_norm):
            y0, x0 = minutiae[i]
            y1, x1 = minutiae[j]
            axes[1].plot([x0, x1], [y0, y1],
                         color=cmap(wn), linewidth=0.8, alpha=0.7)
    axes[1].scatter(minutiae[:, 1], minutiae[:, 0],
                    s=20, c="yellow", edgecolors="black", linewidths=0.3, zorder=5)
    axes[1].set_title(f"Delaunay Graph  (E={len(edges)} edges)")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                norm=plt.Normalize(weights.min(), weights.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], label="Ridge Count")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, f"{tag}_phase1_2_delaunay.png")


def visualise_phase2(result: dict, tag: str = "sample") -> None:
    """
    Save Phase 2 visualisation: ridge count weights on each edge.
    """
    prep     = result["prep"]
    minutiae = prep["minutiae"]
    edges    = result["edges"]
    weights  = result["weights"]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(prep["gray"], cmap="gray", alpha=0.65)

    if len(edges):
        w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        cmap   = plt.cm.coolwarm
        for (i, j), w, wn in zip(edges, weights, w_norm):
            y0, x0 = minutiae[i]
            y1, x1 = minutiae[j]
            ax.plot([x0, x1], [y0, y1],
                    color=cmap(wn), linewidth=1.0, alpha=0.8)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, str(w), fontsize=4, color="white",
                    ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.4, pad=0.5, linewidth=0))

    ax.scatter(minutiae[:, 1], minutiae[:, 0],
               s=20, c="lime", edgecolors="black", linewidths=0.3, zorder=5)
    ax.set_title("Phase 2 – Ridge-Count Edge Weights", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, f"{tag}_phase2_ridge_weights.png")


def visualise_phase3(result: dict, tag: str = "sample", node_example: int = 0) -> None:
    """
    Save Phase 3 visualisation: k-hop subgraph + spectral descriptor bars.
    """
    prep        = result["prep"]
    minutiae    = prep["minutiae"]
    adjacency   = result["adjacency"]
    weight_dict = result["weight_dict"]
    node_descs  = result["node_descs"]
    k           = CFG["k_hop"]
    size        = CFG["descriptor_size"]

    # Pick a node near the centre of the fingerprint
    h, w   = prep["gray"].shape
    cy, cx = h / 2, w / 2
    dists  = np.linalg.norm(minutiae - [cy, cx], axis=1)
    node_example = int(np.argmin(dists))

    sub_nodes, sub_edges = extract_k_hop_subgraph(node_example, adjacency, k)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f"Phase 3 – Spectral Descriptors  (k={k}-hop, dim={size})",
                 fontsize=13, y=1.01)

    # Left: full graph
    axes[0].imshow(prep["gray"], cmap="gray", alpha=0.55)
    for (i, j) in result["edges"]:
        y0, x0 = minutiae[i]; y1, x1 = minutiae[j]
        axes[0].plot([x0, x1], [y0, y1], "c-", lw=0.5, alpha=0.5)
    axes[0].scatter(minutiae[:, 1], minutiae[:, 0],
                    s=15, c="yellow", zorder=4)
    py, px = minutiae[node_example]
    axes[0].scatter([px], [py], s=80, c="red", zorder=5, label=f"Centre node {node_example}")
    axes[0].set_title("Full Delaunay Graph")
    axes[0].legend(fontsize=7)
    axes[0].axis("off")

    # Middle: k-hop subgraph
    axes[1].imshow(prep["gray"], cmap="gray", alpha=0.55)
    sub_set = set(sub_nodes)
    for (i, j) in sub_edges:
        y0, x0 = minutiae[i]; y1, x1 = minutiae[j]
        axes[1].plot([x0, x1], [y0, y1], "r-", lw=1.2, alpha=0.85)
    axes[1].scatter(minutiae[sub_nodes, 1], minutiae[sub_nodes, 0],
                    s=40, c="orange", edgecolors="black", lw=0.5, zorder=5)
    axes[1].scatter([px], [py], s=100, c="red", marker="*", zorder=6)
    axes[1].set_title(f"{k}-hop Subgraph  ({len(sub_nodes)} nodes, {len(sub_edges)} edges)")
    axes[1].axis("off")

    # Right: spectral descriptor bar chart (all nodes, averaged)
    mean_desc = node_descs.mean(axis=0)
    axes[2].bar(range(size), mean_desc, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[2].set_xlabel("Eigenvalue index")
    axes[2].set_ylabel("Eigenvalue")
    axes[2].set_title("Avg. Spectral Descriptor across all nodes")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, f"{tag}_phase3_spectral.png")


def visualise_phase4(metrics: dict, db_name: str, tag: str = "sample") -> None:
    """
    Save Phase 4 visualisation: genuine/impostor score distributions + ROC curve.
    """
    gen = np.array(metrics["genuine_scores"])
    imp = np.array(metrics["impostor_scores"])
    far = metrics["far_curve"]
    frr = metrics["frr_curve"]
    thr = metrics["thresholds"]
    eer = metrics["eer"]
    thr_eer = metrics["eer_threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Phase 4 – Matching Evaluation  [{db_name}]", fontsize=13, y=1.01)

    # Histogram of scores
    bins = np.linspace(0, 1, 40)
    axes[0].hist(gen, bins=bins, color="forestgreen", alpha=0.7, label=f"Genuine  (n={len(gen)})")
    axes[0].hist(imp, bins=bins, color="tomato",      alpha=0.7, label=f"Impostor (n={len(imp)})")
    axes[0].axvline(thr_eer, color="navy", linestyle="--", label=f"EER thr={thr_eer:.3f}")
    axes[0].set_xlabel("Cosine Similarity Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distributions")
    axes[0].legend(fontsize=8)

    # FAR / FRR vs threshold
    axes[1].plot(thr, far, "r-", lw=2, label="FAR")
    axes[1].plot(thr, frr, "b-", lw=2, label="FRR")
    axes[1].axvline(thr_eer, color="k", linestyle=":", label=f"EER={eer:.2%}")
    axes[1].scatter([thr_eer], [eer], s=80, color="black", zorder=5)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Error Rate")
    axes[1].set_title("FAR / FRR Curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # ROC curve
    order = np.argsort(far)
    axes[2].plot(far[order], 1 - frr[order], "purple", lw=2, label=f"AUC={metrics['auc']:.3f}")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    axes[2].scatter([eer], [1 - eer], s=80, color="red", zorder=5, label=f"EER={eer:.2%}")
    axes[2].set_xlabel("False Accept Rate (FAR)")
    axes[2].set_ylabel("True Accept Rate (TAR = 1–FRR)")
    axes[2].set_title("ROC Curve")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, f"{tag}_phase4_evaluation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _progress(iterable, desc="", total=None):
    """Simple fallback progress logger (used when tqdm is absent)."""
    items = list(iterable)
    n     = total or len(items)
    t0    = time.time()
    for i, item in enumerate(items, 1):
        elapsed  = time.time() - t0
        eta      = (elapsed / i) * (n - i)
        pct      = 100 * i // n
        bar      = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{desc}] |{bar}| {i}/{n}  ETA {eta:.0f}s", end="", flush=True)
        yield item
    print()


def main():
    t_start = time.time()

    log.info("=" * 65)
    log.info("  TRC-SD Fingerprint Recognition Pipeline")
    log.info("  DSM 410 – Computer Vision | IIT Indore")
    log.info("=" * 65)

    # ── 0. Load dataset ──────────────────────────────────────────────────────
    log.info("\n[Phase 0]  Loading FVC2002 dataset …")
    dataset = load_dataset(CFG["dataset_dir"])
    if not dataset:
        log.error("Dataset is empty.  Check path: " + CFG["dataset_dir"])
        sys.exit(1)

    log.info(f"  Loaded {len(dataset)} databases: {list(dataset.keys())}")

    # ── Per-DB evaluation ─────────────────────────────────────────────────────
    all_summary = {}

    vis_saved = False    # save visualisations only for the very first image

    for db_name, subject_map in sorted(dataset.items()):

        log.info(f"\n{'─'*60}")
        log.info(f"  Database: {db_name}  |  {len(subject_map)} subjects")
        log.info(f"{'─'*60}")

        # Process all images ──────────────────────────────────────────────────
        log.info(f"\n[Phase 1-3]  Feature extraction …")

        gallery_descs, gallery_labels = [], []
        query_descs,   query_labels   = [], []
        by_subject                    = defaultdict(list)

        all_paths = []
        for subj_id in sorted(subject_map.keys()):
            for path in sorted(subject_map[subj_id]):
                all_paths.append((subj_id, path))

        n_ok = 0
        n_fail = 0

        for subj_id, path in tqdm(all_paths, desc=db_name):
            res = process_fingerprint(path)
            if res is None:
                n_fail += 1
                continue
            n_ok += 1

            label = f"{subj_id}_{Path(path).stem}"
            by_subject[subj_id].append(res["global_desc"])

            # gallery = first impression; rest = queries
            if len(by_subject[subj_id]) == 1:
                gallery_descs.append(res["global_desc"])
                gallery_labels.append(label)
            else:
                query_descs.append(res["global_desc"])
                query_labels.append(label)

            # ── Save intermediate visualisations for the very first image ──
            if not vis_saved:
                log.info(f"\n  Saving pipeline visualisations for: {Path(path).name}")
                visualise_phase1_1(res, tag=db_name)
                visualise_phase1_2(res, tag=db_name)
                visualise_phase2(res,   tag=db_name)
                visualise_phase3(res,   tag=db_name)
                vis_saved = True
                log.info("")

        log.info(f"  Processed: {n_ok} / {len(all_paths)}  "
                 f"(skipped {n_fail} with too few minutiae)")

        if n_ok == 0:
            log.warning(f"  {db_name}: no images could be processed – skipping.")
            continue

        # ── Verification evaluation ───────────────────────────────────────────
        log.info(f"\n[Phase 4 / Eval]  Verification …")
        metrics = evaluate_verification(dict(by_subject))

        if metrics:
            log.info(f"  Genuine pairs   : {metrics['n_genuine_pairs']}")
            log.info(f"  Impostor pairs  : {metrics['n_impostor_pairs']}")
            log.info(f"  EER             : {metrics['eer']:.2%}  (threshold={metrics['eer_threshold']:.3f})")
            log.info(f"  FNMR @ FMR=0.1%: {metrics['fnmr_at_fmr0001']:.2%}")
            log.info(f"  AUC             : {metrics['auc']:.4f}")

            visualise_phase4(metrics, db_name, tag=db_name)

        # ── Identification evaluation ─────────────────────────────────────────
        if gallery_descs and query_descs:
            log.info(f"\n[Phase 4]  Identification (Rank-1) …")
            t_idx   = time.time()
            rank1   = evaluate_identification(gallery_labels, gallery_descs,
                                              query_labels,   query_descs, rank=1)
            rank5   = evaluate_identification(gallery_labels, gallery_descs,
                                              query_labels,   query_descs, rank=5)
            t_match = time.time() - t_idx

            log.info(f"  Rank-1 accuracy  : {rank1:.2%}")
            log.info(f"  Rank-5 accuracy  : {rank5:.2%}")
            log.info(f"  KD-tree query time (all queries): {t_match*1000:.1f} ms  "
                     f"→ {t_match*1000/max(1,len(query_descs)):.2f} ms/query")
        else:
            rank1, rank5, t_match = 0.0, 0.0, 0.0

        all_summary[db_name] = {
            "n_images"      : n_ok,
            "eer"           : metrics.get("eer",      "N/A"),
            "fnmr_0001"     : metrics.get("fnmr_at_fmr0001", "N/A"),
            "auc"           : metrics.get("auc",      "N/A"),
            "rank1"         : rank1,
            "rank5"         : rank5,
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    t_total = time.time() - t_start

    log.info(f"\n{'═'*65}")
    log.info("  VALIDATION RESULTS SUMMARY")
    log.info(f"{'═'*65}")
    log.info(f"  {'Database':<12} {'#Images':>8} {'EER':>8} {'FNMR@0.1%':>12} "
             f"{'AUC':>8} {'Rank-1':>9} {'Rank-5':>9}")
    log.info(f"  {'-'*67}")
    for db, v in sorted(all_summary.items()):
        eer_str  = f"{v['eer']:.2%}"  if isinstance(v['eer'], float) else "N/A"
        fnmr_str = f"{v['fnmr_0001']:.2%}" if isinstance(v['fnmr_0001'], float) else "N/A"
        auc_str  = f"{v['auc']:.4f}" if isinstance(v['auc'], float) else "N/A"
        r1_str   = f"{v['rank1']:.2%}"
        r5_str   = f"{v['rank5']:.2%}"
        log.info(f"  {db:<12} {v['n_images']:>8} {eer_str:>8} {fnmr_str:>12} "
                 f"{auc_str:>8} {r1_str:>9} {r5_str:>9}")

    log.info(f"\n  Total wall-clock time : {t_total:.1f} s")
    log.info(f"  Outputs saved to      : {Path(CFG['output_dir']).resolve()}")
    log.info(f"  Sample visualisations : {Path(CFG['vis_dir']).resolve()}")
    log.info(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
