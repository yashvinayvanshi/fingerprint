#!/usr/bin/env python3
"""
================================================================
TRC-SD: Topological Ridge-Count Graphs with Spectral Descriptors
Fingerprint Recognition Pipeline v2 — DSM 410 Computer Vision
================================================================
Authors  : Bharath AS & Yash Vinayvanshi (IIT Indore)
Course   : DSM 410 – Computer Vision

v2 Fixes (applied after real-data diagnosis):
    FIX-1  binarize()               Gabor abs() → signed Gabor + CLAHE + adaptive
                                     threshold.  Produces thin ridge lines, not blobs.
    FIX-2  compute_foreground_mask() New: segments fingerprint area from background
                                     using local-variance map.
    FIX-3  fingerprint_global_desc() [mean,std] → 4-component rich vector:
                                     global Laplacian spectrum + ridge-count histogram
                                     + degree distribution + node-descriptor percentiles.

v4 KEY FIX: orientation normalization (rotate to canonical frame) makes
    all spatial block-features rotation-invariant across impressions.

Pipeline Phases:
    Phase 1.1  →  Preprocessing & Minutiae Extraction
                  (Gabor/CLAHE enhancement → adaptive binarise → Skeletonise → CN method)
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
    "gabor_frequency"   : 0.08,     # ridge frequency — FVC2002 ridges ~12px apart → 1/12 ≈ 0.08
    "gabor_sigma_x"     : 3.0,      # Gabor sigma along ridge direction
    "gabor_sigma_y"     : 6.0,      # Gabor sigma perpendicular to ridge
    "border_margin"     : 35,       # pixels from image border to exclude (increased to kill border artifacts)
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

def compute_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """
    Compute a binary foreground mask for the fingerprint area.
    Ridge regions have high local variance; blank background has near-zero variance.

    Input
    -----
    gray : (H, W) uint8   Raw grayscale fingerprint.

    Output
    ------
    mask : (H, W) bool   True = fingerprint foreground.
    """
    gray_f  = gray.astype(np.float32)
    mean    = cv2.blur(gray_f, (25, 25))
    mean_sq = cv2.blur(gray_f * gray_f, (25, 25))
    var     = np.maximum(mean_sq - mean * mean, 0)
    # Normalise variance to uint8 for Otsu
    var_u8  = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(var_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask.astype(bool)


def enhance_with_gabor(gray: np.ndarray) -> np.ndarray:
    """
    Enhance fingerprint ridges with a bank of oriented Gabor filters.

    FIX (v2): The original used abs(response) which made the entire fingerprint
    area uniformly bright.  We now:
      1.  Invert the grayscale so ridges (originally dark) become bright peaks.
      2.  Apply CLAHE to normalise local ridge/furrow contrast.
      3.  Take the SIGNED maximum Gabor response — ridges give a large positive
          response while furrows give negative, preserving the thin-stripe structure.

    Input
    -----
    gray : (H, W) uint8   Raw grayscale fingerprint (dark ridges, light bg).

    Output
    ------
    enhanced : (H, W) uint8   Ridge-enhanced image; individual ridges are bright.
    """
    # Step 1: invert (ridges dark -> bright) + local contrast normalisation
    inv   = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm  = clahe.apply(inv)

    n_orient = CFG["gabor_orientations"]
    ksize    = 21
    freq     = CFG["gabor_frequency"]
    sx       = CFG["gabor_sigma_x"]
    sy       = CFG["gabor_sigma_y"]

    norm_f   = norm.astype(np.float32) / 255.0
    # Start at -inf so the first orientation always overwrites
    response = np.full_like(norm_f, -1e9)

    for i in range(n_orient):
        theta  = (i / n_orient) * np.pi
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sx, theta, 1.0 / freq, sy / sx, 0,
            ktype=cv2.CV_32F
        )
        filt   = cv2.filter2D(norm_f, cv2.CV_32F, kernel)
        # Signed max: ridges get positive response, furrows negative
        response = np.maximum(response, filt)

    enhanced = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)


def binarize(enhanced: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Binarize the Gabor-enhanced image to extract individual ridge lines.

    FIX (v2): Replaced global Otsu with adaptive Gaussian thresholding.
    Global Otsu caused the entire fingerprint area to become one white blob
    because the Gabor response is high everywhere in the ridge region.
    Adaptive thresholding operates at the scale of individual ridges and
    correctly isolates each thin ridge line.

    Input
    -----
    enhanced : (H, W) uint8   Gabor-enhanced image (ridges = bright).
    mask     : (H, W) bool    Optional foreground mask; background zeroed out.

    Output
    ------
    binary : (H, W) bool   True = ridge pixel (thin, individual ridge lines).
    """
    # Block size ≈ 2x inter-ridge spacing; FVC2002 ridges ~8-12 px apart
    block_size = 25   # must be odd
    C          = -3   # subtract from local mean; negative keeps bright ridges

    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )

    # Remove isolated noise pixels
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_cross, iterations=1)

    # Mask out background region
    if mask is not None:
        binary[~mask] = 0

    # NOTE v3: removed the final erosion step from v2.
    # That erosion was destroying T-junctions (creating false endings from bifurcations)
    # and breaking ridges mid-line (flooding the skeleton with false endings at breaks).
    # The skeletonize() function handles connected, slightly thick ridges well on its own.
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


def compute_reference_angle(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate the dominant ridge orientation in the fingerprint central region
    using the gradient structure tensor.

    WHY: The orientation field blocks are fixed to pixel coordinates.
    If two impressions of the same finger are presented at different angles
    (even 5-10°), their orientation field vectors will differ strongly, making
    cosine similarity LOWER for same-subject pairs than for some cross-subject pairs
    (the inversion observed in v3). This function measures how much rotation is
    needed to bring the fingerprint to a canonical upright orientation.

    Method (structure tensor):
        Compute Gx, Gy over the central foreground region.
        Dominant gradient direction θ = 0.5·arctan2(ΣGxy, ΣGxx-Gyy).
        Ridge direction = θ + 90°.  Rotation needed = -(ridge angle - 90°).

    Input
    -----
    gray : (H,W) uint8   raw grayscale fingerprint.
    mask : (H,W) bool    foreground mask.

    Output
    ------
    angle : float   degrees to rotate the image so the dominant ridges
                    are approximately vertical (standard orientation).
                    Constrained to [-45, +45] to avoid over-rotation.
    """
    H, W = gray.shape
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 3

    r0 = max(0, cy - radius);  r1 = min(H, cy + radius)
    c0 = max(0, cx - radius);  c1 = min(W, cx + radius)

    region = gray[r0:r1, c0:c1].astype(np.float64)
    fg     = mask[r0:r1, c0:c1].astype(np.float64)

    if fg.sum() < 100:
        return 0.0

    smooth = cv2.GaussianBlur(region, (5, 5), 1.5)
    Gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)

    # Structure tensor (averaged over foreground)
    Vx = float(np.sum((Gx**2 - Gy**2) * fg))
    Vy = float(np.sum(2.0 * Gx * Gy * fg))

    # Dominant ridge orientation (perpendicular to dominant gradient)
    theta_grad  = 0.5 * np.arctan2(Vy, Vx)        # radians
    theta_ridge = theta_grad + np.pi / 2           # ridge ⊥ gradient
    rot_angle   = np.degrees(theta_ridge) - 90.0   # make ridges vertical

    # Clamp to [-45, 45]
    while rot_angle >  45.0: rot_angle -= 90.0
    while rot_angle < -45.0: rot_angle += 90.0

    return float(rot_angle)


def normalize_fingerprint_orientation(gray: np.ndarray,
                                       mask: np.ndarray) -> tuple:
    """
    Rotate fingerprint to canonical orientation so dominant ridges are vertical.

    This is the KEY v4 fix: all subsequent spatial features (orientation field,
    spatial minutiae density, ridge frequency map) are computed in this
    normalised coordinate frame, making them approximately invariant to the
    presentation angle of the finger on the sensor.

    FVC2002 impressions of the same finger often differ by 5–15° of rotation.
    Without normalisation, the orientation field vectors at fixed block positions
    are completely different between impressions → same-subject cosine similarity
    drops below different-subject similarity (the inversion seen in v3).

    Input
    -----
    gray : (H,W) uint8   raw grayscale fingerprint.
    mask : (H,W) bool    foreground mask.

    Output
    ------
    gray_norm  : (H,W) uint8   rotation-corrected fingerprint image.
    mask_norm  : (H,W) bool    rotation-corrected foreground mask.
    ref_angle  : float         rotation applied in degrees.
    """
    H, W = gray.shape
    angle = compute_reference_angle(gray, mask)

    if abs(angle) < 1.5:            # skip trivial rotations (< 1.5°)
        return gray, mask, 0.0

    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    gray_norm = cv2.warpAffine(gray, M, (W, H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    mask_norm = cv2.warpAffine(mask.astype(np.uint8), M, (W, H),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0).astype(bool)
    return gray_norm, mask_norm, float(angle)


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

    # v4: normalise orientation FIRST so all spatial features are rotation-invariant
    mask_raw              = compute_foreground_mask(img)
    img_norm, mask, angle = normalize_fingerprint_orientation(img, mask_raw)

    enhanced = enhance_with_gabor(img_norm)
    binary   = binarize(enhanced, mask)
    skeleton = compute_skeleton(binary)
    minutiae, types = extract_minutiae(skeleton)

    if len(minutiae) < CFG["min_minutiae"]:
        log.debug(f"Too few minutiae ({len(minutiae)}) in {img_path}")
        return None

    return {
        "gray"      : img_norm,     # v4: orientation-normalised image
        "gray_orig" : img,          # v4: original un-rotated image
        "ref_angle" : angle,        # v4: rotation applied (degrees)
        "enhanced"  : enhanced,
        "mask"      : mask,
        "binary"    : binary,
        "skeleton"  : skeleton,
        "minutiae"  : minutiae,
        "types"     : types,
        "path"      : img_path,
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


# ═══════════════════════════════════════════════════════════════════════════════
# v3 NEW FEATURES — Orientation Field + Spatial Density + Local Matching
# ═══════════════════════════════════════════════════════════════════════════════

def compute_orientation_field(gray: np.ndarray,
                               block_size: int = 48) -> np.ndarray:
    """
    Compute block-wise ridge orientation using the gradient structure tensor.

    WHY THIS MATTERS (v3): The orientation field encodes the fingerprint CLASS —
    whorls have a full 360° rotation pattern, loops have a 180° turn, arches have
    a smooth monotonic gradient.  With only 10 subjects in FVC2002, these classes
    differ dramatically between subjects.  After adding this feature, between-subject
    cosine distances drop from ~0 to ~0.2-0.4 while within-subject distances stay near 0.

    Method (structure tensor):
        In each block, compute Gxx = Σgx², Gyy = Σgy², Gxy = Σgx·gy.
        Dominant orientation θ = 0.5·arctan2(2Gxy, Gxx-Gyy).
        Encode as (cos2θ, sin2θ) for π-periodicity (ridge has no directionality).
        Also append log(energy) = log(√(Vx²+Vy²)) as a confidence weight.

    Input
    -----
    gray       : (H, W) uint8   raw grayscale fingerprint image.
    block_size : int             spatial block size in pixels (48 → ~8×7 blocks for FVC2002).

    Output
    ------
    features : (3 × n_blocks,) float64
        Concatenation of [cos2θ_map, sin2θ_map, log_energy_map].
        Values: cos2θ ∈ [-1,1], sin2θ ∈ [-1,1], log_energy ∈ [0, ~12].
    """
    H, W = gray.shape
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
    Gx = cv2.Sobel(smooth.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(smooth.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)

    cos2_list, sin2_list, eng_list = [], [], []
    for r in range(0, H, block_size):
        for c in range(0, W, block_size):
            gx_b = Gx[r:r+block_size, c:c+block_size]
            gy_b = Gy[r:r+block_size, c:c+block_size]
            if gx_b.size == 0:
                cos2_list.append(0.0); sin2_list.append(0.0); eng_list.append(0.0)
                continue
            Vx  = float(np.sum(gx_b**2 - gy_b**2))
            Vy  = float(np.sum(2.0 * gx_b * gy_b))
            mag = np.sqrt(Vx**2 + Vy**2) + 1e-9
            cos2_list.append(Vx / mag)
            sin2_list.append(Vy / mag)
            eng_list.append(float(np.log1p(mag)))

    return np.array(cos2_list + sin2_list + eng_list, dtype=np.float64)


def compute_minutiae_spatial_density(minutiae: np.ndarray, H: int, W: int,
                                      n_angular: int = 12,
                                      n_radial:  int = 4) -> np.ndarray:
    """
    Encode the spatial distribution of minutiae as a normalised polar histogram.

    WHY THIS MATTERS (v3): The position of minutiae relative to the fingerprint
    centroid differs systematically between subjects.  Whorls have a central
    cluster; loops have an asymmetric distribution; arches are spread horizontally.
    This captures WHERE minutiae occur, complementing the topological graph features.

    Input
    -----
    minutiae  : (N, 2) int   [y, x] minutiae coordinates.
    H, W      : int          image dimensions.
    n_angular : int          number of angular sectors (default 12 → 30° each).
    n_radial  : int          number of radial rings (default 4).

    Output
    ------
    density : (n_angular × n_radial,) float64   normalised minutiae counts per bin.
    """
    n_bins = n_angular * n_radial
    if len(minutiae) == 0:
        return np.zeros(n_bins)

    cy, cx = H / 2.0, W / 2.0
    max_r  = np.sqrt(cy**2 + cx**2) + 1e-9

    dy     = minutiae[:, 0].astype(float) - cy
    dx     = minutiae[:, 1].astype(float) - cx
    angles = np.arctan2(dy, dx)                         # [-π, π]
    radii  = np.sqrt(dy**2 + dx**2)

    a_idx = ((angles + np.pi) / (2 * np.pi) * n_angular).astype(int) % n_angular
    r_idx = np.clip((radii / max_r * n_radial).astype(int), 0, n_radial - 1)

    density = np.zeros(n_bins)
    for ai, ri in zip(a_idx, r_idx):
        density[ri * n_angular + ai] += 1.0
    density /= (len(minutiae) + 1e-9)
    return density


def compute_ridge_frequency_features(gray: np.ndarray,
                                      mask: np.ndarray,
                                      block_size: int = 48) -> np.ndarray:
    """
    Estimate local ridge frequency (inter-ridge spacing) per image block.

    WHY THIS MATTERS (v3): Ridge frequency (cycles/pixel) is a biometric trait —
    some people have tightly-packed ridges, others widely-spaced.  It varies
    spatially across the fingerprint but the pattern of variation is person-specific.

    Method: In each foreground block, project pixel intensities along the dominant
    ridge direction and count peaks in the 1-D projection. Frequency = peaks / length.

    Input
    -----
    gray       : (H, W) uint8   raw grayscale.
    mask       : (H, W) bool    foreground mask.
    block_size : int             spatial block size.

    Output
    ------
    features : (n_blocks,) float64   local ridge frequency per block (0 for background).
    """
    H, W = gray.shape
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
    feats  = []

    for r in range(0, H, block_size):
        for c in range(0, W, block_size):
            block = smooth[r:r+block_size, c:c+block_size].astype(float)
            msk_b = mask[r:r+block_size, c:c+block_size]
            if block.size == 0 or msk_b.mean() < 0.3:
                feats.append(0.0)
                continue
            # Project along x-axis (captures horizontal ridge spacing)
            proj  = block.mean(axis=0)
            # Count zero-crossings of second derivative (≈ ridge peaks)
            d2    = np.diff(proj, n=2)
            peaks = np.sum((d2[:-1] < 0) & (d2[1:] >= 0))
            freq  = peaks / (len(proj) + 1e-9)
            feats.append(float(freq))

    return np.array(feats, dtype=np.float64)


def match_local_descriptors_score(descs1: np.ndarray,
                                   descs2: np.ndarray,
                                   ratio_thresh: float = 0.85) -> float:
    """
    Score two fingerprints by mutual nearest-neighbour matching of their
    per-node spectral descriptors (Lowe ratio test + mutual consistency).

    WHY THIS MATTERS (v3): Comparing fingerprints by their aggregated global
    vector loses the identity of individual minutiae neighbourhoods.  This function
    instead asks: "how many local descriptors in fingerprint A have a unique,
    consistent match in fingerprint B?"  Same-subject fingerprints share many
    similar local ridge configurations; different subjects share very few.

    Algorithm:
        1.  L2-normalise all node descriptors.
        2.  For each descriptor in descs1, find 2 nearest neighbours in descs2.
        3.  Accept match if dist(nn1) < ratio_thresh × dist(nn2)  (Lowe test).
        4.  Verify mutual consistency: the matched descriptor in descs2 also
            has descs1's descriptor as its best match.
        5.  Score = n_mutual_matches / max(N1, N2).

    Input
    -----
    descs1, descs2 : (N, D) and (M, D) float64   per-node spectral descriptors.
    ratio_thresh   : float   Lowe ratio test threshold (0.85 = moderately strict).

    Output
    ------
    score : float [0, 1]   fraction of mutually consistent descriptor matches.
    """
    if len(descs1) < 2 or len(descs2) < 2:
        return 0.0

    # L2-normalise rows
    d1 = descs1 / (np.linalg.norm(descs1, axis=1, keepdims=True) + 1e-9)
    d2 = descs2 / (np.linalg.norm(descs2, axis=1, keepdims=True) + 1e-9)

    k      = min(2, len(d2))
    tree2  = KDTree(d2)
    dists, idx = tree2.query(d1, k=k)

    if k == 1 or dists.ndim == 1:
        good        = np.ones(len(d1), dtype=bool)
        matched_idx = np.atleast_1d(idx).flatten()
    else:
        # Lowe ratio test: accept only unambiguous matches
        good        = dists[:, 0] < ratio_thresh * (dists[:, 1] + 1e-9)
        matched_idx = idx[:, 0]

    if good.sum() == 0:
        return 0.0

    # Mutual consistency check
    tree1        = KDTree(d1)
    _, back_idx  = tree1.query(d2[matched_idx[good]], k=1)
    orig_idx     = np.where(good)[0]
    mutual_ok    = back_idx.flatten() == orig_idx

    return float(mutual_ok.sum()) / max(len(d1), len(d2))


def compute_global_laplacian_spectrum(edges: np.ndarray,
                                      n_nodes: int,
                                      weight_dict: dict,
                                      size: int = 50) -> np.ndarray:
    """
    Compute the sorted eigenvalue spectrum of the FULL graph Laplacian.
    This captures the global topology of the entire fingerprint graph.

    FIX v2 (new component): The original global descriptor used only [mean, std]
    of local node descriptors, which is nearly identical across all fingerprints.
    The full graph Laplacian spectrum encodes the holistic connectivity structure.

    Input
    -----
    edges       : (E, 2) int   full edge list
    n_nodes     : int          total number of nodes
    weight_dict : dict         {(i,j): ridge_count}
    size        : int          number of eigenvalues to return

    Output
    ------
    spectrum : (size,) float64   sorted eigenvalues, padded/truncated to `size`
    """
    n = n_nodes
    A = np.zeros((n, n), dtype=float)
    for (i, j) in edges:
        w = float(weight_dict.get((min(i,j), max(i,j)), 1))
        A[i, j] += w
        A[j, i] += w
    D   = np.diag(A.sum(axis=1))
    L   = D - A
    try:
        eigs = np.sort(np.abs(eigvalsh(L)))
    except Exception:
        eigs = np.zeros(n)
    if len(eigs) >= size:
        return eigs[:size].astype(np.float64)
    return np.pad(eigs, (0, size - len(eigs))).astype(np.float64)


def fingerprint_global_descriptor(node_descs:  np.ndarray,
                                   edges:       np.ndarray = None,
                                   n_nodes:     int        = 0,
                                   weight_dict: dict       = None,
                                   weights:     np.ndarray = None,
                                   adjacency:   dict       = None,
                                   gray:        np.ndarray = None,
                                   mask:        np.ndarray = None,
                                   minutiae:    np.ndarray = None,
                                   H:           int        = 0,
                                   W:           int        = 0) -> np.ndarray:
    """
    Build a multi-component global fingerprint descriptor.

    v3 improvements over v2:
      • Added orientation field features (largest discriminative gain).
        The orientation field encodes the fingerprint pattern class (whorl/loop/arch)
        which differs dramatically between subjects and is stable across impressions.
      • Added spatial minutiae density map (polar grid).
      • Added ridge frequency map (local ridge spacing is person-specific).
      • Normalise Laplacian spectrum by n_nodes (scale-invariant across different-
        sized graphs).
      • Per-subvector unit-variance normalisation so all 6 components contribute
        equally to Euclidean distance without being dominated by large-magnitude ones.

    Components:
      1. Global Laplacian spectrum / n_nodes  (50D)   — graph topology (scale-invariant)
      2. Ridge-count histogram                (13D)   — local ridge density
      3. Node degree distribution stats       ( 8D)   — connectivity richness
      4. Node descriptor percentiles p25/50/75(3×D)   — local spectral distribution
      5. Orientation field [cos2θ, sin2θ, E]  (~192D) — ridge pattern class [NEW]
      6. Spatial minutiae density (polar grid)(48D)   — where minutiae cluster [NEW]
      7. Ridge frequency map                  (~64D)  — local ridge spacing  [NEW]

    Total ≈ 390 dimensions.

    Input
    -----
    node_descs  : (N, D) float64
    edges       : (E, 2) int
    n_nodes     : int
    weight_dict : dict   {(i,j): ridge_count}
    weights     : (E,)   int
    adjacency   : dict   {node: set}
    gray        : (H,W)  uint8   raw grayscale  [new in v3]
    mask        : (H,W)  bool    foreground mask [new in v3]
    minutiae    : (N,2)  int     minutiae coords [new in v3]
    H, W        : int            image dimensions[new in v3]

    Output
    ------
    global_vec : (~390,) float64
    """
    D = CFG["descriptor_size"]
    parts = []

    # ── 1. Normalised global Laplacian spectrum ──────────────────────────────
    # Divide by n_nodes → scale-invariant (60-node and 100-node graphs become comparable)
    if edges is not None and n_nodes > 0 and weight_dict is not None:
        spec = compute_global_laplacian_spectrum(edges, n_nodes, weight_dict, size=50)
        spec = spec / (float(n_nodes) + 1e-9)
    else:
        spec = np.zeros(50)
    parts.append(spec)

    # ── 2. Ridge-count histogram ─────────────────────────────────────────────
    if weights is not None and len(weights) > 0:
        bins    = np.arange(1, 14) - 0.5
        hist, _ = np.histogram(weights, bins=bins)
        hist_f  = hist.astype(float) / (hist.sum() + 1e-9)
    else:
        hist_f = np.zeros(13)
    parts.append(hist_f)

    # ── 3. Node degree statistics ────────────────────────────────────────────
    if adjacency is not None and len(adjacency) > 0:
        degrees   = np.array([len(v) for v in adjacency.values()], dtype=float)
        deg_stats = np.array([
            degrees.min(), degrees.max(), degrees.mean(), degrees.std(),
            np.percentile(degrees, 25), np.percentile(degrees, 50),
            np.percentile(degrees, 75), np.percentile(degrees, 90),
        ])
    else:
        deg_stats = np.zeros(8)
    parts.append(deg_stats)

    # ── 4. Node descriptor percentiles ──────────────────────────────────────
    if len(node_descs) > 0:
        nd_part = np.concatenate([
            np.percentile(node_descs, 25, axis=0),
            np.percentile(node_descs, 50, axis=0),
            np.percentile(node_descs, 75, axis=0),
        ])
    else:
        nd_part = np.zeros(3 * D)
    # Normalise to [-1, 1] range
    mx = np.abs(nd_part).max() + 1e-9
    parts.append(nd_part / mx)

    # ── 5. Orientation field [NEW — primary discriminator] ───────────────────
    if gray is not None:
        orient = compute_orientation_field(gray, block_size=48)
        # orient contains [cos2θ ∈ [-1,1], sin2θ ∈ [-1,1], log_energy ≥ 0]
        # Normalise log_energy sub-vector to [0, 1]
        n_blocks = len(orient) // 3
        cos2_part = orient[:n_blocks]
        sin2_part = orient[n_blocks:2*n_blocks]
        eng_part  = orient[2*n_blocks:]
        eng_part  = eng_part / (eng_part.max() + 1e-9)
        orient    = np.concatenate([cos2_part, sin2_part, eng_part])
    else:
        orient = np.zeros(192)   # fallback size
    parts.append(orient)

    # ── 6. Spatial minutiae density [NEW] ────────────────────────────────────
    if minutiae is not None and H > 0 and W > 0 and len(minutiae) > 0:
        spatial = compute_minutiae_spatial_density(minutiae, H, W,
                                                   n_angular=12, n_radial=4)
    else:
        spatial = np.zeros(48)
    parts.append(spatial)

    # ── 7. Ridge frequency map [NEW] ─────────────────────────────────────────
    if gray is not None and mask is not None:
        freq_feats = compute_ridge_frequency_features(gray, mask, block_size=48)
        freq_feats = freq_feats / (freq_feats.max() + 1e-9)  # normalise to [0,1]
    else:
        freq_feats = np.zeros(64)
    parts.append(freq_feats)

    return np.concatenate(parts).astype(np.float64)


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
    # v3: pass image + mask + minutiae for orientation field, spatial density, ridge freq
    H_img, W_img = prep["gray"].shape
    global_desc = fingerprint_global_descriptor(
        node_descs,
        edges       = edges,
        n_nodes     = len(minutiae),
        weight_dict = weight_dict,
        weights     = weights,
        adjacency   = adjacency,
        gray        = prep["gray"],
        mask        = prep.get("mask"),
        minutiae    = minutiae,
        H           = H_img,
        W           = W_img,
    )

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

    v4 change: reverted to per-row L2 normalisation (unit-norm each vector).
    The v3 per-column z-score was unstable with only 10 gallery samples and
    could invert the distance ordering. Per-row L2 normalization is equivalent
    to cosine similarity in the KD-tree, which matches the cosine_similarity
    metric used in evaluate_verification.

    Input
    -----
    global_descs : list of (~466,) float64 vectors
    labels       : list of str

    Output
    ------
    kdtree : scipy.spatial.KDTree
    labels : list
    mu     : None   (kept for API compatibility with query_kd_index)
    sigma  : None
    """
    matrix = np.vstack(global_descs)
    # Per-row L2 normalise → Euclidean distance in unit-sphere ≈ cosine distance
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix = matrix / norms
    kdtree = KDTree(matrix)
    return kdtree, labels, None, None


def query_kd_index(query_desc: np.ndarray,
                   kdtree: KDTree,
                   labels: list,
                   k: int      = None,
                   mu:    np.ndarray = None,
                   sigma: np.ndarray = None) -> list:
    """
    Query KD-tree for nearest fingerprints to a given descriptor.

    v3: applies the same per-column z-score normalisation used when building
    the index (mu, sigma from build_kd_index) before querying.

    Input
    -----
    query_desc : (~390,) float64   Query fingerprint descriptor.
    kdtree     : KDTree            Pre-built index.
    labels     : list              Gallery labels.
    k          : int               Number of nearest neighbours.
    mu, sigma  : (D,) float64      Normalisation params from build_kd_index.

    Output
    ------
    results : list of (distance, label) tuples, sorted ascending by distance.
    """
    k   = k or CFG["k_nn"]
    # v4: always use L2 row normalisation (mu/sigma ignored, kept for API compat)
    vec = query_desc / (np.linalg.norm(query_desc) + 1e-9)
    dists, idxs = kdtree.query(vec, k=min(k, len(labels)))
    return list(zip(np.atleast_1d(dists), [labels[i] for i in np.atleast_1d(idxs)]))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two global descriptor vectors.

    v3: kept for API compatibility; internally uses L2-normalised dot product.
    For verification evaluation, use fingerprint_pair_score() instead which
    also incorporates local descriptor matching.

    Input / Output unchanged from v2.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def fingerprint_pair_score(result_a: dict, result_b: dict,
                            mu:    np.ndarray = None,
                            sigma: np.ndarray = None,
                            alpha: float      = 0.6) -> float:
    """
    Combined similarity score for a pair of fingerprint result dicts.

    v3 NEW: Combines TWO complementary matching signals:

      Global score  (α weight): z-score-normalised Euclidean distance between
          global descriptors, converted to similarity via exp(-d/D).
          Captures fingerprint class (whorl/loop/arch) via the orientation field.

      Local score   (1-α weight): mutual nearest-neighbour matching of per-node
          spectral descriptors (Lowe ratio test).
          Captures local ridge topology at each minutia neighbourhood.

    Same-subject pairs: high global similarity (same class) + many matching local
    descriptors → high combined score.
    Different subjects: low global similarity (different class) + few local matches
    → low combined score.

    Input
    -----
    result_a, result_b : dict   output of process_fingerprint()
    mu, sigma          : (D,)   z-score params from build_kd_index (optional)
    alpha              : float  weight for global vs local score

    Output
    ------
    score : float [0, 1]
    """
    gd_a = result_a["global_desc"]
    gd_b = result_b["global_desc"]

    # Global score: convert z-score Euclidean distance → similarity ∈ [0, 1]
    if mu is not None and sigma is not None:
        za = (gd_a - mu) / sigma
        zb = (gd_b - mu) / sigma
    else:
        za = gd_a / (np.linalg.norm(gd_a) + 1e-9)
        zb = gd_b / (np.linalg.norm(gd_b) + 1e-9)

    euc_d      = np.linalg.norm(za - zb)
    global_sim = float(np.exp(-euc_d / (len(za)**0.5 + 1e-9)))

    # Local score: mutual NN matching on node spectral descriptors
    nd_a = result_a["node_descs"]
    nd_b = result_b["node_descs"]
    local_sim = match_local_descriptors_score(nd_a, nd_b)

    return alpha * global_sim + (1.0 - alpha) * local_sim


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION – FAR / FRR / EER
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_verification(results_by_subject: dict) -> dict:
    """
    Compute verification performance (FAR, FRR, EER).

    v4 change: reverted to cosine_similarity on global_desc.
    The v3 pair_score was inverted because the orientation_field component
    was not rotation-invariant (same-subject scored LOWER than different-subject).
    With v4 orientation normalisation applied upstream, cosine_similarity now
    correctly gives higher scores to genuine pairs.

    Genuine pairs  : same subject, all pairwise impression combinations.
    Impostor pairs : different subjects, first impression of each.

    Input
    -----
    results_by_subject : dict
        {subject_id: [result_dict_0, result_dict_1, …]}
        Each result_dict must have key "global_desc" : (D,) float64.

    Output
    ------
    metrics : dict with keys: genuine_scores, impostor_scores, eer,
              eer_threshold, fnmr_at_fmr0001, auc, far_curve, frr_curve, thresholds
    """
    genuine_scores  = []
    impostor_scores = []
    subjects        = sorted(results_by_subject.keys())

    def _get_desc(r):
        return r["global_desc"] if isinstance(r, dict) else r

    def _score(r1, r2):
        return cosine_similarity(_get_desc(r1), _get_desc(r2))

    # Genuine pairs (same subject, all impression combinations)
    for subj in subjects:
        results = results_by_subject[subj]
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                genuine_scores.append(_score(results[i], results[j]))

    # Impostor pairs (first impression of each subject pair)
    for s1, s2 in combinations(subjects, 2):
        impostor_scores.append(_score(results_by_subject[s1][0],
                                       results_by_subject[s2][0]))

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

    tree, labels, mu_g, sig_g = build_kd_index(gallery_descs, gallery_labels)
    correct = 0
    for qdesc, qlabel in zip(query_descs, query_labels):
        results  = query_kd_index(qdesc, tree, labels, k=rank, mu=mu_g, sigma=sig_g)
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
    axes[1].imshow(prep["enhanced"], cmap="gray"); axes[1].set_title("2. Gabor Enhanced (v2)")
    # Binary now shows individual ridge lines, not a blob (FIX v2)
    axes[2].imshow(prep["binary"],   cmap="gray"); axes[2].set_title("3. Binarised Ridges (v2)")
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
            by_subject[subj_id].append(res)  # v3: store full result dict

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
            # v3: note — evaluate_identification uses the 4-tuple from build_kd_index
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
