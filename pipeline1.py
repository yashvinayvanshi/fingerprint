#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Fingerprint Recognition Pipeline v2 — TRC-SD (Improved)                   ║
║  DSM 410: Computer Vision | Group Project                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KEY IMPROVEMENTS OVER v1                                                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Phase 0   : Gabor filter bank enhancement (replaces simple CLAHE+Otsu)    ║
║              → fixes merged-blob skeleton problem from v1                  ║
║  Phase 1.1 : Adaptive binarisation + branch-length quality filtering        ║
║              → removes spurious minutiae from noise/blob boundaries        ║
║              MAX_MINUTIAE raised 60→100, quality-ranked selection          ║
║  Phase 1.2 : Max-distance pruning on Delaunay edges (removes long jumps)   ║
║  Phase 2   : Same ridge-count weighting (anatomical invariant)             ║
║  Phase 3   : GLOBAL graph Laplacian spectral embedding                     ║
║              → top-K Fiedler eigenvectors of full Lsym (most impactful)    ║
║              + per-node k-hop descriptors (normalised Lsym, k=3)           ║
║              + ridge-count histogram (distribution-level feature)          ║
║              + structural fingerprint features (degree, clustering)        ║
║              Total descriptor: ~134-dim, L2-normalised                     ║
║  Phase 4   : Same KD-Tree, now on L2-normalised vectors                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, logging, warnings, math, re
from collections import defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

from scipy.spatial import Delaunay, KDTree
from scipy.linalg import eigvalsh, eigh
from skimage.morphology import skeletonize, remove_small_objects

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT   = "./datasets/PAMI Lab"
DB_NAMES       = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]
OUTPUT_DIR     = "./outputs_v2"
SAMPLE_DIR     = os.path.join(OUTPUT_DIR, "intermediate_steps")

# Minutiae
MAX_MINUTIAE   = 100
MIN_MINUTIAE   = 8
MARGIN         = 20
MIN_BRANCH_LEN = 8

# Graph
MAX_EDGE_DIST  = 80    # prune Delaunay edges longer than this (px)

# Descriptor
DESC_SIZE      = 20    # eigenvalues per k-hop node descriptor
K_HOP          = 3
GLOBAL_K       = 30    # top-K global Laplacian eigenvectors
RIDGE_HIST_BINS= 15

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "pipeline_v2.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)


def progress(iterable, desc="", total=None):
    """Lightweight progress bar."""
    items = list(iterable) if not hasattr(iterable, "__len__") else iterable
    n = total or len(items)
    start = time.time()
    for i, item in enumerate(items):
        pct = (i+1) / n * 100
        bar = "█" * int(pct//4) + "░" * (25 - int(pct//4))
        eta = (time.time()-start)/(i+1)*(n-i-1) if i > 0 else 0
        sys.stdout.write(f"\r  {desc} [{bar}] {i+1}/{n} ({pct:.0f}%) ETA:{eta:.0f}s")
        sys.stdout.flush()
        yield item
    print(f"  {'─'*50} done in {time.time()-start:.1f}s")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Gabor Filter Bank Enhancement
# ═════════════════════════════════════════════════════════════════════════════

def gabor_enhance(gray: np.ndarray,
                   n_angles: int = 8,
                   freq: float  = 0.11,
                   ksize: int   = 21,
                   sigma: float = 4.0) -> np.ndarray:
    """
    [FIX #1] Multi-orientation Gabor filter bank — max-response envelope.
    This replaces CLAHE+Otsu which caused ridges to merge into blobs.

    Input:
        gray     : (H,W) uint8 — greyscale fingerprint
        n_angles : int         — number of orientations
        freq     : float       — ridge spatial frequency (cycles/px)
        ksize    : int         — kernel size
        sigma    : float       — Gabor bandwidth

    Output:
        enhanced : (H,W) uint8 — Gabor-enhanced (ridges=bright)
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    f32   = eq.astype(np.float32) / 255.0

    responses = []
    for k in range(n_angles):
        theta = k * math.pi / n_angles
        kern  = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F
        )
        resp = cv2.filter2D(f32, cv2.CV_32F, kern)
        responses.append(resp)

    envelope = np.max(responses, axis=0)
    cv2.normalize(envelope, envelope, 0, 255, cv2.NORM_MINMAX)
    return envelope.astype(np.uint8)


def load_and_preprocess(image_path: str) -> tuple:
    """
    Load and produce both CLAHE and Gabor-enhanced images.

    Input:
        image_path : str — path to fingerprint image

    Output:
        gray      : (H,W) uint8 — original greyscale
        enhanced  : (H,W) uint8 — Gabor-enhanced (used for binarisation)
        clahe_img : (H,W) uint8 — CLAHE only (for display comparison)
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    gray      = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    enhanced  = gabor_enhance(gray)
    return gray, enhanced, clahe_img


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1.1 — Binarisation, Skeleton, Quality-Filtered Minutiae
# ═════════════════════════════════════════════════════════════════════════════

def extract_ridge_mask(enhanced: np.ndarray,
                        gray_fallback: np.ndarray = None) -> tuple:
    """
    [FIX #2] Adaptive threshold on Gabor-enhanced image.
    Avoids global Otsu merge that created blobs in v1.
    Falls back to CLAHE+Otsu if adaptive yields a degenerate skeleton
    (guards against unusual image statistics in edge cases).

    Input:
        enhanced       : (H,W) uint8 — Gabor-enhanced image
        gray_fallback  : (H,W) uint8 — original CLAHE image for fallback

    Output:
        skeleton : (H,W) uint8 — thinned skeleton (255=ridge)
        binary   : (H,W) uint8 — binary ridge mask
    """
    def _binarise_adaptive(img):
        b = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, blockSize=21, C=8
        )
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        b  = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k2, iterations=1)
        cl = remove_small_objects(b.astype(bool), min_size=50)
        return cl.astype(np.uint8)*255, skeletonize(cl).astype(np.uint8)*255

    def _binarise_otsu(img):
        """Fallback: CLAHE + global Otsu (v1 style, larger kernels)."""
        _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        b  = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k3, iterations=2)
        b  = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k3, iterations=1)
        cl = b.astype(bool)
        return b, skeletonize(cl).astype(np.uint8)*255

    # Try Gabor + adaptive first
    binary, skeleton = _binarise_adaptive(enhanced)

    # Sanity check: count crossing-number candidates (a well-formed skeleton
    # should have some). If zero across the whole image, fall back to Otsu.
    bin_skel = (skeleton > 0).astype(np.uint8)
    H, W = skeleton.shape
    sample_margin = 10
    patch = bin_skel[sample_margin:H-sample_margin, sample_margin:W-sample_margin]
    has_cn = False
    for y in range(0, patch.shape[0], 4):      # coarse scan for speed
        for x in range(0, patch.shape[1], 4):
            if patch[y, x] == 1:
                nbr = bin_skel[y+sample_margin-1:y+sample_margin+2,
                               x+sample_margin-1:x+sample_margin+2]
                if nbr.shape == (3,3) and _crossing_number(nbr) in (1,3):
                    has_cn = True
                    break
        if has_cn:
            break

    if not has_cn:
        # Fallback to Otsu on CLAHE or on Gabor-enhanced image
        fb = gray_fallback if gray_fallback is not None else enhanced
        binary, skeleton = _binarise_otsu(fb)

    return skeleton, binary


def _trace_branch(skel: np.ndarray, sy: int, sx: int,
                   py: int, px: int, max_len: int = 40) -> int:
    """
    Trace a ridge branch from a minutia to measure its pixel length.

    Input:
        skel    : (H,W) uint8 — skeleton
        sy, sx  : int — start position (minutia)
        py, px  : int — previous position (direction seed)
        max_len : int — cap for efficiency

    Output:
        length : int — branch pixel length
    """
    H, W   = skel.shape
    cy, cx = sy, sx
    length = 0
    for _ in range(max_len):
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] > 0:
                    if ny != py or nx != px:
                        nbrs.append((ny, nx))
        if not nbrs:
            break
        py, px = cy, cx
        cy, cx = nbrs[0]
        length += 1
        nbr = skel[cy-1:cy+2, cx-1:cx+2].astype(np.uint8) // 255
        if nbr.shape == (3,3):
            order = [nbr[0,1],nbr[0,2],nbr[1,2],nbr[2,2],
                      nbr[2,1],nbr[2,0],nbr[1,0],nbr[0,0]]
            cn = sum(abs(int(order[i])-int(order[(i+1)%8])) for i in range(8))//2
            if cn in (1,3) and length > 1:
                break
    return length


def _crossing_number(nbr: np.ndarray) -> int:
    """Compute crossing number of a 3×3 neighbourhood."""
    p     = nbr.flatten()
    order = [p[1],p[2],p[5],p[8],p[7],p[6],p[3],p[0]]
    return sum(abs(int(order[i])-int(order[(i+1)%8])) for i in range(8)) // 2


def extract_minutiae(skeleton: np.ndarray,
                     margin: int     = MARGIN,
                     max_pts: int    = MAX_MINUTIAE,
                     min_dist: float = 10.0,
                     min_branch: int = MIN_BRANCH_LEN) -> tuple:
    """
    [FIX #3] Crossing-number minutiae with branch-length quality filter.
    Short branches (<8px) are almost always noise artefacts from imperfect
    skeletonisation — filtering them out dramatically cleans up the graph.

    Input:
        skeleton   : (H,W) uint8  — ridge skeleton
        margin     : int          — border exclusion
        max_pts    : int          — upper limit
        min_dist   : float        — min pixel separation
        min_branch : int          — min branch length to accept

    Output:
        minutiae : (N,2) float32 — (x,y)
        types    : (N,)  int8   — 1=ending, 3=bifurcation
    """
    bin_skel = (skeleton > 0).astype(np.uint8)
    H, W     = bin_skel.shape
    pts, tps, quals = [], [], []

    for y in range(margin, H-margin):
        for x in range(margin, W-margin):
            if bin_skel[y, x] == 1:
                nbr = bin_skel[y-1:y+2, x-1:x+2]
                if nbr.shape == (3,3):
                    cn = _crossing_number(nbr)
                    if cn in (1, 3):
                        bl = _trace_branch(skeleton, y, x, y, x)
                        if bl >= min_branch:
                            pts.append([x, y])
                            tps.append(cn)
                            quals.append(bl)

    if not pts:
        return np.zeros((0,2), np.float32), np.zeros(0, np.int8)

    pts   = np.array(pts,   np.float32)
    tps   = np.array(tps,   np.int8)
    quals = np.array(quals, np.float32)

    # Quality-ranked NMS
    order = np.argsort(-quals)
    kept_p, kept_t = [], []
    used = np.zeros(len(pts), bool)
    for i in order:
        if not used[i]:
            kept_p.append(pts[i]); kept_t.append(tps[i])
            used[np.linalg.norm(pts-pts[i], axis=1) < min_dist] = True

    pts = np.array(kept_p, np.float32)
    tps = np.array(kept_t, np.int8)
    if len(pts) > max_pts:
        pts = pts[:max_pts]; tps = tps[:max_pts]
    return pts, tps


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1.2 — Pruned Delaunay Graph
# ═════════════════════════════════════════════════════════════════════════════

def build_delaunay_graph(minutiae: np.ndarray,
                          max_dist: float = MAX_EDGE_DIST) -> list:
    """
    [FIX #4] Delaunay triangulation with max-distance edge pruning.
    Removes long edges that connect topologically distant minutiae.

    Input:
        minutiae : (N,2) float32
        max_dist : float — max allowed edge length in pixels

    Output:
        edges : list of (int, int) — undirected edge index pairs
    """
    n = len(minutiae)
    if n < 3:
        return [(i,j) for i in range(n) for j in range(i+1,n)]

    tri      = Delaunay(minutiae)
    edge_set = set()
    for simplex in tri.simplices:
        for k in range(3):
            a, b = simplex[k], simplex[(k+1)%3]
            if np.linalg.norm(minutiae[a]-minutiae[b]) <= max_dist:
                edge_set.add((min(a,b), max(a,b)))

    # Reconnect isolated nodes
    adj_counts = defaultdict(int)
    for (a,b) in edge_set:
        adj_counts[a]+=1; adj_counts[b]+=1
    for i in range(n):
        if adj_counts[i] == 0:
            dists = np.linalg.norm(minutiae-minutiae[i], axis=1)
            dists[i] = np.inf
            j = int(np.argmin(dists))
            edge_set.add((min(i,j), max(i,j)))

    return list(edge_set)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Ridge-Count Edge Weighting
# ═════════════════════════════════════════════════════════════════════════════

def _bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0<x1 else -1
    sy = 1 if y0<y1 else -1
    err = dx - dy
    while True:
        pts.append((x0,y0))
        if x0==x1 and y0==y1: break
        e2 = 2*err
        if e2>-dy: err-=dy; x0+=sx
        if e2< dx: err+=dx; y0+=sy
    return pts


def count_ridges(pt_a: np.ndarray, pt_b: np.ndarray,
                  skeleton: np.ndarray) -> int:
    """
    Count ridge crossings between two minutiae (anatomical constant,
    invariant to elastic skin distortion).

    Input:
        pt_a, pt_b : (2,) float32 — (x,y) of minutiae
        skeleton   : (H,W) uint8

    Output:
        count : int ≥ 1
    """
    H, W     = skeleton.shape
    bin_skel = (skeleton > 0).astype(np.uint8)
    pixels   = _bresenham(int(pt_a[0]),int(pt_a[1]),int(pt_b[0]),int(pt_b[1]))
    count, prev = 0, 0
    for (px,py) in pixels:
        if 0<=py<H and 0<=px<W:
            curr = int(bin_skel[py,px])
            if prev==0 and curr==1: count+=1
            prev = curr
    return max(count, 1)


def build_weighted_graph(minutiae: np.ndarray, edges: list,
                          skeleton: np.ndarray, types: np.ndarray) -> tuple:
    """
    Attach ridge-count weights to every edge.

    Input:
        minutiae : (N,2), edges : list, skeleton : (H,W), types : (N,)

    Output:
        adj     : dict {i: [(j, weight), ...]}
        weights : dict {(i,j): weight}
    """
    adj     = defaultdict(list)
    weights = {}
    for (i,j) in edges:
        w = count_ridges(minutiae[i], minutiae[j], skeleton)
        adj[i].append((j,w)); adj[j].append((i,w))
        weights[(min(i,j), max(i,j))] = w
    return dict(adj), weights


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Multi-Level Spectral Descriptors
# ═════════════════════════════════════════════════════════════════════════════

def build_laplacian(n: int, edges_w: dict, normalised: bool = True) -> np.ndarray:
    """
    Build graph Laplacian from weighted edges.

    Input:
        n          : int  — node count
        edges_w    : dict {(i,j): weight}
        normalised : bool — True → Lsym = D^{-½} L D^{-½} ∈ [0,2]

    Output:
        L : (n,n) float64
    """
    A = np.zeros((n,n), np.float64)
    for (i,j), w in edges_w.items():
        A[i,j]=w; A[j,i]=w
    d = A.sum(axis=1)
    L = np.diag(d) - A
    if normalised:
        d_inv_sqrt = np.where(d>0, 1.0/np.sqrt(d), 0.0)
        D = np.diag(d_inv_sqrt)
        L = D @ L @ D
    return L


def compute_global_spectral_embedding(minutiae: np.ndarray,
                                       edges_w: dict,
                                       k: int = GLOBAL_K) -> tuple:
    """
    [FIX #5 — MOST IMPACTFUL] Global Fiedler spectral embedding.

    Compute top-k eigenvectors of the full graph's normalised Laplacian.
    These encode the entire topological fingerprint structure in one shot
    and are far more discriminative than per-node k-hop aggregations.

    Input:
        minutiae : (N,2)
        edges_w  : dict
        k        : int — eigenvectors to extract

    Output:
        global_desc : (4*k,) float32 — aggregated embedding
        global_vals : (k,)   float64 — eigenvalues for display
    """
    n    = len(minutiae)
    k_eff= min(k, n-1)
    zero = np.zeros(4*GLOBAL_K, np.float32)
    if n < 4 or not edges_w:
        return zero, np.zeros(GLOBAL_K)

    L = build_laplacian(n, edges_w, normalised=True)
    try:
        vals, vecs = eigh(L, subset_by_index=[0, k_eff])
    except Exception:
        return zero, np.zeros(GLOBAL_K)

    # Skip trivial λ₀=0
    vecs = vecs[:, 1:]; vals = vals[1:]
    if vecs.shape[1] == 0:
        return zero, np.zeros(GLOBAL_K)

    if vecs.shape[1] < GLOBAL_K:
        vecs = np.hstack([vecs, np.zeros((n, GLOBAL_K-vecs.shape[1]))])
        vals = np.concatenate([vals, np.zeros(GLOBAL_K-len(vals))])
    vecs = vecs[:, :GLOBAL_K]; vals = vals[:GLOBAL_K]

    # Aggregation: sorted spectral coordinates are permutation-invariant
    sorted_vecs = np.sort(vecs, axis=0)
    # Node-level max magnitudes — padded to ensure always GLOBAL_K elements
    node_max = np.sort(np.abs(vecs).max(axis=1))
    if len(node_max) >= GLOBAL_K:
        node_max = node_max[-GLOBAL_K:]
    else:
        node_max = np.pad(node_max, (0, GLOBAL_K - len(node_max)))
    desc = np.concatenate([
        sorted_vecs.mean(axis=0),
        sorted_vecs.std(axis=0),
        vals,
        node_max,
    ])
    return desc.astype(np.float32), vals


def compute_ridge_count_histogram(weights: dict,
                                   n_bins: int = RIDGE_HIST_BINS) -> np.ndarray:
    """
    [FIX #6] L1-normalised histogram of ridge counts across all edges.
    Captures the coarse topological density — a global invariant.

    Input:
        weights : dict {(i,j): ridge_count}
        n_bins  : int

    Output:
        hist : (n_bins,) float32
    """
    if not weights:
        return np.zeros(n_bins, np.float32)
    counts = np.array(list(weights.values()), np.float32)
    hist, _ = np.histogram(counts, bins=n_bins, range=(1,30))
    hist = hist.astype(np.float32)
    if hist.sum() > 0: hist /= hist.sum()
    return hist


def _get_khop_subgraph(node, adj, k):
    """Extract k-hop neighbourhood subgraph around a node."""
    visited = {node}; frontier = {node}
    for _ in range(k):
        nxt = set()
        for n in frontier:
            for (nb,_) in adj.get(n,[]):
                if nb not in visited:
                    nxt.add(nb); visited.add(nb)
        frontier = nxt
    local_nodes = sorted(visited)
    nmap = {n:i for i,n in enumerate(local_nodes)}
    seen = {}
    for n in local_nodes:
        for (nb,w) in adj.get(n,[]):
            if nb in nmap:
                a,b = min(nmap[n],nmap[nb]), max(nmap[n],nmap[nb])
                seen[(a,b)] = w
    return local_nodes, seen


def compute_local_spectral_desc(minutiae: np.ndarray, adj: dict,
                                  k: int = K_HOP,
                                  desc_sz: int = DESC_SIZE) -> np.ndarray:
    """
    [FIX #7] Per-node k-hop Lsym eigenvalues → aggregated statistics.

    Input:
        minutiae : (N,2), adj : dict, k : int, desc_sz : int

    Output:
        desc : (4*desc_sz,) float32
    """
    n = len(minutiae)
    if n == 0: return np.zeros(4*desc_sz, np.float32)
    node_descs = []
    for node in range(n):
        loc_n, loc_e = _get_khop_subgraph(node, adj, k)
        m = len(loc_n)
        if m < 2:
            node_descs.append(np.zeros(desc_sz, np.float32))
            continue
        L = build_laplacian(m, loc_e, normalised=True)
        try:    vals = np.sort(eigvalsh(L))
        except: vals = np.zeros(m)
        out = vals[:desc_sz] if len(vals)>=desc_sz else np.pad(vals,(0,desc_sz-len(vals)))
        node_descs.append(out.astype(np.float32))
    M = np.array(node_descs)
    return np.concatenate([M.mean(0), M.std(0), M.min(0), M.max(0)]).astype(np.float32)


def compute_structural_features(minutiae: np.ndarray, adj: dict,
                                  weights: dict, types: np.ndarray) -> np.ndarray:
    """
    [FIX #8] Compact hand-crafted structural features:
    degree stats, ending/bifurcation ratio, ridge count stats, spatial spread.

    Input:
        minutiae : (N,2), adj : dict, weights : dict, types : (N,)

    Output:
        feats : (9,) float32
    """
    n = len(minutiae)
    if n == 0: return np.zeros(9, np.float32)
    deg  = np.array([len(adj.get(i,[])) for i in range(n)], np.float32)
    eb   = float((types==1).sum()) / max(n,1)
    wv   = np.array(list(weights.values()), np.float32) if weights else np.array([0.0])
    diag = 400.0
    return np.array([
        deg.mean(), deg.std(), deg.max(),
        eb,
        wv.mean()/20.0, wv.std()/20.0, wv.max()/30.0,
        minutiae[:,0].std()/diag if n>1 else 0.0,
        minutiae[:,1].std()/diag if n>1 else 0.0,
    ], np.float32)


def compute_fingerprint_descriptor(minutiae: np.ndarray, adj: dict,
                                    weights: dict, types: np.ndarray) -> np.ndarray:
    """
    [FIX #9] Combine all descriptor components → L2-normalised vector.

    Components:
      A. Global spectral embedding (4*GLOBAL_K dims)   — finger-level topology
      B. Local k-hop spectral desc (4*DESC_SIZE dims)  — local topology
      C. Ridge-count histogram (RIDGE_HIST_BINS dims)  — ridge density
      D. Structural features (9 dims)                  — graph statistics

    Input:
        minutiae : (N,2), adj : dict, weights : dict, types : (N,)

    Output:
        descriptor : (D,) float32 — L2-normalised
    """
    A, _ = compute_global_spectral_embedding(minutiae, weights)
    B    = compute_local_spectral_desc(minutiae, adj)
    C    = compute_ridge_count_histogram(weights)
    D    = compute_structural_features(minutiae, adj, weights, types)
    raw  = np.concatenate([A, B, C, D])
    norm = np.linalg.norm(raw)
    if norm > 1e-8: raw = raw / norm
    return raw.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — KD-Tree Indexing & Matching
# ═════════════════════════════════════════════════════════════════════════════

def build_kd_index(descriptors: np.ndarray) -> KDTree:
    """Build KD-Tree for O(log N) nearest-neighbour search."""
    return KDTree(descriptors)


def kd_search(query: np.ndarray, tree: KDTree, k: int = 1) -> tuple:
    """KD-Tree query."""
    dists, idxs = tree.query(query.reshape(1,-1), k=k)
    return dists[0], idxs[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (L2-normalised inputs → dot product)."""
    na,nb = np.linalg.norm(a), np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def save_all_steps(sample_path, gray, clahe_img, enhanced,
                    binary, skeleton, minutiae, types, edges,
                    adj, weights, fp_desc, node_descs, global_vals):
    """Save all intermediate visualisation images for one sample."""
    label = os.path.basename(sample_path)

    def _save(name, img, cmap="gray", title=""):
        fig,ax = plt.subplots(figsize=(5,5))
        ax.imshow(img,cmap=cmap); ax.set_title(title,fontsize=9,fontweight="bold")
        ax.axis("off"); plt.tight_layout()
        fig.savefig(os.path.join(SAMPLE_DIR,name),dpi=130,bbox_inches="tight")
        plt.close(fig)

    _save("step0a_original.png",  gray,     "gray", f"Phase 0 │ Original\n{label}")
    _save("step0b_clahe.png",     clahe_img,"gray", "Phase 0 │ CLAHE only")
    _save("step0c_gabor.png",     enhanced, "gray", "Phase 0 │ Gabor Enhanced [FIX #1]")
    _save("step1a_binary.png",    binary,   "gray", "Phase 1.1 │ Adaptive Binarised [FIX #2]")
    _save("step1b_skeleton.png",  skeleton, "gray", "Phase 1.1 │ Clean Skeleton")

    # Minutiae type-coded
    fig,ax = plt.subplots(figsize=(5,5))
    ax.imshow(skeleton,cmap="gray")
    if len(minutiae):
        end_ = minutiae[types==1]; bif_ = minutiae[types==3]
        if len(end_): ax.scatter(end_[:,0],end_[:,1],c="red",s=18,label=f"Endings ({len(end_)})",zorder=5)
        if len(bif_): ax.scatter(bif_[:,0],bif_[:,1],c="lime",s=18,marker="^",label=f"Bifurc. ({len(bif_)})",zorder=5)
        ax.legend(fontsize=7,loc="upper right")
    ax.set_title(f"Phase 1.1 │ Minutiae N={len(minutiae)} [FIX #3 type-coded]",
                 fontsize=9,fontweight="bold"); ax.axis("off"); plt.tight_layout()
    fig.savefig(os.path.join(SAMPLE_DIR,"step1c_minutiae.png"),dpi=130,bbox_inches="tight")
    plt.close(fig)

    # Pruned Delaunay
    fig,ax = plt.subplots(figsize=(5,5))
    ax.imshow(gray,cmap="gray",alpha=0.45)
    for (i,j) in edges:
        ax.plot([minutiae[i,0],minutiae[j,0]],[minutiae[i,1],minutiae[j,1]],"y-",lw=0.7,alpha=0.7)
    ax.scatter(minutiae[:,0],minutiae[:,1],c="red",s=10,zorder=5)
    ax.set_title(f"Phase 1.2 │ Pruned Delaunay ({len(edges)} edges) [FIX #4]",
                 fontsize=9,fontweight="bold"); ax.axis("off"); plt.tight_layout()
    fig.savefig(os.path.join(SAMPLE_DIR,"step2a_delaunay.png"),dpi=130,bbox_inches="tight"); plt.close(fig)

    # Weighted
    fig,ax = plt.subplots(figsize=(5,5))
    ax.imshow(gray,cmap="gray",alpha=0.35)
    if edges and weights:
        ew = np.array([weights.get((min(i,j),max(i,j)),1) for (i,j) in edges],float)
        nrm= Normalize(vmin=ew.min(),vmax=ew.max())
        cmap_e = cm.plasma
        for idx,(i,j) in enumerate(edges):
            ax.plot([minutiae[i,0],minutiae[j,0]],[minutiae[i,1],minutiae[j,1]],
                    color=cmap_e(nrm(ew[idx])),lw=1.1)
        ax.scatter(minutiae[:,0],minutiae[:,1],c="cyan",s=8,zorder=5)
        plt.colorbar(cm.ScalarMappable(norm=nrm,cmap=cmap_e),ax=ax,
                     fraction=0.046,pad=0.04,label="Ridge Count",shrink=0.85)
    ax.set_title("Phase 2 │ Ridge-Count Weighted Graph",fontsize=9,fontweight="bold")
    ax.axis("off"); plt.tight_layout()
    fig.savefig(os.path.join(SAMPLE_DIR,"step2b_weighted.png"),dpi=130,bbox_inches="tight"); plt.close(fig)

    # Global eigenvalue spectrum
    fig,ax = plt.subplots(figsize=(6,3.5))
    ax.plot(global_vals[:GLOBAL_K],"o-",ms=4,lw=1.5,color="steelblue")
    ax.fill_between(range(len(global_vals[:GLOBAL_K])),global_vals[:GLOBAL_K],alpha=0.25)
    ax.set_title("Phase 3 │ Global Laplacian Eigenvalue Spectrum [FIX #5]",
                 fontsize=10,fontweight="bold")
    ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("λ (Lsym)")
    ax.grid(True,alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(SAMPLE_DIR,"step3a_global_spectrum.png"),dpi=130,bbox_inches="tight"); plt.close(fig)

    # Node descriptor matrix
    fig,ax = plt.subplots(figsize=(6,3))
    if len(node_descs):
        im = ax.imshow(node_descs.T,aspect="auto",cmap="viridis")
        plt.colorbar(im,ax=ax,shrink=0.85)
    ax.set_xlabel("Node"); ax.set_ylabel("Eigenvalue index")
    ax.set_title("Phase 3 │ Local k-hop Descriptor Matrix (k=3) [FIX #6]",fontsize=9,fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(SAMPLE_DIR,"step3b_local_desc.png"),dpi=130,bbox_inches="tight"); plt.close(fig)

    # Ridge count histogram
    fig,ax = plt.subplots(figsize=(5,3))
    if weights:
        ax.hist(list(weights.values()),bins=20,color="coral",edgecolor="black",alpha=0.8)
    ax.set_title("Phase 3 │ Ridge-Count Distribution [FIX #7]",fontsize=9,fontweight="bold")
    ax.set_xlabel("Ridge count"); ax.set_ylabel("Freq"); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAMPLE_DIR,"step3c_ridge_hist.png"),dpi=130,bbox_inches="tight"); plt.close(fig)

    # Final descriptor
    D      = len(fp_desc)
    colors = (["steelblue"]*(4*GLOBAL_K)+["coral"]*(4*DESC_SIZE)+
               ["green"]*RIDGE_HIST_BINS+["purple"]*9)[:D]
    fig,ax = plt.subplots(figsize=(8,3))
    ax.bar(range(D),fp_desc,color=colors[:D],alpha=0.8,width=1.0)
    patches = [mpatches.Patch(color="steelblue",label=f"Global Spectral ({4*GLOBAL_K}d)"),
               mpatches.Patch(color="coral",    label=f"Local k-hop ({4*DESC_SIZE}d)"),
               mpatches.Patch(color="green",    label=f"Ridge Hist ({RIDGE_HIST_BINS}d)"),
               mpatches.Patch(color="purple",   label="Structural (9d)")]
    ax.legend(handles=patches,fontsize=7,ncol=2)
    ax.set_title(f"Phase 4 │ L2-Norm Descriptor ({D}-dim) [FIX #8,9]",fontsize=9,fontweight="bold")
    ax.set_xlabel("Dim"); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAMPLE_DIR,"step4_descriptor.png"),dpi=130,bbox_inches="tight"); plt.close(fig)

    # ── Composite overview ───────────────────────────────────────────────
    fig, axes = plt.subplots(2,6,figsize=(30,10))
    fig.suptitle(f"TRC-SD Pipeline v2 — Improved │ {label}",fontsize=15,fontweight="bold")

    for k_,(img,cm_,ttl) in enumerate([
        (gray,    "gray","Ph0 │ Original"),
        (clahe_img,"gray","Ph0 │ CLAHE"),
        (enhanced,"gray","Ph0 │ Gabor [FIX#1]"),
        (binary,  "gray","Ph1.1 │ Adaptive [FIX#2]"),
        (skeleton,"gray","Ph1.1 │ Skeleton"),
    ]):
        axes[0,k_].imshow(img,cmap=cm_); axes[0,k_].set_title(ttl,fontsize=9,fontweight="bold"); axes[0,k_].axis("off")

    ax = axes[0,5]; ax.imshow(skeleton,cmap="gray")
    if len(minutiae):
        e_=minutiae[types==1]; b_=minutiae[types==3]
        if len(e_): ax.scatter(e_[:,0],e_[:,1],c="red",s=8,zorder=5,label=f"End {len(e_)}")
        if len(b_): ax.scatter(b_[:,0],b_[:,1],c="lime",s=8,marker="^",zorder=5,label=f"Bif {len(b_)}")
        ax.legend(fontsize=6)
    ax.set_title(f"Ph1.1 │ Minutiae N={len(minutiae)} [FIX#3]",fontsize=9,fontweight="bold"); ax.axis("off")

    ax=axes[1,0]; ax.imshow(gray,cmap="gray",alpha=0.45)
    for (i,j) in edges:
        ax.plot([minutiae[i,0],minutiae[j,0]],[minutiae[i,1],minutiae[j,1]],"y-",lw=0.5,alpha=0.7)
    ax.scatter(minutiae[:,0],minutiae[:,1],c="red",s=6,zorder=5)
    ax.set_title(f"Ph1.2 │ Pruned Delaunay ({len(edges)}e) [FIX#4]",fontsize=9,fontweight="bold"); ax.axis("off")

    ax=axes[1,1]; ax.imshow(gray,cmap="gray",alpha=0.35)
    if edges and weights:
        ew_=np.array([weights.get((min(i,j),max(i,j)),1) for (i,j) in edges],float)
        nrm_=Normalize(vmin=ew_.min(),vmax=ew_.max())
        for idx,(i,j) in enumerate(edges):
            ax.plot([minutiae[i,0],minutiae[j,0]],[minutiae[i,1],minutiae[j,1]],
                    color=cm.plasma(nrm_(ew_[idx])),lw=0.9)
        ax.scatter(minutiae[:,0],minutiae[:,1],c="cyan",s=6,zorder=5)
    ax.set_title("Ph2 │ Ridge-Count Weighted",fontsize=9,fontweight="bold"); ax.axis("off")

    ax=axes[1,2]; ax.plot(global_vals[:GLOBAL_K],"o-",ms=3,lw=1.5,color="steelblue")
    ax.fill_between(range(len(global_vals[:GLOBAL_K])),global_vals[:GLOBAL_K],alpha=0.2)
    ax.set_title("Ph3 │ Global Eigenvalues [FIX#5]",fontsize=9,fontweight="bold")
    ax.set_xlabel("Eigenvalue idx",fontsize=7); ax.grid(True,alpha=0.3)

    ax=axes[1,3]
    if len(node_descs): ax.imshow(node_descs.T,aspect="auto",cmap="viridis")
    ax.set_title("Ph3 │ Local Desc Matrix [FIX#6]",fontsize=9,fontweight="bold")

    ax=axes[1,4]
    if weights: ax.hist(list(weights.values()),bins=15,color="coral",edgecolor="black",alpha=0.8)
    ax.set_title("Ph3 │ Ridge Count Dist [FIX#7]",fontsize=9,fontweight="bold")
    ax.set_xlabel("Ridge count",fontsize=7); ax.grid(True,alpha=0.3)

    ax=axes[1,5]; ax.bar(range(len(fp_desc)),fp_desc,color=colors[:len(fp_desc)],alpha=0.8,width=1.0)
    ax.set_title(f"Ph4 │ L2 Descriptor ({len(fp_desc)}d) [FIX#8,9]",fontsize=9,fontweight="bold")
    ax.set_xlabel("Dim",fontsize=7); ax.grid(True,alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAMPLE_DIR,"pipeline_overview_v2.png")
    fig.savefig(path,dpi=120,bbox_inches="tight"); plt.close(fig)
    log.info(f"  ✓ Composite overview → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION — FAR / FRR / EER
# ═════════════════════════════════════════════════════════════════════════════

def build_match_pairs(records: list) -> tuple:
    """
    Build genuine (same finger) and impostor (different finger) score arrays.

    Input:
        records : list of dicts with keys: db, finger_id, descriptor

    Output:
        genuine  : np.ndarray — cosine similarity scores for genuine pairs
        impostor : np.ndarray — cosine similarity scores for impostor pairs
    """
    genuine, impostor = [], []
    groups = defaultdict(list)
    for r in records:
        groups[(r["db"], r["finger_id"])].append(r["descriptor"])

    for descs in groups.values():
        for i in range(len(descs)):
            for j in range(i+1, len(descs)):
                genuine.append(cosine_similarity(descs[i], descs[j]))

    keys  = list(groups.keys())
    rng   = np.random.default_rng(42)
    n_imp = min(len(genuine)*3, 8000)
    for _ in range(n_imp):
        if len(keys) < 2: break
        a, b = rng.choice(len(keys), 2, replace=False)
        if keys[a] != keys[b]:
            da = rng.integers(len(groups[keys[a]]))
            db = rng.integers(len(groups[keys[b]]))
            impostor.append(cosine_similarity(groups[keys[a]][da], groups[keys[b]][db]))

    return np.array(genuine), np.array(impostor)


def compute_far_frr(genuine, impostor, thresholds):
    """Compute FAR and FRR arrays over threshold sweep."""
    far = np.array([(impostor>=t).mean() for t in thresholds])
    frr = np.array([(genuine <t).mean()  for t in thresholds])
    return far, frr


def find_eer(far, frr, thresholds):
    """Find Equal Error Rate and its threshold."""
    idx = np.argmin(np.abs(far-frr))
    return float((far[idx]+frr[idx])/2), float(thresholds[idx])


def plot_roc(far, frr, thresholds, eer, eer_thresh, db_name):
    """Save FAR/FRR curve + DET curve for a database."""
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle(f"Validation — {db_name}  [Pipeline v2]",fontsize=13,fontweight="bold")
    ax1.plot(thresholds, far*100,"r-",lw=2,label="FAR")
    ax1.plot(thresholds, frr*100,"b-",lw=2,label="FRR")
    ax1.axvline(eer_thresh,color="green",ls="--",lw=1.5,label=f"EER thresh={eer_thresh:.3f}")
    ax1.axhline(eer*100,color="green",ls=":",lw=1.5,label=f"EER={eer*100:.2f}%")
    ax1.set_xlabel("Threshold"); ax1.set_ylabel("Error Rate (%)"); ax1.set_title("FAR & FRR")
    ax1.legend(fontsize=8); ax1.grid(True,alpha=0.3); ax1.set_ylim(-2,102)
    ax2.plot(far*100,frr*100,"purple",lw=2)
    ax2.scatter([eer*100],[eer*100],c="green",s=80,zorder=5,label=f"EER={eer*100:.2f}%")
    ax2.set_xlabel("FAR (%)"); ax2.set_ylabel("FRR (%)"); ax2.set_title("DET Curve")
    ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"roc_{db_name}_v2.png")
    fig.savefig(out,dpi=130,bbox_inches="tight"); plt.close(fig)
    log.info(f"  ✓ ROC → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FULL IMAGE PROCESSING WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

def process_image(image_path: str, save_steps: bool = False) -> tuple:
    """
    Run full v2 pipeline on one fingerprint image.

    Input:
        image_path : str  — path to fingerprint image
        save_steps : bool — save intermediate vis if True

    Output:
        descriptor : (D,) float32 — L2-normalised fingerprint vector
        meta       : dict         — counts and timing breakdown
    """
    t_total = time.time(); meta = {}

    t = time.time()
    gray, enhanced, clahe_img = load_and_preprocess(image_path)
    meta["t_load"] = time.time()-t

    t = time.time()
    skeleton, binary = extract_ridge_mask(enhanced, gray_fallback=clahe_img)
    minutiae, types  = extract_minutiae(skeleton)
    meta["t_minutiae"] = time.time()-t
    meta["n_minutiae"] = len(minutiae)

    if len(minutiae) < MIN_MINUTIAE:
        return None, meta

    t = time.time()
    edges = build_delaunay_graph(minutiae)
    meta["t_delaunay"] = time.time()-t
    meta["n_edges"]    = len(edges)

    t = time.time()
    adj, weights = build_weighted_graph(minutiae, edges, skeleton, types)
    meta["t_weighting"] = time.time()-t

    t = time.time()
    # Per-node descriptors for display
    node_descs = []
    for node in range(len(minutiae)):
        loc_n, loc_e = _get_khop_subgraph(node, adj, K_HOP)
        L = build_laplacian(len(loc_n), loc_e, normalised=True)
        try:    vals = np.sort(eigvalsh(L))
        except: vals = np.zeros(len(loc_n))
        out = vals[:DESC_SIZE] if len(vals)>=DESC_SIZE else np.pad(vals,(0,DESC_SIZE-len(vals)))
        node_descs.append(out.astype(np.float32))
    node_descs = np.array(node_descs)

    # Global eigenvalues for display
    n = len(minutiae); k_g = min(GLOBAL_K+1, n-1)
    try:
        gv, _ = eigh(build_laplacian(n,weights,True), subset_by_index=[0,k_g])
        gv = gv[1:]
    except: gv = np.zeros(GLOBAL_K)

    fp_desc = compute_fingerprint_descriptor(minutiae, adj, weights, types)
    meta["t_spectral"] = time.time()-t
    meta["t_total"]    = time.time()-t_total
    meta["desc_dim"]   = len(fp_desc)

    if save_steps:
        save_all_steps(image_path, gray, clahe_img, enhanced,
                        binary, skeleton, minutiae, types, edges,
                        adj, weights, fp_desc, node_descs, gv)
    return fp_desc, meta


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("="*68)
    log.info("  TRC-SD Fingerprint Recognition Pipeline v2 — IMPROVED")
    log.info("="*68)
    for i, fix in enumerate([
        "Gabor filter bank → clean ridge skeleton (replaces CLAHE+Otsu blob)",
        "Adaptive binarisation → no merged ridges",
        "Branch-length quality filter → removes spurious minutiae",
        "Pruned Delaunay (≤80px) → no spurious long edges",
        "Global Lsym Fiedler embedding → finger-specific topology [main fix]",
        "Normalised Laplacian Lsym for k-hop (k=3) → stable eigenvalues",
        "Ridge-count histogram → distribution-level invariant",
        "Structural hand-crafted features → degree/bifurcation context",
        "L2-normalised descriptor → correct cosine similarity range",
    ], 1):
        log.info(f"  [{i}] {fix}")
    log.info("")

    if not all(os.path.isdir(os.path.join(DATASET_ROOT,d)) for d in DB_NAMES):
        log.error("Dataset not found: " + DATASET_ROOT); sys.exit(1)
    log.info(f"Dataset: {DATASET_ROOT}")

    all_paths = []
    for db in DB_NAMES:
        imgs = sorted(os.path.join(DATASET_ROOT,db,f)
                      for f in os.listdir(os.path.join(DATASET_ROOT,db))
                      if f.lower().endswith((".tif",".bmp",".png",".jpg")))
        all_paths.extend(imgs)
        log.info(f"  {db}: {len(imgs)} images")
    log.info(f"  Total: {len(all_paths)}\n")

    log.info("─── PHASE 0-3: Feature Extraction ─────────────────────────────")
    records=[]; timing_log=[]; sample_done=False
    img_re = re.compile(r"(\d+)_(\d+)\.")

    for img_path in progress(all_paths, desc="Extracting v2"):
        fname = os.path.basename(img_path)
        db    = os.path.basename(os.path.dirname(img_path))
        m     = img_re.match(fname)
        if not m: continue
        finger_id=int(m.group(1)); imp_id=int(m.group(2))

        save = not sample_done
        desc, meta = process_image(img_path, save_steps=save)
        if desc is None:
            log.warning(f"  Skip {fname}: {meta.get('n_minutiae',0)} minutiae")
            continue
        if save:
            log.info(f"\n  ✓ Steps saved for: {fname}")
            log.info(f"    Minutiae:{meta['n_minutiae']}  Edges:{meta['n_edges']}  Desc:{meta['desc_dim']}d")
            log.info(f"    Timing → load:{meta['t_load']*1e3:.1f}ms "
                      f"minutiae:{meta['t_minutiae']*1e3:.1f}ms "
                      f"spectral:{meta['t_spectral']*1e3:.1f}ms "
                      f"total:{meta['t_total']*1e3:.1f}ms\n")
            sample_done=True

        records.append(dict(path=img_path,db=db,finger_id=finger_id,
                             impression_id=imp_id,descriptor=desc))
        timing_log.append(meta)

    log.info(f"\n  Processed: {len(records)} / {len(all_paths)}")
    if len(records)<10: log.error("Too few."); sys.exit(1)

    log.info("\n─── PHASE 4: KD-Tree ──────────────────────────────────────────")
    descs = np.array([r["descriptor"] for r in records])
    tree  = build_kd_index(descs)
    log.info(f"  KD-Tree │ {descs.shape}  "
              f"[Global:{4*GLOBAL_K} + Local:{4*DESC_SIZE} + Hist:{RIDGE_HIST_BINS} + Struct:9]")

    log.info("\n─── Validation ─────────────────────────────────────────────────")
    thresholds  = np.linspace(0.0,1.0,500)
    all_results = {}
    per_db      = defaultdict(list)
    for r in records: per_db[r["db"]].append(r)

    for db_name, db_recs in per_db.items():
        gen, imp = build_match_pairs(db_recs)
        if len(gen)==0 or len(imp)==0: continue
        far,frr         = compute_far_frr(gen,imp,thresholds)
        eer,eer_thresh  = find_eer(far,frr,thresholds)
        all_results[db_name]=dict(genuine=gen,impostor=imp,
                                   far=far,frr=frr,eer=eer,eer_thresh=eer_thresh)
        plot_roc(far,frr,thresholds,eer,eer_thresh,db_name)

    log.info("\n─── Speed Benchmark ────────────────────────────────────────────")
    q=descs[0]; t0=time.perf_counter(); N=1000
    for _ in range(N): kd_search(q,tree,k=1)
    avg_ms=(time.perf_counter()-t0)/N*1000
    log.info(f"  KD-Tree │ N={len(records)} │ {avg_ms:.4f}ms/q ({1000/avg_ms:.0f} qps)")

    # ── Report ─────────────────────────────────────────────────────────────
    SEP="═"*70
    print(f"\n{SEP}")
    print("  ██████  VALIDATION RESULTS — TRC-SD v2  ██████")
    print(SEP)
    print(f"\n  {'Database':<10}  {'Genuine':<10}  {'Impostor':<10}  "
          f"{'EER':>7}  {'Thresh':>8}  {'FAR@EER':>8}  {'FRR@EER':>8}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
    all_eers=[]
    for db_name, res in all_results.items():
        gen=res["genuine"]; imp=res["impostor"]
        eer=res["eer"];     et =res["eer_thresh"]
        fe=(imp>=et).mean(); fr=(gen<et).mean()
        all_eers.append(eer)
        print(f"  {db_name:<10}  {len(gen):<10}  {len(imp):<10}  "
              f"{eer*100:>6.2f}%  {et:>8.4f}  {fe*100:>7.2f}%  {fr*100:>7.2f}%")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
    mean_eer=np.mean(all_eers) if all_eers else float("nan")
    print(f"  {'AVERAGE':<10}  {'':10}  {'':10}  {mean_eer*100:>6.2f}%\n")

    t_v=[m["t_total"]    for m in timing_log if "t_total"    in m]
    m_v=[m["n_minutiae"] for m in timing_log if "n_minutiae" in m]
    e_v=[m["n_edges"]    for m in timing_log if "n_edges"    in m]

    print(SEP); print("  PIPELINE v2 STATISTICS"); print(SEP)
    print(f"  Images processed         : {len(records)}")
    print(f"  Minutiae / image          : min={min(m_v):.0f}  mean={np.mean(m_v):.1f}  max={max(m_v):.0f}")
    print(f"  Pruned Delaunay edges/img : min={min(e_v):.0f}  mean={np.mean(e_v):.1f}  max={max(e_v):.0f}")
    print(f"  Processing time / image   : min={min(t_v)*1e3:.1f}ms  mean={np.mean(t_v)*1e3:.1f}ms")
    print(f"  Descriptor dim            : {descs.shape[1]}  "
          f"= Global({4*GLOBAL_K}) + Local({4*DESC_SIZE}) + Hist({RIDGE_HIST_BINS}) + Struct(9)")
    print(f"  1:N KD-Tree latency       : {avg_ms:.4f}ms  (~{1000/avg_ms:.0f} qps)\n")

    print(SEP); print("  v1 → v2 IMPROVEMENT SUMMARY"); print(SEP)
    fixes = [
        ("#1","Gabor filter bank (8 orientations)","Eliminates merged-blob skeleton from v1"),
        ("#2","Adaptive threshold (not global Otsu)","Handles local contrast variation"),
        ("#3","Branch-length quality filter (≥8px)","Removes spurious short-branch minutiae"),
        ("#4","Pruned Delaunay (≤80px edges)","Removes spurious cross-fingerprint connections"),
        ("#5","Global Lsym Fiedler embedding","PRIMARY FIX: encodes full finger topology"),
        ("#6","Normalised Laplacian k=3 hop","Stable eigenvalue range [0,2], richer context"),
        ("#7","Ridge-count histogram (15 bins)","Distribution-level distortion invariant"),
        ("#8","Structural feature vector (9d)","Degree/bifurcation statistics"),
        ("#9","L2-normalised final descriptor","Correct cosine similarity [-1,1]"),
    ]
    print(f"  {'Fix':<5}  {'What changed':<40}  Why it helps")
    print(f"  {'─'*5}  {'─'*40}  {'─'*35}")
    for fix, what, why in fixes:
        print(f"  {fix:<5}  {what:<40}  {why}")
    print()

    print(SEP); print("  OUTPUT FILES"); print(SEP)
    print(f"  Steps  → {SAMPLE_DIR}/")
    for f in sorted(os.listdir(SAMPLE_DIR)):
        print(f"    • {f}")
    print(f"  ROC    → {OUTPUT_DIR}/roc_*_v2.png")
    print(f"  Log    → {OUTPUT_DIR}/pipeline_v2.log")
    print(SEP+"\n")


if __name__ == "__main__":
    main()
