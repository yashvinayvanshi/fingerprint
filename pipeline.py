"""
Fingerprint Recognition using Topological Ridge Count Graph + Spectral Descriptors

Pipeline:
1. Image preprocessing
2. Minutiae extraction
3. Graph construction (Delaunay triangulation)
4. Ridge count weighting
5. Local k-hop subgraphs
6. Spectral descriptors (Laplacian eigenvalues)
7. KD-tree indexing
8. Matching + evaluation metrics

Dataset expected at:
./datasets/PAMI Lab/DB*_B
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from sklearn.metrics import roc_curve, auc
from skimage.morphology import skeletonize
from skimage.feature import canny
from skimage.util import img_as_float
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

DATASET_PATH = "./datasets/PAMI Lab"

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

def load_dataset(dataset_path):
    """
    Loads fingerprint images.

    Input:
        dataset_path: root dataset directory

    Output:
        images: list of (image, label, path)
    """

    images = []

    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".tif") or f.endswith(".png") or f.endswith(".jpg"):
                path = os.path.join(root, f)

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # label = finger ID (first part of filename)
                label = f.split("_")[0]

                images.append((img, label, path))

    logging.info(f"Loaded {len(images)} fingerprint images")

    return images


# ------------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------------

def preprocess_image(img):
    """
    Enhance fingerprint image.

    Input:
        img: grayscale fingerprint image

    Output:
        binary ridge map
    """

    img = cv2.equalizeHist(img)

    edges = canny(img_as_float(img), sigma=2)

    skeleton = skeletonize(edges)

    return skeleton.astype(np.uint8)


# ------------------------------------------------------------
# 3. MINUTIAE EXTRACTION
# ------------------------------------------------------------

def extract_minutiae(skeleton):
    """
    Extract ridge endings and bifurcations.

    Input:
        skeleton image

    Output:
        list of minutiae points [(x,y)]
    """

    minutiae = []

    h, w = skeleton.shape

    for y in range(1, h-1):
        for x in range(1, w-1):

            if skeleton[y, x]:

                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1

                if neighbors == 1 or neighbors >= 3:
                    minutiae.append((x, y))

    return np.array(minutiae)


# ------------------------------------------------------------
# 4. DELAUNAY GRAPH
# ------------------------------------------------------------

def build_delaunay_graph(minutiae):
    """
    Build graph using Delaunay triangulation.

    Input:
        minutiae points

    Output:
        NetworkX graph
    """

    tri = Delaunay(minutiae)

    G = nx.Graph()

    for i, p in enumerate(minutiae):
        G.add_node(i, pos=p)

    for simplex in tri.simplices:

        for i in range(3):
            a = simplex[i]
            b = simplex[(i+1) % 3]

            G.add_edge(a, b)

    return G


# ------------------------------------------------------------
# 5. RIDGE COUNT WEIGHTING
# ------------------------------------------------------------

def ridge_count(img, p1, p2):
    """
    Estimate ridge count between two points.

    Input:
        img: skeleton image
        p1,p2: points

    Output:
        ridge count
    """

    line = np.linspace(p1, p2, 100)

    count = 0

    for x, y in line:
        x = int(x)
        y = int(y)

        if img[y, x]:
            count += 1

    return count


def weight_graph(G, skeleton):
    """
    Assign ridge count weights to graph edges.

    Input:
        G: graph
        skeleton: skeleton image

    Output:
        weighted graph
    """

    for u, v in G.edges():

        p1 = G.nodes[u]["pos"]
        p2 = G.nodes[v]["pos"]

        rc = ridge_count(skeleton, p1, p2)

        G[u][v]["weight"] = rc

    return G


# ------------------------------------------------------------
# 6. LOCAL SUBGRAPH
# ------------------------------------------------------------

def extract_khop_subgraph(G, node, k=2):
    """
    Extract k-hop neighborhood.

    Input:
        G: graph
        node: center node
        k: hop size

    Output:
        subgraph
    """

    nodes = nx.single_source_shortest_path_length(G, node, cutoff=k).keys()

    return G.subgraph(nodes)


# ------------------------------------------------------------
# 7. SPECTRAL DESCRIPTOR
# ------------------------------------------------------------

def spectral_descriptor(subgraph, k=6):
    """
    Compute eigenvalues of graph Laplacian.

    Input:
        subgraph
        k: descriptor length

    Output:
        descriptor vector
    """

    A = nx.to_numpy_array(subgraph, weight="weight")

    D = np.diag(A.sum(axis=1))

    L = D - A

    eigvals = np.linalg.eigvals(L)

    eigvals = np.sort(np.real(eigvals))

    if len(eigvals) < k:
        eigvals = np.pad(eigvals, (0, k-len(eigvals)))

    return eigvals[:k]


# ------------------------------------------------------------
# 8. FINGERPRINT DESCRIPTOR
# ------------------------------------------------------------

def fingerprint_descriptor(G):
    """
    Generate descriptors for fingerprint.

    Input:
        graph

    Output:
        array of spectral descriptors
    """

    desc = []

    for node in G.nodes():

        sg = extract_khop_subgraph(G, node, k=2)

        d = spectral_descriptor(sg)

        desc.append(d)

    return np.array(desc)


# ------------------------------------------------------------
# 9. MATCHING
# ------------------------------------------------------------

def build_index(descriptors):
    """
    Build KD-tree index.

    Input:
        descriptor list

    Output:
        KDTree
    """

    all_desc = np.vstack(descriptors)

    tree = KDTree(all_desc)

    return tree


def match(desc1, desc2):
    """
    Compute similarity between fingerprints.

    Input:
        descriptor sets

    Output:
        similarity score
    """

    from scipy.spatial.distance import cdist

    d = cdist(desc1, desc2)

    return np.min(d)


# ------------------------------------------------------------
# 10. VISUALIZATION
# ------------------------------------------------------------

def visualize_pipeline(img, skeleton, minutiae, G):

    plt.figure(figsize=(12,4))

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title("Fingerprint")

    plt.subplot(132)
    plt.imshow(skeleton, cmap='gray')
    plt.title("Skeleton")

    plt.subplot(133)
    plt.imshow(img, cmap='gray')

    xs, ys = minutiae[:,0], minutiae[:,1]

    plt.scatter(xs, ys, s=5, c='red')

    plt.title("Minutiae")

    plt.savefig(f"{OUTPUT_DIR}/pipeline.png")

    plt.show()


# ------------------------------------------------------------
# 11. EVALUATION
# ------------------------------------------------------------

def evaluate(matches, labels):

    scores = []
    truths = []

    for i in range(len(matches)):
        for j in range(i+1, len(matches)):

            score = match(matches[i], matches[j])

            scores.append(score)

            truths.append(labels[i] == labels[j])

    fpr, tpr, _ = roc_curve(truths, -np.array(scores))

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.title("ROC Curve")

    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")

    logging.info(f"AUC = {roc_auc}")

    return roc_auc


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main():

    dataset = load_dataset(DATASET_PATH)

    descriptors = []
    labels = []

    for img, label, path in tqdm(dataset[:50]):  # limit samples for demo

        skel = preprocess_image(img)

        minutiae = extract_minutiae(skel)

        if len(minutiae) < 10:
            continue

        G = build_delaunay_graph(minutiae)

        G = weight_graph(G, skel)

        desc = fingerprint_descriptor(G)

        descriptors.append(desc)
        labels.append(label)

        visualize_pipeline(img, skel, minutiae, G)

    logging.info("Building index")

    tree = build_index(descriptors)

    auc_score = evaluate(descriptors, labels)

    logging.info("Finished pipeline")
    logging.info(f"AUC Score: {auc_score}")


if __name__ == "__main__":
    main()