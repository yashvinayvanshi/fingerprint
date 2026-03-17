"""
Fingerprint Recognition Pipeline
Topological Ridge Count Graph + Spectral Descriptors

Dataset structure expected:

datasets/
   PAMI Lab/
        DB1_B/
        DB2_B/
        DB3_B/
        DB4_B/
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
from tqdm import tqdm
import logging

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATASET_PATH = "./datasets/PAMI Lab"
OUTPUT_DIR = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

def load_dataset(path):

    images = []

    for root, _, files in os.walk(path):

        for f in files:

            if f.endswith(".tif") or f.endswith(".png"):

                full = os.path.join(root, f)

                img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)

                label = f.split("_")[0]

                images.append((img, label, full))

    logging.info(f"Loaded {len(images)} fingerprint images")

    return images


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

def preprocess_image(img):

    # normalize contrast
    img = cv2.equalizeHist(img)

    # smooth
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # ridge segmentation
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # remove noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # skeletonize
    skeleton = skeletonize(binary.astype(bool))

    return img, binary, skeleton.astype(np.uint8)


# ------------------------------------------------------------
# MINUTIAE DETECTION (Crossing Number)
# ------------------------------------------------------------

def extract_minutiae(skel):

    minutiae = []

    h, w = skel.shape

    for y in range(1, h-1):
        for x in range(1, w-1):

            if skel[y, x] == 1:

                n = [
                    int(skel[y-1,x]),
                    int(skel[y-1,x+1]),
                    int(skel[y,x+1]),
                    int(skel[y+1,x+1]),
                    int(skel[y+1,x]),
                    int(skel[y+1,x-1]),
                    int(skel[y,x-1]),
                    int(skel[y-1,x-1])
                ]

                transitions = 0

                for i in range(8):
                    transitions += abs(n[i] - n[(i+1)%8])

                transitions = transitions // 2

                if transitions == 1 or transitions == 3:
                    minutiae.append((x,y))

    minutiae = np.array(minutiae)

    if len(minutiae) > 0:
        minutiae = np.unique(minutiae, axis=0)

    return minutiae


# ------------------------------------------------------------
# REMOVE BORDER MINUTIAE
# ------------------------------------------------------------

def clean_minutiae(minutiae, shape):

    if len(minutiae) == 0:
        return minutiae

    h, w = shape

    filtered = []

    for x,y in minutiae:

        if 20 < x < w-20 and 20 < y < h-20:
            filtered.append((x,y))

    return np.array(filtered)


# ------------------------------------------------------------
# DELAUNAY GRAPH
# ------------------------------------------------------------

def build_delaunay_graph(minutiae):

    tri = Delaunay(minutiae)

    G = nx.Graph()

    for i,p in enumerate(minutiae):
        G.add_node(i, pos=p)

    for simplex in tri.simplices:

        for i in range(3):

            a = simplex[i]
            b = simplex[(i+1)%3]

            G.add_edge(a,b)

    return G


# ------------------------------------------------------------
# RIDGE COUNT
# ------------------------------------------------------------

def ridge_count(skel, p1, p2):

    samples = 50

    xs = np.linspace(p1[0], p2[0], samples).astype(int)
    ys = np.linspace(p1[1], p2[1], samples).astype(int)

    xs = np.clip(xs,0,skel.shape[1]-1)
    ys = np.clip(ys,0,skel.shape[0]-1)

    vals = skel[ys,xs]

    crossings = np.sum(np.abs(np.diff(vals)))

    return crossings


def weight_graph(G, skel):

    for u,v in G.edges():

        p1 = G.nodes[u]["pos"]
        p2 = G.nodes[v]["pos"]

        w = ridge_count(skel,p1,p2)

        G[u][v]["weight"] = w + 1e-5

    return G


# ------------------------------------------------------------
# LOCAL SUBGRAPH
# ------------------------------------------------------------

def khop_subgraph(G,node,k=2):

    nodes = nx.single_source_shortest_path_length(G,node,k).keys()

    return G.subgraph(nodes)


# ------------------------------------------------------------
# SPECTRAL DESCRIPTOR
# ------------------------------------------------------------

def spectral_descriptor(subgraph,k=25):

    A = nx.to_numpy_array(subgraph,weight="weight")

    if A.shape[0] < 2:
        return np.zeros(k)

    D = np.diag(A.sum(axis=1))

    L = D - A

    eigvals = np.sort(np.real(np.linalg.eigvals(L)))

    if len(eigvals) < k:
        eigvals = np.pad(eigvals,(0,k-len(eigvals)))

    desc = eigvals[:k]

    desc = desc / (np.linalg.norm(desc)+1e-8)

    return desc


# ------------------------------------------------------------
# FINGERPRINT DESCRIPTOR
# ------------------------------------------------------------

def fingerprint_descriptor(G):

    desc = []

    for node in G.nodes():

        sg = khop_subgraph(G,node)

        d = spectral_descriptor(sg)

        desc.append(d)

    return np.mean(desc,axis=0)


# ------------------------------------------------------------
# SAVE VISUALIZATION
# ------------------------------------------------------------

def save_sample(img,binary,skel,minutiae,G):

    plt.figure(figsize=(16,6))

    plt.subplot(231)
    plt.imshow(img,cmap="gray")
    plt.title("Fingerprint")

    plt.subplot(232)
    plt.imshow(binary,cmap="gray")
    plt.title("Binary Ridges")

    plt.subplot(233)
    plt.imshow(skel,cmap="gray")
    plt.title("Skeleton")

    plt.subplot(234)
    plt.imshow(img,cmap="gray")
    plt.scatter(minutiae[:,0],minutiae[:,1],s=5,c="red")
    plt.title("Minutiae")

    plt.subplot(235)

    pos = nx.get_node_attributes(G,"pos")
    nx.draw(G,pos,node_size=5)

    plt.title("Delaunay Graph")

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/sample_pipeline.png")

    plt.close()


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def evaluate(descriptors,labels):

    scores = []
    truths = []

    n = len(descriptors)

    for i in range(n):

        for j in range(i+1,n):

            d = np.linalg.norm(descriptors[i]-descriptors[j])

            scores.append(d)

            truths.append(labels[i]==labels[j])

    fpr,tpr,_ = roc_curve(truths,-np.array(scores))

    roc_auc = auc(fpr,tpr)

    plt.figure()

    plt.plot(fpr,tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")

    logging.info(f"AUC = {roc_auc:.3f}")

    return roc_auc


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    dataset = load_dataset(DATASET_PATH)

    descriptors = []
    labels = []

    sample_saved = False

    logging.info("Starting feature extraction")

    for img,label,_ in tqdm(dataset):

        img,binary,skel = preprocess_image(img)

        logging.info(f"Skeleton pixels: {np.sum(skel)}")

        minutiae = extract_minutiae(skel)

        minutiae = clean_minutiae(minutiae,img.shape)

        logging.info(f"Minutiae detected: {len(minutiae)}")

        if len(minutiae) < 10:
            continue

        G = build_delaunay_graph(minutiae)

        G = weight_graph(G,skel)

        desc = fingerprint_descriptor(G)

        descriptors.append(desc)
        labels.append(label)

        if not sample_saved:

            save_sample(img,binary,skel,minutiae,G)

            sample_saved = True

    if len(descriptors) == 0:

        logging.error("No valid fingerprints processed")

        return

    logging.info(f"Valid fingerprints processed: {len(descriptors)}")

    logging.info("Building KDTree index")

    tree = KDTree(np.array(descriptors))

    logging.info("Evaluating recognition performance")

    auc_score = evaluate(descriptors,labels)

    logging.info(f"Final AUC Score: {auc_score:.3f}")

    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()