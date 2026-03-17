import os
import glob
import logging
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin
from skimage.draw import line
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from collections import Counter

# Configure Progress Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_fvc_dataset(base_path):
    """
    Input: String path to the base dataset directory (e.g., './datasets/PAMI Lab').
    Output: List of tuples (file_path, finger_id).
    """
    image_paths = []
    # Search through DB1_B, DB2_B, DB3_B, DB4_B
    search_pattern = os.path.join(base_path, "DB*_B", "*.*")
    files = glob.glob(search_pattern)
    
    for file in files:
        filename = os.path.basename(file)
        # Assuming standard FVC naming convention: '101_1.tif' -> ID 101, Impression 1
        try:
            finger_id = filename.split('_')[0]
            image_paths.append((file, finger_id))
        except IndexError:
            continue
            
    logging.info(f"Loaded {len(image_paths)} images from dataset.")
    return image_paths

def preprocess_and_thin(image_path, save_debug=False, debug_dir="./debug"):
    """
    Phase 1.1: Standard robust edge-detection and morphological thinning.
    Input: Path to the image file.
    Output: Binary thinned skeleton image (numpy array).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Enhance contrast and binarize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological thinning to 1-pixel wide ridges
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        plt.imsave(os.path.join(debug_dir, "1_original.png"), img, cmap='gray')
        plt.imsave(os.path.join(debug_dir, "2_skeleton.png"), skeleton, cmap='gray')
        
    return skeleton

def extract_minutiae(skeleton, max_points=100):
    """
    Phase 1.1 (cont): Isolate minutiae points (vertices V).
    Input: Thinned boolean image.
    Output: List of (x, y) coordinates representing nodes.
    """
    # For demonstration, we use a proxy for minutiae (Harris corners on skeleton)
    # A true minutiae extractor would use a 3x3 crossing number algorithm.
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(skeleton_uint8, max_points, 0.01, 10)
    
    if corners is not None:
        minutiae = [tuple(map(int, pt[0])) for pt in corners]
        return minutiae
    return []

def build_delaunay_graph(minutiae, save_debug=False, debug_dir="./debug"):
    """
    Phase 1.2: Apply Delaunay Triangulation to form edges E.
    Input: List of (x,y) minutiae coordinates.
    Output: List of edges (index1, index2).
    """
    if len(minutiae) < 4:
        return []
        
    points = np.array(minutiae)
    tri = Delaunay(points)
    
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted([simplex[0], simplex[1]])))
        edges.add(tuple(sorted([simplex[1], simplex[2]])))
        edges.add(tuple(sorted([simplex[2], simplex[0]])))
        
    if save_debug:
        plt.figure()
        plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
        plt.plot(points[:, 0], points[:, 1], 'ro')
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(debug_dir, "3_delaunay.png"))
        plt.close()
        
    return list(edges)

def calculate_ridge_counts(skeleton, minutiae, edges):
    """
    Phase 2: Distortion-Proof Edge Weights (Ridge Counting).
    Input: Skeleton image, minutiae points, edges.
    Output: Dictionary mapping edge (i, j) to an integer ridge count.
    """
    ridge_counts = {}
    for (i, j) in edges:
        x0, y0 = minutiae[i]
        x1, y1 = minutiae[j]
        
        # Get pixels along the line between two minutiae
        rr, cc = line(y0, x0, y1, x1)
        
        # Ensure coordinates are within image bounds
        valid = (rr >= 0) & (rr < skeleton.shape[0]) & (cc >= 0) & (cc < skeleton.shape[1])
        rr, cc = rr[valid], cc[valid]
        
        line_pixels = skeleton[rr, cc]
        
        # Count transitions from 0 to 1 (ridge crossings)
        transitions = np.sum(np.diff(line_pixels.astype(int)) > 0)
        ridge_counts[(i, j)] = transitions
        
    return ridge_counts

def extract_spectral_descriptors(minutiae, edges, ridge_counts, k_hops=2, desc_len=5):
    """
    Phase 3: Spectral Description of local subgraphs.
    Input: Graph data and parameters.
    Output: List of fixed-length numerical vectors (Spectral Descriptors).
    """
    G = nx.Graph()
    for idx, pt in enumerate(minutiae):
        G.add_node(idx, pos=pt)
        
    for (i, j), weight in ridge_counts.items():
        G.add_edge(i, j, weight=weight)
        
    descriptors = []
    
    for node in G.nodes():
        # 1. Extract local k-hop neighborhood subgraph
        subgraph_nodes = nx.single_source_shortest_path_length(G, node, cutoff=k_hops).keys()
        subgraph = G.subgraph(subgraph_nodes)
        
        if len(subgraph) < 3: # Ignore trivially small subgraphs
            continue
            
        # 2. Compute Graph Laplacian L = D - A
        L = nx.laplacian_matrix(subgraph, weight='weight').todense()
        
        # 3. Extract ordered eigenvalues
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)
        
        # Pad or truncate to fixed length
        if len(eigenvalues) >= desc_len:
            descriptor = eigenvalues[:desc_len]
        else:
            descriptor = np.pad(eigenvalues, (0, desc_len - len(eigenvalues)), 'constant')
            
        descriptors.append(descriptor)
        
    return descriptors

def build_database_index(dataset_paths):
    """
    Phase 4.1: Convert all dataset images into descriptors and index them.
    Input: List of dataset tuples (path, finger_id).
    Output: KDTree, array of all descriptors, array of corresponding finger IDs.
    """
    all_descriptors = []
    descriptor_labels = []
    
    first_sample = True
    
    for path, finger_id in dataset_paths:
        skeleton = preprocess_and_thin(path, save_debug=first_sample)
        if skeleton is None:
            continue
            
        minutiae = extract_minutiae(skeleton)
        edges = build_delaunay_graph(minutiae, save_debug=first_sample)
        ridge_counts = calculate_ridge_counts(skeleton, minutiae, edges)
        
        descriptors = extract_spectral_descriptors(minutiae, edges, ridge_counts)
        
        for desc in descriptors:
            all_descriptors.append(desc)
            descriptor_labels.append(finger_id)
            
        if first_sample:
            logging.info(f"Saved intermediate debug images for sample: {path}")
            first_sample = False
            
    if not all_descriptors:
        raise ValueError("No descriptors could be extracted from the dataset.")
        
    X = np.array(all_descriptors)
    tree = KDTree(X, metric='euclidean') # Phase 4: Ultrafast Vector Matching
    
    return tree, X, np.array(descriptor_labels)

def match_fingerprint_query(query_path, kd_tree, database_labels):
    """
    Phase 4.2: Match a single query image against the KD-Tree.
    Input: Path to query image, the populated KDTree, and the DB labels.
    Output: Predicted Finger ID.
    """
    skeleton = preprocess_and_thin(query_path)
    if skeleton is None: return None
    
    minutiae = extract_minutiae(skeleton)
    edges = build_delaunay_graph(minutiae)
    ridge_counts = calculate_ridge_counts(skeleton, minutiae, edges)
    query_descriptors = extract_spectral_descriptors(minutiae, edges, ridge_counts)
    
    if not query_descriptors:
        return None
        
    # Search tree in O(logN) time
    query_vectors = np.array(query_descriptors)
    distances, indices = kd_tree.query(query_vectors, k=3)
    
    # Majority vote from nearest neighbors across all descriptors
    votes = []
    for idx_list in indices:
        for idx in idx_list:
            votes.append(database_labels[idx])
            
    if votes:
        most_common = Counter(votes).most_common(1)[0][0]
        return most_common
    return None

def main():
    logging.info("Starting TRC-SD Fingerprint Pipeline...")
    
    dataset_path = "./datasets/PAMI Lab"
    all_images = load_fvc_dataset(dataset_path)
    
    if not all_images:
        logging.error("No images found. Please check dataset path.")
        return
        
    # Validation Setup: Split into enrollment (impression 1-6) and query (impression 7-8)
    enrollment_set = [item for item in all_images if int(os.path.basename(item[0]).split('_')[1].split('.')[0]) <= 6]
    query_set = [item for item in all_images if int(os.path.basename(item[0]).split('_')[1].split('.')[0]) > 6]
    
    logging.info(f"Building KD-Tree index with {len(enrollment_set)} enrollment images...")
    kd_tree, db_vectors, db_labels = build_database_index(enrollment_set)
    logging.info(f"Indexing complete. Total Spectral Descriptors stored: {len(db_vectors)}")
    
    logging.info("Running Validation on Query Set...")
    correct_matches = 0
    total_queries = len(query_set)
    
    for q_path, true_id in query_set:
        predicted_id = match_fingerprint_query(q_path, kd_tree, db_labels)
        if predicted_id == true_id:
            correct_matches += 1
            
    accuracy = (correct_matches / total_queries) * 100 if total_queries > 0 else 0
    
    print("\n" + "="*40)
    print("         VALIDATION RESULTS")
    print("="*40)
    print(f"Total Queries Executed: {total_queries}")
    print(f"Correct Identifications: {correct_matches}")
    print(f"Rank-1 Accuracy: {accuracy:.2f}%")
    print(f"Computational complexity per descriptor match: O(log N)")
    print("="*40)

if __name__ == "__main__":
    main()