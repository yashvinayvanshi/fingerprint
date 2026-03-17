import os
import glob
import logging
import math
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.draw import line
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from collections import Counter

# Configure Progress Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_fvc_dataset(base_path):
    """Loads image paths and finger IDs from the FVC dataset structure."""
    image_paths = []
    search_pattern = os.path.join(base_path, "DB*_B", "*.*")
    files = glob.glob(search_pattern)
    
    for file in files:
        filename = os.path.basename(file)
        try:
            finger_id = filename.split('_')[0]
            image_paths.append((file, finger_id))
        except IndexError:
            continue
            
    logging.info(f"Loaded {len(image_paths)} images from dataset.")
    return image_paths

def get_roi_mask(image):
    """
    Generates a binary Region of Interest (ROI) mask to segment the 
    fingerprint from the background noise.
    """
    # 1. Blur heavily to merge the ridges together
    blur = cv2.GaussianBlur(image, (11, 11), 0)
    
    # 2. Threshold to get the general foreground
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Morphological operations to create a solid blob (close holes, open noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 4. Erode the mask to trim away the noisy outer boundary of the print
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask

def preprocess_and_thin(image_path, save_debug=False, debug_dir="./debug"):
    """Phase 1.1: Enhancement, ROI Masking, and Morphological thinning."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply the ROI Mask to clear background noise
    roi_mask = get_roi_mask(img)
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask)
    
    # Clear the borders just in case the ROI touches the edge
    margin = 15
    binary[:margin, :] = 0        
    binary[-margin:, :] = 0       
    binary[:, :margin] = 0        
    binary[:, -margin:] = 0       
    
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        plt.imsave(os.path.join(debug_dir, "1_original.png"), img, cmap='gray')
        plt.imsave(os.path.join(debug_dir, "2_roi_mask.png"), roi_mask, cmap='gray')
        plt.imsave(os.path.join(debug_dir, "3_skeleton.png"), skeleton, cmap='gray')
        
    return skeleton

def extract_minutiae_crossing_number(skeleton):
    """Extracts true ridge endings and bifurcations using the 3x3 Crossing Number method."""
    skeleton_bool = skeleton > 0
    minutiae = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    rows, cols = skeleton_bool.shape
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if skeleton_bool[r, c]:
                p = [skeleton_bool[r + dr, c + dc] for dr, dc in neighbors]
                transitions = sum(1 for i in range(8) if p[i] != p[i+1])
                cn = transitions / 2
                
                # CN == 1 (Ending), CN == 3 (Bifurcation)
                if cn == 1 or cn == 3:
                    minutiae.append((c, r)) 
    return minutiae

def filter_minutiae(minutiae, image_shape, border_dist=20, min_dist=10):
    """Filters spurious minutiae near borders and merges dense noise clusters."""
    filtered = []
    rows, cols = image_shape
    
    # 1. Border Filtering
    for x, y in minutiae:
        if (x > border_dist and x < cols - border_dist and 
            y > border_dist and y < rows - border_dist):
            filtered.append((x, y))
            
    # 2. Greedy Distance Filtering
    final_minutiae = []
    for p in filtered:
        is_too_close = False
        for kept_p in final_minutiae:
            dist = math.hypot(p[0] - kept_p[0], p[1] - kept_p[1])
            if dist < min_dist:
                is_too_close = True
                break
        
        if not is_too_close:
            final_minutiae.append(p)
            
    return final_minutiae

def build_delaunay_graph(minutiae, save_debug=False, debug_dir="./debug"):
    """Phase 1.2: Applies Delaunay Triangulation to form the localized mesh."""
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
        plt.plot(points[:, 0], points[:, 1], 'ro', markersize=3)
        plt.gca().invert_yaxis()
        plt.title(f"Filtered Minutiae ({len(points)}) & Delaunay Topology")
        plt.savefig(os.path.join(debug_dir, "4_delaunay.png"))
        plt.close()
        
    return list(edges)

def calculate_ridge_counts(skeleton, minutiae, edges):
    """Phase 2: Computes distortion-proof edge weights via ridge counting."""
    ridge_counts = {}
    for (i, j) in edges:
        x0, y0 = minutiae[i]
        x1, y1 = minutiae[j]
        
        rr, cc = line(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < skeleton.shape[0]) & (cc >= 0) & (cc < skeleton.shape[1])
        rr, cc = rr[valid], cc[valid]
        
        line_pixels = skeleton[rr, cc]
        transitions = np.sum(np.diff(line_pixels.astype(int)) > 0)
        ridge_counts[(i, j)] = transitions
        
    return ridge_counts

def extract_spectral_descriptors(minutiae, edges, ridge_counts, k_hops=2, desc_len=5):
    """Phase 3: Generates localized invariant vectors (eigenvalues of Laplacian)."""
    G = nx.Graph()
    for idx, pt in enumerate(minutiae):
        G.add_node(idx, pos=pt)
    for (i, j), weight in ridge_counts.items():
        G.add_edge(i, j, weight=weight)
        
    descriptors = []
    for node in G.nodes():
        subgraph_nodes = nx.single_source_shortest_path_length(G, node, cutoff=k_hops).keys()
        subgraph = G.subgraph(subgraph_nodes)
        
        if len(subgraph) < 3: 
            continue
            
        L = nx.laplacian_matrix(subgraph, weight='weight').todense()
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        
        if len(eigenvalues) >= desc_len:
            descriptor = eigenvalues[:desc_len]
        else:
            descriptor = np.pad(eigenvalues, (0, desc_len - len(eigenvalues)), 'constant')
            
        descriptors.append(descriptor)
    return descriptors

def build_database_index(dataset_paths):
    """Phase 4.1: Vectorizes enrollment images and builds O(logN) spatial index."""
    all_descriptors = []
    descriptor_labels = []
    first_sample = True
    
    for path, finger_id in dataset_paths:
        skeleton = preprocess_and_thin(path, save_debug=first_sample)
        if skeleton is None: continue
            
        raw_minutiae = extract_minutiae_crossing_number(skeleton)
        minutiae = filter_minutiae(raw_minutiae, skeleton.shape)
        edges = build_delaunay_graph(minutiae, save_debug=first_sample)
        ridge_counts = calculate_ridge_counts(skeleton, minutiae, edges)
        descriptors = extract_spectral_descriptors(minutiae, edges, ridge_counts)
        
        for desc in descriptors:
            all_descriptors.append(desc)
            descriptor_labels.append(finger_id)
            
        if first_sample:
            logging.info(f"Saved intermediate debug images for sample: {path}")
            first_sample = False
            
    X = np.array(all_descriptors)
    tree = KDTree(X, metric='euclidean')
    return tree, X, np.array(descriptor_labels)

def match_fingerprint_query(query_path, kd_tree, database_labels, max_l2_distance=0.4):
    """Phase 4.2: Matches query image against index with strict FRR/FAR thresholding."""
    skeleton = preprocess_and_thin(query_path)
    if skeleton is None: return None
    
    raw_minutiae = extract_minutiae_crossing_number(skeleton)
    minutiae = filter_minutiae(raw_minutiae, skeleton.shape)
    edges = build_delaunay_graph(minutiae)
    ridge_counts = calculate_ridge_counts(skeleton, minutiae, edges)
    query_descriptors = extract_spectral_descriptors(minutiae, edges, ridge_counts)
    
    if not query_descriptors:
        return None
        
    query_vectors = np.array(query_descriptors)
    distances, indices = kd_tree.query(query_vectors, k=3)
    
    votes = []
    for i, idx_list in enumerate(indices):
        for j, db_idx in enumerate(idx_list):
            if distances[i][j] < max_l2_distance:
                votes.append(database_labels[db_idx])
                
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
        
    # Split: Enrollment (1-6) and Query (7-8)
    enrollment_set = [item for item in all_images if int(os.path.basename(item[0]).split('_')[1].split('.')[0]) <= 6]
    query_set = [item for item in all_images if int(os.path.basename(item[0]).split('_')[1].split('.')[0]) > 6]
    
    logging.info(f"Building KD-Tree index with {len(enrollment_set)} enrollment images...")
    kd_tree, db_vectors, db_labels = build_database_index(enrollment_set)
    logging.info(f"Indexing complete. Total Spectral Descriptors stored: {len(db_vectors)}")
    
    logging.info("Running Validation on Query Set...")
    correct_matches = 0
    total_queries = len(query_set)
    
    # Distance threshold for matching - can be tuned!
    distance_threshold = 0.4 
    
    for q_path, true_id in query_set:
        predicted_id = match_fingerprint_query(q_path, kd_tree, db_labels, max_l2_distance=distance_threshold)
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