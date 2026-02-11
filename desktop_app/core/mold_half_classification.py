"""
Mold Half Classification

Classifies boundary triangles of a mold cavity into two mold halves (H₁ and H₂)
based on parting directions.

Algorithm:
1. Identify outer boundary triangles by matching against hull geometry
2. Initial classification using directional scoring (dot product with parting directions)
3. Morphological opening (erosion + dilation) to remove thin peninsulas
4. Laplacian smoothing with 2-ring neighborhood for smooth boundaries
5. Orphan region removal using connected component analysis
6. Re-apply directional constraints for strong alignments

Color convention:
- H₁ (Green): Triangles whose normals align with d1 (dot(n, d1) >= 0)
- H₂ (Orange): Triangles whose normals align with d2 (dot(n, d2) >= 0)
- Inner boundary (Dark gray): Triangles from the original part surface

Algorithm matches the React frontend implementation exactly.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple

import numpy as np
import trimesh

# Check for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Check for C++ acceleration
try:
    from . import fast_algorithms as _cpp
    CPP_FAST_AVAILABLE = True
except ImportError:
    CPP_FAST_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Classification thresholds
STRONG_ALIGNMENT_THRESHOLD = 0.7      # Dot product threshold for strong directional alignment
STRONG_ALIGNMENT_MARGIN = 0.3         # Required margin over competing direction
OUTER_BOUNDARY_TOLERANCE_FRACTION = 0.001  # 0.1% of hull size for outer boundary detection
TET_OUTER_BOUNDARY_TOLERANCE_FRACTION = 0.02  # 2% of hull size for tet-based detection

# Boundary zone parameters
DEFAULT_BOUNDARY_ZONE_THRESHOLD = 0.05  # Default threshold for boundary zone (5% of bbox diagonal per paper)

# Area-based filtering
ORPHAN_REGION_AREA_THRESHOLD = 0.02   # 2% of total area - regions smaller are orphans

# Minimum outer triangle fraction before warning
MIN_OUTER_TRIANGLE_FRACTION = 0.1     # Warn if <10% of triangles are outer


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MoldHalfClassificationResult:
    """Result of mold half classification."""
    
    # Map from triangle index to mold half (1 or 2), only for outer boundary triangles
    side_map: Dict[int, int]
    
    # Triangle indices belonging to mold half 1 (H₁)
    h1_triangles: Set[int]
    
    # Triangle indices belonging to mold half 2 (H₂)
    h2_triangles: Set[int]
    
    # Triangle indices in the boundary zone between H₁ and H₂ (grey, no label)
    boundary_zone_triangles: Set[int]
    
    # Triangle indices belonging to inner boundary (part surface) - not classified
    inner_boundary_triangles: Set[int]
    
    # Total triangles in boundary mesh
    total_triangles: int
    
    # Total outer boundary triangles (H₁ + H₂ + boundary zone)
    outer_boundary_count: int


@dataclass
class TriangleInfo:
    """Information about a triangle."""
    normal: np.ndarray
    centroid: np.ndarray
    area: float


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_triangle_info(mesh: trimesh.Trimesh) -> List[TriangleInfo]:
    """
    Extract triangle information from a trimesh mesh.
    
    Returns array of normals, centroids, and areas for each triangle.
    """
    triangles = []
    
    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals
    
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        centroid = (v0 + v1 + v2) / 3.0
        
        # Compute area using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        
        triangles.append(TriangleInfo(
            normal=face_normals[i].copy(),
            centroid=centroid,
            area=area,
        ))
    
    return triangles


def build_triangle_adjacency(mesh: trimesh.Trimesh) -> Dict[int, List[int]]:
    """
    Build triangle adjacency map using trimesh's built-in face_adjacency.
    
    This is O(n) using trimesh's optimized implementation instead of
    the O(n²) string-based approach.
    
    Each triangle index maps to array of neighboring triangle indices
    (triangles sharing at least one edge).
    """
    n_faces = len(mesh.faces)
    
    # Initialize adjacency dict with empty lists
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n_faces)}
    
    # trimesh.face_adjacency is an (n, 2) array of pairs of adjacent face indices
    # This is computed efficiently using the half-edge structure
    face_adj = mesh.face_adjacency
    
    for i in range(len(face_adj)):
        a, b = face_adj[i]
        adjacency[a].append(b)
        adjacency[b].append(a)
    
    return adjacency


def classify_hull_faces_fast(
    hull_mesh: trimesh.Trimesh,
    d1: np.ndarray,
    d2: np.ndarray,
    boundary_zone_threshold: float = 0.05
) -> MoldHalfClassificationResult:
    """
    Fast hull face classification following the paper's algorithm exactly.
    
    Per paper Section 4.1: "Given the two parting directions, we partition the boundary ∂H
    into two parts, ∂H1 and ∂H2: we select the two faces F1 and F2 of ∂H whose 
    normals best align with d1 and d2 and then use a greedy region-growing 
    approach from F1 and F2 to assign faces to ∂H1 and ∂H2."
    
    Per paper: "We compute the shortest paths towards all vertices of ∂H, except for 
    those whose distance from the boundary between ∂H1 and ∂H2 is less than a fixed 
    threshold... we set the threshold to 15% of the convex hull bounding box diagonal."
    
    IMPORTANT: The boundary zone is defined by PHYSICAL DISTANCE, not hop count.
    This ensures consistent boundary zone width regardless of mesh density.
    
    Args:
        hull_mesh: The hull mesh (∂H) to partition
        d1: First parting direction (normalized)
        d2: Second parting direction (normalized)
        boundary_zone_threshold: Fraction of bbox diagonal for boundary zone (default 0.15 = 15%)
    
    Returns:
        MoldHalfClassificationResult with H1, H2, and boundary zone
    """
    import time
    start_time = time.time()
    
    n_faces = len(hull_mesh.faces)
    face_normals = hull_mesh.face_normals
    face_centroids = hull_mesh.triangles_center
    
    # Compute bounding box diagonal for distance threshold
    bbox_diagonal = np.linalg.norm(hull_mesh.bounds[1] - hull_mesh.bounds[0])
    distance_threshold = boundary_zone_threshold * bbox_diagonal
    
    # Normalize directions
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    # Compute alignment scores for all faces (vectorized)
    dot1 = face_normals @ d1  # (n_faces,)
    dot2 = face_normals @ d2  # (n_faces,)
    
    # Find seed faces: F1 = best alignment with d1, F2 = best alignment with d2
    f1 = int(np.argmax(dot1))
    f2 = int(np.argmax(dot2))
    
    logger.debug(f"Seed faces: F1={f1} (dot={dot1[f1]:.3f}), F2={f2} (dot={dot2[f2]:.3f})")
    
    # Build adjacency
    adjacency = build_triangle_adjacency(hull_mesh)
    
    # Initialize labels: 0 = unassigned, 1 = H1, 2 = H2
    labels = np.zeros(n_faces, dtype=np.int8)
    labels[f1] = 1
    labels[f2] = 2
    
    # Priority queues for greedy expansion
    # Use numpy arrays for faster processing
    import heapq
    
    # (negative_score, face_idx) - negative so highest score pops first
    pq1 = []  # H1 candidates
    pq2 = []  # H2 candidates
    in_queue = np.zeros(n_faces, dtype=bool)
    
    # Add neighbors of seeds to queues
    for n_idx in adjacency[f1]:
        if labels[n_idx] == 0:
            heapq.heappush(pq1, (-dot1[n_idx], n_idx))
            in_queue[n_idx] = True
    
    for n_idx in adjacency[f2]:
        if labels[n_idx] == 0 and not in_queue[n_idx]:
            heapq.heappush(pq2, (-dot2[n_idx], n_idx))
            in_queue[n_idx] = True
    
    # Greedy expansion - alternate between H1 and H2
    while pq1 or pq2:
        # Expand H1
        while pq1:
            neg_score, idx = heapq.heappop(pq1)
            if labels[idx] != 0:
                continue
            
            # Assign based on which direction this face prefers
            if dot1[idx] >= dot2[idx]:
                labels[idx] = 1
                # Add unassigned neighbors to H1 queue
                for n_idx in adjacency[idx]:
                    if labels[n_idx] == 0 and not in_queue[n_idx]:
                        heapq.heappush(pq1, (-dot1[n_idx], n_idx))
                        in_queue[n_idx] = True
                break
            else:
                # This face prefers H2, move it there
                heapq.heappush(pq2, (-dot2[idx], idx))
        
        # Expand H2
        while pq2:
            neg_score, idx = heapq.heappop(pq2)
            if labels[idx] != 0:
                continue
            
            if dot2[idx] >= dot1[idx]:
                labels[idx] = 2
                # Add unassigned neighbors to H2 queue
                for n_idx in adjacency[idx]:
                    if labels[n_idx] == 0 and not in_queue[n_idx]:
                        heapq.heappush(pq2, (-dot2[n_idx], n_idx))
                        in_queue[n_idx] = True
                break
            else:
                # This face prefers H1, move it there
                heapq.heappush(pq1, (-dot1[idx], idx))
    
    # Handle any remaining unassigned (disconnected components)
    unassigned = labels == 0
    if np.any(unassigned):
        labels[unassigned] = np.where(dot1[unassigned] >= dot2[unassigned], 1, 2)
    
    # =========================================================================
    # BOUNDARY ZONE: Use PHYSICAL DISTANCE as per paper Section 4.1
    # "We compute the shortest paths towards all vertices of ∂H, except for 
    # those whose distance from the boundary between ∂H1 and ∂H2 is less than 
    # a fixed threshold... we set the threshold to 15% of the convex hull 
    # bounding box diagonal."
    # =========================================================================
    
    # Find interface edges (edges where H1 face meets H2 face)
    # These define the boundary line between ∂H1 and ∂H2
    interface_edges = []
    for i in range(n_faces):
        my_label = labels[i]
        face = hull_mesh.faces[i]
        for j in range(3):
            v0, v1 = face[j], face[(j + 1) % 3]
            edge_key = (min(v0, v1), max(v0, v1))
            # Check if this edge is shared with a face of different label
            for n_idx in adjacency[i]:
                if labels[n_idx] != my_label:
                    n_face = hull_mesh.faces[n_idx]
                    n_verts = set(n_face)
                    if v0 in n_verts and v1 in n_verts:
                        interface_edges.append((v0, v1))
                        break
    
    # Remove duplicates and get interface edge midpoints
    interface_edges = list(set(interface_edges))
    
    if len(interface_edges) > 0:
        # Compute interface line points (edge midpoints)
        vertices = hull_mesh.vertices
        interface_points = np.array([
            (vertices[e[0]] + vertices[e[1]]) / 2 for e in interface_edges
        ])
        
        # Build KDTree for fast distance queries
        from scipy.spatial import cKDTree
        interface_tree = cKDTree(interface_points)
        
        # Compute distance from each face centroid to the interface
        distances_to_interface, _ = interface_tree.query(face_centroids, k=1)
        
        # Faces within threshold distance are in the boundary zone
        boundary_zone_mask = distances_to_interface <= distance_threshold
        
        logger.debug(f"Boundary zone: threshold={distance_threshold:.2f} ({boundary_zone_threshold*100:.0f}% of {bbox_diagonal:.2f}), "
                    f"{np.sum(boundary_zone_mask)} faces within threshold")
    else:
        # No interface found (all faces same label) - no boundary zone
        boundary_zone_mask = np.zeros(n_faces, dtype=bool)
        logger.warning("No interface edges found between H1 and H2")
    
    # Build result sets
    h1_triangles = set(np.where((labels == 1) & ~boundary_zone_mask)[0])
    h2_triangles = set(np.where((labels == 2) & ~boundary_zone_mask)[0])
    boundary_zone_triangles = set(np.where(boundary_zone_mask)[0])
    
    # Build side_map (excluding boundary zone)
    side_map = {}
    for i in h1_triangles:
        side_map[i] = 1
    for i in h2_triangles:
        side_map[i] = 2
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Hull classification: H1={len(h1_triangles)}, H2={len(h2_triangles)}, "
               f"boundary={len(boundary_zone_triangles)} (threshold={distance_threshold:.2f}) in {elapsed:.0f}ms")
    
    return MoldHalfClassificationResult(
        side_map=side_map,
        h1_triangles=h1_triangles,
        h2_triangles=h2_triangles,
        boundary_zone_triangles=boundary_zone_triangles,
        inner_boundary_triangles=set(),  # Hull has no inner boundary
        total_triangles=n_faces,
        outer_boundary_count=n_faces,
    )


def _classify_mold_halves_cpp(
    boundary_mesh: trimesh.Trimesh,
    outer_triangles: Set[int],
    d1: np.ndarray,
    d2: np.ndarray
) -> Dict[int, int]:
    """
    C++ implementation wrapper for mold half classification with OpenMP.
    
    Much faster than Python for large meshes (10-50x speedup).
    Uses optimized version that builds adjacency in C++ directly.
    """
    import time
    start_time = time.time()
    
    n_faces = len(boundary_mesh.faces)
    n_outer = len(outer_triangles)
    
    if n_outer == 0:
        return {}
    
    # Get face normals and faces
    face_normals = np.ascontiguousarray(boundary_mesh.face_normals, dtype=np.float64)
    faces = np.ascontiguousarray(boundary_mesh.faces, dtype=np.int64)
    
    # Convert outer_triangles set to array
    outer_indices = np.array(list(outer_triangles), dtype=np.int64)
    
    # Normalize directions
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    # Use optimized function that builds adjacency internally
    result = _cpp.classify_mold_halves_from_faces(
        face_normals,
        faces,
        outer_indices,
        d1,
        d2
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    # Convert result to dict
    labels = np.asarray(result.side_labels)
    indices = np.asarray(result.outer_indices)
    
    side_map = {int(indices[i]): int(labels[i]) for i in range(len(indices))}
    
    logger.info(
        f"[C++] Mold half classification complete in {elapsed:.0f}ms: "
        f"H1={result.n_h1}, H2={result.n_h2} from {n_outer} outer triangles"
    )
    
    return side_map


def _classify_mold_halves_complete_cpp(
    cavity_mesh: trimesh.Trimesh,
    tet_vertices: np.ndarray,
    tet_boundary_labels: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    boundary_zone_hops: int
) -> MoldHalfClassificationResult:
    """
    Complete mold half classification pipeline in C++.
    
    Handles everything in one C++ call:
    - Outer boundary detection (from tet labels)
    - Classification (greedy region growing)
    - Boundary zone computation (BFS)
    - Color generation
    
    This is MUCH faster than the Python pipeline for large meshes.
    """
    import time
    start_time = time.time()
    
    n_faces = len(cavity_mesh.faces)
    
    # Prepare arrays
    face_normals = np.ascontiguousarray(cavity_mesh.face_normals, dtype=np.float64)
    faces = np.ascontiguousarray(cavity_mesh.faces, dtype=np.int64)
    mesh_vertices = np.ascontiguousarray(cavity_mesh.vertices, dtype=np.float64)
    tet_verts = np.ascontiguousarray(tet_vertices, dtype=np.float64)
    tet_labels = np.ascontiguousarray(tet_boundary_labels, dtype=np.int8)
    
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    # Call complete C++ pipeline
    result = _cpp.classify_mold_halves_complete(
        face_normals,
        faces,
        mesh_vertices,
        tet_verts,
        tet_labels,
        d1,
        d2,
        boundary_zone_hops
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    # Convert result to Python structures
    face_labels = np.asarray(result.face_labels)
    
    # Build sets from labels
    h1_triangles = set(np.where(face_labels == 1)[0])
    h2_triangles = set(np.where(face_labels == 2)[0])
    boundary_zone_triangles = set(np.where(face_labels == 0)[0])
    inner_boundary_triangles = set(np.where(face_labels == -1)[0])
    
    # Build side_map (excluding boundary zone and inner)
    side_map = {}
    for i in h1_triangles:
        side_map[i] = 1
    for i in h2_triangles:
        side_map[i] = 2
    
    logger.info(
        f"[C++ Complete] Mold half classification in {elapsed:.0f}ms: "
        f"H1={result.n_h1}, H2={result.n_h2}, boundary={result.n_boundary_zone}, "
        f"inner={result.n_inner}"
    )
    
    return MoldHalfClassificationResult(
        side_map=side_map,
        h1_triangles=h1_triangles,
        h2_triangles=h2_triangles,
        boundary_zone_triangles=boundary_zone_triangles,
        inner_boundary_triangles=inner_boundary_triangles,
        total_triangles=n_faces,
        outer_boundary_count=result.n_h1 + result.n_h2 + result.n_boundary_zone,
    )


def _classify_mold_halves_complete_cuda(
    cavity_mesh: trimesh.Trimesh,
    tet_vertices: np.ndarray,
    tet_boundary_labels: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    boundary_zone_hops: int
) -> MoldHalfClassificationResult:
    """
    Complete CUDA-accelerated mold half classification pipeline.
    
    Uses PyTorch CUDA for:
    - Nearest neighbor search (mesh vertices → tet vertices)
    - Face classification (vectorized)
    - Dot product computation (vectorized)
    - Label propagation (parallel)
    - BFS boundary zone (parallel)
    """
    import torch
    import time
    
    start_time = time.time()
    device = torch.device('cuda')
    
    n_faces = len(cavity_mesh.faces)
    n_mesh_verts = len(cavity_mesh.vertices)
    n_tet_verts = len(tet_vertices)
    
    # Move data to GPU
    mesh_verts_t = torch.tensor(cavity_mesh.vertices, dtype=torch.float32, device=device)
    tet_verts_t = torch.tensor(tet_vertices, dtype=torch.float32, device=device)
    tet_labels_t = torch.tensor(tet_boundary_labels, dtype=torch.int8, device=device)
    faces_t = torch.tensor(cavity_mesh.faces, dtype=torch.int64, device=device)
    face_normals_t = torch.tensor(cavity_mesh.face_normals, dtype=torch.float32, device=device)
    d1_t = torch.tensor(d1 / np.linalg.norm(d1), dtype=torch.float32, device=device)
    d2_t = torch.tensor(d2 / np.linalg.norm(d2), dtype=torch.float32, device=device)
    
    # =========================================================================
    # Step 1: Nearest neighbor search (mesh vertices → tet vertices) on GPU
    # Use batched distance computation to avoid OOM
    # =========================================================================
    batch_size = 5000
    mesh_to_tet = torch.zeros(n_mesh_verts, dtype=torch.int64, device=device)
    
    for start in range(0, n_mesh_verts, batch_size):
        end = min(start + batch_size, n_mesh_verts)
        batch_verts = mesh_verts_t[start:end]  # (batch, 3)
        
        # Compute distances: (batch, n_tet_verts)
        # Use squared distance for efficiency
        dists = torch.cdist(batch_verts, tet_verts_t, p=2)
        mesh_to_tet[start:end] = torch.argmin(dists, dim=1)
    
    # Get vertex labels from tet mesh
    vertex_labels = tet_labels_t[mesh_to_tet]  # (n_mesh_verts,)
    
    # =========================================================================
    # Step 2: Classify faces - inner if ALL vertices are inner (-1)
    # =========================================================================
    face_vertex_labels = vertex_labels[faces_t]  # (n_faces, 3)
    is_inner = (face_vertex_labels == -1).all(dim=1)  # (n_faces,)
    is_outer = ~is_inner
    
    outer_indices = torch.where(is_outer)[0]  # Indices of outer faces
    n_outer = len(outer_indices)
    n_inner = n_faces - n_outer
    
    if n_outer == 0:
        # All faces are inner
        torch.cuda.empty_cache()
        return MoldHalfClassificationResult(
            side_map={},
            h1_triangles=set(),
            h2_triangles=set(),
            boundary_zone_triangles=set(),
            inner_boundary_triangles=set(range(n_faces)),
            total_triangles=n_faces,
            outer_boundary_count=0,
        )
    
    # =========================================================================
    # Step 3: Build face adjacency on GPU
    # =========================================================================
    # Create edge -> face mapping
    # Extract edges from faces
    edges = torch.stack([
        torch.stack([faces_t[:, 0], faces_t[:, 1]], dim=1),
        torch.stack([faces_t[:, 1], faces_t[:, 2]], dim=1),
        torch.stack([faces_t[:, 2], faces_t[:, 0]], dim=1),
    ], dim=1).reshape(-1, 2)  # (n_faces * 3, 2)
    
    # Sort each edge (min, max) for consistent keys
    edges_sorted = torch.sort(edges, dim=1)[0]
    
    # Create face indices for each edge
    face_indices = torch.arange(n_faces, device=device).repeat_interleave(3)
    
    # Build adjacency using CPU (GPU hash maps are complex)
    edges_cpu = edges_sorted.cpu().numpy()
    face_indices_cpu = face_indices.cpu().numpy()
    
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    for i, (e0, e1) in enumerate(edges_cpu):
        edge_to_faces[(e0, e1)].append(face_indices_cpu[i])
    
    # Build adjacency lists
    adj_lists = [[] for _ in range(n_faces)]
    for face_list in edge_to_faces.values():
        if len(face_list) == 2:
            adj_lists[face_list[0]].append(face_list[1])
            adj_lists[face_list[1]].append(face_list[0])
    
    # =========================================================================
    # Step 4: Classify outer triangles using vectorized operations
    # =========================================================================
    outer_normals = face_normals_t[outer_indices]  # (n_outer, 3)
    dot1 = outer_normals @ d1_t  # (n_outer,)
    dot2 = outer_normals @ d2_t  # (n_outer,)
    
    # Initial classification: assign based on better alignment
    outer_labels = torch.where(dot1 >= dot2,
                               torch.ones(n_outer, dtype=torch.int8, device=device),
                               torch.full((n_outer,), 2, dtype=torch.int8, device=device))
    
    # Find seeds (best alignment)
    f1_local = torch.argmax(dot1).item()
    f2_local = torch.argmax(dot2).item()
    outer_labels[f1_local] = 1
    outer_labels[f2_local] = 2
    
    # Create global->local mapping for outer triangles
    global_to_local = torch.full((n_faces,), -1, dtype=torch.int64, device=device)
    global_to_local[outer_indices] = torch.arange(n_outer, device=device)
    
    # Build local adjacency for outer triangles only
    outer_indices_cpu = outer_indices.cpu().numpy()
    outer_set = set(outer_indices_cpu)
    global_to_local_cpu = {int(outer_indices_cpu[i]): i for i in range(n_outer)}
    
    local_adj = [[] for _ in range(n_outer)]
    for i in range(n_outer):
        global_idx = int(outer_indices_cpu[i])
        for neighbor in adj_lists[global_idx]:
            if neighbor in outer_set:
                local_adj[i].append(global_to_local_cpu[neighbor])
    
    # Convert to tensors for GPU operations
    adj_flat = []
    adj_ptr = [0]
    for neighbors in local_adj:
        adj_flat.extend(neighbors)
        adj_ptr.append(len(adj_flat))
    
    adj_flat_t = torch.tensor(adj_flat, dtype=torch.int64, device=device)
    adj_ptr_t = torch.tensor(adj_ptr, dtype=torch.int64, device=device)
    neighbor_counts = torch.tensor([len(n) for n in local_adj], dtype=torch.int32, device=device)
    
    # =========================================================================
    # Step 5: Parallel label propagation (Laplacian smoothing)
    # =========================================================================
    for iteration in range(5):
        # Count H1 and H2 neighbors for each triangle
        h1_counts = torch.zeros(n_outer, dtype=torch.int32, device=device)
        h2_counts = torch.zeros(n_outer, dtype=torch.int32, device=device)
        
        # Batch process neighbor counting
        for i in range(n_outer):
            start = adj_ptr_t[i].item()
            end = adj_ptr_t[i + 1].item()
            if end > start:
                neighbor_labels = outer_labels[adj_flat_t[start:end]]
                h1_counts[i] = (neighbor_labels == 1).sum()
                h2_counts[i] = (neighbor_labels == 2).sum()
        
        # Apply voting with threshold
        valid = neighbor_counts > 0
        flip_to_h1 = valid & (outer_labels == 2) & (h1_counts > h2_counts * 2)
        flip_to_h2 = valid & (outer_labels == 1) & (h2_counts > h1_counts * 2)
        
        outer_labels[flip_to_h1] = 1
        outer_labels[flip_to_h2] = 2
        
        if not flip_to_h1.any() and not flip_to_h2.any():
            break
    
    # Re-apply strong directional constraints
    strong_h1 = (dot1 > STRONG_ALIGNMENT_THRESHOLD) & (dot1 > dot2 + STRONG_ALIGNMENT_MARGIN)
    strong_h2 = (dot2 > STRONG_ALIGNMENT_THRESHOLD) & (dot2 > dot1 + STRONG_ALIGNMENT_MARGIN)
    outer_labels[strong_h1] = 1
    outer_labels[strong_h2] = 2
    
    # =========================================================================
    # Step 6: Find interface and compute boundary zone
    # =========================================================================
    is_interface = torch.zeros(n_outer, dtype=torch.bool, device=device)
    
    for i in range(n_outer):
        start = adj_ptr_t[i].item()
        end = adj_ptr_t[i + 1].item()
        if end > start:
            neighbor_labels = outer_labels[adj_flat_t[start:end]]
            if (neighbor_labels != outer_labels[i]).any():
                is_interface[i] = True
    
    # BFS from interface triangles
    distance = torch.full((n_outer,), -1, dtype=torch.int64, device=device)
    interface_indices = torch.where(is_interface)[0]
    distance[interface_indices] = 0
    
    # BFS on CPU (GPU BFS is complex)
    distance_cpu = distance.cpu().numpy()
    queue = list(interface_indices.cpu().numpy())
    head = 0
    
    while head < len(queue):
        local_idx = queue[head]
        head += 1
        dist = distance_cpu[local_idx]
        if dist >= boundary_zone_hops:
            continue
        
        for neighbor in local_adj[local_idx]:
            if distance_cpu[neighbor] < 0:
                distance_cpu[neighbor] = dist + 1
                queue.append(neighbor)
    
    distance = torch.tensor(distance_cpu, dtype=torch.int64, device=device)
    
    # =========================================================================
    # Step 7: Build final results
    # =========================================================================
    in_boundary_zone = (distance >= 0) & (distance <= boundary_zone_hops)
    
    # Final labels for all faces
    face_labels = torch.full((n_faces,), -1, dtype=torch.int8, device=device)  # Default: inner
    
    # Set outer face labels
    for i in range(n_outer):
        fi = outer_indices[i]
        if in_boundary_zone[i]:
            face_labels[fi] = 0  # Boundary zone
        else:
            face_labels[fi] = outer_labels[i]
    
    # Move to CPU and convert to sets
    face_labels_cpu = face_labels.cpu().numpy()
    
    h1_triangles = set(np.where(face_labels_cpu == 1)[0])
    h2_triangles = set(np.where(face_labels_cpu == 2)[0])
    boundary_zone_triangles = set(np.where(face_labels_cpu == 0)[0])
    inner_boundary_triangles = set(np.where(face_labels_cpu == -1)[0])
    
    # Build side_map
    side_map = {}
    for i in h1_triangles:
        side_map[i] = 1
    for i in h2_triangles:
        side_map[i] = 2
    
    elapsed = (time.time() - start_time) * 1000
    
    logger.info(
        f"[CUDA Complete] Mold half classification in {elapsed:.0f}ms: "
        f"H1={len(h1_triangles)}, H2={len(h2_triangles)}, "
        f"boundary={len(boundary_zone_triangles)}, inner={len(inner_boundary_triangles)}"
    )
    
    torch.cuda.empty_cache()
    
    return MoldHalfClassificationResult(
        side_map=side_map,
        h1_triangles=h1_triangles,
        h2_triangles=h2_triangles,
        boundary_zone_triangles=boundary_zone_triangles,
        inner_boundary_triangles=inner_boundary_triangles,
        total_triangles=n_faces,
        outer_boundary_count=len(h1_triangles) + len(h2_triangles) + len(boundary_zone_triangles),
    )


def classify_mold_halves_fast_gpu(
    boundary_mesh: trimesh.Trimesh,
    outer_triangles: Set[int],
    d1: np.ndarray,
    d2: np.ndarray,
    adjacency: Dict[int, List[int]] = None
) -> Dict[int, int]:
    """
    GPU-accelerated mold half classification using parallel label propagation.
    
    Instead of sequential greedy region-growing, this uses:
    1. Batch compute all dot products on GPU
    2. Parallel label propagation iterations
    
    Args:
        boundary_mesh: The boundary mesh
        outer_triangles: Set of outer boundary triangle indices to classify
        d1: First parting direction (normalized)
        d2: Second parting direction (normalized)
        adjacency: Pre-computed adjacency map (optional, will compute if not provided)
    
    Returns:
        Dict mapping triangle index to mold half (1 or 2)
    """
    import torch
    
    if len(outer_triangles) == 0:
        return {}
    
    device = torch.device('cuda')
    
    # Get face normals
    face_normals = boundary_mesh.face_normals
    outer_array = np.array(list(outer_triangles), dtype=np.int32)
    n_outer = len(outer_array)
    
    # Build adjacency if not provided
    if adjacency is None:
        adjacency = build_triangle_adjacency(boundary_mesh)
    
    # Create mapping from global index to local index
    global_to_local = {int(outer_array[i]): i for i in range(n_outer)}
    
    # Build adjacency in local indices (CSR format for GPU)
    adj_list = []
    adj_ptr = [0]
    for i in range(n_outer):
        global_idx = int(outer_array[i])
        neighbors = [global_to_local[n] for n in adjacency.get(global_idx, []) 
                    if n in global_to_local]
        adj_list.extend(neighbors)
        adj_ptr.append(len(adj_list))
    
    # Move data to GPU
    normals_t = torch.tensor(face_normals[outer_array], dtype=torch.float32, device=device)
    d1_t = torch.tensor(d1, dtype=torch.float32, device=device)
    d2_t = torch.tensor(d2, dtype=torch.float32, device=device)
    
    # Compute all dot products in one go (fully vectorized)
    dot1 = normals_t @ d1_t  # (n_outer,)
    dot2 = normals_t @ d2_t  # (n_outer,)
    
    # Initial classification based on directional preference
    labels = torch.where(dot1 >= dot2, 
                        torch.ones(n_outer, dtype=torch.int32, device=device),
                        torch.full((n_outer,), 2, dtype=torch.int32, device=device))
    
    # Convert adjacency to GPU tensors for parallel propagation
    adj_indices = torch.tensor(adj_list, dtype=torch.int64, device=device)
    adj_ptr_t = torch.tensor(adj_ptr, dtype=torch.int64, device=device)
    
    # Parallel label propagation iterations
    # Each iteration, vertices can switch if majority of neighbors disagree
    max_iterations = 20
    for iteration in range(max_iterations):
        old_labels = labels.clone()
        
        # For each vertex, count H1 and H2 neighbors
        h1_counts = torch.zeros(n_outer, dtype=torch.int32, device=device)
        h2_counts = torch.zeros(n_outer, dtype=torch.int32, device=device)
        
        # Process in parallel using scatter_add
        for i in range(n_outer):
            start = adj_ptr_t[i].item()
            end = adj_ptr_t[i + 1].item()
            if end > start:
                neighbor_labels = labels[adj_indices[start:end]]
                h1_counts[i] = (neighbor_labels == 1).sum()
                h2_counts[i] = (neighbor_labels == 2).sum()
        
        # Apply voting with hysteresis (require strong majority to flip)
        total_neighbors = h1_counts + h2_counts
        valid = total_neighbors > 0
        
        # Flip H2->H1 if h1_count > h2_count * 1.5 OR strong alignment with d1
        flip_to_h1 = valid & (labels == 2) & (
            (h1_counts > h2_counts * 1.5) | (dot1 > dot2 + 0.3)
        )
        # Flip H1->H2 if h2_count > h1_count * 1.5 OR strong alignment with d2  
        flip_to_h2 = valid & (labels == 1) & (
            (h2_counts > h1_counts * 1.5) | (dot2 > dot1 + 0.3)
        )
        
        labels[flip_to_h1] = 1
        labels[flip_to_h2] = 2
        
        # Check convergence
        changed = (labels != old_labels).sum().item()
        if changed == 0:
            logger.debug(f"GPU label propagation converged at iteration {iteration}")
            break
    
    # Convert back to dict
    labels_cpu = labels.cpu().numpy()
    result = {int(outer_array[i]): int(labels_cpu[i]) for i in range(n_outer)}
    
    torch.cuda.empty_cache()
    return result


def classify_mold_halves_fast(
    boundary_mesh: trimesh.Trimesh,
    outer_triangles: Set[int],
    d1: np.ndarray,
    d2: np.ndarray,
    adjacency: Dict[int, List[int]] = None,
    use_gpu: bool = True
) -> Dict[int, int]:
    """
    Fast mold half classification using greedy region growing (per research paper).
    
    Algorithm from paper:
    "Given the two parting directions, we partition the boundary ∂H into two parts, 
    ∂H1 and ∂H2: we select the two faces F1 and F2 of ∂H whose normals best align 
    with d1 and d2 and then use a greedy region-growing approach from F1 and F2 
    to assign faces to ∂H1 and ∂H2, according to the alignment of their normal to d1 and d2"
    
    This is O(n) where n = number of triangles, much faster than the proximity-based approach.
    Uses C++ with OpenMP for large meshes, GPU for medium meshes, Python for small meshes.
    
    Args:
        boundary_mesh: The boundary mesh
        outer_triangles: Set of outer boundary triangle indices to classify
        d1: First parting direction (normalized)
        d2: Second parting direction (normalized)
        adjacency: Pre-computed adjacency map (optional, will compute if not provided)
        use_gpu: Whether to use GPU acceleration if available
    
    Returns:
        Dict mapping triangle index to mold half (1 or 2)
    """
    if len(outer_triangles) == 0:
        return {}
    
    n_outer = len(outer_triangles)
    
    # Use C++ with OpenMP for large meshes (fastest)
    if CPP_FAST_AVAILABLE and n_outer > 1000:
        logger.debug(f"Using C++ (OpenMP) classification for {n_outer} outer triangles")
        return _classify_mold_halves_cpp(boundary_mesh, outer_triangles, d1, d2)
    
    # Use GPU path for medium-large meshes
    if use_gpu and CUDA_AVAILABLE and n_outer > 10000:
        logger.debug(f"Using GPU classification for {n_outer} outer triangles")
        return classify_mold_halves_fast_gpu(boundary_mesh, outer_triangles, d1, d2, adjacency)
    
    # Get face normals
    face_normals = boundary_mesh.face_normals
    outer_array = np.array(list(outer_triangles), dtype=np.int32)
    
    # Compute dot products with parting directions (vectorized)
    normals = face_normals[outer_array]  # (n_outer, 3)
    dot1 = normals @ d1  # (n_outer,)
    dot2 = normals @ d2  # (n_outer,)
    
    # Find seed faces: F1 = best alignment with d1, F2 = best alignment with d2
    f1_local_idx = np.argmax(dot1)
    f2_local_idx = np.argmax(dot2)
    f1 = outer_array[f1_local_idx]
    f2 = outer_array[f2_local_idx]
    
    # Build adjacency if not provided
    if adjacency is None:
        adjacency = build_triangle_adjacency(boundary_mesh)
    
    # Create mapping from triangle idx to local index for fast lookup
    outer_set = set(outer_triangles)
    
    # Greedy region growing using priority queues
    # Priority = negative dot product (so highest alignment is processed first)
    import heapq
    
    side = {}  # Result: triangle_idx -> 1 or 2
    
    # Initialize with seed faces
    side[f1] = 1
    side[f2] = 2
    
    # Priority queues: (-alignment_score, triangle_idx)
    # Use negative so highest alignment is popped first
    pq1 = []  # For H1 expansion
    pq2 = []  # For H2 expansion
    
    # Add neighbors of seeds to queues
    for neighbor in adjacency.get(f1, []):
        if neighbor in outer_set and neighbor not in side:
            # Alignment with d1 for H1 candidates
            n_vec = face_normals[neighbor]
            score = np.dot(n_vec, d1)
            heapq.heappush(pq1, (-score, neighbor))
    
    for neighbor in adjacency.get(f2, []):
        if neighbor in outer_set and neighbor not in side:
            # Alignment with d2 for H2 candidates
            n_vec = face_normals[neighbor]
            score = np.dot(n_vec, d2)
            heapq.heappush(pq2, (-score, neighbor))
    
    # Greedy expansion - alternate between H1 and H2, always taking best-aligned
    in_queue = set()  # Track which triangles are already queued
    for _, tri in pq1:
        in_queue.add(tri)
    for _, tri in pq2:
        in_queue.add(tri)
    
    while pq1 or pq2:
        # Try H1 expansion
        assigned_h1 = False
        while pq1:
            neg_score, tri = heapq.heappop(pq1)
            if tri in side:
                continue
            
            # Assign to H1 if this face aligns better with d1 than d2
            n_vec = face_normals[tri]
            align1 = np.dot(n_vec, d1)
            align2 = np.dot(n_vec, d2)
            
            if align1 >= align2:
                side[tri] = 1
                assigned_h1 = True
                # Add unassigned neighbors to H1 queue
                for neighbor in adjacency.get(tri, []):
                    if neighbor in outer_set and neighbor not in side and neighbor not in in_queue:
                        n_vec2 = face_normals[neighbor]
                        score = np.dot(n_vec2, d1)
                        heapq.heappush(pq1, (-score, neighbor))
                        in_queue.add(neighbor)
                break
            else:
                # This face prefers H2, put it in H2 queue
                heapq.heappush(pq2, (-align2, tri))
        
        # Try H2 expansion
        assigned_h2 = False
        while pq2:
            neg_score, tri = heapq.heappop(pq2)
            if tri in side:
                continue
            
            # Assign to H2 if this face aligns better with d2 than d1
            n_vec = face_normals[tri]
            align1 = np.dot(n_vec, d1)
            align2 = np.dot(n_vec, d2)
            
            if align2 >= align1:
                side[tri] = 2
                assigned_h2 = True
                # Add unassigned neighbors to H2 queue
                for neighbor in adjacency.get(tri, []):
                    if neighbor in outer_set and neighbor not in side and neighbor not in in_queue:
                        n_vec2 = face_normals[neighbor]
                        score = np.dot(n_vec2, d2)
                        heapq.heappush(pq2, (-score, neighbor))
                        in_queue.add(neighbor)
                break
            else:
                # This face prefers H1, put it in H1 queue
                heapq.heappush(pq1, (-align1, tri))
        
        # Safety: if neither queue made progress, break
        if not assigned_h1 and not assigned_h2 and not pq1 and not pq2:
            break
    
    # Handle any remaining unassigned triangles (disconnected components)
    for tri in outer_triangles:
        if tri not in side:
            n_vec = face_normals[tri]
            align1 = np.dot(n_vec, d1)
            align2 = np.dot(n_vec, d2)
            side[tri] = 1 if align1 >= align2 else 2
    
    return side


def identify_outer_boundary_by_distance(
    boundary_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    tolerance_fraction: float = 0.02
) -> Tuple[Set[int], Set[int]]:
    """
    Fast outer boundary detection using vertex distance to part mesh.
    
    A face is "inner" (from part surface) if ALL its vertices are within 
    tolerance of the part mesh surface. Otherwise it's "outer" (from hull surface).
    
    Uses KDTree for fast vertex-to-vertex distance computation.
    For most cases, this is accurate enough since boundary mesh vertices that
    are on the part surface will be very close to part mesh vertices.
    
    Args:
        boundary_mesh: The boundary mesh to classify
        part_mesh: The original part mesh
        tolerance_fraction: Fraction of mesh size for tolerance (default 2%)
    
    Returns:
        Tuple of (outer_triangles, inner_triangles) sets
    """
    import time
    from scipy.spatial import cKDTree
    
    start = time.time()
    
    # Compute tolerance based on mesh size
    bounds = boundary_mesh.bounds
    mesh_size = np.linalg.norm(bounds[1] - bounds[0])
    tolerance = mesh_size * tolerance_fraction
    
    # Build KDTree on part mesh vertices (fast O(n log n))
    boundary_verts = boundary_mesh.vertices
    part_verts = part_mesh.vertices
    
    kdtree_start = time.time()
    part_tree = cKDTree(part_verts)
    distances, _ = part_tree.query(boundary_verts, k=1)
    kdtree_time = (time.time() - kdtree_start) * 1000
    
    # Mark vertices as "near part surface" if within tolerance
    near_part = distances < tolerance  # (n_verts,)
    
    # Log distance statistics for debugging
    logger.debug(f"Distance to part (KDTree): min={distances.min():.4f}, max={distances.max():.4f}, "
                f"tolerance={tolerance:.4f}, query_time={kdtree_time:.0f}ms")
    logger.debug(f"Vertices near part: {np.sum(near_part)} / {len(boundary_verts)}")
    
    # A face is inner if ALL its vertices are near part surface
    faces = boundary_mesh.faces  # (n_faces, 3)
    face_near_part = near_part[faces]  # (n_faces, 3)
    all_near_part = np.all(face_near_part, axis=1)  # (n_faces,)
    
    inner_triangles = set(np.where(all_near_part)[0])
    outer_triangles = set(np.where(~all_near_part)[0])
    
    elapsed = (time.time() - start) * 1000
    logger.debug(f"KDTree-based outer detection: {len(outer_triangles)} outer, "
                f"{len(inner_triangles)} inner in {elapsed:.0f}ms")
    
    return outer_triangles, inner_triangles


def identify_outer_boundary_from_tet_labels(
    boundary_mesh: trimesh.Trimesh,
    tet_vertices: np.ndarray,
    boundary_labels: np.ndarray
) -> Tuple[Set[int], Set[int]]:
    """
    Identify outer vs inner boundary triangles using pre-computed tet boundary labels.
    
    This is O(n) - much faster than proximity-based detection which is O(n*m).
    
    The boundary_labels array marks each tet vertex as:
    - 0: interior (not on boundary)
    - 1: H1 (outer boundary aligned with d1)
    - 2: H2 (outer boundary aligned with d2)
    - -1: inner boundary (part surface)
    
    A triangle is on the inner boundary if ALL its vertices are on the inner boundary.
    Otherwise it's on the outer boundary (or mixed, treated as outer).
    
    Args:
        boundary_mesh: The boundary mesh extracted from tet mesh
        tet_vertices: (N, 3) tetrahedral mesh vertices
        boundary_labels: (N,) boundary labels from tet mesh (-1=inner, 1=H1, 2=H2)
    
    Returns:
        Tuple of (outer_triangles, inner_triangles) sets
    """
    import time
    start = time.time()
    
    from scipy.spatial import cKDTree
    
    # Map boundary mesh vertices to tet mesh vertices
    boundary_verts = np.asarray(boundary_mesh.vertices)
    n_boundary_verts = len(boundary_verts)
    n_tet_verts = len(tet_vertices)
    
    kdtree_start = time.time()
    tet_tree = cKDTree(tet_vertices)
    _, boundary_to_tet = tet_tree.query(boundary_verts, k=1)
    kdtree_time = (time.time() - kdtree_start) * 1000
    
    # Get labels for boundary mesh vertices
    vertex_labels = boundary_labels[boundary_to_tet]  # (n_boundary_verts,)
    
    # Classify each triangle
    faces = boundary_mesh.faces  # (n_faces, 3)
    face_vertex_labels = vertex_labels[faces]  # (n_faces, 3)
    
    # A face is "inner" if ALL its vertices are inner boundary (-1)
    # This is conservative - mixed faces go to outer
    all_inner = np.all(face_vertex_labels == -1, axis=1)  # (n_faces,)
    
    inner_triangles = set(np.where(all_inner)[0])
    outer_triangles = set(np.where(~all_inner)[0])
    
    elapsed = (time.time() - start) * 1000
    logger.debug(f"From tet labels: {len(outer_triangles)} outer, {len(inner_triangles)} inner "
                f"({n_boundary_verts} boundary verts → {n_tet_verts} tet verts, KDTree: {kdtree_time:.0f}ms, total: {elapsed:.0f}ms)")
    
    return outer_triangles, inner_triangles


def identify_outer_boundary_from_hull(
    cavity_triangles: List[TriangleInfo],
    hull_mesh: trimesh.Trimesh,
    use_gpu: bool = True
) -> Set[int]:
    """
    Identify outer boundary triangles by checking if their centroids lie on the hull surface.
    
    Uses the same plane distance method as the React app:
    For each cavity triangle centroid, check if it lies on any hull face plane.
    
    Uses GPU acceleration when available for large meshes.
    
    The cavity mesh is created by CSG: Hull - Part
    Outer boundary = triangles on the Hull surface (distance to hull surface ≈ 0)
    Inner boundary = triangles from the Part surface (inside the hull)
    """
    if len(cavity_triangles) == 0:
        return set()
    
    # Compute hull bounding box to determine appropriate tolerance
    bounds = hull_mesh.bounds
    hull_size = bounds[1] - bounds[0]
    max_dim = np.max(hull_size)
    tolerance = max_dim * OUTER_BOUNDARY_TOLERANCE_FRACTION
    
    # Get hull face normals and compute plane constants
    face_normals = hull_mesh.face_normals  # Shape: (n_faces, 3)
    vertices = hull_mesh.vertices
    faces = hull_mesh.faces
    
    # Deduplicate planes by normal direction (convex hull faces)
    # This matches the React app's extractHullFacePlanes behavior
    unique_planes = []
    normal_keys = set()
    
    for i in range(len(faces)):
        normal = face_normals[i]
        normal_key = f"{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}"
        if normal_key in normal_keys:
            continue
        normal_keys.add(normal_key)
        
        # Plane equation: n · p = d
        v0 = vertices[faces[i, 0]]
        d = np.dot(normal, v0)
        unique_planes.append((normal, d))
    
    # Convert to arrays for vectorized computation
    plane_normals = np.array([p[0] for p in unique_planes])  # Shape: (n_planes, 3)
    plane_d = np.array([p[1] for p in unique_planes])  # Shape: (n_planes,)
    
    # Get all cavity centroids as a single array
    centroids = np.array([tri.centroid for tri in cavity_triangles])  # Shape: (n_tris, 3)
    n_tris = len(centroids)
    
    # Use GPU for large meshes
    if use_gpu and CUDA_AVAILABLE and n_tris > 50000:
        return _identify_outer_boundary_from_hull_gpu(centroids, plane_normals, plane_d, tolerance)
    
    outer_triangles = set()
    
    # Process in batches to avoid memory explosion
    batch_size = 10000
    
    for start in range(0, len(centroids), batch_size):
        end = min(start + batch_size, len(centroids))
        batch_centroids = centroids[start:end]  # Shape: (batch, 3)
        
        # Compute distances to all planes: |n·c - d|
        # batch_centroids @ plane_normals.T gives shape (batch, n_planes)
        dots = batch_centroids @ plane_normals.T  # Shape: (batch, n_planes)
        distances = np.abs(dots - plane_d)  # Shape: (batch, n_planes)
        
        # Check if any plane distance is within tolerance
        on_hull = np.any(distances < tolerance, axis=1)  # Shape: (batch,)
        
        # Add indices of triangles on hull
        for i in np.where(on_hull)[0]:
            outer_triangles.add(start + i)
    
    return outer_triangles


def _identify_outer_boundary_from_hull_gpu(
    centroids: np.ndarray,
    plane_normals: np.ndarray,
    plane_d: np.ndarray,
    tolerance: float
) -> Set[int]:
    """
    GPU-accelerated outer boundary identification using plane distances.
    
    Args:
        centroids: (N, 3) triangle centroids
        plane_normals: (P, 3) plane normal vectors
        plane_d: (P,) plane distance constants
        tolerance: Distance tolerance for on-hull detection
    
    Returns:
        Set of triangle indices on the hull surface
    """
    import torch
    
    device = torch.device('cuda')
    n_tris = len(centroids)
    n_planes = len(plane_normals)
    
    logger.debug(f"GPU: Computing plane distances for {n_tris} triangles against {n_planes} planes")
    
    # Move data to GPU
    centroids_t = torch.tensor(centroids, dtype=torch.float32, device=device)
    normals_t = torch.tensor(plane_normals, dtype=torch.float32, device=device)
    d_t = torch.tensor(plane_d, dtype=torch.float32, device=device)
    
    # Process in batches to manage GPU memory
    batch_size = 100000
    on_hull_mask = torch.zeros(n_tris, dtype=torch.bool, device=device)
    
    for start in range(0, n_tris, batch_size):
        end = min(start + batch_size, n_tris)
        batch = centroids_t[start:end]  # (B, 3)
        
        # Compute distances to all planes: |n·c - d|
        dots = batch @ normals_t.T  # (B, P)
        distances = torch.abs(dots - d_t)  # (B, P)
        
        # Check if any plane distance is within tolerance
        on_hull_mask[start:end] = torch.any(distances < tolerance, dim=1)
    
    # Convert to set
    outer_indices = torch.where(on_hull_mask)[0].cpu().numpy()
    torch.cuda.empty_cache()
    
    return set(outer_indices.tolist())


def identify_outer_boundary_by_proximity(
    boundary_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh = None,
    use_gpu: bool = True
) -> Set[int]:
    """
    Identify outer boundary triangles by proximity to hull surface vs part surface.
    
    This method works better for tetrahedral boundary meshes where triangles
    don't align exactly with hull planes.
    
    Uses GPU acceleration when available for faster distance computation.
    
    For each triangle centroid:
    - If closer to hull surface than to part surface -> outer boundary
    - Otherwise -> inner boundary (part surface)
    
    If part_mesh is not provided, uses distance to hull surface with a threshold.
    
    Args:
        boundary_mesh: The boundary surface mesh (e.g., from tetrahedralization)
        hull_mesh: The original hull mesh
        part_mesh: The original part mesh (optional)
        use_gpu: Whether to use GPU acceleration if available
    
    Returns:
        Set of triangle indices that are on the outer boundary (hull surface)
    """
    n_faces = len(boundary_mesh.faces)
    if n_faces == 0:
        return set()
    
    # Compute centroids for all triangles
    centroids = boundary_mesh.triangles_center  # (n_faces, 3)
    
    # Compute hull size for tolerance
    bounds = hull_mesh.bounds
    hull_size = np.linalg.norm(bounds[1] - bounds[0])
    tolerance = hull_size * TET_OUTER_BOUNDARY_TOLERANCE_FRACTION
    
    # Use GPU for large meshes
    if use_gpu and CUDA_AVAILABLE and n_faces > 5000:
        hull_distances = _compute_distances_to_mesh_gpu(centroids, hull_mesh)
        
        if part_mesh is not None:
            part_distances = _compute_distances_to_mesh_gpu(centroids, part_mesh)
            outer_mask = hull_distances < part_distances
            logger.debug(f"GPU proximity classification: {np.sum(outer_mask)} outer, {np.sum(~outer_mask)} inner")
        else:
            outer_mask = hull_distances < tolerance
            logger.debug(f"GPU threshold classification (tol={tolerance:.4f}): {np.sum(outer_mask)} outer, {np.sum(~outer_mask)} inner")
    else:
        # CPU path using trimesh
        _, hull_distances, _ = hull_mesh.nearest.on_surface(centroids)
        
        if part_mesh is not None:
            _, part_distances, _ = part_mesh.nearest.on_surface(centroids)
            outer_mask = hull_distances < part_distances
            logger.debug(f"CPU proximity classification: {np.sum(outer_mask)} outer, {np.sum(~outer_mask)} inner")
        else:
            outer_mask = hull_distances < tolerance
            logger.debug(f"CPU threshold classification (tol={tolerance:.4f}): {np.sum(outer_mask)} outer, {np.sum(~outer_mask)} inner")
    
    outer_triangles = set(np.where(outer_mask)[0])
    return outer_triangles


def _compute_distances_to_mesh_gpu(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    batch_size: int = 50000
) -> np.ndarray:
    """
    GPU-accelerated distance computation from points to mesh surface.
    
    Uses PyTorch CUDA to compute point-to-triangle distances in parallel.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        batch_size: Points per GPU batch
    
    Returns:
        (N,) array of unsigned distances to mesh surface
    """
    import torch
    
    device = torch.device('cuda')
    n_points = len(query_points)
    n_faces = len(target_mesh.faces)
    
    logger.debug(f"GPU: Computing distances for {n_points} points to {n_faces} triangles")
    
    # Get mesh data on GPU
    vertices = torch.tensor(target_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(target_mesh.faces, dtype=torch.int64, device=device)
    
    # Pre-compute triangle vertices
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Pre-compute triangle edge data
    edge0 = v1 - v0  # (F, 3)
    edge1 = v2 - v0  # (F, 3)
    dot00 = (edge0 * edge0).sum(dim=-1)  # (F,)
    dot01 = (edge0 * edge1).sum(dim=-1)  # (F,)
    dot11 = (edge1 * edge1).sum(dim=-1)  # (F,)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)  # (F,)
    
    # Adjust batch size based on triangle count
    max_pairs = 150_000_000
    adaptive_batch = max(100, min(batch_size, max_pairs // n_faces))
    
    all_distances = torch.zeros(n_points, dtype=torch.float32, device=device)
    
    for start_idx in range(0, n_points, adaptive_batch):
        end_idx = min(start_idx + adaptive_batch, n_points)
        batch_points = torch.tensor(
            query_points[start_idx:end_idx], 
            dtype=torch.float32, 
            device=device
        )  # (B, 3)
        
        # Compute distances
        batch_distances = _point_to_triangles_distance_gpu(
            batch_points, v0, edge0, edge1, dot00, dot01, dot11, inv_denom
        )
        
        all_distances[start_idx:end_idx] = batch_distances
        
        if (start_idx // adaptive_batch) % 10 == 0:
            torch.cuda.empty_cache()
    
    result = all_distances.cpu().numpy().astype(np.float64)
    torch.cuda.empty_cache()
    return result


def _point_to_triangles_distance_gpu(
    points: 'torch.Tensor',
    v0: 'torch.Tensor',
    edge0: 'torch.Tensor',
    edge1: 'torch.Tensor',
    dot00: 'torch.Tensor',
    dot01: 'torch.Tensor',
    dot11: 'torch.Tensor',
    inv_denom: 'torch.Tensor'
) -> 'torch.Tensor':
    """
    Compute minimum distance from each point to any triangle (GPU).
    
    Args:
        points: (B, 3) query points
        v0: (F, 3) first vertex of each triangle
        edge0, edge1: (F, 3) triangle edges
        dot00, dot01, dot11, inv_denom: (F,) pre-computed values
    
    Returns:
        (B,) minimum distance for each point
    """
    # Expand for broadcasting: points (B, 1, 3), triangles (1, F, 3)
    p = points.unsqueeze(1)  # (B, 1, 3)
    
    # Vector from v0 to point
    v0_to_p = p - v0.unsqueeze(0)  # (B, F, 3)
    
    # Compute dot products for barycentric coordinates
    dot02 = (v0_to_p * edge0.unsqueeze(0)).sum(dim=-1)  # (B, F)
    dot12 = (v0_to_p * edge1.unsqueeze(0)).sum(dim=-1)  # (B, F)
    
    # Barycentric coordinates
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom  # (B, F)
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom  # (B, F)
    
    # Clamp to triangle
    import torch
    u_clamped = torch.clamp(u, 0, 1)
    v_clamped = torch.clamp(v, 0, 1)
    
    # Ensure u + v <= 1
    uv_sum = u_clamped + v_clamped
    scale = torch.where(uv_sum > 1, 1.0 / uv_sum, torch.ones_like(uv_sum))
    u_final = u_clamped * scale
    v_final = v_clamped * scale
    
    # Compute squared distance directly
    diff = v0_to_p - u_final.unsqueeze(-1) * edge0.unsqueeze(0) - v_final.unsqueeze(-1) * edge1.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=-1)  # (B, F)
    
    # Minimum distance across all triangles
    min_dist = torch.sqrt(dist_sq.min(dim=1).values)  # (B,)
    
    return min_dist


def classify_and_smooth(
    triangles: List[TriangleInfo],
    adjacency: Dict[int, List[int]],
    outer_triangles: Set[int],
    d1: np.ndarray,
    d2: np.ndarray
) -> Dict[int, int]:
    """
    Classify outer boundary triangles and apply morphological smoothing.
    
    Optimized version using NumPy arrays for vectorized operations.
    
    Algorithm:
    1. Initial classification by directional preference
    2. Morphological opening (erode + dilate) to remove thin peninsulas
    3. Laplacian smoothing for final cleanup
    4. Orphan removal for isolated regions
    5. Re-apply strong directional constraints
    """
    outer_array = np.array(list(outer_triangles), dtype=np.int32)
    n_outer = len(outer_array)
    
    if n_outer == 0:
        return {}
    
    max_tri_idx = outer_array.max() + 1
    
    # Create lookup array: outer_idx -> position in outer_array (-1 if not outer)
    is_outer = np.zeros(max_tri_idx, dtype=bool)
    is_outer[outer_array] = True
    
    # Extract all normals at once (vectorized)
    normals = np.array([triangles[i].normal for i in outer_array])  # Shape: (n_outer, 3)
    
    # Step 1: Initial classification by directional preference (vectorized)
    dot1 = normals @ d1  # Shape: (n_outer,)
    dot2 = normals @ d2  # Shape: (n_outer,)
    
    # side_labels: 1=H1, 2=H2 (indexed by position in outer_array)
    side_labels = np.where(dot1 >= dot2, 1, 2).astype(np.int8)
    
    # Build neighbor index arrays for vectorized access
    # neighbors_flat: flat array of all neighbor indices
    # neighbors_ptr: pointer array (neighbors of outer_array[i] are at neighbors_flat[ptr[i]:ptr[i+1]])
    neighbors_list = []
    neighbors_ptr = [0]
    
    for i in outer_array:
        adj = adjacency.get(i, [])
        # Filter to only outer triangles
        filtered = [n for n in adj if n < max_tri_idx and is_outer[n]]
        neighbors_list.extend(filtered)
        neighbors_ptr.append(len(neighbors_list))
    
    neighbors_flat = np.array(neighbors_list, dtype=np.int32)
    neighbors_ptr = np.array(neighbors_ptr, dtype=np.int32)
    
    # Create mapping from triangle index to position in outer_array
    tri_to_pos = np.full(max_tri_idx, -1, dtype=np.int32)
    tri_to_pos[outer_array] = np.arange(n_outer, dtype=np.int32)
    
    # Convert neighbor indices to positions in outer_array
    neighbors_pos = tri_to_pos[neighbors_flat]
    
    # Precompute neighbor counts per triangle
    neighbor_counts = neighbors_ptr[1:] - neighbors_ptr[:-1]
    
    # Build CSR-style row indices for vectorized neighbor counting
    row_indices = np.repeat(np.arange(n_outer), neighbor_counts)
    
    # =========================================================================
    # VECTORIZED SMOOTHING FUNCTIONS
    # =========================================================================
    
    def count_neighbor_labels_vectorized(labels: np.ndarray, target_label: int) -> np.ndarray:
        """Count how many neighbors of each triangle have the target label."""
        # Get labels of all neighbors
        neighbor_labels = labels[neighbors_pos]  # Shape: (total_neighbors,)
        # Check which neighbors match target
        matches = (neighbor_labels == target_label).astype(np.int32)
        # Sum matches per triangle using bincount
        counts = np.bincount(row_indices, weights=matches, minlength=n_outer)
        return counts.astype(np.int32)
    
    def morph_pass_vectorized(target_side: int, flip_to: int, threshold: float, iterations: int):
        """Vectorized morphological operation."""
        nonlocal side_labels
        for _ in range(iterations):
            # Find triangles with target side
            is_target = side_labels == target_side
            
            # Count neighbors with flip_to label
            opp_counts = count_neighbor_labels_vectorized(side_labels, flip_to)
            
            # Flip condition: has neighbors AND opposite count >= threshold * neighbor_count
            # opp_count > 0 AND opp_count >= neighbor_counts * threshold
            flip_mask = (
                is_target & 
                (neighbor_counts > 0) &
                (opp_counts > 0) & 
                (opp_counts >= neighbor_counts * threshold)
            )
            
            if not np.any(flip_mask):
                break
            
            side_labels[flip_mask] = flip_to
    
    # Step 2: Morphological opening (same iterations as React app: 4 erode, 4 dilate)
    # Erode H1, Dilate H1, Erode H2, Dilate H2
    morph_pass_vectorized(1, 2, 0.5, 4)
    morph_pass_vectorized(2, 1, 0.5, 4)
    morph_pass_vectorized(2, 1, 0.5, 4)
    morph_pass_vectorized(1, 2, 0.5, 4)
    
    # Step 3: Laplacian smoothing - flip if outnumbered by neighbors (10 passes like React)
    for _ in range(10):
        h1_counts = count_neighbor_labels_vectorized(side_labels, 1)
        h2_counts = count_neighbor_labels_vectorized(side_labels, 2)
        
        # Flip H1->H2 if h2_count > h1_count * 1.5
        flip_to_h2 = (side_labels == 1) & (neighbor_counts > 0) & (h2_counts > h1_counts * 1.5)
        # Flip H2->H1 if h1_count > h2_count * 1.5
        flip_to_h1 = (side_labels == 2) & (neighbor_counts > 0) & (h1_counts > h2_counts * 1.5)
        
        if not np.any(flip_to_h2) and not np.any(flip_to_h1):
            break
        
        side_labels[flip_to_h2] = 2
        side_labels[flip_to_h1] = 1
    
    # Step 4: Remove orphan regions (increased threshold for large triangles)
    side = {int(outer_array[i]): int(side_labels[i]) for i in range(n_outer)}
    remove_orphans(side, adjacency, outer_triangles, 150)  # Increased from 80
    
    # Sync back
    for i in range(n_outer):
        side_labels[i] = side.get(int(outer_array[i]), side_labels[i])
    
    # Step 5: Re-apply strong directional constraints (vectorized)
    strong_h1 = (dot1 > STRONG_ALIGNMENT_THRESHOLD) & (dot1 > dot2 + STRONG_ALIGNMENT_MARGIN)
    strong_h2 = (dot2 > STRONG_ALIGNMENT_THRESHOLD) & (dot2 > dot1 + STRONG_ALIGNMENT_MARGIN)
    side_labels[strong_h1] = 1
    side_labels[strong_h2] = 2
    
    # Final orphan cleanup (larger threshold)
    side = {int(outer_array[i]): int(side_labels[i]) for i in range(n_outer)}
    remove_orphans(side, adjacency, outer_triangles, 120)  # Increased from 60
    
    # Area-based orphan removal to handle large triangles creating peninsulas
    remove_orphans_by_area(side, adjacency, outer_triangles, triangles, area_threshold_ratio=0.02)
    
    # Step 6: Final boundary smoothing pass (5 iterations like React app) - VECTORIZED
    for i in range(n_outer):
        side_labels[i] = side.get(int(outer_array[i]), side_labels[i])
    
    for _ in range(5):
        h1_counts = count_neighbor_labels_vectorized(side_labels, 1)
        h2_counts = count_neighbor_labels_vectorized(side_labels, 2)
        
        # Only flip at boundaries (where both H1 and H2 neighbors exist) with >= 3 neighbors
        at_boundary = (h1_counts > 0) & (h2_counts > 0) & (neighbor_counts >= 3)
        
        # Flip H1->H2 if h2_count >= h1_count * 2
        flip_to_h2 = (side_labels == 1) & at_boundary & (h2_counts >= h1_counts * 2)
        # Flip H2->H1 if h1_count >= h2_count * 2
        flip_to_h1 = (side_labels == 2) & at_boundary & (h1_counts >= h2_counts * 2)
        
        if not np.any(flip_to_h2) and not np.any(flip_to_h1):
            break
        
        side_labels[flip_to_h2] = 2
        side_labels[flip_to_h1] = 1
    
    # Convert back to dict
    return {int(outer_array[i]): int(side_labels[i]) for i in range(n_outer)}


def remove_orphans(
    side: Dict[int, int],
    adjacency: Dict[int, List[int]],
    outer_triangles: Set[int],
    max_size: int
) -> None:
    """Remove small isolated regions by flipping them to match surroundings."""
    processed: Set[int] = set()
    
    for start_idx in outer_triangles:
        if start_idx in processed:
            continue
        
        if start_idx not in side:
            continue
        
        start_side = side[start_idx]
        
        # BFS to find connected component
        component = [start_idx]
        processed.add(start_idx)
        head = 0
        
        while head < len(component):
            idx = component[head]
            head += 1
            
            for neighbor in adjacency.get(idx, []):
                if neighbor in processed:
                    continue
                if neighbor not in outer_triangles:
                    continue
                if side.get(neighbor) != start_side:
                    continue
                
                processed.add(neighbor)
                component.append(neighbor)
        
        # Flip small components
        if len(component) <= max_size:
            new_side = 2 if start_side == 1 else 1
            for idx in component:
                side[idx] = new_side


def remove_orphans_by_area(
    side: Dict[int, int],
    adjacency: Dict[int, List[int]],
    outer_triangles: Set[int],
    triangles: List[TriangleInfo],
    area_threshold_ratio: float = ORPHAN_REGION_AREA_THRESHOLD
) -> None:
    """
    Remove isolated regions based on total area rather than triangle count.
    
    This handles cases where a few large triangles create visual peninsulas
    that pass the count-based orphan test but are still visually isolated.
    
    Args:
        side: Triangle side assignments (modified in place)
        adjacency: Triangle adjacency map
        outer_triangles: Set of outer boundary triangles
        triangles: Triangle info including areas
        area_threshold_ratio: Components with area < this ratio of total outer area are flipped
    """
    # Compute total area of outer triangles for each side
    h1_total_area = sum(triangles[i].area for i in outer_triangles if side.get(i) == 1)
    h2_total_area = sum(triangles[i].area for i in outer_triangles if side.get(i) == 2)
    
    # Use the smaller side's area as threshold basis
    min_side_area = min(h1_total_area, h2_total_area) if h1_total_area > 0 and h2_total_area > 0 else 0
    area_threshold = min_side_area * area_threshold_ratio
    
    if area_threshold <= 0:
        return
    
    processed: Set[int] = set()
    
    for start_idx in outer_triangles:
        if start_idx in processed:
            continue
        
        if start_idx not in side:
            continue
        
        start_side = side[start_idx]
        
        # BFS to find connected component
        component = [start_idx]
        component_area = triangles[start_idx].area
        processed.add(start_idx)
        head = 0
        
        while head < len(component):
            idx = component[head]
            head += 1
            
            for neighbor in adjacency.get(idx, []):
                if neighbor in processed:
                    continue
                if neighbor not in outer_triangles:
                    continue
                if side.get(neighbor) != start_side:
                    continue
                
                processed.add(neighbor)
                component.append(neighbor)
                component_area += triangles[neighbor].area
        
        # Flip small-area components
        if component_area < area_threshold:
            new_side = 2 if start_side == 1 else 1
            for idx in component:
                side[idx] = new_side


# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================

def classify_mold_halves(
    cavity_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    d1: np.ndarray,
    d2: np.ndarray,
    boundary_zone_threshold: float = DEFAULT_BOUNDARY_ZONE_THRESHOLD,
    part_mesh: trimesh.Trimesh = None,
    use_proximity_method: bool = False,
    use_fast_method: bool = True,
    use_gpu: bool = True,
    tet_vertices: np.ndarray = None,
    tet_boundary_labels: np.ndarray = None
) -> MoldHalfClassificationResult:
    """
    Classify boundary triangles of mold cavity into two mold halves.
    
    Only classifies the OUTER boundary (hull surface), not the inner boundary (part surface).
    
    Uses GPU acceleration when available for large meshes (>10k triangles).
    
    Args:
        cavity_mesh: The mold cavity mesh (or tetrahedral boundary mesh)
        hull_mesh: The original hull mesh (used to identify outer boundary)
        d1: First parting direction (pull direction for H₁)
        d2: Second parting direction (pull direction for H₂)
        boundary_zone_threshold: Threshold for boundary zone (0-1, fraction of interface)
        part_mesh: Original part mesh (optional, used for proximity-based classification)
        use_proximity_method: If True, use proximity-based outer boundary detection
                             (better for tetrahedral boundary meshes)
        use_fast_method: If True, use fast greedy region-growing (O(n) per paper)
                        If False, use original morphological smoothing approach
        tet_vertices: (N, 3) tet mesh vertices (if available, enables fast label-based detection)
        tet_boundary_labels: (N,) boundary labels from tet mesh (-1=inner, 1=H1, 2=H2)
    
    Returns:
        MoldHalfClassificationResult with classification data
    """
    import time
    start_time = time.time()
    
    # Normalize directions
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    n_triangles = len(cavity_mesh.faces)
    n_hull_faces = len(hull_mesh.faces)
    
    # Log available acceleration options
    logger.info(f"Mold half classification: {n_triangles} boundary faces, {n_hull_faces} hull faces")
    logger.debug(f"Acceleration: CUDA={CUDA_AVAILABLE}, C++={CPP_FAST_AVAILABLE}, "
                f"tet_labels={'yes' if tet_boundary_labels is not None else 'no'}, "
                f"part_mesh={'yes' if part_mesh is not None else 'no'}")
    
    # Compute boundary zone hops from threshold
    # Scale: 0% = 0 hops, 15% = 3 hops, 30% = 8 hops
    boundary_zone_hops = round(boundary_zone_threshold * boundary_zone_threshold * 180)
    
    # =========================================================================
    # FASTEST PATH: Use CUDA GPU pipeline when available
    # =========================================================================
    if CUDA_AVAILABLE and use_gpu and tet_vertices is not None and tet_boundary_labels is not None:
        logger.debug(f"Using CUDA GPU pipeline for {n_triangles} faces")
        return _classify_mold_halves_complete_cuda(
            cavity_mesh, tet_vertices, tet_boundary_labels,
            d1, d2, boundary_zone_hops
        )
    
    # =========================================================================
    # FAST PATH: Use complete C++ pipeline when tet labels available
    # =========================================================================
    if CPP_FAST_AVAILABLE and tet_vertices is not None and tet_boundary_labels is not None:
        logger.debug(f"Using complete C++ pipeline for {n_triangles} faces")
        return _classify_mold_halves_complete_cpp(
            cavity_mesh, tet_vertices, tet_boundary_labels,
            d1, d2, boundary_zone_hops
        )
    
    # =========================================================================
    # FAST PATH: Hull-based classification (classify small hull, map to boundary)
    # This is much faster than the Python fallback because:
    # 1. Hull classification uses fast region-growing O(hull_faces)
    # 2. Outer detection uses trimesh's optimized on_surface() 
    # 3. Mapping uses KDTree O(boundary_faces * log(hull_faces))
    # Always prefer this path when part_mesh is available.
    # =========================================================================
    if part_mesh is not None:
        # Use fast hull-based approach - always faster than Python fallback
        logger.info(f"Using fast hull-based classification (hull={len(hull_mesh.faces)}, boundary={n_triangles})")
        return classify_mold_halves_via_hull(
            cavity_mesh, hull_mesh, part_mesh,
            d1, d2, boundary_zone_threshold
        )
    
    # =========================================================================
    # FALLBACK: Python pipeline (for when tet labels not available)
    # WARNING: This path is slow for large meshes (>50k faces)
    # Consider using the hull-based path by providing part_mesh
    # =========================================================================
    logger.warning(f"Using slow Python fallback for {n_triangles} faces. "
                  f"Consider providing part_mesh for fast hull-based classification.")
    
    step_start = time.time()
    
    # Step 1: Identify outer boundary triangles
    if tet_vertices is not None and tet_boundary_labels is not None:
        # Fast O(n) path using pre-computed tet boundary labels
        outer_triangles, inner_boundary_triangles = identify_outer_boundary_from_tet_labels(
            cavity_mesh, tet_vertices, tet_boundary_labels
        )
        logger.info(f"Step 1 (tet labels): {len(outer_triangles)} outer, {len(inner_boundary_triangles)} inner in {(time.time()-step_start)*1000:.0f}ms")
    elif part_mesh is not None:
        # FAST: Use distance-based detection (O(n) with KD-tree)
        outer_triangles, inner_boundary_triangles = identify_outer_boundary_by_distance(
            cavity_mesh, part_mesh
        )
        logger.debug(f"Using distance method: {len(outer_triangles)} outer, {len(inner_boundary_triangles)} inner")
    elif use_proximity_method:
        # Slower proximity-based method (for legacy compatibility)
        triangles_info = extract_triangle_info(cavity_mesh)
        outer_triangles = identify_outer_boundary_by_proximity(cavity_mesh, hull_mesh, part_mesh)
        inner_boundary_triangles = set(range(n_triangles)) - outer_triangles
    else:
        # Plane-based method (original CSG cavity mesh)
        triangles_info = extract_triangle_info(cavity_mesh)
        outer_triangles = identify_outer_boundary_from_hull(triangles_info, hull_mesh)
        inner_boundary_triangles = set(range(n_triangles)) - outer_triangles
    
    logger.debug(f"Outer boundary: {len(outer_triangles)} triangles")
    logger.debug(f"Inner boundary (part surface): {len(inner_boundary_triangles)} triangles")
    
    # If hull matching found very few outer triangles, treat ALL as outer
    effective_outer_triangles = outer_triangles
    if len(outer_triangles) < n_triangles * MIN_OUTER_TRIANGLE_FRACTION:
        logger.warning("Hull matching found too few triangles, using all triangles")
        effective_outer_triangles = set(range(n_triangles))
    
    # Step 2: Build triangle adjacency on the FULL cavity mesh
    step_start = time.time()
    adjacency = build_triangle_adjacency(cavity_mesh)
    logger.info(f"Step 2 (adjacency): built in {(time.time()-step_start)*1000:.0f}ms")
    
    # Early exit if no outer triangles
    if len(effective_outer_triangles) == 0:
        logger.warning("No outer triangles to classify")
        return MoldHalfClassificationResult(
            side_map={},
            h1_triangles=set(),
            h2_triangles=set(),
            boundary_zone_triangles=set(),
            inner_boundary_triangles=inner_boundary_triangles,
            total_triangles=n_triangles,
            outer_boundary_count=0,
        )
    
    # Step 3: Classify using chosen method
    # Step 3: Classify using chosen method
    step_start = time.time()
    if use_fast_method:
        # Fast O(n) greedy region-growing (per research paper)
        # Uses GPU when available for large meshes
        side = classify_mold_halves_fast(cavity_mesh, effective_outer_triangles, d1, d2, adjacency, use_gpu=use_gpu)
    else:
        # Original morphological smoothing approach (requires triangle info)
        triangles_info = extract_triangle_info(cavity_mesh)
        side = classify_and_smooth(triangles_info, adjacency, effective_outer_triangles, d1, d2)
    logger.info(f"Step 3 (classification): completed in {(time.time()-step_start)*1000:.0f}ms")
    
    # Step 4: Compute boundary zone parameters
    step_start = time.time()
    # Pre-compute neighbors filtered to outer triangles - VECTORIZED
    # Convert outer triangles to array for fast lookup
    outer_array = np.array(list(effective_outer_triangles), dtype=np.int64)
    outer_set = effective_outer_triangles  # Keep set for O(1) membership test
    
    # Build neighbors using vectorized numpy operations where possible
    neighbors: Dict[int, List[int]] = {}
    for i in outer_array:
        neighbors[i] = [n for n in adjacency.get(i, []) if n in outer_set]
    
    # Find interface triangles (where H₁ meets H₂)
    interface_triangles: Set[int] = set()
    for tri_idx in outer_array:
        my_side = side.get(tri_idx)
        if my_side is None:
            continue
        
        for neighbor_idx in neighbors.get(tri_idx, []):
            neighbor_side = side.get(neighbor_idx)
            if neighbor_side is not None and neighbor_side != my_side:
                interface_triangles.add(tri_idx)
                break
    
    logger.info(f"Step 4 (interface): {len(interface_triangles)} triangles in {(time.time()-step_start)*1000:.0f}ms")
    
    # Step 5: Expand boundary zone using BFS from interface triangles
    # Scale: 0% = 0 hops, 15% = 3 hops, 30% = 8 hops
    boundary_zone_hops = round(boundary_zone_threshold * boundary_zone_threshold * 180)
    logger.debug(f"Boundary zone threshold: {boundary_zone_threshold*100:.0f}% → {boundary_zone_hops} hops")
    
    boundary_zone_triangles = set(interface_triangles)
    distance: Dict[int, int] = {}
    
    # Initialize distances for interface triangles
    for tri_idx in interface_triangles:
        distance[tri_idx] = 0
    # Step 5: Expand boundary zone using BFS from interface triangles
    step_start = time.time()
    
    # BFS to expand boundary zone
    queue = list(interface_triangles)
    head = 0
    
    while head < len(queue):
        tri_idx = queue[head]
        head += 1
        dist = distance[tri_idx]
        
        if dist >= boundary_zone_hops:
            continue
        
        for neighbor_idx in neighbors.get(tri_idx, []):
            if neighbor_idx not in distance:
                distance[neighbor_idx] = dist + 1
                boundary_zone_triangles.add(neighbor_idx)
                queue.append(neighbor_idx)
    
    logger.info(f"Step 5 (BFS): {len(boundary_zone_triangles)} boundary zone in {(time.time()-step_start)*1000:.0f}ms")
    
    # Step 6: Build result sets
    step_start = time.time()
    
    # Remove boundary zone triangles from H₁ and H₂ sets
    h1_triangles: Set[int] = set()
    h2_triangles: Set[int] = set()
    
    # First pass: collect H1 and H2 triangles (excluding boundary zone)
    for tri_idx, mold_side in side.items():
        if tri_idx in boundary_zone_triangles:
            continue  # Skip boundary zone triangles
        elif mold_side == 1:
            h1_triangles.add(tri_idx)
        else:
            h2_triangles.add(tri_idx)
    
    # Second pass: remove boundary zone triangles from side map
    for tri_idx in boundary_zone_triangles:
        side.pop(tri_idx, None)
    
    logger.info(f"Step 6 (result sets): built in {(time.time()-step_start)*1000:.0f}ms")
    
    elapsed = time.time() - start_time
    
    # Summary logging
    logger.info(
        f"Mold Half Classification: H1={len(h1_triangles)}, H2={len(h2_triangles)}, "
        f"boundary={len(boundary_zone_triangles)}, inner={len(inner_boundary_triangles)}, "
        f"time={elapsed*1000:.0f}ms"
    )
    
    return MoldHalfClassificationResult(
        side_map=side,
        h1_triangles=h1_triangles,
        h2_triangles=h2_triangles,
        boundary_zone_triangles=boundary_zone_triangles,
        inner_boundary_triangles=inner_boundary_triangles,
        total_triangles=n_triangles,
        outer_boundary_count=len(effective_outer_triangles),
    )


def classify_mold_halves_via_hull(
    boundary_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    d1: np.ndarray,
    d2: np.ndarray,
    boundary_zone_threshold: float = DEFAULT_BOUNDARY_ZONE_THRESHOLD
) -> MoldHalfClassificationResult:
    """
    Fast mold half classification by classifying hull first, then mapping to boundary mesh.
    
    This is MUCH faster than classifying the full boundary mesh because:
    - Hull mesh: typically 1-5k faces
    - Tet boundary mesh: can be 50-500k faces
    
    Algorithm:
    1. Classify hull mesh directly using fast region growing (O(hull_faces))
    2. Identify outer vs inner triangles on boundary mesh using distance to part (O(boundary_faces))
    3. Map outer triangles to hull classification using nearest centroid (O(outer_faces * log(hull_faces)))
    
    Args:
        boundary_mesh: The tetrahedral boundary mesh (hull + part surfaces)
        hull_mesh: The original hull mesh (much smaller)
        part_mesh: The original part mesh
        d1: First parting direction
        d2: Second parting direction
        boundary_zone_threshold: Threshold for boundary zone
    
    Returns:
        MoldHalfClassificationResult
    """
    from scipy.spatial import cKDTree
    import time
    start_time = time.time()
    
    n_boundary_faces = len(boundary_mesh.faces)
    n_hull_faces = len(hull_mesh.faces)
    
    # Normalize directions
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    # =========================================================================
    # Step 1: Classify hull mesh (FAST - small mesh)
    # 
    # Per paper Section 4.1: boundary zone is defined by PHYSICAL DISTANCE
    # "we set the threshold to 15% of the convex hull bounding box diagonal"
    # =========================================================================
    hull_start = time.time()
    hull_classification = classify_hull_faces_fast(hull_mesh, d1, d2, boundary_zone_threshold)
    hull_time = (time.time() - hull_start) * 1000
    
    logger.debug(f"Hull classification in {hull_time:.0f}ms: "
                f"H1={len(hull_classification.h1_triangles)}, H2={len(hull_classification.h2_triangles)}")
    
    # =========================================================================
    # Step 2: Identify outer vs inner triangles on boundary mesh
    # =========================================================================
    outer_start = time.time()
    outer_triangles, inner_triangles = identify_outer_boundary_by_distance(boundary_mesh, part_mesh)
    outer_time = (time.time() - outer_start) * 1000
    
    logger.debug(f"Outer detection in {outer_time:.0f}ms: "
                f"{len(outer_triangles)} outer, {len(inner_triangles)} inner")
    
    if len(outer_triangles) == 0:
        # All triangles are inner (shouldn't happen normally)
        return MoldHalfClassificationResult(
            side_map={},
            h1_triangles=set(),
            h2_triangles=set(),
            boundary_zone_triangles=set(),
            inner_boundary_triangles=inner_triangles,
            total_triangles=n_boundary_faces,
            outer_boundary_count=0,
        )
    
    # =========================================================================
    # Step 3: Map outer triangles to hull classification using nearest centroid
    # =========================================================================
    map_start = time.time()
    
    # Compute hull face centroids
    hull_verts = hull_mesh.vertices
    hull_faces = hull_mesh.faces
    hull_centroids = hull_verts[hull_faces].mean(axis=1)  # (n_hull, 3)
    
    # Build KD-tree for hull centroids
    hull_tree = cKDTree(hull_centroids)
    
    # Compute boundary face centroids (only for outer faces)
    boundary_verts = boundary_mesh.vertices
    boundary_faces = boundary_mesh.faces
    outer_indices = np.array(list(outer_triangles), dtype=np.int64)
    outer_centroids = boundary_verts[boundary_faces[outer_indices]].mean(axis=1)  # (n_outer, 3)
    
    # Find nearest hull face for each outer boundary face
    _, nearest_hull = hull_tree.query(outer_centroids, k=1)  # (n_outer,)
    
    # Create label array for hull faces: 1=H1, 2=H2, 0=boundary_zone
    hull_labels = np.zeros(n_hull_faces, dtype=np.int8)
    for idx in hull_classification.h1_triangles:
        hull_labels[idx] = 1
    for idx in hull_classification.h2_triangles:
        hull_labels[idx] = 2
    # boundary_zone stays 0
    
    # Map labels to outer boundary faces
    outer_labels = hull_labels[nearest_hull]  # (n_outer,)
    
    map_time = (time.time() - map_start) * 1000
    
    # =========================================================================
    # Step 4: Build result sets
    # =========================================================================
    h1_triangles = set(outer_indices[outer_labels == 1])
    h2_triangles = set(outer_indices[outer_labels == 2])
    boundary_zone_triangles = set(outer_indices[outer_labels == 0])
    
    # Build side_map
    side_map = {}
    for idx in h1_triangles:
        side_map[idx] = 1
    for idx in h2_triangles:
        side_map[idx] = 2
    
    total_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"[Hull-based] Mold half classification in {total_time:.0f}ms "
        f"(hull:{hull_time:.0f}ms, outer:{outer_time:.0f}ms, map:{map_time:.0f}ms): "
        f"H1={len(h1_triangles)}, H2={len(h2_triangles)}, "
        f"boundary={len(boundary_zone_triangles)}, inner={len(inner_triangles)}"
    )
    
    return MoldHalfClassificationResult(
        side_map=side_map,
        h1_triangles=h1_triangles,
        h2_triangles=h2_triangles,
        boundary_zone_triangles=boundary_zone_triangles,
        inner_boundary_triangles=inner_triangles,
        total_triangles=n_boundary_faces,
        outer_boundary_count=len(outer_triangles),
    )
