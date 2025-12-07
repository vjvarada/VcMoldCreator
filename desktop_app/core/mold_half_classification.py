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
from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict

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

logger = logging.getLogger(__name__)


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
# COLORS FOR MOLD HALVES (Same as parting directions)
# ============================================================================

class MoldHalfColors:
    """Colors for mold half visualization (RGBA, 0-255)."""
    H1 = np.array([0, 255, 0, 255], dtype=np.uint8)           # Green (same as D1)
    H2 = np.array([255, 102, 0, 255], dtype=np.uint8)         # Orange (same as D2)
    BOUNDARY_ZONE = np.array([180, 180, 180, 255], dtype=np.uint8)  # Light gray
    INNER = np.array([80, 80, 80, 255], dtype=np.uint8)       # Dark gray
    UNCLASSIFIED = np.array([136, 136, 136, 255], dtype=np.uint8)   # Gray


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
    tolerance = max_dim * 0.001  # 0.1% of hull size
    
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
    tolerance = hull_size * 0.02  # 2% of hull size - more generous for tet meshes
    
    # Use GPU for large meshes
    if use_gpu and CUDA_AVAILABLE and n_faces > 5000:
        hull_distances = _compute_distances_to_mesh_gpu(centroids, hull_mesh)
        
        if part_mesh is not None:
            part_distances = _compute_distances_to_mesh_gpu(centroids, part_mesh)
            outer_mask = hull_distances < part_distances
            n_outer = np.sum(outer_mask)
            n_inner = np.sum(~outer_mask)
            logger.info(f"GPU proximity classification: {n_outer} outer, {n_inner} inner")
            logger.info(f"  Hull distances: min={hull_distances.min():.4f}, max={hull_distances.max():.4f}, mean={hull_distances.mean():.4f}")
            logger.info(f"  Part distances: min={part_distances.min():.4f}, max={part_distances.max():.4f}, mean={part_distances.mean():.4f}")
        else:
            outer_mask = hull_distances < tolerance
            logger.debug(f"GPU threshold classification (tol={tolerance:.4f}): {np.sum(outer_mask)} outer, {np.sum(~outer_mask)} inner")
    else:
        # CPU path using trimesh
        _, hull_distances, _ = hull_mesh.nearest.on_surface(centroids)
        
        if part_mesh is not None:
            _, part_distances, _ = part_mesh.nearest.on_surface(centroids)
            outer_mask = hull_distances < part_distances
            n_outer = np.sum(outer_mask)
            n_inner = np.sum(~outer_mask)
            logger.info(f"Proximity classification: {n_outer} outer, {n_inner} inner")
            logger.info(f"  Hull distances: min={hull_distances.min():.4f}, max={hull_distances.max():.4f}, mean={hull_distances.mean():.4f}")
            logger.info(f"  Part distances: min={part_distances.min():.4f}, max={part_distances.max():.4f}, mean={part_distances.mean():.4f}")
        else:
            outer_mask = hull_distances < tolerance
            logger.debug(f"Threshold classification (tol={tolerance:.4f}): {np.sum(outer_mask)} outer, {np.sum(~outer_mask)} inner")
    
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
    strong_threshold = 0.7
    strong_h1 = (dot1 > strong_threshold) & (dot1 > dot2 + 0.3)
    strong_h2 = (dot2 > strong_threshold) & (dot2 > dot1 + 0.3)
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
    area_threshold_ratio: float = 0.02
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
    boundary_zone_threshold: float = 0.15,
    part_mesh: trimesh.Trimesh = None,
    use_proximity_method: bool = False
) -> MoldHalfClassificationResult:
    """
    Classify boundary triangles of mold cavity into two mold halves.
    
    Only classifies the OUTER boundary (hull surface), not the inner boundary (part surface).
    
    Args:
        cavity_mesh: The mold cavity mesh (or tetrahedral boundary mesh)
        hull_mesh: The original hull mesh (used to identify outer boundary)
        d1: First parting direction (pull direction for H₁)
        d2: Second parting direction (pull direction for H₂)
        boundary_zone_threshold: Threshold for boundary zone (0-1, fraction of interface)
        part_mesh: Original part mesh (optional, used for proximity-based classification)
        use_proximity_method: If True, use proximity-based outer boundary detection
                             (better for tetrahedral boundary meshes)
    
    Returns:
        MoldHalfClassificationResult with classification data
    """
    import time
    start_time = time.time()
    
    # Normalize directions
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    # Step 1: Extract triangle information from cavity
    triangles = extract_triangle_info(cavity_mesh)
    
    # Step 2: Identify outer boundary triangles
    if use_proximity_method or part_mesh is not None:
        # Use proximity-based method (better for tetrahedral boundary meshes)
        outer_triangles = identify_outer_boundary_by_proximity(cavity_mesh, hull_mesh, part_mesh)
    else:
        # Use plane-based method (original CSG cavity mesh)
        outer_triangles = identify_outer_boundary_from_hull(triangles, hull_mesh)
    
    inner_boundary_triangles = set(range(len(triangles))) - outer_triangles
    
    logger.debug(f"Outer boundary: {len(outer_triangles)} triangles")
    logger.debug(f"Inner boundary (part surface): {len(inner_boundary_triangles)} triangles")
    
    # If hull matching found very few outer triangles, treat ALL as outer
    effective_outer_triangles = outer_triangles
    if len(outer_triangles) < len(triangles) * 0.1:
        logger.warning("Hull matching found too few triangles, using all triangles")
        effective_outer_triangles = set(range(len(triangles)))
    
    # Step 3: Build triangle adjacency on the FULL cavity mesh
    adjacency = build_triangle_adjacency(cavity_mesh)
    
    # Early exit if no outer triangles
    if len(effective_outer_triangles) == 0:
        logger.warning("No outer triangles to classify")
        return MoldHalfClassificationResult(
            side_map={},
            h1_triangles=set(),
            h2_triangles=set(),
            boundary_zone_triangles=set(),
            inner_boundary_triangles=inner_boundary_triangles,
            total_triangles=len(triangles),
            outer_boundary_count=0,
        )
    
    # Step 4: Classify and smooth
    side = classify_and_smooth(triangles, adjacency, effective_outer_triangles, d1, d2)
    
    # Step 5: Compute boundary zone parameters
    # Pre-compute neighbors filtered to outer triangles
    neighbors: Dict[int, List[int]] = {}
    for i in effective_outer_triangles:
        neighbors[i] = [n for n in adjacency.get(i, []) if n in effective_outer_triangles]
    
    # Find interface triangles (where H₁ meets H₂)
    interface_triangles: Set[int] = set()
    for tri_idx in effective_outer_triangles:
        my_side = side.get(tri_idx)
        if my_side is None:
            continue
        
        for neighbor_idx in neighbors.get(tri_idx, []):
            neighbor_side = side.get(neighbor_idx)
            if neighbor_side is not None and neighbor_side != my_side:
                interface_triangles.add(tri_idx)
                break
    
    logger.debug(f"Interface triangles (H₁/H₂ border): {len(interface_triangles)}")
    
    # Step 6: Expand boundary zone using BFS from interface triangles
    # Scale: 0% = 0 hops, 15% = 3 hops, 30% = 8 hops
    boundary_zone_hops = round(boundary_zone_threshold * boundary_zone_threshold * 180)
    logger.debug(f"Boundary zone threshold: {boundary_zone_threshold*100:.0f}% → {boundary_zone_hops} hops")
    
    boundary_zone_triangles = set(interface_triangles)
    distance: Dict[int, int] = {}
    
    # Initialize distances for interface triangles
    for tri_idx in interface_triangles:
        distance[tri_idx] = 0
    
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
    
    logger.debug(f"Boundary zone triangles: {len(boundary_zone_triangles)}")
    
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
        total_triangles=len(triangles),
        outer_boundary_count=len(effective_outer_triangles),
    )


def get_mold_half_face_colors(
    n_faces: int,
    classification: MoldHalfClassificationResult
) -> np.ndarray:
    """
    Generate face colors array for mold half visualization.
    
    Args:
        n_faces: Number of faces in the mesh
        classification: The mold half classification result
    
    Returns:
        Array of shape (n_faces, 4) with RGBA colors (0-255)
    """
    colors = np.tile(MoldHalfColors.UNCLASSIFIED, (n_faces, 1))
    
    # Color H₁ triangles (green)
    for tri_idx in classification.h1_triangles:
        if tri_idx < n_faces:
            colors[tri_idx] = MoldHalfColors.H1
    
    # Color H₂ triangles (orange)
    for tri_idx in classification.h2_triangles:
        if tri_idx < n_faces:
            colors[tri_idx] = MoldHalfColors.H2
    
    # Color boundary zone triangles (light gray)
    for tri_idx in classification.boundary_zone_triangles:
        if tri_idx < n_faces:
            colors[tri_idx] = MoldHalfColors.BOUNDARY_ZONE
    
    # Color inner boundary triangles (dark gray)
    for tri_idx in classification.inner_boundary_triangles:
        if tri_idx < n_faces:
            colors[tri_idx] = MoldHalfColors.INNER
    
    return colors
