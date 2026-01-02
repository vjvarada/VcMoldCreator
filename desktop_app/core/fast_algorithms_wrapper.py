"""
Fast Algorithms Wrapper

This module provides a unified interface to fast C++ implementations
with automatic fallback to Python implementations if C++ is not available.

Usage:
    from core.fast_algorithms_wrapper import run_dijkstra_escape_labeling_fast
    
    result = run_dijkstra_escape_labeling_fast(tet_result, use_weighted_edges=True)

Building C++ Module:
    cd desktop_app/core
    python setup_cpp.py build_ext --inplace
"""

import logging
import time
from typing import Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import C++ module
try:
    from . import fast_algorithms as _cpp
    CPP_AVAILABLE = True
    logger.info("C++ fast_algorithms module loaded successfully - using optimized implementations")
except ImportError:
    CPP_AVAILABLE = False
    logger.info("C++ fast_algorithms not compiled - using Python fallback. "
                "For 10-20x speedup, run: cd desktop_app/core && python setup_cpp.py build_ext --inplace")


def run_dijkstra_escape_labeling_fast(
    tet_result,  # TetrahedralMeshResult
    use_weighted_edges: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """
    Run Dijkstra's algorithm for escape labeling using C++ if available.
    
    This function automatically selects between C++ (fast) and Python (fallback)
    implementations based on availability.
    
    Args:
        tet_result: TetrahedralMeshResult with boundary_labels and edge_weights computed
        use_weighted_edges: If True, use edge_length * edge_weight as cost
    
    Returns:
        Tuple of:
            - escape_labels: (I,) int8 array (1=H1, 2=H2, 0=unreachable)
            - vertex_indices: (I,) int64 array of interior vertex indices
            - distances: (I,) float64 array of shortest distances
            - destinations: (I,) int64 array of destination boundary vertices
            - paths: List of paths (each path is a list of vertex indices)
    """
    if CPP_AVAILABLE:
        return _run_dijkstra_cpp(tet_result, use_weighted_edges)
    else:
        return _run_dijkstra_python(tet_result, use_weighted_edges)


def _run_dijkstra_cpp(tet_result, use_weighted_edges: bool):
    """C++ implementation wrapper."""
    start_time = time.time()
    
    edges = np.asarray(tet_result.edges, dtype=np.int64)
    edge_lengths = np.asarray(tet_result.edge_lengths, dtype=np.float64)
    edge_weights = np.asarray(tet_result.edge_weights, dtype=np.float64)
    boundary_labels = np.asarray(tet_result.boundary_labels, dtype=np.int8)
    
    result = _cpp.dijkstra_escape_labeling(
        edges, edge_lengths, edge_weights, boundary_labels, use_weighted_edges
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    # Count results
    labels = np.asarray(result.escape_labels)
    n_h1 = np.sum(labels == 1)
    n_h2 = np.sum(labels == 2)
    n_unreached = np.sum(labels == 0)
    
    logger.info(f"[C++] Dijkstra complete in {elapsed:.0f}ms: {n_h1}→H1, {n_h2}→H2, {n_unreached} unreached")
    
    return (
        np.asarray(result.escape_labels),
        np.asarray(result.vertex_indices),
        np.asarray(result.distances),
        np.asarray(result.destinations),
        list(result.paths)
    )


def _run_dijkstra_python(tet_result, use_weighted_edges: bool):
    """Python fallback implementation."""
    import heapq
    
    start_time = time.time()
    
    vertices = tet_result.vertices
    edges = tet_result.edges
    edge_lengths = tet_result.edge_lengths
    edge_weights = tet_result.edge_weights
    boundary_labels = tet_result.boundary_labels
    
    n_vertices = len(vertices)
    
    # Compute edge costs
    if use_weighted_edges and edge_weights is not None:
        edge_costs = edge_lengths * edge_weights
    else:
        edge_costs = edge_lengths.copy()
    
    # Build adjacency list
    adjacency = {i: [] for i in range(n_vertices)}
    for edge_idx, (v0, v1) in enumerate(edges):
        cost = edge_costs[edge_idx]
        adjacency[v0].append((v1, cost))
        adjacency[v1].append((v0, cost))
    
    # Identify interior vertices and boundary sets
    interior_vertex_indices = np.where((boundary_labels != 1) & (boundary_labels != 2))[0]
    h1_vertices = set(np.where(boundary_labels == 1)[0])
    h2_vertices = set(np.where(boundary_labels == 2)[0])
    
    n_interior = len(interior_vertex_indices)
    
    # Run Dijkstra from each interior vertex
    escape_labels = np.zeros(n_interior, dtype=np.int8)
    distances = np.full(n_interior, np.inf, dtype=np.float64)
    destinations = np.full(n_interior, -1, dtype=np.int64)
    paths = []
    
    for idx, start_v in enumerate(interior_vertex_indices):
        dist = np.full(n_vertices, np.inf, dtype=np.float64)
        dist[start_v] = 0.0
        predecessor = np.full(n_vertices, -1, dtype=np.int64)
        visited = np.zeros(n_vertices, dtype=bool)
        
        pq = [(0.0, start_v)]
        destination = -1
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if visited[u]:
                continue
            visited[u] = True
            
            if u in h1_vertices:
                escape_labels[idx] = 1
                distances[idx] = d
                destination = u
                break
            elif u in h2_vertices:
                escape_labels[idx] = 2
                distances[idx] = d
                destination = u
                break
            
            for v, cost in adjacency[u]:
                if visited[v]:
                    continue
                new_dist = d + cost
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    predecessor[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        destinations[idx] = destination
        
        # Reconstruct path
        if destination >= 0:
            path = []
            current = destination
            while current >= 0:
                path.append(current)
                current = predecessor[current]
            path.reverse()
            paths.append(path)
        else:
            paths.append([])
    
    elapsed = (time.time() - start_time) * 1000
    
    n_h1 = np.sum(escape_labels == 1)
    n_h2 = np.sum(escape_labels == 2)
    n_unreached = np.sum(escape_labels == 0)
    
    logger.info(f"[Python] Dijkstra complete in {elapsed:.0f}ms: {n_h1}→H1, {n_h2}→H2, {n_unreached} unreached")
    
    return escape_labels, interior_vertex_indices, distances, destinations, paths


def compute_edge_boundary_labels_fast(
    tet_result,  # TetrahedralMeshResult
) -> np.ndarray:
    """
    Compute edge boundary labels using C++ if available.
    
    Args:
        tet_result: TetrahedralMeshResult with boundary_labels and boundary_mesh
    
    Returns:
        (E,) int8 array of edge labels: 0=interior, 1=H1, 2=H2, -1=inner, -2=mixed
    """
    if CPP_AVAILABLE:
        return _compute_edge_labels_cpp(tet_result)
    else:
        return _compute_edge_labels_python(tet_result)


def _compute_edge_labels_cpp(tet_result) -> np.ndarray:
    """C++ implementation wrapper."""
    start_time = time.time()
    
    edges = np.asarray(tet_result.edges, dtype=np.int64)
    boundary_labels = np.asarray(tet_result.boundary_labels, dtype=np.int8)
    boundary_faces = np.asarray(tet_result.boundary_mesh.faces, dtype=np.int64)
    boundary_mesh_vertices = np.asarray(tet_result.boundary_mesh.vertices, dtype=np.float64)
    tet_vertices = np.asarray(tet_result.vertices, dtype=np.float64)
    
    result = _cpp.compute_edge_boundary_labels(
        edges, boundary_labels, boundary_faces, boundary_mesh_vertices, tet_vertices
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    # Count stats
    n_interior = np.sum(result == 0)
    n_h1 = np.sum(result == 1)
    n_h2 = np.sum(result == 2)
    n_inner = np.sum(result == -1)
    n_mixed = np.sum(result == -2)
    
    logger.info(
        f"[C++] Edge labels computed in {elapsed:.0f}ms: "
        f"interior={n_interior}, H1={n_h1}, H2={n_h2}, inner={n_inner}, mixed={n_mixed}"
    )
    
    return np.asarray(result)


def _compute_edge_labels_python(tet_result) -> np.ndarray:
    """Python fallback - calls the original implementation."""
    # Import here to avoid circular imports
    from .tetrahedral_mesh import compute_edge_boundary_labels as _original
    return _original(tet_result)


def find_secondary_cuts_fast(
    edge_membrane_data: list,
    seed_triangles: list,
    seed_triangle_positions: list,
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int = 20,
    min_membrane_thickness: float = 0.0
) -> list:
    """
    Find secondary cutting edges using C++ if available, with Python fallback.
    
    This function automatically selects between C++ (fast, with OpenMP parallelization)
    and Python (fallback) implementations based on availability.
    
    Uses boolean intersection detection with minimum count threshold.
    
    Args:
        edge_membrane_data: List of (vi, vj, path_i, path_j, boundary_path) tuples
        seed_triangles: List of (v0, v1, v2) vertex index tuples
        seed_triangle_positions: List of (pos0, pos1, pos2) position tuples
        vertices: (N, 3) array of tetrahedral mesh vertices
        boundary_verts: (B, 3) array of boundary mesh vertices
        min_intersection_count: Minimum number of segment-triangle intersections required (1-50)
        min_membrane_thickness: Skip membranes thinner than this
    
    Returns:
        List of (vi, vj) tuples representing secondary cutting edges
    """
    if CPP_AVAILABLE:
        return _find_secondary_cuts_cpp(
            edge_membrane_data, seed_triangles, seed_triangle_positions,
            vertices, boundary_verts, min_intersection_count, min_membrane_thickness
        )
    else:
        return _find_secondary_cuts_python(
            edge_membrane_data, seed_triangles, seed_triangle_positions,
            vertices, boundary_verts, min_intersection_count, min_membrane_thickness
        )


def _find_secondary_cuts_cpp(
    edge_membrane_data: list,
    seed_triangles: list,
    seed_triangle_positions: list,
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int,
    min_membrane_thickness: float
) -> list:
    """C++ implementation wrapper for secondary cuts detection."""
    start_time = time.time()
    
    n_membranes = len(edge_membrane_data)
    n_seed_tris = len(seed_triangles)
    
    if n_membranes == 0 or n_seed_tris == 0:
        return []
    
    # Prepare flattened arrays for C++ (membrane data)
    membrane_edge_vi = np.zeros(n_membranes, dtype=np.int64)
    membrane_edge_vj = np.zeros(n_membranes, dtype=np.int64)
    
    path_i_list = []
    path_i_offsets = np.zeros(n_membranes, dtype=np.int64)
    path_j_list = []
    path_j_offsets = np.zeros(n_membranes, dtype=np.int64)
    boundary_list = []
    boundary_offsets = np.zeros(n_membranes, dtype=np.int64)
    
    for m, (vi, vj, path_i, path_j, boundary_path) in enumerate(edge_membrane_data):
        membrane_edge_vi[m] = vi
        membrane_edge_vj[m] = vj
        
        path_i_offsets[m] = len(path_i_list)
        path_i_list.extend(path_i)
        
        path_j_offsets[m] = len(path_j_list)
        path_j_list.extend(path_j)
        
        boundary_offsets[m] = len(boundary_list)
        boundary_list.extend(boundary_path)
    
    membrane_path_i = np.array(path_i_list, dtype=np.int64)
    membrane_path_j = np.array(path_j_list, dtype=np.int64)
    membrane_boundary_path = np.array(boundary_list, dtype=np.int64)
    
    # Prepare seed triangle arrays for C++
    # seed_triangles: (T, 3, 3) positions
    # seed_triangle_vertices: (T, 3) vertex indices
    seed_tri_positions = np.zeros((n_seed_tris, 3, 3), dtype=np.float64)
    seed_tri_vertices = np.zeros((n_seed_tris, 3), dtype=np.int64)
    
    for t, ((v0, v1, v2), (p0, p1, p2)) in enumerate(zip(seed_triangles, seed_triangle_positions)):
        seed_tri_vertices[t] = [v0, v1, v2]
        seed_tri_positions[t, 0] = p0
        seed_tri_positions[t, 1] = p1
        seed_tri_positions[t, 2] = p2
    
    # Ensure vertices arrays are contiguous
    tet_vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    boundary_vertices = np.ascontiguousarray(boundary_verts, dtype=np.float64)
    
    # Call C++ implementation
    result = _cpp.find_secondary_cuts(
        membrane_edge_vi,
        membrane_edge_vj,
        membrane_path_i,
        path_i_offsets,
        membrane_path_j,
        path_j_offsets,
        membrane_boundary_path,
        boundary_offsets,
        seed_tri_positions,
        seed_tri_vertices,
        tet_vertices,
        boundary_vertices,
        min_intersection_count,
        min_membrane_thickness
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    # Convert result to list of tuples
    cutting_edges = [(vi, vj) for vi, vj in result.cutting_edges]
    
    logger.info(
        f"[C++] Secondary cuts complete in {elapsed:.0f}ms: "
        f"{len(cutting_edges)} cuts found from {result.n_membranes_checked} membranes "
        f"(min_intersection_count={min_intersection_count})"
    )
    
    return cutting_edges


def _find_secondary_cuts_python(
    edge_membrane_data: list,
    seed_triangles: list,
    seed_triangle_positions: list,
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int,
    min_membrane_thickness: float
) -> list:
    """Python fallback - calls the original implementation."""
    # Import here to avoid circular imports
    from .tetrahedral_mesh import _find_secondary_cuts_by_triangle_intersection
    return _find_secondary_cuts_by_triangle_intersection(
        edge_membrane_data, seed_triangles, seed_triangle_positions,
        vertices, boundary_verts, min_intersection_count, min_membrane_thickness,
        use_gpu=True  # Let Python implementation decide GPU vs CPU
    )


# ============================================================================
# MOLD HALF CLASSIFICATION
# ============================================================================

def classify_mold_halves_fast(
    boundary_mesh,  # trimesh.Trimesh
    outer_triangles: set,
    d1: np.ndarray,
    d2: np.ndarray,
    adjacency: dict = None
) -> dict:
    """
    Fast mold half classification using C++ with OpenMP if available.
    
    Args:
        boundary_mesh: The boundary mesh
        outer_triangles: Set of outer boundary triangle indices to classify
        d1: First parting direction (normalized)
        d2: Second parting direction (normalized)
        adjacency: Pre-computed adjacency map (optional)
    
    Returns:
        Dict mapping triangle index to mold half (1 or 2)
    """
    if CPP_AVAILABLE and len(outer_triangles) > 1000:
        return _classify_mold_halves_cpp(boundary_mesh, outer_triangles, d1, d2)
    else:
        # Use Python implementation for small meshes or if C++ unavailable
        from .mold_half_classification import classify_mold_halves_fast as _python_impl
        return _python_impl(boundary_mesh, outer_triangles, d1, d2, adjacency, use_gpu=True)


def _classify_mold_halves_cpp(
    boundary_mesh,
    outer_triangles: set,
    d1: np.ndarray,
    d2: np.ndarray
) -> dict:
    """C++ implementation wrapper for mold half classification."""
    start_time = time.time()
    
    n_faces = len(boundary_mesh.faces)
    n_outer = len(outer_triangles)
    
    if n_outer == 0:
        return {}
    
    # Get face normals
    face_normals = np.ascontiguousarray(boundary_mesh.face_normals, dtype=np.float64)
    
    # Convert outer_triangles set to array
    outer_indices = np.array(list(outer_triangles), dtype=np.int64)
    
    # Normalize directions
    d1 = d1.astype(np.float64) / np.linalg.norm(d1)
    d2 = d2.astype(np.float64) / np.linalg.norm(d2)
    
    # Build CSR adjacency from trimesh face_adjacency
    # trimesh.face_adjacency is (n, 2) pairs of adjacent faces
    adj_lists = [[] for _ in range(n_faces)]
    for i in range(len(boundary_mesh.face_adjacency)):
        a, b = boundary_mesh.face_adjacency[i]
        adj_lists[a].append(b)
        adj_lists[b].append(a)
    
    # Convert to CSR format
    adj_indices_list = []
    adj_ptr = [0]
    for fi in range(n_faces):
        adj_indices_list.extend(adj_lists[fi])
        adj_ptr.append(len(adj_indices_list))
    
    adj_indices = np.array(adj_indices_list, dtype=np.int64)
    adj_ptr = np.array(adj_ptr, dtype=np.int64)
    
    # Call C++ implementation
    result = _cpp.classify_mold_halves(
        face_normals,
        outer_indices,
        d1,
        d2,
        adj_indices,
        adj_ptr,
        n_faces
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


def is_cpp_available() -> bool:
    """Check if C++ module is available."""
    return CPP_AVAILABLE


def get_implementation_info() -> dict:
    """Get information about available implementations."""
    return {
        'cpp_available': CPP_AVAILABLE,
        'dijkstra': 'C++' if CPP_AVAILABLE else 'Python',
        'edge_labels': 'C++' if CPP_AVAILABLE else 'Python',
        'secondary_cuts': 'C++ (OpenMP)' if CPP_AVAILABLE else 'Python/GPU',
        'mold_half_classification': 'C++ (OpenMP)' if CPP_AVAILABLE else 'Python/GPU',
        'laplacian_smoothing': 'C++ (OpenMP)' if CPP_AVAILABLE else 'Python',
        'speedup_estimate': '10-50x' if CPP_AVAILABLE else 'N/A (using Python)',
    }


# =============================================================================
# FAST LAPLACIAN SMOOTHING
# =============================================================================

def run_fast_laplacian_smoothing(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertices: np.ndarray,
    excluded_vertices: Optional[np.ndarray] = None,
    iterations: int = 5,
    lambda_factor: float = 0.5
) -> Tuple[np.ndarray, float]:
    """
    Run fast Laplacian smoothing using C++ with OpenMP if available.
    
    This is useful for simple smoothing where all boundary/excluded vertices
    are fixed throughout the operation.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face vertex indices
        boundary_vertices: (B,) indices of boundary vertices to preserve
        excluded_vertices: (E,) additional vertex indices to exclude from smoothing
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing factor (0.0-1.0)
    
    Returns:
        Tuple of (smoothed_vertices, elapsed_ms)
    """
    if excluded_vertices is None:
        excluded_vertices = np.array([], dtype=np.int64)
    
    if CPP_AVAILABLE:
        return _run_laplacian_smoothing_cpp(
            vertices, faces, boundary_vertices, excluded_vertices,
            iterations, lambda_factor
        )
    else:
        return _run_laplacian_smoothing_python(
            vertices, faces, boundary_vertices, excluded_vertices,
            iterations, lambda_factor
        )


def _run_laplacian_smoothing_cpp(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertices: np.ndarray,
    excluded_vertices: np.ndarray,
    iterations: int,
    lambda_factor: float
) -> Tuple[np.ndarray, float]:
    """C++ implementation of Laplacian smoothing."""
    verts = np.ascontiguousarray(vertices, dtype=np.float64)
    fcs = np.ascontiguousarray(faces, dtype=np.int64)
    boundary = np.ascontiguousarray(boundary_vertices, dtype=np.int64)
    excluded = np.ascontiguousarray(excluded_vertices, dtype=np.int64)
    
    result = _cpp.fast_laplacian_smoothing(
        verts, fcs, boundary, excluded, iterations, lambda_factor
    )
    
    logger.info(f"[C++] Laplacian smoothing: {result.iterations_performed} iterations "
                f"in {result.elapsed_ms:.1f}ms")
    
    return np.asarray(result.vertices), result.elapsed_ms


def _run_laplacian_smoothing_python(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertices: np.ndarray,
    excluded_vertices: np.ndarray,
    iterations: int,
    lambda_factor: float
) -> Tuple[np.ndarray, float]:
    """Python fallback implementation of Laplacian smoothing."""
    start_time = time.time()
    
    n_verts = len(vertices)
    verts = vertices.copy()
    
    # Build set of fixed vertices
    fixed_set = set(boundary_vertices.tolist())
    fixed_set.update(excluded_vertices.tolist())
    
    # Build vertex adjacency
    neighbors = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            v = f[i]
            neighbors[v].add(f[(i+1) % 3])
            neighbors[v].add(f[(i+2) % 3])
    
    # Smoothing iterations
    for _ in range(iterations):
        new_verts = verts.copy()
        
        for v in range(n_verts):
            if v in fixed_set:
                continue
            
            if neighbors[v]:
                neighbor_list = list(neighbors[v])
                avg_pos = verts[neighbor_list].mean(axis=0)
                new_verts[v] = verts[v] + lambda_factor * (avg_pos - verts[v])
        
        verts = new_verts
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"[Python] Laplacian smoothing: {iterations} iterations in {elapsed_ms:.1f}ms")
    
    return verts, elapsed_ms


def run_fast_boundary_interior_smoothing(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertex_indices: np.ndarray,
    boundary_edges: list,  # List of (v0, v1) tuples
    excluded_vertices: Optional[np.ndarray] = None,
    lambda_factor: float = 0.5
) -> np.ndarray:
    """
    Run one iteration of boundary + interior smoothing using C++ if available.
    
    This performs:
    1. Smooth boundary vertices along boundary polylines
    2. Smooth interior vertices (boundary fixed)
    
    Designed to be called in a loop with re-projection between iterations.
    
    Args:
        vertices: (N, 3) current vertex positions
        faces: (F, 3) face vertex indices
        boundary_vertex_indices: (B,) indices of boundary vertices
        boundary_edges: List of (v0, v1) tuples representing boundary edges
        excluded_vertices: (E,) vertex indices to completely exclude
        lambda_factor: Smoothing factor
    
    Returns:
        Smoothed vertex positions (N, 3)
    """
    if excluded_vertices is None:
        excluded_vertices = np.array([], dtype=np.int64)
    
    if CPP_AVAILABLE:
        return _run_boundary_interior_smoothing_cpp(
            vertices, faces, boundary_vertex_indices, boundary_edges,
            excluded_vertices, lambda_factor
        )
    else:
        return _run_boundary_interior_smoothing_python(
            vertices, faces, boundary_vertex_indices, boundary_edges,
            excluded_vertices, lambda_factor
        )


def _run_boundary_interior_smoothing_cpp(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertex_indices: np.ndarray,
    boundary_edges: list,
    excluded_vertices: np.ndarray,
    lambda_factor: float
) -> np.ndarray:
    """C++ implementation of boundary + interior smoothing."""
    # Build boundary neighbor data in CSR-like format
    boundary_set = set(boundary_vertex_indices.tolist())
    
    # Build boundary adjacency
    boundary_adj = {v: [] for v in boundary_vertex_indices}
    for v0, v1 in boundary_edges:
        if v0 in boundary_adj:
            boundary_adj[v0].append(v1)
        if v1 in boundary_adj:
            boundary_adj[v1].append(v0)
    
    # Convert to flat arrays with offsets
    boundary_neighbors_list = []
    boundary_offsets = [0]
    
    for vi in boundary_vertex_indices:
        neighbors = boundary_adj.get(vi, [])
        boundary_neighbors_list.extend(neighbors)
        boundary_offsets.append(len(boundary_neighbors_list))
    
    verts = np.ascontiguousarray(vertices, dtype=np.float64)
    fcs = np.ascontiguousarray(faces, dtype=np.int64)
    boundary_idx = np.ascontiguousarray(boundary_vertex_indices, dtype=np.int64)
    boundary_nbrs = np.array(boundary_neighbors_list, dtype=np.int64)
    boundary_offs = np.array(boundary_offsets, dtype=np.int64)
    excluded = np.ascontiguousarray(excluded_vertices, dtype=np.int64)
    
    result = _cpp.fast_boundary_and_interior_smoothing(
        verts, fcs, boundary_idx, boundary_nbrs, boundary_offs,
        excluded, lambda_factor
    )
    
    return np.asarray(result)


def _run_boundary_interior_smoothing_python(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertex_indices: np.ndarray,
    boundary_edges: list,
    excluded_vertices: np.ndarray,
    lambda_factor: float
) -> np.ndarray:
    """Python fallback for boundary + interior smoothing."""
    n_verts = len(vertices)
    verts = vertices.copy()
    
    boundary_set = set(boundary_vertex_indices.tolist())
    excluded_set = set(excluded_vertices.tolist())
    
    # Build boundary adjacency
    boundary_adj = {v: [] for v in boundary_vertex_indices}
    for v0, v1 in boundary_edges:
        if v0 in boundary_adj:
            boundary_adj[v0].append(v1)
        if v1 in boundary_adj:
            boundary_adj[v1].append(v0)
    
    # Build full vertex adjacency
    neighbors = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            v = f[i]
            neighbors[v].add(f[(i+1) % 3])
            neighbors[v].add(f[(i+2) % 3])
    
    # Step 1: Smooth boundary vertices
    new_verts = verts.copy()
    for vi in boundary_vertex_indices:
        if vi in excluded_set:
            continue
        nbrs = boundary_adj.get(vi, [])
        if len(nbrs) >= 2:
            avg_pos = verts[nbrs].mean(axis=0)
            new_verts[vi] = verts[vi] + lambda_factor * (avg_pos - verts[vi])
    
    verts = new_verts
    
    # Step 2: Smooth interior vertices
    new_verts = verts.copy()
    for vi in range(n_verts):
        if vi in boundary_set or vi in excluded_set:
            continue
        if neighbors[vi]:
            neighbor_list = list(neighbors[vi])
            avg_pos = verts[neighbor_list].mean(axis=0)
            new_verts[vi] = verts[vi] + lambda_factor * (avg_pos - verts[vi])
    
    return new_verts

