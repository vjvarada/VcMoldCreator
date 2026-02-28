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

