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


def is_cpp_available() -> bool:
    """Check if C++ module is available."""
    return CPP_AVAILABLE


def get_implementation_info() -> dict:
    """Get information about available implementations."""
    return {
        'cpp_available': CPP_AVAILABLE,
        'dijkstra': 'C++' if CPP_AVAILABLE else 'Python',
        'edge_labels': 'C++' if CPP_AVAILABLE else 'Python',
        'speedup_estimate': '10-20x' if CPP_AVAILABLE else 'N/A (using Python)',
    }
