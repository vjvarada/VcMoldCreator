"""
Tetrahedral Mesh Generation for Mold Volume

This module generates a tetrahedral mesh of the mold cavity volume using fTetWild
(via pytetwild). The tetrahedral mesh replaces the voxel grid approach for:
- More accurate volume representation
- Better boundary conforming
- Edge-based weight assignment for parting surface computation

Algorithm:
1. Take the cavity mesh (Hull - Part CSG result)
2. Tetrahedralize the interior volume using fTetWild
3. Build edge connectivity for weight assignment
4. Compute distances from tet vertices/edges to part surface and shell boundary

The edge weights are used for Dijkstra-based escape labeling to find the parting surface.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import trimesh

try:
    import pytetwild
    PYTETWILD_AVAILABLE = True
except ImportError:
    PYTETWILD_AVAILABLE = False

# Check for GPU acceleration options
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Check for C++ fast algorithms
try:
    from . import fast_algorithms as _cpp_fast
    CPP_FAST_AVAILABLE = True
except ImportError:
    CPP_FAST_AVAILABLE = False

logger = logging.getLogger(__name__)

if CPP_FAST_AVAILABLE:
    logger.info("C++ fast_algorithms available - using optimized Dijkstra and edge labeling")
else:
    logger.info("C++ fast_algorithms not compiled - using Python implementations. "
                "For 10-20x speedup: cd desktop_app/core && python setup_cpp.py build_ext --inplace")


@dataclass
class TetrahedralMeshResult:
    """Result of tetrahedral mesh generation."""
    
    # Vertex positions (N x 3) - these are the CURRENT (possibly inflated) positions
    vertices: np.ndarray
    
    # Tetrahedra indices (M x 4) - each row is 4 vertex indices
    tetrahedra: np.ndarray
    
    # Edge list (E x 2) - unique edges extracted from tetrahedra
    edges: np.ndarray
    
    # Edge lengths (E,) - computed on CURRENT vertex positions
    edge_lengths: np.ndarray
    
    # Boundary surface mesh (extracted from tetrahedral mesh outer faces)
    # Uses CURRENT (possibly inflated) vertex positions
    boundary_mesh: Optional[trimesh.Trimesh] = None
    
    # ORIGINAL (non-inflated) vertex positions (N x 3)
    # Used for parting surface construction
    vertices_original: Optional[np.ndarray] = None
    
    # ORIGINAL boundary surface mesh (non-inflated)
    # Used for parting surface construction
    boundary_mesh_original: Optional[trimesh.Trimesh] = None
    
    # Edge midpoint distances to part mesh (E,) - for weight computation
    edge_dist_to_part: Optional[np.ndarray] = None
    
    # Edge weights (E,) - weight = 1 / (dist^2 + epsilon)
    edge_weights: Optional[np.ndarray] = None
    
    # Weighted edge lengths (E,) = edge_length * edge_weight
    weighted_edge_lengths: Optional[np.ndarray] = None
    
    # Boundary vertex mask - True if vertex is on boundary surface (N,)
    boundary_vertices: Optional[np.ndarray] = None
    
    # Shell boundary vertex labels: 0=interior, 1=H1, 2=H2, -1=inner boundary (N,)
    boundary_labels: Optional[np.ndarray] = None
    
    # Edge boundary labels: 0=interior, 1=H1, 2=H2, -1=inner boundary, -2=boundary zone (E,)
    edge_boundary_labels: Optional[np.ndarray] = None
    
    # R value - maximum distance from hull vertices to part surface
    r_value: Optional[float] = None
    r_hull_point: Optional[np.ndarray] = None  # Point on hull with max distance
    r_part_point: Optional[np.ndarray] = None  # Closest point on part surface
    
    # Whether the mesh has been inflated
    is_inflated: bool = False
    
    # Statistics
    num_vertices: int = 0
    num_tetrahedra: int = 0
    num_edges: int = 0
    num_boundary_faces: int = 0
    
    # Dijkstra results - interior vertex escape labels (includes both part surface and cavity interior)
    # seed_escape_labels: (I,) int8 - 1=escapes to H1, 2=escapes to H2, 0=unreachable
    seed_escape_labels: Optional[np.ndarray] = None
    
    # Interior vertex indices in the tet mesh (I,) - indices of all interior vertices (not on H1/H2)
    seed_vertex_indices: Optional[np.ndarray] = None
    
    # Interior vertex distances to boundary (I,) - shortest path distance to H1/H2
    seed_distances: Optional[np.ndarray] = None
    
    # Escape destination vertices for each interior vertex (I,) - the boundary vertex reached
    seed_escape_destinations: Optional[np.ndarray] = None
    
    # Escape paths for each interior vertex - list of vertex index lists
    # seed_escape_paths[i] = [start_v, v1, v2, ..., dest_v] path from interior vertex to boundary
    seed_escape_paths: Optional[List[List[int]]] = None
    
    # Secondary cutting edges - edges between same-label seeds where membrane intersects part
    # List of (vi, vj) tuples representing secondary cut edges
    secondary_cut_edges: Optional[List[Tuple[int, int]]] = None
    
    # Timing
    tetrahedralize_time_ms: float = 0.0
    total_time_ms: float = 0.0


def tetrahedralize_mesh(
    cavity_mesh: trimesh.Trimesh,
    edge_length_fac: float = 0.05,
    optimize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tetrahedral mesh from a surface mesh using fTetWild.
    
    Args:
        cavity_mesh: The cavity surface mesh (should be watertight)
        edge_length_fac: Target edge length as fraction of bounding box diagonal
                        Default 0.05 = bbox/20. Smaller = finer mesh.
        optimize: Whether to optimize mesh quality (slower but better quality)
    
    Returns:
        Tuple of (vertices, tetrahedra)
        - vertices: (N, 3) float64 array of vertex positions
        - tetrahedra: (M, 4) int32 array of tetrahedron vertex indices
    """
    if not PYTETWILD_AVAILABLE:
        raise ImportError("pytetwild is not installed. Install with: pip install pytetwild")
    
    # Get vertices and faces from trimesh
    vertices = np.asarray(cavity_mesh.vertices, dtype=np.float64)
    faces = np.asarray(cavity_mesh.faces, dtype=np.int32)
    
    logger.info(f"Tetrahedralizing mesh with {len(vertices)} vertices, {len(faces)} faces")
    logger.info(f"Parameters: edge_length_fac={edge_length_fac}, optimize={optimize}")
    
    # Run fTetWild
    tet_vertices, tetrahedra = pytetwild.tetrahedralize(
        vertices, 
        faces,
        optimize=optimize,
        edge_length_fac=edge_length_fac
    )
    
    logger.info(f"Generated {len(tet_vertices)} vertices, {len(tetrahedra)} tetrahedra")
    
    return tet_vertices, tetrahedra


def filter_tetrahedra_outside_part(
    tet_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    part_mesh: trimesh.Trimesh,
    tolerance: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out tetrahedra that are inside the part mesh.
    
    This removes tetrahedra whose centroids are inside the part mesh,
    which can happen due to imperfect CSG operations or mesh issues.
    
    Args:
        tet_vertices: (N, 3) tetrahedral mesh vertices
        tetrahedra: (M, 4) tetrahedron vertex indices
        part_mesh: The original part mesh to check against
        tolerance: Small positive value - centroids within this distance
                   inside the part are also removed (default 0.0)
    
    Returns:
        Tuple of (filtered_vertices, filtered_tetrahedra)
        Note: vertices are reindexed to remove unused ones
    """
    logger.info(f"Filtering tetrahedra inside part mesh (tolerance={tolerance})...")
    
    # Compute tetrahedra centroids
    centroids = tet_vertices[tetrahedra].mean(axis=1)  # (M, 3)
    
    # Check which centroids are inside the part mesh
    # trimesh.contains returns True for points inside the mesh
    inside_part = part_mesh.contains(centroids)
    
    # If tolerance > 0, also check distance to surface for points just outside
    if tolerance > 0:
        # For points just outside, check if they're very close to the surface
        _, distances, _ = part_mesh.nearest.on_surface(centroids)
        # Points very close to the surface (but outside) are kept
        # Only truly inside points are removed
        inside_part = inside_part & (distances > tolerance)
    
    # Keep tetrahedra whose centroids are OUTSIDE the part
    outside_mask = ~inside_part
    n_inside = np.sum(inside_part)
    n_outside = np.sum(outside_mask)
    
    logger.info(f"Tetrahedra inside part: {n_inside}, outside part: {n_outside}")
    
    if n_inside == 0:
        logger.info("No tetrahedra inside part mesh - no filtering needed")
        return tet_vertices, tetrahedra
    
    # Filter tetrahedra
    filtered_tetrahedra = tetrahedra[outside_mask]
    
    # Reindex vertices to remove unused ones
    used_vertices = np.unique(filtered_tetrahedra.ravel())
    vertex_remap = np.full(len(tet_vertices), -1, dtype=np.int32)
    vertex_remap[used_vertices] = np.arange(len(used_vertices))
    
    filtered_vertices = tet_vertices[used_vertices]
    filtered_tetrahedra = vertex_remap[filtered_tetrahedra]
    
    logger.info(f"After filtering: {len(filtered_vertices)} vertices, {len(filtered_tetrahedra)} tetrahedra")
    
    return filtered_vertices, filtered_tetrahedra


def extract_boundary_surface(
    vertices: np.ndarray,
    tetrahedra: np.ndarray
) -> trimesh.Trimesh:
    """
    Extract the boundary surface from a tetrahedral mesh.
    
    The boundary surface consists of triangular faces that belong to only
    one tetrahedron (i.e., they are on the outer surface).
    
    Each tetrahedron has 4 faces:
    - Face 0: vertices (1, 2, 3) - opposite to vertex 0
    - Face 1: vertices (0, 2, 3) - opposite to vertex 1  
    - Face 2: vertices (0, 1, 3) - opposite to vertex 2
    - Face 3: vertices (0, 1, 2) - opposite to vertex 3
    
    Args:
        vertices: (N, 3) vertex positions
        tetrahedra: (M, 4) tetrahedron vertex indices
    
    Returns:
        trimesh.Trimesh of the boundary surface
    """
    # Face vertex indices for each tetrahedron (ordered for consistent normals)
    # Using right-hand rule: normals point outward
    face_local_indices = [
        (1, 3, 2),  # Face opposite vertex 0
        (0, 2, 3),  # Face opposite vertex 1
        (0, 3, 1),  # Face opposite vertex 2
        (0, 1, 2),  # Face opposite vertex 3
    ]
    
    # Extract all faces from all tetrahedra
    all_faces = []
    for i, j, k in face_local_indices:
        faces = tetrahedra[:, [i, j, k]]
        all_faces.append(faces)
    
    all_faces = np.vstack(all_faces)  # (4*M, 3)
    
    # Sort vertices within each face for comparison (to find duplicates)
    sorted_faces = np.sort(all_faces, axis=1)
    
    # Find unique faces and their counts
    # Boundary faces appear exactly once, interior faces appear twice
    unique_faces, inverse, counts = np.unique(
        sorted_faces, axis=0, return_inverse=True, return_counts=True
    )
    
    # Get boundary face indices (faces that appear only once)
    boundary_mask = counts[inverse] == 1
    boundary_faces = all_faces[boundary_mask]
    
    logger.info(f"Extracted {len(boundary_faces)} boundary faces from {len(tetrahedra)} tetrahedra")
    
    # Create trimesh with only the vertices used by boundary faces
    # This removes unused vertices and reindexes faces
    unique_vertex_indices = np.unique(boundary_faces.ravel())
    vertex_remap = np.full(len(vertices), -1, dtype=np.int32)
    vertex_remap[unique_vertex_indices] = np.arange(len(unique_vertex_indices))
    
    boundary_vertices = vertices[unique_vertex_indices]
    remapped_faces = vertex_remap[boundary_faces]
    
    # Create trimesh
    boundary_mesh = trimesh.Trimesh(vertices=boundary_vertices, faces=remapped_faces)
    
    # Merge any duplicate vertices (within tolerance)
    boundary_mesh.merge_vertices()
    
    # Remove any degenerate faces (zero area)
    boundary_mesh.remove_degenerate_faces()
    
    # Remove any duplicate faces
    boundary_mesh.remove_duplicate_faces()
    
    # Fix normals to point outward (consistent winding)
    boundary_mesh.fix_normals()
    
    logger.info(f"Boundary mesh cleaned: {len(boundary_mesh.vertices)} vertices, {len(boundary_mesh.faces)} faces")
    
    return boundary_mesh


def inflate_boundary_vertices(
    tet_result: 'TetrahedralMeshResult',
    part_mesh: trimesh.Trimesh,
    r_value: float,
    use_gpu: bool = True,
    inner_boundary_threshold: float = None
) -> 'TetrahedralMeshResult':
    """
    Inflate OUTER boundary vertices of the tetrahedral mesh outward.
    
    Only outer boundary vertices (hull surface) are inflated.
    Inner boundary vertices (part surface) are kept fixed to preserve
    the part cavity shape.
    
    For each OUTER boundary vertex, move it outward by:
        displacement = R - distance_to_part
    
    This creates a more uniform thickness in the mold volume.
    Vertices close to the part move more (up to R), vertices far from
    the part move less.
    
    Args:
        tet_result: TetrahedralMeshResult to modify
        part_mesh: Original part mesh (for distance computation)
        r_value: Maximum distance R (computed earlier)
        use_gpu: Whether to use GPU acceleration
        inner_boundary_threshold: Distance threshold to identify inner boundary
                                  vertices. If None, auto-computed as 2% of mesh size.
    
    Returns:
        Modified TetrahedralMeshResult with inflated outer boundary vertices
    """
    import time
    start_time = time.time()
    
    boundary_mesh = tet_result.boundary_mesh
    if boundary_mesh is None:
        logger.warning("No boundary mesh available for inflation")
        return tet_result
    
    # Get boundary vertices
    boundary_verts = np.asarray(boundary_mesh.vertices, dtype=np.float64)
    n_boundary = len(boundary_verts)
    
    # Auto-compute threshold if not provided (same as visualization uses)
    if inner_boundary_threshold is None:
        bounds = part_mesh.bounds
        mesh_size = np.linalg.norm(bounds[1] - bounds[0])
        inner_boundary_threshold = mesh_size * 0.02  # 2% of mesh size
    
    logger.info(f"Processing {n_boundary} boundary vertices with R={r_value:.4f}, inner_threshold={inner_boundary_threshold:.4f}")
    
    # Compute distance from each boundary vertex to part surface
    if use_gpu and CUDA_AVAILABLE:
        distances, closest_points = compute_distances_and_closest_points_gpu(
            boundary_verts, part_mesh
        )
    else:
        closest_points, distances, _ = part_mesh.nearest.on_surface(boundary_verts)
    
    # Identify inner boundary vertices (those very close to part surface)
    # These should NOT be moved - they define the part cavity
    inner_boundary_mask = distances < inner_boundary_threshold
    outer_boundary_mask = ~inner_boundary_mask
    
    n_inner = np.sum(inner_boundary_mask)
    n_outer = np.sum(outer_boundary_mask)
    logger.info(f"Inner boundary (fixed): {n_inner} vertices, Outer boundary (to inflate): {n_outer} vertices")
    
    # Compute displacement for outer boundary vertices only: R - distance
    displacements = np.zeros(n_boundary, dtype=np.float64)
    displacements[outer_boundary_mask] = r_value - distances[outer_boundary_mask]
    
    # Clamp displacements to be non-negative (don't move inward)
    displacements = np.maximum(displacements, 0.0)
    
    logger.info(f"Outer boundary displacement range: [{displacements[outer_boundary_mask].min():.4f}, {displacements[outer_boundary_mask].max():.4f}]")
    
    # Compute outward direction for each vertex (vertex normal)
    # Use the boundary mesh vertex normals
    vertex_normals = boundary_mesh.vertex_normals
    if vertex_normals is None or len(vertex_normals) != n_boundary:
        # Fallback: compute direction from closest point on part to vertex
        directions = boundary_verts - closest_points
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        vertex_normals = directions / norms
    
    # Ensure normals are unit vectors
    normal_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    normal_lengths = np.where(normal_lengths > 1e-10, normal_lengths, 1.0)
    vertex_normals = vertex_normals / normal_lengths
    
    # Move boundary vertices outward (only outer boundary will move due to displacement=0 for inner)
    inflated_boundary_verts = boundary_verts + vertex_normals * displacements[:, np.newaxis]
    
    # Update the boundary mesh with new vertex positions
    inflated_boundary_mesh = trimesh.Trimesh(
        vertices=inflated_boundary_verts,
        faces=boundary_mesh.faces.copy(),
        process=False
    )
    inflated_boundary_mesh.fix_normals()
    
    # Create mapping from boundary mesh vertices to tetrahedral mesh vertices
    # The tetrahedral vertices include interior vertices, but we only modify boundary ones
    # We need to identify which tet vertices correspond to boundary mesh vertices
    
    # Copy the original tet vertices
    new_tet_verts = tet_result.vertices.copy()
    
    # For each boundary mesh vertex, find the corresponding tet vertex and update it
    # Use nearest neighbor matching since boundary_mesh may have merged vertices
    from scipy.spatial import cKDTree
    tet_tree = cKDTree(tet_result.vertices)
    
    # For each original boundary vertex, find nearest tet vertex
    orig_boundary_verts = np.asarray(boundary_mesh.vertices, dtype=np.float64)
    _, tet_indices = tet_tree.query(orig_boundary_verts, k=1)
    
    # Update those tet vertices with inflated positions
    new_tet_verts[tet_indices] = inflated_boundary_verts
    
    # Recompute edge lengths with new vertices
    edges = tet_result.edges
    v0 = new_tet_verts[edges[:, 0]]
    v1 = new_tet_verts[edges[:, 1]]
    new_edge_lengths = np.linalg.norm(v1 - v0, axis=1)
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Boundary inflation complete in {elapsed:.0f}ms")
    logger.info(f"  Updated {len(tet_indices)} tet vertices")
    
    # Store original vertices and boundary mesh (for parting surface construction later)
    # If already inflated, keep the original from previous result
    original_verts = tet_result.vertices_original if tet_result.vertices_original is not None else tet_result.vertices.copy()
    original_boundary = tet_result.boundary_mesh_original if tet_result.boundary_mesh_original is not None else boundary_mesh
    
    # Create new result with updated data
    inflated_result = TetrahedralMeshResult(
        vertices=new_tet_verts,
        tetrahedra=tet_result.tetrahedra,
        edges=tet_result.edges,
        edge_lengths=new_edge_lengths,
        boundary_mesh=inflated_boundary_mesh,
        vertices_original=original_verts,
        boundary_mesh_original=original_boundary,
        edge_dist_to_part=None,  # Needs recomputation
        edge_weights=None,  # Needs recomputation
        weighted_edge_lengths=None,
        boundary_vertices=tet_result.boundary_vertices,
        boundary_labels=tet_result.boundary_labels,
        edge_boundary_labels=None,  # Will be computed with edge weights
        r_value=r_value,
        r_hull_point=tet_result.r_hull_point,
        r_part_point=tet_result.r_part_point,
        is_inflated=True,
        num_vertices=len(new_tet_verts),
        num_tetrahedra=tet_result.num_tetrahedra,
        num_edges=tet_result.num_edges,
        num_boundary_faces=len(inflated_boundary_mesh.faces),
        tetrahedralize_time_ms=tet_result.tetrahedralize_time_ms,
        total_time_ms=tet_result.total_time_ms + elapsed
    )
    
    return inflated_result


def extract_edges(tetrahedra: np.ndarray) -> np.ndarray:
    """
    Extract unique edges from tetrahedra.
    
    Each tetrahedron has 6 edges. We extract all edges and remove duplicates.
    
    Args:
        tetrahedra: (M, 4) array of tetrahedron vertex indices
    
    Returns:
        (E, 2) array of unique edge vertex indices (sorted so v0 < v1)
    """
    # Each tetrahedron has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edge_pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]
    
    # Extract all edges
    all_edges = []
    for i, j in edge_pairs:
        edges = np.column_stack([tetrahedra[:, i], tetrahedra[:, j]])
        all_edges.append(edges)
    
    all_edges = np.vstack(all_edges)
    
    # Sort each edge so smaller index comes first
    all_edges = np.sort(all_edges, axis=1)
    
    # Remove duplicates
    unique_edges = np.unique(all_edges, axis=0)
    
    return unique_edges


def compute_edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Compute the length of each edge.
    
    Args:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge vertex indices
    
    Returns:
        (E,) array of edge lengths
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    return lengths


def compute_edge_weights_simple(
    tet_result: 'TetrahedralMeshResult',
    part_mesh: trimesh.Trimesh,
    epsilon: float = 0.25,
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute edge weights based on distance to part surface.
    
    Simple formula (no boundary bias since inflation already handles that):
        weight = 1 / (distance^2 + epsilon)
    
    Higher weights mean lower traversal cost (edges near the part are "cheaper").
    
    Args:
        tet_result: Tetrahedral mesh result (uses inflated vertex positions)
        part_mesh: Original part mesh for distance computation
        epsilon: Small constant to avoid division by zero (default 0.25)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Tuple of:
            - edge_weights: (E,) array of edge weights
            - edge_dist_to_part: (E,) array of edge midpoint distances to part
    """
    import time
    start_time = time.time()
    
    vertices = tet_result.vertices
    edges = tet_result.edges
    n_edges = len(edges)
    
    # Compute edge midpoints
    edge_midpoints = (vertices[edges[:, 0]] + vertices[edges[:, 1]]) / 2
    
    logger.info(f"Computing edge weights for {n_edges} edges...")
    
    # Compute distances from edge midpoints to part surface
    if use_gpu and CUDA_AVAILABLE:
        logger.debug("Using GPU for edge midpoint distance computation")
        edge_dist_to_part, _ = compute_distances_and_closest_points_gpu(
            edge_midpoints, part_mesh
        )
    else:
        logger.debug("Using CPU for edge midpoint distance computation")
        _, edge_dist_to_part, _ = part_mesh.nearest.on_surface(edge_midpoints)
    
    # Compute weights: weight = 1 / (distance^2 + epsilon)
    # Higher weight = lower cost for edges near the part surface
    edge_weights = 1.0 / (edge_dist_to_part ** 2 + epsilon)
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Edge weights computed in {elapsed:.0f}ms")
    logger.info(f"  Distance range: [{edge_dist_to_part.min():.4f}, {edge_dist_to_part.max():.4f}]")
    logger.info(f"  Weight range: [{edge_weights.min():.4f}, {edge_weights.max():.4f}]")
    
    return edge_weights, edge_dist_to_part


def compute_edge_boundary_labels(
    tet_result: 'TetrahedralMeshResult'
) -> np.ndarray:
    """
    Compute labels for each edge based on vertex boundary labels.
    
    Edge labels:
        0  = Interior edge (not on boundary surface, or connects interior to boundary)
        1  = H1 boundary edge (on boundary surface, both vertices on H1)
        2  = H2 boundary edge (on boundary surface, both vertices on H2)
        -1 = Inner boundary edge (on boundary surface, both vertices on inner/part boundary)
        -2 = Mixed boundary edge (on boundary surface, both on boundary but different types)
    
    IMPORTANT: Only edges that are actually ON THE BOUNDARY MESH SURFACE are labeled
    as boundary edges. Tetrahedral edges that connect boundary vertices but pass through
    the interior (through concave regions) are labeled as interior edges (0).
    
    Uses C++ implementation when available for faster performance.
    
    Args:
        tet_result: Tetrahedral mesh result with boundary_labels computed
    
    Returns:
        (E,) int8 array of edge boundary labels
    """
    # Use C++ implementation if available
    if CPP_FAST_AVAILABLE:
        return _compute_edge_boundary_labels_cpp(tet_result)
    else:
        return _compute_edge_boundary_labels_python(tet_result)


def _compute_edge_boundary_labels_cpp(
    tet_result: 'TetrahedralMeshResult'
) -> np.ndarray:
    """C++ implementation of edge boundary label computation."""
    import time
    start_time = time.time()
    
    edges = tet_result.edges
    n_edges = len(edges)
    
    boundary_labels = tet_result.boundary_labels
    if boundary_labels is None:
        logger.warning("No boundary labels available for edge labeling")
        return np.zeros(n_edges, dtype=np.int8)
    
    boundary_mesh = tet_result.boundary_mesh
    if boundary_mesh is None:
        logger.warning("No boundary_mesh available for edge labeling")
        return np.zeros(n_edges, dtype=np.int8)
    
    # Prepare arrays for C++ call
    edges_arr = np.asarray(edges, dtype=np.int64)
    boundary_labels_arr = np.asarray(boundary_labels, dtype=np.int8)
    boundary_faces = np.asarray(boundary_mesh.faces, dtype=np.int64)
    boundary_mesh_vertices = np.asarray(boundary_mesh.vertices, dtype=np.float64)
    tet_vertices = np.asarray(tet_result.vertices, dtype=np.float64)
    
    # Call C++ implementation
    edge_labels = _cpp_fast.compute_edge_boundary_labels(
        edges_arr, boundary_labels_arr, boundary_faces, boundary_mesh_vertices, tet_vertices
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    # Count stats
    n_interior = np.sum(edge_labels == 0)
    n_h1 = np.sum(edge_labels == 1)
    n_h2 = np.sum(edge_labels == 2)
    n_inner = np.sum(edge_labels == -1)
    n_mixed = np.sum(edge_labels == -2)
    
    logger.info(
        f"[C++] Edge boundary labels computed in {elapsed:.0f}ms: "
        f"interior={n_interior}, H1={n_h1}, H2={n_h2}, inner={n_inner}, mixed={n_mixed}"
    )
    
    return np.asarray(edge_labels)


def _compute_edge_boundary_labels_python(
    tet_result: 'TetrahedralMeshResult'
) -> np.ndarray:
    """Python fallback implementation of edge boundary label computation."""
    edges = tet_result.edges
    n_edges = len(edges)
    
    edge_labels = np.zeros(n_edges, dtype=np.int8)
    
    boundary_labels = tet_result.boundary_labels
    if boundary_labels is None:
        logger.warning("No boundary labels available for edge labeling")
        return edge_labels
    
    boundary_verts = tet_result.boundary_vertices
    if boundary_verts is None:
        logger.warning("No boundary_vertices mask available for edge labeling")
        return edge_labels
    
    boundary_mesh = tet_result.boundary_mesh
    if boundary_mesh is None:
        logger.warning("No boundary_mesh available for edge labeling")
        return edge_labels
    
    # Build set of edges that are ON THE BOUNDARY MESH SURFACE
    # These are edges of boundary triangles, not just edges between boundary vertices
    boundary_faces = np.asarray(boundary_mesh.faces)
    
    # First, we need to map boundary mesh vertex indices to tet mesh vertex indices
    # The boundary mesh may have different vertex indexing than the tet mesh
    boundary_mesh_verts = np.asarray(boundary_mesh.vertices)
    tet_verts = tet_result.vertices
    
    # Build mapping from boundary mesh vertices to tet vertices
    # Use spatial matching since indices may not correspond directly
    from scipy.spatial import cKDTree
    tet_tree = cKDTree(tet_verts)
    _, boundary_to_tet_map = tet_tree.query(boundary_mesh_verts, k=1)
    
    # Build set of boundary surface edges (in tet vertex indices)
    boundary_surface_edges = set()
    for face in boundary_faces:
        # Convert boundary mesh indices to tet indices
        v0_tet = boundary_to_tet_map[face[0]]
        v1_tet = boundary_to_tet_map[face[1]]
        v2_tet = boundary_to_tet_map[face[2]]
        
        # Add edges (sorted to ensure consistent ordering)
        boundary_surface_edges.add((min(v0_tet, v1_tet), max(v0_tet, v1_tet)))
        boundary_surface_edges.add((min(v1_tet, v2_tet), max(v1_tet, v2_tet)))
        boundary_surface_edges.add((min(v2_tet, v0_tet), max(v2_tet, v0_tet)))
    
    logger.info(f"Found {len(boundary_surface_edges)} edges on boundary surface")
    
    # Get labels for both endpoints of each edge
    v0_labels = boundary_labels[edges[:, 0]]
    v1_labels = boundary_labels[edges[:, 1]]
    
    # Check if each tet edge is on the boundary surface
    on_boundary_surface = np.zeros(n_edges, dtype=bool)
    for i, (v0, v1) in enumerate(edges):
        edge_key = (min(v0, v1), max(v0, v1))
        on_boundary_surface[i] = edge_key in boundary_surface_edges
    
    # Edge label rules (only for edges that are ON THE BOUNDARY SURFACE):
    # - Both H1: edge is H1 (1)
    # - Both H2: edge is H2 (2)
    # - Both inner: edge is inner (-1)
    # - Both on boundary surface but different types: mixed (-2)
    # - Otherwise: interior edge (0) - colored by weight
    
    # H1 edges (on surface, both on H1 boundary)
    h1_mask = on_boundary_surface & (v0_labels == 1) & (v1_labels == 1)
    edge_labels[h1_mask] = 1
    
    # H2 edges (on surface, both on H2 boundary)
    h2_mask = on_boundary_surface & (v0_labels == 2) & (v1_labels == 2)
    edge_labels[h2_mask] = 2
    
    # Inner boundary edges (on surface, both on inner boundary / part surface)
    inner_mask = on_boundary_surface & (v0_labels == -1) & (v1_labels == -1)
    edge_labels[inner_mask] = -1
    
    # Mixed boundary edges (on surface but not same type)
    mixed_mask = on_boundary_surface & ~h1_mask & ~h2_mask & ~inner_mask
    edge_labels[mixed_mask] = -2
    
    # All other edges (including interior-to-boundary connections) stay at 0
    # These will be colored by weight
    
    # Count stats
    n_interior = np.sum(edge_labels == 0)
    n_h1 = np.sum(edge_labels == 1)
    n_h2 = np.sum(edge_labels == 2)
    n_inner = np.sum(edge_labels == -1)
    n_mixed = np.sum(edge_labels == -2)
    
    logger.info(
        f"Edge boundary labels: interior={n_interior}, H1={n_h1}, H2={n_h2}, "
        f"inner={n_inner}, mixed={n_mixed}"
    )
    
    return edge_labels


def compute_distances_to_mesh_gpu(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    batch_size: int = 50000
) -> np.ndarray:
    """
    GPU-accelerated distance computation from points to mesh surface.
    
    Uses PyTorch CUDA to compute point-to-triangle distances in parallel on GPU.
    Falls back to CPU if CUDA is not available.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        batch_size: Points per GPU batch (to manage VRAM)
    
    Returns:
        (N,) array of unsigned distances to mesh surface
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        logger.warning("GPU not available, falling back to CPU computation")
        return compute_distances_to_mesh_cpu(query_points, target_mesh)
    
    import torch
    
    device = torch.device('cuda')
    n_points = len(query_points)
    n_faces = len(target_mesh.faces)
    
    # Get mesh data on GPU
    vertices = torch.tensor(target_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(target_mesh.faces, dtype=torch.int64, device=device)
    
    # Pre-compute triangle vertices
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Pre-compute triangle edge data (constant for all points)
    edge0 = v1 - v0  # (F, 3)
    edge1 = v2 - v0  # (F, 3)
    dot00 = (edge0 * edge0).sum(dim=-1)  # (F,)
    dot01 = (edge0 * edge1).sum(dim=-1)  # (F,)
    dot11 = (edge1 * edge1).sum(dim=-1)  # (F,)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)  # (F,)
    
    # Adjust batch size based on triangle count to fit in GPU memory
    # Memory per point-triangle pair ~ 12 bytes (3 floats for intermediate)
    # Target ~2GB working memory
    max_pairs = 150_000_000  # ~2GB for intermediates
    adaptive_batch = max(100, min(batch_size, max_pairs // n_faces))
    
    logger.debug(f"GPU: Processing {n_points} points against {n_faces} triangles (batch={adaptive_batch})")
    
    all_distances = torch.zeros(n_points, dtype=torch.float32, device=device)
    
    for start_idx in range(0, n_points, adaptive_batch):
        end_idx = min(start_idx + adaptive_batch, n_points)
        batch_points = torch.tensor(
            query_points[start_idx:end_idx], 
            dtype=torch.float32, 
            device=device
        )  # (B, 3)
        
        # Compute distances using memory-efficient method
        batch_distances = _point_to_triangles_distance_gpu_efficient(
            batch_points, v0, edge0, edge1, dot00, dot01, dot11, inv_denom
        )
        
        all_distances[start_idx:end_idx] = batch_distances
        
        # Clear cache periodically
        if (start_idx // adaptive_batch) % 10 == 0:
            torch.cuda.empty_cache()
            progress = 100 * end_idx / n_points
            logger.debug(f"  GPU: {end_idx}/{n_points} ({progress:.0f}%)")
    
    result = all_distances.cpu().numpy().astype(np.float64)
    torch.cuda.empty_cache()
    return result


def compute_distances_and_closest_points_gpu(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    batch_size: int = 30000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated computation of distances AND closest points from query points to mesh.
    
    This is useful for finding R (max hull-to-part distance) where we need both
    the distance and the closest point on the target mesh.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        batch_size: Points per GPU batch (smaller than distance-only due to more memory)
    
    Returns:
        Tuple of:
        - distances: (N,) array of unsigned distances to mesh surface
        - closest_points: (N, 3) array of closest points on mesh surface
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        logger.warning("GPU not available, falling back to CPU computation")
        closest_points, distances, _ = target_mesh.nearest.on_surface(query_points)
        return distances, closest_points
    
    import torch
    
    device = torch.device('cuda')
    n_points = len(query_points)
    n_faces = len(target_mesh.faces)
    
    logger.info(f"GPU: Computing distances and closest points for {n_points} points against {n_faces} triangles")
    
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
    
    # Adjust batch size - need more memory for storing closest points
    max_pairs = 100_000_000  # Conservative for storing closest points
    adaptive_batch = max(100, min(batch_size, max_pairs // n_faces))
    
    logger.debug(f"  Batch size: {adaptive_batch}")
    
    all_distances = torch.zeros(n_points, dtype=torch.float32, device=device)
    all_closest = torch.zeros((n_points, 3), dtype=torch.float32, device=device)
    
    for start_idx in range(0, n_points, adaptive_batch):
        end_idx = min(start_idx + adaptive_batch, n_points)
        batch_points = torch.tensor(
            query_points[start_idx:end_idx], 
            dtype=torch.float32, 
            device=device
        )  # (B, 3)
        
        # Compute distances and closest points
        batch_distances, batch_closest = _point_to_triangles_with_closest_gpu(
            batch_points, v0, edge0, edge1, dot00, dot01, dot11, inv_denom
        )
        
        all_distances[start_idx:end_idx] = batch_distances
        all_closest[start_idx:end_idx] = batch_closest
        
        # Clear cache periodically
        if (start_idx // adaptive_batch) % 5 == 0:
            torch.cuda.empty_cache()
            progress = 100 * end_idx / n_points
            logger.debug(f"  GPU: {end_idx}/{n_points} ({progress:.0f}%)")
    
    distances = all_distances.cpu().numpy().astype(np.float64)
    closest_points = all_closest.cpu().numpy().astype(np.float64)
    torch.cuda.empty_cache()
    
    logger.info(f"GPU: Completed distance computation")
    return distances, closest_points


def _point_to_triangles_with_closest_gpu(
    points: 'torch.Tensor',
    v0: 'torch.Tensor',
    edge0: 'torch.Tensor',
    edge1: 'torch.Tensor',
    dot00: 'torch.Tensor',
    dot01: 'torch.Tensor',
    dot11: 'torch.Tensor',
    inv_denom: 'torch.Tensor'
) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Compute minimum distance and closest point from each point to any triangle (GPU).
    
    Args:
        points: (B, 3) query points
        v0: (F, 3) first vertex of each triangle
        edge0, edge1: (F, 3) triangle edges
        dot00, dot01, dot11, inv_denom: (F,) pre-computed values
    
    Returns:
        Tuple of:
        - (B,) minimum distance for each point
        - (B, 3) closest point on mesh for each query point
    """
    import torch
    
    B = points.shape[0]
    F = v0.shape[0]
    
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
    u_clamped = torch.clamp(u, 0, 1)
    v_clamped = torch.clamp(v, 0, 1)
    
    # Ensure u + v <= 1
    uv_sum = u_clamped + v_clamped
    scale = torch.where(uv_sum > 1, 1.0 / uv_sum, torch.ones_like(uv_sum))
    u_final = u_clamped * scale
    v_final = v_clamped * scale
    
    # Closest point on each triangle: v0 + u*edge0 + v*edge1
    closest_on_tri = v0.unsqueeze(0) + u_final.unsqueeze(-1) * edge0.unsqueeze(0) + v_final.unsqueeze(-1) * edge1.unsqueeze(0)  # (B, F, 3)
    
    # Compute squared distance to each triangle
    diff = p - closest_on_tri  # (B, F, 3)
    dist_sq = (diff * diff).sum(dim=-1)  # (B, F)
    
    # Find minimum distance triangle for each point
    min_dist_sq, min_indices = dist_sq.min(dim=1)  # (B,), (B,)
    min_dist = torch.sqrt(min_dist_sq)  # (B,)
    
    # Gather closest points using the minimum indices
    # closest_on_tri is (B, F, 3), min_indices is (B,)
    batch_indices = torch.arange(B, device=points.device)
    closest_points = closest_on_tri[batch_indices, min_indices]  # (B, 3)
    
    return min_dist, closest_points


def _point_to_triangles_distance_gpu_efficient(
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
    Memory-efficient version using pre-computed triangle data.
    
    Args:
        points: (B, 3) query points
        v0: (F, 3) first vertex of each triangle
        edge0, edge1: (F, 3) triangle edges
        dot00, dot01, dot11, inv_denom: (F,) pre-computed values
    
    Returns:
        (B,) minimum distance for each point
    """
    import torch
    
    B = points.shape[0]
    
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
    u_clamped = torch.clamp(u, 0, 1)
    v_clamped = torch.clamp(v, 0, 1)
    
    # Ensure u + v <= 1
    uv_sum = u_clamped + v_clamped
    scale = torch.where(uv_sum > 1, 1.0 / uv_sum, torch.ones_like(uv_sum))
    u_final = u_clamped * scale
    v_final = v_clamped * scale
    
    # Closest point on triangle: v0 + u*edge0 + v*edge1
    # Compute squared distance directly without storing closest point
    # closest = v0 + u*edge0 + v*edge1
    # diff = p - closest = p - v0 - u*edge0 - v*edge1 = v0_to_p - u*edge0 - v*edge1
    diff = v0_to_p - u_final.unsqueeze(-1) * edge0.unsqueeze(0) - v_final.unsqueeze(-1) * edge1.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=-1)  # (B, F)
    
    # Minimum distance across all triangles
    min_dist = torch.sqrt(dist_sq.min(dim=1).values)  # (B,)
    
    return min_dist


def compute_distances_to_mesh_cpu(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    batch_size: int = 50000
) -> np.ndarray:
    """
    CPU distance computation from points to mesh surface.
    
    Uses trimesh's proximity query with batched processing to manage memory.
    Note: trimesh's rtree is NOT thread-safe, so we process sequentially.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        batch_size: Number of points to process per batch
    
    Returns:
        (N,) array of unsigned distances to mesh surface
    """
    n_points = len(query_points)
    
    if n_points <= batch_size:
        # Small enough to process in one go
        closest_points, distances, triangle_ids = target_mesh.nearest.on_surface(query_points)
        return distances
    
    # Process in batches sequentially (rtree is not thread-safe)
    logger.debug(f"CPU: Processing {n_points} points in batches of {batch_size}")
    
    all_distances = np.zeros(n_points, dtype=np.float64)
    
    n_batches = (n_points + batch_size - 1) // batch_size
    for i, start_idx in enumerate(range(0, n_points, batch_size)):
        end_idx = min(start_idx + batch_size, n_points)
        batch_points = query_points[start_idx:end_idx]
        
        closest_points, distances, triangle_ids = target_mesh.nearest.on_surface(batch_points)
        all_distances[start_idx:end_idx] = distances
        
        if (i + 1) % 5 == 0 or (i + 1) == n_batches:
            logger.debug(f"  CPU: Batch {i+1}/{n_batches} ({100*(i+1)/n_batches:.0f}%)")
    
    return all_distances


def compute_distances_to_mesh(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    use_gpu: bool = True,
    batch_size: int = None
) -> np.ndarray:
    """
    Compute distances from query points to target mesh surface.
    
    Automatically selects GPU or CPU computation based on availability.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        use_gpu: Whether to prefer GPU if available
        batch_size: Batch size (auto-selected if None)
    
    Returns:
        (N,) array of unsigned distances to mesh surface
    """
    if use_gpu and CUDA_AVAILABLE:
        logger.info(f"Using GPU acceleration for distance computation ({len(query_points)} points)")
        return compute_distances_to_mesh_gpu(
            query_points, 
            target_mesh, 
            batch_size=batch_size or 50000
        )
    else:
        if use_gpu and not CUDA_AVAILABLE:
            logger.info("GPU requested but CUDA not available, using CPU")
        return compute_distances_to_mesh_cpu(
            query_points, 
            target_mesh, 
            batch_size=batch_size or 50000
        )


def identify_boundary_vertices(
    tet_vertices: np.ndarray,
    cavity_mesh: trimesh.Trimesh,
    tolerance: float = None
) -> np.ndarray:
    """
    Identify which tet vertices lie on the cavity boundary surface.
    
    Uses trimesh's fast proximity query (CPU only, but efficient for this task)
    since we just need a boolean mask, not exact distances.
    
    Args:
        tet_vertices: (N, 3) tetrahedral mesh vertices
        cavity_mesh: Original cavity surface mesh
        tolerance: Distance tolerance for boundary detection.
                   If None, uses 0.1% of bounding box diagonal.
    
    Returns:
        (N,) boolean array - True if vertex is on boundary
    """
    if tolerance is None:
        bbox_diag = np.linalg.norm(cavity_mesh.bounds[1] - cavity_mesh.bounds[0])
        tolerance = bbox_diag * 0.001
    
    # Use trimesh's fast proximity query directly (CPU, but efficient for boundary check)
    # This is faster than our GPU implementation for this specific use case
    # because we're checking ALL vertices against the surface they came from
    closest_points, distances, triangle_ids = cavity_mesh.nearest.on_surface(tet_vertices)
    
    # Vertices within tolerance are on boundary
    boundary_mask = distances < tolerance
    
    return boundary_mask


def label_boundary_from_classification(
    tet_vertices: np.ndarray,
    boundary_mask: np.ndarray,
    boundary_mesh: trimesh.Trimesh,
    classification_result,  # MoldHalfClassificationResult
    hull_mesh: trimesh.Trimesh = None  # Not used but kept for backwards compatibility
) -> np.ndarray:
    """
    Label boundary vertices as H1, H2, or inner boundary based on mold half classification.
    
    This function now works directly with the tetrahedral boundary mesh classification.
    Face labels are propagated to vertices by majority voting from adjacent faces.
    
    Args:
        tet_vertices: (N, 3) tetrahedral mesh vertices
        boundary_mask: (N,) boolean mask of boundary vertices
        boundary_mesh: The tetrahedral boundary mesh (classification was done on this mesh)
        classification_result: Result from mold half classification on boundary_mesh
        hull_mesh: Not used - kept for backwards compatibility
    
    Returns:
        (N,) int8 array: 0=interior/boundary zone, 1=H1 boundary, 2=H2 boundary, -1=inner boundary (part surface)
    """
    labels = np.zeros(len(tet_vertices), dtype=np.int8)
    
    # Get boundary vertex indices
    boundary_indices = np.where(boundary_mask)[0]
    
    if len(boundary_indices) == 0:
        return labels
    
    # Get classification sets
    h1_set = classification_result.h1_triangles
    h2_set = classification_result.h2_triangles
    inner_set = classification_result.inner_boundary_triangles
    
    # Build vertex-to-face adjacency from the boundary mesh
    # Each vertex gets labels from all faces it belongs to
    vertex_h1_count = np.zeros(len(tet_vertices), dtype=np.int32)
    vertex_h2_count = np.zeros(len(tet_vertices), dtype=np.int32)
    vertex_inner_count = np.zeros(len(tet_vertices), dtype=np.int32)
    
    # The boundary mesh vertices are indexed from 0, but they correspond to 
    # tetrahedral vertices. We need to map boundary mesh faces to tet vertices.
    # boundary_mesh.faces contains indices into boundary_mesh.vertices
    
    # Find the mapping from boundary mesh vertices to tet vertices
    # boundary_mesh.vertices should be a subset of tet_vertices (boundary vertices)
    # Use KDTree for efficient lookup
    from scipy.spatial import cKDTree
    
    tet_tree = cKDTree(tet_vertices)
    
    # Find which tet vertex each boundary mesh vertex corresponds to
    boundary_to_tet = tet_tree.query(boundary_mesh.vertices, k=1)[1]
    
    # Now iterate over faces and accumulate labels for each vertex
    for face_idx, face in enumerate(boundary_mesh.faces):
        # Map face vertex indices to tet vertex indices
        tet_v0 = boundary_to_tet[face[0]]
        tet_v1 = boundary_to_tet[face[1]]
        tet_v2 = boundary_to_tet[face[2]]
        
        if face_idx in h1_set:
            vertex_h1_count[tet_v0] += 1
            vertex_h1_count[tet_v1] += 1
            vertex_h1_count[tet_v2] += 1
        elif face_idx in h2_set:
            vertex_h2_count[tet_v0] += 1
            vertex_h2_count[tet_v1] += 1
            vertex_h2_count[tet_v2] += 1
        elif face_idx in inner_set:
            vertex_inner_count[tet_v0] += 1
            vertex_inner_count[tet_v1] += 1
            vertex_inner_count[tet_v2] += 1
        # Boundary zone faces contribute to no count (vertex stays 0)
    
    # For each boundary vertex, assign label based on majority vote
    # Strategy: Use majority voting across all face types
    # Inner boundary vertices should ONLY be marked as seeds if they don't touch H1/H2
    for vert_idx in boundary_indices:
        h1_count = vertex_h1_count[vert_idx]
        h2_count = vertex_h2_count[vert_idx]
        inner_count = vertex_inner_count[vert_idx]
        
        total = h1_count + h2_count + inner_count
        
        if total == 0:
            # No classified faces touch this vertex - stays 0 (boundary zone)
            continue
        
        # Use majority voting
        if h1_count >= h2_count and h1_count >= inner_count:
            labels[vert_idx] = 1
        elif h2_count >= h1_count and h2_count >= inner_count:
            labels[vert_idx] = 2
        else:
            # Inner has majority - this is a seed vertex
            labels[vert_idx] = -1
    
    # Log distribution
    n_h1 = np.sum(labels == 1)
    n_h2 = np.sum(labels == 2)
    n_inner = np.sum(labels == -1)
    n_unclassified = np.sum(labels == 0)
    logger.info(f"label_boundary_from_classification: H1={n_h1}, H2={n_h2}, inner(seeds)={n_inner}, unclassified={n_unclassified}")
    
    return labels


def generate_tetrahedral_mesh(
    cavity_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    classification_result=None,  # MoldHalfClassificationResult
    edge_length_fac: float = 0.05,
    optimize: bool = True,
    compute_distances: bool = True
) -> TetrahedralMeshResult:
    """
    Generate tetrahedral mesh with edge weights for parting surface computation.
    
    This is the main entry point for tetrahedral mesh generation.
    
    Algorithm for edge weights:
    1. For each edge, compute midpoint distance to part mesh surface
    2. edge_weight = 1 / (distance^2 + 0.25)
    3. weighted_edge_length = edge_length * edge_weight
    
    Args:
        cavity_mesh: The cavity mesh (Hull - Part CSG result)
        part_mesh: Original part mesh (for distance computation)
        hull_mesh: Original hull mesh (not used currently, kept for API compatibility)
        classification_result: Mold half classification result (for boundary labeling)
        edge_length_fac: Target edge length as fraction of bbox diagonal
        optimize: Whether to optimize tet quality
        compute_distances: Whether to compute distance fields and weights
    
    Returns:
        TetrahedralMeshResult with all mesh data and weights
    """
    import time
    start_time = time.time()
    
    # Step 1: Tetrahedralize
    tet_start = time.time()
    tet_vertices, tetrahedra = tetrahedralize_mesh(
        cavity_mesh, 
        edge_length_fac=edge_length_fac,
        optimize=optimize
    )
    tet_time = (time.time() - tet_start) * 1000
    
    # Step 2: Extract edges
    edges = extract_edges(tetrahedra)
    edge_lengths = compute_edge_lengths(tet_vertices, edges)
    
    logger.info(f"Extracted {len(edges)} unique edges")
    
    # Step 3: Extract boundary surface from tetrahedral mesh
    boundary_mesh = extract_boundary_surface(tet_vertices, tetrahedra)
    
    # Initialize result
    result = TetrahedralMeshResult(
        vertices=tet_vertices,
        tetrahedra=tetrahedra,
        edges=edges,
        edge_lengths=edge_lengths,
        boundary_mesh=boundary_mesh,
        num_vertices=len(tet_vertices),
        num_tetrahedra=len(tetrahedra),
        num_edges=len(edges),
        num_boundary_faces=len(boundary_mesh.faces),
        tetrahedralize_time_ms=tet_time
    )
    
    # Step 4: Identify boundary vertices (vertices on the boundary surface)
    boundary_mask = identify_boundary_vertices(tet_vertices, boundary_mesh)
    result.boundary_vertices = boundary_mask
    logger.info(f"Found {np.sum(boundary_mask)} boundary vertices out of {len(tet_vertices)}")
    
    # Step 4: Compute edge weights based on distance to part surface
    if compute_distances:
        # Compute edge midpoint distances to part mesh
        edge_midpoints = (tet_vertices[edges[:, 0]] + tet_vertices[edges[:, 1]]) / 2
        
        logger.debug("Computing edge midpoint distances to part mesh...")
        edge_dist_to_part = compute_distances_to_mesh(edge_midpoints, part_mesh)
        result.edge_dist_to_part = edge_dist_to_part
        
        # Compute edge weights: weight = 1 / (dist^2 + 0.25)
        # This gives higher weight (shorter effective distance) near the part surface
        result.edge_weights = 1.0 / (edge_dist_to_part ** 2 + 0.25)
        
        # Compute weighted edge lengths
        result.weighted_edge_lengths = edge_lengths * result.edge_weights
        
        logger.info(f"Edge weights: min={result.edge_weights.min():.4f}, max={result.edge_weights.max():.4f}, mean={result.edge_weights.mean():.4f}")
    
    # Step 5: Label boundary vertices from classification (optional)
    # Note: The preferred path is to use EdgeWeightsWorker which classifies
    # the tetrahedral boundary mesh directly for better accuracy.
    if classification_result is not None:
        logger.debug("Labeling boundary vertices from classification...")
        result.boundary_labels = label_boundary_from_classification(
            tet_vertices,
            boundary_mask,
            boundary_mesh,  # Use tet boundary mesh for vertex mapping
            classification_result,
            hull_mesh
        )
        
        n_h1 = np.sum(result.boundary_labels == 1)
        n_h2 = np.sum(result.boundary_labels == 2)
        n_inner = np.sum(result.boundary_labels == -1)
        logger.info(f"Boundary labels: H1={n_h1}, H2={n_h2}, inner(seeds)={n_inner}")
    
    result.total_time_ms = (time.time() - start_time) * 1000
    
    logger.info(
        f"Tetrahedral mesh complete: {result.num_vertices} verts, "
        f"{result.num_tetrahedra} tets, {result.num_edges} edges, "
        f"time={result.total_time_ms:.0f}ms"
    )
    
    return result


def build_vertex_adjacency(edges: np.ndarray, num_vertices: int) -> Dict[int, List[int]]:
    """
    Build vertex adjacency map from edge list.
    
    Args:
        edges: (E, 2) edge vertex indices
        num_vertices: Total number of vertices
    
    Returns:
        Dict mapping vertex index to list of adjacent vertex indices
    """
    adjacency: Dict[int, List[int]] = {i: [] for i in range(num_vertices)}
    
    for v0, v1 in edges:
        adjacency[v0].append(v1)
        adjacency[v1].append(v0)
    
    return adjacency


def build_edge_index_map(edges: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Build map from edge vertex pair to edge index.
    
    Args:
        edges: (E, 2) edge vertex indices (sorted)
    
    Returns:
        Dict mapping (v0, v1) tuple to edge index
    """
    edge_map = {}
    for i, (v0, v1) in enumerate(edges):
        # Store with smaller index first
        key = (min(v0, v1), max(v0, v1))
        edge_map[key] = i
    
    return edge_map


def compute_edge_weights(
    tet_result: TetrahedralMeshResult,
    shell_radius: float = None,
    bias_boundary_layers: int = 1
) -> np.ndarray:
    """
    Compute edge weights for Dijkstra-based escape labeling.
    
    The weight scheme follows the React app's volumetric grid approach but applied to edges:
    
    For each edge, the cost is: L * wt
    where:
        L = edge length (Euclidean distance)
        wt = weighting factor based on distance to part mesh
    
    Weighting factor computation:
    - For boundary-adjacent edges:
        biasedDist = δ_part + λ_w  where λ_w = R - δ_shell
        wt = 1 / (biasedDist² + 0.25)
    - For interior edges:
        wt = 1 / (δ_part² + 0.25)
    
    This biases paths toward the shell boundary while preferring thick regions.
    
    Args:
        tet_result: Tetrahedral mesh result with distances computed
        shell_radius: Radius R for shell bias. If None, computed from max shell distance.
        bias_boundary_layers: Number of edge layers from boundary to apply shell bias
    
    Returns:
        (E,) array of edge weights (cost = edge_length * weight)
    """
    if tet_result.edge_dist_to_part is None:
        raise ValueError("Edge distances to part not computed. Run with compute_distances=True")
    
    edges = tet_result.edges
    edge_lengths = tet_result.edge_lengths
    dist_to_part = tet_result.edge_dist_to_part
    dist_to_shell = tet_result.edge_dist_to_shell
    
    n_edges = len(edges)
    
    # Compute shell radius if not provided
    if shell_radius is None and dist_to_shell is not None:
        shell_radius = np.max(dist_to_shell) * 1.1  # 10% buffer
    
    # Identify boundary-adjacent edges using vertex boundary mask
    boundary_verts = tet_result.boundary_vertices
    if boundary_verts is None:
        # No boundary info, treat all as interior
        is_boundary_edge = np.zeros(n_edges, dtype=bool)
    else:
        # Edge is boundary-adjacent if either endpoint is on boundary
        v0_boundary = boundary_verts[edges[:, 0]]
        v1_boundary = boundary_verts[edges[:, 1]]
        is_boundary_edge = v0_boundary | v1_boundary
    
    # Compute weighting factors
    weights = np.zeros(n_edges, dtype=np.float64)
    
    # Interior edges: wt = 1 / (δ_part² + 0.25)
    interior_mask = ~is_boundary_edge
    weights[interior_mask] = 1.0 / (dist_to_part[interior_mask] ** 2 + 0.25)
    
    # Boundary-adjacent edges: biased weight
    if dist_to_shell is not None and shell_radius is not None:
        boundary_mask = is_boundary_edge
        lambda_w = shell_radius - dist_to_shell[boundary_mask]
        biased_dist = dist_to_part[boundary_mask] + np.maximum(lambda_w, 0)
        weights[boundary_mask] = 1.0 / (biased_dist ** 2 + 0.25)
    else:
        # Fall back to unbiased for boundary edges
        boundary_mask = is_boundary_edge
        weights[boundary_mask] = 1.0 / (dist_to_part[boundary_mask] ** 2 + 0.25)
    
    logger.debug(f"Computed edge weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    
    return weights


def compute_edge_costs(
    tet_result: TetrahedralMeshResult,
    shell_radius: float = None
) -> np.ndarray:
    """
    Compute edge traversal costs for Dijkstra (length * weight).
    
    Args:
        tet_result: Tetrahedral mesh result
        shell_radius: Radius for shell bias computation
    
    Returns:
        (E,) array of edge costs
    """
    weights = compute_edge_weights(tet_result, shell_radius)
    costs = tet_result.edge_lengths * weights
    
    logger.debug(f"Computed edge costs: min={costs.min():.6f}, max={costs.max():.6f}, mean={costs.mean():.6f}")
    
    return costs


def run_dijkstra_escape_labeling(
    tet_result: 'TetrahedralMeshResult',
    use_weighted_edges: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """
    Run Dijkstra's algorithm FROM each INTERIOR vertex (not on H1/H2 boundary)
    TO the external boundary (H1 or H2).
    
    This runs from ALL interior vertices - both those on the part surface (seeds)
    and those in the cavity interior - to classify every interior vertex and edge.
    
    Each interior vertex independently walks through the tetrahedral mesh graph to reach
    either the H1 or H2 boundary. The vertex is labeled based on which boundary it 
    reaches first via the cheapest path.
    
    Uses C++ implementation when available for 10-20x speedup.
    
    Args:
        tet_result: Tetrahedral mesh result with boundary_labels and edge_weights computed
                   (uses the INFLATED mesh where edge weights were computed)
        use_weighted_edges: If True, use edge_length / edge_weight as cost
                           If False, use just edge_length as cost
    
    Returns:
        Tuple of:
            - interior_escape_labels: (I,) int8 array for each interior vertex where:
                1 = vertex walks to H1
                2 = vertex walks to H2
                0 = unreachable (no path to H1 or H2)
            - interior_vertex_indices: (I,) int array of interior vertex indices in the tet mesh
            - interior_distances: (I,) float array of shortest path distances to boundary
            - interior_escape_destinations: (I,) int array of destination boundary vertex indices
            - interior_escape_paths: List of paths, each path is a list of vertex indices
    """
    # Use C++ implementation if available
    if CPP_FAST_AVAILABLE:
        return _run_dijkstra_escape_labeling_cpp(tet_result, use_weighted_edges)
    else:
        return _run_dijkstra_escape_labeling_python(tet_result, use_weighted_edges)


def _run_dijkstra_escape_labeling_cpp(
    tet_result: 'TetrahedralMeshResult',
    use_weighted_edges: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """
    C++ implementation of Dijkstra escape labeling - 10-20x faster than Python.
    """
    import time
    start_time = time.time()
    
    boundary_labels = tet_result.boundary_labels
    if boundary_labels is None:
        raise ValueError("boundary_labels must be computed before running Dijkstra")
    
    # Prepare arrays for C++ call
    edges = np.asarray(tet_result.edges, dtype=np.int64)
    edge_lengths = np.asarray(tet_result.edge_lengths, dtype=np.float64)
    edge_weights = np.asarray(tet_result.edge_weights, dtype=np.float64)
    boundary_labels_arr = np.asarray(boundary_labels, dtype=np.int8)
    
    # Log info about interior vertices
    interior_count = np.sum((boundary_labels_arr != 1) & (boundary_labels_arr != 2))
    seed_count = np.sum(boundary_labels_arr == -1)
    h1_count = np.sum(boundary_labels_arr == 1)
    h2_count = np.sum(boundary_labels_arr == 2)
    
    logger.info(f"[C++] Dijkstra: {interior_count} interior ({seed_count} seeds), {h1_count} H1, {h2_count} H2 targets")
    
    # Call C++ implementation
    result = _cpp_fast.dijkstra_escape_labeling(
        edges, edge_lengths, edge_weights, boundary_labels_arr, use_weighted_edges
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    # Extract results
    escape_labels = np.asarray(result.escape_labels)
    vertex_indices = np.asarray(result.vertex_indices)
    distances = np.asarray(result.distances)
    destinations = np.asarray(result.destinations)
    paths = [list(p) for p in result.paths]
    
    # Count results
    n_h1 = np.sum(escape_labels == 1)
    n_h2 = np.sum(escape_labels == 2)
    n_unreached = np.sum(escape_labels == 0)
    
    logger.info(f"[C++] Dijkstra complete in {elapsed:.0f}ms: {n_h1}→H1, {n_h2}→H2, {n_unreached} unreached")
    
    return escape_labels, vertex_indices, distances, destinations, paths


def _run_dijkstra_escape_labeling_python(
    tet_result: 'TetrahedralMeshResult',
    use_weighted_edges: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """
    Python fallback implementation of Dijkstra escape labeling.
    """
    import heapq
    import time
    
    start_time = time.time()
    
    vertices = tet_result.vertices  # INFLATED vertices
    edges = tet_result.edges
    edge_lengths = tet_result.edge_lengths
    edge_weights = tet_result.edge_weights
    boundary_labels = tet_result.boundary_labels
    
    n_vertices = len(vertices)
    
    if boundary_labels is None:
        raise ValueError("boundary_labels must be computed before running Dijkstra")
    
    # Compute edge costs
    # edge_weights = 1/(dist² + ε), so higher weight = closer to part
    # cost = length × weight: HIGH cost near part, guides paths along gradient
    if use_weighted_edges and edge_weights is not None:
        edge_costs = edge_lengths * edge_weights
    else:
        edge_costs = edge_lengths.copy()
    
    logger.debug(f"Edge costs: min={edge_costs.min():.4f}, max={edge_costs.max():.4f}, mean={edge_costs.mean():.4f}")
    
    # Build adjacency list with edge costs
    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_vertices)}
    
    for edge_idx, (v0, v1) in enumerate(edges):
        cost = edge_costs[edge_idx]
        adjacency[v0].append((v1, cost))
        adjacency[v1].append((v0, cost))
    
    # Identify ALL interior vertices (both seeds on part surface AND cavity interior)
    # Interior = NOT on H1 and NOT on H2
    # boundary_labels: 0=interior, 1=H1, 2=H2, -1=inner boundary (part surface)
    interior_vertex_indices = np.where((boundary_labels != 1) & (boundary_labels != 2))[0]
    h1_vertices = set(np.where(boundary_labels == 1)[0])
    h2_vertices = set(np.where(boundary_labels == 2)[0])
    
    # Also track which interior vertices are on the part surface (seeds) vs truly interior
    seed_vertex_set = set(np.where(boundary_labels == -1)[0])
    n_seeds = len(seed_vertex_set)
    n_interior = len(interior_vertex_indices)
    n_truly_interior = n_interior - n_seeds
    
    logger.info(f"Dijkstra: {n_interior} interior vertices ({n_seeds} on part surface, {n_truly_interior} in cavity)")
    logger.info(f"Dijkstra: {len(h1_vertices)} H1 targets, {len(h2_vertices)} H2 targets")
    
    if n_interior == 0:
        logger.warning("No interior vertices found")
        return (np.array([], dtype=np.int8), np.array([], dtype=np.int64), 
                np.array([], dtype=np.float64), np.array([], dtype=np.int64), [])
    
    if len(h1_vertices) == 0 and len(h2_vertices) == 0:
        logger.warning("No H1 or H2 boundary vertices found")
        return (np.zeros(n_interior, dtype=np.int8), interior_vertex_indices, 
                np.full(n_interior, np.inf), np.full(n_interior, -1, dtype=np.int64), 
                [[] for _ in range(n_interior)])
    
    # Run Dijkstra FROM each interior vertex to find shortest path to H1 or H2
    interior_escape_labels = np.zeros(n_interior, dtype=np.int8)
    interior_distances = np.full(n_interior, np.inf, dtype=np.float64)
    interior_escape_destinations = np.full(n_interior, -1, dtype=np.int64)
    interior_escape_paths: List[List[int]] = []
    
    for idx, start_v in enumerate(interior_vertex_indices):
        dist = np.full(n_vertices, np.inf, dtype=np.float64)
        dist[start_v] = 0.0
        
        # Predecessor array to reconstruct path
        predecessor = np.full(n_vertices, -1, dtype=np.int64)
        
        pq = [(0.0, start_v)]
        visited = np.zeros(n_vertices, dtype=bool)
        
        destination = -1
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if visited[u]:
                continue
            visited[u] = True
            
            # Check if we reached a target
            if u in h1_vertices:
                interior_escape_labels[idx] = 1
                interior_distances[idx] = d
                destination = u
                break
            elif u in h2_vertices:
                interior_escape_labels[idx] = 2
                interior_distances[idx] = d
                destination = u
                break
            
            # Propagate to neighbors
            for v, cost in adjacency[u]:
                if visited[v]:
                    continue
                new_dist = d + cost
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    predecessor[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        interior_escape_destinations[idx] = destination
        
        # Reconstruct path from start to destination
        if destination >= 0:
            path = []
            current = destination
            while current >= 0:
                path.append(current)
                current = predecessor[current]
            path.reverse()  # Now path goes from start to destination
            interior_escape_paths.append(path)
        else:
            interior_escape_paths.append([])
    
    # Count results
    n_h1 = np.sum(interior_escape_labels == 1)
    n_h2 = np.sum(interior_escape_labels == 2)
    n_unreached = np.sum(interior_escape_labels == 0)
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Dijkstra complete in {elapsed:.0f}ms: {n_h1} vertices→H1, {n_h2} vertices→H2, {n_unreached} unreached")
    
    return interior_escape_labels, interior_vertex_indices, interior_distances, interior_escape_destinations, interior_escape_paths


def find_secondary_cutting_edges(
    tet_result: TetrahedralMeshResult,
    part_mesh: trimesh.Trimesh,
    boundary_mesh: trimesh.Trimesh = None,
    sensitivity: float = 0.5,
    use_gpu: bool = True
) -> List[Tuple[int, int]]:
    """
    Find secondary cutting edges where the membrane between same-label interior vertices 
    intersects the part mesh representation.
    
    This works on ALL interior edges (edges between any two interior vertices that
    both escape to the same boundary H1 or H2), not just edges on the part surface.
    
    For each interior edge e = (vi, vj) between two interior vertices with the SAME escape label
    (both escape to H1 or both escape to H2):
    
    1. Get escape paths: vi → wi (on ∂H) and vj → wj (on ∂H)
    2. Compute shortest path wi → wj on the boundary mesh ∂H
    3. Form a "membrane" surface bounded by:
       - Edge (vi, vj)
       - Path vi → wi  
       - Path vj → wj
       - Path wi → wj on ∂H
    4. If this membrane intersects the part mesh M, mark edge as secondary cut
    
    Args:
        tet_result: TetrahedralMeshResult with Dijkstra results (paths, destinations)
        part_mesh: The original part mesh M to check intersection against
        boundary_mesh: The boundary mesh ∂H (uses tet_result.boundary_mesh_original if not provided)
        sensitivity: 0.0 to 1.0, controls how aggressively to detect secondary cuts
                    0.0 = very strict (only clear intersections)
                    1.0 = very sensitive (catches near-misses)
        use_gpu: Whether to use CUDA acceleration if available (default True)
    
    Returns:
        List of (vi, vj) tuples representing secondary cutting edges
    """
    import time
    import heapq
    
    start_time = time.time()
    
    if tet_result.seed_escape_labels is None:
        raise ValueError("Dijkstra must be run before finding secondary cuts")
    
    if tet_result.seed_escape_paths is None:
        raise ValueError("Escape paths must be computed (run updated Dijkstra)")
    
    # Use the INFLATED geometry for membrane construction - this is where Dijkstra paths were computed
    # The escape paths are vertex indices that refer to the inflated mesh vertices
    vertices = tet_result.vertices  # Inflated mesh for membrane geometry
    
    # For the boundary mesh, use the current (inflated) boundary mesh
    if boundary_mesh is None:
        boundary_mesh = tet_result.boundary_mesh
    
    if boundary_mesh is None:
        logger.warning("No boundary mesh available for secondary cuts")
        return []
    
    logger.info(f"Secondary cuts: using inflated vertices ({len(vertices)} verts), boundary_mesh ({len(boundary_mesh.vertices)} verts)")
    
    # These now contain ALL interior vertices (not just those on part surface)
    interior_vertex_indices = tet_result.seed_vertex_indices
    interior_escape_labels = tet_result.seed_escape_labels
    interior_escape_destinations = tet_result.seed_escape_destinations
    interior_escape_paths = tet_result.seed_escape_paths
    boundary_labels = tet_result.boundary_labels
    
    n_interior = len(interior_vertex_indices)
    
    # Build mapping from vertex index to interior array index
    vertex_to_interior_idx = {}
    for idx, v in enumerate(interior_vertex_indices):
        vertex_to_interior_idx[v] = idx
    
    interior_set = set(interior_vertex_indices)
    
    # Identify which interior vertices are on the part surface (seeds)
    # boundary_labels == -1 means on part surface
    part_surface_vertices = set()
    if boundary_labels is not None:
        for v in interior_vertex_indices:
            if boundary_labels[v] == -1:
                part_surface_vertices.add(v)
    else:
        # If no boundary labels, assume all interior vertices could be part surface
        part_surface_vertices = interior_set
    
    logger.info(f"Secondary cuts: {n_interior} interior vertices, {len(part_surface_vertices)} on part surface")
    
    # Build seed triangles from PART MESH faces
    # These represent the part cross-section that membranes should not pass through
    from scipy.spatial import cKDTree
    
    # Map part mesh vertices to tet mesh vertices
    part_verts = np.asarray(part_mesh.vertices)
    tet_tree = cKDTree(vertices)
    _, part_to_tet = tet_tree.query(part_verts, k=1)
    
    # Find TRIANGULAR FACES from PART MESH where ALL THREE vertices map to part surface vertices
    # These triangles represent the part surface in tetrahedral mesh resolution
    seed_triangles = []  # List of (v0, v1, v2) tet vertex indices
    seed_triangle_positions = []  # List of (pos0, pos1, pos2) positions
    
    for face in part_mesh.faces:
        tet_v0 = part_to_tet[face[0]]
        tet_v1 = part_to_tet[face[1]]
        tet_v2 = part_to_tet[face[2]]
        
        # Check if all three vertices are on part surface
        if tet_v0 in part_surface_vertices and tet_v1 in part_surface_vertices and tet_v2 in part_surface_vertices:
            # Avoid degenerate triangles
            if tet_v0 != tet_v1 and tet_v1 != tet_v2 and tet_v2 != tet_v0:
                seed_triangles.append((tet_v0, tet_v1, tet_v2))
                seed_triangle_positions.append((
                    vertices[tet_v0].copy(),
                    vertices[tet_v1].copy(),
                    vertices[tet_v2].copy()
                ))
    
    logger.info(f"Secondary cuts: found {len(seed_triangles)} seed triangles (part surface representation)")
    
    # Find ALL interior edges from tetrahedral mesh where both vertices are interior
    # and have the SAME escape label (both H1 or both H2)
    # IMPORTANT: Exclude edges where BOTH vertices are on the part surface - those are
    # not secondary cuts, they're edges ON the part itself. Their membranes would
    # trivially intersect the part surface, causing false positives.
    candidate_edges = []
    tet_edges = tet_result.edges
    n_both_on_part_surface = 0  # Counter for edges filtered out
    
    for v0, v1 in tet_edges:
        if v0 in interior_set and v1 in interior_set:
            # Skip edges where BOTH vertices are on the part surface
            # These are not secondary cuts - the membrane would trivially touch the part
            if v0 in part_surface_vertices and v1 in part_surface_vertices:
                n_both_on_part_surface += 1
                continue
            
            idx0 = vertex_to_interior_idx[v0]
            idx1 = vertex_to_interior_idx[v1]
            
            label0 = interior_escape_labels[idx0]
            label1 = interior_escape_labels[idx1]
            
            # Only consider same-label edges (both H1 or both H2)
            # Different label edges are primary cuts (yellow), not secondary
            if label0 == label1 and label0 > 0:
                candidate_edges.append((v0, v1, idx0, idx1))
    
    logger.info(f"Secondary cuts: {len(candidate_edges)} same-label interior edges to check (skipped {n_both_on_part_surface} edges on part surface)")
    
    if len(candidate_edges) == 0 or len(seed_triangles) == 0:
        return []
    
    # Build outer boundary mesh adjacency for shortest path computation
    boundary_verts = np.asarray(boundary_mesh.vertices)
    boundary_adjacency = _build_boundary_adjacency(boundary_mesh)
    
    # Calculate minimum membrane thickness to filter out nearly-degenerate membranes
    # Higher sensitivity = smaller threshold = more detections
    # Lower sensitivity = larger threshold = fewer detections (filters out borderline cases)
    # NOTE: Disabled for now as it was causing issues - set to 0 to not filter
    avg_edge_length = np.mean(tet_result.edge_lengths) if tet_result.edge_lengths is not None else 1.0
    # min_membrane_thickness = avg_edge_length * 0.1 * (1.0 - sensitivity)
    min_membrane_thickness = 0.0  # Disabled - rely on triangle intersection only
    
    logger.info(f"Secondary cuts: sensitivity={sensitivity:.2f}, avg_edge_length={avg_edge_length:.4f}")
    
    # Gather all membrane data first
    edge_membrane_data = []
    
    for vi, vj, idx_i, idx_j in candidate_edges:
        # Get escape paths
        path_i = interior_escape_paths[idx_i]  # vi → wi
        path_j = interior_escape_paths[idx_j]  # vj → wj
        
        if len(path_i) < 1 or len(path_j) < 1:
            continue
        
        wi = interior_escape_destinations[idx_i]
        wj = interior_escape_destinations[idx_j]
        
        if wi < 0 or wj < 0:
            continue
        
        # Verify that path ends at the boundary destination
        # The path should be [start_vertex, ..., boundary_vertex]
        if path_i[-1] != wi:
            logger.warning(f"Path {idx_i} does not end at destination: path[-1]={path_i[-1]}, wi={wi}")
        if path_j[-1] != wj:
            logger.warning(f"Path {idx_j} does not end at destination: path[-1]={path_j[-1]}, wj={wj}")
        
        # Map wi, wj to boundary mesh vertex indices
        wi_boundary = _find_nearest_boundary_vertex(vertices[wi], boundary_verts)
        wj_boundary = _find_nearest_boundary_vertex(vertices[wj], boundary_verts)
        
        # Compute shortest path wi → wj on boundary mesh
        boundary_path = _shortest_path_on_boundary(
            wi_boundary, wj_boundary, boundary_mesh, boundary_adjacency
        )
        
        edge_membrane_data.append((vi, vj, path_i, path_j, boundary_path))
    
    logger.info(f"Secondary cuts: built {len(edge_membrane_data)} membranes to check")
    logger.info(f"Secondary cuts: min_membrane_thickness={min_membrane_thickness:.4f}")
    
    if len(edge_membrane_data) == 0:
        return []
    
    # Check if membrane cuts through seed TRIANGLES (faces representing the part)
    # This uses tetrahedral mesh resolution and checks against actual surface area
    secondary_cuts = _find_secondary_cuts_by_triangle_intersection(
        edge_membrane_data, 
        seed_triangles,
        seed_triangle_positions,
        vertices, 
        boundary_verts,
        sensitivity,
        min_membrane_thickness,
        use_gpu=use_gpu
    )
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Secondary cuts complete in {elapsed:.0f}ms: {len(secondary_cuts)} secondary cutting edges found")
    
    return secondary_cuts


def _find_secondary_cuts_by_triangle_intersection(
    edge_membrane_data: List[Tuple],
    seed_triangles: List[Tuple[int, int, int]],
    seed_triangle_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    sensitivity: float,
    min_membrane_thickness: float = 0.0,
    use_gpu: bool = True
) -> List[Tuple[int, int]]:
    """
    Find secondary cuts by checking if membranes intersect seed TRIANGLES (faces).
    
    This approach checks if the membrane surface cuts through the triangular faces
    formed by seed vertices, which represent the part surface in tetrahedral mesh
    resolution. This is more robust than checking individual edges.
    
    Args:
        min_membrane_thickness: Skip membranes thinner than this (avoids false positives
                               on flat surfaces where escape paths are nearly identical)
    
    Uses GPU acceleration if available for massive speedup.
    """
    n_triangles = len(seed_triangles)
    n_membranes = len(edge_membrane_data)
    logger.info(f"Secondary cuts: checking {n_membranes} membranes against {n_triangles} seed triangles")
    
    if n_triangles == 0:
        logger.warning("No seed triangles found - cannot check for secondary cuts")
        return []
    
    # Check GPU availability
    gpu_available = use_gpu and CUDA_AVAILABLE and TORCH_AVAILABLE
    
    if gpu_available:
        logger.info("Secondary cuts: Using GPU acceleration")
        return _find_secondary_cuts_triangle_gpu(
            edge_membrane_data, seed_triangles, seed_triangle_positions,
            vertices, boundary_verts, sensitivity, min_membrane_thickness
        )
    else:
        logger.info("Secondary cuts: Using CPU")
        return _find_secondary_cuts_triangle_cpu(
            edge_membrane_data, seed_triangles, seed_triangle_positions,
            vertices, boundary_verts, sensitivity, min_membrane_thickness
        )


def _find_secondary_cuts_triangle_cpu(
    edge_membrane_data: List[Tuple],
    seed_triangles: List[Tuple[int, int, int]],
    seed_triangle_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    sensitivity: float,
    min_membrane_thickness: float = 0.0
) -> List[Tuple[int, int]]:
    """CPU implementation of secondary cuts by triangle-triangle intersection."""
    secondary_cuts = []
    
    for vi, vj, path_i, path_j, boundary_path in edge_membrane_data:
        # Build membrane triangles (skip if too thin - flat surface case)
        membrane_triangles = _build_membrane_triangles(
            path_i, path_j, boundary_path, vertices, boundary_verts,
            min_thickness=min_membrane_thickness
        )
        
        if len(membrane_triangles) == 0:
            continue
        
        # Skip triangles that share vertices with the test edge
        skip_vertices = {vi, vj}
        for v in path_i[:min(3, len(path_i))]:
            skip_vertices.add(v)
        for v in path_j[:min(3, len(path_j))]:
            skip_vertices.add(v)
        
        # Check if any membrane triangle intersects any seed triangle
        has_intersection = False
        for tri_idx, (tv0, tv1, tv2) in enumerate(seed_triangles):
            # Skip triangles that share vertices with the membrane edges
            if tv0 in skip_vertices or tv1 in skip_vertices or tv2 in skip_vertices:
                continue
            
            seed_tri_pos = seed_triangle_positions[tri_idx]
            
            for mem_tri in membrane_triangles:
                if _triangles_intersect(mem_tri, seed_tri_pos, sensitivity):
                    has_intersection = True
                    break
            
            if has_intersection:
                break
        
        if has_intersection:
            secondary_cuts.append((vi, vj))
    
    return secondary_cuts


def _find_secondary_cuts_triangle_gpu(
    edge_membrane_data: List[Tuple],
    seed_triangles: List[Tuple[int, int, int]],
    seed_triangle_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    sensitivity: float,
    min_membrane_thickness: float = 0.0
) -> List[Tuple[int, int]]:
    """
    GPU-accelerated secondary cuts using triangle-triangle intersection.
    
    For each membrane, we check if any of its triangles intersect any seed triangle.
    We use a two-step approach:
    1. Check if membrane edges intersect seed triangles
    2. Check if seed triangle edges intersect membrane triangles
    """
    import torch
    device = torch.device('cuda')
    
    # Pre-compute all seed triangle data on GPU
    n_seed_tris = len(seed_triangles)
    seed_tri_v0 = np.array([pos[0] for pos in seed_triangle_positions], dtype=np.float32)
    seed_tri_v1 = np.array([pos[1] for pos in seed_triangle_positions], dtype=np.float32)
    seed_tri_v2 = np.array([pos[2] for pos in seed_triangle_positions], dtype=np.float32)
    
    seed_v0_gpu = torch.tensor(seed_tri_v0, device=device)  # (S, 3)
    seed_v1_gpu = torch.tensor(seed_tri_v1, device=device)  # (S, 3)
    seed_v2_gpu = torch.tensor(seed_tri_v2, device=device)  # (S, 3)
    
    # Pre-compute seed triangle edges for intersection tests
    # Each triangle has 3 edges
    seed_edge_p0 = np.concatenate([seed_tri_v0, seed_tri_v1, seed_tri_v2], axis=0)  # (3S, 3)
    seed_edge_p1 = np.concatenate([seed_tri_v1, seed_tri_v2, seed_tri_v0], axis=0)  # (3S, 3)
    seed_edge_p0_gpu = torch.tensor(seed_edge_p0, device=device)
    seed_edge_p1_gpu = torch.tensor(seed_edge_p1, device=device)
    
    # Map from edge index to triangle index
    edge_to_tri_idx = np.concatenate([
        np.arange(n_seed_tris),
        np.arange(n_seed_tris),
        np.arange(n_seed_tris)
    ])
    
    secondary_cuts = []
    
    for membrane_idx, (vi, vj, path_i, path_j, boundary_path) in enumerate(edge_membrane_data):
        # Build membrane triangles (skip if too thin - flat surface case)
        membrane_triangles = _build_membrane_triangles(
            path_i, path_j, boundary_path, vertices, boundary_verts,
            min_thickness=min_membrane_thickness
        )
        
        if len(membrane_triangles) == 0:
            continue
        
        # Skip triangles that share vertices with the test edge
        skip_vertices = {vi, vj}
        for v in path_i[:min(3, len(path_i))]:
            skip_vertices.add(v)
        for v in path_j[:min(3, len(path_j))]:
            skip_vertices.add(v)
        
        # Build skip mask for seed triangles
        skip_mask = torch.zeros(n_seed_tris, dtype=torch.bool, device=device)
        for tri_idx, (tv0, tv1, tv2) in enumerate(seed_triangles):
            if tv0 in skip_vertices or tv1 in skip_vertices or tv2 in skip_vertices:
                skip_mask[tri_idx] = True
        
        # Expand skip mask to edges (3 edges per triangle)
        skip_edge_mask = skip_mask.repeat(3)  # (3S,)
        
        # Convert membrane triangles to GPU tensors
        mem_tri_v0 = np.array([t[0] for t in membrane_triangles], dtype=np.float32)
        mem_tri_v1 = np.array([t[1] for t in membrane_triangles], dtype=np.float32)
        mem_tri_v2 = np.array([t[2] for t in membrane_triangles], dtype=np.float32)
        
        mem_v0_gpu = torch.tensor(mem_tri_v0, device=device)  # (M, 3)
        mem_v1_gpu = torch.tensor(mem_tri_v1, device=device)  # (M, 3)
        mem_v2_gpu = torch.tensor(mem_tri_v2, device=device)  # (M, 3)
        
        # Membrane edges
        mem_edge_p0 = np.concatenate([mem_tri_v0, mem_tri_v1, mem_tri_v2], axis=0)  # (3M, 3)
        mem_edge_p1 = np.concatenate([mem_tri_v1, mem_tri_v2, mem_tri_v0], axis=0)  # (3M, 3)
        mem_edge_p0_gpu = torch.tensor(mem_edge_p0, device=device)
        mem_edge_p1_gpu = torch.tensor(mem_edge_p1, device=device)
        
        # Test 1: Do membrane edges intersect seed triangles?
        intersects_1 = _batch_segment_triangle_intersection_gpu(
            mem_edge_p0_gpu, mem_edge_p1_gpu,
            seed_v0_gpu, seed_v1_gpu, seed_v2_gpu,
            skip_mask, sensitivity
        )
        
        # Test 2: Do seed triangle edges intersect membrane triangles?
        # (No skip mask for membrane triangles - we check all)
        no_skip = torch.zeros(len(mem_edge_p0_gpu), dtype=torch.bool, device=device)
        intersects_2 = _batch_segment_triangle_intersection_gpu(
            seed_edge_p0_gpu, seed_edge_p1_gpu,
            mem_v0_gpu, mem_v1_gpu, mem_v2_gpu,
            skip_edge_mask, sensitivity
        )
        
        if intersects_1 or intersects_2:
            secondary_cuts.append((vi, vj))
        
        # Periodic cleanup
        if membrane_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    return secondary_cuts


def _triangles_intersect(
    tri1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tri2: Tuple[np.ndarray, np.ndarray, np.ndarray],
    sensitivity: float
) -> bool:
    """
    Check if two triangles intersect.
    
    Two triangles intersect if:
    1. Any edge of tri1 passes through tri2, OR
    2. Any edge of tri2 passes through tri1
    """
    v0, v1, v2 = tri1
    u0, u1, u2 = tri2
    
    # Check edges of tri1 against tri2
    if _segment_intersects_triangle(v0, v1, u0, u1, u2, sensitivity):
        return True
    if _segment_intersects_triangle(v1, v2, u0, u1, u2, sensitivity):
        return True
    if _segment_intersects_triangle(v2, v0, u0, u1, u2, sensitivity):
        return True
    
    # Check edges of tri2 against tri1
    if _segment_intersects_triangle(u0, u1, v0, v1, v2, sensitivity):
        return True
    if _segment_intersects_triangle(u1, u2, v0, v1, v2, sensitivity):
        return True
    if _segment_intersects_triangle(u2, u0, v0, v1, v2, sensitivity):
        return True
    
    return False


def _batch_segment_triangle_intersection_gpu(
    seg_p0: 'torch.Tensor',  # (E, 3) segment start points
    seg_p1: 'torch.Tensor',  # (E, 3) segment end points
    tri_v0: 'torch.Tensor',  # (T, 3) triangle vertex 0
    tri_v1: 'torch.Tensor',  # (T, 3) triangle vertex 1
    tri_v2: 'torch.Tensor',  # (T, 3) triangle vertex 2
    skip_mask: 'torch.Tensor',  # (T,) or (E,) bool - True to skip
    sensitivity: float
) -> bool:
    """
    GPU-batched Möller–Trumbore intersection test.
    
    Tests all segments against all triangles in parallel.
    Returns True if ANY non-skipped segment intersects ANY triangle.
    """
    import torch
    
    EPSILON = 1e-7
    
    n_segs = seg_p0.shape[0]
    n_tris = tri_v0.shape[0]
    
    if n_segs == 0 or n_tris == 0:
        return False
    
    # Expand dimensions for broadcasting: (E, 1, 3) x (1, T, 3)
    p0 = seg_p0.unsqueeze(1)  # (E, 1, 3)
    p1 = seg_p1.unsqueeze(1)  # (E, 1, 3)
    v0 = tri_v0.unsqueeze(0)  # (1, T, 3)
    v1 = tri_v1.unsqueeze(0)  # (1, T, 3)
    v2 = tri_v2.unsqueeze(0)  # (1, T, 3)
    
    # Segment direction and length
    direction = p1 - p0  # (E, 1, 3)
    seg_length = torch.norm(direction, dim=-1, keepdim=True)  # (E, 1, 1)
    seg_length = seg_length.clamp(min=EPSILON)
    direction = direction / seg_length  # Normalized (E, 1, 3)
    seg_length = seg_length.squeeze(-1)  # (E, 1)
    
    # Triangle edges
    edge1 = v1 - v0  # (1, T, 3)
    edge2 = v2 - v0  # (1, T, 3)
    
    # Cross product: h = direction × edge2
    h = torch.cross(direction.expand(-1, n_tris, -1), edge2.expand(n_segs, -1, -1), dim=-1)  # (E, T, 3)
    
    # Determinant: a = edge1 · h
    a = (edge1.expand(n_segs, -1, -1) * h).sum(dim=-1)  # (E, T)
    
    # Check for parallel rays (|a| < epsilon)
    parallel_mask = torch.abs(a) < EPSILON  # (E, T)
    
    # Avoid division by zero
    a_safe = torch.where(parallel_mask, torch.ones_like(a), a)
    f = 1.0 / a_safe  # (E, T)
    
    # s = p0 - v0
    s = p0 - v0  # (E, T, 3)
    
    # u = f * (s · h)
    u = f * (s * h).sum(dim=-1)  # (E, T)
    
    # Check u bounds
    u_valid = (u >= -EPSILON) & (u <= 1.0 + EPSILON)  # (E, T)
    
    # q = s × edge1
    q = torch.cross(s, edge1.expand(n_segs, -1, -1), dim=-1)  # (E, T, 3)
    
    # v = f * (direction · q)
    v = f * (direction.expand(-1, n_tris, -1) * q).sum(dim=-1)  # (E, T)
    
    # Check v bounds and u+v bounds
    v_valid = (v >= -EPSILON) & ((u + v) <= 1.0 + EPSILON)  # (E, T)
    
    # t = f * (edge2 · q) - distance along ray to intersection
    t = f * (edge2.expand(n_segs, -1, -1) * q).sum(dim=-1)  # (E, T)
    
    # Tolerance based on sensitivity
    # sensitivity < 1.0: positive tolerance (more lenient, fewer detections)
    # sensitivity = 1.0: zero tolerance (exact intersection)
    # sensitivity > 1.0: negative tolerance (expanded detection zone, more detections)
    tolerance = seg_length * 0.05 * (1.0 - sensitivity)  # (E, 1)
    
    # Check if t is within segment bounds
    t_valid = (t > -tolerance) & (t < seg_length + tolerance)  # (E, T)
    
    # Combine all conditions
    intersects = ~parallel_mask & u_valid & v_valid & t_valid  # (E, T)
    
    # Apply skip mask - handle both (T,) and (E,) shapes
    if skip_mask.shape[0] == n_tris:
        # Skip mask is for triangles - broadcast across edges
        intersects = intersects & ~skip_mask.unsqueeze(0)  # (E, T)
    elif skip_mask.shape[0] == n_segs:
        # Skip mask is for edges - broadcast across triangles
        intersects = intersects & ~skip_mask.unsqueeze(1)  # (E, T)
    
    # Return True if any intersection
    return intersects.any().item()


def _build_membrane_triangles(
    path_i: List[int],
    path_j: List[int],
    boundary_path: List[int],
    tet_vertices: np.ndarray,
    boundary_vertices: np.ndarray,
    min_thickness: float = 0.0
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build triangulated membrane surface from escape paths.
    
    Returns empty list if the membrane is too thin (paths nearly identical),
    which happens on flat surfaces where adjacent vertices escape the same way.
    """
    path_i_positions = tet_vertices[path_i]
    path_j_positions = tet_vertices[path_j]
    
    n_i = len(path_i_positions)
    n_j = len(path_j_positions)
    
    if n_i < 1 or n_j < 1:
        return []
    
    # Check if paths are too similar (flat surface case)
    # Compare positions along the paths - if they're nearly identical, skip
    if min_thickness > 0:
        max_separation = 0.0
        n_check = min(n_i, n_j, 5)
        for k in range(n_check):
            t = k / max(1, n_check - 1)
            idx_i = min(int(t * (n_i - 1)), n_i - 1)
            idx_j = min(int(t * (n_j - 1)), n_j - 1)
            sep = np.linalg.norm(path_i_positions[idx_i] - path_j_positions[idx_j])
            max_separation = max(max_separation, sep)
        
        if max_separation < min_thickness:
            # Paths are nearly identical - membrane is too thin
            return []
    
    membrane_triangles = []
    
    # Resample paths to have same number of points
    n_samples = max(n_i, n_j, 5)
    
    path_i_resampled = []
    path_j_resampled = []
    
    for s in range(n_samples):
        t = s / max(1, n_samples - 1)
        
        # Interpolate along path_i
        idx_i = min(int(t * (n_i - 1)), n_i - 1)
        frac_i = t * (n_i - 1) - idx_i
        if idx_i < n_i - 1:
            pos_i = path_i_positions[idx_i] * (1 - frac_i) + path_i_positions[idx_i + 1] * frac_i
        else:
            pos_i = path_i_positions[idx_i]
        path_i_resampled.append(pos_i)
        
        # Interpolate along path_j
        idx_j = min(int(t * (n_j - 1)), n_j - 1)
        frac_j = t * (n_j - 1) - idx_j
        if idx_j < n_j - 1:
            pos_j = path_j_positions[idx_j] * (1 - frac_j) + path_j_positions[idx_j + 1] * frac_j
        else:
            pos_j = path_j_positions[idx_j]
        path_j_resampled.append(pos_j)
    
    # Create triangles between the two paths
    for s in range(n_samples - 1):
        p0 = path_i_resampled[s]
        p1 = path_i_resampled[s + 1]
        p2 = path_j_resampled[s]
        p3 = path_j_resampled[s + 1]
        
        membrane_triangles.append((p0, p1, p2))
        membrane_triangles.append((p1, p3, p2))
    
    # Add boundary path lid
    if len(boundary_path) > 1:
        boundary_path_positions = boundary_vertices[boundary_path]
        wi_pos = path_i_resampled[-1]
        wj_pos = path_j_resampled[-1]
        mid = (wi_pos + wj_pos) / 2
        
        for i in range(len(boundary_path_positions) - 1):
            bp0 = boundary_path_positions[i]
            bp1 = boundary_path_positions[i + 1]
            membrane_triangles.append((bp0, bp1, mid))
    
    return membrane_triangles


def _segment_intersects_triangle(
    p0: np.ndarray, p1: np.ndarray,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    sensitivity: float = 0.5
) -> bool:
    """
    Check if line segment (p0, p1) intersects triangle (v0, v1, v2).
    
    Uses Möller–Trumbore intersection algorithm.
    """
    EPSILON = 1e-9
    
    # Direction of segment
    direction = p1 - p0
    seg_length = np.linalg.norm(direction)
    if seg_length < EPSILON:
        return False
    direction = direction / seg_length
    
    # Triangle edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Begin calculating determinant
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < EPSILON:
        # Ray is parallel to triangle
        return False
    
    f = 1.0 / a
    s = p0 - v0
    u = f * np.dot(s, h)
    
    if u < -EPSILON or u > 1.0 + EPSILON:
        return False
    
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    
    if v < -EPSILON or u + v > 1.0 + EPSILON:
        return False
    
    # Compute t to find intersection point
    t = f * np.dot(edge2, q)
    
    # Check if intersection is within the segment (0 <= t <= seg_length)
    # Tolerance based on sensitivity:
    # sensitivity < 1.0: positive tolerance (more lenient, fewer detections)
    # sensitivity = 1.0: zero tolerance (exact intersection)
    # sensitivity > 1.0: negative tolerance (expanded detection, more detections)
    tolerance = seg_length * 0.05 * (1.0 - sensitivity)
    
    if t > -tolerance and t < seg_length + tolerance:
        return True
    
    return False
    
    # Create triangles between the two paths
    for s in range(n_samples - 1):
        p0 = path_i_resampled[s]
        p1 = path_i_resampled[s + 1]
        p2 = path_j_resampled[s]
        p3 = path_j_resampled[s + 1]
        
        membrane_triangles.append((p0, p1, p2))
        membrane_triangles.append((p1, p3, p2))
    
    # Add boundary path lid
    if len(boundary_path) > 1:
        boundary_path_positions = boundary_vertices[boundary_path]
        wi_pos = path_i_resampled[-1]
        wj_pos = path_j_resampled[-1]
        mid = (wi_pos + wj_pos) / 2
        
        for i in range(len(boundary_path_positions) - 1):
            bp0 = boundary_path_positions[i]
            bp1 = boundary_path_positions[i + 1]
            membrane_triangles.append((bp0, bp1, mid))
    
    return membrane_triangles
    
    for vi, vj, path_i, path_j, boundary_path in edge_membrane_data:
        # The edge being tested - we skip intersection tests with this edge and its adjacent edges
        test_edge = (min(vi, vj), max(vi, vj))
        
        # Check if membrane from (vi, vj) intersects any OTHER seed edge
        if _membrane_intersects_seed_edges(
            vi, vj, path_i, path_j, boundary_path,
            vertices, boundary_verts,
            all_seed_edges, all_seed_edge_positions,
            test_edge, sensitivity
        ):
            secondary_cuts.append((vi, vj))
    
    return secondary_cuts


def _build_boundary_adjacency(boundary_mesh: trimesh.Trimesh) -> Dict[int, List[Tuple[int, float]]]:
    """Build adjacency list for boundary mesh with edge lengths."""
    n_verts = len(boundary_mesh.vertices)
    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_verts)}
    
    vertices = boundary_mesh.vertices
    
    # Extract edges from faces
    edges_seen = set()
    for face in boundary_mesh.faces:
        for i in range(3):
            v0, v1 = face[i], face[(i + 1) % 3]
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edges_seen:
                edges_seen.add(edge_key)
                length = np.linalg.norm(vertices[v1] - vertices[v0])
                adjacency[v0].append((v1, length))
                adjacency[v1].append((v0, length))
    
    return adjacency


def _find_nearest_boundary_vertex(point: np.ndarray, boundary_verts: np.ndarray) -> int:
    """Find nearest vertex in boundary mesh to a given point."""
    distances = np.linalg.norm(boundary_verts - point, axis=1)
    return int(np.argmin(distances))


def _shortest_path_on_boundary(
    start: int, 
    end: int, 
    boundary_mesh: trimesh.Trimesh,
    adjacency: Dict[int, List[Tuple[int, float]]]
) -> List[int]:
    """
    Compute shortest path between two vertices on the boundary mesh using Dijkstra.
    
    Returns list of boundary mesh vertex indices from start to end.
    """
    import heapq
    
    if start == end:
        return [start]
    
    n_verts = len(boundary_mesh.vertices)
    dist = np.full(n_verts, np.inf, dtype=np.float64)
    dist[start] = 0.0
    predecessor = np.full(n_verts, -1, dtype=np.int64)
    
    pq = [(0.0, start)]
    visited = np.zeros(n_verts, dtype=bool)
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if visited[u]:
            continue
        visited[u] = True
        
        if u == end:
            break
        
        for v, cost in adjacency.get(u, []):
            if visited[v]:
                continue
            new_dist = d + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                predecessor[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    # Reconstruct path
    if predecessor[end] < 0 and start != end:
        return []  # No path found
    
    path = []
    current = end
    while current >= 0:
        path.append(current)
        current = predecessor[current]
    path.reverse()
    
    return path


def _membrane_intersects_part_v2(
    vi: int, vj: int,
    path_i: List[int], path_j: List[int],
    boundary_path: List[int],
    tet_vertices: np.ndarray,
    boundary_vertices: np.ndarray,
    part_mesh: trimesh.Trimesh,
    distance_tolerance: float,
    sensitivity: float
) -> bool:
    """
    Check if the membrane surface bounded by the given paths intersects the part mesh.
    
    The membrane is bounded by 4 edges (all paths go OUTWARD from seeds):
    1. Edge (vi, vj) - on the part surface (inner boundary)
    2. Path vi → wi - escape path from seed vi outward to boundary point wi
    3. Path vj → wj - escape path from seed vj outward to boundary point wj
    4. Path wi → wj - shortest path on outer boundary ∂H
    
    Visually:
    
        wi -----(boundary path)---- wj
        ↑                            ↑
     path_i                       path_j
     (outward)                   (outward)
        |                            |
        vi -----(part edge)------- vj
    
    If this membrane passes THROUGH the part mesh, the mold cannot separate
    without cutting through the part at this edge → secondary cut needed.
    """
    # Get positions of escape path vertices (paths contain tet mesh vertex indices)
    # path_i: [vi, v1, v2, ..., wi] - from seed OUTWARD to boundary
    # path_j: [vj, v1, v2, ..., wj] - from seed OUTWARD to boundary
    path_i_positions = tet_vertices[path_i]
    path_j_positions = tet_vertices[path_j]
    
    # Get boundary path positions (indices into boundary mesh)
    # boundary_path: [wi_idx, ..., wj_idx] on the outer boundary mesh
    if len(boundary_path) > 0:
        boundary_path_positions = boundary_vertices[boundary_path]
    else:
        boundary_path_positions = np.empty((0, 3))
    
    # The membrane is a quadrilateral-like surface with 4 sides:
    # Side 1: vi → vj (part surface edge) - implicit, connects the two seeds
    # Side 2: vi → wi (path_i going outward)
    # Side 3: wi → wj (boundary path)
    # Side 4: wj → vj (NOT the reverse of path_j, but the boundary path connects to vj via path_j end)
    
    # Sample points across the membrane surface
    # The key sampling is between the two OUTWARD paths (path_i and path_j)
    sample_points = []
    
    n_i = len(path_i_positions)
    n_j = len(path_j_positions)
    
    if n_i < 1 or n_j < 1:
        return False
    
    # Sample across the "ribbon" between the two outward escape paths
    # At each depth level t (0=seeds, 1=boundary), sample across from path_i to path_j
    n_depth_samples = max(n_i, n_j, 5)
    
    for s in range(n_depth_samples):
        t = s / max(1, n_depth_samples - 1)
        
        # Get position along path_i at parameter t (0=vi, 1=wi)
        idx_i = min(int(t * (n_i - 1)), n_i - 1)
        pos_i = path_i_positions[idx_i]
        
        # Get position along path_j at parameter t (0=vj, 1=wj)
        idx_j = min(int(t * (n_j - 1)), n_j - 1)
        pos_j = path_j_positions[idx_j]
        
        # Sample across the ribbon at this depth (between the two paths)
        for u in [0.2, 0.35, 0.5, 0.65, 0.8]:
            sample = pos_i * (1 - u) + pos_j * u
            sample_points.append(sample)
    
    # Also sample the "top" of the membrane (boundary path wi → wj)
    if len(boundary_path_positions) > 1:
        wi_pos = path_i_positions[-1] if n_i > 0 else None
        wj_pos = path_j_positions[-1] if n_j > 0 else None
        
        if wi_pos is not None and wj_pos is not None:
            # Sample along the boundary path and slightly inward
            for i in range(len(boundary_path_positions)):
                bp = boundary_path_positions[i]
                # Sample points between boundary and the membrane interior
                # (toward the midpoint of the escape paths)
                mid_depth_i = path_i_positions[n_i // 2] if n_i > 1 else path_i_positions[0]
                mid_depth_j = path_j_positions[n_j // 2] if n_j > 1 else path_j_positions[0]
                interior_target = (mid_depth_i + mid_depth_j) / 2
                
                for t in [0.2, 0.4]:
                    sample = bp * (1 - t) + interior_target * t
                    sample_points.append(sample)
    
    if len(sample_points) == 0:
        return False
    
    sample_points = np.array(sample_points)
    
    # Filter out points that are ON the part surface (seeds are on the surface)
    # We only want to check interior membrane points
    try:
        _, distances_to_surface, _ = part_mesh.nearest.on_surface(sample_points)
        
        # Only keep points that are NOT on the surface
        surface_threshold = distance_tolerance * 0.1
        interior_mask = distances_to_surface > surface_threshold
        
        if not np.any(interior_mask):
            # All samples are on the surface - no interior to check
            return False
        
        interior_samples = sample_points[interior_mask]
        
        # Check if any interior sample point is INSIDE the part mesh
        inside = part_mesh.contains(interior_samples)
        n_inside = np.sum(inside)
        
        if n_inside > 0:
            logger.debug(f"Edge ({vi}, {vj}): {n_inside}/{len(interior_samples)} membrane samples inside part")
            return True
        
        # With higher sensitivity, check for samples close to surface
        if sensitivity > 0.3:
            interior_distances = distances_to_surface[interior_mask]
            n_close = np.sum(interior_distances < distance_tolerance)
            threshold_fraction = 0.1 * (1.0 - sensitivity)
            if n_close > len(interior_distances) * threshold_fraction:
                logger.debug(f"Edge ({vi}, {vj}): {n_close}/{len(interior_distances)} samples within tolerance")
                return True
                
                
    except Exception as e:
        logger.warning(f"Error checking membrane intersection: {e}")
    
    return False
