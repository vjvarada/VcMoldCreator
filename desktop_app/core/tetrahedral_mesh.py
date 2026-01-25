"""
Tetrahedral Mesh Generation for Mold Volume

This module generates a tetrahedral mesh of the mold bounding volume using fTetWild
via the pytetwild package.

The tetrahedral mesh replaces the voxel grid approach for:
- More accurate volume representation
- Better boundary conforming
- Edge-based weight assignment for parting surface computation

Algorithm:
1. Tetrahedralize the complete bounding hull mesh
2. Classify tetrahedra as inside or outside the part mesh (using centroid test)
3. Filter to keep only tetrahedra in the mold cavity (outside the part)
4. Build edge connectivity (walk graph) from the filtered tetrahedra
5. Compute distances from tet vertices/edges to part surface and shell boundary

The walk graph (edges from outside-part tetrahedra) is used for Dijkstra-based 
escape labeling to find the parting surface.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import trimesh

# Import pytetwild for tetrahedralization
try:
    import pytetwild
    PYTETWILD_AVAILABLE = True
except ImportError:
    PYTETWILD_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log pytetwild availability
if PYTETWILD_AVAILABLE:
    logger.info("pytetwild available for tetrahedralization")
else:
    logger.warning("pytetwild not available - tetrahedralization will fail. Install with: pip install pytetwild")

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
    
    # Full vertex mold labels (N,) - final H1(1) or H2(2) classification for ALL vertices
    # After Dijkstra + propagation, every vertex should have label 1 or 2
    # Used by marching tetrahedra for parting surface extraction
    vertex_mold_labels: Optional[np.ndarray] = None
    
    # Primary cutting edges - edges where one vertex escapes to H1 and the other to H2
    # These are the edges the parting surface passes through (shown in yellow)
    # List of (vi, vj) tuples representing primary cut edges
    primary_cut_edges: Optional[List[Tuple[int, int]]] = None
    
    # Secondary cutting edges - edges between same-label seeds where membrane intersects part
    # List of (vi, vj) tuples representing secondary cut edges
    secondary_cut_edges: Optional[List[Tuple[int, int]]] = None
    
    # =========================================================================
    # CACHED DATA FOR SECONDARY CUT DETECTION (computed once, reused)
    # =========================================================================
    
    # Boundary mesh adjacency list: Dict[vertex_idx, List[(neighbor_idx, edge_length)]]
    # Built from boundary_mesh edges, used for shortest path computation on ∂H
    # Caching this avoids rebuilding it each time secondary cuts are computed
    boundary_adjacency: Optional[Dict[int, List[Tuple[int, float]]]] = None
    
    # Part mesh vertex to tet mesh vertex mapping (computed via KDTree)
    # part_to_tet_vertex[part_mesh_vertex_idx] = closest_tet_vertex_idx
    # Caching this avoids expensive KDTree queries during secondary cut detection
    part_to_tet_vertex: Optional[np.ndarray] = None
    
    # Seed triangles for secondary cut detection
    # List of (tet_v0, tet_v1, tet_v2) representing part surface in tet mesh resolution
    # Only includes triangles where all 3 vertices are on part surface (boundary_labels == -1)
    cached_seed_triangles: Optional[List[Tuple[int, int, int]]] = None
    cached_seed_triangle_positions: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
    
    # Cached boundary paths for secondary cut detection
    # Dict[(min_vertex, max_vertex), path] - shortest paths on boundary mesh ∂H
    # Persisted across secondary cut detection runs to avoid redundant Dijkstra
    cached_boundary_paths: Optional[Dict[Tuple[int, int], List[int]]] = None
    
    # =========================================================================
    # PARTING SURFACE EXTRACTION DATA
    # =========================================================================
    
    # Cut edge flags - True if edge is cut (primary OR secondary)
    # Indexed by global edge ID (same ordering as self.edges)
    # cut_edge_flags[e] = 1 means edge e is cut by parting/secondary surface
    cut_edge_flags: Optional[np.ndarray] = None  # (E,) bool or int8
    
    # Edge index lookup - maps (min(vi,vj), max(vi,vj)) to global edge index
    # Used for fast edge-to-index lookup during marching tets
    edge_to_index: Optional[Dict[Tuple[int, int], int]] = None
    
    # Tet-to-edge mapping - for each tet, its 6 global edge indices
    # tet_edge_indices[t, :] = [e0, e1, e2, e3, e4, e5] for tet t
    # Edge order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    tet_edge_indices: Optional[np.ndarray] = None  # (M, 6) int64
    
    # Timing
    tetrahedralize_time_ms: float = 0.0
    total_time_ms: float = 0.0


def tetrahedralize_mesh(
    surface_mesh: trimesh.Trimesh,
    edge_length_fac: float = 0.05,
    optimize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tetrahedral mesh from a surface mesh using fTetWild.
    
    fTetWild is extremely robust - handles non-manifold, self-intersecting,
    and defective meshes gracefully.
    
    Args:
        surface_mesh: The surface mesh to tetrahedralize (e.g., hull mesh)
        edge_length_fac: Target edge length as fraction of bounding box diagonal
                        Default 0.05 = bbox/20. Smaller = finer mesh.
        optimize: Whether to optimize mesh quality (slower but better quality)
    
    Returns:
        Tuple of (vertices, tetrahedra)
        - vertices: (N, 3) float64 array of vertex positions
        - tetrahedra: (M, 4) int32 array of tetrahedron vertex indices
    """
    import time
    start = time.time()
    
    # Get vertices and faces from trimesh
    vertices = np.asarray(surface_mesh.vertices, dtype=np.float64)
    faces = np.asarray(surface_mesh.faces, dtype=np.int32)
    
    logger.info(f"Tetrahedralizing mesh with {len(vertices)} vertices, {len(faces)} faces")
    logger.info(f"Parameters: edge_length_fac={edge_length_fac}, optimize={optimize}")
    
    if not PYTETWILD_AVAILABLE:
        raise ImportError(
            "pytetwild not available. Install with: pip install pytetwild"
        )
    
    # Run fTetWild via pytetwild
    tet_vertices, tetrahedra = pytetwild.tetrahedralize(
        vertices, 
        faces,
        optimize=optimize,
        edge_length_fac=edge_length_fac
    )
    
    elapsed = (time.time() - start) * 1000
    logger.info(f"Generated {len(tet_vertices)} vertices, {len(tetrahedra)} tetrahedra in {elapsed:.0f}ms")
    
    return tet_vertices, tetrahedra


def classify_tetrahedra(
    tet_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    part_mesh: trimesh.Trimesh,
    tolerance: float = 0.0
) -> np.ndarray:
    """
    Classify each tetrahedron as inside or outside the part mesh.
    
    Uses the centroid of each tetrahedron to determine if it's inside the part.
    Tetrahedra in the mold cavity (outside the part) are marked as True,
    tetrahedra inside the part are marked as False.
    
    Args:
        tet_vertices: (N, 3) tetrahedral mesh vertices
        tetrahedra: (M, 4) tetrahedron vertex indices
        part_mesh: The original part mesh to check against
        tolerance: Small positive value - centroids within this distance
                   inside the part are also considered inside (default 0.0)
    
    Returns:
        (M,) boolean array - True = outside part (mold cavity), False = inside part
    """
    logger.info(f"Classifying {len(tetrahedra)} tetrahedra...")
    
    # Compute tetrahedra centroids
    centroids = tet_vertices[tetrahedra].mean(axis=1)  # (M, 3)
    
    # Check which centroids are inside the part mesh
    # trimesh.contains returns True for points inside the mesh
    inside_part = part_mesh.contains(centroids)
    
    # If tolerance > 0, also check distance to surface for points just outside
    if tolerance > 0:
        _, distances, _ = part_mesh.nearest.on_surface(centroids)
        inside_part = inside_part & (distances > tolerance)
    
    # Outside part = mold cavity
    outside_part = ~inside_part
    
    n_inside = np.sum(inside_part)
    n_outside = np.sum(outside_part)
    logger.info(f"Classification: {n_inside} tetrahedra inside part, {n_outside} outside part (mold cavity)")
    
    return outside_part


def filter_tetrahedra_outside_part(
    tet_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    part_mesh: trimesh.Trimesh,
    tolerance: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out tetrahedra that are inside the part mesh.
    
    Returns only the tetrahedra in the mold cavity (outside the part),
    which form the walk graph for the mold half algorithm.
    
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
    logger.info(f"Filtering tetrahedra to build walk graph from mold cavity (tolerance={tolerance})...")
    
    # Use classify_tetrahedra to get the outside/inside mask
    outside_mask = classify_tetrahedra(tet_vertices, tetrahedra, part_mesh, tolerance)
    
    n_outside = np.sum(outside_mask)
    n_inside = len(tetrahedra) - n_outside
    
    if n_inside == 0:
        logger.info("No tetrahedra inside part mesh - no filtering needed")
        return tet_vertices, tetrahedra
    
    # Filter to keep only tetrahedra outside the part (mold cavity)
    filtered_tetrahedra = tetrahedra[outside_mask]
    
    # Reindex vertices to remove unused ones
    used_vertices = np.unique(filtered_tetrahedra.ravel())
    vertex_remap = np.full(len(tet_vertices), -1, dtype=np.int32)
    vertex_remap[used_vertices] = np.arange(len(used_vertices))
    
    filtered_vertices = tet_vertices[used_vertices]
    filtered_tetrahedra = vertex_remap[filtered_tetrahedra]
    
    logger.info(f"Walk graph built from {len(filtered_tetrahedra)} tetrahedra ({len(filtered_vertices)} vertices)")
    
    return filtered_vertices, filtered_tetrahedra


def extract_labeled_boundary_meshes(
    tet_result: 'TetrahedralMeshResult',
    use_original: bool = True
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """
    Extract separate boundary meshes for inner (part) and outer (hull) boundaries.
    
    This is useful for membrane smoothing where we want to re-project boundary
    vertices to the TETRAHEDRAL boundary surfaces rather than the original
    input meshes. This ensures exact alignment since the membrane vertices
    come from cut points on tetrahedral edges.
    
    Boundary labels:
    - -1 = part surface (inner boundary M)
    -  0 = boundary zone (grey zone between H1 and H2 on hull, OR interior)
    -  1 = H1 (hull half 1)
    -  2 = H2 (hull half 2)
    
    For the OUTER boundary mesh, we include ALL hull faces:
    - Faces where ALL vertices are on hull (label in {0, 1, 2} AND not on part)
    - This includes H1, H2, AND the grey boundary zone between them
    
    Args:
        tet_result: TetrahedralMeshResult with boundary_labels
        use_original: If True, use non-inflated vertices (vertices_original)
                     If False, use current (possibly inflated) vertices
    
    Returns:
        Tuple of (inner_boundary_mesh, outer_boundary_mesh)
        - inner_boundary_mesh: Faces where all vertices have boundary_labels == -1 (part surface M)
        - outer_boundary_mesh: Faces on hull ∂H (H1 + H2 + boundary zone, excludes part surface)
        Either can be None if no such faces exist.
    """
    if tet_result.boundary_labels is None:
        logger.warning("No boundary_labels available - cannot extract labeled boundary meshes")
        return None, None
    
    if tet_result.boundary_mesh is None:
        logger.warning("No boundary_mesh available")
        return None, None
    
    # Choose vertex positions
    if use_original and tet_result.boundary_mesh_original is not None:
        boundary_mesh = tet_result.boundary_mesh_original
        logger.debug("Using original (non-inflated) boundary mesh")
    else:
        boundary_mesh = tet_result.boundary_mesh
        logger.debug("Using current (possibly inflated) boundary mesh")
    
    vertices = np.array(boundary_mesh.vertices, dtype=np.float64)
    faces = np.array(boundary_mesh.faces, dtype=np.int64)
    
    # Get boundary labels for boundary mesh vertices
    # Note: boundary_mesh vertices are a subset of tet_result.vertices
    # We need to map boundary mesh vertex indices to tet vertex indices
    
    # Build mapping from boundary mesh vertices to tet vertices using spatial matching
    from scipy.spatial import cKDTree
    
    if use_original and tet_result.vertices_original is not None:
        tet_verts = tet_result.vertices_original
    else:
        tet_verts = tet_result.vertices
    
    tet_tree = cKDTree(tet_verts)
    _, tet_indices = tet_tree.query(vertices, k=1)
    
    # Get boundary labels for each boundary mesh vertex
    vertex_labels = tet_result.boundary_labels[tet_indices]
    
    # Also need to know which vertices are actually ON the boundary surface
    # (not interior). Use boundary_vertices mask from tet_result.
    if tet_result.boundary_vertices is not None:
        is_on_boundary = tet_result.boundary_vertices[tet_indices]
    else:
        # If no boundary_vertices mask, assume all boundary_mesh vertices are on boundary
        is_on_boundary = np.ones(len(vertices), dtype=bool)
    
    # Classify faces by their vertex labels
    face_v0_labels = vertex_labels[faces[:, 0]]
    face_v1_labels = vertex_labels[faces[:, 1]]
    face_v2_labels = vertex_labels[faces[:, 2]]
    
    # Inner boundary: all 3 vertices on part surface (label == -1)
    inner_mask = (face_v0_labels == -1) & (face_v1_labels == -1) & (face_v2_labels == -1)
    
    # Outer boundary: all 3 vertices are NOT on part surface (label != -1)
    # This includes H1 (1), H2 (2), AND boundary zone (0) vertices that are on hull
    # A vertex with label 0 could be interior OR boundary zone - but since we're 
    # iterating over boundary_mesh faces, all vertices are guaranteed to be on boundary
    outer_v0 = face_v0_labels != -1  # Not on part surface
    outer_v1 = face_v1_labels != -1  # Not on part surface
    outer_v2 = face_v2_labels != -1  # Not on part surface
    outer_mask = outer_v0 & outer_v1 & outer_v2
    
    inner_mesh = None
    outer_mesh = None
    
    # Extract inner boundary mesh (part surface M)
    inner_faces = faces[inner_mask]
    if len(inner_faces) > 0:
        # Get unique vertices used by inner faces
        inner_verts_used = np.unique(inner_faces.ravel())
        inner_vert_remap = np.full(len(vertices), -1, dtype=np.int64)
        inner_vert_remap[inner_verts_used] = np.arange(len(inner_verts_used))
        
        inner_mesh = trimesh.Trimesh(
            vertices=vertices[inner_verts_used],
            faces=inner_vert_remap[inner_faces],
            process=False
        )
        inner_mesh.fix_normals()
        logger.info(f"Extracted inner boundary (part M): {len(inner_mesh.vertices)} verts, {len(inner_mesh.faces)} faces")
    else:
        logger.warning("No inner boundary faces found (label == -1)")
    
    # Extract outer boundary mesh (hull ∂H = H1 + H2 + boundary zone)
    outer_faces = faces[outer_mask]
    if len(outer_faces) > 0:
        # Get unique vertices used by outer faces
        outer_verts_used = np.unique(outer_faces.ravel())
        outer_vert_remap = np.full(len(vertices), -1, dtype=np.int64)
        outer_vert_remap[outer_verts_used] = np.arange(len(outer_verts_used))
        
        outer_mesh = trimesh.Trimesh(
            vertices=vertices[outer_verts_used],
            faces=outer_vert_remap[outer_faces],
            process=False
        )
        outer_mesh.fix_normals()
        
        # Log breakdown of outer boundary
        outer_vertex_labels = vertex_labels[outer_verts_used]
        n_h1 = np.sum(outer_vertex_labels == 1)
        n_h2 = np.sum(outer_vertex_labels == 2)
        n_zone = np.sum(outer_vertex_labels == 0)
        logger.info(f"Extracted outer boundary (hull ∂H): {len(outer_mesh.vertices)} verts "
                   f"({n_h1} H1, {n_h2} H2, {n_zone} boundary zone), {len(outer_mesh.faces)} faces")
    else:
        logger.warning("No outer boundary faces found")
    
    # Log faces that have mixed part/hull vertices (seam faces - should be rare)
    n_mixed = len(faces) - np.sum(inner_mask) - np.sum(outer_mask)
    if n_mixed > 0:
        logger.debug(f"{n_mixed} boundary faces have mixed part/hull vertices (seam faces)")
    
    return inner_mesh, outer_mesh


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
    
    # Remove any degenerate faces (zero area) - trimesh 4.x API
    nondegenerate = boundary_mesh.nondegenerate_faces()
    if nondegenerate.sum() < len(boundary_mesh.faces):
        boundary_mesh.update_faces(nondegenerate)
    
    # Remove any duplicate faces - trimesh 4.x API
    unique_mask = boundary_mesh.unique_faces()
    if unique_mask.sum() < len(boundary_mesh.faces):
        boundary_mesh.update_faces(unique_mask)
    
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


# =============================================================================
# PARTING SURFACE EXTRACTION HELPERS
# =============================================================================

def build_edge_to_index_map(edges: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Build a dictionary mapping (vi, vj) edge tuples to global edge indices.
    
    Args:
        edges: (E, 2) array of unique edges (sorted so v0 < v1)
    
    Returns:
        Dictionary mapping (min(vi, vj), max(vi, vj)) -> edge index
    """
    edge_to_index = {}
    for idx, (v0, v1) in enumerate(edges):
        # Edges are already sorted, but ensure consistency
        key = (int(min(v0, v1)), int(max(v0, v1)))
        edge_to_index[key] = idx
    return edge_to_index


def build_tet_edge_indices(
    tetrahedra: np.ndarray,
    edge_to_index: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """
    Build tet-to-edge mapping: for each tetrahedron, get its 6 global edge indices.
    
    Edge order within each tet: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    
    Args:
        tetrahedra: (M, 4) array of tetrahedron vertex indices
        edge_to_index: Dictionary mapping (vi, vj) -> global edge index
    
    Returns:
        (M, 6) array where tet_edge_indices[t, k] is the global edge index
        for the k-th edge of tetrahedron t
    """
    n_tets = len(tetrahedra)
    tet_edge_indices = np.zeros((n_tets, 6), dtype=np.int64)
    
    # Edge pairs within a tetrahedron (local vertex indices)
    edge_pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]
    
    for t in range(n_tets):
        tet = tetrahedra[t]
        for k, (i, j) in enumerate(edge_pairs):
            vi, vj = int(tet[i]), int(tet[j])
            key = (min(vi, vj), max(vi, vj))
            tet_edge_indices[t, k] = edge_to_index[key]
    
    return tet_edge_indices


def compute_cut_edge_flags(
    edges: np.ndarray,
    primary_cut_edges: Optional[List[Tuple[int, int]]],
    secondary_cut_edges: Optional[List[Tuple[int, int]]],
    edge_to_index: Optional[Dict[Tuple[int, int], int]] = None
) -> np.ndarray:
    """
    Compute boolean array marking which edges are cut (primary OR secondary).
    
    Args:
        edges: (E, 2) array of unique edges
        primary_cut_edges: List of (vi, vj) primary cut edge tuples
        secondary_cut_edges: List of (vi, vj) secondary cut edge tuples
        edge_to_index: Optional pre-computed edge-to-index map
    
    Returns:
        (E,) boolean array where cut_edge_flags[e] = True if edge e is cut
    """
    n_edges = len(edges)
    cut_flags = np.zeros(n_edges, dtype=np.int8)
    
    # Build edge-to-index map if not provided
    if edge_to_index is None:
        edge_to_index = build_edge_to_index_map(edges)
    
    # Mark primary cut edges
    if primary_cut_edges is not None:
        for vi, vj in primary_cut_edges:
            key = (min(vi, vj), max(vi, vj))
            if key in edge_to_index:
                cut_flags[edge_to_index[key]] = 1
    
    # Mark secondary cut edges
    if secondary_cut_edges is not None:
        for vi, vj in secondary_cut_edges:
            key = (min(vi, vj), max(vi, vj))
            if key in edge_to_index:
                cut_flags[edge_to_index[key]] = 1
    
    return cut_flags


def compute_primary_cut_edge_flags(
    edges: np.ndarray,
    primary_cut_edges: Optional[List[Tuple[int, int]]],
    edge_to_index: Optional[Dict[Tuple[int, int], int]] = None
) -> np.ndarray:
    """
    Compute boolean array marking which edges are PRIMARY cut edges only.
    
    Args:
        edges: (E, 2) array of unique edges
        primary_cut_edges: List of (vi, vj) primary cut edge tuples
        edge_to_index: Optional pre-computed edge-to-index map
    
    Returns:
        (E,) boolean array where flags[e] = True if edge e is a primary cut
    """
    n_edges = len(edges)
    cut_flags = np.zeros(n_edges, dtype=np.int8)
    
    if primary_cut_edges is None:
        return cut_flags
    
    # Build edge-to-index map if not provided
    if edge_to_index is None:
        edge_to_index = build_edge_to_index_map(edges)
    
    # Mark primary cut edges
    for vi, vj in primary_cut_edges:
        key = (min(vi, vj), max(vi, vj))
        if key in edge_to_index:
            cut_flags[edge_to_index[key]] = 1
    
    return cut_flags


def compute_secondary_cut_edge_flags(
    edges: np.ndarray,
    secondary_cut_edges: Optional[List[Tuple[int, int]]],
    edge_to_index: Optional[Dict[Tuple[int, int], int]] = None
) -> np.ndarray:
    """
    Compute boolean array marking which edges are SECONDARY cut edges only.
    
    Args:
        edges: (E, 2) array of unique edges
        secondary_cut_edges: List of (vi, vj) secondary cut edge tuples
        edge_to_index: Optional pre-computed edge-to-index map
    
    Returns:
        (E,) boolean array where flags[e] = True if edge e is a secondary cut
    """
    n_edges = len(edges)
    cut_flags = np.zeros(n_edges, dtype=np.int8)
    
    if secondary_cut_edges is None:
        return cut_flags
    
    # Build edge-to-index map if not provided
    if edge_to_index is None:
        edge_to_index = build_edge_to_index_map(edges)
    
    # Mark secondary cut edges
    for vi, vj in secondary_cut_edges:
        key = (min(vi, vj), max(vi, vj))
        if key in edge_to_index:
            cut_flags[edge_to_index[key]] = 1
    
    return cut_flags


def compute_extended_secondary_cut_edge_flags(
    edges: np.ndarray,
    tetrahedra: np.ndarray,
    tet_edge_indices: np.ndarray,
    primary_cut_edges: Optional[List[Tuple[int, int]]],
    secondary_cut_edges: Optional[List[Tuple[int, int]]],
    edge_to_index: Optional[Dict[Tuple[int, int], int]] = None
) -> np.ndarray:
    """
    Compute extended secondary cut edge flags that include primary edges in tets
    where both primary and secondary edges exist.
    
    This allows the secondary parting surface to extend and connect to the 
    primary parting surface where they meet.
    
    Logic:
    - Start with all secondary cut edges marked
    - For each tetrahedron that has at least one secondary edge:
      - If it also has primary edges, mark those primary edges as "cut" too
    - This makes the secondary surface extend through primary edges in shared tets
    
    Args:
        edges: (E, 2) array of unique edges
        tetrahedra: (M, 4) tetrahedron vertex indices
        tet_edge_indices: (M, 6) mapping from tet to its 6 edge indices
        primary_cut_edges: List of (vi, vj) primary cut edge tuples
        secondary_cut_edges: List of (vi, vj) secondary cut edge tuples
        edge_to_index: Optional pre-computed edge-to-index map
    
    Returns:
        (E,) boolean array - extended secondary cut flags
    """
    n_edges = len(edges)
    
    # Build edge-to-index map if not provided
    if edge_to_index is None:
        edge_to_index = build_edge_to_index_map(edges)
    
    # Start with basic secondary flags
    secondary_flags = np.zeros(n_edges, dtype=np.int8)
    if secondary_cut_edges is not None:
        for vi, vj in secondary_cut_edges:
            key = (min(vi, vj), max(vi, vj))
            if key in edge_to_index:
                secondary_flags[edge_to_index[key]] = 1
    
    # Compute primary flags
    primary_flags = np.zeros(n_edges, dtype=np.int8)
    if primary_cut_edges is not None:
        for vi, vj in primary_cut_edges:
            key = (min(vi, vj), max(vi, vj))
            if key in edge_to_index:
                primary_flags[edge_to_index[key]] = 1
    
    # If no secondary edges, nothing to extend
    if secondary_cut_edges is None or len(secondary_cut_edges) == 0:
        return secondary_flags
    
    # If no primary edges, just return secondary
    if primary_cut_edges is None or len(primary_cut_edges) == 0:
        return secondary_flags
    
    # Extended flags start as copy of secondary
    extended_flags = secondary_flags.copy()
    
    # For each tetrahedron, check if it has both primary and secondary edges
    n_extended = 0
    for t in range(len(tetrahedra)):
        tet_edges = tet_edge_indices[t]  # 6 edge indices
        
        has_secondary = False
        has_primary = False
        primary_edges_in_tet = []
        
        for local_e in range(6):
            global_e = tet_edges[local_e]
            if secondary_flags[global_e]:
                has_secondary = True
            if primary_flags[global_e]:
                has_primary = True
                primary_edges_in_tet.append(global_e)
        
        # If this tet has both, extend secondary to include primary edges
        if has_secondary and has_primary:
            for global_e in primary_edges_in_tet:
                if extended_flags[global_e] == 0:
                    extended_flags[global_e] = 1
                    n_extended += 1
    
    logger.info(f"Extended secondary cut edges: {np.sum(secondary_flags)} -> {np.sum(extended_flags)} (+{n_extended} from shared tets)")
    
    return extended_flags


def compute_secondary_vertex_labels(
    vertices: np.ndarray,
    edges: np.ndarray,
    secondary_cut_edges: List[Tuple[int, int]],
    tetrahedra: np.ndarray,
    vertex_mold_labels: Optional[np.ndarray] = None,
    edge_to_index: Optional[Dict[Tuple[int, int], int]] = None
) -> np.ndarray:
    """
    Compute BINARY vertex labels for secondary surface extraction.
    
    This function assigns a binary label (1=SideA, 2=SideB) to each vertex
    based on which "side" of the secondary cut edges they are on.
    
    The algorithm uses connected components:
    1. Build a graph of vertices connected by NON-cut edges (edges not in secondary_cut_edges)
    2. Use flood fill to identify connected components
    3. Assign alternating labels (1, 2) to adjacent components
    
    This ensures that marching tetrahedra only sees 0, 3, or 4-edge configs
    (valid binary configurations), avoiding the problematic 5 and 6-edge configs.
    
    Args:
        vertices: (N, 3) vertex positions
        edges: (E, 2) unique edges
        secondary_cut_edges: List of (vi, vj) secondary cut edges
        tetrahedra: (M, 4) tetrahedra vertex indices
        vertex_mold_labels: (N,) optional primary mold labels (1=H1, 2=H2)
                           Used to initialize labels consistently with primary surface
        edge_to_index: Optional pre-computed edge-to-index map
    
    Returns:
        (N,) array of binary labels: 1=SideA, 2=SideB, 0=unlabeled
        
    Note:
        The labels are arbitrary (SideA vs SideB has no geometric meaning),
        but they are consistent: vertices connected by non-cut edges have
        the same label, vertices connected by cut edges have different labels.
    """
    from collections import deque
    
    n_vertices = len(vertices)
    n_edges = len(edges)
    
    if secondary_cut_edges is None or len(secondary_cut_edges) == 0:
        logger.warning("No secondary cut edges - cannot compute secondary vertex labels")
        return np.zeros(n_vertices, dtype=np.int8)
    
    # Build edge-to-index map if not provided
    if edge_to_index is None:
        edge_to_index = build_edge_to_index_map(edges)
    
    # Mark which edges are secondary cuts
    secondary_cut_set = set()
    for vi, vj in secondary_cut_edges:
        secondary_cut_set.add((min(vi, vj), max(vi, vj)))
    
    logger.info(f"Computing secondary vertex labels: {len(secondary_cut_set)} cut edges")
    
    # Build adjacency list using NON-cut edges only
    # Two vertices are "connected" if they share an edge that is NOT a secondary cut
    adjacency = [[] for _ in range(n_vertices)]
    cut_adjacency = [[] for _ in range(n_vertices)]  # Adjacency via CUT edges
    
    for e_idx in range(n_edges):
        v0, v1 = int(edges[e_idx, 0]), int(edges[e_idx, 1])
        key = (min(v0, v1), max(v0, v1))
        
        if key in secondary_cut_set:
            # This is a cut edge - track separately for label propagation
            cut_adjacency[v0].append(v1)
            cut_adjacency[v1].append(v0)
        else:
            # Non-cut edge - vertices are on same side
            adjacency[v0].append(v1)
            adjacency[v1].append(v0)
    
    # Find vertices involved in tetrahedra with secondary cut edges
    # We only need to label vertices that are part of tets with secondary cuts
    involved_vertices = set()
    for vi, vj in secondary_cut_edges:
        involved_vertices.add(vi)
        involved_vertices.add(vj)
    
    # Expand to all vertices in tetrahedra that contain cut edges
    for tet in tetrahedra:
        tet_verts = set(int(v) for v in tet)
        for vi, vj in secondary_cut_edges:
            if vi in tet_verts and vj in tet_verts:
                # This tet has a secondary cut edge - include all its vertices
                involved_vertices.update(tet_verts)
                break
    
    logger.info(f"Secondary labeling: {len(involved_vertices)} vertices involved in cut tets")
    
    # Initialize labels array
    labels = np.zeros(n_vertices, dtype=np.int8)
    
    # BFS to assign labels using connected components
    # Start from a vertex connected to a cut edge and assign label 1
    # Vertices connected via non-cut edges get the same label
    # Vertices connected via cut edges get the opposite label
    
    labeled_count = 0
    component_count = 0
    
    # Process each connected component
    for start_v in involved_vertices:
        if labels[start_v] != 0:
            continue  # Already labeled
        
        # Start new component with label 1
        component_count += 1
        queue = deque()
        queue.append((start_v, 1))  # (vertex, label)
        
        while queue:
            v, lbl = queue.popleft()
            
            if labels[v] != 0:
                # Already labeled - check for consistency
                if labels[v] != lbl:
                    # Inconsistency - this can happen at complex junctions
                    # Just log it and continue
                    pass
                continue
            
            labels[v] = lbl
            labeled_count += 1
            
            # Propagate same label to neighbors via non-cut edges
            for neighbor in adjacency[v]:
                if neighbor in involved_vertices and labels[neighbor] == 0:
                    queue.append((neighbor, lbl))
            
            # Propagate OPPOSITE label to neighbors via cut edges
            opposite_lbl = 2 if lbl == 1 else 1
            for neighbor in cut_adjacency[v]:
                if neighbor in involved_vertices and labels[neighbor] == 0:
                    queue.append((neighbor, opposite_lbl))
    
    n_side_a = np.sum(labels == 1)
    n_side_b = np.sum(labels == 2)
    n_unlabeled = np.sum(labels == 0)
    
    logger.info(f"Secondary vertex labels computed: {n_side_a} SideA, {n_side_b} SideB, "
                f"{n_unlabeled} unlabeled, {component_count} components")
    
    return labels


def prepare_parting_surface_data(tet_result: 'TetrahedralMeshResult') -> 'TetrahedralMeshResult':
    """
    Prepare all data structures needed for parting surface extraction.
    
    This computes:
    - edge_to_index: Dict mapping (vi, vj) -> global edge index
    - tet_edge_indices: (M, 6) mapping from tet to its 6 edge indices
    - cut_edge_flags: (E,) boolean marking cut edges
    
    Args:
        tet_result: TetrahedralMeshResult with primary_cut_edges and secondary_cut_edges
    
    Returns:
        Updated TetrahedralMeshResult with parting surface data populated
    """
    import time
    start = time.time()
    
    # Build edge-to-index map
    tet_result.edge_to_index = build_edge_to_index_map(tet_result.edges)
    
    # Build tet-to-edge mapping
    tet_result.tet_edge_indices = build_tet_edge_indices(
        tet_result.tetrahedra,
        tet_result.edge_to_index
    )
    
    # Compute cut edge flags
    tet_result.cut_edge_flags = compute_cut_edge_flags(
        tet_result.edges,
        tet_result.primary_cut_edges,
        tet_result.secondary_cut_edges,
        tet_result.edge_to_index
    )
    
    elapsed_ms = (time.time() - start) * 1000
    
    n_primary = len(tet_result.primary_cut_edges) if tet_result.primary_cut_edges else 0
    n_secondary = len(tet_result.secondary_cut_edges) if tet_result.secondary_cut_edges else 0
    n_cut = np.sum(tet_result.cut_edge_flags)
    
    logger.info(f"Prepared parting surface data in {elapsed_ms:.1f}ms: "
                f"{n_primary} primary + {n_secondary} secondary = {n_cut} cut edges")
    
    return tet_result


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
    import time
    start_time = time.time()
    
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
    boundary_faces = np.asarray(boundary_mesh.faces, dtype=np.int64)
    
    # First, we need to map boundary mesh vertex indices to tet mesh vertex indices
    # The boundary mesh may have different vertex indexing than the tet mesh
    boundary_mesh_verts = np.asarray(boundary_mesh.vertices)
    tet_verts = tet_result.vertices
    
    # Build mapping from boundary mesh vertices to tet vertices
    # Use spatial matching since indices may not correspond directly
    from scipy.spatial import cKDTree
    kdtree_start = time.time()
    tet_tree = cKDTree(tet_verts)
    _, boundary_to_tet_map = tet_tree.query(boundary_mesh_verts, k=1)
    kdtree_time = (time.time() - kdtree_start) * 1000
    
    # Vectorized extraction of boundary surface edges (in tet vertex indices)
    # Convert all face vertex indices to tet vertex indices
    tet_faces = boundary_to_tet_map[boundary_faces]  # (n_faces, 3)
    
    # Extract edges from faces (each face has 3 edges)
    edge0_v0 = tet_faces[:, 0]
    edge0_v1 = tet_faces[:, 1]
    edge1_v0 = tet_faces[:, 1]
    edge1_v1 = tet_faces[:, 2]
    edge2_v0 = tet_faces[:, 2]
    edge2_v1 = tet_faces[:, 0]
    
    # Stack all edges and sort each edge (min, max)
    all_edges_v0 = np.concatenate([edge0_v0, edge1_v0, edge2_v0])
    all_edges_v1 = np.concatenate([edge0_v1, edge1_v1, edge2_v1])
    sorted_min = np.minimum(all_edges_v0, all_edges_v1)
    sorted_max = np.maximum(all_edges_v0, all_edges_v1)
    
    # Create a unique key for each edge using Cantor pairing function
    # For efficiency, use simpler hashing: v_min * max_vertex_id + v_max
    max_v = max(tet_verts.shape[0], np.max(sorted_max) + 1)
    boundary_edge_keys = sorted_min.astype(np.int64) * max_v + sorted_max.astype(np.int64)
    boundary_edge_keys_set = set(boundary_edge_keys)
    
    logger.info(f"Found {len(boundary_edge_keys_set)} unique edges on boundary surface")
    
    # Get labels for both endpoints of each edge
    v0_labels = boundary_labels[edges[:, 0]]
    v1_labels = boundary_labels[edges[:, 1]]
    
    # Vectorized check if each tet edge is on the boundary surface
    edges_arr = np.asarray(edges, dtype=np.int64)
    tet_edge_min = np.minimum(edges_arr[:, 0], edges_arr[:, 1])
    tet_edge_max = np.maximum(edges_arr[:, 0], edges_arr[:, 1])
    tet_edge_keys = tet_edge_min * max_v + tet_edge_max
    
    # Use numpy isin for vectorized set membership (can be slow for large sets)
    # Instead, convert to sorted array and use searchsorted
    boundary_keys_sorted = np.sort(np.array(list(boundary_edge_keys_set), dtype=np.int64))
    insertion_idx = np.searchsorted(boundary_keys_sorted, tet_edge_keys)
    insertion_idx = np.clip(insertion_idx, 0, len(boundary_keys_sorted) - 1)
    on_boundary_surface = boundary_keys_sorted[insertion_idx] == tet_edge_keys
    
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
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(
        f"[Python] Edge boundary labels in {elapsed:.0f}ms (KDTree: {kdtree_time:.0f}ms): "
        f"interior={n_interior}, H1={n_h1}, H2={n_h2}, inner={n_inner}, mixed={n_mixed}"
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
    boundary_mesh: trimesh.Trimesh,
    tolerance: float = None
) -> np.ndarray:
    """
    Identify which tet vertices lie on the boundary surface.
    
    Uses trimesh's fast proximity query (CPU only, but efficient for this task)
    since we just need a boolean mask, not exact distances.
    
    Args:
        tet_vertices: (N, 3) tetrahedral mesh vertices
        boundary_mesh: The boundary surface mesh (e.g., tetrahedral boundary)
        tolerance: Distance tolerance for boundary detection.
                   If None, uses 0.1% of bounding box diagonal.
    
    Returns:
        (N,) boolean array - True if vertex is on boundary
    """
    if tolerance is None:
        bbox_diag = np.linalg.norm(boundary_mesh.bounds[1] - boundary_mesh.bounds[0])
        tolerance = bbox_diag * 0.001
    
    # Use trimesh's fast proximity query directly (CPU, but efficient for boundary check)
    # This is faster than our GPU implementation for this specific use case
    # because we're checking ALL vertices against the surface they came from
    closest_points, distances, triangle_ids = boundary_mesh.nearest.on_surface(tet_vertices)
    
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
    import time
    start_time = time.time()
    
    n_tet_verts = len(tet_vertices)
    labels = np.zeros(n_tet_verts, dtype=np.int8)
    
    # Get boundary vertex indices
    boundary_indices = np.where(boundary_mask)[0]
    
    if len(boundary_indices) == 0:
        return labels
    
    # Get classification sets
    h1_set = classification_result.h1_triangles
    h2_set = classification_result.h2_triangles
    inner_set = classification_result.inner_boundary_triangles
    
    # Find the mapping from boundary mesh vertices to tet vertices
    # Use KDTree for efficient lookup
    from scipy.spatial import cKDTree
    
    kdtree_start = time.time()
    tet_tree = cKDTree(tet_vertices)
    boundary_to_tet = tet_tree.query(boundary_mesh.vertices, k=1)[1]
    kdtree_time = (time.time() - kdtree_start) * 1000
    
    # Get boundary mesh faces
    boundary_faces = np.asarray(boundary_mesh.faces)
    n_faces = len(boundary_faces)
    
    # Create face label array (vectorized)
    face_labels = np.zeros(n_faces, dtype=np.int8)  # 0=boundary_zone, 1=H1, 2=H2, -1=inner
    
    # Convert sets to arrays for vectorized assignment
    if len(h1_set) > 0:
        h1_indices = np.array(list(h1_set), dtype=np.int64)
        h1_indices = h1_indices[h1_indices < n_faces]  # Bounds check
        face_labels[h1_indices] = 1
    if len(h2_set) > 0:
        h2_indices = np.array(list(h2_set), dtype=np.int64)
        h2_indices = h2_indices[h2_indices < n_faces]
        face_labels[h2_indices] = 2
    if len(inner_set) > 0:
        inner_indices = np.array(list(inner_set), dtype=np.int64)
        inner_indices = inner_indices[inner_indices < n_faces]
        face_labels[inner_indices] = -1
    
    # Map boundary mesh face vertex indices to tet vertex indices
    tet_faces = boundary_to_tet[boundary_faces]  # (n_faces, 3) tet vertex indices
    
    # Initialize vertex counts
    vertex_h1_count = np.zeros(n_tet_verts, dtype=np.int32)
    vertex_h2_count = np.zeros(n_tet_verts, dtype=np.int32)
    vertex_inner_count = np.zeros(n_tet_verts, dtype=np.int32)
    
    # Vectorized counting using np.add.at
    h1_mask = face_labels == 1
    h2_mask = face_labels == 2
    inner_mask = face_labels == -1
    
    h1_faces = tet_faces[h1_mask]
    if len(h1_faces) > 0:
        np.add.at(vertex_h1_count, h1_faces[:, 0], 1)
        np.add.at(vertex_h1_count, h1_faces[:, 1], 1)
        np.add.at(vertex_h1_count, h1_faces[:, 2], 1)
    
    h2_faces = tet_faces[h2_mask]
    if len(h2_faces) > 0:
        np.add.at(vertex_h2_count, h2_faces[:, 0], 1)
        np.add.at(vertex_h2_count, h2_faces[:, 1], 1)
        np.add.at(vertex_h2_count, h2_faces[:, 2], 1)
    
    inner_faces = tet_faces[inner_mask]
    if len(inner_faces) > 0:
        np.add.at(vertex_inner_count, inner_faces[:, 0], 1)
        np.add.at(vertex_inner_count, inner_faces[:, 1], 1)
        np.add.at(vertex_inner_count, inner_faces[:, 2], 1)
    
    # Vectorized label assignment using majority voting
    # H1 wins if h1 >= h2 AND h1 >= inner
    h1_wins = (vertex_h1_count >= vertex_h2_count) & (vertex_h1_count >= vertex_inner_count) & (vertex_h1_count > 0)
    labels[h1_wins] = 1
    
    # H2 wins if h2 > h1 AND h2 >= inner (strict > for h1 since h1 checked first)
    h2_wins = (vertex_h2_count > vertex_h1_count) & (vertex_h2_count >= vertex_inner_count) & (vertex_h2_count > 0) & ~h1_wins
    labels[h2_wins] = 2
    
    # Inner wins if inner > h1 AND inner > h2
    inner_wins = (vertex_inner_count > vertex_h1_count) & (vertex_inner_count > vertex_h2_count) & ~h1_wins & ~h2_wins
    labels[inner_wins] = -1
    
    # Log distribution
    n_h1 = np.sum(labels == 1)
    n_h2 = np.sum(labels == 2)
    n_inner = np.sum(labels == -1)
    n_unclassified = np.sum(labels == 0)
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"label_boundary_from_classification in {elapsed:.0f}ms (KDTree: {kdtree_time:.0f}ms): "
               f"H1={n_h1}, H2={n_h2}, inner(seeds)={n_inner}, unclassified={n_unclassified}")
    
    return labels


def generate_tetrahedral_mesh(
    hull_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
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
        hull_mesh: The inflated hull mesh (bounding volume to tetrahedralize)
        part_mesh: Original part mesh (for distance computation and filtering)
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
        hull_mesh, 
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


def prepare_secondary_cuts_cache(
    tet_result: TetrahedralMeshResult,
    part_mesh: trimesh.Trimesh,
    boundary_mesh: trimesh.Trimesh = None
) -> TetrahedralMeshResult:
    """
    Precompute and cache data structures needed for secondary cut detection.
    
    This function computes expensive data ONCE that would otherwise be
    recomputed every time find_secondary_cutting_edges() is called:
    
    1. boundary_adjacency: Adjacency list for boundary mesh (for shortest path on ∂H)
    2. part_to_tet_vertex: KDTree-based mapping from part mesh to tet mesh vertices
    3. cached_seed_triangles: Part mesh triangles mapped to tet mesh vertex indices
    
    Call this function AFTER Dijkstra labeling is complete and BEFORE
    calling find_secondary_cutting_edges() for optimal performance.
    
    If running secondary cuts multiple times (e.g., with different thresholds),
    this cache provides significant speedup.
    
    Args:
        tet_result: TetrahedralMeshResult with Dijkstra results complete
        part_mesh: The original part mesh M
        boundary_mesh: Optional boundary mesh (uses tet_result.boundary_mesh if None)
    
    Returns:
        Updated TetrahedralMeshResult with cache fields populated
    """
    import time
    from scipy.spatial import cKDTree
    
    start_time = time.time()
    
    # Use provided boundary mesh or fall back to tet_result's
    if boundary_mesh is None:
        boundary_mesh = tet_result.boundary_mesh
    
    if boundary_mesh is None:
        logger.warning("No boundary mesh available - cannot prepare secondary cuts cache")
        return tet_result
    
    vertices = tet_result.vertices
    boundary_labels = tet_result.boundary_labels
    
    # =========================================================================
    # 1. Build boundary mesh adjacency (for shortest path on ∂H)
    # =========================================================================
    if tet_result.boundary_adjacency is None:
        logger.info("Building boundary mesh adjacency...")
        tet_result.boundary_adjacency = _build_boundary_adjacency(boundary_mesh)
        logger.info(f"  Built adjacency for {len(tet_result.boundary_adjacency)} boundary vertices")
    else:
        logger.info("Using cached boundary adjacency")
    
    # =========================================================================
    # 2. Build part mesh to tet mesh vertex mapping (expensive KDTree query)
    # =========================================================================
    if tet_result.part_to_tet_vertex is None:
        logger.info("Building part-to-tet vertex mapping (KDTree)...")
        part_verts = np.asarray(part_mesh.vertices)
        tet_tree = cKDTree(vertices)
        _, part_to_tet = tet_tree.query(part_verts, k=1)
        tet_result.part_to_tet_vertex = part_to_tet
        logger.info(f"  Mapped {len(part_verts)} part vertices to tet mesh")
    else:
        logger.info("Using cached part-to-tet mapping")
    
    # =========================================================================
    # 3. Build seed triangles (part surface in tet mesh resolution)
    # =========================================================================
    if tet_result.cached_seed_triangles is None:
        logger.info("Building seed triangles...")
        
        # Identify part surface vertices
        part_surface_vertices = set()
        if boundary_labels is not None:
            part_surface_vertices = set(np.where(boundary_labels == -1)[0])
        
        # Build seed triangles from part mesh faces
        seed_triangles = []
        seed_triangle_positions = []
        part_to_tet = tet_result.part_to_tet_vertex
        
        for face in part_mesh.faces:
            tet_v0 = part_to_tet[face[0]]
            tet_v1 = part_to_tet[face[1]]
            tet_v2 = part_to_tet[face[2]]
            
            # Check if all three vertices are on part surface
            if (tet_v0 in part_surface_vertices and 
                tet_v1 in part_surface_vertices and 
                tet_v2 in part_surface_vertices):
                # Avoid degenerate triangles
                if tet_v0 != tet_v1 and tet_v1 != tet_v2 and tet_v2 != tet_v0:
                    seed_triangles.append((tet_v0, tet_v1, tet_v2))
                    seed_triangle_positions.append((
                        vertices[tet_v0].copy(),
                        vertices[tet_v1].copy(),
                        vertices[tet_v2].copy()
                    ))
        
        tet_result.cached_seed_triangles = seed_triangles
        tet_result.cached_seed_triangle_positions = seed_triangle_positions
        logger.info(f"  Built {len(seed_triangles)} seed triangles")
    else:
        logger.info("Using cached seed triangles")
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"Secondary cuts cache prepared in {elapsed_ms:.0f}ms")
    
    return tet_result


def find_secondary_cutting_edges(
    tet_result: TetrahedralMeshResult,
    part_mesh: trimesh.Trimesh,
    boundary_mesh: trimesh.Trimesh = None,
    min_intersection_count: int = 1,
    use_gpu: bool = True,
    method: str = 'triangle'  # 'triangle' (default) or 'ray' (experimental)
) -> List[Tuple[int, int]]:
    """
    Find secondary cutting edges where the membrane between same-label interior vertices 
    intersects the part mesh representation.
    
    Per Paper Section 4.2 "Additional Membranes":
    "We introduce a cutting membrane across the edge e to separate vi and vj if a 
    discrete approximation of the minimal surface bounded by the edge (vi, vj), 
    the escape paths (vi, wi), (vj, wj), and the shortest path (wi, wj) 
    (computed on ∂H), intersects the object M."
    
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
    4. If this membrane intersects the part mesh M enough times, mark edge as secondary cut
    
    OPTIMIZATION: Call prepare_secondary_cuts_cache() BEFORE this function to
    precompute expensive data structures. If the cache is not prepared, this
    function will build them on-the-fly (slower for repeated calls).
    
    Args:
        tet_result: TetrahedralMeshResult with Dijkstra results (paths, destinations)
        part_mesh: The original part mesh M to check intersection against
        boundary_mesh: The boundary mesh ∂H (uses tet_result.boundary_mesh_original if not provided)
        min_intersection_count: Minimum number of intersections required (1-20)
                               Higher values filter out false positives from edge cases
                               1 = any intersection triggers secondary cut
                               Higher = requires multiple intersections (more confidence)
        use_gpu: Whether to use CUDA acceleration if available (default True)
        method: Intersection testing method:
                'triangle' - Triangle-triangle intersection (default, correct per paper)
                             Builds membrane surface and checks intersection with seed triangles
                'ray' - (EXPERIMENTAL) Ray casting - may not correctly detect all intersections
    
    Returns:
        List of (vi, vj) tuples representing secondary cutting edges
    """
    import time
    
    start_time = time.time()
    timing_log = {}  # Track timing for each phase
    
    if tet_result.seed_escape_labels is None:
        raise ValueError("Dijkstra must be run before finding secondary cuts")
    
    if tet_result.seed_escape_paths is None:
        raise ValueError("Escape paths must be computed (run updated Dijkstra)")
    
    # Store method and part_mesh for Phase 7
    use_ray_method = (method.lower() == 'ray')
    
    # Use the INFLATED geometry for membrane construction - this is where Dijkstra paths were computed
    # The escape paths are vertex indices that refer to the inflated mesh vertices
    vertices = tet_result.vertices  # Inflated mesh for membrane geometry
    
    # For the boundary mesh, use the current (inflated) boundary mesh
    if boundary_mesh is None:
        boundary_mesh = tet_result.boundary_mesh
    
    if boundary_mesh is None:
        logger.warning("No boundary mesh available for secondary cuts")
        return []
    
    logger.info(f"=== SECONDARY CUTS DETAILED TIMING (method={method}) ===")
    logger.info(f"Input: {len(vertices)} vertices, {len(boundary_mesh.vertices)} boundary verts")
    
    # These now contain ALL interior vertices (not just those on part surface)
    interior_vertex_indices = tet_result.seed_vertex_indices
    interior_escape_labels = tet_result.seed_escape_labels
    interior_escape_destinations = tet_result.seed_escape_destinations
    interior_escape_paths = tet_result.seed_escape_paths
    boundary_labels = tet_result.boundary_labels
    
    n_interior = len(interior_vertex_indices)
    
    # =========================================================================
    # PHASE 1: Build vertex mappings
    # =========================================================================
    phase_start = time.time()
    
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
    
    timing_log['phase1_vertex_mappings'] = (time.time() - phase_start) * 1000
    logger.info(f"Phase 1 (vertex mappings): {timing_log['phase1_vertex_mappings']:.1f}ms - {n_interior} interior, {len(part_surface_vertices)} on part surface")
    
    # =========================================================================
    # PHASE 2: GET/BUILD SEED TRIANGLES
    # =========================================================================
    phase_start = time.time()
    
    # Check if cache is available
    use_cache = (tet_result.cached_seed_triangles is not None and 
                 tet_result.cached_seed_triangle_positions is not None)
    
    if use_cache:
        logger.info("Phase 2: Using cached seed triangles")
        seed_triangles = tet_result.cached_seed_triangles
        seed_triangle_positions = tet_result.cached_seed_triangle_positions
    else:
        # Build seed triangles from PART MESH faces (slower path)
        logger.info("Phase 2: Building seed triangles (not cached)")
        from scipy.spatial import cKDTree
        
        # Use cached part-to-tet mapping if available
        if tet_result.part_to_tet_vertex is not None:
            part_to_tet = tet_result.part_to_tet_vertex
        else:
            part_verts = np.asarray(part_mesh.vertices)
            tet_tree = cKDTree(vertices)
            _, part_to_tet = tet_tree.query(part_verts, k=1)
        
        seed_triangles = []
        seed_triangle_positions = []
        
        for face in part_mesh.faces:
            tet_v0 = part_to_tet[face[0]]
            tet_v1 = part_to_tet[face[1]]
            tet_v2 = part_to_tet[face[2]]
            
            if (tet_v0 in part_surface_vertices and 
                tet_v1 in part_surface_vertices and 
                tet_v2 in part_surface_vertices):
                if tet_v0 != tet_v1 and tet_v1 != tet_v2 and tet_v2 != tet_v0:
                    seed_triangles.append((tet_v0, tet_v1, tet_v2))
                    seed_triangle_positions.append((
                        vertices[tet_v0].copy(),
                        vertices[tet_v1].copy(),
                        vertices[tet_v2].copy()
                    ))
    
    timing_log['phase2_seed_triangles'] = (time.time() - phase_start) * 1000
    logger.info(f"Phase 2 (seed triangles): {timing_log['phase2_seed_triangles']:.1f}ms - {len(seed_triangles)} triangles")
    
    # =========================================================================
    # PHASE 3: FIND CANDIDATE EDGES (same-label interior edges)
    # =========================================================================
    phase_start = time.time()
    
    tet_edges = tet_result.edges
    n_edges = len(tet_edges)
    
    # Vectorized check: both vertices must be interior
    # Build a mask for interior vertices
    interior_mask = np.zeros(len(vertices), dtype=bool)
    interior_mask[interior_vertex_indices] = True
    
    v0_interior = interior_mask[tet_edges[:, 0]]
    v1_interior = interior_mask[tet_edges[:, 1]]
    both_interior = v0_interior & v1_interior
    
    # For interior edges, check if same label
    candidate_edges = []
    n_on_part_surface = 0
    
    for e_idx in np.where(both_interior)[0]:
        v0, v1 = int(tet_edges[e_idx, 0]), int(tet_edges[e_idx, 1])
        idx0 = vertex_to_interior_idx[v0]
        idx1 = vertex_to_interior_idx[v1]
        
        label0 = interior_escape_labels[idx0]
        label1 = interior_escape_labels[idx1]
        
        # Only consider same-label edges (both H1 or both H2)
        if label0 == label1 and label0 > 0:
            candidate_edges.append((v0, v1, idx0, idx1))
            if v0 in part_surface_vertices and v1 in part_surface_vertices:
                n_on_part_surface += 1
    
    timing_log['phase3_candidate_edges'] = (time.time() - phase_start) * 1000
    logger.info(f"Phase 3 (candidate edges): {timing_log['phase3_candidate_edges']:.1f}ms - {len(candidate_edges)} edges ({n_on_part_surface} on part surface)")
    
    if len(candidate_edges) == 0 or len(seed_triangles) == 0:
        return []
    
    # =========================================================================
    # PHASE 4: BUILD/GET BOUNDARY ADJACENCY
    # =========================================================================
    phase_start = time.time()
    
    boundary_verts = np.asarray(boundary_mesh.vertices)
    
    if tet_result.boundary_adjacency is not None:
        logger.info("Phase 4: Using cached boundary adjacency")
        boundary_adjacency = tet_result.boundary_adjacency
    else:
        logger.info("Phase 4: Building boundary adjacency (not cached)")
        boundary_adjacency = _build_boundary_adjacency(boundary_mesh)
    
    timing_log['phase4_boundary_adjacency'] = (time.time() - phase_start) * 1000
    logger.info(f"Phase 4 (boundary adjacency): {timing_log['phase4_boundary_adjacency']:.1f}ms")
    
    # Calculate minimum membrane thickness to filter out nearly-degenerate membranes
    avg_edge_length = np.mean(tet_result.edge_lengths) if tet_result.edge_lengths is not None else 1.0
    min_membrane_thickness = 0.0  # Disabled - rely on intersection counting
    
    # =========================================================================
    # PHASE 5: PRE-COMPUTE BOUNDARY MAPPINGS
    # =========================================================================
    phase_start = time.time()
    
    # Pre-compute boundary mappings for all unique destinations
    unique_destinations = set()
    for vi, vj, idx_i, idx_j in candidate_edges:
        wi = interior_escape_destinations[idx_i]
        wj = interior_escape_destinations[idx_j]
        if wi >= 0:
            unique_destinations.add(wi)
        if wj >= 0:
            unique_destinations.add(wj)
    
    # Pre-compute tet vertex -> boundary vertex mapping for all destinations
    dest_to_boundary = {}
    for dest in unique_destinations:
        dest_to_boundary[dest] = _find_nearest_boundary_vertex(vertices[dest], boundary_verts)
    
    timing_log['phase5_dest_to_boundary'] = (time.time() - phase_start) * 1000
    logger.info(f"Phase 5 (dest-to-boundary mapping): {timing_log['phase5_dest_to_boundary']:.1f}ms - {len(unique_destinations)} destinations")
    
    # =========================================================================
    # PHASE 6: BUILD MEMBRANE DATA (with boundary path computation)
    # =========================================================================
    phase_start = time.time()
    
    # OPTIMIZATION: First pass - collect all unique boundary pairs needed
    # This allows us to batch-compute shortest paths more efficiently
    needed_boundary_pairs = set()
    edge_to_boundary_pair = []  # (vi, vj, idx_i, idx_j, wi_boundary, wj_boundary)
    skipped_same_dest = 0
    skipped_no_path = 0
    
    for vi, vj, idx_i, idx_j in candidate_edges:
        # Get escape paths (already cached from Dijkstra)
        path_i = interior_escape_paths[idx_i]  # vi → wi
        path_j = interior_escape_paths[idx_j]  # vj → wj
        
        if len(path_i) < 1 or len(path_j) < 1:
            skipped_no_path += 1
            continue
        
        wi = interior_escape_destinations[idx_i]
        wj = interior_escape_destinations[idx_j]
        
        if wi < 0 or wj < 0:
            skipped_no_path += 1
            continue
        
        # Use pre-computed boundary mappings
        wi_boundary = dest_to_boundary[wi]
        wj_boundary = dest_to_boundary[wj]
        
        # OPTIMIZATION: Skip if destinations are the same (no membrane possible)
        if wi_boundary == wj_boundary:
            skipped_same_dest += 1
            continue
        
        # Record the boundary pair needed
        cache_key = (min(wi_boundary, wj_boundary), max(wi_boundary, wj_boundary))
        needed_boundary_pairs.add(cache_key)
        edge_to_boundary_pair.append((vi, vj, idx_i, idx_j, wi_boundary, wj_boundary, cache_key))
    
    pair_collection_time = (time.time() - phase_start) * 1000
    logger.info(f"Phase 6a (collect pairs): {pair_collection_time:.1f}ms - {len(needed_boundary_pairs)} unique boundary pairs from {len(edge_to_boundary_pair)} edges")
    
    # =========================================================================
    # PHASE 6b: BATCH COMPUTE BOUNDARY PATHS
    # Use multi-source Dijkstra for efficiency
    # =========================================================================
    batch_start = time.time()
    
    # Get unique source vertices (first element of each pair)
    unique_sources = set()
    for a, b in needed_boundary_pairs:
        unique_sources.add(a)
        unique_sources.add(b)
    
    logger.info(f"Phase 6b: Computing shortest paths from {len(unique_sources)} unique boundary sources")
    
    # Use cached boundary paths from tet_result if available, or compute
    boundary_path_cache = {}
    
    if tet_result.cached_boundary_paths is not None:
        # Use cached paths
        cached_hits = 0
        for pair in needed_boundary_pairs:
            if pair in tet_result.cached_boundary_paths:
                boundary_path_cache[pair] = tet_result.cached_boundary_paths[pair]
                cached_hits += 1
        logger.info(f"  - Cache hits: {cached_hits}/{len(needed_boundary_pairs)}")
    
    # Compute missing paths using multi-source Dijkstra
    missing_pairs = [p for p in needed_boundary_pairs if p not in boundary_path_cache]
    
    if len(missing_pairs) > 0:
        # Group by source for efficient multi-target Dijkstra
        source_to_targets = {}
        for a, b in missing_pairs:
            if a not in source_to_targets:
                source_to_targets[a] = set()
            source_to_targets[a].add(b)
            # Also add reverse (b->a) since we need both directions
            if b not in source_to_targets:
                source_to_targets[b] = set()
            source_to_targets[b].add(a)
        
        logger.info(f"  - Computing {len(missing_pairs)} missing paths from {len(source_to_targets)} sources")
        
        # Run multi-target Dijkstra from each source
        path_compute_start = time.time()
        for source, targets in source_to_targets.items():
            # Run Dijkstra from source to all targets at once
            paths = _multi_target_dijkstra_boundary(source, targets, boundary_mesh, boundary_adjacency)
            for target, path in paths.items():
                cache_key = (min(source, target), max(source, target))
                if cache_key not in boundary_path_cache:
                    # Store path in canonical direction (smaller vertex first)
                    if source < target:
                        boundary_path_cache[cache_key] = path
                    else:
                        boundary_path_cache[cache_key] = path[::-1] if path else []
        
        path_compute_time = (time.time() - path_compute_start) * 1000
        logger.info(f"  - Path computation: {path_compute_time:.1f}ms")
    
    # Cache the computed paths for future use
    if tet_result.cached_boundary_paths is None:
        tet_result.cached_boundary_paths = {}
    tet_result.cached_boundary_paths.update(boundary_path_cache)
    
    batch_time = (time.time() - batch_start) * 1000
    logger.info(f"Phase 6b (batch boundary paths): {batch_time:.1f}ms - {len(boundary_path_cache)} paths computed/cached")
    
    # =========================================================================
    # PHASE 6c: BUILD FINAL MEMBRANE DATA
    # =========================================================================
    assemble_start = time.time()
    edge_membrane_data = []
    
    for vi, vj, idx_i, idx_j, wi_boundary, wj_boundary, cache_key in edge_to_boundary_pair:
        path_i = interior_escape_paths[idx_i]
        path_j = interior_escape_paths[idx_j]
        
        boundary_path = boundary_path_cache.get(cache_key, [])
        # Reverse if needed based on direction
        if wi_boundary > wj_boundary:
            boundary_path = boundary_path[::-1]
        
        edge_membrane_data.append((vi, vj, path_i, path_j, boundary_path))
    
    assemble_time = (time.time() - assemble_start) * 1000
    
    timing_log['phase6_membrane_data'] = (time.time() - phase_start) * 1000
    timing_log['phase6_boundary_paths'] = batch_time
    logger.info(f"Phase 6 total: {timing_log['phase6_membrane_data']:.1f}ms")
    logger.info(f"  - Pair collection: {pair_collection_time:.1f}ms")
    logger.info(f"  - Batch path compute: {batch_time:.1f}ms")
    logger.info(f"  - Assembly: {assemble_time:.1f}ms")
    logger.info(f"  - Membranes: {len(edge_membrane_data)}, skipped: {skipped_same_dest} same-dest, {skipped_no_path} no-path")
    
    if len(edge_membrane_data) == 0:
        return []
    
    # =========================================================================
    # PHASE 7: INTERSECTION TESTING
    # =========================================================================
    phase_start = time.time()
    logger.info(f"Phase 7: Starting intersection tests (method={method}, min_intersection_count={min_intersection_count})")
    
    # Default minimum membrane thickness to avoid false positives from flat surfaces
    min_membrane_thickness = 0.0
    
    if use_ray_method:
        # EXPERIMENTAL: Ray-based method (may miss some intersections)
        logger.info("Phase 7: Using ray-based method (EXPERIMENTAL)")
        secondary_cuts = _find_secondary_cuts_ray_based(
            edge_membrane_data,
            part_mesh,
            vertices,
            boundary_verts,
            min_intersection_count,
            min_membrane_thickness
        )
    else:
        # PAPER-FAITHFUL: Collision-based method
        # Uses trimesh CollisionManager with FCL/BVH for O(log n) collision queries
        # IMPORTANT: Check against SEED TRIANGLES (tet mesh resolution), not original part mesh
        # This avoids false positives from resolution mismatch
        logger.info("Phase 7: Using collision-based method (BVH-accelerated, tet-resolution)")
        secondary_cuts = _find_secondary_cuts_collision_based(
            edge_membrane_data,
            seed_triangles,
            seed_triangle_positions,
            vertices,
            boundary_verts,
            min_intersection_count,
            min_membrane_thickness,
            part_mesh  # Fallback only
        )
    
    timing_log['phase7_intersection_test'] = (time.time() - phase_start) * 1000
    
    # =========================================================================
    # TIMING SUMMARY
    # =========================================================================
    total_elapsed = (time.time() - start_time) * 1000
    
    logger.info(f"=== SECONDARY CUTS TIMING SUMMARY ===")
    logger.info(f"Phase 1 (vertex mappings):      {timing_log.get('phase1_vertex_mappings', 0):.1f}ms")
    logger.info(f"Phase 2 (seed triangles):       {timing_log.get('phase2_seed_triangles', 0):.1f}ms")
    logger.info(f"Phase 3 (candidate edges):      {timing_log.get('phase3_candidate_edges', 0):.1f}ms")
    logger.info(f"Phase 4 (boundary adjacency):   {timing_log.get('phase4_boundary_adjacency', 0):.1f}ms")
    logger.info(f"Phase 5 (dest-to-boundary):     {timing_log.get('phase5_dest_to_boundary', 0):.1f}ms")
    logger.info(f"Phase 6 (membrane data):        {timing_log.get('phase6_membrane_data', 0):.1f}ms")
    logger.info(f"  └─ Boundary paths:            {timing_log.get('phase6_boundary_paths', 0):.1f}ms")
    logger.info(f"Phase 7 (intersection test):    {timing_log.get('phase7_intersection_test', 0):.1f}ms")
    logger.info(f"TOTAL:                          {total_elapsed:.1f}ms")
    logger.info(f"Result: {len(secondary_cuts)} secondary cutting edges found")
    
    return secondary_cuts


def _find_secondary_cuts_by_triangle_intersection(
    edge_membrane_data: List[Tuple],
    seed_triangles: List[Tuple[int, int, int]],
    seed_triangle_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int = 1,
    min_membrane_thickness: float = 0.0,
    use_gpu: bool = True
) -> List[Tuple[int, int]]:
    """
    Find secondary cuts by checking if membranes intersect seed TRIANGLES (faces).
    
    This approach checks if the membrane surface cuts through the triangular faces
    formed by seed vertices, which represent the part surface in tetrahedral mesh
    resolution. This is more robust than checking individual edges.
    
    Args:
        min_intersection_count: Minimum number of segment-triangle intersections required (1-20)
        min_membrane_thickness: Skip membranes thinner than this (avoids false positives
                               on flat surfaces where escape paths are nearly identical)
    
    Uses GPU acceleration if available for massive speedup.
    """
    import time
    
    n_triangles = len(seed_triangles)
    n_membranes = len(edge_membrane_data)
    logger.info(f"Phase 7: checking {n_membranes} membranes against {n_triangles} seed triangles (min_intersections={min_intersection_count})")
    
    if n_triangles == 0:
        logger.warning("No seed triangles found - cannot check for secondary cuts")
        return []
    
    # Check GPU availability
    gpu_available = use_gpu and CUDA_AVAILABLE and TORCH_AVAILABLE
    
    if gpu_available:
        logger.info("Secondary cuts: Using GPU acceleration")
        return _find_secondary_cuts_triangle_gpu(
            edge_membrane_data, seed_triangles, seed_triangle_positions,
            vertices, boundary_verts, min_intersection_count, min_membrane_thickness
        )
    else:
        logger.info("Secondary cuts: Using CPU")
        return _find_secondary_cuts_triangle_cpu(
            edge_membrane_data, seed_triangles, seed_triangle_positions,
            vertices, boundary_verts, min_intersection_count, min_membrane_thickness
        )


def _find_secondary_cuts_ray_based(
    edge_membrane_data: List[Tuple],
    part_mesh: 'trimesh.Trimesh',
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int = 1,
    min_membrane_thickness: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Paper-faithful secondary cuts detection using ray casting.
    
    Per Section 4.2: Check if "minimal surface bounded by escape paths intersects M".
    
    Instead of building full membrane triangles, we cast rays across the membrane
    region and check for intersections with the original part mesh M.
    This is much faster because:
    1. trimesh uses BVH internally for O(log n) ray queries
    2. We don't need to build membrane triangles
    3. The part mesh BVH is built once and reused for all membranes
    
    The "cross-path" segments (between path_i and path_j at each depth) 
    approximate the minimal surface check.
    """
    import time
    
    n_membranes = len(edge_membrane_data)
    logger.info(f"Phase 7 (ray-based): checking {n_membranes} membranes against part mesh")
    
    # Build ray intersector once for the part mesh (uses BVH internally)
    ray_intersector = part_mesh.ray
    
    secondary_cuts = []
    
    # Progress tracking
    progress_interval = max(1, n_membranes // 10)
    last_progress_time = time.time()
    
    for membrane_idx, (vi, vj, path_i, path_j, boundary_path) in enumerate(edge_membrane_data):
        n_i = len(path_i)
        n_j = len(path_j)
        
        if n_i < 2 or n_j < 2:
            continue
        
        # Get path positions
        path_i_positions = vertices[path_i]
        path_j_positions = vertices[path_j]
        
        # Check membrane thickness (skip flat surfaces)
        if min_membrane_thickness > 0:
            max_dist = 0.0
            for k in range(min(n_i, n_j)):
                dist = np.linalg.norm(path_i_positions[k] - path_j_positions[k])
                max_dist = max(max_dist, dist)
            if max_dist < min_membrane_thickness:
                continue
        
        # Sample cross-path segments (rays between path_i and path_j)
        # This approximates the minimal surface
        intersection_count = 0
        n_samples = min(max(n_i, n_j), 10)  # Sample up to 10 depth levels
        
        ray_origins = []
        ray_directions = []
        ray_lengths = []
        
        for s in range(n_samples):
            t = s / max(1, n_samples - 1)
            
            idx_i = min(int(t * (n_i - 1)), n_i - 1)
            idx_j = min(int(t * (n_j - 1)), n_j - 1)
            
            p_i = path_i_positions[idx_i]
            p_j = path_j_positions[idx_j]
            
            direction = p_j - p_i
            length = np.linalg.norm(direction)
            
            if length > 1e-6:
                ray_origins.append(p_i)
                ray_directions.append(direction / length)
                ray_lengths.append(length)
        
        if len(ray_origins) == 0:
            continue
        
        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)
        ray_lengths = np.array(ray_lengths)
        
        # Cast rays and check for intersections
        locations, index_ray, index_tri = ray_intersector.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
        
        # Count valid intersections (within the segment, not beyond)
        for hit_idx, ray_idx in enumerate(index_ray):
            hit_point = locations[hit_idx]
            origin = ray_origins[ray_idx]
            hit_dist = np.linalg.norm(hit_point - origin)
            
            # Check if hit is within the segment (with small tolerance)
            if hit_dist <= ray_lengths[ray_idx] * 1.01:
                intersection_count += 1
                if intersection_count >= min_intersection_count:
                    break
        
        if intersection_count >= min_intersection_count:
            secondary_cuts.append((vi, vj))
        
        # Progress logging
        if membrane_idx > 0 and membrane_idx % progress_interval == 0:
            elapsed = time.time() - last_progress_time
            pct = (membrane_idx / n_membranes) * 100
            rate = progress_interval / elapsed if elapsed > 0 else 0
            eta = (n_membranes - membrane_idx) / rate if rate > 0 else 0
            logger.info(f"  Ray progress: {membrane_idx}/{n_membranes} ({pct:.0f}%) - {len(secondary_cuts)} cuts found - ETA: {eta:.1f}s")
            last_progress_time = time.time()
    
    return secondary_cuts


def _find_secondary_cuts_collision_based(
    edge_membrane_data: List[Tuple],
    seed_triangles: List[Tuple[int, int, int]],
    seed_triangle_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int = 1,
    min_membrane_thickness: float = 0.0,
    part_mesh: 'trimesh.Trimesh' = None  # Fallback only
) -> List[Tuple[int, int]]:
    """
    Paper-faithful secondary cuts detection using trimesh CollisionManager.
    
    Per Section 4.2: Check if "minimal surface bounded by escape paths intersects M".
    
    IMPORTANT: We check collision against the SEED TRIANGLES (part surface in tet mesh
    resolution), NOT the original part mesh. This avoids false positives caused by
    resolution mismatch between the tet mesh and original mesh.
    
    The membranes are built from tet mesh vertices, so checking against a surface
    also in tet mesh resolution ensures consistent collision detection.
    
    This method:
    1. Builds a mesh from seed triangles (part surface in tet resolution)
    2. Creates a CollisionManager with this mesh (uses FCL with BVH internally)
    3. For each candidate edge, constructs the membrane mesh
    4. Checks collision between membrane and seed surface - O(log n) per query
    """
    import time
    
    n_membranes = len(edge_membrane_data)
    n_seed_tris = len(seed_triangles)
    logger.info(f"Phase 7 (collision-based): checking {n_membranes} membranes against {n_seed_tris} seed triangles (tet resolution)")
    
    if n_seed_tris == 0:
        logger.warning("No seed triangles - cannot check for secondary cuts")
        return []
    
    # Try to use trimesh's collision module (requires python-fcl)
    try:
        from trimesh.collision import CollisionManager
        # Test that FCL is actually available by creating a manager
        manager = CollisionManager()
    except (ImportError, ValueError) as e:
        logger.warning(f"trimesh.collision/FCL not available ({e}) - falling back to CPU triangle method")
        return _find_secondary_cuts_triangle_cpu(
            edge_membrane_data, seed_triangles, seed_triangle_positions,
            vertices, boundary_verts, min_intersection_count, min_membrane_thickness
        )
    
    # Build mesh from seed triangles (part surface in tet mesh resolution)
    seed_verts = []
    seed_faces = []
    for tri_idx, (p0, p1, p2) in enumerate(seed_triangle_positions):
        base = len(seed_verts)
        seed_verts.extend([p0, p1, p2])
        seed_faces.append([base, base + 1, base + 2])
    
    seed_mesh = trimesh.Trimesh(
        vertices=np.array(seed_verts),
        faces=np.array(seed_faces),
        process=False
    )
    
    # Add seed mesh to collision manager (builds BVH internally)
    manager.add_object("seed_surface", seed_mesh)
    logger.info(f"  Built seed surface mesh: {len(seed_verts)} verts, {len(seed_faces)} faces")
    
    # Compute average edge length for penetration depth threshold
    # This scales the threshold appropriately for different mesh sizes
    avg_edge_length = seed_mesh.edges_unique_length.mean() if len(seed_mesh.edges_unique) > 0 else 1.0
    logger.info(f"  Average seed mesh edge length: {avg_edge_length:.4f}")
    
    secondary_cuts = []
    
    # Progress tracking
    progress_interval = max(1, n_membranes // 10)
    last_progress_time = time.time()
    
    for membrane_idx, (vi, vj, path_i, path_j, boundary_path) in enumerate(edge_membrane_data):
        # Build membrane mesh, skipping the first triangle pair (touches part surface)
        # This avoids false positives where the membrane origin touches the seed surface
        membrane_triangles = _build_membrane_triangles(
            path_i, path_j, boundary_path, vertices, boundary_verts,
            min_thickness=min_membrane_thickness,
            skip_first_n=2  # Skip first 2 triangle pairs (near part surface)
        )
        
        if len(membrane_triangles) == 0:
            continue
        
        # Convert triangle positions to trimesh
        membrane_verts = []
        membrane_faces = []
        for tri_idx, (p0, p1, p2) in enumerate(membrane_triangles):
            base = len(membrane_verts)
            membrane_verts.extend([p0, p1, p2])
            membrane_faces.append([base, base + 1, base + 2])
        
        if len(membrane_verts) == 0:
            continue
        
        membrane_mesh = trimesh.Trimesh(
            vertices=np.array(membrane_verts),
            faces=np.array(membrane_faces),
            process=False
        )
        
        # Check collision with seed surface, requesting contact data for depth filtering
        # This filters out grazing contacts in concave regions
        is_collision, _, contacts = manager.in_collision_single(
            membrane_mesh, return_names=True, return_data=True
        )
        
        if is_collision and contacts:
            # Filter by penetration depth - grazing contacts have very small depth
            # Use min_penetration_depth as threshold (scaled by average edge length)
            min_penetration_depth = avg_edge_length * 0.05  # 5% of avg edge length
            
            significant_collision = False
            for contact in contacts:
                if contact.depth > min_penetration_depth:
                    significant_collision = True
                    break
            
            if significant_collision:
                secondary_cuts.append((vi, vj))
        
        # Progress logging
        if membrane_idx > 0 and membrane_idx % progress_interval == 0:
            elapsed = time.time() - last_progress_time
            pct = (membrane_idx / n_membranes) * 100
            rate = progress_interval / elapsed if elapsed > 0 else 0
            eta = (n_membranes - membrane_idx) / rate if rate > 0 else 0
            logger.info(f"  Collision progress: {membrane_idx}/{n_membranes} ({pct:.0f}%) - {len(secondary_cuts)} cuts found - ETA: {eta:.1f}s")
            last_progress_time = time.time()
    
    return secondary_cuts


def _find_secondary_cuts_fallback_cpu(
    edge_membrane_data: List[Tuple],
    part_mesh: 'trimesh.Trimesh',
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int = 1,
    min_membrane_thickness: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Fallback CPU implementation when trimesh.collision is not available.
    
    Builds seed triangles from part mesh and uses triangle-triangle intersection.
    """
    import time
    from scipy.spatial import cKDTree
    
    logger.info("Phase 7 (fallback CPU): Building seed triangles from part mesh")
    
    # Build seed triangles from part mesh
    part_verts = np.asarray(part_mesh.vertices)
    tet_tree = cKDTree(vertices)
    _, part_to_tet = tet_tree.query(part_verts, k=1)
    
    # All tet vertices near part mesh
    part_surface_set = set(part_to_tet)
    
    seed_triangles = []
    seed_triangle_positions = []
    
    for face in part_mesh.faces:
        tet_v0 = part_to_tet[face[0]]
        tet_v1 = part_to_tet[face[1]]
        tet_v2 = part_to_tet[face[2]]
        
        # Avoid degenerate
        if tet_v0 != tet_v1 and tet_v1 != tet_v2 and tet_v2 != tet_v0:
            seed_triangles.append((tet_v0, tet_v1, tet_v2))
            seed_triangle_positions.append((
                vertices[tet_v0].copy(),
                vertices[tet_v1].copy(),
                vertices[tet_v2].copy()
            ))
    
    logger.info(f"  Built {len(seed_triangles)} seed triangles")
    
    # Now use the existing triangle-triangle CPU method
    return _find_secondary_cuts_triangle_cpu(
        edge_membrane_data, seed_triangles, seed_triangle_positions,
        vertices, boundary_verts, min_intersection_count, min_membrane_thickness
    )


def _find_secondary_cuts_triangle_cpu(
    edge_membrane_data: List[Tuple],
    seed_triangles: List[Tuple[int, int, int]],
    seed_triangle_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int = 1,
    min_membrane_thickness: float = 0.0
) -> List[Tuple[int, int]]:
    """CPU implementation of secondary cuts by triangle-triangle intersection."""
    import time
    
    secondary_cuts = []
    n_membranes = len(edge_membrane_data)
    
    # Progress tracking
    progress_interval = max(1, n_membranes // 10)  # Log every 10%
    last_progress_time = time.time()
    membranes_processed = 0
    total_membrane_build_time = 0.0
    total_intersection_time = 0.0
    
    for membrane_idx, (vi, vj, path_i, path_j, boundary_path) in enumerate(edge_membrane_data):
        # Build membrane triangles (skip if too thin - flat surface case)
        build_start = time.time()
        membrane_triangles = _build_membrane_triangles(
            path_i, path_j, boundary_path, vertices, boundary_verts,
            min_thickness=min_membrane_thickness
        )
        total_membrane_build_time += (time.time() - build_start) * 1000
        
        if len(membrane_triangles) == 0:
            membranes_processed += 1
            continue
        
        # Skip triangles that share vertices with the test edge
        skip_vertices = {vi, vj}
        for v in path_i[:min(3, len(path_i))]:
            skip_vertices.add(v)
        for v in path_j[:min(3, len(path_j))]:
            skip_vertices.add(v)
        
        # Count intersections (require minimum threshold to trigger)
        intersection_start = time.time()
        intersection_count = 0
        
        for tri_idx, (tv0, tv1, tv2) in enumerate(seed_triangles):
            # Skip triangles that share vertices with the membrane edges
            if tv0 in skip_vertices or tv1 in skip_vertices or tv2 in skip_vertices:
                continue
            
            seed_tri_pos = seed_triangle_positions[tri_idx]
            
            for mem_tri in membrane_triangles:
                if _triangles_intersect(mem_tri, seed_tri_pos):
                    intersection_count += 1
                    if intersection_count >= min_intersection_count:
                        break
            
            if intersection_count >= min_intersection_count:
                break
        
        total_intersection_time += (time.time() - intersection_start) * 1000
        
        if intersection_count >= min_intersection_count:
            secondary_cuts.append((vi, vj))
        
        membranes_processed += 1
        
        # Progress logging
        if membrane_idx > 0 and membrane_idx % progress_interval == 0:
            elapsed = time.time() - last_progress_time
            pct = (membrane_idx / n_membranes) * 100
            rate = progress_interval / elapsed if elapsed > 0 else 0
            eta = (n_membranes - membrane_idx) / rate if rate > 0 else 0
            logger.info(f"  CPU progress: {membrane_idx}/{n_membranes} ({pct:.0f}%) - {len(secondary_cuts)} cuts found - ETA: {eta:.1f}s")
            last_progress_time = time.time()
    
    logger.info(f"  CPU complete: membrane_build={total_membrane_build_time:.0f}ms, intersection_test={total_intersection_time:.0f}ms")
    return secondary_cuts


def _find_secondary_cuts_triangle_gpu(
    edge_membrane_data: List[Tuple],
    seed_triangles: List[Tuple[int, int, int]],
    seed_triangle_positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: np.ndarray,
    boundary_verts: np.ndarray,
    min_intersection_count: int = 1,
    min_membrane_thickness: float = 0.0
) -> List[Tuple[int, int]]:
    """
    GPU-accelerated secondary cuts using triangle-triangle intersection.
    
    For each membrane, we check if any of its triangles intersect any seed triangle.
    We use a two-step approach:
    1. Check if membrane edges intersect seed triangles
    2. Check if seed triangle edges intersect membrane triangles
    
    Now counts intersections and requires minimum threshold to trigger.
    """
    import torch
    import time
    
    device = torch.device('cuda')
    n_membranes = len(edge_membrane_data)
    
    # Progress tracking
    progress_interval = max(1, n_membranes // 10)  # Log every 10%
    last_progress_time = time.time()
    total_gpu_time = 0.0
    total_membrane_build_time = 0.0
    
    # Pre-compute all seed triangle data on GPU
    gpu_prep_start = time.time()
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
    
    gpu_prep_time = (time.time() - gpu_prep_start) * 1000
    logger.info(f"  GPU prep (seed triangles to CUDA): {gpu_prep_time:.1f}ms")
    
    secondary_cuts = []
    
    for membrane_idx, (vi, vj, path_i, path_j, boundary_path) in enumerate(edge_membrane_data):
        # Build membrane triangles (skip if too thin - flat surface case)
        build_start = time.time()
        membrane_triangles = _build_membrane_triangles(
            path_i, path_j, boundary_path, vertices, boundary_verts,
            min_thickness=min_membrane_thickness
        )
        total_membrane_build_time += (time.time() - build_start) * 1000
        
        if len(membrane_triangles) == 0:
            continue
        
        # Skip triangles that share vertices with the test edge
        skip_vertices = {vi, vj}
        for v in path_i[:min(3, len(path_i))]:
            skip_vertices.add(v)
        for v in path_j[:min(3, len(path_j))]:
            skip_vertices.add(v)
        
        # Build skip mask for seed triangles
        gpu_start = time.time()
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
        count_1 = _batch_segment_triangle_intersection_gpu(
            mem_edge_p0_gpu, mem_edge_p1_gpu,
            seed_v0_gpu, seed_v1_gpu, seed_v2_gpu,
            skip_mask
        )
        
        # Test 2: Do seed triangle edges intersect membrane triangles?
        # (No skip mask for membrane triangles - we check all)
        no_skip = torch.zeros(len(mem_edge_p0_gpu), dtype=torch.bool, device=device)
        count_2 = _batch_segment_triangle_intersection_gpu(
            seed_edge_p0_gpu, seed_edge_p1_gpu,
            mem_v0_gpu, mem_v1_gpu, mem_v2_gpu,
            skip_edge_mask
        )
        
        total_gpu_time += (time.time() - gpu_start) * 1000
        
        total_intersections = count_1 + count_2
        if total_intersections >= min_intersection_count:
            secondary_cuts.append((vi, vj))
        
        # Progress logging
        if membrane_idx > 0 and membrane_idx % progress_interval == 0:
            elapsed = time.time() - last_progress_time
            pct = (membrane_idx / n_membranes) * 100
            rate = progress_interval / elapsed if elapsed > 0 else 0
            eta = (n_membranes - membrane_idx) / rate if rate > 0 else 0
            logger.info(f"  GPU progress: {membrane_idx}/{n_membranes} ({pct:.0f}%) - {len(secondary_cuts)} cuts found - ETA: {eta:.1f}s")
            last_progress_time = time.time()
        
        # Periodic cleanup
        if membrane_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    logger.info(f"  GPU complete: membrane_build={total_membrane_build_time:.0f}ms, gpu_intersection={total_gpu_time:.0f}ms")
    return secondary_cuts


def _triangles_intersect(
    tri1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tri2: Tuple[np.ndarray, np.ndarray, np.ndarray]
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
    if _segment_intersects_triangle(v0, v1, u0, u1, u2):
        return True
    if _segment_intersects_triangle(v1, v2, u0, u1, u2):
        return True
    if _segment_intersects_triangle(v2, v0, u0, u1, u2):
        return True
    
    # Check edges of tri2 against tri1
    if _segment_intersects_triangle(u0, u1, v0, v1, v2):
        return True
    if _segment_intersects_triangle(u1, u2, v0, v1, v2):
        return True
    if _segment_intersects_triangle(u2, u0, v0, v1, v2):
        return True
    
    return False


def _batch_segment_triangle_intersection_gpu(
    seg_p0: 'torch.Tensor',  # (E, 3) segment start points
    seg_p1: 'torch.Tensor',  # (E, 3) segment end points
    tri_v0: 'torch.Tensor',  # (T, 3) triangle vertex 0
    tri_v1: 'torch.Tensor',  # (T, 3) triangle vertex 1
    tri_v2: 'torch.Tensor',  # (T, 3) triangle vertex 2
    skip_mask: 'torch.Tensor'  # (T,) or (E,) bool - True to skip
) -> int:
    """
    GPU-batched Möller–Trumbore intersection test.
    
    Tests all segments against all triangles in parallel.
    Returns COUNT of non-skipped segment-triangle intersections.
    Boolean intersection detection - no tolerance scaling.
    """
    import torch
    
    EPSILON = 1e-7
    
    n_segs = seg_p0.shape[0]
    n_tris = tri_v0.shape[0]
    
    if n_segs == 0 or n_tris == 0:
        return 0
    
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
    
    # Check if t is within segment bounds (boolean - no tolerance scaling)
    t_valid = (t >= -EPSILON) & (t <= seg_length + EPSILON)  # (E, T)
    
    # Combine all conditions
    intersects = ~parallel_mask & u_valid & v_valid & t_valid  # (E, T)
    
    # Apply skip mask - handle both (T,) and (E,) shapes
    if skip_mask.shape[0] == n_tris:
        # Skip mask is for triangles - broadcast across edges
        intersects = intersects & ~skip_mask.unsqueeze(0)  # (E, T)
    elif skip_mask.shape[0] == n_segs:
        # Skip mask is for edges - broadcast across triangles
        intersects = intersects & ~skip_mask.unsqueeze(1)  # (E, T)
    
    # Return count of intersections
    return intersects.sum().item()


def _build_membrane_triangles(
    path_i: List[int],
    path_j: List[int],
    boundary_path: List[int],
    tet_vertices: np.ndarray,
    boundary_vertices: np.ndarray,
    min_thickness: float = 0.0,
    skip_first_n: int = 0
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build triangulated membrane surface from escape paths.
    
    Returns empty list if the membrane is too thin (paths nearly identical),
    which happens on flat surfaces where adjacent vertices escape the same way.
    
    Args:
        path_i: Escape path from vi to wi (vertex indices)
        path_j: Escape path from vj to wj (vertex indices)
        boundary_path: Path from wi to wj on boundary (vertex indices)
        tet_vertices: Tetrahedral mesh vertices
        boundary_vertices: Boundary mesh vertices
        min_thickness: Minimum membrane thickness (skip if too thin)
        skip_first_n: Number of triangle pairs to skip at the start (near part surface)
                      This helps avoid false positives where membrane touches its origin
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
    # Skip the first skip_first_n triangle pairs (they touch the part surface)
    start_s = skip_first_n
    for s in range(start_s, n_samples - 1):
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
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> bool:
    """
    Check if line segment (p0, p1) intersects triangle (v0, v1, v2).
    
    Uses Möller–Trumbore intersection algorithm.
    Returns True if segment passes through triangle (boolean, no tolerance scaling).
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
    
    # Check if intersection is within the segment (boolean - no tolerance scaling)
    return (t >= -EPSILON) and (t <= seg_length + EPSILON)


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


def _multi_target_dijkstra_boundary(
    source: int,
    targets: set,
    boundary_mesh: trimesh.Trimesh,
    adjacency: Dict[int, List[Tuple[int, float]]]
) -> Dict[int, List[int]]:
    """
    Compute shortest paths from a single source to multiple targets on the boundary mesh.
    
    This is more efficient than running Dijkstra separately for each target,
    as we can stop early once all targets are found.
    
    Args:
        source: Source vertex index on boundary mesh
        targets: Set of target vertex indices
        boundary_mesh: The boundary mesh
        adjacency: Pre-built adjacency list with edge lengths
        
    Returns:
        Dict mapping target -> path (list of vertex indices from source to target)
    """
    import heapq
    
    if not targets:
        return {}
    
    # Quick check for trivial case
    results = {}
    remaining_targets = set(targets)
    
    if source in remaining_targets:
        results[source] = [source]
        remaining_targets.discard(source)
        if not remaining_targets:
            return results
    
    n_verts = len(boundary_mesh.vertices)
    dist = np.full(n_verts, np.inf, dtype=np.float64)
    dist[source] = 0.0
    predecessor = np.full(n_verts, -1, dtype=np.int64)
    
    pq = [(0.0, source)]
    visited = np.zeros(n_verts, dtype=bool)
    
    while pq and remaining_targets:
        d, u = heapq.heappop(pq)
        
        if visited[u]:
            continue
        visited[u] = True
        
        # Check if we reached a target
        if u in remaining_targets:
            # Reconstruct path to this target
            path = []
            current = u
            while current >= 0:
                path.append(current)
                current = predecessor[current]
            path.reverse()
            results[u] = path
            remaining_targets.discard(u)
            
            if not remaining_targets:
                break
        
        # Explore neighbors
        for v, cost in adjacency.get(u, []):
            if visited[v]:
                continue
            new_dist = d + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                predecessor[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    # Return empty paths for any targets we couldn't reach
    for t in remaining_targets:
        results[t] = []
    
    return results


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
