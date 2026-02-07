"""
Parting Surface Extraction using Marching Tetrahedra

This module extracts a continuous parting surface mesh from a tetrahedral mesh
where certain edges have been marked as "cut" (either primary or secondary cuts).

Algorithm: Marching Tetrahedra for Binary Labeling (H1 vs H2)
- Each tetrahedron has 6 edges
- Cut edges connect vertices with different labels (H1 vs H2)
- In a proper binary system, only 3-edge and 4-edge configurations are valid
- All vertices must be labeled H1 or H2 before extraction (see label propagation)

Valid configurations in binary (2-class) system:
- 0-edge: All vertices same label → no surface (trivial)
- 3-edge: One vertex isolated (3+1 split) → single triangle
- 4-edge: Two vertices vs two vertices (2+2 split) → quad (2 triangles)

Invalid configurations (indicate labeling issues):
- 1, 2, 5, 6-edge configs should NOT occur if all vertices are properly labeled

Edge numbering within a tetrahedron (vertices 0,1,2,3):
    Edge 0: (0,1)
    Edge 1: (0,2)
    Edge 2: (0,3)
    Edge 3: (1,2)
    Edge 4: (1,3)
    Edge 5: (2,3)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Gap filling thresholds (as fractions of edge length)
MIN_PROJECTION_DISTANCE_FRACTION = 0.1   # Minimum distance from boundary to projection point
PROJECTION_OFFSET_FRACTION = 0.3         # Offset distance when adjusting too-close projections

# Triangle area threshold (as fraction of median area)
MIN_TRIANGLE_AREA_FRACTION = 0.01        # 1% of median area considered degenerate

# Distance threshold multiplier for auto-computing gap fill threshold
# Higher value = more aggressive gap filling (catches edges farther from target surfaces)
DISTANCE_THRESHOLD_EDGE_MULTIPLIER = 3.0  # threshold = median_edge_length * this value

# Small polyline removal thresholds
DEFAULT_MIN_LOOP_PERIMETER = 4.0  # Default minimum perimeter in mm for closed loops to keep

# Tubular section detection thresholds
# A "tubular" section is a small-diameter closed boundary loop
# We detect these by comparing perimeter to the "span" (max distance across loop)
TUBULAR_ASPECT_RATIO_THRESHOLD = 6.0  # perimeter / span ratio above this is tubular
TUBULAR_MAX_PERIMETER = 15.0  # Maximum perimeter (mm) to consider for tubular detection
MIN_TUBULAR_VERTICES = 6  # Minimum vertices in loop to consider tubular

# Island removal for primary surfaces
PRIMARY_MIN_ISLAND_TRIANGLES = 10  # Minimum triangles to keep an island (primary surface)
PRIMARY_MIN_ISLAND_AREA_FRACTION = 0.01  # Islands with area < 1% of total are removed

# Floating edge detection and filling thresholds
# After smoothing, boundary vertices are re-projected to the part surface, but
# edges connecting them may "float" away from the part. We detect these floating
# edges and create fill triangles to close the gaps.
#
# An edge is "floating" if its MIDPOINT is not on the part surface.
# This is mathematically optimal because the midpoint of a chord is where
# maximum deviation from a curve occurs.
#
# For CSG operations to produce manifold results, there must be NO gaps between
# the parting surface and the part mesh. We use extremely tight tolerances.
FLOATING_EDGE_TOLERANCE_FRACTION = 0.001  # Edge is floating if midpoint is > 0.1% of edge length from part
FLOATING_EDGE_MIN_TOLERANCE = 1e-6        # Minimum tolerance: 1 nanometer (essentially zero for CSG)


# =============================================================================
# MARCHING TETRAHEDRA LOOKUP TABLES
# =============================================================================

# Edge definitions: edge index -> (vertex_i, vertex_j)
TET_EDGES = [
    (0, 1),  # Edge 0
    (0, 2),  # Edge 1
    (0, 3),  # Edge 2
    (1, 2),  # Edge 3
    (1, 3),  # Edge 4
    (2, 3),  # Edge 5
]


def _build_marching_tet_table() -> Dict[int, List[Tuple]]:
    """
    Build the marching tetrahedra lookup table for BINARY (2-class) labeling.
    
    In a proper binary system where every vertex is labeled H1 or H2:
    - 0-edge config: All 4 vertices same label → no surface
    - 3-edge config: 1 vertex isolated (3+1 split) → single triangle
    - 4-edge config: 2 vs 2 vertices (2+2 split) → quad (2 triangles)
    
    5-edge and 6-edge configs should NOT occur in a binary system (they require
    3+ classes). If they do occur, it indicates a labeling bug upstream.
    We handle them as warnings but still attempt reasonable triangulation.
    
    Edge numbering:
        Edge 0: (v0, v1)    Edge 3: (v1, v2)
        Edge 1: (v0, v2)    Edge 4: (v1, v3)
        Edge 2: (v0, v3)    Edge 5: (v2, v3)
    
    Returns:
        Dictionary mapping 6-bit config -> list of triangle specs
        Each triangle is (edge0, edge1, edge2) using local edge indices
    """
    table = {}
    
    # Initialize all 64 configs to empty
    for i in range(64):
        table[i] = []
    
    # ===========================================================================
    # 3-EDGE CONFIGS: Single triangle (one vertex isolated from the other three)
    # These are the ONLY valid 3-edge configs in a binary system.
    # ===========================================================================
    # Vertex 0 isolated (label differs from v1,v2,v3): edges 0,1,2 cut
    # Config 7 = 000111 (binary)
    table[7] = [(0, 1, 2)]
    
    # Vertex 1 isolated: edges 0,3,4 cut
    # Config 25 = 011001
    table[25] = [(0, 3, 4)]
    
    # Vertex 2 isolated: edges 1,3,5 cut
    # Config 42 = 101010
    table[42] = [(1, 3, 5)]
    
    # Vertex 3 isolated: edges 2,4,5 cut
    # Config 52 = 110100
    table[52] = [(2, 4, 5)]
    
    # ===========================================================================
    # 4-EDGE CONFIGS: Quadrilateral surface (two triangles)
    # Two vertices on one side, two on the other.
    # 
    # CRITICAL: Diagonal choice must be CONSISTENT across adjacent tetrahedra.
    # Per Nielson & Franke paper: "After this is done for each prism, it is seen 
    # that the diagonal on the separating quadrilateral is arbitrary, and we 
    # choose the m_jk to m_il segment."
    #
    # For each 4-edge config, we have a quad with 4 cut points. We need to
    # triangulate consistently. We use the rule: connect cut points on edges
    # that share the LOWEST-indexed vertex (v0).
    #
    # Quad vertex order (CCW when viewed from H2 side):
    # For config 30: m1(v0-v2) → m3(v1-v2) → m4(v1-v3) → m2(v0-v3)
    # Diagonal through m1-m4: triangles (1,3,4) and (1,4,2)
    # ===========================================================================
    # Split {v0,v1} vs {v2,v3}: edges 1,2,3,4 cut (edges 0,5 uncut)
    # Cut points: m1(v0-v2), m2(v0-v3), m3(v1-v2), m4(v1-v3)
    # Quad order: m1 → m3 → m4 → m2
    # Config 30 = 011110
    # Diagonal m1-m4: triangles (m1,m3,m4) and (m1,m4,m2) = (1,3,4) and (1,4,2)
    table[30] = [(1, 3, 4), (1, 4, 2)]
    
    # Split {v0,v2} vs {v1,v3}: edges 0,2,3,5 cut (edges 1,4 uncut)
    # Cut points: m0(v0-v1), m2(v0-v3), m3(v1-v2), m5(v2-v3)
    # Quad order: m0 → m3 → m5 → m2
    # Config 45 = 101101
    # Diagonal m0-m5: triangles (m0,m3,m5) and (m0,m5,m2) = (0,3,5) and (0,5,2)
    table[45] = [(0, 3, 5), (0, 5, 2)]
    
    # Split {v0,v3} vs {v1,v2}: edges 0,1,4,5 cut (edges 2,3 uncut)
    # Cut points: m0(v0-v1), m1(v0-v2), m4(v1-v3), m5(v2-v3)
    # Quad order: m0 → m4 → m5 → m1
    # Config 51 = 110011
    # Diagonal m0-m5: triangles (m0,m4,m5) and (m0,m5,m1) = (0,4,5) and (0,5,1)
    table[51] = [(0, 4, 5), (0, 5, 1)]
    
    # ===========================================================================
    # INVALID CONFIGS (should not occur in binary system)
    # These indicate a labeling bug. We mark them for warning but don't generate
    # triangles since doing so incorrectly causes self-intersections.
    # ===========================================================================
    # 5-edge configs (would require 3 classes)
    table[62] = 'SKIP'  # Edge 0 not cut - invalid in binary
    table[61] = 'SKIP'  # Edge 1 not cut
    table[59] = 'SKIP'  # Edge 2 not cut
    table[55] = 'SKIP'  # Edge 3 not cut
    table[47] = 'SKIP'  # Edge 4 not cut
    table[31] = 'SKIP'  # Edge 5 not cut
    
    # 6-edge config (would require 4 classes)
    table[63] = 'SKIP'  # All edges cut - invalid in binary
    
    return table


# Pre-build the lookup table
MARCHING_TET_TABLE = _build_marching_tet_table()


# =============================================================================
# SELF-INTERSECTION DETECTION
# =============================================================================

def detect_mesh_self_intersections(
    mesh: trimesh.Trimesh,
    sample_fraction: float = 1.0,
    max_pairs_to_check: int = 1000000
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Detect self-intersecting triangles in a mesh.
    
    Two triangles "self-intersect" if they are non-adjacent (don't share an edge
    or vertex) but their geometry overlaps in 3D space. This causes CSG operations
    to fail or produce incorrect results.
    
    Algorithm:
    1. Build adjacency map (which triangles share vertices)
    2. Use spatial hashing/BVH to find candidate triangle pairs
    3. For non-adjacent pairs with overlapping bboxes, test intersection
    
    Args:
        mesh: The mesh to check for self-intersections
        sample_fraction: Fraction of triangles to check (1.0 = all). Use < 1.0 for
            large meshes to get a fast estimate.
        max_pairs_to_check: Maximum number of triangle pairs to test (for performance)
        
    Returns:
        Tuple of (count, pairs) where:
        - count: Number of self-intersecting triangle pairs found
        - pairs: List of (face_idx_1, face_idx_2) pairs that intersect
    """
    import time
    start = time.time()
    
    if mesh is None or len(mesh.faces) == 0:
        return 0, []
    
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)
    
    if n_faces < 2:
        return 0, []
    
    # Step 1: Build vertex-to-face adjacency
    # face_idx is adjacent to another if they share at least one vertex
    vertex_to_faces: Dict[int, Set[int]] = {}
    for fi, face in enumerate(faces):
        for vi in face:
            if vi not in vertex_to_faces:
                vertex_to_faces[vi] = set()
            vertex_to_faces[vi].add(fi)
    
    # Build face adjacency set (faces that share a vertex are adjacent)
    face_adjacent: Dict[int, Set[int]] = {fi: set() for fi in range(n_faces)}
    for vi, face_set in vertex_to_faces.items():
        for fi in face_set:
            face_adjacent[fi].update(face_set)
    # Remove self-adjacency
    for fi in range(n_faces):
        face_adjacent[fi].discard(fi)
    
    # Step 2: Compute per-face bounding boxes for spatial filtering
    tri_verts = vertices[faces]  # (F, 3, 3)
    bbox_min = tri_verts.min(axis=1)  # (F, 3)
    bbox_max = tri_verts.max(axis=1)  # (F, 3)
    
    # Step 3: Find candidate pairs using bbox overlap
    # For efficiency, use sampling if mesh is large
    if sample_fraction < 1.0 and n_faces > 100:
        n_sample = max(int(n_faces * sample_fraction), 50)
        sample_indices = np.random.choice(n_faces, size=n_sample, replace=False)
    else:
        sample_indices = np.arange(n_faces)
    
    intersecting_pairs = []
    pairs_checked = 0
    
    for i_idx, fi in enumerate(sample_indices):
        if pairs_checked >= max_pairs_to_check:
            break
            
        # Get triangle fi's bbox
        fi_min = bbox_min[fi]
        fi_max = bbox_max[fi]
        
        # Find candidate triangles with overlapping bbox (only check fj > fi to avoid duplicates)
        for fj in range(fi + 1, n_faces):
            if pairs_checked >= max_pairs_to_check:
                break
                
            # Skip adjacent triangles (they share vertices, so they "touch" but don't "intersect")
            if fj in face_adjacent[fi]:
                continue
            
            # Quick bbox overlap check
            fj_min = bbox_min[fj]
            fj_max = bbox_max[fj]
            
            # Check if bboxes overlap in all 3 dimensions
            if (fi_max[0] < fj_min[0] or fj_max[0] < fi_min[0] or
                fi_max[1] < fj_min[1] or fj_max[1] < fi_min[1] or
                fi_max[2] < fj_min[2] or fj_max[2] < fi_min[2]):
                continue  # No overlap
            
            pairs_checked += 1
            
            # Bboxes overlap - do full triangle-triangle intersection test
            tri_i = (vertices[faces[fi, 0]], vertices[faces[fi, 1]], vertices[faces[fi, 2]])
            tri_j = (vertices[faces[fj, 0]], vertices[faces[fj, 1]], vertices[faces[fj, 2]])
            
            if _triangles_intersect_fast(tri_i, tri_j):
                intersecting_pairs.append((fi, fj))
    
    elapsed = time.time() - start
    
    if intersecting_pairs:
        logger.warning(f"SELF-INTERSECTION: Found {len(intersecting_pairs)} intersecting triangle pairs "
                      f"(checked {pairs_checked} pairs in {elapsed*1000:.1f}ms)")
    else:
        logger.debug(f"Self-intersection check: OK (checked {pairs_checked} pairs in {elapsed*1000:.1f}ms)")
    
    return len(intersecting_pairs), intersecting_pairs


def _triangles_intersect_fast(
    tri1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tri2: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> bool:
    """
    Fast check if two triangles intersect using Möller's algorithm.
    
    Two triangles intersect if:
    1. Any edge of tri1 passes through tri2, OR
    2. Any edge of tri2 passes through tri1
    
    This is the same logic as _triangles_intersect in tetrahedral_mesh.py,
    duplicated here to avoid circular imports.
    """
    v0, v1, v2 = tri1
    u0, u1, u2 = tri2
    
    # Check edges of tri1 against tri2
    if _segment_intersects_triangle_fast(v0, v1, u0, u1, u2):
        return True
    if _segment_intersects_triangle_fast(v1, v2, u0, u1, u2):
        return True
    if _segment_intersects_triangle_fast(v2, v0, u0, u1, u2):
        return True
    
    # Check edges of tri2 against tri1
    if _segment_intersects_triangle_fast(u0, u1, v0, v1, v2):
        return True
    if _segment_intersects_triangle_fast(u1, u2, v0, v1, v2):
        return True
    if _segment_intersects_triangle_fast(u2, u0, v0, v1, v2):
        return True
    
    return False


def _segment_intersects_triangle_fast(
    p0: np.ndarray, p1: np.ndarray,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> bool:
    """
    Möller–Trumbore ray-triangle intersection test for a line segment.
    Returns True if segment (p0, p1) intersects triangle (v0, v1, v2).
    """
    EPSILON = 1e-9
    
    direction = p1 - p0
    seg_length = np.linalg.norm(direction)
    if seg_length < EPSILON:
        return False
    direction = direction / seg_length
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < EPSILON:
        return False  # Ray parallel to triangle
    
    f = 1.0 / a
    s = p0 - v0
    u = f * np.dot(s, h)
    
    if u < -EPSILON or u > 1.0 + EPSILON:
        return False
    
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    
    if v < -EPSILON or (u + v) > 1.0 + EPSILON:
        return False
    
    t = f * np.dot(edge2, q)
    
    # Check if intersection is within segment bounds
    if t < -EPSILON or t > seg_length + EPSILON:
        return False
    
    return True


# =============================================================================
# PARTING SURFACE EXTRACTION
# =============================================================================

@dataclass
class PartingSurfaceResult:
    """Result of parting surface extraction."""
    
    # The extracted parting surface mesh
    mesh: Optional[trimesh.Trimesh] = None
    
    # Surface vertices (V x 3) - midpoints of cut edges
    vertices: Optional[np.ndarray] = None
    
    # Surface faces (F x 3) - triangles
    faces: Optional[np.ndarray] = None
    
    # Mapping from surface vertex index to global edge index
    # vertex_to_edge[sv] = global edge index that this vertex sits on
    vertex_to_edge: Optional[np.ndarray] = None
    
    # Boundary type for each vertex (per paper Section 4.3-4.4):
    # -1 = placed on part surface M (INNER boundary - re-project to part)
    #  0 = interior (midpoint - no boundary constraint)
    #  1 = placed on hull H1 boundary (OUTER boundary - re-project to hull)
    #  2 = placed on hull H2 boundary (OUTER boundary - re-project to hull)
    # This is used during smoothing to correctly re-project boundary vertices
    vertex_boundary_type: Optional[np.ndarray] = None
    
    # Statistics
    num_vertices: int = 0
    num_faces: int = 0
    num_tets_processed: int = 0
    num_tets_contributing: int = 0
    
    # Self-intersection detection results
    # Count of self-intersecting triangle pairs (0 = clean mesh)
    self_intersection_count: int = 0
    # List of (face_idx_1, face_idx_2) pairs that intersect
    self_intersecting_pairs: Optional[List[Tuple[int, int]]] = None
    
    # Timing
    extraction_time_ms: float = 0.0


def extract_parting_surface(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    edges: np.ndarray,
    cut_edge_flags: np.ndarray,
    tet_edge_indices: np.ndarray,
    use_original_vertices: bool = True,
    vertices_original: Optional[np.ndarray] = None,
    boundary_labels: Optional[np.ndarray] = None,
    vertex_mold_labels: Optional[np.ndarray] = None,
    vertex_escape_distances: Optional[np.ndarray] = None,
    use_label_derived_cuts: bool = True
) -> PartingSurfaceResult:
    """
    Extract the parting surface mesh using Marching Tetrahedra.
    
    Per paper Section 4.3: "The triangulated surface C encoding the cut layout 
    is composed using a set of patches that are interconnected by chains of 
    non-manifold edges that are bounded by construction by the object surface 
    mesh M and the external boundary ∂H."
    
    For each tetrahedron:
    1. Look up which of its 6 edges are cut (using cut_edge_flags)
    2. Build a 6-bit configuration index
    3. Look up triangles from MARCHING_TET_TABLE
    4. Emit triangles using boundary-aware cut point placement:
       - If edge touches part surface M (boundary_label == -1): use M vertex position
       - If edge touches hull boundary ∂H (boundary_label == 1 or 2): use ∂H vertex position
       - Otherwise: use edge midpoint
    
    This ensures the parting surface is "bounded by construction" by M and ∂H.
    
    Cut Point Placement (per Nielson & Franke 1997):
        If vertex_escape_distances is provided, we use WEIGHTED interpolation to place
        cut points at the estimated location where the classification actually changes:
        
            t = d0 / (d0 + d1)
            cut_point = (1-t) * v0 + t * v1
        
        where d0, d1 are the geodesic escape distances of the two vertices.
        This places the cut point closer to the vertex with the shorter escape path,
        which better approximates the true parting surface location.
        
        If vertex_escape_distances is None, we fall back to geometric midpoint (t=0.5).
    
    Args:
        vertices: (N, 3) vertex positions (inflated mesh)
        tetrahedra: (M, 4) tetrahedron vertex indices
        edges: (E, 2) global edge list
        cut_edge_flags: (E,) boolean flags for cut edges
        tet_edge_indices: (M, 6) tet-to-edge mapping
        use_original_vertices: If True, use vertices_original for surface construction
        vertices_original: (N, 3) original (non-inflated) vertex positions
        boundary_labels: (N,) vertex boundary labels: 0=interior, 1=H1, 2=H2, -1=part surface
        vertex_mold_labels: (N,) mold half labels: 1=H1, 2=H2 for all vertices
        vertex_escape_distances: (N,) geodesic distance from each vertex to its escape boundary.
            Used for weighted cut point placement per Nielson & Franke. If None, uses midpoint.
    
    Returns:
        PartingSurfaceResult with the extracted surface mesh
    """
    import time
    start = time.time()
    
    result = PartingSurfaceResult()
    result.num_tets_processed = len(tetrahedra)
    
    # Choose which vertex positions to use for surface construction
    if use_original_vertices and vertices_original is not None:
        verts = vertices_original
        logger.info("Using original (non-inflated) vertices for parting surface")
    else:
        verts = vertices
        logger.info("Using current (possibly inflated) vertices for parting surface")
    
    # CRITICAL FIX: For PRIMARY surfaces, if vertex_mold_labels is provided, 
    # recompute cut edges directly from the labels. This ensures cut edges are 
    # CONSISTENT with the binary labeling.
    # 
    # For SECONDARY surfaces, we MUST use the pre-computed cut_edge_flags directly
    # because secondary cuts connect vertices with the SAME mold label (both H1 or
    # both H2) - deriving cuts from labels would produce zero edges!
    if vertex_mold_labels is not None and use_label_derived_cuts:
        logger.info("Recomputing cut edges from vertex_mold_labels for PRIMARY surface consistency...")
        n_edges = len(edges)
        cut_edge_flags = np.zeros(n_edges, dtype=np.int8)
        for e_idx in range(n_edges):
            v0, v1 = edges[e_idx]
            l0 = vertex_mold_labels[v0]
            l1 = vertex_mold_labels[v1]
            # Edge is cut if labels differ (1 vs 2)
            if l0 != l1 and l0 in (1, 2) and l1 in (1, 2):
                cut_edge_flags[e_idx] = 1
        logger.info(f"Computed {np.sum(cut_edge_flags)} cut edges from vertex labels")
    elif not use_label_derived_cuts:
        logger.info("Using pre-computed cut_edge_flags (SECONDARY surface mode)")
    
    # Count cut edges
    n_cut = np.sum(cut_edge_flags)
    logger.info(f"Extracting parting surface from {len(tetrahedra)} tets, {n_cut} cut edges")
    
    if n_cut == 0:
        logger.warning("No cut edges found - cannot extract parting surface")
        result.extraction_time_ms = (time.time() - start) * 1000
        return result
    
    # Step 1: Compute cut point positions for all cut edges
    # Per paper Section 4.3: surface is "bounded by construction" by M and ∂H
    # For edges touching a boundary surface, place vertex ON that surface (not at midpoint)
    cut_edge_indices = np.where(cut_edge_flags)[0]
    
    # Map from global edge index to surface vertex index
    edge_to_surface_vertex = {int(e): i for i, e in enumerate(cut_edge_indices)}
    
    # Compute cut point positions using WEIGHTED interpolation if distances available
    # Per Nielson & Franke 1997: "In some applications where there is additional 
    # information on which to base any bias or adjustment... weights may be used"
    # We use the geodesic escape distances to place cut points at the estimated
    # location where the classification changes (d0 = d1 boundary).
    surface_vertices = np.zeros((len(cut_edge_indices), 3), dtype=np.float64)
    
    # Track boundary type for each vertex (per paper Section 4.4):
    # -1 = on part surface M (INNER boundary - re-project to part during smoothing)
    #  0 = interior (no boundary constraint)
    #  1 = on hull H1 boundary (OUTER boundary - re-project to hull)
    #  2 = on hull H2 boundary (OUTER boundary - re-project to hull)
    # NOTE: This is for LATER re-projection, NOT for initial placement!
    vertex_boundary_type = np.zeros(len(cut_edge_indices), dtype=np.int8)
    
    # Track statistics for logging
    n_part_boundary = 0   # Cut points near part surface M (will re-project later)
    n_hull_boundary = 0   # Cut points near hull boundary ∂H (will re-project later)
    n_interior = 0        # Cut points in interior (no re-projection needed)
    n_weighted = 0        # Cut points placed using weighted interpolation
    n_midpoint = 0        # Cut points placed at midpoint (fallback)
    
    # Check if weighted placement is possible
    use_weighted = (vertex_escape_distances is not None and 
                    len(vertex_escape_distances) == len(vertices))
    if use_weighted:
        logger.info("Using WEIGHTED cut point placement (per Nielson & Franke 1997)")
    else:
        logger.info("Using MIDPOINT cut point placement (no escape distances available)")
    
    for i, e_idx in enumerate(cut_edge_indices):
        v0, v1 = edges[e_idx]
        
        # Compute cut point position using weighted interpolation if available
        # Per Nielson & Franke: t = d0 / (d0 + d1), cut_point = (1-t)*v0 + t*v1
        # This places the cut point at the estimated classification boundary
        if use_weighted:
            d0 = vertex_escape_distances[v0]
            d1 = vertex_escape_distances[v1]
            
            # Validate distances - both should be positive and finite
            if d0 > 0 and d1 > 0 and np.isfinite(d0) and np.isfinite(d1):
                # t = d0 / (d0 + d1) means cut point is closer to v0 when d0 is small
                # (v0 has shorter escape path → is closer to its boundary)
                t = d0 / (d0 + d1)
                # Clamp t to [0.1, 0.9] to avoid placing cut points too close to vertices
                # which could create degenerate triangles
                t = np.clip(t, 0.1, 0.9)
                surface_vertices[i] = (1.0 - t) * verts[v0] + t * verts[v1]
                n_weighted += 1
            else:
                # Fallback to midpoint if distances are invalid
                surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
                n_midpoint += 1
        else:
            # No distance data - use geometric midpoint (default per Nielson)
            surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
            n_midpoint += 1
        
        # Determine boundary type for LATER re-projection (not for placement)
        # boundary_labels: -1 = on part surface M, 0 = interior, 1 = on H1 hull, 2 = on H2 hull
        if boundary_labels is not None:
            bl0 = boundary_labels[v0]
            bl1 = boundary_labels[v1]
            
            # Priority: Part surface M (-1) > Hull (1,2) > Interior (0)
            # If either endpoint is on part surface, mark for re-projection to part M
            if bl0 == -1 or bl1 == -1:
                vertex_boundary_type[i] = -1  # Will re-project to part M later
                n_part_boundary += 1
            # Else if either endpoint is on hull boundary, mark for re-projection to hull
            elif bl0 in (1, 2) or bl1 in (1, 2):
                # Use the non-zero hull label (prefer H1 if both are hull vertices)
                if bl0 in (1, 2):
                    vertex_boundary_type[i] = bl0
                else:
                    vertex_boundary_type[i] = bl1
                n_hull_boundary += 1
            else:
                # Both interior - no re-projection needed
                vertex_boundary_type[i] = 0
                n_interior += 1
        else:
            # No boundary labels available
            vertex_boundary_type[i] = 0
            n_interior += 1
    
    logger.info(f"Cut point placement: {n_weighted} weighted, {n_midpoint} midpoint")
    logger.info(f"Cut point boundary types: {n_part_boundary} near part M, "
                f"{n_hull_boundary} near hull ∂H, {n_interior} interior")
    
    # Step 2: Process each tetrahedron using marching tetrahedra
    # 
    # For PRIMARY surfaces (use_label_derived_cuts=True): compute cut edges directly
    # from vertex_mold_labels. This GUARANTEES valid configurations (0, 3, or 4 edges only).
    # 
    # For SECONDARY surfaces (use_label_derived_cuts=False): use the pre-computed 
    # cut_edge_flags directly because secondary cuts connect same-label vertices.
    triangles = []
    tets_contributing = 0
    
    # Track configuration statistics for debugging
    config_counts = {}
    n_skipped_configs = 0  # Count of invalid 5/6-edge configs (indicates labeling issues)
    n_label_derived = 0    # Count of configs derived from vertex labels
    n_flag_derived = 0     # Count of configs derived from cut_edge_flags
    
    # Local edge definitions: (v0_local, v1_local) for each of 6 edges
    LOCAL_TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for t in range(len(tetrahedra)):
        tet_verts = tetrahedra[t]  # Shape (4,) - the 4 vertex indices
        tet_edges = tet_edge_indices[t]  # Shape (6,) - the 6 global edge indices
        
        # Determine configuration: use vertex_mold_labels for PRIMARY surfaces,
        # use cut_edge_flags directly for SECONDARY surfaces
        if vertex_mold_labels is not None and use_label_derived_cuts:
            # PRIMARY: Compute cut edges directly from vertex labels
            # An edge is cut if its endpoints have different labels (1 vs 2)
            config = 0
            for local_e in range(6):
                v0_local, v1_local = LOCAL_TET_EDGES[local_e]
                v0_global = tet_verts[v0_local]
                v1_global = tet_verts[v1_local]
                label0 = vertex_mold_labels[v0_global]
                label1 = vertex_mold_labels[v1_global]
                
                # Edge is cut if labels differ (1 vs 2)
                if label0 != label1 and label0 in (1, 2) and label1 in (1, 2):
                    config |= (1 << local_e)
            n_label_derived += 1
        else:
            # SECONDARY (or fallback): use pre-computed cut_edge_flags
            config = 0
            for local_e in range(6):
                global_e = tet_edges[local_e]
                if cut_edge_flags[global_e]:
                    config |= (1 << local_e)
            n_flag_derived += 1
        
        # Track config counts
        config_counts[config] = config_counts.get(config, 0) + 1
        
        # Look up triangles from table
        table_entry = MARCHING_TET_TABLE.get(config, [])
        
        # Build local edge-to-vertex mapping for this tet
        local_edge_to_vertex = {}
        for local_e in range(6):
            global_e = int(tet_edges[local_e])
            if global_e in edge_to_surface_vertex:
                local_edge_to_vertex[local_e] = edge_to_surface_vertex[global_e]
        
        if table_entry == 'SKIP':
            # =================================================================
            # INVALID CONFIGURATION (5 or 6 edges cut)
            # This should not occur in a proper binary labeling system.
            # It indicates that some vertices are unlabeled or have inconsistent labels.
            # Skip to avoid generating broken geometry.
            # =================================================================
            n_skipped_configs += 1
            continue
            
        elif table_entry:
            # =================================================================
            # VALID CONFIGURATIONS (3 or 4 edges cut)
            # Generate triangles using mid-edge vertices
            # Orient triangles so normal points from H1 towards H2
            # =================================================================
            tets_contributing += 1
            
            for local_tri in table_entry:
                # Convert local edge indices to surface vertex indices
                tri_verts = []
                valid = True
                
                for local_e in local_tri:
                    global_e = int(tet_edges[local_e])
                    if global_e in edge_to_surface_vertex:
                        tri_verts.append(edge_to_surface_vertex[global_e])
                    else:
                        # Edge not cut - shouldn't happen with valid table
                        valid = False
                        break
                
                if valid and len(tri_verts) == 3:
                    # Orient triangle normals consistently
                    # For PRIMARY surfaces: normal points from H1 towards H2
                    # For SECONDARY surfaces: normal points towards the "secondary side"
                    #   (we use escape distances as a proxy for "side")
                    if vertex_mold_labels is not None and use_label_derived_cuts:
                        # PRIMARY: Get positions of the 3 cut points
                        p0 = surface_vertices[tri_verts[0]]
                        p1 = surface_vertices[tri_verts[1]]
                        p2 = surface_vertices[tri_verts[2]]
                        
                        # Compute triangle centroid and normal
                        tri_center = (p0 + p1 + p2) / 3.0
                        edge1 = p1 - p0
                        edge2 = p2 - p0
                        normal = np.cross(edge1, edge2)
                        
                        # Find the H2 centroid in this tet (vertices with label 2)
                        h2_positions = []
                        for v_local in range(4):
                            v_global = tet_verts[v_local]
                            if vertex_mold_labels[v_global] == 2:
                                h2_positions.append(verts[v_global])
                        
                        if len(h2_positions) > 0:
                            h2_centroid = np.mean(h2_positions, axis=0)
                            # Vector from triangle center towards H2
                            to_h2 = h2_centroid - tri_center
                            
                            # If normal points away from H2, flip the triangle
                            if np.dot(normal, to_h2) < 0:
                                tri_verts = [tri_verts[0], tri_verts[2], tri_verts[1]]  # Flip winding
                    else:
                        # SECONDARY: Orient based on escape distances if available
                        # The side with LONGER escape distance should be the "outside"
                        # (secondary cuts occur where paths diverge around the part M)
                        if vertex_escape_distances is not None:
                            p0 = surface_vertices[tri_verts[0]]
                            p1 = surface_vertices[tri_verts[1]]
                            p2 = surface_vertices[tri_verts[2]]
                            
                            tri_center = (p0 + p1 + p2) / 3.0
                            edge1 = p1 - p0
                            edge2 = p2 - p0
                            normal = np.cross(edge1, edge2)
                            
                            # For secondary cuts, find the tet vertex with LONGEST escape distance
                            # and orient AWAY from it (it's on the "far side" of the cut)
                            max_dist = -np.inf
                            max_dist_pos = None
                            for v_local in range(4):
                                v_global = tet_verts[v_local]
                                d = vertex_escape_distances[v_global]
                                if np.isfinite(d) and d > max_dist:
                                    max_dist = d
                                    max_dist_pos = verts[v_global]
                            
                            if max_dist_pos is not None:
                                to_far = max_dist_pos - tri_center
                                # Normal should point AWAY from the far vertex
                                if np.dot(normal, to_far) > 0:
                                    tri_verts = [tri_verts[0], tri_verts[2], tri_verts[1]]  # Flip winding
                    
                    triangles.append(tri_verts)
    
    # Store vertex_to_edge now (before any vertex merging which would invalidate it)
    # The mapping will be invalidated by any vertex merge later
    result.vertex_to_edge = cut_edge_indices.copy()
    
    # === VALIDATION: Check that all triangle indices are within bounds ===
    n_total_vertices = len(surface_vertices)
    invalid_triangles = []
    for tri_idx, tri in enumerate(triangles):
        for vi in tri:
            if vi < 0 or vi >= n_total_vertices:
                invalid_triangles.append((tri_idx, tri, vi))
    
    if invalid_triangles:
        logger.error(f"VALIDATION ERROR: {len(invalid_triangles)} triangles have out-of-bounds vertex indices "
                    f"(total vertices: {n_total_vertices})")
        for tri_idx, tri, bad_vi in invalid_triangles[:5]:  # Log first 5
            logger.error(f"  Triangle {tri_idx}: {tri} - vertex {bad_vi} out of bounds")
        # Filter out invalid triangles to prevent crash
        valid_triangle_mask = np.ones(len(triangles), dtype=bool)
        for tri_idx, _, _ in invalid_triangles:
            valid_triangle_mask[tri_idx] = False
        triangles = [t for i, t in enumerate(triangles) if valid_triangle_mask[i]]
        logger.warning(f"Removed {len(invalid_triangles)} invalid triangles, {len(triangles)} remain")
    
    # Log configuration statistics
    config_source = "vertex_mold_labels" if n_label_derived > 0 else "cut_edge_flags"
    logger.info(f"Configuration statistics ({len(config_counts)} unique configs, source: {config_source}):")
    invalid_config_tets = 0  # Count tets with invalid configs (1, 2, 5, 6 edges)
    for config in sorted(config_counts.keys()):
        n_edges = bin(config).count('1')
        table_entry = MARCHING_TET_TABLE.get(config, [])
        if table_entry == 'SKIP':
            status = "-> SKIPPED (invalid in binary system)"
            invalid_config_tets += config_counts[config]
        elif table_entry:
            status = "-> triangles"
        else:
            status = "-> EMPTY (no triangles)"
            if n_edges in (1, 2):
                invalid_config_tets += config_counts[config]
        if config_counts[config] > 10:  # Only log significant configs
            logger.info(f"  Config {config:2d} ({config:06b}, {n_edges} edges): {config_counts[config]:5d} tets {status}")
    
    if n_skipped_configs > 0:
        logger.warning(f"  {n_skipped_configs} tets had 5/6-edge configs (SKIPPED) - indicates labeling issues upstream")
    if invalid_config_tets > 0:
        logger.warning(f"  {invalid_config_tets} total tets with invalid/empty configs (1, 2, 5, or 6 edges)")
    
    result.num_tets_contributing = tets_contributing
    
    if not triangles:
        logger.warning("No triangles generated - check cut edge configuration")
        result.extraction_time_ms = (time.time() - start) * 1000
        return result
    
    # Step 3: Build the mesh
    faces = np.array(triangles, dtype=np.int64)
    
    # Remove degenerate triangles (where vertices are the same)
    valid_mask = (faces[:, 0] != faces[:, 1]) & \
                 (faces[:, 1] != faces[:, 2]) & \
                 (faces[:, 0] != faces[:, 2])
    n_degenerate_idx = np.sum(~valid_mask)
    faces = faces[valid_mask]
    
    if len(faces) == 0:
        logger.warning("All triangles were degenerate")
        result.extraction_time_ms = (time.time() - start) * 1000
        return result
    
    # Step 3b: Merge nearly-coincident vertices (per Bloomenthal paper)
    # "each pair of connected vertices whose distance is less than ε is replaced
    # by the average of the two vertices"
    # This helps eliminate thin/sliver triangles that cause visual artifacts.
    merge_epsilon = 1e-8  # Tight threshold for near-coincident points
    
    # Build KD-tree to find close vertex pairs
    from scipy.spatial import cKDTree
    tree = cKDTree(surface_vertices)
    pairs = tree.query_pairs(merge_epsilon)
    
    if pairs:
        # Build vertex merge mapping
        n_verts = len(surface_vertices)
        vertex_map = np.arange(n_verts)
        
        # Union-find for merging
        for i, j in pairs:
            # Find roots
            root_i = i
            while vertex_map[root_i] != root_i:
                root_i = vertex_map[root_i]
            root_j = j
            while vertex_map[root_j] != root_j:
                root_j = vertex_map[root_j]
            # Merge to lower index
            if root_i != root_j:
                vertex_map[max(root_i, root_j)] = min(root_i, root_j)
        
        # Flatten mapping
        for i in range(n_verts):
            root = i
            while vertex_map[root] != root:
                root = vertex_map[root]
            vertex_map[i] = root
        
        # Count unique vertices
        unique_roots = np.unique(vertex_map)
        if len(unique_roots) < n_verts:
            logger.debug(f"Merging {n_verts - len(unique_roots)} near-coincident vertices")
            
            # Remap vertices
            old_to_new = np.full(n_verts, -1, dtype=np.int64)
            old_to_new[unique_roots] = np.arange(len(unique_roots))
            
            new_surface_vertices = surface_vertices[unique_roots]
            new_vertex_boundary_type = vertex_boundary_type[unique_roots]
            
            # Remap face indices through union-find then to new indices
            new_faces = old_to_new[vertex_map[faces]]
            
            # Remove degenerate faces after merge
            valid_after_merge = (new_faces[:, 0] != new_faces[:, 1]) & \
                               (new_faces[:, 1] != new_faces[:, 2]) & \
                               (new_faces[:, 0] != new_faces[:, 2])
            n_degenerate_merge = np.sum(~valid_after_merge)
            new_faces = new_faces[valid_after_merge]
            
            if n_degenerate_merge > 0:
                logger.debug(f"Removed {n_degenerate_merge} triangles degenerate after vertex merge")
            
            surface_vertices = new_surface_vertices
            vertex_boundary_type = new_vertex_boundary_type
            faces = new_faces
    
    result.vertices = surface_vertices
    result.faces = faces
    result.vertex_boundary_type = vertex_boundary_type.copy()  # Store after any vertex merge
    result.num_vertices = len(surface_vertices)
    result.num_faces = len(faces)
    
    # Check for and FIX non-manifold edges (edges shared by more than 2 triangles)
    # Non-manifold edges cause self-intersections and prevent proper mesh operations
    surface_vertices, faces, vertex_boundary_type = _fix_non_manifold_edges(
        surface_vertices, faces, vertex_boundary_type
    )
    
    # Update result after non-manifold fix
    result.vertices = surface_vertices
    result.faces = faces
    result.vertex_boundary_type = vertex_boundary_type.copy()
    result.num_vertices = len(surface_vertices)
    result.num_faces = len(faces)
    
    # Create trimesh
    try:
        result.mesh = trimesh.Trimesh(
            vertices=surface_vertices,
            faces=faces,
            process=False  # Don't merge vertices or remove degenerates
        )
        
        # Apply local normal consistency repair before global fix_normals
        # This fixes locally flipped triangles that cause self-folding
        result.mesh, n_flipped = _repair_local_normal_consistency(result.mesh)
        if n_flipped > 0:
            logger.info(f"Flipped {n_flipped} locally inconsistent triangles")
        
        # Then apply global normal consistency
        result.mesh.fix_normals()
        
    except Exception as e:
        logger.error(f"Failed to create trimesh: {e}")
    
    # Step 4: Check for self-intersecting triangles
    # Self-intersections cause CSG operations (blade thickening) to fail
    if result.mesh is not None:
        si_count, si_pairs = detect_mesh_self_intersections(result.mesh)
        result.self_intersection_count = si_count
        result.self_intersecting_pairs = si_pairs
        
        if si_count > 0:
            logger.warning(f"PARTING SURFACE HAS {si_count} SELF-INTERSECTING TRIANGLE PAIRS")
            logger.warning("This may cause CSG operations (blade thickening) to fail!")
            # Log a few examples
            for fi, fj in si_pairs[:5]:
                logger.debug(f"  Faces {fi} and {fj} intersect")
    
    result.extraction_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Parting surface: {result.num_vertices} vertices, {result.num_faces} faces "
                f"from {result.num_tets_contributing}/{result.num_tets_processed} tets "
                f"(self-intersections: {result.self_intersection_count}) "
                f"in {result.extraction_time_ms:.1f}ms")
    
    return result


def _fix_non_manifold_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_boundary_type: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fix non-manifold edges by removing duplicate triangles that share the same edge.
    
    Non-manifold edges occur when more than 2 triangles share the same edge.
    This typically happens due to:
    1. Duplicate triangles generated from adjacent tetrahedra
    2. Vertex merging that incorrectly combines separate surface sheets
    
    The fix strategy:
    1. Identify edges shared by more than 2 triangles
    2. For each such edge, keep only the 2 triangles with most consistent normals
    3. Remove the extra triangles
    
    This is simpler and more robust than vertex duplication because it removes
    the erroneous geometry rather than trying to separate it.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        vertex_boundary_type: (N,) boundary type for each vertex
    
    Returns:
        Updated (vertices, faces, vertex_boundary_type) tuple
    """
    # Build edge-to-faces map
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(fi)
    
    # Find non-manifold edges
    non_manifold_edges = {e: fs for e, fs in edge_to_faces.items() if len(fs) > 2}
    
    if not non_manifold_edges:
        return vertices, faces, vertex_boundary_type
    
    logger.warning(f"Found {len(non_manifold_edges)} non-manifold edges - fixing...")
    for edge, face_list in list(non_manifold_edges.items())[:5]:
        logger.debug(f"  Edge {edge}: shared by {len(face_list)} triangles")
    
    # Compute face normals for comparison
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0  # Avoid division by zero
    face_normals = face_normals / norms
    
    # For each non-manifold edge, keep only the 2 best triangles
    # "Best" = the 2 triangles whose normals are most similar (likely from same surface sheet)
    faces_to_remove = set()
    
    for edge, face_list in non_manifold_edges.items():
        if len(face_list) <= 2:
            continue
        
        # Find the pair of faces with most similar normals (same sheet)
        # For a proper manifold edge, the 2 triangles should have similar normals
        # (pointing roughly the same direction on the surface)
        best_pair = None
        best_similarity = -2.0
        
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                fi, fj = face_list[i], face_list[j]
                similarity = np.dot(face_normals[fi], face_normals[fj])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (fi, fj)
        
        # Mark all faces except the best pair for removal
        for fi in face_list:
            if fi not in best_pair:
                faces_to_remove.add(fi)
    
    if faces_to_remove:
        logger.info(f"Removing {len(faces_to_remove)} duplicate triangles from non-manifold edges")
        
        # Create mask for faces to keep
        keep_mask = np.ones(len(faces), dtype=bool)
        keep_mask[list(faces_to_remove)] = False
        faces = faces[keep_mask]
        
        # Remove unreferenced vertices
        referenced = np.unique(faces.ravel())
        if len(referenced) < len(vertices):
            old_to_new = np.full(len(vertices), -1, dtype=np.int64)
            old_to_new[referenced] = np.arange(len(referenced))
            
            vertices = vertices[referenced]
            vertex_boundary_type = vertex_boundary_type[referenced]
            faces = old_to_new[faces]
    
    return vertices, faces, vertex_boundary_type


def _repair_local_normal_consistency(mesh: trimesh.Trimesh, max_iterations: int = 5) -> Tuple[trimesh.Trimesh, int]:
    """
    Repair normal consistency using BFS propagation from a seed triangle.
    
    This is more robust than majority-voting because it ensures global consistency
    by propagating orientation from a known-good starting point.
    
    The algorithm:
    1. Build face adjacency graph (faces sharing edges)
    2. For each connected component:
       a. Pick a seed face (the one with most area, likely to be reliable)
       b. BFS propagate orientation: if neighbor disagrees, flip it
    3. After propagation, use majority-vote to fix any remaining edge cases
    
    Args:
        mesh: Input mesh with potentially inconsistent normals
        max_iterations: Maximum iterations for final majority-vote cleanup
    
    Returns:
        Tuple of (repaired_mesh, num_triangles_flipped)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    n_faces = len(faces)
    
    # Build edge-to-faces adjacency
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(fi)
    
    # Build face adjacency list
    face_neighbors = [[] for _ in range(n_faces)]
    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            fi, fj = face_list
            face_neighbors[fi].append(fj)
            face_neighbors[fj].append(fi)
    
    # Track which faces have been visited and total flips
    visited = np.zeros(n_faces, dtype=bool)
    total_flipped = 0
    
    # Process each connected component
    component_id = 0
    for start_face in range(n_faces):
        if visited[start_face]:
            continue
        
        # Find all faces in this component using BFS
        component_faces = []
        queue = [start_face]
        visited[start_face] = True
        
        while queue:
            fi = queue.pop(0)
            component_faces.append(fi)
            for fj in face_neighbors[fi]:
                if not visited[fj]:
                    visited[fj] = True
                    queue.append(fj)
        
        if len(component_faces) == 0:
            continue
        
        component_id += 1
        
        # Choose seed as largest triangle in component (most reliable normal)
        temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        component_areas = temp_mesh.area_faces[component_faces]
        seed_idx = component_faces[np.argmax(component_areas)]
        
        # BFS propagate orientation from seed
        propagation_visited = np.zeros(n_faces, dtype=bool)
        propagation_queue = [seed_idx]
        propagation_visited[seed_idx] = True
        component_flipped = 0
        
        while propagation_queue:
            fi = propagation_queue.pop(0)
            face_i = faces[fi]
            
            # Compute normal for face i
            p0 = vertices[face_i[0]]
            p1 = vertices[face_i[1]]
            p2 = vertices[face_i[2]]
            normal_i = np.cross(p1 - p0, p2 - p0)
            norm_len = np.linalg.norm(normal_i)
            if norm_len > 1e-10:
                normal_i = normal_i / norm_len
            
            for fj in face_neighbors[fi]:
                if propagation_visited[fj]:
                    continue
                propagation_visited[fj] = True
                propagation_queue.append(fj)
                
                # Compute normal for face j
                face_j = faces[fj]
                q0 = vertices[face_j[0]]
                q1 = vertices[face_j[1]]
                q2 = vertices[face_j[2]]
                normal_j = np.cross(q1 - q0, q2 - q0)
                norm_len = np.linalg.norm(normal_j)
                if norm_len > 1e-10:
                    normal_j = normal_j / norm_len
                
                # Check if normals disagree
                if np.dot(normal_i, normal_j) < 0:
                    # Flip face j
                    faces[fj, 1], faces[fj, 2] = faces[fj, 2], faces[fj, 1]
                    component_flipped += 1
        
        total_flipped += component_flipped
        
        if component_flipped > 0:
            logger.debug(f"Component {component_id}: propagation flipped {component_flipped}/{len(component_faces)} triangles")
    
    # Final cleanup: majority vote for any remaining inconsistencies
    # (can happen at T-junctions or non-manifold regions)
    for iteration in range(max_iterations):
        temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        face_normals = temp_mesh.face_normals
        
        # Rebuild edge-to-faces for current faces
        edge_to_faces = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = (min(v0, v1), max(v0, v1))
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append(fi)
        
        inconsistent = []
        for fi in range(n_faces):
            face = faces[fi]
            neighbor_normals = []
            
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = (min(v0, v1), max(v0, v1))
                for fj in edge_to_faces.get(edge_key, []):
                    if fj != fi:
                        neighbor_normals.append(face_normals[fj])
            
            if len(neighbor_normals) == 0:
                continue
            
            my_normal = face_normals[fi]
            n_agree = sum(1 for nn in neighbor_normals if np.dot(my_normal, nn) > 0)
            n_disagree = len(neighbor_normals) - n_agree
            
            if n_disagree > n_agree:
                inconsistent.append(fi)
        
        if len(inconsistent) == 0:
            break
        
        for fi in inconsistent:
            faces[fi, 1], faces[fi, 2] = faces[fi, 2], faces[fi, 1]
        
        total_flipped += len(inconsistent)
        logger.debug(f"Normal consistency cleanup iteration {iteration + 1}: flipped {len(inconsistent)} triangles")
    
    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return result_mesh, total_flipped


def extract_parting_surface_from_tet_result(
    tet_result,
    use_original_vertices: bool = True,
    prepare_data: bool = True,
    cut_type: str = 'both',
    extend_to_primary: bool = True,
    use_improved_secondary: bool = True,
    include_orphan_primary_in_secondary: bool = True
) -> PartingSurfaceResult:
    """
    Convenience function to extract parting surface from a TetrahedralMeshResult.
    
    Args:
        tet_result: TetrahedralMeshResult with cut edges computed
        use_original_vertices: If True, use non-inflated vertices
        prepare_data: If True, call prepare_parting_surface_data if needed
        cut_type: Which cut edges to use: 'primary', 'secondary', or 'both'
        extend_to_primary: If True and cut_type='secondary', extend secondary surface
                          to connect with primary surface in shared tetrahedra
        use_improved_secondary: If True, use the improved secondary membrane module
                               that properly handles primary-secondary junction tetrahedra
        include_orphan_primary_in_secondary: If True and cut_type='secondary', include
                                            orphan primary edges (edges not used by primary
                                            surface due to invalid tet configurations)
    
    Returns:
        PartingSurfaceResult
    """
    # Import here to avoid circular imports
    from . import tetrahedral_mesh as tm
    
    # Debug: Check vertex_mold_labels status
    if tet_result.vertex_mold_labels is not None:
        n_labels = len(tet_result.vertex_mold_labels)
        n_h1 = np.sum(tet_result.vertex_mold_labels == 1)
        n_h2 = np.sum(tet_result.vertex_mold_labels == 2)
        n_unlabeled = np.sum(tet_result.vertex_mold_labels == 0)
        logger.info(f"vertex_mold_labels available: {n_labels} total, {n_h1} H1, {n_h2} H2, {n_unlabeled} unlabeled")
    else:
        logger.warning("vertex_mold_labels is None - will use cut_edge_flags (may have inconsistencies)")
    
    # Prepare data structures if not already done
    if prepare_data and tet_result.tet_edge_indices is None:
        logger.info("Preparing parting surface data structures...")
        tet_result = tm.prepare_parting_surface_data(tet_result)
    
    # Validate required data
    if tet_result.tet_edge_indices is None:
        raise ValueError("tet_edge_indices not computed - run prepare_parting_surface_data first")
    
    # Determine which vertex labels to use for label-derived cuts
    # - PRIMARY: Use vertex_mold_labels (H1 vs H2)
    # - SECONDARY: Compute binary labels based on cut edge sidedness
    vertex_labels_for_cuts = None
    use_label_derived_cuts = False
    
    # Compute appropriate cut edge flags based on cut_type
    if cut_type == 'primary':
        if tet_result.primary_cut_edges is None:
            logger.warning("No primary cut edges found")
            return PartingSurfaceResult()
        cut_flags = tm.compute_primary_cut_edge_flags(
            tet_result.edges,
            tet_result.primary_cut_edges,
            tet_result.edge_to_index
        )
        logger.info(f"Extracting PRIMARY parting surface ({np.sum(cut_flags)} edges)")
        
        # PRIMARY: Use vertex_mold_labels for label-derived cuts
        use_label_derived_cuts = True
        vertex_labels_for_cuts = tet_result.vertex_mold_labels
        
    elif cut_type == 'secondary':
        if tet_result.secondary_cut_edges is None:
            logger.warning("No secondary cut edges found")
            return PartingSurfaceResult()
        
        # Determine which secondary edges to use
        # If include_orphan_primary_in_secondary=True, we enhance the secondary edges
        # with orphan primary edges (primary edges not used by primary surface)
        from . import secondary_membrane as sm
        
        if include_orphan_primary_in_secondary and use_improved_secondary:
            # Create enhanced secondary edges including orphan primary edges
            enhanced_secondary_edges = sm.create_enhanced_secondary_cut_edges(
                tet_result,
                include_orphan_primary=True
            )
            logger.info(f"Enhanced secondary edges: {len(enhanced_secondary_edges)} "
                       f"(original: {len(tet_result.secondary_cut_edges)})")
        else:
            enhanced_secondary_edges = tet_result.secondary_cut_edges
        
        if extend_to_primary and use_improved_secondary:
            # Use IMPROVED extended flags that only include adjacent primary edges
            cut_flags = sm.compute_extended_secondary_cut_flags_improved(
                tet_result,
                enhanced_secondary_edges,
                include_junction_primary_edges=True
            )
            logger.info(f"Extracting IMPROVED EXTENDED SECONDARY parting surface ({np.sum(cut_flags)} edges)")
        elif extend_to_primary:
            # Use original extended flags that include primary edges in shared tets
            cut_flags = tm.compute_extended_secondary_cut_edge_flags(
                tet_result.edges,
                tet_result.tetrahedra,
                tet_result.tet_edge_indices,
                tet_result.primary_cut_edges,
                enhanced_secondary_edges,
                tet_result.edge_to_index
            )
            logger.info(f"Extracting EXTENDED SECONDARY parting surface ({np.sum(cut_flags)} edges, connected to primary)")
        else:
            cut_flags = tm.compute_secondary_cut_edge_flags(
                tet_result.edges,
                enhanced_secondary_edges,
                tet_result.edge_to_index
            )
            logger.info(f"Extracting SECONDARY parting surface ({np.sum(cut_flags)} edges)")
        
        # SECONDARY: Use pre-computed cut_edge_flags directly (NOT label-derived)
        #
        # IMPORTANT: For secondary surfaces, we MUST use the pre-computed cut_edge_flags
        # directly because:
        # 1. Secondary cuts connect vertices with the SAME escape label (both H1 or both H2)
        # 2. Label-derived cuts look for vertices with DIFFERENT labels (1 vs 2)
        # 3. The BFS-based label propagation doesn't guarantee all vertices get labeled
        # 4. Using label-derived cuts for secondary causes edges to be missed when:
        #    - Vertices have label=0 (not involved in propagation)
        #    - The condition "label0 in (1,2) and label1 in (1,2)" fails
        #
        # The cut_edge_flags are already computed correctly based on:
        # - Original secondary_cut_edges (edges with same escape label but divergent paths)
        # - Enhanced edges including orphan primary edges (unused by primary surface)
        # - Junction primary edges (connecting secondary to primary at shared tets)
        #
        # The marching tetrahedra algorithm will use these flags directly to determine
        # which edges are cut, regardless of vertex labels.
        
        use_label_derived_cuts = False
        vertex_labels_for_cuts = None
        logger.info("SECONDARY surface: Using pre-computed cut_edge_flags directly "
                   f"({np.sum(cut_flags)} cut edges, NOT label-derived)")
        
    else:  # 'both'
        if tet_result.cut_edge_flags is None:
            raise ValueError("cut_edge_flags not computed - run prepare_parting_surface_data first")
        cut_flags = tet_result.cut_edge_flags
        logger.info(f"Extracting combined (PRIMARY + SECONDARY) parting surface ({np.sum(cut_flags)} edges)")
        
        # 'both': Use vertex_mold_labels for primary portion
        # Secondary edges may still produce inconsistent configs, but primary will be clean
        use_label_derived_cuts = True
        vertex_labels_for_cuts = tet_result.vertex_mold_labels
    
    # Build full vertex_escape_distances array from seed data if available
    # seed_distances is indexed by interior vertex index, we need global vertex index
    vertex_escape_distances = None
    if (tet_result.seed_distances is not None and 
        tet_result.seed_vertex_indices is not None and
        len(tet_result.seed_distances) == len(tet_result.seed_vertex_indices)):
        
        n_vertices = len(tet_result.vertices)
        vertex_escape_distances = np.full(n_vertices, np.inf, dtype=np.float64)
        
        # Map interior vertex distances to global array
        for idx, v_global in enumerate(tet_result.seed_vertex_indices):
            vertex_escape_distances[v_global] = tet_result.seed_distances[idx]
        
        # For boundary vertices (H1/H2), distance is 0 (they ARE on the boundary)
        if tet_result.boundary_labels is not None:
            boundary_mask = (tet_result.boundary_labels == 1) | (tet_result.boundary_labels == 2)
            vertex_escape_distances[boundary_mask] = 0.0
        
        n_finite = np.sum(np.isfinite(vertex_escape_distances))
        logger.info(f"Built vertex_escape_distances: {n_finite}/{n_vertices} vertices with valid distances")
    else:
        logger.info("No seed_distances available - will use midpoint cut point placement")
    
    return extract_parting_surface(
        vertices=tet_result.vertices,
        tetrahedra=tet_result.tetrahedra,
        edges=tet_result.edges,
        cut_edge_flags=cut_flags,
        tet_edge_indices=tet_result.tet_edge_indices,
        use_original_vertices=use_original_vertices,
        vertices_original=tet_result.vertices_original,
        boundary_labels=tet_result.boundary_labels,  # Pass for boundary-aware cut point placement
        vertex_mold_labels=vertex_labels_for_cuts,  # Pass computed labels (primary only, None for secondary)
        vertex_escape_distances=vertex_escape_distances,  # Pass for weighted cut point placement
        use_label_derived_cuts=use_label_derived_cuts  # True for PRIMARY, False for SECONDARY
    )


# =============================================================================
# SURFACE NETS STYLE SMOOTHING (Optional post-processing)
# =============================================================================

def smooth_parting_surface(
    surface: PartingSurfaceResult,
    iterations: int = 3,
    lambda_factor: float = 0.5
) -> PartingSurfaceResult:
    """
    Apply Laplacian smoothing to the parting surface.
    
    This is inspired by Surface Nets' approach of averaging positions
    to create smoother surfaces.
    
    Args:
        surface: PartingSurfaceResult to smooth
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing factor (0-1), higher = more smoothing
    
    Returns:
        New PartingSurfaceResult with smoothed vertices
    """
    if surface.mesh is None or surface.vertices is None:
        return surface
    
    import time
    start = time.time()
    
    vertices = surface.vertices.copy()
    faces = surface.faces
    
    # Build vertex adjacency from faces
    n_verts = len(vertices)
    neighbors = [set() for _ in range(n_verts)]
    
    for f in faces:
        for i in range(3):
            v = f[i]
            neighbors[v].add(f[(i+1) % 3])
            neighbors[v].add(f[(i+2) % 3])
    
    # Laplacian smoothing iterations
    for _ in range(iterations):
        new_vertices = vertices.copy()
        
        for v in range(n_verts):
            if neighbors[v]:
                neighbor_list = list(neighbors[v])
                centroid = np.mean(vertices[neighbor_list], axis=0)
                new_vertices[v] = vertices[v] + lambda_factor * (centroid - vertices[v])
        
        vertices = new_vertices
    
    # Create result - IMPORTANT: preserve vertex_boundary_type for floating edge detection!
    result = PartingSurfaceResult(
        vertices=vertices,
        faces=surface.faces,
        vertex_to_edge=surface.vertex_to_edge,
        vertex_boundary_type=surface.vertex_boundary_type,  # Preserve boundary type!
        num_vertices=surface.num_vertices,
        num_faces=surface.num_faces,
        num_tets_processed=surface.num_tets_processed,
        num_tets_contributing=surface.num_tets_contributing,
        extraction_time_ms=surface.extraction_time_ms
    )
    
    try:
        result.mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )
        result.mesh.fix_normals()
    except Exception as e:
        logger.error(f"Failed to create smoothed mesh: {e}")
    
    elapsed = (time.time() - start) * 1000
    logger.info(f"Smoothed parting surface ({iterations} iterations) in {elapsed:.1f}ms")
    
    return result


def repair_parting_surface(
    surface: PartingSurfaceResult,
    merge_vertices: bool = True,
    merge_threshold: float = 1e-8
) -> PartingSurfaceResult:
    """
    Clean the parting surface mesh by merging vertices and removing degenerate faces.
    
    This performs:
    1. Merge duplicate/close vertices (preserving boundary types)
    2. Remove degenerate faces (zero area)
    3. Remove small area triangles (< min_area)
    
    IMPORTANT: This function preserves vertex_boundary_type through the repair process
    by tracking vertex merging and keeping boundary types from merged vertices.
    
    Args:
        surface: PartingSurfaceResult to clean
        merge_vertices: Whether to merge close vertices
        merge_threshold: Distance threshold for vertex merging
    
    Returns:
        Cleaned PartingSurfaceResult with updated vertex_boundary_type
    """
    if surface.mesh is None:
        return surface
    
    import time
    start = time.time()
    
    try:
        mesh = surface.mesh.copy()
        initial_verts = len(mesh.vertices)
        initial_faces = len(mesh.faces)
        
        # Track vertex boundary type through the repair process
        has_boundary_type = surface.vertex_boundary_type is not None and len(surface.vertex_boundary_type) == initial_verts
        current_boundary_type = surface.vertex_boundary_type.copy() if has_boundary_type else None
        
        # Step 1: Merge close vertices
        if merge_vertices:
            # Before merge, we need to track which vertices get merged
            # trimesh.merge_vertices() doesn't give us a mapping, so we rebuild manually
            from scipy.spatial import cKDTree
            
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            # Find unique vertices
            tree = cKDTree(vertices)
            # pairs contains (i, j) pairs where dist(v_i, v_j) < merge_threshold
            # We need the first unique representative for each group
            used = np.zeros(len(vertices), dtype=bool)
            vertex_map = np.arange(len(vertices))  # Maps old indices to new indices
            new_vertices = []
            new_boundary_type = [] if has_boundary_type else None
            
            for i in range(len(vertices)):
                if used[i]:
                    continue
                    
                # Find all vertices close to this one
                neighbors = tree.query_ball_point(vertices[i], merge_threshold)
                
                # Mark all neighbors as used and map them to this vertex
                for j in neighbors:
                    used[j] = True
                    vertex_map[j] = len(new_vertices)
                
                new_vertices.append(vertices[i])
                
                # For boundary type, take the most "extreme" type from merged vertices
                # Priority: 1/2 (hull/outer) > -1 (part/inner) > 0 (interior)
                # 
                # IMPORTANT: Hull takes priority because for the PRIMARY parting surface,
                # we want the OUTER boundary to project to the hull ∂H, not the part M.
                # If a part vertex and hull vertex are close enough to merge, they're
                # at a seam edge and the outer boundary projection is more critical.
                if has_boundary_type:
                    merged_types = [current_boundary_type[j] for j in neighbors]
                    if 1 in merged_types or 2 in merged_types:
                        # Outer boundary (hull) takes priority
                        hull_types = [t for t in merged_types if t in (1, 2)]
                        new_boundary_type.append(hull_types[0])
                    elif -1 in merged_types:
                        # Inner boundary (part)
                        new_boundary_type.append(-1)
                    else:
                        new_boundary_type.append(0)
            
            new_vertices = np.array(new_vertices)
            new_faces = vertex_map[faces]
            
            # Remove degenerate faces (where two or more vertices are the same after merge)
            valid_faces_mask = (new_faces[:, 0] != new_faces[:, 1]) & \
                              (new_faces[:, 1] != new_faces[:, 2]) & \
                              (new_faces[:, 0] != new_faces[:, 2])
            new_faces = new_faces[valid_faces_mask]
            
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
            current_boundary_type = np.array(new_boundary_type, dtype=np.int8) if has_boundary_type else None
            
            logger.debug(f"Vertex merge: {initial_verts} -> {len(new_vertices)} vertices")
        
        # Step 2: Remove degenerate faces (zero area)
        valid_faces = mesh.nondegenerate_faces()
        if np.sum(~valid_faces) > 0:
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[valid_faces],
                process=False  # Don't auto-process - we handle vertex removal in Step 4
            )
            # Boundary type stays the same - only faces changed
        
        # Step 3: Remove very small triangles (potential self-intersection sources)
        if len(mesh.faces) > 0:
            areas = mesh.area_faces
            median_area = np.median(areas)
            min_area_threshold = median_area * MIN_TRIANGLE_AREA_FRACTION
            valid_area_mask = areas >= min_area_threshold
            small_removed = np.sum(~valid_area_mask)
            if small_removed > 0:
                logger.info(f"Removing {small_removed} very small triangles (< {min_area_threshold:.6f} area)")
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces[valid_area_mask],
                    process=False  # Don't auto-process - we handle vertex removal in Step 4
                )
        
        # Step 4: Remove unreferenced vertices (and update boundary type)
        if len(mesh.faces) > 0:
            referenced = np.unique(mesh.faces.ravel())
            if len(referenced) < len(mesh.vertices):
                # Need to remap vertices
                old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int64)
                old_to_new[referenced] = np.arange(len(referenced))
                
                new_vertices = mesh.vertices[referenced]
                new_faces = old_to_new[mesh.faces]
                
                mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
                
                # Update boundary type to match new vertex indices
                if has_boundary_type and current_boundary_type is not None:
                    current_boundary_type = current_boundary_type[referenced]
        
        result = PartingSurfaceResult(
            mesh=mesh,
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            vertex_to_edge=None,  # Invalidated by merge - mapping no longer valid
            vertex_boundary_type=current_boundary_type,  # PRESERVED through repair
            num_vertices=len(mesh.vertices),
            num_faces=len(mesh.faces),
            num_tets_processed=surface.num_tets_processed,
            num_tets_contributing=surface.num_tets_contributing,
            extraction_time_ms=surface.extraction_time_ms
        )
        
        elapsed = (time.time() - start) * 1000
        vert_diff = initial_verts - result.num_vertices
        face_diff = initial_faces - result.num_faces
        logger.info(f"Cleaned parting surface: merged {vert_diff} verts, removed {face_diff} faces in {elapsed:.1f}ms")
        if has_boundary_type:
            bt = result.vertex_boundary_type
            logger.info(f"Preserved boundary types: {np.sum(bt == -1)} part (inner), "
                       f"{np.sum(bt == 1) + np.sum(bt == 2)} hull (outer), {np.sum(bt == 0)} interior")
        
        return result
        
    except Exception as e:
        logger.warning(f"Surface cleaning failed: {e}")
        return surface


def repair_parting_surface_with_part(
    surface: PartingSurfaceResult,
    part_mesh: trimesh.Trimesh
) -> PartingSurfaceResult:
    """
    Clean parting surface (merge vertices, remove degenerates).
    
    Args:
        surface: PartingSurfaceResult to clean
        part_mesh: The original part mesh (reserved for future use)
    
    Returns:
        Cleaned PartingSurfaceResult
    """
    if surface.mesh is None:
        return surface
    
    # Basic cleanup: merge vertices and remove degenerate faces
    return repair_parting_surface(surface, merge_vertices=True)


def remove_small_islands(
    surface: PartingSurfaceResult,
    min_triangles: int = PRIMARY_MIN_ISLAND_TRIANGLES,
    min_area_fraction: float = PRIMARY_MIN_ISLAND_AREA_FRACTION
) -> PartingSurfaceResult:
    """
    Remove small disconnected components (islands) from the parting surface.
    
    Small islands are typically noise from isolated tetrahedra with valid
    configurations that don't connect to the main surface. These can cause
    issues during smoothing and CSG operations.
    
    This is particularly important for SECONDARY surfaces which may have
    multiple disconnected patches - we keep all significant patches but
    remove tiny noise fragments.
    
    Args:
        surface: PartingSurfaceResult to clean
        min_triangles: Minimum triangles to keep an island (default: 10)
        min_area_fraction: Minimum area fraction of total (default: 0.01 = 1%)
    
    Returns:
        Cleaned PartingSurfaceResult with small islands removed
    """
    if surface.mesh is None or len(surface.mesh.faces) == 0:
        return surface
    
    try:
        mesh = surface.mesh.copy()
        
        # Split into connected components
        try:
            components = mesh.split(only_watertight=False)
        except Exception as e:
            logger.warning(f"Could not split mesh for island removal: {e}")
            return surface
        
        if len(components) <= 1:
            # Only one component - nothing to remove
            return surface
        
        # Calculate total area for percentage threshold
        total_area = sum(c.area for c in components)
        min_area = total_area * min_area_fraction
        
        # Identify components to keep
        # Keep a component if it has enough triangles OR enough area
        kept_components = []
        removed_count = 0
        removed_triangles = 0
        removed_area = 0.0
        
        for comp in components:
            n_tris = len(comp.faces)
            area = comp.area
            
            if n_tris >= min_triangles or area >= min_area:
                kept_components.append(comp)
            else:
                removed_count += 1
                removed_triangles += n_tris
                removed_area += area
        
        if removed_count == 0:
            # Nothing to remove
            return surface
        
        logger.info(f"Removing {removed_count} small islands "
                   f"({removed_triangles} triangles, {removed_area:.2f} area)")
        
        if len(kept_components) == 0:
            logger.warning("All components were too small - keeping original mesh")
            return surface
        
        # Merge kept components back into a single mesh
        if len(kept_components) == 1:
            combined = kept_components[0]
        else:
            combined = trimesh.util.concatenate(kept_components)
        
        # Build result
        # Note: vertex_boundary_type is invalidated because vertices are re-indexed
        # If needed, the caller should track vertex_boundary_type through this operation
        result = PartingSurfaceResult(
            mesh=combined,
            vertices=np.array(combined.vertices),
            faces=np.array(combined.faces),
            vertex_to_edge=None,  # Invalidated
            vertex_boundary_type=None,  # Invalidated by component merge
            num_vertices=len(combined.vertices),
            num_faces=len(combined.faces),
            num_tets_processed=surface.num_tets_processed,
            num_tets_contributing=surface.num_tets_contributing,
            extraction_time_ms=surface.extraction_time_ms
        )
        
        logger.info(f"Island removal complete: {len(components)} -> {len(kept_components)} components, "
                   f"{surface.num_faces} -> {result.num_faces} faces")
        
        return result
        
    except Exception as e:
        logger.warning(f"Island removal failed: {e}")
        return surface


# =============================================================================
# GAP CLOSING BETWEEN PARTING SURFACE AND PART
# =============================================================================

@dataclass
class GapClosingResult:
    """Result of gap closing operation."""
    mesh: Optional[trimesh.Trimesh] = None
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    
    # Indices of vertices that should be constrained to part surface during smoothing
    # These are the NEW vertices projected onto the part (outer rim of fill)
    part_constrained_vertices: Optional[np.ndarray] = None
    
    # Indices of original boundary vertices used in gap fill (inner rim of fill)
    # These should be treated as boundary vertices but NOT re-projected
    # They maintain the connection between parting surface and fill geometry
    fill_boundary_vertices: Optional[np.ndarray] = None
    
    # Edges of the inner rim (original boundary edges that become internal after fill)
    # These are needed to build proper boundary_neighbors for smoothing
    # Format: Nx2 array of vertex index pairs
    inner_rim_edges: Optional[np.ndarray] = None
    
    # Statistics
    boundary_edges_found: int = 0
    boundary_edges_near_part: int = 0
    boundary_loops_found: int = 0
    fill_faces_added: int = 0
    new_vertices_added: int = 0
    
    # Timing
    processing_time_ms: float = 0.0


def close_parting_surface_gaps(
    surface: PartingSurfaceResult,
    part_mesh: trimesh.Trimesh,
    distance_threshold: float = None,
    method: str = 'smart_curtain',
    primary_surface_mesh: trimesh.Trimesh = None,
    hull_mesh: trimesh.Trimesh = None,
    vertex_boundary_type: np.ndarray = None
) -> GapClosingResult:
    """
    Close gaps between parting surface boundary edges and target surfaces.
    
    Per the paper Section 4.3: "The triangulated surface C is bounded by 
    construction by the object surface mesh M and the external boundary ∂H."
    
    This function connects parting surface boundary edges to:
    1. The part mesh M (inner boundary) - for vertices with boundary_type == -1
    2. The hull mesh ∂H (outer boundary) - for vertices with boundary_type == 1/2
    3. The primary surface (for secondary surfaces)
    
    IMPORTANT: This function returns the mesh with properly connected fill geometry.
    The `part_constrained_vertices` field contains indices of vertices that MUST
    be constrained to stay on the part/hull surface during any subsequent smoothing.
    
    Algorithm:
    1. Find boundary edges near target surfaces (part, hull, or primary)
    2. Build ordered chains from these edges
    3. Project boundary vertices to closest point on nearest target surface
    4. Create fill triangles that:
       - Reference original surface vertices directly (no duplication)
       - Add only the projected-to-target vertices as new vertices
       - Use consistent winding direction
    
    Args:
        surface: PartingSurfaceResult with boundary edges to close
        part_mesh: The original part mesh to close gaps against (inner boundary M)
        distance_threshold: Max distance to consider "near target" (auto-computed if None)
        method: Gap closing method (currently 'smart_curtain' is recommended)
        primary_surface_mesh: Optional primary parting surface mesh (for secondary surface gap filling)
        hull_mesh: Optional hull mesh to close gaps against (outer boundary ∂H)
    
    Returns:
        GapClosingResult with new mesh including fill geometry and constraint info
    """
    import time
    
    start = time.time()
    result = GapClosingResult()
    
    if surface.mesh is None:
        logger.warning("No parting surface mesh to close gaps for")
        return result
    
    if part_mesh is None:
        logger.warning("No part mesh provided for gap closing")
        return result
    
    mesh = surface.mesh
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    n_orig_verts = len(vertices)
    
    # === Step 1: Find boundary edges and their adjacent face info ===
    edge_to_faces = {}
    edge_to_adjacent_verts = {}  # For determining winding
    
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = face[i], face[(i+1) % 3]
            v_opposite = face[(i+2) % 3]  # Third vertex of face
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
                edge_to_adjacent_verts[edge_key] = []
            edge_to_faces[edge_key].append(fi)
            # Store which vertex is "inside" relative to this edge
            edge_to_adjacent_verts[edge_key].append((v0, v1, v_opposite))
    
    # Boundary edges appear in exactly one face
    boundary_edges = []
    boundary_edge_info = {}  # edge_key -> (v0, v1, v_opposite) for winding
    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) == 1:
            boundary_edges.append(edge_key)
            boundary_edge_info[edge_key] = edge_to_adjacent_verts[edge_key][0]
    
    result.boundary_edges_found = len(boundary_edges)
    logger.info(f"Found {len(boundary_edges)} boundary edges on parting surface")
    
    if len(boundary_edges) == 0:
        logger.info("Parting surface is watertight - no gaps to close")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # === Step 2: Compute distances to target surfaces (part, hull, primary) ===
    # Use MINIMUM distance from EITHER edge vertex to target surface
    # This is more robust than using midpoint distance - if either vertex is near the target,
    # we should create a fill triangle to ensure proper connection.
    boundary_edge_array = np.array(boundary_edges)
    
    # Get positions of both vertices for each edge
    v0_positions = vertices[boundary_edge_array[:, 0]]
    v1_positions = vertices[boundary_edge_array[:, 1]]
    
    # Compute distances from both vertices to part mesh
    _, dist_v0_part, _ = trimesh.proximity.closest_point(part_mesh, v0_positions)
    _, dist_v1_part, _ = trimesh.proximity.closest_point(part_mesh, v1_positions)
    distances_part = np.minimum(dist_v0_part, dist_v1_part)
    distances = distances_part.copy()
    
    # Also compute distances to hull mesh (outer boundary ∂H) 
    distances_hull = None
    if hull_mesh is not None:
        _, dist_v0_hull, _ = trimesh.proximity.closest_point(hull_mesh, v0_positions)
        _, dist_v1_hull, _ = trimesh.proximity.closest_point(hull_mesh, v1_positions)
        distances_hull = np.minimum(dist_v0_hull, dist_v1_hull)
        distances = np.minimum(distances, distances_hull)
        logger.info(f"Computing distances to part AND hull mesh")
    
    # If primary surface provided, also compute distances to it
    distances_primary = None
    if primary_surface_mesh is not None:
        _, dist_v0_primary, _ = trimesh.proximity.closest_point(primary_surface_mesh, v0_positions)
        _, dist_v1_primary, _ = trimesh.proximity.closest_point(primary_surface_mesh, v1_positions)
        distances_primary = np.minimum(dist_v0_primary, dist_v1_primary)
        distances = np.minimum(distances, distances_primary)
        logger.info(f"Computing distances to primary surface")
    
    # Auto-compute threshold if not provided
    if distance_threshold is None:
        edge_lengths = np.linalg.norm(
            vertices[boundary_edge_array[:, 1]] - vertices[boundary_edge_array[:, 0]], 
            axis=1
        )
        distance_threshold = np.median(edge_lengths) * DISTANCE_THRESHOLD_EDGE_MULTIPLIER
        logger.info(f"Auto distance threshold: {distance_threshold:.4f}")
    
    # Find edges near target surfaces
    near_target_mask = distances < distance_threshold
    near_target_edges = boundary_edge_array[near_target_mask]
    result.boundary_edges_near_part = len(near_target_edges)
    
    # Build target description for logging
    targets = ["part"]
    if hull_mesh is not None:
        targets.append("hull")
    if primary_surface_mesh is not None:
        targets.append("primary")
    target_desc = "/".join(targets)
    
    logger.info(f"Found {len(near_target_edges)} boundary edges near {target_desc} (< {distance_threshold:.4f})")
    
    if len(near_target_edges) == 0:
        logger.info(f"No boundary edges near {target_desc} - no gaps to close")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # === Step 3: Build boundary chains from near-target edges ===
    boundary_chains = _build_boundary_chains(near_target_edges)
    result.boundary_loops_found = len(boundary_chains)
    logger.info(f"Found {len(boundary_chains)} boundary chains near {target_desc}")
    
    # Get vertex_boundary_type from surface if not provided
    if vertex_boundary_type is None:
        vertex_boundary_type = surface.vertex_boundary_type
    
    # === Step 4: Create fill geometry with proper vertex sharing ===
    fill_result = _create_connected_fill(
        vertices, faces, boundary_chains, part_mesh, boundary_edge_info, distance_threshold,
        primary_surface_mesh=primary_surface_mesh,
        hull_mesh=hull_mesh,
        vertex_boundary_type=vertex_boundary_type
    )
    new_vertices = fill_result['new_vertices']
    new_faces = fill_result['new_faces']
    part_constrained_indices = fill_result['part_constrained_indices']
    fill_boundary_indices = fill_result['fill_boundary_indices']
    inner_rim_edges = fill_result['inner_rim_edges']
    
    if len(new_faces) == 0:
        logger.info("No fill faces generated")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    result.fill_faces_added = len(new_faces)
    result.new_vertices_added = len(new_vertices)
    
    # === Step 5: Combine original mesh with fill geometry ===
    # new_vertices are ONLY the projected points (new vertices to add)
    # new_faces already reference original vertex indices correctly
    
    combined_vertices = np.vstack([vertices, new_vertices])
    combined_faces = np.vstack([faces, new_faces])
    
    # Track which vertices are constrained to part (projected vertices - outer rim)
    result.part_constrained_vertices = np.array(part_constrained_indices, dtype=np.int64)
    
    # Track original boundary vertices used in fill (inner rim)
    # These should be boundary-smoothed but NOT re-projected
    result.fill_boundary_vertices = np.array(fill_boundary_indices, dtype=np.int64)
    
    # Track inner rim edges (original boundary edges that become internal after fill)
    # These are needed for proper boundary neighbor computation during smoothing
    result.inner_rim_edges = inner_rim_edges
    
    # Create combined mesh
    try:
        combined_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            process=False
        )
        
        # Remove any degenerate faces but do NOT merge vertices
        # (we want to preserve the connection topology)
        valid = combined_mesh.nondegenerate_faces()
        if np.sum(~valid) > 0:
            combined_mesh = trimesh.Trimesh(
                vertices=combined_mesh.vertices,
                faces=combined_mesh.faces[valid],
                process=False
            )
        
        # Fix normals to ensure consistent orientation
        combined_mesh.fix_normals()
        
        result.mesh = combined_mesh
        result.vertices = np.array(combined_mesh.vertices)
        result.faces = np.array(combined_mesh.faces)
        
    except Exception as e:
        logger.error(f"Failed to create combined mesh: {e}")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
    
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Gap closing: added {result.fill_faces_added} fill faces, "
                f"{result.new_vertices_added} new projected verts, "
                f"{len(result.part_constrained_vertices)} part-constrained (outer rim), "
                f"{len(result.fill_boundary_vertices)} fill-boundary (inner rim), "
                f"{len(result.inner_rim_edges)} inner rim edges "
                f"in {result.processing_time_ms:.1f}ms")
    
    return result


# =============================================================================
# FLOATING EDGE FILLING RESULT
# =============================================================================

@dataclass
class FloatingEdgeFillingResult:
    """Result of collar extension for inner boundary edges."""
    mesh: Optional[trimesh.Trimesh] = None
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    
    # Statistics
    boundary_edges_checked: int = 0
    floating_edges_found: int = 0
    fill_triangles_added: int = 0
    new_vertices_added: int = 0
    
    # Indices of new vertices that should be constrained to part surface
    part_constrained_vertices: Optional[np.ndarray] = None
    
    processing_time_ms: float = 0.0


# =============================================================================
# HELPER FUNCTIONS FOR BOUNDARY EDGE AND FAN VERTEX DETECTION
# =============================================================================

def _classify_boundary_edge(
    v0: int,
    v1: int,
    vertex_boundary_type: Optional[np.ndarray],
    vert_to_part_dist: Dict[int, float],
    vert_to_hull_dist: Dict[int, float],
    part_proximity_threshold: float = 0.5
) -> bool:
    """
    Classify a boundary edge as inner (part) or outer (hull).
    
    Classification priority:
    1. HIGHEST: Both vertices have vertex_boundary_type == -1 → inner
    2. PRIMARY: Both vertices closer to part than hull → inner (requires hull distances)
    3. SECONDARY: vertex_boundary_type with distance fallback
    4. FALLBACK: Both vertices within part_proximity_threshold → inner
    
    Args:
        v0, v1: Vertex indices
        vertex_boundary_type: Array with -1=part, 0=interior, 1/2=hull
        vert_to_part_dist: Precomputed distances to part mesh
        vert_to_hull_dist: Precomputed distances to hull mesh (empty if no hull)
        part_proximity_threshold: Distance threshold for fallback classification
    
    Returns:
        True if edge is inner (part) boundary, False if outer (hull) boundary
    """
    has_boundary_type = vertex_boundary_type is not None and len(vertex_boundary_type) > 0
    has_hull_dist = len(vert_to_hull_dist) > 0
    
    # Get boundary types
    bt0 = vertex_boundary_type[v0] if (has_boundary_type and v0 < len(vertex_boundary_type)) else None
    bt1 = vertex_boundary_type[v1] if (has_boundary_type and v1 < len(vertex_boundary_type)) else None
    
    # HIGHEST PRIORITY: Both vertices explicitly part boundary
    if bt0 == -1 and bt1 == -1:
        return True
    
    # Get distances
    d0_part = vert_to_part_dist.get(v0, 999)
    d1_part = vert_to_part_dist.get(v1, 999)
    d0_hull = vert_to_hull_dist.get(v0, 999)
    d1_hull = vert_to_hull_dist.get(v1, 999)
    
    # PRIMARY: Hull vs part distance comparison
    if has_hull_dist:
        v0_closer_to_part = d0_part < d0_hull
        v1_closer_to_part = d1_part < d1_hull
        
        if v0_closer_to_part and v1_closer_to_part:
            return True
        elif not v0_closer_to_part and not v1_closer_to_part:
            return False
        # Mixed case - falls through to other criteria
    
    # SECONDARY: Use boundary type with distance fallback
    if has_boundary_type:
        bt0_val = bt0 if bt0 is not None else 0
        bt1_val = bt1 if bt1 is not None else 0
        
        # Outer if either vertex is on hull
        if bt0_val in (1, 2) or bt1_val in (1, 2):
            return False
        # One vertex on part - check if other is close
        if bt0_val == -1 or bt1_val == -1:
            other_dist = d1_part if bt0_val == -1 else d0_part
            return other_dist < part_proximity_threshold
    
    # FALLBACK: Pure distance heuristic
    return d0_part < part_proximity_threshold and d1_part < part_proximity_threshold


# =============================================================================
# CONSTANTS FOR IMPROVED VERTEX DETECTION
# =============================================================================

# Angle threshold for "sharp corner" detection (degrees)
# Vertices where boundary edges meet at angles sharper than this get extra fanning
SHARP_CORNER_ANGLE_THRESHOLD = 60.0

# Angle threshold for "near-tip" detection (degrees) 
# Vertices with 2 edges from different faces, but edges pointing in similar directions
# (angle between outgoing edge vectors < this threshold) are classified as "near-tips"
NEAR_TIP_ANGLE_THRESHOLD = 45.0

# Maximum vertices in a "small loop" that needs complete fanning
SMALL_LOOP_MAX_VERTICES = 5

# Minimum divergence angle for enhanced fanning (degrees)
# When collar directions at a vertex spread by more than this, use enhanced fans
COLLAR_DIVERGENCE_THRESHOLD = 90.0


@dataclass 
class BoundaryVertexInfo:
    """Classification info for boundary vertices with enhanced detection.
    
    Vertex classification (mutually exclusive, in priority order):
    1. isolated_tips: 2 boundary edges from SAME face (single triangle tips)
    2. near_tips: 2 edges from DIFFERENT faces but pointing in similar direction
    3. sharp_corners: 2+ edges meeting at sharp angles (< SHARP_CORNER_ANGLE_THRESHOLD)
    4. divergent_corners: 2+ edges where collar directions spread significantly
    5. high_valence: 3+ edges (complex junctions)
    6. corners: Regular corners (2 edges from different faces, not sharp)
    7. endpoints: 1 edge (chain endpoints)
    
    Vertices in categories 1-5 get fanned triangle collars.
    Regular corners (6) may or may not need fans depending on geometry.
    """
    # PRIMARY categories (definitely need fans)
    corners: List[int]              # Regular corners (2+ edges from DIFFERENT faces)
    isolated_tips: List[int]        # 2 edges from SAME face (single triangle tip)
    endpoints: List[int]            # 1 boundary edge (chain end)
    
    # ENHANCED categories (additional detection for better fanning)
    near_tips: List[int]            # 2 edges from diff faces but similar direction
    sharp_corners: List[int]        # Edges meet at sharp angle (< 60°)
    divergent_corners: List[int]    # Collar directions spread > 90°
    high_valence: List[int]         # 3+ boundary edges (complex junctions)
    small_loop_vertices: List[int]  # Vertices part of small boundary loops
    
    # Edge data
    vertex_to_edges: Dict[int, List[Tuple]]  # vertex -> [(edge_key, this_v, other_v), ...]
    
    # Edge angle data for each vertex (for debugging/tuning)
    vertex_edge_angles: Dict[int, float]  # vertex -> angle between edges (degrees)
    
    def __init__(self):
        """Initialize with empty lists."""
        self.corners = []
        self.isolated_tips = []
        self.endpoints = []
        self.near_tips = []
        self.sharp_corners = []
        self.divergent_corners = []
        self.high_valence = []
        self.small_loop_vertices = []
        self.vertex_to_edges = {}
        self.vertex_edge_angles = {}
    
    def get_all_fan_vertices(self) -> List[int]:
        """Get all vertices that should have fanned triangle collars.
        
        Returns combined list of isolated_tips + near_tips + sharp_corners + 
        divergent_corners + high_valence + corners (deduped).
        """
        fan_verts = set(self.isolated_tips)
        fan_verts.update(self.near_tips)
        fan_verts.update(self.sharp_corners)
        fan_verts.update(self.divergent_corners)
        fan_verts.update(self.high_valence)
        fan_verts.update(self.corners)
        fan_verts.update(self.small_loop_vertices)
        return list(fan_verts)


def _compute_edge_angle(
    vi_pos: np.ndarray,
    other_v1_pos: np.ndarray,
    other_v2_pos: np.ndarray
) -> float:
    """
    Compute angle between two boundary edges at a vertex.
    
    Args:
        vi_pos: Position of the vertex where edges meet
        other_v1_pos: Position of first edge's other endpoint
        other_v2_pos: Position of second edge's other endpoint
    
    Returns:
        Angle in degrees (0-180) between the two edges
    """
    # Edge vectors pointing outward from vi
    e1 = other_v1_pos - vi_pos
    e2 = other_v2_pos - vi_pos
    
    len1 = np.linalg.norm(e1)
    len2 = np.linalg.norm(e2)
    
    if len1 < 1e-10 or len2 < 1e-10:
        return 180.0  # Degenerate - treat as straight
    
    e1_unit = e1 / len1
    e2_unit = e2 / len2
    
    cos_angle = np.clip(np.dot(e1_unit, e2_unit), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    
    return angle_deg


def _detect_small_boundary_loops(
    inner_boundary_edges: List[Tuple[int, int]],
    max_vertices: int = SMALL_LOOP_MAX_VERTICES
) -> List[Set[int]]:
    """
    Detect small closed boundary loops.
    
    Small loops (e.g., 3-4 vertices forming a closed ring around a protrusion)
    need all their vertices to get fanned collars for proper sealing.
    
    Args:
        inner_boundary_edges: List of inner boundary edge tuples
        max_vertices: Maximum vertices in a loop to consider "small"
    
    Returns:
        List of sets, each set contains vertex indices forming a small loop
    """
    from collections import defaultdict
    
    # Build adjacency
    adj = defaultdict(set)
    for v0, v1 in inner_boundary_edges:
        adj[v0].add(v1)
        adj[v1].add(v0)
    
    # Find connected components using DFS
    visited = set()
    small_loops = []
    
    for start in adj:
        if start in visited:
            continue
        
        # DFS to find component
        stack = [start]
        component = set()
        
        while stack:
            v = stack.pop()
            if v in component:
                continue
            component.add(v)
            visited.add(v)
            
            for neighbor in adj[v]:
                if neighbor not in component:
                    stack.append(neighbor)
        
        # Check if it's a small closed loop
        # A closed loop has all vertices with exactly 2 neighbors in the component
        if len(component) <= max_vertices and len(component) >= 3:
            is_closed_loop = all(len(adj[v] & component) == 2 for v in component)
            if is_closed_loop:
                small_loops.append(component)
    
    return small_loops


def _detect_boundary_vertices(
    inner_boundary_edges: List[Tuple[int, int]],
    edge_to_face: Dict[Tuple[int, int], Tuple[int, int]],
    vertices_arr: np.ndarray,
    restored_corner_positions: Optional[np.ndarray] = None
) -> BoundaryVertexInfo:
    """
    Detect and classify all boundary vertices with ENHANCED detection.
    
    This function uses multiple criteria to identify vertices that need 
    fanned triangle collars for proper mesh sealing:
    
    Vertex types (in classification priority):
    1. isolated_tips: 2 boundary edges from SAME face (single triangle tip)
    2. near_tips: 2 edges from DIFFERENT faces but pointing in similar direction (<45°)
    3. sharp_corners: 2+ edges meeting at sharp angles (<60°)
    4. divergent_corners: 2+ edges where collar directions would spread >90°
    5. high_valence: 3+ boundary edges (complex junctions)
    6. corners: Regular corners (2 edges from different faces, moderate angle)
    7. endpoints: 1 boundary edge (chain endpoints, no fans)
    
    Args:
        inner_boundary_edges: List of inner boundary edge tuples
        edge_to_face: Mapping from edge_key to (face_idx, third_vertex)
        vertices_arr: Vertex positions array
        restored_corner_positions: Optional positions of restored concave corners
    
    Returns:
        BoundaryVertexInfo with classified vertices and edge angle data
    """
    result = BoundaryVertexInfo()
    
    # Build vertex -> edge mapping
    for v0, v1 in inner_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        for vi, other in [(v0, v1), (v1, v0)]:
            if vi not in result.vertex_to_edges:
                result.vertex_to_edges[vi] = []
            result.vertex_to_edges[vi].append((edge_key, vi, other))
    
    # Detect small boundary loops FIRST (all their vertices need fans)
    small_loops = _detect_small_boundary_loops(inner_boundary_edges)
    small_loop_vertex_set = set()
    for loop in small_loops:
        small_loop_vertex_set.update(loop)
        result.small_loop_vertices.extend(loop)
    
    if small_loops:
        logger.info(f"Detected {len(small_loops)} small boundary loops "
                   f"({len(small_loop_vertex_set)} total vertices)")
    
    # Classify each vertex
    for vi, edges in result.vertex_to_edges.items():
        n_edges = len(edges)
        
        # Skip if already classified as small loop vertex (will get fans)
        if vi in small_loop_vertex_set:
            continue
        
        # 1 edge: endpoint (no fan needed)
        if n_edges == 1:
            result.endpoints.append(vi)
            continue
        
        # 3+ edges: high-valence junction (always needs fans)
        if n_edges >= 3:
            result.high_valence.append(vi)
            # Store angle as 360/n_edges for reference
            result.vertex_edge_angles[vi] = 360.0 / n_edges
            continue
        
        # 2 edges: detailed classification
        assert n_edges == 2
        
        edge_a, edge_b = edges[0], edges[1]
        edge_key_a, _, other_a = edge_a
        edge_key_b, _, other_b = edge_b
        
        # Get face info
        face_a = edge_to_face.get(edge_key_a)
        face_b = edge_to_face.get(edge_key_b)
        same_face = (face_a is not None and face_b is not None and face_a[0] == face_b[0])
        
        # Compute angle between the two edges
        vi_pos = vertices_arr[vi]
        other_a_pos = vertices_arr[other_a]
        other_b_pos = vertices_arr[other_b]
        edge_angle = _compute_edge_angle(vi_pos, other_a_pos, other_b_pos)
        result.vertex_edge_angles[vi] = edge_angle
        
        # Classification based on face sharing and angle
        if same_face:
            # Both edges from same face = isolated tip (classic single triangle tip)
            result.isolated_tips.append(vi)
            
        elif edge_angle < NEAR_TIP_ANGLE_THRESHOLD:
            # Edges point in similar direction (< 45°) = near-tip
            # This catches cases like two adjacent triangles forming a "finger"
            result.near_tips.append(vi)
            
        elif edge_angle < SHARP_CORNER_ANGLE_THRESHOLD:
            # Sharp angle (< 60°) = sharp corner needing enhanced fans
            result.sharp_corners.append(vi)
            
        elif edge_angle > (180.0 - COLLAR_DIVERGENCE_THRESHOLD / 2):
            # Very obtuse angle (> 135°) means collar directions will diverge significantly
            # when projected onto part surface - may need enhanced fanning
            result.divergent_corners.append(vi)
            
        else:
            # Regular corner (moderate angle: 60° to 135°)
            result.corners.append(vi)
    
    # Add restored concave corners (from smoothing step)
    if restored_corner_positions is not None and len(restored_corner_positions) > 0:
        mesh_scale = np.linalg.norm(np.ptp(vertices_arr, axis=0))
        match_tolerance = mesh_scale * 0.001
        
        inner_verts = set(result.vertex_to_edges.keys())
        all_classified = (set(result.corners) | set(result.isolated_tips) | 
                         set(result.near_tips) | set(result.sharp_corners) |
                         set(result.divergent_corners) | set(result.high_valence) |
                         small_loop_vertex_set)
        
        for pos in restored_corner_positions:
            dists = np.linalg.norm(vertices_arr - pos, axis=1)
            min_idx = np.argmin(dists)
            
            if dists[min_idx] < match_tolerance and min_idx in inner_verts and min_idx not in all_classified:
                # Add to sharp_corners (concave corners need enhanced fans)
                result.sharp_corners.append(min_idx)
                all_classified.add(min_idx)
    
    # Log classification results
    logger.info(f"Boundary vertex classification: "
               f"{len(result.isolated_tips)} isolated tips, "
               f"{len(result.near_tips)} near tips, "
               f"{len(result.sharp_corners)} sharp corners, "
               f"{len(result.divergent_corners)} divergent corners, "
               f"{len(result.high_valence)} high-valence, "
               f"{len(result.corners)} regular corners, "
               f"{len(result.endpoints)} endpoints, "
               f"{len(result.small_loop_vertices)} small-loop vertices")
    
    return result


def _create_collar_vertex(
    vi_pos: np.ndarray,
    part_mesh: trimesh.Trimesh,
    part_face_normals: np.ndarray,
    collar_depth: float,
    collar_dir_hint: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create a collar vertex by projecting to part and offsetting inward.
    
    Args:
        vi_pos: Position of the membrane vertex
        part_mesh: The part mesh
        part_face_normals: Pre-computed part face normals
        collar_depth: How far to offset into the part
        collar_dir_hint: Optional hint direction for collar (used for isolated tips)
    
    Returns:
        Position of the collar vertex
    """
    # If we have a direction hint (for isolated tips), try planar point first
    if collar_dir_hint is not None:
        planar_pt = vi_pos + collar_depth * collar_dir_hint
        try:
            if part_mesh.contains([planar_pt])[0]:
                return planar_pt
        except:
            pass
    
    # Standard approach: project to part and offset inward
    try:
        closest_pts, _, closest_faces = trimesh.proximity.closest_point(part_mesh, [vi_pos])
        closest_pt = closest_pts[0]
        closest_face = closest_faces[0]
        
        if closest_face < len(part_face_normals):
            into_part = -part_face_normals[closest_face]
        else:
            into_part = collar_dir_hint if collar_dir_hint is not None else np.array([0, 0, -1])
        
        into_part = into_part / (np.linalg.norm(into_part) + 1e-8)
        collar_pt = closest_pt + collar_depth * into_part
        
        # Verify inside
        try:
            if not part_mesh.contains([collar_pt])[0]:
                alt_pt = closest_pt - collar_depth * into_part
                if part_mesh.contains([alt_pt])[0]:
                    collar_pt = alt_pt
        except:
            pass
        
        return collar_pt
    except:
        # Fallback to simple offset
        return vi_pos + collar_depth * (collar_dir_hint if collar_dir_hint is not None else np.array([0, 0, -1]))


def _create_fan_triangles(
    vi: int,
    vi_pos: np.ndarray,
    collar_infos: List[Tuple[int, np.ndarray]],  # [(collar_idx, collar_pos), ...]
    ref_normal: np.ndarray,
    vertices: List[np.ndarray],
    faces: List[List[int]],
    part_mesh: trimesh.Trimesh,
    part_face_normals: np.ndarray,
    collar_depth: float,
    fan_subdivisions: int
) -> Tuple[int, int]:
    """
    Create fan triangles between consecutive collar vertices at a corner/tip.
    
    This unified function handles both isolated tips and regular corners.
    
    Args:
        vi: Apex vertex index
        vi_pos: Apex vertex position
        collar_infos: List of (collar_idx, collar_pos) in angular order
        ref_normal: Reference normal for triangle winding
        vertices: Mutable list of vertex positions
        faces: Mutable list of face indices
        part_mesh: Part mesh for containment checks
        part_face_normals: Part face normals
        collar_depth: Collar depth for arc vertices
        fan_subdivisions: Base number of subdivisions
    
    Returns:
        (arc_vertices_created, triangles_created)
    """
    arc_verts_created = 0
    triangles_created = 0
    
    if len(collar_infos) < 2:
        return 0, 0
    
    # Process each consecutive pair of collars
    for i in range(len(collar_infos)):
        c_a, c_a_pos = collar_infos[i]
        c_b, c_b_pos = collar_infos[(i + 1) % len(collar_infos)]
        
        if c_a == c_b or np.linalg.norm(c_a_pos - c_b_pos) < 1e-6:
            continue
        
        # Collar directions
        dir_a = c_a_pos - vi_pos
        dir_b = c_b_pos - vi_pos
        len_a, len_b = np.linalg.norm(dir_a), np.linalg.norm(dir_b)
        
        if len_a < 1e-8 or len_b < 1e-8:
            continue
        
        dir_a_unit = dir_a / len_a
        dir_b_unit = dir_b / len_b
        
        # Angle between collar directions
        cos_angle = np.clip(np.dot(dir_a_unit, dir_b_unit), -1, 1)
        angle = np.arccos(cos_angle)
        angle_deg = np.degrees(angle)
        
        # Fan plane normal
        fan_normal = np.cross(dir_a_unit, dir_b_unit)
        fan_len = np.linalg.norm(fan_normal)
        
        if fan_len < 1e-8:
            # Parallel - single triangle
            e1, e2 = c_a_pos - vi_pos, c_b_pos - vi_pos
            tri_n = np.cross(e1, e2)
            if np.linalg.norm(tri_n) > 1e-10:
                if np.dot(tri_n, ref_normal) >= 0:
                    faces.append([vi, c_a, c_b])
                else:
                    faces.append([vi, c_b, c_a])
                triangles_created += 1
            continue
        
        fan_normal = fan_normal / fan_len
        if np.dot(fan_normal, ref_normal) < 0:
            fan_normal = -fan_normal
        
        # Subdivisions based on angle
        n_subs = 2 + max(0, int((angle_deg - 30) / 30))
        n_subs = min(n_subs, fan_subdivisions + 3)
        
        # Create arc vertices via SLERP
        arc_collars = [(c_a, c_a_pos)]
        
        sin_angle = np.sin(angle)
        for j in range(1, n_subs):
            t = j / n_subs
            
            # SLERP interpolation
            if abs(sin_angle) < 1e-8:
                interp_dir = (1 - t) * dir_a_unit + t * dir_b_unit
            else:
                w_a = np.sin((1 - t) * angle) / sin_angle
                w_b = np.sin(t * angle) / sin_angle
                interp_dir = w_a * dir_a_unit + w_b * dir_b_unit
            
            interp_dir = interp_dir / (np.linalg.norm(interp_dir) + 1e-8)
            interp_radius = (1 - t) * len_a + t * len_b
            arc_pt = vi_pos + interp_radius * interp_dir
            
            # Check containment
            try:
                inside = part_mesh.contains([arc_pt])[0]
            except:
                inside = False
            
            if not inside:
                arc_pt = _create_collar_vertex(arc_pt, part_mesh, part_face_normals, 
                                              collar_depth, interp_dir)
            
            arc_idx = len(vertices)
            vertices.append(arc_pt.copy())
            arc_collars.append((arc_idx, arc_pt))
            arc_verts_created += 1
        
        arc_collars.append((c_b, c_b_pos))
        
        # Create triangles
        for j in range(len(arc_collars) - 1):
            c_curr, c_curr_pos = arc_collars[j]
            c_next, c_next_pos = arc_collars[j + 1]
            
            e1, e2 = c_curr_pos - vi_pos, c_next_pos - vi_pos
            tri_n = np.cross(e1, e2)
            
            if np.linalg.norm(tri_n) < 1e-10:
                continue
            
            if np.dot(tri_n, fan_normal) >= 0:
                faces.append([vi, c_curr, c_next])
            else:
                faces.append([vi, c_next, c_curr])
            triangles_created += 1
    
    return arc_verts_created, triangles_created


# =============================================================================
# MAIN COLLAR EXTENSION FUNCTION
# =============================================================================

def create_robust_collar_extension(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    hull_mesh: Optional[trimesh.Trimesh] = None,
    vertex_boundary_type: Optional[np.ndarray] = None,
    collar_depth: float = 0.5,
    fan_subdivisions: int = 4,
    restored_corner_positions: Optional[np.ndarray] = None
) -> FloatingEdgeFillingResult:
    """
    Create robust collar extension for inner boundary edges.
    
    Algorithm:
    1. Find inner boundary edges (touching part, not hull)
    2. Create collar vertices by projecting edge endpoints onto part and offsetting inward
    3. Create quad collars connecting membrane edges to collar edges
    4. Create fan triangles at corners and isolated tip vertices
    
    Args:
        membrane_mesh: The smoothed membrane mesh
        part_mesh: The part mesh to connect to
        hull_mesh: Optional hull mesh for accurate boundary classification
        vertex_boundary_type: Array with -1=part, 0=interior, 1/2=hull
        collar_depth: How far to extend into part (mm)
        fan_subdivisions: Number of subdivisions for corner fans
        restored_corner_positions: Optional positions of restored concave corners
    
    Returns:
        FloatingEdgeFillingResult with the collared mesh
    """
    import time
    start = time.time()
    
    result = FloatingEdgeFillingResult()
    
    if membrane_mesh is None or part_mesh is None:
        logger.warning("Missing mesh for collar extension")
        return result
    
    vertices = list(membrane_mesh.vertices)
    faces = list(membrane_mesh.faces)
    n_orig_verts = len(vertices)
    n_orig_faces = len(faces)
    
    vertices_arr = np.array(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int64)
    
    # =========================================================================
    # STEP 1: Build edge-to-face mapping and find boundary edges
    # =========================================================================
    
    edge_to_face = {}  # edge_key -> (face_idx, third_vertex)
    edge_face_count = {}
    
    for fi, face in enumerate(faces_arr):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            v2 = int(face[(i + 2) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_face_count[edge_key] = edge_face_count.get(edge_key, 0) + 1
            edge_to_face[edge_key] = (fi, v2)
    
    boundary_edges = [(v0, v1) for (v0, v1), c in edge_face_count.items() if c == 1]
    
    if not boundary_edges:
        logger.info("No boundary edges found")
        result.mesh = membrane_mesh
        result.vertices = np.array(membrane_mesh.vertices)
        result.faces = np.array(membrane_mesh.faces)
        return result
    
    logger.info(f"Found {len(boundary_edges)} mesh boundary edges")
    
    # =========================================================================
    # STEP 2: Classify boundary edges using consolidated helper
    # =========================================================================
    
    boundary_verts = list(set([v for e in boundary_edges for v in e]))
    boundary_vert_positions = vertices_arr[boundary_verts]
    
    # Precompute distances
    try:
        _, dists_to_part, _ = trimesh.proximity.closest_point(part_mesh, boundary_vert_positions)
        vert_to_part_dist = dict(zip(boundary_verts, dists_to_part))
    except Exception as e:
        logger.warning(f"Distance computation failed: {e}")
        result.mesh = membrane_mesh
        return result
    
    vert_to_hull_dist = {}
    if hull_mesh is not None:
        try:
            _, dists_to_hull, _ = trimesh.proximity.closest_point(hull_mesh, boundary_vert_positions)
            vert_to_hull_dist = dict(zip(boundary_verts, dists_to_hull))
        except:
            pass
    
    inner_boundary_edges = []
    outer_count = 0
    
    for v0, v1 in boundary_edges:
        is_inner = _classify_boundary_edge(
            v0, v1, vertex_boundary_type, vert_to_part_dist, vert_to_hull_dist
        )
        if is_inner:
            inner_boundary_edges.append((v0, v1))
        else:
            outer_count += 1
    
    logger.info(f"Inner boundary edges: {len(inner_boundary_edges)}, outer (hull): {outer_count}")
    
    if len(inner_boundary_edges) == 0:
        logger.info("No inner boundary edges to collar")
        result.mesh = membrane_mesh
        result.vertices = np.array(membrane_mesh.vertices)
        result.faces = np.array(membrane_mesh.faces)
        return result
    
    # =========================================================================
    # STEP 3: Detect boundary vertices using consolidated helper
    # =========================================================================
    
    boundary_info = _detect_boundary_vertices(
        inner_boundary_edges, edge_to_face, vertices_arr, restored_corner_positions
    )
    
    # Log summary - detailed breakdown is in _detect_boundary_vertices
    total_fan_verts = len(boundary_info.get_all_fan_vertices())
    logger.info(f"Boundary detection: {total_fan_verts} fan vertices, "
               f"{len(boundary_info.endpoints)} chain endpoints")
    
    # =========================================================================
    # STEP 4: Create collar vertices for each edge endpoint
    # =========================================================================
    
    membrane_face_normals = membrane_mesh.face_normals
    part_face_normals = part_mesh.face_normals
    
    # Map: (edge_key, vertex) -> collar_idx
    edge_endpoint_collar = {}
    
    # Vertices that need special collar direction hints (tip-like vertices)
    # Include both isolated_tips AND near_tips (they behave similarly)
    tip_like_vertices = set(boundary_info.isolated_tips)
    tip_like_vertices.update(boundary_info.near_tips)
    
    for v0, v1 in inner_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        
        if edge_key not in edge_endpoint_collar:
            edge_endpoint_collar[edge_key] = {}
        
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi, _ = face_info
        face_normal = membrane_face_normals[fi] if fi < len(membrane_face_normals) else np.array([0, 0, 1])
        
        v0_pos, v1_pos = vertices_arr[v0], vertices_arr[v1]
        edge_vec = v1_pos - v0_pos
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-8:
            continue
        edge_dir = edge_vec / edge_len
        
        for vi in [v0, v1]:
            if vi in edge_endpoint_collar[edge_key]:
                continue
            
            vi_pos = vertices_arr[vi]
            
            # For tip-like vertices (isolated tips + near tips), use perpendicular direction in face plane
            collar_hint = None
            if vi in tip_like_vertices:
                perp = np.cross(face_normal, edge_dir)
                perp_len = np.linalg.norm(perp)
                if perp_len > 1e-8:
                    perp = perp / perp_len
                    # Flip if pointing toward interior
                    face_verts = faces_arr[fi]
                    third_v = [fv for fv in face_verts if fv != v0 and fv != v1]
                    if third_v:
                        to_third = vertices_arr[third_v[0]] - vi_pos
                        if np.dot(perp, to_third) > 0:
                            perp = -perp
                    collar_hint = perp
            
            collar_pt = _create_collar_vertex(vi_pos, part_mesh, part_face_normals, collar_depth, collar_hint)
            collar_idx = len(vertices)
            vertices.append(collar_pt.copy())
            edge_endpoint_collar[edge_key][vi] = collar_idx
    
    logger.info(f"Created {len(vertices) - n_orig_verts} collar vertices")
    
    # =========================================================================
    # STEP 5: Create quad collars for each edge
    # =========================================================================
    
    quads_created = 0
    
    for v0, v1 in inner_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        collar_data = edge_endpoint_collar.get(edge_key, {})
        c0, c1 = collar_data.get(v0), collar_data.get(v1)
        
        if c0 is None or c1 is None:
            continue
        
        v0_pos, v1_pos = vertices_arr[v0], vertices_arr[v1]
        c0_pos, c1_pos = np.array(vertices[c0]), np.array(vertices[c1])
        
        face_info = edge_to_face.get(edge_key)
        ref_normal = membrane_face_normals[face_info[0]] if face_info and face_info[0] < len(membrane_face_normals) else np.array([0, 0, 1])
        
        # Create quad as two triangles with correct winding
        for tri_verts, ref_e1, ref_e2 in [([v0, v1, c1], v1_pos - v0_pos, c1_pos - v0_pos),
                                          ([v0, c1, c0], c1_pos - v0_pos, c0_pos - v0_pos)]:
            tri_n = np.cross(ref_e1, ref_e2)
            if np.linalg.norm(tri_n) > 1e-10:
                if np.dot(tri_n, ref_normal) >= 0:
                    faces.append(tri_verts)
                else:
                    faces.append([tri_verts[0], tri_verts[2], tri_verts[1]])
        
        quads_created += 1
    
    logger.info(f"Created {quads_created} quad collars")
    
    # =========================================================================
    # STEP 6: Create fan triangles at all fan vertices
    # =========================================================================
    
    total_arc_verts = 0
    total_fan_tris = 0
    
    # Use the new get_all_fan_vertices() method for comprehensive detection
    # This includes: isolated_tips, near_tips, sharp_corners, divergent_corners,
    # high_valence, corners, and small_loop_vertices
    all_fan_vertices = boundary_info.get_all_fan_vertices()
    
    logger.info(f"Creating fans for {len(all_fan_vertices)} vertices "
               f"(isolated tips: {len(boundary_info.isolated_tips)}, "
               f"near tips: {len(boundary_info.near_tips)}, "
               f"sharp corners: {len(boundary_info.sharp_corners)}, "
               f"other: {len(all_fan_vertices) - len(boundary_info.isolated_tips) - len(boundary_info.near_tips) - len(boundary_info.sharp_corners)})")
    
    for vi in all_fan_vertices:
        vi_pos = vertices_arr[vi]
        edges_at_vi = boundary_info.vertex_to_edges.get(vi, [])
        
        if len(edges_at_vi) < 2:
            continue
        
        # Collect collar info for each edge
        collar_infos = []
        corner_normals = []
        
        for edge_key, _, other_v in edges_at_vi:
            collar_data = edge_endpoint_collar.get(edge_key, {})
            c_idx = collar_data.get(vi)
            if c_idx is not None:
                collar_infos.append((c_idx, np.array(vertices[c_idx])))
                face_info = edge_to_face.get(edge_key)
                if face_info and face_info[0] < len(membrane_face_normals):
                    corner_normals.append(membrane_face_normals[face_info[0]])
        
        if len(collar_infos) < 2:
            continue
        
        # Compute average reference normal
        ref_normal = np.mean(corner_normals, axis=0) if corner_normals else np.array([0, 0, 1])
        ref_len = np.linalg.norm(ref_normal)
        ref_normal = ref_normal / ref_len if ref_len > 1e-8 else np.array([0, 0, 1])
        
        # Pre-compute reference x_axis and y_axis for angle sorting
        # (avoids closure late-binding issues with collar_infos[0])
        first_collar_vec = collar_infos[0][1] - vi_pos
        first_collar_proj = first_collar_vec - np.dot(first_collar_vec, ref_normal) * ref_normal
        fc_len = np.linalg.norm(first_collar_proj)
        x_axis_ref = first_collar_proj / fc_len if fc_len > 1e-8 else np.array([1, 0, 0])
        y_axis_ref = np.cross(ref_normal, x_axis_ref)
        
        # Sort collar vertices by angle around vi
        def collar_angle(ci, vi_pos=vi_pos, ref_normal=ref_normal, 
                         x_axis=x_axis_ref, y_axis=y_axis_ref):
            _, pos = ci
            d = pos - vi_pos
            d_proj = d - np.dot(d, ref_normal) * ref_normal
            d_len = np.linalg.norm(d_proj)
            if d_len < 1e-8:
                return 0.0
            d_proj = d_proj / d_len
            return np.arctan2(np.dot(d_proj, y_axis), np.dot(d_proj, x_axis))
        
        collar_infos.sort(key=collar_angle)
        
        # Create fan triangles
        arc_v, fan_t = _create_fan_triangles(
            vi, vi_pos, collar_infos, ref_normal, vertices, faces,
            part_mesh, part_face_normals, collar_depth, fan_subdivisions
        )
        total_arc_verts += arc_v
        total_fan_tris += fan_t
    
    logger.info(f"Created {total_fan_tris} fan triangles ({total_arc_verts} arc vertices)")
    
    # =========================================================================
    # STEP 7: Create result mesh
    # =========================================================================
    
    result.new_vertices_added = len(vertices) - n_orig_verts
    result.fill_triangles_added = len(faces) - n_orig_faces
    
    try:
        result.mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
            process=False
        )
        result.mesh.fix_normals()
        result.vertices = np.array(result.mesh.vertices)
        result.faces = np.array(result.mesh.faces)
    except Exception as e:
        logger.error(f"Failed to create result mesh: {e}")
        result.mesh = membrane_mesh
        result.vertices = np.array(membrane_mesh.vertices)
        result.faces = np.array(membrane_mesh.faces)
    
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Collar extension complete: {len(inner_boundary_edges)} edges, "
               f"{quads_created} quads, {total_fan_tris} fan triangles, "
               f"{result.new_vertices_added} new vertices in {result.processing_time_ms:.1f}ms")
    
    return result


def fill_floating_edge_gaps_parting_surface(
    surface: 'PartingSurfaceResult',
    part_mesh: trimesh.Trimesh,
    tolerance_fraction: float = FLOATING_EDGE_TOLERANCE_FRACTION,
    min_tolerance: float = FLOATING_EDGE_MIN_TOLERANCE,
    collar_depth: float = 0.5
) -> 'PartingSurfaceResult':
    """
    Fill floating edge gaps in a PartingSurfaceResult using collar extension.
    
    This is a wrapper around create_robust_collar_extension that integrates with
    the PartingSurfaceResult data structure used in the pipeline.
    
    Should be called AFTER smoothing, as smoothing can cause boundary edges
    to "float" away from the part surface.
    
    Args:
        surface: The parting surface result (after smoothing)
        part_mesh: The part mesh to connect to
        tolerance_fraction: Fraction of edge length as tolerance (unused, kept for API compat)
        min_tolerance: Minimum absolute tolerance in mm (unused, kept for API compat)
        collar_depth: How far to extend INTO the part (mm), default 0.5mm
    
    Returns:
        Updated PartingSurfaceResult with floating edges filled
    """
    if surface.mesh is None or part_mesh is None:
        logger.warning("Missing surface or part mesh for floating edge filling")
        return surface
    
    logger.info(f"Using robust collar extension for inner boundary edges (depth={collar_depth}mm)")
    
    # Call the robust collar extension function
    fill_result = create_robust_collar_extension(
        membrane_mesh=surface.mesh,
        part_mesh=part_mesh,
        hull_mesh=None,  # No hull mesh in this context
        vertex_boundary_type=surface.vertex_boundary_type,
        collar_depth=collar_depth,
        fan_subdivisions=4,
        restored_corner_positions=None
    )
    
    if fill_result.mesh is None or fill_result.fill_triangles_added == 0:
        # No changes made
        return surface
    
    # Update the vertex_boundary_type array to include new vertices
    # New vertices from fill are constrained to part surface (-1)
    n_orig_verts = len(surface.vertices) if surface.vertices is not None else 0
    n_new_verts = fill_result.new_vertices_added
    
    if surface.vertex_boundary_type is not None and n_new_verts > 0:
        new_boundary_type = np.zeros(n_orig_verts + n_new_verts, dtype=np.int32)
        new_boundary_type[:n_orig_verts] = surface.vertex_boundary_type
        # New vertices are on part surface (type -1)
        new_boundary_type[n_orig_verts:] = -1
    else:
        new_boundary_type = surface.vertex_boundary_type
    
    # Create updated result
    result = PartingSurfaceResult(
        mesh=fill_result.mesh,
        vertices=fill_result.vertices,
        faces=fill_result.faces,
        vertex_to_edge=surface.vertex_to_edge,  # Original mapping still valid
        vertex_boundary_type=new_boundary_type,
        num_vertices=len(fill_result.vertices),
        num_faces=len(fill_result.faces),
        num_tets_processed=surface.num_tets_processed,
        num_tets_contributing=surface.num_tets_contributing,
        extraction_time_ms=surface.extraction_time_ms + fill_result.processing_time_ms
    )
    
    logger.info(f"Collar extension complete: {fill_result.fill_triangles_added} triangles added, "
                f"{fill_result.new_vertices_added} new vertices")
    
    return result


# =============================================================================
# TUBULAR SECTION DETECTION AND FILLING
# =============================================================================

@dataclass
class TubularSectionInfo:
    """Information about a detected tubular (thin, elongated) boundary loop."""
    loop_vertices: List[int] = None  # Vertex indices forming the loop
    perimeter: float = 0.0           # Total perimeter of the loop
    span: float = 0.0                # Maximum distance across the loop
    aspect_ratio: float = 0.0        # perimeter / span ratio
    centroid: np.ndarray = None      # Center of the loop
    is_tubular: bool = False         # Whether this meets tubular criteria


@dataclass
class TubularDetectionResult:
    """Result of tubular section detection and filling."""
    mesh: Optional[trimesh.Trimesh] = None
    tubular_sections_found: int = 0
    tubular_sections_filled: int = 0
    fill_faces_added: int = 0
    
    # Details of each tubular section
    tubular_info: List[TubularSectionInfo] = None
    
    processing_time_ms: float = 0.0


def detect_tubular_sections(
    mesh: trimesh.Trimesh,
    aspect_threshold: float = TUBULAR_ASPECT_RATIO_THRESHOLD,
    max_perimeter: float = TUBULAR_MAX_PERIMETER,
    min_vertices: int = MIN_TUBULAR_VERTICES
) -> List[TubularSectionInfo]:
    """
    Detect tubular (thin, elongated) boundary loops in a mesh.
    
    Tubular sections are characterized by:
    - High perimeter-to-span ratio (elongated shape)
    - Small overall size (limited perimeter)
    - Closed boundary loops
    
    These often form at the ends of thin protrusions or where the surface
    pinches down to a narrow tube.
    
    Args:
        mesh: Input mesh to analyze
        aspect_threshold: Minimum perimeter/span ratio to consider tubular
        max_perimeter: Maximum perimeter to consider for tubular detection
        min_vertices: Minimum vertices in loop to consider
    
    Returns:
        List of TubularSectionInfo for each detected tubular section
    """
    if mesh is None or len(mesh.faces) == 0:
        return []
    
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    
    # Find boundary edges
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(fi)
    
    boundary_edges = [(v0, v1) for (v0, v1), flist in edge_to_faces.items() if len(flist) == 1]
    
    if not boundary_edges:
        return []
    
    # Build closed loops from boundary edges
    boundary_chains = _build_boundary_chains(np.array(boundary_edges, dtype=np.int64))
    closed_loops = [(chain, is_closed) for chain, is_closed in boundary_chains if is_closed]
    
    tubular_sections = []
    
    for chain_verts, _ in closed_loops:
        if len(chain_verts) < min_vertices:
            continue
        
        # Calculate perimeter
        perimeter = 0.0
        n_verts = len(chain_verts)
        for i in range(n_verts):
            v0 = chain_verts[i]
            v1 = chain_verts[(i + 1) % n_verts]
            perimeter += np.linalg.norm(vertices[v1] - vertices[v0])
        
        if perimeter > max_perimeter:
            continue
        
        # Calculate span (maximum distance between any two loop vertices)
        loop_positions = vertices[chain_verts]
        if len(loop_positions) >= 2:
            from scipy.spatial.distance import pdist
            distances = pdist(loop_positions)
            span = np.max(distances) if len(distances) > 0 else 0.0
        else:
            span = 0.0
        
        # Avoid division by zero
        if span < 1e-6:
            span = 1e-6
        
        aspect_ratio = perimeter / span
        centroid = np.mean(loop_positions, axis=0)
        
        is_tubular = aspect_ratio > aspect_threshold
        
        info = TubularSectionInfo(
            loop_vertices=list(chain_verts),
            perimeter=perimeter,
            span=span,
            aspect_ratio=aspect_ratio,
            centroid=centroid,
            is_tubular=is_tubular
        )
        
        if is_tubular:
            logger.debug(f"Tubular section: {len(chain_verts)} verts, "
                        f"perimeter={perimeter:.2f}, span={span:.2f}, "
                        f"aspect={aspect_ratio:.2f}")
        
        tubular_sections.append(info)
    
    n_tubular = sum(1 for s in tubular_sections if s.is_tubular)
    logger.info(f"Tubular detection: found {n_tubular} tubular sections out of "
                f"{len(closed_loops)} closed loops")
    
    return tubular_sections


def fill_tubular_sections(
    mesh: trimesh.Trimesh,
    tubular_infos: List[TubularSectionInfo] = None,
    aspect_threshold: float = TUBULAR_ASPECT_RATIO_THRESHOLD,
    max_perimeter: float = TUBULAR_MAX_PERIMETER
) -> TubularDetectionResult:
    """
    Fill tubular (thin, elongated) boundary loops with fan triangulation.
    
    This closes thin "tubes" that often form at the ends of protrusions
    or narrow regions of the parting surface.
    
    Args:
        mesh: Input mesh
        tubular_infos: Pre-computed tubular section info (or None to detect)
        aspect_threshold: Minimum perimeter/span ratio to fill
        max_perimeter: Maximum perimeter to consider
    
    Returns:
        TubularDetectionResult with filled mesh
    """
    import time
    start = time.time()
    
    result = TubularDetectionResult()
    result.tubular_info = []
    
    if mesh is None or len(mesh.faces) == 0:
        return result
    
    # Detect tubular sections if not provided
    if tubular_infos is None:
        tubular_infos = detect_tubular_sections(
            mesh, aspect_threshold, max_perimeter
        )
    
    result.tubular_info = tubular_infos
    result.tubular_sections_found = sum(1 for s in tubular_infos if s.is_tubular)
    
    if result.tubular_sections_found == 0:
        result.mesh = mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # Fill tubular sections with fan triangulation
    vertices = list(mesh.vertices)
    faces = list(mesh.faces)
    
    for info in tubular_infos:
        if not info.is_tubular:
            continue
        
        # Add centroid as new vertex
        centroid_idx = len(vertices)
        vertices.append(info.centroid)
        
        # Create fan triangulation
        loop = info.loop_vertices
        n_loop = len(loop)
        for i in range(n_loop):
            v0 = loop[i]
            v1 = loop[(i + 1) % n_loop]
            faces.append([v0, v1, centroid_idx])
            result.fill_faces_added += 1
        
        result.tubular_sections_filled += 1
    
    # Create result mesh
    try:
        result.mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
            process=False
        )
        result.mesh.fix_normals()
    except Exception as e:
        logger.error(f"Failed to create filled mesh: {e}")
        result.mesh = mesh
    
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Tubular filling: filled {result.tubular_sections_filled} sections, "
                f"added {result.fill_faces_added} faces in {result.processing_time_ms:.1f}ms")
    
    return result


# =============================================================================
# THIN FLAT SECTION DETECTION
# =============================================================================

@dataclass
class ThinSectionInfo:
    """Information about a detected thin flat section of the mesh."""
    face_indices: List[int] = None   # Triangle indices in this section
    area: float = 0.0                # Total area of the section
    perimeter: float = 0.0           # Boundary perimeter of the section
    bounding_box_dims: np.ndarray = None  # [width, height, depth]
    aspect_ratio: float = 0.0        # Ratio of longest to shortest bbox dimension
    is_thin: bool = False            # Whether this meets thin criteria


@dataclass
class ThinSectionResult:
    """Result of thin section detection."""
    thin_sections_found: int = 0
    thin_sections_removed: int = 0
    faces_removed: int = 0
    mesh: Optional[trimesh.Trimesh] = None
    thin_section_info: List[ThinSectionInfo] = None
    processing_time_ms: float = 0.0


def detect_thin_flat_sections(
    mesh: trimesh.Trimesh,
    area_threshold_fraction: float = 0.02,
    aspect_ratio_threshold: float = 10.0
) -> List[ThinSectionInfo]:
    """
    Detect thin, flat sections of the mesh that may be problematic.
    
    Thin sections are characterized by:
    - Small area relative to total mesh area
    - High aspect ratio (elongated in one direction)
    - Nearly coplanar triangles
    
    These often form as degenerate regions that should be removed or repaired.
    
    Args:
        mesh: Input mesh to analyze
        area_threshold_fraction: Maximum area fraction to consider thin
        aspect_ratio_threshold: Minimum aspect ratio to consider thin
    
    Returns:
        List of ThinSectionInfo for detected thin sections
    """
    if mesh is None or len(mesh.faces) == 0:
        return []
    
    # Split mesh into connected components
    try:
        components = mesh.split(only_watertight=False)
    except Exception as e:
        logger.warning(f"Could not split mesh for thin section detection: {e}")
        return []
    
    if len(components) <= 1:
        return []
    
    total_area = mesh.area
    thin_sections = []
    
    for comp in components:
        if len(comp.faces) < 3:
            continue
        
        comp_area = comp.area
        area_fraction = comp_area / total_area if total_area > 0 else 0
        
        # Skip components that are too large
        if area_fraction > area_threshold_fraction:
            continue
        
        # Calculate bounding box dimensions
        bbox = comp.bounds
        dims = bbox[1] - bbox[0]  # [width, height, depth]
        
        # Calculate aspect ratio (longest / shortest dimension)
        dims_sorted = np.sort(dims)
        if dims_sorted[0] > 1e-6:
            aspect_ratio = dims_sorted[2] / dims_sorted[0]
        else:
            aspect_ratio = float('inf')
        
        # Calculate boundary perimeter
        boundary_edges = []
        edge_count = {}
        for face in comp.faces:
            for i in range(3):
                e = tuple(sorted([face[i], face[(i+1)%3]]))
                edge_count[e] = edge_count.get(e, 0) + 1
        
        perimeter = 0.0
        for e, count in edge_count.items():
            if count == 1:  # Boundary edge
                perimeter += np.linalg.norm(comp.vertices[e[1]] - comp.vertices[e[0]])
        
        is_thin = aspect_ratio > aspect_ratio_threshold
        
        info = ThinSectionInfo(
            face_indices=list(range(len(comp.faces))),
            area=comp_area,
            perimeter=perimeter,
            bounding_box_dims=dims,
            aspect_ratio=aspect_ratio,
            is_thin=is_thin
        )
        
        if is_thin:
            logger.debug(f"Thin section: {len(comp.faces)} faces, "
                        f"area={comp_area:.4f}, aspect={aspect_ratio:.2f}")
        
        thin_sections.append(info)
    
    n_thin = sum(1 for s in thin_sections if s.is_thin)
    logger.info(f"Thin section detection: found {n_thin} thin sections out of "
                f"{len(components)-1} small components")
    
    return thin_sections


def remove_thin_flat_sections(
    mesh: trimesh.Trimesh,
    area_threshold_fraction: float = 0.02,
    aspect_ratio_threshold: float = 10.0
) -> ThinSectionResult:
    """
    Remove thin, flat sections from the mesh.
    
    Args:
        mesh: Input mesh
        area_threshold_fraction: Maximum area fraction to consider thin
        aspect_ratio_threshold: Minimum aspect ratio to consider thin
    
    Returns:
        ThinSectionResult with cleaned mesh
    """
    import time
    start = time.time()
    
    result = ThinSectionResult()
    result.thin_section_info = []
    
    if mesh is None or len(mesh.faces) == 0:
        return result
    
    # Split mesh into connected components
    try:
        components = mesh.split(only_watertight=False)
    except Exception as e:
        logger.warning(f"Could not split mesh: {e}")
        result.mesh = mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    if len(components) <= 1:
        result.mesh = mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    total_area = mesh.area
    kept_components = []
    
    for comp in components:
        if len(comp.faces) < 3:
            result.faces_removed += len(comp.faces)
            continue
        
        comp_area = comp.area
        area_fraction = comp_area / total_area if total_area > 0 else 0
        
        # Large components are always kept
        if area_fraction > area_threshold_fraction:
            kept_components.append(comp)
            continue
        
        # Calculate aspect ratio
        bbox = comp.bounds
        dims = bbox[1] - bbox[0]
        dims_sorted = np.sort(dims)
        if dims_sorted[0] > 1e-6:
            aspect_ratio = dims_sorted[2] / dims_sorted[0]
        else:
            aspect_ratio = float('inf')
        
        info = ThinSectionInfo(
            face_indices=list(range(len(comp.faces))),
            area=comp_area,
            bounding_box_dims=dims,
            aspect_ratio=aspect_ratio,
            is_thin=(aspect_ratio > aspect_ratio_threshold)
        )
        result.thin_section_info.append(info)
        
        if info.is_thin:
            result.thin_sections_found += 1
            result.thin_sections_removed += 1
            result.faces_removed += len(comp.faces)
            logger.debug(f"Removing thin section: {len(comp.faces)} faces, aspect={aspect_ratio:.2f}")
        else:
            kept_components.append(comp)
    
    # Reconstruct mesh from kept components
    if len(kept_components) == 0:
        logger.warning("All components removed as thin sections!")
        result.mesh = mesh  # Return original if all would be removed
    elif len(kept_components) == 1:
        result.mesh = kept_components[0]
    else:
        result.mesh = trimesh.util.concatenate(kept_components)
    
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Thin section removal: removed {result.thin_sections_removed} sections, "
                f"{result.faces_removed} faces in {result.processing_time_ms:.1f}ms")
    
    return result


# =============================================================================
# SELF-FOLDING / SELF-INTERSECTION DETECTION
# =============================================================================

@dataclass
class SelfFoldingInfo:
    """Information about detected self-folding regions."""
    problem_face_indices: List[int] = None   # Faces with normal inconsistency
    flipped_normal_count: int = 0            # Number of faces with flipped normals
    self_intersection_count: int = 0         # Number of self-intersecting face pairs
    avg_normal_consistency: float = 1.0      # Average dot product of adjacent normals (1.0 = consistent)


@dataclass
class SelfFoldingResult:
    """Result of self-folding detection and repair."""
    mesh: Optional[trimesh.Trimesh] = None
    self_folding_detected: bool = False
    flipped_faces_found: int = 0
    flipped_faces_fixed: int = 0
    self_intersections_found: int = 0
    faces_removed_for_intersection: int = 0
    info: Optional[SelfFoldingInfo] = None
    processing_time_ms: float = 0.0


def detect_self_folding(
    mesh: trimesh.Trimesh,
    normal_consistency_threshold: float = 0.0  # cos(90°) - adjacent faces shouldn't fold >90°
) -> SelfFoldingInfo:
    """
    Detect self-folding regions where the mesh folds back on itself.
    
    Self-folding is detected by:
    1. Adjacent faces with inconsistent normal directions
    2. Sharp changes in normal direction (>90 degree turns)
    
    This does NOT perform self-intersection tests (expensive).
    Use detect_self_intersections() for that.
    
    Args:
        mesh: Input mesh to analyze
        normal_consistency_threshold: Minimum dot product of adjacent face normals
                                     (0.0 = allow up to 90°, -1.0 = allow all)
    
    Returns:
        SelfFoldingInfo with detection results
    """
    info = SelfFoldingInfo(problem_face_indices=[])
    
    if mesh is None or len(mesh.faces) == 0:
        return info
    
    # Get face normals
    try:
        face_normals = mesh.face_normals
    except Exception as e:
        logger.warning(f"Could not compute face normals: {e}")
        return info
    
    if face_normals is None or len(face_normals) == 0:
        return info
    
    # Build face adjacency (faces sharing an edge)
    edge_to_faces = {}
    faces = mesh.faces
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(fi)
    
    # Check normal consistency between adjacent faces
    normal_consistency_sum = 0.0
    n_comparisons = 0
    problem_faces = set()
    
    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue  # Boundary edge or non-manifold
        
        fi, fj = face_list
        n1 = face_normals[fi]
        n2 = face_normals[fj]
        
        # Dot product indicates alignment (1 = same direction, -1 = opposite)
        dot = np.dot(n1, n2)
        normal_consistency_sum += dot
        n_comparisons += 1
        
        if dot < normal_consistency_threshold:
            # These faces have inconsistent normals (folding)
            problem_faces.add(fi)
            problem_faces.add(fj)
            info.flipped_normal_count += 1
    
    info.problem_face_indices = list(problem_faces)
    
    if n_comparisons > 0:
        info.avg_normal_consistency = normal_consistency_sum / n_comparisons
    
    if len(problem_faces) > 0:
        logger.info(f"Self-folding detection: {len(problem_faces)} problem faces, "
                   f"{info.flipped_normal_count} inconsistent normal pairs, "
                   f"avg consistency: {info.avg_normal_consistency:.3f}")
    
    return info


def detect_self_intersections(
    mesh: trimesh.Trimesh,
    sample_fraction: float = 0.1
) -> List[Tuple[int, int]]:
    """
    Detect self-intersecting triangles in the mesh.
    
    This is an expensive operation, so we optionally sample a fraction of faces.
    
    Args:
        mesh: Input mesh to analyze
        sample_fraction: Fraction of faces to sample (1.0 = all faces)
    
    Returns:
        List of (face_i, face_j) pairs that intersect
    """
    if mesh is None or len(mesh.faces) < 2:
        return []
    
    try:
        # Use trimesh's built-in intersection detection if available
        if hasattr(mesh, 'ray') and hasattr(mesh.ray, 'intersects_id'):
            # Ray-based approximate detection
            pass
        
        # Simple bounding box-based culling for potential intersections
        # (Full triangle-triangle intersection is O(n²) expensive)
        
        faces = mesh.faces
        vertices = mesh.vertices
        n_faces = len(faces)
        
        # Compute face bounding boxes
        face_mins = np.zeros((n_faces, 3))
        face_maxs = np.zeros((n_faces, 3))
        
        for fi in range(n_faces):
            face_verts = vertices[faces[fi]]
            face_mins[fi] = face_verts.min(axis=0)
            face_maxs[fi] = face_verts.max(axis=0)
        
        # Sample faces if requested
        if sample_fraction < 1.0:
            n_sample = max(1, int(n_faces * sample_fraction))
            sample_indices = np.random.choice(n_faces, n_sample, replace=False)
        else:
            sample_indices = np.arange(n_faces)
        
        intersecting_pairs = []
        
        # Check each sampled face against others with overlapping bboxes
        for fi in sample_indices:
            # Find faces with potentially overlapping bboxes
            overlap_mask = np.all(face_mins <= face_maxs[fi], axis=1) & \
                          np.all(face_maxs >= face_mins[fi], axis=1)
            overlap_indices = np.where(overlap_mask)[0]
            
            for fj in overlap_indices:
                if fj <= fi:  # Skip self and already-checked pairs
                    continue
                
                # Check if triangles share a vertex (adjacent triangles aren't self-intersecting)
                if len(set(faces[fi]) & set(faces[fj])) > 0:
                    continue
                
                # Do detailed triangle-triangle intersection test
                # (Using simplified edge-plane test)
                tri_i = vertices[faces[fi]]
                tri_j = vertices[faces[fj]]
                
                if _triangles_intersect(tri_i, tri_j):
                    intersecting_pairs.append((fi, fj))
        
        if len(intersecting_pairs) > 0:
            logger.info(f"Self-intersection detection: found {len(intersecting_pairs)} "
                       f"intersecting face pairs (sampled {len(sample_indices)}/{n_faces} faces)")
        
        return intersecting_pairs
        
    except Exception as e:
        logger.warning(f"Self-intersection detection failed: {e}")
        return []


def _triangles_intersect(tri1: np.ndarray, tri2: np.ndarray) -> bool:
    """
    Test if two triangles intersect in 3D space.
    
    Uses the Möller triangle-triangle intersection test.
    
    Args:
        tri1: 3x3 array of triangle 1 vertices
        tri2: 3x3 array of triangle 2 vertices
    
    Returns:
        True if triangles intersect
    """
    # Simple separation axis test (approximate)
    # Check if all vertices of one triangle are on one side of the other's plane
    
    def plane_side(point, tri):
        """Return signed distance from point to triangle's plane."""
        v0 = tri[0]
        n = np.cross(tri[1] - v0, tri[2] - v0)
        n_len = np.linalg.norm(n)
        if n_len < 1e-10:
            return 0  # Degenerate triangle
        n = n / n_len
        return np.dot(point - v0, n)
    
    # Check if all vertices of tri2 are on same side of tri1's plane
    sides1 = [plane_side(tri2[i], tri1) for i in range(3)]
    if all(s > 1e-8 for s in sides1) or all(s < -1e-8 for s in sides1):
        return False
    
    # Check if all vertices of tri1 are on same side of tri2's plane
    sides2 = [plane_side(tri1[i], tri2) for i in range(3)]
    if all(s > 1e-8 for s in sides2) or all(s < -1e-8 for s in sides2):
        return False
    
    # Triangles potentially intersect (conservative - may have false positives)
    # For a full implementation, would need the interval overlap test
    return True


def repair_self_folding(
    mesh: trimesh.Trimesh,
    normal_consistency_threshold: float = 0.0,
    remove_intersecting: bool = False,
    max_iterations: int = 3,
    unfold_iterations: int = 5,
    unfold_strength: float = 0.3
) -> SelfFoldingResult:
    """
    Attempt to repair self-folding regions in the mesh.
    
    Self-folding occurs when portions of the membrane fold back on themselves,
    creating regions where faces have opposite normals to their neighbors.
    
    Repair strategies:
    1. Identify folded regions by detecting faces with inconsistent normals
    2. Apply targeted local smoothing to "unfold" these regions
    3. Fix face normals for consistency
    4. Optionally remove faces that can't be repaired
    
    Args:
        mesh: Input mesh
        normal_consistency_threshold: Minimum dot product for normal consistency
                                     (0.0 = allow up to 90°)
        remove_intersecting: If True, remove self-intersecting faces
        max_iterations: Maximum iterations for the repair loop
        unfold_iterations: Number of local smoothing iterations for folded regions
        unfold_strength: Strength of local smoothing (0-1)
    
    Returns:
        SelfFoldingResult with repaired mesh
    """
    import time
    start = time.time()
    
    result = SelfFoldingResult()
    
    if mesh is None or len(mesh.faces) == 0:
        return result
    
    # Work with a copy
    working_mesh = mesh.copy()
    vertices = working_mesh.vertices.copy()
    faces = working_mesh.faces
    
    # Step 1: Detect self-folding
    info = detect_self_folding(working_mesh, normal_consistency_threshold)
    result.info = info
    result.flipped_faces_found = info.flipped_normal_count
    
    if info.flipped_normal_count == 0 and not remove_intersecting:
        result.mesh = working_mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    result.self_folding_detected = True
    
    # Step 2: Identify vertices in folded regions
    problem_faces = set(info.problem_face_indices)
    problem_vertices = set()
    for fi in problem_faces:
        problem_vertices.update(faces[fi])
    
    if len(problem_vertices) > 0:
        logger.debug(f"Found {len(problem_vertices)} vertices in folded regions")
        
        # Build vertex adjacency
        vertex_neighbors = {v: set() for v in range(len(vertices))}
        for face in faces:
            for i in range(3):
                vi = face[i]
                for j in range(3):
                    if i != j:
                        vertex_neighbors[vi].add(face[j])
        
        # Step 3: Apply targeted local smoothing to unfold regions
        # Only smooth vertices that are part of folded faces
        for unfold_iter in range(unfold_iterations):
            new_vertices = vertices.copy()
            
            for vi in problem_vertices:
                neighbors = list(vertex_neighbors[vi])
                if len(neighbors) < 2:
                    continue
                
                # Compute centroid of neighbors
                neighbor_positions = vertices[neighbors]
                centroid = neighbor_positions.mean(axis=0)
                
                # Move toward centroid (unfold)
                new_vertices[vi] = vertices[vi] + unfold_strength * (centroid - vertices[vi])
            
            vertices = new_vertices
        
        # Update mesh with unfolded vertices
        working_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Step 4: Fix normals using trimesh's built-in method
    for iteration in range(max_iterations):
        try:
            working_mesh.fix_normals()
            
            # Re-check after fix
            new_info = detect_self_folding(working_mesh, normal_consistency_threshold)
            result.flipped_faces_fixed = info.flipped_normal_count - new_info.flipped_normal_count
            
            if new_info.flipped_normal_count == 0:
                break
                
        except Exception as e:
            logger.warning(f"Normal fix iteration {iteration} failed: {e}")
            break
    
    # Step 5: If still have problems, try removing the worst offending faces
    final_info = detect_self_folding(working_mesh, normal_consistency_threshold)
    
    if final_info.flipped_normal_count > 0 and len(final_info.problem_face_indices) > 0:
        # Find faces that are severely folded (normal opposite to neighbors)
        face_normals = working_mesh.face_normals
        edge_to_faces = {}
        for fi, face in enumerate(working_mesh.faces):
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = (min(v0, v1), max(v0, v1))
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append(fi)
        
        # Score each problem face by how badly it disagrees with neighbors
        face_scores = {}
        for fi in final_info.problem_face_indices:
            score = 0
            n_neighbors = 0
            for i in range(3):
                v0, v1 = int(working_mesh.faces[fi, i]), int(working_mesh.faces[fi, (i + 1) % 3])
                edge_key = (min(v0, v1), max(v0, v1))
                for fj in edge_to_faces.get(edge_key, []):
                    if fj != fi:
                        dot = np.dot(face_normals[fi], face_normals[fj])
                        score += (1.0 - dot) / 2.0  # 0 = aligned, 1 = opposite
                        n_neighbors += 1
            if n_neighbors > 0:
                face_scores[fi] = score / n_neighbors
        
        # Remove faces that are severely reversed (score > 0.7 means mostly opposite)
        faces_to_remove = [fi for fi, score in face_scores.items() if score > 0.7]
        
        if faces_to_remove and len(faces_to_remove) < len(working_mesh.faces) * 0.1:
            # Only remove if less than 10% of faces
            keep_mask = np.ones(len(working_mesh.faces), dtype=bool)
            keep_mask[faces_to_remove] = False
            
            new_faces = working_mesh.faces[keep_mask]
            working_mesh = trimesh.Trimesh(
                vertices=working_mesh.vertices,
                faces=new_faces,
                process=False
            )
            working_mesh.remove_unreferenced_vertices()
            
            result.faces_removed_for_intersection = len(faces_to_remove)
            logger.info(f"Removed {len(faces_to_remove)} severely folded faces")
    
    # Step 6: Optionally detect and remove self-intersecting faces
    if remove_intersecting:
        intersecting_pairs = detect_self_intersections(working_mesh)
        result.self_intersections_found = len(intersecting_pairs)
        
        if len(intersecting_pairs) > 0:
            # Collect all faces involved in intersections
            problem_faces = set()
            for fi, fj in intersecting_pairs:
                problem_faces.add(fi)
                problem_faces.add(fj)
            
            # Remove problematic faces
            keep_mask = np.ones(len(working_mesh.faces), dtype=bool)
            keep_mask[list(problem_faces)] = False
            
            new_faces = working_mesh.faces[keep_mask]
            working_mesh = trimesh.Trimesh(
                vertices=working_mesh.vertices,
                faces=new_faces,
                process=False
            )
            working_mesh.remove_unreferenced_vertices()
            
            result.faces_removed_for_intersection += len(problem_faces)
    
    result.mesh = working_mesh
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Self-folding repair: fixed {result.flipped_faces_fixed} normals, "
                f"removed {result.faces_removed_for_intersection} folded/intersecting faces "
                f"in {result.processing_time_ms:.1f}ms")
    
    return result


# =============================================================================
# MESH QUALITY IMPROVEMENT - FIX THIN TRIANGLES
# =============================================================================

@dataclass
class MeshQualityResult:
    """Result of mesh quality improvement."""
    mesh: Optional[trimesh.Trimesh] = None
    
    # Statistics
    thin_triangles_found: int = 0
    triangles_collapsed: int = 0
    triangles_split: int = 0
    edges_collapsed: int = 0
    vertex_only_removed: int = 0  # Triangles only connected by vertices, not edges
    degenerate_removed: int = 0   # Zero/near-zero area triangles
    duplicate_removed: int = 0    # Duplicate faces
    ear_removed: int = 0          # Triangles with 2 boundary edges
    non_manifold_removed: int = 0 # Triangles on non-manifold edges
    tiny_removed: int = 0         # Very small area triangles
    
    # Quality metrics before/after
    min_aspect_before: float = 0.0
    avg_aspect_before: float = 0.0
    min_aspect_after: float = 0.0
    avg_aspect_after: float = 0.0
    
    # Timing
    processing_time_ms: float = 0.0


def _compute_all_aspect_ratios(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute aspect ratio for all triangles in mesh."""
    triangles = mesh.triangles  # (n_faces, 3, 3)
    
    # Edge vectors
    e0 = triangles[:, 1] - triangles[:, 0]  # V0 -> V1
    e1 = triangles[:, 2] - triangles[:, 1]  # V1 -> V2
    e2 = triangles[:, 0] - triangles[:, 2]  # V2 -> V0
    
    # Edge lengths
    a = np.linalg.norm(e0, axis=1)
    b = np.linalg.norm(e1, axis=1)
    c = np.linalg.norm(e2, axis=1)
    
    # Semi-perimeter
    s = (a + b + c) / 2
    
    # Area via Heron's formula (with numerical stability)
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = np.maximum(area_sq, 0)  # Clamp negative values
    area = np.sqrt(area_sq)
    
    # Aspect ratio = 4 * area^2 / (s * a * b * c) * 2
    # Normalized so equilateral = 1
    denom = s * a * b * c
    denom = np.where(denom < 1e-20, 1e-20, denom)  # Avoid division by zero
    
    ratio = (4 * area * area) / denom
    aspect = np.minimum(ratio * 2, 1.0)
    
    return aspect


def improve_mesh_quality(
    mesh: trimesh.Trimesh,
    min_aspect_ratio: float = 0.15,
    max_iterations: int = 3,
    collapse_thin_triangles: bool = True,
    split_thin_triangles: bool = False
) -> MeshQualityResult:
    """
    Improve mesh quality by fixing thin/degenerate triangles.
    
    This function identifies triangles with low aspect ratios and improves them by:
    1. Edge collapse - collapse the shortest edge of thin triangles
    2. Triangle splitting - split thin triangles at centroid (optional)
    
    Args:
        mesh: Input mesh to improve
        min_aspect_ratio: Triangles below this threshold are considered thin
        max_iterations: Maximum improvement iterations
        collapse_thin_triangles: Whether to collapse edges of thin triangles
        split_thin_triangles: Whether to split thin triangles (adds vertices)
    
    Returns:
        MeshQualityResult with improved mesh and statistics
    """
    import time
    start = time.time()
    
    result = MeshQualityResult()
    
    if mesh is None or len(mesh.faces) == 0:
        return result
    
    # Compute initial quality metrics
    initial_aspects = _compute_all_aspect_ratios(mesh)
    result.min_aspect_before = float(np.min(initial_aspects))
    result.avg_aspect_before = float(np.mean(initial_aspects))
    
    thin_mask = initial_aspects < min_aspect_ratio
    result.thin_triangles_found = int(np.sum(thin_mask))
    
    if result.thin_triangles_found == 0:
        logger.info(f"No thin triangles found (min aspect ratio: {result.min_aspect_before:.4f})")
        result.mesh = mesh
        result.min_aspect_after = result.min_aspect_before
        result.avg_aspect_after = result.avg_aspect_before
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    logger.info(f"Found {result.thin_triangles_found} thin triangles (aspect < {min_aspect_ratio})")
    
    working_mesh = mesh.copy()
    
    for iteration in range(max_iterations):
        # Recompute aspects
        aspects = _compute_all_aspect_ratios(working_mesh)
        thin_mask = aspects < min_aspect_ratio
        n_thin = np.sum(thin_mask)
        
        if n_thin == 0:
            break
        
        logger.debug(f"Iteration {iteration + 1}: {n_thin} thin triangles remaining")
        
        if collapse_thin_triangles:
            # Strategy: collapse shortest edge in thin triangles
            working_mesh, n_collapsed = _collapse_thin_triangle_edges(
                working_mesh, thin_mask, aspects
            )
            result.edges_collapsed += n_collapsed
            result.triangles_collapsed += n_collapsed
        
        if split_thin_triangles and n_thin > 0:
            # Recompute after collapse
            aspects = _compute_all_aspect_ratios(working_mesh)
            thin_mask = aspects < min_aspect_ratio
            
            if np.sum(thin_mask) > 0:
                working_mesh, n_split = _split_thin_triangles(working_mesh, thin_mask)
                result.triangles_split += n_split
    
    # Final quality metrics
    final_aspects = _compute_all_aspect_ratios(working_mesh)
    result.min_aspect_after = float(np.min(final_aspects))
    result.avg_aspect_after = float(np.mean(final_aspects))
    
    # === Additional triangle cleaning passes ===
    
    # 1. Remove degenerate (zero-area) triangles
    working_mesh, n_degen = _remove_degenerate_triangles(working_mesh)
    result.degenerate_removed = n_degen
    
    # 2. Remove duplicate triangles
    working_mesh, n_dup = _remove_duplicate_triangles(working_mesh)
    result.duplicate_removed = n_dup
    
    # 3. Remove ear triangles (2 boundary edges)
    working_mesh, n_ear = _remove_ear_triangles(working_mesh)
    result.ear_removed = n_ear
    
    # 4. Remove triangles on non-manifold edges (>2 faces sharing edge)
    working_mesh, n_nm = _remove_non_manifold_triangles(working_mesh)
    result.non_manifold_removed = n_nm
    
    # 5. Remove vertex-only connected triangles (no shared edges)
    working_mesh, n_vo = _remove_vertex_only_triangles(working_mesh)
    result.vertex_only_removed = n_vo
    
    # 6. Remove very small triangles (area < 1% of median)
    working_mesh, n_tiny = _remove_tiny_triangles(working_mesh, area_threshold_fraction=0.01)
    result.tiny_removed = n_tiny
    
    total_removed = n_degen + n_dup + n_ear + n_nm + n_vo + n_tiny
    
    result.mesh = working_mesh
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Mesh quality improvement: "
                f"collapsed {result.edges_collapsed} edges, split {result.triangles_split} triangles, "
                f"removed: {result.degenerate_removed} degenerate, {result.duplicate_removed} duplicate, "
                f"{result.ear_removed} ear, {result.non_manifold_removed} non-manifold, "
                f"{result.vertex_only_removed} vertex-only, {result.tiny_removed} tiny "
                f"(total {total_removed} removed), "
                f"aspect ratio: {result.min_aspect_before:.4f} → {result.min_aspect_after:.4f} (min) "
                f"in {result.processing_time_ms:.1f}ms")
    
    return result


def _remove_vertex_only_triangles(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, int]:
    """
    Remove triangles that are only connected to the mesh by vertices, not edges.
    
    These are "dangling" triangles that don't share any edges with neighboring triangles,
    only sharing vertices. Such triangles create non-manifold geometry.
    
    Returns:
        Tuple of (new_mesh, num_removed)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    faces = mesh.faces
    n_faces = len(faces)
    
    # Build edge-to-face mapping
    # Each edge is represented as a sorted tuple (min_vertex, max_vertex)
    edge_to_faces = {}
    
    for fi, face in enumerate(faces):
        v0, v1, v2 = face[0], face[1], face[2]
        
        # Three edges per face
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    # Find triangles with no shared edges (all 3 edges have only 1 face)
    vertex_only_triangles = []
    
    for fi, face in enumerate(faces):
        v0, v1, v2 = face[0], face[1], face[2]
        
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        
        # Count edges shared with other triangles
        shared_edges = 0
        for edge in edges:
            if len(edge_to_faces[edge]) > 1:
                shared_edges += 1
        
        # If no edges are shared, this triangle is only vertex-connected
        if shared_edges == 0:
            vertex_only_triangles.append(fi)
    
    if len(vertex_only_triangles) == 0:
        return mesh, 0
    
    logger.debug(f"Found {len(vertex_only_triangles)} vertex-only connected triangles to remove")
    
    # Create mask for faces to keep
    keep_mask = np.ones(n_faces, dtype=bool)
    keep_mask[vertex_only_triangles] = False
    
    new_faces = faces[keep_mask]
    
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=new_faces,
        process=False
    )
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh, len(vertex_only_triangles)


def _remove_degenerate_triangles(mesh: trimesh.Trimesh, area_threshold: float = 1e-10) -> Tuple[trimesh.Trimesh, int]:
    """
    Remove degenerate (zero or near-zero area) triangles.
    
    These are triangles where vertices are collinear or nearly collinear.
    
    Args:
        mesh: Input mesh
        area_threshold: Absolute area below which triangles are considered degenerate
    
    Returns:
        Tuple of (new_mesh, num_removed)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    # Compute triangle areas
    areas = mesh.area_faces
    
    # Find degenerate triangles
    degenerate_mask = areas < area_threshold
    n_degenerate = np.sum(degenerate_mask)
    
    if n_degenerate == 0:
        return mesh, 0
    
    logger.debug(f"Found {n_degenerate} degenerate (zero-area) triangles to remove")
    
    # Keep non-degenerate faces
    keep_mask = ~degenerate_mask
    new_faces = mesh.faces[keep_mask]
    
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=new_faces,
        process=False
    )
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh, int(n_degenerate)


def _remove_duplicate_triangles(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, int]:
    """
    Remove duplicate triangles (same vertices, any winding order).
    
    Returns:
        Tuple of (new_mesh, num_removed)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    faces = mesh.faces
    n_faces = len(faces)
    
    # Create canonical form of each face (sorted vertices)
    canonical = np.sort(faces, axis=1)
    
    # Find unique faces
    _, unique_indices, counts = np.unique(
        canonical, axis=0, return_index=True, return_counts=True
    )
    
    n_duplicates = n_faces - len(unique_indices)
    
    if n_duplicates == 0:
        return mesh, 0
    
    logger.debug(f"Found {n_duplicates} duplicate triangles to remove")
    
    # Keep only unique faces
    new_faces = faces[np.sort(unique_indices)]
    
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=new_faces,
        process=False
    )
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh, n_duplicates


def _remove_ear_triangles(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, int]:
    """
    Remove ear triangles - triangles with 2 boundary edges.
    
    These are triangles that stick out from the mesh with only one edge
    connected to the interior. They're often artifacts from mesh operations.
    
    Returns:
        Tuple of (new_mesh, num_removed)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    faces = mesh.faces
    n_faces = len(faces)
    
    # Build edge-to-face mapping
    edge_to_faces = {}
    
    for fi, face in enumerate(faces):
        v0, v1, v2 = face[0], face[1], face[2]
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    # Find ear triangles (2 boundary edges = 2 edges with only 1 face)
    ear_triangles = []
    
    for fi, face in enumerate(faces):
        v0, v1, v2 = face[0], face[1], face[2]
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        
        # Count boundary edges (edges with only 1 face)
        boundary_edges = sum(1 for e in edges if len(edge_to_faces[e]) == 1)
        
        if boundary_edges >= 2:
            ear_triangles.append(fi)
    
    if len(ear_triangles) == 0:
        return mesh, 0
    
    logger.debug(f"Found {len(ear_triangles)} ear triangles (2+ boundary edges) to remove")
    
    # Create mask for faces to keep
    keep_mask = np.ones(n_faces, dtype=bool)
    keep_mask[ear_triangles] = False
    
    new_faces = faces[keep_mask]
    
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=new_faces,
        process=False
    )
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh, len(ear_triangles)


def _remove_non_manifold_triangles(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, int]:
    """
    Remove triangles that share non-manifold edges (edges with >2 faces).
    
    Non-manifold edges cause problems for CSG operations and make the mesh
    not watertight. We remove all triangles incident to such edges.
    
    Returns:
        Tuple of (new_mesh, num_removed)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    faces = mesh.faces
    n_faces = len(faces)
    
    # Build edge-to-face mapping
    edge_to_faces = {}
    
    for fi, face in enumerate(faces):
        v0, v1, v2 = face[0], face[1], face[2]
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    # Find non-manifold edges (>2 faces)
    non_manifold_edges = {e for e, flist in edge_to_faces.items() if len(flist) > 2}
    
    if len(non_manifold_edges) == 0:
        return mesh, 0
    
    # Find all triangles incident to non-manifold edges
    triangles_to_remove = set()
    for edge in non_manifold_edges:
        triangles_to_remove.update(edge_to_faces[edge])
    
    if len(triangles_to_remove) == 0:
        return mesh, 0
    
    logger.debug(f"Found {len(non_manifold_edges)} non-manifold edges, "
                 f"removing {len(triangles_to_remove)} incident triangles")
    
    # Create mask for faces to keep
    keep_mask = np.ones(n_faces, dtype=bool)
    keep_mask[list(triangles_to_remove)] = False
    
    new_faces = faces[keep_mask]
    
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=new_faces,
        process=False
    )
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh, len(triangles_to_remove)


def _remove_tiny_triangles(mesh: trimesh.Trimesh, area_threshold_fraction: float = 0.01) -> Tuple[trimesh.Trimesh, int]:
    """
    Remove very small triangles relative to the mesh's median triangle area.
    
    Args:
        mesh: Input mesh
        area_threshold_fraction: Remove triangles with area < fraction * median_area
    
    Returns:
        Tuple of (new_mesh, num_removed)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    # Compute triangle areas
    areas = mesh.area_faces
    
    if len(areas) == 0:
        return mesh, 0
    
    # Use median to be robust to outliers
    median_area = np.median(areas)
    threshold = area_threshold_fraction * median_area
    
    # Find tiny triangles
    tiny_mask = areas < threshold
    n_tiny = np.sum(tiny_mask)
    
    if n_tiny == 0:
        return mesh, 0
    
    logger.debug(f"Found {n_tiny} tiny triangles (area < {threshold:.6f}) to remove")
    
    # Keep non-tiny faces
    keep_mask = ~tiny_mask
    new_faces = mesh.faces[keep_mask]
    
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=new_faces,
        process=False
    )
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh, int(n_tiny)


def _collapse_thin_triangle_edges(
    mesh: trimesh.Trimesh,
    thin_mask: np.ndarray,
    aspects: np.ndarray
) -> Tuple[trimesh.Trimesh, int]:
    """
    Collapse shortest edges in thin triangles.
    
    For each thin triangle, identifies the shortest edge and collapses it
    by merging the two vertices to their midpoint.
    
    Returns:
        Tuple of (new_mesh, num_collapsed)
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    thin_indices = np.where(thin_mask)[0]
    
    # Sort by aspect ratio (worst first) to prioritize the thinnest
    sorted_indices = thin_indices[np.argsort(aspects[thin_indices])]
    
    # Track which vertices have been merged
    vertex_map = np.arange(len(vertices))  # Maps old index to new index
    collapsed_count = 0
    max_collapses = min(len(sorted_indices), len(faces) // 10)  # Limit to 10% of faces per iteration
    
    for fi in sorted_indices[:max_collapses]:
        if fi >= len(faces):
            continue
        
        face = faces[fi]
        v0, v1, v2 = face[0], face[1], face[2]
        
        # Map to current vertex indices
        v0, v1, v2 = vertex_map[v0], vertex_map[v1], vertex_map[v2]
        
        # Skip if already collapsed (vertices merged)
        if v0 == v1 or v1 == v2 or v2 == v0:
            continue
        
        # Get vertex positions
        p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
        
        # Find shortest edge
        e01 = np.linalg.norm(p1 - p0)
        e12 = np.linalg.norm(p2 - p1)
        e20 = np.linalg.norm(p0 - p2)
        
        min_edge = min(e01, e12, e20)
        
        # Collapse shortest edge
        if min_edge == e01:
            # Merge v0 and v1 to midpoint
            midpoint = (p0 + p1) / 2
            vertices[v0] = midpoint
            vertex_map[vertex_map == v1] = v0
        elif min_edge == e12:
            # Merge v1 and v2 to midpoint
            midpoint = (p1 + p2) / 2
            vertices[v1] = midpoint
            vertex_map[vertex_map == v2] = v1
        else:
            # Merge v2 and v0 to midpoint
            midpoint = (p2 + p0) / 2
            vertices[v2] = midpoint
            vertex_map[vertex_map == v0] = v2
        
        collapsed_count += 1
    
    if collapsed_count == 0:
        return mesh, 0
    
    # Remap faces to new vertex indices
    new_faces = vertex_map[faces]
    
    # Remove degenerate faces (where any two vertices are the same)
    valid_mask = (new_faces[:, 0] != new_faces[:, 1]) & \
                 (new_faces[:, 1] != new_faces[:, 2]) & \
                 (new_faces[:, 2] != new_faces[:, 0])
    
    new_faces = new_faces[valid_mask]
    
    # Create new mesh
    new_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=new_faces,
        process=False
    )
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh, collapsed_count


def _split_thin_triangles(
    mesh: trimesh.Trimesh,
    thin_mask: np.ndarray
) -> Tuple[trimesh.Trimesh, int]:
    """
    Split thin triangles by adding a vertex at centroid.
    
    Each thin triangle is replaced by 3 triangles sharing the centroid.
    This can help improve aspect ratios in some cases.
    
    Returns:
        Tuple of (new_mesh, num_split)
    """
    vertices = list(mesh.vertices)
    faces = list(mesh.faces)
    
    thin_indices = np.where(thin_mask)[0]
    split_count = 0
    
    # Collect faces to remove and new faces to add
    faces_to_remove = set()
    new_faces = []
    
    for fi in thin_indices:
        face = faces[fi]
        v0, v1, v2 = face[0], face[1], face[2]
        
        # Compute centroid
        p0, p1, p2 = mesh.vertices[v0], mesh.vertices[v1], mesh.vertices[v2]
        centroid = (p0 + p1 + p2) / 3
        
        # Add centroid as new vertex
        centroid_idx = len(vertices)
        vertices.append(centroid)
        
        # Create 3 new faces
        new_faces.append([v0, v1, centroid_idx])
        new_faces.append([v1, v2, centroid_idx])
        new_faces.append([v2, v0, centroid_idx])
        
        faces_to_remove.add(fi)
        split_count += 1
    
    # Build final face list
    final_faces = [f for i, f in enumerate(faces) if i not in faces_to_remove]
    final_faces.extend(new_faces)
    
    new_mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(final_faces),
        process=False
    )
    
    return new_mesh, split_count


# =============================================================================
# COMPREHENSIVE COMPONENT CLEANUP - KEEP ONLY MAIN MEMBRANE
# =============================================================================

@dataclass
class ComponentCleanupResult:
    """Result of comprehensive component cleanup."""
    mesh: Optional[trimesh.Trimesh] = None
    
    # Statistics
    total_components_found: int = 0
    components_removed: int = 0
    faces_removed: int = 0
    
    # Main component info
    main_component_faces: int = 0
    main_component_area: float = 0.0
    
    # Reasons for removal
    removed_by_size: int = 0
    removed_by_area: int = 0
    removed_by_intersection: int = 0
    removed_by_distance: int = 0
    
    # Timing
    processing_time_ms: float = 0.0


def _check_meshes_intersect(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, 
                            sample_count: int = 50) -> bool:
    """
    Check if two meshes intersect by sampling points and checking containment.
    
    Uses a fast approximation: samples points from mesh1's faces and checks
    if any are inside mesh2's convex hull or very close to mesh2.
    
    Args:
        mesh1: First mesh
        mesh2: Second mesh
        sample_count: Number of sample points to check
    
    Returns:
        True if meshes likely intersect
    """
    try:
        # Quick bounding box check first
        min1, max1 = mesh1.bounds
        min2, max2 = mesh2.bounds
        
        # If bounding boxes don't overlap, meshes can't intersect
        if (max1[0] < min2[0] or min1[0] > max2[0] or
            max1[1] < min2[1] or min1[1] > max2[1] or
            max1[2] < min2[2] or min1[2] > max2[2]):
            return False
        
        # Sample points from mesh1's face centroids
        n_faces = len(mesh1.faces)
        if n_faces == 0:
            return False
        
        sample_indices = np.random.choice(n_faces, min(sample_count, n_faces), replace=False)
        sample_points = mesh1.triangles_center[sample_indices]
        
        # Check proximity to mesh2
        _, distances, _ = trimesh.proximity.closest_point(mesh2, sample_points)
        
        # If any sample point is very close to mesh2, consider it intersecting
        # Use a threshold based on mesh2's average edge length
        edges = mesh2.edges_unique
        if len(edges) > 0:
            edge_lengths = np.linalg.norm(
                mesh2.vertices[edges[:, 1]] - mesh2.vertices[edges[:, 0]], axis=1
            )
            threshold = np.median(edge_lengths) * 0.5
        else:
            threshold = 0.1
        
        if np.any(distances < threshold):
            return True
        
        return False
        
    except Exception as e:
        logger.debug(f"Intersection check failed: {e}")
        return False


def _compute_component_distance_to_main(component: trimesh.Trimesh, 
                                        main_mesh: trimesh.Trimesh) -> float:
    """
    Compute the minimum distance from a component to the main mesh.
    
    Args:
        component: The component mesh
        main_mesh: The main/largest mesh
    
    Returns:
        Minimum distance between component boundary and main mesh
    """
    try:
        # Sample points from component
        n_verts = len(component.vertices)
        sample_count = min(50, n_verts)
        sample_indices = np.random.choice(n_verts, sample_count, replace=False)
        sample_points = component.vertices[sample_indices]
        
        # Find closest points on main mesh
        _, distances, _ = trimesh.proximity.closest_point(main_mesh, sample_points)
        
        return float(np.min(distances))
        
    except Exception as e:
        logger.debug(f"Distance computation failed: {e}")
        return float('inf')


def cleanup_membrane_components(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: Optional[trimesh.Trimesh] = None,
    min_area_fraction: float = 0.02,
    min_face_count: int = 20,
    max_distance_from_main: float = None,
    remove_intersecting: bool = True
) -> ComponentCleanupResult:
    """
    Comprehensive cleanup of membrane mesh to keep only the main component.
    
    This function identifies and removes:
    1. Small disconnected islands (by face count and area)
    2. Components that intersect with the main membrane (self-intersection artifacts)
    3. Components that are too far from the main membrane (orphans)
    
    The "main" component is identified as the largest by area.
    
    Args:
        membrane_mesh: The membrane mesh to clean up
        part_mesh: Optional part mesh for context (unused currently but reserved)
        min_area_fraction: Components with area < this fraction of main are removed
        min_face_count: Components with fewer faces than this are always removed
        max_distance_from_main: Components farther than this from main are removed (auto if None)
        remove_intersecting: Whether to remove components that intersect the main
    
    Returns:
        ComponentCleanupResult with cleaned mesh and statistics
    """
    import time
    start = time.time()
    
    result = ComponentCleanupResult()
    
    if membrane_mesh is None or len(membrane_mesh.faces) == 0:
        logger.warning("No membrane mesh provided for component cleanup")
        return result
    
    # Split into connected components
    try:
        components = membrane_mesh.split(only_watertight=False)
    except Exception as e:
        logger.warning(f"Could not split mesh into components: {e}")
        result.mesh = membrane_mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    result.total_components_found = len(components)
    
    if len(components) <= 1:
        # Single component - nothing to clean
        result.mesh = membrane_mesh
        if len(components) == 1:
            result.main_component_faces = len(components[0].faces)
            result.main_component_area = float(components[0].area)
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    logger.info(f"Found {len(components)} connected components in membrane")
    
    # Compute area for each component and find the main (largest) one
    component_info = []
    for i, comp in enumerate(components):
        area = float(comp.area) if len(comp.faces) > 0 else 0.0
        n_faces = len(comp.faces)
        component_info.append({
            'index': i,
            'mesh': comp,
            'area': area,
            'n_faces': n_faces
        })
    
    # Sort by area (descending) - largest is main
    component_info.sort(key=lambda x: x['area'], reverse=True)
    
    main_component = component_info[0]
    main_mesh = main_component['mesh']
    main_area = main_component['area']
    
    result.main_component_faces = main_component['n_faces']
    result.main_component_area = main_area
    
    logger.info(f"Main component: {main_component['n_faces']} faces, area={main_area:.4f}")
    
    # Auto-compute max distance if not provided
    if max_distance_from_main is None:
        mesh_scale = np.linalg.norm(membrane_mesh.bounds[1] - membrane_mesh.bounds[0])
        max_distance_from_main = mesh_scale * 0.1  # 10% of mesh scale
    
    # Evaluate each non-main component
    kept_components = [main_mesh]
    
    for info in component_info[1:]:  # Skip main (index 0)
        comp = info['mesh']
        area = info['area']
        n_faces = info['n_faces']
        should_remove = False
        removal_reason = None
        
        # Check 1: Minimum face count
        if n_faces < min_face_count:
            should_remove = True
            removal_reason = 'size'
            result.removed_by_size += 1
        
        # Check 2: Minimum area fraction
        elif main_area > 0 and (area / main_area) < min_area_fraction:
            should_remove = True
            removal_reason = 'area'
            result.removed_by_area += 1
        
        # Check 3: Intersection with main (artifact)
        elif remove_intersecting and _check_meshes_intersect(comp, main_mesh):
            should_remove = True
            removal_reason = 'intersection'
            result.removed_by_intersection += 1
        
        # Check 4: Distance from main
        else:
            dist = _compute_component_distance_to_main(comp, main_mesh)
            if dist > max_distance_from_main:
                should_remove = True
                removal_reason = 'distance'
                result.removed_by_distance += 1
        
        if should_remove:
            result.components_removed += 1
            result.faces_removed += n_faces
            logger.debug(f"Removing component with {n_faces} faces, area={area:.4f} (reason: {removal_reason})")
        else:
            kept_components.append(comp)
            logger.debug(f"Keeping component with {n_faces} faces, area={area:.4f}")
    
    # Combine kept components
    if len(kept_components) == 1:
        result.mesh = kept_components[0]
    else:
        result.mesh = trimesh.util.concatenate(kept_components)
    
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Component cleanup: removed {result.components_removed}/{result.total_components_found-1} "
                f"secondary components ({result.faces_removed} faces) - "
                f"by size: {result.removed_by_size}, by area: {result.removed_by_area}, "
                f"by intersection: {result.removed_by_intersection}, by distance: {result.removed_by_distance} "
                f"in {result.processing_time_ms:.1f}ms")
    
    return result


# =============================================================================
# ROBUST GAP CLOSING WITH QUALITY-CONTROLLED TRIANGULATION
# =============================================================================

@dataclass
class GapClosingResult:
    """Result of closing gaps between membrane boundary and part surface."""
    mesh: Optional[trimesh.Trimesh] = None
    
    # Statistics
    boundary_loops_found: int = 0
    inner_boundary_loops: int = 0
    gap_faces_added: int = 0
    gap_vertices_added: int = 0
    edges_subdivided: int = 0
    thin_triangles_avoided: int = 0
    
    # Average gap distance before closing
    avg_gap_distance: float = 0.0
    max_gap_distance: float = 0.0
    
    # Quality metrics
    min_triangle_aspect_ratio: float = 0.0
    avg_triangle_aspect_ratio: float = 0.0
    
    # Timing
    processing_time_ms: float = 0.0


def _compute_triangle_aspect_ratio(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute aspect ratio of a triangle (0 = degenerate, 1 = equilateral).
    
    Uses the ratio of the inscribed circle radius to the circumscribed circle radius,
    normalized to 1 for an equilateral triangle.
    """
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p0 - p2)
    
    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        return 0.0
    
    # Semi-perimeter
    s = (a + b + c) / 2
    
    # Area via Heron's formula
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return 0.0
    area = np.sqrt(area_sq)
    
    # Inradius = area / s
    # Circumradius = (a * b * c) / (4 * area)
    # Ratio = inradius / circumradius = 4 * area^2 / (s * a * b * c)
    # For equilateral: ratio = 0.5, so we normalize by 2
    ratio = (4 * area * area) / (s * a * b * c)
    return min(1.0, ratio * 2)


def _find_best_projection_on_part(
    point: np.ndarray,
    part_mesh: trimesh.Trimesh,
    neighbor_projections: List[np.ndarray],
    search_radius: float
) -> Tuple[np.ndarray, float]:
    """
    Find the best projection point on part surface considering neighbors.
    
    Instead of just closest point, this considers:
    1. Distance to original point
    2. Smoothness with neighboring projections
    3. Avoids projecting to topologically distant parts
    
    Returns:
        Tuple of (projection_point, distance)
    """
    # Get closest point as baseline
    closest_pts, closest_dists, closest_faces = trimesh.proximity.closest_point(
        part_mesh, point.reshape(1, 3)
    )
    closest_pt = closest_pts[0]
    closest_dist = closest_dists[0]
    
    # If no neighbors or very close, just use closest
    if len(neighbor_projections) == 0 or closest_dist < 1e-6:
        return closest_pt, closest_dist
    
    # Compute neighbor centroid
    neighbor_center = np.mean(neighbor_projections, axis=0)
    
    # If closest point is near the neighbor centroid, use it
    dist_to_neighbors = np.linalg.norm(closest_pt - neighbor_center)
    avg_neighbor_spread = np.mean([np.linalg.norm(p - neighbor_center) for p in neighbor_projections])
    
    # If within expected spread, accept the closest point
    if dist_to_neighbors < avg_neighbor_spread * 3 + search_radius:
        return closest_pt, closest_dist
    
    # Otherwise, try to find a better point by sampling
    # This handles cases where closest point jumps to wrong part of mesh
    
    # Sample points in direction from point toward neighbor center
    direction = neighbor_center - point
    if np.linalg.norm(direction) > 1e-10:
        direction = direction / np.linalg.norm(direction)
        
        # Try a point closer to neighbor center
        candidate = point + direction * closest_dist * 0.5
        cand_pts, cand_dists, _ = trimesh.proximity.closest_point(
            part_mesh, candidate.reshape(1, 3)
        )
        
        cand_pt = cand_pts[0]
        cand_dist_to_neighbors = np.linalg.norm(cand_pt - neighbor_center)
        
        # Use candidate if it's closer to neighbors and not too far from original point
        total_dist = np.linalg.norm(cand_pt - point)
        if cand_dist_to_neighbors < dist_to_neighbors * 0.7 and total_dist < search_radius:
            return cand_pt, total_dist
    
    return closest_pt, closest_dist


def _subdivide_gap_edge(
    v0: np.ndarray, v1: np.ndarray,
    p0: np.ndarray, p1: np.ndarray,
    part_mesh: trimesh.Trimesh,
    target_edge_length: float,
    min_aspect_ratio: float = 0.15
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Subdivide a gap edge if it would create thin triangles.
    
    Given boundary edge (v0, v1) and their projections (p0, p1), determines
    if subdivision is needed and returns interpolated vertices.
    
    Returns:
        Tuple of (boundary_points, projection_points) including endpoints
    """
    boundary_len = np.linalg.norm(v1 - v0)
    projection_len = np.linalg.norm(p1 - p0)
    gap_dist = (np.linalg.norm(p0 - v0) + np.linalg.norm(p1 - v1)) / 2
    
    # Check if quad triangles would be thin
    # Test the two triangles that would be created: (v0, v1, p1) and (v0, p1, p0)
    aspect1 = _compute_triangle_aspect_ratio(v0, v1, p1)
    aspect2 = _compute_triangle_aspect_ratio(v0, p1, p0)
    
    # Also check alternative diagonal: (v0, v1, p0) and (v1, p1, p0)
    aspect3 = _compute_triangle_aspect_ratio(v0, v1, p0)
    aspect4 = _compute_triangle_aspect_ratio(v1, p1, p0)
    
    # Use diagonal that gives better minimum aspect ratio
    min_aspect_diag1 = min(aspect1, aspect2)
    min_aspect_diag2 = min(aspect3, aspect4)
    
    # If both diagonals give acceptable triangles, no subdivision needed
    if max(min_aspect_diag1, min_aspect_diag2) >= min_aspect_ratio:
        return [v0, v1], [p0, p1]
    
    # Determine number of subdivisions needed
    # Based on longest edge relative to target length
    max_edge = max(boundary_len, projection_len, gap_dist * 2)
    n_subdivisions = max(1, int(np.ceil(max_edge / target_edge_length)))
    n_subdivisions = min(n_subdivisions, 10)  # Cap at 10 subdivisions
    
    if n_subdivisions <= 1:
        return [v0, v1], [p0, p1]
    
    # Create subdivided points with batched projection query
    boundary_points = []
    interp_points = []
    
    for i in range(n_subdivisions + 1):
        t = i / n_subdivisions
        
        # Interpolate on boundary
        b_pt = v0 + t * (v1 - v0)
        boundary_points.append(b_pt)
        
        # Interpolate projection (will be re-projected in batch)
        p_interp = p0 + t * (p1 - p0)
        interp_points.append(p_interp)
    
    # Batch re-project all interpolated points to part surface
    interp_array = np.array(interp_points)
    proj_pts, _, _ = trimesh.proximity.closest_point(part_mesh, interp_array)
    projection_points = list(proj_pts)
    
    return boundary_points, projection_points


def close_membrane_gaps_robust(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    vertex_boundary_type: Optional[np.ndarray] = None,
    max_gap_distance: float = None,
    min_gap_distance: float = 0.001,
    target_edge_length: float = None,
    min_aspect_ratio: float = 0.15
) -> GapClosingResult:
    """
    Robustly close gaps between membrane boundary and part surface.
    
    This improved algorithm addresses issues with the simple flange approach:
    1. Quality-controlled triangulation - avoids thin triangles
    2. Adaptive edge subdivision - adds vertices when needed
    3. Better projection finding - avoids projecting to wrong part of mesh
    4. Consistent winding - ensures proper normals
    
    Algorithm:
    1. Find all boundary loops of the membrane
    2. Identify inner loops (should connect to part)
    3. For each inner loop:
       a. Project vertices to part with neighbor-aware projection
       b. Check triangle quality; subdivide edges if needed
       c. Create triangles with proper winding
    4. Handle any remaining small holes with fan triangulation
    
    Args:
        membrane_mesh: The membrane mesh with boundary edges
        part_mesh: The part/object surface to connect to
        vertex_boundary_type: Optional array of boundary types (-1=inner, 0=interior, 1/2=outer)
        max_gap_distance: Maximum gap to bridge (auto-computed if None)
        min_gap_distance: Gaps smaller than this are snapped
        target_edge_length: Target edge length for subdivision (auto-computed if None)
        min_aspect_ratio: Minimum acceptable triangle aspect ratio (0-1)
    
    Returns:
        GapClosingResult with the combined mesh
    """
    import time
    start = time.time()
    
    result = GapClosingResult()
    
    if membrane_mesh is None or len(membrane_mesh.faces) == 0:
        logger.warning("No membrane mesh provided for gap closing")
        return result
    
    if part_mesh is None or len(part_mesh.faces) == 0:
        logger.warning("No part mesh provided for gap closing")
        result.mesh = membrane_mesh
        return result
    
    vertices = list(membrane_mesh.vertices)
    faces = list(membrane_mesh.faces)
    n_orig_verts = len(vertices)
    
    # Auto-compute parameters
    mesh_scale = np.linalg.norm(membrane_mesh.bounds[1] - membrane_mesh.bounds[0])
    if max_gap_distance is None:
        max_gap_distance = mesh_scale * 0.15  # 15% of mesh scale
    if target_edge_length is None:
        # Use average edge length of membrane
        edges = membrane_mesh.edges_unique
        edge_lengths = np.linalg.norm(
            np.array(vertices)[edges[:, 1]] - np.array(vertices)[edges[:, 0]], 
            axis=1
        )
        target_edge_length = np.median(edge_lengths) * 1.5
    
    # Step 1: Find boundary edges
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i+1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(fi)
    
    boundary_edges = [(v0, v1) for (v0, v1), flist in edge_to_faces.items() if len(flist) == 1]
    
    if len(boundary_edges) == 0:
        logger.info("Membrane is watertight - no gap closing needed")
        result.mesh = membrane_mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # Step 2: Build boundary loops
    boundary_loops = _build_boundary_chains(np.array(boundary_edges, dtype=np.int64))
    result.boundary_loops_found = len(boundary_loops)
    
    # Step 3: Identify inner boundary loops
    inner_loops = []
    vertices_np = np.array(vertices)
    
    for loop_verts, is_closed in boundary_loops:
        if not is_closed or len(loop_verts) < 3:
            continue
        
        # Determine if this is an inner loop (should connect to part)
        if vertex_boundary_type is not None and len(vertex_boundary_type) >= n_orig_verts:
            inner_count = sum(1 for v in loop_verts if v < len(vertex_boundary_type) and vertex_boundary_type[v] == -1)
            is_inner = inner_count > len(loop_verts) * 0.5
        else:
            loop_positions = vertices_np[loop_verts]
            _, distances, _ = trimesh.proximity.closest_point(part_mesh, loop_positions)
            avg_dist = np.mean(distances)
            is_inner = avg_dist < max_gap_distance
        
        if is_inner:
            inner_loops.append(list(loop_verts))
    
    result.inner_boundary_loops = len(inner_loops)
    
    if len(inner_loops) == 0:
        logger.info("No inner boundary loops found - no gap closing needed")
        result.mesh = membrane_mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # Log loop sizes for debugging
    loop_sizes = [len(loop) for loop in inner_loops]
    total_loop_verts = sum(loop_sizes)
    logger.info(f"Closing gaps for {len(inner_loops)} inner boundary loops "
               f"(total {total_loop_verts} vertices, sizes: min={min(loop_sizes)}, max={max(loop_sizes)}, avg={np.mean(loop_sizes):.1f})")
    
    # === Pre-compute face normals and edge-to-face map for normal consistency ===
    membrane_normals = membrane_mesh.face_normals
    
    # Build edge-to-face mapping (edge_key -> face_index)
    boundary_edge_to_face = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0_f, v1_f = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0_f, v1_f), max(v0_f, v1_f))
            if edge_key not in boundary_edge_to_face:
                boundary_edge_to_face[edge_key] = fi  # First face with this edge
    
    def get_adjacent_normal(v0_idx, v1_idx):
        """Get normal of adjacent face for boundary edge (v0, v1)"""
        edge_key = (min(v0_idx, v1_idx), max(v0_idx, v1_idx))
        if edge_key in boundary_edge_to_face:
            return membrane_normals[boundary_edge_to_face[edge_key]]
        return None
    
    def compute_normal(p0, p1, p2):
        """Compute triangle normal"""
        edge1 = np.array(p1) - np.array(p0)
        edge2 = np.array(p2) - np.array(p0)
        normal = np.cross(edge1, edge2)
        length = np.linalg.norm(normal)
        return normal / length if length > 1e-10 else np.array([0, 0, 1])
    
    def create_consistent_face(v0, v1, v2, adj_normal, verts_list, new_verts_list, n_orig):
        """Create triangle with normal consistent with adjacent face"""
        # Get positions
        p0 = np.array(verts_list[v0]) if v0 < n_orig else np.array(new_verts_list[v0 - n_orig])
        p1 = np.array(verts_list[v1]) if v1 < n_orig else np.array(new_verts_list[v1 - n_orig])
        p2 = np.array(verts_list[v2]) if v2 < n_orig else np.array(new_verts_list[v2 - n_orig])
        
        normal = compute_normal(p0, p1, p2)
        
        if adj_normal is not None:
            if np.dot(normal, adj_normal) < 0:
                return [v0, v2, v1]  # Flip winding
        return [v0, v1, v2]
    
    # Step 4: Process each inner loop
    new_vertices = []
    new_faces = []
    gap_distances = []
    triangle_aspects = []
    edges_subdivided = 0
    thin_avoided = 0
    
    # Calculate total vertices for progress logging
    total_loop_verts = sum(len(loop) for loop in inner_loops)
    processed_verts = 0
    last_progress_log = 0
    
    for loop_idx, loop_verts in enumerate(inner_loops):
        n_loop = len(loop_verts)
        
        # Progress logging every 20% or for large loops
        progress_pct = int(100 * processed_verts / max(1, total_loop_verts))
        if progress_pct >= last_progress_log + 20 or n_loop > 100:
            logger.info(f"Gap closing progress: loop {loop_idx + 1}/{len(inner_loops)}, "
                       f"{progress_pct}% vertices processed ({n_loop} verts in current loop)")
            last_progress_log = progress_pct
        
        # First pass: compute initial projections for all vertices (batched)
        loop_positions = np.array([vertices[v] for v in loop_verts])
        initial_proj, initial_dists, _ = trimesh.proximity.closest_point(part_mesh, loop_positions)
        
        gap_distances.extend(initial_dists)
        
        # Second pass: refine projections considering neighbors (optimized)
        # Only refine if initial projection seems far from neighbors
        projections = list(initial_proj)  # Start with initial projections
        
        # For small loops or simple cases, skip refinement entirely
        if n_loop <= 3 or np.max(initial_dists) < max_gap_distance * 0.5:
            pass  # Use initial projections as-is
        else:
            # Only refine vertices whose projections seem inconsistent
            neighbor_dists = np.zeros(n_loop)
            for i in range(n_loop):
                prev_i = (i - 1) % n_loop
                next_i = (i + 1) % n_loop
                neighbor_center = (initial_proj[prev_i] + initial_proj[next_i]) / 2
                neighbor_dists[i] = np.linalg.norm(initial_proj[i] - neighbor_center)
            
            # Only refine vertices that are outliers (>2x median distance from neighbors)
            median_neighbor_dist = np.median(neighbor_dists)
            refinement_threshold = max(median_neighbor_dist * 2, max_gap_distance * 0.3)
            
            for i in range(n_loop):
                if neighbor_dists[i] > refinement_threshold:
                    prev_i = (i - 1) % n_loop
                    next_i = (i + 1) % n_loop
                    neighbor_projs = [initial_proj[prev_i], initial_proj[next_i]]
                    
                    refined_proj, _ = _find_best_projection_on_part(
                        loop_positions[i],
                        part_mesh,
                        neighbor_projs,
                        max_gap_distance
                    )
                    projections[i] = refined_proj
        
        processed_verts += n_loop
        
        # Third pass: create triangles with quality control
        for i in range(n_loop):
            next_i = (i + 1) % n_loop            
            # Get adjacent face normal for normal consistency
            adj_normal = get_adjacent_normal(loop_verts[i], loop_verts[next_i])
            
            v0_pos = loop_positions[i]
            v1_pos = loop_positions[next_i]
            p0_pos = projections[i]
            p1_pos = projections[next_i]
            
            dist0 = np.linalg.norm(p0_pos - v0_pos)
            dist1 = np.linalg.norm(p1_pos - v1_pos)
            
            # Handle snap cases
            if dist0 < min_gap_distance and dist1 < min_gap_distance:
                # Both vertices snap - update original vertices
                vertices[loop_verts[i]] = p0_pos.tolist()
                vertices[loop_verts[next_i]] = p1_pos.tolist()
                continue
            
            # Check if we need subdivision
            boundary_pts, proj_pts = _subdivide_gap_edge(
                v0_pos, v1_pos, p0_pos, p1_pos,
                part_mesh, target_edge_length, min_aspect_ratio
            )
            
            if len(boundary_pts) > 2:
                edges_subdivided += 1
            
            # Create vertices for subdivision points (skip endpoints which are original verts)
            sub_boundary_indices = []
            sub_proj_indices = []
            
            for j, (b_pt, p_pt) in enumerate(zip(boundary_pts, proj_pts)):
                if j == 0:
                    # First point is loop_verts[i]
                    b_idx = loop_verts[i]
                    # Check if it should snap
                    if dist0 < min_gap_distance:
                        vertices[b_idx] = p_pt.tolist()
                        p_idx = b_idx
                    else:
                        p_idx = n_orig_verts + len(new_vertices)
                        new_vertices.append(p_pt.tolist())
                elif j == len(boundary_pts) - 1:
                    # Last point is loop_verts[next_i]
                    b_idx = loop_verts[next_i]
                    if dist1 < min_gap_distance:
                        vertices[b_idx] = p_pt.tolist()
                        p_idx = b_idx
                    else:
                        p_idx = n_orig_verts + len(new_vertices)
                        new_vertices.append(p_pt.tolist())
                else:
                    # Intermediate subdivision point
                    b_idx = n_orig_verts + len(new_vertices)
                    new_vertices.append(b_pt.tolist())
                    p_idx = n_orig_verts + len(new_vertices)
                    new_vertices.append(p_pt.tolist())
                
                sub_boundary_indices.append(b_idx)
                sub_proj_indices.append(p_idx)
            
            # Create triangles for each sub-segment
            for j in range(len(sub_boundary_indices) - 1):
                bv0 = sub_boundary_indices[j]
                bv1 = sub_boundary_indices[j + 1]
                pv0 = sub_proj_indices[j]
                pv1 = sub_proj_indices[j + 1]
                
                # Get vertex positions for triangle creation
                bv0_pos = np.array(vertices[bv0]) if bv0 < len(vertices) else np.array(new_vertices[bv0 - n_orig_verts])
                bv1_pos = np.array(vertices[bv1]) if bv1 < len(vertices) else np.array(new_vertices[bv1 - n_orig_verts])
                pv0_pos = np.array(vertices[pv0]) if pv0 < len(vertices) else np.array(new_vertices[pv0 - n_orig_verts])
                pv1_pos = np.array(vertices[pv1]) if pv1 < len(vertices) else np.array(new_vertices[pv1 - n_orig_verts])
                
                # Skip degenerate cases
                if bv0 == pv0 and bv1 == pv1:
                    continue
                if bv0 == pv0:
                    # Single triangle
                    aspect = _compute_triangle_aspect_ratio(bv0_pos, bv1_pos, pv1_pos)
                    if aspect >= min_aspect_ratio * 0.5:  # Allow slightly lower for single triangles
                        face = create_consistent_face(bv0, bv1, pv1, adj_normal, vertices, new_vertices, n_orig_verts)
                        new_faces.append(face)
                        triangle_aspects.append(aspect)
                    else:
                        thin_avoided += 1
                    continue
                if bv1 == pv1:
                    aspect = _compute_triangle_aspect_ratio(bv0_pos, bv1_pos, pv0_pos)
                    if aspect >= min_aspect_ratio * 0.5:
                        face = create_consistent_face(bv0, bv1, pv0, adj_normal, vertices, new_vertices, n_orig_verts)
                        new_faces.append(face)
                        triangle_aspects.append(aspect)
                    else:
                        thin_avoided += 1
                    continue
                if pv0 == pv1:
                    aspect = _compute_triangle_aspect_ratio(bv0_pos, bv1_pos, pv0_pos)
                    if aspect >= min_aspect_ratio * 0.5:
                        face = create_consistent_face(bv0, bv1, pv0, adj_normal, vertices, new_vertices, n_orig_verts)
                        new_faces.append(face)
                        triangle_aspects.append(aspect)
                    else:
                        thin_avoided += 1
                    continue
                
                # Full quad - choose better diagonal
                aspect1 = _compute_triangle_aspect_ratio(bv0_pos, bv1_pos, pv1_pos)
                aspect2 = _compute_triangle_aspect_ratio(bv0_pos, pv1_pos, pv0_pos)
                aspect3 = _compute_triangle_aspect_ratio(bv0_pos, bv1_pos, pv0_pos)
                aspect4 = _compute_triangle_aspect_ratio(bv1_pos, pv1_pos, pv0_pos)
                
                min_diag1 = min(aspect1, aspect2)
                min_diag2 = min(aspect3, aspect4)
                
                if min_diag1 >= min_diag2:
                    # Use diagonal 1: bv0-pv1
                    if aspect1 >= min_aspect_ratio * 0.3:
                        face = create_consistent_face(bv0, bv1, pv1, adj_normal, vertices, new_vertices, n_orig_verts)
                        new_faces.append(face)
                        triangle_aspects.append(aspect1)
                    else:
                        thin_avoided += 1
                    if aspect2 >= min_aspect_ratio * 0.3:
                        face = create_consistent_face(bv0, pv1, pv0, adj_normal, vertices, new_vertices, n_orig_verts)
                        new_faces.append(face)
                        triangle_aspects.append(aspect2)
                    else:
                        thin_avoided += 1
                else:
                    # Use diagonal 2: bv1-pv0
                    if aspect3 >= min_aspect_ratio * 0.3:
                        face = create_consistent_face(bv0, bv1, pv0, adj_normal, vertices, new_vertices, n_orig_verts)
                        new_faces.append(face)
                        triangle_aspects.append(aspect3)
                    else:
                        thin_avoided += 1
                    if aspect4 >= min_aspect_ratio * 0.3:
                        face = create_consistent_face(bv1, pv1, pv0, adj_normal, vertices, new_vertices, n_orig_verts)
                        new_faces.append(face)
                        triangle_aspects.append(aspect4)
                    else:
                        thin_avoided += 1
    
    result.gap_vertices_added = len(new_vertices)
    result.gap_faces_added = len(new_faces)
    result.edges_subdivided = edges_subdivided
    result.thin_triangles_avoided = thin_avoided
    
    if len(gap_distances) > 0:
        result.avg_gap_distance = float(np.mean(gap_distances))
        result.max_gap_distance = float(np.max(gap_distances))
    
    if len(triangle_aspects) > 0:
        result.min_triangle_aspect_ratio = float(np.min(triangle_aspects))
        result.avg_triangle_aspect_ratio = float(np.mean(triangle_aspects))
    
    if len(new_faces) == 0 and len(new_vertices) == 0:
        logger.info("No gap geometry needed - boundaries already on part surface")
        result.mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # Combine mesh
    if len(new_vertices) > 0:
        combined_vertices = np.vstack([np.array(vertices), np.array(new_vertices)])
    else:
        combined_vertices = np.array(vertices)
    
    if len(new_faces) > 0:
        combined_faces = np.vstack([np.array(faces), np.array(new_faces, dtype=np.int64)])
    else:
        combined_faces = np.array(faces)
    
    combined_mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False
    )
    
    combined_mesh.fix_normals()
    
    result.mesh = combined_mesh
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Gap closing: added {result.gap_faces_added} faces, "
                f"{result.gap_vertices_added} vertices, "
                f"subdivided {result.edges_subdivided} edges, "
                f"avoided {result.thin_triangles_avoided} thin triangles, "
                f"avg gap: {result.avg_gap_distance:.4f}mm, "
                f"aspect ratio: min={result.min_triangle_aspect_ratio:.3f}, avg={result.avg_triangle_aspect_ratio:.3f} "
                f"in {result.processing_time_ms:.1f}ms")
    
    return result


# =============================================================================
# REMAINING HOLE FILLING WITH FAN TRIANGULATION
# =============================================================================

def fill_remaining_holes_with_fans(
    mesh: trimesh.Trimesh,
    max_hole_perimeter: float = None
) -> Tuple[trimesh.Trimesh, int]:
    """
    Fill any remaining small holes using fan triangulation from centroid.
    
    This is a final cleanup step after gap closing to handle any small holes
    that weren't fully closed by the projection-based approach.
    
    Args:
        mesh: The mesh with potential holes
        max_hole_perimeter: Maximum hole perimeter to fill (auto-computed if None)
    
    Returns:
        Tuple of (filled_mesh, holes_filled_count)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    
    vertices = list(mesh.vertices)
    faces = list(mesh.faces)
    n_orig_verts = len(vertices)
    
    if max_hole_perimeter is None:
        mesh_scale = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        max_hole_perimeter = mesh_scale * 0.3  # 30% of mesh scale
    
    # Find boundary edges
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i+1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(fi)
    
    boundary_edges = [(v0, v1) for (v0, v1), flist in edge_to_faces.items() if len(flist) == 1]
    
    if len(boundary_edges) == 0:
        return mesh, 0
    
    # Build boundary loops
    boundary_loops = _build_boundary_chains(np.array(boundary_edges, dtype=np.int64))
    
    new_vertices = []
    new_faces = []
    holes_filled = 0
    
    for loop_verts, is_closed in boundary_loops:
        if not is_closed or len(loop_verts) < 3:
            continue
        
        # Compute perimeter
        loop_positions = np.array([vertices[v] for v in loop_verts])
        perimeter = 0.0
        for i in range(len(loop_verts)):
            next_i = (i + 1) % len(loop_verts)
            perimeter += np.linalg.norm(loop_positions[next_i] - loop_positions[i])
        
        if perimeter > max_hole_perimeter:
            continue
        
        # Create centroid vertex
        centroid = np.mean(loop_positions, axis=0)
        centroid_idx = n_orig_verts + len(new_vertices)
        new_vertices.append(centroid.tolist())
        
        # Create fan triangles
        for i in range(len(loop_verts)):
            next_i = (i + 1) % len(loop_verts)
            new_faces.append([loop_verts[i], loop_verts[next_i], centroid_idx])
        
        holes_filled += 1
    
    if holes_filled == 0:
        return mesh, 0
    
    # Combine
    combined_vertices = np.vstack([np.array(vertices), np.array(new_vertices)])
    combined_faces = np.vstack([np.array(faces), np.array(new_faces, dtype=np.int64)])
    
    filled_mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False
    )
    filled_mesh.fix_normals()
    
    logger.info(f"Fan triangulation: filled {holes_filled} remaining holes")
    
    return filled_mesh, holes_filled


# =============================================================================
# LEGACY FLANGE CREATION (kept for compatibility)
# =============================================================================

@dataclass
class FlangeCreationResult:
    """Result of creating a flange to connect membrane boundary to part surface."""
    mesh: Optional[trimesh.Trimesh] = None
    
    # Statistics
    boundary_loops_found: int = 0
    inner_boundary_loops: int = 0
    flange_faces_added: int = 0
    flange_vertices_added: int = 0
    
    # Average gap distance before flange
    avg_gap_distance: float = 0.0
    max_gap_distance: float = 0.0
    
    # Timing
    processing_time_ms: float = 0.0


def create_inner_boundary_flange(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    vertex_boundary_type: Optional[np.ndarray] = None,
    max_flange_width: float = None,
    min_flange_width: float = 0.01
) -> FlangeCreationResult:
    """
    Create a flange of triangles that connects the membrane's inner boundary to the part surface.
    
    NOTE: This is now a wrapper around close_membrane_gaps_robust() for better results.
    
    Args:
        membrane_mesh: The membrane mesh with boundary edges
        part_mesh: The part/object surface to connect to
        vertex_boundary_type: Optional array of boundary types (-1=inner, 0=interior, 1/2=outer)
        max_flange_width: Maximum allowed flange width (auto-computed if None)
        min_flange_width: Minimum flange width to create triangles for
    
    Returns:
        FlangeCreationResult with the combined mesh
    """
    # Use the new robust implementation
    gap_result = close_membrane_gaps_robust(
        membrane_mesh=membrane_mesh,
        part_mesh=part_mesh,
        vertex_boundary_type=vertex_boundary_type,
        max_gap_distance=max_flange_width,
        min_gap_distance=min_flange_width
    )
    
    # Also fill any remaining small holes
    if gap_result.mesh is not None:
        filled_mesh, holes_filled = fill_remaining_holes_with_fans(gap_result.mesh)
        if filled_mesh is not None:
            gap_result.mesh = filled_mesh
    
    # Convert to legacy result format
    result = FlangeCreationResult()
    result.mesh = gap_result.mesh
    result.boundary_loops_found = gap_result.boundary_loops_found
    result.inner_boundary_loops = gap_result.inner_boundary_loops
    result.flange_faces_added = gap_result.gap_faces_added
    result.flange_vertices_added = gap_result.gap_vertices_added
    result.avg_gap_distance = gap_result.avg_gap_distance
    result.max_gap_distance = gap_result.max_gap_distance
    result.processing_time_ms = gap_result.processing_time_ms
    
    return result


# =============================================================================
# SMALL POLYLINE REMOVAL AND HOLE FILLING
# =============================================================================

@dataclass
class SmallHoleRemovalResult:
    """Result of small polyline/hole removal operation."""
    mesh: Optional[trimesh.Trimesh] = None
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    
    # Statistics
    loops_found: int = 0
    loops_removed: int = 0
    holes_filled: int = 0
    fill_faces_added: int = 0
    
    # Perimeter info for removed loops
    removed_loop_perimeters: List[float] = None
    
    # Indices of new centroid vertices (for boundary_type extension)
    new_centroid_indices: Optional[List[int]] = None
    
    # Timing
    processing_time_ms: float = 0.0


def remove_small_boundary_loops(
    surface: PartingSurfaceResult,
    min_perimeter: float = DEFAULT_MIN_LOOP_PERIMETER,
    fill_holes: bool = True,
    smooth_fill: bool = True,
    smooth_iterations: int = 2
) -> SmallHoleRemovalResult:
    """
    Remove small closed boundary loops (holes) from the parting surface.
    
    This function identifies closed boundary loops with perimeter less than
    min_perimeter, removes them, fills the resulting holes, and optionally
    smooths the fill region.
    
    Per the paper Section 4.4: After extracting the membrane mesh, small holes
    or artifacts should be cleaned up to create a smooth parting surface.
    
    Algorithm:
    1. Find all boundary edges (edges with only one adjacent face)
    2. Build closed loops from boundary edges
    3. Calculate perimeter of each loop
    4. For loops with perimeter < min_perimeter:
       a. Compute centroid of loop vertices
       b. Create fan triangulation from centroid to loop edges
       c. Apply local Laplacian smoothing to fill region
    
    Args:
        surface: PartingSurfaceResult with the parting surface mesh
        min_perimeter: Minimum perimeter (in mesh units) for loops to keep.
                      Loops smaller than this will be filled. Default 4.0mm
        fill_holes: If True, fill the holes after identifying small loops
        smooth_fill: If True, apply local smoothing to filled regions
        smooth_iterations: Number of smoothing iterations for fill regions
    
    Returns:
        SmallHoleRemovalResult with cleaned mesh and statistics
    """
    import time
    start = time.time()
    
    result = SmallHoleRemovalResult(removed_loop_perimeters=[])
    
    if surface.mesh is None:
        logger.warning("No mesh provided for small loop removal")
        return result
    
    mesh = surface.mesh
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    
    # === Step 1: Find all boundary edges ===
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(fi)
    
    boundary_edges = [(v0, v1) for (v0, v1), flist in edge_to_faces.items() if len(flist) == 1]
    
    if not boundary_edges:
        logger.info("No boundary edges found - mesh is watertight")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # === Step 2: Build closed loops from boundary edges ===
    boundary_chains = _build_boundary_chains(np.array(boundary_edges, dtype=np.int64))
    
    # Filter to only closed loops
    closed_loops = [(chain, is_closed) for chain, is_closed in boundary_chains if is_closed]
    result.loops_found = len(closed_loops)
    
    logger.info(f"Found {len(closed_loops)} closed boundary loops")
    
    if len(closed_loops) == 0:
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # === Step 3: Calculate perimeter and identify small loops ===
    small_loops = []
    for chain_verts, is_closed in closed_loops:
        perimeter = 0.0
        n_verts = len(chain_verts)
        for i in range(n_verts):
            v0 = chain_verts[i]
            v1 = chain_verts[(i + 1) % n_verts]
            edge_len = np.linalg.norm(vertices[v1] - vertices[v0])
            perimeter += edge_len
        
        if perimeter < min_perimeter:
            small_loops.append((chain_verts, perimeter))
            logger.info(f"Small loop found: {len(chain_verts)} vertices, perimeter={perimeter:.3f}mm")
    
    if not small_loops:
        logger.info(f"No loops with perimeter < {min_perimeter}mm found")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    logger.info(f"Found {len(small_loops)} small loops to fill (perimeter < {min_perimeter}mm)")
    
    if not fill_holes:
        # Just report statistics without filling
        result.loops_removed = len(small_loops)
        result.removed_loop_perimeters = [p for _, p in small_loops]
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # === Step 4: Fill small holes ===
    new_faces_list = list(faces)  # Start with existing faces
    new_vertices_list = list(vertices)  # Start with existing vertices
    fill_vertex_indices = []  # Track centroid vertices for smoothing
    
    for chain_verts, perimeter in small_loops:
        result.removed_loop_perimeters.append(perimeter)
        
        # Compute centroid of loop
        loop_positions = vertices[chain_verts]
        centroid = np.mean(loop_positions, axis=0)
        
        # Add centroid as new vertex
        centroid_idx = len(new_vertices_list)
        new_vertices_list.append(centroid)
        fill_vertex_indices.append(centroid_idx)
        
        # Create fan triangulation from centroid to loop edges
        n_loop = len(chain_verts)
        for i in range(n_loop):
            v0 = chain_verts[i]
            v1 = chain_verts[(i + 1) % n_loop]
            # Triangle: v0, v1, centroid (consistent winding)
            new_faces_list.append([v0, v1, centroid_idx])
            result.fill_faces_added += 1
        
        result.holes_filled += 1
    
    result.loops_removed = len(small_loops)
    
    # Store centroid indices for boundary_type extension
    result.new_centroid_indices = fill_vertex_indices
    
    # Convert to arrays
    filled_vertices = np.array(new_vertices_list, dtype=np.float64)
    filled_faces = np.array(new_faces_list, dtype=np.int64)
    
    # === Step 5: Optional local smoothing of fill regions ===
    if smooth_fill and fill_vertex_indices:
        filled_vertices = _smooth_fill_vertices(
            filled_vertices,
            filled_faces, 
            fill_vertex_indices, 
            iterations=smooth_iterations
        )
    
    # === Step 6: Create result mesh ===
    try:
        result_mesh = trimesh.Trimesh(
            vertices=filled_vertices,
            faces=filled_faces,
            process=False
        )
        result_mesh.fix_normals()
        
        result.mesh = result_mesh
        result.vertices = np.array(result_mesh.vertices)
        result.faces = np.array(result_mesh.faces)
        
    except Exception as e:
        logger.error(f"Failed to create filled mesh: {e}")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
    
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Small loop removal: filled {result.holes_filled} holes, "
                f"added {result.fill_faces_added} faces "
                f"in {result.processing_time_ms:.1f}ms")
    
    return result


def _smooth_fill_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    fill_vertex_indices: List[int],
    iterations: int = 2,
    lambda_factor: float = 0.5
) -> np.ndarray:
    """
    Apply local Laplacian smoothing to fill vertices (centroids).
    
    Only the centroid vertices are moved; boundary loop vertices stay fixed.
    
    Args:
        vertices: All mesh vertices
        faces: All mesh faces
        fill_vertex_indices: Indices of fill centroid vertices to smooth
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing factor (0-1)
    
    Returns:
        Updated vertex array
    """
    vertices = vertices.copy()
    fill_set = set(fill_vertex_indices)
    
    # Build vertex adjacency from faces
    n_verts = len(vertices)
    neighbors = [set() for _ in range(n_verts)]
    
    for f in faces:
        for i in range(3):
            v = int(f[i])
            neighbors[v].add(int(f[(i + 1) % 3]))
            neighbors[v].add(int(f[(i + 2) % 3]))
    
    # Smooth only fill vertices
    for _ in range(iterations):
        new_positions = vertices.copy()
        
        for v in fill_vertex_indices:
            if neighbors[v]:
                neighbor_list = list(neighbors[v])
                centroid = np.mean(vertices[neighbor_list], axis=0)
                new_positions[v] = vertices[v] + lambda_factor * (centroid - vertices[v])
        
        vertices = new_positions
    
    return vertices


def _build_boundary_chains(edges: np.ndarray) -> List[Tuple[List[int], bool]]:
    """
    Build ordered boundary chains from a set of boundary edges.
    Handles both closed loops and open chains.
    
    Args:
        edges: (N, 2) array of edge vertex indices
    
    Returns:
        List of (vertex_list, is_closed) tuples
    """
    if len(edges) == 0:
        return []
    
    # Build adjacency for boundary vertices
    adj = {}
    for v0, v1 in edges:
        if v0 not in adj:
            adj[v0] = []
        if v1 not in adj:
            adj[v1] = []
        adj[v0].append(v1)
        adj[v1].append(v0)
    
    chains = []
    visited_edges = set()
    
    # Find endpoint vertices (degree 1) to start open chains
    endpoints = [v for v, neighbors in adj.items() if len(neighbors) == 1]
    
    # Process endpoints first (open chains)
    for start_v in endpoints:
        if not adj[start_v]:  # Already processed
            continue
        
        # Check if the edge from this endpoint is already visited
        neighbor = adj[start_v][0]
        edge_key = (min(start_v, neighbor), max(start_v, neighbor))
        if edge_key in visited_edges:
            continue
        
        # Walk from endpoint
        chain = [start_v]
        visited_edges.add(edge_key)
        
        current = neighbor
        prev = start_v
        chain.append(current)
        
        while True:
            neighbors = [n for n in adj.get(current, []) if n != prev]
            if not neighbors:
                break
            
            next_v = neighbors[0]
            edge_key = (min(current, next_v), max(current, next_v))
            if edge_key in visited_edges:
                break
            
            visited_edges.add(edge_key)
            chain.append(next_v)
            prev = current
            current = next_v
        
        if len(chain) >= 2:
            chains.append((chain, False))  # Open chain
    
    # Process remaining edges (closed loops)
    for edge in edges:
        edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        if edge_key in visited_edges:
            continue
        
        # Start a new loop from this edge
        loop = [edge[0], edge[1]]
        visited_edges.add(edge_key)
        
        current = edge[1]
        prev = edge[0]
        
        while True:
            neighbors = [n for n in adj.get(current, []) if n != prev]
            if not neighbors:
                break
            
            next_v = neighbors[0]
            edge_key = (min(current, next_v), max(current, next_v))
            if edge_key in visited_edges:
                break
            
            if next_v == loop[0]:
                # Loop closed
                visited_edges.add(edge_key)
                break
            
            visited_edges.add(edge_key)
            loop.append(next_v)
            prev = current
            current = next_v
        
        if len(loop) >= 3:
            chains.append((loop, True))  # Closed loop
    
    return chains


def _create_connected_fill(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    boundary_chains: List[Tuple[List[int], bool]],
    part_mesh: trimesh.Trimesh,
    boundary_edge_info: Dict,
    max_distance: float,
    primary_surface_mesh: trimesh.Trimesh = None,
    hull_mesh: trimesh.Trimesh = None,
    vertex_boundary_type: np.ndarray = None
) -> Dict:
    """
    Create fill geometry that connects parting surface boundary to target surfaces.
    
    Per the paper Section 4.3: "The triangulated surface C is bounded by 
    construction by the object surface mesh M and the external boundary ∂H."
    
    This function creates a RIBBON of quads (2 triangles each) that form a proper
    connection from the parting surface boundary to the target surface (part M, 
    hull ∂H, or primary surface).
    
    IMPORTANT (per paper Section 4.4): Boundary vertices must be projected to
    their ORIGINAL surface:
    - Inner boundary (vertex_boundary_type == -1): project to part mesh M
    - Outer boundary (vertex_boundary_type == 1 or 2): project to hull mesh ∂H
    - If vertex_boundary_type is None, fall back to closest surface
    
    APPROACH: For each boundary vertex in a chain:
    1. Determine which surface to project to based on boundary type
    2. Project the vertex to the appropriate target surface
    3. Create quads between adjacent (boundary_vertex, projected_vertex) pairs
    
    Args:
        surface_vertices: Original parting surface vertices
        surface_faces: Original parting surface faces
        boundary_chains: List of (vertex_list, is_closed) tuples
        part_mesh: Part mesh for projection (inner boundary M)
        boundary_edge_info: Dict mapping edge_key to (v0, v1, v_opposite) for winding
        max_distance: Maximum allowed projection distance
        primary_surface_mesh: Optional primary parting surface (for secondary surfaces)
        hull_mesh: Optional hull mesh for projection (outer boundary ∂H)
        vertex_boundary_type: Array of boundary type per vertex 
                             (-1=part M inner, 0=interior, 1/2=hull ∂H outer)
    
    Returns:
        Dict with keys:
        - 'new_vertices': Projected vertices (to be appended to surface_vertices)
        - 'new_faces': Face indices referencing combined vertex array
        - 'part_constrained_indices': Indices of projected vertices (should re-project to target)
        - 'fill_boundary_indices': Indices of original boundary vertices used (inner rim)
        - 'inner_rim_edges': Nx2 array of inner rim edge vertex pairs
    """
    n_orig_verts = len(surface_vertices)
    new_vertices = []  # Projected boundary vertices
    new_faces = []
    part_constrained_indices = []  # Track which new vertices should stay on target
    fill_boundary_indices = set()  # Track original boundary vertices used in fill
    inner_rim_edges = []  # Track edges along the inner rim
    
    # Check if we have proper boundary type info
    use_boundary_type = vertex_boundary_type is not None and len(vertex_boundary_type) >= n_orig_verts
    if use_boundary_type:
        logger.info("Using vertex_boundary_type for projection target selection")
    else:
        logger.info("No vertex_boundary_type - using closest surface for projection")
    
    for chain_idx, (chain_verts, is_closed) in enumerate(boundary_chains):
        if len(chain_verts) < 2:
            continue
        
        n_pts = len(chain_verts)
        
        # Get boundary positions
        boundary_positions = np.array([surface_vertices[v] for v in chain_verts])
        
        # Initialize projection arrays
        proj_pts = np.zeros_like(boundary_positions)
        proj_dists = np.full(n_pts, np.inf)
        
        if use_boundary_type:
            # === NEW CORRECT PROJECTION: Use vertex_boundary_type ===
            # Inner boundary vertices (-1) → project to part mesh M
            # Outer boundary vertices (1, 2) → project to hull mesh ∂H
            
            for i, bv in enumerate(chain_verts):
                bt = vertex_boundary_type[bv] if bv < len(vertex_boundary_type) else 0
                
                if bt == -1:
                    # Inner boundary - project to part mesh M
                    proj_pt, dist, _ = trimesh.proximity.closest_point(part_mesh, boundary_positions[i:i+1])
                    proj_pts[i] = proj_pt[0]
                    proj_dists[i] = dist[0]
                elif bt in (1, 2) and hull_mesh is not None:
                    # Outer boundary - project to hull mesh ∂H
                    proj_pt, dist, _ = trimesh.proximity.closest_point(hull_mesh, boundary_positions[i:i+1])
                    proj_pts[i] = proj_pt[0]
                    proj_dists[i] = dist[0]
                else:
                    # Interior vertex or no appropriate mesh - use closest of all available
                    proj_pt_part, dist_part, _ = trimesh.proximity.closest_point(part_mesh, boundary_positions[i:i+1])
                    proj_pts[i] = proj_pt_part[0]
                    proj_dists[i] = dist_part[0]
                    
                    if hull_mesh is not None:
                        proj_pt_hull, dist_hull, _ = trimesh.proximity.closest_point(hull_mesh, boundary_positions[i:i+1])
                        if dist_hull[0] < proj_dists[i]:
                            proj_pts[i] = proj_pt_hull[0]
                            proj_dists[i] = dist_hull[0]
                    
                    if primary_surface_mesh is not None:
                        proj_pt_prim, dist_prim, _ = trimesh.proximity.closest_point(primary_surface_mesh, boundary_positions[i:i+1])
                        if dist_prim[0] < proj_dists[i]:
                            proj_pts[i] = proj_pt_prim[0]
                            proj_dists[i] = dist_prim[0]
        else:
            # === FALLBACK: Project to closest surface ===
            # This is the old behavior when no boundary type info is available
            
            # Project to part mesh
            proj_pts_part, dist_part, _ = trimesh.proximity.closest_point(part_mesh, boundary_positions)
            proj_pts = proj_pts_part.copy()
            proj_dists = dist_part.copy()
            
            # Check hull mesh if available
            if hull_mesh is not None:
                proj_pts_hull, dist_hull, _ = trimesh.proximity.closest_point(hull_mesh, boundary_positions)
                hull_closer = dist_hull < proj_dists
                proj_pts[hull_closer] = proj_pts_hull[hull_closer]
                proj_dists[hull_closer] = dist_hull[hull_closer]
            
            # Check primary mesh if available  
            if primary_surface_mesh is not None:
                proj_pts_primary, dist_primary, _ = trimesh.proximity.closest_point(primary_surface_mesh, boundary_positions)
                primary_closer = dist_primary < proj_dists
                proj_pts[primary_closer] = proj_pts_primary[primary_closer]
                proj_dists[primary_closer] = dist_primary[primary_closer]
        
        # Create mapping from boundary vertex to its projected vertex index
        # Only include vertices within max_distance
        boundary_to_proj = {}  # boundary_vertex_idx -> new_vertex_idx
        
        for i, bv in enumerate(chain_verts):
            if proj_dists[i] <= max_distance:
                # Check that projected point is actually different from boundary vertex
                # (if they're the same, no fill needed)
                dist_moved = np.linalg.norm(proj_pts[i] - boundary_positions[i])
                if dist_moved > 1e-6:
                    new_vert_idx = n_orig_verts + len(new_vertices)
                    new_vertices.append(proj_pts[i])
                    part_constrained_indices.append(new_vert_idx)
                    boundary_to_proj[bv] = new_vert_idx
                    fill_boundary_indices.add(bv)
        
        # Now create fill triangles (quads split into 2 triangles)
        # For each edge in the boundary chain, create a quad if both vertices have projections
        n_edges = n_pts if is_closed else n_pts - 1
        
        for i in range(n_edges):
            next_i = (i + 1) % n_pts
            
            bv0 = chain_verts[i]
            bv1 = chain_verts[next_i]
            
            # Check if both boundary vertices have projections
            if bv0 not in boundary_to_proj or bv1 not in boundary_to_proj:
                # One or both vertices didn't get projected - skip or create partial fill
                # Try single triangle fill as fallback
                if bv0 in boundary_to_proj:
                    pv0 = boundary_to_proj[bv0]
                    # Triangle: bv0, bv1, pv0
                    new_faces.append([bv0, bv1, pv0])
                    inner_rim_edges.append([bv0, bv1])
                elif bv1 in boundary_to_proj:
                    pv1 = boundary_to_proj[bv1]
                    # Triangle: bv0, bv1, pv1
                    new_faces.append([bv0, bv1, pv1])
                    inner_rim_edges.append([bv0, bv1])
                continue
            
            pv0 = boundary_to_proj[bv0]
            pv1 = boundary_to_proj[bv1]
            
            # Determine winding direction from adjacent face
            edge_key = (min(bv0, bv1), max(bv0, bv1))
            
            # Check if pv0 and pv1 are the same (degenerate quad)
            if pv0 == pv1:
                # Single triangle
                new_faces.append([bv0, bv1, pv0])
            else:
                # Full quad as two triangles
                # The quad has vertices: bv0, bv1, pv1, pv0 (in order)
                # We need consistent winding
                
                if edge_key in boundary_edge_info:
                    adj_v0, adj_v1, adj_opp = boundary_edge_info[edge_key]
                    
                    # Determine which side the adjacent face's interior is on
                    edge_mid = 0.5 * (surface_vertices[bv0] + surface_vertices[bv1])
                    to_inside = surface_vertices[adj_opp] - edge_mid
                    to_proj = new_vertices[pv0 - n_orig_verts] - edge_mid
                    
                    same_side = np.dot(to_inside, to_proj) > 0
                    
                    if same_side:
                        # Projected points are on same side as interior - flip winding
                        # Quad: bv0 -> bv1 -> pv1 -> pv0 (but we need CCW from outside)
                        new_faces.append([bv0, bv1, pv1])
                        new_faces.append([bv0, pv1, pv0])
                    else:
                        # Normal winding
                        new_faces.append([bv0, pv0, pv1])
                        new_faces.append([bv0, pv1, bv1])
                else:
                    # No edge info - default winding (assume projection is "outside")
                    new_faces.append([bv0, pv0, pv1])
                    new_faces.append([bv0, pv1, bv1])
            
            # Track inner rim edge
            inner_rim_edges.append([bv0, bv1])
    
    if len(new_vertices) == 0:
        return {
            'new_vertices': np.zeros((0, 3)),
            'new_faces': np.zeros((0, 3), dtype=np.int64),
            'part_constrained_indices': [],
            'fill_boundary_indices': [],
            'inner_rim_edges': np.zeros((0, 2), dtype=np.int64)
        }
    
    return {
        'new_vertices': np.array(new_vertices, dtype=np.float64),
        'new_faces': np.array(new_faces, dtype=np.int64),
        'part_constrained_indices': part_constrained_indices,
        'fill_boundary_indices': list(fill_boundary_indices),
        'inner_rim_edges': np.array(inner_rim_edges, dtype=np.int64) if inner_rim_edges else np.zeros((0, 2), dtype=np.int64)
    }
