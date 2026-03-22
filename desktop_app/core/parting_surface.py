"""
Parting Surface Extraction using Marching Tetrahedra (Paper Section 4.3)

This module extracts a continuous parting surface mesh from a tetrahedral mesh
where certain edges have been marked as "cut" (either primary or secondary cuts).

Algorithm Overview (Alderighi et al. SIGGRAPH 2019):
=====================================================
The triangulated surface C encoding the cut layout is composed using a set of patches
that are interconnected by chains of non-manifold edges that are bounded by construction
by the object surface mesh M and the external boundary ∂H.

Marching Tetrahedra for Binary Labeling (H1 vs H2):
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

Cut Point Placement (per Nielson & Franke 1997):
- If vertex_escape_distances available: use WEIGHTED interpolation
  t = d0 / (d0 + d1), cut_point = (1-t)*v0 + t*v1
- Otherwise: use geometric midpoint (t=0.5)

Edge numbering within a tetrahedron (vertices 0,1,2,3):
    Edge 0: (0,1)    Edge 3: (1,2)
    Edge 1: (0,2)    Edge 4: (1,3)
    Edge 2: (0,3)    Edge 5: (2,3)

Key Functions:
- extract_parting_surface(): Main Marching Tetrahedra algorithm
- extract_parting_surface_from_tet_result(): Convenience wrapper
- smooth_parting_surface(): Surface nets style smoothing
- repair_parting_surface(): Gap filling and cleanup

References:
- Alderighi et al., "Volume-Aware Design of Composite Molds", SIGGRAPH 2019, Section 4.3
- Nielson & Franke, "Computing the Separating Surface for Segmented Data", 1997
- Bloomenthal & Ferguson, "Polygonization of Non-Manifold Implicit Surfaces", 1995
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - CUT POINT PLACEMENT
# =============================================================================

# Weighted interpolation clamp bounds (prevents degenerate triangles)
# When using t = d0/(d0+d1), clamp t to [MIN_T, MAX_T]
CUT_POINT_INTERPOLATION_MIN_T = 0.1  # Don't place cut point too close to v0
CUT_POINT_INTERPOLATION_MAX_T = 0.9  # Don't place cut point too close to v1

# Vertex merge epsilon for near-coincident cut points (per Bloomenthal paper)
VERTEX_MERGE_EPSILON = 1e-8


# =============================================================================
# CONSTANTS - GAP FILLING AND REPAIR
# =============================================================================

# Gap filling thresholds (as fractions of edge length)
MIN_PROJECTION_DISTANCE_FRACTION = 0.1   # Minimum distance from boundary to projection point
PROJECTION_OFFSET_FRACTION = 0.3         # Offset distance when adjusting too-close projections

# Distance threshold multiplier for auto-computing gap fill threshold
# Higher value = more aggressive gap filling (catches edges farther from target surfaces)
DISTANCE_THRESHOLD_EDGE_MULTIPLIER = 3.0  # threshold = median_edge_length * this value


# =============================================================================
# CONSTANTS - TRIANGLE QUALITY THRESHOLDS
# =============================================================================

# Triangle area threshold (as fraction of median area)
MIN_TRIANGLE_AREA_FRACTION = 0.01        # 1% of median area considered degenerate


# =============================================================================
# CONSTANTS - ISLAND AND LOOP REMOVAL
# =============================================================================

# Small polyline removal thresholds
DEFAULT_MIN_LOOP_PERIMETER = 4.0  # Default minimum perimeter in mm for closed loops to keep

# Island removal for primary surfaces
PRIMARY_MIN_ISLAND_TRIANGLES = 10  # Minimum triangles to keep an island (primary surface)
PRIMARY_MIN_ISLAND_AREA_FRACTION = 0.01  # Islands with area < 1% of total are removed


# =============================================================================
# CONSTANTS - FLOATING EDGE DETECTION
# =============================================================================

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
# CONSTANTS - BOUNDARY TYPE CODES
# =============================================================================

# Boundary type codes for vertex classification (used during smoothing)
BOUNDARY_TYPE_PART = -1       # On part surface M (inner boundary - reproject to part)
BOUNDARY_TYPE_INTERIOR = 0    # Interior (midpoint - no boundary constraint)
BOUNDARY_TYPE_H1 = 1          # On hull H1 boundary (outer boundary - reproject to hull)
BOUNDARY_TYPE_H2 = 2          # On hull H2 boundary (outer boundary - reproject to hull)
BOUNDARY_TYPE_PRIMARY_JUNCTION = 3  # At primary-secondary junction (reproject to primary surface)
BOUNDARY_TYPE_PRIMARY_THEN_PART = 4  # Triple junction: reproject to primary, then to part


# =============================================================================
# MARCHING TETRAHEDRA LOOKUP TABLES
# =============================================================================
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

# Face definitions: face index -> (v_i, v_j, v_k) and which edges bound the face
# Each tetrahedral face is a triangle of 3 vertices.
# TET_FACE_EDGES[face_idx] = list of (local_edge_idx) that bound that face.
TET_FACES = [
    (0, 1, 2),  # Face 0: v0-v1-v2  (edges 0,1,3)
    (0, 1, 3),  # Face 1: v0-v1-v3  (edges 0,2,4)
    (0, 2, 3),  # Face 2: v0-v2-v3  (edges 1,2,5)
    (1, 2, 3),  # Face 3: v1-v2-v3  (edges 3,4,5)
]

TET_FACE_EDGES = [
    [0, 1, 3],  # Face 0: edges (v0-v1), (v0-v2), (v1-v2)
    [0, 2, 4],  # Face 1: edges (v0-v1), (v0-v3), (v1-v3)
    [1, 2, 5],  # Face 2: edges (v0-v2), (v0-v3), (v2-v3)
    [3, 4, 5],  # Face 3: edges (v1-v2), (v1-v3), (v2-v3)
]


def _build_marching_tet_table() -> Dict[int, object]:
    """
    Build the marching tetrahedra lookup table for BINARY and MULTI-REGION labeling.
    
    In a proper binary system where every vertex is labeled H1 or H2:
    - 0-edge config: All 4 vertices same label → no surface
    - 3-edge config: 1 vertex isolated (3+1 split) → single triangle
    - 4-edge config: 2 vs 2 vertices (2+2 split) → quad (2 triangles)
    
    For multi-region systems (secondary membranes meeting each other or primary):
    - 5-edge config: 3 regions meet → 2 face vertices + 1 inner vertex + 8 triangles
      Per Bloomenthal & Ferguson 1995, Section 4-5: TWO faces have 3 cut edges,
      each gets a face vertex. An inner vertex connects them. Each face line
      (edge midpoint to face vertex) + inner vertex = 1 triangle.
      Faces with 3 cut edges: 3 triangles each = 6. Faces with 2 cut edges:
      1 triangle each = 2. Total = 8 triangles, 3 new vertices.
    - 6-edge config: 4 regions meet → 4 face vertices + 1 inner vertex + 12 triangles
      Per Bloomenthal 1995, Section 4: "one tetrahedron may contain four face
      vertices... our polygonizer supports an inner vertex within the tetrahedron"
    
    Edge numbering:
        Edge 0: (v0, v1)    Edge 3: (v1, v2)
        Edge 1: (v0, v2)    Edge 4: (v1, v3)
        Edge 2: (v0, v3)    Edge 5: (v2, v3)
    
    Returns:
        Dictionary mapping 6-bit config -> triangulation specification:
        - For 0/3/4-edge: List of (edge0, edge1, edge2) tuples
        - For 5-edge: 'FACE_VERTEX' sentinel (handled procedurally)
        - For 6-edge: 'INNER_VERTEX' sentinel (handled procedurally)
    """
    table = {}
    
    # Initialize all 64 configs to empty
    for i in range(64):
        table[i] = []
    
    # ===========================================================================
    # 3-EDGE CONFIGS: Single triangle from 3 edge midpoints.
    #
    # In a BINARY system, only 4 configs are valid (one vertex isolated).
    # However, for MULTI-REGION secondary surfaces (mixed secondary + junction
    # primary edges), ANY 3-edge combination can occur. All produce a single
    # triangle from the 3 cut edge midpoints.
    #
    # We enumerate ALL C(6,3) = 20 three-edge configs so that secondary
    # surfaces at junctions don't leave gaps.
    # ===========================================================================
    # --- Binary-valid configs (one vertex isolated) ---
    table[7]  = [(0, 1, 2)]   # v0 isolated: edges 0,1,2
    table[25] = [(0, 3, 4)]   # v1 isolated: edges 0,3,4
    table[42] = [(1, 3, 5)]   # v2 isolated: edges 1,3,5
    table[52] = [(2, 4, 5)]   # v3 isolated: edges 2,4,5
    
    # --- Multi-region 3-edge configs (not binary-valid) ---
    # These occur at secondary-primary junctions where the mixed cut flags
    # don't correspond to a binary vertex labeling.
    table[11] = [(0, 1, 3)]   # edges 0,1,3 (face 0: v0-v1-v2)
    table[13] = [(0, 2, 3)]   # edges 0,2,3
    table[14] = [(1, 2, 3)]   # edges 1,2,3
    table[19] = [(0, 1, 4)]   # edges 0,1,4
    table[21] = [(0, 2, 4)]   # edges 0,2,4 (face 1: v0-v1-v3)
    table[22] = [(1, 2, 4)]   # edges 1,2,4
    table[26] = [(1, 3, 4)]   # edges 1,3,4
    table[28] = [(2, 3, 4)]   # edges 2,3,4
    table[35] = [(0, 1, 5)]   # edges 0,1,5
    table[37] = [(0, 2, 5)]   # edges 0,2,5
    table[38] = [(1, 2, 5)]   # edges 1,2,5 (face 2: v0-v2-v3)
    table[41] = [(0, 3, 5)]   # edges 0,3,5
    table[44] = [(2, 3, 5)]   # edges 2,3,5
    table[49] = [(0, 4, 5)]   # edges 0,4,5
    table[50] = [(1, 4, 5)]   # edges 1,4,5
    table[56] = [(3, 4, 5)]   # edges 3,4,5 (face 3: v1-v2-v3)
    
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
    # 4-EDGE MULTI-REGION CONFIGS: face vertex + connecting triangles
    #
    # In multi-region secondary surfaces, 4-edge configs that aren't binary-valid
    # occur at junctions. ALL 12 unhandled 4-edge configs have exactly one face
    # with all 3 edges cut. We treat them as FACE_VERTEX: the single-face
    # fallback creates a face vertex on the 3-cut face + connecting triangles
    # for the extra edge.
    # ===========================================================================
    table[15] = 'FACE_VERTEX'   # edges 0,1,2,3: face 0 (0,1,3) + edge 2
    table[23] = 'FACE_VERTEX'   # edges 0,1,2,4: face 1 (0,2,4) + edge 1
    table[27] = 'FACE_VERTEX'   # edges 0,1,3,4: face 0 (0,1,3) + edge 4
    table[29] = 'FACE_VERTEX'   # edges 0,2,3,4: face 1 (0,2,4) + edge 3
    table[39] = 'FACE_VERTEX'   # edges 0,1,2,5: face 2 (1,2,5) + edge 0
    table[43] = 'FACE_VERTEX'   # edges 0,1,3,5: face 0 (0,1,3) + edge 5
    table[46] = 'FACE_VERTEX'   # edges 1,2,3,5: face 2 (1,2,5) + edge 3
    table[53] = 'FACE_VERTEX'   # edges 0,2,4,5: face 1 (0,2,4) + edge 5
    table[54] = 'FACE_VERTEX'   # edges 1,2,4,5: face 2 (1,2,5) + edge 4
    table[57] = 'FACE_VERTEX'   # edges 0,3,4,5: face 3 (3,4,5) + edge 0
    table[58] = 'FACE_VERTEX'   # edges 1,3,4,5: face 3 (3,4,5) + edge 1
    table[60] = 'FACE_VERTEX'   # edges 2,3,4,5: face 3 (3,4,5) + edge 2
    
    # ===========================================================================
    # 5-EDGE CONFIGS: Three regions meet → 2 face vertices + 1 inner vertex
    #
    # Per Bloomenthal & Ferguson 1995, Section 4-5:
    # In a 5-edge config, TWO faces have all 3 edges cut (both faces contain
    # exactly one endpoint of the uncut edge plus the other 2 vertices).
    # Each of these faces spans 3 regions → each gets a face vertex.
    # The two face vertices connect through an inner vertex.
    #
    # Generates 8 triangles: 3 per 3-cut face + 1 per 2-cut face
    # 3 new vertices: 2 face vertices + 1 inner vertex
    #
    # These are handled procedurally by _generate_5_edge_triangles()
    # ===========================================================================
    table[62] = 'FACE_VERTEX'  # Edge 0 (v0-v1) not cut → face 3 (v1,v2,v3) has 3 cuts but that's wrong
    table[61] = 'FACE_VERTEX'  # Edge 1 (v0-v2) not cut
    table[59] = 'FACE_VERTEX'  # Edge 2 (v0-v3) not cut
    table[55] = 'FACE_VERTEX'  # Edge 3 (v1-v2) not cut
    table[47] = 'FACE_VERTEX'  # Edge 4 (v1-v3) not cut
    table[31] = 'FACE_VERTEX'  # Edge 5 (v2-v3) not cut
    
    # ===========================================================================
    # 6-EDGE CONFIG: Four regions meet → inner vertex required
    #
    # Per Bloomenthal & Ferguson 1995, Section 4:
    # "one tetrahedron may contain four face vertices... our polygonizer
    # supports an inner vertex within the tetrahedron that connects face
    # and edge vertices"
    #
    # All 4 faces have 3 cut edges each → each gets a face vertex.
    # These 4 face vertices connect through 1 inner vertex at their average.
    # This produces 12 triangles (3 per face, each: edge_v, face_v, inner_v).
    #
    # Handled procedurally by _generate_6_edge_triangles()
    # ===========================================================================
    table[63] = 'INNER_VERTEX'  # All 6 edges cut
    
    return table


# Pre-build the lookup table
MARCHING_TET_TABLE = _build_marching_tet_table()


# =============================================================================
# 5-EDGE AND 6-EDGE CONFIGURATION HELPERS (Bloomenthal & Ferguson 1995)
# =============================================================================

def _find_faces_with_3_cut_edges(config: int) -> List[int]:
    """
    For a 5-edge config, find ALL faces that have all 3 edges cut.
    
    In a 5-edge config (one edge uncut), EXACTLY TWO faces have all 3 edges cut.
    These are the two faces that each contain exactly one endpoint of the uncut
    edge plus the other two vertices.
    
    For example, with edge 0 (v0-v1) uncut:
    - Face 2 (v0,v2,v3): edges 1,2,5 → all cut
    - Face 3 (v1,v2,v3): edges 3,4,5 → all cut
    
    Per Bloomenthal & Ferguson 1995 Section 4: Both faces have 3 regions meeting,
    so both need face vertices. These face vertices connect through an inner vertex.
    
    Args:
        config: 6-bit configuration bitmask (5 bits set)
        
    Returns:
        List of face indices (0-3) where all 3 edges are cut (should be exactly 2)
    """
    faces = []
    for face_idx, face_edges in enumerate(TET_FACE_EDGES):
        all_cut = all((config & (1 << e)) for e in face_edges)
        if all_cut:
            faces.append(face_idx)
    return faces


def _generate_5_edge_triangles(
    config: int,
    local_edge_to_sv: Dict[int, int],
    surface_vertices: np.ndarray,
    next_vertex_index: int
) -> Tuple[List[List[int]], np.ndarray, np.ndarray, int]:
    """
    Generate triangles for a 5-edge configuration using TWO face vertices +
    ONE inner vertex per Bloomenthal & Ferguson 1995.
    
    In a 5-edge config, the single uncut edge connects two tet vertices that
    share the same region (say A). The other two vertices are in regions B and C.
    Three surfaces meet inside this tet: A|B, A|C, and B|C.
    
    EXACTLY TWO faces have all 3 edges cut (the faces not containing the uncut
    edge's vertex pair together). Each of these faces spans 3 regions → each
    needs a face vertex. The two remaining faces each have exactly 2 cut edges.
    
    Per Bloomenthal Section 5: "In the case of non-disjoint surfaces, there
    must be at least two face vertices that separate regions of interest. We
    create an inner vertex whose location is the average location of those
    face vertices. Each face line, together with this inner vertex, creates
    one triangle."
    
    Triangulation:
    - For each face with 3 cut edges: 3 triangles (edge_midpoint, face_vertex, inner_vertex)
    - For each face with 2 cut edges: 1 triangle (edge_midpoint_a, edge_midpoint_b, inner_vertex)  
    - Total: 3 + 3 + 1 + 1 = 8 triangles, 3 new vertices (fv_a, fv_b, inner)
    
    Args:
        config: 6-bit configuration (5 edges cut)
        local_edge_to_sv: Map from local edge index to surface vertex index
        surface_vertices: Current array of surface vertex positions
        next_vertex_index: Index to assign to the first new vertex
        
    Returns:
        Tuple of:
        - triangles: List of [sv0, sv1, sv2] triangle vertex index lists (8 triangles)
        - new_vertex_positions: (3, 3) positions of 2 face vertices + 1 inner vertex
        - new_vertex_bts: (3,) boundary types (all 0 = interior)
        - n_new_vertices: Number of new vertices created (3)
    """
    # Find BOTH faces with all 3 edges cut
    faces_3cut = _find_faces_with_3_cut_edges(config)
    
    if len(faces_3cut) < 2:
        # Single-face approach: 4-edge multi-region configs have exactly 1 face
        # with 3 cuts. Also used as fallback for any unexpected 5-edge case.
        n_cut = bin(config).count('1')
        if n_cut != 5:
            logger.debug(f"{n_cut}-edge config {config:06b}: 1 face with 3 cuts — "
                        f"using single-face handler")
        else:
            logger.warning(f"5-edge config {config:06b}: expected 2 faces with 3 cut edges, "
                          f"found {len(faces_3cut)} — using single-face fallback")
        if len(faces_3cut) == 1:
            return _generate_5_edge_triangles_single_face_fallback(
                config, faces_3cut[0], local_edge_to_sv, surface_vertices, next_vertex_index
            )
        return [], np.zeros((0, 3)), np.zeros(0, dtype=np.int8), 0
    
    # Identify the 2 faces with 2 cut edges (the other two faces)
    faces_2cut = [f for f in range(4) if f not in faces_3cut]
    
    # Step 1: Create face vertices for the two 3-cut faces
    face_vertex_positions = []
    face_vertex_indices = []
    
    for i, face_idx in enumerate(faces_3cut):
        face_edge_indices = TET_FACE_EDGES[face_idx]
        positions = []
        for le in face_edge_indices:
            sv = local_edge_to_sv.get(le)
            if sv is None:
                logger.warning(f"5-edge config {config:06b}: missing surface vertex "
                             f"for edge {le} on face {face_idx}")
                return [], np.zeros((0, 3)), np.zeros(0, dtype=np.int8), 0
            positions.append(surface_vertices[sv])
        
        fv_pos = np.mean(positions, axis=0)
        face_vertex_positions.append(fv_pos)
        face_vertex_indices.append(next_vertex_index + i)
    
    # Step 2: Create inner vertex at centroid of face vertices
    inner_pos = np.mean(face_vertex_positions, axis=0)
    inner_idx = next_vertex_index + 2
    
    # Step 3: Generate triangles
    triangles = []
    
    # For each face with 3 cut edges: 3 triangles (sv, fv, iv)
    # Per Bloomenthal: "Each face line, together with this inner vertex,
    # creates one triangle."
    for i, face_idx in enumerate(faces_3cut):
        fv_idx = face_vertex_indices[i]
        face_edge_indices = TET_FACE_EDGES[face_idx]
        
        for le in face_edge_indices:
            sv = local_edge_to_sv[le]
            triangles.append([sv, fv_idx, inner_idx])
    
    # For each face with 2 cut edges: 1 triangle (sv_a, sv_b, iv)
    # These faces have 2 cut edges and 1 uncut edge. The 2 edge midpoints
    # connect directly through the inner vertex.
    for face_idx in faces_2cut:
        face_edge_indices = TET_FACE_EDGES[face_idx]
        cut_svs = []
        for le in face_edge_indices:
            if config & (1 << le):  # Edge is cut
                sv = local_edge_to_sv.get(le)
                if sv is not None:
                    cut_svs.append(sv)
        
        if len(cut_svs) == 2:
            triangles.append([cut_svs[0], cut_svs[1], inner_idx])
        elif len(cut_svs) > 0:
            logger.warning(f"5-edge config {config:06b} face {face_idx}: "
                         f"expected 2 cut edges, found {len(cut_svs)}")
    
    # Build new vertex arrays: [fv_a, fv_b, inner]
    all_new_positions = np.array(
        face_vertex_positions + [inner_pos], dtype=np.float64
    )
    all_new_bts = np.zeros(3, dtype=np.int8)  # All interior
    
    return triangles, all_new_positions, all_new_bts, 3


def _generate_5_edge_triangles_single_face_fallback(
    config: int,
    face_idx: int,
    local_edge_to_sv: Dict[int, int],
    surface_vertices: np.ndarray,
    next_vertex_index: int
) -> Tuple[List[List[int]], np.ndarray, np.ndarray, int]:
    """
    Degraded fallback for 5-edge config when only 1 face with 3 cut edges is found.
    
    This creates a single face vertex and fans 3 triangles around it on the
    3-cut face, plus 2 connecting triangles to the remaining cut edges.
    This is geometrically incorrect but provides some surface coverage as a fallback.
    
    Args:
        config: 6-bit configuration
        face_idx: The single face with 3 cut edges
        local_edge_to_sv: Edge-to-surface-vertex mapping
        surface_vertices: Current vertex positions
        next_vertex_index: Next available vertex index
        
    Returns:
        Same tuple format as _generate_5_edge_triangles
    """
    face_edge_indices = TET_FACE_EDGES[face_idx]
    
    face_svs = []
    face_positions = []
    for le in face_edge_indices:
        sv = local_edge_to_sv.get(le)
        if sv is None:
            return [], np.zeros((0, 3)), np.zeros(0, dtype=np.int8), 0
        face_svs.append(sv)
        face_positions.append(surface_vertices[sv])
    
    face_vertex_pos = np.mean(face_positions, axis=0)
    fv_idx = next_vertex_index
    
    triangles = []
    for i in range(3):
        sv_a = face_svs[i]
        sv_b = face_svs[(i + 1) % 3]
        triangles.append([sv_a, sv_b, fv_idx])
    
    # Connect remaining cut edges (not on the 3-cut face) through face vertex.
    # For each extra cut edge, find all face-edges it shares a tet face with
    # and create connecting triangles. This ensures proper surface coverage
    # at junctions where secondary meets primary.
    remaining_cut_edges = [e for e in range(6)
                          if (config & (1 << e)) and e not in face_edge_indices]
    
    for re in remaining_cut_edges:
        sv_r = local_edge_to_sv.get(re)
        if sv_r is not None:
            re_verts = set(TET_EDGES[re])
            for i, fe in enumerate(face_edge_indices):
                fe_verts = set(TET_EDGES[fe])
                if fe_verts & re_verts:
                    triangles.append([face_svs[i], sv_r, fv_idx])
    
    return triangles, face_vertex_pos.reshape(1, 3), np.array([0], dtype=np.int8), 1


def _generate_6_edge_triangles(
    local_edge_to_sv: Dict[int, int],
    surface_vertices: np.ndarray,
    next_vertex_index: int
) -> Tuple[List[List[int]], np.ndarray, np.ndarray, int]:
    """
    Generate triangles for a 6-edge configuration using face + inner vertices.
    
    Per Bloomenthal & Ferguson 1995 Section 4, Figure 11:
    When all 4 tet corners have different region labels, each of the 4 faces
    has 3 cut edges. We create:
    - 4 face vertices (one per face, at centroid of that face's 3 edge midpoints)
    - 1 inner vertex (at the centroid of the 4 face vertices)
    - 12 triangles (3 per face): each triangle is (edge_midpoint, face_vertex, inner_vertex)
    
    Per Bloomenthal: "each face line, together with this inner vertex, creates
    one triangle"
    
    Args:
        local_edge_to_sv: Map from local edge index to surface vertex index
        surface_vertices: Current array of surface vertex positions
        next_vertex_index: Index to assign to the first new vertex
        
    Returns:
        Tuple of:
        - triangles: List of [sv0, sv1, sv2] triangle vertex index lists
        - new_vertex_positions: (5, 3) positions of the 4 face vertices + 1 inner vertex
        - new_vertex_bts: (5,) boundary types (all 0 = interior)
        - n_new_vertices: Number of new vertices created (5)
    """
    # Verify all 6 edge midpoints exist
    for e in range(6):
        if e not in local_edge_to_sv:
            logger.warning(f"6-edge config: missing surface vertex for edge {e}")
            return [], np.zeros((0, 3)), np.zeros(0, dtype=np.int8), 0
    
    # Step 1: Compute 4 face vertices
    face_vertex_positions = []
    face_vertex_indices = []  # Will be next_vertex_index, +1, +2, +3
    
    for face_idx in range(4):
        face_edges = TET_FACE_EDGES[face_idx]
        positions = [surface_vertices[local_edge_to_sv[e]] for e in face_edges]
        fv_pos = np.mean(positions, axis=0)
        face_vertex_positions.append(fv_pos)
        face_vertex_indices.append(next_vertex_index + face_idx)
    
    # Step 2: Compute inner vertex = centroid of face vertices
    inner_pos = np.mean(face_vertex_positions, axis=0)
    inner_idx = next_vertex_index + 4
    
    # Step 3: Generate 12 triangles (3 per face)
    # On each face, the 3 edge midpoints and the face vertex create 3 triangles.
    # Each triangle connects: (edge_midpoint_i, edge_midpoint_j, face_vertex)
    # But per Bloomenthal, with the inner vertex, each face line (between two
    # edge midpoints through the face vertex) becomes a triangle:
    # (edge_midpoint, face_vertex, inner_vertex)
    #
    # Actually Bloomenthal says "Each face line, together with this inner vertex,
    # creates one triangle." A face line connects two edge vertices through the
    # face vertex. So each face has 3 face lines (connecting pairs of edge midpoints
    # via the face vertex), and each line + inner_vertex = 1 triangle.
    #
    # Triangle for each edge midpoint on the face:
    # (edge_midpoint_i, face_vertex, inner_vertex)
    # 3 such triangles per face, 4 faces = 12 triangles.
    
    triangles = []
    for face_idx in range(4):
        fv_idx = face_vertex_indices[face_idx]
        face_edges = TET_FACE_EDGES[face_idx]
        
        for i in range(3):
            sv_a = local_edge_to_sv[face_edges[i]]
            sv_b = local_edge_to_sv[face_edges[(i + 1) % 3]]
            triangles.append([sv_a, sv_b, fv_idx])
            # Also fan to inner vertex
        
    # Wait — the above produces 12 triangles on the face surfaces, but doesn't
    # connect through the inner vertex. Let me reconsider.
    #
    # The correct interpretation per Bloomenthal Figure 11 and Section 5:
    # - On each face, 3 edge midpoints connect to 1 face vertex → 3 triangles
    #   (this triangulates the face's portion of the separating surface)
    # - The face vertices of different faces are then connected through the
    #   inner vertex to form the "interior" portion of the separating surfaces.
    #
    # For a full 4-region separation, each pair of adjacent faces shares one
    # edge midpoint. The surface between their two regions goes through:
    #   edge_midpoint → face_vertex_A → inner_vertex → face_vertex_B → edge_midpoint
    #
    # The simplest correct approach: for each face, create 3 triangles fanning
    # the edge midpoints around the face vertex (same as 5-edge face treatment),
    # then for each edge of the tet, create a triangle connecting:
    #   (face_vertex_of_face_A, face_vertex_of_face_B, inner_vertex)
    # where face_A and face_B are the two faces that share that edge.
    #
    # Total: 4×3 + 6 = 18 triangles? That's too many.
    #
    # Actually the correct Bloomenthal approach is simpler:
    # Each face has 3 "face lines" (segments from edge_midpoint through face_vertex).
    # Each face line + inner_vertex = 1 triangle.
    # So: 4 faces × 3 face lines = 12 triangles.
    # Triangle = (edge_midpoint_i, face_vertex_of_this_face, inner_vertex)
    
    triangles = []
    for face_idx in range(4):
        fv_idx = face_vertex_indices[face_idx]
        face_edges = TET_FACE_EDGES[face_idx]
        
        for e_local in face_edges:
            sv = local_edge_to_sv[e_local]
            triangles.append([sv, fv_idx, inner_idx])
    
    # Build new vertex arrays
    all_new_positions = np.array(face_vertex_positions + [inner_pos], dtype=np.float64)
    all_new_bts = np.zeros(5, dtype=np.int8)  # All interior
    
    return triangles, all_new_positions, all_new_bts, 5


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


# =============================================================================
# CUT POINT PLACEMENT HELPERS
# =============================================================================

def _compute_cut_point_position(
    v0_pos: np.ndarray,
    v1_pos: np.ndarray,
    d0: Optional[float] = None,
    d1: Optional[float] = None
) -> Tuple[np.ndarray, bool]:
    """
    Compute the position of a cut point on an edge.
    
    Per Nielson & Franke 1997: "In some applications where there is additional
    information on which to base any bias or adjustment... weights may be used"
    
    If escape distances are available, use weighted interpolation:
        t = d0 / (d0 + d1)
        cut_point = (1-t) * v0 + t * v1
    
    This places the cut point closer to the vertex with shorter escape path,
    which better approximates the true parting surface location.
    
    Args:
        v0_pos: Position of first vertex
        v1_pos: Position of second vertex
        d0: Escape distance of first vertex (None for midpoint)
        d1: Escape distance of second vertex (None for midpoint)
    
    Returns:
        Tuple of (cut_point_position, used_weighted) where used_weighted indicates
        whether weighted interpolation was used (True) or midpoint fallback (False).
    """
    # Check if weighted placement is possible
    if (d0 is not None and d1 is not None and 
        d0 > 0 and d1 > 0 and 
        np.isfinite(d0) and np.isfinite(d1)):
        
        # t = d0 / (d0 + d1) means cut point is closer to v0 when d0 is small
        t = d0 / (d0 + d1)
        # Clamp to avoid degenerate triangles
        t = np.clip(t, CUT_POINT_INTERPOLATION_MIN_T, CUT_POINT_INTERPOLATION_MAX_T)
        cut_point = (1.0 - t) * v0_pos + t * v1_pos
        return cut_point, True
    else:
        # Fallback to midpoint
        return 0.5 * (v0_pos + v1_pos), False


def _determine_boundary_type_primary(
    bl0: int,
    bl1: int
) -> int:
    """
    Determine boundary type for a PRIMARY surface cut vertex.
    
    Priority: Part surface M (-1) > Hull (1,2) > Interior (0)
    
    Args:
        bl0: Boundary label of first edge endpoint (-1=part, 0=interior, 1=H1, 2=H2)
        bl1: Boundary label of second edge endpoint
    
    Returns:
        Boundary type code for the cut vertex:
        -1 = part boundary, 1/2 = hull boundary, 0 = interior
    """
    # If either endpoint is on part surface, mark for re-projection to part M
    if bl0 == BOUNDARY_TYPE_PART or bl1 == BOUNDARY_TYPE_PART:
        return BOUNDARY_TYPE_PART
    
    # Else if either endpoint is on hull, use that hull label
    if bl0 in (BOUNDARY_TYPE_H1, BOUNDARY_TYPE_H2):
        return bl0
    if bl1 in (BOUNDARY_TYPE_H1, BOUNDARY_TYPE_H2):
        return bl1
    
    # Both interior
    return BOUNDARY_TYPE_INTERIOR


def _determine_boundary_type_secondary(
    bl0: int,
    bl1: int,
    is_primary_adjacent: bool
) -> int:
    """
    Determine boundary type for a SECONDARY surface cut vertex.
    
    For secondary membranes, the boundary condition is the PRIMARY parting
    surface, not the hull. We use two complementary signals:
    
    1. PRIMARY ADJACENCY: If edge is adjacent to primary cut, mark as bt=3
       (primary junction) so it gets re-projected to the primary surface.
    
    2. BOUNDARY LABELS: For non-junction edges:
       - BOTH on part (-1/-1) → -1 (genuine part boundary)
       - ONE on part → 0 (interior, free to smooth)
       - Hull endpoints → hull label (outer boundary)
       - BOTH interior → 0 (interior, free to smooth)
    
    Args:
        bl0: Boundary label of first edge endpoint
        bl1: Boundary label of second edge endpoint
        is_primary_adjacent: True if edge is adjacent to a primary cut edge
    
    Returns:
        Boundary type code for the cut vertex
    """
    # Primary adjacency takes precedence
    if is_primary_adjacent:
        return BOUNDARY_TYPE_PRIMARY_JUNCTION
    
    # Both endpoints on part surface - genuine part boundary
    if bl0 == BOUNDARY_TYPE_PART and bl1 == BOUNDARY_TYPE_PART:
        return BOUNDARY_TYPE_PART
    
    # ONE endpoint on part - do NOT pin (would fold membrane)
    if bl0 == BOUNDARY_TYPE_PART or bl1 == BOUNDARY_TYPE_PART:
        return BOUNDARY_TYPE_INTERIOR
    
    # Hull endpoints
    if bl0 in (BOUNDARY_TYPE_H1, BOUNDARY_TYPE_H2):
        return bl0
    if bl1 in (BOUNDARY_TYPE_H1, BOUNDARY_TYPE_H2):
        return bl1
    
    # Both interior
    return BOUNDARY_TYPE_INTERIOR


def _orient_triangle_by_mold_labels(
    tri_verts: List[int],
    surface_vertices: np.ndarray,
    tet_verts: np.ndarray,
    vertex_mold_labels: np.ndarray,
    verts: np.ndarray
) -> List[int]:
    """
    Orient triangle so normal points from H1 towards H2 (for PRIMARY surfaces).
    
    Uses the H2 vertices in the tetrahedron to determine which side is "H2",
    then ensures the triangle normal points towards that side.
    
    Args:
        tri_verts: Triangle vertex indices (3 elements)
        surface_vertices: All surface vertex positions
        tet_verts: The 4 vertex indices of the tetrahedron
        vertex_mold_labels: Mold labels for all vertices (1=H1, 2=H2)
        verts: All vertex positions
    
    Returns:
        Possibly reordered tri_verts with consistent orientation
    """
    # Get positions of the 3 cut points
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
            return [tri_verts[0], tri_verts[2], tri_verts[1]]  # Flip winding
    
    return tri_verts


def _orient_triangle_by_escape_distance(
    tri_verts: List[int],
    surface_vertices: np.ndarray,
    tet_verts: np.ndarray,
    vertex_escape_distances: np.ndarray,
    verts: np.ndarray
) -> List[int]:
    """
    Orient triangle based on escape distances (for SECONDARY surfaces).
    
    The side with LONGER escape distance is the "outside" - secondary cuts
    occur where paths diverge around the part M.
    
    Args:
        tri_verts: Triangle vertex indices (3 elements)
        surface_vertices: All surface vertex positions
        tet_verts: The 4 vertex indices of the tetrahedron
        vertex_escape_distances: Escape distances for all vertices
        verts: All vertex positions
    
    Returns:
        Possibly reordered tri_verts with consistent orientation
    """
    # Get triangle geometry
    p0 = surface_vertices[tri_verts[0]]
    p1 = surface_vertices[tri_verts[1]]
    p2 = surface_vertices[tri_verts[2]]
    
    tri_center = (p0 + p1 + p2) / 3.0
    edge1 = p1 - p0
    edge2 = p2 - p0
    normal = np.cross(edge1, edge2)
    
    # Find the tet vertex with LONGEST escape distance
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
            return [tri_verts[0], tri_verts[2], tri_verts[1]]  # Flip winding
    
    return tri_verts


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
    use_label_derived_cuts: bool = True,
    is_secondary: bool = False,
    primary_cut_vertex_mask: Optional[np.ndarray] = None
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
        is_secondary: If True, use stricter vertex_boundary_type assignment for secondary surfaces.
            For secondary surfaces, a cut vertex is only labeled -1 (part) if BOTH tet endpoints
            are on the part surface. If only one endpoint is on the part, the vertex is labeled 0
            (interior) so it can be re-projected to the primary membrane instead.
        primary_cut_vertex_mask: (N,) boolean mask where True means the vertex is an endpoint
            of a primary cut edge. Used only when is_secondary=True to detect junction edges —
            secondary cut edges whose tet-mesh endpoints touch primary cut vertices get
            vertex_boundary_type=3 (primary junction), meaning they should be re-projected
            to the closest primary surface during smoothing.
    
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
    n_primary_junction = 0  # Cut points at primary junction (will re-project to primary)
    n_interior = 0        # Cut points in interior (no re-projection needed)
    n_weighted = 0        # Cut points placed using weighted interpolation
    n_midpoint = 0        # Cut points placed at midpoint (fallback)
    
    # For secondary surfaces: identify cut edges adjacent to primary cuts.
    # A secondary cut edge is "adjacent" if either of its tet-mesh endpoints
    # is also an endpoint of a primary cut edge. Such vertices lie at the
    # junction between the secondary and primary surfaces and must be
    # re-projected to the primary surface during smoothing.
    has_primary_vertex_mask = (is_secondary and 
                               primary_cut_vertex_mask is not None and
                               len(primary_cut_vertex_mask) > 0)
    
    # Check if weighted placement is possible
    use_weighted = (vertex_escape_distances is not None and 
                    len(vertex_escape_distances) == len(vertices))
    if use_weighted:
        logger.info("Using WEIGHTED cut point placement (per Nielson & Franke 1997)")
    else:
        logger.info("Using MIDPOINT cut point placement (no escape distances available)")
    
    for i, e_idx in enumerate(cut_edge_indices):
        v0, v1 = edges[e_idx]
        
        # Compute cut point position using helper
        if use_weighted:
            d0 = vertex_escape_distances[v0]
            d1 = vertex_escape_distances[v1]
        else:
            d0, d1 = None, None
        
        cut_point, was_weighted = _compute_cut_point_position(
            verts[v0], verts[v1], d0, d1
        )
        surface_vertices[i] = cut_point
        if was_weighted:
            n_weighted += 1
        else:
            n_midpoint += 1
        
        # Determine boundary type for LATER re-projection (not for placement)
        if boundary_labels is not None:
            bl0 = boundary_labels[v0]
            bl1 = boundary_labels[v1]
            
            if is_secondary:
                # Check if edge is adjacent to a primary cut edge
                is_primary_adjacent = (has_primary_vertex_mask and 
                                       (primary_cut_vertex_mask[v0] or primary_cut_vertex_mask[v1]))
                bt = _determine_boundary_type_secondary(bl0, bl1, is_primary_adjacent)
            else:
                bt = _determine_boundary_type_primary(bl0, bl1)
            
            vertex_boundary_type[i] = bt
            
            # Track statistics
            if bt == BOUNDARY_TYPE_PART:
                n_part_boundary += 1
            elif bt in (BOUNDARY_TYPE_H1, BOUNDARY_TYPE_H2):
                n_hull_boundary += 1
            elif bt == BOUNDARY_TYPE_PRIMARY_JUNCTION:
                n_primary_junction += 1
            else:
                n_interior += 1
        else:
            vertex_boundary_type[i] = BOUNDARY_TYPE_INTERIOR
            n_interior += 1
    
    logger.info(f"Cut point placement: {n_weighted} weighted, {n_midpoint} midpoint")
    if is_secondary:
        logger.info(f"Cut point boundary types: {n_part_boundary} part M, "
                    f"{n_hull_boundary} hull ∂H, {n_primary_junction} primary junction, "
                    f"{n_interior} interior")
    else:
        logger.info(f"Cut point boundary types: {n_part_boundary} near part M, "
                    f"{n_hull_boundary} near hull ∂H, {n_interior} interior")
    
    # Post-processing for secondary surfaces: detect bt=4 (primary-then-part)
    # 
    # A bt=3 vertex that is connected by a surface edge to a bt=-1 vertex sits
    # at the triple junction where secondary membrane, primary membrane, and part
    # all meet. This vertex needs DUAL reprojection: first to the primary surface,
    # then to the part surface — pinning it to the curve where primary meets part.
    n_primary_then_part = 0
    if is_secondary and len(cut_edge_indices) > 0:
        # Build adjacency from the triangles we're about to create — but we don't
        # have triangles yet. Instead, check all pairs of cut vertices that share
        # a tetrahedron edge. Two cut vertices are adjacent in the surface if they
        # appear in the same tet's marching configuration.
        # 
        # Simpler approach: iterate over all cut edge pairs within each tet.
        # If both edges are cut, the resulting surface vertices are adjacent.
        cut_edge_set = set(int(e) for e in cut_edge_indices)
        
        # For each tet, find pairs of cut edges → pairs of surface vertices
        bt3_adjacent_to_part = set()
        for t in range(len(tetrahedra)):
            tet_edges_global = tet_edge_indices[t]
            # Find which of the 6 edges are cut
            cut_local = [k for k in range(6) if int(tet_edges_global[k]) in cut_edge_set]
            if len(cut_local) < 2:
                continue
            # Check all pairs of cut edges in this tet
            for a_idx in range(len(cut_local)):
                for b_idx in range(a_idx + 1, len(cut_local)):
                    ea = int(tet_edges_global[cut_local[a_idx]])
                    eb = int(tet_edges_global[cut_local[b_idx]])
                    sa = edge_to_surface_vertex.get(ea)
                    sb = edge_to_surface_vertex.get(eb)
                    if sa is None or sb is None:
                        continue
                    bta = vertex_boundary_type[sa]
                    btb = vertex_boundary_type[sb]
                    # If one is part (-1) and other is primary junction (3),
                    # upgrade the bt=3 vertex to bt=4
                    if bta == -1 and btb == 3:
                        bt3_adjacent_to_part.add(sb)
                    elif btb == -1 and bta == 3:
                        bt3_adjacent_to_part.add(sa)
        
        for si in bt3_adjacent_to_part:
            vertex_boundary_type[si] = 4
            n_primary_then_part += 1
            n_primary_junction -= 1
        
        if n_primary_then_part > 0:
            logger.info(f"Detected {n_primary_then_part} primary-then-part (bt=4) vertices "
                       f"at triple junctions (primary ∩ part ∩ secondary)")
    
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
    n_skipped_configs = 0  # Count of unresolvable configs
    n_5edge_configs = 0    # Count of 5-edge configs resolved with face vertex
    n_6edge_configs = 0    # Count of 6-edge configs resolved with inner vertex
    n_label_derived = 0    # Count of configs derived from vertex labels
    n_flag_derived = 0     # Count of configs derived from cut_edge_flags
    
    # For 5-edge and 6-edge configs: we create NEW vertices (face/inner vertices)
    # that don't sit on any tet edge. We collect them here and append after the loop.
    new_vertex_positions = []  # List of np.ndarray (K, 3) arrays
    new_vertex_bts = []        # List of np.ndarray (K,) arrays
    next_new_vertex_index = len(surface_vertices)  # Starting index for new vertices
    
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
            # INVALID CONFIGURATION (5 or 6 edges cut) - LEGACY SKIP
            # Should not reach here anymore; FACE_VERTEX/INNER_VERTEX handle these.
            # =================================================================
            n_skipped_configs += 1
            continue
        
        elif table_entry == 'FACE_VERTEX':
            # =================================================================
            # 5-EDGE CONFIGURATION: 2 Face vertices + 1 inner vertex required
            # Per Bloomenthal & Ferguson 1995, Sections 4-5:
            # Three regions meet → 2 faces with 3 cut edges each get a face
            # vertex, connected through an inner vertex. 8 triangles total.
            # =================================================================
            n_5edge_configs += 1
            tets_contributing += 1
            
            new_tris, new_pos, new_bt, n_new = _generate_5_edge_triangles(
                config, local_edge_to_vertex, surface_vertices,
                next_new_vertex_index
            )
            
            if n_new > 0 and new_tris:
                # Append the face vertex to our growing lists
                new_vertex_positions.append(new_pos)
                new_vertex_bts.append(new_bt)
                
                # Temporarily extend surface_vertices so orientation can access
                # the new face vertex positions by index
                surface_vertices = np.concatenate(
                    [surface_vertices, new_pos], axis=0
                )
                next_new_vertex_index += n_new
                
                # Orient and append triangles
                for tri_verts in new_tris:
                    if vertex_escape_distances is not None:
                        tri_verts = _orient_triangle_by_escape_distance(
                            tri_verts, surface_vertices, tet_verts,
                            vertex_escape_distances, verts
                        )
                    triangles.append(tri_verts)
            else:
                n_skipped_configs += 1
            continue
        
        elif table_entry == 'INNER_VERTEX':
            # =================================================================
            # 6-EDGE CONFIGURATION: Inner vertex required
            # Per Bloomenthal & Ferguson 1995, Section 4, Figure 11:
            # Four regions meet → 4 face vertices + 1 inner vertex → 12 triangles
            # =================================================================
            n_6edge_configs += 1
            tets_contributing += 1
            
            new_tris, new_pos, new_bt, n_new = _generate_6_edge_triangles(
                local_edge_to_vertex, surface_vertices,
                next_new_vertex_index
            )
            
            if n_new > 0 and new_tris:
                # Append face vertices + inner vertex to our growing lists
                new_vertex_positions.append(new_pos)
                new_vertex_bts.append(new_bt)
                
                # Temporarily extend surface_vertices so orientation can access
                # the new face/inner vertex positions by index
                surface_vertices = np.concatenate(
                    [surface_vertices, new_pos], axis=0
                )
                next_new_vertex_index += n_new
                
                # Orient and append triangles
                for tri_verts in new_tris:
                    if vertex_escape_distances is not None:
                        tri_verts = _orient_triangle_by_escape_distance(
                            tri_verts, surface_vertices, tet_verts,
                            vertex_escape_distances, verts
                        )
                    triangles.append(tri_verts)
            else:
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
                    # Orient triangle normals consistently using helper functions
                    if vertex_mold_labels is not None and use_label_derived_cuts:
                        # PRIMARY surfaces: normal points from H1 towards H2
                        tri_verts = _orient_triangle_by_mold_labels(
                            tri_verts, surface_vertices, tet_verts,
                            vertex_mold_labels, verts
                        )
                    elif vertex_escape_distances is not None:
                        # SECONDARY surfaces: orient based on escape distances
                        tri_verts = _orient_triangle_by_escape_distance(
                            tri_verts, surface_vertices, tet_verts,
                            vertex_escape_distances, verts
                        )
                    
                    triangles.append(tri_verts)
    
    # Store vertex_to_edge now (before any vertex merging which would invalidate it)
    # The mapping will be invalidated by any vertex merge later
    result.vertex_to_edge = cut_edge_indices.copy()
    
    # =========================================================================
    # Update vertex_boundary_type for face/inner vertices created by 5/6-edge configs.
    # NOTE: surface_vertices was already extended incrementally in the loop above
    # (to allow orientation functions to access new vertex positions by index).
    # Only vertex_boundary_type needs to be extended here.
    # =========================================================================
    if new_vertex_bts:
        all_new_bts = np.concatenate(new_vertex_bts, axis=0)
        vertex_boundary_type = np.concatenate(
            [vertex_boundary_type, all_new_bts], axis=0
        )
        logger.info(
            f"Created {len(all_new_bts)} face/inner vertices "
            f"({n_5edge_configs} from 5-edge, {n_6edge_configs} from 6-edge configs)"
        )
    
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
    invalid_config_tets = 0  # Count tets with invalid configs (1, 2 edges)
    for config in sorted(config_counts.keys()):
        n_edges = bin(config).count('1')
        table_entry = MARCHING_TET_TABLE.get(config, [])
        if table_entry == 'FACE_VERTEX':
            status = f"-> FACE_VERTEX ({n_edges}-edge, face vertex generated)"
        elif table_entry == 'INNER_VERTEX':
            status = "-> INNER_VERTEX (6-edge, inner vertex generated)"
        elif table_entry:
            status = "-> triangles"
        else:
            status = "-> EMPTY (no triangles)"
            if n_edges in (1, 2):
                invalid_config_tets += config_counts[config]
        if config_counts[config] > 10:  # Only log significant configs
            logger.info(f"  Config {config:2d} ({config:06b}, {n_edges} edges): {config_counts[config]:5d} tets {status}")
    
    if n_5edge_configs > 0:
        logger.info(f"  {n_5edge_configs} tets resolved via 5-edge face vertex (Bloomenthal)")
    if n_6edge_configs > 0:
        logger.info(f"  {n_6edge_configs} tets resolved via 6-edge inner vertex (Bloomenthal)")
    if n_skipped_configs > 0:
        logger.warning(f"  {n_skipped_configs} tets had unresolvable configs (SKIPPED)")
    if invalid_config_tets > 0:
        logger.warning(f"  {invalid_config_tets} total tets with invalid/empty configs (1 or 2 edges)")
    
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
    
    # For secondary surfaces: build a vertex mask of primary cut edge endpoints.
    # Any secondary cut edge whose tet-mesh endpoint touches a primary cut
    # vertex is at the junction between secondary and primary surfaces.
    # The cut vertex there gets vertex_boundary_type=3 → reprojected to primary.
    primary_vertex_mask = None
    if cut_type == 'secondary' and tet_result.primary_cut_edges is not None:
        n_verts = len(tet_result.vertices)
        primary_vertex_mask = np.zeros(n_verts, dtype=bool)
        for vi, vj in tet_result.primary_cut_edges:
            primary_vertex_mask[vi] = True
            primary_vertex_mask[vj] = True
        n_primary_verts = np.sum(primary_vertex_mask)
        logger.info(f"Primary cut vertex mask: {n_primary_verts} vertices "
                   f"from {len(tet_result.primary_cut_edges)} primary cut edges")
        
        # Diagnostic: count how many secondary cut edges are adjacent to primary
        secondary_cut_edge_indices = np.where(cut_flags)[0]
        n_adjacent = 0
        for e_idx in secondary_cut_edge_indices:
            v0, v1 = tet_result.edges[e_idx]
            if primary_vertex_mask[v0] or primary_vertex_mask[v1]:
                n_adjacent += 1
        logger.info(f"Secondary cut edges adjacent to primary: {n_adjacent} / "
                   f"{len(secondary_cut_edge_indices)} "
                   f"({100*n_adjacent/max(1,len(secondary_cut_edge_indices)):.1f}%)")
    elif cut_type == 'secondary':
        logger.warning("No primary_cut_edges on tet_result — bt=3 junction detection disabled")
    
    return extract_parting_surface(
        vertices=tet_result.vertices,
        tetrahedra=tet_result.tetrahedra,
        edges=tet_result.edges,
        cut_edge_flags=cut_flags,
        tet_edge_indices=tet_result.tet_edge_indices,
        use_original_vertices=use_original_vertices,
        vertices_original=tet_result.vertices_original,
        boundary_labels=tet_result.boundary_labels,
        vertex_mold_labels=vertex_labels_for_cuts,
        vertex_escape_distances=vertex_escape_distances,
        use_label_derived_cuts=use_label_derived_cuts,
        is_secondary=(cut_type == 'secondary'),
        primary_cut_vertex_mask=primary_vertex_mask
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
    merge_threshold: float = 1e-8,
    is_secondary: bool = False
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
        is_secondary: If True, use part-priority merge (part > hull > interior)
                      to prevent secondary vertices from being re-projected
                      to the primary membrane instead of staying on the part.
    
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
                
                # For boundary type, choose the most appropriate type from merged vertices.
                #
                # PRIMARY surfaces: hull(1/2) > part(-1) > interior(0)
                #   Hull takes priority because the OUTER boundary should project to
                #   hull ∂H, not part M. If a part and hull vertex merge at a seam,
                #   the outer boundary projection is more critical.
                #
                # SECONDARY surfaces: part(-1) > primary_junction(3) > hull(1/2) > interior(0)
                #   Part takes priority because secondary part-boundary vertices must
                #   stay on the part surface M. Primary junction (3) next so that
                #   junction vertices re-project to the primary membrane. Hull last
                #   among boundary types.
                if has_boundary_type:
                    merged_types = [current_boundary_type[j] for j in neighbors]
                    if is_secondary:
                        # Secondary priority: part > primary_then_part > primary_junction > hull > interior
                        if -1 in merged_types:
                            new_boundary_type.append(-1)
                        elif 4 in merged_types:
                            new_boundary_type.append(4)
                        elif 3 in merged_types:
                            new_boundary_type.append(3)
                        elif 1 in merged_types or 2 in merged_types:
                            hull_types = [t for t in merged_types if t in (1, 2)]
                            new_boundary_type.append(hull_types[0])
                        else:
                            new_boundary_type.append(0)
                    else:
                        # Primary priority: hull > part > interior
                        if 1 in merged_types or 2 in merged_types:
                            hull_types = [t for t in merged_types if t in (1, 2)]
                            new_boundary_type.append(hull_types[0])
                        elif -1 in merged_types:
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
            
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
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
        if has_boundary_type and result.vertex_boundary_type is not None:
            bt = result.vertex_boundary_type
            if len(bt) != result.num_vertices:
                logger.error(f"vertex_boundary_type length mismatch after repair: "
                           f"{len(bt)} entries vs {result.num_vertices} vertices")
            else:
                logger.info(f"Preserved boundary types: {np.sum(bt == -1)} part, "
                           f"{np.sum(bt == 3)} primary junction, "
                           f"{np.sum(bt == 4)} primary→part, "
                           f"{np.sum(bt == 1) + np.sum(bt == 2)} hull, "
                           f"{np.sum(bt == 0)} interior")
        
        return result
        
    except Exception as e:
        logger.warning(f"Surface cleaning failed: {e}")
        return surface


def repair_parting_surface_with_part(
    surface: PartingSurfaceResult,
    part_mesh: trimesh.Trimesh,
    is_secondary: bool = False
) -> PartingSurfaceResult:
    """
    Clean parting surface (merge vertices, remove degenerates).
    
    Args:
        surface: PartingSurfaceResult to clean
        part_mesh: The original part mesh (reserved for future use)
        is_secondary: If True, use part-priority merge for secondary surfaces
    
    Returns:
        Cleaned PartingSurfaceResult
    """
    if surface.mesh is None:
        return surface
    
    # Basic cleanup: merge vertices and remove degenerate faces
    return repair_parting_surface(surface, merge_vertices=True, is_secondary=is_secondary)


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
    
    IMPORTANT: This function preserves vertex_boundary_type through the
    removal process by using face-level component labeling and vertex
    compaction (rather than split/concatenate which loses vertex metadata).
    
    Args:
        surface: PartingSurfaceResult to clean
        min_triangles: Minimum triangles to keep an island (default: 10)
        min_area_fraction: Minimum area fraction of total (default: 0.01 = 1%)
    
    Returns:
        Cleaned PartingSurfaceResult with small islands removed and
        vertex_boundary_type preserved
    """
    if surface.mesh is None or len(surface.mesh.faces) == 0:
        return surface
    
    try:
        mesh = surface.mesh
        faces = np.array(mesh.faces)
        vertices = np.array(mesh.vertices)
        n_faces = len(faces)
        n_verts = len(vertices)
        
        has_boundary_type = (
            surface.vertex_boundary_type is not None and
            len(surface.vertex_boundary_type) == n_verts
        )
        
        # Build face adjacency to find connected components via BFS
        # Two faces are adjacent if they share an edge
        from collections import deque
        
        # Build edge-to-face map
        edge_to_faces = {}
        for fi in range(n_faces):
            f = faces[fi]
            for e in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]:
                edge_key = (min(e), max(e))
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append(fi)
        
        # BFS to label connected components
        component_label = np.full(n_faces, -1, dtype=np.int32)
        current_label = 0
        
        for start_face in range(n_faces):
            if component_label[start_face] >= 0:
                continue
            
            # BFS from this face
            queue = deque([start_face])
            component_label[start_face] = current_label
            
            while queue:
                fi = queue.popleft()
                f = faces[fi]
                for e in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]:
                    edge_key = (min(e), max(e))
                    for neighbor_fi in edge_to_faces.get(edge_key, []):
                        if component_label[neighbor_fi] < 0:
                            component_label[neighbor_fi] = current_label
                            queue.append(neighbor_fi)
            
            current_label += 1
        
        n_components = current_label
        
        if n_components <= 1:
            # Only one component - nothing to remove
            return surface
        
        # Calculate per-component statistics
        # Compute face areas
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        
        total_area = np.sum(face_areas)
        min_area = total_area * min_area_fraction
        
        # Decide which components to keep
        keep_component = np.zeros(n_components, dtype=bool)
        removed_count = 0
        removed_triangles = 0
        removed_area = 0.0
        
        for ci in range(n_components):
            comp_mask = component_label == ci
            comp_n_tris = np.sum(comp_mask)
            comp_area = np.sum(face_areas[comp_mask])
            
            if comp_n_tris >= min_triangles or comp_area >= min_area:
                keep_component[ci] = True
            else:
                removed_count += 1
                removed_triangles += comp_n_tris
                removed_area += comp_area
        
        if removed_count == 0:
            # Nothing to remove
            return surface
        
        logger.info(f"Removing {removed_count} small islands "
                   f"({removed_triangles} triangles, {removed_area:.2f} area)")
        
        if not np.any(keep_component):
            logger.warning("All components were too small - keeping original mesh")
            return surface
        
        # Build mask of faces to keep
        faces_to_keep = keep_component[component_label]
        kept_faces = faces[faces_to_keep]
        
        # Compact vertices: keep only referenced vertices and remap face indices
        referenced = np.unique(kept_faces.ravel())
        old_to_new = np.full(n_verts, -1, dtype=np.int64)
        old_to_new[referenced] = np.arange(len(referenced))
        
        new_vertices = vertices[referenced]
        new_faces = old_to_new[kept_faces]
        
        # Preserve vertex_boundary_type through the compaction
        new_boundary_type = None
        if has_boundary_type:
            new_boundary_type = surface.vertex_boundary_type[referenced]
        
        combined = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
        
        # Build result with preserved vertex_boundary_type
        result = PartingSurfaceResult(
            mesh=combined,
            vertices=new_vertices,
            faces=new_faces,
            vertex_to_edge=None,  # Invalidated
            vertex_boundary_type=new_boundary_type,  # PRESERVED through compaction
            num_vertices=len(new_vertices),
            num_faces=len(new_faces),
            num_tets_processed=surface.num_tets_processed,
            num_tets_contributing=surface.num_tets_contributing,
            extraction_time_ms=surface.extraction_time_ms
        )
        
        kept_count = np.sum(keep_component)
        logger.info(f"Island removal complete: {n_components} -> {kept_count} components, "
                   f"{surface.num_faces} -> {result.num_faces} faces")
        if has_boundary_type and new_boundary_type is not None:
            logger.info(f"Preserved boundary types: {np.sum(new_boundary_type == -1)} part, "
                       f"{np.sum((new_boundary_type == 1) | (new_boundary_type == 2))} hull, "
                       f"{np.sum(new_boundary_type == 0)} interior")
        
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
    
    # Self-intersection detection results
    self_intersection_count: int = 0
    self_intersections_repaired: int = 0
    
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
    part_proximity_threshold: float = 0.1
) -> Tuple[bool, str]:
    """
    Classify a boundary edge as inner (part) or outer (hull).

    Classification priority:

    1. **Stale-hull override** — If either vertex has bt 1/2 but BOTH are
       close to the part surface (d_part < threshold) and at least as
       close to part as to hull (d_part <= d_hull + eps), the hull label
       is considered stale (moved during smoothing) and the edge is
       accepted as inner.
    2. **Hull reject** — If either vertex has bt 1/2 and the override
       does not apply → outer.
    3. **Both part** — Both bt == -1 → inner.
    4. **Mixed part/interior** — One bt == -1, other 0 → inner.
    5. **Distance-based** — Both bt == 0 (or no bt): compare d_part vs
       d_hull with 80 % margin; fall back to proximity threshold.

    A separate proximity rescue pass (STEP 2b in the caller) catches any
    remaining outer edges where both vertices and midpoint are genuinely
    close to the part.

    Args:
        v0, v1: Vertex indices.
        vertex_boundary_type: Array with -1=part, 0=interior, 1/2=hull.
        vert_to_part_dist: Pre-computed distances to part mesh.
        vert_to_hull_dist: Pre-computed distances to hull mesh (empty if
            no hull).
        part_proximity_threshold: Distance threshold for fallback
            classification.

    Returns:
        ``(is_inner, reason)`` — *is_inner* is True when the edge should
        receive a collar; *reason* is a human-readable tag for logging.
    """
    has_boundary_type = vertex_boundary_type is not None and len(vertex_boundary_type) > 0
    has_hull_dist = len(vert_to_hull_dist) > 0
    
    # Get boundary types
    bt0 = vertex_boundary_type[v0] if (has_boundary_type and v0 < len(vertex_boundary_type)) else None
    bt1 = vertex_boundary_type[v1] if (has_boundary_type and v1 < len(vertex_boundary_type)) else None
    
    # =========================================================================
    # STRICT CHECK: If we have vertex_boundary_type, use it definitively
    # =========================================================================
    
    if has_boundary_type:
        # -----------------------------------------------------------------
        # STALE LABEL OVERRIDE: After smoothing + re-projection, vertices
        # originally labeled hull (bt=1/2) may have been moved directly
        # ONTO the part surface.  Their boundary_type is stale.  Detect
        # this by checking actual distances: if BOTH vertices are very
        # close to the part surface (and significantly closer to part
        # than to hull when hull distances are available), override the
        # hull label and treat the edge as inner.
        #
        # Threshold: 10% of part_proximity_threshold gives a tight
        # "on-surface" band (typically < 0.5 mm).  We cap at 1.0 mm
        # to avoid false positives on large models.
        # -----------------------------------------------------------------
        if bt0 in (1, 2) or bt1 in (1, 2):
            d0_p = vert_to_part_dist.get(v0, 999)
            d1_p = vert_to_part_dist.get(v1, 999)
            on_surface_thresh = min(part_proximity_threshold, 1.0)

            both_on_part = d0_p < on_surface_thresh and d1_p < on_surface_thresh

            if both_on_part:
                # Extra check when hull distances available: vertices must
                # be at least as close to part as to hull.  Use <= with a
                # small epsilon so that vertices sitting exactly at the
                # part/hull intersection (d_part ≈ 0, d_hull ≈ 0) are
                # accepted rather than rejected.
                if has_hull_dist:
                    d0_h = vert_to_hull_dist.get(v0, 999)
                    d1_h = vert_to_hull_dist.get(v1, 999)
                    _eps = 1e-6
                    if d0_p <= d0_h + _eps and d1_p <= d1_h + _eps:
                        return True, (
                            f"stale_hull_override(bt0={bt0},bt1={bt1},"
                            f"d_part=[{d0_p:.4f},{d1_p:.4f}],"
                            f"d_hull=[{d0_h:.4f},{d1_h:.4f}])")
                    # Hull distances don't confirm → still reject
                    return False, (
                        f"hull_vertex_confirmed(bt0={bt0},bt1={bt1},"
                        f"d_part=[{d0_p:.4f},{d1_p:.4f}],"
                        f"d_hull=[{d0_h:.4f},{d1_h:.4f}])")
                else:
                    # No hull distances available — trust the distance-
                    # to-part measurement as ground truth.
                    return True, (
                        f"stale_hull_override_no_hull(bt0={bt0},bt1={bt1},"
                        f"d_part=[{d0_p:.4f},{d1_p:.4f}])")

            # Not on part surface → genuinely hull
            return False, f"hull_vertex(bt0={bt0},bt1={bt1})"
        
        # ACCEPT: Both vertices explicitly on part → inner
        if bt0 == -1 and bt1 == -1:
            return True, "both_part(bt=-1,-1)"
        
        # MIXED: One on part, one interior (0) or unknown.
        # Since this is a MESH BOUNDARY edge and one vertex is confirmed
        # on the part surface, the other vertex is almost certainly also
        # on the inner boundary (boundary_type may be 0 after smoothing).
        if bt0 == -1 or bt1 == -1:
            return True, f"mixed_part_interior(bt0={bt0},bt1={bt1})"
        
        # Both interior (0) or unknown — after smoothing, inner boundary
        # vertices may have boundary_type 0.  Fall through to distance-
        # based classification instead of rejecting outright.
    
    # =========================================================================
    # DISTANCE-BASED CLASSIFICATION (both-0 with boundary_type, or no
    # vertex_boundary_type at all)
    # =========================================================================
    
    # Get distances
    d0_part = vert_to_part_dist.get(v0, 999)
    d1_part = vert_to_part_dist.get(v1, 999)
    d0_hull = vert_to_hull_dist.get(v0, 999)
    d1_hull = vert_to_hull_dist.get(v1, 999)
    
    # PRIMARY: Hull vs part distance comparison (with margin)
    if has_hull_dist:
        # Require BOTH vertices to be significantly closer to part than hull
        # Use a margin factor to avoid edge cases
        margin = 0.8  # Part distance must be < 80% of hull distance
        v0_clearly_part = d0_part < (d0_hull * margin)
        v1_clearly_part = d1_part < (d1_hull * margin)
        
        if v0_clearly_part and v1_clearly_part:
            return True, f"distance_closer_to_part(d_part=[{d0_part:.4f},{d1_part:.4f}],d_hull=[{d0_hull:.4f},{d1_hull:.4f}])"
        
        # If either vertex is closer to hull, it's an outer edge
        if d0_part > d0_hull or d1_part > d1_hull:
            return False, f"closer_to_hull(d_part=[{d0_part:.4f},{d1_part:.4f}],d_hull=[{d0_hull:.4f},{d1_hull:.4f}])"
    
    # FALLBACK: Pure distance heuristic (very conservative)
    # Only classify as inner if BOTH vertices are very close to part
    is_inner = d0_part < part_proximity_threshold and d1_part < part_proximity_threshold
    if is_inner:
        return True, f"proximity_fallback(d_part=[{d0_part:.4f},{d1_part:.4f}],thresh={part_proximity_threshold:.4f})"
    else:
        return False, f"too_far_from_part(bt0={bt0},bt1={bt1},d_part=[{d0_part:.4f},{d1_part:.4f}],thresh={part_proximity_threshold:.4f})"


# =============================================================================
# CONSTANTS FOR IMPROVED VERTEX DETECTION
# =============================================================================

# Angle threshold for "sharp corner" detection (degrees)
# Vertices where boundary edges meet at angles sharper than this get fans
# Edge angle < 90° means the boundary turns by > 90°, which is significant
# REDUCED from 100° to 90° to allow more quads
SHARP_CORNER_ANGLE_THRESHOLD = 90.0

# Minimum divergence angle for enhanced fanning (degrees)
# When collar directions at a vertex spread by more than this, use enhanced fans
# This catches near-180° angles where collars point away from each other
COLLAR_DIVERGENCE_THRESHOLD = 120.0

# ============== ENHANCED DETECTION CONSTANTS ==============

# Face normal divergence threshold (degrees)
# If adjacent faces at a boundary vertex have normals differing by more than this,
# the vertex needs enhanced fanning to handle the surface twist
FACE_NORMAL_DIVERGENCE_THRESHOLD = 60.0

# Boundary chain curvature threshold (degrees)
# Measures how sharply the boundary "turns" at a vertex by looking at the direction
# change in the boundary chain (prev -> vi -> next)
BOUNDARY_CURVATURE_THRESHOLD = 120.0  # Must be >= COLLAR_DIVERGENCE_THRESHOLD to avoid contradictory classification

# Protrusion detection - convexity angle threshold (degrees)
# A vertex is considered "protruding" if the angle between the two boundary edges
# and the face normals suggests the vertex sticks out relative to neighbors
PROTRUSION_CONVEXITY_THRESHOLD = 30.0

# Distance ratio for protrusion detection
# If vertex is farther from part surface than avg of neighbors by this factor,
# it's likely at a protrusion tip
PROTRUSION_DISTANCE_RATIO = 1.5

# Collar prediction - predict if collar directions will spread when projected to part
# Uses the perpendicular to edge direction + face normal to estimate collar direction
COLLAR_SPREAD_PREDICTION_THRESHOLD = 90.0  # degrees


@dataclass 
class BoundaryVertexInfo:
    """Classification info for boundary vertices with enhanced detection.
    
    Vertex classification (mutually exclusive, in priority order):
    1. sharp_corners: 2 edges meeting at angle < 90° -> FANS
    2. high_valence: 3+ edges (complex junctions) -> FANS
    3. divergent_corners: 2 edges with angle > 120° (nearly straight) -> QUADS (upgradable)
    4. corners: Regular corners (angle 90-120°) -> QUADS (upgradable)
    5. endpoints: 1 edge (chain endpoints) -> NO COLLAR
    
    NOTE: Triangles with 2 disconnected boundary edges (same_face) are NO
    LONGER auto-classified as fans.  They use the same angle-based primary
    classification and enhanced concavity checks as all other 2-edge vertices.
    isolated_tips is retained but only populated by legacy/external callers.
    
    ENHANCED DETECTION (additive, can upgrade corners to fans):
    - face_normal_divergent: Adjacent faces have significantly different normals
    - boundary_curvature_high: Boundary chain turns sharply at this vertex
    - protrusion_tips: Vertices that protrude into space
    - collar_spread_predicted: Collar directions predicted to spread when projected
    
    Categories 1-2 and enhanced categories ALWAYS get fanned triangle collars.
    Categories 3-4 (divergent_corners, corners) use QUADS unless enhanced detection
    upgrades them via concavity / protrusion / curvature / collar-spread checks.
    Endpoints (5) do NOT get any collar triangles.
    """
    # QUAD categories (use quads unless upgraded by enhanced detection)
    corners: List[int]              # Regular corners (angle 90-120°)
    divergent_corners: List[int]    # Nearly straight (angle > 120°)
    endpoints: List[int]            # 1 boundary edge (chain end)
    
    # FAN categories (always use fans)
    isolated_tips: List[int]        # Legacy; no longer auto-populated
    sharp_corners: List[int]        # Edges meet at angle < 90° -> FANS
    high_valence: List[int]         # 3+ boundary edges (complex junctions)
    
    # ENHANCED DETECTION categories (upgrade corners to fans)
    face_normal_divergent: List[int]    # Adjacent faces have different normals
    boundary_curvature_high: List[int]  # High curvature along boundary chain
    protrusion_tips: List[int]          # Vertices that stick out (protrusions)
    collar_spread_predicted: List[int]  # Collar spread predicted geometrically
    
    # Edge data
    vertex_to_edges: Dict[int, List[Tuple]]  # vertex -> [(edge_key, this_v, other_v), ...]
    
    # Edge angle data for each vertex (for debugging/tuning)
    vertex_edge_angles: Dict[int, float]  # vertex -> angle between edges (degrees)
    
    # Additional diagnostic data
    vertex_face_normal_angles: Dict[int, float]  # vertex -> angle between adjacent face normals
    vertex_boundary_curvature: Dict[int, float]  # vertex -> boundary chain curvature (degrees)
    vertex_protrusion_score: Dict[int, float]    # vertex -> protrusion score (higher = more protruding)
    
    def __init__(self):
        """Initialize with empty lists."""
        self.corners = []
        self.divergent_corners = []
        self.endpoints = []
        self.isolated_tips = []
        self.sharp_corners = []
        self.high_valence = []
        # Enhanced detection categories
        self.face_normal_divergent = []
        self.boundary_curvature_high = []
        self.protrusion_tips = []
        self.collar_spread_predicted = []
        # Data dictionaries
        self.vertex_to_edges = {}
        self.vertex_edge_angles = {}
        self.vertex_face_normal_angles = {}
        self.vertex_boundary_curvature = {}
        self.vertex_protrusion_score = {}
    
    def get_all_fan_vertices(self) -> List[int]:
        """Get all vertices that should have fanned triangle collars.
        
        Returns combined list of fan-worthy vertices (deduped).
        Does NOT include corners or divergent_corners (they use quads).
        """
        fan_verts = set(self.isolated_tips)
        fan_verts.update(self.sharp_corners)
        fan_verts.update(self.high_valence)
        # Add enhanced detection categories
        fan_verts.update(self.face_normal_divergent)
        fan_verts.update(self.boundary_curvature_high)
        fan_verts.update(self.protrusion_tips)
        fan_verts.update(self.collar_spread_predicted)
        return list(fan_verts)
    
    def get_all_quad_vertices(self) -> List[int]:
        """Get all vertices that should use quad collars (not fans).
        
        Returns vertices in corners and divergent_corners that weren't
        upgraded to fans by enhanced detection.
        """
        fan_set = set(self.get_all_fan_vertices())
        quad_verts = []
        for vi in self.corners:
            if vi not in fan_set:
                quad_verts.append(vi)
        for vi in self.divergent_corners:
            if vi not in fan_set:
                quad_verts.append(vi)
        return quad_verts
    
    def get_total_classified(self) -> int:
        """Get total number of classified vertices (for diagnostic purposes)."""
        return (len(self.get_all_fan_vertices()) + 
                len(self.get_all_quad_vertices()) + 
                len(self.endpoints))
    
    def get_classification_summary(self) -> str:
        """Get a formatted summary of all classifications."""
        lines = [
            f"  sharp_corners (fan): {len(self.sharp_corners)}",
            f"  high_valence (fan): {len(self.high_valence)}",
            f"  corners (quad): {len(self.corners)}",
            f"  divergent_corners (quad): {len(self.divergent_corners)}",
            f"  face_normal_divergent: {len(self.face_normal_divergent)}",
            f"  boundary_curvature_high: {len(self.boundary_curvature_high)}",
            f"  protrusion_tips: {len(self.protrusion_tips)}",
            f"  collar_spread_predicted: {len(self.collar_spread_predicted)}",
            f"  endpoints (no collar): {len(self.endpoints)}",
        ]
        return "\n".join(lines)


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


def _compute_face_normal_divergence(
    vi: int,
    edges: List[Tuple],
    edge_to_face: Dict[Tuple[int, int], Tuple[int, int]],
    faces_arr: np.ndarray,
    vertices_arr: np.ndarray
) -> float:
    """
    Compute the divergence between face normals at adjacent faces meeting at a boundary vertex.
    
    Args:
        vi: The boundary vertex index
        edges: List of (edge_key, vi, other_v) tuples for edges at this vertex
        edge_to_face: Mapping from edge_key to (face_idx, third_vertex)
        faces_arr: Faces array for normal computation
        vertices_arr: Vertex positions array
    
    Returns:
        Maximum angle (degrees) between any two adjacent face normals at this vertex.
        Returns 0.0 if insufficient face data.
    """
    face_normals = []
    
    for edge_key, _, _ in edges:
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi, _ = face_info
        if fi >= len(faces_arr):
            continue
        
        # Compute face normal
        face = faces_arr[fi]
        v0, v1, v2 = vertices_arr[face[0]], vertices_arr[face[1]], vertices_arr[face[2]]
        e1, e2 = v1 - v0, v2 - v0
        normal = np.cross(e1, e2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-10:
            face_normals.append(normal / norm_len)
    
    if len(face_normals) < 2:
        return 0.0
    
    # Find maximum angle between any pair of face normals
    max_angle = 0.0
    for i in range(len(face_normals)):
        for j in range(i + 1, len(face_normals)):
            cos_angle = np.clip(np.dot(face_normals[i], face_normals[j]), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            max_angle = max(max_angle, angle)
    
    return max_angle


def _compute_boundary_chain_curvature(
    vi: int,
    edges: List[Tuple],
    vertices_arr: np.ndarray,
    adj_map: Dict[int, Set[int]]
) -> float:
    """
    Compute the curvature of the boundary chain at a vertex.
    
    Curvature is measured as the angle of the "turn" at this vertex,
    i.e., 180° minus the angle between (prev->vi) and (vi->next) vectors.
    
    A straight chain has 0° curvature, a 90° turn has 90° curvature.
    
    Args:
        vi: The boundary vertex index
        edges: List of (edge_key, vi, other_v) tuples for edges at this vertex
        vertices_arr: Vertex positions array
        adj_map: Adjacency map from vertex to set of connected boundary vertices
    
    Returns:
        Curvature in degrees (0-180). Higher = sharper turn.
        Returns 0.0 if vertex doesn't have exactly 2 neighbors.
    """
    neighbors = list(adj_map.get(vi, set()))
    
    if len(neighbors) != 2:
        return 0.0  # Not a chain vertex (endpoint or junction)
    
    prev_v, next_v = neighbors[0], neighbors[1]
    vi_pos = vertices_arr[vi]
    prev_pos = vertices_arr[prev_v]
    next_pos = vertices_arr[next_v]
    
    # Vectors along the chain
    vec_in = vi_pos - prev_pos  # from prev to vi
    vec_out = next_pos - vi_pos  # from vi to next
    
    len_in = np.linalg.norm(vec_in)
    len_out = np.linalg.norm(vec_out)
    
    if len_in < 1e-10 or len_out < 1e-10:
        return 0.0
    
    vec_in_unit = vec_in / len_in
    vec_out_unit = vec_out / len_out
    
    # Angle between incoming and outgoing directions
    cos_angle = np.clip(np.dot(vec_in_unit, vec_out_unit), -1.0, 1.0)
    chain_angle = np.degrees(np.arccos(cos_angle))  # 0-180
    
    # Curvature = 180 - chain_angle
    # Straight line: chain_angle=180, curvature=0
    # 90° turn: chain_angle=90, curvature=90
    curvature = 180.0 - chain_angle
    
    return curvature


def _compute_protrusion_score(
    vi: int,
    edges: List[Tuple],
    vertices_arr: np.ndarray,
    edge_to_face: Dict[Tuple[int, int], Tuple[int, int]],
    faces_arr: np.ndarray
) -> float:
    """
    Compute a score indicating how much a vertex "protrudes" into space.
    
    A high protrusion score indicates the vertex sticks out relative to its
    neighbors - typical at the tips of thin protrusions or fingers.
    
    The score is based on:
    1. How much farther vi is from the centroid of its neighbors
    2. Whether the edge vectors point "inward" from vi
    
    Args:
        vi: The boundary vertex index
        edges: List of (edge_key, vi, other_v) tuples for edges at this vertex
        vertices_arr: Vertex positions array
        edge_to_face: Mapping from edge_key to (face_idx, third_vertex)
        faces_arr: Faces array
    
    Returns:
        Protrusion score (0.0 to 1.0+). Higher = more protruding.
    """
    if len(edges) < 2:
        return 0.0
    
    vi_pos = vertices_arr[vi]
    
    # Collect neighbor positions
    neighbor_positions = []
    for edge_key, _, other_v in edges:
        neighbor_positions.append(vertices_arr[other_v])
    
    if len(neighbor_positions) < 2:
        return 0.0
    
    neighbor_positions = np.array(neighbor_positions)
    neighbor_centroid = np.mean(neighbor_positions, axis=0)
    
    # Vector from vi to neighbor centroid
    vi_to_centroid = neighbor_centroid - vi_pos
    dist_to_centroid = np.linalg.norm(vi_to_centroid)
    
    if dist_to_centroid < 1e-10:
        return 0.0
    
    # Average distance from vertices to their neighbors
    avg_neighbor_dist = np.mean([np.linalg.norm(p - vi_pos) for p in neighbor_positions])
    
    if avg_neighbor_dist < 1e-10:
        return 0.0
    
    # Protrusion score: how far vi is from centroid relative to edge lengths
    # High score = vi is "sticking out" from its neighbors
    protrusion_score = dist_to_centroid / avg_neighbor_dist
    
    # Also check if face normals point "outward" from vi
    # (indicates convex protrusion vs concave valley)
    outward_count = 0
    total_faces = 0
    
    for edge_key, _, _ in edges:
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi, third_v = face_info
        if fi >= len(faces_arr):
            continue
        
        # Compute face normal
        face = faces_arr[fi]
        v0, v1, v2 = vertices_arr[face[0]], vertices_arr[face[1]], vertices_arr[face[2]]
        e1, e2 = v1 - v0, v2 - v0
        normal = np.cross(e1, e2)
        norm_len = np.linalg.norm(normal)
        
        if norm_len > 1e-10:
            normal = normal / norm_len
            # Check if normal points "away" from centroid
            centroid_dir = vi_to_centroid / dist_to_centroid
            if np.dot(normal, centroid_dir) < 0:
                outward_count += 1
            total_faces += 1
    
    # Boost score if faces point outward (convex protrusion)
    if total_faces > 0 and outward_count > total_faces / 2:
        protrusion_score *= 1.5
    
    return protrusion_score


def _predict_collar_spread(
    vi: int,
    edges: List[Tuple],
    vertices_arr: np.ndarray,
    edge_to_face: Dict[Tuple[int, int], Tuple[int, int]],
    faces_arr: np.ndarray
) -> float:
    """
    Predict how much collar directions will spread when projected to the part surface.
    
    For each boundary edge at vi, the collar direction is perpendicular to the edge
    in the plane of the adjacent face. This function computes the angle between
    these predicted collar directions.
    
    Args:
        vi: The boundary vertex index
        edges: List of (edge_key, vi, other_v) tuples for edges at this vertex
        vertices_arr: Vertex positions array
        edge_to_face: Mapping from edge_key to (face_idx, third_vertex)
        faces_arr: Faces array
    
    Returns:
        Predicted collar spread angle (degrees). Higher = more spread.
    """
    collar_dirs = []
    
    for edge_key, _, other_v in edges:
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi, third_v = face_info
        if fi >= len(faces_arr):
            continue
        
        vi_pos = vertices_arr[vi]
        other_pos = vertices_arr[other_v]
        
        # Edge direction
        edge_vec = other_pos - vi_pos
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-10:
            continue
        edge_dir = edge_vec / edge_len
        
        # Compute face normal
        face = faces_arr[fi]
        v0, v1, v2 = vertices_arr[face[0]], vertices_arr[face[1]], vertices_arr[face[2]]
        e1, e2 = v1 - v0, v2 - v0
        face_normal = np.cross(e1, e2)
        fn_len = np.linalg.norm(face_normal)
        if fn_len < 1e-10:
            continue
        face_normal = face_normal / fn_len
        
        # Collar direction = perpendicular to edge in face plane
        # = cross(face_normal, edge_dir)
        collar_dir = np.cross(face_normal, edge_dir)
        collar_len = np.linalg.norm(collar_dir)
        if collar_len < 1e-10:
            continue
        collar_dir = collar_dir / collar_len
        
        # Orient collar direction away from third vertex (into the mold)
        third_pos = vertices_arr[third_v]
        to_third = third_pos - vi_pos
        if np.dot(collar_dir, to_third) > 0:
            collar_dir = -collar_dir
        
        collar_dirs.append(collar_dir)
    
    if len(collar_dirs) < 2:
        return 0.0
    
    # Find maximum angle between collar directions
    max_angle = 0.0
    for i in range(len(collar_dirs)):
        for j in range(i + 1, len(collar_dirs)):
            cos_angle = np.clip(np.dot(collar_dirs[i], collar_dirs[j]), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            max_angle = max(max_angle, angle)
    
    return max_angle


def _detect_boundary_vertices(
    inner_boundary_edges: List[Tuple[int, int]],
    edge_to_face: Dict[Tuple[int, int], Tuple[int, int]],
    vertices_arr: np.ndarray,
    restored_corner_positions: Optional[np.ndarray] = None,
    faces_arr: Optional[np.ndarray] = None
) -> BoundaryVertexInfo:
    """
    Detect and classify all boundary vertices with ENHANCED detection.
    
    This function uses multiple criteria to identify vertices that need 
    fanned triangle collars for proper mesh sealing:
    
    PRIMARY classification (mutually exclusive, in priority order):
    1. sharp_corners: 2 edges meeting at angle < 90° -> FANS
    2. high_valence: 3+ boundary edges (complex junctions) -> FANS
    3. divergent_corners: 2 edges with angle > 120° -> QUADS (upgradable)
    4. corners: Regular corners (2 edges, angle 90-120°) -> QUADS (upgradable)
    5. endpoints: 1 boundary edge (chain endpoints, no fans)
    
    NOTE: same_face (both edges from same triangle) is no longer treated
    specially.  These vertices go through the same angle + enhanced concavity
    checks as all other 2-edge vertices.
    
    ENHANCED DETECTION (additive, can flag vertices already in other categories):
    - face_normal_divergent: Adjacent faces have different normals (twist/fold)
    - boundary_curvature_high: Boundary chain turns sharply at this vertex
    - protrusion_tips: Vertex protrudes into space (thin feature tip)
    - collar_spread_predicted: Collar directions predicted to spread when projected
    
    Args:
        inner_boundary_edges: List of inner boundary edge tuples
        edge_to_face: Mapping from edge_key to (face_idx, third_vertex)
        vertices_arr: Vertex positions array
        restored_corner_positions: Optional positions of restored concave corners
        faces_arr: Optional faces array for enhanced detection (from membrane mesh)
    
    Returns:
        BoundaryVertexInfo with classified vertices and edge angle data
    """
    from collections import defaultdict
    
    result = BoundaryVertexInfo()
    
    # Build vertex -> edge mapping
    for v0, v1 in inner_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        for vi, other in [(v0, v1), (v1, v0)]:
            if vi not in result.vertex_to_edges:
                result.vertex_to_edges[vi] = []
            result.vertex_to_edges[vi].append((edge_key, vi, other))
    
    # Build adjacency map for boundary chain analysis
    boundary_adj = defaultdict(set)
    for v0, v1 in inner_boundary_edges:
        boundary_adj[v0].add(v1)
        boundary_adj[v1].add(v0)
    
    # Try to infer faces_arr from edge_to_face if not provided
    if faces_arr is None:
        # Build faces array from edge_to_face data
        face_indices = set()
        for edge_key, (fi, third_v) in edge_to_face.items():
            face_indices.add(fi)
        
        if face_indices:
            max_face = max(face_indices) + 1
            # We can't fully reconstruct faces without more info, 
            # but we can still use edge-based analysis
            faces_arr = None  # Leave as None; enhanced detection will handle gracefully
    
    # =========================================================================
    # STEP 1: Detect small and medium boundary loops
    # =========================================================================
    
    # =========================================================================
    # STEP 1: Primary classification for each vertex
    # =========================================================================
    
    for vi, edges in result.vertex_to_edges.items():
        n_edges = len(edges)
        
        # 1 edge: endpoint (no fan needed)
        if n_edges == 1:
            result.endpoints.append(vi)
            continue
        
        # 3+ edges: high-valence junction (always needs fans)
        if n_edges >= 3:
            result.high_valence.append(vi)
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
        
        # Primary classification based on angle only.
        # same_face is computed above but NOT used for special treatment;
        # triangles with two disconnected boundary edges go through the
        # same angle-based classification + enhanced concavity checks as
        # all other 2-edge vertices.
        #
        # Classification scheme:
        # - sharp_corners: angle < 90°    -> FANS
        # - divergent_corners: angle > 120° -> QUADS (upgradable)
        # - corners: angle 90-120°          -> QUADS (upgradable)
        
        if edge_angle < SHARP_CORNER_ANGLE_THRESHOLD:
            # Sharp corner (includes tips): angle < 90° means boundary
            # turns > 90° — quads fail at sharp turns, use fans.
            result.sharp_corners.append(vi)
            
        elif edge_angle > (180.0 - COLLAR_DIVERGENCE_THRESHOLD / 2):
            # Divergent: angle > 120°, boundary is relatively straight
            # Collars spread apart but quads should work here
            result.divergent_corners.append(vi)
            
        else:
            # Regular corner: angle 90-120° (moderate turn)
            # Quads work well for these moderate angles
            result.corners.append(vi)
    
    # =========================================================================
    # STEP 3: Enhanced detection for additional criteria (additive)
    # =========================================================================
    
    # Collect all currently classified vertices (to avoid duplicate work)
    already_fan_verts = (
        set(result.isolated_tips) | set(result.sharp_corners) |
        set(result.high_valence)
    )
    
    # Vertices in "corners" may need upgrade based on enhanced concavity
    # criteria (protrusion, curvature, collar_spread).  Divergent_corners
    # are NOT included here because boundary curvature == edge_angle for
    # 2-edge chain vertices, so the curvature check would always fire for
    # divergent_corners (>120°), making all of them fans and defeating
    # the purpose.  Divergent_corners can still be upgraded to fans via
    # face_normal_divergence (check 1), which applies to ALL non-fan verts.
    corner_verts_to_check = set(result.corners)
    
    # Also check endpoints in case enhanced detection reveals they need fans
    # (though typically endpoints remain as endpoints)
    
    for vi, edges in result.vertex_to_edges.items():
        n_edges = len(edges)
        
        # Skip endpoints and already-classified vertices for most checks
        if vi in result.endpoints:
            continue
        
        # =====================================================================
        # Enhanced Check 1: Face Normal Divergence
        # =====================================================================
        if n_edges >= 2 and faces_arr is not None:
            face_divergence = _compute_face_normal_divergence(
                vi, edges, edge_to_face, faces_arr, vertices_arr
            )
            result.vertex_face_normal_angles[vi] = face_divergence
            
            if face_divergence > FACE_NORMAL_DIVERGENCE_THRESHOLD:
                if vi not in already_fan_verts:
                    result.face_normal_divergent.append(vi)
                    already_fan_verts.add(vi)
        
        # =====================================================================
        # Enhanced Check 2: Boundary Chain Curvature
        # =====================================================================
        if n_edges == 2:  # Chain vertices only
            curvature = _compute_boundary_chain_curvature(
                vi, edges, vertices_arr, boundary_adj
            )
            result.vertex_boundary_curvature[vi] = curvature
            
            if curvature > BOUNDARY_CURVATURE_THRESHOLD:
                if vi not in already_fan_verts and vi in corner_verts_to_check:
                    # Upgrade from corner to boundary_curvature_high
                    result.boundary_curvature_high.append(vi)
                    already_fan_verts.add(vi)
        
        # =====================================================================
        # Enhanced Check 3: Protrusion Detection
        # =====================================================================
        if n_edges >= 2 and faces_arr is not None:
            protrusion_score = _compute_protrusion_score(
                vi, edges, vertices_arr, edge_to_face, faces_arr
            )
            result.vertex_protrusion_score[vi] = protrusion_score
            
            if protrusion_score > PROTRUSION_DISTANCE_RATIO:
                if vi not in already_fan_verts and vi in corner_verts_to_check:
                    result.protrusion_tips.append(vi)
                    already_fan_verts.add(vi)
        
        # =====================================================================
        # Enhanced Check 4: Collar Spread Prediction
        # =====================================================================
        if n_edges >= 2 and faces_arr is not None:
            collar_spread = _predict_collar_spread(
                vi, edges, vertices_arr, edge_to_face, faces_arr
            )
            
            if collar_spread > COLLAR_SPREAD_PREDICTION_THRESHOLD:
                if vi not in already_fan_verts and vi in corner_verts_to_check:
                    result.collar_spread_predicted.append(vi)
                    already_fan_verts.add(vi)
    
    # =========================================================================
    # STEP 4: Add restored concave corners (from smoothing step)
    # =========================================================================
    
    if restored_corner_positions is not None and len(restored_corner_positions) > 0:
        mesh_scale = np.linalg.norm(np.ptp(vertices_arr, axis=0))
        match_tolerance = mesh_scale * 0.001
        
        inner_verts = set(result.vertex_to_edges.keys())
        
        for pos in restored_corner_positions:
            dists = np.linalg.norm(vertices_arr - pos, axis=1)
            min_idx = np.argmin(dists)
            
            if dists[min_idx] < match_tolerance and min_idx in inner_verts and min_idx not in already_fan_verts:
                result.sharp_corners.append(min_idx)
                already_fan_verts.add(min_idx)
    
    # =========================================================================
    # STEP 5: Log classification results
    # =========================================================================
    
    logger.info(f"Boundary vertex classification (FANS):\n"
               f"  sharp_corners: {len(result.sharp_corners)}, "
               f"high_valence: {len(result.high_valence)}")
    
    logger.info(f"Boundary vertex classification (QUADS):\n"
               f"  corners: {len(result.corners)}, "
               f"divergent_corners: {len(result.divergent_corners)}, "
               f"endpoints: {len(result.endpoints)}")
    
    logger.info(f"Boundary vertex classification (ENHANCED):\n"
               f"  face_normal_div: {len(result.face_normal_divergent)}, "
               f"boundary_curv: {len(result.boundary_curvature_high)}, "
               f"protrusion: {len(result.protrusion_tips)}, "
               f"collar_spread: {len(result.collar_spread_predicted)}")
    
    total_fan = len(result.get_all_fan_vertices())
    total_quad = len(result.get_all_quad_vertices())
    logger.info(f"Total: {total_fan} fan vertices, {total_quad} quad vertices")
    
    return result


# Number of sample points along each edge for containment testing
_EDGE_CONTAINMENT_SAMPLES = 5
# t-values for sampling: avoid exact endpoints (0 and 1) to prevent
# surface-coincident false positives; sample interior of the edge
_EDGE_SAMPLE_T = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def _test_edges_inside_part(
    edges: List[Tuple[int, int]],
    verts_arr: np.ndarray,
    part_mesh: trimesh.Trimesh,
    n_samples: int = _EDGE_CONTAINMENT_SAMPLES
) -> np.ndarray:
    """Test whether edges lie fully inside the part mesh using multi-sample containment.

    For each edge, sample ``n_samples`` evenly-spaced interior points and test
    containment.  An edge is considered "outside" if ANY sample point is outside.
    This is much more robust than a single midpoint check — it catches edges
    that pass through concave regions of the part surface where both endpoints
    are inside but the connecting line segment exits the volume.

    Args:
        edges: List of (v0, v1) vertex index pairs.
        verts_arr: (N, 3) vertex positions array.
        part_mesh: The part mesh (must support ``contains``).
        n_samples: Number of interior sample points per edge (default 5).

    Returns:
        Boolean array of shape (len(edges),).  True = edge fully inside,
        False = at least one sample is outside the part.
    """
    if len(edges) == 0:
        return np.array([], dtype=bool)

    t_vals = _EDGE_SAMPLE_T[:n_samples]
    edge_arr = np.array(edges, dtype=np.int64)
    p0 = verts_arr[edge_arr[:, 0]]  # (E, 3)
    p1 = verts_arr[edge_arr[:, 1]]  # (E, 3)

    n_edges = len(edges)
    # Build all sample points in one flat array for a single batch call
    # Shape: (E * n_samples, 3)
    all_samples = np.empty((n_edges * len(t_vals), 3), dtype=np.float64)
    for ti, t in enumerate(t_vals):
        all_samples[ti * n_edges:(ti + 1) * n_edges] = p0 + t * (p1 - p0)

    try:
        inside_flat = part_mesh.contains(all_samples)  # (E * n_samples,)
    except Exception:
        # Fallback: single midpoint test (legacy behaviour)
        midpoints = (p0 + p1) / 2.0
        try:
            return part_mesh.contains(midpoints)
        except Exception:
            return np.ones(n_edges, dtype=bool)  # assume inside on failure

    # Reshape to (n_samples, E) and require ALL samples inside for the edge
    inside_matrix = inside_flat.reshape(len(t_vals), n_edges)
    return np.all(inside_matrix, axis=0)  # (E,)  True = fully inside


def _resolve_degenerate_collar(
    vi: int,
    vi_pos: np.ndarray,
    part_mesh: trimesh.Trimesh,
    part_face_normals: np.ndarray,
    collar_depth: float,
    vertex_normals: Dict[int, np.ndarray],
    boundary_adj: Dict[int, set],
    quad_vertex_collar: Dict[int, int],
    vertices: list,
    vertices_arr: np.ndarray,
) -> np.ndarray:
    """Try several fallback directions when the primary collar vertex is degenerate.

    Called when ``_create_collar_vertex`` returns a position coincident with
    the source vertex.  Tries, in order:

    1. Negative vertex normal (into the part).
    2. Direction copied from an adjacent quad collar vertex.
    3. Closest-point inward surface normal with reduced depth.
    4. Forced tiny offset (last resort).

    Args:
        vi: Vertex index.
        vi_pos: Vertex position on the membrane boundary.
        part_mesh: The part mesh.
        part_face_normals: Pre-computed face normals of part_mesh.
        collar_depth: Desired collar depth (mm).
        vertex_normals: Pre-computed averaged vertex normals from membrane.
        boundary_adj: Adjacency map among inner boundary vertices.
        quad_vertex_collar: Mapping from vertex → collar index (Phase 1).
        vertices: Mutable list of all vertex positions (collar verts appended here).
        vertices_arr: Original membrane vertex positions as ndarray.

    Returns:
        Position for the collar vertex (may still be degenerate in very
        rare pathological cases — caller should check).
    """
    # Fallback 1: push along negative vertex normal (into the part)
    vtx_normal = vertex_normals.get(vi)
    if vtx_normal is not None:
        into_dir = -vtx_normal
        into_dir = into_dir / (np.linalg.norm(into_dir) + 1e-10)
        candidate = _create_collar_vertex(
            vi_pos, part_mesh, part_face_normals, collar_depth, into_dir)
        if np.linalg.norm(candidate - vi_pos) >= 1e-8:
            return candidate

    # Fallback 2: borrow direction from an adjacent vertex that already
    # has a working collar (Phase 1 quad collars).
    for adj_vi in boundary_adj.get(vi, set()):
        adj_ci = quad_vertex_collar.get(adj_vi)
        if adj_ci is not None and adj_ci < len(vertices):
            adj_collar_pos = np.asarray(vertices[adj_ci], dtype=np.float64)
            adj_src_pos = vertices_arr[adj_vi] if adj_vi < len(vertices_arr) else vi_pos
            adj_dir = adj_collar_pos - adj_src_pos
            adj_dir_len = np.linalg.norm(adj_dir)
            if adj_dir_len > 1e-8:
                candidate = _create_collar_vertex(
                    vi_pos, part_mesh, part_face_normals,
                    collar_depth, adj_dir / adj_dir_len)
                if np.linalg.norm(candidate - vi_pos) >= 1e-8:
                    return candidate

    # Fallback 3: closest-point inward normal with half collar depth
    try:
        cp, _, cf = trimesh.proximity.closest_point(part_mesh, [vi_pos])
        pfi = cf[0]
        if pfi < len(part_face_normals):
            into = -part_face_normals[pfi]
            into = into / (np.linalg.norm(into) + 1e-10)
            candidate = vi_pos + collar_depth * 0.5 * into
            if np.linalg.norm(candidate - vi_pos) >= 1e-8:
                return candidate
    except Exception:
        pass

    # Fallback 4 (last resort): forced tiny offset.  Won't be inside the
    # part in general, but Step 4b snap-to-surface will correct it.
    logger.warning("Vertex %d: all collar fallbacks exhausted — forced offset", vi)
    return vi_pos + np.array([0.0, 0.0, -collar_depth * 0.1])


def _create_collar_vertex(
    vi_pos: np.ndarray,
    part_mesh: trimesh.Trimesh,
    part_face_normals: np.ndarray,
    collar_depth: float,
    collar_dir_hint: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create a collar vertex by offsetting in the membrane plane direction.

    Strategy:
    1. If a coplanar direction hint is available, compute the candidate
       position along that direction.
    2. Test whether the candidate is inside the part mesh.
    3. If outside, fall back to projecting onto the closest point of the
       part surface and then pushing inward along the inward surface normal
       by ``collar_depth``.  This guarantees the collar vertex ends up
       inside the part.

    The collar is a tiny extension (typically 0.2 mm) whose purpose is
    to create continuous geometry for CSG operations.

    Args:
        vi_pos: Position of the membrane vertex (should be ON part surface)
        part_mesh: The part mesh
        part_face_normals: Pre-computed part face normals
        collar_depth: How far to offset (mm)
        collar_dir_hint: Direction for collar (perpendicular to edge in membrane plane)

    Returns:
        Position of the collar vertex
    """
    candidate = None

    # --- Primary: use coplanar hint direction ---
    if collar_dir_hint is not None:
        collar_dir = collar_dir_hint / (np.linalg.norm(collar_dir_hint) + 1e-8)
        candidate = vi_pos + collar_depth * collar_dir

        # Quick containment check – if the candidate is inside the part,
        # accept it immediately (cheapest happy path).
        try:
            if part_mesh.contains([candidate])[0]:
                return candidate
        except Exception:
            # contains() can fail on degenerate meshes – accept the
            # candidate optimistically in that case.
            return candidate

        # Candidate is outside the part.  Try blending the hint direction
        # with the inward part-surface normal so the result has a stronger
        # inward component while preserving some coplanar alignment.
        try:
            closest_pts, _, closest_faces = trimesh.proximity.closest_point(
                part_mesh, [vi_pos])
            closest_face = closest_faces[0]
            if closest_face < len(part_face_normals):
                into_part = -part_face_normals[closest_face]
                into_part = into_part / (np.linalg.norm(into_part) + 1e-8)
                # 50/50 blend of hint and inward normal
                blended = 0.5 * collar_dir + 0.5 * into_part
                bl_len = np.linalg.norm(blended)
                if bl_len > 1e-8:
                    blended = blended / bl_len
                    blended_candidate = vi_pos + collar_depth * blended
                    if part_mesh.contains([blended_candidate])[0]:
                        return blended_candidate
        except Exception:
            pass

        # Blend didn't help — fall through to surface-projection fallback.

    # --- Fallback: project to closest part surface point + push inward ---
    try:
        closest_pts, _, closest_faces = trimesh.proximity.closest_point(
            part_mesh, [vi_pos])
        closest_pt = closest_pts[0]
        closest_face = closest_faces[0]

        if closest_face < len(part_face_normals):
            into_part = -part_face_normals[closest_face]
        else:
            into_part = np.array([0, 0, -1])

        into_part = into_part / (np.linalg.norm(into_part) + 1e-8)
        return closest_pt + collar_depth * into_part
    except Exception:
        # Last resort: return the hint-based candidate if we have one,
        # otherwise a naive offset.
        if candidate is not None:
            return candidate
        return vi_pos + collar_depth * np.array([0, 0, -1])


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
    wrap_around: bool = True
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
        wrap_around: If True, connect last collar back to first (for all-fan vertices).
                     If False, create open arc only (when quads handle the outer edges).
    
    Returns:
        (arc_vertices_created, triangles_created)
    """
    arc_verts_created = 0
    triangles_created = 0
    
    if len(collar_infos) < 2:
        return 0, 0
    
    # Determine iteration range based on wrap_around
    # wrap_around=True: create arcs between all consecutive pairs including last→first
    # wrap_around=False: create arcs between consecutive pairs WITHOUT wrapping
    if wrap_around:
        n_arcs = len(collar_infos)  # includes wrap from last to first
    else:
        n_arcs = len(collar_infos) - 1  # open arc, no wrap
    
    # Process each consecutive pair of collars
    for i in range(n_arcs):
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
        
        # -----------------------------------------------------------------
        # NO in-plane projection.  Collar vertices are placed coplanar
        # with their specific faces (using per-edge face normals in
        # Phase 2).  SLERP between the actual 3D directions so the arc
        # smoothly transitions between the two face planes rather than
        # being forced into an averaged vertex-normal plane that may
        # not match either face.
        # -----------------------------------------------------------------
        
        # Angle between (projected) collar directions
        cos_angle = np.clip(np.dot(dir_a_unit, dir_b_unit), -1, 1)
        angle = np.arccos(cos_angle)
        angle_deg = np.degrees(angle)
        
        # Fan plane normal — after projection both directions lie in the
        # ref_normal plane, so their cross product is ~parallel to ref_normal.
        fan_normal = np.cross(dir_a_unit, dir_b_unit)
        fan_len = np.linalg.norm(fan_normal)
        
        if fan_len < 1e-8:
            # Parallel / nearly parallel — single triangle
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
        
        # Dynamic subdivisions: ~1 subdivision per 30° of arc angle.
        # Small angles get fewer fan triangles, large angles get more.
        #   0-30°   → 1 (single triangle, no arc vertices)
        #  30-60°   → 2
        #  60-90°   → 3
        #  90-120°  → 4
        # 120-150°  → 5
        # 150-180°  → 6
        n_subs = max(1, int(np.ceil(angle_deg / 30.0)))
        
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
            
            # Use ref_normal for winding consistency with the membrane
            if np.dot(tri_n, ref_normal) >= 0:
                faces.append([vi, c_curr, c_next])
            else:
                faces.append([vi, c_next, c_curr])
            triangles_created += 1
    
    return arc_verts_created, triangles_created


# =============================================================================
# COLLAR SELF-INTERSECTION DETECTION AND REPAIR
# =============================================================================

# Maximum repair iterations for self-intersection resolution
_MAX_SELF_INTERSECTION_REPAIR_ITERATIONS = 5

# After repair, if this many or fewer pairs remain, accept the result
_ACCEPTABLE_REMAINING_INTERSECTIONS = 0


def _detect_collar_self_intersections(
    mesh: trimesh.Trimesh,
    collar_face_start: int
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Detect self-intersections involving collar faces.

    Only checks pairs where at least one face is a collar face
    (index >= collar_face_start). This is much cheaper than a full
    mesh self-intersection test since we only care about the collar
    region.

    Args:
        mesh: The combined membrane + collar mesh.
        collar_face_start: Index of the first collar face.

    Returns:
        (count, pairs) — number of intersecting pairs and the pair list.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)

    if collar_face_start >= n_faces:
        return 0, []

    # Build vertex → face adjacency for adjacency filtering
    vertex_to_faces: Dict[int, Set[int]] = {}
    for fi, face in enumerate(faces):
        for vi in face:
            vi_int = int(vi)
            if vi_int not in vertex_to_faces:
                vertex_to_faces[vi_int] = set()
            vertex_to_faces[vi_int].add(fi)

    # Face adjacency: faces sharing a vertex are adjacent (non-intersecting
    # by definition — they share geometry, they don't cross each other)
    def are_adjacent(fi: int, fj: int) -> bool:
        for vi in faces[fi]:
            if fj in vertex_to_faces.get(int(vi), set()):
                return True
        return False

    # Compute per-face bounding boxes
    tri_verts = vertices[faces]  # (F, 3, 3)
    bbox_min = tri_verts.min(axis=1)  # (F, 3)
    bbox_max = tri_verts.max(axis=1)  # (F, 3)

    intersecting_pairs: List[Tuple[int, int]] = []

    # Only test collar faces against ALL other faces
    collar_range = range(collar_face_start, n_faces)

    for fi in collar_range:
        fi_min = bbox_min[fi]
        fi_max = bbox_max[fi]

        # Test against every other face (both membrane and collar)
        for fj in range(n_faces):
            if fj == fi:
                continue
            # Only record each pair once (fi < fj or fi is the collar face)
            if fj >= collar_face_start and fj <= fi:
                continue  # will be tested when fj is the outer loop

            # Quick bbox overlap check
            if (fi_max[0] < bbox_min[fj][0] or bbox_max[fj][0] < fi_min[0] or
                fi_max[1] < bbox_min[fj][1] or bbox_max[fj][1] < fi_min[1] or
                fi_max[2] < bbox_min[fj][2] or bbox_max[fj][2] < fi_min[2]):
                continue

            # Skip adjacent faces
            if are_adjacent(fi, fj):
                continue

            # Full triangle-triangle intersection test
            tri_i = (vertices[faces[fi, 0]], vertices[faces[fi, 1]], vertices[faces[fi, 2]])
            tri_j = (vertices[faces[fj, 0]], vertices[faces[fj, 1]], vertices[faces[fj, 2]])

            if _triangles_intersect_fast(tri_i, tri_j):
                pair = (min(fi, fj), max(fi, fj))
                intersecting_pairs.append(pair)

    # Deduplicate (shouldn't be needed but defensive)
    intersecting_pairs = list(set(intersecting_pairs))

    if intersecting_pairs:
        logger.warning(f"Collar self-intersection: {len(intersecting_pairs)} "
                      f"intersecting pairs detected among "
                      f"{n_faces - collar_face_start} collar faces")
    else:
        logger.debug(f"Collar self-intersection check: clean "
                    f"({n_faces - collar_face_start} collar faces)")

    return len(intersecting_pairs), intersecting_pairs


def _detect_and_repair_collar_self_intersections(
    mesh: trimesh.Trimesh,
    collar_face_start: int,
    part_mesh: trimesh.Trimesh,
    collar_depth: float
) -> Tuple[trimesh.Trimesh, int, int]:
    """Detect self-intersecting collar triangles and repair them.

    Two-phase repair strategy (preferred order):

    **Phase A — Vertex shrink** (iterated up to
    ``_MAX_SELF_INTERSECTION_REPAIR_ITERATIONS``):
      For each collar vertex involved in a self-intersecting face, move it
      toward the nearest *membrane* vertex it is connected to.  This shrinks
      the collar triangle(s) at that vertex, reducing overlap.  Each
      iteration uses a progressively stronger shrink factor so that
      vertices converge to the membrane boundary.

    **Phase B — Face replacement** (if Phase A doesn't fully resolve):
      For each remaining self-intersecting collar face, replace it with a
      single collapsed triangle from the apex vertex (membrane vertex) to
      the two adjacent membrane-boundary vertices.  This eliminates the
      complex fan/quad geometry at that face while preserving coverage.
      Only faces that still self-intersect are replaced; all other collar
      geometry is kept.

    Args:
        mesh: The combined membrane + collar mesh.
        collar_face_start: Index of the first collar face.
        part_mesh: The part mesh for re-projection.
        collar_depth: Original collar depth (mm).

    Returns:
        (repaired_mesh, initial_count, repaired_count)
    """
    # Initial detection
    initial_count, pairs = _detect_collar_self_intersections(mesh, collar_face_start)

    if initial_count == 0:
        return mesh, 0, 0

    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # Identify collar-only vertices (not shared with original membrane)
    collar_faces_arr = faces[collar_face_start:]
    if len(collar_faces_arr) == 0:
        return mesh, initial_count, 0

    all_collar_verts = set(collar_faces_arr.flatten().tolist())
    orig_verts_in_collar = (
        set(faces[:collar_face_start].flatten().tolist())
        if collar_face_start > 0 else set()
    )
    collar_only_verts = all_collar_verts - orig_verts_in_collar

    # Build collar-vertex → connected membrane-vertex mapping
    # A collar-only vertex is connected to a membrane vertex if they share
    # a collar face.  Among those, pick the original membrane vertex
    # (index in orig_verts_in_collar) so we know where to shrink toward.
    collar_to_membrane: Dict[int, int] = {}
    for fi in range(collar_face_start, len(faces)):
        face = faces[fi]
        membrane_vis = [int(v) for v in face if int(v) in orig_verts_in_collar]
        collar_vis = [int(v) for v in face if int(v) in collar_only_verts]
        if membrane_vis:
            for cv in collar_vis:
                if cv not in collar_to_membrane:
                    collar_to_membrane[cv] = membrane_vis[0]

    remaining_count = initial_count
    total_repaired = 0

    # =====================================================================
    # Phase A: Iterative vertex shrink
    # =====================================================================
    for repair_iter in range(_MAX_SELF_INTERSECTION_REPAIR_ITERATIONS):
        if remaining_count <= _ACCEPTABLE_REMAINING_INTERSECTIONS:
            break

        bad_collar_faces = set()
        for fi, fj in pairs:
            if fi >= collar_face_start:
                bad_collar_faces.add(fi)
            if fj >= collar_face_start:
                bad_collar_faces.add(fj)

        verts_to_fix = set()
        for fi in bad_collar_faces:
            if fi < len(faces):
                for vi in faces[fi]:
                    vi_int = int(vi)
                    if vi_int in collar_only_verts:
                        verts_to_fix.add(vi_int)

        if not verts_to_fix:
            break

        # Shrink factor increases each iteration: 0.5, 0.7, 0.85, 0.93, 0.98
        shrink_factors = [0.5, 0.7, 0.85, 0.93, 0.98]
        shrink_factor = shrink_factors[min(repair_iter, len(shrink_factors) - 1)]
        fix_list = sorted(verts_to_fix)
        n_shrunk = 0

        for vi in fix_list:
            membrane_vi = collar_to_membrane.get(vi)
            if membrane_vi is not None:
                # Move collar vertex toward its membrane anchor
                src_pos = vertices[membrane_vi]
                collar_pos = vertices[vi]
                vertices[vi] = src_pos + (1.0 - shrink_factor) * (collar_pos - src_pos)
                n_shrunk += 1
            else:
                # No membrane anchor found — fall back to closest-point
                # re-projection with reduced depth
                try:
                    cp, _, cf = trimesh.proximity.closest_point(
                        part_mesh, [vertices[vi]])
                    reduced_depth = collar_depth * (0.3 / (repair_iter + 1))
                    pfi = cf[0]
                    if pfi < len(part_mesh.face_normals):
                        pn = part_mesh.face_normals[pfi]
                        pn_len = np.linalg.norm(pn)
                        if pn_len > 1e-10:
                            into = -pn / pn_len
                            vertices[vi] = cp[0] + reduced_depth * into
                        else:
                            vertices[vi] = cp[0]
                    else:
                        vertices[vi] = cp[0]
                    n_shrunk += 1
                except Exception:
                    pass

        repaired_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False)
        repaired_mesh.fix_normals()

        new_count, pairs = _detect_collar_self_intersections(
            repaired_mesh, collar_face_start)

        previously_remaining = remaining_count
        remaining_count = new_count
        iter_repaired = previously_remaining - remaining_count

        logger.info(
            f"Collar SI repair iter {repair_iter + 1}: shrunk "
            f"{n_shrunk} vertices (factor={shrink_factor:.1f}), "
            f"intersections {previously_remaining} → {remaining_count} "
            f"(fixed {iter_repaired})")

        if iter_repaired > 0:
            total_repaired += iter_repaired

        mesh = repaired_mesh
        vertices = mesh.vertices.copy()

    # =====================================================================
    # Phase B: Collapse remaining self-intersecting collar faces
    # =====================================================================
    # For each remaining self-intersecting collar face, aggressively
    # collapse collar vertices toward their membrane anchors.  This makes
    # the faces very small (near-degenerate) so they cannot self-intersect
    # with distant geometry.  Faces are preserved (not removed) to
    # avoid creating gaps.
    if remaining_count > 0:
        bad_face_set = set()
        for fi, fj in pairs:
            if fi >= collar_face_start:
                bad_face_set.add(fi)
            if fj >= collar_face_start:
                bad_face_set.add(fj)

        if bad_face_set:
            # Phase B-1: Aggressive vertex collapse (99% toward anchor)
            bad_collar_verts = set()
            for fi in bad_face_set:
                if fi < len(faces):
                    for vi in faces[fi]:
                        vi_int = int(vi)
                        if vi_int in collar_only_verts:
                            bad_collar_verts.add(vi_int)

            collapse_count = 0
            for vi in sorted(bad_collar_verts):
                membrane_vi = collar_to_membrane.get(vi)
                if membrane_vi is not None:
                    src_pos = vertices[membrane_vi]
                    collar_pos = vertices[vi]
                    # Move 99% toward anchor — face becomes tiny but persists
                    vertices[vi] = src_pos + 0.01 * (collar_pos - src_pos)
                    collapse_count += 1
                else:
                    # No anchor — project to closest point on part surface
                    try:
                        cp, _, _ = trimesh.proximity.closest_point(
                            part_mesh, [vertices[vi]])
                        vertices[vi] = cp[0]
                        collapse_count += 1
                    except Exception:
                        pass

            if collapse_count > 0:
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    process=False)
                mesh.fix_normals()

                new_count, pairs = _detect_collar_self_intersections(
                    mesh, collar_face_start)

                resolved_b1 = remaining_count - new_count
                if resolved_b1 > 0:
                    total_repaired += resolved_b1
                remaining_count = new_count

                logger.info(
                    f"Collar SI Phase B-1: collapsed {collapse_count} "
                    f"vertices (99%% toward anchor), "
                    f"intersections → {remaining_count} "
                    f"(fixed {resolved_b1})")

                vertices = mesh.vertices.copy()
                faces = mesh.faces.copy()

        # Phase B-2: If still intersecting, remove as last resort
        if remaining_count > 0:
            bad_face_set2 = set()
            for fi, fj in pairs:
                if fi >= collar_face_start:
                    bad_face_set2.add(fi)
                if fj >= collar_face_start:
                    bad_face_set2.add(fj)

            if bad_face_set2:
                faces_list = faces.tolist()
                for fi in sorted(bad_face_set2):
                    if fi < len(faces_list):
                        faces_list[fi] = None

                new_faces = [f for f in faces_list if f is not None]
                n_removed = len(faces_list) - len(new_faces)

                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=np.array(new_faces, dtype=np.int64),
                    process=False)
                mesh.remove_unreferenced_vertices()
                mesh.fix_normals()

                total_repaired += remaining_count
                if n_removed > 0:
                    logger.warning(
                        f"Collar SI Phase B-2: removed {n_removed} "
                        f"unrepairable collar faces as last resort")
                remaining_count = 0
            remaining_count = 0

    total_resolved = initial_count - remaining_count
    if total_resolved > 0:
        logger.info(
            f"Collar self-intersection repair: {initial_count} detected, "
            f"{total_resolved} resolved")

    return mesh, initial_count, total_resolved


# =============================================================================
# MAIN COLLAR EXTENSION FUNCTION
# =============================================================================

def create_robust_collar_extension(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    hull_mesh: Optional[trimesh.Trimesh] = None,
    vertex_boundary_type: Optional[np.ndarray] = None,
    collar_depth: float = 0.2,
    restored_corner_positions: Optional[np.ndarray] = None
) -> FloatingEdgeFillingResult:
    """
    Create robust collar extension for inner boundary edges.
    
    Extends the membrane toward the part surface along every inner
    (part-touching) boundary edge so that the final mold cavity has a
    clean seal against the cast object.
    
    Pipeline:
        1. Find boundary edges (edge_face_count == 1).
        2. Classify each edge as inner/outer via ``_classify_boundary_edge``
           with stale-hull-label override + a proximity rescue pass
           (STEP 2b) that promotes outer edges close to the part.
        3. Detect boundary-vertex topology (fans, quads, corners, tips).
        3b. Greedily downgrade adjacent fans to quads for strip continuity.
        4. Phase 1 — shared collar vertex per QUAD vertex (averaged perp).
           Phase 2 — per-edge collar per FAN vertex (face-normal perp),
           with automatic quad-linking and geometry-less fallback.
        4b. Snap-to-surface: push any collar vertex outside the part
            back onto the surface.
        5. Assemble quad strips along each inner boundary edge.
        6. Create fan triangles at fan vertices.
        6b. Iterative extension (up to ``MAX_COLLAR_EXTENSION_ITERATIONS``).
        6c. Final frontier inward push.
        7. Build result mesh.
    
    Args:
        membrane_mesh: The smoothed membrane mesh.
        part_mesh: The part mesh to connect to.
        hull_mesh: Optional hull mesh for boundary classification.
        vertex_boundary_type: Array with -1=part, 0=interior, 1/2=hull.
        collar_depth: How far to extend into part (mm).
        restored_corner_positions: Optional positions of restored concave
            corners (currently unused, reserved for future parity).
    
    Returns:
        ``FloatingEdgeFillingResult`` with the collared mesh.
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
    # DEFENSIVE CHECK: Validate vertex_boundary_type alignment
    # If smoothing used trimesh with process=True, merge_vertices() may have
    # renumbered vertex indices, making vertex_boundary_type misaligned.
    # =========================================================================
    if vertex_boundary_type is not None:
        if len(vertex_boundary_type) != n_orig_verts:
            logger.error(
                "vertex_boundary_type length (%d) != mesh vertex count (%d). "
                "This likely means vertex indices were renumbered during smoothing "
                "(trimesh merge_vertices). Collar classification will be unreliable. "
                "Falling back to distance-based classification only.",
                len(vertex_boundary_type), n_orig_verts
            )
            # Set to None so downstream code uses distance-based fallback
            vertex_boundary_type = None
    
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
        except Exception:
            logger.debug("Hull distance computation failed, using empty hull distances", exc_info=True)
    
    inner_boundary_edges = []
    outer_count = 0
    
    # Compute mesh-scale-relative threshold for the both-0 / distance-based
    # fallback in _classify_boundary_edge.  Use 5% of the mesh diagonal so
    # vertices that drifted slightly during smoothing are still captured.
    mesh_scale = np.linalg.norm(np.ptp(vertices_arr, axis=0))
    relative_proximity_threshold = max(mesh_scale * 0.05, 0.05)
    
    # Track classification reasons for debugging
    classification_reasons: Dict[str, int] = {}
    outer_edge_reasons: List[Tuple[Tuple[int, int], str]] = []
    
    for v0, v1 in boundary_edges:
        is_inner, reason = _classify_boundary_edge(
            v0, v1, vertex_boundary_type, vert_to_part_dist, vert_to_hull_dist,
            part_proximity_threshold=relative_proximity_threshold
        )
        # Bin by reason category (first word before '(')
        reason_key = reason.split('(')[0] if '(' in reason else reason
        classification_reasons[reason_key] = classification_reasons.get(reason_key, 0) + 1
        
        if is_inner:
            inner_boundary_edges.append((v0, v1))
        else:
            outer_count += 1
            outer_edge_reasons.append(((v0, v1), reason))
    
    logger.info(f"Inner boundary edges: {len(inner_boundary_edges)}, outer (hull): {outer_count} "
               f"(proximity threshold: {relative_proximity_threshold:.4f})")
    logger.info(f"Classification breakdown: {classification_reasons}")
    
    # =========================================================================
    # STEP 2b: Proximity-based inner edge rescue
    # =========================================================================
    # After label-based classification, rescue any OUTER-classified mesh
    # boundary edges where both vertices are genuinely close to the part
    # surface.  This catches edges at the part/hull intersection where
    # boundary_type labels are stale after smoothing and the hull distance
    # nearly equals the part distance (making the label check ambiguous).
    #
    # Criteria — promote an outer edge to inner if:
    #   1. BOTH vertices are within `rescue_threshold` of the part surface
    #   2. The edge midpoint is also within `rescue_threshold` of the part
    # This double-check avoids promoting edges that just graze the part.
    
    # Threshold: generous but bounded.  Use the same proximity threshold
    # that drives fallback classification, capped at 2.0 mm to avoid
    # rescuing edges that happen to be far above the part.
    rescue_threshold = min(relative_proximity_threshold, 2.0)
    
    _rescued_edges: List[Tuple[int, int]] = []
    _remaining_outer: List[Tuple[Tuple[int, int], str]] = []
    
    if outer_edge_reasons:
        # Compute midpoint distances for all outer edges in one batch
        _outer_edge_list = [edge for edge, _ in outer_edge_reasons]
        _outer_midpoints = np.array([
            (vertices_arr[v0] + vertices_arr[v1]) / 2.0
            for v0, v1 in _outer_edge_list
        ], dtype=np.float64)
        
        try:
            _, _mid_dists, _ = trimesh.proximity.closest_point(
                part_mesh, _outer_midpoints)
        except Exception:
            _mid_dists = np.full(len(_outer_edge_list), 999.0)
        
        for i, ((v0, v1), reason) in enumerate(outer_edge_reasons):
            d0 = vert_to_part_dist.get(v0, 999)
            d1 = vert_to_part_dist.get(v1, 999)
            d_mid = _mid_dists[i]
            
            if (d0 < rescue_threshold and
                    d1 < rescue_threshold and
                    d_mid < rescue_threshold):
                _rescued_edges.append((v0, v1))
            else:
                _remaining_outer.append(((v0, v1), reason))
        
        if _rescued_edges:
            inner_boundary_edges.extend(_rescued_edges)
            outer_count -= len(_rescued_edges)
            outer_edge_reasons = _remaining_outer
            
            logger.info(
                f"STEP 2b: Rescued {len(_rescued_edges)} outer edges as "
                f"inner (proximity to part < {rescue_threshold:.3f} mm). "
                f"Inner now: {len(inner_boundary_edges)}, "
                f"outer now: {outer_count}")
            
            # Log a sample of rescued edges for debugging
            for v0, v1 in _rescued_edges[:10]:
                d0 = vert_to_part_dist.get(v0, -1)
                d1 = vert_to_part_dist.get(v1, -1)
                bt0 = vertex_boundary_type[v0] if vertex_boundary_type is not None and v0 < len(vertex_boundary_type) else None
                bt1 = vertex_boundary_type[v1] if vertex_boundary_type is not None and v1 < len(vertex_boundary_type) else None
                logger.info(
                    f"  Rescued edge ({v0},{v1}): bt=({bt0},{bt1}), "
                    f"d_part=({d0:.4f},{d1:.4f})")
            if len(_rescued_edges) > 10:
                logger.info(f"  ... and {len(_rescued_edges) - 10} more")
        else:
            logger.info("STEP 2b: No outer edges eligible for proximity rescue")
    
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
        inner_boundary_edges, edge_to_face, vertices_arr, restored_corner_positions,
        faces_arr=faces_arr  # Pass faces array for enhanced detection
    )
    
    # Log summary - detailed breakdown is in _detect_boundary_vertices
    total_fan_verts = len(boundary_info.get_all_fan_vertices())
    logger.info(f"Boundary detection: {total_fan_verts} fan vertices, "
               f"{len(boundary_info.endpoints)} chain endpoints")
    
    # =========================================================================
    # STEP 3b: Downgrade adjacent fan vertices to quads
    # =========================================================================
    # When two adjacent boundary vertices are both fans, their fan triangles
    # can twist or overlap.  Keep the sharper vertex (smaller edge angle) as
    # a fan and downgrade the other to a quad.
    
    fan_category_sets = [
        boundary_info.sharp_corners,
        boundary_info.high_valence,
        boundary_info.isolated_tips,
        boundary_info.face_normal_divergent,
        boundary_info.boundary_curvature_high,
        boundary_info.protrusion_tips,
        boundary_info.collar_spread_predicted,
    ]
    current_fans = set()
    for fs in fan_category_sets:
        current_fans.update(fs)
    
    # Build adjacency from inner boundary edges
    boundary_adj: Dict[int, set] = {}
    for bv0, bv1 in inner_boundary_edges:
        boundary_adj.setdefault(bv0, set()).add(bv1)
        boundary_adj.setdefault(bv1, set()).add(bv0)
    
    # Greedy: iterate fans sorted by edge angle ascending (sharpest first).
    # The sharpest vertex keeps its fan; its neighbours are downgraded.
    fan_angles = []
    for vi in current_fans:
        angle = boundary_info.vertex_edge_angles.get(vi, 180.0)
        fan_angles.append((angle, vi))
    fan_angles.sort()  # sharpest (smallest angle) first
    
    kept_fans: set = set()
    downgraded_fans: set = set()
    for angle, vi in fan_angles:
        if vi in downgraded_fans:
            continue
        kept_fans.add(vi)
        # Downgrade all fan neighbours of this vertex
        for nb in boundary_adj.get(vi, set()):
            if nb in current_fans and nb not in kept_fans:
                downgraded_fans.add(nb)
    
    if downgraded_fans:
        logger.info(f"Downgraded {len(downgraded_fans)} adjacent fan vertices to quads: "
                   f"{sorted(downgraded_fans)}")
        # Remove downgraded vertices from every fan list
        for fs in fan_category_sets:
            for vi in list(fs):
                if vi in downgraded_fans:
                    fs.remove(vi)
        # Add them to corners (quad treatment)
        for vi in downgraded_fans:
            if vi not in boundary_info.corners:
                boundary_info.corners.append(vi)
    
    # =========================================================================
    # STEP 4: Create collar vertices for each edge endpoint
    # =========================================================================
    # IMPORTANT: Each collar vertex must be EDGE-SPECIFIC, not just vertex-specific.
    # At corner vertices where multiple edges meet, each edge contributes its own
    # collar direction (perpendicular to the edge in the face plane). This ensures
    # distinct collar positions that can be connected by fan triangles.
    
    membrane_face_normals = membrane_mesh.face_normals
    part_face_normals = part_mesh.face_normals
    
    # Map: (edge_key, vertex) -> collar_idx
    edge_endpoint_collar = {}
    
    # All fan vertices need edge-aware collar directions (separate collar per edge)
    all_fan_verts_set = set(boundary_info.get_all_fan_vertices())
    
    # PHASE 1: For QUAD vertices, create a SINGLE shared collar vertex
    # This ensures adjacent edges connect properly at non-corner vertices
    # We average the collar directions from all adjacent edges
    quad_vertex_collar = {}  # vi -> collar_idx (shared across all edges at this vertex)
    
    # Build vertex -> adjacent boundary edges mapping
    vertex_to_boundary_edges = {}
    for v0, v1 in inner_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        for vi in [v0, v1]:
            if vi not in vertex_to_boundary_edges:
                vertex_to_boundary_edges[vi] = []
            vertex_to_boundary_edges[vi].append((edge_key, v0, v1))
    
    # -----------------------------------------------------------------
    # Pre-compute VERTEX NORMALS for all inner boundary vertices.
    # Average of normals from ALL adjacent membrane faces, giving a
    # smooth surface normal at each vertex.  Using vertex normals
    # (instead of per-edge face normals) ensures collar directions at
    # each vertex lie in a consistent plane, preventing fold-in on
    # curved surfaces and keeping fans / quads coplanar.
    # -----------------------------------------------------------------
    vertex_to_faces_map = {}
    for fi in range(len(faces_arr)):
        for fv_idx in range(3):
            fv = int(faces_arr[fi][fv_idx])
            if fv not in vertex_to_faces_map:
                vertex_to_faces_map[fv] = []
            vertex_to_faces_map[fv].append(fi)
    
    vertex_normals = {}
    for vi in vertex_to_boundary_edges:
        adj_faces = vertex_to_faces_map.get(vi, [])
        normals = []
        for fi in adj_faces:
            if fi < len(membrane_face_normals):
                n = membrane_face_normals[fi]
                if np.linalg.norm(n) > 1e-10:
                    normals.append(n)
        if normals:
            avg_n = np.mean(normals, axis=0)
            avg_len = np.linalg.norm(avg_n)
            vertex_normals[vi] = avg_n / avg_len if avg_len > 1e-8 else np.array([0.0, 0.0, 1.0])
        else:
            vertex_normals[vi] = np.array([0.0, 0.0, 1.0])
    
    # Create shared collar vertices for quad (non-fan) vertices
    for vi, edge_list in vertex_to_boundary_edges.items():
        if vi in all_fan_verts_set:
            continue  # Fan vertices get per-edge collars in Phase 2
        
        vi_pos = vertices_arr[vi]
        vtx_normal = vertex_normals.get(vi, np.array([0.0, 0.0, 1.0]))
        
        # Average the collar directions from all adjacent edges
        collar_dirs = []
        for edge_key, v0, v1 in edge_list:
            face_info = edge_to_face.get(edge_key)
            if face_info is None:
                continue
            
            fi, _ = face_info
            
            v0_pos, v1_pos = vertices_arr[v0], vertices_arr[v1]
            edge_vec = v1_pos - v0_pos
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-8:
                continue
            edge_dir = edge_vec / edge_len
            
            # Perpendicular to edge using vertex normal (averaged from
            # all adjacent faces).  This keeps the collar in a consistent
            # plane across all edges at this vertex.
            perp = np.cross(vtx_normal, edge_dir)
            perp_len = np.linalg.norm(perp)
            if perp_len > 1e-8:
                perp = perp / perp_len
                # Orient outward
                face_verts = faces_arr[fi]
                third_v = [fv for fv in face_verts if fv != v0 and fv != v1]
                if third_v:
                    to_third = vertices_arr[third_v[0]] - vi_pos
                    if np.dot(perp, to_third) > 0:
                        perp = -perp
                collar_dirs.append(perp)
        
        if collar_dirs:
            # Average the directions
            avg_dir = np.mean(collar_dirs, axis=0)
            avg_len = np.linalg.norm(avg_dir)
            if avg_len > 1e-8:
                avg_dir = avg_dir / avg_len
            else:
                avg_dir = collar_dirs[0]  # Fallback to first direction
            
            collar_pt = _create_collar_vertex(vi_pos, part_mesh, part_face_normals, collar_depth, avg_dir)
        else:
            # Fallback: no valid perpendicular direction found for any
            # adjacent edge (all degenerate or parallel to vertex normal).
            # Use closest-point-on-part + inward push fallback.
            logger.debug("Phase 1: vertex %d has no valid collar direction, "
                         "using closest-point fallback", vi)
            collar_pt = _create_collar_vertex(vi_pos, part_mesh, part_face_normals, collar_depth, None)
        
        # Guard against degenerate collar (coincident with source vertex).
        # Phase 1 must always produce a collar vertex for strip continuity.
        if np.linalg.norm(collar_pt - vi_pos) < 1e-8:
            collar_pt = _resolve_degenerate_collar(
                vi, vi_pos, part_mesh, part_face_normals,
                collar_depth, vertex_normals, boundary_adj,
                quad_vertex_collar, vertices, vertices_arr)
        
        collar_idx = len(vertices)
        vertices.append(collar_pt.copy())
        quad_vertex_collar[vi] = collar_idx
    
    # PHASE 2: Create per-edge collar vertices for FAN vertices and link
    # quad vertices from Phase 1.  Fan vertices get a separate collar per
    # edge so the fan triangle arcs remain coplanar with each face.
    phase2_fan_created = 0
    phase2_skipped = 0
    
    for v0, v1 in inner_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        
        if edge_key not in edge_endpoint_collar:
            edge_endpoint_collar[edge_key] = {}
        
        face_info = edge_to_face.get(edge_key)
        fi = face_info[0] if face_info is not None else None
        
        v0_pos, v1_pos = vertices_arr[v0], vertices_arr[v1]
        edge_vec = v1_pos - v0_pos
        edge_len = np.linalg.norm(edge_vec)
        has_geometry = face_info is not None and edge_len >= 1e-8
        edge_dir = edge_vec / edge_len if has_geometry else None
        
        if not has_geometry:
            phase2_skipped += 1
        
        for vi in [v0, v1]:
            if vi in edge_endpoint_collar[edge_key]:
                continue
            
            # QUAD vertices: link to the shared collar from Phase 1
            if vi in quad_vertex_collar:
                edge_endpoint_collar[edge_key][vi] = quad_vertex_collar[vi]
                continue
            
            # FAN vertices: create per-edge collar using the adjacent
            # face normal (keeps collar coplanar with that face).
            vi_pos = vertices_arr[vi]
            collar_hint = None
            
            if has_geometry:
                face_n = (membrane_face_normals[fi]
                          if fi < len(membrane_face_normals)
                          else vertex_normals.get(vi, np.array([0.0, 0.0, 1.0])))
                face_n_len = np.linalg.norm(face_n)
                if face_n_len < 1e-10:
                    face_n = vertex_normals.get(vi, np.array([0.0, 0.0, 1.0]))
                else:
                    face_n = face_n / face_n_len
                
                perp = np.cross(face_n, edge_dir)
                perp_len = np.linalg.norm(perp)
                if perp_len > 1e-8:
                    perp = perp / perp_len
                    face_verts = faces_arr[fi]
                    third_v = [fv for fv in face_verts if fv != v0 and fv != v1]
                    if third_v:
                        to_third = vertices_arr[third_v[0]] - vi_pos
                        if np.dot(perp, to_third) > 0:
                            perp = -perp
                    collar_hint = perp
            
            collar_pt = _create_collar_vertex(
                vi_pos, part_mesh, part_face_normals, collar_depth, collar_hint)
            # If degenerate (coincident with source vertex), try fallbacks
            if np.linalg.norm(collar_pt - vi_pos) < 1e-8:
                collar_pt = _resolve_degenerate_collar(
                    vi, vi_pos, part_mesh, part_face_normals,
                    collar_depth, vertex_normals, boundary_adj,
                    quad_vertex_collar, vertices, vertices_arr)
                # Final degenerate check — only skip if truly unresolvable
                if np.linalg.norm(collar_pt - vi_pos) < 1e-8:
                    logger.warning(
                        "Phase 2: fan vertex %d still degenerate after "
                        "all fallbacks — skipping", vi)
                    continue
            collar_idx = len(vertices)
            vertices.append(collar_pt.copy())
            edge_endpoint_collar[edge_key][vi] = collar_idx
            phase2_fan_created += 1
    
    logger.info(f"Phase 2: created {phase2_fan_created} fan collars"
               f"{f', {phase2_skipped} edges lacked geometry' if phase2_skipped else ''}")
    logger.info(f"Created {len(vertices) - n_orig_verts} collar vertices total")

    # =========================================================================
    # STEP 4b: Snap-to-surface validation for collar vertices
    # =========================================================================
    # After all collar vertices are created (Phase 1 + Phase 2), verify that
    # each one is inside the part mesh.  Any that are outside get projected
    # to the closest point on the part surface and pushed inward along the
    # inward surface normal.  This catches cases where the hint-based
    # direction was tangential to the part and the collar vertex ended up
    # just outside the surface.
    n_collar_verts = len(vertices) - n_orig_verts
    if n_collar_verts > 0:
        collar_indices = list(range(n_orig_verts, len(vertices)))
        collar_positions = np.array([vertices[ci] for ci in collar_indices], dtype=np.float64)
        try:
            collar_inside = part_mesh.contains(collar_positions)
            n_outside = int(np.sum(~collar_inside))
            if n_outside > 0:
                outside_local = np.where(~collar_inside)[0]
                outside_positions = collar_positions[outside_local]
                cp_pts, _, cp_faces = trimesh.proximity.closest_point(
                    part_mesh, outside_positions)
                snap_push = collar_depth * 0.5  # push half collar_depth inward
                n_snapped = 0
                for oi, li in enumerate(outside_local):
                    ci = collar_indices[li]
                    pfi = cp_faces[oi]
                    if pfi < len(part_face_normals):
                        pn = part_face_normals[pfi]
                        pn_len = np.linalg.norm(pn)
                        if pn_len > 1e-10:
                            into = -pn / pn_len
                            vertices[ci] = (cp_pts[oi] + snap_push * into).copy()
                        else:
                            vertices[ci] = cp_pts[oi].copy()
                    else:
                        vertices[ci] = cp_pts[oi].copy()
                    n_snapped += 1
                logger.info(
                    f"Step 4b: snapped {n_snapped}/{n_collar_verts} outside "
                    "collar vertices to part surface + inward push")
            else:
                logger.info(
                    f"Step 4b: all {n_collar_verts} collar vertices "
                    "already inside part mesh")
        except Exception as ex:
            logger.warning(f"Step 4b: containment check failed ({ex}), "
                          "skipping snap-to-surface")

    # =========================================================================
    # STEP 5: Create quad collars for ALL boundary edges
    # =========================================================================
    # ALWAYS create quads for every inner boundary edge, regardless of fan status.
    # Quads handle the strip ALONG each edge: [v0, v1, c1, c0].
    # Fans handle the angular GAP BETWEEN edges at a vertex.
    # These are complementary, not overlapping — quads and fans share an edge
    # at the fan vertex ([vi, c_vi]) but create triangles on opposite sides.
    
    quads_created = 0
    both_fan_skipped = 0
    
    for v0, v1 in inner_boundary_edges:
        v0_is_fan = v0 in all_fan_verts_set
        v1_is_fan = v1 in all_fan_verts_set
        
        edge_key = (min(v0, v1), max(v0, v1))
        collar_data = edge_endpoint_collar.get(edge_key, {})
        c0, c1 = collar_data.get(v0), collar_data.get(v1)
        
        if c0 is None or c1 is None:
            continue
        
        v0_pos, v1_pos = vertices_arr[v0], vertices_arr[v1]
        c0_pos, c1_pos = np.array(vertices[c0]), np.array(vertices[c1])
        
        # Use average of vertex normals of both endpoints for consistent
        # reference normal across the quad
        vn0 = vertex_normals.get(v0, np.array([0.0, 0.0, 1.0]))
        vn1 = vertex_normals.get(v1, np.array([0.0, 0.0, 1.0]))
        avg_ref = vn0 + vn1
        avg_ref_len = np.linalg.norm(avg_ref)
        ref_normal = avg_ref / avg_ref_len if avg_ref_len > 1e-8 else np.array([0.0, 0.0, 1.0])
        
        if v0_is_fan and v1_is_fan:
            both_fan_skipped += 1
            # NOTE: We still create the quad below! Quads handle the strip
            # along the edge; fans handle the angular gap at each vertex.
        
        # Create full quad (2 triangles) to connect boundary edge to collar edge.
        # This applies to ALL edges: both-quad, mixed fan+quad, AND both-fan.
        #
        # Choose the diagonal that produces better-shaped triangles on
        # non-planar quads (pick the diagonal whose two triangle normals
        # are most consistent with ref_normal).
        diag1 = [([v0, v1, c1], v1_pos - v0_pos, c1_pos - v0_pos),
                 ([v0, c1, c0], c1_pos - v0_pos, c0_pos - v0_pos)]
        diag2 = [([v0, v1, c0], v1_pos - v0_pos, c0_pos - v0_pos),
                 ([v1, c1, c0], c1_pos - v1_pos, c0_pos - v1_pos)]
        
        best_diag = diag1
        best_score = -2.0
        for diag in (diag1, diag2):
            score = 0.0
            valid = True
            for tri_v, e1, e2 in diag:
                tn = np.cross(e1, e2)
                tn_len = np.linalg.norm(tn)
                if tn_len < 1e-10:
                    valid = False
                    break
                score += abs(np.dot(tn / tn_len, ref_normal))
            if valid and score > best_score:
                best_score = score
                best_diag = diag
        
        for tri_verts, ref_e1, ref_e2 in best_diag:
            tri_n = np.cross(ref_e1, ref_e2)
            if np.linalg.norm(tri_n) > 1e-10:
                if np.dot(tri_n, ref_normal) >= 0:
                    faces.append(tri_verts)
                else:
                    faces.append([tri_verts[0], tri_verts[2], tri_verts[1]])
        quads_created += 1
    
    logger.info(f"Created {quads_created} quad collars "
               f"({both_fan_skipped} edges with both fan endpoints)")
    
    edges_without_quads = len(inner_boundary_edges) - quads_created
    if edges_without_quads > 0:
        logger.warning(
            f"{edges_without_quads}/{len(inner_boundary_edges)} inner boundary "
            "edges did not receive collar quads (missing collar vertices)")
    
    # =========================================================================
    # STEP 5b: Patch edges that didn't receive quads
    # =========================================================================
    # When a collar vertex is missing for one endpoint of an edge, the quad
    # is skipped entirely.  Instead, create the collar vertex on-the-fly
    # using _resolve_degenerate_collar and add the quad.  This guarantees
    # every inner boundary edge gets collar geometry.
    
    if edges_without_quads > 0:
        patched_count = 0
        for v0, v1 in inner_boundary_edges:
            edge_key = (min(v0, v1), max(v0, v1))
            collar_data = edge_endpoint_collar.get(edge_key, {})
            c0, c1 = collar_data.get(v0), collar_data.get(v1)
            
            if c0 is not None and c1 is not None:
                continue  # already has a quad
            
            # Create missing collar vertex/vertices
            for vi, ci in [(v0, c0), (v1, c1)]:
                if ci is not None:
                    continue
                vi_pos = vertices_arr[vi]
                # Try the phase-1/phase-2 approach first
                collar_pt = _create_collar_vertex(
                    vi_pos, part_mesh, part_face_normals, collar_depth, None)
                if np.linalg.norm(collar_pt - vi_pos) < 1e-8:
                    collar_pt = _resolve_degenerate_collar(
                        vi, vi_pos, part_mesh, part_face_normals,
                        collar_depth, vertex_normals, boundary_adj,
                        quad_vertex_collar, vertices, vertices_arr)
                if np.linalg.norm(collar_pt - vi_pos) < 1e-8:
                    continue  # truly unresolvable (pathological)
                new_ci = len(vertices)
                vertices.append(collar_pt.copy())
                collar_data[vi] = new_ci
                edge_endpoint_collar[edge_key] = collar_data
            
            # Re-check after patching
            c0, c1 = collar_data.get(v0), collar_data.get(v1)
            if c0 is None or c1 is None:
                continue
            
            v0_pos, v1_pos = vertices_arr[v0], vertices_arr[v1]
            c0_pos = np.array(vertices[c0], dtype=np.float64)
            c1_pos = np.array(vertices[c1], dtype=np.float64)
            
            vn0 = vertex_normals.get(v0, np.array([0.0, 0.0, 1.0]))
            vn1 = vertex_normals.get(v1, np.array([0.0, 0.0, 1.0]))
            avg_ref = vn0 + vn1
            avg_ref_len = np.linalg.norm(avg_ref)
            ref_normal = avg_ref / avg_ref_len if avg_ref_len > 1e-8 else np.array([0.0, 0.0, 1.0])
            
            # Create the quad (same logic as Step 5)
            diag1 = [([v0, v1, c1], v1_pos - v0_pos, c1_pos - v0_pos),
                     ([v0, c1, c0], c1_pos - v0_pos, c0_pos - v0_pos)]
            diag2 = [([v0, v1, c0], v1_pos - v0_pos, c0_pos - v0_pos),
                     ([v1, c1, c0], c1_pos - v1_pos, c0_pos - v1_pos)]
            
            best_diag = diag1
            best_score = -2.0
            for diag in (diag1, diag2):
                score = 0.0
                valid = True
                for tri_v, e1, e2 in diag:
                    tn = np.cross(e1, e2)
                    tn_len = np.linalg.norm(tn)
                    if tn_len < 1e-10:
                        valid = False
                        break
                    score += abs(np.dot(tn / tn_len, ref_normal))
                if valid and score > best_score:
                    best_score = score
                    best_diag = diag
            
            for tri_verts, ref_e1, ref_e2 in best_diag:
                tri_n = np.cross(ref_e1, ref_e2)
                if np.linalg.norm(tri_n) > 1e-10:
                    if np.dot(tri_n, ref_normal) >= 0:
                        faces.append(tri_verts)
                    else:
                        faces.append([tri_verts[0], tri_verts[2], tri_verts[1]])
            patched_count += 1
        
        if patched_count > 0:
            logger.info(f"Step 5b: patched {patched_count} edges that were "
                       f"missing collar quads")
    
    # =========================================================================
    # STEP 6: Create fan triangles at all fan vertices
    # =========================================================================
    
    total_arc_verts = 0
    total_fan_tris = 0
    
    # Use the pre-computed fan vertices
    all_fan_vertices = list(all_fan_verts_set)
    
    # Detailed breakdown for logging
    n_angle_based = (len(boundary_info.isolated_tips) +
                     len(boundary_info.sharp_corners) + len(boundary_info.high_valence))
    n_enhanced = (len(boundary_info.face_normal_divergent) + len(boundary_info.boundary_curvature_high) +
                  len(boundary_info.protrusion_tips) + len(boundary_info.collar_spread_predicted))
    
    logger.info(f"Creating fans for {len(all_fan_vertices)} vertices "
               f"(angle-based: {n_angle_based}, enhanced: {n_enhanced})")
    
    for vi in all_fan_vertices:
        vi_pos = vertices_arr[vi]
        edges_at_vi = boundary_info.vertex_to_edges.get(vi, [])
        
        if len(edges_at_vi) < 2:
            continue
        
        # Collect collar info for each edge
        collar_infos = []
        
        for edge_key, _, other_v in edges_at_vi:
            collar_data = edge_endpoint_collar.get(edge_key, {})
            c_idx = collar_data.get(vi)
            if c_idx is not None:
                collar_infos.append((c_idx, np.array(vertices[c_idx])))
        
        if len(collar_infos) < 2:
            continue
        
        # Use pre-computed vertex normal (averaged from ALL adjacent
        # face normals).  This is the same normal used for collar
        # direction computation, ensuring fan arcs stay in the same
        # plane as the collar vertices.
        ref_normal = vertex_normals.get(vi, np.array([0.0, 0.0, 1.0]))
        
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
            """Compute angle of collar vertex around vi for sorting."""
            _, pos = ci
            d = pos - vi_pos
            d_proj = d - np.dot(d, ref_normal) * ref_normal
            d_len = np.linalg.norm(d_proj)
            if d_len < 1e-8:
                return 0.0
            d_proj = d_proj / d_len
            return np.arctan2(np.dot(d_proj, y_axis), np.dot(d_proj, x_axis))
        
        collar_infos.sort(key=collar_angle)
        
        # Determine if we should wrap around:
        # - 3+ edges (high-valence): wrap to close the full loop of fan sectors.
        # - 2 edges: open arc only (wrap_around=False). Quads handle the edge
        #   strips, and the fan fills just the angular gap between the two
        #   collar directions. No wrapping needed.
        wrap_around = len(edges_at_vi) >= 3
        
        # Create fan triangles
        arc_v, fan_t = _create_fan_triangles(
            vi, vi_pos, collar_infos, ref_normal, vertices, faces,
            part_mesh, part_face_normals, collar_depth,
            wrap_around=wrap_around
        )
        total_arc_verts += arc_v
        total_fan_tris += fan_t
    
    logger.info(f"Created {total_fan_tris} fan triangles ({total_arc_verts} arc vertices)")
    
    # =========================================================================
    # STEP 6b: Iterative collar extension for boundary edges outside part mesh
    # =========================================================================
    # After collar creation, check every boundary edge of the collar region.
    # Uses MULTI-SAMPLE containment testing: sample 5 interior points along
    # each edge and require ALL to be inside the part.  This catches edges
    # that pass through concave surface regions where both endpoints are
    # inside but the connecting segment exits the volume.
    # For edges that fail, first re-project vertices to the closest part
    # surface, then extend with another collar quad.
    # Repeat up to 5 iterations or until all collar boundary edges are inside.
    #
    # Adjacent extension quads share new vertices so the extension strip
    # forms a continuous boundary rather than disconnected quads.
    #
    # Frontier tracking: only edges where BOTH vertices fall in the current
    # frontier range are candidates, preventing exponential growth.
    
    MAX_COLLAR_EXTENSION_ITERATIONS = 5
    total_extension_quads = 0
    total_reprojected = 0
    _frontier_start = n_orig_verts  # first iteration: all collar verts
    
    for _ext_iter in range(MAX_COLLAR_EXTENSION_ITERATIONS):
        _iter_vert_start = len(vertices)  # track new verts created this iter
        
        # ----- Build edge-face counts for current mesh state -----
        _efc: Dict[Tuple[int, int], int] = {}
        _etf: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for _fi, _face in enumerate(faces):
            for _ei in range(3):
                _va, _vb = int(_face[_ei]), int(_face[(_ei + 1) % 3])
                _vc = int(_face[(_ei + 2) % 3])
                _ek = (min(_va, _vb), max(_va, _vb))
                _efc[_ek] = _efc.get(_ek, 0) + 1
                _etf[_ek] = (_fi, _vc)

        # Frontier collar boundary edges: boundary (count==1) with BOTH
        # vertices in the current frontier range (>= _frontier_start).
        _collar_be = [
            (va, vb) for (va, vb), cnt in _efc.items()
            if cnt == 1 and va >= _frontier_start and vb >= _frontier_start
        ]

        if not _collar_be:
            logger.info(f"Collar extension iter {_ext_iter + 1}: "
                        "no frontier collar boundary edges found")
            break

        # ----- Multi-sample containment test along each edge -----
        _verts_arr = np.array(vertices, dtype=np.float64)

        try:
            _inside = _test_edges_inside_part(
                _collar_be, _verts_arr, part_mesh)
        except Exception as _ex:
            logger.warning(f"Collar extension iter {_ext_iter + 1}: "
                           f"edge containment test failed ({_ex}), "
                           "stopping iterations")
            break

        _outside_indices = [
            i for i in range(len(_collar_be)) if not _inside[i]
        ]

        if not _outside_indices:
            logger.info(f"Collar extension iter {_ext_iter + 1}: all "
                        f"{len(_collar_be)} collar boundary edges "
                        "are fully inside the part mesh")
            break

        # =================================================================
        # SUB-STEP A: Re-project outside frontier vertices to part surface
        # =================================================================
        # Collect the unique frontier vertices that belong to outside edges
        _outside_edges_pre = [_collar_be[i] for i in _outside_indices]
        _reproj_verts = set()
        for _va, _vb in _outside_edges_pre:
            _reproj_verts.add(_va)
            _reproj_verts.add(_vb)
        _reproj_list = sorted(_reproj_verts)

        if _reproj_list:
            _reproj_positions = _verts_arr[_reproj_list]
            try:
                # Check which vertices are themselves outside the part
                _v_inside = part_mesh.contains(_reproj_positions)
                _v_outside_mask = ~_v_inside

                if np.any(_v_outside_mask):
                    _outside_positions = _reproj_positions[_v_outside_mask]
                    _closest_pts, _, _ = trimesh.proximity.closest_point(
                        part_mesh, _outside_positions)

                    # Write re-projected positions back into vertices list
                    _oi = 0
                    _n_reprojected = 0
                    for _ri, _vi in enumerate(_reproj_list):
                        if _v_outside_mask[_ri]:
                            vertices[_vi] = _closest_pts[_oi].copy()
                            _oi += 1
                            _n_reprojected += 1

                    if _n_reprojected > 0:
                        total_reprojected += _n_reprojected
                        logger.info(
                            f"Collar extension iter {_ext_iter + 1}: "
                            f"re-projected {_n_reprojected} vertices "
                            "to part surface")

                        # Rebuild vertex array after re-projection
                        _verts_arr = np.array(vertices, dtype=np.float64)
            except Exception as _ex:
                logger.warning(
                    f"Collar extension iter {_ext_iter + 1}: "
                    f"re-projection failed ({_ex}), proceeding without")

        # Always extend all originally-outside edges after re-projection
        _outside_edges = [_collar_be[i] for i in _outside_indices]
        logger.info(f"Collar extension iter {_ext_iter + 1}: "
                    f"{len(_outside_edges)}/{len(_collar_be)} collar "
                    "boundary edges outside part — extending")

        # =================================================================
        # SUB-STEP B: Build adjacency among outside edges for vertex sharing
        # =================================================================
        # Two outside edges are *adjacent* when they share a vertex.
        # When extending, adjacent edges share the new vertex at their
        # common endpoint so the extension strip is continuous.
        #
        # _vert_new_idx[vi] = index of the NEW vertex created for
        # frontier vertex vi during this iteration (shared across edges)
        _vert_new_idx: Dict[int, int] = {}

        # Pre-compute per-edge extension geometry (direction + face normal)
        _edge_ext_info: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for _va, _vb in _outside_edges:
            _va_pos = _verts_arr[_va]
            _vb_pos = _verts_arr[_vb]
            _ek = (min(_va, _vb), max(_va, _vb))
            _finfo = _etf.get(_ek)
            if _finfo is None:
                continue
            _fi, _vc = _finfo
            _vc_pos = _verts_arr[_vc]

            _edge_vec = _vb_pos - _va_pos
            _edge_len = np.linalg.norm(_edge_vec)
            if _edge_len < 1e-10:
                continue
            _edge_dir = _edge_vec / _edge_len

            _fverts = faces[_fi]
            _p0 = _verts_arr[int(_fverts[0])]
            _p1 = _verts_arr[int(_fverts[1])]
            _p2 = _verts_arr[int(_fverts[2])]
            _fn = np.cross(_p1 - _p0, _p2 - _p0)
            _fn_len = np.linalg.norm(_fn)
            if _fn_len < 1e-10:
                continue
            _fn = _fn / _fn_len

            _perp = np.cross(_fn, _edge_dir)
            _perp_len = np.linalg.norm(_perp)
            if _perp_len < 1e-10:
                continue
            _perp = _perp / _perp_len
            _to_third = _vc_pos - (_va_pos + _vb_pos) / 2.0
            if np.dot(_perp, _to_third) > 0:
                _perp = -_perp

            _edge_ext_info[_ek] = (_perp, _fn)

        # Fallback for edges whose geometry computation failed (degenerate
        # face, zero-length edge, or edge parallel to face normal).  Use
        # the inward part-surface normal at the edge midpoint so the
        # extension still makes progress into the part volume.
        _n_geom_fallback = 0
        for _va, _vb in _outside_edges:
            _ek = (min(_va, _vb), max(_va, _vb))
            if _ek in _edge_ext_info:
                continue
            _mid = (_verts_arr[_va] + _verts_arr[_vb]) / 2.0
            try:
                _, _, _fb_faces = trimesh.proximity.closest_point(
                    part_mesh, [_mid])
                _fb_fi = _fb_faces[0]
                if _fb_fi < len(part_mesh.face_normals):
                    _into = -part_mesh.face_normals[_fb_fi]
                    _into_len = np.linalg.norm(_into)
                    if _into_len > 1e-10:
                        _into = _into / _into_len
                        _edge_ext_info[_ek] = (_into, _into)
                        _n_geom_fallback += 1
                        continue
            except Exception:
                pass
            # Absolute last resort — straight down
            _fb_dir = np.array([0.0, 0.0, -1.0])
            _edge_ext_info[_ek] = (_fb_dir, _fb_dir)
            _n_geom_fallback += 1

        if _n_geom_fallback > 0:
            logger.info(
                f"Collar extension iter {_ext_iter + 1}: "
                f"{_n_geom_fallback} edges used inward-normal "
                "fallback direction (degenerate geometry)")

        # Build adjacency: vertex -> list of outside edge keys that use it
        _vert_to_outside_edges: Dict[int, List[Tuple[int, int]]] = {}
        for _va, _vb in _outside_edges:
            _ek = (min(_va, _vb), max(_va, _vb))
            if _ek not in _edge_ext_info:
                continue
            _vert_to_outside_edges.setdefault(_va, []).append(_ek)
            _vert_to_outside_edges.setdefault(_vb, []).append(_ek)

        # =================================================================
        # SUB-STEP C: Create new vertices (shared at adjacency points)
        # =================================================================
        # For each frontier vertex that appears in outside edges, compute
        # one new position.  If the vertex is shared by multiple edges,
        # average their perpendicular directions for a smooth join.
        # Skip creating vertices that would be nearly coincident with
        # their source (degenerate after re-projection).
        #
        # FIXED INCREMENT EXTENSION: Each iteration extends by exactly
        # collar_depth (mm).  Up to MAX_COLLAR_EXTENSION_ITERATIONS (5)
        # iterations ensures the collar reaches the part interior in
        # small, controlled steps without overshooting.
        #
        # ROBUSTNESS: Blend edge-perpendicular with the inward part
        # surface normal at each frontier vertex.  This ensures the
        # extension always has a component pointing INTO the part,
        # even when the membrane meets the surface at a shallow angle
        # (where the edge perpendicular would be tangential).
        _frontier_into_part: Dict[int, np.ndarray] = {}
        _all_frontier_vi = list(_vert_to_outside_edges.keys())
        if _all_frontier_vi:
            _frontier_positions = _verts_arr[_all_frontier_vi]
            try:
                _, _, _f_closest_faces = trimesh.proximity.closest_point(
                    part_mesh, _frontier_positions)
                for _fi_idx, _vi in enumerate(_all_frontier_vi):
                    _pfi = _f_closest_faces[_fi_idx]
                    if _pfi < len(part_mesh.face_normals):
                        _pn = part_mesh.face_normals[_pfi]
                        _pn_len = np.linalg.norm(_pn)
                        if _pn_len > 1e-10:
                            _frontier_into_part[_vi] = -_pn / _pn_len
            except Exception:
                logger.debug("Part normal blending failed, using edge perp only", exc_info=True)

        # Pre-compute blended directions for all frontier vertices.
        _vi_directions: Dict[int, np.ndarray] = {}  # vi -> unit direction
        for _vi, _ek_list in _vert_to_outside_edges.items():
            if _vi in _vert_new_idx:
                continue

            # Average perpendicular directions from all adjacent edges
            _perp_sum = np.zeros(3)
            _count = 0
            for _ek in _ek_list:
                if _ek in _edge_ext_info:
                    _perp_sum += _edge_ext_info[_ek][0]
                    _count += 1
            if _count == 0:
                # No edge has valid extension geometry at this vertex.
                # Fall back to inward part-surface normal so the
                # extension still progresses into the part volume.
                _into_dir = _frontier_into_part.get(_vi)
                if _into_dir is not None:
                    _vi_directions[_vi] = _into_dir
                continue

            if _count == 1:
                _avg_perp = _perp_sum  # already unit length from above
            else:
                _avg_len = np.linalg.norm(_perp_sum)
                if _avg_len < 1e-10:
                    # Opposing directions cancel out — use first edge's perp
                    _avg_perp = _edge_ext_info[_ek_list[0]][0]
                else:
                    _avg_perp = _perp_sum / _avg_len

            # Blend with inward part normal for robust penetration
            _into_dir = _frontier_into_part.get(_vi)
            if _into_dir is not None:
                _alignment = np.dot(_avg_perp, _into_dir)
                # If edge perp is mostly tangent or points away from
                # part interior, add significant into-part bias
                _blend = 0.6 if _alignment < 0.3 else 0.2
                _blended = _avg_perp * (1 - _blend) + _into_dir * _blend
                _bl_len = np.linalg.norm(_blended)
                if _bl_len > 1e-10:
                    _avg_perp = _blended / _bl_len

            _vi_directions[_vi] = _avg_perp

        # -----------------------------------------------------------------
        # Create new vertex positions using fixed collar_depth increments.
        # Each iteration extends by exactly collar_depth — small steps
        # repeated up to MAX_COLLAR_EXTENSION_ITERATIONS (5) times.
        # -----------------------------------------------------------------
        _ray_vi_list = [vi for vi in _vi_directions if vi not in _vert_new_idx]
        for _vi in _ray_vi_list:
            if _vi in _vert_new_idx:
                continue
            _vi_pos = _verts_arr[_vi]
            _dir = _vi_directions[_vi]
            _new_pos = _vi_pos + collar_depth * _dir

            # Skip if new position is coincident with source (post re-proj)
            if np.linalg.norm(_new_pos - _vi_pos) < 1e-6:
                continue

            _new_idx = len(vertices)
            vertices.append(_new_pos.copy())
            _vert_new_idx[_vi] = _new_idx

        # -----------------------------------------------------------------
        # Post-extension containment snap: ensure all new extension
        # vertices are inside the part mesh.  Any that overshot (e.g.
        # through a thin wall) get snapped to the closest part surface
        # point and pushed inward by collar_depth.
        # -----------------------------------------------------------------
        if _vert_new_idx:
            _new_vi_list = sorted(_vert_new_idx.values())
            _new_positions = np.array(
                [vertices[ni] for ni in _new_vi_list], dtype=np.float64)
            try:
                _new_inside = part_mesh.contains(_new_positions)
                _new_outside_count = int(np.sum(~_new_inside))
                if _new_outside_count > 0:
                    _nout_local = np.where(~_new_inside)[0]
                    _nout_pos = _new_positions[_nout_local]
                    _snap_pts, _, _snap_faces = trimesh.proximity.closest_point(
                        part_mesh, _nout_pos)
                    _snap_push = collar_depth * 0.5
                    for _si, _li in enumerate(_nout_local):
                        _ni = _new_vi_list[_li]
                        _sf = _snap_faces[_si]
                        if _sf < len(part_mesh.face_normals):
                            _sn = part_mesh.face_normals[_sf]
                            _sn_len = np.linalg.norm(_sn)
                            if _sn_len > 1e-10:
                                _into = -_sn / _sn_len
                                vertices[_ni] = (
                                    _snap_pts[_si] + _snap_push * _into
                                ).copy()
                            else:
                                vertices[_ni] = _snap_pts[_si].copy()
                        else:
                            vertices[_ni] = _snap_pts[_si].copy()
                    logger.info(
                        f"Collar extension iter {_ext_iter + 1}: "
                        f"snapped {_new_outside_count}/{len(_new_vi_list)} "
                        "overshot extension vertices to part surface")
            except Exception as _snap_ex:
                logger.debug(
                    f"Extension snap failed ({_snap_ex})", exc_info=True)

        # =================================================================
        # SUB-STEP D: Create quad faces for each outside edge
        # =================================================================
        _extended = 0
        _skipped_degenerate = 0
        for _va, _vb in _outside_edges:
            _ek = (min(_va, _vb), max(_va, _vb))
            if _ek not in _edge_ext_info:
                continue
            _, _fn = _edge_ext_info[_ek]

            _new_va_idx = _vert_new_idx.get(_va)
            _new_vb_idx = _vert_new_idx.get(_vb)
            if _new_va_idx is None or _new_vb_idx is None:
                continue

            # Skip if new vertices are nearly coincident with source
            # (happens when re-projection already brought them to part
            # surface and extension collapses back to the same spot)
            _va_pos = np.array(vertices[_va], dtype=np.float64)
            _vb_pos = np.array(vertices[_vb], dtype=np.float64)
            _new_va_pos = np.array(vertices[_new_va_idx], dtype=np.float64)
            _new_vb_pos = np.array(vertices[_new_vb_idx], dtype=np.float64)
            _disp_a = np.linalg.norm(_new_va_pos - _va_pos)
            _disp_b = np.linalg.norm(_new_vb_pos - _vb_pos)
            if _disp_a < 1e-6 and _disp_b < 1e-6:
                _skipped_degenerate += 1
                continue

            # Build two triangles for the quad [va, vb, new_vb, new_va]
            _tri1 = [_va, _vb, _new_vb_idx]
            _tri2 = [_va, _new_vb_idx, _new_va_idx]
            for _tri in (_tri1, _tri2):
                _tp0 = np.array(vertices[_tri[0]], dtype=np.float64)
                _tp1 = np.array(vertices[_tri[1]], dtype=np.float64)
                _tp2 = np.array(vertices[_tri[2]], dtype=np.float64)
                _tn = np.cross(_tp1 - _tp0, _tp2 - _tp0)
                if np.dot(_tn, _fn) < 0:
                    _tri[1], _tri[2] = _tri[2], _tri[1]
            faces.append(_tri1)
            faces.append(_tri2)
            _extended += 1

        # =================================================================
        # SUB-STEP E: Join adjacent extension quads with stitching triangles
        # =================================================================
        # Where two outside edges share a vertex, the extension quads
        # already share the new vertex (from SUB-STEP C).  However, when
        # frontiers have side gaps (e.g., one edge was inside, neighbors
        # were outside), fill the lateral gap with a triangle connecting
        # the outside edge's new side vertex to the shared vertex.
        # This is handled implicitly by vertex sharing above — no extra
        # stitching faces needed because both quads reference the same
        # new vertex index, making the strip manifold-continuous.

        total_extension_quads += _extended
        # Advance frontier to the vertices created in this iteration
        _frontier_start = _iter_vert_start
        logger.info(f"Collar extension iter {_ext_iter + 1}: "
                    f"extended {_extended} edges "
                    f"({len(_vert_new_idx)} shared new vertices"
                    f"{f', {_skipped_degenerate} degenerate skipped' if _skipped_degenerate else ''})")
        if _extended == 0:
            break

    if total_extension_quads > 0 or total_reprojected > 0:
        logger.info(f"Iterative collar extension: {total_extension_quads} "
                    f"extension quads, {total_reprojected} vertex "
                    "re-projections total")

    # =========================================================================
    # STEP 6c: Final frontier validation and inward push
    # =========================================================================
    # Safety net: after all iterative extensions, ensure the outermost collar
    # boundary vertices are inside the part mesh.  Any that are still outside
    # get re-projected to the closest part surface point, then pushed inward
    # along the negative surface normal by a small fraction of collar_depth.
    # Also logs a diagnostic of how many final collar boundary edge midpoints
    # are inside vs outside the part mesh.

    if len(vertices) > n_orig_verts:
        _final_verts_arr = np.array(vertices, dtype=np.float64)

        # Build edge-face counts for final mesh state
        _final_efc: Dict[Tuple[int, int], int] = {}
        for _fi, _face in enumerate(faces):
            for _ei in range(3):
                _va, _vb = int(_face[_ei]), int(_face[(_ei + 1) % 3])
                _ek = (min(_va, _vb), max(_va, _vb))
                _final_efc[_ek] = _final_efc.get(_ek, 0) + 1

        # Collar boundary vertices: boundary (count==1) and index >= n_orig_verts
        _collar_bv = set()
        for (_va, _vb), cnt in _final_efc.items():
            if cnt == 1:
                if _va >= n_orig_verts:
                    _collar_bv.add(_va)
                if _vb >= n_orig_verts:
                    _collar_bv.add(_vb)

        if _collar_bv:
            _cbv_list = sorted(_collar_bv)
            _cbv_positions = _final_verts_arr[_cbv_list]
            try:
                _cbv_inside = part_mesh.contains(_cbv_positions)
                _cbv_outside_vi = [
                    _cbv_list[i] for i in range(len(_cbv_list))
                    if not _cbv_inside[i]
                ]

                if _cbv_outside_vi:
                    _cbv_out_pos = _final_verts_arr[_cbv_outside_vi]
                    _cp, _, _cf = trimesh.proximity.closest_point(
                        part_mesh, _cbv_out_pos)
                    _push_depth = collar_depth * 0.15
                    _n_pushed = 0
                    for _oi, _vi in enumerate(_cbv_outside_vi):
                        _pfi = _cf[_oi]
                        if _pfi < len(part_mesh.face_normals):
                            _pn = part_mesh.face_normals[_pfi]
                            _pn_len = np.linalg.norm(_pn)
                            if _pn_len > 1e-10:
                                _into = -_pn / _pn_len
                                vertices[_vi] = (
                                    _cp[_oi] + _push_depth * _into
                                ).copy()
                            else:
                                vertices[_vi] = _cp[_oi].copy()
                        else:
                            vertices[_vi] = _cp[_oi].copy()
                        _n_pushed += 1

                    logger.info(
                        f"Final frontier: pushed {_n_pushed}/"
                        f"{len(_cbv_list)} outside collar boundary "
                        "vertices inward")
                else:
                    logger.info(
                        f"Final frontier: all {len(_cbv_list)} collar "
                        "boundary vertices inside part")

                # Diagnostic: multi-sample containment of collar boundary edges
                _collar_be_final = [
                    (va, vb) for (va, vb), cnt in _final_efc.items()
                    if cnt == 1
                    and (va >= n_orig_verts or vb >= n_orig_verts)
                ]
                if _collar_be_final:
                    _diag_verts = np.array(vertices, dtype=np.float64)
                    try:
                        _fmid_inside = _test_edges_inside_part(
                            _collar_be_final, _diag_verts, part_mesh)
                        _n_in = int(np.sum(_fmid_inside))
                        _n_out = len(_collar_be_final) - _n_in
                        if _n_out > 0:
                            logger.warning(
                                f"Final collar validation: {_n_out}/"
                                f"{len(_collar_be_final)} edges have "
                                "samples outside part mesh")
                        else:
                            logger.info(
                                f"Final collar validation: all "
                                f"{len(_collar_be_final)} edges fully "
                                "inside part mesh")
                    except Exception as _diag_ex:
                        logger.warning(
                            f"Final collar validation diagnostic failed: "
                            f"{_diag_ex}")
            except Exception as _ex:
                logger.warning(
                    f"Final frontier validation failed: {_ex}")

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
    
    # =========================================================================
    # STEP 7b: Detect and repair collar self-intersections
    # =========================================================================
    # Collar triangles (fans, quads, extension strips) can self-intersect on
    # complex geometry — especially at sharp corners where fan arcs from
    # adjacent vertices overlap, or where iterative extension quads twist.
    #
    # Strategy:
    #   1. Detect self-intersecting pairs among collar faces only
    #      (faces with index >= n_orig_faces).
    #   2. For each intersecting pair, identify the collar vertices involved
    #      and re-project them to the closest point on the part surface.
    #   3. After re-projection, re-check. If self-intersections remain,
    #      remove the offending faces entirely (safe because collar faces
    #      are cosmetic extensions, not structural membrane geometry).
    #   4. Record counts in result for UI reporting.
    
    if result.mesh is not None and result.fill_triangles_added > 0:
        result.mesh, result.self_intersection_count, result.self_intersections_repaired = \
            _detect_and_repair_collar_self_intersections(
                result.mesh, n_orig_faces, part_mesh, collar_depth)
        result.vertices = np.array(result.mesh.vertices)
        result.faces = np.array(result.mesh.faces)
        # Update fill count in case faces were removed
        result.fill_triangles_added = len(result.faces) - n_orig_faces
    
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Collar extension complete: {len(inner_boundary_edges)} edges, "
               f"{quads_created} quads, {total_fan_tris} fan triangles, "
               f"{result.new_vertices_added} new vertices"
               f"{f', {result.self_intersections_repaired} self-intersections repaired' if result.self_intersections_repaired > 0 else ''}"
               f" in {result.processing_time_ms:.1f}ms")
    
    return result


def fill_floating_edge_gaps_parting_surface(
    surface: 'PartingSurfaceResult',
    part_mesh: trimesh.Trimesh,
    tolerance_fraction: float = FLOATING_EDGE_TOLERANCE_FRACTION,
    min_tolerance: float = FLOATING_EDGE_MIN_TOLERANCE,
    collar_depth: float = 0.2
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
        collar_depth: How far to extend INTO the part (mm), default 0.2mm
    
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
