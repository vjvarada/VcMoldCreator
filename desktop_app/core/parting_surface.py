"""
Parting Surface Extraction using Marching Tetrahedra

This module extracts a continuous parting surface mesh from a tetrahedral mesh
where certain edges have been marked as "cut" (either primary or secondary cuts).

Algorithm: Marching Tetrahedra (paper-based vertex classification)
- Each tetrahedron has 6 edges
- Cut edges are derived from vertex labels (H1 vs H2)
- Only 3-edge and 4-edge configurations generate triangles
- 5-edge and 6-edge configs are skipped to avoid self-intersections

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
from typing import Dict, List, Tuple, Optional
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


def _build_marching_tet_table() -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Build the marching tetrahedra lookup table.
    
    Based on the paper's vertex-classification approach, we only generate
    triangles for configurations that arise from clean vertex separation
    (H1 vs H2 labels), which produces only 3-edge and 4-edge configurations.
    
    Edge numbering (for reference):
        Edge 0: (v0, v1)    Edge 3: (v1, v2)
        Edge 1: (v0, v2)    Edge 4: (v1, v3)
        Edge 2: (v0, v3)    Edge 5: (v2, v3)
    
    Active configurations:
        0 edges: Config 0 - no surface
        3 edges: 4 configs (7, 25, 42, 52) - single triangle (vertex isolated)
        4 edges: 3 configs (30, 45, 51) - quad (2 triangles)
    
    Disabled configurations (produce self-intersections):
        5 edges: 6 configs - non-manifold geometry
        6 edges: Config 63 - tetrahedral singularity
    
    Returns:
        Dictionary mapping 6-bit config -> list of triangles
        Each triangle is (edge_a, edge_b, edge_c) using local edge indices.
    """
    table = {}
    
    # Initialize all 64 configs to empty
    for i in range(64):
        table[i] = []
    
    # ===========================================================================
    # 3-EDGE CONFIGS: Single triangle (vertex isolated from others)
    # ===========================================================================
    # Vertex 0 isolated: edges 0,1,2 cut (edges from v0)
    # Config = 1 + 2 + 4 = 7
    table[7] = [(0, 1, 2)]
    
    # Vertex 1 isolated: edges 0,3,4 cut (edges from v1)
    # Config = 1 + 8 + 16 = 25
    table[25] = [(0, 3, 4)]
    
    # Vertex 2 isolated: edges 1,3,5 cut (edges from v2)
    # Config = 2 + 8 + 32 = 42
    table[42] = [(1, 3, 5)]
    
    # Vertex 3 isolated: edges 2,4,5 cut (edges from v3)
    # Config = 4 + 16 + 32 = 52
    table[52] = [(2, 4, 5)]
    
    # ===========================================================================
    # 4-EDGE CONFIGS: Quadrilateral surface (2 triangles)
    # ===========================================================================
    # Split {v0,v1} vs {v2,v3}: edges 1,2,3,4 cut
    # Config = 2 + 4 + 8 + 16 = 30
    table[30] = [(1, 3, 4), (1, 4, 2)]
    
    # Split {v0,v2} vs {v1,v3}: edges 0,2,3,5 cut
    # Config = 1 + 4 + 8 + 32 = 45
    table[45] = [(0, 3, 5), (0, 5, 2)]
    
    # Split {v0,v3} vs {v1,v2}: edges 0,1,4,5 cut
    # Config = 1 + 2 + 16 + 32 = 51
    table[51] = [(0, 4, 5), (0, 5, 1)]
    
    # ===========================================================================
    # 5-EDGE CONFIGS: DISABLED - These produce non-manifold/self-intersecting geometry
    # ===========================================================================
    # When 5 edges are cut, the resulting triangles share vertices in ways that
    # create self-intersections. Following the original paper approach, we skip
    # these configurations. Tetrahedra with 5 cut edges will not contribute triangles.
    #
    # Configs 62, 61, 59, 55, 47, 31 are left empty (no triangles).
    
    # ===========================================================================
    # 6-EDGE CONFIG: DISABLED - Tetrahedral singularity causes self-intersections
    # ===========================================================================
    # When all 6 edges are cut, the 4 triangles (one per face) self-intersect.
    # Following the original paper approach, we skip this configuration.
    #
    # Config 63 is left empty (no triangles).
    
    return table


# Pre-build the lookup table
MARCHING_TET_TABLE = _build_marching_tet_table()


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
    boundary_labels: Optional[np.ndarray] = None
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
    
    Args:
        vertices: (N, 3) vertex positions (inflated mesh)
        tetrahedra: (M, 4) tetrahedron vertex indices
        edges: (E, 2) global edge list
        cut_edge_flags: (E,) boolean flags for cut edges
        tet_edge_indices: (M, 6) tet-to-edge mapping
        use_original_vertices: If True, use vertices_original for surface construction
        vertices_original: (N, 3) original (non-inflated) vertex positions
        boundary_labels: (N,) vertex boundary labels: 0=interior, 1=H1, 2=H2, -1=part surface
    
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
    
    # Compute cut point positions with boundary-aware placement
    surface_vertices = np.zeros((len(cut_edge_indices), 3), dtype=np.float64)
    
    # Track boundary type for each vertex (per paper Section 4.4):
    # -1 = on part surface M (INNER boundary - re-project to part during smoothing)
    #  0 = interior midpoint (no boundary constraint)
    #  1 = on hull H1 boundary (OUTER boundary - re-project to hull)
    #  2 = on hull H2 boundary (OUTER boundary - re-project to hull)
    vertex_boundary_type = np.zeros(len(cut_edge_indices), dtype=np.int8)
    
    # Track statistics for logging
    n_part_boundary = 0   # Cut points placed on part surface M
    n_hull_boundary = 0   # Cut points placed on hull boundary ∂H
    n_midpoint = 0        # Cut points at edge midpoints (interior)
    
    for i, e_idx in enumerate(cut_edge_indices):
        v0, v1 = edges[e_idx]
        
        # Determine cut point placement based on boundary labels
        if boundary_labels is not None:
            bl0 = boundary_labels[v0]
            bl1 = boundary_labels[v1]
            
            # Check if either vertex is on part surface M (boundary_label == -1)
            # If so, place cut point at that vertex to ensure surface is bounded by M
            # Mark as INNER boundary (type -1) for re-projection to part during smoothing
            if bl0 == -1 and bl1 != -1:
                # v0 is on part surface, v1 is not → place at v0
                surface_vertices[i] = verts[v0]
                vertex_boundary_type[i] = -1  # INNER boundary - re-project to part M
                n_part_boundary += 1
            elif bl1 == -1 and bl0 != -1:
                # v1 is on part surface, v0 is not → place at v1
                surface_vertices[i] = verts[v1]
                vertex_boundary_type[i] = -1  # INNER boundary - re-project to part M
                n_part_boundary += 1
            elif bl0 == -1 and bl1 == -1:
                # BOTH on part surface → use midpoint but still mark as part boundary
                surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
                vertex_boundary_type[i] = -1  # INNER boundary - re-project to part M
                n_part_boundary += 1
            # Check if either vertex is on hull boundary ∂H (boundary_label == 1 or 2)
            # If so, place cut point at that vertex to ensure surface is bounded by ∂H
            # Mark as OUTER boundary (type 1 or 2) for re-projection to hull during smoothing
            elif bl0 in (1, 2) and bl1 == 0:
                # v0 is on hull boundary, v1 is interior → place at v0
                surface_vertices[i] = verts[v0]
                vertex_boundary_type[i] = bl0  # OUTER boundary - re-project to hull ∂H
                n_hull_boundary += 1
            elif bl1 in (1, 2) and bl0 == 0:
                # v1 is on hull boundary, v0 is interior → place at v1
                surface_vertices[i] = verts[v1]
                vertex_boundary_type[i] = bl1  # OUTER boundary - re-project to hull ∂H
                n_hull_boundary += 1
            elif bl0 in (1, 2) and bl1 in (1, 2):
                # BOTH on hull boundary → use midpoint but still mark as hull boundary
                surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
                vertex_boundary_type[i] = bl0  # OUTER boundary - re-project to hull ∂H
                n_hull_boundary += 1
            elif (bl0 in (1, 2) and bl1 == -1) or (bl1 in (1, 2) and bl0 == -1):
                # One on hull, one on part - this is a seam edge
                # Place at midpoint, but should still connect to BOTH boundaries
                # For re-projection, prefer hull since outer boundary projection is critical
                surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
                # Mark as hull boundary since we want outer edges to project to hull
                vertex_boundary_type[i] = bl0 if bl0 in (1, 2) else bl1  # OUTER boundary
                n_hull_boundary += 1
            else:
                # Both interior (bl0 == 0 and bl1 == 0) → use midpoint
                surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
                vertex_boundary_type[i] = 0  # Interior - no boundary constraint
                n_midpoint += 1
        else:
            # No boundary labels available → use midpoint (fallback)
            surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
            vertex_boundary_type[i] = 0  # Interior - no boundary constraint
            n_midpoint += 1
    
    logger.info(f"Cut point placement: {n_part_boundary} on part M (inner), "
                f"{n_hull_boundary} on hull ∂H (outer), {n_midpoint} midpoints (interior)")
    
    result.vertex_to_edge = cut_edge_indices.copy()
    result.vertex_boundary_type = vertex_boundary_type
    
    # Step 2: Process each tetrahedron
    triangles = []
    tets_contributing = 0
    
    # Track configuration statistics for debugging
    config_counts = {}
    
    for t in range(len(tetrahedra)):
        # Get the 6 global edge indices for this tet
        tet_edges = tet_edge_indices[t]  # Shape (6,)
        
        # Build 6-bit configuration: bit i = 1 if edge i is cut
        config = 0
        for local_e in range(6):
            global_e = tet_edges[local_e]
            if cut_edge_flags[global_e]:
                config |= (1 << local_e)
        
        # Track config counts
        config_counts[config] = config_counts.get(config, 0) + 1
        
        # Look up triangles from table
        table_triangles = MARCHING_TET_TABLE.get(config, [])
        
        if table_triangles:
            tets_contributing += 1
            
            for local_tri in table_triangles:
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
                    triangles.append(tri_verts)
    
    # Log configuration statistics
    logger.info(f"Configuration statistics ({len(config_counts)} unique configs):")
    high_config_tets = 0  # Count tets with 5+ edges cut
    for config in sorted(config_counts.keys()):
        n_edges = bin(config).count('1')
        has_triangles = len(MARCHING_TET_TABLE.get(config, [])) > 0
        status = "→ triangles" if has_triangles else "→ EMPTY (no triangles)"
        if config_counts[config] > 10:  # Only log significant configs
            logger.info(f"  Config {config:2d} ({config:06b}, {n_edges} edges): {config_counts[config]:5d} tets {status}")
        if n_edges >= 5:
            high_config_tets += config_counts[config]
    
    if high_config_tets > 0:
        logger.warning(f"  {high_config_tets} tets with 5+ edges cut - potential self-intersections")
    
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
    faces = faces[valid_mask]
    
    if len(faces) == 0:
        logger.warning("All triangles were degenerate")
        result.extraction_time_ms = (time.time() - start) * 1000
        return result
    
    result.vertices = surface_vertices
    result.faces = faces
    result.num_vertices = len(surface_vertices)
    result.num_faces = len(faces)
    
    # Create trimesh
    try:
        result.mesh = trimesh.Trimesh(
            vertices=surface_vertices,
            faces=faces,
            process=False  # Don't merge vertices or remove degenerates
        )
        
        # Optionally fix winding for consistent normals
        result.mesh.fix_normals()
        
    except Exception as e:
        logger.error(f"Failed to create trimesh: {e}")
    
    result.extraction_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Parting surface: {result.num_vertices} vertices, {result.num_faces} faces "
                f"from {result.num_tets_contributing}/{result.num_tets_processed} tets "
                f"in {result.extraction_time_ms:.1f}ms")
    
    return result


def extract_parting_surface_from_tet_result(
    tet_result,
    use_original_vertices: bool = True,
    prepare_data: bool = True,
    cut_type: str = 'both',
    extend_to_primary: bool = True
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
    
    Returns:
        PartingSurfaceResult
    """
    # Import here to avoid circular imports
    from . import tetrahedral_mesh as tm
    
    # Prepare data structures if not already done
    if prepare_data and tet_result.tet_edge_indices is None:
        logger.info("Preparing parting surface data structures...")
        tet_result = tm.prepare_parting_surface_data(tet_result)
    
    # Validate required data
    if tet_result.tet_edge_indices is None:
        raise ValueError("tet_edge_indices not computed - run prepare_parting_surface_data first")
    
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
    elif cut_type == 'secondary':
        if tet_result.secondary_cut_edges is None:
            logger.warning("No secondary cut edges found")
            return PartingSurfaceResult()
        
        if extend_to_primary:
            # Use extended flags that include primary edges in shared tets
            cut_flags = tm.compute_extended_secondary_cut_edge_flags(
                tet_result.edges,
                tet_result.tetrahedra,
                tet_result.tet_edge_indices,
                tet_result.primary_cut_edges,
                tet_result.secondary_cut_edges,
                tet_result.edge_to_index
            )
            logger.info(f"Extracting EXTENDED SECONDARY parting surface ({np.sum(cut_flags)} edges, connected to primary)")
        else:
            cut_flags = tm.compute_secondary_cut_edge_flags(
                tet_result.edges,
                tet_result.secondary_cut_edges,
                tet_result.edge_to_index
            )
            logger.info(f"Extracting SECONDARY parting surface ({np.sum(cut_flags)} edges)")
    else:  # 'both'
        if tet_result.cut_edge_flags is None:
            raise ValueError("cut_edge_flags not computed - run prepare_parting_surface_data first")
        cut_flags = tet_result.cut_edge_flags
        logger.info(f"Extracting combined (PRIMARY + SECONDARY) parting surface ({np.sum(cut_flags)} edges)")
    
    return extract_parting_surface(
        vertices=tet_result.vertices,
        tetrahedra=tet_result.tetrahedra,
        edges=tet_result.edges,
        cut_edge_flags=cut_flags,
        tet_edge_indices=tet_result.tet_edge_indices,
        use_original_vertices=use_original_vertices,
        vertices_original=tet_result.vertices_original,
        boundary_labels=tet_result.boundary_labels  # Pass for boundary-aware cut point placement
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
    
    # Create result
    result = PartingSurfaceResult(
        vertices=vertices,
        faces=surface.faces,
        vertex_to_edge=surface.vertex_to_edge,
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
                faces=mesh.faces[valid_faces]
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
                    faces=mesh.faces[valid_area_mask]
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
                
                mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                
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
    max_iterations: int = 3
) -> SelfFoldingResult:
    """
    Attempt to repair self-folding regions in the mesh.
    
    Repair strategies:
    1. Fix inconsistent face normals
    2. Optionally remove self-intersecting faces
    
    Args:
        mesh: Input mesh
        normal_consistency_threshold: Minimum dot product for normal consistency
        remove_intersecting: If True, remove self-intersecting faces
        max_iterations: Maximum iterations for normal fixing
    
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
    
    # Step 1: Detect self-folding
    info = detect_self_folding(working_mesh, normal_consistency_threshold)
    result.info = info
    result.flipped_faces_found = info.flipped_normal_count
    
    if info.flipped_normal_count == 0 and not remove_intersecting:
        result.mesh = working_mesh
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    result.self_folding_detected = True
    
    # Step 2: Try to fix normals using trimesh's built-in method
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
    
    # Step 3: Optionally detect and remove self-intersecting faces
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
                faces=new_faces
            )
            working_mesh.remove_unreferenced_vertices()
            
            result.faces_removed_for_intersection = len(problem_faces)
    
    result.mesh = working_mesh
    result.processing_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Self-folding repair: fixed {result.flipped_faces_fixed} normals, "
                f"removed {result.faces_removed_for_intersection} intersecting faces "
                f"in {result.processing_time_ms:.1f}ms")
    
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
