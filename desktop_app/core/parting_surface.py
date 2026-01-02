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
    vertices_original: Optional[np.ndarray] = None
) -> PartingSurfaceResult:
    """
    Extract the parting surface mesh using Marching Tetrahedra.
    
    For each tetrahedron:
    1. Look up which of its 6 edges are cut (using cut_edge_flags)
    2. Build a 6-bit configuration index
    3. Look up triangles from MARCHING_TET_TABLE
    4. Emit triangles using edge midpoints as vertices
    
    Args:
        vertices: (N, 3) vertex positions (inflated mesh)
        tetrahedra: (M, 4) tetrahedron vertex indices
        edges: (E, 2) global edge list
        cut_edge_flags: (E,) boolean flags for cut edges
        tet_edge_indices: (M, 6) tet-to-edge mapping
        use_original_vertices: If True, use vertices_original for surface construction
        vertices_original: (N, 3) original (non-inflated) vertex positions
    
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
    
    # Step 1: Compute edge midpoints for all cut edges
    # We'll create vertices only for cut edges
    cut_edge_indices = np.where(cut_edge_flags)[0]
    
    # Map from global edge index to surface vertex index
    edge_to_surface_vertex = {int(e): i for i, e in enumerate(cut_edge_indices)}
    
    # Compute midpoints
    surface_vertices = np.zeros((len(cut_edge_indices), 3), dtype=np.float64)
    for i, e_idx in enumerate(cut_edge_indices):
        v0, v1 = edges[e_idx]
        surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
    
    result.vertex_to_edge = cut_edge_indices.copy()
    
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
        vertices_original=tet_result.vertices_original
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
    1. Merge duplicate/close vertices
    2. Remove degenerate faces (zero area)
    3. Remove small area triangles (< min_area)
    
    Args:
        surface: PartingSurfaceResult to clean
        merge_vertices: Whether to merge close vertices
        merge_threshold: Distance threshold for vertex merging
    
    Returns:
        Cleaned PartingSurfaceResult
    """
    if surface.mesh is None:
        return surface
    
    import time
    start = time.time()
    
    try:
        mesh = surface.mesh.copy()
        initial_verts = len(mesh.vertices)
        initial_faces = len(mesh.faces)
        
        # Step 1: Merge close vertices
        if merge_vertices:
            mesh.merge_vertices(merge_tex=False, merge_norm=False)
        
        # Step 2: Remove degenerate faces (zero area)
        valid_faces = mesh.nondegenerate_faces()
        if np.sum(~valid_faces) > 0:
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[valid_faces]
            )
        
        # Step 3: Remove very small triangles (potential self-intersection sources)
        # These often result from 5-edge configs where triangles nearly collapse
        if len(mesh.faces) > 0:
            areas = mesh.area_faces
            median_area = np.median(areas)
            min_area_threshold = median_area * 0.01  # 1% of median area
            valid_area_mask = areas >= min_area_threshold
            small_removed = np.sum(~valid_area_mask)
            if small_removed > 0:
                logger.info(f"Removing {small_removed} very small triangles (< {min_area_threshold:.6f} area)")
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces[valid_area_mask]
                )
                mesh.remove_unreferenced_vertices()
        
        result = PartingSurfaceResult(
            mesh=mesh,
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            vertex_to_edge=None,  # Invalidated by merge
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
    method: str = 'smart_curtain'
) -> GapClosingResult:
    """
    Close gaps between parting surface boundary edges and the part mesh.
    
    IMPORTANT: This function returns the mesh with properly connected fill geometry.
    The `part_constrained_vertices` field contains indices of vertices that MUST
    be constrained to stay on the part surface during any subsequent smoothing.
    
    Algorithm:
    1. Find boundary edges near the part
    2. Build ordered chains from these edges
    3. Project boundary vertices to closest point on part surface
    4. Create fill triangles that:
       - Reference original surface vertices directly (no duplication)
       - Add only the projected-to-part vertices as new vertices
       - Use consistent winding direction
    
    Args:
        surface: PartingSurfaceResult with boundary edges to close
        part_mesh: The original part mesh to close gaps against
        distance_threshold: Max distance to consider "near part" (auto-computed if None)
        method: Gap closing method (currently 'smart_curtain' is recommended)
    
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
    
    # === Step 2: Compute distances to part surface ===
    boundary_edge_array = np.array(boundary_edges)
    midpoints = 0.5 * (vertices[boundary_edge_array[:, 0]] + vertices[boundary_edge_array[:, 1]])
    
    # Use trimesh's closest_point for accurate surface distance
    closest_pts, distances, _ = trimesh.proximity.closest_point(part_mesh, midpoints)
    
    # Auto-compute threshold if not provided
    if distance_threshold is None:
        edge_lengths = np.linalg.norm(
            vertices[boundary_edge_array[:, 1]] - vertices[boundary_edge_array[:, 0]], 
            axis=1
        )
        distance_threshold = np.median(edge_lengths) * 2.0
        logger.info(f"Auto distance threshold: {distance_threshold:.4f}")
    
    # Find edges near part
    near_part_mask = distances < distance_threshold
    near_part_edges = boundary_edge_array[near_part_mask]
    result.boundary_edges_near_part = len(near_part_edges)
    
    logger.info(f"Found {len(near_part_edges)} boundary edges near part (< {distance_threshold:.4f})")
    
    if len(near_part_edges) == 0:
        logger.info("No boundary edges near part - no gaps to close")
        result.mesh = mesh
        result.vertices = vertices
        result.faces = faces
        result.processing_time_ms = (time.time() - start) * 1000
        return result
    
    # === Step 3: Build boundary chains from near-part edges ===
    boundary_chains = _build_boundary_chains(near_part_edges)
    result.boundary_loops_found = len(boundary_chains)
    logger.info(f"Found {len(boundary_chains)} boundary chains near part")
    
    # === Step 4: Create fill geometry with proper vertex sharing ===
    fill_result = _create_connected_fill(
        vertices, faces, boundary_chains, part_mesh, boundary_edge_info, distance_threshold
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
    max_distance: float
) -> Dict:
    """
    Create SINGLE fill triangles per boundary edge that connect to part mesh.
    
    SIMPLIFIED APPROACH: For each boundary edge (bv0, bv1), create ONE triangle
    that connects to a single projected point on the part mesh. This avoids
    self-intersecting geometry from the "curtain" approach.
    
    The projected point is the edge midpoint projected to the part surface.
    Triangle: (bv0, bv1, projected_midpoint)
    
    The projected_midpoint vertex should be re-projected to part during smoothing,
    just like the parting surface vertices are re-projected.
    
    Args:
        surface_vertices: Original parting surface vertices
        surface_faces: Original parting surface faces
        boundary_chains: List of (vertex_list, is_closed) tuples
        part_mesh: Part mesh for projection
        boundary_edge_info: Dict mapping edge_key to (v0, v1, v_opposite) for winding
        max_distance: Maximum allowed projection distance
    
    Returns:
        Dict with keys:
        - 'new_vertices': Projected midpoints (to be appended to surface_vertices)
        - 'new_faces': Face indices referencing combined vertex array
        - 'part_constrained_indices': Indices of projected vertices (should re-project to part)
        - 'fill_boundary_indices': Indices of original boundary vertices used (inner rim)
        - 'inner_rim_edges': Nx2 array of inner rim edge vertex pairs
    """
    n_orig_verts = len(surface_vertices)
    new_vertices = []  # Projected midpoints
    new_faces = []
    part_constrained_indices = []  # Track which new vertices should stay on part
    fill_boundary_indices = set()  # Track original boundary vertices used in fill
    inner_rim_edges = []  # Track edges along the inner rim for boundary neighbor computation
    
    for chain_idx, (chain_verts, is_closed) in enumerate(boundary_chains):
        if len(chain_verts) < 2:
            continue
        
        n_pts = len(chain_verts)
        n_edges = n_pts if is_closed else n_pts - 1
        
        for i in range(n_edges):
            next_i = (i + 1) % n_pts
            
            # Original boundary vertex indices
            bv0 = chain_verts[i]
            bv1 = chain_verts[next_i]
            
            # Get boundary edge positions
            b0_pos = surface_vertices[bv0]
            b1_pos = surface_vertices[bv1]
            
            # Check if edge is degenerate
            edge_len = np.linalg.norm(b1_pos - b0_pos)
            if edge_len < 1e-8:
                continue
            
            # Compute edge midpoint
            edge_midpoint = 0.5 * (b0_pos + b1_pos)
            
            # Project midpoint to part surface
            proj_pts, distances, face_indices = trimesh.proximity.closest_point(part_mesh, [edge_midpoint])
            proj_pt = proj_pts[0]
            proj_dist = distances[0]
            
            # Skip if projection is too far
            if proj_dist > max_distance:
                continue
            
            # Check that projected point is not too close to either boundary vertex
            # (would create degenerate triangle)
            dist_to_b0 = np.linalg.norm(proj_pt - b0_pos)
            dist_to_b1 = np.linalg.norm(proj_pt - b1_pos)
            min_dist = min(dist_to_b0, dist_to_b1)
            
            # If the projected point is too close, it means the boundary is already
            # on the part surface. In this case, move along the part surface normal
            # to find a valid projection point.
            if min_dist < edge_len * 0.1:
                # Get the part surface normal at this point
                face_idx = face_indices[0]
                part_normal = part_mesh.face_normals[face_idx]
                
                # Also get the direction from the adjacent face's interior to the edge
                edge_key = (min(bv0, bv1), max(bv0, bv1))
                if edge_key in boundary_edge_info:
                    adj_v0, adj_v1, adj_opp = boundary_edge_info[edge_key]
                    interior_dir = edge_midpoint - surface_vertices[adj_opp]
                    interior_dir = interior_dir / (np.linalg.norm(interior_dir) + 1e-10)
                    
                    # Move the projection point along the part surface, away from interior
                    # Project interior_dir onto the tangent plane of the part surface
                    tangent_dir = interior_dir - np.dot(interior_dir, part_normal) * part_normal
                    tangent_len = np.linalg.norm(tangent_dir)
                    
                    if tangent_len > 1e-6:
                        tangent_dir = tangent_dir / tangent_len
                        # Move by a fraction of the edge length along this direction
                        offset = edge_len * 0.3
                        candidate_pt = proj_pt + tangent_dir * offset
                        
                        # Re-project to ensure it's on the part surface
                        new_proj_pts, new_dists, _ = trimesh.proximity.closest_point(part_mesh, [candidate_pt])
                        proj_pt = new_proj_pts[0]
                        
                        # Re-check distances
                        dist_to_b0 = np.linalg.norm(proj_pt - b0_pos)
                        dist_to_b1 = np.linalg.norm(proj_pt - b1_pos)
                        min_dist = min(dist_to_b0, dist_to_b1)
                        
                        # Still too close? Skip this edge
                        if min_dist < edge_len * 0.1:
                            continue
                    else:
                        # Can't find a good tangent direction, skip
                        continue
                else:
                    # No edge info, skip
                    continue
            
            # Create new vertex for the projected midpoint
            new_vert_idx = n_orig_verts + len(new_vertices)
            new_vertices.append(proj_pt)
            part_constrained_indices.append(new_vert_idx)
            
            # Track boundary vertices and inner rim edge
            fill_boundary_indices.add(bv0)
            fill_boundary_indices.add(bv1)
            inner_rim_edges.append([bv0, bv1])
            
            # Determine winding direction from adjacent face
            edge_key = (min(bv0, bv1), max(bv0, bv1))
            
            if edge_key in boundary_edge_info:
                adj_v0, adj_v1, adj_opp = boundary_edge_info[edge_key]
                
                # Compute direction from edge to adjacent face's opposite vertex
                edge_mid_adj = 0.5 * (surface_vertices[adj_v0] + surface_vertices[adj_v1])
                to_inside = surface_vertices[adj_opp] - edge_mid_adj
                
                # Direction to projected point
                to_proj = proj_pt - edge_midpoint
                
                # If projection is on same side as the opposite vertex, flip winding
                same_side = np.dot(to_inside, to_proj) > 0
                
                if same_side:
                    # Flip winding: normal points away from inside
                    new_faces.append([bv0, bv1, new_vert_idx])
                else:
                    # Normal winding
                    new_faces.append([bv0, new_vert_idx, bv1])
            else:
                # No edge info - default winding
                new_faces.append([bv0, new_vert_idx, bv1])
    
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


def _build_boundary_loops(edges: np.ndarray) -> List[List[int]]:
    """
    Build ordered boundary loops from a set of boundary edges.
    
    Args:
        edges: (N, 2) array of edge vertex indices
    
    Returns:
        List of loops, each loop is a list of vertex indices in order
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
    
    loops = []
    visited_edges = set()
    
    for start_edge in edges:
        edge_key = (min(start_edge[0], start_edge[1]), max(start_edge[0], start_edge[1]))
        if edge_key in visited_edges:
            continue
        
        # Start a new loop from this edge
        loop = [start_edge[0], start_edge[1]]
        visited_edges.add(edge_key)
        
        # Walk forward from end vertex
        current = start_edge[1]
        prev = start_edge[0]
        
        while True:
            # Find next vertex
            neighbors = adj.get(current, [])
            next_vert = None
            for n in neighbors:
                if n != prev:
                    edge_key = (min(current, n), max(current, n))
                    if edge_key not in visited_edges:
                        next_vert = n
                        visited_edges.add(edge_key)
                        break
            
            if next_vert is None:
                break
            
            if next_vert == loop[0]:
                # Loop closed
                break
            
            loop.append(next_vert)
            prev = current
            current = next_vert
        
        if len(loop) >= 3:
            loops.append(loop)
    
    return loops
