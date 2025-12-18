"""
Parting Surface Extraction using Marching Tetrahedra

This module extracts a continuous parting surface mesh from a tetrahedral mesh
where certain edges have been marked as "cut" (either primary or secondary cuts).

Algorithm: Extended Marching Tetrahedra
- Each tetrahedron has 6 edges
- Each edge can be either CUT (1) or NOT CUT (0)
- This gives 2^6 = 64 possible configurations
- For each configuration, we have a pre-computed lookup table that defines
  which triangles to emit (using edge midpoints as vertices)

The lookup table encodes surface triangles that pass through the cut edges,
creating a manifold surface that separates the two mold halves.

Edge numbering within a tetrahedron (vertices 0,1,2,3):
    Edge 0: (0,1)
    Edge 1: (0,2)
    Edge 2: (0,3)
    Edge 3: (1,2)
    Edge 4: (1,3)
    Edge 5: (2,3)

Face definitions (for understanding triangulation):
    Face 0: vertices (1,2,3) - opposite vertex 0
    Face 1: vertices (0,2,3) - opposite vertex 1
    Face 2: vertices (0,1,3) - opposite vertex 2
    Face 3: vertices (0,1,2) - opposite vertex 3
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
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

# For each edge, which two faces share it
# Useful for consistent triangle orientation
EDGE_TO_FACES = {
    0: (2, 3),  # Edge (0,1) shared by face 2 (0,1,3) and face 3 (0,1,2)
    1: (1, 3),  # Edge (0,2) shared by face 1 (0,2,3) and face 3 (0,1,2)
    2: (1, 2),  # Edge (0,3) shared by face 1 (0,2,3) and face 2 (0,1,3)
    3: (0, 3),  # Edge (1,2) shared by face 0 (1,2,3) and face 3 (0,1,2)
    4: (0, 2),  # Edge (1,3) shared by face 0 (1,2,3) and face 2 (0,1,3)
    5: (0, 1),  # Edge (2,3) shared by face 0 (1,2,3) and face 1 (0,2,3)
}


def _build_marching_tet_table() -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Build the 64-entry marching tetrahedra lookup table.
    
    For a tetrahedron with vertices labeled H1 or H2, an edge is "cut" if
    its two vertices have different labels. This gives us specific valid
    configurations based on how many vertices are H1 vs H2:
    
    Valid configurations from vertex labeling:
    - 0 edges cut: all vertices same label (no surface)
    - 3 edges cut: exactly 1 vertex differs (vertex isolated) -> 1 triangle
    - 4 edges cut: 2 vertices H1, 2 vertices H2 -> 2 triangles (quad)
    
    The 7 valid non-empty configurations are:
    - Config 7 (000111): vertex 0 isolated, edges 0,1,2 cut
    - Config 25 (011001): vertex 1 isolated, edges 0,3,4 cut
    - Config 42 (101010): vertex 2 isolated, edges 1,3,5 cut
    - Config 52 (110100): vertex 3 isolated, edges 2,4,5 cut
    - Config 30 (011110): 2-2 split {0,1} vs {2,3}, edges 1,2,3,4 cut
    - Config 45 (101101): 2-2 split {0,2} vs {1,3}, edges 0,2,3,5 cut
    - Config 51 (110011): 2-2 split {0,3} vs {1,2}, edges 0,1,4,5 cut
    
    Returns:
        Dictionary mapping 6-bit config -> list of triangles
        Each triangle is (edge_a, edge_b, edge_c) using local edge indices.
    """
    table = {}
    
    # Initialize all 64 configs to empty
    for i in range(64):
        table[i] = []
    
    # 3-edge configs: vertex isolated (one vertex has different label)
    # The triangle passes through the 3 edges emanating from the isolated vertex
    # 
    # Vertex 0 isolated: cut edges are (0,1), (0,2), (0,3) = edges 0, 1, 2
    # Config = 1 + 2 + 4 = 7
    table[7] = [(0, 1, 2)]
    
    # Vertex 1 isolated: cut edges are (0,1), (1,2), (1,3) = edges 0, 3, 4
    # Config = 1 + 8 + 16 = 25
    table[25] = [(0, 3, 4)]
    
    # Vertex 2 isolated: cut edges are (0,2), (1,2), (2,3) = edges 1, 3, 5
    # Config = 2 + 8 + 32 = 42
    table[42] = [(1, 3, 5)]
    
    # Vertex 3 isolated: cut edges are (0,3), (1,3), (2,3) = edges 2, 4, 5
    # Config = 4 + 16 + 32 = 52
    table[52] = [(2, 4, 5)]
    
    # 4-edge configs: 2-2 vertex split (two vertices H1, two H2)
    # The surface forms a quad through 4 edges, split into 2 triangles
    #
    # Split {0,1} vs {2,3}: cut edges are (0,2), (0,3), (1,2), (1,3) = edges 1,2,3,4
    # Config = 2 + 4 + 8 + 16 = 30
    # Quad edges in order: 1,3,4,2 form a cycle -> triangles (1,3,4) and (1,4,2)
    table[30] = [(1, 3, 4), (1, 4, 2)]
    
    # Split {0,2} vs {1,3}: cut edges are (0,1), (0,3), (1,2), (2,3) = edges 0,2,3,5
    # Config = 1 + 4 + 8 + 32 = 45
    # Quad edges in order: 0,3,5,2 form a cycle -> triangles (0,3,5) and (0,5,2)
    table[45] = [(0, 3, 5), (0, 5, 2)]
    
    # Split {0,3} vs {1,2}: cut edges are (0,1), (0,2), (1,3), (2,3) = edges 0,1,4,5
    # Config = 1 + 2 + 16 + 32 = 51
    # Quad edges in order: 0,4,5,1 form a cycle -> triangles (0,4,5) and (0,5,1)
    table[51] = [(0, 4, 5), (0, 5, 1)]
    
    return table


def _add_3_edge_configs_UNUSED(table: Dict[int, List[Tuple[int, int, int]]]):
    """
    DEPRECATED: Old implementation that used face-edge patterns.
    Kept for reference only.
    
    The correct pattern is vertex-isolation, not face-edges.
    Vertex isolation: 3 edges from an isolated vertex
    Face edges: 3 edges forming a face (WRONG)
    """
    pass


def _add_4_edge_configs_UNUSED(table: Dict[int, List[Tuple[int, int, int]]]):
    """
    DEPRECATED: Old implementation.
    Kept for reference only.
    """
    pass


# Pre-build the lookup table
MARCHING_TET_TABLE = _build_marching_tet_table()


def _log_table_stats():
    """Log statistics about the lookup table."""
    stats = {}
    for config, triangles in MARCHING_TET_TABLE.items():
        n_tris = len(triangles)
        stats[n_tris] = stats.get(n_tris, 0) + 1
    
    logger.debug(f"Marching tet table: {sum(stats.values())} configs")
    for n_tris, count in sorted(stats.items()):
        logger.debug(f"  {count} configs with {n_tris} triangle(s)")


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
    
    for t in range(len(tetrahedra)):
        # Get the 6 global edge indices for this tet
        tet_edges = tet_edge_indices[t]  # Shape (6,)
        
        # Build 6-bit configuration: bit i = 1 if edge i is cut
        config = 0
        for local_e in range(6):
            global_e = tet_edges[local_e]
            if cut_edge_flags[global_e]:
                config |= (1 << local_e)
        
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


def fill_holes_in_surface(
    surface: PartingSurfaceResult,
    max_hole_edges: int = 200
) -> PartingSurfaceResult:
    """
    Fill holes in the parting surface mesh.
    
    Uses trimesh's hole filling capabilities to close gaps
    in the surface that may result from labeling inconsistencies.
    
    Args:
        surface: PartingSurfaceResult with holes
        max_hole_edges: Maximum number of edges in a hole to fill (larger holes skipped)
                       Default increased to 200 for better hole detection.
    
    Returns:
        New PartingSurfaceResult with holes filled
    """
    if surface.mesh is None:
        return surface
    
    import time
    start = time.time()
    
    try:
        mesh = surface.mesh.copy()
        
        # Get initial hole count
        initial_holes = len(mesh.outline().entities) if hasattr(mesh.outline(), 'entities') else 0
        
        # Fill holes using trimesh
        # This triangulates boundary loops
        filled = mesh.fill_holes()
        
        if filled:
            # Get final hole count
            final_holes = len(mesh.outline().entities) if hasattr(mesh.outline(), 'entities') else 0
            
            # Update result
            result = PartingSurfaceResult(
                mesh=mesh,
                vertices=np.array(mesh.vertices),
                faces=np.array(mesh.faces),
                vertex_to_edge=surface.vertex_to_edge,  # May be stale for new verts
                num_vertices=len(mesh.vertices),
                num_faces=len(mesh.faces),
                num_tets_processed=surface.num_tets_processed,
                num_tets_contributing=surface.num_tets_contributing,
                extraction_time_ms=surface.extraction_time_ms
            )
            
            elapsed = (time.time() - start) * 1000
            holes_filled = initial_holes - final_holes
            new_faces = result.num_faces - surface.num_faces
            logger.info(f"Filled {holes_filled} holes, added {new_faces} faces in {elapsed:.1f}ms")
            
            return result
        else:
            logger.info("No holes to fill")
            return surface
            
    except Exception as e:
        logger.warning(f"Hole filling failed: {e}")
        return surface


def repair_parting_surface(
    surface: PartingSurfaceResult,
    fill_holes: bool = True,
    merge_vertices: bool = True,
    merge_threshold: float = 1e-8,
    max_hole_edges: int = 500
) -> PartingSurfaceResult:
    """
    Repair the parting surface mesh to ensure it's watertight.
    
    This performs several repair operations:
    1. Merge duplicate/close vertices
    2. Remove degenerate faces
    3. Fill holes (including larger holes up to max_hole_edges)
    4. Fix normals for consistency
    
    Args:
        surface: PartingSurfaceResult to repair
        fill_holes: Whether to fill holes
        merge_vertices: Whether to merge close vertices
        merge_threshold: Distance threshold for vertex merging
        max_hole_edges: Maximum hole size to fill (number of boundary edges)
                       Increased default (500) to catch larger holes.
    
    Returns:
        Repaired PartingSurfaceResult
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
        
        # Step 3: Remove duplicate faces
        mesh.remove_duplicate_faces()
        
        # Step 4: Fill holes - use custom implementation for larger holes
        if fill_holes:
            holes_filled = _fill_holes_comprehensive(mesh, max_hole_edges=max_hole_edges)
            logger.debug(f"Filled {holes_filled} holes")
        
        # Step 5: Fix normals
        mesh.fix_normals()
        
        # Create result
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
        face_diff = result.num_faces - initial_faces
        logger.info(f"Repaired parting surface: merged {vert_diff} verts, "
                    f"added {face_diff} faces in {elapsed:.1f}ms")
        
        return result
        
    except Exception as e:
        logger.warning(f"Surface repair failed: {e}")
        return surface


def _fill_holes_comprehensive(mesh: trimesh.Trimesh, max_hole_edges: int = 500) -> int:
    """
    Fill holes in a mesh, including larger holes.
    
    This function finds boundary loops and triangulates them to fill holes.
    Unlike trimesh's default fill_holes(), it can handle larger holes.
    
    Args:
        mesh: Trimesh to modify in-place
        max_hole_edges: Maximum number of edges in a hole to fill
        
    Returns:
        Number of holes filled
    """
    from scipy.spatial import Delaunay
    
    holes_filled = 0
    
    try:
        # Get boundary edges (edges that appear in only one face)
        edges = mesh.edges_unique
        edges_sorted = np.sort(edges, axis=1)
        
        # Count edge occurrences in faces
        face_edges = mesh.faces_unique_edges
        edge_counts = np.zeros(len(edges), dtype=np.int32)
        for fe in face_edges:
            edge_counts[fe] += 1
        
        # Boundary edges appear exactly once
        boundary_mask = edge_counts == 1
        boundary_edges = edges[boundary_mask]
        
        if len(boundary_edges) == 0:
            return 0
        
        # Find connected boundary loops
        loops = _find_boundary_loops(boundary_edges)
        
        for loop in loops:
            if len(loop) > max_hole_edges:
                logger.debug(f"Skipping large hole with {len(loop)} edges")
                continue
            
            if len(loop) < 3:
                continue
            
            # Get the vertices in order around the loop
            loop_vertices = mesh.vertices[loop]
            
            # Triangulate the hole using ear clipping or fan triangulation
            new_faces = _triangulate_polygon_3d(loop, loop_vertices)
            
            if new_faces is not None and len(new_faces) > 0:
                # Add new faces to mesh
                current_faces = list(mesh.faces)
                current_faces.extend(new_faces)
                mesh.faces = np.array(current_faces)
                holes_filled += 1
        
        # Let trimesh clean up - remove degenerate faces
        valid_faces = mesh.nondegenerate_faces()
        if np.sum(~valid_faces) > 0:
            mesh.update_faces(valid_faces)
        mesh.remove_duplicate_faces()
        
    except Exception as e:
        logger.warning(f"Comprehensive hole filling failed: {e}")
        # Fallback to trimesh's default
        try:
            mesh.fill_holes()
        except:
            pass
    
    return holes_filled


def _find_boundary_loops(edges: np.ndarray) -> List[List[int]]:
    """
    Find closed loops from boundary edges.
    
    Args:
        edges: (N, 2) array of edge vertex indices
        
    Returns:
        List of vertex index lists, each forming a closed loop
    """
    if len(edges) == 0:
        return []
    
    # Build adjacency from edges
    adj = {}
    for e in edges:
        v0, v1 = int(e[0]), int(e[1])
        if v0 not in adj:
            adj[v0] = []
        if v1 not in adj:
            adj[v1] = []
        adj[v0].append(v1)
        adj[v1].append(v0)
    
    # Find loops by walking
    visited_edges = set()
    loops = []
    
    for start in adj.keys():
        if len(adj[start]) != 2:
            continue  # Not a manifold boundary vertex
        
        # Try to walk a loop starting from this vertex
        for next_v in adj[start]:
            edge_key = (min(start, next_v), max(start, next_v))
            if edge_key in visited_edges:
                continue
            
            # Walk the loop
            loop = [start]
            current = next_v
            prev = start
            
            while current != start and len(loop) < 10000:
                loop.append(current)
                visited_edges.add((min(prev, current), max(prev, current)))
                
                # Find next vertex (not the one we came from)
                neighbors = adj.get(current, [])
                next_candidates = [n for n in neighbors if n != prev]
                
                if not next_candidates:
                    break
                
                prev = current
                current = next_candidates[0]
            
            if current == start and len(loop) >= 3:
                # Mark the closing edge as visited
                visited_edges.add((min(prev, start), max(prev, start)))
                loops.append(loop)
    
    return loops


def _triangulate_polygon_3d(vertex_indices: List[int], vertices: np.ndarray) -> Optional[np.ndarray]:
    """
    Triangulate a 3D polygon using fan triangulation.
    
    For convex or near-convex polygons, fan triangulation works well.
    For complex polygons, we project to 2D and use ear clipping.
    
    Args:
        vertex_indices: List of vertex indices forming the polygon boundary
        vertices: (N, 3) array of vertex positions
        
    Returns:
        (M, 3) array of triangle vertex indices, or None if failed
    """
    n = len(vertex_indices)
    if n < 3:
        return None
    
    if n == 3:
        return np.array([vertex_indices])
    
    try:
        # Compute polygon normal
        v0 = vertices[0]
        edges_vec = vertices[1:] - v0
        
        # Use cross product of first two edges as normal estimate
        if len(edges_vec) >= 2:
            normal = np.cross(edges_vec[0], edges_vec[1])
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len
            else:
                normal = np.array([0, 0, 1])
        else:
            normal = np.array([0, 0, 1])
        
        # Project to 2D
        # Find two orthogonal axes in the polygon plane
        up = np.array([0, 1, 0]) if abs(normal[1]) < 0.9 else np.array([1, 0, 0])
        axis_x = np.cross(up, normal)
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = np.cross(normal, axis_x)
        
        # Project vertices to 2D
        centered = vertices - vertices.mean(axis=0)
        pts_2d = np.column_stack([
            np.dot(centered, axis_x),
            np.dot(centered, axis_y)
        ])
        
        # Use ear clipping triangulation
        triangles = _ear_clip_triangulate(pts_2d)
        
        if triangles is not None and len(triangles) > 0:
            # Map back to original vertex indices
            return np.array([[vertex_indices[t[0]], vertex_indices[t[1]], vertex_indices[t[2]]] 
                           for t in triangles])
        
    except Exception as e:
        logger.debug(f"Polygon triangulation failed: {e}")
    
    # Fallback: simple fan triangulation from centroid
    # This works for convex polygons
    triangles = []
    for i in range(1, n - 1):
        triangles.append([vertex_indices[0], vertex_indices[i], vertex_indices[i + 1]])
    
    return np.array(triangles) if triangles else None


def _ear_clip_triangulate(points: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
    """
    Simple ear clipping triangulation for 2D polygon.
    
    Args:
        points: (N, 2) array of 2D points in order around polygon
        
    Returns:
        List of (i, j, k) triangle index tuples
    """
    n = len(points)
    if n < 3:
        return None
    if n == 3:
        return [(0, 1, 2)]
    
    # Initialize vertex list
    remaining = list(range(n))
    triangles = []
    
    max_iterations = n * n  # Safety limit
    iterations = 0
    
    while len(remaining) > 3 and iterations < max_iterations:
        iterations += 1
        found_ear = False
        
        for i in range(len(remaining)):
            prev_idx = remaining[(i - 1) % len(remaining)]
            curr_idx = remaining[i]
            next_idx = remaining[(i + 1) % len(remaining)]
            
            # Check if this is an ear (convex vertex with no points inside)
            if _is_ear(points, remaining, prev_idx, curr_idx, next_idx):
                triangles.append((prev_idx, curr_idx, next_idx))
                remaining.remove(curr_idx)
                found_ear = True
                break
        
        if not found_ear:
            # No ear found, polygon might be degenerate
            # Fall back to fan triangulation for remaining
            for i in range(1, len(remaining) - 1):
                triangles.append((remaining[0], remaining[i], remaining[i + 1]))
            break
    
    # Handle final triangle
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    
    return triangles


def _is_ear(points: np.ndarray, remaining: List[int], 
            prev_idx: int, curr_idx: int, next_idx: int) -> bool:
    """Check if vertex curr_idx forms an ear in the polygon."""
    p0 = points[prev_idx]
    p1 = points[curr_idx]
    p2 = points[next_idx]
    
    # Check if the vertex is convex (left turn)
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if cross <= 0:
        return False
    
    # Check if any other vertex is inside the triangle
    for idx in remaining:
        if idx in (prev_idx, curr_idx, next_idx):
            continue
        if _point_in_triangle(points[idx], p0, p1, p2):
            return False
    
    return True


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if point p is inside triangle abc using barycentric coordinates."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-10:
        return False
    
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= 0) and (v >= 0) and (u + v < 1)


# =============================================================================
# BRIDGE GAPS TO PART MESH
# =============================================================================

def bridge_to_part_mesh(
    surface: PartingSurfaceResult,
    part_mesh: trimesh.Trimesh,
    max_bridge_distance: float = 5.0,
    min_edge_length: float = 0.1
) -> PartingSurfaceResult:
    """
    Bridge gaps between the parting surface boundary and the part mesh.
    
    This function finds boundary edges of the parting surface that are close
    to the part mesh and creates triangles to close the gap.
    
    Args:
        surface: PartingSurfaceResult with potential gaps
        part_mesh: The original part mesh to bridge to
        max_bridge_distance: Maximum distance to bridge (vertices farther away are ignored)
        min_edge_length: Minimum edge length to consider (avoids degenerate triangles)
    
    Returns:
        New PartingSurfaceResult with gaps bridged
    """
    if surface.mesh is None or part_mesh is None:
        return surface
    
    import time
    from trimesh.proximity import ProximityQuery
    
    start = time.time()
    
    try:
        mesh = surface.mesh.copy()
        initial_faces = len(mesh.faces)
        
        # Find boundary vertices
        boundary_verts = _get_boundary_vertices(mesh)
        
        if len(boundary_verts) == 0:
            logger.debug("No boundary vertices found - surface may be closed")
            return surface
        
        # Create proximity query for part mesh
        proximity = ProximityQuery(part_mesh)
        
        # Get boundary vertex positions
        boundary_positions = mesh.vertices[boundary_verts]
        
        # Find closest points on part mesh
        closest_points, distances, face_indices = proximity.on_surface(boundary_positions)
        
        # Filter by distance - only bridge vertices close enough to part mesh
        close_mask = distances < max_bridge_distance
        
        if not np.any(close_mask):
            logger.debug(f"No boundary vertices within {max_bridge_distance} of part mesh")
            return surface
        
        # Get the boundary loops for ordered traversal
        boundary_edges = _get_boundary_edges(mesh)
        loops = _find_boundary_loops(boundary_edges)
        
        new_faces = []
        new_vertices = list(mesh.vertices)
        
        for loop in loops:
            if len(loop) < 3:
                continue
            
            # For each edge in the loop, check if we should bridge
            for i in range(len(loop)):
                v0_idx = loop[i]
                v1_idx = loop[(i + 1) % len(loop)]
                
                # Check if both vertices are in our boundary set and close to part mesh
                if v0_idx not in boundary_verts or v1_idx not in boundary_verts:
                    continue
                
                # Get indices into boundary_verts array
                bv0_pos = np.where(boundary_verts == v0_idx)[0]
                bv1_pos = np.where(boundary_verts == v1_idx)[0]
                
                if len(bv0_pos) == 0 or len(bv1_pos) == 0:
                    continue
                
                bv0_idx = bv0_pos[0]
                bv1_idx = bv1_pos[0]
                
                # Check if close enough to bridge
                if not close_mask[bv0_idx] or not close_mask[bv1_idx]:
                    continue
                
                # Get the closest points on part mesh
                p0_on_part = closest_points[bv0_idx]
                p1_on_part = closest_points[bv1_idx]
                
                # Get parting surface boundary positions
                p0_boundary = mesh.vertices[v0_idx]
                p1_boundary = mesh.vertices[v1_idx]
                
                # Check if the projection points are different enough
                proj_dist = np.linalg.norm(p0_on_part - p1_on_part)
                if proj_dist < min_edge_length:
                    # Points project to nearly same location - make single triangle
                    # Add the midpoint on part mesh as new vertex
                    mid_on_part = 0.5 * (p0_on_part + p1_on_part)
                    new_vert_idx = len(new_vertices)
                    new_vertices.append(mid_on_part)
                    
                    # Triangle from boundary edge to new point
                    new_faces.append([v0_idx, v1_idx, new_vert_idx])
                else:
                    # Create two triangles (a quad between boundary edge and part mesh)
                    # Add two new vertices on part mesh
                    new_v0_idx = len(new_vertices)
                    new_vertices.append(p0_on_part)
                    new_v1_idx = len(new_vertices)
                    new_vertices.append(p1_on_part)
                    
                    # Two triangles forming a quad
                    new_faces.append([v0_idx, v1_idx, new_v1_idx])
                    new_faces.append([v0_idx, new_v1_idx, new_v0_idx])
        
        if not new_faces:
            logger.debug("No bridging faces created")
            return surface
        
        # Build new mesh with added vertices and faces
        all_vertices = np.array(new_vertices)
        all_faces = np.vstack([mesh.faces, np.array(new_faces)])
        
        new_mesh = trimesh.Trimesh(
            vertices=all_vertices,
            faces=all_faces,
            process=False
        )
        
        # Clean up
        new_mesh.merge_vertices()
        valid_faces = new_mesh.nondegenerate_faces()
        if np.sum(~valid_faces) > 0:
            new_mesh = trimesh.Trimesh(
                vertices=new_mesh.vertices,
                faces=new_mesh.faces[valid_faces]
            )
        new_mesh.remove_duplicate_faces()
        new_mesh.fix_normals()
        
        result = PartingSurfaceResult(
            mesh=new_mesh,
            vertices=np.array(new_mesh.vertices),
            faces=np.array(new_mesh.faces),
            vertex_to_edge=None,
            num_vertices=len(new_mesh.vertices),
            num_faces=len(new_mesh.faces),
            num_tets_processed=surface.num_tets_processed,
            num_tets_contributing=surface.num_tets_contributing,
            extraction_time_ms=surface.extraction_time_ms
        )
        
        elapsed = (time.time() - start) * 1000
        faces_added = result.num_faces - initial_faces
        logger.info(f"Bridged parting surface to part mesh: added {faces_added} faces in {elapsed:.1f}ms")
        
        return result
        
    except Exception as e:
        logger.warning(f"Bridge to part mesh failed: {e}")
        return surface


def _get_boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Get indices of boundary vertices (vertices on boundary edges)."""
    boundary_edges = _get_boundary_edges(mesh)
    if len(boundary_edges) == 0:
        return np.array([], dtype=np.int64)
    
    return np.unique(boundary_edges.flatten())


def _get_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Get boundary edges (edges appearing in exactly one face)."""
    edges = mesh.edges_unique
    
    # Count how many faces each edge belongs to
    edge_counts = np.zeros(len(edges), dtype=np.int32)
    face_edges = mesh.faces_unique_edges
    for fe in face_edges:
        edge_counts[fe] += 1
    
    # Boundary edges appear exactly once
    boundary_mask = edge_counts == 1
    return edges[boundary_mask]


def repair_parting_surface_with_part(
    surface: PartingSurfaceResult,
    part_mesh: trimesh.Trimesh,
    fill_internal_holes: bool = True,
    bridge_to_part: bool = True,
    max_hole_edges: int = 500,
    max_bridge_distance: float = 5.0
) -> PartingSurfaceResult:
    """
    Comprehensive repair of parting surface including bridging to part mesh.
    
    This is an enhanced version of repair_parting_surface that also
    closes gaps between the parting surface boundary and the original part.
    
    Args:
        surface: PartingSurfaceResult to repair
        part_mesh: The original part mesh
        fill_internal_holes: Whether to fill internal holes in the surface
        bridge_to_part: Whether to bridge gaps to the part mesh
        max_hole_edges: Maximum hole size to fill
        max_bridge_distance: Maximum distance to bridge to part mesh
    
    Returns:
        Repaired PartingSurfaceResult
    """
    if surface.mesh is None:
        return surface
    
    # Step 1: Basic repair (merge vertices, remove degenerates, fill internal holes)
    result = repair_parting_surface(
        surface,
        fill_holes=fill_internal_holes,
        merge_vertices=True,
        max_hole_edges=max_hole_edges
    )
    
    # Step 2: Bridge gaps to part mesh
    if bridge_to_part and part_mesh is not None:
        result = bridge_to_part_mesh(
            result,
            part_mesh,
            max_bridge_distance=max_bridge_distance
        )
    
    return result


# Log table stats on module load (debug level)
_log_table_stats()
