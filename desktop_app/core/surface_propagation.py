"""
Surface Propagation and Smoothing

This module provides algorithms for:
1. Removing isolated triangle islands (connected components with < N triangles)
2. Propagating/growing secondary parting surfaces to connect with:
   - The primary parting surface
   - The part mesh surface

The propagation extends boundary edges of the secondary surface until they
reach target surfaces.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Smoothing tolerance thresholds (as fraction of mesh bounding box diagonal)
ON_SURFACE_TOLERANCE_FRACTION = 0.05      # 5% - for determining if vertex is ON a surface
MAX_REPROJECT_DISTANCE_FRACTION = 0.15    # 15% - for allowing re-projection to nearby surfaces
INNER_BOUNDARY_THRESHOLD_FRACTION = 0.02  # 2% - for classifying boundary vertices as inner

# Edge weight computation
EDGE_WEIGHT_EPSILON = 0.25  # Minimum edge weight to prevent division issues

# Minimum triangle area threshold (as fraction of median area)
MIN_TRIANGLE_AREA_FRACTION = 0.01  # Triangles with area < 1% of median are considered degenerate


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PropagationResult:
    """Result of surface propagation."""
    
    # The propagated surface mesh
    mesh: Optional[trimesh.Trimesh] = None
    
    # Original mesh before propagation
    original_mesh: Optional[trimesh.Trimesh] = None
    
    # Statistics
    original_vertices: int = 0
    original_faces: int = 0
    final_vertices: int = 0
    final_faces: int = 0
    
    # Removed islands
    islands_removed: int = 0
    triangles_removed: int = 0
    
    # Propagation stats
    boundary_vertices_extended: int = 0
    new_faces_added: int = 0
    
    # Timing
    cleanup_time_ms: float = 0.0
    propagation_time_ms: float = 0.0
    total_time_ms: float = 0.0


@dataclass
class SmoothingResult:
    """Result of membrane smoothing."""
    
    # The smoothed surface mesh
    mesh: Optional[trimesh.Trimesh] = None
    
    # Original mesh before smoothing
    original_mesh: Optional[trimesh.Trimesh] = None
    
    # Statistics
    original_vertices: int = 0
    final_vertices: int = 0
    
    # Boundary stats
    boundary_vertices: int = 0
    interior_vertices: int = 0
    
    # Iterations performed
    iterations: int = 0
    damping_factor: float = 0.5
    
    # Timing
    total_time_ms: float = 0.0


# =============================================================================
# ISLAND REMOVAL
# =============================================================================

def remove_isolated_islands(
    mesh: trimesh.Trimesh,
    min_triangles: int = 3
) -> Tuple[trimesh.Trimesh, int, int]:
    """
    Remove connected components (islands) with fewer than min_triangles.
    
    Args:
        mesh: Input trimesh
        min_triangles: Minimum number of triangles to keep a component
    
    Returns:
        Tuple of (cleaned_mesh, islands_removed, triangles_removed)
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0, 0
    
    # Get connected components using face adjacency
    # trimesh splits by connectivity
    try:
        components = mesh.split(only_watertight=False)
    except Exception as e:
        logger.warning(f"Could not split mesh into components: {e}")
        return mesh, 0, 0
    
    if len(components) <= 1:
        # Single component or empty
        if len(components) == 1 and len(components[0].faces) < min_triangles:
            logger.info(f"Single component with {len(components[0].faces)} faces < {min_triangles}, removing all")
            return trimesh.Trimesh(), 1, len(mesh.faces)
        return mesh, 0, 0
    
    # Filter components by size
    kept_components = []
    islands_removed = 0
    triangles_removed = 0
    
    for comp in components:
        n_faces = len(comp.faces)
        if n_faces >= min_triangles:
            kept_components.append(comp)
        else:
            islands_removed += 1
            triangles_removed += n_faces
            logger.debug(f"Removing island with {n_faces} faces")
    
    if len(kept_components) == 0:
        logger.warning("All components removed! Returning empty mesh")
        return trimesh.Trimesh(), islands_removed, triangles_removed
    
    # Concatenate remaining components
    if len(kept_components) == 1:
        cleaned_mesh = kept_components[0]
    else:
        cleaned_mesh = trimesh.util.concatenate(kept_components)
    
    logger.info(f"Removed {islands_removed} islands ({triangles_removed} triangles), "
                f"kept {len(kept_components)} components ({len(cleaned_mesh.faces)} triangles)")
    
    return cleaned_mesh, islands_removed, triangles_removed


# =============================================================================
# BOUNDARY DETECTION
# =============================================================================

def find_boundary_edges(mesh: trimesh.Trimesh) -> List[Tuple[int, int]]:
    """
    Find boundary edges of a mesh (edges shared by only one face).
    
    Args:
        mesh: Input trimesh
    
    Returns:
        List of (v0, v1) vertex index pairs for boundary edges
    """
    if mesh is None or len(mesh.faces) == 0:
        return []
    
    # Count how many faces share each edge
    edge_face_count = {}
    
    for face in mesh.faces:
        # Each face has 3 edges
        edges = [
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0])),
        ]
        for edge in edges:
            edge_face_count[edge] = edge_face_count.get(edge, 0) + 1
    
    # Boundary edges are those shared by exactly one face
    boundary_edges = [edge for edge, count in edge_face_count.items() if count == 1]
    
    return boundary_edges


def find_boundary_vertices(mesh: trimesh.Trimesh) -> Set[int]:
    """
    Find all vertices that lie on boundary edges.
    
    Args:
        mesh: Input trimesh
    
    Returns:
        Set of vertex indices on the boundary
    """
    boundary_edges = find_boundary_edges(mesh)
    boundary_verts = set()
    for v0, v1 in boundary_edges:
        boundary_verts.add(v0)
        boundary_verts.add(v1)
    return boundary_verts


def get_ordered_boundary_loops(mesh: trimesh.Trimesh) -> List[List[int]]:
    """
    Get ordered boundary loops (closed chains of boundary vertices).
    
    Args:
        mesh: Input trimesh
    
    Returns:
        List of vertex index lists, each representing a closed boundary loop
    """
    boundary_edges = find_boundary_edges(mesh)
    
    if not boundary_edges:
        return []
    
    # Build adjacency for boundary vertices
    adj = {}
    for v0, v1 in boundary_edges:
        if v0 not in adj:
            adj[v0] = []
        if v1 not in adj:
            adj[v1] = []
        adj[v0].append(v1)
        adj[v1].append(v0)
    
    # Extract loops by traversing
    visited_edges = set()
    loops = []
    
    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue
        
        # Start a new loop
        loop = [start_edge[0]]
        current = start_edge[1]
        prev = start_edge[0]
        visited_edges.add(start_edge)
        
        while current != loop[0]:
            loop.append(current)
            # Find next vertex
            neighbors = adj.get(current, [])
            next_v = None
            for n in neighbors:
                edge = (min(current, n), max(current, n))
                if edge not in visited_edges:
                    next_v = n
                    visited_edges.add(edge)
                    break
            
            if next_v is None:
                # Dead end or already visited
                break
            
            prev = current
            current = next_v
        
        if len(loop) >= 3:
            loops.append(loop)
    
    return loops


# =============================================================================
# SURFACE PROPAGATION
# =============================================================================

def _snap_to_mesh_vertex(position: np.ndarray, mesh: trimesh.Trimesh, tolerance: float) -> Tuple[np.ndarray, bool]:
    """
    Try to snap a position to an existing vertex on the mesh if within tolerance.
    
    Returns:
        Tuple of (snapped_position, was_snapped)
    """
    if mesh is None or len(mesh.vertices) == 0:
        return position, False
    
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)
    dist, idx = tree.query(position)
    
    if dist <= tolerance:
        return mesh.vertices[idx].copy(), True
    return position, False


def propagate_to_target_surfaces(
    secondary_mesh: trimesh.Trimesh,
    primary_mesh: Optional[trimesh.Trimesh],
    part_mesh: Optional[trimesh.Trimesh],
    max_distance: float = 100.0,
    step_size: float = 0.5,
    max_iterations: int = 50
) -> Tuple[trimesh.Trimesh, int, int]:
    """
    Extend secondary surface boundaries to DIRECTLY CONNECT to the primary 
    parting surface or part mesh surface.
    
    Each boundary vertex is connected to its closest point ON the target surface,
    creating a watertight connection suitable for manifold volume creation.
    
    Algorithm:
    1. Find boundary loops of secondary mesh
    2. For each boundary vertex, find the closest point ON the primary/part surface
    3. Snap to existing mesh vertices when possible for better connectivity
    4. Create triangles that directly connect boundary to those exact surface points
    5. Fill any remaining gaps with additional triangles
    
    Args:
        secondary_mesh: The secondary parting surface to extend
        primary_mesh: The primary parting surface (target)
        part_mesh: The part mesh surface (alternative target)
        max_distance: Maximum distance to extend (vertices beyond this won't connect)
        step_size: Not used in direct connection mode
        max_iterations: Not used in direct connection mode
    
    Returns:
        Tuple of (extended_mesh, boundary_vertices_connected, new_faces_added)
    """
    if secondary_mesh is None or len(secondary_mesh.faces) == 0:
        return secondary_mesh, 0, 0
    
    if primary_mesh is None and part_mesh is None:
        logger.warning("No target surfaces provided for propagation")
        return secondary_mesh, 0, 0
    
    # Use trimesh's proximity query to find closest point ON the surface
    from trimesh.proximity import ProximityQuery
    from scipy.spatial import cKDTree
    
    primary_proximity = None
    part_proximity = None
    primary_vertex_tree = None
    part_vertex_tree = None
    
    if primary_mesh is not None and len(primary_mesh.faces) > 0:
        primary_proximity = ProximityQuery(primary_mesh)
        primary_vertex_tree = cKDTree(primary_mesh.vertices)
        logger.debug(f"Primary surface: {len(primary_mesh.faces)} faces, {len(primary_mesh.vertices)} vertices")
    
    if part_mesh is not None and len(part_mesh.faces) > 0:
        part_proximity = ProximityQuery(part_mesh)
        part_vertex_tree = cKDTree(part_mesh.vertices)
        logger.debug(f"Part surface: {len(part_mesh.faces)} faces")
    
    # Get boundary loops of secondary mesh
    boundary_loops = get_ordered_boundary_loops(secondary_mesh)
    
    if not boundary_loops:
        logger.info("No boundary loops found - mesh may be closed")
        return secondary_mesh, 0, 0
    
    logger.info(f"Found {len(boundary_loops)} boundary loops to connect to target surfaces")
    
    # Calculate snap tolerance based on mesh scale
    mesh_scale = np.linalg.norm(secondary_mesh.bounds[1] - secondary_mesh.bounds[0])
    snap_tolerance = mesh_scale * 0.01  # 1% of mesh size
    logger.debug(f"Snap tolerance: {snap_tolerance:.4f} (mesh scale: {mesh_scale:.2f})")
    
    # Start with original mesh data
    all_vertices = list(secondary_mesh.vertices)
    all_faces = list(secondary_mesh.faces)
    
    total_connected = 0
    total_new_faces = 0
    
    for loop_idx, loop in enumerate(boundary_loops):
        if len(loop) < 3:
            continue
        
        # Get boundary vertex positions
        boundary_positions = np.array([secondary_mesh.vertices[vi] for vi in loop])
        n_boundary = len(loop)
        
        # Find closest point on target surface for each boundary vertex
        target_positions = np.zeros_like(boundary_positions)
        target_distances = np.full(n_boundary, np.inf)
        target_source = ['none'] * n_boundary  # Track which surface each connects to
        target_mesh_ref = [None] * n_boundary  # Reference to which mesh
        
        for i in range(n_boundary):
            pos = boundary_positions[i]
            
            # Check primary surface
            if primary_proximity is not None:
                closest_pts, dists, _ = primary_proximity.on_surface([pos])
                if dists[0] < target_distances[i]:
                    target_distances[i] = dists[0]
                    target_positions[i] = closest_pts[0]
                    target_source[i] = 'primary'
                    target_mesh_ref[i] = primary_mesh
            
            # Check part surface (take closer one)
            if part_proximity is not None:
                closest_pts, dists, _ = part_proximity.on_surface([pos])
                if dists[0] < target_distances[i]:
                    target_distances[i] = dists[0]
                    target_positions[i] = closest_pts[0]
                    target_source[i] = 'part'
                    target_mesh_ref[i] = part_mesh
        
        # Try to snap target positions to existing vertices on target mesh
        snapped_count = 0
        for i in range(n_boundary):
            if target_mesh_ref[i] is not None:
                snapped_pos, was_snapped = _snap_to_mesh_vertex(
                    target_positions[i], target_mesh_ref[i], snap_tolerance
                )
                if was_snapped:
                    target_positions[i] = snapped_pos
                    snapped_count += 1
        
        if snapped_count > 0:
            logger.debug(f"Loop {loop_idx}: snapped {snapped_count}/{n_boundary} vertices to existing mesh vertices")
        
        # Filter: only connect vertices within max_distance
        connect_mask = target_distances <= max_distance
        n_to_connect = np.sum(connect_mask)
        
        if n_to_connect == 0:
            logger.debug(f"Loop {loop_idx}: no vertices within max_distance {max_distance}")
            continue
        
        logger.debug(f"Loop {loop_idx}: connecting {n_to_connect}/{n_boundary} vertices "
                    f"(distances: {target_distances[connect_mask].min():.4f} - {target_distances[connect_mask].max():.4f})")
        
        # Add target positions as new vertices
        new_vertex_start = len(all_vertices)
        for pos in target_positions:
            all_vertices.append(pos.copy())
        
        # Create triangles connecting boundary to target positions
        for i in range(n_boundary):
            next_i = (i + 1) % n_boundary
            
            # Original boundary vertices
            old_v0 = loop[i]
            old_v1 = loop[next_i]
            
            # New vertices on target surface
            new_v0 = new_vertex_start + i
            new_v1 = new_vertex_start + next_i
            
            # Skip if both vertices are too far (beyond max_distance)
            if not connect_mask[i] and not connect_mask[next_i]:
                continue
            
            # Calculate distances moved
            dist_v0 = np.linalg.norm(boundary_positions[i] - target_positions[i])
            dist_v1 = np.linalg.norm(boundary_positions[next_i] - target_positions[next_i])
            
            # If one vertex didn't move much but the other did, create a single triangle
            min_move_threshold = 1e-6
            
            if dist_v0 < min_move_threshold and dist_v1 < min_move_threshold:
                # Both already on surface, no triangle needed
                continue
            elif dist_v0 < min_move_threshold:
                # Only v1 moved - single triangle: old0, old1, new1
                all_faces.append([old_v0, old_v1, new_v1])
                total_new_faces += 1
            elif dist_v1 < min_move_threshold:
                # Only v0 moved - single triangle: old0, old1, new0
                all_faces.append([old_v0, old_v1, new_v0])
                total_new_faces += 1
            else:
                # Both moved - create quad as two triangles
                # Check if new vertices are very close (collapsed edge)
                if np.linalg.norm(target_positions[i] - target_positions[next_i]) < min_move_threshold:
                    # New edge collapsed - single triangle
                    all_faces.append([old_v0, old_v1, new_v0])
                    total_new_faces += 1
                else:
                    # Normal quad
                    all_faces.append([old_v0, old_v1, new_v1])
                    all_faces.append([old_v0, new_v1, new_v0])
                    total_new_faces += 2
        
        total_connected += n_to_connect
        
        # Log which surfaces vertices connected to
        primary_count = sum(1 for s in target_source if s == 'primary')
        part_count = sum(1 for s in target_source if s == 'part')
        logger.debug(f"Loop {loop_idx}: {primary_count} to primary, {part_count} to part surface")
    
    # Create the extended mesh
    if total_new_faces > 0:
        vertices_array = np.array(all_vertices)
        faces_array = np.array(all_faces)
        extended_mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
        
        # Merge vertices with a reasonable tolerance to close small gaps
        # Use a tolerance based on mesh scale
        merge_tolerance = mesh_scale * 0.001  # 0.1% of mesh size
        extended_mesh.merge_vertices(merge_tex=True, merge_norm=True)
        
        # Remove degenerate faces
        valid_faces = extended_mesh.nondegenerate_faces()
        if np.sum(~valid_faces) > 0:
            logger.debug(f"Removing {np.sum(~valid_faces)} degenerate faces")
            extended_mesh = trimesh.Trimesh(
                vertices=extended_mesh.vertices,
                faces=extended_mesh.faces[valid_faces]
            )
        
        extended_mesh.remove_unreferenced_vertices()
        
        # Check for remaining holes (informational only - don't fill per paper)
        remaining_boundaries = find_boundary_edges(extended_mesh)
        if remaining_boundaries:
            logger.debug(f"Extended mesh has {len(remaining_boundaries)} remaining boundary edges")
        
        
        logger.info(f"Extension complete: {total_connected} vertices connected to target surfaces, "
                   f"{total_new_faces} triangles added")
        
        return extended_mesh, total_connected, total_new_faces
    else:
        logger.info("No extension needed - boundary already on target surfaces")
        return secondary_mesh, 0, 0


# =============================================================================
# SURFACE SMOOTHING
# =============================================================================

def smooth_surface_laplacian(
    mesh: trimesh.Trimesh,
    iterations: int = 3,
    lambda_factor: float = 0.5,
    preserve_boundary: bool = True
) -> trimesh.Trimesh:
    """
    Apply Laplacian smoothing to a surface mesh.
    
    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing factor (0-1)
        preserve_boundary: If True, don't move boundary vertices
    
    Returns:
        Smoothed mesh
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh
    
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    
    # Find boundary vertices if preserving
    boundary_verts = set()
    if preserve_boundary:
        boundary_verts = find_boundary_vertices(mesh)
    
    # Build vertex adjacency
    n_verts = len(vertices)
    neighbors = [set() for _ in range(n_verts)]
    
    for f in faces:
        for i in range(3):
            v = f[i]
            neighbors[v].add(f[(i+1) % 3])
            neighbors[v].add(f[(i+2) % 3])
    
    # Smoothing iterations
    for _ in range(iterations):
        new_vertices = vertices.copy()
        
        for v in range(n_verts):
            if v in boundary_verts:
                continue
            
            if neighbors[v]:
                neighbor_list = list(neighbors[v])
                neighbor_positions = vertices[neighbor_list]
                avg_pos = neighbor_positions.mean(axis=0)
                new_vertices[v] = vertices[v] + lambda_factor * (avg_pos - vertices[v])
        
        vertices = new_vertices
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def smooth_membrane_with_boundary_reprojection(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: Optional[trimesh.Trimesh],
    hull_mesh: Optional[trimesh.Trimesh],
    primary_mesh: Optional[trimesh.Trimesh] = None,
    iterations: int = 5,
    damping_factor: float = 0.5,
    excluded_vertices: Optional[np.ndarray] = None,
    use_fast_smoothing: bool = True,
    vertex_boundary_type: Optional[np.ndarray] = None,
    reproject_interior_to_part: bool = True,
    edge_endpoint_0: Optional[np.ndarray] = None,
    edge_endpoint_1: Optional[np.ndarray] = None
) -> SmoothingResult:
    """
    Smooth the membrane surface following the algorithm from Section 4.4 of the paper.
    
    Per paper Section 4.4: "The smoothing is performed by alternating two main steps:
    1. First we smooth the polyline that includes the boundary vertices only.
       After this smooth step, we re-project those vertices onto the original
       surface of M or the external boundary ∂H, depending on which of the two
       surfaces they belonged to originally.
    2. Then, we smooth all the interior vertices, keeping the ones on the
       boundary fixed (the ones smoothed in the previous step).
    Each smoothing step is performed using a damping factor (0.5 in our experiments)
    to ensure a proper convergence to the final solution."
    
    EXTENSION (not in paper): We optionally add a 4th step that re-projects interior
    vertices that originated from the part surface (bt=-1) back to the part mesh.
    This helps maintain contact between the membrane and the part surface.
    
    IMPORTANT: The key phrase is "which of the two surfaces they belonged to ORIGINALLY".
    This means we must track vertex boundary origin during extraction, NOT re-classify
    by closest distance during smoothing:
    - INNER boundary vertices (on part surface M) must re-project to part M
    - OUTER boundary vertices (on hull boundary ∂H) must re-project to hull ∂H
    
    For type -2 vertices (inner-inner midpoints on concave/convex edges), we use
    EDGE-CONSTRAINED smoothing: the vertex can slide along its original edge line
    but not perpendicular to it. This handles concave edges correctly.
    
    Args:
        membrane_mesh: The triangulated cut surface C to smooth
        part_mesh: The part/object surface mesh M (boundary constraint)
        hull_mesh: The external boundary mesh ∂H (boundary constraint)
        primary_mesh: The primary parting surface (for secondary surface smoothing)
        iterations: Number of alternating smooth iterations
        damping_factor: Smoothing damping factor (0.5 per paper)
        excluded_vertices: Optional array of vertex indices to exclude from smoothing
                          (e.g., gap-fill vertices that should remain fixed)
        vertex_boundary_type: Array tracking original boundary type for each vertex
                             -2 = inner-inner midpoint (edge-constrained smoothing)
                             -1 = on part surface M (INNER boundary, re-project to part)
                              0 = interior (no boundary constraint)
                              1/2 = on hull boundary ∂H (OUTER boundary, re-project to hull)
                             If None, falls back to distance-based classification
        reproject_interior_to_part: If True, also re-project interior vertices with bt=-1
                                   to part mesh after interior smoothing. This is our 
                                   extension, not in the original paper. Default: True.
        edge_endpoint_0: (V, 3) array of first edge endpoint for type -2 vertices
        edge_endpoint_1: (V, 3) array of second edge endpoint for type -2 vertices
    
    Returns:
        SmoothingResult with smoothed mesh and statistics
    """
    import time
    from trimesh.proximity import ProximityQuery
    
    result = SmoothingResult()
    result.original_mesh = membrane_mesh
    result.iterations = iterations
    result.damping_factor = damping_factor
    
    if membrane_mesh is None or len(membrane_mesh.faces) == 0:
        logger.warning("No membrane mesh provided for smoothing")
        return result
    
    start_time = time.time()
    
    result.original_vertices = len(membrane_mesh.vertices)
    
    vertices = membrane_mesh.vertices.copy()
    faces = membrane_mesh.faces.copy()
    n_verts = len(vertices)
    
    # Build proximity queries for re-projection
    part_proximity = None
    hull_proximity = None
    primary_proximity = None
    
    if part_mesh is not None and len(part_mesh.faces) > 0:
        part_proximity = ProximityQuery(part_mesh)
        logger.debug(f"Part mesh for re-projection: {len(part_mesh.faces)} faces")
    
    if hull_mesh is not None and len(hull_mesh.faces) > 0:
        hull_proximity = ProximityQuery(hull_mesh)
        logger.debug(f"Hull mesh for re-projection: {len(hull_mesh.faces)} faces")
    
    if primary_mesh is not None and len(primary_mesh.faces) > 0:
        primary_proximity = ProximityQuery(primary_mesh)
        logger.debug(f"Primary mesh for re-projection: {len(primary_mesh.faces)} faces")
    
    # Find boundary vertices and classify which surface they belong to
    boundary_edges = find_boundary_edges(trimesh.Trimesh(vertices=vertices, faces=faces))
    boundary_verts = set()
    for v0, v1 in boundary_edges:
        boundary_verts.add(v0)
        boundary_verts.add(v1)
    
    # Build set of excluded vertices (e.g., gap-fill vertices that shouldn't be smoothed)
    excluded_set = set()
    if excluded_vertices is not None and len(excluded_vertices) > 0:
        excluded_set = set(excluded_vertices.tolist())
        logger.info(f"Excluding {len(excluded_set)} vertices from smoothing (gap-fill vertices)")
    
    boundary_verts = sorted(boundary_verts)
    # Interior verts are those not on boundary AND not excluded
    interior_verts = [v for v in range(n_verts) if v not in boundary_verts and v not in excluded_set]
    
    result.boundary_vertices = len(boundary_verts)
    result.interior_vertices = len(interior_verts)
    
    logger.info(f"Smoothing membrane: {n_verts} vertices "
               f"({len(boundary_verts)} boundary, {len(interior_verts)} interior, {len(excluded_set)} excluded)")
    
    # Classify boundary vertices: which surface do they belong to?
    # Per paper Section 4.4: re-project to "the original surface ... they belonged to originally"
    # 
    # Use vertex_boundary_type from extraction if available:
    #   -1 = on part surface M (INNER boundary) → re-project to part
    #    0 = interior midpoint OR boundary zone vertex on hull
    #   1/2 = on hull boundary ∂H (OUTER boundary) → re-project to hull
    #
    # IMPORTANT: The boundary zone (label 0 on tet boundary) is still part of the 
    # hull surface ∂H, just not classified as H1 or H2. Membrane boundary vertices
    # with type 0 that are on the membrane boundary should be re-projected to hull
    # since the boundary zone is ON the hull, not interior.
    #
    # FALLBACK: Distance-based classification for unclassified vertices
    
    boundary_surface = {}  # vertex index -> 'part', 'hull', 'primary', or 'patch'
    
    mesh_scale = np.linalg.norm(membrane_mesh.bounds[1] - membrane_mesh.bounds[0])
    
    # Check if we have proper boundary type information from extraction
    use_boundary_type_classification = (
        vertex_boundary_type is not None and 
        len(vertex_boundary_type) == n_verts
    )
    
    if use_boundary_type_classification:
        # === CLASSIFICATION FROM EXTRACTION ===
        # Use the vertex_boundary_type from extraction (per paper Section 4.4)
        logger.info("Using vertex_boundary_type for boundary classification (per paper Section 4.4)")
        
        for vi in boundary_verts:
            bt = vertex_boundary_type[vi]
            if bt == -2:
                # INNER-INNER MIDPOINT - midpoint of edge where BOTH endpoints touch part M
                # These vertices should NOT be reprojected because:
                # 1. They're already very close to part M (both endpoints are ON M)
                # 2. Reprojecting to nearest surface point moves them OFF their edge line
                # 3. On curved surfaces, this creates gaps in the membrane
                # Mark as 'fixed' to exclude from reprojection entirely
                boundary_surface[vi] = 'fixed'
            elif bt == -1:
                # INNER boundary - originally on part surface M
                boundary_surface[vi] = 'part'
            elif bt in (1, 2):
                # OUTER boundary - originally on hull boundary ∂H (H1 or H2)
                boundary_surface[vi] = 'hull'
            elif bt == 0:
                # Type 0 = either:
                # 1. Interior midpoint of an edge where both tet verts were interior/boundary zone
                # 2. Boundary zone vertex on hull (grey zone between H1 and H2)
                #
                # Since this vertex is on the MEMBRANE BOUNDARY (not interior), it should
                # be re-projected to the hull. The boundary zone is ON the hull surface,
                # not inside the volume. Mark as hull for re-projection.
                boundary_surface[vi] = 'hull'
            else:
                # Unexpected value - use fallback
                boundary_surface[vi] = 'patch'
        
        part_count = sum(1 for s in boundary_surface.values() if s == 'part')
        hull_count = sum(1 for s in boundary_surface.values() if s == 'hull')
        fixed_count = sum(1 for s in boundary_surface.values() if s == 'fixed')
        patch_count = sum(1 for s in boundary_surface.values() if s == 'patch')
        
        logger.info(f"Boundary classification (from extraction): {part_count} to part (inner), "
                   f"{hull_count} to hull (outer, including boundary zone), "
                   f"{fixed_count} fixed (inner-inner midpoint), {patch_count} patch (needs fallback)")
        
        # For 'patch' vertices, use NEIGHBOR PROPAGATION first, then distance as last resort
        # This is important because the inflated hull CONTAINS the part mesh, so interior-
        # turned-boundary vertices may be closer to part by distance but topologically
        # belong to the outer (hull) boundary chain.
        if patch_count > 0:
            from trimesh.proximity import ProximityQuery
            
            # Build boundary adjacency to find neighbors along boundary edges
            boundary_adjacency = {v: set() for v in boundary_verts}
            for v0, v1 in boundary_edges:
                boundary_adjacency[v0].add(v1)
                boundary_adjacency[v1].add(v0)
            
            # STEP 1: Propagate from already-classified vertices to their patch neighbors
            # This uses the boundary chain topology rather than distance
            # Use CONVERGENCE-BASED stopping: continue until no changes occur
            # Maximum rounds is set very high as a safety limit only
            propagated_count = 0
            changed = True
            max_propagation_rounds = len(boundary_verts)  # Worst case: propagate through entire chain
            propagation_round = 0
            
            while changed and propagation_round < max_propagation_rounds:
                changed = False
                propagation_round += 1
                for vi in list(boundary_verts):
                    if boundary_surface.get(vi) != 'patch':
                        continue
                    
                    # Look at neighbors along boundary edges
                    neighbors = boundary_adjacency.get(vi, set())
                    neighbor_types = [boundary_surface.get(n) for n in neighbors if n in boundary_surface]
                    
                    # Count non-patch classifications (including propagated ones)
                    part_neighbors = sum(1 for t in neighbor_types if t in ('part', 'closest_part', 'propagated_part'))
                    hull_neighbors = sum(1 for t in neighbor_types if t in ('hull', 'closest_hull', 'propagated_hull'))
                    
                    # If ALL classified neighbors agree, propagate that classification
                    if part_neighbors > 0 and hull_neighbors == 0:
                        boundary_surface[vi] = 'propagated_part'
                        changed = True
                        propagated_count += 1
                    elif hull_neighbors > 0 and part_neighbors == 0:
                        boundary_surface[vi] = 'propagated_hull'
                        changed = True
                        propagated_count += 1
                    # If neighbors disagree (mixed), leave as 'patch' for distance fallback
            
            logger.info(f"Propagation classified {propagated_count} patch vertices in {propagation_round} rounds "
                       f"(converged: {not changed or propagation_round < max_propagation_rounds})")
            
            # STEP 2: For remaining 'patch' vertices, use distance-based fallback
            remaining_patches = [vi for vi in boundary_verts if boundary_surface.get(vi) == 'patch']
            
            if len(remaining_patches) > 0 and (part_mesh is not None or hull_mesh is not None):
                patch_positions = vertices[remaining_patches]
                
                # Build proximity queries
                if part_mesh is not None and len(part_mesh.faces) > 0:
                    part_prox = ProximityQuery(part_mesh)
                    _, dist_to_part, _ = part_prox.on_surface(patch_positions)
                else:
                    dist_to_part = np.full(len(remaining_patches), np.inf)
                
                if hull_mesh is not None and len(hull_mesh.faces) > 0:
                    hull_prox = ProximityQuery(hull_mesh)
                    _, dist_to_hull, _ = hull_prox.on_surface(patch_positions)
                else:
                    dist_to_hull = np.full(len(remaining_patches), np.inf)
                
                for i, vi in enumerate(remaining_patches):
                    if dist_to_part[i] < dist_to_hull[i]:
                        boundary_surface[vi] = 'closest_part'
                    else:
                        boundary_surface[vi] = 'closest_hull'
                
                logger.info(f"Distance fallback for {len(remaining_patches)} remaining patch vertices: "
                           f"{sum(1 for vi in remaining_patches if boundary_surface[vi] == 'closest_part')} to part, "
                           f"{sum(1 for vi in remaining_patches if boundary_surface[vi] == 'closest_hull')} to hull")
            
            # Summary of all fallback classifications
            total_part = sum(1 for vi in boundary_verts if boundary_surface.get(vi) in ('propagated_part', 'closest_part'))
            total_hull = sum(1 for vi in boundary_verts if boundary_surface.get(vi) in ('propagated_hull', 'closest_hull'))
            logger.info(f"Total fallback classification: {total_part} to part, {total_hull} to hull")
        
        primary_count = 0  # Not used for primary surfaces
        closest_count = sum(1 for s in boundary_surface.values() if s.startswith('closest_'))
        
    else:
        # === FALLBACK: DISTANCE-BASED CLASSIFICATION ===
        # This is the old method - may cause incorrect re-projection
        logger.warning("No vertex_boundary_type provided - using distance-based classification (may be inaccurate)")
        
        # Tight tolerance for classification (which surface is this vertex ON)
        on_surface_tolerance = mesh_scale * ON_SURFACE_TOLERANCE_FRACTION
        
        # Loose tolerance for re-projection (allow bridging gaps)
        max_reproject_distance = mesh_scale * MAX_REPROJECT_DISTANCE_FRACTION
        
        # OPTIMIZATION: Batch all distance queries instead of one-by-one
        boundary_verts_array = np.array(boundary_verts, dtype=np.int64)
        boundary_positions = vertices[boundary_verts_array]
        n_boundary = len(boundary_verts_array)
        
        # Compute distances to all surfaces in batch
        dist_to_part = np.full(n_boundary, np.inf)
        dist_to_hull = np.full(n_boundary, np.inf)
        dist_to_primary = np.full(n_boundary, np.inf)
        
        if part_proximity is not None:
            _, dist_to_part, _ = part_proximity.on_surface(boundary_positions)
        
        if hull_proximity is not None:
            _, dist_to_hull, _ = hull_proximity.on_surface(boundary_positions)
        
        if primary_proximity is not None:
            _, dist_to_primary, _ = primary_proximity.on_surface(boundary_positions)
        
        # Classify each boundary vertex based on distances
        for i, vi in enumerate(boundary_verts):
            d_part = dist_to_part[i]
            d_hull = dist_to_hull[i]
            d_primary = dist_to_primary[i]
            
            # First, check if vertex is clearly ON one surface (within tight tolerance)
            # Priority: primary > part > hull (for secondary surfaces touching primary)
            if primary_proximity is not None and d_primary < on_surface_tolerance:
                boundary_surface[vi] = 'primary'
            elif d_part < on_surface_tolerance:
                boundary_surface[vi] = 'part'
            elif d_hull < on_surface_tolerance:
                boundary_surface[vi] = 'hull'
            else:
                # Not clearly ON any surface - find the CLOSEST surface within max distance
                min_dist = min(d_part, d_hull, d_primary if primary_proximity else np.inf)
                
                if min_dist < max_reproject_distance:
                    if d_part == min_dist:
                        boundary_surface[vi] = 'closest_part'
                    elif d_hull == min_dist:
                        boundary_surface[vi] = 'closest_hull'
                    else:
                        boundary_surface[vi] = 'closest_primary'
                else:
                    boundary_surface[vi] = 'patch'
        
        part_count = sum(1 for s in boundary_surface.values() if s in ('part', 'closest_part'))
        hull_count = sum(1 for s in boundary_surface.values() if s in ('hull', 'closest_hull'))
        primary_count = sum(1 for s in boundary_surface.values() if s in ('primary', 'closest_primary'))
        patch_count = sum(1 for s in boundary_surface.values() if s == 'patch')
        closest_count = sum(1 for s in boundary_surface.values() if s.startswith('closest_'))
        
        # Log distance statistics for debugging
        if part_proximity is not None:
            logger.info(f"Part distances - min: {dist_to_part.min():.4f}, max: {dist_to_part.max():.4f}, "
                       f"mean: {dist_to_part.mean():.4f}")
        if hull_proximity is not None:
            logger.info(f"Hull distances - min: {dist_to_hull.min():.4f}, max: {dist_to_hull.max():.4f}, "
                       f"mean: {dist_to_hull.mean():.4f}")
        if primary_proximity is not None:
            logger.info(f"Primary distances - min: {dist_to_primary.min():.4f}, max: {dist_to_primary.max():.4f}, "
                       f"mean: {dist_to_primary.mean():.4f}")
        logger.info(f"Tolerances: on-surface={on_surface_tolerance:.4f}, max-reproject={max_reproject_distance:.4f} (mesh scale: {mesh_scale:.4f})")
        logger.info(f"Boundary classification (distance-based): {part_count} to part, "
                   f"{hull_count} to hull, {primary_count} to primary, {patch_count} patch ({closest_count} gap-bridging)")
    
    # Build vertex adjacency (along boundary and overall)
    vertex_neighbors = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            v = f[i]
            vertex_neighbors[v].add(f[(i+1) % 3])
            vertex_neighbors[v].add(f[(i+2) % 3])
    
    # Build boundary adjacency (neighbors along boundary edges only)
    boundary_neighbors = {v: set() for v in boundary_verts}
    for v0, v1 in boundary_edges:
        boundary_neighbors[v0].add(v1)
        boundary_neighbors[v1].add(v0)
    
    # Try to use fast C++ smoothing if available
    try:
        from .fast_algorithms_wrapper import (
            run_fast_boundary_interior_smoothing,
            is_cpp_available
        )
        can_use_fast = use_fast_smoothing and is_cpp_available()
    except ImportError:
        can_use_fast = False
    
    if can_use_fast:
        logger.info("Using fast C++ smoothing with OpenMP")
    else:
        logger.info("Using Python smoothing (rebuild C++ module for speedup)")
    
    # Pre-categorize boundary vertices for batched re-projection
    part_boundary_indices = []
    hull_boundary_indices = []
    primary_boundary_indices = []
    edge_constrained_indices = []  # Type -2: inner-inner midpoints (edge-constrained smoothing)
    
    for vi in boundary_verts:
        if vi in excluded_set:
            continue
        surface_type = boundary_surface.get(vi, 'patch')
        if surface_type == 'fixed':
            # Type -2 vertices (inner-inner midpoints on concave/convex edges)
            # Use EDGE-CONSTRAINED smoothing: vertex can slide along original edge
            # but not perpendicular to it. This handles concave edges correctly.
            edge_constrained_indices.append(vi)
        elif surface_type in ('part', 'closest_part', 'propagated_part'):
            part_boundary_indices.append(vi)
        elif surface_type in ('hull', 'closest_hull', 'propagated_hull'):
            hull_boundary_indices.append(vi)
        elif surface_type in ('primary', 'closest_primary'):
            primary_boundary_indices.append(vi)
    
    # Check if we have edge constraint data for edge-constrained smoothing
    has_edge_constraints = (edge_endpoint_0 is not None and 
                           edge_endpoint_1 is not None and
                           len(edge_endpoint_0) == n_verts)
    
    # If no edge constraints available, fall back to excluding fixed vertices from smoothing
    if len(edge_constrained_indices) > 0:
        if has_edge_constraints:
            logger.info(f"Using EDGE-CONSTRAINED smoothing for {len(edge_constrained_indices)} inner-inner midpoint vertices "
                       f"(type -2: on concave/convex edges of part M)")
        else:
            logger.warning(f"No edge constraint data - excluding {len(edge_constrained_indices)} inner-inner midpoints from smoothing")
            for vi in edge_constrained_indices:
                excluded_set.add(vi)
            edge_constrained_indices = []  # Clear since we're excluding them instead
    
    part_boundary_indices = np.array(part_boundary_indices, dtype=np.int64)
    hull_boundary_indices = np.array(hull_boundary_indices, dtype=np.int64)
    primary_boundary_indices = np.array(primary_boundary_indices, dtype=np.int64)
    edge_constrained_indices = np.array(edge_constrained_indices, dtype=np.int64)
    
    # NEW: Identify INTERIOR vertices that should be reprojected to part mesh
    # These are vertices created during inner-inner edge processing that ended up
    # as interior vertices in the parting surface mesh, not on its boundary.
    # They have vertex_boundary_type == -1 but are in interior_verts, not boundary_verts.
    interior_part_indices = []
    if vertex_boundary_type is not None and len(vertex_boundary_type) == n_verts:
        for vi in interior_verts:
            if vertex_boundary_type[vi] == -1:
                interior_part_indices.append(vi)
    interior_part_indices = np.array(interior_part_indices, dtype=np.int64)
    
    # NEW: Identify and REMOVE triangles that are LYING FLAT on the part mesh surface.
    # These triangles:
    # 1. Have all 3 vertices with bt == -1 (on part mesh)
    # 2. Have all 3 vertices at distance ≈ 0 from part
    # 3. Have their CENTROID also at distance ≈ 0 from part (this distinguishes flat triangles
    #    from triangles that have vertices on part but extend perpendicular as a membrane)
    #
    # Triangles lying flat on the part overlap with the part surface and don't contribute 
    # to the parting membrane - they should be filtered out.
    part_contact_tolerance = mesh_scale * 0.001  # 0.1% of mesh scale
    
    triangles_to_remove = set()
    if vertex_boundary_type is not None and len(vertex_boundary_type) == n_verts and part_proximity is not None:
        # First, compute distance to part mesh for all vertices
        _, all_dists, _ = part_proximity.on_surface(vertices)
        
        for fi, face in enumerate(faces):
            v0, v1, v2 = face
            bt0, bt1, bt2 = vertex_boundary_type[v0], vertex_boundary_type[v1], vertex_boundary_type[v2]
            
            # Check if all 3 vertices are on the part mesh (bt == -1)
            if bt0 == -1 and bt1 == -1 and bt2 == -1:
                # Check if all 3 vertices are very close to the part surface
                d0, d1, d2 = all_dists[v0], all_dists[v1], all_dists[v2]
                if d0 < part_contact_tolerance and d1 < part_contact_tolerance and d2 < part_contact_tolerance:
                    # Also check if the CENTROID is close to the part surface
                    # This distinguishes flat-lying triangles from perpendicular membrane triangles
                    centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
                    _, centroid_dist, _ = part_proximity.on_surface(centroid.reshape(1, 3))
                    
                    if centroid_dist[0] < part_contact_tolerance:
                        # This triangle lies flat on the part surface - mark for removal
                        triangles_to_remove.add(fi)
        
        if triangles_to_remove:
            logger.info(f"Removing {len(triangles_to_remove)} triangles that lie flat on part mesh surface")
            
            # Filter out the triangles
            keep_mask = np.ones(len(faces), dtype=bool)
            keep_mask[list(triangles_to_remove)] = False
            faces = faces[keep_mask]
            
            # Remove unreferenced vertices and update vertex_boundary_type
            referenced_verts = np.unique(faces.ravel())
            if len(referenced_verts) < n_verts:
                old_to_new = np.full(n_verts, -1, dtype=np.int64)
                old_to_new[referenced_verts] = np.arange(len(referenced_verts))
                
                vertices = vertices[referenced_verts]
                faces = old_to_new[faces]
                vertex_boundary_type = vertex_boundary_type[referenced_verts]
                n_verts = len(vertices)
                
                # Update boundary_surface dictionary with new indices
                new_boundary_surface = {}
                for old_vi, surface_type in boundary_surface.items():
                    if old_to_new[old_vi] >= 0:
                        new_boundary_surface[int(old_to_new[old_vi])] = surface_type
                boundary_surface = new_boundary_surface
                
                # Update boundary_verts, interior_verts, and indices
                boundary_verts_old = set(boundary_verts)
                boundary_verts = sorted([int(old_to_new[v]) for v in boundary_verts if old_to_new[v] >= 0])
                interior_verts = [v for v in range(n_verts) if v not in set(boundary_verts) and v not in excluded_set]
                
                # Update part/hull/primary boundary indices
                part_boundary_indices = np.array([old_to_new[v] for v in part_boundary_indices if old_to_new[v] >= 0], dtype=np.int64)
                hull_boundary_indices = np.array([old_to_new[v] for v in hull_boundary_indices if old_to_new[v] >= 0], dtype=np.int64)
                primary_boundary_indices = np.array([old_to_new[v] for v in primary_boundary_indices if old_to_new[v] >= 0], dtype=np.int64)
                interior_part_indices = np.array([old_to_new[v] for v in interior_part_indices if old_to_new[v] >= 0], dtype=np.int64)
                
                # Rebuild boundary edges and neighbors using a temporary mesh
                temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                boundary_edges = find_boundary_edges(temp_mesh)
                boundary_verts = sorted(set(v for edge in boundary_edges for v in edge))
                boundary_neighbors = {v: set() for v in boundary_verts}
                for v0, v1 in boundary_edges:
                    if v0 in boundary_neighbors:
                        boundary_neighbors[v0].add(v1)
                    if v1 in boundary_neighbors:
                        boundary_neighbors[v1].add(v0)
                
                # Rebuild vertex_neighbors
                vertex_neighbors = [set() for _ in range(n_verts)]
                for f in faces:
                    for i in range(3):
                        v = f[i]
                        vertex_neighbors[v].add(f[(i+1) % 3])
                        vertex_neighbors[v].add(f[(i+2) % 3])
                
                # Update interior_verts based on new boundary_verts
                interior_verts = [v for v in range(n_verts) if v not in set(boundary_verts) and v not in excluded_set]
                
                logger.info(f"After removing overlapping triangles: {n_verts} vertices, {len(faces)} faces, "
                           f"{len(boundary_verts)} boundary verts")
    
    # NOTE: We no longer exclude part-contact vertices from smoothing.
    # Instead, ALL part-boundary vertices are smoothed using SURFACE-CONSTRAINED smoothing,
    # which keeps them on the part surface while smoothing the polyline.
    # This ensures the polyline on the part mesh remains smooth.
    
    # Only exclude interior vertices that are completely surrounded by part-contact faces
    # (these are truly interior to a flat contact patch, not on the polyline)
    smoothing_excluded_vertices = set()
    
    # Don't add any vertices to excluded_set for smoothing purposes
    # (gap-fill vertices from earlier are already in excluded_set if needed)
    
    # Update interior_verts - don't exclude part vertices, they need interior smoothing too
    # Only exclude vertices that were explicitly marked as gap-fill
    
    logger.info(f"Batched re-projection: {len(part_boundary_indices)} part (reproject), "
               f"{len(hull_boundary_indices)} hull, {len(primary_boundary_indices)} primary, "
               f"{len(edge_constrained_indices)} edge-constrained (type -2), "
               f"{len(interior_part_indices)} interior-part")
    
    # Build set of edge-constrained vertices for edge-constrained smoothing
    edge_constrained_set = set(edge_constrained_indices.tolist()) if len(edge_constrained_indices) > 0 else set()
    
    # Build set of part-boundary vertices for surface-constrained smoothing
    part_boundary_set = set(part_boundary_indices.tolist()) if len(part_boundary_indices) > 0 else set()
    
    # Alternating smoothing iterations per paper Section 4.4:
    # "The smoothing is performed by alternating two main steps:
    #  1. First we smooth the polyline that includes the boundary vertices only.
    #     After this smooth step, we re-project those vertices onto the original 
    #     surface of M or the external boundary ∂H.
    #  2. Then, we smooth all the interior vertices, keeping the ones on the 
    #     boundary fixed (the ones smoothed in the previous step)."
    for iteration in range(iterations):
        if can_use_fast:
            # === Fast C++ smoothing: Step 1 - Smooth boundary vertices (polyline smoothing) ===
            # Note: C++ function only does boundary smoothing, not interior
            vertices = run_fast_boundary_interior_smoothing(
                vertices=vertices,
                faces=faces,
                boundary_vertex_indices=np.array(boundary_verts, dtype=np.int64),
                boundary_edges=boundary_edges,
                excluded_vertices=np.array(list(excluded_set), dtype=np.int64) if excluded_set else None,
                lambda_factor=damping_factor
            )
        else:
            # === Python fallback: Step 1 - Smooth boundary vertices (polyline smoothing) ===
            # Per paper Section 4.4: "First we smooth the polyline that includes the 
            # boundary vertices only."
            # 
            # IMPORTANT: Paper says to use standard Laplacian smoothing for the polyline,
            # then re-project AFTER. NOT surface-constrained smoothing during the smooth step.
            # This is a standard Laplacian smooth in 3D space for ALL boundary vertices.
            # 
            # EXTENSION: Type -2 vertices use EDGE-CONSTRAINED smoothing - they can slide
            # along their original edge line but not perpendicular to it.
            new_vertices = vertices.copy()
            
            for vi in boundary_verts:
                # Skip excluded vertices (gap-fill vertices, etc.)
                if vi in excluded_set:
                    continue
                
                neighbors = list(boundary_neighbors[vi])
                if len(neighbors) < 2:
                    continue
                    
                neighbor_positions = vertices[neighbors]
                avg_pos = neighbor_positions.mean(axis=0)
                target = vertices[vi] + damping_factor * (avg_pos - vertices[vi])
                
                # For type -2 (edge-constrained) vertices: project onto original edge line
                if vi in edge_constrained_set and has_edge_constraints:
                    # Project target onto the original edge line
                    p0 = edge_endpoint_0[vi]
                    p1 = edge_endpoint_1[vi]
                    edge_vec = p1 - p0
                    edge_len_sq = np.dot(edge_vec, edge_vec)
                    
                    if edge_len_sq > 1e-12:
                        # Project target onto edge line: t = ((target - p0) · edge_vec) / |edge_vec|²
                        t = np.dot(target - p0, edge_vec) / edge_len_sq
                        # Clamp t to [0, 1] to stay within edge segment
                        t = np.clip(t, 0.0, 1.0)
                        # Compute position on edge
                        new_vertices[vi] = p0 + t * edge_vec
                    # else: degenerate edge, skip
                else:
                    # Standard 3D Laplacian smoothing for ALL other boundary vertices
                    # Reprojection happens in the next step
                    new_vertices[vi] = target
            
            vertices = new_vertices
        
        # === Step 2: Re-project boundary vertices onto target surfaces ===
        # Per paper: "After this smooth step, we re-project those vertices onto the 
        # original surface of M or the external boundary ∂H"
        
        # Re-project to part mesh M (batched)
        if len(part_boundary_indices) > 0 and part_proximity is not None:
            part_positions = vertices[part_boundary_indices]
            closest_pts, dists, _ = part_proximity.on_surface(part_positions)
            
            # Log large movements that might cause degenerate triangles
            if iteration == 0:
                movements = np.linalg.norm(closest_pts - part_positions, axis=1)
                large_moves = movements > 0.1
                if np.any(large_moves):
                    logger.debug(f"Part boundary reprojection: {np.sum(large_moves)} vertices moved > 0.1, "
                               f"max move = {movements.max():.4f}")
            
            vertices[part_boundary_indices] = closest_pts
        
        # Re-project to hull mesh ∂H (batched)
        if len(hull_boundary_indices) > 0 and hull_proximity is not None:
            hull_positions = vertices[hull_boundary_indices]
            closest_pts, _, _ = hull_proximity.on_surface(hull_positions)
            vertices[hull_boundary_indices] = closest_pts
        
        # Re-project to primary mesh (for secondary surfaces)
        if len(primary_boundary_indices) > 0 and primary_proximity is not None:
            primary_positions = vertices[primary_boundary_indices]
            closest_pts, _, _ = primary_proximity.on_surface(primary_positions)
            vertices[primary_boundary_indices] = closest_pts
        
        # === Step 3: Smooth interior vertices (boundary vertices now fixed) ===
        # Per paper: "Then, we smooth all the interior vertices, keeping the ones 
        # on the boundary fixed"
        new_vertices = vertices.copy()
        
        for vi in interior_verts:
            neighbors = list(vertex_neighbors[vi])
            if neighbors:
                neighbor_positions = vertices[neighbors]
                avg_pos = neighbor_positions.mean(axis=0)
                new_vertices[vi] = vertices[vi] + damping_factor * (avg_pos - vertices[vi])
        
        vertices = new_vertices
        
        # === Step 4 (EXTENSION - not in paper): Re-project interior vertices that should be on part mesh ===
        # These are inner-inner edge vertices (both tet verts on part surface) that
        # ended up as interior vertices in the parting surface mesh. They were
        # projected to the part mesh during extraction but drift during interior
        # smoothing. Re-project them to maintain contact with the part surface.
        #
        # NOTE: This step is OUR EXTENSION and is NOT described in the paper.
        # The paper only describes: (1) boundary smooth, (2) boundary reproject, (3) interior smooth.
        if reproject_interior_to_part and len(interior_part_indices) > 0 and part_proximity is not None:
            interior_part_positions = vertices[interior_part_indices]
            closest_pts, _, _ = part_proximity.on_surface(interior_part_positions)
            vertices[interior_part_indices] = closest_pts
        
        if (iteration + 1) % 2 == 0:
            logger.debug(f"Smoothing iteration {iteration + 1}/{iterations} complete")
    
    # Create smoothed mesh
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Clean up any degenerate faces that might have been created
    valid_faces = smoothed_mesh.nondegenerate_faces()
    n_degenerate = np.sum(~valid_faces)
    if n_degenerate > 0:
        logger.warning(f"Removing {n_degenerate} degenerate faces after smoothing")
        
        # Log details about degenerate faces for debugging
        degenerate_indices = np.where(~valid_faces)[0]
        for idx in degenerate_indices[:10]:  # Limit to first 10
            face = faces[idx]
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge_lengths = [
                np.linalg.norm(v1 - v0),
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v0 - v2)
            ]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            bt_info = ""
            if vertex_boundary_type is not None:
                bt_info = f", bt=[{vertex_boundary_type[face[0]]}, {vertex_boundary_type[face[1]]}, {vertex_boundary_type[face[2]]}]"
            logger.warning(f"  Degenerate face {idx}: verts={face.tolist()}, "
                          f"edges=[{edge_lengths[0]:.4f}, {edge_lengths[1]:.4f}, {edge_lengths[2]:.4f}], "
                          f"area={area:.6f}{bt_info}")
        
        if n_degenerate > 10:
            logger.warning(f"  ... and {n_degenerate - 10} more degenerate faces")
        
        smoothed_mesh = trimesh.Trimesh(
            vertices=smoothed_mesh.vertices,
            faces=smoothed_mesh.faces[valid_faces]
        )
    
    result.mesh = smoothed_mesh
    result.final_vertices = len(smoothed_mesh.vertices)
    result.total_time_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Membrane smoothing complete: {iterations} iterations, "
               f"{result.total_time_ms:.1f}ms")
    
    return result


# =============================================================================
# MAIN PROPAGATION FUNCTION
# =============================================================================

def propagate_secondary_surface(
    secondary_mesh: trimesh.Trimesh,
    primary_mesh: Optional[trimesh.Trimesh] = None,
    part_mesh: Optional[trimesh.Trimesh] = None,
    min_island_triangles: int = 3,
    max_propagation_distance: float = 10.0,
    propagation_step_size: float = 0.5,
    smooth_iterations: int = 2,
    smooth_factor: float = 0.3
) -> PropagationResult:
    """
    Full pipeline for secondary surface propagation and smoothing.
    
    Steps:
    1. Remove isolated islands with < min_island_triangles
    2. Propagate boundaries to touch primary surface and part mesh
    3. Smooth the result
    
    Args:
        secondary_mesh: The secondary parting surface mesh
        primary_mesh: The primary parting surface (target for propagation)
        part_mesh: The part mesh (target for propagation)
        min_island_triangles: Minimum triangles to keep an island
        max_propagation_distance: Maximum distance to propagate
        propagation_step_size: Step size for propagation
        smooth_iterations: Number of smoothing iterations
        smooth_factor: Smoothing lambda factor
    
    Returns:
        PropagationResult with processed mesh and statistics
    """
    import time
    
    result = PropagationResult()
    result.original_mesh = secondary_mesh
    
    if secondary_mesh is None:
        logger.warning("No secondary mesh provided")
        return result
    
    result.original_vertices = len(secondary_mesh.vertices)
    result.original_faces = len(secondary_mesh.faces)
    
    total_start = time.time()
    
    # Step 1: Remove isolated islands
    cleanup_start = time.time()
    cleaned_mesh, islands_removed, triangles_removed = remove_isolated_islands(
        secondary_mesh, 
        min_triangles=min_island_triangles
    )
    result.cleanup_time_ms = (time.time() - cleanup_start) * 1000
    result.islands_removed = islands_removed
    result.triangles_removed = triangles_removed
    
    if cleaned_mesh is None or len(cleaned_mesh.faces) == 0:
        logger.warning("No triangles remaining after island removal")
        result.mesh = cleaned_mesh
        result.final_vertices = 0
        result.final_faces = 0
        result.total_time_ms = (time.time() - total_start) * 1000
        return result
    
    # Step 2: Propagate to target surfaces
    propagation_start = time.time()
    propagated_mesh, extended_verts, new_faces = propagate_to_target_surfaces(
        cleaned_mesh,
        primary_mesh,
        part_mesh,
        max_distance=max_propagation_distance,
        step_size=propagation_step_size
    )
    result.propagation_time_ms = (time.time() - propagation_start) * 1000
    result.boundary_vertices_extended = extended_verts
    result.new_faces_added = new_faces
    
    # Step 3: Smooth the result
    if smooth_iterations > 0 and propagated_mesh is not None and len(propagated_mesh.faces) > 0:
        propagated_mesh = smooth_surface_laplacian(
            propagated_mesh,
            iterations=smooth_iterations,
            lambda_factor=smooth_factor,
            preserve_boundary=True
        )
    
    result.mesh = propagated_mesh
    result.final_vertices = len(propagated_mesh.vertices) if propagated_mesh else 0
    result.final_faces = len(propagated_mesh.faces) if propagated_mesh else 0
    result.total_time_ms = (time.time() - total_start) * 1000
    
    logger.info(f"Surface propagation complete: "
                f"{result.original_faces} -> {result.final_faces} faces, "
                f"{result.islands_removed} islands removed, "
                f"{result.new_faces_added} faces added, "
                f"{result.total_time_ms:.1f}ms")
    
    return result
