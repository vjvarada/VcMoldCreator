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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import trimesh

logger = logging.getLogger(__name__)


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
# HOLE FILLING
# =============================================================================

def _fill_small_holes(mesh: trimesh.Trimesh, max_hole_edges: int = 6) -> trimesh.Trimesh:
    """
    Fill small holes in the mesh (holes with up to max_hole_edges boundary edges).
    
    Uses ear-clipping triangulation for small polygon holes.
    
    Args:
        mesh: Input mesh with holes
        max_hole_edges: Maximum number of edges in a hole to fill
    
    Returns:
        Mesh with small holes filled
    """
    boundary_loops = get_ordered_boundary_loops(mesh)
    
    if not boundary_loops:
        return mesh
    
    # Filter to small holes only
    small_holes = [loop for loop in boundary_loops if len(loop) <= max_hole_edges]
    
    if not small_holes:
        return mesh
    
    logger.debug(f"Filling {len(small_holes)} small holes (max {max_hole_edges} edges)")
    
    all_vertices = list(mesh.vertices)
    all_faces = list(mesh.faces)
    holes_filled = 0
    
    for loop in small_holes:
        n = len(loop)
        if n < 3:
            continue
        
        # Get vertex positions for the hole
        positions = np.array([mesh.vertices[vi] for vi in loop])
        
        # Compute hole normal (average of edge cross products)
        center = positions.mean(axis=0)
        normal = np.zeros(3)
        for i in range(n):
            v0 = positions[i] - center
            v1 = positions[(i + 1) % n] - center
            normal += np.cross(v0, v1)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-10:
            normal /= normal_len
        
        # Simple triangulation: fan from first vertex for small holes
        if n == 3:
            # Triangle - just add it
            all_faces.append([loop[0], loop[1], loop[2]])
            holes_filled += 1
        elif n == 4:
            # Quad - split into two triangles
            all_faces.append([loop[0], loop[1], loop[2]])
            all_faces.append([loop[0], loop[2], loop[3]])
            holes_filled += 1
        else:
            # Fan triangulation for 5-6 sided holes
            for i in range(1, n - 1):
                all_faces.append([loop[0], loop[i], loop[i + 1]])
            holes_filled += 1
    
    if holes_filled > 0:
        logger.info(f"Filled {holes_filled} small holes")
        filled_mesh = trimesh.Trimesh(
            vertices=np.array(all_vertices),
            faces=np.array(all_faces)
        )
        # Clean up
        valid_faces = filled_mesh.nondegenerate_faces()
        if np.sum(~valid_faces) > 0:
            filled_mesh = trimesh.Trimesh(
                vertices=filled_mesh.vertices,
                faces=filled_mesh.faces[valid_faces]
            )
        return filled_mesh
    
    return mesh


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
        
        # Check for remaining holes and try to fill small ones
        remaining_boundaries = find_boundary_edges(extended_mesh)
        if remaining_boundaries:
            logger.debug(f"Extended mesh has {len(remaining_boundaries)} remaining boundary edges")
            # Try to fill small holes (triangular or quad holes)
            extended_mesh = _fill_small_holes(extended_mesh, max_hole_edges=6)
        
        
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
    excluded_vertices: Optional[np.ndarray] = None
) -> SmoothingResult:
    """
    Smooth the membrane surface while preserving boundaries on the part mesh (M),
    hull boundary (∂H), and optionally the primary parting surface.
    
    Implements the algorithm from the paper:
    1. Smooth boundary vertices (polylines)
    2. Re-project boundary vertices onto their original surfaces (M, ∂H, or primary)
    3. Smooth interior vertices while keeping boundary vertices fixed
    4. Repeat for specified iterations
    
    Args:
        membrane_mesh: The triangulated cut surface C to smooth
        part_mesh: The part/object surface mesh M (boundary constraint)
        hull_mesh: The external boundary mesh ∂H (boundary constraint)
        primary_mesh: The primary parting surface (for secondary surface smoothing)
        iterations: Number of alternating smooth/re-project iterations
        damping_factor: Smoothing damping factor (0.5 recommended)
        excluded_vertices: Optional array of vertex indices to exclude from smoothing
                          (e.g., gap-fill vertices that should remain fixed)
    
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
    # We use TWO thresholds:
    # 1. A tight threshold to determine which surface the vertex is ON
    # 2. A looser threshold to allow re-projection even if not exactly on surface
    boundary_surface = {}  # vertex index -> 'part', 'hull', 'primary', or 'closest'
    
    mesh_scale = np.linalg.norm(membrane_mesh.bounds[1] - membrane_mesh.bounds[0])
    
    # Tight tolerance for classification (which surface is this vertex ON)
    on_surface_tolerance = mesh_scale * 0.05  # 5% of mesh size
    
    # Loose tolerance for re-projection (allow bridging gaps)
    # Any boundary vertex within this distance will be re-projected to closest surface
    max_reproject_distance = mesh_scale * 0.15  # 15% of mesh size - much more generous
    
    # Track distances for debugging
    all_part_dists = []
    all_hull_dists = []
    all_primary_dists = []
    
    for vi in boundary_verts:
        pos = vertices[vi]
        dist_to_part = np.inf
        dist_to_hull = np.inf
        dist_to_primary = np.inf
        
        if part_proximity is not None:
            _, d, _ = part_proximity.on_surface([pos])
            dist_to_part = d[0]
            all_part_dists.append(dist_to_part)
        
        if hull_proximity is not None:
            _, d, _ = hull_proximity.on_surface([pos])
            dist_to_hull = d[0]
            all_hull_dists.append(dist_to_hull)
        
        if primary_proximity is not None:
            _, d, _ = primary_proximity.on_surface([pos])
            dist_to_primary = d[0]
            all_primary_dists.append(dist_to_primary)
        
        # First, check if vertex is clearly ON one surface (within tight tolerance)
        # Priority: primary > part > hull (for secondary surfaces touching primary)
        # For primary surfaces: part > hull
        if primary_proximity is not None and dist_to_primary < on_surface_tolerance:
            boundary_surface[vi] = 'primary'
        elif dist_to_part < on_surface_tolerance:
            boundary_surface[vi] = 'part'
        elif dist_to_hull < on_surface_tolerance:
            boundary_surface[vi] = 'hull'
        else:
            # Not clearly ON any surface - find the CLOSEST surface within max distance
            # This handles corners and gap-bridging cases
            min_dist = min(dist_to_part, dist_to_hull, dist_to_primary if primary_proximity else np.inf)
            
            if min_dist < max_reproject_distance:
                # Re-project to closest surface to bridge the gap
                if dist_to_part == min_dist:
                    boundary_surface[vi] = 'closest_part'
                elif dist_to_hull == min_dist:
                    boundary_surface[vi] = 'closest_hull'
                else:
                    boundary_surface[vi] = 'closest_primary'
            else:
                # Too far from all surfaces - leave as 'patch'
                boundary_surface[vi] = 'patch'
    
    part_count = sum(1 for s in boundary_surface.values() if s in ('part', 'closest_part'))
    hull_count = sum(1 for s in boundary_surface.values() if s in ('hull', 'closest_hull'))
    primary_count = sum(1 for s in boundary_surface.values() if s in ('primary', 'closest_primary'))
    patch_count = sum(1 for s in boundary_surface.values() if s == 'patch')
    closest_count = sum(1 for s in boundary_surface.values() if s.startswith('closest_'))
    
    # Log distance statistics for debugging
    if all_part_dists:
        logger.info(f"Part distances - min: {min(all_part_dists):.4f}, max: {max(all_part_dists):.4f}, "
                   f"mean: {np.mean(all_part_dists):.4f}")
    if all_hull_dists:
        logger.info(f"Hull distances - min: {min(all_hull_dists):.4f}, max: {max(all_hull_dists):.4f}, "
                   f"mean: {np.mean(all_hull_dists):.4f}")
    if all_primary_dists:
        logger.info(f"Primary distances - min: {min(all_primary_dists):.4f}, max: {max(all_primary_dists):.4f}, "
                   f"mean: {np.mean(all_primary_dists):.4f}")
    logger.info(f"Tolerances: on-surface={on_surface_tolerance:.4f}, max-reproject={max_reproject_distance:.4f} (mesh scale: {mesh_scale:.4f})")
    logger.info(f"Boundary classification: {part_count} to part, "
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
    
    # Alternating smoothing iterations
    for iteration in range(iterations):
        # === Step 1: Smooth boundary vertices (polyline smoothing) ===
        new_vertices = vertices.copy()
        
        for vi in boundary_verts:
            # Skip excluded vertices (gap-fill vertices)
            if vi in excluded_set:
                continue
            neighbors = list(boundary_neighbors[vi])
            if len(neighbors) >= 2:
                # Average of neighbors along the boundary
                neighbor_positions = vertices[neighbors]
                avg_pos = neighbor_positions.mean(axis=0)
                new_vertices[vi] = vertices[vi] + damping_factor * (avg_pos - vertices[vi])
        
        vertices = new_vertices
        
        # === Step 2: Re-project boundary vertices onto target surfaces ===
        # Per paper algorithm: after smoothing boundary polylines, re-project to M or ∂H
        # - 'part': Boundary vertices near part mesh → re-project to part
        # - 'hull': Boundary vertices near hull mesh → re-project to hull
        # - 'primary': Boundary vertices near primary surface → re-project to primary
        # - 'patch': Patch boundaries → NO re-projection
        for vi in boundary_verts:
            # Skip excluded vertices (gap-fill vertices)
            if vi in excluded_set:
                continue
            surface_type = boundary_surface[vi]
            pos = vertices[vi]
            
            if surface_type in ('part', 'closest_part') and part_proximity is not None:
                closest_pts, _, _ = part_proximity.on_surface([pos])
                vertices[vi] = closest_pts[0]
            elif surface_type in ('hull', 'closest_hull') and hull_proximity is not None:
                closest_pts, _, _ = hull_proximity.on_surface([pos])
                vertices[vi] = closest_pts[0]
            elif surface_type in ('primary', 'closest_primary') and primary_proximity is not None:
                closest_pts, _, _ = primary_proximity.on_surface([pos])
                vertices[vi] = closest_pts[0]
            # 'patch' vertices are not re-projected
        
        # === Step 3: Smooth interior vertices (boundary vertices fixed) ===
        # Note: interior_verts already excludes excluded_set
        new_vertices = vertices.copy()
        
        for vi in interior_verts:
            neighbors = list(vertex_neighbors[vi])
            if neighbors:
                neighbor_positions = vertices[neighbors]
                avg_pos = neighbor_positions.mean(axis=0)
                new_vertices[vi] = vertices[vi] + damping_factor * (avg_pos - vertices[vi])
        
        vertices = new_vertices
        
        if (iteration + 1) % 2 == 0:
            logger.debug(f"Smoothing iteration {iteration + 1}/{iterations} complete")
    
    # Create smoothed mesh
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Clean up any degenerate faces that might have been created
    valid_faces = smoothed_mesh.nondegenerate_faces()
    if np.sum(~valid_faces) > 0:
        logger.debug(f"Removing {np.sum(~valid_faces)} degenerate faces after smoothing")
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
