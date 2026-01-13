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

# Feature detection thresholds for sharp edge/corner classification
SHARP_EDGE_ANGLE_THRESHOLD = 30.0  # degrees - edges sharper than this are classified as "sharp"
CORNER_SHARP_EDGE_COUNT = 2        # A vertex with >= this many sharp edges meeting is a "corner"


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


@dataclass
class FeatureDebugVisualization:
    """Debug visualization data for sharp edge/corner feature classification."""
    
    # Target mesh (part or hull) feature data
    target_mesh_vertices: Optional[np.ndarray] = None       # (N, 3) vertex positions
    target_feature_types: Optional[np.ndarray] = None       # (N,) 0=smooth, 1=sharp_edge, 2=corner
    target_sharp_edges: Optional[List[Tuple[int, int]]] = None  # List of (v0, v1) edge indices
    
    # Membrane boundary vertex data  
    membrane_boundary_positions: Optional[np.ndarray] = None  # (M, 3) boundary vertex positions
    membrane_boundary_projected_types: Optional[np.ndarray] = None  # (M,) feature type each projects to
    membrane_boundary_nearest_target: Optional[np.ndarray] = None  # (M,) nearest target vertex index


def get_feature_debug_visualization(
    target_mesh: trimesh.Trimesh,
    membrane_mesh: trimesh.Trimesh,
    membrane_boundary_indices: np.ndarray = None,
    angle_threshold: float = SHARP_EDGE_ANGLE_THRESHOLD
) -> FeatureDebugVisualization:
    """
    Generate debug visualization data for feature classification.
    
    This function returns data needed to visualize:
    1. Which vertices on the target mesh are classified as sharp edges vs corners
    2. Which membrane boundary vertices will be projected to these features
    
    Args:
        target_mesh: The target mesh (part or hull) to classify features on
        membrane_mesh: The membrane mesh being smoothed
        membrane_boundary_indices: Indices of membrane boundary vertices (computed if not provided)
        angle_threshold: Dihedral angle threshold for sharp edge detection
        
    Returns:
        FeatureDebugVisualization with all data needed for visualization
    """
    from scipy.spatial import cKDTree
    
    debug = FeatureDebugVisualization()
    
    if target_mesh is None or len(target_mesh.faces) == 0:
        return debug
    
    # Classify target mesh vertices
    feature_types, sharp_edge_info = classify_mesh_vertex_features(
        target_mesh, angle_threshold
    )
    
    debug.target_mesh_vertices = target_mesh.vertices.copy()
    debug.target_feature_types = feature_types
    
    # Extract sharp edges from the target mesh
    sharp_edges = []
    edge_to_faces = {}
    for fi, face in enumerate(target_mesh.faces):
        for i in range(3):
            v0, v1 = face[i], face[(i+1) % 3]
            edge = (min(v0, v1), max(v0, v1))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) != 2:
            # Boundary edge - treat as sharp
            sharp_edges.append(edge)
            continue
        
        n0 = target_mesh.face_normals[face_indices[0]]
        n1 = target_mesh.face_normals[face_indices[1]]
        dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        
        if angle_deg > angle_threshold:
            sharp_edges.append(edge)
    
    debug.target_sharp_edges = sharp_edges
    
    # Compute membrane boundary indices if not provided
    if membrane_mesh is not None and membrane_boundary_indices is None:
        # Find boundary vertices by looking at edges that appear in only one face
        edge_count = {}
        for face in membrane_mesh.faces:
            for i in range(3):
                v0, v1 = face[i], face[(i+1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary_verts = set()
        for edge, count in edge_count.items():
            if count == 1:  # Boundary edge
                boundary_verts.add(edge[0])
                boundary_verts.add(edge[1])
        
        membrane_boundary_indices = np.array(list(boundary_verts), dtype=np.int64)
        logger.info(f"Feature debug: Computed {len(membrane_boundary_indices)} membrane boundary vertices")
    
    # Map membrane boundary vertices to target mesh features
    if membrane_mesh is not None and membrane_boundary_indices is not None and len(membrane_boundary_indices) > 0:
        target_kdtree = cKDTree(target_mesh.vertices)
        
        boundary_positions = membrane_mesh.vertices[membrane_boundary_indices]
        _, nearest_target_indices = target_kdtree.query(boundary_positions)
        
        # Get feature type for each membrane boundary vertex based on nearest target
        projected_types = feature_types[nearest_target_indices]
        
        debug.membrane_boundary_positions = boundary_positions.copy()
        debug.membrane_boundary_projected_types = projected_types
        debug.membrane_boundary_nearest_target = nearest_target_indices
        
        logger.info(
            f"Feature debug: {np.sum(projected_types == 0)} smooth, "
            f"{np.sum(projected_types == 1)} sharp edge, "
            f"{np.sum(projected_types == 2)} corner membrane boundary vertices"
        )
    
    return debug


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
# DEGENERATE TRIANGLE REMOVAL
# =============================================================================

def remove_on_surface_triangles(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    tolerance_fraction: float = 0.0005,
    vertex_boundary_type: Optional[np.ndarray] = None
) -> Tuple[trimesh.Trimesh, int, Optional[np.ndarray]]:
    """
    Remove degenerate triangles that lie entirely on the part mesh surface.
    
    These are triangles where ALL three vertices AND the centroid are within
    tolerance of the part mesh surface. Such triangles are typically artifacts
    from the marching tetrahedra extraction and don't contribute to the membrane's
    cutting function - they lie flat on M rather than spanning from it.
    
    This cleanup step should be run BEFORE smoothing to prevent these degenerate
    triangles from causing issues during Laplacian smoothing.
    
    Args:
        membrane_mesh: The primary surface membrane mesh
        part_mesh: The part mesh (M) to check against
        tolerance_fraction: Distance tolerance as fraction of mesh scale (default 0.5%)
        vertex_boundary_type: Optional per-vertex boundary type array from extraction
                              (will be filtered to match remaining vertices)
    
    Returns:
        Tuple of (cleaned_mesh, num_triangles_removed, updated_vertex_boundary_type)
    """
    if membrane_mesh is None or len(membrane_mesh.faces) == 0:
        return membrane_mesh, 0, vertex_boundary_type
    
    if part_mesh is None or len(part_mesh.faces) == 0:
        logger.warning("No part mesh provided for on-surface triangle removal")
        return membrane_mesh, 0, vertex_boundary_type
    
    from trimesh.proximity import ProximityQuery
    
    vertices = membrane_mesh.vertices
    faces = membrane_mesh.faces
    n_faces = len(faces)
    
    # Compute tolerance based on mesh scale
    mesh_scale = np.linalg.norm(membrane_mesh.bounds[1] - membrane_mesh.bounds[0])
    tolerance = mesh_scale * tolerance_fraction
    
    logger.info(f"Checking {n_faces} triangles for on-surface degeneracy "
               f"(tolerance: {tolerance:.4f}, {tolerance_fraction*100:.1f}% of mesh scale)")
    
    # Build proximity query for part mesh
    part_proximity = ProximityQuery(part_mesh)
    
    # Check distances for all vertices first
    _, vertex_distances, _ = part_proximity.on_surface(vertices)
    
    # For each face, check if all 3 vertices AND centroid are on the part mesh
    faces_to_keep = []
    
    for fi, face in enumerate(faces):
        v0, v1, v2 = face
        
        # Check vertex distances
        d0 = vertex_distances[v0]
        d1 = vertex_distances[v1]
        d2 = vertex_distances[v2]
        
        all_vertices_on_surface = (d0 < tolerance and d1 < tolerance and d2 < tolerance)
        
        if all_vertices_on_surface:
            # Also check centroid
            centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
            _, centroid_dist, _ = part_proximity.on_surface(centroid.reshape(1, 3))
            
            if centroid_dist[0] < tolerance:
                # This triangle is entirely on the part mesh - remove it
                continue
        
        faces_to_keep.append(fi)
    
    num_removed = n_faces - len(faces_to_keep)
    
    if num_removed == 0:
        logger.info("No on-surface degenerate triangles found")
        return membrane_mesh, 0, vertex_boundary_type
    
    # Build new mesh with only kept faces
    kept_faces = faces[faces_to_keep]
    
    # Find which vertices are still used
    used_vertices = np.unique(kept_faces.flatten())
    
    # Create vertex mapping from old to new indices
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[used_vertices] = np.arange(len(used_vertices))
    
    # Remap face indices
    new_faces = old_to_new[kept_faces]
    new_vertices = vertices[used_vertices]
    
    # Update vertex_boundary_type if provided
    new_boundary_type = None
    if vertex_boundary_type is not None and len(vertex_boundary_type) == len(vertices):
        new_boundary_type = vertex_boundary_type[used_vertices]
    
    cleaned_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    logger.info(f"Removed {num_removed} on-surface degenerate triangles "
               f"({n_faces} → {len(cleaned_mesh.faces)} faces)")
    
    return cleaned_mesh, num_removed, new_boundary_type


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


# =============================================================================
# FEATURE DETECTION FOR SHARP EDGES/CORNERS
# =============================================================================

def classify_mesh_vertex_features(
    mesh: trimesh.Trimesh,
    angle_threshold: float = SHARP_EDGE_ANGLE_THRESHOLD
) -> Tuple[np.ndarray, dict]:
    """
    Classify mesh vertices as smooth, sharp_edge, or corner.
    Also tracks whether sharp features are concave or convex.
    
    This is used for feature-preserving smoothing where:
    - Concave corner vertices should NOT be smoothed (keep fixed)
    - Convex corner vertices CAN be smoothed normally
    - Sharp edge vertices should only move along the edge direction
    - Smooth vertices can use normal Laplacian smoothing + surface reprojection
    
    Args:
        mesh: Input trimesh
        angle_threshold: Dihedral angle (degrees) above which an edge is considered sharp
    
    Returns:
        Tuple of:
        - vertex_feature_type: Array with values:
            0 = smooth
            1 = sharp_edge (convex)
            2 = corner (convex) 
            3 = sharp_edge (concave)
            4 = corner (concave)
        - sharp_edge_info: Dict mapping vertex index to list of (neighbor_vertex, edge_direction)
                          for sharp edge vertices, so they can be projected along the edge
    """
    n_verts = len(mesh.vertices)
    vertex_feature_type = np.zeros(n_verts, dtype=np.int8)
    sharp_edge_info = {}
    
    if mesh is None or len(mesh.faces) == 0:
        return vertex_feature_type, sharp_edge_info
    
    # Build edge -> faces adjacency
    edge_to_faces = {}
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            v0, v1 = face[i], face[(i+1) % 3]
            edge = (min(v0, v1), max(v0, v1))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    # Find sharp edges by checking dihedral angle, and track concavity
    sharp_edges = set()
    edge_is_concave = {}  # edge -> bool (True = concave, False = convex)
    
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) != 2:
            # Boundary edge - treat as sharp convex (can be smoothed)
            sharp_edges.add(edge)
            edge_is_concave[edge] = False
            continue
        
        # Get face normals and centroids
        n0 = mesh.face_normals[face_indices[0]]
        n1 = mesh.face_normals[face_indices[1]]
        
        # Compute dihedral angle
        dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        
        if angle_deg > angle_threshold:
            sharp_edges.add(edge)
            
            # Determine concavity using signed dihedral angle
            # The standard method: check if the opposite vertex of one face 
            # is "above" or "below" the plane of the other face
            v0, v1 = edge
            
            # Get the vertices in each face that are NOT on the shared edge
            face0 = mesh.faces[face_indices[0]]
            face1 = mesh.faces[face_indices[1]]
            opposite_v0 = [v for v in face0 if v != v0 and v != v1][0]
            opposite_v1 = [v for v in face1 if v != v0 and v != v1][0]
            
            # Get the positions
            p_opposite0 = mesh.vertices[opposite_v0]
            p_opposite1 = mesh.vertices[opposite_v1]
            p_edge0 = mesh.vertices[v0]
            
            # Check: is the opposite vertex of face1 "above" or "below" 
            # the plane of face0?
            # If it's on the SAME side as n0 points, the edge is CONVEX
            # If it's on the OPPOSITE side, the edge is CONCAVE
            vec_to_opp1 = p_opposite1 - p_edge0
            side = np.dot(vec_to_opp1, n0)
            
            # For CONVEX edge: opposite vertex of face1 is BELOW face0's plane
            #   (on opposite side from where n0 points) → side < 0
            # For CONCAVE edge: opposite vertex of face1 is ABOVE face0's plane
            #   (on same side as where n0 points) → side > 0
            is_concave = side > 0
            edge_is_concave[edge] = is_concave
    
    # Count sharp edges meeting at each vertex and track concavity
    vertex_sharp_edge_count = np.zeros(n_verts, dtype=np.int32)
    vertex_concave_edge_count = np.zeros(n_verts, dtype=np.int32)
    vertex_sharp_neighbors = {i: [] for i in range(n_verts)}
    
    for edge in sharp_edges:
        v0, v1 = edge
        vertex_sharp_edge_count[v0] += 1
        vertex_sharp_edge_count[v1] += 1
        
        if edge_is_concave.get(edge, False):
            vertex_concave_edge_count[v0] += 1
            vertex_concave_edge_count[v1] += 1
        
        # Store the neighbor and edge direction for sharp edge vertices
        edge_dir = mesh.vertices[v1] - mesh.vertices[v0]
        edge_dir_norm = edge_dir / (np.linalg.norm(edge_dir) + 1e-10)
        
        vertex_sharp_neighbors[v0].append((v1, edge_dir_norm))
        vertex_sharp_neighbors[v1].append((v0, -edge_dir_norm))
    
    # Classify vertices (now with concavity)
    # 0 = smooth, 1 = sharp_edge (convex), 2 = corner (convex)
    # 3 = sharp_edge (concave), 4 = corner (concave)
    for vi in range(n_verts):
        sharp_count = vertex_sharp_edge_count[vi]
        concave_count = vertex_concave_edge_count[vi]
        
        if sharp_count >= CORNER_SHARP_EDGE_COUNT:
            # Corner: where multiple sharp edges meet
            # It's a concave corner if ANY of the edges are concave
            if concave_count > 0:
                vertex_feature_type[vi] = 4  # concave corner
            else:
                vertex_feature_type[vi] = 2  # convex corner
        elif sharp_count == 1:
            # On a sharp edge (but not a corner)
            if concave_count > 0:
                vertex_feature_type[vi] = 3  # concave sharp edge
            else:
                vertex_feature_type[vi] = 1  # convex sharp edge
            sharp_edge_info[vi] = vertex_sharp_neighbors[vi]
        # else: smooth (remains 0)
    
    n_smooth = np.sum(vertex_feature_type == 0)
    n_sharp_convex = np.sum(vertex_feature_type == 1)
    n_corner_convex = np.sum(vertex_feature_type == 2)
    n_sharp_concave = np.sum(vertex_feature_type == 3)
    n_corner_concave = np.sum(vertex_feature_type == 4)
    logger.debug(f"Feature classification: {n_smooth} smooth, "
                f"{n_sharp_convex} sharp_edge(convex), {n_corner_convex} corner(convex), "
                f"{n_sharp_concave} sharp_edge(concave), {n_corner_concave} corner(concave)")
    
    return vertex_feature_type, sharp_edge_info


def project_to_sharp_edge(
    vertex_pos: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray
) -> np.ndarray:
    """
    Project a point onto a line segment (sharp edge).
    
    This is used for sharp edge reprojection where we constrain
    the vertex to move only along the edge, not off it.
    
    Args:
        vertex_pos: Current vertex position (3,)
        edge_start: Start point of the edge (3,)
        edge_end: End point of the edge (3,)
    
    Returns:
        Projected position on the edge segment (3,)
    """
    edge_vec = edge_end - edge_start
    edge_len_sq = np.dot(edge_vec, edge_vec)
    
    if edge_len_sq < 1e-12:
        # Degenerate edge - return start point
        return edge_start.copy()
    
    # Project onto line and clamp to segment
    t = np.dot(vertex_pos - edge_start, edge_vec) / edge_len_sq
    t = np.clip(t, 0.0, 1.0)
    
    return edge_start + t * edge_vec


def reproject_with_feature_awareness(
    vertices: np.ndarray,
    boundary_indices: np.ndarray,
    target_mesh: trimesh.Trimesh,
    feature_types: np.ndarray,
    sharp_edge_info: dict,
    search_radius: Optional[float] = None
) -> np.ndarray:
    """
    Re-project boundary vertices to target mesh with feature-awareness.
    
    For smooth regions: use nearest-point projection (standard)
    For sharp edges: project along the edge direction only
    For concave corners: keep fixed (no projection) - prevents drifting
    For convex corners: use standard projection (safe to smooth)
    
    This prevents vertices from jumping across concave features during smoothing.
    
    Feature type codes:
        0 = smooth
        1 = sharp_edge (convex)
        2 = corner (convex) - CAN be projected normally
        3 = sharp_edge (concave)
        4 = corner (concave) - should NOT be projected
    
    Args:
        vertices: (N, 3) all mesh vertices (membrane mesh)
        boundary_indices: Indices of boundary vertices to reproject (membrane mesh indices)
        target_mesh: The target mesh to project onto (part/hull mesh)
        feature_types: Per-vertex feature type for TARGET MESH
        sharp_edge_info: Dict from classify_mesh_vertex_features (for target mesh)
        search_radius: Optional max search distance (auto-computed if None)
    
    Returns:
        Updated vertices array (modified in place and returned)
    """
    from trimesh.proximity import ProximityQuery
    from scipy.spatial import cKDTree
    
    if target_mesh is None or len(target_mesh.faces) == 0:
        return vertices
    
    if len(boundary_indices) == 0:
        return vertices
    
    proximity = ProximityQuery(target_mesh)
    
    # Auto-compute search radius if not provided
    if search_radius is None:
        mesh_scale = np.linalg.norm(target_mesh.bounds[1] - target_mesh.bounds[0])
        search_radius = mesh_scale * 0.1  # 10% of mesh size
    
    # Build KD-tree for target mesh vertices to find nearest target vertex for each membrane vertex
    target_kdtree = cKDTree(target_mesh.vertices)
    
    # For each membrane boundary vertex, find its nearest target mesh vertex
    # to determine which feature type applies
    boundary_positions = vertices[boundary_indices]
    _, nearest_target_indices = target_kdtree.query(boundary_positions)
    
    # Build a KD-tree of ONLY CONCAVE corner vertices for radius-based corner detection
    # Only concave corners (type 4) need special handling; convex corners can be smoothed
    concave_corner_mask = (feature_types == 4)
    concave_corner_indices = np.where(concave_corner_mask)[0]
    corner_detection_radius = search_radius * 0.3  # Use 30% of search radius for corner detection
    
    concave_corner_kdtree = None
    if len(concave_corner_indices) > 0:
        corner_positions = target_mesh.vertices[concave_corner_indices]
        concave_corner_kdtree = cKDTree(corner_positions)
    
    # Separate membrane vertices by the feature type of their nearest target vertex
    # IMPORTANT: Use radius-based CONCAVE corner detection to avoid misclassifying vertices
    smooth_indices = []
    sharp_edge_indices = []
    concave_corner_indices_membrane = []
    
    for i, vi in enumerate(boundary_indices):
        pos = boundary_positions[i]
        
        # First, check if this vertex is within radius of ANY CONCAVE corner
        # Only concave corners need to be kept fixed; convex corners can move
        if concave_corner_kdtree is not None:
            corner_dist, _ = concave_corner_kdtree.query(pos)
            if corner_dist < corner_detection_radius:
                concave_corner_indices_membrane.append(vi)
                continue
        
        # Not near a concave corner - use nearest vertex classification
        target_vi = nearest_target_indices[i]
        ft = feature_types[target_vi]
        
        # Feature types: 0=smooth, 1=sharp_convex, 2=corner_convex, 3=sharp_concave, 4=corner_concave
        if ft == 0 or ft == 2:  # smooth OR convex corner - can use normal projection
            smooth_indices.append(vi)
        elif ft == 1 or ft == 3:  # sharp edge (convex or concave)
            sharp_edge_indices.append(vi)
        elif ft == 4:  # concave corner (nearest is concave corner)
            concave_corner_indices_membrane.append(vi)
    
    # === Smooth vertices: standard nearest-point with bounded search ===
    if len(smooth_indices) > 0:
        smooth_indices_arr = np.array(smooth_indices, dtype=np.int64)
        smooth_positions = vertices[smooth_indices_arr]
        closest_pts, distances, _ = proximity.on_surface(smooth_positions)
        
        # Only apply projection if within search radius
        within_radius = distances < search_radius
        vertices[smooth_indices_arr[within_radius]] = closest_pts[within_radius]
    
    # === Sharp edge vertices: project to nearest point ON an edge ===
    # OPTIMIZED: Use vectorized edge distance computation instead of O(E) loop per vertex
    if len(sharp_edge_indices) > 0:
        # Get target mesh edges
        target_edges = target_mesh.edges_unique
        edge_starts = target_mesh.vertices[target_edges[:, 0]]  # (E, 3)
        edge_ends = target_mesh.vertices[target_edges[:, 1]]    # (E, 3)
        edge_midpoints = (edge_starts + edge_ends) / 2          # (E, 3)
        
        # Build KD-tree of edge midpoints for fast spatial lookup
        from scipy.spatial import cKDTree
        edge_midpoint_tree = cKDTree(edge_midpoints)
        
        # For each sharp edge vertex, find nearby edges and pick the closest
        sharp_positions = vertices[sharp_edge_indices]
        
        # Query k nearest edge midpoints (edges within reasonable distance)
        k_neighbors = min(50, len(target_edges))  # Check up to 50 nearby edges
        _, nearby_edge_indices = edge_midpoint_tree.query(sharp_positions, k=k_neighbors)
        
        # Vectorized projection for all sharp edge vertices
        for i, vi in enumerate(sharp_edge_indices):
            pos = vertices[vi]
            candidate_edges = nearby_edge_indices[i] if k_neighbors > 1 else [nearby_edge_indices[i]]
            
            best_dist = np.inf
            best_proj = pos.copy()
            
            for edge_idx in candidate_edges:
                proj = project_to_sharp_edge(pos, edge_starts[edge_idx], edge_ends[edge_idx])
                dist = np.linalg.norm(proj - pos)
                
                if dist < best_dist:
                    best_dist = dist
                    best_proj = proj
            
            if best_dist < search_radius:
                vertices[vi] = best_proj
    
    # === Concave corner vertices: keep fixed (no projection) ===
    # Concave corners should not be smoothed or reprojected - they would drift away
    # Convex corners are handled in smooth_indices and can be reprojected normally
    n_concave_corner = len(concave_corner_indices_membrane)
    if n_concave_corner > 0:
        logger.debug(f"Keeping {n_concave_corner} concave corner vertices fixed (no reprojection)")
    
    return vertices


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
    vertex_boundary_type: Optional[np.ndarray] = None
) -> SmoothingResult:
    """
    Smooth the membrane surface following the algorithm from Section 4.4 of the paper.
    
    Per paper: "The smoothing is performed by alternating two main steps:
    1. First we smooth the polyline that includes the boundary vertices only.
       After this smooth step, we re-project those vertices onto the original
       surface of M or the external boundary ∂H, depending on which of the two
       surfaces they belonged to originally.
    2. Then, we smooth all the interior vertices, keeping the ones on the
       boundary fixed (the ones smoothed in the previous step).
    Each smoothing step is performed using a damping factor (0.5 in our experiments)
    to ensure a proper convergence to the final solution."
    
    IMPORTANT: The key phrase is "which of the two surfaces they belonged to ORIGINALLY".
    This means we must track vertex boundary origin during extraction, NOT re-classify
    by closest distance during smoothing:
    - INNER boundary vertices (on part surface M) must re-project to part M
    - OUTER boundary vertices (on hull boundary ∂H) must re-project to hull ∂H
    
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
                             -1 = on part surface M (INNER boundary, re-project to part)
                              0 = interior (no boundary constraint)
                              1/2 = on hull boundary ∂H (OUTER boundary, re-project to hull)
                             If None, falls back to distance-based classification
    
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
            if bt == -1:
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
        patch_count = sum(1 for s in boundary_surface.values() if s == 'patch')
        
        logger.info(f"Boundary classification (from extraction): {part_count} to part (inner), "
                   f"{hull_count} to hull (outer, including boundary zone), {patch_count} patch (needs fallback)")
        
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
    
    for vi in boundary_verts:
        if vi in excluded_set:
            continue
        surface_type = boundary_surface.get(vi, 'patch')
        if surface_type in ('part', 'closest_part', 'propagated_part'):
            part_boundary_indices.append(vi)
        elif surface_type in ('hull', 'closest_hull', 'propagated_hull'):
            hull_boundary_indices.append(vi)
        elif surface_type in ('primary', 'closest_primary'):
            primary_boundary_indices.append(vi)
    
    part_boundary_indices = np.array(part_boundary_indices, dtype=np.int64)
    hull_boundary_indices = np.array(hull_boundary_indices, dtype=np.int64)
    primary_boundary_indices = np.array(primary_boundary_indices, dtype=np.int64)
    
    logger.info(f"Batched re-projection: {len(part_boundary_indices)} part, "
               f"{len(hull_boundary_indices)} hull, {len(primary_boundary_indices)} primary")
    
    # === Feature detection for target meshes (for feature-aware reprojection) ===
    # This handles sharp concave edges where standard nearest-point projection can
    # jump to the wrong side of the feature. We detect sharp edges/corners on each
    # target mesh and use edge-based projection for sharp features.
    part_feature_types = None
    part_sharp_edge_info = None
    hull_feature_types = None
    hull_sharp_edge_info = None
    primary_feature_types = None
    primary_sharp_edge_info = None
    
    if part_mesh is not None and len(part_mesh.faces) > 0:
        part_feature_types, part_sharp_edge_info = classify_mesh_vertex_features(part_mesh)
        n_sharp_convex = np.sum(part_feature_types == 1)
        n_corner_convex = np.sum(part_feature_types == 2)
        n_sharp_concave = np.sum(part_feature_types == 3)
        n_corner_concave = np.sum(part_feature_types == 4)
        if n_sharp_convex + n_corner_convex + n_sharp_concave + n_corner_concave > 0:
            logger.info(f"Part mesh features: {n_sharp_convex + n_sharp_concave} sharp edges "
                       f"({n_sharp_concave} concave), {n_corner_convex + n_corner_concave} corners "
                       f"({n_corner_concave} concave)")
    
    if hull_mesh is not None and len(hull_mesh.faces) > 0:
        hull_feature_types, hull_sharp_edge_info = classify_mesh_vertex_features(hull_mesh)
        n_sharp_convex = np.sum(hull_feature_types == 1)
        n_corner_convex = np.sum(hull_feature_types == 2)
        n_sharp_concave = np.sum(hull_feature_types == 3)
        n_corner_concave = np.sum(hull_feature_types == 4)
        if n_sharp_convex + n_corner_convex + n_sharp_concave + n_corner_concave > 0:
            logger.info(f"Hull mesh features: {n_sharp_convex + n_sharp_concave} sharp edges "
                       f"({n_sharp_concave} concave), {n_corner_convex + n_corner_concave} corners "
                       f"({n_corner_concave} concave)")
    
    if primary_mesh is not None and len(primary_mesh.faces) > 0:
        primary_feature_types, primary_sharp_edge_info = classify_mesh_vertex_features(primary_mesh)
        n_sharp_convex = np.sum(primary_feature_types == 1)
        n_corner_convex = np.sum(primary_feature_types == 2)
        n_sharp_concave = np.sum(primary_feature_types == 3)
        n_corner_concave = np.sum(primary_feature_types == 4)
        if n_sharp_convex + n_corner_convex + n_sharp_concave + n_corner_concave > 0:
            logger.info(f"Primary mesh features: {n_sharp_convex + n_sharp_concave} sharp edges "
                       f"({n_sharp_concave} concave), {n_corner_convex + n_corner_concave} corners "
                       f"({n_corner_concave} concave)")
    
    # Compute bounded search radius for reprojection
    search_radius = mesh_scale * MAX_REPROJECT_DISTANCE_FRACTION
    
    # === Pre-compute corner vertices that should be kept fixed ===
    # Per paper Section 4.4: corners should be "kept fixed" during smoothing
    # We identify membrane boundary vertices that correspond to corners on the target mesh
    # and exclude them from BOTH boundary smoothing AND interior smoothing
    corner_vertex_set = set()
    
    # Tolerance for corner detection - if a membrane vertex is within this distance
    # of ANY corner vertex on the target mesh, treat it as a corner vertex
    # This handles cases where the membrane boundary is slightly offset from the exact corner
    corner_detection_radius = mesh_scale * 0.03  # 3% of mesh size
    
    def identify_corner_membrane_vertices(membrane_indices, target_mesh, feature_types):
        """Find membrane boundary vertices near target mesh CONCAVE corners.
        
        Uses a radius-based search to find membrane vertices that are near any CONCAVE
        corner on the target mesh. Only concave corners need to be fixed during smoothing;
        convex corners can be smoothed normally without drifting issues.
        
        Feature type 4 = concave corner (the only ones we need to fix)
        Feature type 2 = convex corner (can be smoothed)
        """
        if len(membrane_indices) == 0 or target_mesh is None or feature_types is None:
            return set()
        
        from scipy.spatial import cKDTree
        
        # Find only CONCAVE corner vertices on the target mesh (feature_type == 4)
        concave_corner_mask = (feature_types == 4)
        corner_indices = np.where(concave_corner_mask)[0]
        
        if len(corner_indices) == 0:
            return set()
        
        # Build KD-tree of only the concave corner positions
        corner_positions = target_mesh.vertices[corner_indices]
        corner_kdtree = cKDTree(corner_positions)
        
        corner_verts = set()
        membrane_positions = vertices[membrane_indices]
        
        # For each membrane vertex, check if ANY concave corner is within the detection radius
        distances, _ = corner_kdtree.query(membrane_positions)
        
        for i, vi in enumerate(membrane_indices):
            if distances[i] < corner_detection_radius:
                corner_verts.add(vi)
        
        return corner_verts
    
    # Find corners for each target mesh
    if part_feature_types is not None and len(part_boundary_indices) > 0:
        part_corners = identify_corner_membrane_vertices(
            part_boundary_indices, part_mesh, part_feature_types
        )
        corner_vertex_set.update(part_corners)
        if part_corners:
            logger.info(f"Found {len(part_corners)} membrane vertices at part mesh CONCAVE corners (will be kept fixed)")
    
    if hull_feature_types is not None and len(hull_boundary_indices) > 0:
        hull_corners = identify_corner_membrane_vertices(
            hull_boundary_indices, hull_mesh, hull_feature_types
        )
        corner_vertex_set.update(hull_corners)
        if hull_corners:
            logger.info(f"Found {len(hull_corners)} membrane vertices at hull mesh CONCAVE corners (will be kept fixed)")
    
    if primary_feature_types is not None and len(primary_boundary_indices) > 0:
        primary_corners = identify_corner_membrane_vertices(
            primary_boundary_indices, primary_mesh, primary_feature_types
        )
        corner_vertex_set.update(primary_corners)
        if primary_corners:
            logger.info(f"Found {len(primary_corners)} membrane vertices at primary mesh CONCAVE corners (will be kept fixed)")
    
    # Add corner vertices to excluded set (they won't be smoothed at all)
    excluded_set.update(corner_vertex_set)
    if corner_vertex_set:
        logger.info(f"Total {len(corner_vertex_set)} CONCAVE corner vertices excluded from smoothing")
    
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
            new_vertices = vertices.copy()
            
            for vi in boundary_verts:
                # Skip excluded vertices (gap-fill vertices)
                if vi in excluded_set:
                    continue
                neighbors = list(boundary_neighbors[vi])
                if len(neighbors) >= 2:
                    # Average of neighbors along the boundary polyline
                    neighbor_positions = vertices[neighbors]
                    avg_pos = neighbor_positions.mean(axis=0)
                    new_vertices[vi] = vertices[vi] + damping_factor * (avg_pos - vertices[vi])
            
            vertices = new_vertices
        
        # === Step 2: Re-project boundary vertices onto target surfaces ===
        # Per paper: "After this smooth step, we re-project those vertices onto the 
        # original surface of M or the external boundary ∂H"
        #
        # We use feature-aware reprojection to handle sharp concave edges:
        # - For sharp edges: project to nearest edge (1D) instead of surface (2D)
        # - For corners: keep fixed (don't reproject)
        # - For smooth regions: use standard nearest-point projection
        
        # Re-project to part mesh M (with feature awareness)
        if len(part_boundary_indices) > 0 and part_mesh is not None:
            if part_feature_types is not None:
                # Use feature-aware reprojection for sharp edges/corners
                vertices = reproject_with_feature_awareness(
                    vertices=vertices,
                    boundary_indices=part_boundary_indices,
                    target_mesh=part_mesh,
                    feature_types=part_feature_types,
                    sharp_edge_info=part_sharp_edge_info,
                    search_radius=search_radius
                )
            elif part_proximity is not None:
                # Fallback to standard reprojection
                part_positions = vertices[part_boundary_indices]
                closest_pts, _, _ = part_proximity.on_surface(part_positions)
                vertices[part_boundary_indices] = closest_pts
        
        # Re-project to hull mesh ∂H (with feature awareness)
        if len(hull_boundary_indices) > 0 and hull_mesh is not None:
            if hull_feature_types is not None:
                # Use feature-aware reprojection for sharp edges/corners
                vertices = reproject_with_feature_awareness(
                    vertices=vertices,
                    boundary_indices=hull_boundary_indices,
                    target_mesh=hull_mesh,
                    feature_types=hull_feature_types,
                    sharp_edge_info=hull_sharp_edge_info,
                    search_radius=search_radius
                )
            elif hull_proximity is not None:
                # Fallback to standard reprojection
                hull_positions = vertices[hull_boundary_indices]
                closest_pts, _, _ = hull_proximity.on_surface(hull_positions)
                vertices[hull_boundary_indices] = closest_pts
        
        # Re-project to primary mesh (for secondary surfaces, with feature awareness)
        if len(primary_boundary_indices) > 0 and primary_mesh is not None:
            if primary_feature_types is not None:
                # Use feature-aware reprojection for sharp edges/corners
                vertices = reproject_with_feature_awareness(
                    vertices=vertices,
                    boundary_indices=primary_boundary_indices,
                    target_mesh=primary_mesh,
                    feature_types=primary_feature_types,
                    sharp_edge_info=primary_sharp_edge_info,
                    search_radius=search_radius
                )
            elif primary_proximity is not None:
                # Fallback to standard reprojection
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
