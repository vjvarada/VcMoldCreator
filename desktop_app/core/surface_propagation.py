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

# Feature detection thresholds - only detect concave corners
SHARP_EDGE_ANGLE_THRESHOLD = 60.0  # degrees - edges sharper than this are classified as "sharp"
CORNER_SHARP_EDGE_COUNT = 2        # A vertex with >= this many sharp edges meeting is a "corner"


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _check_edge_concavity(
    v0: int,
    v1: int,
    mesh: trimesh.Trimesh,
    edge_to_faces: dict,
    angle_threshold: float
) -> bool:
    """
    Check if the edge (v0, v1) is concave on the given mesh.
    
    An edge is concave if the dihedral angle exceeds the threshold
    and the adjacent faces fold inward. Used for feature detection
    on both part meshes and target meshes.
    
    Args:
        v0: First vertex index of the edge
        v1: Second vertex index of the edge
        mesh: The mesh to check concavity on
        edge_to_faces: Dict mapping (min_v, max_v) -> list of face indices
        angle_threshold: Minimum dihedral angle in degrees to consider sharp
        
    Returns:
        True if the edge is concave (folds inward), False otherwise
    """
    edge = (min(v0, v1), max(v0, v1))
    face_indices = edge_to_faces.get(edge, [])
    
    if len(face_indices) != 2:
        return False  # Boundary edge - not concave
    
    n0 = mesh.face_normals[face_indices[0]]
    n1 = mesh.face_normals[face_indices[1]]
    
    dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))
    
    if angle_deg <= angle_threshold:
        return False  # Not sharp
    
    # Determine concavity: check if faces fold inward
    face1 = mesh.faces[face_indices[1]]
    opposite_v1 = [v for v in face1 if v != v0 and v != v1][0]
    
    p_opposite1 = mesh.vertices[opposite_v1]
    p_edge0 = mesh.vertices[v0]
    
    vec_to_opp1 = p_opposite1 - p_edge0
    side = np.dot(vec_to_opp1, n0)
    
    return side > 0  # side > 0 means faces fold INWARD = CONCAVE


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
    
    # Restored fixed vertices (concave corners AND isolated triangle tips snapped back after smoothing)
    restored_corner_positions: Optional[np.ndarray] = None  # (K, 3) positions of restored vertices


@dataclass
class FeatureDebugVisualization:
    """Debug visualization data for sharp edge/corner feature classification.
    
    Membrane boundary vertex projected types:
        0 = smooth (normal projection)
        1 = convex sharp edge (can be smoothed)
        2 = convex corner (can be smoothed)
        3 = concave sharp edge (can be smoothed)
        4 = concave corner (FIXED - no smoothing)
        5 = isolated triangle tip (FIXED - same detection as inner collar creation)
    """
    
    # Target mesh (part or hull) feature data
    target_mesh_vertices: Optional[np.ndarray] = None       # (N, 3) vertex positions
    target_feature_types: Optional[np.ndarray] = None       # (N,) 0=smooth, 1=sharp_edge, 2=corner
    target_sharp_edges: Optional[List[Tuple[int, int]]] = None  # List of (v0, v1) edge indices
    
    # Membrane boundary vertex data  
    membrane_boundary_positions: Optional[np.ndarray] = None  # (M, 3) boundary vertex positions
    membrane_boundary_projected_types: Optional[np.ndarray] = None  # (M,) feature type each projects to (0-5)
    membrane_boundary_nearest_target: Optional[np.ndarray] = None  # (M,) nearest target vertex index
    
    # Restored corner vertices (concave corners AND single-triangle tips snapped back after smoothing)
    restored_corner_positions: Optional[np.ndarray] = None  # (K, 3) positions of restored vertices (blue spheres)


def get_feature_debug_visualization(
    target_mesh: trimesh.Trimesh,
    membrane_mesh: trimesh.Trimesh,
    membrane_boundary_indices: np.ndarray = None,
    angle_threshold: float = SHARP_EDGE_ANGLE_THRESHOLD,
    restored_corner_positions: np.ndarray = None
) -> FeatureDebugVisualization:
    """
    Generate debug visualization data for feature classification.
    
    This function uses the SAME detection logic as the actual smoothing process:
    - Directly analyzes membrane-part connections at each boundary vertex
    - Classifies based on whether closest point is at a concave edge/vertex
    - NOT based on nearest target mesh vertex (old approach)
    
    This displays:
    1. Which vertices on the target mesh are classified as sharp edges vs corners
    2. Which membrane boundary vertices connect to concave features (using direct analysis)
    3. Restored corner vertices (concave corners snapped back after smoothing)
    
    Args:
        target_mesh: The target mesh (part) to classify features on
        membrane_mesh: The membrane mesh being smoothed
        membrane_boundary_indices: Indices of membrane boundary vertices (computed if not provided)
        angle_threshold: Dihedral angle threshold for sharp edge detection
        restored_corner_positions: Optional array of restored corner vertex positions (from smoothing)
        
    Returns:
        FeatureDebugVisualization with all data needed for visualization
    """
    from trimesh.proximity import ProximityQuery
    
    debug = FeatureDebugVisualization()
    
    if target_mesh is None or len(target_mesh.faces) == 0:
        return debug
    
    # Classify target mesh vertices (for visualization of target mesh features)
    feature_types, sharp_edge_info = classify_mesh_vertex_features(
        target_mesh, angle_threshold
    )
    
    debug.target_mesh_vertices = target_mesh.vertices.copy()
    debug.target_feature_types = feature_types
    
    # === Build adjacency structures for direct connection analysis ===
    edge_to_faces = {}
    for fi, face in enumerate(target_mesh.faces):
        for i in range(3):
            v0, v1 = face[i], face[(i+1) % 3]
            edge = (min(v0, v1), max(v0, v1))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    vertex_to_faces = {v: [] for v in range(len(target_mesh.vertices))}
    for fi, face in enumerate(target_mesh.faces):
        for v in face:
            vertex_to_faces[v].append(fi)
    
    # Extract sharp edges from the target mesh (for visualization)
    sharp_edges = []
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
    
    # === Classify membrane boundary vertices using DIRECT CONNECTION ANALYSIS ===
    # This matches the actual detection logic in identify_concave_membrane_vertices_at_part_boundary
    if membrane_mesh is not None and membrane_boundary_indices is not None and len(membrane_boundary_indices) > 0:
        
        def is_edge_concave(v0: int, v1: int) -> bool:
            """Check if the edge (v0, v1) is concave on the target mesh."""
            return _check_edge_concavity(v0, v1, target_mesh, edge_to_faces, angle_threshold)
        
        def is_edge_convex_sharp(v0: int, v1: int) -> bool:
            """Check if the edge (v0, v1) is convex sharp on the target mesh."""
            edge = (min(v0, v1), max(v0, v1))
            face_indices = edge_to_faces.get(edge, [])
            
            if len(face_indices) != 2:
                return True  # Boundary edge - treat as convex sharp
            
            n0 = target_mesh.face_normals[face_indices[0]]
            n1 = target_mesh.face_normals[face_indices[1]]
            
            dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot))
            
            if angle_deg <= angle_threshold:
                return False  # Not sharp
            
            # If sharp but not concave, it's convex
            return not is_edge_concave(v0, v1)
        
        def get_vertex_feature_type_at_connection(vi: int) -> int:
            """Get the feature type when closest to a target vertex."""
            face_indices = vertex_to_faces.get(vi, [])
            if len(face_indices) < 2:
                return 0  # Smooth
            
            # Get all edges incident to this vertex
            incident_edges = set()
            for fi in face_indices:
                face = target_mesh.faces[fi]
                for i in range(3):
                    if face[i] == vi or face[(i+1) % 3] == vi:
                        v0, v1 = face[i], face[(i+1) % 3]
                        incident_edges.add((min(v0, v1), max(v0, v1)))
            
            # Check for concave edges first (higher priority)
            has_concave = False
            has_convex_sharp = False
            for edge in incident_edges:
                if is_edge_concave(edge[0], edge[1]):
                    has_concave = True
                    break
                if is_edge_convex_sharp(edge[0], edge[1]):
                    has_convex_sharp = True
            
            if has_concave:
                # Multiple concave edges = corner, single = edge
                concave_count = sum(1 for e in incident_edges if is_edge_concave(e[0], e[1]))
                return 4 if concave_count >= 2 else 3  # 4=concave_corner, 3=concave_edge
            elif has_convex_sharp:
                convex_count = sum(1 for e in incident_edges if is_edge_convex_sharp(e[0], e[1]))
                return 2 if convex_count >= 2 else 1  # 2=convex_corner, 1=convex_edge
            
            return 0  # Smooth
        
        # Use ProximityQuery to find closest points (same as actual detection)
        target_prox = ProximityQuery(target_mesh)
        boundary_positions = membrane_mesh.vertices[membrane_boundary_indices]
        closest_pts, distances, triangle_ids = target_prox.on_surface(boundary_positions)
        
        # Compute mesh scale for tolerance
        bounds = target_mesh.bounds
        mesh_scale = np.max(bounds[1] - bounds[0])
        tolerance = mesh_scale * 0.005  # 0.5% of mesh size
        
        # Classify each membrane boundary vertex using direct connection analysis
        projected_types = np.zeros(len(membrane_boundary_indices), dtype=np.int32)
        nearest_target_indices = np.zeros(len(membrane_boundary_indices), dtype=np.int64)
        
        for i, (closest_pt, tri_id) in enumerate(zip(closest_pts, triangle_ids)):
            if tri_id < 0 or tri_id >= len(target_mesh.faces):
                continue
            
            face = target_mesh.faces[tri_id]
            face_verts = target_mesh.vertices[face]
            
            # Find distance from closest_pt to each face vertex
            vert_distances = np.linalg.norm(face_verts - closest_pt, axis=1)
            min_vert_dist = np.min(vert_distances)
            min_vert_idx = np.argmin(vert_distances)
            
            # Check if closest to a vertex
            if min_vert_dist < tolerance:
                part_vertex_idx = face[min_vert_idx]
                nearest_target_indices[i] = part_vertex_idx
                projected_types[i] = get_vertex_feature_type_at_connection(part_vertex_idx)
                continue
            
            # Check if closest to an edge
            found_edge = False
            for edge_idx in range(3):
                ev0, ev1 = face[edge_idx], face[(edge_idx + 1) % 3]
                p0, p1 = target_mesh.vertices[ev0], target_mesh.vertices[ev1]
                
                edge_vec = p1 - p0
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-10:
                    continue
                edge_dir = edge_vec / edge_len
                
                t = np.dot(closest_pt - p0, edge_dir)
                t = np.clip(t, 0, edge_len)
                proj_pt = p0 + t * edge_dir
                edge_dist = np.linalg.norm(closest_pt - proj_pt)
                
                if edge_dist < tolerance:
                    nearest_target_indices[i] = ev0  # Arbitrary choice for visualization
                    if is_edge_concave(ev0, ev1):
                        projected_types[i] = 3  # concave_edge
                    elif is_edge_convex_sharp(ev0, ev1):
                        projected_types[i] = 1  # convex_edge
                    found_edge = True
                    break
            
            if not found_edge:
                # Closest to face interior - smooth
                nearest_target_indices[i] = face[0]
                projected_types[i] = 0
        
        debug.membrane_boundary_positions = boundary_positions.copy()
        debug.membrane_boundary_projected_types = projected_types
        debug.membrane_boundary_nearest_target = nearest_target_indices
        
        # === Also detect isolated triangle tip vertices ===
        # Uses the SAME edge-based logic as inner collar creation and smoothing step:
        # A tip vertex has exactly 2 boundary edges that share the SAME face.
        
        # Build edge_to_face mapping for the membrane mesh
        edge_to_face_membrane = {}
        edge_face_count_membrane = {}
        for fi, face in enumerate(membrane_mesh.faces):
            for ei in range(3):
                v0, v1 = int(face[ei]), int(face[(ei + 1) % 3])
                edge_key = (min(v0, v1), max(v0, v1))
                edge_face_count_membrane[edge_key] = edge_face_count_membrane.get(edge_key, 0) + 1
                edge_to_face_membrane[edge_key] = (fi, int(face[(ei + 2) % 3]))
        
        # Find boundary edges (edges with only 1 adjacent face)
        vis_boundary_edges = [(v0, v1) for (v0, v1), count in edge_face_count_membrane.items() if count == 1]
        
        # Build vertex -> boundary edges mapping
        vertex_to_vis_boundary_edges = {}
        for v0, v1 in vis_boundary_edges:
            if v0 not in vertex_to_vis_boundary_edges:
                vertex_to_vis_boundary_edges[v0] = []
            if v1 not in vertex_to_vis_boundary_edges:
                vertex_to_vis_boundary_edges[v1] = []
            vertex_to_vis_boundary_edges[v0].append(((v0, v1), v0, v1))
            vertex_to_vis_boundary_edges[v1].append(((v0, v1), v1, v0))
        
        # Mark isolated tip vertices as type 5 (same as inner collar detection)
        # Tip = vertex with exactly 2 boundary edges from the SAME face
        # NOTE: This marks ALL boundary tips for visualization; smoothing only fixes PART boundary tips
        for i, vi in enumerate(membrane_boundary_indices):
            if projected_types[i] == 4:
                continue  # Already a concave corner, don't override
            
            edges = vertex_to_vis_boundary_edges.get(vi, [])
            if len(edges) == 2:
                edge_key_a = (min(edges[0][0][0], edges[0][0][1]), max(edges[0][0][0], edges[0][0][1]))
                edge_key_b = (min(edges[1][0][0], edges[1][0][1]), max(edges[1][0][0], edges[1][0][1]))
                
                face_a = edge_to_face_membrane.get(edge_key_a)
                face_b = edge_to_face_membrane.get(edge_key_b)
                
                if face_a is not None and face_b is not None and face_a[0] == face_b[0]:
                    # Same face - this is an isolated triangle tip!
                    projected_types[i] = 5
        
        logger.info(
            f"Feature debug (direct analysis): {np.sum(projected_types == 0)} smooth, "
            f"{np.sum(projected_types == 1)} convex edge, {np.sum(projected_types == 2)} convex corner, "
            f"{np.sum(projected_types == 3)} concave edge, {np.sum(projected_types == 4)} concave corner, "
            f"{np.sum(projected_types == 5)} isolated triangle tip"
        )
    
    # Add restored corner positions if provided
    if restored_corner_positions is not None and len(restored_corner_positions) > 0:
        debug.restored_corner_positions = restored_corner_positions
        logger.info(f"Feature debug: {len(restored_corner_positions)} restored corner vertices (blue spheres)")
    
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
    Classify mesh vertices into feature types for visualization and smoothing.
    
    Per paper Section 4.4: "corners should be kept fixed" during smoothing.
    Only concave corners (type 4) need to be kept fixed during smoothing.
    
    Convex vs Concave terminology (from the MOLD's perspective):
    - CONVEX edge/corner on part = part sticks OUT = forms a pocket in the mold
      These are the "sharp outer edges" of the original object.
    - CONCAVE edge/corner on part = part goes IN = forms a protrusion in the mold
      These are the "inner crevices" of the original object.
    
    For MOLD EXTRACTION, concave features on the PART are problematic because
    they create CONVEX protrusions in the mold that lock in place.
    
    Args:
        mesh: Input trimesh
        angle_threshold: Dihedral angle (degrees) above which an edge is considered sharp
    
    Returns:
        Tuple of:
        - vertex_feature_type: Array with values:
            0 = smooth (no sharp edges)
            1 = convex sharp edge (on 1-2 sharp edges, all convex)
            2 = convex corner (on 3+ sharp edges, all convex)
            3 = concave sharp edge (on 1-2 sharp edges, at least one concave)
            4 = concave corner (on 3+ sharp edges, at least one concave) - FIXED during smoothing
        - sharp_edge_info: Dict containing edge classification data
    """
    n_verts = len(mesh.vertices)
    vertex_feature_type = np.zeros(n_verts, dtype=np.int8)
    sharp_edge_info = {'sharp_edges': [], 'edge_is_concave': {}}
    
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
        
        # Get face normals
        n0 = mesh.face_normals[face_indices[0]]
        n1 = mesh.face_normals[face_indices[1]]
        
        # Compute dihedral angle
        dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        
        if angle_deg > angle_threshold:
            sharp_edges.add(edge)
            
            # Determine concavity using signed dihedral angle
            # Check if the opposite vertex of one face is "above" or "below" 
            # the plane of the other face
            v0, v1 = edge
            
            # Get the vertices in each face that are NOT on the shared edge
            face0 = mesh.faces[face_indices[0]]
            face1 = mesh.faces[face_indices[1]]
            opposite_v0 = [v for v in face0 if v != v0 and v != v1][0]
            opposite_v1 = [v for v in face1 if v != v0 and v != v1][0]
            
            # Get the positions
            p_opposite1 = mesh.vertices[opposite_v1]
            p_edge0 = mesh.vertices[v0]
            
            # Check: is the opposite vertex of face1 "above" or "below" face0's plane?
            vec_to_opp1 = p_opposite1 - p_edge0
            side = np.dot(vec_to_opp1, n0)
            
            # CORRECTED LOGIC:
            # If opposite vertex of face1 is on the SAME side as n0 points (side > 0),
            # the faces fold INWARD creating a valley/groove = CONCAVE on part
            # If opposite vertex is on the OPPOSITE side (side < 0),
            # the faces fold OUTWARD creating a ridge = CONVEX on part
            is_concave = side > 0
            edge_is_concave[edge] = is_concave
    
    # Store in sharp_edge_info for debugging
    sharp_edge_info['sharp_edges'] = list(sharp_edges)
    sharp_edge_info['edge_is_concave'] = edge_is_concave
    
    # Count sharp edges meeting at each vertex and track concavity
    vertex_sharp_edge_count = np.zeros(n_verts, dtype=np.int32)
    vertex_concave_edge_count = np.zeros(n_verts, dtype=np.int32)
    vertex_convex_edge_count = np.zeros(n_verts, dtype=np.int32)
    
    for edge in sharp_edges:
        v0, v1 = edge
        vertex_sharp_edge_count[v0] += 1
        vertex_sharp_edge_count[v1] += 1
        
        if edge_is_concave.get(edge, False):
            vertex_concave_edge_count[v0] += 1
            vertex_concave_edge_count[v1] += 1
        else:
            vertex_convex_edge_count[v0] += 1
            vertex_convex_edge_count[v1] += 1
    
    # Classify vertices using 5-type scheme:
    # 0 = smooth, 1 = convex_edge, 2 = convex_corner, 3 = concave_edge, 4 = concave_corner
    for vi in range(n_verts):
        sharp_count = vertex_sharp_edge_count[vi]
        concave_count = vertex_concave_edge_count[vi]
        convex_count = vertex_convex_edge_count[vi]
        
        if sharp_count == 0:
            vertex_feature_type[vi] = 0  # smooth
        elif sharp_count >= CORNER_SHARP_EDGE_COUNT:
            # Corner (3+ sharp edges)
            if concave_count > 0:
                vertex_feature_type[vi] = 4  # concave corner - FIXED during smoothing
            else:
                vertex_feature_type[vi] = 2  # convex corner
        else:
            # Edge vertex (1-2 sharp edges)
            if concave_count > 0:
                vertex_feature_type[vi] = 3  # concave edge
            else:
                vertex_feature_type[vi] = 1  # convex edge
    
    n_smooth = np.sum(vertex_feature_type == 0)
    n_convex_edge = np.sum(vertex_feature_type == 1)
    n_convex_corner = np.sum(vertex_feature_type == 2)
    n_concave_edge = np.sum(vertex_feature_type == 3)
    n_concave_corner = np.sum(vertex_feature_type == 4)
    logger.debug(f"Feature classification: {n_smooth} smooth, "
                 f"{n_convex_edge} convex edge, {n_convex_corner} convex corner, "
                 f"{n_concave_edge} concave edge, {n_concave_corner} concave corner (fixed)")
    
    return vertex_feature_type, sharp_edge_info


def reproject_with_feature_awareness(
    vertices: np.ndarray,
    boundary_indices: np.ndarray,
    target_mesh: trimesh.Trimesh,
    feature_types: np.ndarray,
    sharp_edge_info: dict,
    search_radius: Optional[float] = None
) -> np.ndarray:
    """
    Re-project boundary vertices to target mesh.
    
    All vertices are now reprojected using standard nearest-point projection.
    Concave corners are handled separately - they are smoothed normally during
    iterations, then restored to original positions after smoothing completes.
    
    Feature type codes (from classify_mesh_vertex_features):
        0 = smooth (normal reprojection)
        1 = convex sharp edge (normal reprojection)
        2 = convex corner (normal reprojection)
        3 = concave sharp edge (normal reprojection)
        4 = concave corner (normal reprojection - restored after smoothing)
    
    Args:
        vertices: (N, 3) all mesh vertices (membrane mesh)
        boundary_indices: Indices of boundary vertices to reproject (membrane mesh indices)
        target_mesh: The target mesh to project onto (part/hull mesh)
        feature_types: Per-vertex feature type for TARGET MESH (unused, kept for API)
        sharp_edge_info: Unused (kept for API compatibility)
        search_radius: Optional max search distance (auto-computed if None)
    
    Returns:
        Updated vertices array (modified in place and returned)
    """
    from trimesh.proximity import ProximityQuery
    
    if target_mesh is None or len(target_mesh.faces) == 0:
        return vertices
    
    if len(boundary_indices) == 0:
        return vertices
    
    proximity = ProximityQuery(target_mesh)
    
    # Auto-compute search radius if not provided
    if search_radius is None:
        mesh_scale = np.linalg.norm(target_mesh.bounds[1] - target_mesh.bounds[0])
        search_radius = mesh_scale * 0.1  # 10% of mesh size
    
    # === All vertices: standard nearest-point projection with bounded search ===
    boundary_indices_arr = np.array(boundary_indices, dtype=np.int64)
    boundary_positions = vertices[boundary_indices_arr]
    closest_pts, distances, _ = proximity.on_surface(boundary_positions)
    
    # Only apply projection if within search radius
    within_radius = distances < search_radius
    vertices[boundary_indices_arr[within_radius]] = closest_pts[within_radius]
    
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
    vertex_boundary_type: Optional[np.ndarray] = None,
    feature_aware_smoothing: bool = True,
    reproject_to_hull: bool = True,
    reproject_to_primary: bool = False
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
        feature_aware_smoothing: If True (default), detect concave corners and keep them fixed.
                                If False, smooth all vertices normally without feature detection.
        reproject_to_hull: If True (default), re-project hull boundary vertices to hull mesh.
                          If False, hull boundary vertices are free to move during smoothing.
                          Use False for secondary surfaces where only part re-projection is needed.
        reproject_to_primary: If True, re-project non-part boundary vertices to primary mesh
                             (if primary_mesh is provided and they are close enough).
                             Use True for secondary surfaces to connect to primary membrane.
    
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
    # CRITICAL: Use process=False to prevent merge_vertices() from renumbering
    # vertex indices. The boundary edge indices must match the original vertices
    # array — if merge_vertices() removes or reindexes vertices, boundary
    # classification will index the wrong vertices.
    boundary_edges = find_boundary_edges(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
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
    
    # Debug: log what we received
    if vertex_boundary_type is not None:
        logger.debug(f"vertex_boundary_type: {len(vertex_boundary_type)} entries, n_verts={n_verts}, match={len(vertex_boundary_type) == n_verts}")
    else:
        logger.debug(f"vertex_boundary_type: None, n_verts={n_verts}")
    
    if use_boundary_type_classification:
        # === CLASSIFICATION FROM EXTRACTION ===
        # Use the vertex_boundary_type from extraction (per paper Section 4.4)
        logger.info("Using vertex_boundary_type for boundary classification (per paper Section 4.4)")
        
        # Count vertex_boundary_type values for ALL membrane vertices (not just boundary)
        if vertex_boundary_type is not None:
            n_type_part = np.sum(vertex_boundary_type == -1)
            n_type_hull = np.sum((vertex_boundary_type == 1) | (vertex_boundary_type == 2))
            n_type_primary_junc = np.sum(vertex_boundary_type == 3)
            n_type_primary_part = np.sum(vertex_boundary_type == 4)
            n_type_interior = np.sum(vertex_boundary_type == 0)
            logger.info(f"vertex_boundary_type distribution (all {len(vertex_boundary_type)} verts): "
                       f"{n_type_part} part (-1), {n_type_hull} hull (1/2), "
                       f"{n_type_primary_junc} primary junction (3), "
                       f"{n_type_primary_part} primary+part (4), "
                       f"{n_type_interior} interior (0)")
        
        for vi in boundary_verts:
            bt = vertex_boundary_type[vi]
            if bt == -1:
                # INNER boundary - originally on part surface M
                boundary_surface[vi] = 'part'
            elif bt == 4:
                # PRIMARY-THEN-PART - triple junction vertex where secondary,
                # primary, and part all meet. First reprojected to primary,
                # then reprojected to part → pinned to primary∩part curve.
                if primary_mesh is not None:
                    boundary_surface[vi] = 'primary_then_part'
                else:
                    boundary_surface[vi] = 'part'
            elif bt == 3:
                # PRIMARY JUNCTION - secondary vertex on a primary cut edge.
                # This vertex must lie on the primary surface to ensure the
                # secondary membrane connects seamlessly to the primary.
                if primary_mesh is not None:
                    boundary_surface[vi] = 'primary'
                else:
                    boundary_surface[vi] = 'hull'
            elif bt in (1, 2):
                # OUTER boundary - originally on hull boundary ∂H (H1 or H2)
                boundary_surface[vi] = 'hull'
            elif bt == 0:
                # Type 0 = interior or boundary-zone vertex.
                # Classified as 'hull'; when reproject_to_hull=False (secondary),
                # these become free-floating.
                boundary_surface[vi] = 'hull'
            else:
                boundary_surface[vi] = 'patch'
        
        part_count = sum(1 for s in boundary_surface.values() if s == 'part')
        hull_count = sum(1 for s in boundary_surface.values() if s == 'hull')
        primary_count = sum(1 for s in boundary_surface.values() if s == 'primary')
        ptp_count = sum(1 for s in boundary_surface.values() if s == 'primary_then_part')
        patch_count = sum(1 for s in boundary_surface.values() if s == 'patch')
        
        logger.info(f"Boundary classification (from extraction): {part_count} part, "
                   f"{hull_count} hull, {primary_count} primary, "
                   f"{ptp_count} primary→part, {patch_count} patch")
        
        # For 'patch' vertices, use NEIGHBOR PROPAGATION first, then distance as last resort
        # This is important because the inflated hull CONTAINS the part mesh, so interior-
        # turned-boundary vertices may be closer to part by distance but topologically
        # belong to the outer (hull) boundary chain.
        #
        # For SECONDARY surfaces: skip distance fallback entirely — patch vertices stay free.
        is_secondary_surface = reproject_to_primary and not reproject_to_hull
        
        if patch_count > 0:
            from trimesh.proximity import ProximityQuery
            
            if is_secondary_surface:
                # SECONDARY: Make all patch vertices free (hull with reproject_to_hull=False)
                for vi in boundary_verts:
                    if boundary_surface.get(vi) == 'patch':
                        boundary_surface[vi] = 'hull'  # Free (reproject_to_hull=False)
                logger.info(f"Secondary surface: {patch_count} patch vertices set to free (no distance fallback)")
            else:
                # PRIMARY: Use neighbor propagation + distance fallback
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
        
        # For SECONDARY surfaces: prefer free-floating over distance-guessing.
        # Only pin to part if clearly on it; everything else stays free.
        # Distance-based primary detection is unreliable and can cause folding.
        is_secondary_surface = reproject_to_primary and not reproject_to_hull
        if is_secondary_surface:
            logger.info("Secondary surface: using part-only distance classification (all other verts free)")
        
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
        
        if not is_secondary_surface:
            # Only compute hull/primary distances for primary surfaces
            if hull_proximity is not None:
                _, dist_to_hull, _ = hull_proximity.on_surface(boundary_positions)
            
            if primary_proximity is not None:
                _, dist_to_primary, _ = primary_proximity.on_surface(boundary_positions)
        
        # Classify each boundary vertex based on distances
        for i, vi in enumerate(boundary_verts):
            d_part = dist_to_part[i]
            d_hull = dist_to_hull[i]
            d_primary = dist_to_primary[i]
            
            if is_secondary_surface:
                # SECONDARY: Only pin to part if clearly on it. Everything else free.
                if d_part < on_surface_tolerance:
                    boundary_surface[vi] = 'part'
                else:
                    boundary_surface[vi] = 'hull'  # Free (reproject_to_hull=False)
            else:
                # PRIMARY: Full distance-based classification
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
    primary_then_part_indices = []  # bt=4: dual reprojection (primary first, then part)
    
    for vi in boundary_verts:
        if vi in excluded_set:
            continue
        surface_type = boundary_surface.get(vi, 'patch')
        if surface_type in ('part', 'closest_part', 'propagated_part'):
            part_boundary_indices.append(vi)
        elif surface_type == 'primary_then_part':
            primary_then_part_indices.append(vi)
        elif surface_type in ('hull', 'closest_hull', 'propagated_hull'):
            # Only add to hull re-projection list if reproject_to_hull is True
            if reproject_to_hull:
                hull_boundary_indices.append(vi)
            # Otherwise, these vertices are free to move during smoothing
        elif surface_type in ('primary', 'closest_primary'):
            primary_boundary_indices.append(vi)
    
    part_boundary_indices = np.array(part_boundary_indices, dtype=np.int64)
    hull_boundary_indices = np.array(hull_boundary_indices, dtype=np.int64)
    primary_boundary_indices = np.array(primary_boundary_indices, dtype=np.int64)
    primary_then_part_indices = np.array(primary_then_part_indices, dtype=np.int64)
    
    # Count how many boundary vertices are FREE (not re-projected anywhere)
    n_pinned = len(part_boundary_indices) + len(hull_boundary_indices) + len(primary_boundary_indices) + len(primary_then_part_indices)
    n_free_boundary = len(boundary_verts) - n_pinned
    
    logger.info(f"Batched re-projection: {len(part_boundary_indices)} part, "
               f"{len(primary_boundary_indices)} primary, "
               f"{len(primary_then_part_indices)} primary→part, "
               f"{len(hull_boundary_indices)} hull, "
               f"{n_free_boundary} free")
    
    # === Feature detection is now done inline for part boundary only ===
    # The new approach analyzes membrane-part connections directly rather than
    # pre-classifying all vertices on target meshes.
    # Hull and primary boundaries are smoothed normally without feature detection.
    
    # === Pre-compute corner vertices that should be kept fixed ===
    # Per paper Section 4.4: corners should be "kept fixed" during smoothing
    # 
    # NEW APPROACH: Instead of finding concave corners on the part mesh first and then
    # searching for nearby membrane vertices, we:
    # 1. Find all membrane boundary vertices classified as 'part' (on part inner boundary)
    # 2. For each, find the closest point on the part mesh (vertex, edge, or face)
    # 3. If it's at a vertex or edge, analyze the adjacent faces for concavity
    # 4. If concave, mark for restoration after smoothing
    #
    # IMPORTANT: This classification is ONLY for part boundary vertices.
    # Hull boundary vertices are smoothed normally without feature detection.
    corner_vertex_set = set()
    
    def identify_concave_membrane_vertices_at_part_boundary(
        membrane_indices: np.ndarray,
        membrane_positions: np.ndarray,
        part_mesh: trimesh.Trimesh,
        angle_threshold: float = SHARP_EDGE_ANGLE_THRESHOLD
    ) -> set:
        """
        Find membrane boundary vertices that connect to concave regions of the part mesh.
        
        For each membrane vertex on the part boundary:
        1. Find closest point on part mesh (could be on vertex, edge, or face)
        2. If closest to a vertex: check if that vertex is at a concave corner/edge
        3. If closest to an edge: check if that edge is concave
        4. If closest to face interior: not a sharp feature, skip
        
        Args:
            membrane_indices: Indices of membrane boundary vertices classified as 'part'
            membrane_positions: Positions of all membrane vertices
            part_mesh: The part mesh
            angle_threshold: Dihedral angle threshold for sharp edges
            
        Returns:
            Set of membrane vertex indices that should be restored after smoothing
        """
        if len(membrane_indices) == 0 or part_mesh is None or len(part_mesh.faces) == 0:
            return set()
        
        from trimesh.proximity import ProximityQuery
        
        concave_verts = set()
        
        # Build edge -> faces adjacency for part mesh
        edge_to_faces = {}
        for fi, face in enumerate(part_mesh.faces):
            for i in range(3):
                v0, v1 = face[i], face[(i+1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(fi)
        
        # Build vertex -> faces adjacency for part mesh
        vertex_to_faces = {v: [] for v in range(len(part_mesh.vertices))}
        for fi, face in enumerate(part_mesh.faces):
            for v in face:
                vertex_to_faces[v].append(fi)
        
        # Pre-compute which edges are concave
        def is_edge_concave(v0: int, v1: int) -> bool:
            """Check if the edge (v0, v1) is concave on the part mesh."""
            return _check_edge_concavity(v0, v1, part_mesh, edge_to_faces, angle_threshold)
        
        def is_vertex_at_concave_feature(vi: int) -> bool:
            """Check if vertex vi is at a concave edge or corner on the part mesh."""
            face_indices = vertex_to_faces.get(vi, [])
            if len(face_indices) < 2:
                return False
            
            # Get all edges incident to this vertex
            incident_edges = set()
            for fi in face_indices:
                face = part_mesh.faces[fi]
                for i in range(3):
                    if face[i] == vi or face[(i+1) % 3] == vi:
                        v0, v1 = face[i], face[(i+1) % 3]
                        incident_edges.add((min(v0, v1), max(v0, v1)))
            
            # Check if ANY incident edge is concave
            for edge in incident_edges:
                if is_edge_concave(edge[0], edge[1]):
                    return True
            
            return False
        
        # Get closest points on part mesh for all membrane vertices
        # Use try/except to handle rtree errors that can occur with certain mesh configurations
        try:
            part_prox = ProximityQuery(part_mesh)
            query_positions = membrane_positions[membrane_indices]
            closest_pts, distances, triangle_ids = part_prox.on_surface(query_positions)
        except Exception as e:
            logger.warning(f"ProximityQuery failed in concave vertex detection: {e}")
            logger.warning("Skipping concave corner detection - all part boundary vertices will be smoothed normally")
            return set()
        
        # Tolerance for being "at" a vertex or edge vs. on face interior
        # Use a small fraction of mesh scale
        tolerance = mesh_scale * 0.005  # 0.5% of mesh size
        
        for i, vi in enumerate(membrane_indices):
            closest_pt = closest_pts[i]
            tri_id = triangle_ids[i]
            
            if tri_id < 0 or tri_id >= len(part_mesh.faces):
                continue
            
            face = part_mesh.faces[tri_id]
            face_verts = part_mesh.vertices[face]
            
            # Compute barycentric-like distances to determine if closest to vertex, edge, or face
            # Find distance from closest_pt to each face vertex
            vert_distances = np.linalg.norm(face_verts - closest_pt, axis=1)
            min_vert_dist = np.min(vert_distances)
            min_vert_idx = np.argmin(vert_distances)
            
            # Check if closest to a vertex
            if min_vert_dist < tolerance:
                # Closest to a vertex - check if vertex is at concave feature
                part_vertex_idx = face[min_vert_idx]
                if is_vertex_at_concave_feature(part_vertex_idx):
                    concave_verts.add(vi)
                continue
            
            # Check if closest to an edge (distance to nearest edge < tolerance)
            # Test each of the 3 edges of the triangle
            for edge_idx in range(3):
                ev0, ev1 = face[edge_idx], face[(edge_idx + 1) % 3]
                p0, p1 = part_mesh.vertices[ev0], part_mesh.vertices[ev1]
                
                # Distance from closest_pt to line segment (p0, p1)
                edge_vec = p1 - p0
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-10:
                    continue
                edge_dir = edge_vec / edge_len
                
                # Project closest_pt onto the edge line
                t = np.dot(closest_pt - p0, edge_dir)
                t = np.clip(t, 0, edge_len)
                proj_pt = p0 + t * edge_dir
                edge_dist = np.linalg.norm(closest_pt - proj_pt)
                
                if edge_dist < tolerance:
                    # Closest to this edge - check if edge is concave
                    if is_edge_concave(ev0, ev1):
                        concave_verts.add(vi)
                    break
            
            # If closest to face interior (not vertex or edge), it's not a sharp feature
            # No action needed
        
        return concave_verts
    
    def identify_isolated_tip_boundary_vertices(
        membrane_faces: np.ndarray,
        inner_boundary_edges: List[Tuple[int, int]],
        edge_to_face: dict,
        part_boundary_indices: np.ndarray
    ) -> set:
        """
        Find isolated triangle tip vertices on the part boundary.
        
        Uses the SAME detection logic as inner collar creation in parting_surface.py:
        An isolated tip is a boundary vertex where exactly 2 boundary edges 
        share the SAME face. This indicates the vertex is at the tip of an 
        isolated triangle.
        
        These tip vertices should be kept fixed during smoothing to prevent 
        them from drifting. They represent critical contact points with the 
        part mesh (often at thin protrusions or sharp features).
        
        Args:
            membrane_faces: (F, 3) array of membrane face indices
            inner_boundary_edges: List of (v0, v1) inner boundary edges
            edge_to_face: Dict mapping edge_key -> (face_idx, third_vertex)
            part_boundary_indices: Indices of membrane vertices on the part boundary
            
        Returns:
            Set of membrane vertex indices that are isolated triangle tips
        """
        if len(part_boundary_indices) == 0 or len(inner_boundary_edges) == 0:
            return set()
        
        # Build mapping from vertex to its adjacent inner boundary edges
        # (same logic as in parting_surface.py create_robust_collar_extension)
        vertex_to_boundary_edges = {}
        for v0, v1 in inner_boundary_edges:
            if v0 not in vertex_to_boundary_edges:
                vertex_to_boundary_edges[v0] = []
            if v1 not in vertex_to_boundary_edges:
                vertex_to_boundary_edges[v1] = []
            vertex_to_boundary_edges[v0].append(((v0, v1), v0, v1))
            vertex_to_boundary_edges[v1].append(((v0, v1), v1, v0))
        
        # Find isolated tip vertices: vertices with exactly 2 boundary edges
        # from the SAME face (same as inner collar detection)
        part_boundary_set = set(part_boundary_indices)
        isolated_tip_verts = set()
        
        for vi, edges in vertex_to_boundary_edges.items():
            if vi not in part_boundary_set:
                continue
                
            if len(edges) == 2:
                # Check if these 2 boundary edges share the same face
                edge_key_a = (min(edges[0][0][0], edges[0][0][1]), max(edges[0][0][0], edges[0][0][1]))
                edge_key_b = (min(edges[1][0][0], edges[1][0][1]), max(edges[1][0][0], edges[1][0][1]))
                
                face_a = edge_to_face.get(edge_key_a)
                face_b = edge_to_face.get(edge_key_b)
                
                if face_a is not None and face_b is not None and face_a[0] == face_b[0]:
                    # Same face - this is an isolated triangle tip!
                    isolated_tip_verts.add(vi)
        
        return isolated_tip_verts
    
    # Only apply feature detection to part boundary, NOT hull boundary
    if feature_aware_smoothing and part_mesh is not None and len(part_boundary_indices) > 0:
        part_concave_verts = identify_concave_membrane_vertices_at_part_boundary(
            membrane_indices=part_boundary_indices,
            membrane_positions=vertices,
            part_mesh=part_mesh,
            angle_threshold=SHARP_EDGE_ANGLE_THRESHOLD
        )
        corner_vertex_set.update(part_concave_verts)
        if part_concave_verts:
            logger.info(f"Found {len(part_concave_verts)} membrane vertices at part concave features (will restore after smoothing)")
        
        # Also detect isolated triangle tip vertices (same logic as inner collar creation)
        # First, build edge_to_face mapping for the membrane mesh
        edge_to_face_membrane = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = (min(v0, v1), max(v0, v1))
                edge_to_face_membrane[edge_key] = (fi, int(face[(i + 2) % 3]))
        
        # Build list of inner boundary edges (edges where at least one vertex is on part)
        part_boundary_set = set(part_boundary_indices)
        inner_boundary_edges_for_tips = []
        for v0, v1 in boundary_edges:
            # Edge is "inner" if at least one vertex is on part boundary
            if v0 in part_boundary_set or v1 in part_boundary_set:
                inner_boundary_edges_for_tips.append((v0, v1))
        
        # Detect isolated tip vertices using same logic as inner collar
        isolated_tip_verts = identify_isolated_tip_boundary_vertices(
            membrane_faces=faces,
            inner_boundary_edges=inner_boundary_edges_for_tips,
            edge_to_face=edge_to_face_membrane,
            part_boundary_indices=part_boundary_indices
        )
        # Only add if not already classified as concave corner
        new_tip_verts = isolated_tip_verts - corner_vertex_set
        corner_vertex_set.update(new_tip_verts)
        if new_tip_verts:
            logger.info(f"Found {len(new_tip_verts)} isolated triangle tip vertices (same as inner collar detection, will restore after smoothing)")
        
    elif not feature_aware_smoothing:
        logger.info("Feature-aware smoothing disabled - all part boundary vertices smoothed normally")
    
    # NOTE: Hull boundary vertices are smoothed normally - no concave feature detection
    # This is because the hull is the external boundary and doesn't need special handling
    
    # Store original positions of concave corner vertices BEFORE smoothing
    # These will be restored after smoothing completes
    corner_original_positions = {}
    for vi in corner_vertex_set:
        corner_original_positions[vi] = vertices[vi].copy()
    
    if corner_vertex_set:
        logger.info(f"Total {len(corner_vertex_set)} fixed vertices (concave corners + tip vertices) - will smooth then restore positions")
    
    # NOTE: We no longer add corner vertices to excluded_set
    # They will be smoothed normally, then snapped back to original positions after
    
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
        
        # === Step 1b: Restore concave corner vertices after boundary smoothing ===
        # This ensures concave corners stay pinned during each iteration,
        # not just at the very end. This prevents smoothing from gradually
        # pulling them away from their correct positions.
        if corner_original_positions:
            for vi, original_pos in corner_original_positions.items():
                vertices[vi] = original_pos
        
        # === Step 2: Re-project boundary vertices onto target surfaces ===
        # Per paper: "After this smooth step, we re-project those vertices onto the 
        # original surface of M or the external boundary ∂H"
        #
        # Simplified approach: Only concave corners are kept fixed (not reprojected).
        # All other vertices use standard nearest-point reprojection.
        
        # Re-project to part mesh M (excluding concave corners)
        if len(part_boundary_indices) > 0 and part_mesh is not None:
            if part_proximity is not None:
                part_positions = vertices[part_boundary_indices]
                closest_pts, _, _ = part_proximity.on_surface(part_positions)
                # Guard against NaN/Inf from degenerate geometry
                valid_mask = np.all(np.isfinite(closest_pts), axis=1)
                if np.all(valid_mask):
                    vertices[part_boundary_indices] = closest_pts
                else:
                    n_invalid = int(np.sum(~valid_mask))
                    logger.warning(f"Part reprojection: {n_invalid} vertices returned NaN/Inf, keeping original positions")
                    vertices[part_boundary_indices[valid_mask]] = closest_pts[valid_mask]
        
        # Re-project to hull mesh ∂H (excluding concave corners)
        if len(hull_boundary_indices) > 0 and hull_mesh is not None:
            if hull_proximity is not None:
                hull_positions = vertices[hull_boundary_indices]
                closest_pts, _, _ = hull_proximity.on_surface(hull_positions)
                valid_mask = np.all(np.isfinite(closest_pts), axis=1)
                if np.all(valid_mask):
                    vertices[hull_boundary_indices] = closest_pts
                else:
                    n_invalid = int(np.sum(~valid_mask))
                    logger.warning(f"Hull reprojection: {n_invalid} vertices returned NaN/Inf, keeping original positions")
                    vertices[hull_boundary_indices[valid_mask]] = closest_pts[valid_mask]
        
        # Re-project to primary mesh (for secondary surfaces)
        # bt=3 vertices go to primary only.
        # bt=4 vertices go to primary first, then part (handled below).
        all_primary_indices = primary_boundary_indices
        if len(primary_then_part_indices) > 0:
            all_primary_indices = np.concatenate([primary_boundary_indices, primary_then_part_indices])
        
        if len(all_primary_indices) > 0 and primary_mesh is not None:
            if primary_proximity is not None:
                primary_positions = vertices[all_primary_indices]
                closest_pts, _, _ = primary_proximity.on_surface(primary_positions)
                valid_mask = np.all(np.isfinite(closest_pts), axis=1)
                if np.all(valid_mask):
                    vertices[all_primary_indices] = closest_pts
                else:
                    n_invalid = int(np.sum(~valid_mask))
                    logger.warning(f"Primary reprojection: {n_invalid} vertices returned NaN/Inf, keeping original positions")
                    vertices[all_primary_indices[valid_mask]] = closest_pts[valid_mask]
        
        # Dual re-projection for bt=4 (primary-then-part):
        # After being placed on the primary surface above, now snap to part.
        # This pins them to the curve where primary surface intersects part.
        if len(primary_then_part_indices) > 0 and part_mesh is not None:
            if part_proximity is not None:
                ptp_positions = vertices[primary_then_part_indices]
                closest_pts, _, _ = part_proximity.on_surface(ptp_positions)
                valid_mask = np.all(np.isfinite(closest_pts), axis=1)
                if np.all(valid_mask):
                    vertices[primary_then_part_indices] = closest_pts
                else:
                    n_invalid = int(np.sum(~valid_mask))
                    logger.warning(f"Primary→part reprojection: {n_invalid} vertices returned NaN/Inf")
                    vertices[primary_then_part_indices[valid_mask]] = closest_pts[valid_mask]
        
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
    
    # === Final restoration (for consistency after last interior smoothing) ===
    # Since interior smoothing can affect boundary vertices through neighbor averaging,
    # we do one final restoration to ensure concave corners are exactly at original positions.
    # Also store positions for visualization (blue spheres).
    if corner_original_positions:
        for vi, original_pos in corner_original_positions.items():
            vertices[vi] = original_pos
        logger.info(f"Restored {len(corner_original_positions)} concave corner vertices to original positions")
        
        # Store the restored positions for visualization (blue spheres)
        result.restored_corner_positions = np.array(list(corner_original_positions.values()))
    
    # Create smoothed mesh
    # CRITICAL: Use process=False to prevent merge_vertices() from renumbering
    # vertex indices. The caller relies on vertex_boundary_type staying aligned
    # with the vertex array — if merge_vertices() removes or reindexes vertices,
    # collar creation will look up wrong boundary types, causing inner/outer
    # edge misclassification and collar failures.
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Clean up any degenerate faces that might have been created
    valid_faces = smoothed_mesh.nondegenerate_faces()
    if np.sum(~valid_faces) > 0:
        n_removed = np.sum(~valid_faces)
        logger.debug(f"Removing {n_removed} degenerate faces after smoothing")
        # CRITICAL: Use process=False here too — a second process=True would
        # merge vertices again, further corrupting the index alignment.
        smoothed_mesh = trimesh.Trimesh(
            vertices=smoothed_mesh.vertices,
            faces=smoothed_mesh.faces[valid_faces],
            process=False
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
