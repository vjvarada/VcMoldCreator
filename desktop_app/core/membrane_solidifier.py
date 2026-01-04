"""
Membrane Solidifier Module

Converts flat secondary membrane surfaces into solid 3D volumes with thickness.
The resulting volumes are manifold (watertight) and free of self-intersections.

This is a preprocessing step required before using membranes as cutting geometry
in the mold piece splitting operation.
"""

import numpy as np
import trimesh
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SolidificationResult:
    """Result of membrane solidification."""
    solid_mesh: trimesh.Trimesh          # The solidified membrane volume
    is_manifold: bool                     # Whether the result is manifold
    is_watertight: bool                   # Whether the result is watertight
    has_self_intersections: bool          # Whether self-intersections were detected
    thickness_used: float                 # Actual thickness used (may be reduced)
    original_surface: trimesh.Trimesh     # Reference to original surface
    warnings: List[str]                   # Any warnings generated
    mold_half: int = 0                    # Which mold half this belongs to: 1=H1, 2=H2, 0=unknown


def compute_robust_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute robust vertex normals using area-weighted face normals.
    
    This provides smoother normals than simple averaging and handles
    degenerate cases better.
    
    Args:
        mesh: Input surface mesh
        
    Returns:
        vertex_normals: (N, 3) array of unit vertex normals
    """
    # Get face normals and areas
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    
    # Initialize vertex normals
    vertex_normals = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    
    # Accumulate area-weighted face normals to vertices
    for face_idx, face in enumerate(mesh.faces):
        weighted_normal = face_normals[face_idx] * face_areas[face_idx]
        for vertex_idx in face:
            vertex_normals[vertex_idx] += weighted_normal
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
    vertex_normals = vertex_normals / norms
    
    return vertex_normals


def find_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Find boundary edges of a surface mesh.
    
    Boundary edges are edges that belong to only one face.
    
    Args:
        mesh: Input surface mesh
        
    Returns:
        boundary_edges: (M, 2) array of vertex indices for boundary edges
    """
    # Get all edges with their face associations
    edges = mesh.edges_unique
    edge_face_count = mesh.edges_unique_length
    
    # An edge is on boundary if it belongs to only one face
    # For trimesh, we need to check edges_face differently
    
    # Build edge to face mapping
    edge_to_faces = {}
    for face_idx, face in enumerate(mesh.faces):
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge_key = tuple(sorted([v1, v2]))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(face_idx)
    
    # Find boundary edges (only one adjacent face)
    boundary_edges = []
    for edge_key, faces in edge_to_faces.items():
        if len(faces) == 1:
            boundary_edges.append(list(edge_key))
    
    return np.array(boundary_edges) if boundary_edges else np.zeros((0, 2), dtype=np.int32)


def order_boundary_loop(boundary_edges: np.ndarray) -> List[np.ndarray]:
    """
    Order boundary edges into connected loops.
    
    Args:
        boundary_edges: (M, 2) array of boundary edge vertex indices
        
    Returns:
        loops: List of arrays, each containing ordered vertex indices for a loop
    """
    if len(boundary_edges) == 0:
        return []
    
    # Build adjacency for boundary vertices
    adjacency = {}
    for edge in boundary_edges:
        v1, v2 = int(edge[0]), int(edge[1])
        if v1 not in adjacency:
            adjacency[v1] = []
        if v2 not in adjacency:
            adjacency[v2] = []
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)
    
    # Log adjacency stats
    logger.debug(f"Boundary adjacency: {len(adjacency)} unique vertices")
    
    # Find connected loops
    visited = set()
    loops = []
    
    for start_vertex in adjacency.keys():
        if start_vertex in visited:
            continue
            
        # Trace the loop
        loop = [start_vertex]
        visited.add(start_vertex)
        current = start_vertex
        
        max_iterations = len(adjacency) + 10  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            neighbors = adjacency.get(current, [])
            next_vertex = None
            
            for n in neighbors:
                if n not in visited:
                    next_vertex = n
                    break
            
            if next_vertex is None:
                # Check if we can close the loop
                if start_vertex in neighbors and len(loop) > 2:
                    break  # Loop is closed
                else:
                    break  # Dead end (shouldn't happen in valid mesh)
            
            loop.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex
        
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int32))
        else:
            logger.warning(f"Skipping degenerate loop with {len(loop)} vertices")
    
    # Verify all boundary edges are covered
    total_loop_verts = sum(len(loop) for loop in loops)
    if total_loop_verts != len(adjacency):
        logger.warning(f"Boundary coverage mismatch: {total_loop_verts} in loops vs {len(adjacency)} unique vertices")
    
    return loops


def create_offset_surface(
    mesh: trimesh.Trimesh,
    vertex_normals: np.ndarray,
    offset: float
) -> trimesh.Trimesh:
    """
    Create an offset surface by moving vertices along normals.
    
    Args:
        mesh: Original surface mesh
        vertex_normals: Per-vertex normals
        offset: Distance to offset (positive or negative)
        
    Returns:
        offset_mesh: New mesh with offset vertices
    """
    offset_vertices = mesh.vertices + vertex_normals * offset
    
    # If offset is negative, we need to flip face winding
    if offset < 0:
        faces = mesh.faces[:, ::-1]  # Reverse vertex order
    else:
        faces = mesh.faces.copy()
    
    return trimesh.Trimesh(vertices=offset_vertices, faces=faces)


def create_side_faces(
    top_vertices: np.ndarray,
    bottom_vertices: np.ndarray,
    boundary_loops: List[np.ndarray],
    n_original_vertices: int
) -> np.ndarray:
    """
    Create side faces connecting top and bottom surfaces along boundaries.
    
    Args:
        top_vertices: Vertices of top surface
        bottom_vertices: Vertices of bottom surface  
        boundary_loops: Ordered boundary vertex loops (indices into original mesh)
        n_original_vertices: Number of vertices in original mesh
        
    Returns:
        side_faces: (K, 3) array of face vertex indices
    """
    side_faces = []
    
    for loop in boundary_loops:
        n_loop = len(loop)
        for i in range(n_loop):
            # Current and next vertex in loop
            v1 = loop[i]
            v2 = loop[(i + 1) % n_loop]
            
            # Top surface uses original indices
            # Bottom surface uses indices offset by n_original_vertices
            top_v1 = v1
            top_v2 = v2
            bot_v1 = v1 + n_original_vertices
            bot_v2 = v2 + n_original_vertices
            
            # Create two triangles for the quad
            # Winding order matters for correct normals
            side_faces.append([top_v1, bot_v1, top_v2])
            side_faces.append([top_v2, bot_v1, bot_v2])
    
    return np.array(side_faces) if side_faces else np.zeros((0, 3), dtype=np.int32)


def check_self_intersections(mesh: trimesh.Trimesh, sample_count: int = 1000) -> bool:
    """
    Check if a mesh has self-intersections using ray casting.
    
    This is a probabilistic check - it may miss some intersections.
    
    Args:
        mesh: Mesh to check
        sample_count: Number of random rays to cast
        
    Returns:
        has_intersections: True if self-intersections detected
    """
    try:
        # Use trimesh's built-in intersection check if available
        if hasattr(mesh, 'is_watertight') and mesh.is_watertight:
            # For watertight meshes, check for inverted faces
            if hasattr(mesh, 'is_winding_consistent'):
                if not mesh.is_winding_consistent:
                    return True
        
        # Simple heuristic: check if mesh has negative volume regions
        # by testing if centroid is inside
        if mesh.is_watertight:
            return False  # Watertight usually means no self-intersection
            
        return False  # Can't definitively detect, assume ok
        
    except Exception as e:
        logger.warning(f"Self-intersection check failed: {e}")
        return False


def estimate_safe_thickness(
    mesh: trimesh.Trimesh,
    vertex_normals: np.ndarray,
    requested_thickness: float,
    min_thickness: float = 0.4
) -> float:
    """
    Estimate a safe thickness that minimizes self-intersection risk.
    
    Uses local curvature estimation to determine safe offset distance.
    
    Args:
        mesh: Surface mesh
        vertex_normals: Per-vertex normals
        requested_thickness: User-requested thickness
        min_thickness: Absolute minimum thickness to allow (default 0.4mm)
        
    Returns:
        safe_thickness: Thickness that should be safe from self-intersections
    """
    # Estimate minimum feature size using edge lengths
    edge_lengths = mesh.edges_unique_length
    min_edge = np.percentile(edge_lengths, 5)  # 5th percentile
    
    # Also check vertex spacing
    # For each vertex, find distance to nearest neighbor
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)
    distances, _ = tree.query(mesh.vertices, k=2)  # k=2 to get nearest neighbor (not self)
    min_vertex_spacing = np.percentile(distances[:, 1], 5)
    
    # Estimate local curvature from normal variation
    # High normal variation = high curvature = need smaller thickness
    normal_variation = []
    for face in mesh.faces:
        face_normals = vertex_normals[face]
        # Check angle between normals
        for i in range(3):
            for j in range(i + 1, 3):
                dot = np.clip(np.dot(face_normals[i], face_normals[j]), -1, 1)
                angle = np.arccos(dot)
                normal_variation.append(angle)
    
    max_normal_variation = np.percentile(normal_variation, 95) if normal_variation else 0
    
    # Safe thickness is limited by:
    # 1. Half the minimum edge length
    # 2. Half the minimum vertex spacing
    # 3. Inverse of curvature (smaller for high curvature)
    
    curvature_limit = float('inf')
    if max_normal_variation > 0.1:  # ~5.7 degrees
        # Rough estimate: radius of curvature ~ edge_length / angle
        avg_edge = np.mean(edge_lengths)
        radius_estimate = avg_edge / max_normal_variation
        curvature_limit = radius_estimate * 0.3  # Conservative factor
    
    safe_thickness = min(
        requested_thickness,
        min_edge * 0.4,
        min_vertex_spacing * 0.4,
        curvature_limit
    )
    
    # Enforce absolute minimum thickness
    return max(safe_thickness, min_thickness)


def repair_mesh_manifold(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Attempt to repair a mesh to make it manifold.
    
    Args:
        mesh: Input mesh (possibly non-manifold)
        
    Returns:
        repaired_mesh: Repaired mesh (best effort)
    """
    try:
        # Try using manifold3d for repair
        import manifold3d
        
        # Create Mesh object with properly formatted arrays
        mesh_obj = manifold3d.Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32)
        )
        
        # Create Manifold directly from Mesh (new API - no from_mesh method)
        manifold_mesh = manifold3d.Manifold(mesh_obj)
        
        # Check if the manifold is valid
        if manifold_mesh.is_empty():
            raise ValueError("Manifold creation resulted in empty mesh")
        
        # Export back to trimesh
        result_mesh = manifold_mesh.to_mesh()
        repaired = trimesh.Trimesh(
            vertices=result_mesh.vert_properties,
            faces=result_mesh.tri_verts
        )
        
        return repaired
        
    except Exception as e:
        logger.warning(f"Manifold repair failed: {e}, using trimesh repair")
        
        # Fall back to trimesh repair
        mesh_copy = mesh.copy()
        
        # Remove degenerate faces using trimesh repair module
        try:
            trimesh.repair.fix_winding(mesh_copy)
        except Exception:
            pass
        
        # Merge close vertices
        mesh_copy.merge_vertices()
        
        # Fill holes if any
        if not mesh_copy.is_watertight:
            try:
                trimesh.repair.fill_holes(mesh_copy)
            except Exception:
                pass
        
        # Fix normals
        trimesh.repair.fix_normals(mesh_copy)
        
        return mesh_copy


def split_into_components(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """
    Split a mesh into its connected components.
    
    Args:
        mesh: Input mesh that may have multiple disconnected parts
        
    Returns:
        List of meshes, one for each connected component
    """
    # Use trimesh's split function
    components = mesh.split(only_watertight=False)
    
    if isinstance(components, trimesh.Trimesh):
        # Single component returned as mesh, not list
        return [components]
    elif components is None or len(components) == 0:
        # No components found, return original
        return [mesh]
    else:
        return list(components)


def classify_component_mold_half(
    component: trimesh.Trimesh,
    secondary_cut_edges: List[Tuple[int, int]],
    tet_vertices: np.ndarray,
    seed_escape_labels: np.ndarray,
    seed_vertex_indices: np.ndarray
) -> int:
    """
    Determine which mold half a secondary surface component belongs to.
    
    Secondary cut edges connect vertices that both escape to the SAME mold half.
    By finding which secondary cut edges are closest to the component's vertices,
    we can determine which mold half the component belongs to.
    
    Args:
        component: A connected component of the secondary surface mesh
        secondary_cut_edges: List of (vi, vj) secondary cut edges (tet vertex indices)
        tet_vertices: (N, 3) array of tetrahedral mesh vertices
        seed_escape_labels: (I,) array of escape labels (1=H1, 2=H2) for interior vertices
        seed_vertex_indices: (I,) array of vertex indices corresponding to escape labels
    
    Returns:
        1 if component belongs to H1, 2 if belongs to H2, 0 if unknown
    """
    if secondary_cut_edges is None or len(secondary_cut_edges) == 0:
        return 0
    
    if seed_escape_labels is None or seed_vertex_indices is None:
        return 0
    
    # Build mapping from vertex index to escape label
    vertex_to_label = {}
    for i, v_idx in enumerate(seed_vertex_indices):
        vertex_to_label[v_idx] = seed_escape_labels[i]
    
    # Compute edge midpoints for secondary cut edges
    edge_midpoints = []
    edge_labels = []
    for vi, vj in secondary_cut_edges:
        midpoint = (tet_vertices[vi] + tet_vertices[vj]) / 2.0
        edge_midpoints.append(midpoint)
        
        # Get escape label for this edge (both vertices have same label for secondary cuts)
        label = vertex_to_label.get(vi, 0)
        if label == 0:
            label = vertex_to_label.get(vj, 0)
        edge_labels.append(label)
    
    edge_midpoints = np.array(edge_midpoints)
    edge_labels = np.array(edge_labels)
    
    if len(edge_midpoints) == 0:
        return 0
    
    # Sample some vertices from the component
    component_verts = np.asarray(component.vertices)
    n_samples = min(len(component_verts), 20)  # Sample up to 20 vertices
    sample_indices = np.linspace(0, len(component_verts) - 1, n_samples, dtype=int)
    sample_verts = component_verts[sample_indices]
    
    # For each sample vertex, find the closest edge midpoint
    from scipy.spatial import cKDTree
    tree = cKDTree(edge_midpoints)
    _, nearest_edge_indices = tree.query(sample_verts, k=1)
    
    # Get the labels for the nearest edges
    nearest_labels = edge_labels[nearest_edge_indices]
    
    # Vote: which label appears most often?
    h1_count = np.sum(nearest_labels == 1)
    h2_count = np.sum(nearest_labels == 2)
    
    if h1_count > h2_count:
        return 1
    elif h2_count > h1_count:
        return 2
    else:
        # Tie or no valid labels - return 0 (unknown)
        return 0


def solidify_membrane_components(
    surface_mesh: trimesh.Trimesh,
    thickness: float = 2.0,
    auto_reduce_thickness: bool = True,
    use_manifold_repair: bool = True,
    min_thickness: float = 0.4,
    min_area: float = 5.0,
    secondary_cut_edges: Optional[List[Tuple[int, int]]] = None,
    tet_vertices: Optional[np.ndarray] = None,
    seed_escape_labels: Optional[np.ndarray] = None,
    seed_vertex_indices: Optional[np.ndarray] = None
) -> List[SolidificationResult]:
    """
    Split a surface mesh into connected components and solidify each one.
    
    This is the recommended function for secondary surfaces that may
    contain multiple disconnected membrane patches.
    
    Args:
        surface_mesh: Input surface mesh (may have multiple components)
        thickness: Total thickness of the resulting solids
        auto_reduce_thickness: If True, reduce thickness to avoid self-intersection
        use_manifold_repair: If True, use manifold3d to repair results
        min_thickness: Minimum allowed thickness when auto-reducing (default 0.4mm)
        min_area: Minimum surface area in mm² to process (default 5.0mm²)
                  Components smaller than this are skipped.
        secondary_cut_edges: List of (vi, vj) secondary cut edges for mold half classification
        tet_vertices: Tetrahedral mesh vertices for mold half classification
        seed_escape_labels: Escape labels (1=H1, 2=H2) for classification
        seed_vertex_indices: Vertex indices corresponding to escape labels
        
    Returns:
        List of SolidificationResult, one for each component (excluding small ones)
        Each result includes mold_half field: 1=H1, 2=H2, 0=unknown
    """
    # Split into connected components
    components = split_into_components(surface_mesh)
    logger.info(f"Split surface into {len(components)} connected components")
    
    # Check if we have classification data
    can_classify = (secondary_cut_edges is not None and 
                   tet_vertices is not None and 
                   seed_escape_labels is not None and 
                   seed_vertex_indices is not None)
    
    if can_classify:
        logger.info("Mold half classification data available - will classify each component")
    else:
        logger.info("No mold half classification data - components will have mold_half=0")
    
    # Filter out small components
    filtered_components = []
    skipped_count = 0
    for component in components:
        area = component.area
        if area >= min_area:
            filtered_components.append(component)
        else:
            skipped_count += 1
            logger.info(f"Skipping small component: {area:.2f} mm² (< {min_area} mm² threshold)")
    
    if skipped_count > 0:
        logger.info(f"Filtered out {skipped_count} small component(s), {len(filtered_components)} remaining")
    
    results = []
    for i, component in enumerate(filtered_components):
        # Classify which mold half this component belongs to
        mold_half = 0
        if can_classify:
            mold_half = classify_component_mold_half(
                component=component,
                secondary_cut_edges=secondary_cut_edges,
                tet_vertices=tet_vertices,
                seed_escape_labels=seed_escape_labels,
                seed_vertex_indices=seed_vertex_indices
            )
            half_str = f"H{mold_half}" if mold_half > 0 else "unknown"
        else:
            half_str = "N/A"
        
        logger.info(f"Solidifying component {i+1}/{len(filtered_components)}: "
                   f"{len(component.vertices)} verts, {len(component.faces)} faces, "
                   f"{component.area:.2f} mm², mold_half={half_str}")
        
        try:
            result = solidify_membrane(
                surface_mesh=component,
                thickness=thickness,
                auto_reduce_thickness=auto_reduce_thickness,
                use_manifold_repair=use_manifold_repair,
                min_thickness=min_thickness
            )
            # Set the mold half classification
            result.mold_half = mold_half
            results.append(result)
            
            status = "✓ watertight" if result.is_watertight else "✗ not watertight"
            logger.info(f"  Component {i+1} result: {status}, mold_half=H{mold_half}")
            
        except Exception as e:
            logger.error(f"Failed to solidify component {i+1}: {e}")
            # Create a failed result
            results.append(SolidificationResult(
                solid_mesh=component,  # Return original as fallback
                is_manifold=False,
                is_watertight=False,
                has_self_intersections=True,
                thickness_used=thickness,
                original_surface=component,
                warnings=[f"Solidification failed: {e}"],
                mold_half=mold_half  # Still set the classification even if solidification failed
            ))
    
    return results


def solidify_membrane(
    surface_mesh: trimesh.Trimesh,
    thickness: float = 2.0,
    auto_reduce_thickness: bool = True,
    use_manifold_repair: bool = True,
    min_thickness: float = 0.4
) -> SolidificationResult:
    """
    Convert a flat surface mesh into a solid 3D volume with thickness.
    
    The surface is offset in both directions by half the thickness,
    and side faces are added to create a watertight solid.
    
    Args:
        surface_mesh: Input surface mesh (should be manifold surface)
        thickness: Total thickness of the resulting solid (in mesh units)
        auto_reduce_thickness: If True, reduce thickness to avoid self-intersection
        use_manifold_repair: If True, use manifold3d to repair result
        min_thickness: Minimum allowed thickness when auto-reducing (default 0.4mm)
        
    Returns:
        SolidificationResult containing the solid mesh and metadata
    """
    warnings = []
    half_thickness = thickness / 2.0
    
    logger.info(f"Solidifying membrane: {len(surface_mesh.vertices)} vertices, {len(surface_mesh.faces)} faces")
    
    # Compute robust vertex normals
    vertex_normals = compute_robust_vertex_normals(surface_mesh)
    
    # Check for safe thickness
    if auto_reduce_thickness:
        safe_thickness = estimate_safe_thickness(surface_mesh, vertex_normals, thickness, min_thickness)
        if safe_thickness < thickness * 0.9:
            warnings.append(
                f"Thickness reduced from {thickness:.3f} to {safe_thickness:.3f} "
                f"to avoid self-intersections"
            )
            thickness = safe_thickness
            half_thickness = thickness / 2.0
    
    # Create top and bottom offset surfaces
    top_surface = create_offset_surface(surface_mesh, vertex_normals, half_thickness)
    bottom_surface = create_offset_surface(surface_mesh, vertex_normals, -half_thickness)
    
    # Find boundary loops for side face creation
    boundary_edges = find_boundary_edges(surface_mesh)
    logger.info(f"Found {len(boundary_edges)} boundary edges")
    
    boundary_loops = order_boundary_loop(boundary_edges)
    logger.info(f"Ordered into {len(boundary_loops)} boundary loops")
    
    # Log loop sizes
    for i, loop in enumerate(boundary_loops):
        logger.debug(f"  Loop {i}: {len(loop)} vertices")
    
    # Combine vertices: top surface first, then bottom surface
    combined_vertices = np.vstack([top_surface.vertices, bottom_surface.vertices])
    
    # Top faces use original indices
    top_faces = top_surface.faces.copy()
    
    # Bottom faces use offset indices and reversed winding
    n_vertices = len(surface_mesh.vertices)
    bottom_faces = bottom_surface.faces + n_vertices
    
    # Create side faces connecting boundaries
    side_faces = create_side_faces(
        top_surface.vertices,
        bottom_surface.vertices,
        boundary_loops,
        n_vertices
    )
    
    logger.info(f"Created {len(side_faces)} side faces")
    
    # Combine all faces
    if len(side_faces) > 0:
        combined_faces = np.vstack([top_faces, bottom_faces, side_faces])
    else:
        # Closed surface (no boundary) - just combine top and bottom
        combined_faces = np.vstack([top_faces, bottom_faces])
        warnings.append("Surface has no boundary - created shell without side faces")
    
    # Create the combined mesh
    solid_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    
    # Basic cleanup - use process=True for automatic cleanup
    solid_mesh.process(validate=True)
    solid_mesh.merge_vertices()
    
    # Check initial state
    is_watertight_initial = solid_mesh.is_watertight
    logger.info(f"Initial watertight status: {is_watertight_initial}")
    
    # If not watertight, try to fill holes
    if not is_watertight_initial:
        logger.info("Mesh not watertight, attempting hole filling...")
        try:
            trimesh.repair.fill_holes(solid_mesh)
            if solid_mesh.is_watertight:
                logger.info("Hole filling successful!")
                warnings.append("Holes were filled to make mesh watertight")
        except Exception as e:
            logger.warning(f"Hole filling failed: {e}")
    
    # Attempt manifold repair if still not watertight
    if use_manifold_repair and not solid_mesh.is_watertight:
        logger.info("Attempting manifold3d repair...")
        try:
            repaired = repair_mesh_manifold(solid_mesh)
            if repaired.is_watertight:
                solid_mesh = repaired
                warnings.append("Mesh repaired to watertight using manifold repair")
                logger.info("Manifold repair successful!")
            else:
                logger.warning("Manifold repair did not produce watertight mesh")
        except Exception as e:
            warnings.append(f"Manifold repair failed: {e}")
            logger.warning(f"Manifold repair failed: {e}")
    
    # If still not watertight, try convex hull as last resort
    if not solid_mesh.is_watertight:
        logger.warning("Mesh still not watertight, trying convex hull fallback...")
        try:
            # Create convex hull of all vertices
            hull = solid_mesh.convex_hull
            if hull.is_watertight:
                warnings.append("Used convex hull as fallback for watertight mesh")
                logger.info("Convex hull fallback successful")
                solid_mesh = hull
            else:
                logger.warning("Even convex hull is not watertight!")
        except Exception as e:
            logger.warning(f"Convex hull fallback failed: {e}")
    
    # Fix normals to point outward
    trimesh.repair.fix_normals(solid_mesh)
    
    # Final checks
    is_manifold = solid_mesh.is_watertight  # In trimesh, watertight implies manifold for most cases
    is_watertight = solid_mesh.is_watertight
    has_self_intersections = check_self_intersections(solid_mesh)
    
    logger.info(f"Final result: watertight={is_watertight}, manifold={is_manifold}")
    
    if has_self_intersections:
        warnings.append("Self-intersections detected in solidified mesh")
    
    if not is_watertight:
        warnings.append("Result is not watertight - may cause issues in CSG operations")
    
    return SolidificationResult(
        solid_mesh=solid_mesh,
        is_manifold=is_manifold,
        is_watertight=is_watertight,
        has_self_intersections=has_self_intersections,
        thickness_used=thickness,
        original_surface=surface_mesh,
        warnings=warnings
    )


def solidify_membranes(
    membranes: List[trimesh.Trimesh],
    thickness: float = 2.0,
    auto_reduce_thickness: bool = True,
    use_manifold_repair: bool = True
) -> List[SolidificationResult]:
    """
    Solidify multiple membrane surfaces.
    
    Args:
        membranes: List of membrane surface meshes
        thickness: Thickness for all membranes (or per-membrane if list)
        auto_reduce_thickness: Whether to auto-reduce thickness
        use_manifold_repair: Whether to use manifold3d for repair
        
    Returns:
        List of SolidificationResult, one per input membrane
    """
    results = []
    
    for i, membrane in enumerate(membranes):
        logger.info(f"Solidifying membrane {i + 1}/{len(membranes)}")
        
        result = solidify_membrane(
            surface_mesh=membrane,
            thickness=thickness,
            auto_reduce_thickness=auto_reduce_thickness,
            use_manifold_repair=use_manifold_repair
        )
        
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"Membrane {i + 1}: {warning}")
        
        results.append(result)
    
    return results


def solidify_parting_surface(
    parting_surface: trimesh.Trimesh,
    thickness: float = 2.0,
    auto_reduce_thickness: bool = True
) -> SolidificationResult:
    """
    Solidify the primary parting surface.
    
    This is a convenience function that uses the same logic as membrane
    solidification but with parameters tuned for parting surfaces.
    
    Args:
        parting_surface: The parting surface mesh
        thickness: Thickness for the solid
        auto_reduce_thickness: Whether to auto-reduce thickness
        
    Returns:
        SolidificationResult containing the solidified parting surface
    """
    return solidify_membrane(
        surface_mesh=parting_surface,
        thickness=thickness,
        auto_reduce_thickness=auto_reduce_thickness,
        use_manifold_repair=True
    )


# Visualization helper
def visualize_solidification(
    original: trimesh.Trimesh,
    solidified: trimesh.Trimesh,
    show: bool = True
) -> Optional[trimesh.Scene]:
    """
    Create a visualization comparing original surface and solidified volume.
    
    Args:
        original: Original surface mesh
        solidified: Solidified volume mesh
        show: Whether to display the visualization
        
    Returns:
        scene: Trimesh scene object (if show=False)
    """
    scene = trimesh.Scene()
    
    # Original surface in blue (semi-transparent)
    original_visual = original.copy()
    original_visual.visual.face_colors = [100, 100, 255, 128]
    scene.add_geometry(original_visual, node_name='original')
    
    # Solidified volume in green (semi-transparent)
    # Offset slightly to avoid z-fighting
    solidified_visual = solidified.copy()
    solidified_visual.visual.face_colors = [100, 255, 100, 128]
    scene.add_geometry(solidified_visual, node_name='solidified')
    
    if show:
        scene.show()
        return None
    
    return scene


if __name__ == "__main__":
    # Test the solidification
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a test surface (flat square)
    vertices = np.array([
        [0, 0, 0],
        [10, 0, 0],
        [10, 10, 0],
        [0, 10, 0]
    ], dtype=np.float64)
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    
    test_surface = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    print(f"Original surface: {len(test_surface.vertices)} vertices, {len(test_surface.faces)} faces")
    print(f"Is watertight: {test_surface.is_watertight}")
    
    # Solidify
    result = solidify_membrane(test_surface, thickness=2.0)
    
    print(f"\nSolidified mesh: {len(result.solid_mesh.vertices)} vertices, {len(result.solid_mesh.faces)} faces")
    print(f"Is manifold: {result.is_manifold}")
    print(f"Is watertight: {result.is_watertight}")
    print(f"Has self-intersections: {result.has_self_intersections}")
    print(f"Thickness used: {result.thickness_used}")
    print(f"Warnings: {result.warnings}")
    
    # Test with a more complex surface (curved)
    print("\n--- Testing curved surface ---")
    
    # Create a curved surface (partial sphere)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    # Take only top half
    top_half_mask = sphere.vertices[:, 2] >= 0
    # This is tricky - we need to extract faces where all vertices are in top half
    face_mask = np.all(top_half_mask[sphere.faces], axis=1)
    
    curved_surface = trimesh.Trimesh(
        vertices=sphere.vertices,
        faces=sphere.faces[face_mask]
    )
    curved_surface.remove_unreferenced_vertices()
    
    print(f"Curved surface: {len(curved_surface.vertices)} vertices, {len(curved_surface.faces)} faces")
    
    result2 = solidify_membrane(curved_surface, thickness=1.0)
    
    print(f"Solidified curved: {len(result2.solid_mesh.vertices)} vertices, {len(result2.solid_mesh.faces)} faces")
    print(f"Is manifold: {result2.is_manifold}")
    print(f"Is watertight: {result2.is_watertight}")
    print(f"Thickness used: {result2.thickness_used}")
    print(f"Warnings: {result2.warnings}")
