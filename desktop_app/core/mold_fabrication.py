"""
Mold Fabrication Module

Creates the hard shell geometry for composite mold fabrication.

From the paper (Section 5):
> "The hard shell is a prism aligned with the resin pouring direction, from which
> we then subtract the inner soft mold volume; the prism's flat base is orthogonal
> to the resin pouring direction and shaped like the silhouette of the convex hull."

This module implements:
1. Outer shell hull creation (offset from current hull by wall_thickness)
2. Outer collar extension (extend parting surface to outer shell boundary)
3. Prism creation from extended parting surface boundary
4. Shell half separation

Author: VcMoldCreator
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set
import numpy as np

import trimesh
from scipy.spatial import ConvexHull

# CSG operations via manifold3d
try:
    import manifold3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("manifold3d not available - CSG operations will be disabled")

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_WALL_THICKNESS_MM = 5.0  # Default hard shell wall thickness
DEFAULT_COLLAR_SUBDIVISIONS = 4  # Fan subdivisions at corners


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OuterShellHullResult:
    """Result of creating the outer shell hull (offset from inner hull)."""
    
    # The outer shell hull mesh
    mesh: trimesh.Trimesh
    
    # The original inner hull for reference
    inner_hull: trimesh.Trimesh
    
    # Wall thickness used
    wall_thickness: float
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass
class OuterCollarResult:
    """Result of extending parting surface outward to shell boundary."""
    
    # The extended parting surface mesh (with outer collar)
    mesh: trimesh.Trimesh
    
    # Vertex positions
    vertices: np.ndarray
    faces: np.ndarray
    
    # Original parting surface vertex count (before extension)
    original_vertex_count: int = 0
    original_face_count: int = 0
    
    # New vertices/faces added for the collar
    collar_vertex_count: int = 0
    collar_face_count: int = 0
    
    # Extension distance used (2x wall thickness to fully cut prism)
    extension_distance: float = 0.0
    
    # Which vertices are on the outer boundary (after extension)
    outer_boundary_vertices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    
    # The outer boundary loop (ordered vertex indices)
    outer_boundary_loop: List[int] = field(default_factory=list)
    
    # Statistics
    n_outer_boundary_edges_processed: int = 0
    n_corner_fans_created: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass 
class HardShellHalfResult:
    """Result of generating one half of the hard shell."""
    
    # The shell half mesh
    mesh: trimesh.Trimesh
    
    # Which mold half this belongs to (1 or 2)
    mold_half: int
    
    # The prism base polygon (2D silhouette)
    prism_base_vertices: np.ndarray  # (N, 2)
    
    # The pouring direction used
    pouring_direction: np.ndarray  # (3,)
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass
class HardShellPrismResult:
    """Result of creating the hard shell prism.
    
    The prism is aligned with the resin pouring direction, with flat bases
    orthogonal to that direction, shaped like the silhouette of the convex hull.
    (per paper Section 5)
    """
    
    # The full prism mesh (before subtracting mold cavity)
    prism_mesh: trimesh.Trimesh
    
    # The 2D silhouette polygon (convex hull projection)
    silhouette_2d: np.ndarray  # (N, 2) ordered vertices of 2D convex hull
    
    # The 3D silhouette polygon (on the base plane)
    silhouette_3d: np.ndarray  # (N, 3) 3D positions on base plane
    
    # The base plane center and normal
    base_plane_center: np.ndarray  # (3,)
    base_plane_normal: np.ndarray  # (3,) = pouring direction
    
    # Prism height (total extrusion distance)
    prism_height: float
    
    # The transformation matrix from world to 2D projection plane
    world_to_2d: np.ndarray  # (3, 3) rotation matrix
    
    # The pouring direction used
    pouring_direction: np.ndarray  # (3,)
    
    # Wall thickness added around the silhouette
    wall_thickness: float
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass
class HardShellSplitResult:
    """Result of CSG operations to create two hard shell halves.
    
    The workflow is:
    1. Subtract hull from prism to create cavity
    2. Split along parting surface (with extended collar) into two halves
    """
    
    # The two shell halves (H1 and H2)
    shell_half_1: Optional[trimesh.Trimesh] = None  # Upper half (aligned with d1)
    shell_half_2: Optional[trimesh.Trimesh] = None  # Lower half (aligned with d2)
    
    # Intermediate result: prism with cavity (before splitting)
    shell_with_cavity: Optional[trimesh.Trimesh] = None
    
    # The pouring direction used
    pouring_direction: Optional[np.ndarray] = None
    
    # Statistics for each half
    half_1_vertex_count: int = 0
    half_1_face_count: int = 0
    half_2_vertex_count: int = 0
    half_2_face_count: int = 0
    
    # Computation times for each step
    cavity_time_ms: float = 0.0
    split_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Success flags
    cavity_success: bool = False
    split_success: bool = False


# ============================================================================
# HARD SHELL PRISM CREATION (Paper Section 5)
# ============================================================================

def create_hard_shell_prism(
    hull_mesh: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    wall_thickness: float = DEFAULT_WALL_THICKNESS_MM,
    margin: float = 2.0
) -> HardShellPrismResult:
    """
    Create the hard shell prism aligned with the pouring direction.
    
    From the paper (Section 5):
    > "The hard shell is a prism aligned with the resin pouring direction, from which
    > we then subtract the inner soft mold volume; the prism's flat base is orthogonal
    > to the resin pouring direction and shaped like the silhouette of the convex hull."
    
    Algorithm:
    1. Define a plane perpendicular to the pouring direction
    2. Project all hull vertices onto this plane
    3. Compute the 2D convex hull of the projected points (silhouette)
    4. Offset the silhouette outward by wall_thickness
    5. Extrude the silhouette along the pouring direction to create a prism
    
    Args:
        hull_mesh: The inflated hull mesh (∂H)
        pouring_direction: Unit vector for resin pouring direction (f_resin)
        wall_thickness: How thick the hard shell wall should be (mm)
        margin: Additional margin beyond the hull bounds (mm)
        
    Returns:
        HardShellPrismResult with the prism mesh and metadata
    """
    start_time = time.time()
    
    logger.info(f"Creating hard shell prism aligned with pouring direction: {pouring_direction}")
    
    # Normalize pouring direction
    pouring_dir = np.array(pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    vertices = np.array(hull_mesh.vertices, dtype=np.float64)
    
    # =========================================================================
    # Step 1: Build orthonormal basis for the projection plane
    # =========================================================================
    
    # The plane normal is the pouring direction
    # Find two orthogonal vectors in the plane
    
    # Start with an arbitrary non-parallel vector
    if abs(pouring_dir[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt to get orthogonal basis
    u_axis = arbitrary - np.dot(arbitrary, pouring_dir) * pouring_dir
    u_axis = u_axis / np.linalg.norm(u_axis)
    
    v_axis = np.cross(pouring_dir, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)
    
    # Rotation matrix: world_to_2d projects world coordinates to (u, v) plane
    world_to_2d = np.column_stack([u_axis, v_axis, pouring_dir]).T  # (3, 3)
    
    # =========================================================================
    # Step 2: Project vertices onto the plane and compute 2D silhouette
    # =========================================================================
    
    # Project vertices: get (u, v) coordinates  
    vertices_transformed = vertices @ world_to_2d.T
    projected_2d = vertices_transformed[:, :2]  # (N, 2)
    heights = vertices_transformed[:, 2]  # projection along pouring direction
    
    # Compute 2D convex hull (silhouette)
    try:
        hull_2d = ConvexHull(projected_2d)
        silhouette_indices = hull_2d.vertices
        silhouette_2d = projected_2d[silhouette_indices]
    except Exception as e:
        logger.error(f"Failed to compute 2D convex hull: {e}")
        # Fallback: use bounding box
        min_2d = np.min(projected_2d, axis=0)
        max_2d = np.max(projected_2d, axis=0)
        silhouette_2d = np.array([
            [min_2d[0], min_2d[1]],
            [max_2d[0], min_2d[1]],
            [max_2d[0], max_2d[1]],
            [min_2d[0], max_2d[1]],
        ], dtype=np.float64)
    
    logger.info(f"2D silhouette has {len(silhouette_2d)} vertices")
    
    # =========================================================================
    # Step 3: Offset the silhouette outward by wall_thickness + margin
    # =========================================================================
    
    total_offset = wall_thickness + margin
    silhouette_2d_offset = _offset_polygon_2d(silhouette_2d, total_offset)
    
    logger.info(f"Offset silhouette has {len(silhouette_2d_offset)} vertices")
    
    # =========================================================================
    # Step 4: Determine prism height (extrusion distance)
    # =========================================================================
    
    min_height = np.min(heights) - total_offset
    max_height = np.max(heights) + total_offset
    prism_height = max_height - min_height
    
    logger.info(f"Prism height: {prism_height:.2f} (from {min_height:.2f} to {max_height:.2f})")
    
    # =========================================================================
    # Step 5: Create prism mesh by extruding the silhouette
    # =========================================================================
    
    prism_mesh, silhouette_3d_bottom, silhouette_3d_top = _extrude_polygon_to_prism(
        silhouette_2d_offset,
        world_to_2d,
        min_height,
        max_height
    )
    
    # Compute base plane center (centroid of bottom silhouette)
    base_plane_center = np.mean(silhouette_3d_bottom, axis=0)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    result = HardShellPrismResult(
        prism_mesh=prism_mesh,
        silhouette_2d=silhouette_2d_offset,
        silhouette_3d=silhouette_3d_bottom,
        base_plane_center=base_plane_center,
        base_plane_normal=pouring_dir,
        prism_height=prism_height,
        world_to_2d=world_to_2d,
        pouring_direction=pouring_dir,
        wall_thickness=wall_thickness,
        vertex_count=len(prism_mesh.vertices),
        face_count=len(prism_mesh.faces),
        computation_time_ms=elapsed_ms
    )
    
    logger.info(f"Hard shell prism created: {result.vertex_count} vertices, "
                f"{result.face_count} faces in {elapsed_ms:.1f}ms")
    
    return result


# ============================================================================
# CSG OPERATIONS FOR HARD SHELL (Cavity + Split)
# ============================================================================

def _trimesh_to_manifold(mesh: trimesh.Trimesh) -> 'manifold3d.Manifold':
    """Convert a trimesh to manifold3d Manifold object."""
    if not MANIFOLD_AVAILABLE:
        raise RuntimeError("manifold3d is not available")
    
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    
    # Create Mesh then Manifold (correct API for manifold3d v3.x)
    m3d_mesh = manifold3d.Mesh(vert_properties=vertices, tri_verts=faces)
    return manifold3d.Manifold(m3d_mesh)


def _manifold_to_trimesh(manifold: 'manifold3d.Manifold') -> trimesh.Trimesh:
    """Convert a manifold3d Manifold object to trimesh."""
    mesh_data = manifold.to_mesh()
    vertices = np.asarray(mesh_data.vert_properties, dtype=np.float64)
    faces = np.asarray(mesh_data.tri_verts, dtype=np.int64)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def create_shell_with_cavity(
    prism_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Subtract the hull (mold cavity) from the prism to create the shell.
    
    This implements: shell = prism - hull
    
    Args:
        prism_mesh: The hard shell prism mesh
        hull_mesh: The inflated hull mesh (defines the cavity)
        
    Returns:
        Tuple of (shell_with_cavity, computation_time_ms, success)
    """
    start_time = time.time()
    
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available - cannot perform CSG operations")
        return None, 0.0, False
    
    try:
        logger.info("Creating shell with cavity (prism - hull)...")
        
        # Convert to manifold
        prism_manifold = _trimesh_to_manifold(prism_mesh)
        hull_manifold = _trimesh_to_manifold(hull_mesh)
        
        # Perform subtraction: shell = prism - hull
        shell_manifold = prism_manifold - hull_manifold
        
        # Convert back to trimesh
        shell_mesh = _manifold_to_trimesh(shell_manifold)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell with cavity created: {len(shell_mesh.vertices)} vertices, "
                   f"{len(shell_mesh.faces)} faces in {elapsed_ms:.1f}ms")
        
        return shell_mesh, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to create shell with cavity: {e}")
        return None, elapsed_ms, False


def _create_cutting_volume_from_surface(
    parting_surface: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    thickness: float = 100.0
) -> trimesh.Trimesh:
    """
    Create a thick volume from the parting surface for CSG splitting.
    
    The volume is created by extruding the parting surface along the pouring direction
    in both directions, creating a "slab" that can be used for CSG operations.
    
    Args:
        parting_surface: The parting surface mesh (with extended collar)
        pouring_direction: The pouring direction (extrusion axis)
        thickness: Total thickness of the slab (extends thickness/2 in each direction)
        
    Returns:
        A watertight volume mesh for CSG operations
    """
    # Normalize direction
    direction = np.array(pouring_direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Get surface vertices and faces
    vertices = np.asarray(parting_surface.vertices, dtype=np.float64)
    faces = np.asarray(parting_surface.faces, dtype=np.int64)
    n_verts = len(vertices)
    n_faces = len(faces)
    
    half_thickness = thickness / 2.0
    
    # Create top and bottom surfaces by offsetting along the direction
    top_vertices = vertices + direction * half_thickness
    bottom_vertices = vertices - direction * half_thickness
    
    # Combined vertices: [bottom, top]
    all_vertices = np.vstack([bottom_vertices, top_vertices])
    
    # Faces for bottom surface (reverse winding for outward normals)
    bottom_faces = faces[:, ::-1]  # Reverse winding
    
    # Faces for top surface (offset indices)
    top_faces = faces + n_verts
    
    # Side faces: find boundary edges and create quads
    edge_count = {}
    edge_to_face = {}
    
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
            # Store oriented edge info for winding determination
            if edge_key not in edge_to_face:
                edge_to_face[edge_key] = (fi, v0, v1)
    
    # Boundary edges have count == 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    
    side_faces = []
    for v0, v1 in boundary_edges:
        # Get the original edge orientation from the face
        _, orig_v0, orig_v1 = edge_to_face[(v0, v1)]
        
        # Bottom vertices
        b0, b1 = v0, v1
        # Top vertices (offset by n_verts)
        t0, t1 = v0 + n_verts, v1 + n_verts
        
        # Create quad (2 triangles) with correct winding
        # The winding depends on the original edge direction in the face
        if orig_v0 == v0:
            # Edge goes v0->v1 in original face, so outward normal faces "right"
            side_faces.append([b0, t0, t1])
            side_faces.append([b0, t1, b1])
        else:
            # Edge goes v1->v0 in original face
            side_faces.append([b0, b1, t1])
            side_faces.append([b0, t1, t0])
    
    # Combine all faces
    all_faces = np.vstack([
        bottom_faces,
        top_faces,
        np.array(side_faces, dtype=np.int64) if side_faces else np.zeros((0, 3), dtype=np.int64)
    ])
    
    volume_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_faces,
        process=True
    )
    volume_mesh.fix_normals()
    
    logger.debug(f"Created cutting volume: {len(all_vertices)} verts, {len(all_faces)} faces, "
                f"from {n_verts} surface verts, {len(boundary_edges)} boundary edges")
    
    return volume_mesh


def _create_half_space_from_membrane(
    membrane: trimesh.Trimesh,
    direction: np.ndarray,
    extrusion_distance: float = 500.0
) -> trimesh.Trimesh:
    """
    Create a watertight half-space volume bounded by the membrane on one side.
    
    This extrudes the membrane along the given direction to create a closed volume
    representing "everything on one side of the membrane".
    
    The resulting volume has:
    - The membrane as its bottom face (with appropriate winding)
    - The extruded membrane as its top face
    - Side faces connecting the membrane boundary to the extruded boundary
    
    Args:
        membrane: The membrane mesh (e.g., outer collar)
        direction: Unit vector for extrusion direction (defines "which side")
        extrusion_distance: How far to extrude (should be larger than the shell)
        
    Returns:
        A watertight volume mesh representing the half-space
    """
    # Normalize direction
    direction = np.array(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Get membrane geometry
    vertices = np.asarray(membrane.vertices, dtype=np.float64)
    faces = np.asarray(membrane.faces, dtype=np.int64)
    n_verts = len(vertices)
    
    # Create extruded vertices (far end of the half-space)
    extruded_vertices = vertices + direction * extrusion_distance
    
    # Combined vertices: [membrane, extruded]
    all_vertices = np.vstack([vertices, extruded_vertices])
    
    # Faces:
    # 1. Membrane faces (bottom) - reverse winding so normals point into the half-space
    #    (i.e., pointing in the negative direction)
    bottom_faces = faces[:, ::-1]  # Reversed winding
    
    # 2. Extruded faces (top) - keep original winding, offset indices
    #    These normals point outward (in positive direction)
    top_faces = faces + n_verts
    
    # 3. Side faces connecting boundary edges
    # Find boundary edges of membrane
    edge_count = {}
    edge_to_face = {}
    
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
            if edge_key not in edge_to_face:
                edge_to_face[edge_key] = (fi, v0, v1)
    
    # Boundary edges have count == 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    
    side_faces = []
    for v0, v1 in boundary_edges:
        _, orig_v0, orig_v1 = edge_to_face[(v0, v1)]
        
        # Bottom (membrane) vertices
        b0, b1 = v0, v1
        # Top (extruded) vertices
        t0, t1 = v0 + n_verts, v1 + n_verts
        
        # Create quad with correct winding for outward-facing normals
        # Since bottom faces are reversed, side faces should create outward normals
        if orig_v0 == v0:
            side_faces.append([b0, b1, t1])
            side_faces.append([b0, t1, t0])
        else:
            side_faces.append([b0, t0, t1])
            side_faces.append([b0, t1, b1])
    
    # Combine all faces
    all_faces_list = [bottom_faces, top_faces]
    if side_faces:
        all_faces_list.append(np.array(side_faces, dtype=np.int64))
    all_faces = np.vstack(all_faces_list)
    
    half_space_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_faces,
        process=True
    )
    half_space_mesh.fix_normals()
    
    logger.debug(f"Created half-space volume: {len(all_vertices)} verts, {len(all_faces)} faces, "
                f"extruded {extrusion_distance:.1f}mm in direction {direction}")
    
    return half_space_mesh


def _create_cutting_blade_from_membrane(
    membrane: trimesh.Trimesh,
    direction: np.ndarray,
    thickness: float = 0.0005
) -> trimesh.Trimesh:
    """
    Create a thin watertight "blade" volume from the membrane for cutting.
    
    The blade is created by extruding the membrane slightly in both directions
    (+thickness/2 and -thickness/2) to create a thin solid volume that can be
    subtracted from the shell to create a gap.
    
    Args:
        membrane: The membrane mesh (e.g., outer collar)
        direction: Unit vector perpendicular to the membrane (pouring direction)
        thickness: Total thickness of the blade (default: 0.0005mm = 0.5 micron)
        
    Returns:
        A watertight blade volume mesh
    """
    # Normalize direction
    direction = np.array(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Get membrane geometry
    vertices = np.asarray(membrane.vertices, dtype=np.float64)
    faces = np.asarray(membrane.faces, dtype=np.int64)
    n_verts = len(vertices)
    
    half_thickness = thickness / 2.0
    
    # Create vertices for both sides of the blade
    # Bottom side: membrane shifted in -direction
    bottom_vertices = vertices - direction * half_thickness
    # Top side: membrane shifted in +direction
    top_vertices = vertices + direction * half_thickness
    
    # Combined vertices: [bottom, top]
    all_vertices = np.vstack([bottom_vertices, top_vertices])
    
    # Faces:
    # 1. Bottom faces - reverse winding so normals point outward (downward)
    bottom_faces = faces[:, ::-1]  # Reversed winding
    
    # 2. Top faces - keep original winding, offset indices
    top_faces = faces + n_verts
    
    # 3. Side faces connecting boundary edges
    edge_count = {}
    edge_to_face = {}
    
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
            if edge_key not in edge_to_face:
                edge_to_face[edge_key] = (fi, v0, v1)
    
    # Boundary edges have count == 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    
    side_faces = []
    for v0, v1 in boundary_edges:
        _, orig_v0, orig_v1 = edge_to_face[(v0, v1)]
        
        # Bottom vertices
        b0, b1 = v0, v1
        # Top vertices (offset by n_verts)
        t0, t1 = v0 + n_verts, v1 + n_verts
        
        # Create quad with correct winding for outward-facing normals
        if orig_v0 == v0:
            side_faces.append([b0, b1, t1])
            side_faces.append([b0, t1, t0])
        else:
            side_faces.append([b0, t0, t1])
            side_faces.append([b0, t1, b1])
    
    # Combine all faces
    all_faces_list = [bottom_faces, top_faces]
    if side_faces:
        all_faces_list.append(np.array(side_faces, dtype=np.int64))
    all_faces = np.vstack(all_faces_list)
    
    blade_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_faces,
        process=True
    )
    blade_mesh.fix_normals()
    
    logger.debug(f"Created cutting blade: {len(all_vertices)} verts, {len(all_faces)} faces, "
                f"thickness={thickness:.6f}mm")
    
    return blade_mesh


def split_shell_with_membrane(
    shell_with_cavity: trimesh.Trimesh,
    membrane: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    blade_thickness: float = 0.00001
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool]:
    """
    Split the shell into two manifold halves using the membrane as a cutting blade.
    
    This creates a very thin volume (blade) from the membrane and subtracts it
    from the shell, leaving a tiny gap that separates the shell into two 
    disconnected manifold components.
    
    Algorithm:
    1. Create thin "blade" by extruding membrane ±thickness/2 in pouring direction
    2. Subtract blade from shell: cut_shell = shell - blade
    3. Split result into connected components
    4. Return the two largest components as half_1 (upper) and half_2 (lower)
    
    Args:
        shell_with_cavity: The shell mesh (prism - hull)
        membrane: The cutting membrane (outer collar extended parting surface)
        pouring_direction: Unit vector defining the "upper" direction
        blade_thickness: Thickness of the cutting blade (default: 0.00001mm = 0.01 micron)
                        The blade is centered on the membrane (±thickness/2 each side)
        
    Returns:
        Tuple of (shell_half_1, shell_half_2, computation_time_ms, success)
        shell_half_1 is the half in the positive pouring direction (upper)
        shell_half_2 is the half in the negative pouring direction (lower)
    """
    start_time = time.time()
    
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available - cannot perform CSG operations")
        return None, None, 0.0, False
    
    try:
        logger.info(f"Splitting shell using thin blade subtraction (thickness={blade_thickness:.6f}mm)...")
        
        # Normalize direction
        direction = np.array(pouring_direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Step 1: Create thin cutting blade from membrane
        logger.info("Creating cutting blade from membrane...")
        blade = _create_cutting_blade_from_membrane(membrane, direction, blade_thickness)
        logger.info(f"Blade: {len(blade.vertices)} verts, {len(blade.faces)} faces")
        
        # Step 2: Convert to manifold3d
        logger.info("Converting meshes to manifold...")
        shell_manifold = _trimesh_to_manifold(shell_with_cavity)
        blade_manifold = _trimesh_to_manifold(blade)
        
        # Step 3: CSG subtraction - cut the shell with the blade
        logger.info("Performing CSG subtraction: shell - blade...")
        cut_shell_manifold = shell_manifold - blade_manifold
        
        # Step 4: Convert back to trimesh
        cut_shell = _manifold_to_trimesh(cut_shell_manifold)
        logger.info(f"Cut shell: {len(cut_shell.vertices)} verts, {len(cut_shell.faces)} faces")
        
        # Step 5: Split into connected components
        logger.info("Splitting into connected components...")
        components = cut_shell.split(only_watertight=False)
        logger.info(f"Found {len(components)} connected components")
        
        if len(components) < 2:
            logger.warning("Failed to split shell into two components - blade may not have cut through")
            elapsed_ms = (time.time() - start_time) * 1000
            if len(components) == 1:
                return components[0], None, elapsed_ms, False
            return None, None, elapsed_ms, False
        
        # Step 6: Sort by size and take the two largest
        components = sorted(components, key=lambda m: len(m.faces), reverse=True)
        comp1, comp2 = components[0], components[1]
        
        # Step 7: Classify which is upper vs lower based on centroid position
        centroid1 = comp1.centroid
        centroid2 = comp2.centroid
        
        # Project centroids onto pouring direction
        proj1 = np.dot(centroid1, direction)
        proj2 = np.dot(centroid2, direction)
        
        # Upper half has higher projection value
        if proj1 > proj2:
            shell_half_1, shell_half_2 = comp1, comp2
        else:
            shell_half_1, shell_half_2 = comp2, comp1
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell split complete in {elapsed_ms:.1f}ms:")
        logger.info(f"  Half 1 (upper): {len(shell_half_1.vertices)} verts, {len(shell_half_1.faces)} faces")
        logger.info(f"  Half 2 (lower): {len(shell_half_2.vertices)} verts, {len(shell_half_2.faces)} faces")
        
        if len(components) > 2:
            logger.warning(f"  Note: {len(components) - 2} additional small fragments were discarded")
        
        return shell_half_1, shell_half_2, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to split shell with membrane: {e}")
        import traceback
        traceback.print_exc()
        return None, None, elapsed_ms, False


def split_shell_along_parting_surface(
    shell_with_cavity: trimesh.Trimesh,
    parting_surface: trimesh.Trimesh,
    pouring_direction: np.ndarray
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool]:
    """
    Split the shell into two halves along the parting surface.
    
    Uses the parting surface (with extended collar) as a cutting plane.
    
    Args:
        shell_with_cavity: The shell mesh with cavity already subtracted
        parting_surface: The parting surface mesh (with extended collar)
        pouring_direction: The pouring direction (used to determine which half is which)
        
    Returns:
        Tuple of (shell_half_1, shell_half_2, computation_time_ms, success)
        shell_half_1 is the positive pouring direction half
        shell_half_2 is the negative pouring direction half
    """
    start_time = time.time()
    
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available - cannot perform CSG operations")
        return None, None, 0.0, False
    
    try:
        logger.info("Splitting shell along parting surface...")
        
        # Create a thick volume from the parting surface for CSG
        # The volume extends far in both directions to ensure complete intersection
        cutting_volume = _create_cutting_volume_from_surface(
            parting_surface,
            pouring_direction,
            thickness=1000.0  # Large enough to extend beyond the shell
        )
        
        # Convert to manifold
        shell_manifold = _trimesh_to_manifold(shell_with_cavity)
        cutting_manifold = _trimesh_to_manifold(cutting_volume)
        
        # Split using intersection and subtraction:
        # half_1 = shell - cutting_volume (part above the parting surface)
        # half_2 = shell ∩ cutting_volume... no wait, that's not right
        
        # Actually, we need to split using the cutting volume as a divider
        # The cutting volume represents the parting surface "thickened"
        # We want: the shell on the positive side of parting surface, 
        # and the shell on the negative side
        
        # Better approach: 
        # 1. Create two half-spaces by offsetting the cutting slab
        # 2. Intersect shell with each half-space
        
        # Simplest approach: 
        # Create a large box on each side of the parting surface
        # and intersect with shell
        
        # Get shell bounds to create large enough cutting boxes
        shell_bounds = shell_with_cavity.bounds
        shell_center = (shell_bounds[0] + shell_bounds[1]) / 2.0
        shell_size = np.linalg.norm(shell_bounds[1] - shell_bounds[0])
        box_size = shell_size * 3  # Make box much larger than shell
        
        direction = np.array(pouring_direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Create half-space boxes using the cutting volume
        # Half 1: above the parting surface (positive direction)
        # Half 2: below the parting surface (negative direction)
        
        # Use the cutting volume to split:
        # half_1 = shell - (shell ∩ cutting_volume)  -- remove the "inside" part
        # This gives us: shell parts NOT inside the cutting volume
        
        # Better: subtract cutting_volume from shell, then we have two disconnected pieces
        # But that doesn't always work if cutting volume is thin
        
        # Most reliable: use intersection with half-space volumes
        
        # Create a large box for each half
        # Box center offset by box_size/2 in the direction
        half_box_size = box_size
        
        # Create half-space box for positive direction (half 1)
        # Center is offset by box_size/2 + small epsilon in positive direction from shell center
        # But we need to use the parting surface centroid, not shell center
        parting_center = parting_surface.centroid
        
        box_center_1 = parting_center + direction * (half_box_size / 2.0 + 0.1)
        box_1 = trimesh.creation.box(
            extents=[box_size, box_size, box_size],
            transform=trimesh.transformations.translation_matrix(box_center_1)
        )
        
        box_center_2 = parting_center - direction * (half_box_size / 2.0 + 0.1)
        box_2 = trimesh.creation.box(
            extents=[box_size, box_size, box_size],
            transform=trimesh.transformations.translation_matrix(box_center_2)
        )
        
        # Convert boxes to manifold
        box_manifold_1 = _trimesh_to_manifold(box_1)
        box_manifold_2 = _trimesh_to_manifold(box_2)
        
        # Intersect shell with each half-space box
        half_1_manifold = shell_manifold ^ box_manifold_1  # Intersection
        half_2_manifold = shell_manifold ^ box_manifold_2  # Intersection
        
        # Convert back to trimesh
        shell_half_1 = _manifold_to_trimesh(half_1_manifold)
        shell_half_2 = _manifold_to_trimesh(half_2_manifold)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell split complete in {elapsed_ms:.1f}ms:")
        logger.info(f"  Half 1: {len(shell_half_1.vertices)} verts, {len(shell_half_1.faces)} faces")
        logger.info(f"  Half 2: {len(shell_half_2.vertices)} verts, {len(shell_half_2.faces)} faces")
        
        return shell_half_1, shell_half_2, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to split shell: {e}")
        import traceback
        traceback.print_exc()
        return None, None, elapsed_ms, False


def create_split_hard_shell(
    prism_result: HardShellPrismResult,
    hull_mesh: trimesh.Trimesh,
    parting_surface_with_collar: trimesh.Trimesh,
    pouring_direction: np.ndarray
) -> HardShellSplitResult:
    """
    Complete CSG pipeline: create shell with cavity and split into two halves.
    
    Pipeline:
    1. Subtract hull from prism to create cavity
    2. Split along parting surface into two halves
    
    Args:
        prism_result: The hard shell prism result
        hull_mesh: The inflated hull mesh (defines mold cavity)
        parting_surface_with_collar: The parting surface with extended collar
        pouring_direction: The resin pouring direction
        
    Returns:
        HardShellSplitResult with both shell halves
    """
    result = HardShellSplitResult(
        pouring_direction=np.array(pouring_direction)
    )
    
    total_start = time.time()
    
    # Step 1: Create shell with cavity
    shell_with_cavity, cavity_time, cavity_success = create_shell_with_cavity(
        prism_result.prism_mesh,
        hull_mesh
    )
    
    result.shell_with_cavity = shell_with_cavity
    result.cavity_time_ms = cavity_time
    result.cavity_success = cavity_success
    
    if not cavity_success or shell_with_cavity is None:
        result.total_time_ms = (time.time() - total_start) * 1000
        return result
    
    # Step 2: Split along parting surface
    shell_half_1, shell_half_2, split_time, split_success = split_shell_along_parting_surface(
        shell_with_cavity,
        parting_surface_with_collar,
        pouring_direction
    )
    
    result.shell_half_1 = shell_half_1
    result.shell_half_2 = shell_half_2
    result.split_time_ms = split_time
    result.split_success = split_success
    
    if shell_half_1 is not None:
        result.half_1_vertex_count = len(shell_half_1.vertices)
        result.half_1_face_count = len(shell_half_1.faces)
    
    if shell_half_2 is not None:
        result.half_2_vertex_count = len(shell_half_2.vertices)
        result.half_2_face_count = len(shell_half_2.faces)
    
    result.total_time_ms = (time.time() - total_start) * 1000
    
    return result


def _offset_polygon_2d(polygon: np.ndarray, offset: float) -> np.ndarray:
    """
    Offset a 2D convex polygon outward by a given distance.
    
    Uses the following approach for convex polygons:
    - Compute the outward normal for each edge
    - Move each vertex along the bisector of adjacent edge normals
    - Scale by offset / cos(half_angle) to maintain proper offset distance
    
    Args:
        polygon: (N, 2) array of 2D vertices in CCW order
        offset: Distance to offset outward
        
    Returns:
        (N, 2) array of offset polygon vertices
    """
    n = len(polygon)
    if n < 3:
        return polygon.copy()
    
    # Ensure CCW ordering
    # Compute signed area
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += polygon[i, 0] * polygon[j, 1]
        signed_area -= polygon[j, 0] * polygon[i, 1]
    
    if signed_area < 0:
        # CW ordering, reverse
        polygon = polygon[::-1].copy()
    
    # Compute edge normals (outward for CCW polygon)
    edge_normals = []
    for i in range(n):
        j = (i + 1) % n
        edge = polygon[j] - polygon[i]
        # Normal: perpendicular to edge, pointing outward (right side for CCW)
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        edge_normals.append(normal)
    
    # Offset each vertex
    offset_polygon = []
    for i in range(n):
        prev_i = (i - 1) % n
        
        # Bisector of prev and current edge normals
        n_prev = edge_normals[prev_i]
        n_curr = edge_normals[i]
        
        bisector = n_prev + n_curr
        bisector_len = np.linalg.norm(bisector)
        
        if bisector_len > 1e-10:
            bisector = bisector / bisector_len
            
            # cos(half_angle) = dot(n_prev, bisector) = dot(n_curr, bisector)
            cos_half = np.dot(n_prev, bisector)
            
            # Offset distance along bisector
            if cos_half > 0.1:  # Avoid very sharp corners
                offset_dist = offset / cos_half
            else:
                offset_dist = offset * 5  # Cap at 5x for sharp corners
            
            new_vertex = polygon[i] + bisector * offset_dist
        else:
            # Normals are opposite (happens at very sharp corners)
            new_vertex = polygon[i] + n_prev * offset
        
        offset_polygon.append(new_vertex)
    
    return np.array(offset_polygon, dtype=np.float64)


def _extrude_polygon_to_prism(
    polygon_2d: np.ndarray,
    world_to_2d: np.ndarray,
    min_height: float,
    max_height: float
) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """
    Extrude a 2D polygon along the normal direction to create a prism.
    
    Args:
        polygon_2d: (N, 2) array of 2D polygon vertices
        world_to_2d: (3, 3) rotation matrix from world to 2D plane
        min_height: Bottom extrusion distance (along the third axis after rotation)
        max_height: Top extrusion distance
        
    Returns:
        Tuple of:
        - trimesh.Trimesh: The prism mesh
        - np.ndarray: (N, 3) bottom polygon vertices in 3D
        - np.ndarray: (N, 3) top polygon vertices in 3D
    """
    n = len(polygon_2d)
    
    # Transform back to 3D world coordinates
    # world_to_2d.T is the inverse (since it's orthonormal)
    two_d_to_world = world_to_2d.T
    
    # Create bottom and top polygons in 3D
    bottom_vertices = []
    top_vertices = []
    
    for i in range(n):
        # 2D to 3D: [u, v, height] -> world
        pt_bottom = np.array([polygon_2d[i, 0], polygon_2d[i, 1], min_height])
        pt_top = np.array([polygon_2d[i, 0], polygon_2d[i, 1], max_height])
        
        bottom_3d = pt_bottom @ two_d_to_world.T
        top_3d = pt_top @ two_d_to_world.T
        
        bottom_vertices.append(bottom_3d)
        top_vertices.append(top_3d)
    
    bottom_vertices = np.array(bottom_vertices, dtype=np.float64)
    top_vertices = np.array(top_vertices, dtype=np.float64)
    
    # Build mesh vertices: bottom ring + top ring
    # Indices: 0..n-1 = bottom, n..2n-1 = top
    all_vertices = np.vstack([bottom_vertices, top_vertices])
    
    faces = []
    
    # Side faces: quads as 2 triangles each
    for i in range(n):
        j = (i + 1) % n
        
        b0 = i          # bottom vertex i
        b1 = j          # bottom vertex i+1
        t0 = i + n      # top vertex i
        t1 = j + n      # top vertex i+1
        
        # Quad: (b0, b1, t1, t0) -> triangles with outward normals
        # CCW when viewed from outside
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    
    # Bottom cap: fan triangulation from center
    # Add center vertex
    bottom_center = np.mean(bottom_vertices, axis=0)
    bottom_center_idx = len(all_vertices)
    all_vertices = np.vstack([all_vertices, bottom_center.reshape(1, 3)])
    
    for i in range(n):
        j = (i + 1) % n
        # Bottom face: normal points down (reversed winding)
        faces.append([bottom_center_idx, j, i])
    
    # Top cap: fan triangulation from center
    top_center = np.mean(top_vertices, axis=0)
    top_center_idx = len(all_vertices)
    all_vertices = np.vstack([all_vertices, top_center.reshape(1, 3)])
    
    for i in range(n):
        j = (i + 1) % n
        # Top face: normal points up
        faces.append([top_center_idx, i + n, j + n])
    
    faces = np.array(faces, dtype=np.int64)
    
    prism_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=faces,
        process=True
    )
    prism_mesh.fix_normals()
    
    return prism_mesh, bottom_vertices, top_vertices


# ============================================================================
# OUTER SHELL HULL CREATION
# ============================================================================

def create_outer_shell_hull(
    inner_hull: trimesh.Trimesh,
    wall_thickness: float = DEFAULT_WALL_THICKNESS_MM
) -> OuterShellHullResult:
    """
    Create the outer shell hull by offsetting the inner hull outward.
    
    The outer shell hull defines the exterior boundary of the hard shell.
    The space between inner hull and outer shell hull is where the hard
    plastic shell wall will be.
    
    Args:
        inner_hull: The current inflated hull mesh (∂H)
        wall_thickness: How thick the hard shell wall should be (mm)
        
    Returns:
        OuterShellHullResult with the offset hull
    """
    start_time = time.time()
    
    logger.info(f"Creating outer shell hull with wall_thickness={wall_thickness}mm")
    
    # Get vertices and compute smooth normals
    vertices = np.array(inner_hull.vertices, dtype=np.float64)
    faces = np.array(inner_hull.faces, dtype=np.int32)
    
    # Compute area-weighted smooth vertex normals
    smooth_normals = _compute_smooth_vertex_normals(vertices, faces)
    
    # Offset vertices outward along normals
    outer_vertices = vertices + smooth_normals * wall_thickness
    
    # Create mesh from offset vertices
    outer_hull_mesh = trimesh.Trimesh(
        vertices=outer_vertices,
        faces=faces.copy(),
        process=True
    )
    
    # Fix normals to point outward
    outer_hull_mesh.fix_normals()
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Outer shell hull created: {len(outer_hull_mesh.vertices)} vertices, "
                f"{len(outer_hull_mesh.faces)} faces in {elapsed_ms:.1f}ms")
    
    return OuterShellHullResult(
        mesh=outer_hull_mesh,
        inner_hull=inner_hull,
        wall_thickness=wall_thickness,
        vertex_count=len(outer_hull_mesh.vertices),
        face_count=len(outer_hull_mesh.faces),
        computation_time_ms=elapsed_ms
    )


def _compute_smooth_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute area-weighted smooth vertex normals.
    
    Each vertex normal is the normalized sum of face normals weighted by face area.
    """
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute face normals via cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    
    # Face normal length = 2 * area
    face_normal_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    areas = face_normal_lengths / 2.0
    
    # Normalize face normals
    face_normals_normalized = face_normals / (face_normal_lengths + 1e-10)
    
    # Weight by area
    weighted_normals = face_normals_normalized * areas
    
    # Accumulate at each vertex
    normal_accum = np.zeros((num_vertices, 3), dtype=np.float64)
    np.add.at(normal_accum, faces[:, 0], weighted_normals)
    np.add.at(normal_accum, faces[:, 1], weighted_normals)
    np.add.at(normal_accum, faces[:, 2], weighted_normals)
    
    # Normalize
    norms = np.linalg.norm(normal_accum, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    smooth_normals = normal_accum / norms
    
    return smooth_normals


# ============================================================================
# OUTER COLLAR EXTENSION
# ============================================================================

def create_outer_collar_extension(
    parting_surface: trimesh.Trimesh,
    inner_hull: trimesh.Trimesh,
    vertex_boundary_type: np.ndarray,
    pouring_direction: np.ndarray,
    extension_distance: float = 10.0,
    collar_subdivisions: int = DEFAULT_COLLAR_SUBDIVISIONS
) -> OuterCollarResult:
    """
    Extend the parting surface outward by a fixed distance.
    
    This creates a "collar" around the outer boundary of the parting surface,
    extending it by extension_distance so it fully passes through the hard shell prism.
    
    The extension is performed perpendicular to the pouring direction to 
    maintain the prism-like structure of the hard shell.
    
    Algorithm:
    1. Find outer boundary edges of parting surface (edges on hull, type 1 or 2)
    2. For each outer boundary vertex:
       - Compute extension direction (perpendicular to pouring, outward from hull)
       - Extend by extension_distance
       - Create collar quad connecting parting surface to extended boundary
    3. Handle corners with fan triangles
    
    Args:
        parting_surface: The smoothed parting surface mesh
        inner_hull: The current hull mesh (∂H)
        vertex_boundary_type: Array with -1=part, 0=interior, 1/2=hull
        pouring_direction: Unit vector for resin pouring direction
        extension_distance: Distance to extend outward (typically 2x wall_thickness)
        collar_subdivisions: Number of fan subdivisions at corners
        
    Returns:
        OuterCollarResult with the extended parting surface
    """
    start_time = time.time()
    
    logger.info(f"Creating outer collar extension for parting surface (extension={extension_distance}mm)")
    
    result = OuterCollarResult(
        mesh=parting_surface,
        vertices=np.array(parting_surface.vertices),
        faces=np.array(parting_surface.faces),
        original_vertex_count=len(parting_surface.vertices),
        original_face_count=len(parting_surface.faces),
        extension_distance=extension_distance
    )
    
    if parting_surface is None:
        logger.warning("No parting surface provided")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    # Convert to working arrays
    vertices = list(parting_surface.vertices)
    faces = list(parting_surface.faces)
    n_orig_verts = len(vertices)
    n_orig_faces = len(faces)
    
    vertices_arr = np.array(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int64)
    
    # Normalize pouring direction
    pouring_dir = np.array(pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    # =========================================================================
    # STEP 1: Find all mesh boundary edges and classify as inner/outer
    # =========================================================================
    
    edge_to_face = {}  # edge_key -> (face_idx, third_vertex)
    edge_face_count = {}  # edge_key -> count
    
    for fi, face in enumerate(faces_arr):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            v2 = int(face[(i + 2) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_face_count[edge_key] = edge_face_count.get(edge_key, 0) + 1
            edge_to_face[edge_key] = (fi, v2)
    
    # Boundary edges = edges with only 1 adjacent face
    boundary_edges = [(v0, v1) for (v0, v1), count in edge_face_count.items() if count == 1]
    
    if len(boundary_edges) == 0:
        logger.info("No boundary edges found in parting surface")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    logger.info(f"Found {len(boundary_edges)} mesh boundary edges")
    
    # =========================================================================
    # STEP 2: Classify boundary edges as inner (part) or outer (hull)
    # =========================================================================
    
    has_boundary_type = vertex_boundary_type is not None and len(vertex_boundary_type) >= n_orig_verts
    
    outer_boundary_edges = []
    inner_boundary_edges = []
    
    for v0, v1 in boundary_edges:
        is_outer = False
        
        if has_boundary_type:
            bt0 = vertex_boundary_type[v0] if v0 < len(vertex_boundary_type) else 0
            bt1 = vertex_boundary_type[v1] if v1 < len(vertex_boundary_type) else 0
            
            # Outer edge: at least one vertex is on hull boundary (type 1 or 2)
            if bt0 in (1, 2) or bt1 in (1, 2):
                is_outer = True
            # Inner edge: at least one vertex is on part (type -1)
            elif bt0 == -1 or bt1 == -1:
                is_outer = False
            else:
                # Both interior - check distance to inner hull vs part
                # Default to outer for now
                is_outer = True
        else:
            # No boundary type - use proximity to hull
            boundary_vert_positions = vertices_arr[[v0, v1]]
            try:
                _, dists_to_hull, _ = trimesh.proximity.closest_point(inner_hull, boundary_vert_positions)
                # If close to hull, it's outer
                is_outer = np.min(dists_to_hull) < 1.0  # 1mm threshold
            except:
                is_outer = True
        
        if is_outer:
            outer_boundary_edges.append((v0, v1))
        else:
            inner_boundary_edges.append((v0, v1))
    
    logger.info(f"Outer boundary edges: {len(outer_boundary_edges)}, inner: {len(inner_boundary_edges)}")
    
    if len(outer_boundary_edges) == 0:
        logger.info("No outer boundary edges to extend")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    # =========================================================================
    # STEP 3: Build vertex adjacency for outer boundary
    # =========================================================================
    
    vertex_to_boundary_edges: Dict[int, List] = {}
    for v0, v1 in outer_boundary_edges:
        if v0 not in vertex_to_boundary_edges:
            vertex_to_boundary_edges[v0] = []
        if v1 not in vertex_to_boundary_edges:
            vertex_to_boundary_edges[v1] = []
        vertex_to_boundary_edges[v0].append(((v0, v1), v0, v1))  # (edge_tuple, this_vert, other_vert)
        vertex_to_boundary_edges[v1].append(((v0, v1), v1, v0))
    
    # Identify corners (2+ boundary edges) and endpoints
    corner_vertices = []
    endpoint_vertices = []
    
    for vi, edges in vertex_to_boundary_edges.items():
        if len(edges) == 1:
            endpoint_vertices.append(vi)
        elif len(edges) >= 2:
            corner_vertices.append(vi)
    
    logger.info(f"Corner vertices: {len(corner_vertices)}, endpoints: {len(endpoint_vertices)}")
    
    # =========================================================================
    # STEP 4: Create collar vertices for outer boundary
    # =========================================================================
    
    # For each outer boundary vertex, create a collar vertex on the outer shell hull
    # The collar direction is PERPENDICULAR to pouring direction
    
    membrane_face_normals = parting_surface.face_normals
    inner_hull_normals = inner_hull.vertex_normals if hasattr(inner_hull, 'vertex_normals') else None
    
    # Map: original vertex index -> collar vertex index
    vert_to_collar: Dict[int, int] = {}
    
    # For corner vertices with multiple edges, we may need multiple collar vertices
    # edge_key -> {vertex: collar_idx}
    edge_endpoint_collar: Dict[Tuple[int, int], Dict[int, int]] = {}
    
    collar_vertices_created = 0
    
    for v0, v1 in outer_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        if edge_key not in edge_endpoint_collar:
            edge_endpoint_collar[edge_key] = {}
        
        # Get face info for normal
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi, _ = face_info
        if fi < len(membrane_face_normals):
            face_normal = membrane_face_normals[fi]
        else:
            face_normal = np.array([0, 0, 1])
        
        for vi in [v0, v1]:
            if vi in edge_endpoint_collar[edge_key]:
                continue
            
            vi_pos = vertices_arr[vi]
            
            # ==================================================================
            # Compute extension direction: perpendicular to pouring, outward from hull
            # ==================================================================
            
            # Get the closest point on inner hull to find hull normal
            try:
                closest_pts, _, closest_faces = trimesh.proximity.closest_point(inner_hull, [vi_pos])
                closest_pt = closest_pts[0]
                closest_face = closest_faces[0]
                hull_normal = inner_hull.face_normals[closest_face]
            except:
                hull_normal = face_normal
            
            # The extension direction should be:
            # 1. In the plane perpendicular to pouring direction
            # 2. Pointing outward (along hull normal projection)
            
            # Project hull normal onto plane perpendicular to pouring direction
            # perp_component = hull_normal - (hull_normal · pouring_dir) * pouring_dir
            dot_product = np.dot(hull_normal, pouring_dir)
            perp_component = hull_normal - dot_product * pouring_dir
            perp_len = np.linalg.norm(perp_component)
            
            if perp_len > 1e-6:
                extension_dir = perp_component / perp_len
            else:
                # Hull normal is parallel to pouring direction
                # Use face normal instead
                dot_product = np.dot(face_normal, pouring_dir)
                perp_component = face_normal - dot_product * pouring_dir
                perp_len = np.linalg.norm(perp_component)
                if perp_len > 1e-6:
                    extension_dir = perp_component / perp_len
                else:
                    # Fallback: use any perpendicular direction
                    extension_dir = np.cross(pouring_dir, [1, 0, 0])
                    if np.linalg.norm(extension_dir) < 1e-6:
                        extension_dir = np.cross(pouring_dir, [0, 1, 0])
                    extension_dir = extension_dir / np.linalg.norm(extension_dir)
            
            # ==================================================================
            # Extend vertex by fixed extension_distance
            # ==================================================================
            
            # Simply offset the vertex by extension_distance along the extension direction
            collar_pt = vi_pos + extension_dir * extension_distance
            
            # Create collar vertex
            collar_idx = len(vertices)
            vertices.append(collar_pt.copy())
            edge_endpoint_collar[edge_key][vi] = collar_idx
            
            # Also store in simple map for non-corner vertices
            if vi not in vert_to_collar:
                vert_to_collar[vi] = collar_idx
            
            collar_vertices_created += 1
    
    logger.info(f"Created {collar_vertices_created} collar vertices")
    
    # =========================================================================
    # STEP 5: Create quad collars for each edge
    # =========================================================================
    
    collar_faces_created = 0
    
    for v0, v1 in outer_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        
        if edge_key not in edge_endpoint_collar:
            continue
        
        c0 = edge_endpoint_collar[edge_key].get(v0)
        c1 = edge_endpoint_collar[edge_key].get(v1)
        
        if c0 is None or c1 is None:
            continue
        
        # Get face to determine winding order
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi, third_v = face_info
        
        # Determine edge direction in the original face
        face = faces_arr[fi]
        edge_forward = False
        for i in range(3):
            if face[i] == v0 and face[(i + 1) % 3] == v1:
                edge_forward = True
                break
        
        # Create quad (2 triangles) connecting edge to collar
        # Winding: if edge is v0->v1 in face, collar should be c1->c0 for consistent normal
        if edge_forward:
            # Edge goes v0 -> v1 in face, so outward normal wants:
            # Triangle 1: v0, c0, v1
            # Triangle 2: v1, c0, c1
            faces.append([v0, c0, v1])
            faces.append([v1, c0, c1])
        else:
            # Edge goes v1 -> v0 in face
            # Triangle 1: v0, v1, c0
            # Triangle 2: v1, c1, c0
            faces.append([v0, v1, c0])
            faces.append([v1, c1, c0])
        
        collar_faces_created += 2
    
    logger.info(f"Created {collar_faces_created} collar faces")
    
    # =========================================================================
    # STEP 6: Handle corners with fan triangles
    # =========================================================================
    
    corner_fans_created = 0
    
    for vi in corner_vertices:
        edges = vertex_to_boundary_edges.get(vi, [])
        if len(edges) < 2:
            continue
        
        # Get collar vertices for each edge at this corner
        collar_verts = []
        for edge_tuple, this_v, other_v in edges:
            edge_key = (min(edge_tuple[0], edge_tuple[1]), max(edge_tuple[0], edge_tuple[1]))
            collar_map = edge_endpoint_collar.get(edge_key, {})
            collar_idx = collar_map.get(vi)
            if collar_idx is not None:
                collar_verts.append((collar_idx, other_v))
        
        if len(collar_verts) < 2:
            continue
        
        # Order collar vertices angularly around the corner
        vi_pos = vertices_arr[vi]
        
        # Compute angles for each collar vertex
        angles = []
        for collar_idx, other_v in collar_verts:
            collar_pos = np.array(vertices[collar_idx])
            vec = collar_pos - vi_pos
            # Project onto plane perpendicular to pouring direction
            vec_perp = vec - np.dot(vec, pouring_dir) * pouring_dir
            angle = np.arctan2(
                np.dot(vec_perp, np.cross(pouring_dir, [1, 0, 0])),
                np.dot(vec_perp, [1, 0, 0])
            )
            angles.append((angle, collar_idx))
        
        # Sort by angle
        angles.sort(key=lambda x: x[0])
        sorted_collars = [c for _, c in angles]
        
        # Create fan triangles connecting consecutive collar vertices
        for i in range(len(sorted_collars)):
            c_curr = sorted_collars[i]
            c_next = sorted_collars[(i + 1) % len(sorted_collars)]
            
            if c_curr != c_next:
                # Fan triangle: corner vertex to two consecutive collar vertices
                faces.append([vi, c_curr, c_next])
                corner_fans_created += 1
    
    logger.info(f"Created {corner_fans_created} corner fan triangles")
    
    # =========================================================================
    # STEP 7: Build result mesh
    # =========================================================================
    
    vertices_arr_final = np.array(vertices, dtype=np.float64)
    faces_arr_final = np.array(faces, dtype=np.int64)
    
    extended_mesh = trimesh.Trimesh(
        vertices=vertices_arr_final,
        faces=faces_arr_final,
        process=False  # Don't merge vertices
    )
    
    # Identify outer boundary vertices (the collar vertices)
    outer_boundary_verts = np.array(list(set(vert_to_collar.values())), dtype=np.int64)
    
    # Try to order the outer boundary as a loop
    outer_boundary_loop = _extract_boundary_loop(faces_arr_final, outer_boundary_verts, vertices_arr_final)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    result = OuterCollarResult(
        mesh=extended_mesh,
        vertices=vertices_arr_final,
        faces=faces_arr_final,
        original_vertex_count=n_orig_verts,
        original_face_count=n_orig_faces,
        collar_vertex_count=collar_vertices_created,
        collar_face_count=collar_faces_created + corner_fans_created,
        extension_distance=extension_distance,
        outer_boundary_vertices=outer_boundary_verts,
        outer_boundary_loop=outer_boundary_loop,
        n_outer_boundary_edges_processed=len(outer_boundary_edges),
        n_corner_fans_created=corner_fans_created,
        computation_time_ms=elapsed_ms
    )
    
    logger.info(f"Outer collar extension complete: {len(vertices)} vertices, {len(faces)} faces "
                f"(+{collar_vertices_created} verts, +{collar_faces_created + corner_fans_created} faces, "
                f"extended {extension_distance}mm) in {elapsed_ms:.1f}ms")
    
    return result


def _extract_boundary_loop(
    faces: np.ndarray,
    boundary_verts: np.ndarray,
    vertices: np.ndarray
) -> List[int]:
    """
    Extract the outer boundary as an ordered loop of vertex indices.
    
    This attempts to order the boundary vertices into a continuous loop
    by following boundary edges.
    """
    if len(boundary_verts) == 0:
        return []
    
    boundary_set = set(boundary_verts)
    
    # Build edge connectivity for boundary vertices only
    # Find edges where both vertices are in boundary_set
    edge_count = {}
    for face in faces:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            if v0 in boundary_set and v1 in boundary_set:
                edge_key = (min(v0, v1), max(v0, v1))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
    
    # True boundary edges are used once
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    
    if len(boundary_edges) == 0:
        # No true boundary - return unordered
        return list(boundary_verts)
    
    # Build adjacency
    adjacency: Dict[int, List[int]] = {}
    for v0, v1 in boundary_edges:
        if v0 not in adjacency:
            adjacency[v0] = []
        if v1 not in adjacency:
            adjacency[v1] = []
        adjacency[v0].append(v1)
        adjacency[v1].append(v0)
    
    # Walk the loop
    loop = []
    visited = set()
    
    # Start from first boundary vertex
    start = boundary_edges[0][0]
    current = start
    
    while current not in visited:
        visited.add(current)
        loop.append(current)
        
        # Find next unvisited neighbor
        neighbors = adjacency.get(current, [])
        next_v = None
        for n in neighbors:
            if n not in visited:
                next_v = n
                break
        
        if next_v is None:
            break
        current = next_v
    
    return loop


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_wall_thickness_from_bbox(mesh: trimesh.Trimesh, fraction: float = 0.02) -> float:
    """
    Compute a reasonable wall thickness based on mesh size.
    
    Args:
        mesh: Input mesh
        fraction: Fraction of bounding box diagonal (default 2%)
        
    Returns:
        Wall thickness in mesh units
    """
    bounds = mesh.bounds
    diagonal = np.linalg.norm(bounds[1] - bounds[0])
    return diagonal * fraction
