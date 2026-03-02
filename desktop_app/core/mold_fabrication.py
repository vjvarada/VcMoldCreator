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
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

import trimesh
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

# CSG operations via manifold3d
try:
    import manifold3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False
    logger.warning("manifold3d not available - CSG operations will be disabled")


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


@dataclass
class MetamoldPrismResult:
    """Result of creating the metamold prism.
    
    The metamold is a mold used to cast the silicone soft mold itself.
    It uses the same silhouette/direction as the hard shell (hull + resin direction)
    but the height is based on the PART MESH extent, not the parting surface.
    
    The prism is aligned with the resin pouring direction (same as hard shell), with:
    - Silhouette derived from the HULL projection (same as hard shell)
    - Height based on PART MESH extent + configurable offsets above/below
    """
    
    # The prism mesh (before subtracting any cavities)
    prism_mesh: trimesh.Trimesh
    
    # The 2D silhouette polygon (convex hull of parting surface projection)
    silhouette_2d: np.ndarray  # (N, 2) ordered vertices of 2D convex hull
    
    # The 3D silhouette polygon (on the base plane)
    silhouette_3d: np.ndarray  # (N, 3) 3D positions on base plane
    
    # The base plane center and normal
    base_plane_center: np.ndarray  # (3,)
    base_plane_normal: np.ndarray  # (3,) = silicone pouring direction
    
    # Prism dimensions
    prism_height: float  # Total extrusion distance
    height_above_surface: float  # Distance from parting surface max to top
    height_below_surface: float  # Distance from parting surface min to bottom
    
    # The transformation matrix from world to 2D projection plane
    world_to_2d: np.ndarray  # (3, 3) rotation matrix
    
    # The silicone pouring direction used
    pouring_direction: np.ndarray  # (3,)
    
    # Wall thickness (horizontal offset from silhouette)
    wall_thickness: float
    
    # Reference heights from part mesh
    part_mesh_min_height: float  # Min projection along pouring direction
    part_mesh_max_height: float  # Max projection along pouring direction
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


# ============================================================================
# METAMOLD PRISM CREATION
# ============================================================================

def create_metamold_prism(
    hull_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    resin_pouring_direction: np.ndarray,
    wall_thickness: float = 5.0,
    margin: float = 0.0,
    height_above: float = 2.0,
    height_below: float = 2.0,
    parting_surface: trimesh.Trimesh = None
) -> MetamoldPrismResult:
    """
    Create a metamold prism aligned with the resin pouring direction.
    
    The metamold is used to cast the silicone soft mold. It uses the EXACT SAME
    silhouette and extrusion direction as the hard shell prism:
    
    - Silhouette: 2D convex hull of HULL vertices projected onto a plane
                  perpendicular to the RESIN pouring direction (same as hard shell)
    - Extrusion: Along RESIN pouring direction (same as hard shell)
    - Wall offset: wall_thickness + margin (same as hard shell)
    - Height: From (hull_min - total_offset) to (hull_max + total_offset)
              where min/max are the HULL MESH extent along pouring direction,
              matching the hard shell prism exactly.  Also extends to include
              part mesh margins and parting_surface if they exceed the hull
              bounds (safety check).
    
    This ensures the metamold has the EXACT same footprint as the hard shell and is
    tall enough to enclose the entire part with 2mm margin on each side.
    
    Args:
        hull_mesh: The inflated hull mesh (used for silhouette, same as hard shell)
        part_mesh: The original part mesh (used for height reference)
        resin_pouring_direction: Unit vector for RESIN pouring direction (same as hard shell)
        wall_thickness: How thick the hard shell wall should be (mm)
        margin: Additional margin beyond the hull bounds (mm) - same as hard shell
        height_above: Distance above part mesh max to top of prism (mm)
        height_below: Distance below part mesh min to bottom of prism (mm)
        parting_surface: Optional parting surface mesh - if provided, prism height
                        will be extended to include it (ensuring blade cuts through)
        
    Returns:
        MetamoldPrismResult with the prism mesh and metadata
    """
    start_time = time.time()
    
    logger.info(f"Creating metamold prism aligned with resin direction: {resin_pouring_direction}")
    logger.info(f"  Wall thickness: {wall_thickness}mm, margin: {margin}mm, height_above: {height_above}mm, height_below: {height_below}mm")
    
    # Normalize pouring direction (same as hard shell)
    pouring_dir = np.array(resin_pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    # Use hull vertices for silhouette (same as hard shell)
    hull_vertices = np.array(hull_mesh.vertices, dtype=np.float64)
    # Use part mesh vertices for height extent
    part_vertices = np.array(part_mesh.vertices, dtype=np.float64)
    
    # =========================================================================
    # Step 1: Build orthonormal basis for the projection plane (same as hard shell)
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
    # Step 2: Project part mesh vertices to get height extent
    # =========================================================================
    
    part_transformed = part_vertices @ world_to_2d.T
    part_heights = part_transformed[:, 2]  # projection along pouring direction
    
    # Get height extent of the part mesh
    part_min_height = np.min(part_heights)
    part_max_height = np.max(part_heights)
    
    logger.info(f"Part mesh height extent: {part_min_height:.2f} to {part_max_height:.2f}")
    
    # =========================================================================
    # Step 3: Project HULL vertices to get 2D silhouette (SAME as hard shell)
    # =========================================================================
    
    hull_transformed = hull_vertices @ world_to_2d.T
    projected_2d = hull_transformed[:, :2]  # (N, 2)
    
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
    
    logger.info(f"2D silhouette from hull has {len(silhouette_2d)} vertices")
    
    # =========================================================================
    # Step 4: Offset the silhouette outward by wall_thickness + margin (SAME as hard shell)
    # =========================================================================
    
    total_offset = wall_thickness + margin
    silhouette_2d_offset = _offset_polygon_2d(silhouette_2d, total_offset)
    
    logger.info(f"Offset silhouette has {len(silhouette_2d_offset)} vertices (offset={total_offset}mm)")
    
    # =========================================================================
    # Step 5: Determine prism height — MATCH HARD SHELL EXACTLY
    #
    # Use HULL heights + total_offset (same as create_hard_shell_prism) so the
    # metamold prism has the identical footprint AND height as the hard shell
    # prism.  This guarantees the cast hard-shell halves fit inside the
    # metamold.  The part-based height_above/height_below and the parting
    # surface are kept as secondary safety checks.
    # =========================================================================
    
    hull_heights = hull_transformed[:, 2]
    hull_min_height = float(np.min(hull_heights))
    hull_max_height = float(np.max(hull_heights))
    
    # Primary height: hull extents ± total_offset (matches hard shell prism)
    min_height = hull_min_height - total_offset
    max_height = hull_max_height + total_offset
    
    logger.info(f"Hull height extent: {hull_min_height:.2f} to {hull_max_height:.2f}")
    logger.info(f"Primary height (hull ± {total_offset}mm): {min_height:.2f} to {max_height:.2f}")
    
    # Safety: also cover part extents ± height_above/below
    part_min_with_margin = part_min_height - height_below
    part_max_with_margin = part_max_height + height_above
    if part_min_with_margin < min_height:
        min_height = part_min_with_margin
    if part_max_with_margin > max_height:
        max_height = part_max_with_margin
    
    # Safety: also cover parting surface with margin
    if parting_surface is not None:
        ps_vertices = np.array(parting_surface.vertices, dtype=np.float64)
        ps_transformed = ps_vertices @ world_to_2d.T
        ps_heights = ps_transformed[:, 2]  # projection along pouring direction
        ps_min_height = np.min(ps_heights)
        ps_max_height = np.max(ps_heights)
        
        logger.info(f"Parting surface height extent: {ps_min_height:.2f} to {ps_max_height:.2f}")
        
        parting_margin = 2.0  # 2mm margin around parting surface
        if ps_min_height - parting_margin < min_height:
            old_min = min_height
            min_height = ps_min_height - parting_margin
            logger.info(f"Extended prism bottom from {old_min:.2f} to {min_height:.2f} to include parting surface")
        
        if ps_max_height + parting_margin > max_height:
            old_max = max_height
            max_height = ps_max_height + parting_margin
            logger.info(f"Extended prism top from {old_max:.2f} to {max_height:.2f} to include parting surface")
    
    prism_height = max_height - min_height
    
    logger.info(f"Prism height: {prism_height:.2f} (from {min_height:.2f} to {max_height:.2f})")
    
    # =========================================================================
    # Step 6: Create prism mesh by extruding the silhouette
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
    
    result = MetamoldPrismResult(
        prism_mesh=prism_mesh,
        silhouette_2d=silhouette_2d_offset,
        silhouette_3d=silhouette_3d_bottom,
        base_plane_center=base_plane_center,
        base_plane_normal=pouring_dir,
        prism_height=prism_height,
        height_above_surface=height_above,
        height_below_surface=height_below,
        world_to_2d=world_to_2d,
        pouring_direction=pouring_dir,
        wall_thickness=wall_thickness,
        part_mesh_min_height=part_min_height,
        part_mesh_max_height=part_max_height,
        vertex_count=len(prism_mesh.vertices),
        face_count=len(prism_mesh.faces),
        computation_time_ms=elapsed_ms
    )
    
    logger.info(f"Metamold prism created: {result.vertex_count} vertices, "
                f"{result.face_count} faces in {elapsed_ms:.1f}ms")
    
    return result


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

# Directory where problematic input/output meshes are saved for visual debugging.
# Auto-initialised to <workspace_root>/.tmp/csg_debug/ so debug STLs are saved
# on every run without any extra wiring from the UI layer.
# Override with set_csg_debug_dir() to use a per-session path.
_CSG_DEBUG_DIR: Optional[Path] = None
try:
    _CSG_DEBUG_DIR = (
        Path(__file__).parent   # desktop_app/core/
        .parent                 # desktop_app/
        .parent                 # VcMoldCreator/ (workspace root)
        / ".tmp"
        / "csg_debug"
    )
    _CSG_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    _CSG_DEBUG_DIR = None  # Disable silently on any filesystem error


def set_csg_debug_dir(path: Optional[Path]) -> None:
    """Set the directory where bad CSG meshes are saved as STL for inspection.

    Called from the main window once a session directory is known.  If *None*,
    debug saving is disabled.
    """
    global _CSG_DEBUG_DIR
    _CSG_DEBUG_DIR = path
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        logger.info("CSG debug meshes will be saved to: %s", path)


def save_debug_mesh(mesh: trimesh.Trimesh, label: str) -> None:
    """Save *mesh* to the CSG debug directory as ``<label>.stl``.

    Silently does nothing if the debug directory has not been set or if the
    mesh has no vertices.
    """
    if _CSG_DEBUG_DIR is None or mesh is None or len(mesh.vertices) == 0:
        return
    _CSG_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitise label for use as a filename
    safe = label.replace('/', '_').replace('\\', '_').replace(' ', '_')
    out = _CSG_DEBUG_DIR / f"{safe}.stl"
    try:
        mesh.export(str(out))
        # Compute component count and open-edge count for instant diagnosis
        try:
            components = mesh.split(only_watertight=False)
            n_components = len(components)
        except Exception:
            n_components = -1
        try:
            ec: dict = {}
            for face in mesh.faces:
                for i in range(3):
                    v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                    ek = (min(v0, v1), max(v0, v1))
                    ec[ek] = ec.get(ek, 0) + 1
            open_edges = sum(1 for c in ec.values() if c == 1)
        except Exception:
            open_edges = -1
        logger.info(
            "  [CSG DEBUG] '%s' → %s  |  %dv %df  watertight=%s  components=%s  open_edges=%s",
            label, out, len(mesh.vertices), len(mesh.faces),
            mesh.is_watertight, n_components, open_edges,
        )
    except Exception as exc:
        logger.warning("  [CSG DEBUG] Could not save '%s': %s", label, exc)


def _repair_mesh_for_csg(mesh: trimesh.Trimesh, label: str = 'mesh') -> trimesh.Trimesh:
    """Repair a trimesh to make it as watertight as possible before CSG with manifold3d.

    Uses meshlib as the primary repair engine (handles non-manifold edges/vertices,
    complex holes, and self-intersections robustly), with trimesh as a fallback.

    Repair sequence:
    1. meshlib: fixMeshDegeneracies  – removes degenerate/non-manifold faces & edges
    2. meshlib: fillHoleNicely       – fills each open boundary loop with quality triangles
    3. trimesh  (fallback)           – process=True + fix_normals + fill_holes

    Args:
        mesh: Input trimesh (may be non-manifold / non-watertight).
        label: Human-readable label for log messages.

    Returns:
        Repaired trimesh.  The input is not modified.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh

    # Fast path: already perfectly watertight – return immediately without any
    # vertex merging or repair.  This is critical for meshes produced by
    # manifold3d (via _manifold_to_trimesh with process=False) which have
    # intentionally split vertices at seam edges.  Merging those vertices
    # creates T-junctions → non-manifold topology, so we must never touch a
    # mesh that is already watertight.
    if mesh.is_watertight:
        return mesh

    # Step 0: merge coincident vertices (fixes triangle soup from STL loading).
    # STL format stores every triangle independently so each face has its own 3
    # vertices even when adjacent faces share an edge position.  Without merging,
    # every edge looks like a boundary edge → fixMeshDegeneracies treats each
    # vertex as a non-manifold junction and explodes the vertex count 300×+.
    # This step is safe here because we already know the mesh is NOT watertight,
    # so any coincident vertices are genuine triangle-soup duplicates.
    if len(mesh.vertices) != len(np.unique(mesh.vertices, axis=0)):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
        logger.debug("  '%s' vertex merge: %d verts after merge", label, len(mesh.vertices))
        if mesh.is_watertight:
            return mesh

    logger.debug("Repairing '%s' before CSG (verts=%d, faces=%d)",
                 label, len(mesh.vertices), len(mesh.faces))

    # ------------------------------------------------------------------
    # Try meshlib first (most robust)
    # ------------------------------------------------------------------
    try:
        import numpy as _np
        import meshlib.mrmeshnumpy as _mrn
        import meshlib.mrmeshpy as _mr

        # Check INPUT for closed non-manifold topology (0 open boundary edges but
        # not watertight).  In that case fixMeshDegeneracies causes a 29x geometry
        # explosion and fillHoleNicely is a no-op (no holes to fill).  Instead, just
        # do the round-trip: meshFromFacesVerts auto-splits non-manifold edges during
        # import via the half-edge structure, producing clean topology at minimal cost.
        _input_ec: dict = {}
        for _f in mesh.faces:
            for _i in range(3):
                _va, _vb = int(_f[_i]), int(_f[(_i + 1) % 3])
                _ek = (min(_va, _vb), max(_va, _vb))
                _input_ec[_ek] = _input_ec.get(_ek, 0) + 1
        _input_open = sum(1 for _c in _input_ec.values() if _c == 1)
        _is_closed_nonmanifold = (_input_open == 0)  # no boundary → topology problem

        mesh_mr = _mrn.meshFromFacesVerts(
            mesh.faces.astype(_np.int32),
            mesh.vertices.astype(_np.float32),
        )

        if not _is_closed_nonmanifold:
            # Real open boundary edges → run full degeneracy + hole-fill repair.
            _mr.fixMeshDegeneracies(mesh_mr, _mr.FixMeshDegeneraciesParams())
            holes = _mr.findRightBoundary(mesh_mr.topology, None)
            if holes:
                for loop in holes:
                    _mr.fillHoleNicely(mesh_mr, loop[0], _mr.FillHoleNicelySettings())
        # else: the round-trip already resolved non-manifold edges; nothing more needed.

        out_verts = _mrn.getNumpyVerts(mesh_mr)
        out_faces = _mrn.getNumpyFaces(mesh_mr.topology)
        result = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
        result.fix_normals()

        if result.is_watertight:
            logger.debug("  '%s' repaired via meshlib: watertight (%d verts, %d faces)",
                         label, len(result.vertices), len(result.faces))
            return result

        ec: dict = {}
        for face in result.faces:
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                ek = (min(v0, v1), max(v0, v1))
                ec[ek] = ec.get(ek, 0) + 1
        open_edges = sum(1 for c in ec.values() if c == 1)

        if open_edges == 0:
            # Zero open boundary edges means the mesh has non-manifold topology
            # (e.g., 3+ faces sharing one edge, or fan vertices).  Applying
            # trimesh process=True here would re-merge the vertices that meshlib
            # just split, recreating the problem and exploding face count 4×.
            #
            # Instead: pre-import into manifold3d and immediately re-export.
            # manifold3d will cleanly discard any non-manifold faces during import
            # (the same faces it would drop mid-CSG), but by doing it here we get
            # back a perfectly watertight trimesh that manifold3d accepts 100% in
            # the subsequent CSG step — preventing holes / bad interior triangles.
            if MANIFOLD_AVAILABLE:
                try:
                    _pre_verts = np.asarray(result.vertices, dtype=np.float32)
                    _pre_faces = np.asarray(result.faces, dtype=np.uint32)
                    _pre_m3d_mesh = manifold3d.Mesh(
                        vert_properties=_pre_verts,
                        tri_verts=_pre_faces,
                    )
                    _pre_mgl = manifold3d.Manifold(_pre_m3d_mesh)
                    if not _pre_mgl.is_empty() and _pre_mgl.num_tri() > 0:
                        _pre_out = _pre_mgl.to_mesh()
                        _pre_clean_v = np.array(_pre_out.vert_properties, dtype=np.float64)
                        _pre_clean_f = np.array(_pre_out.tri_verts, dtype=np.int64)
                        _clean_result = trimesh.Trimesh(
                            vertices=_pre_clean_v,
                            faces=_pre_clean_f,
                            process=False,
                        )
                        _clean_result.fix_normals()
                        dropped = len(result.faces) - _pre_mgl.num_tri()
                        if _clean_result.is_watertight:
                            logger.info(
                                "  '%s' pre-cleaned via manifold3d: %d verts, %d faces "
                                "(watertight, dropped %d non-manifold faces)",
                                label, len(_clean_result.vertices),
                                len(_clean_result.faces), dropped,
                            )
                            return _clean_result
                        logger.warning(
                            "  '%s' manifold3d pre-clean produced non-watertight result "
                            "(%d faces, dropped %d). Returning meshlib result.",
                            label, len(_clean_result.faces), dropped,
                        )
                except Exception as _pre_exc:
                    logger.warning(
                        "  '%s' manifold3d pre-clean failed (%s). "
                        "Returning meshlib result as-is.",
                        label, _pre_exc,
                    )

            logger.warning(
                "  '%s' has 0 open edges but is still non-manifold after meshlib "
                "and manifold3d pre-clean. CSG result may have minor artifacts.",
                label,
            )
            return result

        logger.warning("  '%s' still NOT watertight after meshlib repair (%d open edges). "
                       "Falling back to trimesh repair.", label, open_edges)
        mesh = result

    except Exception as exc:
        logger.warning("  meshlib repair failed for '%s' (%s). Falling back to trimesh.",
                       label, exc)

    # ------------------------------------------------------------------
    # Fallback: trimesh repair (only reached when there are real open boundary edges)
    # ------------------------------------------------------------------
    import trimesh.repair as _tr_repair

    m = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=True,
    )
    _tr_repair.fix_normals(m, multibody=True)
    _tr_repair.fill_holes(m)

    if m.is_watertight:
        logger.debug("  '%s' repaired via trimesh fallback: watertight.", label)
    else:
        ec2: dict = {}
        for face in m.faces:
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                ek = (min(v0, v1), max(v0, v1))
                ec2[ek] = ec2.get(ek, 0) + 1
        open_edges2 = sum(1 for c in ec2.values() if c == 1)
        logger.warning(
            "  '%s' still NOT watertight after trimesh fallback (%d open edges, %d verts, %d faces). "
            "CSG result may have gaps.",
            label, open_edges2, len(m.vertices), len(m.faces)
        )

    return m


def _trimesh_to_manifold(mesh: trimesh.Trimesh, label: str = 'mesh') -> 'manifold3d.Manifold':
    """Convert a trimesh to manifold3d Manifold object.

    Repairs the mesh before conversion so manifold3d receives a clean input.
    A non-watertight input causes manifold3d to silently discard geometry,
    which leaves holes in subsequent CSG outputs.

    Logs the manifold3d status code and saves the input mesh as a debug STL
    whenever manifold3d rejects it (status != NoError or num_tri == 0).
    """
    if not MANIFOLD_AVAILABLE:
        raise RuntimeError("manifold3d is not available")

    # Repair non-manifold / non-watertight meshes before handing to manifold3d
    mesh = _repair_mesh_for_csg(mesh, label)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    # Log what we're handing to manifold3d
    logger.info(
        "  CSG INPUT  '%s': %d verts, %d faces, watertight=%s",
        label, len(vertices), len(faces), mesh.is_watertight
    )

    # Create Mesh then Manifold (correct API for manifold3d v3.x)
    m3d_mesh = manifold3d.Mesh(vert_properties=vertices, tri_verts=faces)
    m = manifold3d.Manifold(m3d_mesh)

    # Validate: manifold3d silently accepts bad inputs but reports them via status()
    status = m.status()
    if status != manifold3d.Error.NoError or m.num_tri() == 0:
        logger.error(
            "  CSG INPUT  '%s' REJECTED by manifold3d: status=%s, "
            "manifold_tris=%d (input watertight=%s, verts=%d, faces=%d). "
            "Saving input mesh for inspection.",
            label, status, m.num_tri(), mesh.is_watertight,
            len(mesh.vertices), len(mesh.faces)
        )
        save_debug_mesh(mesh, f"bad_csg_input_{label}")
    else:
        logger.info(
            "  CSG INPUT  '%s' accepted: manifold_tris=%d",
            label, m.num_tri()
        )

    return m


def _manifold_to_trimesh(manifold: 'manifold3d.Manifold', label: str = 'result') -> trimesh.Trimesh:
    """Convert a manifold3d Manifold object to trimesh.

    Checks whether the CSG result is empty (which indicates a non-manifold operand
    caused manifold3d to discard geometry) and logs the output watertight status.
    Saves an empty-marker file to the debug directory on failure.
    """
    status = manifold.status()
    n_tri = manifold.num_tri()

    if manifold.is_empty() or n_tri == 0:
        logger.error(
            "  CSG OUTPUT '%s' is EMPTY after operation: status=%s, num_tri=%d. "
            "One or more CSG operands were non-manifold — check 'bad_csg_input_*' "
            "files in the debug directory.",
            label, status, n_tri
        )
    else:
        logger.info(
            "  CSG OUTPUT '%s': manifold_tris=%d, volume=%.3f mm^3",
            label, n_tri, manifold.volume()
        )

    mesh_data = manifold.to_mesh()
    vertices = np.asarray(mesh_data.vert_properties, dtype=np.float64)
    faces = np.asarray(mesh_data.tri_verts, dtype=np.int64)

    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Fix normals on the raw output (process=False avoids unwanted vertex merges)
    result.fix_normals()

    logger.info(
        "  CSG OUTPUT '%s' as trimesh: %d verts, %d faces, watertight=%s",
        label, len(result.vertices), len(result.faces), result.is_watertight
    )
    if not result.is_watertight:
        save_debug_mesh(result, f"bad_csg_output_{label}")

    return result


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
        
        # Convert to manifold (repair applied inside _trimesh_to_manifold)
        prism_manifold = _trimesh_to_manifold(prism_mesh, 'prism')
        hull_manifold = _trimesh_to_manifold(hull_mesh, 'hull')

        # Perform subtraction: shell = prism - hull
        shell_manifold = prism_manifold - hull_manifold
        
        # Convert back to trimesh
        shell_mesh = _manifold_to_trimesh(shell_manifold, 'prism_minus_hull')
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell with cavity created: {len(shell_mesh.vertices)} vertices, "
                   f"{len(shell_mesh.faces)} faces in {elapsed_ms:.1f}ms")
        
        return shell_mesh, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to create shell with cavity: {e}")
        return None, elapsed_ms, False


# ============================================================================
# SURFACE THICKENING CORE
# ============================================================================

def _orient_normals_to_direction(
    vertex_normals: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Flip per-vertex normals so they all point into the same half-space
    as *direction*, then re-normalise.

    Args:
        vertex_normals: (N, 3) per-vertex unit normals (modified in-place).
        direction: (3,) reference direction.

    Returns:
        The (possibly flipped) normals array.
    """
    dots = np.dot(vertex_normals, direction)
    vertex_normals[dots < 0] *= -1
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    vertex_normals /= norms
    return vertex_normals


def _thicken_surface_along_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    half_thickness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thicken a triangle surface into a closed shell by offsetting each
    vertex along its normal in both directions and stitching side walls
    at boundary edges.

    This is the shared geometric kernel used by every blade / volume /
    thicken function in this module.

    Args:
        vertices: (N, 3) surface vertex positions.
        faces: (F, 3) triangle indices into *vertices*.
        vertex_normals: (N, 3) unit normals (already oriented).
        half_thickness: Offset distance on each side of the surface.

    Returns:
        ``(all_vertices, all_faces)`` ready for ``trimesh.Trimesh``.
        Vertex layout is ``[bottom_copy, top_copy]``.
    """
    n_verts = len(vertices)

    top_vertices = vertices + vertex_normals * half_thickness
    bottom_vertices = vertices - vertex_normals * half_thickness
    all_vertices = np.vstack([bottom_vertices, top_vertices])

    # Bottom cap (reversed winding) + top cap (offset indices)
    bottom_faces = faces[:, ::-1]
    top_faces = faces + n_verts

    # Side walls along boundary edges (edge_face_count == 1)
    edge_count: Dict[Tuple[int, int], int] = {}
    edge_to_face: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            ek = (min(v0, v1), max(v0, v1))
            edge_count[ek] = edge_count.get(ek, 0) + 1
            if ek not in edge_to_face:
                edge_to_face[ek] = (fi, v0, v1)

    side_faces = []
    for ek, cnt in edge_count.items():
        if cnt != 1:
            continue
        _, a, b = edge_to_face[ek]
        side_faces.append([a, b, b + n_verts])
        side_faces.append([a, b + n_verts, a + n_verts])

    parts = [bottom_faces, top_faces]
    if side_faces:
        parts.append(np.array(side_faces, dtype=np.int64))
    all_faces = np.vstack(parts)

    return all_vertices, all_faces


# ============================================================================
# CUTTING VOLUME / BLADE CONSTRUCTORS
# ============================================================================

def _create_cutting_volume_from_surface(
    parting_surface: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    thickness: float = 100.0
) -> trimesh.Trimesh:
    """
    Create a thick volume from the parting surface for CSG splitting.
    
    The volume is created by offsetting the parting surface along each
    vertex's surface normal in both directions, creating a uniformly
    thick slab that can be used for CSG operations.  This ensures the
    gap thickness is constant across the membrane regardless of its
    orientation relative to the pouring direction.
    
    Args:
        parting_surface: The parting surface mesh (with extended collar)
        pouring_direction: The pouring direction (used to orient normals
            consistently so "top" faces the positive direction)
        thickness: Total thickness of the slab (extends thickness/2 in
            each direction along the surface normal)
        
    Returns:
        A watertight volume mesh for CSG operations
    """
    direction = np.array(pouring_direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    vertices = np.asarray(parting_surface.vertices, dtype=np.float64)
    faces = np.asarray(parting_surface.faces, dtype=np.int64)

    temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vnormals = _orient_normals_to_direction(
        np.array(temp_mesh.vertex_normals, dtype=np.float64), direction
    )

    all_verts, all_faces = _thicken_surface_along_normals(
        vertices, faces, vnormals, thickness / 2.0
    )

    volume_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    volume_mesh.fix_normals()
    volume_mesh = _repair_mesh_for_csg(volume_mesh, 'cutting_volume')

    logger.debug("Created cutting volume: %dv, %df, thickness=%.1fmm",
                 len(volume_mesh.vertices), len(volume_mesh.faces), thickness)
    return volume_mesh


def _create_cutting_blade_from_membrane(
    membrane: trimesh.Trimesh,
    direction: np.ndarray,
    thickness: float = 0.001
) -> trimesh.Trimesh:
    """
    Create a thin watertight "blade" volume from the membrane for cutting.

    The blade is created by offsetting the membrane along each vertex's
    surface normal in both directions (±thickness/2) to create a thin
    solid volume that can be subtracted from the shell to produce a
    uniform-width gap.

    Before thickening the membrane is pre-cleaned (vertex merge +
    meshlib round-trip) to fix triangle-soup connectivity and
    non-manifold T-junctions.

    Args:
        membrane: The membrane mesh (e.g., outer collar).
        direction: Pouring direction (used to orient normals consistently).
        thickness: Total blade thickness (default: 0.001 mm = 1 micron).

    Returns:
        A watertight blade volume mesh.
    """
    direction = np.array(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    save_debug_mesh(membrane, 'blade_input_membrane')
    logger.debug("Blade input membrane: %dv, %df, watertight=%s",
                 len(membrane.vertices), len(membrane.faces), membrane.is_watertight)

    # ------------------------------------------------------------------
    # Pre-clean membrane topology
    # ------------------------------------------------------------------
    # 1. trimesh merge_vertices  → weld coincident verts (triangle soup fix)
    # 2. meshlib round-trip      → split non-manifold T-junction edges
    try:
        import meshlib.mrmeshnumpy as _mrn
        merged = trimesh.Trimesh(
            vertices=membrane.vertices, faces=membrane.faces, process=True)
        logger.debug("Blade pre-clean merge: %dv/%df → %dv/%df",
                     len(membrane.vertices), len(membrane.faces),
                     len(merged.vertices), len(merged.faces))

        mr_mesh = _mrn.meshFromFacesVerts(
            merged.faces.astype(np.int32), merged.vertices.astype(np.float32))
        vertices = _mrn.getNumpyVerts(mr_mesh).astype(np.float64)
        faces = _mrn.getNumpyFaces(mr_mesh.topology).astype(np.int64)
        logger.debug("Blade pre-clean meshlib: %dv/%df → %dv/%df",
                     len(merged.vertices), len(merged.faces),
                     len(vertices), len(faces))

        save_debug_mesh(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            'blade_input_membrane_precleaned')
    except Exception as exc:
        logger.debug("meshlib blade pre-clean skipped (%s), using raw membrane", exc)
        vertices = np.asarray(membrane.vertices, dtype=np.float64)
        faces = np.asarray(membrane.faces, dtype=np.int64)

    # ------------------------------------------------------------------
    # Thicken along per-vertex normals
    # ------------------------------------------------------------------
    temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vnormals = _orient_normals_to_direction(
        np.array(temp_mesh.vertex_normals, dtype=np.float64), direction)

    all_verts, all_faces = _thicken_surface_along_normals(
        vertices, faces, vnormals, thickness / 2.0)

    blade_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    blade_mesh.fix_normals()
    blade_mesh = _repair_mesh_for_csg(blade_mesh, 'cutting_blade')

    logger.debug("Created cutting blade: %dv, %df, thickness=%.6fmm",
                 len(blade_mesh.vertices), len(blade_mesh.faces), thickness)
    return blade_mesh


def extend_membrane_height(
    membrane: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    target_min_height: float,
    target_max_height: float
) -> trimesh.Trimesh:
    """
    Extend a membrane mesh vertically to match target height bounds.
    
    This is needed when using a cutting membrane (designed for one prism) to split
    another prism with different height bounds (e.g., using hard shell's outer collar
    to split the metamold).
    
    Algorithm:
    1. Convert membrane to aligned coordinate system (z = pouring direction)
    2. Find current height bounds of the membrane
    3. For each boundary edge at min_height, extend down to target_min_height
    4. For each boundary edge at max_height, extend up to target_max_height
    5. Create collar faces connecting original boundary to extended boundary
    
    Args:
        membrane: The cutting membrane mesh
        pouring_direction: Unit vector defining the "up" direction
        target_min_height: Target minimum height (along pouring direction)
        target_max_height: Target maximum height (along pouring direction)
        
    Returns:
        Extended membrane trimesh
    """
    if membrane is None or len(membrane.vertices) == 0:
        return membrane
    
    vertices = np.array(membrane.vertices, dtype=np.float64)
    faces = np.array(membrane.faces, dtype=np.int64)
    
    # Normalize pouring direction
    direction = np.array(pouring_direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Project all vertices onto pouring direction
    heights = np.dot(vertices, direction)
    current_min_height = heights.min()
    current_max_height = heights.max()
    
    logger.info(f"Extending membrane height from [{current_min_height:.2f}, {current_max_height:.2f}] "
               f"to [{target_min_height:.2f}, {target_max_height:.2f}]")
    
    # Check if extension is needed
    extension_below = target_min_height - current_min_height  # Should be negative if we need to go lower
    extension_above = target_max_height - current_max_height  # Should be positive if we need to go higher
    
    # Tolerance for identifying boundary vertices
    height_tol = (current_max_height - current_min_height) * 0.05  # 5% of height range
    
    if extension_below >= -0.1 and extension_above <= 0.1:
        logger.info("No significant height extension needed")
        return membrane
    
    # Find boundary edges
    edge_to_face = {}
    edge_count = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            third_v = int(face[(i + 2) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
            if edge_key not in edge_to_face:
                edge_to_face[edge_key] = (fi, v0, v1, third_v)
    
    boundary_edges = [(v0, v1) for (v0, v1), c in edge_count.items() if c == 1]
    
    if not boundary_edges:
        logger.warning("No boundary edges found - cannot extend membrane height")
        return membrane
    
    # Categorize boundary edges by height
    bottom_edges = []  # Edges near current min_height
    top_edges = []     # Edges near current max_height
    
    for v0, v1 in boundary_edges:
        h0 = heights[v0]
        h1 = heights[v1]
        avg_height = (h0 + h1) / 2
        
        # Check if edge is near bottom
        if avg_height < current_min_height + height_tol:
            bottom_edges.append((v0, v1))
        # Check if edge is near top
        elif avg_height > current_max_height - height_tol:
            top_edges.append((v0, v1))
    
    logger.info(f"Found {len(bottom_edges)} bottom edges, {len(top_edges)} top edges")
    
    # Extend vertices and create collar faces
    new_vertices = list(vertices)
    new_faces = list(faces)
    
    # Maps: original_vertex_index -> extended_vertex_index
    bottom_extension_map = {}
    top_extension_map = {}
    
    # Extend bottom boundary if needed
    if extension_below < -0.1 and len(bottom_edges) > 0:
        for v0, v1 in bottom_edges:
            for vi in [v0, v1]:
                if vi not in bottom_extension_map:
                    # Create extended vertex
                    old_pos = vertices[vi]
                    new_pos = old_pos + direction * extension_below  # extension_below is negative
                    new_idx = len(new_vertices)
                    new_vertices.append(new_pos)
                    bottom_extension_map[vi] = new_idx
        
        # Create collar faces for bottom
        for v0, v1 in bottom_edges:
            # Get winding from original face
            edge_key = (min(v0, v1), max(v0, v1))
            face_info = edge_to_face.get(edge_key)
            if face_info is not None:
                fi, orig_v0, orig_v1, _ = face_info
                # Use face traversal order (A=orig_v0 → B=orig_v1) to pick winding.
                # For extrusion in direction D (downward here), outward normal =
                # cross(B-A, D).  CCW from outside: [A, B, B_ext, A_ext] →
                #   tri1: [A, B, B_ext]   tri2: [A, B_ext, A_ext]
                a = orig_v0
                b = orig_v1
                ext_a = bottom_extension_map.get(a)
                ext_b = bottom_extension_map.get(b)
                if ext_a is not None and ext_b is not None:
                    new_faces.append([a, b, ext_b])
                    new_faces.append([a, ext_b, ext_a])
    
    # Extend top boundary if needed
    if extension_above > 0.1 and len(top_edges) > 0:
        for v0, v1 in top_edges:
            for vi in [v0, v1]:
                if vi not in top_extension_map:
                    # Create extended vertex
                    old_pos = vertices[vi]
                    new_pos = old_pos + direction * extension_above  # extension_above is positive
                    new_idx = len(new_vertices)
                    new_vertices.append(new_pos)
                    top_extension_map[vi] = new_idx
        
        # Create collar faces for top
        for v0, v1 in top_edges:
            # Get winding from original face
            edge_key = (min(v0, v1), max(v0, v1))
            face_info = edge_to_face.get(edge_key)
            if face_info is not None:
                fi, orig_v0, orig_v1, _ = face_info
                # Use face traversal order (A=orig_v0 → B=orig_v1) to pick winding.
                # For extrusion in direction D (upward here), outward normal =
                # cross(B-A, D).  CCW from outside: [A, A_ext, B_ext, B] →
                #   tri1: [A, A_ext, B_ext]   tri2: [A, B_ext, B]
                a = orig_v0
                b = orig_v1
                ext_a = top_extension_map.get(a)
                ext_b = top_extension_map.get(b)
                if ext_a is not None and ext_b is not None:
                    new_faces.append([a, ext_a, ext_b])
                    new_faces.append([a, ext_b, b])
    
    # Create extended mesh
    extended_mesh = trimesh.Trimesh(
        vertices=np.array(new_vertices, dtype=np.float64),
        faces=np.array(new_faces, dtype=np.int64),
        process=True
    )
    extended_mesh.fix_normals()
    
    logger.info(f"Extended membrane: {len(vertices)} -> {len(new_vertices)} vertices, "
               f"{len(faces)} -> {len(new_faces)} faces")
    
    return extended_mesh


def split_shell_with_membrane(
    shell_with_cavity: trimesh.Trimesh,
    membrane: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    blade_thickness: float = 0.001,
    prism_mesh: Optional[trimesh.Trimesh] = None,
    subtractor_mesh: Optional[trimesh.Trimesh] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool]:
    """
    Split the shell into two manifold halves using the membrane as a cutting blade.

    Performs the full CSG chain  (prism - subtractor) - blade  in a single
    uninterrupted manifold3d session when ``prism_mesh`` and ``subtractor_mesh``
    are supplied.  This avoids the manifold3d → trimesh → manifold3d round-trip
    that rebuilds internal topology and causes different (incorrect) CSG results.

    When the optional meshes are not provided the function falls back to the
    legacy  shell_with_cavity - blade  path (kept for backward compatibility).

    Args:
        shell_with_cavity: The shell mesh (prism - subtractor), used for display
                           and as the fallback CSG operand.
        membrane: The cutting membrane (outer collar extended parting surface).
        pouring_direction: Unit vector defining the "upper" direction.
        blade_thickness: Thickness of the cutting blade (default: 0.001 mm).
        prism_mesh: Optional raw prism trimesh.  When provided together with
                    ``subtractor_mesh`` the full chain is computed in one
                    manifold3d session.
        subtractor_mesh: Optional hull or part mesh that was subtracted from the
                         prism to form the shell cavity.

    Returns:
        Tuple of (shell_half_1, shell_half_2, blade_mesh, computation_time_ms, success)
        shell_half_1 is the half in the positive pouring direction (upper)
        shell_half_2 is the half in the negative pouring direction (lower)
        blade_mesh is the thin cutting blade used for the split
    """
    start_time = time.time()
    
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available - cannot perform CSG operations")
        return None, None, None, 0.0, False
    
    try:
        logger.info(f"Splitting shell using thin blade subtraction (thickness={blade_thickness:.6f}mm)...")
        
        # Normalize direction
        direction = np.array(pouring_direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Step 1: Create thin cutting blade from membrane
        logger.info("Creating cutting blade from membrane...")
        blade = _create_cutting_blade_from_membrane(membrane, direction, blade_thickness)
        logger.info(f"Blade: {len(blade.vertices)} verts, {len(blade.faces)} faces")
        save_debug_mesh(blade, 'cutting_blade_final')
        save_debug_mesh(shell_with_cavity, 'shell_with_cavity_input')
        
        # Diagnostic: blade and membrane properties (debug level to reduce log noise)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Membrane: {len(membrane.vertices)} verts, {len(membrane.faces)} faces, watertight={membrane.is_watertight}")
            # Count membrane boundary edges
            membrane_edge_count = {}
            for face in membrane.faces:
                for i in range(3):
                    v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                    edge_key = (min(v0, v1), max(v0, v1))
                    membrane_edge_count[edge_key] = membrane_edge_count.get(edge_key, 0) + 1
            membrane_boundary_edges = sum(1 for c in membrane_edge_count.values() if c == 1)
            logger.debug(f"  Membrane boundary edges: {membrane_boundary_edges}")
            
            logger.debug(f"Blade: watertight={blade.is_watertight}")
            # Count blade boundary edges
            blade_edge_count = {}
            for face in blade.faces:
                for i in range(3):
                    v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                    edge_key = (min(v0, v1), max(v0, v1))
                    blade_edge_count[edge_key] = blade_edge_count.get(edge_key, 0) + 1
            blade_boundary_edges = sum(1 for c in blade_edge_count.values() if c == 1)
            if blade_boundary_edges > 0:
                logger.warning(f"Blade has {blade_boundary_edges} open edges - not watertight!")
            
            # Bounding box info
            shell_bounds = shell_with_cavity.bounds
            blade_bounds = blade.bounds
            membrane_bounds = membrane.bounds
            logger.debug(f"Bounds - Shell: {shell_bounds[0]} to {shell_bounds[1]}")
            logger.debug(f"Bounds - Membrane: {membrane_bounds[0]} to {membrane_bounds[1]}")
            logger.debug(f"Bounds - Blade: {blade_bounds[0]} to {blade_bounds[1]}")
            
            # Check overlap along pouring direction
            shell_projections = shell_with_cavity.vertices @ direction
            blade_projections = blade.vertices @ direction
            shell_proj_min = float(shell_projections.min())
            shell_proj_max = float(shell_projections.max())
            blade_proj_min = float(blade_projections.min())
            blade_proj_max = float(blade_projections.max())
            logger.debug(f"Shell extent along pouring dir: [{shell_proj_min:.2f}, {shell_proj_max:.2f}]")
            logger.debug(f"Blade extent along pouring dir: [{blade_proj_min:.2f}, {blade_proj_max:.2f}]")
            
            overlap = not (blade_proj_max < shell_proj_min or blade_proj_min > shell_proj_max)
            if not overlap:
                logger.warning("Blade does NOT overlap with shell along pouring direction - cut may fail")
            
            # Check shell connectivity BEFORE cutting (expensive)
            shell_components_before = shell_with_cavity.split(only_watertight=False)
            logger.debug(f"Shell has {len(shell_components_before)} connected components BEFORE cutting")
            if len(shell_components_before) > 1:
                for i, comp in enumerate(sorted(shell_components_before, key=lambda m: len(m.faces), reverse=True)[:5]):
                    logger.debug(f"  Component {i+1}: {len(comp.vertices)} verts, {len(comp.faces)} faces")
        
        # Step 2: Convert to manifold3d (repair applied inside _trimesh_to_manifold)
        logger.info("Converting meshes to manifold...")
        blade_manifold = _trimesh_to_manifold(blade, 'cutting_blade')

        # Step 3: CSG subtraction
        # Preferred: full chain (prism - subtractor) - blade in one manifold3d
        # session.  This avoids the manifold3d→trimesh→manifold3d round-trip
        # which rebuilds internal topology and causes incorrect split results.
        use_chain = prism_mesh is not None and subtractor_mesh is not None
        if use_chain:
            logger.info("Performing CSG chain: (prism - subtractor) - blade (no round-trip)...")
            prism_manifold    = _trimesh_to_manifold(prism_mesh,     'prism')
            subtractor_manifold = _trimesh_to_manifold(subtractor_mesh, 'subtractor')
            cut_shell_manifold = (prism_manifold - subtractor_manifold) - blade_manifold
        else:
            logger.info("Performing CSG subtraction: shell - blade (legacy path)...")
            shell_manifold = _trimesh_to_manifold(shell_with_cavity, 'shell_with_cavity')
            cut_shell_manifold = shell_manifold - blade_manifold

        # Step 4: Convert back to trimesh
        cut_shell = _manifold_to_trimesh(cut_shell_manifold, 'shell_with_cavity_minus_blade')
        logger.info(f"Cut shell: {len(cut_shell.vertices)} verts, {len(cut_shell.faces)} faces")

        # Step 5: Split into connected components
        logger.info("Splitting into connected components...")
        components = cut_shell.split(only_watertight=False)
        logger.info(f"Found {len(components)} connected components")

        if len(components) < 2:
            logger.warning("Failed to split shell into two components - blade may not have cut through")
            elapsed_ms = (time.time() - start_time) * 1000
            if len(components) == 1:
                return components[0], None, blade, elapsed_ms, False
            return None, None, blade, elapsed_ms, False

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
        logger.info(f"  Half 1 (upper): {len(shell_half_1.vertices)} verts, {len(shell_half_1.faces)} faces "
                   f"watertight={shell_half_1.is_watertight}")
        logger.info(f"  Half 2 (lower): {len(shell_half_2.vertices)} verts, {len(shell_half_2.faces)} faces "
                   f"watertight={shell_half_2.is_watertight}")

        if len(components) > 2:
            logger.warning(f"  Note: {len(components) - 2} additional small fragments were discarded")

        return shell_half_1, shell_half_2, blade, elapsed_ms, True

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to split shell with membrane: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, elapsed_ms, False


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
        shell_manifold = _trimesh_to_manifold(shell_with_cavity, 'shell_with_cavity')
        cutting_manifold = _trimesh_to_manifold(cutting_volume, 'cutting_volume')
        
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
        box_manifold_1 = _trimesh_to_manifold(box_1, 'half_space_box_1')
        box_manifold_2 = _trimesh_to_manifold(box_2, 'half_space_box_2')
        
        # Intersect shell with each half-space box
        half_1_manifold = shell_manifold ^ box_manifold_1  # Intersection
        half_2_manifold = shell_manifold ^ box_manifold_2  # Intersection
        
        # Convert back to trimesh
        shell_half_1 = _manifold_to_trimesh(half_1_manifold, 'shell_half_1_intersection')
        shell_half_2 = _manifold_to_trimesh(half_2_manifold, 'shell_half_2_intersection')
        
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


# ============================================================================
# SURFACE THICKENING (for secondary membranes)
# ============================================================================

def thicken_surface_symmetric(
    surface_mesh: trimesh.Trimesh,
    thickness: float = 0.2
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Thicken a surface mesh symmetrically by offsetting along per-vertex normals.

    Creates a watertight solid by:
    1. Computing per-vertex normals from the surface
    2. Offsetting vertices by ±thickness/2 along normals
    3. Creating side faces to close the boundary

    The result is a manifold solid "slab" centred on the original surface.

    Args:
        surface_mesh: The input surface mesh to thicken.
        thickness: Total thickness (extends thickness/2 on each side).

    Returns:
        Tuple of (thickened_mesh, computation_time_ms, success).
    """
    start_time = time.time()

    if surface_mesh is None or len(surface_mesh.vertices) == 0:
        logger.error("Surface mesh is empty or None")
        return None, 0.0, False

    if thickness <= 0:
        logger.error("Thickness must be positive")
        return None, 0.0, False

    try:
        logger.info("Thickening surface symmetrically by %smm...", thickness)
        logger.info("  Input: %d verts, %d faces",
                     len(surface_mesh.vertices), len(surface_mesh.faces))

        vertices = np.asarray(surface_mesh.vertices, dtype=np.float64)
        faces = np.asarray(surface_mesh.faces, dtype=np.int64)

        # Per-vertex normals (no orientation step — normals used as-is)
        vnormals = np.asarray(surface_mesh.vertex_normals, dtype=np.float64)
        norms = np.linalg.norm(vnormals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        vnormals = vnormals / norms

        all_verts, all_faces = _thicken_surface_along_normals(
            vertices, faces, vnormals, thickness / 2.0)

        thickened_mesh = trimesh.Trimesh(
            vertices=all_verts, faces=all_faces, process=True)
        thickened_mesh.fix_normals()

        elapsed_ms = (time.time() - start_time) * 1000

        if thickened_mesh.is_watertight:
            logger.info("Surface thickened: %d verts, %d faces (watertight) in %.1fms",
                        len(thickened_mesh.vertices), len(thickened_mesh.faces), elapsed_ms)
        else:
            logger.warning("Surface thickened: %d verts, %d faces (NOT watertight) in %.1fms",
                           len(thickened_mesh.vertices), len(thickened_mesh.faces), elapsed_ms)

        return thickened_mesh, elapsed_ms, True

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to thicken surface: %s", e)
        import traceback
        traceback.print_exc()
        return None, elapsed_ms, False


def create_part_with_thickened_secondary(
    part_mesh: trimesh.Trimesh,
    secondary_surface: trimesh.Trimesh,
    secondary_thickness: float = 0.2
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Create a combined mesh of part + thickened secondary surface.
    
    This is used for metamold creation where we want the secondary membrane
    features to be part of the mold cavity.
    
    Args:
        part_mesh: The original part mesh
        secondary_surface: The smoothed secondary parting surface
        secondary_thickness: Thickness for the secondary surface (symmetric)
        
    Returns:
        Tuple of (combined_mesh, computation_time_ms, success)
    """
    start_time = time.time()
    
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available - cannot perform CSG operations")
        return None, 0.0, False
    
    if part_mesh is None or len(part_mesh.vertices) == 0:
        logger.warning("Part mesh is empty - returning None")
        return None, 0.0, False
    
    # If no secondary surface, just return the part mesh
    if secondary_surface is None or len(secondary_surface.vertices) == 0:
        logger.info("No secondary surface provided - using part mesh only")
        elapsed_ms = (time.time() - start_time) * 1000
        return part_mesh.copy(), elapsed_ms, True
    
    try:
        logger.info(f"Creating part mesh with thickened secondary surface...")
        logger.info(f"  Part: {len(part_mesh.vertices)} verts, {len(part_mesh.faces)} faces")
        logger.info(f"  Secondary: {len(secondary_surface.vertices)} verts, {len(secondary_surface.faces)} faces")
        logger.info(f"  Secondary thickness: {secondary_thickness}mm")
        
        # Step 1: Thicken the secondary surface
        thickened_secondary, thicken_time, thicken_success = thicken_surface_symmetric(
            secondary_surface, secondary_thickness
        )
        
        if not thicken_success or thickened_secondary is None:
            logger.warning("Failed to thicken secondary surface - using part mesh only")
            elapsed_ms = (time.time() - start_time) * 1000
            return part_mesh.copy(), elapsed_ms, True
        
        # Step 2: Boolean union part + thickened secondary
        logger.info("Performing boolean union: part + thickened_secondary...")
        
        part_manifold = _trimesh_to_manifold(part_mesh, 'part_mesh')
        secondary_manifold = _trimesh_to_manifold(thickened_secondary, 'thickened_secondary')
        
        combined_manifold = part_manifold + secondary_manifold
        combined_mesh = _manifold_to_trimesh(combined_manifold, 'part_union_thickened_secondary')
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if combined_mesh.is_watertight:
            logger.info(f"Combined mesh created: {len(combined_mesh.vertices)} verts, "
                       f"{len(combined_mesh.faces)} faces (watertight) in {elapsed_ms:.1f}ms")
        else:
            logger.warning(f"Combined mesh created: {len(combined_mesh.vertices)} verts, "
                          f"{len(combined_mesh.faces)} faces (NOT watertight) in {elapsed_ms:.1f}ms")
        
        return combined_mesh, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to create part with thickened secondary: {e}")
        import traceback
        traceback.print_exc()
        return None, elapsed_ms, False


# ============================================================================
# METAMOLD: ADD PART MESH BACK TO HALVES
# ============================================================================

def _remove_coplanar_shard_components(
    mesh: trimesh.Trimesh,
    label: str = 'mesh',
) -> trimesh.Trimesh:
    """Remove small disconnected components ("shards") produced by manifold3d
    when unioning meshes with exactly coplanar/coincident faces.

    manifold3d's symbolic perturbation creates tiny degenerate triangle fragments
    at shared boundaries (known issue — GitHub manifold #1430, #1359).  These
    fragments are individually watertight but have face counts orders of magnitude
    smaller than the main body.

    Strategy: split into components, keep those that pass BOTH a face-count
    threshold AND an area threshold.  A component is kept if it has either:
      - face_count >= 1% of the largest component (min 100), OR
      - surface area >= 1% of the total mesh area.
    The area check prevents removing legitimate geometry (e.g. prism base caps)
    that may have few but large triangles after trim_by_plane().

    Args:
        mesh: The raw union result from manifold3d (process=False).
        label: Human-readable label for logging.

    Returns:
        Cleaned mesh with shard components removed.  If the mesh has only one
        component or no small shards, the original mesh is returned unchanged.
    """
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        return mesh

    if len(components) <= 1:
        return mesh

    components_sorted = sorted(components, key=lambda c: -len(c.faces))
    main_faces = len(components_sorted[0].faces)
    face_threshold = max(main_faces * 0.01, 100)  # 1% of main, minimum 100 faces

    # Area-based threshold: protect components with meaningful surface area
    # even if they have few faces (e.g. prism base cap after trim_by_plane).
    total_area = float(mesh.area)
    area_threshold = total_area * 0.01  # 1% of total surface area

    kept = []
    for c in components_sorted:
        c_area = float(c.area)
        if len(c.faces) >= face_threshold or c_area >= area_threshold:
            kept.append(c)

    removed_count = len(components) - len(kept)
    removed_faces = sum(
        len(c.faces) for c in components_sorted
        if c not in kept
    )

    if removed_count == 0:
        return mesh

    logger.info(
        "  [Shard cleanup] '%s': removed %d small components (%d faces, %.1f%% of total). "
        "Kept %d component(s) (face_threshold=%d, area_threshold=%.2f mm²).",
        label, removed_count, removed_faces,
        100.0 * removed_faces / max(len(mesh.faces), 1),
        len(kept), int(face_threshold), area_threshold,
    )

    if len(kept) == 0:
        # Safety: never return empty — keep original
        logger.warning("  [Shard cleanup] '%s': all components below threshold — keeping original.", label)
        return mesh

    if len(kept) == 1:
        result = kept[0]
    else:
        result = trimesh.util.concatenate(kept)

    # Preserve process=False semantics so we don't merge manifold3d's split seam vertices
    return result


# Default simplification tolerance in mm.  manifold3d's CSG creates very thin
# sliver triangles along intersection curves (blade ∩ part surface).  A
# tolerance of 0.005 mm (5 µm) is well below the resolution of any desktop 3D
# printer (FDM ≈ 100–200 µm, SLA ≈ 25–50 µm) and eliminates ~90 % of the
# degenerate needles while keeping the mesh single-component.
CSG_SIMPLIFY_TOLERANCE: float = 0.005

# Needle edge-ratio threshold for the targeted edge-collapse pass.
# Triangles with longest_edge / shortest_edge > this value are collapsed.
# 50 is aggressive enough to eliminate visually objectionable slivers while
# preserving legitimate sharp-angle geometry.
CSG_NEEDLE_RATIO_THRESHOLD: float = 50.0


def _collapse_needle_edges(
    mesh: trimesh.Trimesh,
    ratio_threshold: float = CSG_NEEDLE_RATIO_THRESHOLD,
    max_iterations: int = 5,
    label: str = 'mesh',
    vertex_near_part: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """Iteratively collapse the shortest edge of needle triangles.

    For every triangle whose longest_edge / shortest_edge exceeds
    *ratio_threshold*, the two vertices of the shortest edge are merged to
    their midpoint.  Faces that become degenerate (two or more identical
    vertex indices) are removed.  The process repeats until no needles
    remain or *max_iterations* is reached.

    This is a targeted edge-collapse that only touches degenerate geometry —
    well-shaped triangles are never modified.

    Args:
        mesh: Input mesh (vertices may be modified in-place on a copy).
        ratio_threshold: Edge-ratio above which a triangle is considered
            a needle and its shortest edge is collapsed.
        max_iterations: Safety limit on collapse rounds.
        label: Human-readable label for logging.
        vertex_near_part: Optional boolean array (per-vertex, length N).
            When provided, only faces where ALL THREE vertices are near the
            part are eligible for collapse.  Faces with any "far" vertex
            are never touched, protecting outer prism geometry.  This array
            is indexed by vertex ID, which is stable across iterations
            (unlike per-face masks that become stale when faces are removed).

    Returns:
        Cleaned mesh with needle triangles eliminated.
    """
    import numpy as np

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    total_merges = 0
    total_removed = 0

    for iteration in range(max_iterations):
        v = verts[faces]
        e0 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
        e1 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
        e2 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
        longest = np.maximum(e0, np.maximum(e1, e2))
        shortest = np.minimum(e0, np.minimum(e1, e2))
        edge_ratio = longest / (shortest + 1e-15)

        bad = edge_ratio > ratio_threshold
        # Apply per-vertex proximity: a face is eligible only if ALL its
        # vertices are near the part.  Vertex IDs are stable across
        # iterations so this never goes stale.
        if vertex_near_part is not None:
            f = faces  # (F, 3)
            face_all_near = (
                vertex_near_part[f[:, 0]]
                & vertex_near_part[f[:, 1]]
                & vertex_near_part[f[:, 2]]
            )
            bad = bad & face_all_near
        if bad.sum() == 0:
            break

        # Compute per-vertex valence (number of adjacent faces).
        # The vertex with higher valence is a "structural" vertex (wall
        # corner, base perimeter, etc.) and must keep its position.
        # Moving it to the midpoint systematically pulls outer vertices
        # inward, shrinking the base — which is the bug we're fixing.
        valence = np.zeros(len(verts), dtype=np.int32)
        for vi in faces.ravel():
            valence[vi] += 1

        # Build a union-find merge map: for each bad face, merge the
        # shortest-edge vertex pair.
        merge_map = np.arange(len(verts), dtype=np.intp)
        merges = 0

        for fi in np.where(bad)[0]:
            f = faces[fi]
            d = [
                np.linalg.norm(verts[f[0]] - verts[f[1]]),
                np.linalg.norm(verts[f[1]] - verts[f[2]]),
                np.linalg.norm(verts[f[0]] - verts[f[2]]),
            ]
            pairs = [
                (int(f[0]), int(f[1])),
                (int(f[1]), int(f[2])),
                (int(f[0]), int(f[2])),
            ]
            a, b = pairs[int(np.argmin(d))]

            # Chase to root
            while merge_map[a] != a:
                a = merge_map[a]
            while merge_map[b] != b:
                b = merge_map[b]

            if a != b:
                # Keep the vertex with higher valence (more structural
                # connectivity) so that base perimeter / wall corners
                # stay in place.  Fall back to lower-index tie-breaking.
                if valence[a] >= valence[b]:
                    keep, drop = a, b
                else:
                    keep, drop = b, a
                merge_map[drop] = keep
                # Do NOT average to midpoint — keep structural vertex
                # position intact.  The dropped vertex is from a
                # degenerate sliver; its position is not meaningful.
                merges += 1

        # Flatten merge chains
        for i in range(len(merge_map)):
            root = i
            while merge_map[root] != root:
                root = merge_map[root]
            merge_map[i] = root

        # Remap faces and remove newly-degenerate ones
        faces = merge_map[faces]
        ok = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        removed = int(np.sum(~ok))
        faces = faces[ok]

        total_merges += merges
        total_removed += removed

    if total_merges > 0:
        result = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        result.remove_unreferenced_vertices()
        # NOTE: Do NOT call merge_vertices() here — the union-find collapse
        # already handled the merges precisely.  An additional merge_vertices()
        # with trimesh's distance-based tolerance can incorrectly merge nearby
        # but distinct vertices (e.g. base cap vs. wall vertices), breaking
        # connectivity and distorting the base outline.
        # Shard cleanup — collapse can disconnect tiny fragments
        result = _remove_coplanar_shard_components(result, f'{label}_collapsed')
        logger.info(
            "  [Edge collapse] '%s': %d merges, %d degenerate faces removed "
            "in %d iteration(s)",
            label, total_merges, total_removed, min(iteration + 1, max_iterations),
        )
        return result

    return mesh


def cleanup_csg_mesh(
    mesh: trimesh.Trimesh,
    label: str = 'mesh',
    tolerance: float = CSG_SIMPLIFY_TOLERANCE,
    needle_threshold: float = CSG_NEEDLE_RATIO_THRESHOLD,
    part_mesh: Optional[trimesh.Trimesh] = None,
    proximity_radius: float = 2.0,
) -> trimesh.Trimesh:
    """Clean up degenerate sliver triangles produced by manifold3d CSG operations.

    manifold3d creates very thin triangles along intersection curves where two
    meshes meet (e.g. where the cutting blade intersects the part surface).
    These slivers have edge ratios in the millions and areas near zero, causing
    visual artifacts and potential slicer issues for 3D printing.

    Two modes of operation:

    **Global cleanup** (``part_mesh=None``, used for hard-shell):
      1. manifold3d simplify — collapses edges shorter than *tolerance*.
      2. Targeted edge collapse — collapses remaining needles.

    **Selective cleanup** (``part_mesh`` provided, used for metamold):
      Skips the global manifold3d simplify (which distorts prism base caps)
      and only performs targeted edge collapse on needle triangles whose
      centroids are within *proximity_radius* mm of the part mesh.  Outer
      prism geometry (walls, base caps) is left untouched.

    Args:
        mesh: Post-CSG trimesh (typically from _manifold_to_trimesh).
        label: Human-readable label for logging.
        tolerance: Maximum surface displacement in mm for the simplify pass.
            Default is CSG_SIMPLIFY_TOLERANCE (0.005 mm / 5 µm).
        needle_threshold: Edge-ratio threshold for the edge-collapse pass.
            Default is CSG_NEEDLE_RATIO_THRESHOLD (50).
        part_mesh: Optional part mesh.  When provided, enables selective
            cleanup: the global simplify pass is skipped, and only needle
            triangles near the part are collapsed.
        proximity_radius: Distance threshold (mm) from part surface within
            which faces are eligible for cleanup.  Default 2.0 mm.

    Returns:
        Cleaned mesh with degenerate triangles eliminated.  The original mesh
        is returned unchanged if manifold3d is unavailable or if cleanup fails.
    """
    if not MANIFOLD_AVAILABLE:
        return mesh
    if mesh is None or len(mesh.faces) == 0:
        return mesh

    try:
        import numpy as np

        # --- pre-cleanup stats (for logging) ---
        areas = mesh.area_faces
        v = mesh.vertices[mesh.faces]
        e0 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
        e1 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
        e2 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
        longest = np.maximum(e0, np.maximum(e1, e2))
        shortest = np.minimum(e0, np.minimum(e1, e2))
        edge_ratio = longest / (shortest + 1e-15)
        pre_zero = int(np.sum(areas < 1e-10))
        pre_needles = int(np.sum(edge_ratio > needle_threshold))

        if pre_needles == 0 and pre_zero == 0:
            logger.info("  [CSG cleanup] '%s': no degenerate triangles — skipping", label)
            return mesh

        # -----------------------------------------------------------------
        # Selective mode: skip global simplify, only collapse near part
        # -----------------------------------------------------------------
        if part_mesh is not None:
            # --- Step A: Remove zero-area degenerate faces ---
            # These are faces with area ≈ 0 produced at CSG intersection
            # curves.  Removing them is non-destructive (no vertex movement).
            zero_area_mask = areas < 1e-10
            if zero_area_mask.sum() > 0:
                logger.info(
                    "  [CSG cleanup] '%s': removing %d zero-area faces",
                    label, int(zero_area_mask.sum()),
                )
                keep_mask = ~zero_area_mask
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces[keep_mask],
                    process=False,
                )
                mesh.remove_unreferenced_vertices()
                # Recompute edge ratios for the filtered mesh
                v = mesh.vertices[mesh.faces]
                e0 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
                e1 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
                e2 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
                longest = np.maximum(e0, np.maximum(e1, e2))
                shortest = np.minimum(e0, np.minimum(e1, e2))
                edge_ratio = longest / (shortest + 1e-15)

            # --- Step B: Per-vertex proximity (stable across iterations) ---
            # Compute once: which vertices are within proximity_radius of
            # the part surface.  Vertex IDs don't change during edge
            # collapse, so this stays valid across all iterations.
            all_verts = np.asarray(mesh.vertices, dtype=np.float64)
            try:
                _, vert_dists, _ = part_mesh.nearest.on_surface(all_verts)
                vertex_near_part = vert_dists < proximity_radius
            except Exception:
                # Fallback: bounding-box distance heuristic
                part_min = part_mesh.bounds[0] - proximity_radius
                part_max = part_mesh.bounds[1] + proximity_radius
                vertex_near_part = np.all(
                    (all_verts >= part_min) & (all_verts <= part_max), axis=1
                )

            near_vert_count = int(vertex_near_part.sum())
            far_vert_count = len(vertex_near_part) - near_vert_count
            # Face is eligible if ALL 3 vertices are near part
            face_all_near = (
                vertex_near_part[mesh.faces[:, 0]]
                & vertex_near_part[mesh.faces[:, 1]]
                & vertex_near_part[mesh.faces[:, 2]]
            )
            near_needles = int(np.sum((edge_ratio > needle_threshold) & face_all_near))
            logger.info(
                "  [CSG cleanup] '%s' (selective): %d/%d verts near part, "
                "%d needles eligible for collapse",
                label, near_vert_count, len(vertex_near_part), near_needles,
            )

            if near_needles == 0:
                logger.info("  [CSG cleanup] '%s': no near-part needles — skipping", label)
                result = mesh
            else:
                result = _collapse_needle_edges(
                    mesh,
                    ratio_threshold=needle_threshold,
                    max_iterations=5,
                    label=label,
                    vertex_near_part=vertex_near_part,
                )

        else:
            # -----------------------------------------------------------------
            # Global mode: full simplify + collapse (for hard shell etc.)
            # -----------------------------------------------------------------

            # --- Pass 1: manifold3d simplify ---
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces_arr = np.asarray(mesh.faces, dtype=np.uint32)
            import manifold3d
            m3_mesh = manifold3d.Mesh(vert_properties=verts, tri_verts=faces_arr)
            m3 = manifold3d.Manifold(m3_mesh)
            simplified = m3.simplify(tolerance)

            out = simplified.to_mesh()
            result = trimesh.Trimesh(
                vertices=np.array(out.vert_properties[:, :3]),
                faces=np.array(out.tri_verts),
                process=False,
            )
            # NOTE: Do NOT call merge_vertices() here.  manifold3d's output has
            # intentionally split vertices at seam edges (see _manifold_to_trimesh
            # which also uses process=False for this reason).  Merging them can
            # create T-junctions that break manifold topology and cause face
            # connectivity issues — in particular, base-cap vertices can get merged
            # with nearby wall vertices, distorting the prism base outline.

            # Shard cleanup (simplify may split off tiny regions)
            result = _remove_coplanar_shard_components(result, f'{label}_simplified')

            # --- Pass 2: targeted edge collapse for remaining needles ---
            result = _collapse_needle_edges(
                result,
                ratio_threshold=needle_threshold,
                max_iterations=5,
                label=label,
            )

        # --- post-cleanup stats ---
        areas2 = result.area_faces
        v2 = result.vertices[result.faces]
        e0 = np.linalg.norm(v2[:, 1] - v2[:, 0], axis=1)
        e1 = np.linalg.norm(v2[:, 2] - v2[:, 1], axis=1)
        e2 = np.linalg.norm(v2[:, 0] - v2[:, 2], axis=1)
        longest2 = np.maximum(e0, np.maximum(e1, e2))
        shortest2 = np.minimum(e0, np.minimum(e1, e2))
        edge_ratio2 = longest2 / (shortest2 + 1e-15)
        post_zero = int(np.sum(areas2 < 1e-10))
        post_needles = int(np.sum(edge_ratio2 > needle_threshold))

        logger.info(
            "  [CSG cleanup] '%s': %df→%df, "
            "zero-area: %d→%d, needles(>%.0f): %d→%d",
            label,
            len(mesh.faces), len(result.faces),
            pre_zero, post_zero,
            needle_threshold, pre_needles, post_needles,
        )

        return result

    except Exception as exc:
        logger.warning("  [CSG cleanup] '%s' failed — returning original: %s", label, exc)
        return mesh


def add_part_to_metamold_half(
    metamold_half: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Boolean ADD (union) the part mesh to a metamold half.

    This operation adds the part mesh back into the metamold half, creating
    a solid where the part would be placed.  Because the metamold cavity was
    created by subtracting the same part mesh, the cavity inner surface and
    the part outer surface are exactly coincident — this triggers manifold3d's
    coplanar-face shard artifacts (see GitHub manifold #1430).

    Post-processing removes the degenerate shard components, yielding a clean
    single-component watertight mesh.

    Args:
        metamold_half: One half of the split metamold (with cavity)
        part_mesh: The original part mesh to add back

    Returns:
        Tuple of (result_mesh, computation_time_ms, success)
        result_mesh is the metamold half with the part added as solid geometry
    """
    start_time = time.time()

    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available - cannot perform CSG operations")
        return None, 0.0, False

    if metamold_half is None or len(metamold_half.vertices) == 0:
        logger.error("Metamold half is empty or None")
        return None, 0.0, False

    if part_mesh is None or len(part_mesh.vertices) == 0:
        logger.error("Part mesh is empty or None")
        return None, 0.0, False

    try:
        logger.info(f"Adding part mesh to metamold half (boolean union)...")
        logger.info(f"  Metamold half: {len(metamold_half.vertices)} verts, {len(metamold_half.faces)} faces")
        logger.info(f"  Part mesh: {len(part_mesh.vertices)} verts, {len(part_mesh.faces)} faces")

        # Convert to manifold
        half_manifold = _trimesh_to_manifold(metamold_half, 'metamold_half')
        part_manifold = _trimesh_to_manifold(part_mesh, 'part_mesh_for_union')

        # Perform union: result = half + part
        result_manifold = half_manifold + part_manifold

        # Convert back to trimesh
        result_mesh = _manifold_to_trimesh(result_manifold, 'metamold_union_part')

        # Save the raw union result BEFORE shard cleanup for debugging
        save_debug_mesh(result_mesh, 'metamold_union_RAW_before_cleanup')

        # -----------------------------------------------------------
        # Post-process: remove coplanar shard components.
        #
        # The cavity was carved from the same part mesh we are now adding
        # back, so ~99% of the part faces are exactly coincident with
        # cavity faces (opposite normals).  manifold3d's symbolic
        # perturbation produces hundreds of tiny degenerate triangle
        # fragments ("shards") at exactly coincident boundaries.
        #
        # Keeping only the largest component(s) removes these shards
        # cleanly while preserving all legitimate geometry.
        # -----------------------------------------------------------
        result_mesh = _remove_coplanar_shard_components(result_mesh, 'metamold_union')

        elapsed_ms = (time.time() - start_time) * 1000

        # Verify manifold output
        if result_mesh.is_watertight:
            logger.info(f"Part added to metamold half: {len(result_mesh.vertices)} verts, "
                       f"{len(result_mesh.faces)} faces (watertight) in {elapsed_ms:.1f}ms")
        else:
            logger.warning(f"Part added to metamold half: {len(result_mesh.vertices)} verts, "
                          f"{len(result_mesh.faces)} faces (NOT watertight) in {elapsed_ms:.1f}ms")

        return result_mesh, elapsed_ms, True

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to add part to metamold half: {e}")
        import traceback
        traceback.print_exc()
        return None, elapsed_ms, False


def add_part_to_metamold_halves(
    metamold_half_1: trimesh.Trimesh,
    metamold_half_2: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool]:
    """
    Boolean ADD (union) the part mesh to both metamold halves.
    
    This is a convenience function that calls add_part_to_metamold_half for both halves.
    
    Args:
        metamold_half_1: First metamold half (upper, positive pouring direction)
        metamold_half_2: Second metamold half (lower, negative pouring direction)
        part_mesh: The original part mesh to add back
        
    Returns:
        Tuple of (half_1_with_part, half_2_with_part, total_computation_time_ms, success)
    """
    start_time = time.time()
    
    logger.info("Adding part mesh to both metamold halves...")
    
    # Process half 1
    half_1_with_part, time_1, success_1 = add_part_to_metamold_half(metamold_half_1, part_mesh)
    
    if not success_1:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to add part to metamold half 1")
        return None, None, elapsed_ms, False
    
    # Process half 2
    half_2_with_part, time_2, success_2 = add_part_to_metamold_half(metamold_half_2, part_mesh)
    
    if not success_2:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to add part to metamold half 2")
        return half_1_with_part, None, elapsed_ms, False
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Part added to both metamold halves in {elapsed_ms:.1f}ms")
    logger.info(f"  Half 1 with part: {len(half_1_with_part.vertices)} verts, {len(half_1_with_part.faces)} faces")
    logger.info(f"  Half 2 with part: {len(half_2_with_part.vertices)} verts, {len(half_2_with_part.faces)} faces")
    
    return half_1_with_part, half_2_with_part, elapsed_ms, True


def trim_metamold_halves(
    upper_half: trimesh.Trimesh,
    lower_half: trimesh.Trimesh,
    parting_surface: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    trim_threshold: float = 4.0
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, float, float, float]:
    """
    Trim each metamold half's BASE to save 3D printing material.
    
    The upper half (positive pouring direction side) has its TOP trimmed.
    The lower half (negative pouring direction side) has its BOTTOM trimmed.
    Each half is trimmed independently so one never cuts into the other.
    
    Must be called AFTER all CSG operations (cavity, split, part union)
    so both prism walls and part geometry are cleanly clipped.
    
    The caller must pass the halves in the correct order as returned by
    ``split_shell_with_membrane``:
        half_1 → upper_half (positive pouring direction)
        half_2 → lower_half (negative pouring direction)
    
    Args:
        upper_half: The metamold half on the positive pouring-direction side.
        lower_half: The metamold half on the negative pouring-direction side.
        parting_surface: The parting surface mesh (height reference).
        pouring_direction: Unit vector for the resin pouring direction.
        trim_threshold: Keep at least this much material (mm) between the
            parting surface and the trimmed base.  Set ≤ 0 to disable.
    
    Returns:
        (trimmed_upper, trimmed_lower, upper_saved_mm, lower_saved_mm,
         computation_time_ms).
    """
    start_time = time.time()
    
    if not MANIFOLD_AVAILABLE:
        logger.warning("manifold3d not available – cannot trim metamold halves")
        return upper_half, lower_half, 0.0, 0.0, 0.0
    
    if trim_threshold <= 0:
        logger.info("Metamold trim disabled (threshold=%s)", trim_threshold)
        elapsed_ms = (time.time() - start_time) * 1000
        return upper_half, lower_half, 0.0, 0.0, elapsed_ms
    
    pouring_dir = np.asarray(pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    # ------------------------------------------------------------------
    # Parting surface height extent along the pouring direction
    # ------------------------------------------------------------------
    ps_verts = np.asarray(parting_surface.vertices, dtype=np.float64)
    ps_heights = ps_verts @ pouring_dir
    ps_min_height = float(np.min(ps_heights))
    ps_max_height = float(np.max(ps_heights))
    
    logger.info(f"Trimming metamold halves (threshold={trim_threshold:.1f}mm)")
    logger.info(f"  Parting surface height: {ps_min_height:.2f} to {ps_max_height:.2f}")
    
    upper_saved = 0.0
    lower_saved = 0.0
    
    # ------------------------------------------------------------------
    # Trim the UPPER half's TOP (its base, away from PS)
    # Keep everything BELOW (ps_max + threshold)
    # ------------------------------------------------------------------
    trimmed_upper = upper_half
    if upper_half is not None and len(upper_half.vertices) > 0:
        try:
            uh = np.asarray(upper_half.vertices, dtype=np.float64) @ pouring_dir
            half_max = float(np.max(uh))
            trim_at = ps_max_height + trim_threshold
            gap = half_max - trim_at
            
            logger.info(f"  Upper half extent: {float(np.min(uh)):.2f} to {half_max:.2f}")
            
            if gap > 0.01:
                manifold = _trimesh_to_manifold(upper_half, 'upper_half')
                # trim_by_plane(normal, offset) keeps geometry where dot(v, normal) >= offset.
                # To keep everything BELOW trim_at (i.e. dot(v, pouring_dir) <= trim_at),
                # we negate both: normal = -pouring_dir, offset = -trim_at.
                manifold = manifold.trim_by_plane(
                    (-pouring_dir).tolist(), float(-trim_at)
                )
                trimmed_upper = _manifold_to_trimesh(manifold, 'upper_half_trimmed')
                save_debug_mesh(trimmed_upper, 'upper_half_trimmed_RAW_before_cleanup')
                trimmed_upper = _remove_coplanar_shard_components(trimmed_upper, 'upper_half_trimmed')
                upper_saved = gap
                logger.info(f"    Trimmed upper top by {gap:.2f}mm (cut at h={trim_at:.2f})")
            else:
                logger.info(f"    Upper half: no trim needed (gap={gap:.2f}mm)")
        except Exception as e:
            logger.error(f"Failed to trim upper half: {e}")
    
    # ------------------------------------------------------------------
    # Trim the LOWER half's BOTTOM (its base, away from PS)
    # Keep everything ABOVE (ps_min - threshold)
    # ------------------------------------------------------------------
    trimmed_lower = lower_half
    if lower_half is not None and len(lower_half.vertices) > 0:
        try:
            lh = np.asarray(lower_half.vertices, dtype=np.float64) @ pouring_dir
            half_min = float(np.min(lh))
            trim_at = ps_min_height - trim_threshold
            gap = trim_at - half_min
            
            logger.info(f"  Lower half extent: {half_min:.2f} to {float(np.max(lh)):.2f}")
            
            if gap > 0.01:
                manifold = _trimesh_to_manifold(lower_half, 'lower_half')
                # normal = pouring_dir, offset = trim_at → keeps above trim_at
                manifold = manifold.trim_by_plane(
                    pouring_dir.tolist(), float(trim_at)
                )
                trimmed_lower = _manifold_to_trimesh(manifold, 'lower_half_trimmed')
                save_debug_mesh(trimmed_lower, 'lower_half_trimmed_RAW_before_cleanup')
                trimmed_lower = _remove_coplanar_shard_components(trimmed_lower, 'lower_half_trimmed')
                lower_saved = gap
                logger.info(f"    Trimmed lower bottom by {gap:.2f}mm (cut at h={trim_at:.2f})")
            else:
                logger.info(f"    Lower half: no trim needed (gap={gap:.2f}mm)")
        except Exception as e:
            logger.error(f"Failed to trim lower half: {e}")
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Metamold trim complete in {elapsed_ms:.1f}ms "
               f"(upper saved: {upper_saved:.2f}mm, lower saved: {lower_saved:.2f}mm)")
    
    return trimmed_upper, trimmed_lower, upper_saved, lower_saved, elapsed_ms


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


def _ray_2d_convex_polygon_exit(
    origin: np.ndarray,
    direction: np.ndarray,
    polygon: np.ndarray
) -> Optional[float]:
    """
    Find the distance from a point inside a convex polygon to its boundary along a ray.
    
    For a point inside a convex polygon, casts a ray along the given direction and
    finds the exit distance (closest forward intersection with a polygon edge).
    
    This is used by the adaptive collar extension to determine exactly how far each
    boundary vertex needs to extend to pass through the prism's offset silhouette.
    
    Args:
        origin: 2D point (2,) - assumed to be inside the polygon
        direction: 2D unit direction vector (2,)
        polygon: (N, 2) convex polygon vertices
        
    Returns:
        Distance along ray to exit the polygon, or None if no intersection found.
    """
    n = len(polygon)
    min_positive_t = None
    
    for i in range(n):
        j = (i + 1) % n
        
        # Edge from polygon[i] to polygon[j]
        p0 = polygon[i]
        edge = polygon[j] - p0
        
        # Solve: origin + t * direction = p0 + s * edge
        # t * direction - s * edge = p0 - origin
        # Using Cramer's rule:
        # denom = direction[0] * (-edge[1]) - direction[1] * (-edge[0])
        denom = direction[1] * edge[0] - direction[0] * edge[1]
        
        if abs(denom) < 1e-12:
            continue  # Ray parallel to edge
        
        diff = p0 - origin
        t = (diff[1] * edge[0] - diff[0] * edge[1]) / denom
        s = (diff[1] * direction[0] - diff[0] * direction[1]) / denom
        
        # Valid intersection: t > 0 (forward along ray) and 0 <= s <= 1 (on edge segment)
        if t > 1e-8 and -1e-8 <= s <= 1.0 + 1e-8:
            if min_positive_t is None or t < min_positive_t:
                min_positive_t = t
    
    return min_positive_t


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
    
    # Use process=False to prevent trimesh from modifying the carefully
    # constructed prism geometry (merge_vertices / remove_degenerate_faces
    # can collapse fan-center vertices or thin cap triangles, shrinking the
    # base outline).  fix_normals() is sufficient for a clean extrusion.
    prism_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=faces,
        process=False
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
    collar_subdivisions: int = DEFAULT_COLLAR_SUBDIVISIONS,
    wall_thickness: Optional[float] = None,
    prism_margin: float = 0.0,
    safety_margin: float = 2.0
) -> OuterCollarResult:
    """
    Extend the parting surface outward to pass through the hard shell prism boundary.
    
    This creates a "collar" around the outer boundary of the parting surface,
    ensuring it fully passes through the hard shell prism for proper CSG cutting.
    
    When wall_thickness is provided, uses adaptive per-vertex extension distances
    computed by ray-casting against the prism's 2D offset silhouette. This
    guarantees each collar vertex exits the prism boundary, even at sharp corners
    where a fixed multiplier would be insufficient.
    
    The extension is performed perpendicular to the pouring direction to 
    maintain the prism-like structure of the hard shell.
    
    Algorithm:
    1. Find outer boundary edges of parting surface (edges on hull, type 1 or 2)
    2. If wall_thickness provided, compute the prism's 2D offset silhouette
    3. For each outer boundary vertex:
       - Compute extension direction (perpendicular to pouring, outward from hull)
       - If adaptive: ray-cast in 2D to find exit distance from offset silhouette
       - Extend by max(computed_distance + safety_margin, extension_distance)
       - Create collar quad connecting parting surface to extended boundary
    4. Handle corners with fan triangles
    
    Args:
        parting_surface: The smoothed parting surface mesh
        inner_hull: The current hull mesh (∂H)
        vertex_boundary_type: Array with -1=part, 0=interior, 1/2=hull
        pouring_direction: Unit vector for resin pouring direction
        extension_distance: Minimum/fallback distance to extend outward (mm)
        collar_subdivisions: Number of fan subdivisions at corners
        wall_thickness: If provided, enables adaptive extension by computing
                       the prism's offset silhouette (same as create_hard_shell_prism)
        prism_margin: Additional margin used by the prism beyond wall_thickness (mm)
        safety_margin: Extra distance beyond the prism boundary for each vertex (mm)
        
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
    # ADAPTIVE EXTENSION: Compute prism's 2D offset silhouette for per-vertex distances
    # =========================================================================
    
    offset_silhouette_2d = None
    _collar_u_axis = None
    _collar_v_axis = None
    
    if wall_thickness is not None:
        # Build orthonormal basis for the projection plane (same as create_hard_shell_prism)
        if abs(pouring_dir[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        
        _collar_u_axis = arbitrary - np.dot(arbitrary, pouring_dir) * pouring_dir
        _collar_u_axis = _collar_u_axis / np.linalg.norm(_collar_u_axis)
        _collar_v_axis = np.cross(pouring_dir, _collar_u_axis)
        _collar_v_axis = _collar_v_axis / np.linalg.norm(_collar_v_axis)
        
        _collar_world_to_2d = np.column_stack([_collar_u_axis, _collar_v_axis, pouring_dir]).T
        
        # Project hull vertices to 2D and compute silhouette (same algorithm as prism)
        hull_vertices_3d = np.array(inner_hull.vertices, dtype=np.float64)
        hull_transformed = hull_vertices_3d @ _collar_world_to_2d.T
        projected_2d = hull_transformed[:, :2]
        
        try:
            hull_2d = ConvexHull(projected_2d)
            silhouette_indices = hull_2d.vertices
            silhouette_2d = projected_2d[silhouette_indices]
            
            # Offset by the same amount as the prism: wall_thickness + prism_margin
            total_offset = wall_thickness + prism_margin
            offset_silhouette_2d = _offset_polygon_2d(silhouette_2d, total_offset)
            
            logger.info(f"Adaptive extension: computed offset silhouette "
                       f"({len(offset_silhouette_2d)} vertices, offset={total_offset:.1f}mm, "
                       f"safety_margin={safety_margin:.1f}mm)")
        except Exception as e:
            logger.warning(f"Failed to compute offset silhouette for adaptive extension: {e}")
            logger.warning("Falling back to fixed extension_distance")
            offset_silhouette_2d = None
    
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
            except Exception:
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
    _adaptive_extensions = []  # Track per-vertex computed extensions for diagnostics
    
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
            except Exception:
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
            # Compute extension distance (adaptive or fixed)
            # ==================================================================
            
            actual_extension = extension_distance  # Default fallback
            
            if offset_silhouette_2d is not None and _collar_u_axis is not None:
                # Project vertex position and extension direction to 2D
                vi_2d = np.array([
                    np.dot(vi_pos, _collar_u_axis),
                    np.dot(vi_pos, _collar_v_axis)
                ])
                ext_2d = np.array([
                    np.dot(extension_dir, _collar_u_axis),
                    np.dot(extension_dir, _collar_v_axis)
                ])
                ext_2d_norm = np.linalg.norm(ext_2d)
                
                if ext_2d_norm > 1e-8:
                    ext_2d_unit = ext_2d / ext_2d_norm
                    exit_dist = _ray_2d_convex_polygon_exit(
                        vi_2d, ext_2d_unit, offset_silhouette_2d
                    )
                    
                    if exit_dist is not None:
                        # Convert 2D exit distance to 3D extension distance
                        # (accounts for ext_dir not being perfectly unit in 2D)
                        required_extension = (exit_dist + safety_margin) / ext_2d_norm
                        actual_extension = max(required_extension, extension_distance)
            
            collar_pt = vi_pos + extension_dir * actual_extension
            _adaptive_extensions.append(actual_extension)
            
            # Create collar vertex
            collar_idx = len(vertices)
            vertices.append(collar_pt.copy())
            edge_endpoint_collar[edge_key][vi] = collar_idx
            
            # Also store in simple map for non-corner vertices
            if vi not in vert_to_collar:
                vert_to_collar[vi] = collar_idx
            
            collar_vertices_created += 1
    
    logger.info(f"Created {collar_vertices_created} collar vertices")
    
    # Log adaptive extension diagnostics
    if _adaptive_extensions and offset_silhouette_2d is not None:
        ext_arr = np.array(_adaptive_extensions)
        n_above_default = int(np.sum(ext_arr > extension_distance + 0.01))
        logger.info(f"Adaptive extension stats: min={ext_arr.min():.2f}mm, max={ext_arr.max():.2f}mm, "
                   f"mean={ext_arr.mean():.2f}mm, {n_above_default}/{len(ext_arr)} exceeded "
                   f"default ({extension_distance:.1f}mm)")
    
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
