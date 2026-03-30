"""
Resin Channel Creation Module

Creates cylindrical holes in metamold halves for resin pouring inlets
and air escape pathways. These holes are placed at the global maximum
(large resin inlet) and local maxima (small air escape holes) identified
during pouring direction optimization.

The cylinders are boolean-subtracted from the metamold half that contains
the maxima, drilling in the opposite direction to the resin pouring direction
(i.e., from the outside surface inward toward the part cavity).

This step runs AFTER the metamold is generated and uses data from the
pouring direction optimization step.

Author: VcMoldCreator
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import trimesh

# CSG operations via manifold3d
try:
    import manifold3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_AIR_ESCAPE_DEPTH_MM = 1.5     # Default depth of air escape holes
DEFAULT_INLET_MIN_DEPTH_MM = 1.5     # Minimum depth of resin inlet hole
DEFAULT_LOCAL_DIAMETER_MM = 1.2       # Diameter for air escape holes (local maxima)
DEFAULT_GLOBAL_DIAMETER_MM = 5.2      # Top diameter for resin inlet hole (global maximum)
DEFAULT_GLOBAL_BOTTOM_DIAMETER_MM = 2.2  # Bottom diameter of tapered resin inlet hole
DEFAULT_SHELL_INLET_DIAMETER_MM = 17.0  # Diameter of hard-shell through-hole for resin inlet
PLUG_WIDE_DIAMETER_MM = 12.2            # Wide/flange section diameter of resin plug (through the shell hole)
PLUG_TOLERANCE_MM = 0.2                   # Tolerance clearance for plug dimensions
PLUG_SHELL_OVERHANG_MM = 5.0           # Extra height the wide section extends above the hard-shell top
PLUG_TAPER_ANGLE_DEG = 20.0               # Taper half-angle in degrees (plug-to-shell transition)
CYLINDER_SEGMENTS = 32                # Number of segments for cylinder approximation
INLET_DEPTH_SAMPLE_POINTS = 24        # Points around circumference for depth ray-casting
MAX_INTO_PART_ANGLE_DEG = 25.0        # Max angle for adaptive into-part direction
CSG_OVERLAP_MM = 0.5                  # Overlap distance for clean CSG union at junctions
INLET_CONTAINMENT_MARGIN_MM = 0.5     # Extra depth beyond first-contained point
SILICONE_POUR_HOLE_DIAMETER_MM = 12.0        # Locating-pin / plug diameter
SILICONE_POUR_SHELL_HOLE_DIAMETER_MM = 17.0  # Shell through-hole diameter (pin + 5 mm diametric tolerance)
SILICONE_POUR_CLEARANCE_MM = 5.0       # Min clearance from any other shell hole
SILICONE_POUR_GRID_STEP_MM = 1.5       # Grid resolution for candidate sampling


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResinChannelResult:
    """Result of creating resin channels in a metamold half.
    
    Contains the modified metamold half with cylindrical holes subtracted
    at the positions of the local and global maxima from pouring direction analysis.
    """
    
    # The modified metamold half with channels
    mesh: Optional[trimesh.Trimesh] = None
    
    # Which mold half was modified (1 = upper/positive pouring dir, 2 = lower)
    mold_half: int = 1
    
    # Channel parameters used
    air_escape_depth_mm: float = DEFAULT_AIR_ESCAPE_DEPTH_MM
    inlet_depth_mm: float = DEFAULT_INLET_MIN_DEPTH_MM
    inlet_depth_auto: float = 0.0  # Auto-computed inlet depth (before clamping to min)
    local_diameter_mm: float = DEFAULT_LOCAL_DIAMETER_MM
    global_diameter_mm: float = DEFAULT_GLOBAL_DIAMETER_MM
    global_bottom_diameter_mm: float = DEFAULT_GLOBAL_BOTTOM_DIAMETER_MM
    
    # Adaptive angled inlet geometry
    into_part_direction: Optional[np.ndarray] = None   # (3,) angled direction for lower section
    inlet_entry_point: Optional[np.ndarray] = None     # (3,) where inlet meets part surface
    inlet_angled_depth_mm: float = 0.0                  # Depth of angled lower section
    
    # Number of channels created
    n_local_channels: int = 0
    n_global_channels: int = 0
    
    # Positions where channels were placed
    local_channel_positions: Optional[np.ndarray] = None   # (N, 3)
    global_channel_position: Optional[np.ndarray] = None   # (3,) original maxima position
    inlet_cylinder_center: Optional[np.ndarray] = None     # (3,) offset center of inlet cylinder
    
    # The drilling direction used (opposite of resin pouring direction)
    drill_direction: Optional[np.ndarray] = None  # (3,) unit vector
    
    # Hard shell through-holes
    shell_half_modified: int = 0          # Which hard shell half was drilled (1 or 2, 0 = none)
    shell_inlet_diameter_mm: float = DEFAULT_SHELL_INLET_DIAMETER_MM
    n_shell_air_escapes: int = 0          # Number of air escape holes drilled in shell
    modified_shell_mesh: Optional[trimesh.Trimesh] = None  # The shell half with through-holes
    
    # Resin plug
    plug_mesh: Optional[trimesh.Trimesh] = None  # The resin pouring plug
    plug_merged_into_metamold: bool = False  # True when plug is unioned into metamold mesh

    # Silicone mold pouring holes (through BOTH hard shell halves)
    silicone_pour_hole_position: Optional[np.ndarray] = None  # (3,) 3-D centre of the hole
    silicone_pour_hole_diameter_mm: float = SILICONE_POUR_HOLE_DIAMETER_MM
    silicone_pour_hole_is_prism: bool = False  # True when prism-shaped cutter was used
    shell_half_1_final: Optional[trimesh.Trimesh] = None  # shell 1 with ALL holes applied
    shell_half_2_final: Optional[trimesh.Trimesh] = None  # shell 2 with ALL holes applied

    # Alignment notches (applied after resin channels and shell drilling)
    alignment_notches_applied: bool = False
    notched_shell_half_1:    Optional[trimesh.Trimesh] = None
    notched_shell_half_2:    Optional[trimesh.Trimesh] = None
    notched_metamold_half_1: Optional[trimesh.Trimesh] = None
    notched_metamold_half_2: Optional[trimesh.Trimesh] = None
    notch_width_mm: float = 4.0
    notch_depth_mm: float = 0.5

    # Metamold clamp
    clamp_mesh: Optional[trimesh.Trimesh] = None
    clamp_created: bool = False

    # Success flag
    success: bool = False
    
    # Computation time
    computation_time_ms: float = 0.0


# ============================================================================
# CYLINDER CREATION
# ============================================================================

def _create_cylinder(
    center: np.ndarray,
    direction: np.ndarray,
    radius: float,
    height: float,
    segments: int = CYLINDER_SEGMENTS
) -> trimesh.Trimesh:
    """
    Create a cylinder mesh centered at a point, extending along a direction.
    
    The cylinder starts at `center` and extends in `direction` by `height`.
    
    Args:
        center: (3,) starting point of the cylinder
        direction: (3,) unit vector for cylinder axis
        radius: Radius of the cylinder
        height: Height (length) of the cylinder
        segments: Number of segments for circular cross-section
        
    Returns:
        trimesh.Trimesh representing the cylinder
    """
    # Create a cylinder along Z-axis, then transform
    cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=segments
    )
    
    # The default cylinder is centered at origin along Z-axis
    # We need to:
    # 1. Align Z-axis to our direction
    # 2. Translate so the cylinder starts at center (not centered at origin)
    
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Build rotation matrix from Z-axis to target direction
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Handle near-parallel case
    dot = np.dot(z_axis, direction)
    if abs(dot - 1.0) < 1e-8:
        # Already aligned
        rotation_matrix = np.eye(4)
    elif abs(dot + 1.0) < 1e-8:
        # Opposite direction: rotate 180° around X-axis
        rotation_matrix = np.eye(4)
        rotation_matrix[1, 1] = -1.0
        rotation_matrix[2, 2] = -1.0
    else:
        # General case: axis-angle rotation
        axis = np.cross(z_axis, direction)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        
        # Rodrigues' rotation formula as 4x4 matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R
    
    # The default cylinder is centered at origin, so it extends from -height/2 to +height/2
    # We want it to start at `center` and go in `direction` for `height`
    # So translate by center + direction * height/2 (to move center of cylinder to correct position)
    midpoint = center + direction * (height / 2.0)
    
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = midpoint
    
    # Apply rotation then translation
    transform = translation_matrix @ rotation_matrix
    cylinder.apply_transform(transform)
    
    return cylinder


def _create_frustum(
    center: np.ndarray,
    direction: np.ndarray,
    top_radius: float,
    bottom_radius: float,
    height: float,
    segments: int = CYLINDER_SEGMENTS
) -> trimesh.Trimesh:
    """
    Create a frustum (truncated cone) mesh starting at center, extending along direction.

    The frustum starts at ``center`` with ``top_radius`` and tapers to
    ``bottom_radius`` at ``center + direction * height``.

    Args:
        center: (3,) starting point (top of frustum)
        direction: (3,) unit vector for frustum axis
        top_radius: Radius at the start (center)
        bottom_radius: Radius at the end (center + direction*height)
        height: Height (length) of the frustum along direction
        segments: Number of segments for circular cross-section

    Returns:
        trimesh.Trimesh representing the frustum
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Build in local space: Z-axis = frustum axis, z=0 = top, z=height = bottom
    vertices = []
    # Top ring (at z=0, radius = top_radius)
    for j in range(segments):
        vertices.append([top_radius * cos_a[j], top_radius * sin_a[j], 0.0])
    # Bottom ring (at z=height, radius = bottom_radius)
    for j in range(segments):
        vertices.append([bottom_radius * cos_a[j], bottom_radius * sin_a[j], height])
    # Cap centers
    top_center_idx = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    bottom_center_idx = len(vertices)
    vertices.append([0.0, 0.0, height])

    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    # Side quads (as triangle pairs) connecting top ring to bottom ring
    for j in range(segments):
        j_next = (j + 1) % segments
        t0 = j                    # top ring
        t1 = j_next               # top ring
        b0 = segments + j         # bottom ring
        b1 = segments + j_next    # bottom ring
        faces.append([t0, b0, t1])
        faces.append([t1, b0, b1])
    # Top cap
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([top_center_idx, j_next, j])
    # Bottom cap
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([bottom_center_idx, segments + j, segments + j_next])

    faces = np.array(faces, dtype=np.int64)

    # Transform from local Z-axis to target direction, positioned at center
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z_axis, direction)
    if abs(dot - 1.0) < 1e-8:
        R = np.eye(3)
    elif abs(dot + 1.0) < 1e-8:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    vertices = (R @ vertices.T).T + center

    frustum = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    frustum.fix_normals()
    return frustum


# ============================================================================
# MANIFOLD CSG HELPERS
# ============================================================================

def _repair_mesh_for_csg(mesh: trimesh.Trimesh, label: str = 'mesh') -> trimesh.Trimesh:
    """Repair a mesh to make it as watertight as possible before CSG operations.

    Uses meshlib as the primary engine (most robust) with trimesh as fallback.
    Non-watertight inputs cause manifold3d to silently discard geometry,
    leaving holes in the CSG output.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    if mesh.is_watertight:
        return mesh

    # ------------------------------------------------------------------
    # meshlib primary repair
    # ------------------------------------------------------------------
    try:
        import numpy as _np
        import meshlib.mrmeshnumpy as _mrn
        import meshlib.mrmeshpy as _mr

        # Skip fixMeshDegeneracies for closed non-manifold inputs (0 open edges)
        # — the round-trip alone resolves the topology at minimal cost.
        _input_ec2: dict = {}
        for _f2 in mesh.faces:
            for _i2 in range(3):
                _va2, _vb2 = int(_f2[_i2]), int(_f2[(_i2 + 1) % 3])
                _ek2 = (min(_va2, _vb2), max(_va2, _vb2))
                _input_ec2[_ek2] = _input_ec2.get(_ek2, 0) + 1
        _is_closed_nm2 = (sum(1 for _c2 in _input_ec2.values() if _c2 == 1) == 0)

        mesh_mr = _mrn.meshFromFacesVerts(
            mesh.faces.astype(_np.int32),
            mesh.vertices.astype(_np.float32),
        )
        if not _is_closed_nm2:
            _mr.fixMeshDegeneracies(mesh_mr, _mr.FixMeshDegeneraciesParams())
            for loop in _mr.findRightBoundary(mesh_mr.topology, None):
                _mr.fillHoleNicely(mesh_mr, loop[0], _mr.FillHoleNicelySettings())

        out_verts = _mrn.getNumpyVerts(mesh_mr)
        out_faces = _mrn.getNumpyFaces(mesh_mr.topology)
        result = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
        result.fix_normals()

        if result.is_watertight:
            return result

        ec: dict = {}
        for face in result.faces:
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                ek = (min(v0, v1), max(v0, v1))
                ec[ek] = ec.get(ek, 0) + 1
        open_edges = sum(1 for c in ec.values() if c == 1)

        if open_edges == 0:
            # Non-manifold topology (no holes), not a hole-fill problem.
            # Trimesh process=True would re-merge split vertices, undoing
            # meshlib's repair and causing geometry explosion.  Return as-is.
            logger.warning(
                "CSG input '%s': 0 open edges but not watertight (non-manifold topology). "
                "Skipping trimesh fallback to avoid geometry explosion.",
                label,
            )
            return result

        logger.warning("CSG input '%s': meshlib repair left %d open edges, falling back to trimesh.",
                       label, open_edges)
        mesh = result
    except Exception as exc:
        logger.warning("meshlib repair failed for '%s' (%s). Falling back to trimesh.", label, exc)

    # ------------------------------------------------------------------
    # trimesh fallback (only reached when there are real open boundary edges)
    # ------------------------------------------------------------------
    import trimesh.repair as _tr

    # Re-create with process=True to strip degenerate/duplicate faces
    # and merge coincident vertices.
    m = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=True,
    )
    _tr.fix_normals(m, multibody=True)
    _tr.fill_holes(m)

    if not m.is_watertight:
        ec2: dict = {}
        for face in m.faces:
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                ek = (min(v0, v1), max(v0, v1))
                ec2[ek] = ec2.get(ek, 0) + 1
        open_edges2 = sum(1 for c in ec2.values() if c == 1)
        logger.warning(
            "CSG input '%s' still NOT watertight after repair (%d open edges). "
            "CSG result may have gaps.",
            label, open_edges2
        )
    return m


def _trimesh_to_manifold(mesh: trimesh.Trimesh, label: str = 'mesh') -> 'manifold3d.Manifold':
    """Convert a trimesh to manifold3d Manifold object (with pre-repair and status check)."""
    if not MANIFOLD_AVAILABLE:
        raise RuntimeError("manifold3d is not available")

    mesh = _repair_mesh_for_csg(mesh, label)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    logger.info(
        "  CSG INPUT  '%s': %d verts, %d faces, watertight=%s",
        label, len(vertices), len(faces), mesh.is_watertight
    )

    m3d_mesh = manifold3d.Mesh(vert_properties=vertices, tri_verts=faces)
    m = manifold3d.Manifold(m3d_mesh)

    status = m.status()
    if status != manifold3d.Error.NoError or m.num_tri() == 0:
        logger.error(
            "  CSG INPUT  '%s' REJECTED by manifold3d: status=%s, "
            "manifold_tris=%d (input watertight=%s, verts=%d, faces=%d). "
            "Saving input mesh for inspection.",
            label, status, m.num_tri(), mesh.is_watertight,
            len(mesh.vertices), len(mesh.faces)
        )
        try:
            from core.mold_fabrication import save_debug_mesh
            save_debug_mesh(mesh, f"bad_csg_input_resin_{label}")
        except Exception:
            logger.debug("Failed to save debug mesh for bad CSG input '%s'", label, exc_info=True)
    else:
        logger.info("  CSG INPUT  '%s' accepted: manifold_tris=%d", label, m.num_tri())

    return m


def _manifold_to_trimesh(manifold: 'manifold3d.Manifold', label: str = 'result') -> trimesh.Trimesh:
    """Convert a manifold3d Manifold object to trimesh with status logging."""
    status = manifold.status()
    n_tri = manifold.num_tri()

    if manifold.is_empty() or n_tri == 0:
        logger.error(
            "  CSG OUTPUT '%s' is EMPTY: status=%s, num_tri=%d. "
            "One or more CSG operands were non-manifold.",
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
    result.fix_normals()

    logger.info(
        "  CSG OUTPUT '%s' as trimesh: %d verts, %d faces, watertight=%s",
        label, len(result.vertices), len(result.faces), result.is_watertight
    )
    if not result.is_watertight:
        try:
            from core.mold_fabrication import save_debug_mesh
            save_debug_mesh(result, f"bad_csg_output_resin_{label}")
        except Exception:
            logger.debug("Failed to save debug mesh for bad CSG output '%s'", label, exc_info=True)

    return result


# ============================================================================
# PLUG MERGE UTILITY
# ============================================================================

def merge_plug_into_metamold(
    metamold_mesh: trimesh.Trimesh,
    plug_mesh: trimesh.Trimesh,
) -> Optional[trimesh.Trimesh]:
    """
    Boolean-union the resin plug into the metamold mesh.

    This produces an integrated metamold where the pour spout is a permanent
    protrusion (no separate plug part is required during casting).

    The metamold already has the tapered inlet hole drilled; unioning the plug
    fills that cavity and adds the protruding stub section above the metamold
    surface.  The silicone is cast around this stub, forming a matching pour
    channel socket.  Resin is later poured into the socket without any
    loose plug piece.

    Args:
        metamold_mesh: The metamold half (already has resin channels drilled).
        plug_mesh:     The plug geometry (as returned by create_resin_plug).

    Returns:
        Merged trimesh, or the original metamold_mesh if the boolean fails.
    """
    if metamold_mesh is None or plug_mesh is None:
        return metamold_mesh

    if not MANIFOLD_AVAILABLE:
        logger.warning(
            "manifold3d not available — cannot merge plug into metamold; "
            "returning original metamold mesh"
        )
        return metamold_mesh

    try:
        mm_mani   = _trimesh_to_manifold(metamold_mesh, 'metamold_mesh')
        plug_mani = _trimesh_to_manifold(plug_mesh, 'plug_mesh')
        merged    = mm_mani + plug_mani   # Manifold boolean UNION
        result    = _manifold_to_trimesh(merged, 'metamold_union_plug')
        logger.info(
            "Plug merged into metamold: %d verts, %d faces",
            len(result.vertices), len(result.faces),
        )
        return result
    except Exception as exc:
        logger.error("merge_plug_into_metamold failed: %s", exc)
        return metamold_mesh


# ============================================================================
# DETERMINE WHICH HALF CONTAINS MAXIMA
# ============================================================================

def determine_maxima_half(
    metamold_half_1: trimesh.Trimesh,
    metamold_half_2: trimesh.Trimesh,
    maxima_positions: np.ndarray,
    resin_direction: np.ndarray
) -> int:
    """
    Determine which metamold half contains the maxima points.
    
    The maxima are bubble-trapping points at the TOP of the cavity when
    oriented for resin pouring. Since half_1 is the "upper" half (positive
    pouring direction), the maxima should typically be in half_1.
    
    We verify by projecting the maxima positions onto the pouring direction
    and comparing with the centroids of each half.
    
    Args:
        metamold_half_1: Upper metamold half
        metamold_half_2: Lower metamold half 
        maxima_positions: (N, 3) positions of maxima points
        resin_direction: (3,) resin pouring direction (up)
        
    Returns:
        1 if maxima belong to half_1, 2 if they belong to half_2
    """
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    
    # Project the centroid of maxima positions along resin direction
    maxima_center = np.mean(maxima_positions, axis=0)
    maxima_proj = np.dot(maxima_center, resin_dir)
    
    # Project centroids of each half
    centroid_1 = np.mean(metamold_half_1.vertices, axis=0)
    centroid_2 = np.mean(metamold_half_2.vertices, axis=0)
    proj_1 = np.dot(centroid_1, resin_dir)
    proj_2 = np.dot(centroid_2, resin_dir)
    
    # Maxima should be on the upper half (higher projection along resin direction)
    # The "upper" half is the one whose centroid is more aligned with resin direction
    if abs(maxima_proj - proj_1) < abs(maxima_proj - proj_2):
        target_half = 1
    else:
        target_half = 2
    
    logger.info(f"Maxima centroid projection: {maxima_proj:.2f}")
    logger.info(f"Half 1 centroid projection: {proj_1:.2f}, Half 2: {proj_2:.2f}")
    logger.info(f"Maxima assigned to half {target_half}")
    
    return target_half


# ============================================================================
# INLET POSITION OFFSET
# ============================================================================

def _compute_inlet_offset(
    inlet_position: np.ndarray,
    part_mesh: trimesh.Trimesh,
    drill_direction: np.ndarray,
    radius: float
) -> np.ndarray:
    """
    Compute the offset to shift the global inlet cylinder center so that the
    original inlet position (global maximum) lies on the circumference of the
    cylinder rather than at its center.

    The cylinder center is shifted TOWARD the centroid of the part's 2D convex
    hull (projected onto the plane perpendicular to the drill direction) by
    exactly one radius.

    Args:
        inlet_position: (3,) the global maximum position (will be on circumference)
        part_mesh: The original part mesh to compute 2D convex hull of
        drill_direction: (3,) unit vector for drilling axis (perpendicular plane)
        radius: Radius of the inlet cylinder (offset distance)

    Returns:
        (3,) the new cylinder center position
    """
    from scipy.spatial import ConvexHull

    drill_dir = np.asarray(drill_direction, dtype=np.float64)
    drill_dir = drill_dir / (np.linalg.norm(drill_dir) + 1e-10)

    # Build an orthonormal basis for the plane perpendicular to drill_direction
    # Pick an arbitrary vector not parallel to drill_dir
    if abs(drill_dir[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    u = np.cross(drill_dir, arbitrary)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(drill_dir, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # Project all part vertices onto the 2D plane (u, v)
    verts = np.asarray(part_mesh.vertices, dtype=np.float64)
    coords_2d = np.column_stack([verts @ u, verts @ v])  # (N, 2)

    # Compute 2D convex hull and its centroid
    try:
        hull_2d = ConvexHull(coords_2d)
        hull_pts = coords_2d[hull_2d.vertices]  # vertices on the hull
        centroid_2d = np.mean(hull_pts, axis=0)  # (2,)
    except Exception as e:
        logger.warning(f"Could not compute 2D convex hull, using mean of vertices: {e}")
        centroid_2d = np.mean(coords_2d, axis=0)

    # Project the inlet position onto the same 2D plane
    inlet_3d = np.asarray(inlet_position, dtype=np.float64)
    inlet_2d = np.array([inlet_3d @ u, inlet_3d @ v])  # (2,)

    # Direction from inlet toward hull centroid in 2D
    offset_dir_2d = centroid_2d - inlet_2d
    dist_2d = np.linalg.norm(offset_dir_2d)

    if dist_2d < 1e-8:
        # Inlet is at the centroid — no meaningful offset direction
        logger.warning("Inlet position coincides with 2D hull centroid, no offset applied")
        return inlet_3d.copy()

    offset_dir_2d = offset_dir_2d / dist_2d  # unit vector in 2D

    # Convert the 2D unit direction back to 3D
    offset_dir_3d = offset_dir_2d[0] * u + offset_dir_2d[1] * v
    offset_dir_3d = offset_dir_3d / (np.linalg.norm(offset_dir_3d) + 1e-10)

    # The new cylinder center is shifted by <radius> toward the centroid
    # so that the original inlet position sits on the circumference
    new_center = inlet_3d + offset_dir_3d * radius

    logger.info(f"  Inlet offset: shifted {radius:.2f}mm toward 2D hull centroid")
    logger.info(f"    Original pos: {inlet_3d}")
    logger.info(f"    New center:   {new_center}")
    logger.info(f"    Offset dir:   {offset_dir_3d}")

    return new_center


def _compute_inlet_depth(
    cylinder_center: np.ndarray,
    drill_direction: np.ndarray,
    radius: float,
    part_mesh: trimesh.Trimesh,
    min_depth_mm: float = DEFAULT_INLET_MIN_DEPTH_MM,
    n_samples: int = INLET_DEPTH_SAMPLE_POINTS
) -> float:
    """
    Compute the required drilling depth for the resin inlet so that a full
    circular cross-section of the given radius is cut into the part, even
    on inclined surfaces.

    For tapered inlets, pass the BOTTOM (smaller) radius so that the depth
    ensures the narrow end is fully cut into the part.

    Samples points around the circumference and ray-casts each one along
    the drill direction onto the part surface. The depth variation across
    the circumference (max - min intersection depth) tells us how much
    extra depth is needed to clear the incline.

    Final depth = depth_variation + min_depth_mm

    Args:
        cylinder_center: (3,) the cylinder center position
        drill_direction: (3,) unit vector for drilling axis
        radius: Radius of the cross-section to ensure is fully cut
            (for tapered inlets, use the bottom/smaller radius)
        part_mesh: Part mesh to ray-cast against
        min_depth_mm: Minimum base depth to drill beyond the incline
        n_samples: Number of sample points around the circumference

    Returns:
        Computed depth in mm (always >= min_depth_mm)
    """
    drill_dir = np.asarray(drill_direction, dtype=np.float64)
    drill_dir = drill_dir / (np.linalg.norm(drill_dir) + 1e-10)

    # Build orthonormal basis perpendicular to drill direction
    if abs(drill_dir[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    u = np.cross(drill_dir, arbitrary)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(drill_dir, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # Sample points on the circumference in the plane at cylinder_center
    center = np.asarray(cylinder_center, dtype=np.float64)
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    sample_origins = []
    for angle in angles:
        offset = radius * (np.cos(angle) * u + np.sin(angle) * v)
        sample_origins.append(center + offset)
    # Also include the center itself
    sample_origins.append(center.copy())
    sample_origins = np.array(sample_origins)  # (n_samples+1, 3)

    # Move sample origins far back along the OPPOSITE of drill direction
    # so rays start well above the surface and shoot downward
    bbox_diag = np.linalg.norm(part_mesh.bounds[1] - part_mesh.bounds[0])
    ray_offset = bbox_diag * 2.0
    ray_origins = sample_origins - drill_dir * ray_offset  # Start far above
    ray_directions = np.tile(drill_dir, (len(sample_origins), 1))

    # Ray-cast onto the part surface
    try:
        locations, index_ray, _ = part_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )
    except Exception as e:
        logger.warning(f"Ray-cast failed for inlet depth computation: {e}")
        return min_depth_mm

    if len(locations) == 0:
        logger.warning("No ray intersections found for inlet depth — using minimum depth")
        return min_depth_mm

    # For each ray, find the FIRST intersection (closest to the ray origin)
    # Project intersection points along drill direction relative to center
    intersection_depths = np.dot(locations - center, drill_dir)

    # Group by ray index and take the first hit for each ray
    first_hit_depth = {}
    for i, ray_idx in enumerate(index_ray):
        depth = intersection_depths[i]
        if ray_idx not in first_hit_depth or depth < first_hit_depth[ray_idx]:
            first_hit_depth[ray_idx] = depth

    if not first_hit_depth:
        logger.warning("No valid ray hits — using minimum depth")
        return min_depth_mm

    depths = list(first_hit_depth.values())
    depth_variation = max(depths) - min(depths)

    computed_depth = depth_variation + min_depth_mm
    logger.info(f"  Inlet depth computation: surface depth variation = {depth_variation:.2f}mm")
    logger.info(f"    Min surface depth: {min(depths):.2f}, Max: {max(depths):.2f}")
    logger.info(f"    Base min depth: {min_depth_mm:.2f}mm")
    logger.info(f"    Computed total depth: {computed_depth:.2f}mm")

    return max(computed_depth, min_depth_mm)


# ============================================================================
# ADAPTIVE ANGLED INLET HELPERS
# ============================================================================

def _find_part_entry_point(
    cylinder_center: np.ndarray,
    drill_direction: np.ndarray,
    part_mesh: trimesh.Trimesh,
    max_search_angle_deg: float = MAX_INTO_PART_ANGLE_DEG,
) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
    """
    Find the **shallowest** (nearest to *cylinder_center*) entry point on
    the part surface by casting rays in a cone around *drill_direction*.

    For convex parts the straight-down ray already gives the closest
    entry.  For concave / hollow parts (cups, bowls, tubes) an angled
    ray may hit a wall near the top at a much shorter distance than the
    straight-down ray which would hit the distant bottom.

    Algorithm:
      1. Generate candidate ray directions: the primary drill ray plus
         rings of tilted rays (5°–45° from drill, 12 azimuth samples
         per ring).
      2. Batch ray-cast all candidates at once.
      3. For each ray keep only the nearest forward hit.
      4. Across all rays pick the one whose first hit is closest to
         *cylinder_center* (Euclidean distance).

    Args:
        cylinder_center: (3,) starting point above metamold surface
        drill_direction: (3,) primary drill direction unit vector
        part_mesh: The original part mesh
        max_search_angle_deg: Maximum cone half-angle to search

    Returns:
        Tuple of ``(entry_point, distance, approach_direction)``.
        *entry_point* is ``None`` if no intersection is found.
        *approach_direction* is the ray direction that found the best hit
        (normalised).  Falls back to *drill_direction* on failure.
    """
    center = np.asarray(cylinder_center, dtype=np.float64)
    drill_dir = np.asarray(drill_direction, dtype=np.float64)
    drill_dir = drill_dir / (np.linalg.norm(drill_dir) + 1e-10)

    # --- build orthonormal basis perpendicular to drill_dir ---------------
    if abs(drill_dir[0]) < 0.9:
        arb = np.array([1.0, 0.0, 0.0])
    else:
        arb = np.array([0.0, 1.0, 0.0])
    u = np.cross(drill_dir, arb)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(drill_dir, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # --- generate candidate ray directions --------------------------------
    candidate_dirs = [drill_dir.copy()]                              # always include primary

    tilt_angles_deg = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    tilt_angles_deg = [t for t in tilt_angles_deg if t <= max_search_angle_deg]
    n_azimuth = 12

    for tilt_deg in tilt_angles_deg:
        tilt_rad = np.radians(tilt_deg)
        for az in np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False):
            perp = np.cos(az) * u + np.sin(az) * v
            d = np.cos(tilt_rad) * drill_dir + np.sin(tilt_rad) * perp
            d = d / (np.linalg.norm(d) + 1e-10)
            candidate_dirs.append(d)

    n_rays = len(candidate_dirs)
    origins = np.tile(center, (n_rays, 1))
    directions = np.array(candidate_dirs)

    # --- batch ray-cast ---------------------------------------------------
    try:
        locations, ray_indices, _ = part_mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions
        )
    except Exception as e:
        logger.warning("  Multi-ray cast to part failed: %s", e)
        return None, 0.0, drill_dir.copy()

    if len(locations) == 0:
        logger.warning("  No part intersection from any ray direction")
        return None, 0.0, drill_dir.copy()

    # --- per-ray nearest forward hit --------------------------------------
    best_per_ray: dict = {}                # ray_idx → (fwd_dist, location)
    for i in range(len(locations)):
        ray_idx = int(ray_indices[i])
        fwd = float(np.dot(locations[i] - center, candidate_dirs[ray_idx]))
        if fwd <= 0.01:
            continue
        if ray_idx not in best_per_ray or fwd < best_per_ray[ray_idx][0]:
            best_per_ray[ray_idx] = (fwd, locations[i].copy())

    if not best_per_ray:
        logger.warning("  No forward intersections with part surface")
        return None, 0.0, drill_dir.copy()

    # --- across all rays, find the shortest distance ----------------------
    best_dist = float("inf")
    best_entry = None
    best_ray_idx = 0
    for ray_idx, (fwd, loc) in best_per_ray.items():
        if fwd < best_dist:
            best_dist = fwd
            best_entry = loc
            best_ray_idx = ray_idx

    approach_dir = candidate_dirs[best_ray_idx].copy()
    tilt = np.degrees(np.arccos(np.clip(np.dot(approach_dir, drill_dir), -1, 1)))

    # Also report the straight-down distance for comparison
    straight_dist = best_per_ray.get(0, (float("inf"), None))[0]
    if straight_dist < float("inf") and best_dist < straight_dist - 0.5:
        logger.info("  Part entry: %.1fmm via %.1f° ray (straight-down was "
                     "%.1fmm — saved %.1fmm by angling)",
                     best_dist, tilt, straight_dist,
                     straight_dist - best_dist)
    else:
        logger.info("  Part entry point found at distance %.2fmm from surface"
                     " (approach %.1f° from drill, searched %d rays)",
                     best_dist, tilt, n_rays)

    return best_entry, best_dist, approach_dir


def _compute_into_part_direction(
    entry_point: np.ndarray,
    drill_direction: np.ndarray,
    part_mesh: trimesh.Trimesh,
    max_angle_deg: float = MAX_INTO_PART_ANGLE_DEG,
    bottom_radius: float = DEFAULT_GLOBAL_BOTTOM_DIAMETER_MM / 2.0,
    n_cone_samples: int = 32,
    n_depth_probes: int = 20,
    max_probe_depth_mm: float = 15.0,
    n_circle_samples: int = 8
) -> np.ndarray:
    """
    Compute the direction from the entry point that lets the frustum tip
    (a circle of *bottom_radius*) fit inside the part at the **shallowest**
    possible depth.

    Instead of maximising how many single-point probes are inside solid
    (which biases toward long paths along thin walls), this checks at
    each probe depth whether a full **circle** of ``bottom_radius`` is
    contained.  The scoring prefers directions where circle containment
    is achieved earliest (shallowest fitted depth).

    Algorithm:
      1. Build orthonormal basis perpendicular to drill_direction.
      2. Sample candidate directions at multiple tilt angles (0° to
         max_angle_deg) × azimuth angles (0° to 360°).
      3. For each candidate direction, build probe **circles** (not
         single points) at increasing depths from entry_point.
      4. Score = negative of the shallowest depth where ALL circle
         sample points are inside the part (lower depth = better).
      5. Among directions with the same shallowest fitted depth,
         prefer the smallest tilt angle (straightest path).
      6. If no direction achieves circle containment at any probed
         depth, fall back to single-point containment scoring.

    Args:
        entry_point: (3,) where the inlet meets the part surface
        drill_direction: (3,) original drill direction (straight into part)
        part_mesh: The original part mesh (should be watertight)
        max_angle_deg: Maximum deviation angle from drill_direction
        bottom_radius: Radius of the frustum bottom circle to check
        n_cone_samples: Number of azimuth samples per tilt ring
        n_depth_probes: Number of probe depths along each direction
        max_probe_depth_mm: Max depth to probe into the part
        n_circle_samples: Points around the circle perimeter to check

    Returns:
        (3,) unit vector for the best into-part direction
    """
    drill_dir = np.asarray(drill_direction, dtype=np.float64)
    drill_dir = drill_dir / (np.linalg.norm(drill_dir) + 1e-10)
    entry = np.asarray(entry_point, dtype=np.float64)

    # Build orthonormal basis perpendicular to drill_dir
    if abs(drill_dir[0]) < 0.9:
        arb = np.array([1.0, 0.0, 0.0])
    else:
        arb = np.array([0.0, 1.0, 0.0])
    u = np.cross(drill_dir, arb)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(drill_dir, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # --- generate candidate directions -----------------------------------
    tilt_angles_deg = [0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 45.0]
    tilt_angles_deg = [t for t in tilt_angles_deg if t <= max_angle_deg]
    if not tilt_angles_deg or tilt_angles_deg[0] != 0.0:
        tilt_angles_deg.insert(0, 0.0)

    candidates = []                    # (direction, tilt_deg)
    for tilt_deg in tilt_angles_deg:
        tilt_rad = np.radians(tilt_deg)
        if tilt_deg < 0.1:
            candidates.append((drill_dir.copy(), 0.0))
            continue
        n_az = max(6, n_cone_samples // len(tilt_angles_deg))
        for az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
            perp = np.cos(az) * u + np.sin(az) * v
            d = np.cos(tilt_rad) * drill_dir + np.sin(tilt_rad) * perp
            d = d / (np.linalg.norm(d) + 1e-10)
            candidates.append((d, tilt_deg))

    n_cand = len(candidates)
    probe_ts = np.linspace(0.3, max_probe_depth_mm, n_depth_probes)
    n_probes = len(probe_ts)

    # --- build per-direction orthonormal bases for circle points ----------
    circle_angles = np.linspace(0, 2 * np.pi, n_circle_samples, endpoint=False)
    cos_ca = np.cos(circle_angles)          # (n_circle,)
    sin_ca = np.sin(circle_angles)

    bases = []                               # (u_d, v_d) per candidate
    for d, _ in candidates:
        if abs(d[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])
        u_d = np.cross(d, a);  u_d /= (np.linalg.norm(u_d) + 1e-10)
        v_d = np.cross(d, u_d); v_d /= (np.linalg.norm(v_d) + 1e-10)
        bases.append((u_d, v_d))

    # --- build ALL probe points (circle + center per depth per cand) ------
    pts_per_depth = n_circle_samples + 1     # perimeter + centre
    total_pts = n_cand * n_probes * pts_per_depth
    all_points = np.empty((total_pts, 3), dtype=np.float64)

    idx = 0
    for i, (d, _) in enumerate(candidates):
        u_d, v_d = bases[i]
        for j, t in enumerate(probe_ts):
            c = entry + d * t
            for k in range(n_circle_samples):
                all_points[idx + k] = (c
                    + bottom_radius * (cos_ca[k] * u_d + sin_ca[k] * v_d))
            all_points[idx + n_circle_samples] = c      # centre point
            idx += pts_per_depth

    # --- batch containment check ------------------------------------------
    try:
        inside = part_mesh.contains(all_points)
    except Exception as e:
        logger.warning("  Containment check failed: %s — falling back to "
                       "drill dir", e)
        return drill_dir.copy()

    inside = inside.reshape(n_cand, n_probes, pts_per_depth)

    # A depth is "circle-contained" if ALL pts (perimeter + centre) inside
    circle_ok = np.all(inside, axis=2)          # (n_cand, n_probes)

    # --- PRIMARY score: shallowest depth where circle is contained --------
    # For each candidate, find the index of the first True along probes
    INF_DEPTH = max_probe_depth_mm + 100.0
    fitted_depth = np.full(n_cand, INF_DEPTH)
    for i in range(n_cand):
        first_ok = np.argmax(circle_ok[i])       # first True index
        if circle_ok[i, first_ok]:                # there IS a True
            fitted_depth[i] = probe_ts[first_ok]

    best_fitted = float(np.min(fitted_depth))

    if best_fitted >= INF_DEPTH:
        # --- FALLBACK: no direction achieves circle containment -----------
        # Use single-point (centre-only) containment count instead.
        centre_inside = inside[:, :, -1]          # (n_cand, n_probes)
        scores = np.sum(centre_inside, axis=1)
        best_score = int(np.max(scores))

        if best_score == 0:
            logger.warning("  Into-part direction: no solid material in any "
                           "direction — using drill dir")
            return drill_dir.copy()

        best_mask = scores == best_score
        tilts = np.array([candidates[i][1] for i in range(n_cand)])
        best_indices = np.where(best_mask)[0]
        winner_idx = best_indices[np.argmin(tilts[best_mask])]
        winner_dir, winner_tilt = candidates[winner_idx]

        logger.info("  Into-part direction (point-fallback): %.1f° from drill, "
                     "score=%d/%d (sampled %d dirs, circle never contained)",
                     winner_tilt, best_score, n_probes, n_cand)
        return winner_dir

    # --- among candidates with smallest fitted depth, pick min tilt -------
    DEPTH_TOL = 0.05                             # mm tolerance for ties
    best_mask = fitted_depth <= best_fitted + DEPTH_TOL
    tilts = np.array([candidates[i][1] for i in range(n_cand)])
    best_indices = np.where(best_mask)[0]
    winner_idx = best_indices[np.argmin(tilts[best_mask])]
    winner_dir, winner_tilt = candidates[winner_idx]

    # How many circle-contained depths does the winner have?
    n_ok = int(np.sum(circle_ok[winner_idx]))

    logger.info("  Into-part direction: %.1f° from drill, circle fits at "
                "%.1fmm depth (%d/%d depths OK, sampled %d dirs)",
                winner_tilt, best_fitted, n_ok, n_probes, n_cand)

    return winner_dir


def _compute_angled_inlet_depth(
    entry_point: np.ndarray,
    into_part_dir: np.ndarray,
    bottom_radius: float,
    part_mesh: trimesh.Trimesh,
    min_depth_mm: float = DEFAULT_INLET_MIN_DEPTH_MM,
    n_circle_samples: int = INLET_DEPTH_SAMPLE_POINTS,
    containment_margin_mm: float = INLET_CONTAINMENT_MARGIN_MM
) -> float:
    """
    Compute the MINIMUM depth along *into_part_dir* from *entry_point*
    such that the bottom circle of *bottom_radius* is fully contained in
    the part, plus a small safety margin.

    This ensures the taper is as SHORT and STEEP as possible — the tip
    dives into the part just enough to be fully embedded.

    Uses a linear scan followed by binary-search refinement to find the
    shallowest depth at which the entire bottom circle is inside the part.

    Args:
        entry_point: (3,) starting point on part surface
        into_part_dir: (3,) direction going into the part interior
        bottom_radius: Radius of the bottom circle (e.g. 0.6 mm)
        part_mesh: The original part mesh (should be watertight)
        min_depth_mm: Minimum depth to return
        n_circle_samples: Number of sample points on the bottom circle
        containment_margin_mm: Extra depth beyond first contained point

    Returns:
        Computed depth in mm (>= *min_depth_mm*)
    """
    into_dir = np.asarray(into_part_dir, dtype=np.float64)
    into_dir = into_dir / (np.linalg.norm(into_dir) + 1e-10)
    entry = np.asarray(entry_point, dtype=np.float64)

    # Orthonormal basis perpendicular to into_part_dir
    if abs(into_dir[0]) < 0.9:
        arb = np.array([1.0, 0.0, 0.0])
    else:
        arb = np.array([0.0, 1.0, 0.0])
    u = np.cross(into_dir, arb)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(into_dir, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # Max possible depth: ray from entry → find exit from part
    try:
        locations, _, _ = part_mesh.ray.intersects_location(
            ray_origins=entry.reshape(1, 3),
            ray_directions=into_dir.reshape(1, 3)
        )
    except Exception:
        logger.warning("  Ray-cast failed for angled depth — using min depth")
        return min_depth_mm

    if len(locations) == 0:
        logger.warning("  No exit intersection from entry — using min depth")
        return min_depth_mm

    dists = np.dot(locations - entry, into_dir)
    pos_mask = dists > 0.1
    if not np.any(pos_mask):
        logger.warning("  No positive-distance exit — using min depth")
        return min_depth_mm

    max_ray_depth = float(np.min(dists[pos_mask]))
    conservative_max = max_ray_depth - bottom_radius - 0.3

    if conservative_max < min_depth_mm:
        logger.info("  Part too thin along angled dir (exit %.2fmm) — min depth",
                     max_ray_depth)
        return min_depth_mm

    # Prepare circle sample offsets
    angles = np.linspace(0, 2 * np.pi, n_circle_samples, endpoint=False)
    circle_u = bottom_radius * np.cos(angles)
    circle_v = bottom_radius * np.sin(angles)

    def _check_depth(depth: float) -> bool:
        """Check if all sample points at this depth are inside the part."""
        c = entry + into_dir * depth
        pts = np.zeros((n_circle_samples + 1, 3))
        pts[-1] = c
        for i in range(n_circle_samples):
            pts[i] = c + circle_u[i] * u + circle_v[i] * v
        try:
            return bool(np.all(part_mesh.contains(pts)))
        except Exception:
            return depth <= conservative_max

    # Linear scan from a small starting depth to find the FIRST depth
    # where the entire bottom circle is inside the part.
    scan_step = 0.25   # mm
    first_contained = None
    test_depth = 0.2   # start just inside, not right on the surface

    while test_depth <= conservative_max:
        if _check_depth(test_depth):
            first_contained = test_depth
            break
        test_depth += scan_step

    if first_contained is None:
        logger.warning("  Circle never contained along angled dir — using min depth")
        return min_depth_mm

    # Binary-search refinement between (first_contained - scan_step) and
    # first_contained to find the exact transition.
    lo = max(0.1, first_contained - scan_step)
    hi = first_contained
    for _ in range(12):
        mid = (lo + hi) / 2.0
        if _check_depth(mid):
            hi = mid   # circle is inside — try shallower
        else:
            lo = mid   # circle is outside — need deeper
    first_contained = hi

    result_depth = first_contained + containment_margin_mm
    result_depth = min(result_depth, conservative_max)

    logger.info("  Angled inlet depth: %.2fmm (first contained: %.2fmm, "
                "exit: %.2fmm)", result_depth, first_contained, max_ray_depth)
    return max(result_depth, min_depth_mm)


def _build_compound_inlet(
    cylinder_center: np.ndarray,
    drill_direction: np.ndarray,
    entry_point: np.ndarray,
    into_part_dir: np.ndarray,
    top_radius: float,
    bottom_radius: float,
    upper_height: float,
    angled_depth: float,
    segments: int = CYLINDER_SEGMENTS
) -> trimesh.Trimesh:
    """
    Build a single-piece compound inlet hole for CSG subtraction.

    Three vertex rings form one continuous, watertight mesh:

      - **Ring 0** at *cylinder_center*  (⌀ ``top_radius*2``)  — top
      - **Ring 1** at *axial junction*   (⌀ ``top_radius*2``)  — on drill axis
      - **Ring 2** at *tip*              (⌀ ``bottom_radius*2``) — bottom

    Rings 0–1 form a straight cylinder along ``drill_direction``
    (stays coaxial — no lateral drift even when entry is offset).
    Rings 1–2 form an angled frustum toward the tip inside the part.
    The axial junction is the projection of entry_point onto the
    drill axis, so the 5 mm section never tilts.

    Args:
        cylinder_center: (3,) hole start above metamold surface
        drill_direction: (3,) straight drill direction (vertical)
        entry_point: (3,) where the drill ray intersects the part surface
        into_part_dir: (3,) adaptive direction into the part (angled)
        top_radius: Radius of upper cylinder / top of lower frustum
        bottom_radius: Radius at the frustum tip
        upper_height: Distance from cylinder_center to entry_point
        angled_depth: Depth of angled section into the part
        segments: Circle resolution

    Returns:
        Single watertight trimesh (no CSG union needed).
    """
    drill_dir = np.asarray(drill_direction, dtype=np.float64)
    drill_dir = drill_dir / (np.linalg.norm(drill_dir) + 1e-10)
    into_dir = np.asarray(into_part_dir, dtype=np.float64)
    into_dir = into_dir / (np.linalg.norm(into_dir) + 1e-10)
    center = np.asarray(cylinder_center, dtype=np.float64)
    entry = np.asarray(entry_point, dtype=np.float64)
    tip = entry + into_dir * angled_depth

    # Axial junction: project entry_point onto the drill axis through
    # cylinder_center.  The upper 5 mm cylinder stays coaxial with
    # drill_direction; only the frustum below angles toward the tip.
    axial_height = float(np.dot(entry - center, drill_dir))
    axial_junction = center + drill_dir * axial_height

    angle_deg = np.degrees(np.arccos(np.clip(
        abs(np.dot(into_dir, drill_dir)), 0, 1)))

    # Orthonormal basis perpendicular to drill_direction.
    # All rings share this basis so vertices line up azimuthally
    # and the junction ring is geometrically seamless.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(drill_dir, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(drill_dir, ref)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(drill_dir, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # Three rings — ring 1 on the drill axis ⇒ 5 mm section stays axial
    ring_specs = [
        (center,          top_radius),    # Ring 0: top of straight cylinder
        (axial_junction,  top_radius),    # Ring 1: axial junction (on drill axis)
        (tip,             bottom_radius), # Ring 2: angled frustum tip
    ]

    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    vertices = []
    for rc, rr in ring_specs:
        for j in range(segments):
            vertices.append(rc + rr * (cos_a[j] * u + sin_a[j] * v))

    top_cap_idx = len(vertices)
    vertices.append(center.copy())
    bot_cap_idx = len(vertices)
    vertices.append(tip.copy())
    vertices = np.array(vertices, dtype=np.float64)

    n_rings = len(ring_specs)
    faces = []

    # Side quads between adjacent rings
    for i in range(n_rings - 1):
        for j in range(segments):
            j_next = (j + 1) % segments
            t0 = i * segments + j
            t1 = i * segments + j_next
            b0 = (i + 1) * segments + j
            b1 = (i + 1) * segments + j_next
            faces.append([t0, b0, t1])
            faces.append([t1, b0, b1])

    # Top cap (ring 0)
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([top_cap_idx, j_next, j])

    # Bottom cap (ring 2 = tip)
    last_ring = (n_rings - 1) * segments
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([bot_cap_idx, last_ring + j, last_ring + j_next])

    faces = np.array(faces, dtype=np.int64)

    frustum_length = float(np.linalg.norm(tip - axial_junction))
    logger.info("  Compound inlet (single mesh): upper ⌀%.1fmm × %.1fmm "
                "(axial) + lower ⌀%.1f→⌀%.1fmm × %.1fmm (%.1f° tilt)",
                top_radius * 2, abs(axial_height),
                top_radius * 2, bottom_radius * 2,
                frustum_length, angle_deg)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def _build_angled_plug(
    inlet_cylinder_center: np.ndarray,
    inlet_entry_point: np.ndarray,
    into_part_direction: np.ndarray,
    resin_direction: np.ndarray,
    shell_half: trimesh.Trimesh,
    inlet_top_radius: float,
    inlet_bottom_radius: float,
    wide_radius: float,
    angled_depth: float,
    taper_angle_deg: float = PLUG_TAPER_ANGLE_DEG,
    segments: int = CYLINDER_SEGMENTS
) -> Optional[trimesh.Trimesh]:
    """
    Build a single-piece resin plug matching the compound inlet hole.

    Five vertex rings form one continuous, watertight mesh:

      - **Ring 0**: *tip*             (⌀ ``inlet_bottom_radius*2``) — angled
      - **Ring 1**: *axial junction*  (⌀ ``inlet_top_radius*2``)   — on axis
      - **Ring 2**: *center level*    (⌀ ``inlet_top_radius*2``)   — axial
      - **Ring 3**: *taper end*       (⌀ ``wide_radius*2``)        — axial
      - **Ring 4**: *shell top*       (⌀ ``wide_radius*2``)        — axial

    Ring 0–1 is the angled frustum (only section that tilts).
    Rings 1–4 are the axial section along ``resin_direction``
    (coaxial with the hard-shell hole at cylinder_center).

    Args:
        inlet_cylinder_center: (3,) frustum start above metamold surface
        inlet_entry_point: (3,) where the inlet meets the part surface
        into_part_direction: (3,) adaptive direction into the part
        resin_direction: (3,) resin pouring direction (upward)
        shell_half: Hard shell mesh (for computing shell top)
        inlet_top_radius: Plug radius at metamold surface (with tolerance)
        inlet_bottom_radius: Plug radius at tip (with tolerance)
        wide_radius: Plug radius in shell section (with tolerance)
        angled_depth: Depth of angled section into the part
        taper_angle_deg: Shell-to-inlet taper half-angle
        segments: Circle resolution

    Returns:
        Plug mesh, or None on failure
    """
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    into_dir = np.asarray(into_part_direction, dtype=np.float64)
    into_dir = into_dir / (np.linalg.norm(into_dir) + 1e-10)
    center = np.asarray(inlet_cylinder_center, dtype=np.float64)
    entry = np.asarray(inlet_entry_point, dtype=np.float64)
    tip = entry + into_dir * angled_depth

    # Axial junction: project entry_point onto the resin axis through
    # cylinder_center.  The 5 mm section stays coaxial with resin_dir;
    # only the frustum below angles toward the tip.
    axial_drop = float(np.dot(entry - center, resin_dir))
    axial_junction = center + resin_dir * axial_drop

    angle_deg = np.degrees(np.arccos(np.clip(
        abs(np.dot(into_dir, resin_dir)), 0, 1)))

    # Heights along resin_dir measured from cylinder_center.
    # Rings 2–4 MUST be centred on cylinder_center’s XY position so
    # the plug aligns with the hard-shell through-hole (which is
    # drilled at cylinder_center).
    taper_angle_rad = np.radians(taper_angle_deg)
    taper_height = (wide_radius - inlet_top_radius) / np.tan(taper_angle_rad)

    shell_projs = np.dot(np.asarray(shell_half.vertices), resin_dir)
    shell_top_proj = float(np.max(shell_projs))
    center_proj = float(np.dot(center, resin_dir))
    # Extend the wide section PLUG_SHELL_OVERHANG_MM above the hard-shell top
    # so the plug protrudes and is easy to grip/push out.
    h_above_center = shell_top_proj - center_proj + PLUG_SHELL_OVERHANG_MM
    if h_above_center < taper_height + 0.5:
        h_above_center = taper_height + 0.5

    # Orthonormal basis perpendicular to resin_direction.
    # All rings share this basis so vertex correspondence is consistent
    # and the junction ring is geometrically seamless.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(resin_dir, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(resin_dir, ref)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(resin_dir, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # Five rings — ring 1 is the axial junction (on drill axis).
    # Rings 1–4 share the cylinder_center XY so the plug is
    # coaxial with the hard-shell hole.  Only ring 0→1 tilts.
    ring_specs = [
        (tip,                                           inlet_bottom_radius),  # Ring 0: tip (angled)
        (axial_junction,                                inlet_top_radius),     # Ring 1: axial junction
        (center,                                        inlet_top_radius),     # Ring 2: cylinder_center
        (center + resin_dir * taper_height,             wide_radius),          # Ring 3: taper end
        (center + resin_dir * h_above_center,           wide_radius),          # Ring 4: shell top
    ]

    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    vertices = []
    for rc, rr in ring_specs:
        for j in range(segments):
            vertices.append(rc + rr * (cos_a[j] * u + sin_a[j] * v))

    bot_cap_idx = len(vertices)
    vertices.append(tip.copy())
    top_cap_idx = len(vertices)
    vertices.append(ring_specs[-1][0].copy())  # shell top center
    vertices = np.array(vertices, dtype=np.float64)

    n_rings = len(ring_specs)
    faces = []

    # Side quads between adjacent rings
    for i in range(n_rings - 1):
        for j in range(segments):
            j_next = (j + 1) % segments
            t0 = i * segments + j
            t1 = i * segments + j_next
            b0 = (i + 1) * segments + j
            b1 = (i + 1) * segments + j_next
            faces.append([t0, b0, t1])
            faces.append([t1, b0, b1])

    # Bottom cap (ring 0 = tip)
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([bot_cap_idx, j_next, j])

    # Top cap (ring 4 = shell top)
    last_ring = (n_rings - 1) * segments
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([top_cap_idx, last_ring + j, last_ring + j_next])

    faces = np.array(faces, dtype=np.int64)

    frustum_length = float(np.linalg.norm(tip - axial_junction))
    h_axial = float(np.linalg.norm(center - axial_junction))
    logger.info("Creating resin plug (single mesh, lower tilt=%.1f°):", angle_deg)
    logger.info("  Lower frustum: ⌀%.1f→⌀%.1fmm × %.1fmm (angled)",
                inlet_bottom_radius * 2, inlet_top_radius * 2, frustum_length)
    logger.info("  Axial section: ⌀%.1fmm × %.1fmm",
                inlet_top_radius * 2, h_axial)
    logger.info("  Shell taper: %.0f° × %.1fmm", taper_angle_deg, taper_height)
    logger.info("  Wide section: ⌀%.1fmm × %.1fmm",
                wide_radius * 2, h_above_center - taper_height)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()

    logger.info("  Plug: %dv, %df, watertight=%s",
                len(mesh.vertices), len(mesh.faces), mesh.is_watertight)
    return mesh


# ============================================================================
# MAIN FUNCTION: CREATE RESIN CHANNELS
# ============================================================================

def create_resin_channels(
    metamold_half: trimesh.Trimesh,
    resin_direction: np.ndarray,
    local_maxima_positions: Optional[np.ndarray],
    global_maximum_position: Optional[np.ndarray],
    part_mesh: Optional[trimesh.Trimesh] = None,
    air_escape_depth_mm: float = DEFAULT_AIR_ESCAPE_DEPTH_MM,
    inlet_min_depth_mm: float = DEFAULT_INLET_MIN_DEPTH_MM,
    local_diameter_mm: float = DEFAULT_LOCAL_DIAMETER_MM,
    global_diameter_mm: float = DEFAULT_GLOBAL_DIAMETER_MM,
    global_bottom_diameter_mm: float = DEFAULT_GLOBAL_BOTTOM_DIAMETER_MM,
    mold_half: int = 1,
    skip_inlet_hole: bool = False,
) -> ResinChannelResult:
    """
    Create resin pouring and air escape channels in a metamold half.
    
    Subtracts shapes from the metamold half at the positions of local and
    global maxima identified during pouring direction optimization.
    
    - Global maximum: tapered frustum (truncated cone) for resin pouring
      inlet, tapering from global_diameter_mm at the metamold surface down
      to global_bottom_diameter_mm at the drilling depth. Center is offset
      toward the part's 2D convex hull centroid so the original maximum
      position lies on the top circumference. Depth is auto-computed via
      ray-casting using the bottom (smaller) radius to ensure the narrow
      end is fully cut into the part.
    - Local maxima: small cylinders for air bubble escape (centered on maxima)
    
    Args:
        metamold_half: The metamold half mesh to modify (with part already added)
        resin_direction: (3,) unit vector for resin pouring direction
        local_maxima_positions: (N, 3) positions of local maxima (air escape)
        global_maximum_position: (3,) position of global maximum (resin inlet)
        part_mesh: Original part mesh, used for 2D hull centroid offset and
            ray-casting for inlet depth. If None, no offset or auto-depth.
        air_escape_depth_mm: Depth of air escape holes in mm
        inlet_min_depth_mm: Minimum depth of resin inlet hole in mm
        local_diameter_mm: Diameter of air escape holes at local maxima
        global_diameter_mm: Top diameter of tapered resin inlet hole
        global_bottom_diameter_mm: Bottom diameter of tapered resin inlet hole
        mold_half: Which mold half this is (1=upper, 2=lower)
        skip_inlet_hole: When True, compute all inlet metadata (so the plug can
            be built) but do NOT subtract the inlet cutter from the metamold.
            Use this when the plug is merged into the metamold as an integrated
            pour spout — no socket hole is needed in that case.
        
    Returns:
        ResinChannelResult with the modified mesh and metadata
    """
    start_time = time.time()
    
    result = ResinChannelResult(
        mold_half=mold_half,
        air_escape_depth_mm=air_escape_depth_mm,
        inlet_depth_mm=inlet_min_depth_mm,  # Will be updated after auto-compute
        local_diameter_mm=local_diameter_mm,
        global_diameter_mm=global_diameter_mm,
        global_bottom_diameter_mm=global_bottom_diameter_mm
    )
    
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available - cannot create resin channels")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    if metamold_half is None or len(metamold_half.vertices) == 0:
        logger.error("Metamold half is empty or None")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    has_local = local_maxima_positions is not None and len(local_maxima_positions) > 0
    has_global = global_maximum_position is not None and len(global_maximum_position) > 0
    
    if not has_local and not has_global:
        logger.warning("No maxima positions provided - nothing to drill")
        result.mesh = metamold_half
        result.success = True
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    # The drill direction is OPPOSITE to resin pouring direction
    # Resin direction points "up" (direction silicone flows when pouring resin)
    # We drill FROM the outside DOWN into the part
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    drill_direction = -resin_dir  # Opposite to resin direction
    result.drill_direction = drill_direction.copy()
    
    logger.info(f"Creating resin channels in metamold half {mold_half}")
    logger.info(f"  Resin direction: {resin_dir}")
    logger.info(f"  Drill direction: {drill_direction}")
    logger.info(f"  Air escape depth: {air_escape_depth_mm:.2f} mm")
    logger.info(f"  Inlet min depth: {inlet_min_depth_mm:.2f} mm (auto-computed below)")
    logger.info(f"  Local maxima: {len(local_maxima_positions) if has_local else 0} "
                f"(diameter: {local_diameter_mm:.2f} mm)")
    logger.info(f"  Global maximum: {'yes' if has_global else 'no'} "
                f"(top ⌀: {global_diameter_mm:.2f} mm → bottom ⌀: {global_bottom_diameter_mm:.2f} mm)")
    
    # Collect all shapes to subtract (cylinders for local, frustum for global)
    cylinders = []
    
    # Create cylinders at local maxima (air escape holes)
    if has_local:
        local_radius = local_diameter_mm / 2.0
        for i, pos in enumerate(local_maxima_positions):
            pos = np.asarray(pos, dtype=np.float64)
            cyl = _create_cylinder(
                center=pos,
                direction=drill_direction,
                radius=local_radius,
                height=air_escape_depth_mm
            )
            cylinders.append(cyl)
            logger.debug(f"  Local channel {i}: pos={pos}, radius={local_radius:.2f}, "
                        f"depth={air_escape_depth_mm:.2f}")
        
        result.n_local_channels = len(local_maxima_positions)
        result.local_channel_positions = np.array(local_maxima_positions, dtype=np.float64)
    
    # Create compound inlet at global maximum (resin inlet):
    #   Upper: constant cylinder (⌀ global_diameter_mm) from metamold surface
    #          to part surface, along the original drill direction.
    #   Lower: tapered frustum (global_diameter_mm → global_bottom_diameter_mm)
    #          from part surface into the part interior, along an adaptive
    #          direction that angles toward the part centroid.
    if has_global:
        global_pos = np.asarray(global_maximum_position, dtype=np.float64)
        global_top_radius = global_diameter_mm / 2.0
        global_bottom_radius = global_bottom_diameter_mm / 2.0

        # Compute offset center: shift toward part's 2D hull centroid by
        # the TOP radius (the wide end sits at the surface)
        if part_mesh is not None:
            cylinder_center = _compute_inlet_offset(
                inlet_position=global_pos,
                part_mesh=part_mesh,
                drill_direction=drill_direction,
                radius=global_top_radius
            )
        else:
            logger.warning("No part mesh provided — placing global inlet centered on maximum")
            cylinder_center = global_pos.copy()

        # Move cylinder_center ABOVE the metamold outer surface.
        # _compute_inlet_offset only shifts laterally, leaving it at the
        # part surface level.  The frustum must start above the metamold
        # so the CSG subtraction creates a proper through-hole.
        resin_dir_up = -drill_direction  # points out of the mold
        metamold_projs = np.dot(
            np.asarray(metamold_half.vertices, dtype=np.float64),
            resin_dir_up)
        metamold_top_proj = float(np.max(metamold_projs))
        center_proj = float(np.dot(cylinder_center, resin_dir_up))
        if metamold_top_proj > center_proj:
            overshoot = 2.0  # mm above metamold surface for clean CSG
            lift = metamold_top_proj - center_proj + overshoot
            cylinder_center = cylinder_center + resin_dir_up * lift
            logger.info("  Lifted inlet start %.1fmm to above metamold surface",
                        lift)

        # Try adaptive angled inlet
        entry_point = None
        into_part_dir = None
        angled_depth = 0.0
        upper_height = 0.0
        approach_dir = drill_direction.copy()

        if part_mesh is not None:
            entry_point, upper_height, approach_dir = _find_part_entry_point(
                cylinder_center=cylinder_center,
                drill_direction=drill_direction,
                part_mesh=part_mesh
            )

        if entry_point is not None and part_mesh is not None:
            # Compute adaptive into-part direction
            into_part_dir = _compute_into_part_direction(
                entry_point=entry_point,
                drill_direction=drill_direction,
                part_mesh=part_mesh,
                bottom_radius=global_bottom_radius
            )

            # Compute angled depth ensuring bottom circle is inside part
            angled_depth = _compute_angled_inlet_depth(
                entry_point=entry_point,
                into_part_dir=into_part_dir,
                bottom_radius=global_bottom_radius,
                part_mesh=part_mesh,
                min_depth_mm=inlet_min_depth_mm
            )

            # Build compound inlet (upper cylinder + lower angled frustum)
            compound = _build_compound_inlet(
                cylinder_center=cylinder_center,
                drill_direction=drill_direction,
                entry_point=entry_point,
                into_part_dir=into_part_dir,
                top_radius=global_top_radius,
                bottom_radius=global_bottom_radius,
                upper_height=upper_height,
                angled_depth=angled_depth
            )
            if not skip_inlet_hole:
                cylinders.append(compound)
            else:
                logger.info(
                    "  skip_inlet_hole=True: inlet geometry computed but NOT subtracted"
                )

            result.inlet_depth_mm = upper_height + angled_depth
            result.inlet_depth_auto = upper_height + angled_depth
            result.into_part_direction = into_part_dir.copy()
            result.inlet_entry_point = entry_point.copy()
            result.inlet_angled_depth_mm = angled_depth

            angle_deg = np.degrees(np.arccos(
                np.clip(np.dot(into_part_dir, drill_direction), -1, 1)))
            logger.info("  Adaptive inlet: upper=%.2fmm, angled=%.2fmm, "
                        "deviation=%.1f\u00b0",
                         upper_height, angled_depth, angle_deg)
        else:
            # Fallback: straight frustum (no part entry found)
            logger.info("  Using straight frustum (no adaptive angle)")
            if part_mesh is not None:
                inlet_depth = _compute_inlet_depth(
                    cylinder_center=cylinder_center,
                    drill_direction=drill_direction,
                    radius=global_bottom_radius,
                    part_mesh=part_mesh,
                    min_depth_mm=inlet_min_depth_mm
                )
            else:
                inlet_depth = inlet_min_depth_mm

            result.inlet_depth_mm = inlet_depth
            result.inlet_depth_auto = inlet_depth

            frustum = _create_frustum(
                center=cylinder_center,
                direction=drill_direction,
                top_radius=global_top_radius,
                bottom_radius=global_bottom_radius,
                height=inlet_depth
            )
            if not skip_inlet_hole:
                cylinders.append(frustum)
            else:
                logger.info(
                    "  skip_inlet_hole=True: straight frustum computed but NOT subtracted"
                )

        result.n_global_channels = 1
        result.global_channel_position = global_pos.copy()
        result.inlet_cylinder_center = cylinder_center.copy()
        logger.info("  Global channel: center=%s, top_r=%.2f, bottom_r=%.2f",
                     cylinder_center, global_top_radius, global_bottom_radius)
    
    if not cylinders:
        logger.warning("No cylinders created - returning original mesh")
        result.mesh = metamold_half
        result.success = True
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    # Perform boolean subtraction: metamold_half - all_cylinders
    try:
        logger.info(f"Performing CSG subtraction: metamold half - {len(cylinders)} cylinder(s)...")
        
        # Convert metamold half to manifold
        half_manifold = _trimesh_to_manifold(metamold_half, 'metamold_half_for_channels')
        
        # Subtract each cylinder sequentially
        for i, cyl in enumerate(cylinders):
            try:
                cyl_manifold = _trimesh_to_manifold(cyl, f'resin_channel_cyl_{i}')
                half_manifold = half_manifold - cyl_manifold
                logger.debug(f"  Subtracted cylinder {i+1}/{len(cylinders)}")
            except Exception as e:
                logger.warning(f"  Failed to subtract cylinder {i+1}: {e}")
                # Continue with remaining cylinders
        
        # Convert back to trimesh
        result_mesh = _manifold_to_trimesh(half_manifold, 'metamold_minus_channels')
        
        result.mesh = result_mesh
        result.success = True
        
        elapsed_ms = (time.time() - start_time) * 1000
        result.computation_time_ms = elapsed_ms
        
        logger.info(f"Resin channels created successfully in {elapsed_ms:.1f}ms")
        logger.info(f"  Result: {len(result_mesh.vertices)} verts, {len(result_mesh.faces)} faces")
        logger.info(f"  Channels: {result.n_local_channels} local + {result.n_global_channels} global = "
                    f"{result.n_local_channels + result.n_global_channels} total")
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        result.computation_time_ms = elapsed_ms
        logger.error(f"Failed to create resin channels: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def create_resin_channels_on_both_halves(
    metamold_half_1_with_part: trimesh.Trimesh,
    metamold_half_2_with_part: trimesh.Trimesh,
    resin_direction: np.ndarray,
    local_maxima_positions: Optional[np.ndarray],
    global_maximum_position: Optional[np.ndarray],
    part_mesh: Optional[trimesh.Trimesh] = None,
    air_escape_depth_mm: float = DEFAULT_AIR_ESCAPE_DEPTH_MM,
    inlet_min_depth_mm: float = DEFAULT_INLET_MIN_DEPTH_MM,
    local_diameter_mm: float = DEFAULT_LOCAL_DIAMETER_MM,
    global_diameter_mm: float = DEFAULT_GLOBAL_DIAMETER_MM,
    global_bottom_diameter_mm: float = DEFAULT_GLOBAL_BOTTOM_DIAMETER_MM,
    skip_inlet_hole: bool = False,
) -> Tuple[ResinChannelResult, Optional[trimesh.Trimesh]]:
    """
    Create resin channels on the correct metamold half.
    
    Determines which half contains the maxima and drills channels only
    into that half. The other half is returned unmodified.
    
    Args:
        metamold_half_1_with_part: Upper metamold half (with part)
        metamold_half_2_with_part: Lower metamold half (with part)
        resin_direction: (3,) resin pouring direction
        local_maxima_positions: (N, 3) local maxima positions
        global_maximum_position: (3,) global maximum position
        part_mesh: Original part mesh for 2D hull centroid + depth ray-casting
        air_escape_depth_mm: Depth for air escape holes
        inlet_min_depth_mm: Minimum depth for resin inlet hole
        local_diameter_mm: Diameter for local maxima holes
        global_diameter_mm: Top diameter for tapered global maximum hole
        global_bottom_diameter_mm: Bottom diameter for tapered global maximum hole
        
    Returns:
        Tuple of (ResinChannelResult for modified half, unmodified other half mesh)
    """
    # Determine which positions to use for half detection
    all_positions = []
    if local_maxima_positions is not None and len(local_maxima_positions) > 0:
        all_positions.extend(local_maxima_positions.tolist())
    if global_maximum_position is not None and len(global_maximum_position) > 0:
        all_positions.append(global_maximum_position.tolist())
    
    if not all_positions:
        logger.warning("No maxima positions - cannot determine target half")
        result = ResinChannelResult(success=False)
        return result, metamold_half_2_with_part
    
    all_positions = np.array(all_positions)
    
    # Determine which half contains the maxima
    target_half = determine_maxima_half(
        metamold_half_1_with_part,
        metamold_half_2_with_part,
        all_positions,
        resin_direction
    )
    
    if target_half == 1:
        target_mesh = metamold_half_1_with_part
        other_mesh = metamold_half_2_with_part
    else:
        target_mesh = metamold_half_2_with_part
        other_mesh = metamold_half_1_with_part
    
    # Create channels in the target half
    channel_result = create_resin_channels(
        metamold_half=target_mesh,
        resin_direction=resin_direction,
        local_maxima_positions=local_maxima_positions,
        global_maximum_position=global_maximum_position,
        part_mesh=part_mesh,
        air_escape_depth_mm=air_escape_depth_mm,
        inlet_min_depth_mm=inlet_min_depth_mm,
        local_diameter_mm=local_diameter_mm,
        global_diameter_mm=global_diameter_mm,
        global_bottom_diameter_mm=global_bottom_diameter_mm,
        mold_half=target_half,
        skip_inlet_hole=skip_inlet_hole,
    )
    
    return channel_result, other_mesh


# ============================================================================
# HARD SHELL THROUGH-HOLE
# ============================================================================

def determine_shell_half_for_inlet(
    shell_half_1: trimesh.Trimesh,
    shell_half_2: trimesh.Trimesh,
    resin_direction: np.ndarray
) -> int:
    """
    Determine which hard shell half should get the resin inlet through-hole.
    
    The correct shell half is the one IN THE DIRECTION of the resin pouring
    direction. Resin is poured from above (along resin_direction), so the
    shell half whose centroid has the higher projection along the resin
    direction is the target.
    
    Args:
        shell_half_1: Hard shell half 1
        shell_half_2: Hard shell half 2
        resin_direction: (3,) unit vector for resin pouring direction
        
    Returns:
        1 if shell_half_1 is the target, 2 if shell_half_2
    """
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    
    centroid_1 = np.mean(shell_half_1.vertices, axis=0)
    centroid_2 = np.mean(shell_half_2.vertices, axis=0)
    
    proj_1 = np.dot(centroid_1, resin_dir)
    proj_2 = np.dot(centroid_2, resin_dir)
    
    target = 1 if proj_1 > proj_2 else 2
    
    logger.info(f"Shell half selection for inlet: "
                f"proj_1={proj_1:.2f}, proj_2={proj_2:.2f} → shell half {target}")
    
    return target


# ============================================================================
# SILICONE POUR HOLE HELPERS
# ============================================================================

def _find_silicone_pour_hole_position(
    shell_half_1: Optional['trimesh.Trimesh'],
    shell_half_2: Optional['trimesh.Trimesh'],
    resin_direction: np.ndarray,
    hole_radius_mm: float,
    inlet_cylinder_center: Optional[np.ndarray],
    inlet_shell_radius_mm: float,
    air_escape_positions: Optional[np.ndarray],
    air_escape_radius_mm: float,
    part_mesh: Optional['trimesh.Trimesh'] = None,
    part_hull_inset_mm: float = 5.0,
    clearance_mm: float = SILICONE_POUR_CLEARANCE_MM,
    grid_step_mm: float = SILICONE_POUR_GRID_STEP_MM,
) -> Optional[np.ndarray]:
    """
    Find the best 3-D position for the silicone mold pouring hole.

    The hole is placed within the 2-D footprint of the *part* mesh projected
    perpendicular to *resin_direction*.  Using the part footprint (rather than
    the larger shell silhouette) guarantees the hole lands over solid material
    and avoids the thin outer shell wall.

    The chosen position:
      - Lies fully inside the part\'s 2-D convex hull with an inset of
        *hole_radius_mm* + *part_hull_inset_mm* from each hull edge
      - Is >= (*clearance_mm* + sum of radii) away from the resin inlet hole
      - Is >= (*clearance_mm* + sum of radii) away from every air-escape hole
      - Maximises the distance from the part centroid so it is off-centre
        (critical for the locating-pin function)

    Falls back to the combined shell silhouette if no part mesh is provided.

    Args:
        shell_half_1:            Hard-shell upper half (used for height / fallback).
        shell_half_2:            Hard-shell lower half (used for height / fallback).
        resin_direction:         Resin pouring direction unit vector.
        hole_radius_mm:          Radius of the silicone pouring hole.
        inlet_cylinder_center:   3-D centre of the resin inlet hole (or None).
        inlet_shell_radius_mm:   Radius of the resin inlet shell through-hole.
        air_escape_positions:    (N, 3) centres of air-escape holes (or None).
        air_escape_radius_mm:    Radius of each air-escape hole.
        part_mesh:               Original part mesh — defines the valid region.
        part_hull_inset_mm:      Extra inset inside the part hull boundary beyond
                                 the hole radius (default 5 mm).
        clearance_mm:            Minimum edge-to-edge gap between holes.
        grid_step_mm:            Sampling grid resolution.

    Returns:
        3-D position of the hole centre, or None if no valid location exists.
    """
    from scipy.spatial import ConvexHull

    # ── Build orthonormal frame ──────────────────────────────────────────────
    d = np.asarray(resin_direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)

    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, ref);  u = u / np.linalg.norm(u)
    v = np.cross(d, u);    v = v / np.linalg.norm(v)

    # ── Height: mid-point of combined shell extent along resin axis ──────────
    pieces = [m for m in [shell_half_1, shell_half_2] if m is not None]
    if not pieces and part_mesh is None:
        logger.warning("No meshes provided for silicone pour hole placement")
        return None

    height_verts = np.vstack([m.vertices for m in pieces]) if pieces else np.zeros((1, 3))
    heights = height_verts @ d
    h_centre = (heights.min() + heights.max()) * 0.5

    # ── Choose bounding region: part hull (preferred) or shell hull (fallback) ─
    if part_mesh is not None:
        boundary_verts_3d = np.asarray(part_mesh.vertices, dtype=np.float64)
        logger.info("Silicone pour hole: using part mesh convex hull as bounding region")
    elif pieces:
        boundary_verts_3d = height_verts
        logger.info("Silicone pour hole: no part mesh — falling back to shell silhouette")
    else:
        return None

    coords_2d = np.column_stack([boundary_verts_3d @ u, boundary_verts_3d @ v])
    centroid_2d = coords_2d.mean(axis=0)

    try:
        hull_2d = ConvexHull(coords_2d)
    except Exception as exc:
        logger.warning("ConvexHull failed for silicone pour hole placement: %s", exc)
        return None

    # Hull half-plane equations: eqs[:, :2] @ p + eqs[:, 2] <= 0  ⟺  inside hull.
    # Required inset = hole_radius + extra part inset (0 when using shell fallback).
    eqs = hull_2d.equations
    required_inset = hole_radius_mm + (part_hull_inset_mm if part_mesh is not None else 0.0)

    # ── Forbidden zones: list of (centre_2d, hole_radius) ───────────────────
    blocked: list = []

    if inlet_cylinder_center is not None:
        ic = np.asarray(inlet_cylinder_center, dtype=np.float64)
        ic_2d = np.array([ic @ u, ic @ v])
        blocked.append((ic_2d, inlet_shell_radius_mm))

    if air_escape_positions is not None and len(air_escape_positions) > 0:
        for ae in air_escape_positions:
            ae_3d = np.asarray(ae, dtype=np.float64)
            ae_2d = np.array([ae_3d @ u, ae_3d @ v])
            blocked.append((ae_2d, air_escape_radius_mm))

    # ── Grid search over bounding box of the region hull ───────────────────
    hull_pts = coords_2d[hull_2d.vertices]
    bb_min = hull_pts.min(axis=0)
    bb_max = hull_pts.max(axis=0)

    xs = np.arange(bb_min[0] + required_inset,
                   bb_max[0] - required_inset + grid_step_mm,
                   grid_step_mm)
    ys = np.arange(bb_min[1] + required_inset,
                   bb_max[1] - required_inset + grid_step_mm,
                   grid_step_mm)

    best_pos_2d: Optional[np.ndarray] = None
    best_dist   = -1.0

    for x in xs:
        for y in ys:
            p = np.array([x, y])

            # Test 1: fully inside hull with the required inset
            margins = -(eqs[:, :2] @ p + eqs[:, 2])
            if np.any(margins < required_inset):
                continue

            # Test 2: edge-to-edge clearance from all existing holes
            clear = True
            for (bc, br) in blocked:
                if np.linalg.norm(p - bc) < hole_radius_mm + br + clearance_mm:
                    clear = False
                    break
            if not clear:
                continue

            # Score: distance from centroid (maximise for off-centre placement)
            dist = np.linalg.norm(p - centroid_2d)
            if dist > best_dist:
                best_dist   = dist
                best_pos_2d = p.copy()

    if best_pos_2d is None:
        logger.warning(
            "No valid position found for silicone pour hole "
            "(clearance %.1f mm, hole r=%.1f mm, grid %.1f mm)",
            clearance_mm, hole_radius_mm, grid_step_mm,
        )
        return None

    # Convert back to 3-D: use the mid-height on the resin axis
    pos_3d = best_pos_2d[0] * u + best_pos_2d[1] * v + h_centre * d
    offset_from_centroid_mm = best_dist
    logger.info(
        "Silicone pour hole: pos_2d=(%.2f, %.2f)  offset_from_centroid=%.1f mm",
        best_pos_2d[0], best_pos_2d[1], offset_from_centroid_mm,
    )
    return pos_3d


def _create_scaled_prism_cutter(
    silhouette_2d: np.ndarray,
    world_to_2d: np.ndarray,
    scale: float,
    min_height: float,
    max_height: float,
) -> trimesh.Trimesh:
    """
    Create a prism-shaped cutter mesh from a 2D silhouette, scaled around its centroid.

    The cutter is extruded along the pouring direction (encoded in *world_to_2d*)
    from *min_height* to *max_height*.

    Args:
        silhouette_2d: (N, 2) ordered polygon vertices of the prism base.
        world_to_2d:   (3, 3) orthonormal rotation matrix (rows = u, v, pouring_dir).
        scale:         Scale factor applied around the polygon centroid (0.5 = 50%).
        min_height:    Bottom extrusion height (in the rotated coordinate system).
        max_height:    Top extrusion height.

    Returns:
        A watertight trimesh prism suitable for CSG subtraction.
    """
    poly = np.asarray(silhouette_2d, dtype=np.float64)
    centroid_2d = poly.mean(axis=0)

    # Scale around centroid
    scaled_poly = centroid_2d + (poly - centroid_2d) * scale

    n = len(scaled_poly)
    two_d_to_world = np.asarray(world_to_2d, dtype=np.float64).T  # inverse (orthonormal)

    # Build bottom and top rings in 3D
    bottom_verts = np.zeros((n, 3), dtype=np.float64)
    top_verts = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        pt_bot = np.array([scaled_poly[i, 0], scaled_poly[i, 1], min_height])
        pt_top = np.array([scaled_poly[i, 0], scaled_poly[i, 1], max_height])
        bottom_verts[i] = pt_bot @ two_d_to_world.T
        top_verts[i] = pt_top @ two_d_to_world.T

    # Assemble vertices: 0..n-1 = bottom, n..2n-1 = top
    all_verts = np.vstack([bottom_verts, top_verts])
    faces = []

    # Side quads (two triangles each)
    for i in range(n):
        j = (i + 1) % n
        b0, b1, t0, t1 = i, j, i + n, j + n
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    # Bottom cap (fan from centre vertex)
    bot_center = bottom_verts.mean(axis=0)
    bot_center_idx = len(all_verts)
    all_verts = np.vstack([all_verts, bot_center.reshape(1, 3)])
    for i in range(n):
        j = (i + 1) % n
        faces.append([bot_center_idx, j, i])

    # Top cap (fan from centre vertex)
    top_center = top_verts.mean(axis=0)
    top_center_idx = len(all_verts)
    all_verts = np.vstack([all_verts, top_center.reshape(1, 3)])
    for i in range(n):
        j = (i + 1) % n
        faces.append([top_center_idx, i + n, j + n])

    faces_arr = np.array(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=all_verts, faces=faces_arr, process=False)
    mesh.fix_normals()
    return mesh


SILICONE_POUR_PRISM_SCALE = 0.5  # 50% of the hard-shell prism silhouette


def create_silicone_pour_holes(
    shell_half_1: Optional[trimesh.Trimesh],
    shell_half_2: Optional[trimesh.Trimesh],
    resin_direction: np.ndarray,
    inlet_cylinder_center: Optional[np.ndarray] = None,
    inlet_shell_diameter_mm: float = DEFAULT_SHELL_INLET_DIAMETER_MM,
    air_escape_positions: Optional[np.ndarray] = None,
    air_escape_diameter_mm: float = DEFAULT_LOCAL_DIAMETER_MM,
    hole_diameter_mm: float = SILICONE_POUR_HOLE_DIAMETER_MM,
    clearance_mm: float = SILICONE_POUR_CLEARANCE_MM,
    part_mesh: Optional[trimesh.Trimesh] = None,
    part_hull_inset_mm: float = 5.0,
    shell_hole_diameter_mm: float = SILICONE_POUR_SHELL_HOLE_DIAMETER_MM,
    prism_silhouette_2d: Optional[np.ndarray] = None,
    prism_world_to_2d: Optional[np.ndarray] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], Optional[np.ndarray]]:
    """
    Cut a silicone mold pouring hole through both hard shell halves.

    When *prism_silhouette_2d* and *prism_world_to_2d* are provided the hole
    is shaped like the hard-shell prism base scaled to 50 %, centred on the
    prism centroid.  This gives the silicone mold a matching key so it can
    only be re-inserted in one orientation.

    When no prism data is supplied the function falls back to the legacy
    behaviour (a cylindrical through-hole placed via grid search).

    All other shell operations (resin inlet, air-escape holes, alignment
    notches) are unaffected.

    Args:
        shell_half_1:            Hard-shell upper half (may be None).
        shell_half_2:            Hard-shell lower half (may be None).
        resin_direction:         Resin pouring direction unit vector.
        inlet_cylinder_center:   3-D centre of the resin inlet hole (or None).
        inlet_shell_diameter_mm: Diameter of the resin inlet through-hole.
        air_escape_positions:    (N, 3) positions of air-escape holes (or None).
        air_escape_diameter_mm:  Diameter of each air-escape hole.
        hole_diameter_mm:        Diameter of the locating-pin / silicone pour pin
                                 (used for clearance spacing from other holes).
        clearance_mm:            Minimum edge-to-edge gap from any other hole.
        part_mesh:               Original part mesh — constrains placement inside
                                 the part footprint.
        part_hull_inset_mm:      Extra inset inside part hull boundary (default 5 mm).
        shell_hole_diameter_mm:  Diameter of the through-hole actually drilled into
                                 the hard shells (legacy cylinder fallback only).
        prism_silhouette_2d:     (N, 2) ordered vertices of the hard-shell prism
                                 base polygon.  When provided together with
                                 *prism_world_to_2d* the prism-shaped cutter is used.
        prism_world_to_2d:       (3, 3) orthonormal rotation matrix that was used to
                                 build the hard-shell prism (rows = u, v, pouring_dir).

    Returns:
        Tuple of ``(shell_half_1_modified, shell_half_2_modified, hole_centre_3d)``.
    """
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available — cannot cut silicone pour holes")
        return shell_half_1, shell_half_2, None

    # ------------------------------------------------------------------
    # Prism-shaped cutter (preferred path)
    # ------------------------------------------------------------------
    if prism_silhouette_2d is not None and prism_world_to_2d is not None:
        silhouette = np.asarray(prism_silhouette_2d, dtype=np.float64)
        w2d = np.asarray(prism_world_to_2d, dtype=np.float64)

        # Compute centroid in 3-D (for reporting / stats)
        centroid_2d = silhouette.mean(axis=0)
        two_d_to_world = w2d.T

        resin_dir = np.asarray(resin_direction, dtype=np.float64)
        resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-12)

        # Determine height range that safely spans both shell halves
        pieces = [m for m in [shell_half_1, shell_half_2] if m is not None]
        if not pieces:
            logger.warning("No shell halves provided for silicone pour hole")
            return shell_half_1, shell_half_2, None

        all_verts = np.vstack([m.vertices for m in pieces])
        heights = all_verts @ resin_dir
        margin = 5.0  # generous margin so cutter fully penetrates
        min_h = float(np.min(heights)) - margin
        max_h = float(np.max(heights)) + margin

        # Project heights into the rotated frame that world_to_2d uses
        # (the third row of world_to_2d is the pouring direction)
        all_transformed = all_verts @ w2d.T
        min_h_local = float(np.min(all_transformed[:, 2])) - margin
        max_h_local = float(np.max(all_transformed[:, 2])) + margin

        cutter = _create_scaled_prism_cutter(
            silhouette_2d=silhouette,
            world_to_2d=w2d,
            scale=SILICONE_POUR_PRISM_SCALE,
            min_height=min_h_local,
            max_height=max_h_local,
        )

        # Centroid in 3-D for stats/reporting  (mid-height along resin axis)
        h_centre = (float(np.min(heights)) + float(np.max(heights))) * 0.5
        centroid_3d = np.array([centroid_2d[0], centroid_2d[1], 0.0]) @ two_d_to_world.T
        # centroid_3d is on the plane at height 0; shift to mean shell height
        centroid_3d = centroid_3d + h_centre * resin_dir

        logger.info(
            "Silicone pour hole: prism-shaped cutter at %.0f%% scale, "
            "centroid=(%.1f, %.1f, %.1f)",
            SILICONE_POUR_PRISM_SCALE * 100, *centroid_3d,
        )

        def _cut_prism_one_shell(
            shell: Optional[trimesh.Trimesh], label: str
        ) -> Optional[trimesh.Trimesh]:
            if shell is None:
                return None
            try:
                result = _manifold_to_trimesh(
                    _trimesh_to_manifold(shell, f'shell_for_sil_prism_{label}')
                    - _trimesh_to_manifold(cutter, f'sil_prism_cutter_{label}'),
                    f'shell_minus_sil_prism_{label}'
                )
                logger.info(
                    "Silicone prism hole cut in %s: %d verts", label, len(result.vertices),
                )
                return result
            except Exception as exc:
                logger.error("Failed to cut silicone prism hole in %s: %s", label, exc)
                return shell

        sh1_out = _cut_prism_one_shell(shell_half_1, "shell_half_1")
        sh2_out = _cut_prism_one_shell(shell_half_2, "shell_half_2")
        return sh1_out, sh2_out, centroid_3d

    # ------------------------------------------------------------------
    # Legacy cylindrical hole (fallback when no prism data available)
    # ------------------------------------------------------------------
    hole_radius = hole_diameter_mm / 2.0
    shell_hole_radius = shell_hole_diameter_mm / 2.0
    pos_3d = _find_silicone_pour_hole_position(
        shell_half_1           = shell_half_1,
        shell_half_2           = shell_half_2,
        resin_direction        = resin_direction,
        hole_radius_mm         = hole_radius,
        inlet_cylinder_center  = inlet_cylinder_center,
        inlet_shell_radius_mm  = inlet_shell_diameter_mm / 2.0,
        air_escape_positions   = air_escape_positions,
        air_escape_radius_mm   = air_escape_diameter_mm / 2.0,
        part_mesh              = part_mesh,
        part_hull_inset_mm     = part_hull_inset_mm,
        clearance_mm           = clearance_mm,
    )

    if pos_3d is None:
        logger.warning("Silicone pour hole skipped: no valid position found")
        return shell_half_1, shell_half_2, None

    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-12)
    drill_dir = -resin_dir

    def _drill_one_shell(
        shell: Optional[trimesh.Trimesh], label: str
    ) -> Optional[trimesh.Trimesh]:
        if shell is None:
            return None
        bbox_diag = np.linalg.norm(shell.bounds[1] - shell.bounds[0])
        cyl_height = bbox_diag * 2.0
        cyl_start  = pos_3d - drill_dir * (cyl_height / 2.0)

        cyl = _create_cylinder(
            center    = cyl_start,
            direction = drill_dir,
            radius    = shell_hole_radius,
            height    = cyl_height,
        )
        try:
            result = _manifold_to_trimesh(
                _trimesh_to_manifold(shell, 'shell_for_silicone_hole')
                - _trimesh_to_manifold(cyl, 'silicone_hole_cyl'),
                'shell_minus_silicone_hole'
            )
            logger.info(
                "Silicone pour hole drilled in %s: %d verts, pos=(%.1f,%.1f,%.1f)",
                label, len(result.vertices), *pos_3d,
            )
            return result
        except Exception as exc:
            logger.error("Failed to drill silicone pour hole in %s: %s", label, exc)
            return shell

    sh1_out = _drill_one_shell(shell_half_1, "shell_half_1")
    sh2_out = _drill_one_shell(shell_half_2, "shell_half_2")

    return sh1_out, sh2_out, pos_3d


def create_hard_shell_inlet(
    shell_half: trimesh.Trimesh,
    inlet_cylinder_center: np.ndarray,
    resin_direction: np.ndarray,
    inlet_diameter_mm: float = DEFAULT_SHELL_INLET_DIAMETER_MM
) -> Optional[trimesh.Trimesh]:
    """
    Drill a through-hole in a hard shell half for the resin pouring inlet.
    
    Creates a cylinder at the inlet center position, aligned with the resin
    direction, tall enough to pass completely through the shell, and subtracts
    it via CSG boolean.
    
    Args:
        shell_half: The hard shell half mesh to modify
        inlet_cylinder_center: (3,) center of the resin inlet cylinder
            (the offset center from the metamold step)
        resin_direction: (3,) resin pouring direction unit vector
        inlet_diameter_mm: Diameter of the through-hole
        
    Returns:
        Modified shell half with through-hole, or None on failure
    """
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available — cannot create shell inlet")
        return None
    
    if shell_half is None or len(shell_half.vertices) == 0:
        logger.error("Shell half is empty or None")
        return None
    
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    drill_direction = -resin_dir  # drill from top down, same as metamold
    
    center = np.asarray(inlet_cylinder_center, dtype=np.float64)
    radius = inlet_diameter_mm / 2.0
    
    # Make the cylinder tall enough to go completely through the shell.
    # Use 2x the bounding box diagonal to guarantee it passes through.
    bbox_diag = np.linalg.norm(shell_half.bounds[1] - shell_half.bounds[0])
    cylinder_height = bbox_diag * 2.0
    
    # Start the cylinder well above the shell so it goes cleanly through
    cylinder_start = center - drill_direction * (cylinder_height / 2.0)
    
    logger.info(f"Creating hard shell inlet: center={center}, "
                f"diameter={inlet_diameter_mm:.1f}mm, height={cylinder_height:.1f}mm")
    
    cyl = _create_cylinder(
        center=cylinder_start,
        direction=drill_direction,
        radius=radius,
        height=cylinder_height
    )
    
    try:
        shell_manifold = _trimesh_to_manifold(shell_half, 'shell_half_for_inlet')
        cyl_manifold = _trimesh_to_manifold(cyl, 'inlet_cyl')
        result_manifold = shell_manifold - cyl_manifold
        result_mesh = _manifold_to_trimesh(result_manifold, 'shell_minus_inlet_cyl')
        
        logger.info(f"Hard shell inlet created: {len(result_mesh.vertices)} verts, "
                    f"{len(result_mesh.faces)} faces")
        return result_mesh
        
    except Exception as e:
        logger.error(f"Failed to create hard shell inlet: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_hard_shell_air_escapes(
    shell_half: trimesh.Trimesh,
    local_channel_positions: np.ndarray,
    resin_direction: np.ndarray,
    air_escape_diameter_mm: float = DEFAULT_LOCAL_DIAMETER_MM
) -> Optional[trimesh.Trimesh]:
    """
    Drill through-holes in a hard shell half for the air escape channels.
    
    Creates cylinders at each local maxima position (matching the air escape
    holes in the metamold) and boolean-subtracts them from the shell half.
    Uses the same diameter as the air escape cylinders.
    
    Args:
        shell_half: The hard shell half mesh to modify (may already have resin inlet hole)
        local_channel_positions: (N, 3) positions of local maxima air escapes
        resin_direction: (3,) resin pouring direction unit vector
        air_escape_diameter_mm: Diameter of each through-hole (matches air escape ⌀)
        
    Returns:
        Modified shell half with air escape through-holes, or None on failure
    """
    if not MANIFOLD_AVAILABLE:
        logger.error("manifold3d not available — cannot create shell air escapes")
        return None
    
    if shell_half is None or len(shell_half.vertices) == 0:
        logger.error("Shell half is empty or None")
        return None
    
    if local_channel_positions is None or len(local_channel_positions) == 0:
        logger.info("No local channel positions — skipping air escape shell holes")
        return shell_half
    
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    drill_direction = -resin_dir
    
    radius = air_escape_diameter_mm / 2.0
    bbox_diag = np.linalg.norm(shell_half.bounds[1] - shell_half.bounds[0])
    cylinder_height = bbox_diag * 2.0
    
    logger.info(f"Drilling {len(local_channel_positions)} air escape through-holes "
                f"in hard shell: ⌀{air_escape_diameter_mm:.1f}mm")
    
    current_mesh = shell_half
    
    for i, pos in enumerate(local_channel_positions):
        pos = np.asarray(pos, dtype=np.float64)
        cylinder_start = pos - drill_direction * (cylinder_height / 2.0)
        
        cyl = _create_cylinder(
            center=cylinder_start,
            direction=drill_direction,
            radius=radius,
            height=cylinder_height
        )
        
        try:
            shell_manifold = _trimesh_to_manifold(current_mesh, f'shell_air_escape_iter_{i}')
            cyl_manifold = _trimesh_to_manifold(cyl, f'air_escape_cyl_{i}')
            result_manifold = shell_manifold - cyl_manifold
            current_mesh = _manifold_to_trimesh(result_manifold, f'shell_minus_air_escape_{i}')
            logger.debug(f"  Air escape hole {i}: pos={pos}")
        except Exception as e:
            logger.warning(f"Failed to drill air escape hole {i}: {e}")
            continue
    
    logger.info(f"Air escape shell holes complete: {len(current_mesh.vertices)} verts, "
                f"{len(current_mesh.faces)} faces")
    return current_mesh


def create_resin_plug(
    inlet_cylinder_center: np.ndarray,
    resin_direction: np.ndarray,
    part_mesh: trimesh.Trimesh,
    shell_half: trimesh.Trimesh,
    inlet_diameter_mm: float,
    shell_inlet_diameter_mm: float,
    inlet_depth_mm: float,
    inlet_bottom_diameter_mm: Optional[float] = None,
    into_part_direction: Optional[np.ndarray] = None,
    inlet_entry_point: Optional[np.ndarray] = None,
    inlet_angled_depth_mm: float = 0.0,
    tolerance_mm: float = PLUG_TOLERANCE_MM,
    taper_angle_deg: float = PLUG_TAPER_ANGLE_DEG,
    segments: int = CYLINDER_SEGMENTS
) -> Optional[trimesh.Trimesh]:
    """
    Create a resin pouring plug that fits into the inlet + shell holes.
    
    When adaptive angled geometry is provided (into_part_direction,
    inlet_entry_point, inlet_angled_depth_mm), builds a compound plug:
      1. Lower frustum (along into_part_dir): inlet_bottom_radius tip
         tapering up to inlet_top_radius at the part surface.
      2. Upper solid of revolution (along resin_dir): constant
         inlet_top_radius through the metamold, expanding to wide_radius
         through the hard shell.
    
    Otherwise falls back to a straight 4-section solid of revolution.
    
    Args:
        inlet_cylinder_center: (3,) center of the resin inlet hole
            (at the metamold surface, the TOP of the hole)
        resin_direction: (3,) resin pouring direction unit vector
        part_mesh: The original part mesh (for computing max height)
        shell_half: The hard shell half (for computing shell top)
        inlet_diameter_mm: Top diameter of the tapered resin inlet hole
        shell_inlet_diameter_mm: Hard shell through-hole diameter
        inlet_depth_mm: Depth of the inlet hole drilled into the metamold
        inlet_bottom_diameter_mm: Bottom diameter of the tapered resin inlet
            hole.  If None, defaults to DEFAULT_GLOBAL_BOTTOM_DIAMETER_MM.
        into_part_direction: (3,) adaptive angled direction for the lower
            section.  If None, uses straight plug.
        inlet_entry_point: (3,) where the inlet meets the part surface.
        inlet_angled_depth_mm: Depth of the angled lower section.
        tolerance_mm: Clearance tolerance subtracted from diameters
        taper_angle_deg: Taper half-angle for plug-to-shell transition (degrees)
        segments: Number of segments for circular cross-section
        
    Returns:
        Plug mesh as trimesh.Trimesh, or None on failure
    """
    if inlet_cylinder_center is None or part_mesh is None or shell_half is None:
        logger.error("Missing data for resin plug creation")
        return None
    
    if inlet_bottom_diameter_mm is None:
        inlet_bottom_diameter_mm = DEFAULT_GLOBAL_BOTTOM_DIAMETER_MM
    
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    center = np.asarray(inlet_cylinder_center, dtype=np.float64)
    
    # Radii with tolerance
    # Top of tapered inlet section (at metamold surface / part surface)
    inlet_top_radius = (inlet_diameter_mm - tolerance_mm) / 2.0
    # Bottom of tapered inlet section (deep in the part)
    inlet_bottom_radius = (inlet_bottom_diameter_mm - tolerance_mm) / 2.0
    # Wide shell section — always 12 mm nominal (independent of shell hole size)
    wide_radius = (PLUG_WIDE_DIAMETER_MM - tolerance_mm) / 2.0
    
    if inlet_top_radius <= 0 or inlet_bottom_radius <= 0 or wide_radius <= 0:
        logger.error(f"Invalid plug radii: top={inlet_top_radius:.2f}, "
                     f"bottom={inlet_bottom_radius:.2f}, wide={wide_radius:.2f}")
        return None
    if wide_radius <= inlet_top_radius:
        logger.error(f"Wide radius ({wide_radius:.2f}) must be > inlet top radius "
                     f"({inlet_top_radius:.2f})")
        return None
    
    # ---- Adaptive angled plug (compound geometry) ----
    has_angled = (into_part_direction is not None and
                  inlet_entry_point is not None and
                  inlet_angled_depth_mm > 0.01 and
                  MANIFOLD_AVAILABLE)

    if has_angled:
        plug = _build_angled_plug(
            inlet_cylinder_center=center,
            inlet_entry_point=np.asarray(inlet_entry_point, dtype=np.float64),
            into_part_direction=np.asarray(into_part_direction, dtype=np.float64),
            resin_direction=resin_dir,
            shell_half=shell_half,
            inlet_top_radius=inlet_top_radius,
            inlet_bottom_radius=inlet_bottom_radius,
            wide_radius=wide_radius,
            angled_depth=inlet_angled_depth_mm,
            taper_angle_deg=taper_angle_deg,
            segments=segments
        )
        if plug is not None:
            return plug
        logger.warning("Angled plug failed — falling back to straight plug")

    # ---- Straight plug fallback ----
    # The plug base is at the BOTTOM of the inlet hole.
    # inlet_cylinder_center is at the metamold surface (top of hole).
    # The hole drills downward (opposite resin_dir) by inlet_depth_mm.
    # So the base = center - resin_dir * inlet_depth_mm
    plug_base = center - resin_dir * inlet_depth_mm
    
    # Project plug base, part vertices, and shell vertices onto resin direction
    base_proj = np.dot(plug_base, resin_dir)
    part_projs = np.dot(np.asarray(part_mesh.vertices), resin_dir)
    shell_projs = np.dot(np.asarray(shell_half.vertices), resin_dir)
    
    part_max_proj = np.max(part_projs)
    shell_top_proj = np.max(shell_projs)
    
    # Section 1: tapered section from hole bottom up to part max height.
    # At the bottom: inlet_bottom_radius.  At the top: inlet_top_radius.
    # This matches the tapered frustum hole in the metamold (minus tolerance).
    tapered_height = part_max_proj - base_proj
    if tapered_height < 0.5:
        logger.warning(f"Tapered section height too small ({tapered_height:.2f}mm), "
                       "clamping to 0.5mm")
        tapered_height = 0.5
    
    # Section 2: plug-to-shell taper from inlet_top_radius to wide_radius
    # half-angle from axis: tan(angle) = (wide_r - inlet_top_r) / taper_height
    taper_angle_rad = np.radians(taper_angle_deg)
    radius_diff = wide_radius - inlet_top_radius
    plug_shell_taper_height = radius_diff / np.tan(taper_angle_rad)
    
    # Section 3: wide cylinder from plug-shell taper top to shell top + overhang
    # The extra PLUG_SHELL_OVERHANG_MM lets the plug protrude above the shell.
    taper_top_proj = base_proj + tapered_height + plug_shell_taper_height
    wide_height = shell_top_proj - taper_top_proj + PLUG_SHELL_OVERHANG_MM
    if wide_height < 0.5:
        logger.warning(f"Wide section height too small ({wide_height:.2f}mm), "
                       "clamping to 0.5mm")
        wide_height = 0.5
    
    total_height = tapered_height + plug_shell_taper_height + wide_height
    
    logger.info("Creating resin plug (tapered, base at hole bottom):")
    logger.info(f"  Tapered: ⌀{inlet_bottom_radius*2:.1f}mm → "
                f"⌀{inlet_top_radius*2:.1f}mm × {tapered_height:.1f}mm")
    logger.info(f"  Shell taper: {taper_angle_deg:.0f}° × "
                f"{plug_shell_taper_height:.1f}mm")
    logger.info(f"  Wide:   ⌀{wide_radius*2:.1f}mm × {wide_height:.1f}mm")
    logger.info(f"  Total height: {total_height:.1f}mm")
    
    # Build as solid of revolution: profile = [(height_from_base, radius), ...]
    profile = [
        (0.0, inlet_bottom_radius),                              # bottom of hole
        (tapered_height, inlet_top_radius),                      # part surface
        (tapered_height + plug_shell_taper_height, wide_radius), # shell transition
        (tapered_height + plug_shell_taper_height + wide_height, wide_radius),  # shell top
    ]
    
    n_rings = len(profile)
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    # Create ring vertices in local space (revolution around Z axis)
    vertices = []
    for h, r in profile:
        for j in range(segments):
            vertices.append([r * cos_a[j], r * sin_a[j], h])
    
    # Add cap center vertices
    bottom_center_idx = len(vertices)
    vertices.append([0.0, 0.0, profile[0][0]])
    top_center_idx = len(vertices)
    vertices.append([0.0, 0.0, profile[-1][0]])
    
    vertices = np.array(vertices, dtype=np.float64)
    
    # Build faces
    faces = []
    
    # Side quads (as triangle pairs) between adjacent rings
    for i in range(n_rings - 1):
        for j in range(segments):
            j_next = (j + 1) % segments
            v0 = i * segments + j
            v1 = i * segments + j_next
            v2 = (i + 1) * segments + j_next
            v3 = (i + 1) * segments + j
            faces.append([v0, v2, v1])
            faces.append([v0, v3, v2])
    
    # Bottom cap
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([bottom_center_idx, j, j_next])
    
    # Top cap
    last_ring = (n_rings - 1) * segments
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([top_center_idx, last_ring + j_next, last_ring + j])
    
    faces = np.array(faces, dtype=np.int64)
    
    # Transform from local Z-axis to resin direction, positioned at plug_base
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z_axis, resin_dir)
    
    if abs(dot - 1.0) < 1e-8:
        R = np.eye(3)
    elif abs(dot + 1.0) < 1e-8:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        axis = np.cross(z_axis, resin_dir)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # Position at plug_base (bottom of hole), not inlet_cylinder_center
    vertices = (R @ vertices.T).T + plug_base
    
    plug = trimesh.Trimesh(vertices=vertices, faces=faces)
    plug.fix_normals()
    
    logger.info(f"Resin plug created: {len(plug.vertices)} verts, {len(plug.faces)} faces")
    
    return plug
