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

# CSG operations via abstraction layer (libigl/CGAL preferred, manifold3d fallback)
from . import csg_backend
CSG_AVAILABLE = csg_backend.CSG_AVAILABLE

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_AIR_ESCAPE_DEPTH_MM = 1.5     # Default depth of air escape holes
DEFAULT_INLET_MIN_DEPTH_MM = 1.5     # Minimum depth of resin inlet hole
DEFAULT_LOCAL_DIAMETER_MM = 1.2       # Diameter for air escape holes (local maxima)
DEFAULT_GLOBAL_DIAMETER_MM = 5.2      # Diameter for resin inlet hole (global maximum)
DEFAULT_SHELL_INLET_DIAMETER_MM = 12.2  # Diameter for hard shell through-hole
PLUG_TOLERANCE_MM = 0.2                   # Tolerance clearance for plug dimensions
PLUG_TAPER_ANGLE_DEG = 20.0               # Taper half-angle in degrees
CYLINDER_SEGMENTS = 32                # Number of segments for cylinder approximation
INLET_DEPTH_SAMPLE_POINTS = 24        # Points around circumference for depth ray-casting


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


# ============================================================================
# CSG HELPERS (now handled by csg_backend)
# ============================================================================

# NOTE: _trimesh_to_manifold / _manifold_to_trimesh removed.
# All CSG ops go through csg_backend.csg_difference(), etc.


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
    circular cross-section is cut into the part, even on inclined surfaces.

    Samples points around the cylinder circumference and ray-casts each one
    along the drill direction onto the part surface. The depth variation
    across the circumference (max - min intersection depth) tells us how
    much extra depth is needed to clear the incline.

    Final depth = depth_variation + min_depth_mm

    Args:
        cylinder_center: (3,) the cylinder center position  
        drill_direction: (3,) unit vector for drilling axis
        radius: Radius of the inlet cylinder
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
    mold_half: int = 1
) -> ResinChannelResult:
    """
    Create resin pouring and air escape channels in a metamold half.
    
    Subtracts cylinders from the metamold half at the positions of local and
    global maxima identified during pouring direction optimization. The cylinders
    are drilled in the OPPOSITE direction to the resin pouring direction
    (from outside surface into the cavity).
    
    - Global maximum: large cylinder for resin pouring inlet, with center
      offset toward the part's 2D convex hull centroid so the original
      maximum position lies on the cylinder's circumference. Depth is
      auto-computed via ray-casting to ensure a full circular cutout
      even on inclined surfaces.
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
        global_diameter_mm: Diameter of resin inlet hole at global maximum
        mold_half: Which mold half this is (1=upper, 2=lower)
        
    Returns:
        ResinChannelResult with the modified mesh and metadata
    """
    start_time = time.time()
    
    result = ResinChannelResult(
        mold_half=mold_half,
        air_escape_depth_mm=air_escape_depth_mm,
        inlet_depth_mm=inlet_min_depth_mm,  # Will be updated after auto-compute
        local_diameter_mm=local_diameter_mm,
        global_diameter_mm=global_diameter_mm
    )
    
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available - cannot create resin channels")
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
                f"(diameter: {global_diameter_mm:.2f} mm)")
    
    # Collect all cylinders to subtract
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
    
    # Create cylinder at global maximum (resin inlet)
    # The cylinder center is offset toward the part's 2D convex hull centroid
    # so that the original global max position sits on the circumference.
    # Depth is auto-computed via ray-casting to ensure full circular cutout.
    if has_global:
        global_pos = np.asarray(global_maximum_position, dtype=np.float64)
        global_radius = global_diameter_mm / 2.0

        # Compute offset center: shift toward part's 2D hull centroid by radius
        if part_mesh is not None:
            cylinder_center = _compute_inlet_offset(
                inlet_position=global_pos,
                part_mesh=part_mesh,
                drill_direction=drill_direction,
                radius=global_radius
            )
        else:
            logger.warning("No part mesh provided — placing global inlet centered on maximum")
            cylinder_center = global_pos.copy()

        # Auto-compute inlet depth: ray-cast from circumference to find
        # the surface depth variation due to incline, then add min_depth
        if part_mesh is not None:
            inlet_depth = _compute_inlet_depth(
                cylinder_center=cylinder_center,
                drill_direction=drill_direction,
                radius=global_radius,
                part_mesh=part_mesh,
                min_depth_mm=inlet_min_depth_mm
            )
        else:
            inlet_depth = inlet_min_depth_mm

        result.inlet_depth_mm = inlet_depth
        result.inlet_depth_auto = inlet_depth

        cyl = _create_cylinder(
            center=cylinder_center,
            direction=drill_direction,
            radius=global_radius,
            height=inlet_depth
        )
        cylinders.append(cyl)
        result.n_global_channels = 1
        result.global_channel_position = global_pos.copy()
        result.inlet_cylinder_center = cylinder_center.copy()
        logger.info(f"  Global channel: original_pos={global_pos}, "
                    f"cylinder_center={cylinder_center}, radius={global_radius:.2f}, "
                    f"depth={inlet_depth:.2f}mm (min={inlet_min_depth_mm:.2f}mm)")
    
    if not cylinders:
        logger.warning("No cylinders created - returning original mesh")
        result.mesh = metamold_half
        result.success = True
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    # Perform boolean subtraction: metamold_half - all_cylinders
    try:
        logger.info(f"Performing CSG subtraction: metamold half - {len(cylinders)} cylinder(s)...")
        
        # Subtract each cylinder sequentially via csg_backend
        current_mesh = metamold_half
        for i, cyl in enumerate(cylinders):
            try:
                current_mesh = csg_backend.csg_difference(current_mesh, cyl)
                logger.debug(f"  Subtracted cylinder {i+1}/{len(cylinders)}")
            except Exception as e:
                logger.warning(f"  Failed to subtract cylinder {i+1}: {e}")
                # Continue with remaining cylinders
        
        result_mesh = current_mesh
        
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
    global_diameter_mm: float = DEFAULT_GLOBAL_DIAMETER_MM
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
        global_diameter_mm: Diameter for global maximum hole
        
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
        mold_half=target_half
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
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available — cannot create shell inlet")
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
        result_mesh = csg_backend.csg_difference(shell_half, cyl)
        
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
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available — cannot create shell air escapes")
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
            current_mesh = csg_backend.csg_difference(current_mesh, cyl)
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
    tolerance_mm: float = PLUG_TOLERANCE_MM,
    taper_angle_deg: float = PLUG_TAPER_ANGLE_DEG,
    segments: int = CYLINDER_SEGMENTS
) -> Optional[trimesh.Trimesh]:
    """
    Create a resin pouring plug that fits into the inlet + shell through-holes.
    
    The plug is a 3-section solid of revolution:
      1. Narrow cylinder (bottom): ⌀ = inlet_diameter - tolerance.
         Starts at the BASE of the inlet hole (where it touches the part,
         i.e. inlet_cylinder_center offset downward by inlet_depth) and
         extends upward to the part's max height along resin direction.
      2. Taper (middle): expands from narrow ⌀ to shell ⌀ - tolerance
         at the specified taper half-angle.
      3. Wide cylinder (top): ⌀ = shell_inlet_diameter - tolerance,
         extends from the taper top to the top of the hard shell.
    
    Args:
        inlet_cylinder_center: (3,) center of the resin inlet cylinder
            (at the metamold surface, the TOP of the hole)
        resin_direction: (3,) resin pouring direction unit vector
        part_mesh: The original part mesh (for computing max height)
        shell_half: The hard shell half (for computing shell top)
        inlet_diameter_mm: Resin inlet hole diameter
        shell_inlet_diameter_mm: Hard shell through-hole diameter
        inlet_depth_mm: Depth of the inlet hole drilled into the metamold
        tolerance_mm: Clearance tolerance subtracted from diameters
        taper_angle_deg: Taper half-angle in degrees (from the vertical axis)
        segments: Number of segments for circular cross-section
        
    Returns:
        Plug mesh as trimesh.Trimesh, or None on failure
    """
    if inlet_cylinder_center is None or part_mesh is None or shell_half is None:
        logger.error("Missing data for resin plug creation")
        return None
    
    resin_dir = np.asarray(resin_direction, dtype=np.float64)
    resin_dir = resin_dir / (np.linalg.norm(resin_dir) + 1e-10)
    center = np.asarray(inlet_cylinder_center, dtype=np.float64)
    
    # Radii with tolerance
    narrow_radius = (inlet_diameter_mm - tolerance_mm) / 2.0
    wide_radius = (shell_inlet_diameter_mm - tolerance_mm) / 2.0
    
    if narrow_radius <= 0 or wide_radius <= 0 or wide_radius <= narrow_radius:
        logger.error(f"Invalid plug radii: narrow={narrow_radius:.2f}, wide={wide_radius:.2f}")
        return None
    
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
    
    # Section 1: narrow cylinder from hole bottom up to part max height
    narrow_height = part_max_proj - base_proj
    if narrow_height < 0.5:
        logger.warning(f"Narrow section height too small ({narrow_height:.2f}mm), clamping to 0.5mm")
        narrow_height = 0.5
    
    # Section 2: taper from narrow_radius to wide_radius at specified angle
    # half-angle from axis: tan(angle) = (wide_r - narrow_r) / taper_height
    # taper_height = (wide_r - narrow_r) / tan(angle)
    taper_angle_rad = np.radians(taper_angle_deg)
    radius_diff = wide_radius - narrow_radius
    taper_height = radius_diff / np.tan(taper_angle_rad)
    
    # Section 3: wide cylinder from taper top to shell top
    taper_top_proj = base_proj + narrow_height + taper_height
    wide_height = shell_top_proj - taper_top_proj
    if wide_height < 0.5:
        logger.warning(f"Wide section height too small ({wide_height:.2f}mm), clamping to 0.5mm")
        wide_height = 0.5
    
    total_height = narrow_height + taper_height + wide_height
    
    logger.info(f"Creating resin plug (base at hole bottom):")
    logger.info(f"  Narrow: ⌀{narrow_radius*2:.1f}mm × {narrow_height:.1f}mm")
    logger.info(f"  Taper:  {taper_angle_deg:.0f}° × {taper_height:.1f}mm")
    logger.info(f"  Wide:   ⌀{wide_radius*2:.1f}mm × {wide_height:.1f}mm")
    logger.info(f"  Total height: {total_height:.1f}mm")
    
    # Build as solid of revolution: profile = [(height_from_base, radius), ...]
    profile = [
        (0.0, narrow_radius),
        (narrow_height, narrow_radius),
        (narrow_height + taper_height, wide_radius),
        (narrow_height + taper_height + wide_height, wide_radius),
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
