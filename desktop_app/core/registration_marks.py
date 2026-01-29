"""
Registration Marks Module

Adds sinusoidal displacement patterns to the parting surface to create registration
features that help align mold halves during assembly.

From the paper (Section 5):
> "Registration features are added to the parting surface between the two silicone 
> pieces to improve alignment. Aligning surfaces with regular features is less 
> error-prone in contrast to flat or very smooth surfaces."

The sinusoidal pattern creates smooth, regular dome-shaped ridges that follow the 
hull contour, making it easy for mold halves to snap together correctly.

The pattern is applied in a band around an intermediate convex hull (at 50% distance
between the part and outer hull), not across the entire interior surface.

Author: VcMoldCreator
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import time

import trimesh

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default parameters (can be overridden by user)
DEFAULT_HULL_OFFSET_FRACTION = 0.50  # 50% between part and outer hull
DEFAULT_BAND_WIDTH_FRACTION = 0.05   # 5% of bounding box diagonal
DEFAULT_AMPLITUDE_MM = 5.0           # +/-5mm (10mm peak-to-peak)
DEFAULT_WAVELENGTH_MM = 10.0         # Distance between ridges
DEFAULT_SMOOTHING_ITERATIONS = 2     # Laplacian smoothing iterations


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SinusoidalRegistrationResult:
    """Result of applying sinusoidal registration pattern to a parting surface."""
    
    # Modified mesh
    mesh: trimesh.Trimesh
    vertices: np.ndarray           # (N, 3) modified vertex positions
    original_vertices: np.ndarray  # (N, 3) original vertex positions
    
    # Which vertices were modified
    modified_mask: np.ndarray      # (N,) bool - True if vertex was displaced
    displacement: np.ndarray       # (N,) displacement amount for each vertex
    
    # Band information
    band_center_distance: float    # Distance from part to band center
    band_width: float              # Width of the pattern band
    n_modified_vertices: int       # Number of vertices that were modified
    
    # Parameters used
    amplitude: float
    wavelength: float
    
    # Timing
    computation_time_ms: float


# Aliases for backward compatibility
RegistrationNoiseResult = SinusoidalRegistrationResult
PerlinRegistrationResult = SinusoidalRegistrationResult


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def apply_registration_noise(
    parting_surface: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    vertex_boundary_type: np.ndarray,
    hull_offset_fraction: float = DEFAULT_HULL_OFFSET_FRACTION,
    band_width_fraction: float = DEFAULT_BAND_WIDTH_FRACTION,
    noise_amplitude_mm: float = DEFAULT_AMPLITUDE_MM,
    noise_interval_mm: float = DEFAULT_WAVELENGTH_MM,
    pattern_type: str = "sinusoidal",  # Ignored, kept for API compatibility
    seed: int = 42,  # Ignored, kept for API compatibility
    smoothing_iterations: int = DEFAULT_SMOOTHING_ITERATIONS
) -> SinusoidalRegistrationResult:
    """
    Apply sinusoidal registration pattern to a parting surface.
    
    The pattern is applied only to interior vertices that fall within a band
    around an intermediate hull (at a specified fraction between part and outer hull).
    
    The sinusoidal ridges follow the hull contour and are smoothed to avoid
    jagged artifacts.
    
    Args:
        parting_surface: The smoothed parting surface mesh
        part_mesh: The original part mesh (for distance computation)
        hull_mesh: The outer hull mesh (for distance computation)
        vertex_boundary_type: (N,) array - -1=part, 0=interior, 1/2=hull
        hull_offset_fraction: Fraction of distance from part to hull for band center
        band_width_fraction: Fraction of bbox diagonal for band width
        noise_amplitude_mm: Pattern amplitude in mm (gives +/- this displacement)
        noise_interval_mm: Wavelength (distance between ridge peaks) in mm
        pattern_type: Ignored (kept for API compatibility)
        seed: Ignored (kept for API compatibility)
        smoothing_iterations: Number of Laplacian smoothing passes on pattern values
    
    Returns:
        SinusoidalRegistrationResult with modified mesh and metadata
    """
    start_time = time.time()
    
    vertices = np.array(parting_surface.vertices, dtype=np.float64)
    faces = np.array(parting_surface.faces, dtype=np.int64)
    n_verts = len(vertices)
    
    # Compute bounding box diagonal for scaling
    bbox_diag = np.linalg.norm(parting_surface.bounds[1] - parting_surface.bounds[0])
    band_width = band_width_fraction * bbox_diag
    
    logger.info(f"Applying sinusoidal registration pattern to {n_verts} vertices")
    logger.info(f"  Bbox diagonal: {bbox_diag:.2f}mm")
    logger.info(f"  Band width: {band_width:.2f}mm ({band_width_fraction*100:.1f}% of diagonal)")
    logger.info(f"  Amplitude: +/-{noise_amplitude_mm:.2f}mm")
    logger.info(f"  Wavelength: {noise_interval_mm:.2f}mm")
    
    # Step 1: Find interior vertices (vertex_boundary_type == 0)
    interior_mask = vertex_boundary_type == 0
    n_interior = np.sum(interior_mask)
    logger.info(f"  Interior vertices: {n_interior}/{n_verts}")
    
    if n_interior == 0:
        logger.warning("No interior vertices found - cannot apply pattern")
        return _create_empty_result(parting_surface, vertices, band_width, 0.0,
                                     noise_amplitude_mm, noise_interval_mm, start_time)
    
    # Step 2: Compute distance from each interior vertex to part mesh
    logger.info("  Computing distances to part mesh...")
    part_proximity = trimesh.proximity.ProximityQuery(part_mesh)
    interior_indices = np.where(interior_mask)[0]
    interior_positions = vertices[interior_indices]
    
    closest_part, dist_to_part, _ = part_proximity.on_surface(interior_positions)
    
    # Step 3: Compute distance from each interior vertex to hull mesh
    logger.info("  Computing distances to hull mesh...")
    hull_proximity = trimesh.proximity.ProximityQuery(hull_mesh)
    closest_hull, dist_to_hull, _ = hull_proximity.on_surface(interior_positions)
    
    # Step 4: Find vertices within the registration band
    intermediate_positions = closest_part + hull_offset_fraction * (closest_hull - closest_part)
    dist_to_intermediate = np.linalg.norm(interior_positions - intermediate_positions, axis=1)
    
    half_band_width = band_width / 2.0
    in_band_mask = dist_to_intermediate <= half_band_width
    
    n_in_band = np.sum(in_band_mask)
    logger.info(f"  Vertices in registration band: {n_in_band}/{n_interior}")
    
    if n_in_band == 0:
        logger.warning("No vertices found in registration band - check parameters")
        return _create_empty_result(parting_surface, vertices, band_width,
                                     hull_offset_fraction * bbox_diag,
                                     noise_amplitude_mm, noise_interval_mm, start_time)
    
    # Step 5: Compute vertex normals for displacement direction
    logger.info("  Computing vertex normals...")
    vertex_normals = _compute_vertex_normals(vertices, faces)
    
    # Step 6: Apply sinusoidal pattern to vertices in band
    logger.info("  Computing sinusoidal pattern...")
    
    band_interior_indices = interior_indices[in_band_mask]
    
    # Create modified mask for all vertices
    modified_mask = np.zeros(n_verts, dtype=bool)
    modified_mask[band_interior_indices] = True
    
    # Initialize displacement array
    displacement = np.zeros(n_verts, dtype=np.float64)
    modified_vertices = vertices.copy()
    
    # Compute normalized falloff within band (1.0 at center, 0.0 at edges)
    # Normalize by LOCAL max distance to ensure uniform amplitude across the part
    dist_in_band = dist_to_intermediate[in_band_mask]
    
    # Use actual max distance in band for normalization (not global half_band_width)
    # This ensures consistent falloff regardless of part geometry
    max_dist_in_band = np.max(dist_in_band) if len(dist_in_band) > 0 else half_band_width
    # Ensure we don't divide by zero and have reasonable falloff
    effective_half_width = max(max_dist_in_band, 1e-6)
    
    # Falloff: 1.0 at dist=0, smooth cosine falloff toward edges
    # Using normalized distance ensures uniform amplitude distribution
    normalized_dist = dist_in_band / effective_half_width
    falloff = 0.5 * (1.0 + np.cos(np.pi * np.clip(normalized_dist, 0.0, 1.0)))
    falloff = np.clip(falloff, 0.0, 1.0)
    
    # Get positions and normals for band vertices
    band_positions = vertices[band_interior_indices]
    band_normals = vertex_normals[band_interior_indices]
    
    # Get the closest hull points for vertices in band (for contour-following)
    band_closest_hull = closest_hull[in_band_mask]
    
    # Compute dome-shaped pattern with pre-smoothed parameterization
    # This creates inherently smooth domes by smoothing the arc-length before computing pattern
    pattern_values = _compute_sinusoidal_pattern(
        band_positions, 
        noise_interval_mm, 
        band_closest_hull,
        band_indices=band_interior_indices,
        faces=faces,
        smoothing_iterations=smoothing_iterations
    )
    
    # Apply displacement: pattern_value * amplitude * falloff (FULL amplitude)
    band_displacements = pattern_values * noise_amplitude_mm * falloff
    
    # Store displacements
    displacement[band_interior_indices] = band_displacements
    
    # Displace vertices along normals
    modified_vertices[band_interior_indices] = (
        band_positions + band_displacements[:, np.newaxis] * band_normals
    )
    
    # Step 7: Create modified mesh
    modified_mesh = trimesh.Trimesh(vertices=modified_vertices, faces=faces)
    
    # Compute statistics
    non_zero_disp = displacement[modified_mask]
    logger.info(f"  Applied sinusoidal pattern to {n_in_band} vertices")
    logger.info(f"  Displacement range: [{non_zero_disp.min():.3f}, {non_zero_disp.max():.3f}] mm")
    logger.info(f"  Displacement mean: {np.mean(np.abs(non_zero_disp)):.3f} mm")
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"  Completed in {elapsed_ms:.1f}ms")
    
    return SinusoidalRegistrationResult(
        mesh=modified_mesh,
        vertices=modified_vertices,
        original_vertices=vertices,
        modified_mask=modified_mask,
        displacement=displacement,
        band_center_distance=hull_offset_fraction * bbox_diag,
        band_width=band_width,
        n_modified_vertices=n_in_band,
        amplitude=noise_amplitude_mm,
        wavelength=noise_interval_mm,
        computation_time_ms=elapsed_ms
    )


def _create_empty_result(
    parting_surface: trimesh.Trimesh,
    vertices: np.ndarray,
    band_width: float,
    band_center_distance: float,
    amplitude: float,
    wavelength: float,
    start_time: float
) -> SinusoidalRegistrationResult:
    """Create an empty result when no vertices are modified."""
    n_verts = len(vertices)
    return SinusoidalRegistrationResult(
        mesh=parting_surface,
        vertices=vertices,
        original_vertices=vertices.copy(),
        modified_mask=np.zeros(n_verts, dtype=bool),
        displacement=np.zeros(n_verts, dtype=np.float64),
        band_center_distance=band_center_distance,
        band_width=band_width,
        n_modified_vertices=0,
        amplitude=amplitude,
        wavelength=wavelength,
        computation_time_ms=(time.time() - start_time) * 1000
    )


# ============================================================================
# PATTERN GENERATION FUNCTIONS
# ============================================================================

def _compute_sinusoidal_pattern(
    positions: np.ndarray,
    interval_mm: float,
    closest_hull: np.ndarray = None,
    band_indices: np.ndarray = None,
    faces: np.ndarray = None,
    smoothing_iterations: int = 2
) -> np.ndarray:
    """
    Compute dome-shaped pattern creating smooth ridges that follow the hull contour.
    
    Creates clean dome shapes by:
    1. Computing arc-length parameterization around the hull
    2. Smoothing the arc-length values on the mesh to remove discretization noise
    3. Computing a raised-cosine dome profile for smooth peaks and valleys
    
    Args:
        positions: (N, 3) vertex positions
        interval_mm: Distance between ridge peaks in mm
        closest_hull: (N, 3) closest points on hull surface for each vertex
        band_indices: (N,) global indices of band vertices (for smoothing)
        faces: (F, 3) mesh faces (for smoothing)
        smoothing_iterations: Iterations to smooth the parameterization
    
    Returns:
        (N,) pattern values in range [-1, 1]
    """
    if closest_hull is None:
        # Fallback to simple position-based pattern
        centered = positions - np.mean(positions, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        primary_axis = eigenvectors[:, 2]
        projection = np.dot(centered, primary_axis)
        frequency = 2.0 * np.pi / interval_mm
        return np.cos(frequency * projection)  # cos for dome at center
    
    # Compute centroid of hull points
    centroid = np.mean(closest_hull, axis=0)
    hull_centered = closest_hull - centroid
    
    # Use PCA to find the plane of the hull points
    cov = np.cov(hull_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Two axes in the plane (largest eigenvalues)
    axis1 = eigenvectors[:, 2]  # Primary direction in plane
    axis2 = eigenvectors[:, 1]  # Secondary direction in plane
    
    # Project hull points onto the 2D plane
    proj_x = np.dot(hull_centered, axis1)
    proj_y = np.dot(hull_centered, axis2)
    
    # Compute angular position around centroid
    angles = np.arctan2(proj_y, proj_x)  # Range [-pi, pi]
    
    # Compute approximate radius for arc-length conversion
    radii = np.sqrt(proj_x**2 + proj_y**2)
    mean_radius = np.mean(radii) if np.mean(radii) > 1e-6 else 1.0
    
    # Total circumference of the pattern band
    total_circumference = 2.0 * np.pi * mean_radius
    
    # Adjust frequency to ensure an INTEGER number of waves around the circumference
    # This prevents start/end point overlap or gaps
    raw_num_waves = total_circumference / interval_mm
    num_waves = max(1, round(raw_num_waves))  # At least 1 wave, rounded to nearest integer
    
    # Adjusted interval that fits exactly
    adjusted_interval = total_circumference / num_waves
    
    logger.debug(f"  Circumference: {total_circumference:.2f}mm, "
                 f"requested interval: {interval_mm:.2f}mm, "
                 f"waves: {num_waves}, adjusted interval: {adjusted_interval:.2f}mm")
    
    # Convert angle to arc-length: shift angles from [-pi, pi] to [0, 2*pi]
    # This gives continuous arc-length from 0 to circumference
    angles_shifted = angles + np.pi  # Now in range [0, 2*pi]
    arc_length = mean_radius * angles_shifted
    
    # Compute dome pattern using sine with adjusted frequency
    # Using sin() places zero-crossings at the seam (arc_length=0 and arc_length=circumference)
    # This avoids double-peaks or artifacts at the start/end junction
    frequency = 2.0 * np.pi * num_waves / total_circumference
    pattern = np.sin(frequency * arc_length)
    
    # Smooth the PATTERN values (not arc-length) to avoid seam discontinuity issues
    # Arc-length has a discontinuity at 0/circumference, but sin() values are continuous
    # because sin(0) = sin(2╧Çn) = 0 at the seam
    if band_indices is not None and faces is not None and smoothing_iterations > 0:
        pattern = _smooth_scalar_field(
            pattern, band_indices, faces, smoothing_iterations
        )
    
    return pattern


def _smooth_scalar_field(
    values: np.ndarray,
    band_indices: np.ndarray,
    faces: np.ndarray,
    iterations: int
) -> np.ndarray:
    """
    Smooth a scalar field on the mesh using Laplacian smoothing.
    
    This is used to smooth the arc-length parameterization before computing
    the pattern, resulting in smoother dome shapes.
    
    Args:
        values: (K,) scalar values for band vertices
        band_indices: (K,) global indices of band vertices
        faces: (F, 3) mesh faces
        iterations: Number of smoothing iterations
    
    Returns:
        (K,) smoothed scalar values
    """
    if len(values) == 0 or iterations <= 0:
        return values
    
    # Build adjacency for band vertices
    band_set = set(band_indices.tolist())
    global_to_band = {g: b for b, g in enumerate(band_indices)}
    
    n_band = len(band_indices)
    neighbors = [[] for _ in range(n_band)]
    
    for face in faces:
        band_verts_in_face = []
        for v in face:
            if v in band_set:
                band_verts_in_face.append(global_to_band[v])
        
        for i in range(len(band_verts_in_face)):
            for j in range(i + 1, len(band_verts_in_face)):
                bi, bj = band_verts_in_face[i], band_verts_in_face[j]
                if bj not in neighbors[bi]:
                    neighbors[bi].append(bj)
                if bi not in neighbors[bj]:
                    neighbors[bj].append(bi)
    
    # Laplacian smoothing
    smoothed = values.copy()
    damping = 0.5
    
    for _ in range(iterations):
        new_values = smoothed.copy()
        for i in range(n_band):
            if len(neighbors[i]) > 0:
                neighbor_avg = np.mean([smoothed[n] for n in neighbors[i]])
                new_values[i] = (1.0 - damping) * smoothed[i] + damping * neighbor_avg
        smoothed = new_values
    
    return smoothed


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute smooth vertex normals by averaging face normals.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face vertex indices
    
    Returns:
        (N, 3) normalized vertex normals
    """
    n_verts = len(vertices)
    vertex_normals = np.zeros((n_verts, 3), dtype=np.float64)
    
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    
    # Accumulate face normals to vertices
    for fi, face in enumerate(faces):
        fn = face_normals[fi]
        vertex_normals[face[0]] += fn
        vertex_normals[face[1]] += fn
        vertex_normals[face[2]] += fn
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    vertex_normals = vertex_normals / norms
    
    return vertex_normals


def get_band_visualization_data(
    parting_surface: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    vertex_boundary_type: np.ndarray,
    hull_offset_fraction: float = DEFAULT_HULL_OFFSET_FRACTION,
    band_width_fraction: float = DEFAULT_BAND_WIDTH_FRACTION
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get visualization data for the registration band (for preview).
    
    Returns:
        Tuple of:
            - band_positions: (K, 3) positions of vertices in band
            - band_indices: (K,) indices of vertices in band
            - intermediate_hull_positions: (K, 3) positions on intermediate hull
    """
    vertices = np.array(parting_surface.vertices, dtype=np.float64)
    bbox_diag = np.linalg.norm(parting_surface.bounds[1] - parting_surface.bounds[0])
    band_width = band_width_fraction * bbox_diag
    
    # Find interior vertices
    interior_mask = vertex_boundary_type == 0
    interior_indices = np.where(interior_mask)[0]
    
    if len(interior_indices) == 0:
        return np.array([]), np.array([]), np.array([])
    
    interior_positions = vertices[interior_indices]
    
    # Compute distances
    part_proximity = trimesh.proximity.ProximityQuery(part_mesh)
    hull_proximity = trimesh.proximity.ProximityQuery(hull_mesh)
    
    closest_part, _, _ = part_proximity.on_surface(interior_positions)
    closest_hull, _, _ = hull_proximity.on_surface(interior_positions)
    
    # Compute intermediate positions
    intermediate_positions = closest_part + hull_offset_fraction * (closest_hull - closest_part)
    dist_to_intermediate = np.linalg.norm(interior_positions - intermediate_positions, axis=1)
    
    # Find vertices in band
    half_band_width = band_width / 2.0
    in_band_mask = dist_to_intermediate <= half_band_width
    
    band_indices = interior_indices[in_band_mask]
    band_positions = vertices[band_indices]
    band_intermediate = intermediate_positions[in_band_mask]
    
    return band_positions, band_indices, band_intermediate
