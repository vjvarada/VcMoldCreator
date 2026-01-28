"""
Sinusoidal Registration Module

Adds sinusoidal displacement patterns to the parting surface to create registration
features that help align mold halves during assembly.

From the paper (Section 5):
> "Registration features are added to the parting surface between the two silicone 
> pieces to improve alignment. Aligning surfaces with regular features is less 
> error-prone in contrast to flat or very smooth surfaces."

The sinusoidal pattern creates smooth, regular ridges that follow the hull contour,
making it easy for mold halves to snap together correctly.

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
    smoothing_iterations: int
    
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
    logger.info(f"  Smoothing iterations: {smoothing_iterations}")
    
    # Step 1: Find interior vertices (vertex_boundary_type == 0)
    interior_mask = vertex_boundary_type == 0
    n_interior = np.sum(interior_mask)
    logger.info(f"  Interior vertices: {n_interior}/{n_verts}")
    
    if n_interior == 0:
        logger.warning("No interior vertices found - cannot apply pattern")
        return _create_empty_result(parting_surface, vertices, band_width, 0.0,
                                     noise_amplitude_mm, noise_interval_mm, smoothing_iterations, start_time)
    
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
                                     noise_amplitude_mm, noise_interval_mm, smoothing_iterations, start_time)
    
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
    
    # Compute raw sinusoidal pattern
    pattern_values = _compute_sinusoidal_pattern(band_positions, noise_interval_mm, band_closest_hull)
    
    # Apply displacement: pattern_value * amplitude * falloff (FULL amplitude)
    band_displacements = pattern_values * noise_amplitude_mm * falloff
    
    # Store displacements
    displacement[band_interior_indices] = band_displacements
    
    # Displace vertices along normals
    modified_vertices[band_interior_indices] = (
        band_positions + band_displacements[:, np.newaxis] * band_normals
    )
    
    # Step 7: Smooth the displaced mesh vertices to reduce jaggedness
    # This preserves the sinusoidal amplitude while smoothing mesh artifacts
    # Peak-preserving: local extrema (peaks/valleys) are kept fixed
    if smoothing_iterations > 0:
        logger.info(f"  Smoothing displaced vertices ({smoothing_iterations} iterations)...")
        modified_vertices = _smooth_displaced_vertices(
            modified_vertices,
            band_interior_indices,
            faces,
            smoothing_iterations,
            displacement  # Pass displacement to identify peaks/valleys
        )
    
    # Step 8: Create modified mesh
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
        smoothing_iterations=smoothing_iterations,
        computation_time_ms=elapsed_ms
    )


def _create_empty_result(
    parting_surface: trimesh.Trimesh,
    vertices: np.ndarray,
    band_width: float,
    band_center_distance: float,
    amplitude: float,
    wavelength: float,
    smoothing_iterations: int,
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
        smoothing_iterations=smoothing_iterations,
        computation_time_ms=(time.time() - start_time) * 1000
    )


# ============================================================================
# PATTERN GENERATION FUNCTIONS
# ============================================================================

def _compute_sinusoidal_pattern(
    positions: np.ndarray,
    interval_mm: float,
    closest_hull: np.ndarray = None
) -> np.ndarray:
    """
    Compute sinusoidal pattern creating ridges that follow the hull contour.
    
    Uses arc-length along the hull boundary to create waves that naturally
    follow the curved contour. This is computed by:
    1. Finding the plane of the hull points (using PCA to get normal)
    2. Computing angular position around the centroid in that plane
    3. Converting angle to arc-length and applying sinusoid
    
    Args:
        positions: (N, 3) vertex positions
        interval_mm: Distance between ridge peaks in mm
        closest_hull: (N, 3) closest points on hull surface for each vertex
    
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
        return np.sin(frequency * projection)
    
    # Compute centroid of hull points
    centroid = np.mean(closest_hull, axis=0)
    hull_centered = closest_hull - centroid
    
    # Use PCA to find the plane of the hull points
    # The smallest eigenvector is the normal to the plane
    cov = np.cov(hull_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Normal to the hull point cloud plane (smallest eigenvalue)
    plane_normal = eigenvectors[:, 0]
    # Two axes in the plane (largest eigenvalues)
    axis1 = eigenvectors[:, 2]  # Primary direction in plane
    axis2 = eigenvectors[:, 1]  # Secondary direction in plane
    
    # Project hull points onto the 2D plane
    proj_x = np.dot(hull_centered, axis1)
    proj_y = np.dot(hull_centered, axis2)
    
    # Compute angular position around centroid (arc-length parameterization)
    angles = np.arctan2(proj_y, proj_x)  # Range [-pi, pi]
    
    # Compute approximate radius for arc-length conversion
    radii = np.sqrt(proj_x**2 + proj_y**2)
    mean_radius = np.mean(radii) if np.mean(radii) > 1e-6 else 1.0
    
    # Convert angle to arc-length: s = r * theta
    arc_length = mean_radius * angles
    
    # Compute sinusoidal pattern based on arc-length
    frequency = 2.0 * np.pi / interval_mm
    pattern = np.sin(frequency * arc_length)
    
    return pattern


def _smooth_displaced_vertices(
    vertices: np.ndarray,
    band_indices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 2,
    displacement: np.ndarray = None
) -> np.ndarray:
    """
    Smooth the displaced vertex positions using peak-preserving Laplacian smoothing.
    
    This reduces jagged mesh artifacts by averaging each vertex's position
    with its neighbors, while keeping local extrema (peaks and valleys) fixed
    to preserve the full sinusoidal amplitude.
    
    Args:
        vertices: (N, 3) all vertex positions (will be copied)
        band_indices: (K,) indices of band vertices to smooth
        faces: (F, 3) face indices of the mesh
        iterations: Number of smoothing iterations
        displacement: (N,) displacement values used to identify extrema
    
    Returns:
        (N, 3) smoothed vertex positions
    """
    if len(band_indices) == 0 or iterations <= 0:
        return vertices
    
    smoothed_verts = vertices.copy()
    
    # Build adjacency for band vertices only
    # Create a mapping from global index to band index
    band_set = set(band_indices.tolist())
    global_to_band = {g: b for b, g in enumerate(band_indices)}
    
    # Build neighbor list for band vertices
    n_band = len(band_indices)
    neighbors = [[] for _ in range(n_band)]
    
    for face in faces:
        # Check which vertices of this face are in the band
        band_verts_in_face = []
        for v in face:
            if v in band_set:
                band_verts_in_face.append(global_to_band[v])
        
        # Add edges between band vertices in this face
        for i in range(len(band_verts_in_face)):
            for j in range(i + 1, len(band_verts_in_face)):
                bi, bj = band_verts_in_face[i], band_verts_in_face[j]
                if bj not in neighbors[bi]:
                    neighbors[bi].append(bj)
                if bi not in neighbors[bj]:
                    neighbors[bj].append(bi)
    
    # Identify local extrema (peaks and valleys) based on displacement values
    # These vertices will be kept fixed during smoothing to preserve amplitude
    is_extremum = np.zeros(n_band, dtype=bool)
    
    if displacement is not None:
        band_displacements = displacement[band_indices]
        
        for i in range(n_band):
            if len(neighbors[i]) > 0:
                my_disp = band_displacements[i]
                neighbor_disps = [band_displacements[n] for n in neighbors[i]]
                
                # Check if this vertex is a local maximum (peak)
                is_local_max = all(my_disp >= nd for nd in neighbor_disps)
                # Check if this vertex is a local minimum (valley)
                is_local_min = all(my_disp <= nd for nd in neighbor_disps)
                
                is_extremum[i] = is_local_max or is_local_min
    
    n_extrema = np.sum(is_extremum)
    logger.debug(f"  Found {n_extrema} local extrema (peaks/valleys) to preserve")
    
    # Laplacian smoothing iterations on 3D positions
    # Skip extrema to preserve full amplitude at peaks and valleys
    damping = 0.5  # Blend factor with neighbors
    
    for _ in range(iterations):
        new_positions = smoothed_verts.copy()
        for i in range(n_band):
            # Skip extrema - keep them at original displaced position
            if is_extremum[i]:
                continue
                
            global_idx = band_indices[i]
            if len(neighbors[i]) > 0:
                # Average neighbor positions (in global coordinates)
                neighbor_global_indices = [band_indices[n] for n in neighbors[i]]
                neighbor_avg = np.mean(smoothed_verts[neighbor_global_indices], axis=0)
                # Blend current position with neighbor average
                new_positions[global_idx] = (
                    (1.0 - damping) * smoothed_verts[global_idx] + 
                    damping * neighbor_avg
                )
        smoothed_verts = new_positions
    
    return smoothed_verts


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
