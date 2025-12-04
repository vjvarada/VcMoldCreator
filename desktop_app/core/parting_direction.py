"""
Parting Direction Estimation for Two-Piece Mold Analysis

Uses Embree-accelerated raycasting to find optimal mold parting directions 
that maximize surface visibility while ensuring mutually exclusive surface ownership.

Key constraints:
- Directions must be > 135° apart (suitable for two-piece molds)
- Each triangle is owned by at most one direction (no double-counting)
- Self-occlusion is handled via Embree raycasting
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Set, Dict
import multiprocessing

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Colors for visualization (matching React frontend)
class PartingColors:
    """Colors for parting direction visualization (hex strings)."""
    D1 = "#00ff00"           # Green - Primary direction
    D2 = "#ff6600"           # Orange - Secondary direction
    NEUTRAL = "#888888"      # Gray - Not visible from either
    OVERLAP = "#ffff00"      # Yellow - Visible from both
    MESH_DEFAULT = "#00aaff" # Light blue - Original mesh color


# RGB values for vertex painting (0-1 range)
PAINT_COLORS = {
    'D1': np.array([0.0, 1.0, 0.0]),        # Green
    'D2': np.array([1.0, 0.4, 0.0]),        # Orange
    'NEUTRAL': np.array([0.533, 0.533, 0.533]),  # Gray
    'OVERLAP': np.array([1.0, 1.0, 0.0]),   # Yellow
}

# Minimum angle between parting directions (degrees)
MIN_ANGLE_DEG = 135
MIN_ANGLE_COS = np.cos(np.radians(MIN_ANGLE_DEG))


# ============================================================================
# RAY TRACING BACKEND DETECTION
# ============================================================================

def _check_embree_available() -> bool:
    """Check if trimesh is using Embree-backed ray tracing."""
    try:
        # Create a simple mesh and check the ray tracer type
        test_mesh = trimesh.creation.box()
        ray_type = type(test_mesh.ray).__name__
        ray_module = type(test_mesh.ray).__module__
        # Check for embree in type name or module path
        is_embree = ('embree' in ray_type.lower() or 
                     'embree' in ray_module.lower() or
                     'pyembree' in ray_module.lower())
        return is_embree
    except Exception:
        return False

EMBREE_AVAILABLE = _check_embree_available()

if EMBREE_AVAILABLE:
    logger.info("Embree ray tracing available via trimesh - using hardware-accelerated raycasting")
else:
    logger.info("Embree not available - using trimesh default raycasting")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DirectionScore:
    """Score for a single candidate direction."""
    direction: np.ndarray          # Unit vector (3,)
    visible_area: float            # Total visible surface area
    visible_triangles: Set[int]    # Set of visible triangle indices


@dataclass
class PartingDirectionResult:
    """Result of parting direction computation."""
    d1: np.ndarray                         # Primary direction (3,) unit vector
    d2: np.ndarray                         # Secondary direction (3,) unit vector
    d1_visible_area: float                 # Area visible from D1
    d2_unique_area: float                  # Area uniquely visible from D2
    total_coverage: float                  # Total coverage percentage (0-100)
    angle_degrees: float                   # Angle between D1 and D2
    computation_time_ms: float             # Time taken in milliseconds
    d1_triangles: Set[int] = field(default_factory=set)  # Triangles visible from D1
    d2_triangles: Set[int] = field(default_factory=set)  # Triangles visible from D2


@dataclass
class VisibilityPaintData:
    """Data for visibility painting."""
    d1: np.ndarray                  # Primary direction
    d2: np.ndarray                  # Secondary direction
    triangle_classes: np.ndarray   # Per-triangle classification: 0=neutral, 1=D1, 2=D2, 3=overlap


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fibonacci_sphere(k: int) -> np.ndarray:
    """
    Generate k uniformly distributed points on a unit sphere using Fibonacci sampling.
    
    This produces a nearly uniform distribution of directions for sampling.
    
    Args:
        k: Number of directions to generate
        
    Returns:
        Array of shape (k, 3) with unit vectors
    """
    indices = np.arange(k, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    
    y = 1.0 - (indices / (k - 1)) * 2.0  # y goes from 1 to -1
    radius = np.sqrt(1.0 - y * y)
    theta = phi * indices
    
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    
    directions = np.column_stack([x, y, z])
    
    # Normalize to ensure unit vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms
    
    return directions


def extract_triangle_data(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract triangle normals, centroids, and areas from a mesh.
    
    Args:
        mesh: The trimesh mesh
        
    Returns:
        Tuple of (normals, centroids, areas) as numpy arrays
    """
    # Get face normals (trimesh computes these automatically)
    normals = mesh.face_normals.astype(np.float32)
    
    # Compute centroids
    vertices = mesh.vertices
    faces = mesh.faces
    centroids = vertices[faces].mean(axis=1).astype(np.float32)
    
    # Get areas
    areas = mesh.area_faces.astype(np.float32)
    
    return normals, centroids, areas


# ============================================================================
# VISIBILITY COMPUTATION - EMBREE (Fast Path)
# ============================================================================

def compute_visibility_for_direction(
    direction: np.ndarray,
    mesh: trimesh.Trimesh,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    ray_offset: float
) -> Tuple[List[int], float]:
    """
    Compute which triangles are visible from a given direction.
    
    Uses trimesh's ray intersection which automatically uses Embree
    when available (via embreex package) for fast BVH-accelerated raycasting.
    
    Args:
        direction: Unit vector direction to check (3,)
        mesh: Trimesh mesh object
        normals: Face normals (n_faces, 3)
        centroids: Face centroids (n_faces, 3)
        areas: Face areas (n_faces,)
        ray_offset: Distance to offset ray origins
        
    Returns:
        Tuple of (visible_triangle_indices, visible_area)
    """
    # Step 1: Filter by normal direction (backface culling)
    dots = np.dot(normals, direction)
    front_facing_mask = dots > 0
    front_facing_indices = np.where(front_facing_mask)[0]
    
    if len(front_facing_indices) == 0:
        return [], 0.0
    
    # Step 2: Ray cast from face centroids toward the mesh
    n_rays = len(front_facing_indices)
    ray_origins = centroids[front_facing_indices] + direction * ray_offset
    ray_directions = np.tile(-direction, (n_rays, 1))
    
    try:
        # Use trimesh's ray intersection (embree-accelerated when available)
        hit_indices = mesh.ray.intersects_first(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )
    except Exception as e:
        logger.warning(f"Ray intersection failed: {e}")
        # Fallback: assume all front-facing triangles are visible
        return front_facing_indices.tolist(), float(areas[front_facing_indices].sum())
    
    # Step 3: Check which rays hit their target triangle (self-hit = visible)
    self_hit_mask = hit_indices == front_facing_indices
    
    visible_indices = front_facing_indices[self_hit_mask].tolist()
    visible_area = float(areas[front_facing_indices[self_hit_mask]].sum())
    
    return visible_indices, visible_area


# ============================================================================
# BEST PAIR SELECTION
# ============================================================================

def find_best_pair(
    direction_scores: Dict[int, DirectionScore],
    areas: np.ndarray,
    total_area: float
) -> Tuple[DirectionScore, DirectionScore]:
    """
    Find the best pair of parting directions.
    
    Selects the pair that:
    1. Has angle > 135° between them
    2. Maximizes total coverage (D1 area + D2 unique area)
    
    Args:
        direction_scores: Dict mapping direction index to DirectionScore
        areas: Triangle areas
        total_area: Total surface area
        
    Returns:
        Tuple of (best_d1, best_d2) DirectionScores
    """
    # Sort by visible area (descending)
    sorted_scores = sorted(
        direction_scores.values(),
        key=lambda s: s.visible_area,
        reverse=True
    )
    
    if len(sorted_scores) < 2:
        return sorted_scores[0], sorted_scores[0]
    
    best_d1 = sorted_scores[0]
    best_d2 = sorted_scores[1]
    best_coverage = 0.0
    best_dot = 1.0
    
    n_scores = len(sorted_scores)
    
    for i in range(n_scores):
        s1 = sorted_scores[i]
        
        for j in range(i + 1, n_scores):
            s2 = sorted_scores[j]
            
            # Check angle constraint
            dot = np.dot(s1.direction, s2.direction)
            if dot > MIN_ANGLE_COS:
                continue  # Angle too narrow
            
            # Quick upper bound check
            if s1.visible_area + s2.visible_area <= best_coverage:
                continue
            
            # Try s1 as primary: compute s2's unique contribution
            s2_unique_area = sum(
                areas[tri] for tri in s2.visible_triangles
                if tri not in s1.visible_triangles
            )
            coverage = s1.visible_area + s2_unique_area
            
            if coverage > best_coverage + 0.0001 or \
               (abs(coverage - best_coverage) < 0.0001 and dot < best_dot):
                best_coverage = coverage
                best_d1 = s1
                best_d2 = s2
                best_dot = dot
            
            # Try s2 as primary: compute s1's unique contribution
            s1_unique_area = sum(
                areas[tri] for tri in s1.visible_triangles
                if tri not in s2.visible_triangles
            )
            coverage = s2.visible_area + s1_unique_area
            
            if coverage > best_coverage + 0.0001 or \
               (abs(coverage - best_coverage) < 0.0001 and dot < best_dot):
                best_coverage = coverage
                best_d1 = s2
                best_d2 = s1
                best_dot = dot
    
    return best_d1, best_d2


# ============================================================================
# MAIN API
# ============================================================================

def find_parting_directions(
    mesh: trimesh.Trimesh,
    k: int = 64,
    num_workers: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> PartingDirectionResult:
    """
    Find optimal parting directions for a two-piece mold.
    
    Uses Embree-accelerated raycasting and Fibonacci sphere sampling to test 
    k candidate directions, then selects the best pair that maximizes 
    coverage while maintaining a minimum angle of 135° between them.
    
    Args:
        mesh: The trimesh mesh to analyze
        k: Number of candidate directions to sample (default: 64)
        num_workers: Number of parallel workers (default: CPU count)
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        PartingDirectionResult with the optimal directions and statistics
    """
    start_time = time.perf_counter()
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"Finding parting directions: {k} samples")
    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    logger.info(f"Ray tracer: {type(mesh.ray).__name__} (Embree: {EMBREE_AVAILABLE})")
    
    # Extract triangle data
    normals, centroids, areas = extract_triangle_data(mesh)
    total_area = float(areas.sum())
    
    logger.debug(f"Total surface area: {total_area:.4f}")
    
    # Compute ray offset from bounding box
    bounds = mesh.bounds
    max_extent = np.max(bounds[1] - bounds[0])
    ray_offset = max_extent * 2.0
    
    # Generate candidate directions
    directions = fibonacci_sphere(k)
    
    # Compute visibility for all directions
    direction_scores: Dict[int, DirectionScore] = {}
    
    for i in range(k):
        direction = directions[i]
        
        visible_indices, visible_area = compute_visibility_for_direction(
            direction, mesh, normals, centroids, areas, ray_offset
        )
        
        direction_scores[i] = DirectionScore(
            direction=direction.copy(),
            visible_area=visible_area,
            visible_triangles=set(visible_indices)
        )
        
        if progress_callback:
            progress_callback(i + 1, k)
    
    # Find best pair
    best_d1, best_d2 = find_best_pair(direction_scores, areas, total_area)
    
    # Negate directions so they point in the pull/extraction direction
    # (i.e., the direction the mold half would move to release the part)
    d1 = -best_d1.direction
    d2 = -best_d2.direction
    
    # Compute statistics
    angle_deg = np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1, 1)))
    
    d2_unique_area = sum(
        areas[tri] for tri in best_d2.visible_triangles
        if tri not in best_d1.visible_triangles
    )
    
    total_coverage = (best_d1.visible_area + d2_unique_area) / total_area * 100
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    result = PartingDirectionResult(
        d1=d1,
        d2=d2,
        d1_visible_area=best_d1.visible_area,
        d2_unique_area=d2_unique_area,
        total_coverage=total_coverage,
        angle_degrees=angle_deg,
        computation_time_ms=elapsed_ms,
        d1_triangles=best_d1.visible_triangles,
        d2_triangles=best_d2.visible_triangles
    )
    
    logger.info(f"Parting directions found:")
    logger.info(f"  D1: [{d1[0]:.3f}, {d1[1]:.3f}, {d1[2]:.3f}] ({best_d1.visible_area/total_area*100:.1f}%)")
    logger.info(f"  D2: [{d2[0]:.3f}, {d2[1]:.3f}, {d2[2]:.3f}] ({d2_unique_area/total_area*100:.1f}%)")
    logger.info(f"  Coverage: {total_coverage:.1f}%, Angle: {angle_deg:.1f}°")
    logger.info(f"  Time: {elapsed_ms:.0f}ms")
    
    return result


# ============================================================================
# VISIBILITY PAINTING
# ============================================================================

def compute_visibility_paint(
    mesh: trimesh.Trimesh,
    d1: np.ndarray,
    d2: np.ndarray,
    show_d1: bool = True,
    show_d2: bool = True
) -> VisibilityPaintData:
    """
    Compute per-triangle visibility classification for painting.
    
    Args:
        mesh: The trimesh mesh
        d1: Primary parting direction (3,) unit vector
        d2: Secondary parting direction (3,) unit vector
        show_d1: Whether to include D1 visibility
        show_d2: Whether to include D2 visibility
        
    Returns:
        VisibilityPaintData with classification for each triangle
    """
    start_time = time.perf_counter()
    
    # Extract triangle data
    normals, centroids, areas = extract_triangle_data(mesh)
    n_faces = len(normals)
    
    # Compute ray offset
    bounds = mesh.bounds
    max_extent = np.max(bounds[1] - bounds[0])
    ray_offset = max_extent * 2.0
    
    # Initialize classification (0 = neutral)
    triangle_classes = np.zeros(n_faces, dtype=np.int32)
    
    # Check visibility for each direction
    d1_visible = set()
    d2_visible = set()
    
    if show_d1:
        d1_indices, _ = compute_visibility_for_direction(
            d1, mesh, normals, centroids, areas, ray_offset
        )
        d1_visible = set(d1_indices)
    
    if show_d2:
        d2_indices, _ = compute_visibility_for_direction(
            d2, mesh, normals, centroids, areas, ray_offset
        )
        d2_visible = set(d2_indices)
    
    # Classify triangles
    # 0 = neutral, 1 = D1 only, 2 = D2 only, 3 = overlap
    for i in range(n_faces):
        is_d1 = i in d1_visible
        is_d2 = i in d2_visible
        
        if is_d1 and is_d2:
            triangle_classes[i] = 3  # Overlap
        elif is_d1:
            triangle_classes[i] = 1  # D1
        elif is_d2:
            triangle_classes[i] = 2  # D2
        # else remains 0 (neutral)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Count results
    d1_count = np.sum(triangle_classes == 1)
    d2_count = np.sum(triangle_classes == 2)
    overlap_count = np.sum(triangle_classes == 3)
    neutral_count = np.sum(triangle_classes == 0)
    coverage = (d1_count + d2_count + overlap_count) / n_faces * 100
    
    logger.info(f"Visibility paint: D1={d1_count}, D2={d2_count}, Overlap={overlap_count}, Neutral={neutral_count}")
    logger.info(f"Coverage: {coverage:.1f}%, Time: {elapsed_ms:.0f}ms")
    
    return VisibilityPaintData(
        d1=d1.copy(),
        d2=d2.copy(),
        triangle_classes=triangle_classes
    )


def get_face_colors(paint_data: VisibilityPaintData) -> np.ndarray:
    """
    Convert visibility classification to face colors.
    
    Args:
        paint_data: VisibilityPaintData from compute_visibility_paint
        
    Returns:
        Array of shape (n_faces, 4) with RGBA colors (0-255)
    """
    n_faces = len(paint_data.triangle_classes)
    colors = np.zeros((n_faces, 4), dtype=np.uint8)
    colors[:, 3] = 255  # Full opacity
    
    # Map classes to colors
    for i in range(n_faces):
        cls = paint_data.triangle_classes[i]
        if cls == 1:  # D1
            colors[i, :3] = (PAINT_COLORS['D1'] * 255).astype(np.uint8)
        elif cls == 2:  # D2
            colors[i, :3] = (PAINT_COLORS['D2'] * 255).astype(np.uint8)
        elif cls == 3:  # Overlap
            colors[i, :3] = (PAINT_COLORS['OVERLAP'] * 255).astype(np.uint8)
        else:  # Neutral
            colors[i, :3] = (PAINT_COLORS['NEUTRAL'] * 255).astype(np.uint8)
    
    return colors
