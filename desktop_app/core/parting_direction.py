"""
Parting Direction Estimation for Two-Piece Mold Analysis

Uses Embree-accelerated raycasting to find optimal mold parting directions 
that maximize surface visibility while ensuring mutually exclusive surface ownership.

Key constraints:
- Directions must be > 135° apart (suitable for two-piece molds)
- Each triangle is owned by at most one direction (no double-counting)
- Self-occlusion is handled via Embree raycasting

Acceleration:
- Embree-accelerated BVH raycasting when available (via embreex package)
- Multi-threaded parallel batch processing for candidate directions

Design Principles:
- Separate concerns: sampling, visibility, pair selection, painting
- Cached mesh data to avoid redundant extraction
- Named constants for all thresholds
- Vectorized NumPy operations where possible
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import multiprocessing

import numpy as np
import trimesh


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# RGB values for vertex painting (0-1 range)
PAINT_COLOR_D1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)        # Green
PAINT_COLOR_D2 = np.array([1.0, 0.4, 0.0], dtype=np.float32)        # Orange
PAINT_COLOR_NEUTRAL = np.array([0.533, 0.533, 0.533], dtype=np.float32)  # Gray
PAINT_COLOR_OVERLAP = np.array([1.0, 1.0, 0.0], dtype=np.float32)   # Yellow

# For backward compatibility
PAINT_COLORS = {
    'D1': PAINT_COLOR_D1,
    'D2': PAINT_COLOR_D2,
    'NEUTRAL': PAINT_COLOR_NEUTRAL,
    'OVERLAP': PAINT_COLOR_OVERLAP,
}

# Minimum angle between parting directions (degrees)
MIN_ANGLE_DEGREES = 135
MIN_ANGLE_COS = np.cos(np.radians(MIN_ANGLE_DEGREES))

# Backward compatibility alias
MIN_ANGLE_DEG = MIN_ANGLE_DEGREES

# Pair selection parameters
PAIR_SELECTION_TOP_K = 30  # Number of top directions to consider for pairing
COVERAGE_TOLERANCE = 1e-4  # Tolerance for coverage comparison
COVERAGE_EPSILON = COVERAGE_TOLERANCE  # Alias for backward compatibility

# Ray tracing parameters
RAY_OFFSET_FACTOR = 2.0  # Multiplier on bounding box extent for ray offset

# Parallel processing parameters
MIN_DIRECTIONS_FOR_PARALLEL = 8  # Below this, process sequentially
BATCH_DIVISOR = 4  # Divide work into (num_workers * BATCH_DIVISOR) batches
PROGRESS_UPDATE_INTERVAL = 5  # Report progress every N completions


# =============================================================================
# RAY TRACING BACKEND DETECTION
# =============================================================================

def _check_embree_available() -> bool:
    """
    Check if trimesh is using Embree-backed ray tracing.
    
    Returns:
        True if Embree acceleration is available
    """
    try:
        test_mesh = trimesh.creation.box()
        ray_type = type(test_mesh.ray).__name__
        ray_module = type(test_mesh.ray).__module__
        return ('embree' in ray_type.lower() or 
                'embree' in ray_module.lower() or
                'pyembree' in ray_module.lower())
    except Exception:
        return False


# Module-level flag (computed once at import)
EMBREE_AVAILABLE = _check_embree_available()

if EMBREE_AVAILABLE:
    logger.info("Embree ray tracing available - using hardware-accelerated raycasting")
else:
    logger.info("Embree not available - using trimesh default raycasting")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MeshTriangleData:
    """
    Cached triangle data extracted from a mesh.
    
    Pre-computing these values avoids redundant calculations during
    visibility computation for multiple directions.
    
    Attributes:
        normals: Face normals (n_faces, 3)
        centroids: Face centroids (n_faces, 3)
        areas: Face areas (n_faces,)
        total_area: Sum of all face areas
        ray_offset: Computed ray offset based on bounding box
    """
    normals: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray
    total_area: float
    ray_offset: float
    
    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh) -> 'MeshTriangleData':
        """
        Extract triangle data from a mesh.
        
        Args:
            mesh: The trimesh mesh
            
        Returns:
            MeshTriangleData with pre-computed values
        """
        normals = mesh.face_normals.astype(np.float32)
        
        vertices = mesh.vertices
        faces = mesh.faces
        centroids = vertices[faces].mean(axis=1).astype(np.float32)
        
        areas = mesh.area_faces.astype(np.float32)
        total_area = float(areas.sum())
        
        bounds = mesh.bounds
        max_extent = float(np.max(bounds[1] - bounds[0]))
        ray_offset = max_extent * RAY_OFFSET_FACTOR
        
        return cls(
            normals=normals,
            centroids=centroids,
            areas=areas,
            total_area=total_area,
            ray_offset=ray_offset
        )


@dataclass
class DirectionScore:
    """
    Visibility score for a single candidate direction.
    
    Attributes:
        direction: Unit vector (3,)
        visible_area: Total visible surface area
        visible_triangles: Set of visible triangle indices
    """
    direction: np.ndarray
    visible_area: float
    visible_triangles: Set[int]
    
    @property
    def visibility_mask(self) -> None:
        """Visibility mask is computed on-demand, not stored."""
        # This property is a hint that we use masks internally
        raise NotImplementedError("Use get_visibility_mask() with n_faces parameter")
    
    def get_visibility_mask(self, n_faces: int) -> np.ndarray:
        """
        Convert visible triangles to a boolean mask.
        
        Args:
            n_faces: Total number of faces in mesh
            
        Returns:
            Boolean array of shape (n_faces,)
        """
        mask = np.zeros(n_faces, dtype=bool)
        for tri in self.visible_triangles:
            mask[tri] = True
        return mask


@dataclass
class PartingDirectionResult:
    """
    Result of parting direction computation.
    
    Attributes:
        d1: Primary direction (3,) unit vector (extraction direction)
        d2: Secondary direction (3,) unit vector (extraction direction)
        d1_visible_area: Surface area visible from D1
        d2_unique_area: Surface area uniquely visible from D2 (not visible from D1)
        total_coverage: Total coverage percentage (0-100)
        angle_degrees: Angle between D1 and D2 in degrees
        computation_time_ms: Computation time in milliseconds
        d1_triangles: Set of triangle indices visible from D1
        d2_triangles: Set of triangle indices visible from D2
    """
    d1: np.ndarray
    d2: np.ndarray
    d1_visible_area: float
    d2_unique_area: float
    total_coverage: float
    angle_degrees: float
    computation_time_ms: float
    d1_triangles: Set[int] = field(default_factory=set)
    d2_triangles: Set[int] = field(default_factory=set)


@dataclass
class VisibilityPaintData:
    """
    Data for visibility painting of mesh faces.
    
    Attributes:
        d1: Primary parting direction
        d2: Secondary parting direction
        triangle_classes: Per-triangle classification array
            - 0 = neutral (not visible from either direction)
            - 1 = D1 only
            - 2 = D2 only
            - 3 = overlap (visible from both)
    """
    d1: np.ndarray
    d2: np.ndarray
    triangle_classes: np.ndarray


# =============================================================================
# DIRECTION SAMPLING
# =============================================================================

def fibonacci_sphere(k: int) -> np.ndarray:
    """
    Generate k uniformly distributed points on a unit sphere using Fibonacci sampling.
    
    This produces a nearly uniform distribution of directions for sampling
    candidate parting directions.
    
    Args:
        k: Number of directions to generate (must be >= 1)
        
    Returns:
        Array of shape (k, 3) with unit vectors
        
    Raises:
        ValueError: If k < 1
        
    Reference:
        Fibonacci lattice method produces approximately uniform spherical distribution.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    
    if k == 1:
        return np.array([[0.0, 0.0, 1.0]])
    
    indices = np.arange(k, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle ~2.4 radians
    
    # y goes from 1 to -1 (top to bottom of sphere)
    y = 1.0 - (indices / (k - 1)) * 2.0
    radius = np.sqrt(1.0 - y * y)
    theta = phi * indices
    
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    
    directions = np.column_stack([x, y, z])
    
    # Normalize to ensure unit vectors (handle numerical precision)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Prevent division by zero
    directions = directions / norms
    
    return directions


# =============================================================================
# BACKWARD COMPATIBILITY - extract_triangle_data
# =============================================================================

def extract_triangle_data(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract triangle normals, centroids, and areas from a mesh.
    
    This function is provided for backward compatibility. New code should
    use MeshTriangleData.from_mesh() instead for better caching.
    
    Args:
        mesh: The trimesh mesh
        
    Returns:
        Tuple of (normals, centroids, areas) as numpy arrays
    """
    data = MeshTriangleData.from_mesh(mesh)
    return data.normals, data.centroids, data.areas


# =============================================================================
# VISIBILITY COMPUTATION
# =============================================================================

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
    when available for fast BVH-accelerated raycasting.
    
    Algorithm:
    1. Filter triangles by normal direction (backface culling)
    2. Cast rays from face centroids toward the mesh
    3. Check which rays hit their source triangle (self-hit = visible)
    
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
    # Step 1: Backface culling - only consider front-facing triangles
    dots = np.dot(normals, direction)
    front_facing_mask = dots > 0
    front_facing_indices = np.where(front_facing_mask)[0]
    
    if len(front_facing_indices) == 0:
        return [], 0.0
    
    # Step 2: Cast rays from face centroids
    n_rays = len(front_facing_indices)
    ray_origins = centroids[front_facing_indices] + direction * ray_offset
    ray_directions = np.tile(-direction, (n_rays, 1))
    
    try:
        # Use trimesh's ray intersection (Embree-accelerated when available)
        hit_indices = mesh.ray.intersects_first(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )
    except Exception as e:
        logger.warning("Ray intersection failed: %s", e)
        # Fallback: assume all front-facing triangles are visible
        return front_facing_indices.tolist(), float(areas[front_facing_indices].sum())
    
    # Step 3: Self-hit check - ray hits its source triangle = visible
    self_hit_mask = hit_indices == front_facing_indices
    
    visible_indices = front_facing_indices[self_hit_mask].tolist()
    visible_area = float(areas[front_facing_indices[self_hit_mask]].sum())
    
    return visible_indices, visible_area


def _compute_visibility_for_direction_fast(
    direction: np.ndarray,
    mesh: trimesh.Trimesh,
    triangle_data: MeshTriangleData
) -> Tuple[List[int], float]:
    """
    Internal fast path using cached MeshTriangleData.
    
    Args:
        direction: Unit vector direction to check (3,)
        mesh: Trimesh mesh object
        triangle_data: Pre-computed triangle data
        
    Returns:
        Tuple of (visible_triangle_indices, visible_area)
    """
    return compute_visibility_for_direction(
        direction,
        mesh,
        triangle_data.normals,
        triangle_data.centroids,
        triangle_data.areas,
        triangle_data.ray_offset
    )


def _compute_visibility_batch(
    directions: np.ndarray,
    direction_indices: List[int],
    mesh: trimesh.Trimesh,
    triangle_data: MeshTriangleData
) -> List[Tuple[int, List[int], float]]:
    """
    Compute visibility for a batch of directions.
    
    Batching reduces Python function call overhead when processing
    many directions.
    
    Args:
        directions: Array of directions to process (batch_size, 3)
        direction_indices: Original indices of these directions
        mesh: Trimesh mesh object
        triangle_data: Pre-computed triangle data
        
    Returns:
        List of (direction_idx, visible_indices, visible_area) tuples
    """
    results = []
    for i, idx in enumerate(direction_indices):
        visible_indices, visible_area = _compute_visibility_for_direction_fast(
            directions[i], mesh, triangle_data
        )
        results.append((idx, visible_indices, visible_area))
    return results


def compute_all_visibility_parallel(
    directions: np.ndarray,
    mesh: trimesh.Trimesh,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    ray_offset: float,
    num_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> Dict[int, DirectionScore]:
    """
    Compute visibility for all directions in parallel using thread pool.
    
    Embree is thread-safe, so we can run multiple ray queries in parallel.
    Uses batched processing to reduce Python function call overhead.
    
    Args:
        directions: Array of shape (k, 3) with unit vectors
        mesh: Trimesh mesh object  
        normals: Face normals (n_faces, 3)
        centroids: Face centroids (n_faces, 3)
        areas: Face areas (n_faces,)
        ray_offset: Distance to offset ray origins
        num_workers: Number of parallel workers
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        Dict mapping direction index to DirectionScore
    """
    # Create cached triangle data
    triangle_data = MeshTriangleData(
        normals=normals,
        centroids=centroids,
        areas=areas,
        total_area=float(areas.sum()),
        ray_offset=ray_offset
    )
    
    return _compute_all_visibility_parallel_fast(
        directions, mesh, triangle_data, num_workers, progress_callback
    )


def _compute_all_visibility_parallel_fast(
    directions: np.ndarray,
    mesh: trimesh.Trimesh,
    triangle_data: MeshTriangleData,
    num_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> Dict[int, DirectionScore]:
    """
    Internal fast path for parallel visibility computation.
    
    Args:
        directions: Array of shape (k, 3) with unit vectors
        mesh: Trimesh mesh object
        triangle_data: Pre-computed triangle data
        num_workers: Number of parallel workers
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        Dict mapping direction index to DirectionScore
    """
    k = len(directions)
    direction_scores: Dict[int, DirectionScore] = {}
    completed = 0
    
    # For small k, process sequentially (threading overhead not worth it)
    if k <= MIN_DIRECTIONS_FOR_PARALLEL:
        for i in range(k):
            visible_indices, visible_area = _compute_visibility_for_direction_fast(
                directions[i], mesh, triangle_data
            )
            direction_scores[i] = DirectionScore(
                direction=directions[i].copy(),
                visible_area=visible_area,
                visible_triangles=set(visible_indices)
            )
            completed += 1
            if progress_callback:
                progress_callback(completed, k)
        return direction_scores
    
    # Create batches for parallel processing
    batch_size = max(2, k // (num_workers * BATCH_DIVISOR))
    batches = []
    for start in range(0, k, batch_size):
        end = min(start + batch_size, k)
        batch_indices = list(range(start, end))
        batch_directions = directions[start:end]
        batches.append((batch_directions, batch_indices))
    
    # Use ThreadPoolExecutor (Embree releases the GIL during ray tracing)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for batch_directions, batch_indices in batches:
            future = executor.submit(
                _compute_visibility_batch,
                batch_directions,
                batch_indices,
                mesh,
                triangle_data
            )
            futures[future] = len(batch_indices)
        
        # Collect results as batches complete
        for future in as_completed(futures):
            batch_count = futures[future]
            try:
                batch_results = future.result()
                for idx, visible_indices, visible_area in batch_results:
                    direction_scores[idx] = DirectionScore(
                        direction=directions[idx].copy(),
                        visible_area=visible_area,
                        visible_triangles=set(visible_indices)
                    )
                    completed += 1
                    if progress_callback and (completed % PROGRESS_UPDATE_INTERVAL == 0 or completed == k):
                        progress_callback(completed, k)
            except Exception as e:
                logger.warning("Batch failed: %s", e)
                completed += batch_count
                if progress_callback:
                    progress_callback(completed, k)
    
    # Fill in any missing directions with empty results
    for i in range(k):
        if i not in direction_scores:
            direction_scores[i] = DirectionScore(
                direction=directions[i].copy(),
                visible_area=0.0,
                visible_triangles=set()
            )
    
    return direction_scores


# =============================================================================
# BACKWARD COMPATIBILITY - compute_visibility_batch
# =============================================================================

def compute_visibility_batch(
    directions: np.ndarray,
    direction_indices: List[int],
    mesh: trimesh.Trimesh,
    normals: np.ndarray,
    centroids: np.ndarray,
    areas: np.ndarray,
    ray_offset: float
) -> List[Tuple[int, List[int], float]]:
    """
    Compute visibility for a batch of directions.
    
    This function is provided for backward compatibility.
    
    Args:
        directions: Array of directions to process (batch_size, 3)
        direction_indices: Original indices of these directions
        mesh: Trimesh mesh object
        normals: Face normals (n_faces, 3)
        centroids: Face centroids (n_faces, 3)
        areas: Face areas (n_faces,)
        ray_offset: Distance to offset ray origins
        
    Returns:
        List of (direction_idx, visible_indices, visible_area) tuples
    """
    results = []
    for i, idx in enumerate(direction_indices):
        direction = directions[i]
        visible_indices, visible_area = compute_visibility_for_direction(
            direction, mesh, normals, centroids, areas, ray_offset
        )
        results.append((idx, visible_indices, visible_area))
    return results


# =============================================================================
# PAIR SELECTION
# =============================================================================

def _prepare_visibility_data(
    sorted_scores: List[DirectionScore],
    areas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-compute visibility data for pair selection.
    
    Converts Python sets to vectorized boolean masks for fast operations.
    
    Args:
        sorted_scores: Scores sorted by visible area (descending)
        areas: Triangle areas array
        
    Returns:
        Tuple of (visibility_masks, directions_array, visible_areas_array)
    """
    n_scores = len(sorted_scores)
    n_faces = len(areas)
    
    visibility_masks = np.zeros((n_scores, n_faces), dtype=bool)
    directions_array = np.zeros((n_scores, 3), dtype=np.float64)
    
    for i, score in enumerate(sorted_scores):
        # Use get_visibility_mask if available, else set manually
        for tri in score.visible_triangles:
            visibility_masks[i, tri] = True
        directions_array[i] = score.direction
    
    visible_areas = np.array([s.visible_area for s in sorted_scores], dtype=np.float64)
    
    return visibility_masks, directions_array, visible_areas


def _evaluate_pair(
    mask1: np.ndarray,
    mask2: np.ndarray,
    visible_area1: float,
    visible_area2: float,
    areas: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate a pair of directions for coverage.
    
    Computes coverage for both configurations (d1 primary vs d2 primary).
    
    Args:
        mask1: Visibility mask for direction 1
        mask2: Visibility mask for direction 2
        visible_area1: Total visible area for direction 1
        visible_area2: Total visible area for direction 2
        areas: Triangle areas array
        
    Returns:
        Tuple of (coverage_d1_primary, coverage_d2_primary)
    """
    # D1 as primary: coverage = d1_area + d2_unique_area
    d2_unique_mask = mask2 & ~mask1
    coverage_d1_primary = visible_area1 + float(areas[d2_unique_mask].sum())
    
    # D2 as primary: coverage = d2_area + d1_unique_area
    d1_unique_mask = mask1 & ~mask2
    coverage_d2_primary = visible_area2 + float(areas[d1_unique_mask].sum())
    
    return coverage_d1_primary, coverage_d2_primary


def find_best_pair(
    direction_scores: Dict[int, DirectionScore],
    areas: np.ndarray,
    total_area: float
) -> Tuple[DirectionScore, DirectionScore]:
    """
    Find the best pair of parting directions.
    
    Selects the pair that:
    1. Has angle > 135° between them (MIN_ANGLE_DEGREES)
    2. Maximizes total coverage (D1 area + D2 unique area)
    
    Uses vectorized NumPy operations for fast set difference calculations.
    
    Algorithm:
    1. Sort directions by visible area (best candidates first)
    2. Pre-compute visibility masks and dot products
    3. For top-k candidates, evaluate all pairs satisfying angle constraint
    4. Track best coverage seen
    
    Args:
        direction_scores: Dict mapping direction index to DirectionScore
        areas: Triangle areas
        total_area: Total surface area (unused but kept for API compatibility)
        
    Returns:
        Tuple of (best_d1, best_d2) DirectionScores
        
    Reference:
        Paper Section 4.1 - Parting direction selection
    """
    # Sort by visible area (descending) - best candidates first
    sorted_scores = sorted(
        direction_scores.values(),
        key=lambda s: s.visible_area,
        reverse=True
    )
    
    if len(sorted_scores) < 2:
        logger.warning("Less than 2 directions available, returning same direction twice")
        return sorted_scores[0], sorted_scores[0]
    
    n_scores = len(sorted_scores)
    
    # Pre-compute visibility data
    visibility_masks, directions_array, visible_areas = _prepare_visibility_data(
        sorted_scores, areas
    )
    
    # Pre-compute all pairwise dot products (for angle checking)
    dot_products = directions_array @ directions_array.T
    
    # Initialize with fallback
    best_d1 = sorted_scores[0]
    best_d2 = sorted_scores[1]
    best_coverage = 0.0
    best_dot = 1.0
    
    # Only check top candidates (prune search space)
    top_k = min(n_scores, PAIR_SELECTION_TOP_K)
    
    for i in range(top_k):
        s1 = sorted_scores[i]
        mask1 = visibility_masks[i]
        
        for j in range(i + 1, n_scores):
            # Check angle constraint using pre-computed dot products
            dot = dot_products[i, j]
            if dot > MIN_ANGLE_COS:
                continue  # Angle < MIN_ANGLE_DEGREES
            
            # Quick upper bound pruning
            if visible_areas[i] + visible_areas[j] <= best_coverage:
                continue
            
            s2 = sorted_scores[j]
            mask2 = visibility_masks[j]
            
            # Evaluate both configurations
            coverage_d1_primary, coverage_d2_primary = _evaluate_pair(
                mask1, mask2, visible_areas[i], visible_areas[j], areas
            )
            
            # Check D1 as primary
            if coverage_d1_primary > best_coverage + COVERAGE_EPSILON or \
               (abs(coverage_d1_primary - best_coverage) < COVERAGE_EPSILON and dot < best_dot):
                best_coverage = coverage_d1_primary
                best_d1 = s1
                best_d2 = s2
                best_dot = dot
            
            # Check D2 as primary
            if coverage_d2_primary > best_coverage + COVERAGE_EPSILON or \
               (abs(coverage_d2_primary - best_coverage) < COVERAGE_EPSILON and dot < best_dot):
                best_coverage = coverage_d2_primary
                best_d1 = s2
                best_d2 = s1
                best_dot = dot
    
    return best_d1, best_d2


# =============================================================================
# MAIN API
# =============================================================================

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
    
    # Compute visibility for all directions in parallel
    # This uses ThreadPoolExecutor since Embree releases the GIL
    logger.info(f"Computing visibility for {k} directions using {num_workers} workers")
    direction_scores = compute_all_visibility_parallel(
        directions, mesh, normals, centroids, areas, ray_offset,
        num_workers=num_workers, progress_callback=progress_callback
    )
    
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
    
    # Check visibility for each direction using boolean masks
    d1_mask = np.zeros(n_faces, dtype=bool)
    d2_mask = np.zeros(n_faces, dtype=bool)
    
    if show_d1:
        d1_indices, _ = compute_visibility_for_direction(
            d1, mesh, normals, centroids, areas, ray_offset
        )
        if d1_indices:
            d1_mask[d1_indices] = True
    
    if show_d2:
        d2_indices, _ = compute_visibility_for_direction(
            d2, mesh, normals, centroids, areas, ray_offset
        )
        if d2_indices:
            d2_mask[d2_indices] = True
    
    # Vectorized classification: 0=neutral, 1=D1, 2=D2, 3=overlap
    # Use bitwise encoding: D1 contributes 1, D2 contributes 2
    triangle_classes = d1_mask.astype(np.int32) + 2 * d2_mask.astype(np.int32)
    
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
    
    Uses vectorized NumPy operations for speed.
    
    Args:
        paint_data: VisibilityPaintData from compute_visibility_paint
        
    Returns:
        Array of shape (n_faces, 4) with RGBA colors (0-255)
    """
    n_faces = len(paint_data.triangle_classes)
    colors = np.zeros((n_faces, 4), dtype=np.uint8)
    colors[:, 3] = 255  # Full opacity
    
    # Pre-compute color lookup table (class 0-3 -> RGB)
    color_lut = np.array([
        PAINT_COLORS['NEUTRAL'] * 255,   # class 0 = neutral
        PAINT_COLORS['D1'] * 255,        # class 1 = D1
        PAINT_COLORS['D2'] * 255,        # class 2 = D2
        PAINT_COLORS['OVERLAP'] * 255,   # class 3 = overlap
    ], dtype=np.uint8)
    
    # Vectorized lookup - directly index into LUT
    colors[:, :3] = color_lut[paint_data.triangle_classes]
    
    return colors
