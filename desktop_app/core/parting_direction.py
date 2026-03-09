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

# Iterative refinement parameters
REFINE_CONE_HALF_ANGLE_DEG = 15.0   # Half-angle of cone around promising directions
REFINE_SAMPLES_PER_CONE = 24        # Fibonacci samples per refinement cone
REFINE_TOP_N_DIRECTIONS = 6          # Number of top directions to refine around
REFINE_GAP_TARGETS = 4               # Diverse gap-targeting centers
REFINE_GAP_CONE_HALF_ANGLE_DEG = 20.0  # Wider cone for gap targeting
REFINE_GAP_CONE_SAMPLES = 12         # Samples per gap-targeting cone
PAIR_SELECTION_TOP_K_REFINED = 60    # Larger search window for refined set
AXIALITY_COVERAGE_TOLERANCE = 0.005  # 0.5% of total area — pairs within
                                      # this tolerance prefer larger angles

# Fast path parameters (new vectorised pipeline)
ANTIPODAL_EXPANSION = True           # Double effective sampling with -d for each d
NORMAL_BIAS_EXTRA = 18               # Extra PCA-guided normal-biased directions
FAST_PAIR_TOP_K = 80                 # Search window for fast matrix pair selection
OVERLAP_MATRIX_FACE_LIMIT = 200_000  # Use BLAS overlap matrix below this face count
OVERLAP_CHUNK_SIZE = 200             # Column-chunk size for overlap GEMM


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
    refinement_rounds: int = 0
    total_candidates_evaluated: int = 0
    uncovered_area_pct: float = 0.0


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


@dataclass
class _VisibilityMatrix:
    """
    Compact internal visibility representation using numpy boolean masks.

    Replaces Dict[int, DirectionScore] for fast vectorised pair evaluation.
    Visibility is stored as a dense (k, n_faces) boolean array so that union
    and intersection operations use vectorised numpy ops instead of Python
    set arithmetic.

    Attributes:
        directions: Unit direction vectors (k, 3)
        masks: Per-face visibility flags (k, n_faces) boolean
        visible_areas: Total visible area per direction (k,)
    """
    directions: np.ndarray       # (k, 3)
    masks: np.ndarray            # (k, n_faces) bool
    visible_areas: np.ndarray    # (k,)

    @property
    def n_dirs(self) -> int:
        return self.directions.shape[0]

    @property
    def n_faces(self) -> int:
        return self.masks.shape[1]

    def merge(self, other: '_VisibilityMatrix') -> '_VisibilityMatrix':
        """Merge another visibility matrix into this one."""
        return _VisibilityMatrix(
            directions=np.vstack([self.directions, other.directions]),
            masks=np.vstack([self.masks, other.masks]),
            visible_areas=np.concatenate([self.visible_areas, other.visible_areas]),
        )

    def to_direction_scores(self) -> Dict[int, DirectionScore]:
        """Convert to legacy Dict[int, DirectionScore] format."""
        scores: Dict[int, DirectionScore] = {}
        for i in range(self.n_dirs):
            scores[i] = DirectionScore(
                direction=self.directions[i].copy(),
                visible_area=float(self.visible_areas[i]),
                visible_triangles=set(np.where(self.masks[i])[0].tolist()),
            )
        return scores

    def pair_coverage(self, i: int, j: int, areas: np.ndarray) -> float:
        """Union coverage area for direction pair (i, j)."""
        return float(areas[self.masks[i] | self.masks[j]].sum())

    def pair_coverage_pct(self, i: int, j: int, areas: np.ndarray,
                          total_area: float) -> float:
        """Union coverage as percentage for direction pair (i, j)."""
        return self.pair_coverage(i, j, areas) / total_area * 100.0

    def best_pair_scores(self, i: int, j: int) -> Tuple['DirectionScore', 'DirectionScore']:
        """Return (primary, secondary) DirectionScores for a pair."""
        d1 = DirectionScore(
            direction=self.directions[i].copy(),
            visible_area=float(self.visible_areas[i]),
            visible_triangles=set(np.where(self.masks[i])[0].tolist()),
        )
        d2 = DirectionScore(
            direction=self.directions[j].copy(),
            visible_area=float(self.visible_areas[j]),
            visible_triangles=set(np.where(self.masks[j])[0].tolist()),
        )
        return d1, d2


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


def generate_cone_directions(
    center: np.ndarray,
    half_angle_deg: float,
    n_samples: int
) -> np.ndarray:
    """
    Generate directions uniformly distributed within a cone around a center.

    Uses Fibonacci spiral sampling within a spherical cap to produce nearly
    uniform directions in a cone of the given half-angle.

    Args:
        center: Unit vector at the center of the cone (3,)
        half_angle_deg: Half-angle of the cone in degrees
        n_samples: Number of samples to generate (must be >= 1)

    Returns:
        Array of shape (n_samples, 3) with unit vectors inside the cone
    """
    if n_samples < 1:
        return np.empty((0, 3), dtype=np.float64)

    center = np.asarray(center, dtype=np.float64)
    center = center / (np.linalg.norm(center) + 1e-15)
    half_angle_rad = np.radians(half_angle_deg)

    # Build orthonormal frame with center as z-axis
    z = center
    if abs(z[2]) < 0.9:
        x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    else:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= (np.linalg.norm(x) + 1e-15)
    y = np.cross(z, x)

    # Fibonacci-style sampling within spherical cap
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    cos_half = np.cos(half_angle_rad)

    indices = np.arange(n_samples, dtype=np.float64)
    # Uniform distribution of cos(theta) in [cos(half_angle), 1]
    cos_theta = 1.0 - (1.0 - cos_half) * (indices + 0.5) / n_samples
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta ** 2, 0.0))
    phi = golden_angle * indices

    # Local spherical to world Cartesian
    lx = sin_theta * np.cos(phi)
    ly = sin_theta * np.sin(phi)
    lz = cos_theta

    directions = (
        lx[:, None] * x[None, :]
        + ly[:, None] * y[None, :]
        + lz[:, None] * z[None, :]
    )

    # Normalize for numerical safety
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-10)

    return directions


def _generate_gap_target_directions(
    triangle_data: MeshTriangleData,
    covered_mask: np.ndarray,
    n_targets: int = REFINE_GAP_TARGETS,
    cone_samples: int = REFINE_GAP_CONE_SAMPLES
) -> np.ndarray:
    """
    Generate candidate directions that target the uncovered (neutral) surface.

    Strategy:
    1. Compute the area-weighted mean normal of uncovered faces.
    2. Use farthest-point sampling on uncovered normals to find diverse
       gap-targeting center directions.
    3. Generate cone samples around each center.

    Args:
        triangle_data: Pre-computed triangle data
        covered_mask: Boolean mask (n_faces,) — True = already covered
        n_targets: Number of diverse target centers
        cone_samples: Fibonacci samples per target cone

    Returns:
        Array of shape (N, 3) with unit vectors targeting uncovered regions
    """
    uncovered_mask = ~covered_mask
    uncovered_areas = triangle_data.areas * uncovered_mask
    total_uncovered = float(uncovered_areas.sum())

    if total_uncovered < 1e-10:
        return np.empty((0, 3), dtype=np.float64)

    # Area-weighted mean normal of uncovered faces
    weighted_normals = triangle_data.normals * uncovered_areas[:, None]
    mean_dir = weighted_normals.sum(axis=0).astype(np.float64)
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-10:
        return np.empty((0, 3), dtype=np.float64)
    mean_dir /= norm

    # Select top uncovered faces by area for farthest-point sampling
    n_uncovered = int(uncovered_mask.sum())
    top_k = min(200, n_uncovered)
    top_indices = np.argsort(-uncovered_areas)[:top_k]
    candidate_normals = triangle_data.normals[top_indices].astype(np.float64)

    # Farthest-point sampling for diverse directions
    selected = [0]
    for _ in range(min(n_targets - 1, len(candidate_normals) - 1)):
        selected_dirs = candidate_normals[selected]
        # Maximum absolute cosine similarity to any already selected direction
        sims = np.abs(candidate_normals @ selected_dirs.T).max(axis=1)
        next_idx = int(np.argmin(sims))
        if next_idx not in selected:
            selected.append(next_idx)

    target_dirs = candidate_normals[selected]

    # Generate cone directions around mean + each target
    all_dirs = [generate_cone_directions(mean_dir, REFINE_GAP_CONE_HALF_ANGLE_DEG, cone_samples)]
    for td in target_dirs:
        all_dirs.append(
            generate_cone_directions(td, REFINE_GAP_CONE_HALF_ANGLE_DEG, cone_samples)
        )

    return np.vstack(all_dirs)


def _expand_with_antipodal(directions: np.ndarray) -> np.ndarray:
    """
    Double the direction set by appending the antipodal (-d) of each direction.

    Since optimal two-piece mold pairs tend to be near-antipodal, this
    ensures at least one sample close to the true optimum for each base
    direction, without additional raycasting budget.

    Duplicate-safe: if d and -d are both already present, the downstream
    pair selector simply evaluates both.

    Args:
        directions: (k, 3) unit vectors

    Returns:
        (2k, 3) array with original + negated directions
    """
    return np.vstack([directions, -directions])


def _generate_normal_biased_directions(
    triangle_data: MeshTriangleData,
    n_extra: int = NORMAL_BIAS_EXTRA,
) -> np.ndarray:
    """
    Generate extra candidate directions biased toward dominant face-normal
    clusters using area-weighted PCA.

    Faces with large area contribute more to the covariance, so the
    principal axes capture the directions most likely to see substantive
    surface.  Cone samples around each axis (and its negative) explore
    the neighbourhood.

    Args:
        triangle_data: Pre-computed triangle data
        n_extra: Total extra directions to generate (split among 6 cones)

    Returns:
        (n_extra, 3) unit vectors  (may be fewer if PCA fails)
    """
    normals = triangle_data.normals.astype(np.float64)
    weights = triangle_data.areas.astype(np.float64)
    w_sum = weights.sum()
    if w_sum < 1e-15:
        return np.empty((0, 3), dtype=np.float64)
    weights = weights / w_sum

    # Weighted normals → SVD to find principal axes
    weighted_normals = normals * np.sqrt(weights[:, None])
    try:
        _, _, Vt = np.linalg.svd(weighted_normals, full_matrices=False)
        principal_axes = Vt[:3]  # (3, 3) — top 3 directions
    except Exception:
        return np.empty((0, 3), dtype=np.float64)

    # Generate cone samples around each axis and its negative (6 cones)
    samples_per_cone = max(1, n_extra // 6)
    dirs_list: List[np.ndarray] = []
    for ax in principal_axes:
        ax_unit = ax / (np.linalg.norm(ax) + 1e-15)
        dirs_list.append(generate_cone_directions(ax_unit, 12.0, samples_per_cone))
        dirs_list.append(generate_cone_directions(-ax_unit, 12.0, samples_per_cone))

    return np.vstack(dirs_list) if dirs_list else np.empty((0, 3), dtype=np.float64)


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
# FAST VISIBILITY MATRIX PIPELINE
# =============================================================================

def _compute_visibility_matrix(
    directions: np.ndarray,
    mesh: trimesh.Trimesh,
    triangle_data: MeshTriangleData,
    num_workers: int = 4,
    progress_callback: Optional[callable] = None,
) -> _VisibilityMatrix:
    """
    Compute visibility for all directions, returning a numpy mask matrix.

    This is the fast replacement for ``_compute_all_visibility_parallel_fast``
    that stores results as a dense boolean ``(k, n_faces)`` array rather
    than ``Dict[int, DirectionScore]`` with Python sets.

    The batch backface-culling dot products are computed with a single
    BLAS matrix multiply before the per-direction raycasting loop.

    Args:
        directions: (k, 3) unit vectors to evaluate
        mesh: Trimesh mesh object (Embree-backed when available)
        triangle_data: Pre-computed MeshTriangleData
        num_workers: Thread pool size
        progress_callback: Optional ``callback(current, total)``

    Returns:
        _VisibilityMatrix with boolean masks and area totals
    """
    k = len(directions)
    n_faces = len(triangle_data.areas)
    masks = np.zeros((k, n_faces), dtype=bool)

    # Batch backface culling — single BLAS call
    # all_dots[f, d] = dot(normals[f], directions[d])
    dirs_f32 = np.ascontiguousarray(directions, dtype=np.float32)
    all_dots = triangle_data.normals @ dirs_f32.T  # (n_faces, k)

    def _process_direction(d_idx: int) -> Tuple[int, np.ndarray]:
        """Raycast a single direction.  Returns (index, visible_face_indices)."""
        direction = directions[d_idx]
        front_indices = np.where(all_dots[:, d_idx] > 0)[0]
        if len(front_indices) == 0:
            return d_idx, np.array([], dtype=np.intp)

        ray_origins = (triangle_data.centroids[front_indices]
                       + direction * triangle_data.ray_offset)
        ray_dirs = np.tile(-direction, (len(front_indices), 1))

        try:
            hits = mesh.ray.intersects_first(ray_origins, ray_dirs)
            visible = front_indices[hits == front_indices]
        except Exception:
            visible = front_indices  # fallback: treat all front-facing as visible
        return d_idx, visible

    # --- Sequential path for small k ---
    if k <= MIN_DIRECTIONS_FOR_PARALLEL:
        for d_idx in range(k):
            idx, visible = _process_direction(d_idx)
            if len(visible) > 0:
                masks[idx, visible] = True
            if progress_callback:
                progress_callback(d_idx + 1, k)
    else:
        # --- Parallel path (Embree releases the GIL) ---
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_direction, i): i for i in range(k)}
            for future in as_completed(futures):
                try:
                    idx, visible = future.result()
                    if len(visible) > 0:
                        masks[idx, visible] = True
                except Exception as e:
                    logger.warning("Direction %d failed: %s", futures[future], e)
                completed += 1
                if progress_callback and (completed % PROGRESS_UPDATE_INTERVAL == 0
                                          or completed == k):
                    progress_callback(completed, k)

    # Compute per-direction visible areas via vector dot
    visible_areas = masks.astype(np.float32) @ triangle_data.areas

    return _VisibilityMatrix(
        directions=directions.copy(),
        masks=masks,
        visible_areas=visible_areas,
    )


def _find_best_pair_from_matrix(
    vm: _VisibilityMatrix,
    areas: np.ndarray,
    total_area: float,
    top_k: int = FAST_PAIR_TOP_K,
    axiality_tol: float = AXIALITY_COVERAGE_TOLERANCE,
) -> Tuple[int, int]:
    """
    Find the best direction pair from a _VisibilityMatrix using BLAS
    overlap-matrix computation.

    For the top *top_k* directions (by individual visible area) and ALL
    remaining directions, we compute the pairwise overlap area in one
    matrix multiply::

        overlap[i, j] = sum_f ( mask_i[f] * mask_j[f] * area[f] )
        coverage[i, j] = vis_area[i] + vis_area[j] - overlap[i, j]

    Pairs violating the minimum-angle constraint are masked out.
    When two pairs have near-equal coverage (within *axiality_tol* of
    *total_area*), the more anti-parallel pair wins.

    Args:
        vm: Visibility matrix from ``_compute_visibility_matrix``
        areas: Triangle areas (n_faces,)
        total_area: Sum of areas
        top_k: Number of top directions to search as primary candidate
        axiality_tol: Fraction of total_area within which angle breaks ties

    Returns:
        (primary_idx, secondary_idx) into ``vm`` arrays, where primary
        has the larger individual visible area.
    """
    k = vm.n_dirs
    if k < 2:
        return 0, min(1, k - 1)

    n_faces = vm.n_faces
    search_k = min(k, top_k)

    # Sort by visible area descending
    order = np.argsort(-vm.visible_areas)
    top_indices = order[:search_k]

    # --- BLAS overlap matrix ---
    # area_weighted_top[i, f] = mask_top[i, f] * area[f]   (float32)
    areas_f32 = areas.astype(np.float32)
    top_masks_f = vm.masks[top_indices].astype(np.float32)
    area_weighted_top = top_masks_f * areas_f32[None, :]  # (search_k, n_faces)

    # Compute overlap in column-chunks to limit peak memory
    overlap = np.zeros((search_k, k), dtype=np.float32)
    for j_start in range(0, k, OVERLAP_CHUNK_SIZE):
        j_end = min(j_start + OVERLAP_CHUNK_SIZE, k)
        chunk_f = vm.masks[j_start:j_end].astype(np.float32)  # (chunk, n_faces)
        overlap[:, j_start:j_end] = area_weighted_top @ chunk_f.T

    # Coverage matrix: vis[i] + vis[j] - overlap[i,j]
    top_vis = vm.visible_areas[top_indices]  # (search_k,)
    all_vis = vm.visible_areas               # (k,)
    coverage = top_vis[:, None] + all_vis[None, :] - overlap  # (search_k, k)

    # Angle validity: dot product between top directions and all directions
    top_dirs = vm.directions[top_indices]  # (search_k, 3)
    all_dirs = vm.directions               # (k, 3)
    dots = (top_dirs @ all_dirs.T).astype(np.float32)  # (search_k, k)
    valid = dots <= MIN_ANGLE_COS

    # Mask invalid pairs and self-pairs
    coverage[~valid] = -np.inf
    for ri, oi in enumerate(top_indices):
        coverage[ri, oi] = -np.inf  # no self-pairing

    # Axiality preference: tiny bonus for more anti-parallel pairs
    # -dot ranges from -1 (parallel) to +1 (anti-parallel)
    axiality_bonus = (-dots + 1.0) * (total_area * axiality_tol * 0.5)
    scored = coverage + axiality_bonus * valid

    # Find best pair
    best_flat = int(np.argmax(scored))
    best_ri, best_j = np.unravel_index(best_flat, scored.shape)
    best_i = int(top_indices[best_ri])
    best_j = int(best_j)

    # Ensure primary has larger visible area
    if vm.visible_areas[best_i] >= vm.visible_areas[best_j]:
        return best_i, best_j
    else:
        return best_j, best_i


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


def find_best_pair_refined(
    direction_scores: Dict[int, DirectionScore],
    areas: np.ndarray,
    total_area: float,
    top_k: int = PAIR_SELECTION_TOP_K_REFINED,
    axiality_tolerance: float = AXIALITY_COVERAGE_TOLERANCE
) -> Tuple[DirectionScore, DirectionScore]:
    """
    Find the best pair of parting directions with axiality preference.

    Like :func:`find_best_pair` but uses a wider search window and
    a relative coverage tolerance so that pairs with nearly identical
    coverage are broken by axiality (larger angle preferred).

    A pair closer to 180° apart is mechanically superior for two-piece
    mold extraction because the two halves pull directly away from each
    other, reducing shear on the parting surface.

    Args:
        direction_scores: All scored directions
        areas: Triangle areas (n_faces,)
        total_area: Total surface area
        top_k: Number of top-scoring directions to search among
        axiality_tolerance: Fraction of total_area within which
            a pair prefers larger angle over marginally higher coverage

    Returns:
        Tuple of (best_d1, best_d2) DirectionScores
    """
    sorted_scores = sorted(
        direction_scores.values(),
        key=lambda s: s.visible_area,
        reverse=True
    )

    if len(sorted_scores) < 2:
        logger.warning("Less than 2 directions available")
        return sorted_scores[0], sorted_scores[0]

    n_scores = len(sorted_scores)

    visibility_masks, directions_array, visible_areas = _prepare_visibility_data(
        sorted_scores, areas
    )

    dot_products = directions_array @ directions_array.T

    best_d1 = sorted_scores[0]
    best_d2 = sorted_scores[1]
    best_coverage = 0.0
    best_dot = 1.0  # lower is better (more anti-parallel)

    # Relative tolerance: coverage differences within this are "equal"
    coverage_tol = total_area * axiality_tolerance

    top_k = min(n_scores, top_k)

    for i in range(top_k):
        mask1 = visibility_masks[i]

        for j in range(i + 1, n_scores):
            dot = dot_products[i, j]
            if dot > MIN_ANGLE_COS:
                continue

            # Quick upper bound pruning
            if visible_areas[i] + visible_areas[j] <= best_coverage:
                continue

            mask2 = visibility_masks[j]

            coverage_d1_primary, coverage_d2_primary = _evaluate_pair(
                mask1, mask2, visible_areas[i], visible_areas[j], areas
            )

            # Check D1-as-primary
            if (coverage_d1_primary > best_coverage + coverage_tol
                    or (abs(coverage_d1_primary - best_coverage) <= coverage_tol
                        and dot < best_dot)):
                best_coverage = coverage_d1_primary
                best_d1 = sorted_scores[i]
                best_d2 = sorted_scores[j]
                best_dot = dot

            # Check D2-as-primary
            if (coverage_d2_primary > best_coverage + coverage_tol
                    or (abs(coverage_d2_primary - best_coverage) <= coverage_tol
                        and dot < best_dot)):
                best_coverage = coverage_d2_primary
                best_d1 = sorted_scores[j]
                best_d2 = sorted_scores[i]
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


def find_parting_directions_iterative(
    mesh: trimesh.Trimesh,
    k: int = 128,
    refine: bool = True,
    num_workers: Optional[int] = None,
    progress_callback: Optional[callable] = None,
    text_callback: Optional[callable] = None
) -> PartingDirectionResult:
    """
    Find optimal parting directions using multi-resolution iterative refinement.

    Improved pipeline using numpy visibility masks and BLAS-accelerated pair
    evaluation.  The algorithm has four phases:

    **Phase 0 — Direction generation:**  Generate *k* Fibonacci-sphere
    directions, expand with antipodal copies (*2k* total), and add
    PCA-guided normal-biased extras (~18).  This ensures near-antipodal
    optimal pairs are always in the candidate set.

    **Phase 1 — Parallel raycasting:**  Evaluate all Phase-0 directions
    via Embree-accelerated raycasting.  Results are stored in a compact
    ``_VisibilityMatrix`` (boolean numpy masks) instead of Python sets.

    **Phase 2 — Cone refinement:**  Take the top-N directions by
    individual visible area plus both members of the current best pair
    and generate dense cone samples around each.

    **Phase 3 — Gap targeting:**  Identify triangles uncovered by the
    best pair, compute diverse directions from uncovered-face normals,
    and generate cone samples around them.

    Pair selection uses a BLAS overlap-matrix computation so that
    evaluating all candidate pairs is a single matrix multiply instead
    of a Python loop with per-pair set operations.  Pairs within 0.5%
    of total area prefer larger angles (axiality preference).

    Args:
        mesh: Trimesh mesh to analyze
        k: Candidate directions for coarse scan (default: 128)
        refine: Whether to run refinement phases (default: True)
        num_workers: Parallel workers (default: CPU count − 1)
        progress_callback: Optional ``callback(current, total)`` for
            direction-level progress.
        text_callback: Optional ``callback(message)`` for phase-level
            status text.

    Returns:
        PartingDirectionResult with optimal directions, statistics, and
        refinement metadata.

    Reference:
        Paper Section 4.1 — extended with vectorised multi-resolution search.
    """
    start_time = time.perf_counter()

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info("Iterative parting direction search: k=%d, refine=%s", k, refine)
    logger.info("Mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
    logger.info("Ray tracer: %s (Embree: %s)", type(mesh.ray).__name__, EMBREE_AVAILABLE)

    triangle_data = MeshTriangleData.from_mesh(mesh)
    total_area = triangle_data.total_area
    n_faces = len(triangle_data.areas)
    areas = triangle_data.areas

    # ------------------------------------------------------------------ #
    # Phase 0: Direction generation (Fibonacci + antipodal + PCA-biased)  #
    # ------------------------------------------------------------------ #
    base_dirs = fibonacci_sphere(k)

    # Antipodal expansion: for each d, also test -d  (doubles to 2k)
    if ANTIPODAL_EXPANSION:
        phase1_dirs = _expand_with_antipodal(base_dirs)
    else:
        phase1_dirs = base_dirs

    # PCA-guided normal-biased extras
    normal_dirs = _generate_normal_biased_directions(triangle_data, NORMAL_BIAS_EXTRA)
    if len(normal_dirs) > 0:
        phase1_dirs = np.vstack([phase1_dirs, normal_dirs])

    total_evaluated = len(phase1_dirs)

    # ------------------------------------------------------------------ #
    # Phase 1: Parallel raycasting → visibility matrix                    #
    # ------------------------------------------------------------------ #
    if text_callback:
        text_callback(f"Phase 1: Scanning {total_evaluated} directions "
                      f"({k} base + {total_evaluated - k} expanded)...")

    vm = _compute_visibility_matrix(
        phase1_dirs, mesh, triangle_data, num_workers, progress_callback
    )

    # Baseline pair via BLAS overlap matrix
    baseline_i, baseline_j = _find_best_pair_from_matrix(
        vm, areas, total_area
    )
    baseline_coverage = vm.pair_coverage_pct(baseline_i, baseline_j, areas, total_area)
    logger.info("Phase 1 baseline: %.1f%% coverage (%d directions)",
                baseline_coverage, total_evaluated)

    if not refine:
        # Non-iterative path
        best_d1, best_d2 = vm.best_pair_scores(baseline_i, baseline_j)
        return _build_iterative_result(
            best_d1, best_d2, triangle_data, start_time,
            total_evaluated, refinement_rounds=0
        )

    # ------------------------------------------------------------------ #
    # Phase 2: Cone refinement around top directions + best pair          #
    # ------------------------------------------------------------------ #
    top_order = np.argsort(-vm.visible_areas)
    refine_centers = set()
    for idx in top_order[:REFINE_TOP_N_DIRECTIONS]:
        refine_centers.add(tuple(vm.directions[idx]))
    refine_centers.add(tuple(vm.directions[baseline_i]))
    refine_centers.add(tuple(vm.directions[baseline_j]))
    # Also add antipodals of baseline pair as refinement seeds
    refine_centers.add(tuple(-vm.directions[baseline_i]))
    refine_centers.add(tuple(-vm.directions[baseline_j]))

    new_directions: List[np.ndarray] = []
    for center in refine_centers:
        cone = generate_cone_directions(
            np.array(center), REFINE_CONE_HALF_ANGLE_DEG, REFINE_SAMPLES_PER_CONE
        )
        new_directions.append(cone)

    phase2_dirs = np.vstack(new_directions)
    n_phase2 = len(phase2_dirs)
    total_evaluated += n_phase2

    if text_callback:
        text_callback(f"Phase 2: Refining around {len(refine_centers)} "
                      f"centres ({n_phase2} samples)...")

    vm2 = _compute_visibility_matrix(
        phase2_dirs, mesh, triangle_data, num_workers, progress_callback
    )
    vm = vm.merge(vm2)

    # ------------------------------------------------------------------ #
    # Phase 3: Gap targeting                                              #
    # ------------------------------------------------------------------ #
    mid_i, mid_j = _find_best_pair_from_matrix(vm, areas, total_area)
    covered_mask = vm.masks[mid_i] | vm.masks[mid_j]
    uncovered_pct = float((~covered_mask).sum()) / n_faces * 100
    logger.info("After Phase 2: %.1f%% faces still uncovered", uncovered_pct)

    if uncovered_pct > 1.0:
        gap_dirs = _generate_gap_target_directions(
            triangle_data, covered_mask,
            n_targets=REFINE_GAP_TARGETS,
            cone_samples=REFINE_GAP_CONE_SAMPLES,
        )
        if len(gap_dirs) > 0:
            if text_callback:
                text_callback(f"Phase 3: Targeting {uncovered_pct:.0f}% "
                              f"uncovered area ({len(gap_dirs)} samples)...")
            total_evaluated += len(gap_dirs)

            vm3 = _compute_visibility_matrix(
                gap_dirs, mesh, triangle_data, num_workers, progress_callback
            )
            vm = vm.merge(vm3)
    else:
        logger.info("Phase 3 skipped — uncovered area ≤ 1%%")

    # ------------------------------------------------------------------ #
    # Final pair selection (BLAS overlap matrix + axiality preference)     #
    # ------------------------------------------------------------------ #
    if text_callback:
        text_callback("Selecting best direction pair...")

    best_i, best_j = _find_best_pair_from_matrix(vm, areas, total_area)
    best_d1, best_d2 = vm.best_pair_scores(best_i, best_j)
    final_coverage = vm.pair_coverage_pct(best_i, best_j, areas, total_area)

    # ---- No-regression safeguard ----
    baseline_d1_score, baseline_d2_score = vm.best_pair_scores(baseline_i, baseline_j)
    coverage_margin = 0.1
    if final_coverage < baseline_coverage + coverage_margin:
        baseline_dot = float(np.dot(
            vm.directions[baseline_i], vm.directions[baseline_j]))
        refined_dot = float(np.dot(
            vm.directions[best_i], vm.directions[best_j]))
        if baseline_dot < refined_dot - 1e-6:
            logger.info(
                "Keeping baseline pair (angle %.1f° vs %.1f°, similar coverage)",
                np.degrees(np.arccos(np.clip(baseline_dot, -1, 1))),
                np.degrees(np.arccos(np.clip(refined_dot, -1, 1))),
            )
            best_d1, best_d2 = baseline_d1_score, baseline_d2_score
            final_coverage = baseline_coverage

    improvement = final_coverage - baseline_coverage
    logger.info(
        "Refinement: %.1f%% → %.1f%% (+%.2f%%), %d total candidates",
        baseline_coverage, final_coverage, improvement, total_evaluated,
    )

    return _build_iterative_result(
        best_d1, best_d2, triangle_data, start_time,
        total_evaluated, refinement_rounds=1,
    )


# =============================================================================
# INTERNAL HELPERS FOR ITERATIVE SEARCH
# =============================================================================

def _pair_coverage(
    d1: DirectionScore,
    d2: DirectionScore,
    areas: np.ndarray,
    total_area: float
) -> float:
    """
    Compute total coverage percentage for a direction pair.

    Args:
        d1: Primary direction score
        d2: Secondary direction score
        areas: Triangle areas
        total_area: Total surface area

    Returns:
        Coverage as a percentage (0-100)
    """
    d2_unique_area = sum(
        areas[tri] for tri in d2.visible_triangles
        if tri not in d1.visible_triangles
    )
    return (d1.visible_area + d2_unique_area) / total_area * 100


def _build_iterative_result(
    best_d1: DirectionScore,
    best_d2: DirectionScore,
    triangle_data: MeshTriangleData,
    start_time: float,
    total_evaluated: int,
    refinement_rounds: int
) -> PartingDirectionResult:
    """
    Build a PartingDirectionResult from the selected pair.

    Direction vectors are negated so they point in the extraction
    (pull) direction, consistent with downstream pipeline expectations.
    """
    areas = triangle_data.areas
    total_area = triangle_data.total_area

    d1 = -best_d1.direction
    d2 = -best_d2.direction

    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1, 1))))

    d2_unique_area = sum(
        areas[tri] for tri in best_d2.visible_triangles
        if tri not in best_d1.visible_triangles
    )
    total_coverage = (best_d1.visible_area + d2_unique_area) / total_area * 100
    uncovered_pct = 100.0 - total_coverage

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
        d2_triangles=best_d2.visible_triangles,
        refinement_rounds=refinement_rounds,
        total_candidates_evaluated=total_evaluated,
        uncovered_area_pct=uncovered_pct,
    )

    logger.info("Parting directions (iterative):")
    logger.info("  D1: [%.3f, %.3f, %.3f] (%.1f%%)",
                d1[0], d1[1], d1[2],
                best_d1.visible_area / total_area * 100)
    logger.info("  D2: [%.3f, %.3f, %.3f] (%.1f%% unique)",
                d2[0], d2[1], d2[2],
                d2_unique_area / total_area * 100)
    logger.info("  Coverage: %.1f%%, Angle: %.1f°, Candidates: %d",
                total_coverage, angle_deg, total_evaluated)
    logger.info("  Time: %.0fms", elapsed_ms)

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
