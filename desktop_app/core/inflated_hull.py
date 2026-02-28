"""
Inflated Convex Hull Module

Creates an inflated convex hull bounding volume around a mesh.
This is the bounding volume ∂H from Paper Section 3.

Algorithm:
1. Compute smooth (area-weighted) vertex normals on the input mesh
2. Inflate vertices outward along their normals by an offset distance
3. Compute the convex hull of the inflated vertices
4. Optionally subdivide for smoother results

This approach produces a denser hull because the input mesh has many more
vertices than a convex hull would, resulting in better triangle distribution
around concave features of the part.

Reference:
    Paper Section 3: "Mold volume O is the space between M and ∂H"
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull
import trimesh


logger = logging.getLogger(__name__)


# =============================================================================
# GPU ACCELERATION DETECTION
# =============================================================================

def _check_torch_available() -> Tuple[bool, bool]:
    """Check if PyTorch and CUDA are available."""
    try:
        import torch
        return True, torch.cuda.is_available()
    except ImportError:
        return False, False


TORCH_AVAILABLE, CUDA_AVAILABLE = _check_torch_available()


# =============================================================================
# CONSTANTS
# =============================================================================

# Default inflation offset as percentage of bounding box diagonal
DEFAULT_INFLATION_PERCENT = 0.10  # 10% of bbox diagonal

# Number of subdivision iterations for smoother hull
DEFAULT_SUBDIVISION_ITERATIONS = 2

# Backward compatibility alias
SUBDIVISION_ITERATIONS = DEFAULT_SUBDIVISION_ITERATIONS

# GPU threshold: use GPU for meshes with more faces than this
GPU_FACE_THRESHOLD = 10000

# Numerical stability epsilon for normalization
NORMALIZATION_EPSILON = 1e-10


# =============================================================================
# VALIDATION
# =============================================================================

class HullGenerationError(Exception):
    """Raised when hull generation fails."""
    pass


def validate_input_mesh(mesh: trimesh.Trimesh) -> None:
    """
    Validate that the input mesh is suitable for hull generation.
    
    Args:
        mesh: The mesh to validate
        
    Raises:
        HullGenerationError: If the mesh is invalid
    """
    if mesh is None:
        raise HullGenerationError("Input mesh is None")
    
    if not hasattr(mesh, 'vertices') or mesh.vertices is None:
        raise HullGenerationError("Mesh has no vertices")
    
    if len(mesh.vertices) < 4:
        raise HullGenerationError(
            "Mesh needs at least 4 vertices for convex hull, "
            "got %d" % len(mesh.vertices)
        )
    
    if not hasattr(mesh, 'faces') or mesh.faces is None or len(mesh.faces) == 0:
        raise HullGenerationError("Mesh has no faces")


def validate_offset(offset: float) -> None:
    """
    Validate that the offset value is reasonable.
    
    Args:
        offset: The inflation offset
        
    Raises:
        HullGenerationError: If the offset is invalid
    """
    if offset <= 0:
        raise HullGenerationError("Offset must be positive, got %f" % offset)


# =============================================================================
# OFFSET COMPUTATION
# =============================================================================

def compute_default_offset(
    mesh: trimesh.Trimesh,
    inflation_percent: float = DEFAULT_INFLATION_PERCENT
) -> float:
    """
    Compute the default inflation offset as a percentage of bounding box diagonal.
    
    Args:
        mesh: The input mesh
        inflation_percent: Fraction of bbox diagonal (default: 0.10 = 10%)
        
    Returns:
        Default offset value in mesh units
        
    Example:
        >>> offset = compute_default_offset(mesh)  # 10% of diagonal
        >>> offset = compute_default_offset(mesh, 0.15)  # 15% of diagonal
    """
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    diagonal = np.linalg.norm(dimensions)
    return diagonal * inflation_percent


def compute_bounding_box_diagonal(mesh: trimesh.Trimesh) -> float:
    """
    Compute the bounding box diagonal length.
    
    Args:
        mesh: The input mesh
        
    Returns:
        Length of the bounding box diagonal
    """
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    return float(np.linalg.norm(dimensions))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ManifoldValidation:
    """
    Results of manifold validation check.
    
    Attributes:
        is_closed: True if mesh is watertight (no boundary edges)
        is_manifold: True if each edge shared by exactly 2 faces
        boundary_edge_count: Number of edges with only 1 adjacent face
        non_manifold_edge_count: Number of edges with >2 adjacent faces
        total_edge_count: Total unique edges in the mesh
        euler_characteristic: V - E + F (should be 2 for closed surface)
    """
    is_closed: bool
    is_manifold: bool
    boundary_edge_count: int
    non_manifold_edge_count: int
    total_edge_count: int
    euler_characteristic: int
    
    @property
    def is_valid_hull(self) -> bool:
        """Return True if this is a valid hull (closed and manifold)."""
        return self.is_closed and self.is_manifold


@dataclass
class InflatedHullResult:
    """
    Result of inflated hull computation.
    
    Contains the inflated convex hull mesh (∂H) and associated metadata.
    
    Attributes:
        mesh: The inflated convex hull mesh (trimesh.Trimesh)
        original_hull: The non-inflated convex hull for reference
        smooth_normals: Area-weighted vertex normals used for inflation (N, 3)
        vertex_count: Number of vertices in the inflated hull
        face_count: Number of faces in the inflated hull
        manifold_validation: Validation results for the hull
        offset: The inflation offset distance used
    """
    mesh: trimesh.Trimesh
    original_hull: trimesh.Trimesh
    smooth_normals: np.ndarray
    vertex_count: int
    face_count: int
    manifold_validation: ManifoldValidation
    offset: float
    
    @property
    def is_valid(self) -> bool:
        """Return True if the hull is valid for further processing."""
        return (
            self.mesh is not None and
            self.vertex_count > 0 and
            self.face_count > 0 and
            self.manifold_validation.is_valid_hull
        )


# =============================================================================
# SMOOTH NORMAL COMPUTATION
# =============================================================================

def compute_smooth_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Compute area-weighted smooth vertex normals.
    
    Each vertex normal is the normalized sum of the normals of all faces
    that share that vertex, weighted by face area. This produces smooth
    normals suitable for vertex inflation.
    
    Uses GPU acceleration when available for large meshes.
    
    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of face vertex indices
        use_gpu: Whether to use GPU acceleration if available
        
    Returns:
        (N, 3) array of normalized vertex normals
    """
    # Use GPU for large meshes (above threshold)
    if use_gpu and CUDA_AVAILABLE and len(faces) > GPU_FACE_THRESHOLD:
        return _compute_smooth_vertex_normals_gpu(vertices, faces)
    else:
        return _compute_smooth_vertex_normals_cpu(vertices, faces)


def _compute_smooth_vertex_normals_gpu(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    GPU-accelerated smooth vertex normal computation.
    
    Fully vectorized using PyTorch CUDA. Computes area-weighted
    vertex normals by accumulating face normals at each vertex.
    
    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of face vertex indices
        
    Returns:
        (N, 3) array of normalized vertex normals
    """
    import torch
    
    device = torch.device('cuda')
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    logger.debug(
        "Computing smooth normals on GPU for %d vertices, %d faces",
        num_vertices, num_faces
    )
    
    # Move data to GPU
    verts_t = torch.tensor(vertices, dtype=torch.float64, device=device)
    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)
    
    # Get triangle vertices: (F, 3, 3)
    v0 = verts_t[faces_t[:, 0]]  # (F, 3)
    v1 = verts_t[faces_t[:, 1]]  # (F, 3)
    v2 = verts_t[faces_t[:, 2]]  # (F, 3)
    
    # Compute face normals via cross product: edge1 x edge2
    edge1 = v1 - v0  # (F, 3)
    edge2 = v2 - v0  # (F, 3)
    face_normals = torch.cross(edge1, edge2, dim=1)  # (F, 3)
    
    # Compute areas (half the magnitude of cross product)
    face_normal_lengths = torch.norm(face_normals, dim=1, keepdim=True)  # (F, 1)
    areas = face_normal_lengths / 2.0  # (F, 1)
    
    # Normalize face normals (avoid div by zero)
    eps = NORMALIZATION_EPSILON
    face_normals_normalized = face_normals / (face_normal_lengths + eps)  # (F, 3)
    
    # Area-weighted normals for accumulation
    weighted_normals = face_normals_normalized * areas  # (F, 3)
    
    # Accumulate normals at each vertex using scatter_add
    normal_accum = torch.zeros((num_vertices, 3), dtype=torch.float64, device=device)
    
    # For each corner of each face, accumulate the weighted normal
    for corner in range(3):
        corner_indices = faces_t[:, corner].unsqueeze(1).expand(-1, 3)  # (F, 3)
        normal_accum.scatter_add_(0, corner_indices, weighted_normals)
    
    # Normalize the accumulated normals
    norms = torch.norm(normal_accum, dim=1, keepdim=True)  # (V, 1)
    norms = torch.where(norms > eps, norms, torch.ones_like(norms))
    smooth_normals = normal_accum / norms
    
    result = smooth_normals.cpu().numpy().astype(np.float32)
    torch.cuda.empty_cache()
    
    logger.debug("GPU smooth normals computed")
    return result


def _compute_smooth_vertex_normals_cpu(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    CPU implementation of smooth vertex normal computation.
    
    Vectorized NumPy implementation. Computes area-weighted
    vertex normals by accumulating face normals at each vertex.
    
    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of face vertex indices
        
    Returns:
        (N, 3) array of normalized vertex normals
    """
    num_vertices = len(vertices)
    eps = NORMALIZATION_EPSILON
    
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Compute face normals via cross product
    edge1 = v1 - v0  # (F, 3)
    edge2 = v2 - v0  # (F, 3)
    face_normals = np.cross(edge1, edge2)  # (F, 3)
    
    # Compute areas (half the magnitude of cross product)
    face_normal_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)  # (F, 1)
    areas = face_normal_lengths / 2.0  # (F, 1)
    
    # Normalize face normals
    face_normals_normalized = face_normals / (face_normal_lengths + eps)  # (F, 3)
    
    # Area-weighted normals
    weighted_normals = face_normals_normalized * areas  # (F, 3)
    
    # Accumulate at each vertex using np.add.at
    normal_accum = np.zeros((num_vertices, 3), dtype=np.float64)
    
    # Add weighted normal to each corner vertex
    np.add.at(normal_accum, faces[:, 0], weighted_normals)
    np.add.at(normal_accum, faces[:, 1], weighted_normals)
    np.add.at(normal_accum, faces[:, 2], weighted_normals)
    
    # Normalize
    norms = np.linalg.norm(normal_accum, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    smooth_normals = normal_accum / norms
    
    return smooth_normals.astype(np.float32)


def inflate_vertices(
    vertices: np.ndarray,
    normals: np.ndarray,
    offset: float
) -> np.ndarray:
    """
    Inflate vertices outward along their normals.
    
    Args:
        vertices: Nx3 array of vertex positions
        normals: Nx3 array of vertex normals (normalized)
        offset: Distance to move each vertex along its normal
        
    Returns:
        Nx3 array of inflated vertex positions
    """
    return vertices + normals * offset


def subdivide_mesh(mesh: trimesh.Trimesh, iterations: int = 1) -> trimesh.Trimesh:
    """
    Subdivide mesh to increase triangle density.
    
    Uses trimesh's built-in subdivision which splits each triangle into 4.
    
    Args:
        mesh: Input mesh to subdivide
        iterations: Number of subdivision iterations (each iteration 4x triangles)
        
    Returns:
        Subdivided mesh
    """
    result_mesh = mesh
    for _ in range(iterations):
        # Use Loop subdivision for smooth results
        result_mesh = result_mesh.subdivide()
    return result_mesh


def validate_manifold(mesh: trimesh.Trimesh) -> ManifoldValidation:
    """
    Validate manifold properties of a mesh.
    
    Checks if mesh is:
    - Closed (watertight, no boundary edges)
    - Manifold (each edge shared by exactly 2 faces)
    
    Args:
        mesh: Mesh to validate
        
    Returns:
        ManifoldValidation with detailed results
    """
    # Get edge info
    edges = mesh.edges_unique
    total_edge_count = len(edges) if edges is not None else 0
    
    # Use trimesh's built-in checks
    is_closed = bool(mesh.is_watertight)
    
    # For convex hulls, we can check manifoldness via edge adjacency
    # trimesh.Trimesh has is_watertight but not a direct is_manifold
    # A watertight mesh is manifold if it has no non-manifold edges
    is_manifold = is_closed  # For convex hulls, watertight implies manifold
    
    # Count boundary and non-manifold edges using trimesh internals
    boundary_edge_count = 0
    non_manifold_edge_count = 0
    
    # Try to get edge face adjacency for more accurate counts
    try:
        if hasattr(mesh, 'edges_unique_length') and hasattr(mesh, 'faces_unique_edges'):
            # Get face count per edge
            edge_face_counts = np.bincount(
                mesh.faces_unique_edges.flatten(),
                minlength=total_edge_count
            )
            boundary_edge_count = int(np.sum(edge_face_counts == 1))
            non_manifold_edge_count = int(np.sum(edge_face_counts > 2))
    except Exception:
        logger.debug("Edge face adjacency computation failed, using defaults", exc_info=True)
    
    # Compute Euler characteristic: V - E + F
    euler_characteristic = len(mesh.vertices) - total_edge_count + len(mesh.faces)
    
    return ManifoldValidation(
        is_closed=is_closed,
        is_manifold=is_manifold,
        boundary_edge_count=boundary_edge_count,
        non_manifold_edge_count=non_manifold_edge_count,
        total_edge_count=total_edge_count,
        euler_characteristic=euler_characteristic
    )


# =============================================================================
# CONVEX HULL HELPERS
# =============================================================================

def _compute_convex_hull_mesh(
    vertices: np.ndarray,
    description: str = "hull"
) -> Optional[trimesh.Trimesh]:
    """
    Compute convex hull from vertices and return as trimesh.
    
    Args:
        vertices: (N, 3) array of vertex positions
        description: Description for logging
        
    Returns:
        Trimesh of the convex hull, or None if computation fails
    """
    try:
        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]
        
        # Reindex faces to use only hull vertices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
        hull_faces = np.array([
            [vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]]]
            for f in hull.simplices
        ], dtype=np.int32)
        
        # Create trimesh from hull
        hull_mesh = trimesh.Trimesh(
            vertices=hull_vertices,
            faces=hull_faces,
            process=True
        )
        
        # Fix face winding to ensure consistent normals (outward-facing)
        hull_mesh.fix_normals()
        
        return hull_mesh
        
    except Exception as e:
        logger.warning("ConvexHull computation failed for %s: %s", description, e)
        return None


# =============================================================================
# MAIN API
# =============================================================================

def generate_inflated_hull(
    mesh: trimesh.Trimesh,
    offset: Optional[float] = None,
    subdivisions: int = DEFAULT_SUBDIVISION_ITERATIONS,
    validate: bool = True
) -> InflatedHullResult:
    """
    Generate an inflated convex hull bounding volume around a mesh.
    
    This creates the bounding hull ∂H from Paper Section 3. The mold volume O
    is the space between the part mesh M and this hull ∂H.
    
    Algorithm:
    1. Compute smooth (area-weighted) vertex normals on the input mesh
    2. Inflate vertices outward along their normals by offset distance
    3. Compute convex hull from the inflated vertices
    4. Subdivide the hull for smoother results
    
    This produces a denser hull because the input mesh has many more vertices
    than a simple convex hull, resulting in better triangle distribution.
    
    Args:
        mesh: Input mesh (part M) to create hull around
        offset: Distance to inflate outward (default: 10% of bbox diagonal)
        subdivisions: Number of subdivision iterations (default: 2)
        validate: Whether to validate input mesh (default: True)
        
    Returns:
        InflatedHullResult containing the inflated hull mesh and metadata
        
    Raises:
        HullGenerationError: If hull generation fails
        
    Reference:
        Paper Section 3: Mold volume O = space between M and ∂H
    """
    # Validate input
    if validate:
        validate_input_mesh(mesh)
    
    # Compute default offset if not provided
    if offset is None:
        offset = compute_default_offset(mesh)
    else:
        validate_offset(offset)
    
    logger.info(
        "Generating inflated hull: offset=%.4f, subdivisions=%d",
        offset, subdivisions
    )
    
    # Get vertices and faces as numpy arrays
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    logger.info(
        "Input mesh: %d vertices, %d faces",
        len(vertices), len(faces)
    )
    
    # Step 1: Compute smooth vertex normals on the INPUT mesh
    smooth_normals = compute_smooth_vertex_normals(vertices, faces)
    logger.info("Computed smooth normals for %d vertices", len(smooth_normals))
    
    # Step 2: Inflate input vertices outward along their smooth normals
    inflated_vertices = inflate_vertices(vertices, smooth_normals, offset)
    
    # Step 3: Compute convex hull from the inflated vertices
    inflated_hull_mesh = _compute_convex_hull_mesh(
        inflated_vertices, "inflated hull"
    )
    
    if inflated_hull_mesh is None:
        raise HullGenerationError("Failed to compute inflated convex hull")
    
    logger.info(
        "Initial hull: %d vertices, %d faces",
        len(inflated_hull_mesh.vertices), len(inflated_hull_mesh.faces)
    )
    
    # Step 4: Subdivide the hull for smoother results
    if subdivisions > 0:
        inflated_hull_mesh = subdivide_mesh(inflated_hull_mesh, subdivisions)
        logger.info(
            "After subdivision: %d vertices, %d faces",
            len(inflated_hull_mesh.vertices), len(inflated_hull_mesh.faces)
        )
    
    # Step 5: Compute the original (non-inflated) convex hull for reference
    original_hull_mesh = _compute_convex_hull_mesh(vertices, "original hull")
    if original_hull_mesh is None:
        logger.warning("Original hull computation failed, using empty mesh")
        original_hull_mesh = trimesh.Trimesh()
    
    # Step 6: Validate manifold properties
    manifold_validation = validate_manifold(inflated_hull_mesh)
    
    logger.info(
        "Inflated hull generated: %d vertices, %d faces, "
        "is_closed=%s, is_manifold=%s",
        len(inflated_hull_mesh.vertices),
        len(inflated_hull_mesh.faces),
        manifold_validation.is_closed,
        manifold_validation.is_manifold
    )
    
    return InflatedHullResult(
        mesh=inflated_hull_mesh,
        original_hull=original_hull_mesh,
        smooth_normals=smooth_normals,
        vertex_count=len(inflated_hull_mesh.vertices),
        face_count=len(inflated_hull_mesh.faces),
        manifold_validation=manifold_validation,
        offset=offset
    )
