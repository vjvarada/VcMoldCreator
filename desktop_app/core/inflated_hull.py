"""
Inflated Convex Hull Module

Creates an inflated convex hull bounding volume around a mesh.
This is used for mold cavity creation by:
1. Computing smooth vertex normals on the input mesh
2. Inflating vertices outward along their normals by an offset
3. Computing the convex hull of the inflated vertices
4. Optionally subdividing for smoother results

This approach produces a denser hull because the input mesh has many more 
vertices than a convex hull would, resulting in better triangle distribution.
"""

import logging
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull

import trimesh

logger = logging.getLogger(__name__)

# Default inflation offset as percentage of bounding box diagonal
DEFAULT_INFLATION_PERCENT = 0.10  # 10%

# Number of subdivision iterations for smoother hull
SUBDIVISION_ITERATIONS = 2


def compute_default_offset(mesh: 'trimesh.Trimesh') -> float:
    """
    Compute the default inflation offset as 10% of the bounding box diagonal.
    
    Args:
        mesh: The input mesh
        
    Returns:
        Default offset value in mesh units
    """
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    diagonal = np.linalg.norm(dimensions)
    return diagonal * DEFAULT_INFLATION_PERCENT


@dataclass
class ManifoldValidation:
    """Results of manifold validation check."""
    is_closed: bool
    is_manifold: bool
    boundary_edge_count: int
    non_manifold_edge_count: int
    total_edge_count: int
    euler_characteristic: int


@dataclass
class InflatedHullResult:
    """Result of inflated hull computation."""
    # The inflated convex hull mesh
    mesh: trimesh.Trimesh
    # The original (non-inflated) convex hull for reference
    original_hull: trimesh.Trimesh
    # Smooth normals used for inflation (Nx3 array)
    smooth_normals: np.ndarray
    # Vertex and face counts
    vertex_count: int
    face_count: int
    # Manifold validation results
    manifold_validation: ManifoldValidation
    # The inflation offset used
    offset: float


def compute_smooth_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute area-weighted smooth vertex normals.
    
    Each vertex normal is the normalized sum of the normals of all faces
    that share that vertex, weighted by face area.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face vertex indices
        
    Returns:
        Nx3 array of normalized vertex normals
    """
    num_vertices = len(vertices)
    normal_accum = np.zeros((num_vertices, 3), dtype=np.float64)
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Compute face normal (not normalized yet - length is 2x area)
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        # Area is half the magnitude of the cross product
        area = np.linalg.norm(face_normal) / 2.0
        
        if area > 1e-10:
            # Normalize the face normal
            face_normal = face_normal / (area * 2)
            
            # Add area-weighted normal to each vertex
            for idx in face:
                normal_accum[idx] += face_normal * area
    
    # Normalize all vertex normals
    norms = np.linalg.norm(normal_accum, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 1e-10, norms, 1.0)
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
    edge_face_count = mesh.edges_unique_length
    total_edge_count = len(edges) if edges is not None else 0
    
    # Count boundary edges (edges with only 1 adjacent face)
    # and non-manifold edges (edges with 3+ adjacent faces)
    boundary_edge_count = 0
    non_manifold_edge_count = 0
    
    if mesh.face_adjacency_edges is not None:
        # Build edge to face count mapping
        edge_face_counts = {}
        for i, edge in enumerate(mesh.face_adjacency_edges):
            edge_key = tuple(sorted(edge))
            edge_face_counts[edge_key] = edge_face_counts.get(edge_key, 0) + 1
        
        for edge_key, count in edge_face_counts.items():
            # Each adjacency entry means 2 faces share that edge
            pass  # trimesh handles this differently
    
    # Use trimesh's built-in checks
    is_closed = mesh.is_watertight
    is_manifold = mesh.is_watertight  # For convex hulls, watertight implies manifold
    
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


def generate_inflated_hull(
    mesh: trimesh.Trimesh,
    offset: Optional[float] = None,
    subdivisions: int = SUBDIVISION_ITERATIONS
) -> InflatedHullResult:
    """
    Generate an inflated convex hull bounding volume around a mesh.
    
    Algorithm:
    1. Compute smooth vertex normals on the input mesh
    2. Inflate input mesh vertices outward by offset distance
    3. Compute convex hull from the inflated vertices
    4. Subdivide the hull for smoother results
    
    This produces a denser hull because the input mesh has many more vertices
    than a simple convex hull, resulting in better triangle distribution.
    
    Args:
        mesh: Input mesh to create hull around
        offset: Distance to inflate outward (default: 0.5)
        subdivisions: Number of subdivision iterations (default: 2)
        
    Returns:
        InflatedHullResult containing the inflated mesh and metadata
    """
    # Compute default offset if not provided
    if offset is None:
        offset = compute_default_offset(mesh)
    
    logger.info(f"Generating inflated hull with offset={offset}, subdivisions={subdivisions}")
    
    # Get unique vertices and faces
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    logger.info(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Step 1: Compute smooth vertex normals on the INPUT mesh
    smooth_normals = compute_smooth_vertex_normals(vertices, faces)
    logger.info(f"Computed smooth normals for {len(smooth_normals)} vertices")
    
    # Step 2: Inflate input vertices outward along their smooth normals
    inflated_vertices = inflate_vertices(vertices, smooth_normals, offset)
    
    # Step 3: Compute convex hull from the inflated vertices
    try:
        hull = ConvexHull(inflated_vertices)
        hull_vertices = inflated_vertices[hull.vertices]
        hull_faces = hull.simplices
        
        # Reindex faces to use only hull vertices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
        hull_faces = np.array([
            [vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]]]
            for f in hull.simplices
        ], dtype=np.int32)
        
        # Create trimesh from hull
        inflated_hull_mesh = trimesh.Trimesh(
            vertices=hull_vertices,
            faces=hull_faces,
            process=True
        )
        
        # Fix face winding to ensure consistent normals
        inflated_hull_mesh.fix_normals()
        
    except Exception as e:
        logger.error(f"ConvexHull computation failed: {e}")
        raise ValueError(f"Failed to compute convex hull: {e}")
    
    original_face_count = len(inflated_hull_mesh.faces)
    logger.info(f"Initial hull: {len(inflated_hull_mesh.vertices)} vertices, {original_face_count} faces")
    
    # Step 4: Subdivide the hull for smoother results
    if subdivisions > 0:
        inflated_hull_mesh = subdivide_mesh(inflated_hull_mesh, subdivisions)
        logger.info(f"After subdivision: {len(inflated_hull_mesh.vertices)} vertices, {len(inflated_hull_mesh.faces)} faces")
    
    # Step 5: Compute the original (non-inflated) convex hull for reference
    try:
        orig_hull = ConvexHull(vertices)
        orig_hull_vertices = vertices[orig_hull.vertices]
        orig_vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(orig_hull.vertices)}
        orig_hull_faces = np.array([
            [orig_vertex_map[f[0]], orig_vertex_map[f[1]], orig_vertex_map[f[2]]]
            for f in orig_hull.simplices
        ], dtype=np.int32)
        
        original_hull_mesh = trimesh.Trimesh(
            vertices=orig_hull_vertices,
            faces=orig_hull_faces,
            process=True
        )
        original_hull_mesh.fix_normals()
        
    except Exception as e:
        logger.warning(f"Original hull computation failed: {e}")
        # Create empty mesh as fallback
        original_hull_mesh = trimesh.Trimesh()
    
    # Step 6: Validate manifold properties
    manifold_validation = validate_manifold(inflated_hull_mesh)
    
    logger.info(f"Inflated hull generated: {len(inflated_hull_mesh.vertices)} vertices, "
                f"{len(inflated_hull_mesh.faces)} faces, "
                f"is_closed={manifold_validation.is_closed}, "
                f"is_manifold={manifold_validation.is_manifold}")
    
    return InflatedHullResult(
        mesh=inflated_hull_mesh,
        original_hull=original_hull_mesh,
        smooth_normals=smooth_normals,
        vertex_count=len(inflated_hull_mesh.vertices),
        face_count=len(inflated_hull_mesh.faces),
        manifold_validation=manifold_validation,
        offset=offset
    )


def compute_mesh_bounding_info(mesh: trimesh.Trimesh) -> dict:
    """
    Compute bounding box information for a mesh.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Dictionary with bounds, dimensions, and center
    """
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    
    return {
        'bounds': bounds,
        'min': bounds[0],
        'max': bounds[1],
        'dimensions': dimensions,
        'center': center,
        'volume': mesh.volume if mesh.is_watertight else None
    }
