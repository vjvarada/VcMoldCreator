"""
Tetrahedral Mesh Generation for Mold Volume

This module generates a tetrahedral mesh of the mold cavity volume using fTetWild
(via pytetwild). The tetrahedral mesh replaces the voxel grid approach for:
- More accurate volume representation
- Better boundary conforming
- Edge-based weight assignment for parting surface computation

Algorithm:
1. Take the cavity mesh (Hull - Part CSG result)
2. Tetrahedralize the interior volume using fTetWild
3. Build edge connectivity for weight assignment
4. Compute distances from tet vertices/edges to part surface and shell boundary

The edge weights are used for Dijkstra-based escape labeling to find the parting surface.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import trimesh

try:
    import pytetwild
    PYTETWILD_AVAILABLE = True
except ImportError:
    PYTETWILD_AVAILABLE = False

# Check for GPU acceleration options
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TetrahedralMeshResult:
    """Result of tetrahedral mesh generation."""
    
    # Vertex positions (N x 3)
    vertices: np.ndarray
    
    # Tetrahedra indices (M x 4) - each row is 4 vertex indices
    tetrahedra: np.ndarray
    
    # Edge list (E x 2) - unique edges extracted from tetrahedra
    edges: np.ndarray
    
    # Edge lengths (E,)
    edge_lengths: np.ndarray
    
    # Boundary surface mesh (extracted from tetrahedral mesh outer faces)
    boundary_mesh: Optional[trimesh.Trimesh] = None
    
    # Edge midpoint distances to part mesh (E,) - for weight computation
    edge_dist_to_part: Optional[np.ndarray] = None
    
    # Edge weights (E,) - weight = 1 / (dist^2 + 0.25)
    edge_weights: Optional[np.ndarray] = None
    
    # Weighted edge lengths (E,) = edge_length * edge_weight
    weighted_edge_lengths: Optional[np.ndarray] = None
    
    # Boundary vertex mask - True if vertex is on boundary surface (N,)
    boundary_vertices: Optional[np.ndarray] = None
    
    # Shell boundary vertex labels: 0=interior, 1=H1, 2=H2, -1=inner boundary (N,)
    boundary_labels: Optional[np.ndarray] = None
    
    # R value - maximum distance from hull vertices to part surface
    r_value: Optional[float] = None
    r_hull_point: Optional[np.ndarray] = None  # Point on hull with max distance
    r_part_point: Optional[np.ndarray] = None  # Closest point on part surface
    
    # Whether the mesh has been inflated
    is_inflated: bool = False
    
    # Statistics
    num_vertices: int = 0
    num_tetrahedra: int = 0
    num_edges: int = 0
    num_boundary_faces: int = 0
    
    # Timing
    tetrahedralize_time_ms: float = 0.0
    total_time_ms: float = 0.0


def tetrahedralize_mesh(
    cavity_mesh: trimesh.Trimesh,
    edge_length_fac: float = 0.05,
    optimize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tetrahedral mesh from a surface mesh using fTetWild.
    
    Args:
        cavity_mesh: The cavity surface mesh (should be watertight)
        edge_length_fac: Target edge length as fraction of bounding box diagonal
                        Default 0.05 = bbox/20. Smaller = finer mesh.
        optimize: Whether to optimize mesh quality (slower but better quality)
    
    Returns:
        Tuple of (vertices, tetrahedra)
        - vertices: (N, 3) float64 array of vertex positions
        - tetrahedra: (M, 4) int32 array of tetrahedron vertex indices
    """
    if not PYTETWILD_AVAILABLE:
        raise ImportError("pytetwild is not installed. Install with: pip install pytetwild")
    
    # Get vertices and faces from trimesh
    vertices = np.asarray(cavity_mesh.vertices, dtype=np.float64)
    faces = np.asarray(cavity_mesh.faces, dtype=np.int32)
    
    logger.info(f"Tetrahedralizing mesh with {len(vertices)} vertices, {len(faces)} faces")
    logger.info(f"Parameters: edge_length_fac={edge_length_fac}, optimize={optimize}")
    
    # Run fTetWild
    tet_vertices, tetrahedra = pytetwild.tetrahedralize(
        vertices, 
        faces,
        optimize=optimize,
        edge_length_fac=edge_length_fac
    )
    
    logger.info(f"Generated {len(tet_vertices)} vertices, {len(tetrahedra)} tetrahedra")
    
    return tet_vertices, tetrahedra


def extract_boundary_surface(
    vertices: np.ndarray,
    tetrahedra: np.ndarray
) -> trimesh.Trimesh:
    """
    Extract the boundary surface from a tetrahedral mesh.
    
    The boundary surface consists of triangular faces that belong to only
    one tetrahedron (i.e., they are on the outer surface).
    
    Each tetrahedron has 4 faces:
    - Face 0: vertices (1, 2, 3) - opposite to vertex 0
    - Face 1: vertices (0, 2, 3) - opposite to vertex 1  
    - Face 2: vertices (0, 1, 3) - opposite to vertex 2
    - Face 3: vertices (0, 1, 2) - opposite to vertex 3
    
    Args:
        vertices: (N, 3) vertex positions
        tetrahedra: (M, 4) tetrahedron vertex indices
    
    Returns:
        trimesh.Trimesh of the boundary surface
    """
    # Face vertex indices for each tetrahedron (ordered for consistent normals)
    # Using right-hand rule: normals point outward
    face_local_indices = [
        (1, 3, 2),  # Face opposite vertex 0
        (0, 2, 3),  # Face opposite vertex 1
        (0, 3, 1),  # Face opposite vertex 2
        (0, 1, 2),  # Face opposite vertex 3
    ]
    
    # Extract all faces from all tetrahedra
    all_faces = []
    for i, j, k in face_local_indices:
        faces = tetrahedra[:, [i, j, k]]
        all_faces.append(faces)
    
    all_faces = np.vstack(all_faces)  # (4*M, 3)
    
    # Sort vertices within each face for comparison (to find duplicates)
    sorted_faces = np.sort(all_faces, axis=1)
    
    # Find unique faces and their counts
    # Boundary faces appear exactly once, interior faces appear twice
    unique_faces, inverse, counts = np.unique(
        sorted_faces, axis=0, return_inverse=True, return_counts=True
    )
    
    # Get boundary face indices (faces that appear only once)
    boundary_mask = counts[inverse] == 1
    boundary_faces = all_faces[boundary_mask]
    
    logger.info(f"Extracted {len(boundary_faces)} boundary faces from {len(tetrahedra)} tetrahedra")
    
    # Create trimesh with only the vertices used by boundary faces
    # This removes unused vertices and reindexes faces
    unique_vertex_indices = np.unique(boundary_faces.ravel())
    vertex_remap = np.full(len(vertices), -1, dtype=np.int32)
    vertex_remap[unique_vertex_indices] = np.arange(len(unique_vertex_indices))
    
    boundary_vertices = vertices[unique_vertex_indices]
    remapped_faces = vertex_remap[boundary_faces]
    
    # Create trimesh
    boundary_mesh = trimesh.Trimesh(vertices=boundary_vertices, faces=remapped_faces)
    
    # Merge any duplicate vertices (within tolerance)
    boundary_mesh.merge_vertices()
    
    # Remove any degenerate faces (zero area)
    boundary_mesh.remove_degenerate_faces()
    
    # Remove any duplicate faces
    boundary_mesh.remove_duplicate_faces()
    
    # Fix normals to point outward (consistent winding)
    boundary_mesh.fix_normals()
    
    logger.info(f"Boundary mesh cleaned: {len(boundary_mesh.vertices)} vertices, {len(boundary_mesh.faces)} faces")
    
    return boundary_mesh


def inflate_boundary_vertices(
    tet_result: 'TetrahedralMeshResult',
    part_mesh: trimesh.Trimesh,
    r_value: float,
    use_gpu: bool = True
) -> 'TetrahedralMeshResult':
    """
    Inflate boundary vertices of the tetrahedral mesh outward.
    
    For each boundary vertex, move it outward by:
        displacement = R - distance_to_part
    
    This creates a more uniform thickness in the mold volume.
    Vertices close to the part move more (up to R), vertices far from
    the part move less.
    
    Args:
        tet_result: TetrahedralMeshResult to modify
        part_mesh: Original part mesh (for distance computation)
        r_value: Maximum distance R (computed earlier)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Modified TetrahedralMeshResult with inflated boundary vertices
    """
    import time
    start_time = time.time()
    
    boundary_mesh = tet_result.boundary_mesh
    if boundary_mesh is None:
        logger.warning("No boundary mesh available for inflation")
        return tet_result
    
    # Get boundary vertices
    boundary_verts = np.asarray(boundary_mesh.vertices, dtype=np.float64)
    n_boundary = len(boundary_verts)
    
    logger.info(f"Inflating {n_boundary} boundary vertices with R={r_value:.4f}")
    
    # Compute distance from each boundary vertex to part surface
    if use_gpu and CUDA_AVAILABLE:
        distances, closest_points = compute_distances_and_closest_points_gpu(
            boundary_verts, part_mesh
        )
    else:
        closest_points, distances, _ = part_mesh.nearest.on_surface(boundary_verts)
    
    # Compute displacement for each vertex: R - distance
    displacements = r_value - distances
    
    # Clamp displacements to be non-negative (don't move inward)
    displacements = np.maximum(displacements, 0.0)
    
    logger.info(f"Displacement range: [{displacements.min():.4f}, {displacements.max():.4f}]")
    
    # Compute outward direction for each vertex (vertex normal)
    # Use the boundary mesh vertex normals
    vertex_normals = boundary_mesh.vertex_normals
    if vertex_normals is None or len(vertex_normals) != n_boundary:
        # Fallback: compute direction from closest point on part to vertex
        directions = boundary_verts - closest_points
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        vertex_normals = directions / norms
    
    # Ensure normals are unit vectors
    normal_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    normal_lengths = np.where(normal_lengths > 1e-10, normal_lengths, 1.0)
    vertex_normals = vertex_normals / normal_lengths
    
    # Move boundary vertices outward
    inflated_boundary_verts = boundary_verts + vertex_normals * displacements[:, np.newaxis]
    
    # Update the boundary mesh with new vertex positions
    inflated_boundary_mesh = trimesh.Trimesh(
        vertices=inflated_boundary_verts,
        faces=boundary_mesh.faces.copy(),
        process=False
    )
    inflated_boundary_mesh.fix_normals()
    
    # Create mapping from boundary mesh vertices to tetrahedral mesh vertices
    # The tetrahedral vertices include interior vertices, but we only modify boundary ones
    # We need to identify which tet vertices correspond to boundary mesh vertices
    
    # Copy the original tet vertices
    new_tet_verts = tet_result.vertices.copy()
    
    # For each boundary mesh vertex, find the corresponding tet vertex and update it
    # Use nearest neighbor matching since boundary_mesh may have merged vertices
    from scipy.spatial import cKDTree
    tet_tree = cKDTree(tet_result.vertices)
    
    # For each original boundary vertex, find nearest tet vertex
    orig_boundary_verts = np.asarray(boundary_mesh.vertices, dtype=np.float64)
    _, tet_indices = tet_tree.query(orig_boundary_verts, k=1)
    
    # Update those tet vertices with inflated positions
    new_tet_verts[tet_indices] = inflated_boundary_verts
    
    # Recompute edge lengths with new vertices
    edges = tet_result.edges
    v0 = new_tet_verts[edges[:, 0]]
    v1 = new_tet_verts[edges[:, 1]]
    new_edge_lengths = np.linalg.norm(v1 - v0, axis=1)
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Boundary inflation complete in {elapsed:.0f}ms")
    logger.info(f"  Updated {len(tet_indices)} tet vertices")
    
    # Create new result with updated data
    inflated_result = TetrahedralMeshResult(
        vertices=new_tet_verts,
        tetrahedra=tet_result.tetrahedra,
        edges=tet_result.edges,
        edge_lengths=new_edge_lengths,
        boundary_mesh=inflated_boundary_mesh,
        edge_dist_to_part=None,  # Needs recomputation
        edge_weights=None,  # Needs recomputation
        weighted_edge_lengths=None,
        boundary_vertices=tet_result.boundary_vertices,
        boundary_labels=tet_result.boundary_labels,
        r_value=r_value,
        r_hull_point=tet_result.r_hull_point,
        r_part_point=tet_result.r_part_point,
        is_inflated=True,
        num_vertices=len(new_tet_verts),
        num_tetrahedra=tet_result.num_tetrahedra,
        num_edges=tet_result.num_edges,
        num_boundary_faces=len(inflated_boundary_mesh.faces),
        tetrahedralize_time_ms=tet_result.tetrahedralize_time_ms,
        total_time_ms=tet_result.total_time_ms + elapsed
    )
    
    return inflated_result


def extract_edges(tetrahedra: np.ndarray) -> np.ndarray:
    """
    Extract unique edges from tetrahedra.
    
    Each tetrahedron has 6 edges. We extract all edges and remove duplicates.
    
    Args:
        tetrahedra: (M, 4) array of tetrahedron vertex indices
    
    Returns:
        (E, 2) array of unique edge vertex indices (sorted so v0 < v1)
    """
    # Each tetrahedron has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edge_pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]
    
    # Extract all edges
    all_edges = []
    for i, j in edge_pairs:
        edges = np.column_stack([tetrahedra[:, i], tetrahedra[:, j]])
        all_edges.append(edges)
    
    all_edges = np.vstack(all_edges)
    
    # Sort each edge so smaller index comes first
    all_edges = np.sort(all_edges, axis=1)
    
    # Remove duplicates
    unique_edges = np.unique(all_edges, axis=0)
    
    return unique_edges


def compute_edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Compute the length of each edge.
    
    Args:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge vertex indices
    
    Returns:
        (E,) array of edge lengths
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    return lengths


def compute_distances_to_mesh_gpu(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    batch_size: int = 50000
) -> np.ndarray:
    """
    GPU-accelerated distance computation from points to mesh surface.
    
    Uses PyTorch CUDA to compute point-to-triangle distances in parallel on GPU.
    Falls back to CPU if CUDA is not available.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        batch_size: Points per GPU batch (to manage VRAM)
    
    Returns:
        (N,) array of unsigned distances to mesh surface
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        logger.warning("GPU not available, falling back to CPU computation")
        return compute_distances_to_mesh_cpu(query_points, target_mesh)
    
    import torch
    
    device = torch.device('cuda')
    n_points = len(query_points)
    n_faces = len(target_mesh.faces)
    
    # Get mesh data on GPU
    vertices = torch.tensor(target_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(target_mesh.faces, dtype=torch.int64, device=device)
    
    # Pre-compute triangle vertices
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Pre-compute triangle edge data (constant for all points)
    edge0 = v1 - v0  # (F, 3)
    edge1 = v2 - v0  # (F, 3)
    dot00 = (edge0 * edge0).sum(dim=-1)  # (F,)
    dot01 = (edge0 * edge1).sum(dim=-1)  # (F,)
    dot11 = (edge1 * edge1).sum(dim=-1)  # (F,)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)  # (F,)
    
    # Adjust batch size based on triangle count to fit in GPU memory
    # Memory per point-triangle pair ~ 12 bytes (3 floats for intermediate)
    # Target ~2GB working memory
    max_pairs = 150_000_000  # ~2GB for intermediates
    adaptive_batch = max(100, min(batch_size, max_pairs // n_faces))
    
    logger.debug(f"GPU: Processing {n_points} points against {n_faces} triangles (batch={adaptive_batch})")
    
    all_distances = torch.zeros(n_points, dtype=torch.float32, device=device)
    
    for start_idx in range(0, n_points, adaptive_batch):
        end_idx = min(start_idx + adaptive_batch, n_points)
        batch_points = torch.tensor(
            query_points[start_idx:end_idx], 
            dtype=torch.float32, 
            device=device
        )  # (B, 3)
        
        # Compute distances using memory-efficient method
        batch_distances = _point_to_triangles_distance_gpu_efficient(
            batch_points, v0, edge0, edge1, dot00, dot01, dot11, inv_denom
        )
        
        all_distances[start_idx:end_idx] = batch_distances
        
        # Clear cache periodically
        if (start_idx // adaptive_batch) % 10 == 0:
            torch.cuda.empty_cache()
            progress = 100 * end_idx / n_points
            logger.debug(f"  GPU: {end_idx}/{n_points} ({progress:.0f}%)")
    
    result = all_distances.cpu().numpy().astype(np.float64)
    torch.cuda.empty_cache()
    return result


def compute_distances_and_closest_points_gpu(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    batch_size: int = 30000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated computation of distances AND closest points from query points to mesh.
    
    This is useful for finding R (max hull-to-part distance) where we need both
    the distance and the closest point on the target mesh.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        batch_size: Points per GPU batch (smaller than distance-only due to more memory)
    
    Returns:
        Tuple of:
        - distances: (N,) array of unsigned distances to mesh surface
        - closest_points: (N, 3) array of closest points on mesh surface
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        logger.warning("GPU not available, falling back to CPU computation")
        closest_points, distances, _ = target_mesh.nearest.on_surface(query_points)
        return distances, closest_points
    
    import torch
    
    device = torch.device('cuda')
    n_points = len(query_points)
    n_faces = len(target_mesh.faces)
    
    logger.info(f"GPU: Computing distances and closest points for {n_points} points against {n_faces} triangles")
    
    # Get mesh data on GPU
    vertices = torch.tensor(target_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(target_mesh.faces, dtype=torch.int64, device=device)
    
    # Pre-compute triangle vertices
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Pre-compute triangle edge data
    edge0 = v1 - v0  # (F, 3)
    edge1 = v2 - v0  # (F, 3)
    dot00 = (edge0 * edge0).sum(dim=-1)  # (F,)
    dot01 = (edge0 * edge1).sum(dim=-1)  # (F,)
    dot11 = (edge1 * edge1).sum(dim=-1)  # (F,)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)  # (F,)
    
    # Adjust batch size - need more memory for storing closest points
    max_pairs = 100_000_000  # Conservative for storing closest points
    adaptive_batch = max(100, min(batch_size, max_pairs // n_faces))
    
    logger.debug(f"  Batch size: {adaptive_batch}")
    
    all_distances = torch.zeros(n_points, dtype=torch.float32, device=device)
    all_closest = torch.zeros((n_points, 3), dtype=torch.float32, device=device)
    
    for start_idx in range(0, n_points, adaptive_batch):
        end_idx = min(start_idx + adaptive_batch, n_points)
        batch_points = torch.tensor(
            query_points[start_idx:end_idx], 
            dtype=torch.float32, 
            device=device
        )  # (B, 3)
        
        # Compute distances and closest points
        batch_distances, batch_closest = _point_to_triangles_with_closest_gpu(
            batch_points, v0, edge0, edge1, dot00, dot01, dot11, inv_denom
        )
        
        all_distances[start_idx:end_idx] = batch_distances
        all_closest[start_idx:end_idx] = batch_closest
        
        # Clear cache periodically
        if (start_idx // adaptive_batch) % 5 == 0:
            torch.cuda.empty_cache()
            progress = 100 * end_idx / n_points
            logger.debug(f"  GPU: {end_idx}/{n_points} ({progress:.0f}%)")
    
    distances = all_distances.cpu().numpy().astype(np.float64)
    closest_points = all_closest.cpu().numpy().astype(np.float64)
    torch.cuda.empty_cache()
    
    logger.info(f"GPU: Completed distance computation")
    return distances, closest_points


def _point_to_triangles_with_closest_gpu(
    points: 'torch.Tensor',
    v0: 'torch.Tensor',
    edge0: 'torch.Tensor',
    edge1: 'torch.Tensor',
    dot00: 'torch.Tensor',
    dot01: 'torch.Tensor',
    dot11: 'torch.Tensor',
    inv_denom: 'torch.Tensor'
) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Compute minimum distance and closest point from each point to any triangle (GPU).
    
    Args:
        points: (B, 3) query points
        v0: (F, 3) first vertex of each triangle
        edge0, edge1: (F, 3) triangle edges
        dot00, dot01, dot11, inv_denom: (F,) pre-computed values
    
    Returns:
        Tuple of:
        - (B,) minimum distance for each point
        - (B, 3) closest point on mesh for each query point
    """
    import torch
    
    B = points.shape[0]
    F = v0.shape[0]
    
    # Expand for broadcasting: points (B, 1, 3), triangles (1, F, 3)
    p = points.unsqueeze(1)  # (B, 1, 3)
    
    # Vector from v0 to point
    v0_to_p = p - v0.unsqueeze(0)  # (B, F, 3)
    
    # Compute dot products for barycentric coordinates
    dot02 = (v0_to_p * edge0.unsqueeze(0)).sum(dim=-1)  # (B, F)
    dot12 = (v0_to_p * edge1.unsqueeze(0)).sum(dim=-1)  # (B, F)
    
    # Barycentric coordinates
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom  # (B, F)
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom  # (B, F)
    
    # Clamp to triangle
    u_clamped = torch.clamp(u, 0, 1)
    v_clamped = torch.clamp(v, 0, 1)
    
    # Ensure u + v <= 1
    uv_sum = u_clamped + v_clamped
    scale = torch.where(uv_sum > 1, 1.0 / uv_sum, torch.ones_like(uv_sum))
    u_final = u_clamped * scale
    v_final = v_clamped * scale
    
    # Closest point on each triangle: v0 + u*edge0 + v*edge1
    closest_on_tri = v0.unsqueeze(0) + u_final.unsqueeze(-1) * edge0.unsqueeze(0) + v_final.unsqueeze(-1) * edge1.unsqueeze(0)  # (B, F, 3)
    
    # Compute squared distance to each triangle
    diff = p - closest_on_tri  # (B, F, 3)
    dist_sq = (diff * diff).sum(dim=-1)  # (B, F)
    
    # Find minimum distance triangle for each point
    min_dist_sq, min_indices = dist_sq.min(dim=1)  # (B,), (B,)
    min_dist = torch.sqrt(min_dist_sq)  # (B,)
    
    # Gather closest points using the minimum indices
    # closest_on_tri is (B, F, 3), min_indices is (B,)
    batch_indices = torch.arange(B, device=points.device)
    closest_points = closest_on_tri[batch_indices, min_indices]  # (B, 3)
    
    return min_dist, closest_points


def _point_to_triangles_distance_gpu_efficient(
    points: 'torch.Tensor',
    v0: 'torch.Tensor',
    edge0: 'torch.Tensor',
    edge1: 'torch.Tensor',
    dot00: 'torch.Tensor',
    dot01: 'torch.Tensor',
    dot11: 'torch.Tensor',
    inv_denom: 'torch.Tensor'
) -> 'torch.Tensor':
    """
    Compute minimum distance from each point to any triangle (GPU).
    Memory-efficient version using pre-computed triangle data.
    
    Args:
        points: (B, 3) query points
        v0: (F, 3) first vertex of each triangle
        edge0, edge1: (F, 3) triangle edges
        dot00, dot01, dot11, inv_denom: (F,) pre-computed values
    
    Returns:
        (B,) minimum distance for each point
    """
    import torch
    
    B = points.shape[0]
    
    # Expand for broadcasting: points (B, 1, 3), triangles (1, F, 3)
    p = points.unsqueeze(1)  # (B, 1, 3)
    
    # Vector from v0 to point
    v0_to_p = p - v0.unsqueeze(0)  # (B, F, 3)
    
    # Compute dot products for barycentric coordinates
    dot02 = (v0_to_p * edge0.unsqueeze(0)).sum(dim=-1)  # (B, F)
    dot12 = (v0_to_p * edge1.unsqueeze(0)).sum(dim=-1)  # (B, F)
    
    # Barycentric coordinates
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom  # (B, F)
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom  # (B, F)
    
    # Clamp to triangle
    u_clamped = torch.clamp(u, 0, 1)
    v_clamped = torch.clamp(v, 0, 1)
    
    # Ensure u + v <= 1
    uv_sum = u_clamped + v_clamped
    scale = torch.where(uv_sum > 1, 1.0 / uv_sum, torch.ones_like(uv_sum))
    u_final = u_clamped * scale
    v_final = v_clamped * scale
    
    # Closest point on triangle: v0 + u*edge0 + v*edge1
    # Compute squared distance directly without storing closest point
    # closest = v0 + u*edge0 + v*edge1
    # diff = p - closest = p - v0 - u*edge0 - v*edge1 = v0_to_p - u*edge0 - v*edge1
    diff = v0_to_p - u_final.unsqueeze(-1) * edge0.unsqueeze(0) - v_final.unsqueeze(-1) * edge1.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=-1)  # (B, F)
    
    # Minimum distance across all triangles
    min_dist = torch.sqrt(dist_sq.min(dim=1).values)  # (B,)
    
    return min_dist


def _point_to_triangles_distance_gpu(
    points: 'torch.Tensor',
    v0: 'torch.Tensor',
    v1: 'torch.Tensor', 
    v2: 'torch.Tensor'
) -> 'torch.Tensor':
    """
    Compute minimum distance from each point to any triangle (GPU).
    Legacy version - use _point_to_triangles_distance_gpu_efficient instead.
    
    Args:
        points: (B, 3) query points
        v0, v1, v2: (F, 3) triangle vertices
    
    Returns:
        (B,) minimum distance for each point
    """
    import torch
    
    # Triangle edges
    edge0 = v1 - v0  # (F, 3)
    edge1 = v2 - v0  # (F, 3)
    dot00 = (edge0 * edge0).sum(dim=-1)  # (F,)
    dot01 = (edge0 * edge1).sum(dim=-1)  # (F,)
    dot11 = (edge1 * edge1).sum(dim=-1)  # (F,)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
    
    return _point_to_triangles_distance_gpu_efficient(
        points, v0, edge0, edge1, dot00, dot01, dot11, inv_denom
    )


def compute_distances_to_mesh_cpu(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    batch_size: int = 50000
) -> np.ndarray:
    """
    CPU distance computation from points to mesh surface.
    
    Uses trimesh's proximity query with batched processing to manage memory.
    Note: trimesh's rtree is NOT thread-safe, so we process sequentially.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        batch_size: Number of points to process per batch
    
    Returns:
        (N,) array of unsigned distances to mesh surface
    """
    n_points = len(query_points)
    
    if n_points <= batch_size:
        # Small enough to process in one go
        closest_points, distances, triangle_ids = target_mesh.nearest.on_surface(query_points)
        return distances
    
    # Process in batches sequentially (rtree is not thread-safe)
    logger.debug(f"CPU: Processing {n_points} points in batches of {batch_size}")
    
    all_distances = np.zeros(n_points, dtype=np.float64)
    
    n_batches = (n_points + batch_size - 1) // batch_size
    for i, start_idx in enumerate(range(0, n_points, batch_size)):
        end_idx = min(start_idx + batch_size, n_points)
        batch_points = query_points[start_idx:end_idx]
        
        closest_points, distances, triangle_ids = target_mesh.nearest.on_surface(batch_points)
        all_distances[start_idx:end_idx] = distances
        
        if (i + 1) % 5 == 0 or (i + 1) == n_batches:
            logger.debug(f"  CPU: Batch {i+1}/{n_batches} ({100*(i+1)/n_batches:.0f}%)")
    
    return all_distances


def compute_distances_to_mesh(
    query_points: np.ndarray,
    target_mesh: trimesh.Trimesh,
    use_gpu: bool = True,
    batch_size: int = None
) -> np.ndarray:
    """
    Compute distances from query points to target mesh surface.
    
    Automatically selects GPU or CPU computation based on availability.
    
    Args:
        query_points: (N, 3) points to query
        target_mesh: Target mesh to compute distance to
        use_gpu: Whether to prefer GPU if available
        batch_size: Batch size (auto-selected if None)
    
    Returns:
        (N,) array of unsigned distances to mesh surface
    """
    if use_gpu and CUDA_AVAILABLE:
        logger.info(f"Using GPU acceleration for distance computation ({len(query_points)} points)")
        return compute_distances_to_mesh_gpu(
            query_points, 
            target_mesh, 
            batch_size=batch_size or 50000
        )
    else:
        if use_gpu and not CUDA_AVAILABLE:
            logger.info("GPU requested but CUDA not available, using CPU")
        return compute_distances_to_mesh_cpu(
            query_points, 
            target_mesh, 
            batch_size=batch_size or 50000
        )


def identify_boundary_vertices(
    tet_vertices: np.ndarray,
    cavity_mesh: trimesh.Trimesh,
    tolerance: float = None
) -> np.ndarray:
    """
    Identify which tet vertices lie on the cavity boundary surface.
    
    Uses trimesh's fast proximity query (CPU only, but efficient for this task)
    since we just need a boolean mask, not exact distances.
    
    Args:
        tet_vertices: (N, 3) tetrahedral mesh vertices
        cavity_mesh: Original cavity surface mesh
        tolerance: Distance tolerance for boundary detection.
                   If None, uses 0.1% of bounding box diagonal.
    
    Returns:
        (N,) boolean array - True if vertex is on boundary
    """
    if tolerance is None:
        bbox_diag = np.linalg.norm(cavity_mesh.bounds[1] - cavity_mesh.bounds[0])
        tolerance = bbox_diag * 0.001
    
    # Use trimesh's fast proximity query directly (CPU, but efficient for boundary check)
    # This is faster than our GPU implementation for this specific use case
    # because we're checking ALL vertices against the surface they came from
    closest_points, distances, triangle_ids = cavity_mesh.nearest.on_surface(tet_vertices)
    
    # Vertices within tolerance are on boundary
    boundary_mask = distances < tolerance
    
    return boundary_mask


def label_boundary_from_classification(
    tet_vertices: np.ndarray,
    boundary_mask: np.ndarray,
    cavity_mesh: trimesh.Trimesh,
    classification_result,  # MoldHalfClassificationResult
    hull_mesh: trimesh.Trimesh
) -> np.ndarray:
    """
    Label boundary vertices as H1, H2, or inner boundary based on mold half classification.
    
    Args:
        tet_vertices: (N, 3) tetrahedral mesh vertices
        boundary_mask: (N,) boolean mask of boundary vertices
        cavity_mesh: Cavity surface mesh
        classification_result: Result from mold half classification
        hull_mesh: Original hull mesh
    
    Returns:
        (N,) int8 array: 0=interior, 1=H1 boundary, 2=H2 boundary, -1=inner boundary (part surface)
    """
    labels = np.zeros(len(tet_vertices), dtype=np.int8)
    
    # Get boundary vertex indices
    boundary_indices = np.where(boundary_mask)[0]
    
    if len(boundary_indices) == 0:
        return labels
    
    # Get boundary vertex positions
    boundary_verts = tet_vertices[boundary_indices]
    
    # Find closest face on cavity mesh for each boundary vertex
    closest_points, distances, face_ids = cavity_mesh.nearest.on_surface(boundary_verts)
    
    # Label based on face classification
    h1_set = classification_result.h1_triangles
    h2_set = classification_result.h2_triangles
    inner_set = classification_result.inner_boundary_triangles
    
    for i, (vert_idx, face_id) in enumerate(zip(boundary_indices, face_ids)):
        if face_id in h1_set:
            labels[vert_idx] = 1
        elif face_id in h2_set:
            labels[vert_idx] = 2
        elif face_id in inner_set:
            labels[vert_idx] = -1  # Inner boundary (part surface) - seed for Dijkstra
        # else: stays 0 (boundary zone or unclassified)
    
    return labels


def generate_tetrahedral_mesh(
    cavity_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    classification_result=None,  # MoldHalfClassificationResult
    edge_length_fac: float = 0.05,
    optimize: bool = True,
    compute_distances: bool = True
) -> TetrahedralMeshResult:
    """
    Generate tetrahedral mesh with edge weights for parting surface computation.
    
    This is the main entry point for tetrahedral mesh generation.
    
    Algorithm for edge weights:
    1. For each edge, compute midpoint distance to part mesh surface
    2. edge_weight = 1 / (distance^2 + 0.25)
    3. weighted_edge_length = edge_length * edge_weight
    
    Args:
        cavity_mesh: The cavity mesh (Hull - Part CSG result)
        part_mesh: Original part mesh (for distance computation)
        hull_mesh: Original hull mesh (not used currently, kept for API compatibility)
        classification_result: Mold half classification result (for boundary labeling)
        edge_length_fac: Target edge length as fraction of bbox diagonal
        optimize: Whether to optimize tet quality
        compute_distances: Whether to compute distance fields and weights
    
    Returns:
        TetrahedralMeshResult with all mesh data and weights
    """
    import time
    start_time = time.time()
    
    # Step 1: Tetrahedralize
    tet_start = time.time()
    tet_vertices, tetrahedra = tetrahedralize_mesh(
        cavity_mesh, 
        edge_length_fac=edge_length_fac,
        optimize=optimize
    )
    tet_time = (time.time() - tet_start) * 1000
    
    # Step 2: Extract edges
    edges = extract_edges(tetrahedra)
    edge_lengths = compute_edge_lengths(tet_vertices, edges)
    
    logger.info(f"Extracted {len(edges)} unique edges")
    
    # Step 3: Extract boundary surface from tetrahedral mesh
    boundary_mesh = extract_boundary_surface(tet_vertices, tetrahedra)
    
    # Initialize result
    result = TetrahedralMeshResult(
        vertices=tet_vertices,
        tetrahedra=tetrahedra,
        edges=edges,
        edge_lengths=edge_lengths,
        boundary_mesh=boundary_mesh,
        num_vertices=len(tet_vertices),
        num_tetrahedra=len(tetrahedra),
        num_edges=len(edges),
        num_boundary_faces=len(boundary_mesh.faces),
        tetrahedralize_time_ms=tet_time
    )
    
    # Step 4: Identify boundary vertices (vertices on the boundary surface)
    boundary_mask = identify_boundary_vertices(tet_vertices, boundary_mesh)
    result.boundary_vertices = boundary_mask
    logger.info(f"Found {np.sum(boundary_mask)} boundary vertices out of {len(tet_vertices)}")
    
    # Step 4: Compute edge weights based on distance to part surface
    if compute_distances:
        # Compute edge midpoint distances to part mesh
        edge_midpoints = (tet_vertices[edges[:, 0]] + tet_vertices[edges[:, 1]]) / 2
        
        logger.debug("Computing edge midpoint distances to part mesh...")
        edge_dist_to_part = compute_distances_to_mesh(edge_midpoints, part_mesh)
        result.edge_dist_to_part = edge_dist_to_part
        
        # Compute edge weights: weight = 1 / (dist^2 + 0.25)
        # This gives higher weight (shorter effective distance) near the part surface
        result.edge_weights = 1.0 / (edge_dist_to_part ** 2 + 0.25)
        
        # Compute weighted edge lengths
        result.weighted_edge_lengths = edge_lengths * result.edge_weights
        
        logger.info(f"Edge weights: min={result.edge_weights.min():.4f}, max={result.edge_weights.max():.4f}, mean={result.edge_weights.mean():.4f}")
    
    # Step 5: Label boundary vertices from classification
    if classification_result is not None:
        logger.debug("Labeling boundary vertices from classification...")
        result.boundary_labels = label_boundary_from_classification(
            tet_vertices,
            boundary_mask,
            cavity_mesh,
            classification_result,
            hull_mesh
        )
        
        n_h1 = np.sum(result.boundary_labels == 1)
        n_h2 = np.sum(result.boundary_labels == 2)
        n_inner = np.sum(result.boundary_labels == -1)
        logger.info(f"Boundary labels: H1={n_h1}, H2={n_h2}, inner(seeds)={n_inner}")
    
    result.total_time_ms = (time.time() - start_time) * 1000
    
    logger.info(
        f"Tetrahedral mesh complete: {result.num_vertices} verts, "
        f"{result.num_tetrahedra} tets, {result.num_edges} edges, "
        f"time={result.total_time_ms:.0f}ms"
    )
    
    return result


def build_vertex_adjacency(edges: np.ndarray, num_vertices: int) -> Dict[int, List[int]]:
    """
    Build vertex adjacency map from edge list.
    
    Args:
        edges: (E, 2) edge vertex indices
        num_vertices: Total number of vertices
    
    Returns:
        Dict mapping vertex index to list of adjacent vertex indices
    """
    adjacency: Dict[int, List[int]] = {i: [] for i in range(num_vertices)}
    
    for v0, v1 in edges:
        adjacency[v0].append(v1)
        adjacency[v1].append(v0)
    
    return adjacency


def build_edge_index_map(edges: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Build map from edge vertex pair to edge index.
    
    Args:
        edges: (E, 2) edge vertex indices (sorted)
    
    Returns:
        Dict mapping (v0, v1) tuple to edge index
    """
    edge_map = {}
    for i, (v0, v1) in enumerate(edges):
        # Store with smaller index first
        key = (min(v0, v1), max(v0, v1))
        edge_map[key] = i
    
    return edge_map


def compute_edge_weights(
    tet_result: TetrahedralMeshResult,
    shell_radius: float = None,
    bias_boundary_layers: int = 1
) -> np.ndarray:
    """
    Compute edge weights for Dijkstra-based escape labeling.
    
    The weight scheme follows the React app's volumetric grid approach but applied to edges:
    
    For each edge, the cost is: L * wt
    where:
        L = edge length (Euclidean distance)
        wt = weighting factor based on distance to part mesh
    
    Weighting factor computation:
    - For boundary-adjacent edges:
        biasedDist = δ_part + λ_w  where λ_w = R - δ_shell
        wt = 1 / (biasedDist² + 0.25)
    - For interior edges:
        wt = 1 / (δ_part² + 0.25)
    
    This biases paths toward the shell boundary while preferring thick regions.
    
    Args:
        tet_result: Tetrahedral mesh result with distances computed
        shell_radius: Radius R for shell bias. If None, computed from max shell distance.
        bias_boundary_layers: Number of edge layers from boundary to apply shell bias
    
    Returns:
        (E,) array of edge weights (cost = edge_length * weight)
    """
    if tet_result.edge_dist_to_part is None:
        raise ValueError("Edge distances to part not computed. Run with compute_distances=True")
    
    edges = tet_result.edges
    edge_lengths = tet_result.edge_lengths
    dist_to_part = tet_result.edge_dist_to_part
    dist_to_shell = tet_result.edge_dist_to_shell
    
    n_edges = len(edges)
    
    # Compute shell radius if not provided
    if shell_radius is None and dist_to_shell is not None:
        shell_radius = np.max(dist_to_shell) * 1.1  # 10% buffer
    
    # Identify boundary-adjacent edges using vertex boundary mask
    boundary_verts = tet_result.boundary_vertices
    if boundary_verts is None:
        # No boundary info, treat all as interior
        is_boundary_edge = np.zeros(n_edges, dtype=bool)
    else:
        # Edge is boundary-adjacent if either endpoint is on boundary
        v0_boundary = boundary_verts[edges[:, 0]]
        v1_boundary = boundary_verts[edges[:, 1]]
        is_boundary_edge = v0_boundary | v1_boundary
    
    # Compute weighting factors
    weights = np.zeros(n_edges, dtype=np.float64)
    
    # Interior edges: wt = 1 / (δ_part² + 0.25)
    interior_mask = ~is_boundary_edge
    weights[interior_mask] = 1.0 / (dist_to_part[interior_mask] ** 2 + 0.25)
    
    # Boundary-adjacent edges: biased weight
    if dist_to_shell is not None and shell_radius is not None:
        boundary_mask = is_boundary_edge
        lambda_w = shell_radius - dist_to_shell[boundary_mask]
        biased_dist = dist_to_part[boundary_mask] + np.maximum(lambda_w, 0)
        weights[boundary_mask] = 1.0 / (biased_dist ** 2 + 0.25)
    else:
        # Fall back to unbiased for boundary edges
        boundary_mask = is_boundary_edge
        weights[boundary_mask] = 1.0 / (dist_to_part[boundary_mask] ** 2 + 0.25)
    
    logger.debug(f"Computed edge weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    
    return weights


def compute_edge_costs(
    tet_result: TetrahedralMeshResult,
    shell_radius: float = None
) -> np.ndarray:
    """
    Compute edge traversal costs for Dijkstra (length * weight).
    
    Args:
        tet_result: Tetrahedral mesh result
        shell_radius: Radius for shell bias computation
    
    Returns:
        (E,) array of edge costs
    """
    weights = compute_edge_weights(tet_result, shell_radius)
    costs = tet_result.edge_lengths * weights
    
    logger.debug(f"Computed edge costs: min={costs.min():.6f}, max={costs.max():.6f}, mean={costs.mean():.6f}")
    
    return costs
