"""
CSG Backend Abstraction Layer

Provides robust Constructive Solid Geometry (CSG) operations by wrapping
libigl's CGAL-based mesh boolean implementation (Zhou et al. 2016,
"Mesh Arrangements for Solid Geometry", SIGGRAPH).

This backend uses exact arithmetic via CGAL's rational number kernel,
ensuring 100% robustness on all inputs — including coplanar faces,
self-intersections, and near-degenerate triangles.

Falls back to manifold3d if libigl's CGAL module is unavailable.

References:
    - Zhou et al., "Mesh Arrangements for Solid Geometry", SIGGRAPH 2016
    - Alderighi et al., "Volume-Aware Design of Composite Molds", SIGGRAPH 2019
      (Section 5: "Robust CSG Boolean operations [Zhou et al. 2016]")

Author: VcMoldCreator
"""

import logging
from typing import Optional, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# ============================================================================
# BACKEND DETECTION — prefer libigl/CGAL, fall back to manifold3d
# ============================================================================

CSG_BACKEND: str = "none"
CSG_AVAILABLE: bool = False

# Try libigl with CGAL copyleft module first (exact arithmetic, Zhou 2016)
try:
    import igl
    import igl.copyleft.cgal as _igl_cgal
    # Quick sanity check: ensure mesh_boolean is callable
    if hasattr(_igl_cgal, 'mesh_boolean'):
        CSG_BACKEND = "libigl_cgal"
        CSG_AVAILABLE = True
        logger.info("CSG backend: libigl/CGAL (exact arithmetic, Zhou et al. 2016)")
    else:
        raise ImportError("igl.copyleft.cgal.mesh_boolean not found")
except ImportError as e:
    logger.debug(f"libigl/CGAL not available: {e}")

# Fall back to manifold3d (floating point, may fail on edge cases)
if not CSG_AVAILABLE:
    try:
        import manifold3d
        CSG_BACKEND = "manifold3d"
        CSG_AVAILABLE = True
        logger.info("CSG backend: manifold3d (floating point — fallback)")
    except ImportError:
        logger.warning(
            "No CSG backend available. Install libigl (recommended) or "
            "manifold3d: pip install libigl manifold3d"
        )


# ============================================================================
# MESH ↔ ARRAY CONVERSION
# ============================================================================

def _trimesh_to_arrays(
    mesh: trimesh.Trimesh,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a trimesh to (V, F) numpy arrays for libigl.

    Args:
        mesh: Input triangle mesh.

    Returns:
        V: (#V, 3) float64 vertex positions.
        F: (#F, 3) int32 face indices.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    return V, F


def _arrays_to_trimesh(
    V: np.ndarray,
    F: np.ndarray,
) -> trimesh.Trimesh:
    """Convert (V, F) numpy arrays back to a trimesh.

    Args:
        V: (#V, 3) vertex positions.
        F: (#F, 3) face indices.

    Returns:
        Resulting trimesh.Trimesh.
    """
    return trimesh.Trimesh(
        vertices=np.asarray(V, dtype=np.float64),
        faces=np.asarray(F, dtype=np.int64),
        process=False,
    )


# ============================================================================
# MANIFOLD3D HELPERS (fallback only)
# ============================================================================

def _trimesh_to_manifold(mesh: trimesh.Trimesh):
    """Convert trimesh → manifold3d.Manifold (fallback path)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    m3d_mesh = manifold3d.Mesh(vert_properties=vertices, tri_verts=faces)
    return manifold3d.Manifold(m3d_mesh)


def _manifold_to_trimesh(manifold) -> trimesh.Trimesh:
    """Convert manifold3d.Manifold → trimesh (fallback path)."""
    mesh_data = manifold.to_mesh()
    vertices = np.asarray(mesh_data.vert_properties, dtype=np.float64)
    faces = np.asarray(mesh_data.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


# ============================================================================
# CORE CSG OPERATIONS
# ============================================================================

def csg_difference(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Compute A minus B (set difference).

    Removes the volume of mesh_b from mesh_a.

    Args:
        mesh_a: The base mesh.
        mesh_b: The mesh to subtract.

    Returns:
        Result mesh (A \\ B).

    Raises:
        RuntimeError: If no CSG backend is available or the operation fails.

    References:
        Zhou et al. 2016, Section 3.2 (variadic extraction).
    """
    if not CSG_AVAILABLE:
        raise RuntimeError("No CSG backend available")

    if CSG_BACKEND == "libigl_cgal":
        VA, FA = _trimesh_to_arrays(mesh_a)
        VB, FB = _trimesh_to_arrays(mesh_b)
        try:
            VC, FC, _ = _igl_cgal.mesh_boolean(VA, FA, VB, FB, "minus")
            return _arrays_to_trimesh(VC, FC)
        except RuntimeError as e:
            raise RuntimeError(f"libigl/CGAL mesh_boolean (minus) failed: {e}") from e

    else:  # manifold3d
        ma = _trimesh_to_manifold(mesh_a)
        mb = _trimesh_to_manifold(mesh_b)
        result = ma - mb
        return _manifold_to_trimesh(result)


def csg_union(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Compute A union B.

    Args:
        mesh_a: First mesh.
        mesh_b: Second mesh.

    Returns:
        Result mesh (A ∪ B).

    Raises:
        RuntimeError: If no CSG backend is available or the operation fails.
    """
    if not CSG_AVAILABLE:
        raise RuntimeError("No CSG backend available")

    if CSG_BACKEND == "libigl_cgal":
        VA, FA = _trimesh_to_arrays(mesh_a)
        VB, FB = _trimesh_to_arrays(mesh_b)
        try:
            VC, FC, _ = _igl_cgal.mesh_boolean(VA, FA, VB, FB, "union")
            return _arrays_to_trimesh(VC, FC)
        except RuntimeError as e:
            raise RuntimeError(f"libigl/CGAL mesh_boolean (union) failed: {e}") from e

    else:  # manifold3d
        ma = _trimesh_to_manifold(mesh_a)
        mb = _trimesh_to_manifold(mesh_b)
        result = ma + mb
        return _manifold_to_trimesh(result)


def csg_intersection(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Compute A intersect B.

    Args:
        mesh_a: First mesh.
        mesh_b: Second mesh.

    Returns:
        Result mesh (A ∩ B).

    Raises:
        RuntimeError: If no CSG backend is available or the operation fails.
    """
    if not CSG_AVAILABLE:
        raise RuntimeError("No CSG backend available")

    if CSG_BACKEND == "libigl_cgal":
        VA, FA = _trimesh_to_arrays(mesh_a)
        VB, FB = _trimesh_to_arrays(mesh_b)
        try:
            VC, FC, _ = _igl_cgal.mesh_boolean(VA, FA, VB, FB, "intersect")
            return _arrays_to_trimesh(VC, FC)
        except RuntimeError as e:
            raise RuntimeError(f"libigl/CGAL mesh_boolean (intersect) failed: {e}") from e

    else:  # manifold3d
        ma = _trimesh_to_manifold(mesh_a)
        mb = _trimesh_to_manifold(mesh_b)
        result = ma ^ mb
        return _manifold_to_trimesh(result)


def csg_trim_by_plane(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    offset: float,
) -> trimesh.Trimesh:
    """Trim a mesh by a plane, keeping geometry on one side.

    Keeps all geometry where dot(v, normal) >= offset.
    This matches the manifold3d.Manifold.trim_by_plane(normal, offset) convention.

    Args:
        mesh: The mesh to trim.
        normal: (3,) plane normal (unit vector preferred).
        offset: Scalar offset along the normal direction.

    Returns:
        Trimmed mesh.

    Raises:
        RuntimeError: If no CSG backend is available or the operation fails.
    """
    if not CSG_AVAILABLE:
        raise RuntimeError("No CSG backend available")

    normal = np.asarray(normal, dtype=np.float64).ravel()

    if CSG_BACKEND == "libigl_cgal":
        V, F = _trimesh_to_arrays(mesh)
        # intersect_with_half_space keeps geometry where dot(v - p, n) <= 0.
        #
        # We want to keep dot(v, normal) >= offset, i.e.:
        #   dot(v, -normal) <= -offset
        #   dot(v - p, -normal) <= 0  where dot(p, -normal) = -offset
        #                              => dot(p, normal) = offset
        #                              => p = offset * normal (for unit normal)
        #
        # So: half_space_normal = -normal, half_space_point = offset * normal
        p = offset * normal
        n = -normal
        try:
            VC, FC, _ = _igl_cgal.intersect_with_half_space(V, F, p, n)
            return _arrays_to_trimesh(VC, FC)
        except RuntimeError as e:
            raise RuntimeError(
                f"libigl/CGAL intersect_with_half_space failed: {e}"
            ) from e

    else:  # manifold3d
        m = _trimesh_to_manifold(mesh)
        m = m.trim_by_plane(normal.tolist(), float(offset))
        return _manifold_to_trimesh(m)


# ============================================================================
# CONVENIENCE / STATUS
# ============================================================================

def get_backend_info() -> str:
    """Return a human-readable string describing the active CSG backend."""
    if CSG_BACKEND == "libigl_cgal":
        return (
            "libigl/CGAL (exact arithmetic, Zhou et al. 2016 — "
            "100% robust on all inputs)"
        )
    elif CSG_BACKEND == "manifold3d":
        return "manifold3d (floating point — may fail on edge cases)"
    else:
        return "No CSG backend available"
