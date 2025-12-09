"""
Mold Cavity Module

Creates a mold cavity by performing CSG (Constructive Solid Geometry) subtraction
of the original mesh from the inflated convex hull. The result is the hollow
space between the hull and the original mesh - the "mold cavity".

Uses manifold3d as the primary CSG engine - it's extremely fast and robust,
built on the Manifold library which is used in 3D printing slicers.

Fallback methods:
1. manifold3d (primary) - Fast GPU-accelerated boolean operations
2. shell approach - Combine hull + inverted original for visualization
"""

import logging
from typing import Optional
from dataclasses import dataclass
import numpy as np

import trimesh

# Try to import manifold3d - the fast CSG library
try:
    import manifold3d
    from manifold3d import Manifold, Mesh
    MANIFOLD_AVAILABLE = True
    logger_msg = "manifold3d CSG library available"
except ImportError:
    MANIFOLD_AVAILABLE = False
    logger_msg = "manifold3d not installed - CSG operations will be slower"

logger = logging.getLogger(__name__)
logger.info(logger_msg)

# Color for cavity visualization (matching React frontend)
CAVITY_COLOR = [0.0, 0.8, 0.8, 0.7]  # Cyan with transparency


@dataclass
class CavityValidation:
    """Results of cavity mesh validation."""
    is_closed: bool
    is_manifold: bool
    volume: Optional[float]
    surface_area: Optional[float]
    vertex_count: int
    face_count: int


@dataclass 
class MoldCavityResult:
    """Result of mold cavity computation."""
    mesh: trimesh.Trimesh
    validation: CavityValidation
    success: bool
    error_message: Optional[str] = None
    used_fallback: bool = False
    engine_used: Optional[str] = None


def validate_cavity_mesh(mesh: trimesh.Trimesh) -> CavityValidation:
    """Validate the properties of a cavity mesh."""
    is_closed = mesh.is_watertight
    is_manifold = mesh.is_watertight
    
    volume = None
    surface_area = None
    
    if is_closed:
        try:
            volume = float(mesh.volume)
        except Exception:
            pass
    
    try:
        surface_area = float(mesh.area)
    except Exception:
        pass
    
    return CavityValidation(
        is_closed=is_closed,
        is_manifold=is_manifold,
        volume=volume,
        surface_area=surface_area,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces)
    )


def _empty_result(error_message: str) -> MoldCavityResult:
    """Create an empty/failed result."""
    return MoldCavityResult(
        mesh=trimesh.Trimesh(),
        validation=CavityValidation(
            is_closed=False,
            is_manifold=False,
            volume=None,
            surface_area=None,
            vertex_count=0,
            face_count=0
        ),
        success=False,
        error_message=error_message
    )


def _trimesh_to_manifold(mesh: trimesh.Trimesh) -> 'Manifold':
    """
    Convert a trimesh mesh to a Manifold object.
    
    Args:
        mesh: The trimesh mesh to convert
        
    Returns:
        Manifold object for CSG operations
    """
    if not MANIFOLD_AVAILABLE:
        raise ImportError("manifold3d not installed")
    
    # Get vertices and faces as numpy arrays with correct dtypes
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    
    # Create Manifold Mesh object
    manifold_mesh = Mesh(vert_properties=vertices, tri_verts=faces)
    
    # Create Manifold from mesh
    return Manifold(mesh=manifold_mesh)


def _manifold_to_trimesh(manifold: 'Manifold') -> trimesh.Trimesh:
    """
    Convert a Manifold object back to trimesh.
    
    Args:
        manifold: The Manifold object to convert
        
    Returns:
        trimesh.Trimesh mesh
    """
    # Get the mesh from the manifold
    manifold_mesh = manifold.to_mesh()
    
    # Extract vertices and faces
    vertices = np.asarray(manifold_mesh.vert_properties, dtype=np.float64)
    faces = np.asarray(manifold_mesh.tri_verts, dtype=np.int64)
    
    # Create trimesh
    result = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True
    )
    
    return result


def _try_manifold_subtraction(
    hull_mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh
) -> Optional[trimesh.Trimesh]:
    """
    Perform CSG subtraction using manifold3d library.
    
    This is the fastest and most robust method available.
    manifold3d uses optimized algorithms and can handle complex meshes
    that would fail with other boolean libraries.
    
    Args:
        hull_mesh: The hull mesh (outer boundary)
        original_mesh: The original mesh to subtract
        
    Returns:
        Result mesh or None if failed
    """
    if not MANIFOLD_AVAILABLE:
        logger.warning("manifold3d not available")
        return None
    
    try:
        logger.info("Converting meshes to Manifold format...")
        
        # Convert both meshes to Manifold
        hull_manifold = _trimesh_to_manifold(hull_mesh)
        original_manifold = _trimesh_to_manifold(original_mesh)
        
        logger.info("Performing CSG subtraction (hull - original)...")
        
        # Perform the boolean subtraction: hull - original = cavity
        result_manifold = hull_manifold - original_manifold
        
        # Check if result is valid
        if result_manifold.is_empty():
            logger.warning("Manifold subtraction returned empty result")
            return None
        
        logger.info("Converting result back to trimesh...")
        
        # Convert back to trimesh
        result_mesh = _manifold_to_trimesh(result_manifold)
        
        if len(result_mesh.vertices) == 0 or len(result_mesh.faces) == 0:
            logger.warning("Manifold result has no geometry")
            return None
        
        logger.info(f"Manifold CSG successful: {len(result_mesh.faces)} faces")
        return result_mesh
        
    except Exception as e:
        logger.warning(f"Manifold subtraction failed: {e}")
        return None


def _try_shell_approach(
    hull_mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh
) -> Optional[trimesh.Trimesh]:
    """
    Fallback: Create cavity by combining hull and inverted original.
    
    This creates a "shell" mesh that represents the cavity visually by:
    1. Taking the outer surface of the hull (facing outward)
    2. Taking the surface of the original mesh (facing inward toward cavity)
    3. Combining them into a single mesh
    
    Note: This doesn't do true CSG but creates a reasonable visual representation.
    """
    try:
        logger.info("Using shell approach (fallback)...")
        
        # Invert the normals of the original mesh (flip it inside-out)
        inverted_original = original_mesh.copy()
        inverted_original.invert()
        
        # Combine the hull and inverted original
        combined = trimesh.util.concatenate([hull_mesh, inverted_original])
        
        # Try to clean up the mesh
        combined.fill_holes()
        combined.fix_normals()
        
        if len(combined.vertices) > 0:
            logger.info(f"Shell approach produced {len(combined.faces)} faces")
            return combined
            
    except Exception as e:
        logger.warning(f"Shell-based approach failed: {e}")
    
    return None


def perform_csg_subtraction(
    hull_mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh,
    method: str = 'auto'
) -> MoldCavityResult:
    """
    Perform CSG subtraction: Hull - Original Mesh = Mold Cavity
    
    Creates a mold cavity by subtracting the original mesh from the
    inflated hull, leaving the hollow space between them.
    
    Args:
        hull_mesh: The inflated convex hull mesh
        original_mesh: The original mesh to subtract from the hull
        method: Method to use ('auto', 'manifold', 'shell')
        
    Returns:
        MoldCavityResult containing the resulting cavity mesh and metadata
    """
    logger.info(f"Performing CSG subtraction using method: {method}")
    logger.info(f"Hull: {len(hull_mesh.vertices)} vertices, {len(hull_mesh.faces)} faces")
    logger.info(f"Original: {len(original_mesh.vertices)} vertices, {len(original_mesh.faces)} faces")
    
    # Validate inputs
    if len(hull_mesh.vertices) == 0 or len(hull_mesh.faces) == 0:
        return _empty_result("Hull mesh is empty")
    
    if len(original_mesh.vertices) == 0 or len(original_mesh.faces) == 0:
        return _empty_result("Original mesh is empty")
    
    result_mesh = None
    engine_used = None
    used_fallback = False
    
    # Determine methods to try - manifold3d is primary (fastest and most robust)
    if method == 'auto':
        methods_to_try = ['manifold', 'shell']
    elif method == 'manifold':
        methods_to_try = ['manifold', 'shell']
    elif method == 'shell':
        methods_to_try = ['shell']
    else:
        methods_to_try = ['manifold', 'shell']
    
    for m in methods_to_try:
        if m == 'manifold' and MANIFOLD_AVAILABLE:
            logger.info("Trying manifold3d CSG subtraction...")
            result_mesh = _try_manifold_subtraction(hull_mesh, original_mesh)
            if result_mesh is not None:
                engine_used = 'manifold3d'
                break
        
        elif m == 'shell':
            logger.info("Trying shell-based approach...")
            result_mesh = _try_shell_approach(hull_mesh, original_mesh)
            if result_mesh is not None:
                engine_used = 'shell'
                used_fallback = True
                break
        
        # Mark as fallback if we need to try next method
        if result_mesh is None and m != methods_to_try[-1]:
            used_fallback = True
    
    # Check if we got a result
    if result_mesh is None or len(result_mesh.vertices) == 0:
        return _empty_result("All CSG methods failed to produce a result")
    
    # Fix normals to ensure consistent orientation
    result_mesh.fix_normals()
    
    # Validate the result
    validation = validate_cavity_mesh(result_mesh)
    
    logger.info(f"Cavity created using {engine_used}: "
                f"{validation.vertex_count} vertices, "
                f"{validation.face_count} faces, "
                f"is_closed={validation.is_closed}")
    if validation.volume is not None:
        logger.info(f"Cavity volume: {validation.volume:.2f} cubic units")
    
    return MoldCavityResult(
        mesh=result_mesh,
        validation=validation,
        success=True,
        used_fallback=used_fallback,
        engine_used=engine_used
    )


def create_mold_cavity(
    hull_mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh
) -> MoldCavityResult:
    """
    Convenience function to create a mold cavity.
    
    Uses automatic method selection - manifold3d if available (fastest),
    otherwise falls back to shell approach.
    
    Args:
        hull_mesh: The inflated convex hull mesh
        original_mesh: The original mesh
        
    Returns:
        MoldCavityResult containing the cavity mesh
    """
    return perform_csg_subtraction(hull_mesh, original_mesh, method='auto')
