"""
STL File Loader

Handles loading of STL files (both ASCII and binary formats) using trimesh.
Provides mesh validation and basic preprocessing.

Design Principles:
- Stateless functions preferred over stateful classes
- Clear separation between validation and loading
- Result objects for consistent error handling
- Backward compatible with existing API
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import trimesh


logger = logging.getLogger(__name__)

# Constants
SUPPORTED_EXTENSIONS = frozenset({'.stl'})
MIN_FILE_SIZE_BYTES = 1  # Minimum valid file size


@dataclass(frozen=True)
class LoadResult:
    """
    Immutable result of STL file loading operation.
    
    Attributes:
        mesh: The loaded trimesh mesh, or None if loading failed
        file_path: Absolute path to the loaded file
        file_name: Base name of the file
        file_size_bytes: Size of the file in bytes
        success: Whether loading succeeded
        error_message: Error description if success is False
        load_time_ms: Time taken to load in milliseconds
    """
    mesh: Optional[trimesh.Trimesh]
    file_path: str
    file_name: str
    file_size_bytes: int
    success: bool
    error_message: Optional[str] = None
    load_time_ms: float = 0.0


class ValidationError(Exception):
    """Raised when file validation fails."""


def validate_stl_path(file_path: Union[str, Path]) -> Tuple[Path, int]:
    """
    Validate that a path points to a valid STL file.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (resolved_path, file_size_bytes)
        
    Raises:
        ValidationError: If the file is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension: {path.suffix}. Expected: .stl"
        )
    
    file_size = path.stat().st_size
    if file_size < MIN_FILE_SIZE_BYTES:
        raise ValidationError("File is empty")
    
    return path.resolve(), file_size


def _extract_mesh_from_scene(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    Extract a Trimesh from a Scene object.
    
    Args:
        scene: A trimesh Scene containing geometry
        
    Returns:
        The first Trimesh found in the scene
        
    Raises:
        ValueError: If no geometry found in scene
    """
    geometries = list(scene.geometry.values())
    if not geometries:
        raise ValueError("No geometry found in STL file")
    return geometries[0]


def _load_mesh_from_path(file_path: Path) -> trimesh.Trimesh:
    """
    Load a mesh from a validated file path.
    
    Args:
        file_path: Validated path to STL file
        
    Returns:
        Loaded and processed Trimesh
        
    Raises:
        ValueError: If the loaded object is not a valid mesh
    """
    mesh = trimesh.load(
        str(file_path),
        file_type='stl',
        force='mesh'
    )
    
    # Handle Scene objects (can occur with some STL files)
    if isinstance(mesh, trimesh.Scene):
        mesh = _extract_mesh_from_scene(mesh)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh, got {type(mesh).__name__}")
    
    # Run standard mesh processing
    mesh.process()
    
    return mesh


def load_stl_file(file_path: Union[str, Path]) -> LoadResult:
    """
    Load an STL file and return a trimesh mesh.
    
    This is the primary entry point for loading STL files. It validates
    the file path, loads the mesh using trimesh, and returns a result
    object containing either the mesh or error information.
    
    Args:
        file_path: Path to the STL file (string or Path object)
        
    Returns:
        LoadResult containing the mesh or error information
        
    Example:
        >>> result = load_stl_file("model.stl")
        >>> if result.success:
        ...     print(f"Loaded {result.mesh.vertices.shape[0]} vertices")
        ... else:
        ...     print(f"Error: {result.error_message}")
    """
    start_time = time.perf_counter()
    path = Path(file_path)
    file_name = path.name
    
    # Validate file path
    try:
        resolved_path, file_size = validate_stl_path(path)
    except ValidationError as e:
        logger.warning("STL validation failed: %s", e)
        return LoadResult(
            mesh=None,
            file_path=str(path.absolute()),
            file_name=file_name,
            file_size_bytes=0,
            success=False,
            error_message=str(e),
            load_time_ms=(time.perf_counter() - start_time) * 1000
        )
    
    # Load the mesh
    try:
        mesh = _load_mesh_from_path(resolved_path)
        load_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "Loaded STL: %s (%d vertices, %d faces) in %.1fms",
            file_name, len(mesh.vertices), len(mesh.faces), load_time_ms
        )
        
        return LoadResult(
            mesh=mesh,
            file_path=str(resolved_path),
            file_name=file_name,
            file_size_bytes=file_size,
            success=True,
            load_time_ms=load_time_ms
        )
        
    except Exception as e:
        load_time_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Failed to load STL %s: %s", file_name, e)
        
        return LoadResult(
            mesh=None,
            file_path=str(path.absolute()),
            file_name=file_name,
            file_size_bytes=file_size,
            success=False,
            error_message=str(e),
            load_time_ms=load_time_ms
        )


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class STLLoader:
    """
    STL file loader with support for both ASCII and binary formats.
    
    This class is provided for backward compatibility. For new code,
    prefer using the `load_stl_file()` function directly.
    
    Uses trimesh library for robust STL parsing and initial mesh construction.
    """
    
    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS
    
    def __init__(self):
        self._last_result: Optional[LoadResult] = None
    
    @property
    def last_result(self) -> Optional[LoadResult]:
        """Get the result of the last load operation."""
        return self._last_result
    
    def is_valid_stl_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Check if a file is a valid STL file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            validate_stl_path(file_path)
            return True, ""
        except ValidationError as e:
            return False, str(e)
    
    def load(self, file_path: str) -> LoadResult:
        """
        Load an STL file and return a trimesh mesh.
        
        Args:
            file_path: Path to the STL file
            
        Returns:
            LoadResult containing the mesh or error information
        """
        self._last_result = load_stl_file(file_path)
        return self._last_result
