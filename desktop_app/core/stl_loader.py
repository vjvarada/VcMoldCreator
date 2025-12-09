"""
STL File Loader

Handles loading of STL files (both ASCII and binary formats) using trimesh.
Provides mesh validation and basic preprocessing.
"""

from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import trimesh


@dataclass
class LoadResult:
    """Result of STL file loading operation."""
    mesh: Optional[trimesh.Trimesh]
    file_path: str
    file_name: str
    file_size_bytes: int
    success: bool
    error_message: Optional[str] = None
    load_time_ms: float = 0.0


class STLLoader:
    """
    STL file loader with support for both ASCII and binary formats.
    
    Uses trimesh library for robust STL parsing and initial mesh construction.
    """
    
    SUPPORTED_EXTENSIONS = {'.stl', '.STL'}
    
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
        path = Path(file_path)
        
        if not path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        if path.suffix.lower() not in {ext.lower() for ext in self.SUPPORTED_EXTENSIONS}:
            return False, f"Unsupported file extension: {path.suffix}. Expected .stl"
        
        # Check file is not empty
        if path.stat().st_size == 0:
            return False, "File is empty"
        
        return True, ""
    
    def load(self, file_path: str) -> LoadResult:
        """
        Load an STL file and return a trimesh mesh.
        
        Args:
            file_path: Path to the STL file
            
        Returns:
            LoadResult containing the mesh or error information
        """
        import time
        start_time = time.perf_counter()
        
        path = Path(file_path)
        file_name = path.name
        file_size = 0
        
        # Validate file
        is_valid, error_msg = self.is_valid_stl_file(file_path)
        if not is_valid:
            self._last_result = LoadResult(
                mesh=None,
                file_path=str(path.absolute()),
                file_name=file_name,
                file_size_bytes=0,
                success=False,
                error_message=error_msg
            )
            return self._last_result
        
        file_size = path.stat().st_size
        
        try:
            # Load the mesh using trimesh
            mesh = trimesh.load(
                file_path,
                file_type='stl',
                force='mesh'  # Ensure we get a Trimesh, not Scene
            )
            
            # Ensure we have a Trimesh object
            if isinstance(mesh, trimesh.Scene):
                # Extract the first mesh from the scene
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in STL file")
            
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Expected Trimesh, got {type(mesh)}")
            
            # Basic preprocessing - use process=True or manual cleanup
            # trimesh.Trimesh automatically processes on load, but we can force it
            if hasattr(mesh, 'process'):
                mesh.process()
            
            load_time = (time.perf_counter() - start_time) * 1000
            
            self._last_result = LoadResult(
                mesh=mesh,
                file_path=str(path.absolute()),
                file_name=file_name,
                file_size_bytes=file_size,
                success=True,
                load_time_ms=load_time
            )
            
        except Exception as e:
            load_time = (time.perf_counter() - start_time) * 1000
            self._last_result = LoadResult(
                mesh=None,
                file_path=str(path.absolute()),
                file_name=file_name,
                file_size_bytes=file_size,
                success=False,
                error_message=str(e),
                load_time_ms=load_time
            )
        
        return self._last_result


def load_stl_file(file_path: str) -> LoadResult:
    """
    Convenience function to load an STL file.
    
    Args:
        file_path: Path to the STL file
        
    Returns:
        LoadResult containing the mesh or error information
    """
    loader = STLLoader()
    return loader.load(file_path)
