# Core module for mesh operations
from core.stl_loader import STLLoader, load_stl_file, LoadResult
from core.mesh_analysis import MeshAnalyzer, MeshDiagnostics
from core.mesh_repair import MeshRepairer, MeshRepairResult
from core.parting_direction import (
    find_parting_directions,
    compute_visibility_paint,
    get_face_colors,
    PartingDirectionResult,
    VisibilityPaintData,
    PartingColors,
)

__all__ = [
    'STLLoader',
    'load_stl_file',
    'LoadResult',
    'MeshAnalyzer',
    'MeshDiagnostics',
    'MeshRepairer',
    'MeshRepairResult',
    # Parting direction
    'find_parting_directions',
    'compute_visibility_paint',
    'get_face_colors',
    'PartingDirectionResult',
    'VisibilityPaintData',
    'PartingColors',
]
