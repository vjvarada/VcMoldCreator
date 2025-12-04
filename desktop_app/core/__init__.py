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
from core.inflated_hull import (
    generate_inflated_hull,
    compute_default_offset,
    compute_smooth_vertex_normals,
    validate_manifold,
    InflatedHullResult,
    ManifoldValidation,
    DEFAULT_INFLATION_PERCENT,
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
    # Inflated hull
    'generate_inflated_hull',
    'compute_default_offset',
    'compute_smooth_vertex_normals',
    'validate_manifold',
    'InflatedHullResult',
    'ManifoldValidation',
    'DEFAULT_INFLATION_PERCENT',
]
