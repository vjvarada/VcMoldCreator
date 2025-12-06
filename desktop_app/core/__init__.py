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
from core.mold_cavity import (
    create_mold_cavity,
    perform_csg_subtraction,
    validate_cavity_mesh,
    MoldCavityResult,
    CavityValidation,
    CAVITY_COLOR,
)
from core.mold_half_classification import (
    classify_mold_halves,
    get_mold_half_face_colors,
    MoldHalfClassificationResult,
    MoldHalfColors,
)
from core.tetrahedral_mesh import (
    generate_tetrahedral_mesh,
    tetrahedralize_mesh,
    extract_edges,
    extract_boundary_surface,
    compute_edge_lengths,
    compute_distances_to_mesh,
    build_vertex_adjacency,
    build_edge_index_map,
    compute_edge_weights,
    compute_edge_costs,
    TetrahedralMeshResult,
    PYTETWILD_AVAILABLE,
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
    # Mold cavity
    'create_mold_cavity',
    'perform_csg_subtraction',
    'validate_cavity_mesh',
    'MoldCavityResult',
    'CavityValidation',
    'CAVITY_COLOR',
    # Mold half classification
    'classify_mold_halves',
    'get_mold_half_face_colors',
    'MoldHalfClassificationResult',
    'MoldHalfColors',
    # Tetrahedral mesh
    'generate_tetrahedral_mesh',
    'tetrahedralize_mesh',
    'extract_edges',
    'compute_edge_lengths',
    'compute_distances_to_mesh',
    'build_vertex_adjacency',
    'build_edge_index_map',
    'compute_edge_weights',
    'compute_edge_costs',
    'TetrahedralMeshResult',
    'PYTETWILD_AVAILABLE',
]
