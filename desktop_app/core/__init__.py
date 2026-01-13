# Core module for mesh operations
from core.stl_loader import STLLoader, load_stl_file, LoadResult
from core.mesh_analysis import (
    MeshAnalyzer,
    MeshDiagnostics,
    compute_height_field,
    build_vertex_neighbors,
    build_face_neighbors,
)
from core.mesh_repair import MeshRepairer, MeshRepairResult, is_meshlib_available
from core.parting_direction import (
    find_parting_directions,
    compute_visibility_paint,
    get_face_colors,
    PartingDirectionResult,
    VisibilityPaintData,
)
from core.pouring_direction import (
    PouringDirectionResult as PouringDirResult,
    PersistencePair,
    evaluate_candidate_directions,
    score_pouring_direction,
    compute_persistence_pairs,
    find_mold_aware_pouring_directions,
    MoldAwarePouringDirections,
)
from core.inflated_hull import (
    generate_inflated_hull,
    compute_default_offset,
    compute_smooth_vertex_normals,
    validate_manifold,
    InflatedHullResult,
    ManifoldValidation,
)
from core.mold_half_classification import (
    classify_mold_halves,
    MoldHalfClassificationResult,
)
from core.tetrahedral_mesh import (
    generate_tetrahedral_mesh,
    tetrahedralize_mesh,
    classify_tetrahedra,
    filter_tetrahedra_outside_part,
    extract_edges,
    extract_boundary_surface,
    compute_edge_lengths,
    compute_distances_to_mesh,
    build_vertex_adjacency,
    build_edge_index_map,
    prepare_parting_surface_data,
    build_edge_to_index_map,
    build_tet_edge_indices,
    compute_cut_edge_flags,
    label_boundary_vertices_direct,
    TetrahedralMeshResult,
    PYTETWILD_AVAILABLE,
)
from core.parting_surface import (
    extract_parting_surface,
    extract_parting_surface_from_tet_result,
    smooth_parting_surface,
    repair_parting_surface,
    remove_small_boundary_loops,
    close_parting_surface_gaps,
    PartingSurfaceResult,
    SmallHoleRemovalResult,
    GapClosingResult,
    MARCHING_TET_TABLE,
    TET_EDGES,
)

__all__ = [
    'STLLoader',
    'load_stl_file',
    'LoadResult',
    'MeshAnalyzer',
    'MeshDiagnostics',
    'MeshRepairer',
    'MeshRepairResult',
    'is_meshlib_available',
    # Mesh analysis utilities
    'compute_height_field',
    'build_vertex_neighbors',
    'build_face_neighbors',
    # Parting direction
    'find_parting_directions',
    'compute_visibility_paint',
    'get_face_colors',
    'PartingDirectionResult',
    'VisibilityPaintData',
    # Pouring direction
    'find_mold_aware_pouring_directions',
    'MoldAwarePouringDirections',
    'PouringDirResult',
    'PersistencePair',
    'evaluate_candidate_directions',
    'score_pouring_direction',
    'compute_persistence_pairs',
    # Inflated hull
    'generate_inflated_hull',
    'compute_default_offset',
    'compute_smooth_vertex_normals',
    'validate_manifold',
    'InflatedHullResult',
    'ManifoldValidation',
    # Mold half classification
    'classify_mold_halves',
    'MoldHalfClassificationResult',
    # Tetrahedral mesh
    'generate_tetrahedral_mesh',
    'tetrahedralize_mesh',
    'extract_edges',
    'compute_edge_lengths',
    'compute_distances_to_mesh',
    'build_vertex_adjacency',
    'build_edge_index_map',
    'prepare_parting_surface_data',
    'build_edge_to_index_map',
    'build_tet_edge_indices',
    'compute_cut_edge_flags',
    'TetrahedralMeshResult',
    'PYTETWILD_AVAILABLE',
    # Parting surface extraction
    'extract_parting_surface',
    'extract_parting_surface_from_tet_result',
    'smooth_parting_surface',
    'repair_parting_surface',
    'remove_small_boundary_loops',
    'close_parting_surface_gaps',
    'PartingSurfaceResult',
    'SmallHoleRemovalResult',
    'GapClosingResult',
    'MARCHING_TET_TABLE',
    'TET_EDGES',
]
