"""
Secondary Membrane Creation and Processing

This module handles the creation of secondary membranes (additional cuts) that are
aware of the primary membrane. Secondary membranes are needed when escape paths of
adjacent vertices traverse different sides of geometric features in the part.

Per Paper Section 4.2 "Additional Membranes":
"Once the silicone mold volume has been partitioned into two pieces, for each piece
we define additional membranes corresponding to features that could prevent the 
mold extraction. A membrane is introduced when the escape paths of two adjacent 
vertices traverse the volume on different sides of a portion of the object M."

Key insight: Secondary membranes are computed PER MOLD HALF (O1 and O2), and they
need to CONNECT to the primary membrane at their boundaries.

The connection between primary and secondary membranes happens at JUNCTION TETRAHEDRA -
tetrahedra that contain both primary cut edges and secondary cut edges.

ORPHAN EDGE HANDLING:
Primary cut edges may not participate in the primary surface if they exist only in 
tetrahedra with invalid configurations (1, 2, 5, or 6 cut edges instead of valid 3 or 4).
These "orphan" edges are added to the secondary membrane to fill gaps in coverage.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import trimesh

logger = logging.getLogger(__name__)


# Import marching tet table for validation
try:
    from .parting_surface import MARCHING_TET_TABLE
except ImportError:
    MARCHING_TET_TABLE = None


@dataclass
class SecondaryMembraneInfo:
    """Information about a secondary membrane and its relationship to the primary."""
    
    # Mold half this secondary belongs to (1=H1, 2=H2)
    mold_half: int
    
    # Secondary cut edges for this membrane
    cut_edges: List[Tuple[int, int]]
    
    # Tetrahedra that contain only secondary cuts (interior to the secondary membrane)
    interior_tets: Set[int]
    
    # Tetrahedra that contain BOTH primary and secondary cuts (junctions)
    junction_tets: Set[int]
    
    # Edges where secondary meets primary (at junction tets)
    junction_edges: Set[Tuple[int, int]]
    
    # Binary vertex labels for this secondary membrane (SideA=1, SideB=2)
    vertex_labels: Optional[np.ndarray] = None


def classify_secondary_membranes_by_mold_half(
    tet_result,
    part_mesh: trimesh.Trimesh
) -> Tuple[List[SecondaryMembraneInfo], List[SecondaryMembraneInfo]]:
    """
    Classify secondary cut edges by which mold half they belong to.
    
    Secondary membranes are computed WITHIN each mold half (H1 or H2). This function
    separates secondary cuts into those that are in H1's volume versus H2's volume.
    
    Args:
        tet_result: TetrahedralMeshResult with secondary_cut_edges and primary_cut_edges
        part_mesh: Original part mesh
        
    Returns:
        Tuple of (h1_membranes, h2_membranes) where each is a list of SecondaryMembraneInfo
    """
    if tet_result.secondary_cut_edges is None or len(tet_result.secondary_cut_edges) == 0:
        return [], []
    
    if tet_result.seed_escape_labels is None:
        raise ValueError("Dijkstra escape labels required for secondary membrane classification")
    
    # Build mapping from vertex index to interior array index
    vertex_to_interior_idx = {}
    for idx, v in enumerate(tet_result.seed_vertex_indices):
        vertex_to_interior_idx[v] = idx
    
    # Classify each secondary cut edge by the escape label of its endpoints
    # Both endpoints should have the SAME escape label (that's the definition of secondary cuts)
    h1_edges = []  # Secondary cuts in H1 volume
    h2_edges = []  # Secondary cuts in H2 volume
    
    for vi, vj in tet_result.secondary_cut_edges:
        # Get escape labels
        idx_i = vertex_to_interior_idx.get(vi)
        idx_j = vertex_to_interior_idx.get(vj)
        
        if idx_i is None or idx_j is None:
            # One or both vertices not in interior - skip
            continue
        
        label_i = tet_result.seed_escape_labels[idx_i]
        label_j = tet_result.seed_escape_labels[idx_j]
        
        # Secondary cuts should have same label
        if label_i == label_j:
            if label_i == 1:
                h1_edges.append((vi, vj))
            elif label_i == 2:
                h2_edges.append((vi, vj))
    
    logger.info(f"Secondary edges by mold half: {len(h1_edges)} in H1, {len(h2_edges)} in H2")
    
    # Build membrane info for each mold half
    h1_membranes = _build_membrane_info(tet_result, h1_edges, mold_half=1)
    h2_membranes = _build_membrane_info(tet_result, h2_edges, mold_half=2)
    
    return h1_membranes, h2_membranes


def _build_membrane_info(
    tet_result,
    secondary_edges: List[Tuple[int, int]],
    mold_half: int
) -> List[SecondaryMembraneInfo]:
    """
    Build SecondaryMembraneInfo for edges in a mold half.
    
    IMPORTANT: This function now separates secondary edges into CONNECTED COMPONENTS
    before building membrane info. Each connected component becomes its own
    SecondaryMembraneInfo, preventing label conflicts between geometrically
    separate secondary membranes (per Bloomenthal 1995, Section 4: multiple
    regions within a single propagation cell).
    
    Args:
        tet_result: TetrahedralMeshResult with edge data
        secondary_edges: List of (vi, vj) secondary cut edges for this mold half
        mold_half: Which mold half (1=H1, 2=H2)
        
    Returns:
        List of SecondaryMembraneInfo, one per connected component
    """
    if len(secondary_edges) == 0:
        return []
    
    # =========================================================================
    # Step 1: Separate secondary edges into connected components
    # Two edges are connected if they share a vertex.
    # =========================================================================
    edge_set_canonical = {(min(vi, vj), max(vi, vj)) for vi, vj in secondary_edges}
    
    # Build vertex-to-edges adjacency
    vertex_to_edges: Dict[int, Set[Tuple[int, int]]] = {}
    for vi, vj in secondary_edges:
        key = (min(vi, vj), max(vi, vj))
        vertex_to_edges.setdefault(vi, set()).add(key)
        vertex_to_edges.setdefault(vj, set()).add(key)
    
    # BFS to find connected components
    visited_edges: Set[Tuple[int, int]] = set()
    components: List[List[Tuple[int, int]]] = []
    
    for start_edge in edge_set_canonical:
        if start_edge in visited_edges:
            continue
        
        component = []
        queue = [start_edge]
        while queue:
            edge = queue.pop(0)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            component.append(edge)
            
            # Find adjacent edges (share a vertex)
            vi, vj = edge
            for neighbor_edge in vertex_to_edges.get(vi, set()):
                if neighbor_edge not in visited_edges:
                    queue.append(neighbor_edge)
            for neighbor_edge in vertex_to_edges.get(vj, set()):
                if neighbor_edge not in visited_edges:
                    queue.append(neighbor_edge)
        
        if component:
            components.append(component)
    
    logger.info(f"Mold half {mold_half}: {len(secondary_edges)} secondary edges "
                f"form {len(components)} connected components")
    
    # =========================================================================
    # Step 2: Build a SecondaryMembraneInfo for EACH connected component
    # =========================================================================
    primary_edge_set = set()
    if tet_result.primary_cut_edges is not None:
        primary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in tet_result.primary_cut_edges}
    
    edge_to_index = tet_result.edge_to_index if tet_result.edge_to_index is not None else {}
    if not edge_to_index:
        from . import tetrahedral_mesh as tm
        edge_to_index = tm.build_edge_to_index_map(tet_result.edges)
    
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    membranes = []
    for comp_idx, comp_edges in enumerate(components):
        comp_edge_set = set(comp_edges)
        
        interior_tets = set()
        junction_tets = set()
        junction_edges = set()
        
        for t_idx, tet in enumerate(tet_result.tetrahedra):
            has_secondary = False
            has_primary = False
            tet_secondary_edges = []
            tet_primary_edges = []
            
            for i, j in edge_pairs:
                vi, vj = int(tet[i]), int(tet[j])
                key = (min(vi, vj), max(vi, vj))
                
                if key in comp_edge_set:
                    has_secondary = True
                    tet_secondary_edges.append(key)
                if key in primary_edge_set:
                    has_primary = True
                    tet_primary_edges.append(key)
            
            if has_secondary:
                if has_primary:
                    junction_tets.add(t_idx)
                    junction_edges.update(tet_primary_edges)
                else:
                    interior_tets.add(t_idx)
        
        membrane = SecondaryMembraneInfo(
            mold_half=mold_half,
            cut_edges=list(comp_edges),
            interior_tets=interior_tets,
            junction_tets=junction_tets,
            junction_edges=junction_edges
        )
        membranes.append(membrane)
        
        logger.info(f"  Component {comp_idx}: {len(comp_edges)} edges, "
                    f"{len(interior_tets)} interior tets, {len(junction_tets)} junction tets")
    
    return membranes


def compute_secondary_vertex_labels_with_primary_awareness(
    tet_result,
    secondary_edges: List[Tuple[int, int]],
    mold_half: int
) -> np.ndarray:
    """
    Compute vertex labels for secondary surface extraction that are AWARE of the primary membrane.
    
    This ensures that at JUNCTION TETRAHEDRA (where secondary meets primary), the vertex
    labels are consistent with the primary membrane's labeling.
    
    The key insight is:
    - Secondary membranes exist WITHIN a mold half (all secondary edge endpoints escape to same H1 or H2)
    - At junctions, the primary membrane already divides space into H1 and H2
    - The secondary membrane adds a THIRD region within the mold half
    
    For proper Marching Tetrahedra, we need to assign labels such that:
    1. Pure secondary tets get binary labels (A=1, B=2) based on which side of secondary cuts
    2. Junction tets get 3-region labels: primary side (from vertex_mold_labels) + secondary side
    
    SIMPLIFICATION: Since marching tet with 3 regions is complex, we use an alternative approach:
    - At junction tets, treat the primary cut as a BOUNDARY condition
    - Secondary vertices on the primary cut get labels from vertex_mold_labels
    - This makes secondary surface TERMINATE at the primary membrane (they connect there)
    
    Args:
        tet_result: TetrahedralMeshResult with computed data
        secondary_edges: List of secondary cut edges for this mold half
        mold_half: Which mold half (1=H1, 2=H2) these secondary edges are in
        
    Returns:
        (N,) array of vertex labels: 1=SideA, 2=SideB, 0=not involved
    """
    from collections import deque
    
    n_vertices = len(tet_result.vertices)
    edges = tet_result.edges
    n_edges = len(edges)
    
    if len(secondary_edges) == 0:
        return np.zeros(n_vertices, dtype=np.int8)
    
    # Build edge-to-index map
    edge_to_index = tet_result.edge_to_index
    if edge_to_index is None:
        from . import tetrahedral_mesh as tm
        edge_to_index = tm.build_edge_to_index_map(edges)
    
    # Mark secondary cut edges
    secondary_cut_set = {(min(vi, vj), max(vi, vj)) for vi, vj in secondary_edges}
    
    # Mark primary cut edges
    primary_cut_set = set()
    if tet_result.primary_cut_edges is not None:
        primary_cut_set = {(min(vi, vj), max(vi, vj)) for vi, vj in tet_result.primary_cut_edges}
    
    # Build adjacency lists
    # Same-side adjacency: edges that are NOT secondary cuts
    # Cross-cut adjacency: edges that ARE secondary cuts
    same_side_adj = [[] for _ in range(n_vertices)]
    cross_cut_adj = [[] for _ in range(n_vertices)]
    
    for e_idx in range(n_edges):
        v0, v1 = int(edges[e_idx, 0]), int(edges[e_idx, 1])
        key = (min(v0, v1), max(v0, v1))
        
        if key in secondary_cut_set:
            cross_cut_adj[v0].append(v1)
            cross_cut_adj[v1].append(v0)
        elif key not in primary_cut_set:
            # Non-cut edge (not primary, not secondary) - same side
            same_side_adj[v0].append(v1)
            same_side_adj[v1].append(v0)
        # If it's a primary cut edge, we DON'T add it - this creates the boundary
    
    # Find vertices involved in tetrahedra with secondary cuts
    involved_vertices = set()
    for vi, vj in secondary_edges:
        involved_vertices.add(vi)
        involved_vertices.add(vj)
    
    # Expand to all vertices in tetrahedra that contain secondary cut edges
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for tet in tet_result.tetrahedra:
        tet_verts = set(int(v) for v in tet)
        for i, j in edge_pairs:
            vi, vj = int(tet[i]), int(tet[j])
            key = (min(vi, vj), max(vi, vj))
            if key in secondary_cut_set:
                involved_vertices.update(tet_verts)
                break
    
    logger.info(f"Secondary labeling (mold half {mold_half}): {len(involved_vertices)} vertices involved")
    
    # Initialize labels
    labels = np.zeros(n_vertices, dtype=np.int8)
    
    # BFS to assign labels
    # Start from a vertex on a secondary cut edge and assign label 1
    # Propagate same label via same_side_adj
    # Propagate opposite label via cross_cut_adj
    
    labeled_count = 0
    component_count = 0
    
    for start_v in involved_vertices:
        if labels[start_v] != 0:
            continue
        
        component_count += 1
        queue = deque()
        queue.append((start_v, 1))
        
        while queue:
            v, lbl = queue.popleft()
            
            if labels[v] != 0:
                # Already labeled - consistency check
                if labels[v] != lbl:
                    # Inconsistency can happen at complex junctions
                    pass
                continue
            
            labels[v] = lbl
            labeled_count += 1
            
            # Propagate same label via non-cut edges
            for neighbor in same_side_adj[v]:
                if neighbor in involved_vertices and labels[neighbor] == 0:
                    queue.append((neighbor, lbl))
            
            # Propagate opposite label via secondary cut edges
            opposite_lbl = 2 if lbl == 1 else 1
            for neighbor in cross_cut_adj[v]:
                if neighbor in involved_vertices and labels[neighbor] == 0:
                    queue.append((neighbor, opposite_lbl))
    
    n_side_a = np.sum(labels == 1)
    n_side_b = np.sum(labels == 2)
    
    logger.info(f"Secondary labels computed: {n_side_a} SideA, {n_side_b} SideB, {component_count} components")
    
    return labels


def compute_multi_region_secondary_labels(
    tet_result,
    membrane_components: List['SecondaryMembraneInfo'],
    mold_half: int,
    primary_region_offset: int = 2
) -> np.ndarray:
    """
    Compute multi-region vertex labels for multiple secondary membrane components.
    
    Per Bloomenthal & Ferguson 1995 Section 4: When multiple surfaces meet,
    we need a multiple regionalization of space rather than binary. Each connected
    secondary membrane component gets its own pair of region labels, ensuring that
    the BFS labeling for one component doesn't conflict with another.
    
    Region assignment scheme:
    - Regions 1, 2: Reserved for the primary membrane (H1, H2)
    - Component 0: regions (3, 4) → SideA=3, SideB=4
    - Component 1: regions (5, 6) → SideA=5, SideB=6
    - Component N: regions (3+2N, 4+2N)
    
    At JUNCTION TETRAHEDRA where secondary meets primary, the primary edge labels
    (H1=1, H2=2) are treated as boundary conditions. Primary edges act as barriers
    that stop the BFS propagation, making the secondary surface terminate at the
    primary membrane.
    
    At CROSS-COMPONENT JUNCTIONS where two secondary components share a tetrahedron,
    each component's edges get their own region pair, and the resulting 5-edge or
    6-edge tet configurations are resolved using face/inner vertices per Bloomenthal.
    
    Args:
        tet_result: TetrahedralMeshResult with computed data
        membrane_components: List of SecondaryMembraneInfo, one per connected component
        mold_half: Which mold half (1=H1, 2=H2)
        primary_region_offset: First region label to use for secondary (default 2 → starts at 3)
        
    Returns:
        (N,) int16 array of vertex region labels:
        0 = not involved in any secondary membrane
        1 = primary H1, 2 = primary H2 (for vertices on primary boundary)
        3, 4 = component 0 SideA/SideB
        5, 6 = component 1 SideA/SideB
        etc.
    """
    from collections import deque
    
    n_vertices = len(tet_result.vertices)
    edges = tet_result.edges
    n_edges = len(edges)
    
    if not membrane_components:
        return np.zeros(n_vertices, dtype=np.int16)
    
    # Build primary cut edge set for barrier detection
    primary_cut_set = set()
    if tet_result.primary_cut_edges is not None:
        primary_cut_set = {(min(vi, vj), max(vi, vj)) for vi, vj in tet_result.primary_cut_edges}
    
    # Initialize region labels
    region_labels = np.zeros(n_vertices, dtype=np.int16)
    
    # Process each component independently with its own region pair
    edge_pairs_local = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for comp_idx, membrane in enumerate(membrane_components):
        region_a = primary_region_offset + 1 + (2 * comp_idx)      # 3, 5, 7, ...
        region_b = primary_region_offset + 1 + (2 * comp_idx) + 1  # 4, 6, 8, ...
        
        # Build cut edge set for this component
        comp_cut_set = {(min(vi, vj), max(vi, vj)) for vi, vj in membrane.cut_edges}
        
        # Build adjacency for this component
        # same_side: edges that are NOT this component's cuts AND not primary cuts
        # cross_cut: edges that ARE this component's cuts
        same_side_adj: Dict[int, List[int]] = {}
        cross_cut_adj: Dict[int, List[int]] = {}
        
        for e_idx in range(n_edges):
            v0, v1 = int(edges[e_idx, 0]), int(edges[e_idx, 1])
            key = (min(v0, v1), max(v0, v1))
            
            if key in comp_cut_set:
                cross_cut_adj.setdefault(v0, []).append(v1)
                cross_cut_adj.setdefault(v1, []).append(v0)
            elif key not in primary_cut_set:
                same_side_adj.setdefault(v0, []).append(v1)
                same_side_adj.setdefault(v1, []).append(v0)
            # Primary edges are barriers — not added to either adjacency
        
        # Identify all vertices involved in tetrahedra with this component's edges
        involved_vertices = set()
        for vi, vj in membrane.cut_edges:
            involved_vertices.add(vi)
            involved_vertices.add(vj)
        
        for tet in tet_result.tetrahedra:
            tet_verts = set(int(v) for v in tet)
            for i, j in edge_pairs_local:
                vi, vj = int(tet[i]), int(tet[j])
                key = (min(vi, vj), max(vi, vj))
                if key in comp_cut_set:
                    involved_vertices.update(tet_verts)
                    break
        
        # BFS to assign region labels for this component
        labeled_count = 0
        for start_v in involved_vertices:
            if region_labels[start_v] != 0:
                # Already labeled by this or another component — skip
                continue
            
            queue = deque()
            queue.append((start_v, region_a))
            
            while queue:
                v, lbl = queue.popleft()
                
                if region_labels[v] != 0:
                    continue
                
                region_labels[v] = lbl
                labeled_count += 1
                
                # Propagate same label via non-cut edges
                for neighbor in same_side_adj.get(v, []):
                    if neighbor in involved_vertices and region_labels[neighbor] == 0:
                        queue.append((neighbor, lbl))
                
                # Propagate opposite label via component's cut edges
                opposite_lbl = region_b if lbl == region_a else region_a
                for neighbor in cross_cut_adj.get(v, []):
                    if neighbor in involved_vertices and region_labels[neighbor] == 0:
                        queue.append((neighbor, opposite_lbl))
        
        n_a = np.sum(region_labels == region_a)
        n_b = np.sum(region_labels == region_b)
        logger.info(f"  Component {comp_idx} (regions {region_a}/{region_b}): "
                    f"{n_a} SideA, {n_b} SideB, {labeled_count} total labeled")
    
    # Assign primary boundary labels (1, 2) for vertices on the primary boundary
    # that are also involved in secondary tets (junction vertices)
    if tet_result.vertex_mold_labels is not None:
        primary_boundary_verts = np.where(
            (tet_result.boundary_labels == 1) | (tet_result.boundary_labels == 2)
        )[0]
        for v in primary_boundary_verts:
            if region_labels[v] == 0:
                region_labels[v] = tet_result.boundary_labels[v]
    
    n_components = len(membrane_components)
    n_labeled = np.sum(region_labels != 0)
    logger.info(f"Multi-region labeling complete: {n_components} components, "
                f"{n_labeled}/{n_vertices} vertices labeled, "
                f"max region = {np.max(region_labels)}")
    
    return region_labels


def build_region_pairs_of_interest(
    membrane_components: List['SecondaryMembraneInfo'],
    include_primary: bool = True,
    primary_region_offset: int = 2
) -> List[Tuple[int, int]]:
    """
    Build the set of region-pairs whose separating surfaces should be polygonized.
    
    Per Bloomenthal 1995 Section 3: "Our surface definition consists of two parts:
    an integer-valued region function and a set of region-pairs of interest."
    
    For secondary membranes, the region-pairs of interest are:
    - Each secondary component's (region_a, region_b) pair → the membrane surface
    - Optionally (1, 2) → the primary parting surface (handled separately)
    
    Cross-component pairs (e.g. (3, 5)) are NOT surfaces of interest — they would
    create spurious surfaces between unrelated secondary membranes. The Bloomenthal
    polygonizer uses this set to decide which faces to generate.
    
    Args:
        membrane_components: List of SecondaryMembraneInfo for one mold half
        include_primary: If True, include (1, 2) for primary surface
        primary_region_offset: Offset for secondary region labels (default 2)
        
    Returns:
        List of (region_a, region_b) pairs, canonically ordered (a < b)
    """
    pairs = []
    
    if include_primary:
        pairs.append((1, 2))
    
    for comp_idx in range(len(membrane_components)):
        region_a = primary_region_offset + 1 + (2 * comp_idx)
        region_b = primary_region_offset + 1 + (2 * comp_idx) + 1
        pairs.append((min(region_a, region_b), max(region_a, region_b)))
    
    return pairs


def extract_secondary_surfaces_per_component(
    tet_result,
    part_mesh: 'trimesh.Trimesh',
    include_orphan_primary: bool = True,
    extend_to_primary: bool = True
) -> List[Tuple['SecondaryMembraneInfo', 'PartingSurfaceResult']]:
    """
    Extract secondary surfaces PER CONNECTED COMPONENT.
    
    This is the main entry point for multi-component secondary surface extraction.
    Instead of extracting all secondary membranes as a single merged surface, this
    function:
    
    1. Classifies secondary edges by mold half (H1/H2)
    2. Separates edges into connected components within each half
    3. Extracts a separate PartingSurfaceResult for each component
    4. At junction tets (where components meet primary or each other),
       5/6-edge configs are resolved using face/inner vertices (Bloomenthal)
    
    The per-component extraction allows:
    - Independent smoothing per membrane (different boundary conditions)
    - Proper junction handling between multiple secondary membranes
    - Better tracking of which membrane belongs to which geometric feature
    
    Per Paper Section 4.2: "Once the silicone mold volume has been partitioned
    into two pieces, for each piece we define additional membranes."
    
    Args:
        tet_result: TetrahedralMeshResult with secondary_cut_edges computed
        part_mesh: Original part mesh for classification
        include_orphan_primary: If True, enhance secondary edges with orphan primary edges
        extend_to_primary: If True, include junction primary edges for connectivity
        
    Returns:
        List of (SecondaryMembraneInfo, PartingSurfaceResult) tuples.
        Each tuple contains the membrane metadata and its extracted surface.
        Empty list if no secondary cut edges exist.
    """
    from .parting_surface import extract_parting_surface_from_tet_result
    from . import tetrahedral_mesh as tm
    
    if tet_result.secondary_cut_edges is None or len(tet_result.secondary_cut_edges) == 0:
        logger.info("No secondary cut edges — nothing to extract")
        return []
    
    # Ensure data structures are ready
    if tet_result.tet_edge_indices is None:
        tet_result = tm.prepare_parting_surface_data(tet_result)
    
    # Step 1: Classify into mold halves and connected components
    h1_membranes, h2_membranes = classify_secondary_membranes_by_mold_half(
        tet_result, part_mesh
    )
    
    all_membranes = h1_membranes + h2_membranes
    
    if not all_membranes:
        logger.warning("No secondary membrane components found after classification")
        return []
    
    logger.info(f"Processing {len(all_membranes)} secondary membrane components "
                f"({len(h1_membranes)} in H1, {len(h2_membranes)} in H2)")
    
    # Step 2: Optionally enhance with orphan primary edges
    if include_orphan_primary:
        _used, unused_edges, _cfg = find_all_unused_primary_edges(tet_result)
        if unused_edges:
            logger.info(f"Found {len(unused_edges)} orphan primary edges to distribute")
            _distribute_orphan_edges_to_components(all_membranes, unused_edges, tet_result)
    
    # Step 3: Extract surface per component
    results: List[Tuple[SecondaryMembraneInfo, 'PartingSurfaceResult']] = []
    
    for comp_idx, membrane in enumerate(all_membranes):
        logger.info(f"Extracting component {comp_idx} (H{membrane.mold_half}): "
                    f"{len(membrane.cut_edges)} edges, "
                    f"{len(membrane.junction_tets)} junction tets")
        
        # Build cut flags for just this component's edges
        if extend_to_primary:
            cut_flags = compute_extended_secondary_cut_flags_improved(
                tet_result,
                membrane.cut_edges,
                include_junction_primary_edges=True
            )
        else:
            n_edges = len(tet_result.edges)
            edge_to_index = tet_result.edge_to_index or {}
            cut_flags = np.zeros(n_edges, dtype=np.int8)
            for vi, vj in membrane.cut_edges:
                key = (min(vi, vj), max(vi, vj))
                if key in edge_to_index:
                    cut_flags[edge_to_index[key]] = 1
        
        n_cut = np.sum(cut_flags)
        if n_cut == 0:
            logger.warning(f"  Component {comp_idx}: no cut edges after flag computation — skipping")
            continue
        
        # Build escape distance array for orientation
        vertex_escape_distances = None
        if (tet_result.seed_distances is not None and
            tet_result.seed_vertex_indices is not None and
            len(tet_result.seed_distances) == len(tet_result.seed_vertex_indices)):
            n_verts = len(tet_result.vertices)
            vertex_escape_distances = np.full(n_verts, np.inf, dtype=np.float64)
            for idx, v_global in enumerate(tet_result.seed_vertex_indices):
                vertex_escape_distances[v_global] = tet_result.seed_distances[idx]
            if tet_result.boundary_labels is not None:
                boundary_mask = (tet_result.boundary_labels == 1) | (tet_result.boundary_labels == 2)
                vertex_escape_distances[boundary_mask] = 0.0
        
        # Build primary vertex mask for junction detection
        primary_vertex_mask = None
        if tet_result.primary_cut_edges is not None:
            n_verts = len(tet_result.vertices)
            primary_vertex_mask = np.zeros(n_verts, dtype=bool)
            for vi, vj in tet_result.primary_cut_edges:
                primary_vertex_mask[vi] = True
                primary_vertex_mask[vj] = True
        
        # Import and call the core extraction function
        from .parting_surface import extract_parting_surface
        
        surface_result = extract_parting_surface(
            vertices=tet_result.vertices,
            tetrahedra=tet_result.tetrahedra,
            edges=tet_result.edges,
            cut_edge_flags=cut_flags,
            tet_edge_indices=tet_result.tet_edge_indices,
            use_original_vertices=True,
            vertices_original=tet_result.vertices_original,
            boundary_labels=tet_result.boundary_labels,
            vertex_mold_labels=None,  # Secondary: no label-derived cuts
            vertex_escape_distances=vertex_escape_distances,
            use_label_derived_cuts=False,
            is_secondary=True,
            primary_cut_vertex_mask=primary_vertex_mask
        )
        
        if surface_result.mesh is not None and surface_result.num_faces > 0:
            results.append((membrane, surface_result))
            logger.info(f"  Component {comp_idx}: extracted {surface_result.num_vertices} verts, "
                        f"{surface_result.num_faces} faces")
        else:
            logger.warning(f"  Component {comp_idx}: empty surface — skipped")
    
    logger.info(f"Extracted {len(results)} non-empty secondary surfaces "
                f"from {len(all_membranes)} components")
    return results


def _distribute_orphan_edges_to_components(
    membranes: List[SecondaryMembraneInfo],
    orphan_edges: List[Tuple[int, int]],
    tet_result
) -> None:
    """
    Distribute orphan primary edges to the nearest secondary membrane component.
    
    Orphan primary edges (primary cut edges not used by the primary surface due to
    invalid tet configs) are assigned to the secondary component that shares the
    most vertices with them. This modifies the membrane objects in-place.
    
    Args:
        membranes: List of SecondaryMembraneInfo to augment (modified in-place)
        orphan_edges: List of orphan primary cut edges
        tet_result: TetrahedralMeshResult for context
    """
    if not membranes or not orphan_edges:
        return
    
    # Build vertex sets for each component
    comp_vertex_sets = []
    for membrane in membranes:
        verts = set()
        for vi, vj in membrane.cut_edges:
            verts.add(vi)
            verts.add(vj)
        comp_vertex_sets.append(verts)
    
    assigned_count = 0
    for vi, vj in orphan_edges:
        # Find component with most vertex overlap
        best_comp = -1
        best_overlap = 0
        for c_idx, vset in enumerate(comp_vertex_sets):
            overlap = (vi in vset) + (vj in vset)
            if overlap > best_overlap:
                best_overlap = overlap
                best_comp = c_idx
        
        if best_comp >= 0 and best_overlap > 0:
            key = (min(vi, vj), max(vi, vj))
            membranes[best_comp].cut_edges.append(key)
            comp_vertex_sets[best_comp].add(vi)
            comp_vertex_sets[best_comp].add(vj)
            assigned_count += 1
    
    logger.info(f"Distributed {assigned_count}/{len(orphan_edges)} orphan edges to components")


def compute_extended_secondary_cut_flags_improved(
    tet_result,
    secondary_edges: List[Tuple[int, int]],
    include_junction_primary_edges: bool = True
) -> np.ndarray:
    """
    Compute extended secondary cut flags with improved handling of junction tetrahedra.
    
    This version properly handles the connection between secondary and primary membranes:
    1. Marks all secondary cut edges
    2. At junction tetrahedra (containing both primary and secondary), optionally includes
       primary edges so the secondary surface can connect to the primary
    
    The key improvement: We only include primary edges that are ADJACENT to secondary edges
    within the junction tetrahedra, not ALL primary edges in the tet.
    
    Args:
        tet_result: TetrahedralMeshResult
        secondary_edges: List of secondary cut edges
        include_junction_primary_edges: If True, include primary edges at junctions
        
    Returns:
        (E,) boolean array of extended cut flags
    """
    n_edges = len(tet_result.edges)
    
    # Build edge-to-index map
    edge_to_index = tet_result.edge_to_index
    if edge_to_index is None:
        from . import tetrahedral_mesh as tm
        edge_to_index = tm.build_edge_to_index_map(tet_result.edges)
    
    # Start with secondary flags
    cut_flags = np.zeros(n_edges, dtype=np.int8)
    secondary_edge_set = set()
    
    for vi, vj in secondary_edges:
        key = (min(vi, vj), max(vi, vj))
        secondary_edge_set.add(key)
        if key in edge_to_index:
            cut_flags[edge_to_index[key]] = 1
    
    if not include_junction_primary_edges:
        return cut_flags
    
    if tet_result.primary_cut_edges is None or len(tet_result.primary_cut_edges) == 0:
        return cut_flags
    
    # Build primary edge set
    primary_edge_set = set()
    for vi, vj in tet_result.primary_cut_edges:
        key = (min(vi, vj), max(vi, vj))
        primary_edge_set.add(key)
    
    # Find junction tetrahedra and add adjacent primary edges
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    n_junction_edges_added = 0
    
    for t_idx, tet in enumerate(tet_result.tetrahedra):
        # Collect secondary and primary edges in this tet
        tet_secondary_edges = []
        tet_primary_edges = []
        
        for i, j in edge_pairs:
            vi, vj = int(tet[i]), int(tet[j])
            key = (min(vi, vj), max(vi, vj))
            
            if key in secondary_edge_set:
                tet_secondary_edges.append((key, set([vi, vj])))
            if key in primary_edge_set:
                tet_primary_edges.append((key, set([vi, vj])))
        
        # If this is a junction tet
        if tet_secondary_edges and tet_primary_edges:
            # Add primary edges that SHARE A VERTEX with secondary edges
            # This ensures proper connection at the junction
            secondary_vertices = set()
            for _, verts in tet_secondary_edges:
                secondary_vertices.update(verts)
            
            for p_key, p_verts in tet_primary_edges:
                # Check if primary edge shares a vertex with any secondary edge
                if p_verts & secondary_vertices:
                    if p_key in edge_to_index and cut_flags[edge_to_index[p_key]] == 0:
                        cut_flags[edge_to_index[p_key]] = 1
                        n_junction_edges_added += 1
    
    logger.info(f"Extended secondary flags: {len(secondary_edges)} secondary + {n_junction_edges_added} junction primary")
    
    return cut_flags


def find_all_unused_primary_edges(
    tet_result,
    vertex_mold_labels: np.ndarray = None
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Dict[int, int]]:
    """
    Find ALL primary cut edges that don't contribute to the primary surface.
    
    This function simulates the Marching Tetrahedra process to identify which
    edges would actually be used in triangle generation. Any primary edge not
    used is returned as "unused".
    
    The key insight is that an edge is USED if:
    1. It's in a tetrahedron with a valid configuration (3 or 4 cut edges)
    2. The edge is one of the cut edges in that configuration
    
    Args:
        tet_result: TetrahedralMeshResult
        vertex_mold_labels: Optional labels array (uses tet_result.vertex_mold_labels if None)
        
    Returns:
        Tuple of:
        - used_edges: Set of primary edges that ARE used in the primary surface
        - unused_edges: Set of primary edges that are NOT used
        - config_counts: Dict mapping n_cut_edges to count of tets with that config
    """
    if tet_result.primary_cut_edges is None:
        return set(), set(), {}
    
    if vertex_mold_labels is None:
        vertex_mold_labels = tet_result.vertex_mold_labels
    
    if vertex_mold_labels is None:
        logger.warning("No vertex_mold_labels available - cannot analyze edge usage")
        return set(), set(), {}
    
    primary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in tet_result.primary_cut_edges}
    
    used_edges = set()
    config_counts = {}
    
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for tet in tet_result.tetrahedra:
        # Compute the configuration for this tet
        config = 0
        tet_cut_edges = []  # (global_key, local_idx) pairs
        
        for local_e, (i, j) in enumerate(edge_pairs):
            vi, vj = int(tet[i]), int(tet[j])
            label_i = vertex_mold_labels[vi]
            label_j = vertex_mold_labels[vj]
            
            # Edge is cut if labels differ and both are valid (1 or 2)
            if label_i != label_j and label_i in (1, 2) and label_j in (1, 2):
                config |= (1 << local_e)
                key = (min(vi, vj), max(vi, vj))
                if key in primary_edge_set:
                    tet_cut_edges.append((key, local_e))
        
        n_cut = bin(config).count('1')
        config_counts[n_cut] = config_counts.get(n_cut, 0) + 1
        
        # Check if this is a valid config (3 or 4 cut edges produce triangles)
        if n_cut in (3, 4):
            # All cut edges in this tet contribute to triangles
            for key, _ in tet_cut_edges:
                used_edges.add(key)
    
    # Unused edges = all primary edges minus used edges
    unused_edges = primary_edge_set - used_edges
    
    logger.info(f"Primary edge usage analysis:")
    logger.info(f"  Total primary cut edges: {len(primary_edge_set)}")
    logger.info(f"  Used in primary surface: {len(used_edges)}")
    logger.info(f"  NOT used (available for secondary): {len(unused_edges)}")
    logger.info(f"  Tet config distribution: {dict(sorted(config_counts.items()))}")
    
    return used_edges, unused_edges, config_counts



def find_primary_adjacent_edges_in_secondary_tets(
    tet_result,
    secondary_edges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Find primary cut edges that should be included to connect secondary membranes.
    
    These are primary edges in tetrahedra that:
    1. Contain at least one secondary cut edge
    2. The primary edge shares a vertex with a secondary edge in that tet
    
    This ensures the secondary surface can properly connect to the primary membrane.
    
    Args:
        tet_result: TetrahedralMeshResult
        secondary_edges: List of secondary cut edges
        
    Returns:
        List of (vi, vj) primary edges to include
    """
    if not secondary_edges:
        return []
    
    if tet_result.primary_cut_edges is None or len(tet_result.primary_cut_edges) == 0:
        return []
    
    secondary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in secondary_edges}
    primary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in tet_result.primary_cut_edges}
    
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    adjacent_primary_edges = set()
    
    for tet in tet_result.tetrahedra:
        tet_secondary_edges = []
        tet_primary_edges = []
        
        for i, j in edge_pairs:
            vi, vj = int(tet[i]), int(tet[j])
            key = (min(vi, vj), max(vi, vj))
            
            if key in secondary_edge_set:
                tet_secondary_edges.append((vi, vj))
            if key in primary_edge_set:
                tet_primary_edges.append((vi, vj))
        
        if tet_secondary_edges and tet_primary_edges:
            # Collect vertices on secondary edges
            secondary_verts = set()
            for vi, vj in tet_secondary_edges:
                secondary_verts.add(vi)
                secondary_verts.add(vj)
            
            # Add primary edges that share a vertex
            for vi, vj in tet_primary_edges:
                if vi in secondary_verts or vj in secondary_verts:
                    adjacent_primary_edges.add((min(vi, vj), max(vi, vj)))
    
    return list(adjacent_primary_edges)

def find_orphan_primary_cut_edges(
    tet_result,
    vertex_mold_labels: np.ndarray = None
) -> Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:
    """
    Find primary cut edges that are NOT used by the primary parting surface.
    
    In Marching Tetrahedra, only tetrahedra with exactly 3 or 4 cut edges produce
    triangles. Tetrahedra with 1, 2, 5, or 6 cut edges are "invalid" configurations
    and produce NO triangles.
    
    This function identifies primary cut edges that exist ONLY in tetrahedra with
    invalid configurations (0, 1, 2, 5, or 6 cut edges). These "orphan" edges could
    potentially contribute to secondary membranes.
    
    Args:
        tet_result: TetrahedralMeshResult with primary_cut_edges
        vertex_mold_labels: (N,) array of vertex labels (1=H1, 2=H2, 0=unlabeled)
                           If None, uses tet_result.vertex_mold_labels
    
    Returns:
        Tuple of:
        - orphan_edges: List of (vi, vj) primary cut edges not used by primary surface
        - tet_configs: Dict mapping tet_idx to list of cut edge indices (0-5) for debugging
    """
    if tet_result.primary_cut_edges is None or len(tet_result.primary_cut_edges) == 0:
        return [], {}
    
    if vertex_mold_labels is None:
        vertex_mold_labels = tet_result.vertex_mold_labels
    
    if vertex_mold_labels is None:
        logger.warning("Cannot find orphan edges - no vertex_mold_labels available")
        return [], {}
    
    # Build set of primary cut edges
    primary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in tet_result.primary_cut_edges}
    
    # Track which primary edges appear in VALID (3 or 4 cut) configurations
    edges_in_valid_configs = set()
    
    # Track which primary edges appear in INVALID configurations
    edges_in_invalid_configs = set()
    
    # Track ALL primary edges that appear in ANY tetrahedra
    edges_in_any_tet = set()
    
    # Debug: track tet configurations
    tet_configs = {}
    
    # Edge pairs within a tetrahedron (local vertex indices)
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    n_valid_tets = 0
    n_invalid_tets = 0
    config_distribution = {}  # Track how many tets have each config size
    
    for t_idx, tet in enumerate(tet_result.tetrahedra):
        # Compute configuration from vertex labels
        # An edge is cut if endpoints have different labels (1 vs 2)
        tet_cut_edges = []
        tet_primary_edges = []
        
        for local_e, (i, j) in enumerate(edge_pairs):
            vi, vj = int(tet[i]), int(tet[j])
            label_i = vertex_mold_labels[vi]
            label_j = vertex_mold_labels[vj]
            
            # Check if this edge is cut (different labels, both labeled)
            if label_i != label_j and label_i in (1, 2) and label_j in (1, 2):
                tet_cut_edges.append(local_e)
                
                # Check if it's a primary cut edge
                key = (min(vi, vj), max(vi, vj))
                if key in primary_edge_set:
                    tet_primary_edges.append((key, local_e))
                    edges_in_any_tet.add(key)
        
        n_cut = len(tet_cut_edges)
        config_distribution[n_cut] = config_distribution.get(n_cut, 0) + 1
        
        # 3-edge and 4-edge configs are VALID (produce triangles)
        # 0, 1, 2, 5, 6-edge configs are INVALID (produce nothing)
        is_valid = n_cut in (3, 4)
        
        if tet_primary_edges:
            tet_configs[t_idx] = [e[1] for e in tet_primary_edges]
            
            if is_valid:
                n_valid_tets += 1
                for key, _ in tet_primary_edges:
                    edges_in_valid_configs.add(key)
            else:
                n_invalid_tets += 1
                for key, _ in tet_primary_edges:
                    edges_in_invalid_configs.add(key)
    
    # Orphan edges are those that appear ONLY in invalid configurations
    # (never in valid 3/4 configurations)
    orphan_edges = edges_in_invalid_configs - edges_in_valid_configs
    
    # Also find edges that are in primary_cut_edges but DON'T appear in any tet
    # (this would indicate an edge list vs tet mesh mismatch)
    edges_not_in_tets = primary_edge_set - edges_in_any_tet
    
    logger.info(f"Orphan edge analysis:")
    logger.info(f"  Primary cut edges total: {len(primary_edge_set)}")
    logger.info(f"  Primary edges found in tets: {len(edges_in_any_tet)}")
    logger.info(f"  Primary edges NOT in any tet: {len(edges_not_in_tets)}")
    logger.info(f"  Edges in valid (3/4-cut) tets: {len(edges_in_valid_configs)}")
    logger.info(f"  Edges in invalid tets only: {len(orphan_edges)}")
    logger.info(f"  Valid tets with primary edges: {n_valid_tets}")
    logger.info(f"  Invalid tets with primary edges: {n_invalid_tets}")
    logger.info(f"  Config distribution: {dict(sorted(config_distribution.items()))}")
    
    # Combine orphan edges with edges not found in any tet
    all_orphans = list(orphan_edges) + list(edges_not_in_tets)
    
    if edges_not_in_tets:
        logger.warning(f"  {len(edges_not_in_tets)} primary edges not found in any tetrahedron!")
    
    return all_orphans, tet_configs


def analyze_orphan_edges_for_secondary(
    tet_result,
    orphan_edges: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Analyze orphan primary cut edges to determine which could contribute to secondary membranes.
    
    Orphan edges are primary cut edges that only appear in tetrahedra with invalid
    configurations (1, 2, 5, or 6 cut edges) and thus don't contribute to the primary surface.
    
    We classify orphans into three categories:
    1. Adjacent to secondary: Share vertices with existing secondary cut edges
    2. Can form valid groups: Multiple orphan edges that together could form valid tet configs
    3. Truly isolated: Single edges that can't form any valid configuration
    
    Args:
        tet_result: TetrahedralMeshResult
        orphan_edges: List of (vi, vj) orphan primary cut edges
        
    Returns:
        Tuple of:
        - adjacent_to_secondary: Edges adjacent to existing secondary edges
        - can_form_groups: Edges that can form valid configs with other orphans
        - truly_isolated: Edges that are completely isolated
    """
    if not orphan_edges:
        return [], [], []
    
    orphan_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in orphan_edges}
    
    # Get vertices on secondary edges (if any)
    secondary_vertices = set()
    if tet_result.secondary_cut_edges is not None:
        for vi, vj in tet_result.secondary_cut_edges:
            secondary_vertices.add(vi)
            secondary_vertices.add(vj)
    
    # Get vertices on orphan edges
    orphan_vertices = set()
    for vi, vj in orphan_edges:
        orphan_vertices.add(vi)
        orphan_vertices.add(vj)
    
    # Build adjacency among orphan edges (which orphans share vertices)
    vertex_to_orphan_edges = {}
    for vi, vj in orphan_edges:
        key = (min(vi, vj), max(vi, vj))
        if vi not in vertex_to_orphan_edges:
            vertex_to_orphan_edges[vi] = []
        if vj not in vertex_to_orphan_edges:
            vertex_to_orphan_edges[vj] = []
        vertex_to_orphan_edges[vi].append(key)
        vertex_to_orphan_edges[vj].append(key)
    
    # Classify each orphan edge
    adjacent_to_secondary = []
    can_form_groups = []
    truly_isolated = []
    
    for vi, vj in orphan_edges:
        key = (min(vi, vj), max(vi, vj))
        
        # Check if adjacent to secondary edges
        is_adjacent_secondary = vi in secondary_vertices or vj in secondary_vertices
        
        # Check if this edge shares vertices with OTHER orphan edges
        # (meaning they could potentially form a valid group together)
        connected_orphans = set()
        for e in vertex_to_orphan_edges.get(vi, []):
            if e != key:
                connected_orphans.add(e)
        for e in vertex_to_orphan_edges.get(vj, []):
            if e != key:
                connected_orphans.add(e)
        
        has_orphan_neighbors = len(connected_orphans) > 0
        
        if is_adjacent_secondary:
            adjacent_to_secondary.append((vi, vj))
        elif has_orphan_neighbors:
            can_form_groups.append((vi, vj))
        else:
            truly_isolated.append((vi, vj))
    
    logger.info(f"Orphan edge classification:")
    logger.info(f"  Total orphan edges: {len(orphan_edges)}")
    logger.info(f"  Adjacent to secondary: {len(adjacent_to_secondary)}")
    logger.info(f"  Can form groups with other orphans: {len(can_form_groups)}")
    logger.info(f"  Truly isolated: {len(truly_isolated)}")
    
    return adjacent_to_secondary, can_form_groups, truly_isolated


def find_orphan_edge_clusters(
    tet_result,
    orphan_edges: List[Tuple[int, int]]
) -> List[Set[Tuple[int, int]]]:
    """
    Find connected clusters of orphan edges.
    
    Orphan edges that share vertices form clusters. Each cluster could potentially
    form its own secondary membrane patch.
    
    Args:
        tet_result: TetrahedralMeshResult
        orphan_edges: List of orphan primary cut edges
        
    Returns:
        List of sets, each set containing connected orphan edges
    """
    if not orphan_edges:
        return []
    
    # Build adjacency graph among orphan edges
    orphan_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in orphan_edges}
    
    # Vertex to edges mapping
    vertex_to_edges = {}
    for vi, vj in orphan_edges:
        key = (min(vi, vj), max(vi, vj))
        if vi not in vertex_to_edges:
            vertex_to_edges[vi] = set()
        if vj not in vertex_to_edges:
            vertex_to_edges[vj] = set()
        vertex_to_edges[vi].add(key)
        vertex_to_edges[vj].add(key)
    
    # Find connected components using BFS
    visited = set()
    clusters = []
    
    for start_edge in orphan_edge_set:
        if start_edge in visited:
            continue
        
        # BFS to find all connected edges
        cluster = set()
        queue = [start_edge]
        
        while queue:
            edge = queue.pop(0)
            if edge in visited:
                continue
            
            visited.add(edge)
            cluster.add(edge)
            
            # Find adjacent edges (share a vertex)
            vi, vj = edge
            for neighbor_edge in vertex_to_edges.get(vi, set()):
                if neighbor_edge not in visited:
                    queue.append(neighbor_edge)
            for neighbor_edge in vertex_to_edges.get(vj, set()):
                if neighbor_edge not in visited:
                    queue.append(neighbor_edge)
        
        if cluster:
            clusters.append(cluster)
    
    logger.info(f"Found {len(clusters)} orphan edge clusters")
    for i, cluster in enumerate(clusters):
        logger.info(f"  Cluster {i}: {len(cluster)} edges")
    
    return clusters


def create_enhanced_secondary_cut_edges(
    tet_result,
    include_orphan_primary: bool = True,
    include_isolated_orphans: bool = True
) -> List[Tuple[int, int]]:
    """
    Create an enhanced secondary cut edge list that includes unused primary edges.
    
    This combines:
    1. Original secondary cut edges (same-label edges with divergent paths)
    2. ALL unused primary edges (primary edges not used in the primary surface)
    
    The unused primary edges fill gaps left by the primary surface where tetrahedra had
    invalid configurations (1, 2, 5, or 6 cut edges instead of the valid 3 or 4).
    
    Args:
        tet_result: TetrahedralMeshResult
        include_orphan_primary: If True, include unused primary edges
        include_isolated_orphans: If True, also include truly isolated orphan edges
                                 (edges with no connection to other cuts)
        
    Returns:
        Enhanced list of secondary cut edges
    """
    # Start with original secondary edges
    secondary_edges = []
    if tet_result.secondary_cut_edges is not None:
        secondary_edges = list(tet_result.secondary_cut_edges)
    
    original_count = len(secondary_edges)
    secondary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in secondary_edges}
    
    if not include_orphan_primary:
        return secondary_edges
    
    # Use the comprehensive function to find ALL unused primary edges
    used_edges, unused_edges, config_counts = find_all_unused_primary_edges(tet_result)
    
    if not unused_edges:
        logger.info("No unused primary edges found - all primary edges contribute to surface")
        return secondary_edges
    
    # Convert to list and remove any that are already in secondary
    unused_list = [e for e in unused_edges if e not in secondary_edge_set]
    
    if not unused_list:
        logger.info("All unused primary edges already in secondary edge list")
        return secondary_edges
    
    # Classify unused edges for logging
    adjacent, groupable, isolated = analyze_orphan_edges_for_secondary(tet_result, unused_list)
    
    # Add ALL unused edges (no filtering)
    # The marching tetrahedra will determine which ones can form valid surfaces
    added_count = 0
    
    if adjacent:
        logger.info(f"Adding {len(adjacent)} unused primary edges adjacent to secondary")
        secondary_edges.extend(adjacent)
        added_count += len(adjacent)
    
    if groupable:
        logger.info(f"Adding {len(groupable)} unused primary edges that can form groups")
        secondary_edges.extend(groupable)
        added_count += len(groupable)
    
    if isolated:
        if include_isolated_orphans:
            logger.info(f"Adding {len(isolated)} isolated unused primary edges")
            secondary_edges.extend(isolated)
            added_count += len(isolated)
        else:
            logger.info(f"Skipping {len(isolated)} isolated unused primary edges")
    
    logger.info(f"Enhanced secondary edges: {original_count} original + "
               f"{added_count} unused primary = {len(secondary_edges)} total")
    
    return secondary_edges
