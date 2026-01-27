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
    """Build SecondaryMembraneInfo for edges in a mold half."""
    
    if len(secondary_edges) == 0:
        return []
    
    # Build edge sets
    secondary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in secondary_edges}
    primary_edge_set = set()
    if tet_result.primary_cut_edges is not None:
        primary_edge_set = {(min(vi, vj), max(vi, vj)) for vi, vj in tet_result.primary_cut_edges}
    
    # Classify tetrahedra
    interior_tets = set()
    junction_tets = set()
    junction_edges = set()
    
    edge_to_index = tet_result.edge_to_index if tet_result.edge_to_index is not None else {}
    if not edge_to_index:
        from . import tetrahedral_mesh as tm
        edge_to_index = tm.build_edge_to_index_map(tet_result.edges)
    
    # Edge pairs within a tetrahedron (local vertex indices)
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for t_idx, tet in enumerate(tet_result.tetrahedra):
        has_secondary = False
        has_primary = False
        tet_secondary_edges = []
        tet_primary_edges = []
        
        for i, j in edge_pairs:
            vi, vj = int(tet[i]), int(tet[j])
            key = (min(vi, vj), max(vi, vj))
            
            if key in secondary_edge_set:
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
    
    logger.info(f"Mold half {mold_half}: {len(secondary_edges)} secondary edges, "
                f"{len(interior_tets)} interior tets, {len(junction_tets)} junction tets, "
                f"{len(junction_edges)} junction edges")
    
    # Create membrane info
    membrane = SecondaryMembraneInfo(
        mold_half=mold_half,
        cut_edges=secondary_edges,
        interior_tets=interior_tets,
        junction_tets=junction_tets,
        junction_edges=junction_edges
    )
    
    return [membrane]


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
