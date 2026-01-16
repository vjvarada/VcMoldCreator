"""
Label Cleanup Module for Parting Surface Generation

This module contains algorithms for cleaning up vertex labels (H1/H2) after
Dijkstra escape labeling but BEFORE marching tetrahedra surface extraction.

The Problem:
------------
Dijkstra assigns each interior vertex a "mold half" label based on which 
boundary (H1 or H2) it escapes to first. Due to local path variations, this
can create isolated pockets of one label surrounded by the other, leading to
"tunnels" in the extracted surface.

The Solution Pipeline:
---------------------
1. Label Propagation: Ensure every vertex has an H1 or H2 label
2. Majority Vote Smoothing: Flip vertices whose label disagrees with neighbors
3. Connected Component Filter: Remove small isolated label regions
4. Tetrahedral Pocket Detection: Find and eliminate tunnel-forming tet pockets
5. High Cut-Valence Detection: Break potential surface loops at junctions

All cleanup operates BEFORE surface extraction to prevent topological artifacts.
Post-extraction fixes (like Bloomenthal's ε-collapse) don't work for our
topological tunnels - they only fix geometric artifacts like degenerate triangles.

Key Insight:
-----------
Our tunnels are TOPOLOGICAL (wrong labels → wrong surface topology), not
GEOMETRIC (bad triangle shapes). Therefore, we must fix the labels before
extracting the surface, not the surface geometry after extraction.
"""

import logging
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LabelCleanupResult:
    """Result of label cleanup processing."""
    
    # Final vertex labels (1=H1, 2=H2)
    vertex_labels: np.ndarray
    
    # Primary cut edges (edges crossing H1/H2 boundary)
    primary_cut_edges: List[Tuple[int, int]]
    
    # Statistics
    propagation_iterations: int = 0
    propagation_vertices_labeled: int = 0
    
    smoothing_iterations: int = 0
    smoothing_vertices_flipped: int = 0
    
    component_filter_vertices_flipped: int = 0
    component_filter_components_removed: int = 0
    
    pocket_detection_pockets_eliminated: int = 0
    pocket_detection_vertices_flipped: int = 0
    
    cut_valence_vertices_flipped: int = 0
    
    # Label distribution
    n_h1_vertices: int = 0
    n_h2_vertices: int = 0


def cleanup_vertex_labels(
    edges: np.ndarray,
    interior_vertex_indices: np.ndarray,
    interior_escape_labels: np.ndarray,
    boundary_labels: np.ndarray,
    tetrahedra: Optional[np.ndarray] = None,
    smooth_labels: bool = True,
    smooth_iterations: int = 2,
    smooth_threshold: float = 0.7,
    detect_tet_pockets: bool = True,
    detect_high_cut_valence: bool = True,
    max_cut_valence: int = 4
) -> LabelCleanupResult:
    """
    Clean up vertex labels to prevent tunnels in the extracted surface.
    
    This function implements the full label cleanup pipeline:
    1. Label propagation (ensure all vertices have H1/H2 labels)
    2. Majority vote smoothing (flip isolated vertices)
    3. Connected component filtering (remove small label regions)
    4. Tetrahedral pocket detection (eliminate tunnel-forming pockets)
    5. High cut-valence detection (break surface loops at junctions)
    
    Args:
        edges: (E, 2) array of edge vertex indices
        interior_vertex_indices: (I,) indices of interior (non-boundary) vertices
        interior_escape_labels: (I,) escape labels (1=H1, 2=H2, 0=unreachable)
        boundary_labels: (V,) boundary classification (-1=part, 1=H1, 2=H2)
        tetrahedra: (T, 4) tetrahedra vertex indices (required for pocket detection)
        smooth_labels: Enable majority vote smoothing
        smooth_iterations: Number of smoothing iterations
        smooth_threshold: Threshold for label flipping (0.5-0.9)
        detect_tet_pockets: Enable tetrahedral pocket detection
        detect_high_cut_valence: Enable cut-valence based loop breaking
        max_cut_valence: Maximum cut edges per vertex before flagging
    
    Returns:
        LabelCleanupResult with cleaned labels and statistics
    """
    result = LabelCleanupResult(
        vertex_labels=np.array([]),
        primary_cut_edges=[]
    )
    
    n_verts = np.max(edges) + 1 if len(edges) > 0 else 0
    
    # Initialize: 0 means "not yet classified"
    vertex_labels = np.zeros(n_verts, dtype=np.int8)
    
    # Build adjacency list (needed for all cleanup steps)
    adjacency = _build_adjacency_list(edges, n_verts)
    
    # =========================================================================
    # STEP 1: Initialize labels from boundary and escape classifications
    # =========================================================================
    logger.info("Label Cleanup Step 1: Initializing vertex labels...")
    
    # Assign H1/H2 boundary vertices their boundary labels
    if boundary_labels is not None:
        h1_mask = boundary_labels == 1
        h2_mask = boundary_labels == 2
        vertex_labels[h1_mask] = 1
        vertex_labels[h2_mask] = 2
    
    # Assign interior vertices their escape labels
    for i, vert_idx in enumerate(interior_vertex_indices):
        escape_label = interior_escape_labels[i]
        if escape_label in (1, 2):
            vertex_labels[vert_idx] = escape_label
    
    # =========================================================================
    # STEP 2: Propagate labels to unlabeled vertices
    # =========================================================================
    unlabeled_count = np.sum(vertex_labels == 0)
    if unlabeled_count > 0:
        vertex_labels, prop_iters, prop_labeled = _propagate_labels(
            vertex_labels, adjacency
        )
        result.propagation_iterations = prop_iters
        result.propagation_vertices_labeled = prop_labeled
    
    # =========================================================================
    # STEP 3: Majority vote smoothing
    # =========================================================================
    interior_set = set(interior_vertex_indices.tolist())
    
    if smooth_labels:
        vertex_labels, smooth_iters, smooth_flipped = _majority_vote_smoothing(
            vertex_labels, adjacency, interior_set,
            iterations=smooth_iterations, threshold=smooth_threshold
        )
        result.smoothing_iterations = smooth_iters
        result.smoothing_vertices_flipped = smooth_flipped
    
    # =========================================================================
    # STEP 4: Connected component filtering
    # =========================================================================
    if smooth_labels:
        vertex_labels, comp_flipped, comp_removed = _connected_component_filter(
            vertex_labels, adjacency, interior_set
        )
        result.component_filter_vertices_flipped = comp_flipped
        result.component_filter_components_removed = comp_removed
    
    # =========================================================================
    # STEP 5: Tetrahedral pocket detection
    # =========================================================================
    if detect_tet_pockets and tetrahedra is not None and len(tetrahedra) > 0:
        vertex_labels, pockets_elim, pocket_verts = _detect_and_eliminate_tet_pockets(
            vertex_labels, tetrahedra, interior_set
        )
        result.pocket_detection_pockets_eliminated = pockets_elim
        result.pocket_detection_vertices_flipped = pocket_verts
    
    # =========================================================================
    # STEP 6: High cut-valence detection
    # =========================================================================
    if detect_high_cut_valence:
        vertex_labels, valence_flipped = _detect_high_cut_valence(
            vertex_labels, edges, adjacency, interior_set,
            max_cut_valence=max_cut_valence
        )
        result.cut_valence_vertices_flipped = valence_flipped
    
    # =========================================================================
    # STEP 7: Compute final cut edges
    # =========================================================================
    primary_cut_edges = _compute_cut_edges(vertex_labels, edges)
    
    # Finalize result
    result.vertex_labels = vertex_labels
    result.primary_cut_edges = primary_cut_edges
    result.n_h1_vertices = int(np.sum(vertex_labels == 1))
    result.n_h2_vertices = int(np.sum(vertex_labels == 2))
    
    logger.info(f"Label cleanup complete: {result.n_h1_vertices} H1, {result.n_h2_vertices} H2, "
               f"{len(primary_cut_edges)} cut edges")
    
    return result


# =============================================================================
# STEP 1: Adjacency Building
# =============================================================================

def _build_adjacency_list(edges: np.ndarray, n_verts: int) -> Dict[int, List[int]]:
    """Build vertex adjacency list from edge array."""
    adjacency = {i: [] for i in range(n_verts)}
    for v0, v1 in edges:
        adjacency[v0].append(v1)
        adjacency[v1].append(v0)
    return adjacency


# =============================================================================
# STEP 2: Label Propagation
# =============================================================================

def _propagate_labels(
    vertex_labels: np.ndarray,
    adjacency: Dict[int, List[int]],
    max_iterations: int = 100
) -> Tuple[np.ndarray, int, int]:
    """
    Propagate H1/H2 labels to unlabeled vertices.
    
    Uses iterative nearest-neighbor propagation: assign unlabeled vertices
    the label of their nearest labeled neighbor. Repeat until all vertices
    are labeled.
    
    This ensures every vertex has a valid label, preventing invalid marching
    tetrahedra configurations.
    
    Returns:
        (updated_labels, iterations_used, vertices_labeled)
    """
    unlabeled_count_initial = np.sum(vertex_labels == 0)
    if unlabeled_count_initial == 0:
        return vertex_labels, 0, 0
    
    logger.info(f"Propagating labels to {unlabeled_count_initial} unlabeled vertices...")
    
    vertex_labels = vertex_labels.copy()
    
    for iteration in range(max_iterations):
        newly_labeled = 0
        unlabeled_indices = np.where(vertex_labels == 0)[0]
        
        if len(unlabeled_indices) == 0:
            break
        
        for vi in unlabeled_indices:
            h1_neighbors = sum(1 for n in adjacency[vi] if vertex_labels[n] == 1)
            h2_neighbors = sum(1 for n in adjacency[vi] if vertex_labels[n] == 2)
            
            if h1_neighbors > 0 or h2_neighbors > 0:
                vertex_labels[vi] = 1 if h1_neighbors >= h2_neighbors else 2
                newly_labeled += 1
        
        if newly_labeled == 0:
            break
    
    # Fallback: assign remaining unlabeled vertices to H1
    final_unlabeled = np.sum(vertex_labels == 0)
    if final_unlabeled > 0:
        logger.warning(f"{final_unlabeled} vertices remain unlabeled after propagation, assigning to H1")
        vertex_labels[vertex_labels == 0] = 1
    else:
        logger.info(f"Label propagation complete in {iteration + 1} iterations")
    
    total_labeled = unlabeled_count_initial - final_unlabeled + (final_unlabeled if final_unlabeled > 0 else 0)
    return vertex_labels, iteration + 1, int(total_labeled)


# =============================================================================
# STEP 3: Majority Vote Smoothing
# =============================================================================

def _majority_vote_smoothing(
    vertex_labels: np.ndarray,
    adjacency: Dict[int, List[int]],
    interior_set: Set[int],
    iterations: int = 2,
    threshold: float = 0.7
) -> Tuple[np.ndarray, int, int]:
    """
    Apply majority vote smoothing to remove isolated label pockets.
    
    Small isolated regions of H1 surrounded by H2 (or vice versa) create tunnels.
    We flip vertices whose label differs from a supermajority of their neighbors.
    
    IMPORTANT: Only smooth INTERIOR vertices. Boundary vertices must keep their
    original labels to maintain proper surface boundaries.
    
    Args:
        vertex_labels: Current label array
        adjacency: Vertex adjacency list
        interior_set: Set of interior vertex indices
        iterations: Number of smoothing iterations
        threshold: Fraction of neighbors that must agree to flip (0.5-0.9)
    
    Returns:
        (updated_labels, iterations_performed, total_vertices_flipped)
    """
    logger.info(f"Majority vote smoothing: {iterations} iterations, threshold={threshold}")
    
    vertex_labels = vertex_labels.copy()
    total_flipped = 0
    
    for smooth_iter in range(iterations):
        flipped_count = 0
        new_labels = vertex_labels.copy()
        
        for vi in interior_set:
            current_label = vertex_labels[vi]
            if current_label not in (1, 2):
                continue
            
            neighbors = adjacency[vi]
            if len(neighbors) < 3:
                continue  # Not enough neighbors for reliable voting
            
            h1_count = sum(1 for n in neighbors if vertex_labels[n] == 1)
            h2_count = sum(1 for n in neighbors if vertex_labels[n] == 2)
            total_labeled = h1_count + h2_count
            
            if total_labeled == 0:
                continue
            
            # Check if current label is in the minority
            if current_label == 1:
                my_fraction = h1_count / total_labeled
                opposite_label = 2
            else:
                my_fraction = h2_count / total_labeled
                opposite_label = 1
            
            # Flip if supermajority of neighbors have opposite label
            if my_fraction < (1.0 - threshold):
                new_labels[vi] = opposite_label
                flipped_count += 1
        
        vertex_labels = new_labels
        total_flipped += flipped_count
        
        if flipped_count > 0:
            logger.info(f"  Iteration {smooth_iter + 1}: flipped {flipped_count} vertices")
        else:
            logger.info(f"  Converged after {smooth_iter + 1} iterations")
            return vertex_labels, smooth_iter + 1, total_flipped
    
    return vertex_labels, iterations, total_flipped


# =============================================================================
# STEP 4: Connected Component Filtering
# =============================================================================

def _connected_component_filter(
    vertex_labels: np.ndarray,
    adjacency: Dict[int, List[int]],
    interior_set: Set[int],
    min_component_fraction: float = 0.001
) -> Tuple[np.ndarray, int, int]:
    """
    Remove small isolated label regions by connected component analysis.
    
    After majority voting, find connected components of same-labeled interior
    vertices and flip small components to the surrounding label.
    
    Args:
        vertex_labels: Current label array
        adjacency: Vertex adjacency list
        interior_set: Set of interior vertex indices
        min_component_fraction: Minimum component size as fraction of interior
    
    Returns:
        (updated_labels, total_vertices_flipped, components_removed)
    """
    vertex_labels = vertex_labels.copy()
    min_component_size = max(10, int(len(interior_set) * min_component_fraction))
    
    logger.info(f"Connected component filter: min size = {min_component_size}")
    
    total_flipped = 0
    total_components_removed = 0
    
    for target_label in [1, 2]:
        opposite_label = 2 if target_label == 1 else 1
        
        # Find all interior vertices with target_label
        target_verts = set(vi for vi in interior_set if vertex_labels[vi] == target_label)
        
        if len(target_verts) == 0:
            continue
        
        # Find connected components using BFS
        visited = set()
        components = []
        
        for start_v in target_verts:
            if start_v in visited:
                continue
            
            component = []
            queue = [start_v]
            visited.add(start_v)
            
            while queue:
                v = queue.pop(0)
                component.append(v)
                
                for neighbor in adjacency[v]:
                    if neighbor in target_verts and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
        
        # Find and flip small components
        for comp in components:
            if len(comp) < min_component_size:
                for vi in comp:
                    vertex_labels[vi] = opposite_label
                    total_flipped += 1
                total_components_removed += 1
        
        n_small = sum(1 for c in components if len(c) < min_component_size)
        if n_small > 0:
            logger.info(f"  H{target_label}: flipped {n_small} small components")
    
    return vertex_labels, total_flipped, total_components_removed


# =============================================================================
# STEP 5: Tetrahedral Pocket Detection
# =============================================================================

def _detect_and_eliminate_tet_pockets(
    vertex_labels: np.ndarray,
    tetrahedra: np.ndarray,
    interior_set: Set[int],
    min_pocket_fraction: float = 0.002
) -> Tuple[np.ndarray, int, int]:
    """
    Detect and eliminate isolated tetrahedral pockets that form tunnels.
    
    Tunnels form when isolated pockets of same-labeled tetrahedra are surrounded
    by cut (mixed-label) tetrahedra. 
    
    A "non-contributing" tet has all vertices with the same label (no cut edges).
    A "contributing" tet has mixed labels (produces cut edges).
    
    Small non-contributing tet clusters completely surrounded by contributing
    tets are tunnel pockets and should have their labels flipped.
    
    Args:
        vertex_labels: Current label array
        tetrahedra: (T, 4) tetrahedra vertex indices
        interior_set: Set of interior vertex indices
        min_pocket_fraction: Minimum pocket size as fraction of total tets
    
    Returns:
        (updated_labels, pockets_eliminated, vertices_flipped)
    """
    vertex_labels = vertex_labels.copy()
    
    logger.info("Tetrahedral pocket detection...")
    
    # Classify each tet based on current vertex labels
    tet_labels = vertex_labels[tetrahedra]  # (T, 4)
    
    # A tet is "contributing" if it has mixed labels
    tet_has_h1 = np.any(tet_labels == 1, axis=1)
    tet_has_h2 = np.any(tet_labels == 2, axis=1)
    contributing_mask = tet_has_h1 & tet_has_h2
    
    non_contributing_indices = np.where(~contributing_mask)[0]
    n_contributing = int(np.sum(contributing_mask))
    n_non_contributing = len(non_contributing_indices)
    
    logger.info(f"  {n_contributing} contributing tets, {n_non_contributing} non-contributing tets")
    
    if n_non_contributing == 0 or n_non_contributing >= len(tetrahedra) * 0.5:
        logger.info("  Skipping pocket detection (too few/many non-contributing tets)")
        return vertex_labels, 0, 0
    
    # Build tet-to-tet adjacency (tets sharing a face)
    face_to_tets = {}
    for ti, tet in enumerate(tetrahedra):
        for face_indices in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            face = frozenset([tet[i] for i in face_indices])
            if face not in face_to_tets:
                face_to_tets[face] = []
            face_to_tets[face].append(ti)
    
    tet_adjacency = {ti: set() for ti in range(len(tetrahedra))}
    for face, tet_list in face_to_tets.items():
        for i, ti in enumerate(tet_list):
            for tj in tet_list[i + 1:]:
                tet_adjacency[ti].add(tj)
                tet_adjacency[tj].add(ti)
    
    # Find connected components of non-contributing tets
    non_contributing_set = set(non_contributing_indices)
    visited_tets = set()
    tet_components = []
    
    for start_ti in non_contributing_indices:
        if start_ti in visited_tets:
            continue
        
        component = []
        queue = [start_ti]
        visited_tets.add(start_ti)
        
        while queue:
            ti = queue.pop(0)
            component.append(ti)
            
            for neighbor_ti in tet_adjacency[ti]:
                if neighbor_ti in non_contributing_set and neighbor_ti not in visited_tets:
                    visited_tets.add(neighbor_ti)
                    queue.append(neighbor_ti)
        
        tet_components.append(component)
    
    logger.info(f"  Found {len(tet_components)} non-contributing tet components")
    
    # Check each component for tunnel pockets
    min_tet_component_size = max(5, int(len(tetrahedra) * min_pocket_fraction))
    pockets_eliminated = 0
    verts_flipped = 0
    
    for comp in tet_components:
        if len(comp) >= min_tet_component_size:
            continue
        
        # Check if completely surrounded by contributing tets
        comp_set = set(comp)
        surrounding_tets = set()
        for ti in comp:
            for neighbor_ti in tet_adjacency[ti]:
                if neighbor_ti not in comp_set:
                    surrounding_tets.add(neighbor_ti)
        
        if len(surrounding_tets) == 0:
            continue
        
        if not all(contributing_mask[ti] for ti in surrounding_tets):
            continue
        
        # This is a tunnel pocket - flip interior vertices
        pocket_verts = set()
        for ti in comp:
            pocket_verts.update(tetrahedra[ti])
        
        interior_pocket_verts = [v for v in pocket_verts if v in interior_set]
        
        if len(interior_pocket_verts) == 0:
            continue
        
        pocket_label = vertex_labels[interior_pocket_verts[0]]
        opposite_label = 2 if pocket_label == 1 else 1
        
        for vi in interior_pocket_verts:
            if vertex_labels[vi] == pocket_label:
                vertex_labels[vi] = opposite_label
                verts_flipped += 1
        
        pockets_eliminated += 1
    
    if pockets_eliminated > 0:
        logger.info(f"  Eliminated {pockets_eliminated} tunnel pockets ({verts_flipped} vertices flipped)")
    
    return vertex_labels, pockets_eliminated, verts_flipped


# =============================================================================
# STEP 6: High Cut-Valence Detection
# =============================================================================

def _detect_high_cut_valence(
    vertex_labels: np.ndarray,
    edges: np.ndarray,
    adjacency: Dict[int, List[int]],
    interior_set: Set[int],
    max_cut_valence: int = 4
) -> Tuple[np.ndarray, int]:
    """
    Detect and fix vertices with high cut-valence to break tunnel loops.
    
    Cut-valence = number of incident cut edges (edges crossing H1/H2 boundary).
    Vertices with high cut-valence are "junction" vertices where multiple
    surface sheets meet, often forming tunnels.
    
    We flip these vertices to the local majority label to simplify the topology.
    
    Args:
        vertex_labels: Current label array
        edges: (E, 2) edge array
        adjacency: Vertex adjacency list
        interior_set: Set of interior vertex indices
        max_cut_valence: Threshold for flagging (default 4)
    
    Returns:
        (updated_labels, vertices_flipped)
    """
    vertex_labels = vertex_labels.copy()
    n_verts = len(vertex_labels)
    
    # Compute temporary cut edges
    edge_labels_v0 = vertex_labels[edges[:, 0]]
    edge_labels_v1 = vertex_labels[edges[:, 1]]
    cut_mask = ((edge_labels_v0 == 1) & (edge_labels_v1 == 2)) | \
               ((edge_labels_v0 == 2) & (edge_labels_v1 == 1))
    temp_cut_edges = edges[cut_mask]
    
    # Compute cut valence for each vertex
    cut_valence = np.zeros(n_verts, dtype=np.int32)
    for v0, v1 in temp_cut_edges:
        cut_valence[v0] += 1
        cut_valence[v1] += 1
    
    # Find high-valence interior vertices
    high_valence_verts = np.where(cut_valence > max_cut_valence)[0]
    high_valence_interior = [v for v in high_valence_verts if v in interior_set]
    
    if len(high_valence_interior) == 0:
        return vertex_labels, 0
    
    logger.info(f"High cut-valence detection: {len(high_valence_interior)} vertices with valence > {max_cut_valence}")
    
    flipped = 0
    for vi in high_valence_interior:
        current_label = vertex_labels[vi]
        neighbors = adjacency[vi]
        
        h1_count = sum(1 for n in neighbors if vertex_labels[n] == 1)
        h2_count = sum(1 for n in neighbors if vertex_labels[n] == 2)
        
        if h1_count + h2_count == 0:
            continue
        
        if h1_count > h2_count and current_label != 1:
            vertex_labels[vi] = 1
            flipped += 1
        elif h2_count > h1_count and current_label != 2:
            vertex_labels[vi] = 2
            flipped += 1
    
    if flipped > 0:
        logger.info(f"  Flipped {flipped} high-valence vertices")
    
    return vertex_labels, flipped


# =============================================================================
# STEP 7: Cut Edge Computation
# =============================================================================

def _compute_cut_edges(
    vertex_labels: np.ndarray,
    edges: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Compute primary cut edges from final vertex labels.
    
    A cut edge connects an H1 vertex to an H2 vertex.
    
    Args:
        vertex_labels: Final label array (1=H1, 2=H2)
        edges: (E, 2) edge array
    
    Returns:
        List of (v0, v1) tuples for cut edges
    """
    edge_labels_v0 = vertex_labels[edges[:, 0]]
    edge_labels_v1 = vertex_labels[edges[:, 1]]
    
    cut_mask = ((edge_labels_v0 == 1) & (edge_labels_v1 == 2)) | \
               ((edge_labels_v0 == 2) & (edge_labels_v1 == 1))
    
    cut_edge_indices = np.where(cut_mask)[0]
    return [(int(edges[i, 0]), int(edges[i, 1])) for i in cut_edge_indices]

# =============================================================================
# SECONDARY SURFACE CLEANUP
# =============================================================================
# Secondary surfaces use direct edge lists (not vertex labels), but still
# need cleanup to remove isolated edge clusters and improve connectivity.
# =============================================================================

@dataclass
class SecondaryEdgeCleanupResult:
    """Result of secondary edge cleanup processing."""
    
    # Final secondary cut edges
    secondary_cut_edges: List[Tuple[int, int]]
    
    # Statistics
    original_edge_count: int = 0
    final_edge_count: int = 0
    isolated_clusters_removed: int = 0
    edges_removed_isolation: int = 0
    dangling_edges_removed: int = 0
    
    # Diagnostics
    cluster_sizes: List[int] = None


def cleanup_secondary_edges(
    secondary_cut_edges: List[Tuple[int, int]],
    all_edges: np.ndarray,
    primary_cut_edges: Optional[List[Tuple[int, int]]] = None,
    min_cluster_edges: int = 5,
    remove_dangling: bool = True,
    require_primary_connection: bool = False
) -> SecondaryEdgeCleanupResult:
    """
    Clean up secondary cut edges to remove isolated clusters and dangling edges.
    
    Secondary cut edges are detected by membrane-part intersection, which can
    produce isolated edge clusters that form disconnected surface patches.
    This cleanup removes small isolated clusters and dangling edges.
    
    Args:
        secondary_cut_edges: List of (vi, vj) secondary cut edges
        all_edges: (E, 2) array of all tetrahedral edges
        primary_cut_edges: List of primary cut edges (for connectivity check)
        min_cluster_edges: Minimum edges in cluster to keep
        remove_dangling: Remove edges that only connect to one other secondary edge
        require_primary_connection: If True, only keep clusters connected to primary
    
    Returns:
        SecondaryEdgeCleanupResult with cleaned edges and statistics
    """
    result = SecondaryEdgeCleanupResult(
        secondary_cut_edges=[],
        original_edge_count=len(secondary_cut_edges)
    )
    
    if len(secondary_cut_edges) == 0:
        return result
    
    logger.info(f"Secondary edge cleanup: {len(secondary_cut_edges)} edges to process")
    
    # Build edge adjacency graph (edges share vertices)
    edge_set = set(tuple(sorted(e)) for e in secondary_cut_edges)
    
    # Build vertex-to-edge mapping
    vertex_to_edges: Dict[int, List[Tuple[int, int]]] = {}
    for edge in edge_set:
        for v in edge:
            if v not in vertex_to_edges:
                vertex_to_edges[v] = []
            vertex_to_edges[v].append(edge)
    
    # Find connected components of secondary edges (BFS via shared vertices)
    visited_edges = set()
    clusters = []
    
    for start_edge in edge_set:
        if start_edge in visited_edges:
            continue
        
        # BFS to find all edges in this cluster
        cluster = []
        queue = [start_edge]
        
        while queue:
            edge = queue.pop(0)
            if edge in visited_edges:
                continue
            
            visited_edges.add(edge)
            cluster.append(edge)
            
            # Find neighboring edges (share a vertex)
            for v in edge:
                for neighbor_edge in vertex_to_edges.get(v, []):
                    if neighbor_edge not in visited_edges:
                        queue.append(neighbor_edge)
        
        clusters.append(cluster)
    
    result.cluster_sizes = [len(c) for c in clusters]
    logger.info(f"Secondary edge cleanup: found {len(clusters)} clusters, sizes: {sorted(result.cluster_sizes, reverse=True)[:10]}")
    
    # Filter clusters by size
    kept_edges = []
    removed_isolation = 0
    removed_clusters = 0
    
    for cluster in clusters:
        if len(cluster) >= min_cluster_edges:
            kept_edges.extend(cluster)
        else:
            removed_isolation += len(cluster)
            removed_clusters += 1
    
    result.isolated_clusters_removed = removed_clusters
    result.edges_removed_isolation = removed_isolation
    
    logger.info(f"Secondary edge cleanup: kept {len(kept_edges)} edges, removed {removed_isolation} from {removed_clusters} small clusters")
    
    # Optional: Remove dangling edges (only one connection to other secondary edges)
    if remove_dangling and len(kept_edges) > 0:
        # Rebuild vertex-to-edge mapping for kept edges
        kept_set = set(kept_edges)
        vertex_to_kept = {}
        for edge in kept_set:
            for v in edge:
                if v not in vertex_to_kept:
                    vertex_to_kept[v] = []
                vertex_to_kept[v].append(edge)
        
        # Find dangling edges (vertex only appears in one edge)
        dangling = []
        for edge in kept_set:
            is_dangling = False
            for v in edge:
                if len(vertex_to_kept.get(v, [])) == 1:
                    # This vertex only appears in this edge
                    is_dangling = True
                    break
            if is_dangling:
                dangling.append(edge)
        
        # Remove dangling edges iteratively
        iterations = 0
        max_iterations = 10
        while dangling and iterations < max_iterations:
            for edge in dangling:
                if edge in kept_set:
                    kept_set.remove(edge)
                    result.dangling_edges_removed += 1
                    # Update vertex mappings
                    for v in edge:
                        if v in vertex_to_kept:
                            vertex_to_kept[v] = [e for e in vertex_to_kept[v] if e != edge]
            
            # Find new dangling edges
            dangling = []
            for edge in kept_set:
                is_dangling = False
                for v in edge:
                    if len(vertex_to_kept.get(v, [])) == 1:
                        is_dangling = True
                        break
                if is_dangling:
                    dangling.append(edge)
            
            iterations += 1
        
        kept_edges = list(kept_set)
        
        if result.dangling_edges_removed > 0:
            logger.info(f"Secondary edge cleanup: removed {result.dangling_edges_removed} dangling edges")
    
    # Optional: Filter to clusters connected to primary surface
    if require_primary_connection and primary_cut_edges is not None and len(primary_cut_edges) > 0:
        # Get vertices on primary cut edges
        primary_vertices = set()
        for v0, v1 in primary_cut_edges:
            primary_vertices.add(v0)
            primary_vertices.add(v1)
        
        # Keep only secondary edges that share a vertex with primary
        connected_edges = []
        for edge in kept_edges:
            if edge[0] in primary_vertices or edge[1] in primary_vertices:
                connected_edges.append(edge)
        
        removed_disconnected = len(kept_edges) - len(connected_edges)
        if removed_disconnected > 0:
            logger.info(f"Secondary edge cleanup: removed {removed_disconnected} edges not connected to primary")
            kept_edges = connected_edges
    
    result.secondary_cut_edges = [(int(e[0]), int(e[1])) for e in kept_edges]
    result.final_edge_count = len(result.secondary_cut_edges)
    
    logger.info(f"Secondary edge cleanup complete: {result.original_edge_count} → {result.final_edge_count} edges")
    
    return result


def compute_secondary_vertex_labels(
    secondary_cut_edges: List[Tuple[int, int]],
    n_vertices: int,
    primary_vertex_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute vertex labels for secondary surface based on secondary cut edges.
    
    Unlike primary surfaces where labels define the cut, secondary surfaces
    cut through SAME-label regions. We assign labels based on connectivity
    to help with surface extraction.
    
    Args:
        secondary_cut_edges: List of (vi, vj) secondary cut edges
        n_vertices: Total number of vertices
        primary_vertex_labels: Primary H1/H2 labels (if available)
    
    Returns:
        Array of labels (1=side A, 2=side B, 0=not near secondary)
    """
    if len(secondary_cut_edges) == 0:
        return np.zeros(n_vertices, dtype=np.int8)
    
    # Get vertices on secondary cuts
    secondary_vertices = set()
    for v0, v1 in secondary_cut_edges:
        secondary_vertices.add(v0)
        secondary_vertices.add(v1)
    
    # Initialize labels
    labels = np.zeros(n_vertices, dtype=np.int8)
    
    # If we have primary labels, use them as a starting point
    if primary_vertex_labels is not None:
        # Secondary cuts separate same-label regions
        # We can use the primary labels directly for the secondary surface
        for v in secondary_vertices:
            if 0 <= v < len(primary_vertex_labels):
                labels[v] = primary_vertex_labels[v]
    else:
        # Without primary labels, just mark all secondary vertices as label 1
        for v in secondary_vertices:
            labels[v] = 1
    
    return labels