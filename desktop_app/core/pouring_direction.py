"""
Pouring Direction Optimization Module

Determines optimal silicone and resin pouring directions by minimizing
air bubble trapping using persistence homology analysis.

IMPORTANT: This module operates on SILICONE MOLD PIECES (P1, P2), NOT on
the input part mesh. The workflow is:
1. Generate silicone mold pieces P1 and P2 (via parting surface creation)
2. For each piece, evaluate candidate pouring directions
3. Select nearly-aligned low-scoring directions f1 (for P1) and f2 (for P2)
4. Select resin direction in a 10° cone around the bisector of f1, f2

Air bubbles get trapped at local maxima relative to the pouring direction.
This module uses persistence pairing (maximum-saddle pairs) to identify
bubble-trapping features and scores candidate directions based on the
total area of trapped regions.

Key concepts:
- Height function f: M -> R projects vertices onto pouring direction
- Persistence pairs (max, saddle) mark bubble-trapping features
- Relevance score = area of trapped region, accounting for mold tilting
- Good directions minimize total relevance score

Based on:
- Edelsbrunner et al. 2000: Topological persistence and simplification
- Milnor 1963: Morse theory

Author: VcMoldCreator
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import logging
import time

import trimesh

from .mesh_analysis import compute_height_field, build_vertex_neighbors, build_face_neighbors
from .parting_direction import fibonacci_sphere

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CriticalPoint:
    """Represents a critical point in the height function."""
    index: int           # Vertex index
    height: float        # Height value f(v)
    point_type: str      # 'maximum', 'minimum', or 'saddle'


@dataclass
class PersistencePair:
    """
    A (maximum, saddle) pair from persistence homology.
    
    Represents a bubble-trapping feature where air would accumulate
    at the maximum and could only escape through the saddle point.
    """
    maximum_idx: int              # Vertex index of maximum (birth)
    saddle_idx: int               # Vertex index of saddle (death)
    maximum_height: float         # f(maximum)
    saddle_height: float          # f(saddle)
    persistence: float            # |f(max) - f(saddle)|
    relevance_score: float = 0.0  # Area of trapped air region
    trapped_faces: Set[int] = field(default_factory=set)  # Face indices in A_m^n - T


@dataclass
class PouringDirectionResult:
    """Result of pouring direction analysis for one direction."""
    direction: np.ndarray         # Unit direction vector
    pairs: List[PersistencePair]  # All persistence pairs found
    total_score: float            # Sum of relevance scores
    relevant_pair_count: int      # Pairs above threshold


@dataclass
class OptimalPouringDirections:
    """Complete result of pouring direction optimization."""
    silicone_direction_1: np.ndarray
    silicone_direction_2: np.ndarray
    silicone_score_1: float
    silicone_score_2: float
    resin_direction: np.ndarray
    resin_score: float
    all_scored_directions: List[PouringDirectionResult]
    computation_time_ms: float = 0.0


# ============================================================================
# UNION-FIND DATA STRUCTURE
# ============================================================================

class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for component tracking.
    
    Used during persistence pairing to efficiently track which connected
    component each vertex belongs to as we sweep through superlevel sets.
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> int:
        """
        Union two components. Returns the new root.
        Uses union by rank for efficiency.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return root_x
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        return root_x


# ============================================================================
# CRITICAL POINT DETECTION
# ============================================================================

def count_connected_components(
    vertices: List[int],
    neighbors: Dict[int, Set[int]]
) -> int:
    """
    Count connected components among a subset of vertices.
    
    Used for saddle detection: a vertex is a saddle if its lower neighbors
    form multiple connected components.
    
    Args:
        vertices: List of vertex indices to check
        neighbors: Full vertex adjacency map
        
    Returns:
        Number of connected components
    """
    if not vertices:
        return 0
    
    vertex_set = set(vertices)
    visited: Set[int] = set()
    components = 0
    
    for start in vertices:
        if start in visited:
            continue
        
        # BFS from start
        components += 1
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            for neighbor in neighbors.get(current, set()):
                if neighbor in vertex_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return components


def classify_vertex(
    vertex_idx: int,
    vertex_heights: np.ndarray,
    neighbors: Dict[int, Set[int]]
) -> Optional[str]:
    """
    Classify a vertex as a critical point based on height comparison with neighbors.
    
    Uses lower/upper link analysis:
    - Maximum: all neighbors have strictly lower height
    - Minimum: all neighbors have strictly higher height
    - Saddle: lower neighbors form multiple disconnected components
    - Regular: none of the above
    
    Args:
        vertex_idx: Index of vertex to classify
        vertex_heights: Height values for all vertices
        neighbors: Vertex adjacency map
        
    Returns:
        'maximum', 'minimum', 'saddle', or None (regular point)
    """
    h = vertex_heights[vertex_idx]
    neighbor_indices = neighbors.get(vertex_idx, set())
    
    if not neighbor_indices:
        return None
    
    lower_neighbors = [n for n in neighbor_indices if vertex_heights[n] < h]
    upper_neighbors = [n for n in neighbor_indices if vertex_heights[n] > h]
    equal_neighbors = [n for n in neighbor_indices if vertex_heights[n] == h]
    
    # Handle degeneracy: if all neighbors have equal height, not a critical point
    if len(equal_neighbors) == len(neighbor_indices):
        return None
    
    # Maximum: all neighbors are strictly lower
    if len(lower_neighbors) == len(neighbor_indices):
        return 'maximum'
    
    # Minimum: all neighbors are strictly higher
    if len(upper_neighbors) == len(neighbor_indices):
        return 'minimum'
    
    # Saddle detection: check if lower link has multiple components
    # A saddle separates the superlevel set into multiple pieces
    if lower_neighbors:
        components = count_connected_components(lower_neighbors, neighbors)
        if components > 1:
            return 'saddle'
    
    return None  # Regular point


def find_critical_points(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    vertex_neighbors: Optional[Dict[int, Set[int]]] = None
) -> Tuple[List[CriticalPoint], np.ndarray]:
    """
    Find all critical points of the height function f(v) = dot(v, direction).
    
    Args:
        mesh: Input mesh
        direction: Pouring direction vector
        vertex_neighbors: Precomputed vertex adjacency (optional)
        
    Returns:
        critical_points: List of CriticalPoint objects
        vertex_heights: Height values for all vertices
    """
    vertex_heights, _ = compute_height_field(mesh, direction)
    
    if vertex_neighbors is None:
        vertex_neighbors = build_vertex_neighbors(mesh)
    
    critical_points = []
    
    for v_idx in range(len(mesh.vertices)):
        point_type = classify_vertex(v_idx, vertex_heights, vertex_neighbors)
        if point_type is not None:
            critical_points.append(CriticalPoint(
                index=v_idx,
                height=vertex_heights[v_idx],
                point_type=point_type
            ))
    
    return critical_points, vertex_heights


# ============================================================================
# PERSISTENCE PAIRING
# ============================================================================

def compute_persistence_pairs(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    vertex_neighbors: Optional[Dict[int, Set[int]]] = None
) -> Tuple[List[PersistencePair], np.ndarray]:
    """
    Compute persistence pairs (maximum, saddle) using superlevel set filtration.
    
    Algorithm (sweep from high to low):
    1. Sort vertices by decreasing height
    2. Process vertices from highest to lowest
    3. When processing vertex v:
       - If v has no processed neighbors: v is a maximum (birth of component)
       - If v connects multiple components: v is a saddle (death of younger components)
       - Otherwise: v joins existing component
    
    This identifies (maximum, saddle) pairs that represent potential
    bubble-trapping features in the mold.
    
    Args:
        mesh: Input mesh
        direction: Pouring direction vector
        vertex_neighbors: Precomputed vertex adjacency (optional)
        
    Returns:
        pairs: List of (maximum, saddle) PersistencePair objects
        vertex_heights: Height values for all vertices
    """
    vertex_heights, _ = compute_height_field(mesh, direction)
    
    if vertex_neighbors is None:
        vertex_neighbors = build_vertex_neighbors(mesh)
    
    n_vertices = len(mesh.vertices)
    
    # Sort vertices by decreasing height (sweep from top to bottom)
    sorted_indices = np.argsort(-vertex_heights)
    
    # Union-Find to track components
    uf = UnionFind(n_vertices)
    
    # Track which vertices have been processed
    processed = np.zeros(n_vertices, dtype=bool)
    
    # Track birth vertex (maximum) for each component root
    component_birth: Dict[int, int] = {}
    
    pairs: List[PersistencePair] = []
    
    for v in sorted_indices:
        h_v = vertex_heights[v]
        
        # Find processed neighbors
        processed_neighbors = [
            n for n in vertex_neighbors.get(v, set())
            if processed[n]
        ]
        
        if not processed_neighbors:
            # v is a maximum - birth of new component
            processed[v] = True
            component_birth[v] = v  # v is its own root initially
        else:
            # Find unique component roots among neighbors
            neighbor_roots = set(uf.find(n) for n in processed_neighbors)
            
            if len(neighbor_roots) == 1:
                # v joins existing component (not a critical point)
                root = neighbor_roots.pop()
                uf.union(v, root)
                processed[v] = True
                
                # Update component_birth to point to new root if changed
                new_root = uf.find(v)
                if new_root != root and root in component_birth:
                    component_birth[new_root] = component_birth.pop(root)
            else:
                # v is a saddle - merges multiple components
                # Sort roots by birth height (descending) - oldest (highest) survives
                roots_by_birth = sorted(
                    neighbor_roots,
                    key=lambda r: vertex_heights[component_birth.get(r, r)],
                    reverse=True
                )
                
                # Oldest (highest) component survives
                survivor_root = roots_by_birth[0]
                
                # Other components die - create persistence pairs
                for dying_root in roots_by_birth[1:]:
                    max_vertex = component_birth.get(dying_root, dying_root)
                    pairs.append(PersistencePair(
                        maximum_idx=max_vertex,
                        saddle_idx=v,
                        maximum_height=vertex_heights[max_vertex],
                        saddle_height=h_v,
                        persistence=vertex_heights[max_vertex] - h_v
                    ))
                
                # Merge all into survivor
                for root in neighbor_roots:
                    if root != survivor_root:
                        uf.union(survivor_root, root)
                        if root in component_birth:
                            del component_birth[root]
                
                # v joins the merged component
                uf.union(v, survivor_root)
                processed[v] = True
                
                # Update survivor's entry to new root
                new_root = uf.find(survivor_root)
                if new_root != survivor_root and survivor_root in component_birth:
                    component_birth[new_root] = component_birth.pop(survivor_root)
    
    return pairs, vertex_heights


# ============================================================================
# BUBBLE RELEVANCE SCORING
# ============================================================================

def grow_trapped_region(
    mesh: trimesh.Trimesh,
    maximum_idx: int,
    saddle_height: float,
    vertex_heights: np.ndarray,
    face_neighbors: Optional[Dict[int, List[int]]] = None
) -> Set[int]:
    """
    Grow region A_m^n from maximum m down to saddle height n.
    
    This is the region where air would be trapped: all faces connected
    to the maximum that are above the saddle height.
    
    Args:
        mesh: Input mesh
        maximum_idx: Vertex index of the maximum
        saddle_height: Height of the saddle point
        vertex_heights: Height values for all vertices
        face_neighbors: Precomputed face adjacency (optional)
        
    Returns:
        Set of face indices in the trapped region
    """
    if face_neighbors is None:
        face_neighbors = build_face_neighbors(mesh)
    
    # Compute face heights as average of vertex heights
    face_vertex_heights = vertex_heights[mesh.faces]
    face_heights = face_vertex_heights.mean(axis=1)
    
    # Find seed face: a face containing the maximum vertex, with highest centroid
    max_vertex_faces = np.where(np.any(mesh.faces == maximum_idx, axis=1))[0]
    if len(max_vertex_faces) == 0:
        return set()
    
    seed_face = max_vertex_faces[np.argmax(face_heights[max_vertex_faces])]
    
    # BFS to grow region above saddle height
    trapped_faces: Set[int] = set()
    queue = [seed_face]
    visited = {seed_face}
    
    while queue:
        face_idx = queue.pop(0)
        
        # Include face if above saddle height
        if face_heights[face_idx] >= saddle_height:
            trapped_faces.add(face_idx)
            
            # Add unvisited neighbors
            for neighbor in face_neighbors.get(face_idx, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return trapped_faces


def compute_tiltable_region(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    trapped_faces: Set[int],
    saddle_idx: int,
    tilt_angle_deg: float = 10.0,
    face_neighbors: Optional[Dict[int, List[int]]] = None
) -> Set[int]:
    """
    Compute region T of faces that can drain with tilting.
    
    If the mold is tilted by angle alpha during casting, some trapped air
    can escape through faces whose normal allows air to flow toward the saddle.
    
    Grows from saddle, including faces whose normal is within tilt_angle
    of the pouring direction (meaning they're upward-facing enough to drain).
    
    Args:
        mesh: Input mesh
        direction: Pouring direction (unit vector)
        trapped_faces: Set of face indices in A_m^n
        saddle_idx: Vertex index of saddle point
        tilt_angle_deg: Maximum tilt angle in degrees
        face_neighbors: Precomputed face adjacency (optional)
        
    Returns:
        Set of face indices that can drain (T)
    """
    if not trapped_faces:
        return set()
    
    if face_neighbors is None:
        face_neighbors = build_face_neighbors(mesh)
    
    # Threshold: cos(90 - tilt) = sin(tilt) for faces that can drain
    tilt_threshold = np.sin(np.radians(tilt_angle_deg))
    normals = mesh.face_normals
    direction = direction / np.linalg.norm(direction)
    
    # Find seed faces containing saddle vertex
    saddle_faces = set(np.where(np.any(mesh.faces == saddle_idx, axis=1))[0])
    seed_faces = saddle_faces & trapped_faces
    
    if not seed_faces:
        return set()
    
    # BFS from saddle faces, only through drainable faces
    tiltable: Set[int] = set()
    queue = list(seed_faces)
    visited = set(seed_faces)
    
    while queue:
        face_idx = queue.pop(0)
        
        # Check if face is drainable: normal has positive component along direction
        # (air can escape upward if we tilt)
        normal_dot = np.dot(normals[face_idx], direction)
        
        # Face can drain if its normal points somewhat upward (within tilt capability)
        if normal_dot > -tilt_threshold:
            tiltable.add(face_idx)
            
            # Propagate to neighbors in trapped region
            for neighbor in face_neighbors.get(face_idx, []):
                if neighbor in trapped_faces and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return tiltable


def compute_pair_relevance(
    mesh: trimesh.Trimesh,
    pair: PersistencePair,
    direction: np.ndarray,
    vertex_heights: np.ndarray,
    tilt_angle_deg: float = 10.0,
    face_neighbors: Optional[Dict[int, List[int]]] = None
) -> PersistencePair:
    """
    Compute relevance score for a persistence pair.
    
    Score = area of (A_m^n - T), the trapped region minus the tiltable region.
    This represents the amount of air that would actually remain trapped.
    
    Args:
        mesh: Input mesh
        pair: PersistencePair to score
        direction: Pouring direction
        vertex_heights: Height values for all vertices
        tilt_angle_deg: Assumed tilt capability during casting
        face_neighbors: Precomputed face adjacency (optional)
        
    Returns:
        Updated PersistencePair with relevance_score and trapped_faces
    """
    if face_neighbors is None:
        face_neighbors = build_face_neighbors(mesh)
    
    # Grow trapped region from maximum to saddle
    trapped = grow_trapped_region(
        mesh, pair.maximum_idx, pair.saddle_height,
        vertex_heights, face_neighbors
    )
    
    # Compute region that can drain with tilting
    tiltable = compute_tiltable_region(
        mesh, direction, trapped, pair.saddle_idx,
        tilt_angle_deg, face_neighbors
    )
    
    # Final trapped region
    final_trapped = trapped - tiltable
    
    # Compute area score
    face_areas = mesh.area_faces
    score = sum(face_areas[f] for f in final_trapped)
    
    pair.trapped_faces = final_trapped
    pair.relevance_score = score
    
    return pair


# ============================================================================
# DIRECTION SCORING
# ============================================================================

def score_pouring_direction(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5,
    vertex_neighbors: Optional[Dict[int, Set[int]]] = None,
    face_neighbors: Optional[Dict[int, List[int]]] = None
) -> PouringDirectionResult:
    """
    Compute bubble-trapping score for a candidate pouring direction.
    
    Lower scores indicate better directions with less trapped air.
    
    Args:
        mesh: Input mesh
        direction: Candidate pouring direction (unit vector)
        tilt_angle_deg: Assumed tilt capability during casting
        area_threshold_mm2: Filter out pairs below this trapped area
        vertex_neighbors: Precomputed vertex adjacency (optional)
        face_neighbors: Precomputed face adjacency (optional)
        
    Returns:
        PouringDirectionResult with pairs and total score
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    
    # Precompute adjacency if not provided
    if vertex_neighbors is None:
        vertex_neighbors = build_vertex_neighbors(mesh)
    if face_neighbors is None:
        face_neighbors = build_face_neighbors(mesh)
    
    # Get persistence pairs
    pairs, vertex_heights = compute_persistence_pairs(
        mesh, direction, vertex_neighbors
    )
    
    # Compute relevance for each pair
    relevant_pairs: List[PersistencePair] = []
    total_score = 0.0
    
    for pair in pairs:
        compute_pair_relevance(
            mesh, pair, direction, vertex_heights,
            tilt_angle_deg, face_neighbors
        )
        
        if pair.relevance_score >= area_threshold_mm2:
            relevant_pairs.append(pair)
            total_score += pair.relevance_score
    
    return PouringDirectionResult(
        direction=direction.copy(),
        pairs=relevant_pairs,
        total_score=total_score,
        relevant_pair_count=len(relevant_pairs)
    )


def evaluate_candidate_directions(
    mesh: trimesh.Trimesh,
    n_directions: int = 64,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5,
    directions: Optional[np.ndarray] = None,
    progress_callback: Optional[callable] = None
) -> List[PouringDirectionResult]:
    """
    Evaluate multiple candidate directions and return sorted by score.
    
    Args:
        mesh: Input mesh
        n_directions: Number of directions to sample (if directions not provided)
        tilt_angle_deg: Assumed tilt capability
        area_threshold_mm2: Filter threshold for trapped area
        directions: Optional pre-specified directions array
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        List of PouringDirectionResult, sorted by total_score (ascending)
    """
    if directions is None:
        directions = fibonacci_sphere(n_directions)
    
    # Precompute adjacency once for efficiency
    vertex_neighbors = build_vertex_neighbors(mesh)
    face_neighbors = build_face_neighbors(mesh)
    
    results: List[PouringDirectionResult] = []
    total = len(directions)
    
    for i, direction in enumerate(directions):
        result = score_pouring_direction(
            mesh, direction, tilt_angle_deg, area_threshold_mm2,
            vertex_neighbors, face_neighbors
        )
        results.append(result)
        
        if progress_callback is not None:
            progress_callback(i + 1, total)
        
        if (i + 1) % 16 == 0:
            logger.debug(f"Evaluated {i + 1}/{len(directions)} directions")
    
    # Sort by score (lower is better)
    results.sort(key=lambda r: r.total_score)
    
    return results


# ============================================================================
# SILICONE AND RESIN DIRECTION SELECTION
# ============================================================================

def select_silicone_directions(
    scored_results: List[PouringDirectionResult],
    max_alignment_angle_deg: float = 30.0,
    n_candidates: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select two nearly-aligned low-scoring directions for silicone pouring.
    
    For each mold piece P1, P2, we want directions f1, f2 that:
    1. Have low bubble-trapping scores
    2. Are nearly aligned (within max_alignment_angle)
    
    Having nearly aligned directions ensures the silicone has a good
    surface to level out while casting.
    
    Args:
        scored_results: Results sorted by score (ascending)
        max_alignment_angle_deg: Maximum angle between f1 and f2
        n_candidates: Number of top candidates to consider
        
    Returns:
        (f1, f2): Two direction vectors for silicone pouring
    """
    if len(scored_results) < 2:
        if len(scored_results) == 1:
            return (scored_results[0].direction, scored_results[0].direction)
        raise ValueError("Need at least one scored direction")
    
    candidates = scored_results[:min(n_candidates, len(scored_results))]
    
    # Cosine threshold for alignment
    min_cos = np.cos(np.radians(max_alignment_angle_deg))
    
    best_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_combined_score = float('inf')
    
    for i, r1 in enumerate(candidates):
        for r2 in candidates[i+1:]:
            # Check alignment (dot product of unit vectors)
            alignment = np.dot(r1.direction, r2.direction)
            
            if alignment >= min_cos:  # Nearly aligned
                combined = r1.total_score + r2.total_score
                
                if combined < best_combined_score:
                    best_combined_score = combined
                    best_pair = (r1.direction.copy(), r2.direction.copy())
    
    if best_pair is None:
        # Fallback: use two best directions regardless of alignment
        logger.warning("No aligned direction pair found, using two best directions")
        return (candidates[0].direction.copy(), candidates[1].direction.copy())
    
    return best_pair


def sample_cone_directions(
    axis: np.ndarray,
    half_angle_deg: float,
    n_samples: int
) -> np.ndarray:
    """
    Sample directions uniformly distributed in a cone around an axis.
    
    Args:
        axis: Central axis of the cone (unit vector)
        half_angle_deg: Half-angle of the cone in degrees
        n_samples: Number of directions to sample
        
    Returns:
        Array of shape (n_samples, 3) with unit direction vectors
    """
    half_angle_rad = np.radians(half_angle_deg)
    axis = axis / np.linalg.norm(axis)
    
    # Create orthonormal basis with axis as z
    if abs(axis[0]) < 0.9:
        perp = np.cross(axis, [1, 0, 0])
    else:
        perp = np.cross(axis, [0, 1, 0])
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(axis, perp)
    
    directions = []
    for i in range(n_samples):
        # Distribute azimuthal angle uniformly
        phi = 2 * np.pi * i / n_samples
        
        # Sample polar angle for uniform distribution in solid angle
        # Use fixed radial samples for reproducibility
        r = (i + 0.5) / n_samples  # Radial fraction
        theta = half_angle_rad * np.sqrt(r)
        
        # Convert to Cartesian in local frame
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Transform to global frame
        direction = x * perp + y * perp2 + z * axis
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)
    
    return np.array(directions)


def select_resin_direction(
    mesh: trimesh.Trimesh,
    f1: np.ndarray,
    f2: np.ndarray,
    cone_half_angle_deg: float = 10.0,
    n_samples_in_cone: int = 16,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5
) -> PouringDirectionResult:
    """
    Select optimal resin pouring direction in cone around f1/f2 bisector.
    
    The resin direction should be the lowest-scoring direction sampled
    in a small cone around the bisector of the silicone directions.
    This ensures good alignment for the leveling surface.
    
    Args:
        mesh: Input mesh
        f1, f2: Silicone pouring directions
        cone_half_angle_deg: Half-angle of sampling cone (10 degrees per paper)
        n_samples_in_cone: Number of directions to sample in cone
        tilt_angle_deg: Assumed tilt capability
        area_threshold_mm2: Filter threshold for trapped area
        
    Returns:
        Best PouringDirectionResult for resin
    """
    # Compute bisector of f1 and f2
    bisector = (f1 + f2) / 2
    norm = np.linalg.norm(bisector)
    if norm < 1e-10:
        # f1 and f2 are opposite - use f1 as bisector
        bisector = f1.copy()
    else:
        bisector = bisector / norm
    
    # Sample directions in cone around bisector
    cone_directions = sample_cone_directions(
        bisector, cone_half_angle_deg, n_samples_in_cone
    )
    
    # Include bisector itself
    all_directions = np.vstack([bisector.reshape(1, 3), cone_directions])
    
    # Score all directions
    results = evaluate_candidate_directions(
        mesh,
        directions=all_directions,
        tilt_angle_deg=tilt_angle_deg,
        area_threshold_mm2=area_threshold_mm2
    )
    
    return results[0]  # Best scoring (lowest)


# ============================================================================
# MAIN API
# ============================================================================

def find_optimal_pouring_directions(
    mesh: trimesh.Trimesh,
    n_candidate_directions: int = 64,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5,
    alignment_angle_deg: float = 30.0,
    resin_cone_angle_deg: float = 10.0,
    progress_callback: Optional[callable] = None
) -> OptimalPouringDirections:
    """
    Find optimal silicone and resin pouring directions for a mold piece.
    
    IMPORTANT: This function should be called on SILICONE MOLD PIECES (P1, P2),
    not on the input part mesh. The typical workflow is:
    1. Call this function on mold piece P1 -> get direction f1
    2. Call this function on mold piece P2 -> get direction f2
    3. Use select_resin_direction() with f1, f2 to get resin direction
    
    Alternatively, for a single mesh this function:
    1. Samples candidate directions uniformly on sphere
    2. Scores each direction for bubble trapping using persistence homology
    3. Selects two nearly-aligned low-scoring silicone directions
    4. Selects optimal resin direction in cone around their bisector
    
    Args:
        mesh: Silicone mold piece mesh (P1 or P2), NOT the input part mesh
        n_candidate_directions: Number of directions to sample (default 64)
        tilt_angle_deg: Assumed tilt capability during casting (default 10 degrees)
        area_threshold_mm2: Ignore trapped regions smaller than this (default 0.5 mm^2)
        alignment_angle_deg: Max angle between silicone directions (default 30 degrees)
        resin_cone_angle_deg: Cone angle for resin direction sampling (default 10 degrees)
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        OptimalPouringDirections with all results
    """
    start_time = time.time()
    
    logger.info(f"Finding optimal pouring directions for mesh with {len(mesh.faces)} faces")
    logger.info(f"Parameters: {n_candidate_directions} directions, tilt={tilt_angle_deg} deg, threshold={area_threshold_mm2} mm^2")
    
    # Step 1: Evaluate all candidate directions
    logger.info(f"Evaluating {n_candidate_directions} candidate directions...")
    all_results = evaluate_candidate_directions(
        mesh, n_candidate_directions, tilt_angle_deg, area_threshold_mm2,
        progress_callback=progress_callback
    )
    
    # Step 2: Select silicone directions
    logger.info("Selecting silicone pouring directions...")
    f1, f2 = select_silicone_directions(
        all_results, alignment_angle_deg
    )
    
    # Find scores for selected directions
    f1_score = next(
        (r.total_score for r in all_results if np.allclose(r.direction, f1)),
        0.0
    )
    f2_score = next(
        (r.total_score for r in all_results if np.allclose(r.direction, f2)),
        0.0
    )
    
    # Step 3: Select resin direction
    logger.info("Selecting resin pouring direction...")
    resin_result = select_resin_direction(
        mesh, f1, f2, resin_cone_angle_deg,
        tilt_angle_deg=tilt_angle_deg,
        area_threshold_mm2=area_threshold_mm2
    )
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Pouring direction optimization complete in {elapsed:.1f} ms")
    logger.info(f"Silicone 1: score={f1_score:.2f}, direction={f1}")
    logger.info(f"Silicone 2: score={f2_score:.2f}, direction={f2}")
    logger.info(f"Resin: score={resin_result.total_score:.2f}, direction={resin_result.direction}")
    
    return OptimalPouringDirections(
        silicone_direction_1=f1,
        silicone_direction_2=f2,
        silicone_score_1=f1_score,
        silicone_score_2=f2_score,
        resin_direction=resin_result.direction,
        resin_score=resin_result.total_score,
        all_scored_directions=all_results,
        computation_time_ms=elapsed
    )


def find_optimal_pouring_for_mold_pieces(
    mold_piece_1: trimesh.Trimesh,
    mold_piece_2: trimesh.Trimesh,
    n_candidate_directions: int = 64,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5,
    alignment_angle_deg: float = 30.0,
    resin_cone_angle_deg: float = 10.0,
    progress_callback: Optional[callable] = None
) -> OptimalPouringDirections:
    """
    Find optimal pouring directions for two silicone mold pieces.
    
    This is the recommended API for production use. It evaluates both mold
    pieces (P1 and P2) to find:
    1. f1: Optimal silicone pouring direction for P1
    2. f2: Optimal silicone pouring direction for P2 (nearly aligned with f1)
    3. Resin direction in a 10° cone around the bisector of f1 and f2
    
    The alignment constraint ensures the silicone has a good surface to 
    level out while casting.
    
    Args:
        mold_piece_1: First silicone mold piece (P1)
        mold_piece_2: Second silicone mold piece (P2)
        n_candidate_directions: Number of directions to sample (default 64)
        tilt_angle_deg: Assumed tilt capability during casting (default 10 degrees)
        area_threshold_mm2: Ignore trapped regions smaller than this (default 0.5 mm^2)
        alignment_angle_deg: Max angle between f1 and f2 (default 30 degrees)
        resin_cone_angle_deg: Cone angle for resin direction sampling (default 10 degrees)
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        OptimalPouringDirections with directions for both pieces and resin
    """
    start_time = time.time()
    total_directions = n_candidate_directions * 2  # Evaluating both pieces
    
    logger.info(f"Finding optimal pouring directions for mold pieces")
    logger.info(f"  P1: {len(mold_piece_1.faces)} faces, P2: {len(mold_piece_2.faces)} faces")
    
    def p1_progress(current, total):
        if progress_callback:
            progress_callback(current, total_directions)
    
    def p2_progress(current, total):
        if progress_callback:
            progress_callback(n_candidate_directions + current, total_directions)
    
    # Step 1: Evaluate directions for P1
    logger.info(f"Evaluating {n_candidate_directions} directions for mold piece P1...")
    p1_results = evaluate_candidate_directions(
        mold_piece_1, n_candidate_directions, tilt_angle_deg, area_threshold_mm2,
        progress_callback=p1_progress
    )
    
    # Step 2: Evaluate directions for P2
    logger.info(f"Evaluating {n_candidate_directions} directions for mold piece P2...")
    p2_results = evaluate_candidate_directions(
        mold_piece_2, n_candidate_directions, tilt_angle_deg, area_threshold_mm2,
        progress_callback=p2_progress
    )
    
    # Step 3: Find best direction for P1
    f1 = p1_results[0].direction  # Best (lowest score) direction for P1
    f1_score = p1_results[0].total_score
    
    # Step 4: Find best direction for P2 that is nearly aligned with f1
    f2 = None
    f2_score = float('inf')
    max_align_rad = np.radians(alignment_angle_deg)
    
    for result in p2_results:
        # Check alignment with f1
        dot = np.clip(np.dot(f1, result.direction), -1.0, 1.0)
        angle = np.arccos(np.abs(dot))  # Allow opposite directions
        
        if angle <= max_align_rad:
            if result.total_score < f2_score:
                f2 = result.direction
                f2_score = result.total_score
    
    if f2 is None:
        # No aligned direction found, use best for P2
        logger.warning("No aligned direction found for P2, using best direction")
        f2 = p2_results[0].direction
        f2_score = p2_results[0].total_score
    
    # Step 5: Select resin direction in cone around bisector
    logger.info("Selecting resin pouring direction...")
    # Use P1 for resin evaluation (could also average both pieces)
    resin_result = select_resin_direction(
        mold_piece_1, f1, f2, resin_cone_angle_deg,
        tilt_angle_deg=tilt_angle_deg,
        area_threshold_mm2=area_threshold_mm2
    )
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Mold piece pouring optimization complete in {elapsed:.1f} ms")
    logger.info(f"P1 direction (f1): score={f1_score:.2f}, direction={f1}")
    logger.info(f"P2 direction (f2): score={f2_score:.2f}, direction={f2}")
    logger.info(f"Resin: score={resin_result.total_score:.2f}, direction={resin_result.direction}")
    
    return OptimalPouringDirections(
        silicone_direction_1=f1,
        silicone_direction_2=f2,
        silicone_score_1=f1_score,
        silicone_score_2=f2_score,
        resin_direction=resin_result.direction,
        resin_score=resin_result.total_score,
        all_scored_directions=p1_results + p2_results,  # Combined results
        computation_time_ms=elapsed
    )
