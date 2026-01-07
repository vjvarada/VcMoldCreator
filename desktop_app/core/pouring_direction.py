"""
Pouring Direction Optimization Module

Determines optimal silicone and resin pouring directions by minimizing
air bubble trapping using persistence homology analysis.

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
from collections import defaultdict, deque
import logging
import time

import trimesh

from .mesh_analysis import compute_height_field, build_vertex_neighbors, build_face_neighbors
from .parting_direction import fibonacci_sphere

logger = logging.getLogger(__name__)


# ============================================================================
# GPU/ACCELERATION DETECTION
# ============================================================================

# Check if C++ fast_algorithms is available (already compiled for this project)
try:
    from . import fast_algorithms as _cpp
    HAS_CPP = True
    logger.info("C++ fast_algorithms available for acceleration")
except ImportError:
    HAS_CPP = False
    _cpp = None


# ============================================================================
# DATA CLASSES
# ============================================================================

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
class MoldAwarePouringDirections:
    """
    Complete result of mold-half-aware pouring direction optimization.
    
    Each mold half (H1 and H2) gets its own optimal silicone pouring direction,
    based on the portion of the part surface that belongs to that mold half.
    The resin direction is optimized for the combined part surface.
    
    Per paper: "Having nearly aligned silicone and resin pouring directions
    is required for the silicone to have a good surface to level out while casting"
    
    NOTE: H1 and H2 are poured in opposite orientations (one with parting surface
    down, the other flipped). The optimal directions for H1 and H2 are typically
    ANTI-ALIGNED (opposite). When is_anti_aligned is True, h2_silicone_direction
    has been NEGATED to represent the effective direction in H2's flipped frame.
    """
    # H1 mold half results
    h1_silicone_direction: np.ndarray
    h1_silicone_score: float
    h1_face_count: int
    h1_all_directions: List[PouringDirectionResult]
    
    # H2 mold half results
    # NOTE: If is_anti_aligned, h2_silicone_direction has been negated!
    h2_silicone_direction: np.ndarray
    h2_silicone_score: float
    h2_face_count: int
    h2_all_directions: List[PouringDirectionResult]
    
    # Resin direction (for the complete part)
    resin_direction: np.ndarray
    resin_score: float
    resin_maxima_positions: Optional[np.ndarray] = None  # (N, 3) positions of bubble-trapping maxima
    resin_global_maximum_position: Optional[np.ndarray] = None  # (3,) position of the highest maximum
    
    # Alignment metrics
    silicone_alignment_angle_deg: float = 0.0  # Angle between h1 and h2 directions (or negated h2)
    directions_well_aligned: bool = True  # True if angle < 30 deg
    is_anti_aligned: bool = False  # True if H1 and H2 optimal directions were opposite
    
    # The split meshes for visualization
    h1_mesh: Optional[trimesh.Trimesh] = None
    h2_mesh: Optional[trimesh.Trimesh] = None
    
    # Computation time
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
# PRECOMPUTED MESH DATA FOR FAST EVALUATION
# ============================================================================

class PrecomputedMeshData:
    """
    Precomputed mesh data structures for fast direction evaluation.
    
    Caches expensive computations that are direction-independent:
    - Vertex and face adjacency
    - Vertex-to-face mapping
    - Face areas
    - Face normals
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """Precompute all direction-independent data."""
        self.mesh = mesh
        self.n_vertices = len(mesh.vertices)
        self.n_faces = len(mesh.faces)
        
        # Vertices and faces as contiguous arrays
        self.vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        
        # Precompute adjacency
        self.vertex_neighbors = build_vertex_neighbors(mesh)
        self.face_neighbors = build_face_neighbors(mesh)
        
        # Precompute vertex-to-face mapping (which faces contain each vertex)
        self.vertex_to_faces = self._build_vertex_to_faces()
        
        # Cache face areas and normals
        self.face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        self.face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        
        # Precompute face vertex indices for vectorized height computation
        # Shape: (n_faces, 3) - already mesh.faces
        
    def _build_vertex_to_faces(self) -> Dict[int, np.ndarray]:
        """Build mapping from vertex index to face indices containing it."""
        vertex_to_faces: Dict[int, List[int]] = defaultdict(list)
        for face_idx, face in enumerate(self.faces):
            for v in face:
                vertex_to_faces[v].append(face_idx)
        # Convert to numpy arrays for faster indexing
        return {v: np.array(faces, dtype=np.int64) for v, faces in vertex_to_faces.items()}
    
    def compute_heights(self, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute vertex and face heights for a direction (vectorized)."""
        direction = direction / np.linalg.norm(direction)
        vertex_heights = self.vertices @ direction
        face_heights = vertex_heights[self.faces].mean(axis=1)
        return vertex_heights, face_heights
    
    def compute_heights_batch(self, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vertex and face heights for multiple directions at once (vectorized).
        
        Args:
            directions: (N, 3) array of direction vectors
            
        Returns:
            vertex_heights: (N, n_vertices) array
            face_heights: (N, n_faces) array
        """
        # Normalize directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        # Vectorized batch computation using numpy
        # Batch matrix multiply: (N, 3) @ (V, 3).T -> (N, V)
        vertex_heights = directions @ self.vertices.T
        
        # Compute face heights: average of vertex heights per face
        # faces: (F, 3) indices, vertex_heights: (N, V)
        # Result: (N, F)
        face_heights = vertex_heights[:, self.faces].mean(axis=2)
        
        return vertex_heights, face_heights
    
    def get_vertex_faces(self, vertex_idx: int) -> np.ndarray:
        """Get faces containing a vertex (fast lookup)."""
        return self.vertex_to_faces.get(vertex_idx, np.array([], dtype=np.int64))


def _score_direction_fast(
    precomputed: PrecomputedMeshData,
    direction: np.ndarray,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5
) -> PouringDirectionResult:
    """
    Fast direction scoring using precomputed mesh data.
    
    This is optimized for repeated evaluation with different directions.
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    
    # Compute heights (vectorized)
    vertex_heights, face_heights = precomputed.compute_heights(direction)
    
    # Get persistence pairs using precomputed neighbors
    pairs, _ = compute_persistence_pairs(
        precomputed.mesh, direction, precomputed.vertex_neighbors
    )
    
    # Score pairs with precomputed data
    relevant_pairs: List[PersistencePair] = []
    total_score = 0.0
    
    # Faces can drain if NOT strongly downward: angle from UP < (90 deg + alpha)
    # This means: normal . direction > -sin(alpha)
    tilt_threshold = np.sin(np.radians(tilt_angle_deg))
    
    for pair in pairs:
        # Skip pairs with negligible persistence
        if pair.persistence < 1e-6:
            continue
        
        # Fast region growing with precomputed data (ceiling faces only)
        trapped = _grow_trapped_region_fast(
            precomputed, pair.maximum_idx, pair.saddle_height, face_heights, direction
        )
        
        if not trapped:
            continue
        
        # Fast tiltable region computation
        tiltable = _compute_tiltable_fast(
            precomputed, direction, trapped, pair.saddle_idx, tilt_threshold
        )
        
        # Final trapped region
        final_trapped = trapped - tiltable
        
        if final_trapped:
            score = np.sum(precomputed.face_areas[list(final_trapped)])
            if score >= area_threshold_mm2:
                pair.trapped_faces = final_trapped
                pair.relevance_score = score
                relevant_pairs.append(pair)
                total_score += score
    
    return PouringDirectionResult(
        direction=direction.copy(),
        pairs=relevant_pairs,
        total_score=total_score,
        relevant_pair_count=len(relevant_pairs)
    )


def _grow_trapped_region_fast(
    precomputed: PrecomputedMeshData,
    maximum_idx: int,
    saddle_height: float,
    face_heights: np.ndarray,
    direction: np.ndarray
) -> Set[int]:
    """
    Fast trapped region growing with precomputed vertex-to-face mapping.
    
    Only includes CEILING faces (normal pointing opposite to pour direction).
    """
    # Compute ceiling mask: faces where normal . direction < 0
    normal_dots = np.dot(precomputed.face_normals, direction)
    is_ceiling = normal_dots < 0
    
    # Find seed face using precomputed mapping (must be a ceiling face)
    max_vertex_faces = precomputed.get_vertex_faces(maximum_idx)
    if len(max_vertex_faces) == 0:
        return set()
    
    # Filter to ceiling faces
    ceiling_mask = is_ceiling[max_vertex_faces]
    ceiling_seed_faces = max_vertex_faces[ceiling_mask]
    if len(ceiling_seed_faces) == 0:
        return set()  # Maximum not on ceiling face
    
    seed_face = ceiling_seed_faces[np.argmax(face_heights[ceiling_seed_faces])]
    
    # BFS with deque - only include ceiling faces above saddle height
    trapped_faces: Set[int] = set()
    queue = deque([seed_face])
    visited = {seed_face}
    
    face_neighbors = precomputed.face_neighbors
    
    while queue:
        face_idx = queue.popleft()
        
        # Only include ceiling faces above saddle height
        if is_ceiling[face_idx] and face_heights[face_idx] >= saddle_height:
            trapped_faces.add(face_idx)
            
            for neighbor in face_neighbors.get(face_idx, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return trapped_faces


def _compute_tiltable_fast(
    precomputed: PrecomputedMeshData,
    direction: np.ndarray,
    trapped_faces: Set[int],
    saddle_idx: int,
    tilt_threshold: float
) -> Set[int]:
    """
    Fast tiltable region computation.
    
    Physical interpretation: When the mold is tilted by angle α, faces that
    are NOT strongly downward-facing can allow air to escape. A face is
    "strongly downward" if it points more than (90° + α) from the UP direction.
    
    Condition: angle(normal, direction) < (90° + α)
              → cos(angle) > cos(90° + α) = -sin(α)
              → normal · direction > -sin(α)
    
    Args:
        tilt_threshold: Should be sin(tilt_angle_deg)
    """
    if not trapped_faces:
        return set()
    
    # Find seed faces containing saddle vertex
    saddle_faces = set(precomputed.get_vertex_faces(saddle_idx))
    seed_faces = saddle_faces & trapped_faces
    
    if not seed_faces:
        return set()
    
    normals = precomputed.face_normals
    face_neighbors = precomputed.face_neighbors
    
    tiltable: Set[int] = set()
    queue = deque(seed_faces)
    visited = set(seed_faces)
    
    while queue:
        face_idx = queue.popleft()
        normal_dot = np.dot(normals[face_idx], direction)
        
        # Face can drain if NOT strongly downward: normal · direction > -sin(α)
        if normal_dot > -tilt_threshold:
            tiltable.add(face_idx)
            
            for neighbor in face_neighbors.get(face_idx, []):
                if neighbor in trapped_faces and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return tiltable


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
    direction: np.ndarray,
    face_neighbors: Optional[Dict[int, List[int]]] = None,
    face_heights: Optional[np.ndarray] = None
) -> Set[int]:
    """
    Grow region A_m^n from maximum m down to saddle height n.
    
    This is the region where air would be trapped: CEILING faces connected
    to the maximum that are above the saddle height.
    
    IMPORTANT: Only ceiling faces (normal pointing opposite to pour direction)
    can trap air. Floor faces allow bubbles to escape upward.
    
    Per paper: "During silicone casting, the mold volume could trap air bubbles
    around downward maxima" - meaning maxima on DOWNWARD-facing surfaces.
    
    Args:
        mesh: Input mesh
        maximum_idx: Vertex index of the maximum
        saddle_height: Height of the saddle point
        vertex_heights: Height values for all vertices
        direction: Pouring direction (UP). Ceiling faces have normal . direction < 0
        face_neighbors: Precomputed face adjacency (optional)
        face_heights: Precomputed face heights (optional, for speed)
        
    Returns:
        Set of face indices in the trapped region (ceiling faces only)
    """
    if face_neighbors is None:
        face_neighbors = build_face_neighbors(mesh)
    
    # Compute face heights if not provided
    if face_heights is None:
        face_vertex_heights = vertex_heights[mesh.faces]
        face_heights = face_vertex_heights.mean(axis=1)
    
    # Precompute ceiling mask: faces where normal . direction < 0 (pointing DOWN)
    normals = mesh.face_normals
    direction = direction / np.linalg.norm(direction)
    normal_dots = np.dot(normals, direction)
    is_ceiling = normal_dots < 0  # Ceiling faces point opposite to pour direction
    
    # Find seed face: a CEILING face containing the maximum vertex, with highest centroid
    max_vertex_faces = np.where(np.any(mesh.faces == maximum_idx, axis=1))[0]
    if len(max_vertex_faces) == 0:
        return set()
    
    # Filter to ceiling faces only
    ceiling_seed_faces = max_vertex_faces[is_ceiling[max_vertex_faces]]
    if len(ceiling_seed_faces) == 0:
        # Maximum is not on a ceiling face - no trapped region
        return set()
    
    seed_face = ceiling_seed_faces[np.argmax(face_heights[ceiling_seed_faces])]
    
    # BFS to grow region: only include CEILING faces above saddle height
    trapped_faces: Set[int] = set()
    queue = deque([seed_face])
    visited = {seed_face}
    
    while queue:
        face_idx = queue.popleft()
        
        # Include face if it's a ceiling face AND above saddle height
        if is_ceiling[face_idx] and face_heights[face_idx] >= saddle_height:
            trapped_faces.add(face_idx)
            
            # Add unvisited neighbors (will check ceiling criterion when processed)
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
    
    Physical interpretation: When the mold is tilted by angle α, faces that
    are NOT strongly downward-facing can allow air to escape. Air trapped at
    ceiling pockets can flow toward the saddle if there's a drainage path.
    
    A face is "strongly downward-facing" if its normal points more than
    (90° + α) from the UP direction (pouring direction f).
    
    Condition: angle(normal, f) < (90° + α)
              → cos(angle) > cos(90° + α) = -sin(α)
              → normal · f > -sin(α)
    
    With α=10°, faces with normal · f > -0.17 can drain, meaning faces
    pointing up to ~100° from UP (10° past horizontal) are drainable.
    
    Args:
        mesh: Input mesh
        direction: Pouring direction (unit vector pointing UP)
        trapped_faces: Set of face indices in the trapped region A_m^n
        saddle_idx: Vertex index of saddle point (drainage point)
        tilt_angle_deg: Maximum tilt angle in degrees (default 10°)
        face_neighbors: Precomputed face adjacency (optional)
        
    Returns:
        Set of face indices that can drain when tilted (region T)
    """
    if not trapped_faces:
        return set()
    
    if face_neighbors is None:
        face_neighbors = build_face_neighbors(mesh)
    
    # Threshold: faces with normal · f > -sin(α) can drain
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
    queue = deque(seed_faces)
    visited = set(seed_faces)
    
    while queue:
        face_idx = queue.popleft()
        normal_dot = np.dot(normals[face_idx], direction)
        
        # Face can drain if NOT strongly downward: normal · f > -sin(α)
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
    face_neighbors: Optional[Dict[int, List[int]]] = None,
    face_heights: Optional[np.ndarray] = None
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
        face_heights: Precomputed face heights (optional, for speed)
        
    Returns:
        Updated PersistencePair with relevance_score and trapped_faces
    """
    if face_neighbors is None:
        face_neighbors = build_face_neighbors(mesh)
    
    # Skip pairs with very small persistence (likely noise)
    if pair.persistence < 1e-6:
        pair.trapped_faces = set()
        pair.relevance_score = 0.0
        return pair
    
    # Grow trapped region from maximum to saddle (ceiling faces only)
    trapped = grow_trapped_region(
        mesh, pair.maximum_idx, pair.saddle_height,
        vertex_heights, direction, face_neighbors, face_heights
    )
    
    if not trapped:
        pair.trapped_faces = set()
        pair.relevance_score = 0.0
        return pair
    
    # Compute region that can drain with tilting
    tiltable = compute_tiltable_region(
        mesh, direction, trapped, pair.saddle_idx,
        tilt_angle_deg, face_neighbors
    )
    
    # Final trapped region
    final_trapped = trapped - tiltable
    
    # Compute area score using numpy for speed
    if final_trapped:
        face_areas = mesh.area_faces
        score = np.sum(face_areas[list(final_trapped)])
    else:
        score = 0.0
    
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
    
    # Precompute face heights once for all pairs (optimization)
    face_vertex_heights = vertex_heights[mesh.faces]
    face_heights = face_vertex_heights.mean(axis=1)
    
    # Compute relevance for each pair
    relevant_pairs: List[PersistencePair] = []
    total_score = 0.0
    
    for pair in pairs:
        compute_pair_relevance(
            mesh, pair, direction, vertex_heights,
            tilt_angle_deg, face_neighbors, face_heights
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
    progress_callback: Optional[callable] = None,
    use_parallel: bool = False,  # Disabled by default - GIL limits benefit
    n_workers: int = None
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
        use_parallel: Unused parameter (kept for API compatibility)
        n_workers: Unused parameter (kept for API compatibility)
        
    Returns:
        List of PouringDirectionResult, sorted by total_score (ascending)
    """
    if directions is None:
        directions = fibonacci_sphere(n_directions)
    
    # Precompute adjacency once for efficiency
    vertex_neighbors = build_vertex_neighbors(mesh)
    face_neighbors = build_face_neighbors(mesh)
    
    # Sequential evaluation (parallel disabled due to GIL limitations)
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


def evaluate_candidate_directions_fast(
    mesh: trimesh.Trimesh,
    n_directions: int = 64,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5,
    progress_callback: Optional[callable] = None
) -> List[PouringDirectionResult]:
    """
    Fast direction evaluation with precomputed mesh data and coarse-to-fine strategy.
    
    Optimizations:
    1. Precompute all direction-independent mesh data once
    2. Coarse pass: Evaluate n_directions/4 uniformly distributed directions
    3. Fine pass: Refine around the top 4 best directions
    
    Args:
        mesh: Input mesh
        n_directions: Target number of directions
        tilt_angle_deg: Assumed tilt capability
        area_threshold_mm2: Filter threshold for trapped area
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        List of PouringDirectionResult, sorted by total_score (ascending)
    """
    # Precompute all mesh data once (major speedup)
    precomputed = PrecomputedMeshData(mesh)
    
    # Phase 1: Coarse evaluation
    n_coarse = max(16, n_directions // 4)
    coarse_directions = fibonacci_sphere(n_coarse)
    
    coarse_results: List[PouringDirectionResult] = []
    
    # Progress tracking
    total_evals = n_coarse + 32  # Estimated total
    current_eval = [0]
    
    def internal_progress(current, total):
        current_eval[0] = current
        if progress_callback:
            progress_callback(current_eval[0], total_evals)
    
    for i, direction in enumerate(coarse_directions):
        # Use fast scorer with precomputed data
        result = _score_direction_fast(
            precomputed, direction, tilt_angle_deg, area_threshold_mm2
        )
        coarse_results.append(result)
        internal_progress(i + 1, total_evals)
    
    # Sort coarse results
    coarse_results.sort(key=lambda r: r.total_score)
    
    # Phase 2: Refine around top N best directions
    n_top = min(4, len(coarse_results))
    top_directions = [r.direction for r in coarse_results[:n_top]]
    
    # Sample additional directions in cones around the best ones
    cone_angle = 15.0  # degrees
    n_per_cone = 8
    
    all_results = list(coarse_results)  # Include coarse results
    
    for i, center_dir in enumerate(top_directions):
        cone_dirs = sample_cone_directions(center_dir, cone_angle, n_per_cone)
        
        for j, direction in enumerate(cone_dirs):
            # Use fast scorer with precomputed data
            result = _score_direction_fast(
                precomputed, direction, tilt_angle_deg, area_threshold_mm2
            )
            all_results.append(result)
            internal_progress(n_coarse + i * n_per_cone + j + 1, total_evals)
    
    # Final sort
    all_results.sort(key=lambda r: r.total_score)
    
    return all_results


# ============================================================================
# SILICONE AND RESIN DIRECTION SELECTION
# ============================================================================

def select_aligned_silicone_directions_across_halves(
    h1_results: List[PouringDirectionResult],
    h2_results: List[PouringDirectionResult],
    max_alignment_angle_deg: float = 30.0,
    n_candidates: int = 20
) -> Tuple[PouringDirectionResult, PouringDirectionResult, float, bool]:
    """
    Select nearly-aligned silicone pouring directions for two mold halves.
    
    Per the paper: "For each piece P1, P2 of the soft silicone part, we evaluate
    the score of a set of candidate directions for silicone pouring and choose
    from those scoring lowest a couple of nearly aligned directions f1, f2."
    
    IMPORTANT: H1 and H2 mold halves have opposite orientations when being poured:
    - When pouring H1, the part sits with parting surface facing DOWN
    - When pouring H2, the part is FLIPPED (parting surface faces UP)
    
    Therefore, the optimal direction for H1 (pointing UP in H1's frame) corresponds
    to the OPPOSITE direction for H2 (pointing DOWN in H1's frame, but UP in H2's
    flipped frame).
    
    We search for pairs where f1 and f2 are either:
    - ALIGNED (same direction): f1 . f2 > cos(threshold)
    - ANTI-ALIGNED (opposite): f1 . f2 < -cos(threshold), meaning f1 ~ -f2
    
    For anti-aligned pairs, we negate f2 to get the effective pouring direction
    in H2's flipped frame, ensuring consistent manufacturing setup.
    
    Args:
        h1_results: Scored results for H1 mold piece (sorted by score ascending)
        h2_results: Scored results for H2 mold piece (sorted by score ascending)
        max_alignment_angle_deg: Maximum angle between f1 and f2 (or f1 and -f2)
        n_candidates: Number of top candidates from each half to consider
        
    Returns:
        Tuple of (h1_best_result, h2_best_result, alignment_angle_deg, is_anti_aligned)
        If is_anti_aligned is True, f2 has been negated to represent the effective
        pouring direction in H2's flipped coordinate frame.
    """
    if not h1_results:
        raise ValueError("No H1 results provided")
    if not h2_results:
        raise ValueError("No H2 results provided")
    
    # Take top candidates from each half - use more candidates for better coverage
    h1_candidates = h1_results[:min(n_candidates, len(h1_results))]
    h2_candidates = h2_results[:min(n_candidates, len(h2_results))]
    
    # Log score ranges for diagnostics
    h1_score_range = f"{h1_candidates[0].total_score:.2f} - {h1_candidates[-1].total_score:.2f}"
    h2_score_range = f"{h2_candidates[0].total_score:.2f} - {h2_candidates[-1].total_score:.2f}"
    logger.info(f"Searching for aligned pair: H1 scores [{h1_score_range}], H2 scores [{h2_score_range}]")
    
    def find_best_pair_within_angle(angle_threshold_deg: float):
        """Find best pair within given angle threshold, allowing anti-alignment."""
        min_cos = np.cos(np.radians(angle_threshold_deg))
        best_h1 = None
        best_h2 = None
        best_combined_score = float('inf')
        best_angle = 0.0
        best_is_anti_aligned = False
        
        for r1 in h1_candidates:
            for r2 in h2_candidates:
                alignment_cos = np.dot(r1.direction, r2.direction)
                
                # Check for alignment (same direction)
                if alignment_cos >= min_cos:
                    combined = r1.total_score + r2.total_score
                    if combined < best_combined_score:
                        best_combined_score = combined
                        best_h1 = r1
                        best_h2 = r2
                        best_angle = np.degrees(np.arccos(np.clip(alignment_cos, -1, 1)))
                        best_is_anti_aligned = False
                
                # Check for anti-alignment (opposite directions)
                # This is the expected case for H1/H2 due to their flipped orientations
                if alignment_cos <= -min_cos:
                    combined = r1.total_score + r2.total_score
                    if combined < best_combined_score:
                        best_combined_score = combined
                        best_h1 = r1
                        best_h2 = r2
                        # Angle from anti-aligned: 180 - angle from aligned
                        best_angle = np.degrees(np.arccos(np.clip(-alignment_cos, -1, 1)))
                        best_is_anti_aligned = True
        
        return best_h1, best_h2, best_angle, best_combined_score, best_is_anti_aligned
    
    # Try progressively relaxed alignment constraints
    angle_thresholds = [max_alignment_angle_deg, 45.0, 60.0, 90.0]
    
    for threshold in angle_thresholds:
        best_h1, best_h2, best_angle, best_score, is_anti_aligned = find_best_pair_within_angle(threshold)
        
        if best_h1 is not None and best_h2 is not None:
            if threshold > max_alignment_angle_deg:
                logger.warning(
                    f"No aligned pair found within {max_alignment_angle_deg}deg. "
                    f"Relaxed to {threshold}deg threshold."
                )
            alignment_type = "ANTI-ALIGNED (opposite)" if is_anti_aligned else "ALIGNED (same)"
            logger.info(
                f"Found {alignment_type} pair: H1 score={best_h1.total_score:.2f}, "
                f"H2 score={best_h2.total_score:.2f}, angle={best_angle:.1f}deg"
            )
            return best_h1, best_h2, best_angle, is_anti_aligned
    
    # Still no pair found even at 90 deg - use independent best and warn
    logger.warning(
        f"No aligned direction pair found even at 90 deg. "
        f"Mold halves may require fundamentally different pouring directions."
    )
    best_h1 = h1_candidates[0]
    best_h2 = h2_candidates[0]
    
    alignment_cos = np.dot(best_h1.direction, best_h2.direction)
    best_alignment_angle = np.degrees(np.arccos(np.clip(abs(alignment_cos), -1, 1)))
    is_anti_aligned = alignment_cos < 0
    
    logger.warning(
        f"Using independent best directions ({best_alignment_angle:.1f}deg apart). "
        f"Consider manual adjustment for better silicone leveling."
    )
    logger.info(
        f"  H1 best: direction={best_h1.direction}, score={best_h1.total_score:.2f}"
    )
    logger.info(
        f"  H2 best: direction={best_h2.direction}, score={best_h2.total_score:.2f}"
    )
    
    return best_h1, best_h2, best_alignment_angle, is_anti_aligned


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
    cone_half_angle_deg: float = 5.0,
    n_samples_in_cone: int = 16,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5
) -> PouringDirectionResult:
    """
    Select optimal resin pouring direction in cone around f1/f2 bisector.
    
    Per the paper: "The resin casting direction is then chosen as the lowest
    scoring one in a set of directions sampled in a small cone around the
    bisector of f1, f2 (opening angle of 10° in our experiments). Having
    nearly aligned silicone and resin pouring directions is required for
    the silicone to have a good surface to level out while casting."
    
    Args:
        mesh: Input mesh (the complete part, not mold halves)
        f1, f2: Silicone pouring directions for each mold piece
        cone_half_angle_deg: Half-angle of sampling cone (5° = 10° opening angle per paper)
        n_samples_in_cone: Number of directions to sample in cone
        tilt_angle_deg: Assumed tilt capability
        area_threshold_mm2: Filter threshold for trapped area
        
    Returns:
        Best PouringDirectionResult for resin (lowest score in cone)
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
# MOLD-HALF AWARE POURING DIRECTION OPTIMIZATION
# ============================================================================

def classify_part_mesh_faces(
    part_mesh: trimesh.Trimesh,
    tet_result,  # TetrahedralMeshResult with escape labels
    distance_threshold: float = None
) -> Tuple[np.ndarray, Set[int], Set[int]]:
    """
    Classify each face of the part mesh as belonging to H1 or H2 mold half.
    
    Uses the Dijkstra escape labels from the tetrahedral mesh to determine
    which mold half each part of the part surface belongs to.
    
    The mapping works as follows:
    1. Get interior vertices from tet mesh that are on the part surface (boundary_labels == -1)
    2. These vertices have escape labels (1=H1, 2=H2) from Dijkstra
    3. Map each part mesh vertex to nearest tet interior vertex
    4. Classify each face by majority vote of its vertex labels
    
    Args:
        part_mesh: The original part mesh
        tet_result: TetrahedralMeshResult with seed_escape_labels and seed_vertex_indices
        distance_threshold: Max distance to match part vertices to tet vertices.
                           If None, auto-computed as 5% of mesh size.
    
    Returns:
        Tuple of:
            - face_labels: (F,) array with 1=H1, 2=H2, 0=unclassified for each face
            - h1_faces: Set of face indices belonging to H1
            - h2_faces: Set of face indices belonging to H2
    """
    from scipy.spatial import cKDTree
    
    # Get escape labels from tet result
    escape_labels = tet_result.seed_escape_labels
    seed_indices = tet_result.seed_vertex_indices
    boundary_labels = tet_result.boundary_labels
    
    if escape_labels is None or seed_indices is None:
        raise ValueError("TetrahedralMeshResult must have seed_escape_labels and seed_vertex_indices computed")
    
    # Get tet mesh vertices - use original (non-inflated) if available for better matching
    if tet_result.vertices_original is not None:
        tet_verts = tet_result.vertices_original
        logger.info("Using original (non-inflated) tet mesh vertices for face classification")
    else:
        tet_verts = tet_result.vertices
        logger.info("Using inflated tet mesh vertices for face classification")
    
    # Filter to only use interior vertices that are on the PART SURFACE (boundary_labels == -1)
    # These are the "seed" vertices that are closest to the actual part mesh
    if boundary_labels is not None:
        # Create mask for seed_indices that are on part surface
        part_surface_mask = np.array([boundary_labels[idx] == -1 for idx in seed_indices])
        part_surface_seed_indices = seed_indices[part_surface_mask]
        part_surface_escape_labels = escape_labels[part_surface_mask]
        logger.info(f"Filtered to {len(part_surface_seed_indices)}/{len(seed_indices)} interior vertices on part surface")
    else:
        # Fallback: use all seed vertices
        part_surface_seed_indices = seed_indices
        part_surface_escape_labels = escape_labels
        logger.warning("No boundary_labels available, using all seed vertices")
    
    if len(part_surface_seed_indices) == 0:
        logger.error("No part surface vertices found!")
        n_faces = len(part_mesh.faces)
        return np.zeros(n_faces, dtype=np.int8), set(), set()
    
    # Get positions of part-surface vertices
    seed_positions = tet_verts[part_surface_seed_indices]
    
    # Build KD-tree of seed positions for fast nearest neighbor lookup
    seed_tree = cKDTree(seed_positions)
    
    # Auto-compute distance threshold - use larger value since tet mesh may be offset
    if distance_threshold is None:
        bounds = part_mesh.bounds
        mesh_size = np.linalg.norm(bounds[1] - bounds[0])
        distance_threshold = mesh_size * 0.05  # 5% of mesh size (larger to handle offset)
    
    # Get part mesh vertices
    part_verts = np.asarray(part_mesh.vertices, dtype=np.float64)
    
    # For each part mesh vertex, find nearest seed vertex and get its label
    distances, seed_neighbors = seed_tree.query(part_verts, k=1)
    
    # Create vertex labels array
    n_verts = len(part_verts)
    vertex_labels = np.zeros(n_verts, dtype=np.int8)
    
    # Label all vertices based on nearest part-surface vertex (no distance threshold for now)
    # Since we're mapping to vertices that ARE on the part surface, they should be close
    vertex_labels = part_surface_escape_labels[seed_neighbors]
    
    # Log distance statistics for debugging
    logger.info(f"Part mesh vertex classification: all {n_verts} vertices mapped")
    logger.info(f"  Distance stats: min={distances.min():.4f}, max={distances.max():.4f}, mean={distances.mean():.4f}")
    logger.info(f"  H1 vertices: {np.sum(vertex_labels == 1)}, H2 vertices: {np.sum(vertex_labels == 2)}, unclassified: {np.sum(vertex_labels == 0)}")
    
    # Classify faces based on vertex majority voting
    faces = part_mesh.faces
    n_faces = len(faces)
    face_labels = np.zeros(n_faces, dtype=np.int8)
    
    for fi in range(n_faces):
        v0, v1, v2 = faces[fi]
        labels = [vertex_labels[v0], vertex_labels[v1], vertex_labels[v2]]
        
        # Count H1 and H2 votes
        h1_count = labels.count(1)
        h2_count = labels.count(2)
        
        # Majority voting
        if h1_count > h2_count:
            face_labels[fi] = 1
        elif h2_count > h1_count:
            face_labels[fi] = 2
        elif h1_count > 0:  # Tie with at least one H1 -> assign to H1
            face_labels[fi] = 1
        elif h2_count > 0:  # Tie with at least one H2 -> assign to H2
            face_labels[fi] = 2
        # else: unclassified (0)
    
    h1_faces = set(np.where(face_labels == 1)[0])
    h2_faces = set(np.where(face_labels == 2)[0])
    
    logger.info(f"Part mesh face classification: H1={len(h1_faces)}, H2={len(h2_faces)}, unclassified={n_faces - len(h1_faces) - len(h2_faces)}")
    
    return face_labels, h1_faces, h2_faces


def split_part_mesh_by_mold_half(
    part_mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    h1_faces: Set[int],
    h2_faces: Set[int]
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """
    Split the part mesh into two submeshes based on mold half classification.
    
    Args:
        part_mesh: The original part mesh
        face_labels: (F,) array with 1=H1, 2=H2, 0=unclassified
        h1_faces: Set of face indices belonging to H1
        h2_faces: Set of face indices belonging to H2
    
    Returns:
        Tuple of (h1_mesh, h2_mesh) - either can be None if no faces belong to that half
    """
    vertices = part_mesh.vertices
    faces = part_mesh.faces
    
    h1_mesh = None
    h2_mesh = None
    
    # Extract H1 submesh
    if h1_faces:
        h1_face_indices = np.array(list(h1_faces), dtype=np.int64)
        h1_faces_arr = faces[h1_face_indices]
        
        # Get unique vertices used by H1 faces
        h1_vert_indices = np.unique(h1_faces_arr.flatten())
        
        # Create vertex remapping
        h1_vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(h1_vert_indices)}
        
        # Remap face indices
        h1_new_faces = np.array([
            [h1_vert_map[v] for v in face]
            for face in h1_faces_arr
        ], dtype=np.int64)
        
        h1_mesh = trimesh.Trimesh(
            vertices=vertices[h1_vert_indices].copy(),
            faces=h1_new_faces,
            process=False
        )
        logger.info(f"H1 submesh: {len(h1_mesh.vertices)} vertices, {len(h1_mesh.faces)} faces")
    
    # Extract H2 submesh
    if h2_faces:
        h2_face_indices = np.array(list(h2_faces), dtype=np.int64)
        h2_faces_arr = faces[h2_face_indices]
        
        # Get unique vertices used by H2 faces
        h2_vert_indices = np.unique(h2_faces_arr.flatten())
        
        # Create vertex remapping
        h2_vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(h2_vert_indices)}
        
        # Remap face indices
        h2_new_faces = np.array([
            [h2_vert_map[v] for v in face]
            for face in h2_faces_arr
        ], dtype=np.int64)
        
        h2_mesh = trimesh.Trimesh(
            vertices=vertices[h2_vert_indices].copy(),
            faces=h2_new_faces,
            process=False
        )
        logger.info(f"H2 submesh: {len(h2_mesh.vertices)} vertices, {len(h2_mesh.faces)} faces")
    
    return h1_mesh, h2_mesh


def invert_mesh_for_cavity_analysis(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Invert a mesh to simulate the mold cavity ceiling for bubble analysis.
    
    When pouring silicone into a mold cavity:
    - The part surface forms the "ceiling" of the cavity
    - Bubbles rise and get trapped at concave pockets in the ceiling
    - The ceiling is the INVERSE of the part surface (normals point into cavity)
    
    This function flips the face winding order to invert the normals,
    effectively converting the part surface to the cavity ceiling surface.
    
    Args:
        mesh: Original part surface mesh (normals pointing outward from part)
        
    Returns:
        Inverted mesh (normals pointing into cavity, i.e., toward the part)
    """
    # Flip face winding order to invert normals
    inverted_faces = mesh.faces[:, ::-1].copy()
    
    inverted_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=inverted_faces,
        process=False
    )
    
    return inverted_mesh


def find_mold_aware_pouring_directions(
    part_mesh: trimesh.Trimesh,
    tet_result,  # TetrahedralMeshResult with escape labels
    n_candidate_directions: int = 64,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5,
    progress_callback: Optional[callable] = None,
    use_fast: bool = True,
    parting_direction: Optional[np.ndarray] = None
) -> MoldAwarePouringDirections:
    """
    Find optimal pouring directions for each mold half separately.
    
    This is the correct approach for silicone mold pouring:
    - Each mold half (H1, H2) is poured separately
    - The silicone pouring direction should be optimized for the portion
      of the part surface that belongs to that mold half
    - The resin pouring direction is optimized for the complete part
    
    Args:
        part_mesh: The original part mesh
        tet_result: TetrahedralMeshResult with seed_escape_labels computed
        n_candidate_directions: Number of directions to sample
        tilt_angle_deg: Assumed tilt capability during casting
        area_threshold_mm2: Ignore trapped regions smaller than this
        progress_callback: Optional callback(current, total) for progress updates
        use_fast: If True, uses coarse-to-fine evaluation for speed (default True)
        parting_direction: Optional parting direction for diagnostic logging
    
    Returns:
        MoldAwarePouringDirections with separate results for H1 and H2
    """
    start_time = time.time()
    
    logger.info(f"Finding mold-aware pouring directions for mesh with {len(part_mesh.faces)} faces")
    
    # Step 1: Classify part mesh faces by mold half
    logger.info("Classifying part mesh faces by mold half...")
    face_labels, h1_faces, h2_faces = classify_part_mesh_faces(part_mesh, tet_result)
    
    # Step 2: Split the part mesh
    logger.info("Splitting part mesh into H1 and H2 submeshes...")
    h1_mesh, h2_mesh = split_part_mesh_by_mold_half(part_mesh, face_labels, h1_faces, h2_faces)
    
    # Step 3: INVERT the meshes for cavity analysis
    # SILICONE POURING ANALYSIS:
    # When pouring silicone around the part, bubbles rise and get trapped
    # against undersides/overhangs of the part surface (faces pointing DOWN).
    # We analyze the ORIGINAL mesh (not inverted) - ceiling pockets are faces
    # with normal · up < 0 (pointing opposite to pouring direction).
    #
    # The h1_mesh and h2_mesh are submeshes of the part surface.
    logger.info(f"Analyzing silicone pour: using original part surface (not inverted)")
    
    # Choose evaluation function
    eval_func = evaluate_candidate_directions_fast if use_fast else evaluate_candidate_directions
    
    # Track progress across both meshes
    total_directions = n_candidate_directions * 2  # For H1 and H2
    current_progress = [0]  # Use list to allow modification in nested function
    
    def h1_progress_callback(current, total):
        current_progress[0] = current
        if progress_callback:
            progress_callback(current_progress[0], total_directions)
    
    def h2_progress_callback(current, total):
        current_progress[0] = n_candidate_directions + current
        if progress_callback:
            progress_callback(current_progress[0], total_directions)
    
    # Step 4: Evaluate candidate directions for H1 (if available)
    h1_all_directions = []
    
    if h1_mesh is not None and len(h1_mesh.faces) > 0:
        logger.info(f"Computing H1 silicone directions ({len(h1_mesh.faces)} faces)...")
        h1_all_directions = eval_func(
            h1_mesh,  # Use ORIGINAL mesh for silicone analysis
            n_candidate_directions,
            tilt_angle_deg,
            area_threshold_mm2,
            progress_callback=h1_progress_callback
        )
        h1_best_str = f"{h1_all_directions[0].total_score:.2f}" if h1_all_directions else "N/A"
        logger.info(f"H1: evaluated {len(h1_all_directions)} directions, best score={h1_best_str}")
    else:
        logger.warning("No H1 faces found")
    
    # Step 5: Evaluate candidate directions for H2 (if available)
    h2_all_directions = []
    
    if h2_mesh is not None and len(h2_mesh.faces) > 0:
        logger.info(f"Computing H2 silicone directions ({len(h2_mesh.faces)} faces)...")
        h2_all_directions = eval_func(
            h2_mesh,  # Use ORIGINAL mesh for silicone analysis
            n_candidate_directions,
            tilt_angle_deg,
            area_threshold_mm2,
            progress_callback=h2_progress_callback
        )
        h2_best_str = f"{h2_all_directions[0].total_score:.2f}" if h2_all_directions else "N/A"
        logger.info(f"H2 cavity: evaluated {len(h2_all_directions)} directions, best score={h2_best_str}")
    else:
        logger.warning("No H2 faces found")
    
    # Step 6: Select ALIGNED silicone directions across both halves
    # Paper: "choose from those scoring lowest a couple of nearly aligned directions f1, f2"
    # 
    # IMPORTANT: H1 and H2 are poured in OPPOSITE orientations:
    # - When pouring H1, the part sits with parting surface facing DOWN
    # - When pouring H2, the part is FLIPPED (parting surface faces UP)
    # 
    # Therefore, optimal directions for H1 and H2 are typically OPPOSITE (anti-aligned).
    # We accept both aligned and anti-aligned pairs. For anti-aligned pairs, we negate
    # f2 to represent the effective pouring direction in H2's flipped coordinate frame.
    logger.info("Selecting aligned silicone directions across both mold halves...")
    
    h1_direction = np.array([0.0, 0.0, 1.0])  # Default
    h1_score = 0.0
    h2_direction = np.array([0.0, 0.0, -1.0])  # Default (opposite)
    h2_score = 0.0
    alignment_angle_deg = 180.0
    well_aligned = False
    is_anti_aligned = False
    
    if h1_all_directions and h2_all_directions:
        # Use joint selection to find aligned or anti-aligned pair
        # Use more candidates for better chance of finding good directions
        h1_best, h2_best, alignment_angle_deg, is_anti_aligned = select_aligned_silicone_directions_across_halves(
            h1_all_directions,
            h2_all_directions,
            max_alignment_angle_deg=30.0,
            n_candidates=min(30, len(h1_all_directions), len(h2_all_directions))
        )
        h1_direction = h1_best.direction.copy()
        h1_score = h1_best.total_score
        h2_direction = h2_best.direction.copy()
        h2_score = h2_best.total_score
        
        # If anti-aligned, negate f2 to get the effective direction in H2's flipped frame
        # This ensures the resin bisector calculation works correctly
        if is_anti_aligned:
            h2_direction = -h2_direction
            logger.info(f"H2 direction negated for anti-aligned pair (H2 is poured in flipped orientation)")
        
        well_aligned = alignment_angle_deg <= 30.0
    elif h1_all_directions:
        # Only H1 available
        h1_direction = h1_all_directions[0].direction.copy()
        h1_score = h1_all_directions[0].total_score
        logger.warning("Only H1 directions available, cannot compute alignment")
    elif h2_all_directions:
        # Only H2 available
        h2_direction = h2_all_directions[0].direction.copy()
        h2_score = h2_all_directions[0].total_score
        logger.warning("Only H2 directions available, cannot compute alignment")
    else:
        logger.warning("No direction results available, using defaults")
    
    # Step 7: Find optimal resin direction for the complete part
    #
    # RESIN POURING ANALYSIS:
    # Resin is poured into the assembled silicone mold (part has been removed).
    # The inner surface of the silicone mold is the NEGATIVE of the part surface:
    # - What was an outward-facing part surface is now the inward-facing mold ceiling
    # - Bubbles rise and trap against faces that originally pointed UP on the part
    #
    # We use the INVERTED part mesh to simulate the mold cavity:
    # - Original face normal · up > 0 (part face up) → inverted normal · up < 0 → ceiling pocket
    #
    # Paper: "The resin casting direction is then chosen as the lowest scoring one
    #         in a set of directions sampled in a small cone around the bisector of f1, f2"
    logger.info("Computing resin direction (using INVERTED mesh for cavity ceiling analysis)...")
    
    inverted_part_mesh = invert_mesh_for_cavity_analysis(part_mesh)
    
    # Sample directions in a cone around the bisector of f1, f2
    # Paper: "opening angle of 10° in our experiments" -> half-angle = 5°
    resin_result = select_resin_direction(
        inverted_part_mesh,  # Use INVERTED mesh - resin fills the mold cavity
        h1_direction,
        h2_direction,
        cone_half_angle_deg=5.0,  # 10° opening angle per paper
        n_samples_in_cone=16,
        tilt_angle_deg=tilt_angle_deg,
        area_threshold_mm2=area_threshold_mm2
    )
    
    # Extract maxima positions for visualization
    # These are the bubble-trapping local maxima on the mold cavity ceiling
    # Note: We use the ORIGINAL part_mesh vertices (not inverted) since
    # the vertex indices from the inverted mesh map to the same positions
    resin_maxima_positions = None
    if resin_result.pairs:
        maxima_indices = [pair.maximum_idx for pair in resin_result.pairs]
        # Get unique indices
        unique_indices = list(set(maxima_indices))
        resin_maxima_positions = part_mesh.vertices[unique_indices].copy()
        logger.info(f"Found {len(unique_indices)} bubble-trapping maxima for resin direction")
    
    # Find the TRUE global maximum (highest point of the ENTIRE mesh in resin direction)
    # This is the absolute highest point when the part is oriented with resin direction up
    resin_dir = resin_result.direction
    all_heights = part_mesh.vertices @ resin_dir
    global_max_vertex_idx = np.argmax(all_heights)
    resin_global_maximum_position = part_mesh.vertices[global_max_vertex_idx].copy()
    logger.info(f"Global maximum vertex {global_max_vertex_idx} at height {all_heights[global_max_vertex_idx]:.2f}: {resin_global_maximum_position}")
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Mold-aware pouring direction optimization complete in {elapsed:.1f} ms")
    logger.info(f"  H1: direction={h1_direction}, score={h1_score:.2f}, faces={len(h1_faces)}")
    logger.info(f"  H2: direction={h2_direction}, score={h2_score:.2f}, faces={len(h2_faces)}")
    logger.info(f"  Resin: direction={resin_result.direction}, score={resin_result.total_score:.2f}")
    logger.info(f"  Alignment: {alignment_angle_deg:.1f}° (well-aligned: {well_aligned})")
    
    # Diagnostic: show angle between pouring directions and parting surface
    if parting_direction is not None:
        parting_dir = np.asarray(parting_direction, dtype=np.float64)
        parting_dir = parting_dir / np.linalg.norm(parting_dir)
        
        # Angle from parting direction (normal to parting surface)
        h1_angle_from_parting = np.degrees(np.arccos(np.clip(np.abs(np.dot(h1_direction, parting_dir)), 0, 1)))
        h2_angle_from_parting = np.degrees(np.arccos(np.clip(np.abs(np.dot(h2_direction, parting_dir)), 0, 1)))
        resin_angle_from_parting = np.degrees(np.arccos(np.clip(np.abs(np.dot(resin_result.direction, parting_dir)), 0, 1)))
        
        # Angle from parting SURFACE (complementary angle)
        h1_angle_from_surface = 90.0 - h1_angle_from_parting
        h2_angle_from_surface = 90.0 - h2_angle_from_parting
        resin_angle_from_surface = 90.0 - resin_angle_from_parting
        
        logger.info(f"  Angles from parting DIRECTION: H1={h1_angle_from_parting:.1f}°, H2={h2_angle_from_parting:.1f}°, Resin={resin_angle_from_parting:.1f}°")
        logger.info(f"  Angles from parting SURFACE: H1={h1_angle_from_surface:.1f}°, H2={h2_angle_from_surface:.1f}°, Resin={resin_angle_from_surface:.1f}°")
        
        # Log the top 5 directions for each mold half with their angles
        logger.info("  Top 5 H1 candidates with parting angles:")
        for i, r in enumerate(h1_all_directions[:5]):
            angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(r.direction, parting_dir)), 0, 1)))
            surface_angle = 90.0 - angle
            logger.info(f"    {i+1}. score={r.total_score:.2f}, surface_angle={surface_angle:.1f} deg, dir={r.direction}")
        
        logger.info("  Top 5 H2 candidates with parting angles:")
        for i, r in enumerate(h2_all_directions[:5]):
            angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(r.direction, parting_dir)), 0, 1)))
            surface_angle = 90.0 - angle
            logger.info(f"    {i+1}. score={r.total_score:.2f}, surface_angle={surface_angle:.1f} deg, dir={r.direction}")
    
    return MoldAwarePouringDirections(
        h1_silicone_direction=h1_direction,
        h1_silicone_score=h1_score,
        h1_face_count=len(h1_faces),
        h1_all_directions=h1_all_directions,
        h2_silicone_direction=h2_direction,
        h2_silicone_score=h2_score,
        h2_face_count=len(h2_faces),
        h2_all_directions=h2_all_directions,
        resin_direction=resin_result.direction,
        resin_score=resin_result.total_score,
        resin_maxima_positions=resin_maxima_positions,
        resin_global_maximum_position=resin_global_maximum_position,
        silicone_alignment_angle_deg=alignment_angle_deg,
        directions_well_aligned=well_aligned,
        is_anti_aligned=is_anti_aligned,
        h1_mesh=h1_mesh,
        h2_mesh=h2_mesh,
        computation_time_ms=elapsed
    )
