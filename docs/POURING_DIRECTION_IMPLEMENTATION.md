# Pouring Direction Optimization Implementation Plan

## Overview

This document outlines the implementation plan for determining optimal silicone and resin pouring directions that minimize air bubble trapping during mold casting. The algorithm uses **persistence homology** to identify and score local maxima where air bubbles could become trapped.

### Problem Statement

Air bubbles may get trapped at local maxima relative to the pouring direction. While slight tilting during casting can help some bubbles escape, others remain trapped depending on the local geometry. We need a criterion to select pouring directions that minimize the presence of relevant (bubble-trapping) local maxima.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Height Function** | `f: M  ℝ` — projects each vertex onto the pouring direction |
| **Persistence Pairing** | Pairs critical points (max, saddle) representing bubble-trapping features |
| **Persistence** | `\|f(m) - f(n)\|` — traditional measure of feature relevance |
| **Relevance Score** | Area of trapped air region, accounting for tilt angle α |
| **Superlevel Set** | `Mₐ = f(a, +)` — all points above height `a` |

---

## Architecture

```
─
                     POURING DIRECTION PIPELINE                   

                                                                  
             
   Height Field  Critical Pt   Persistence         
   Computation       Detection         Pairing             
             
                                                                 
             
   Direction     Score         Bubble Relevance    
   Selection         Aggregation       Scoring             
             
                                                                  

```

---

## Implementation Steps

### Phase 1: Height Function Computation

- [ ] **Step 1.1: Add height field function to `mesh_analysis.py`**

  Location: `desktop_app/core/mesh_analysis.py`

  ```python
  def compute_height_field(mesh: trimesh.Trimesh, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
      """
      Compute height function h(v) = dot(v, direction) for all vertices.
      
      Args:
          mesh: Input mesh
          direction: Unit vector representing pouring direction (points "up")
          
      Returns:
          vertex_heights: (N,) array of vertex height values
          face_heights: (M,) array of face height values (average of vertices)
      """
      direction = direction / np.linalg.norm(direction)
      vertex_heights = mesh.vertices @ direction
      
      # Face heights as average of vertex heights
      face_vertex_heights = vertex_heights[mesh.faces]  # (M, 3)
      face_heights = face_vertex_heights.mean(axis=1)   # (M,)
      
      return vertex_heights, face_heights
  ```

- [ ] **Step 1.2: Add vertex adjacency computation**

  Location: `desktop_app/core/mesh_analysis.py`

  ```python
  def build_vertex_neighbors(mesh: trimesh.Trimesh) -> Dict[int, Set[int]]:
      """
      Build adjacency map: vertex_index -> set of neighboring vertex indices.
      Uses edge connectivity from mesh.
      """
      neighbors = defaultdict(set)
      for edge in mesh.edges:
          neighbors[edge[0]].add(edge[1])
          neighbors[edge[1]].add(edge[0])
      return dict(neighbors)
  ```

---

### Phase 2: Critical Point Detection

- [ ] **Step 2.1: Create new module `pouring_direction.py`**

  Location: `desktop_app/core/pouring_direction.py`

  Create the module with necessary imports and dataclasses:

  ```python
  """
  Pouring Direction Optimization Module
  
  Determines optimal silicone and resin pouring directions by minimizing
  air bubble trapping using persistence homology analysis.
  
  Based on: "Persistence pairing for mold design" concepts from
  Edelsbrunner et al. 2000 and Morse theory (Milnor 1963).
  """
  
  import numpy as np
  from dataclasses import dataclass, field
  from typing import List, Tuple, Dict, Set, Optional
  from collections import defaultdict
  import trimesh
  
  from .mesh_analysis import compute_height_field, build_vertex_neighbors, build_face_neighbors
  from .parting_direction import fibonacci_sphere
  ```

- [ ] **Step 2.2: Define data structures for critical points**

  ```python
  @dataclass
  class CriticalPoint:
      """Represents a critical point in the height function."""
      index: int                    # Vertex index
      height: float                 # Height value f(v)
      point_type: str              # 'maximum', 'minimum', or 'saddle'
      
  @dataclass
  class PersistencePair:
      """A (maximum, saddle) pair from persistence homology."""
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
  ```

- [ ] **Step 2.3: Implement vertex classification**

  Classify each vertex as maximum, minimum, saddle, or regular based on its neighbors:

  ```python
  def classify_vertex(
      vertex_idx: int,
      vertex_heights: np.ndarray,
      neighbors: Dict[int, Set[int]]
  ) -> Optional[str]:
      """
      Classify a vertex as a critical point based on height comparison with neighbors.
      
      Uses lower link analysis:
      - Maximum: all neighbors have lower height
      - Minimum: all neighbors have higher height  
      - Saddle: neighbors form multiple connected components in lower link
      - Regular: single connected component of lower neighbors
      
      Returns:
          'maximum', 'minimum', 'saddle', or None (regular point)
      """
      h = vertex_heights[vertex_idx]
      neighbor_indices = neighbors.get(vertex_idx, set())
      
      if not neighbor_indices:
          return None
      
      lower_neighbors = [n for n in neighbor_indices if vertex_heights[n] < h]
      upper_neighbors = [n for n in neighbor_indices if vertex_heights[n] > h]
      
      # Maximum: no lower neighbors (all neighbors are higher or equal)
      # Actually for maximum: all neighbors should be LOWER
      if len(upper_neighbors) == 0 and len(lower_neighbors) == len(neighbor_indices):
          return 'maximum'
      
      # Minimum: all neighbors are higher
      if len(lower_neighbors) == 0 and len(upper_neighbors) == len(neighbor_indices):
          return 'minimum'
      
      # For saddle detection, check connectivity of lower link
      if lower_neighbors:
          components = count_connected_components(lower_neighbors, neighbors)
          if components > 1:
              return 'saddle'
      
      return None  # Regular point

  def count_connected_components(
      vertices: List[int],
      neighbors: Dict[int, Set[int]]
  ) -> int:
      """Count connected components among a subset of vertices."""
      if not vertices:
          return 0
          
      vertex_set = set(vertices)
      visited = set()
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
  ```

- [ ] **Step 2.4: Find all critical points for a direction**

  ```python
  def find_critical_points(
      mesh: trimesh.Trimesh,
      direction: np.ndarray,
      vertex_neighbors: Dict[int, Set[int]] = None
  ) -> Tuple[List[CriticalPoint], np.ndarray]:
      """
      Find all critical points of the height function f(v) = dot(v, direction).
      
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
  ```

---

### Phase 3: Persistence Pairing

- [ ] **Step 3.1: Implement Union-Find data structure**

  Union-Find is essential for tracking connected components efficiently:

  ```python
  class UnionFind:
      """Union-Find (Disjoint Set Union) data structure for component tracking."""
      
      def __init__(self, n: int):
          self.parent = list(range(n))
          self.rank = [0] * n
          self.component_max = list(range(n))  # Track maximum vertex per component
          
      def find(self, x: int) -> int:
          """Find root with path compression."""
          if self.parent[x] != x:
              self.parent[x] = self.find(self.parent[x])
          return self.parent[x]
      
      def union(self, x: int, y: int) -> Tuple[int, int]:
          """
          Union two components. Returns (kept_root, merged_root).
          The component with higher maximum is kept.
          """
          root_x = self.find(x)
          root_y = self.find(y)
          
          if root_x == root_y:
              return (root_x, root_x)
          
          # Union by rank
          if self.rank[root_x] < self.rank[root_y]:
              root_x, root_y = root_y, root_x
          
          self.parent[root_y] = root_x
          if self.rank[root_x] == self.rank[root_y]:
              self.rank[root_x] += 1
          
          # Update component max
          if self.component_max[root_y] > self.component_max[root_x]:
              self.component_max[root_x] = self.component_max[root_y]
          
          return (root_x, root_y)
      
      def get_component_max(self, x: int) -> int:
          """Get the maximum vertex index in the component containing x."""
          return self.component_max[self.find(x)]
  ```

- [ ] **Step 3.2: Implement persistence pairing algorithm**

  Sweep through superlevel sets from top to bottom, tracking component births (maxima) and deaths (saddles):

  ```python
  def compute_persistence_pairs(
      mesh: trimesh.Trimesh,
      direction: np.ndarray,
      vertex_neighbors: Dict[int, Set[int]] = None
  ) -> Tuple[List[PersistencePair], np.ndarray]:
      """
      Compute persistence pairs (maximum, saddle) using superlevel set filtration.
      
      Algorithm:
      1. Sort vertices by decreasing height
      2. Process vertices from highest to lowest
      3. When processing a vertex v:
         - If v has no processed neighbors: v is a maximum (birth)
         - If v connects multiple components: v is a saddle (death of younger component)
         - Otherwise: v joins existing component
      
      Returns:
          pairs: List of (maximum, saddle) PersistencePair objects
          vertex_heights: Height values for all vertices
      """
      vertex_heights, _ = compute_height_field(mesh, direction)
      
      if vertex_neighbors is None:
          vertex_neighbors = build_vertex_neighbors(mesh)
      
      n_vertices = len(mesh.vertices)
      
      # Sort vertices by decreasing height
      sorted_indices = np.argsort(-vertex_heights)
      
      # Map from vertex index to its position in sorted order
      vertex_to_order = np.zeros(n_vertices, dtype=np.int32)
      vertex_to_order[sorted_indices] = np.arange(n_vertices)
      
      # Union-Find to track components
      uf = UnionFind(n_vertices)
      
      # Track which vertices have been processed
      processed = np.zeros(n_vertices, dtype=bool)
      
      # Track birth vertex (maximum) for each component
      component_birth = {}  # root -> maximum vertex index
      
      pairs = []
      
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
              component_birth[v] = v
          else:
              # Find unique components among neighbors
              neighbor_roots = set(uf.find(n) for n in processed_neighbors)
              
              if len(neighbor_roots) == 1:
                  # v joins existing component
                  root = neighbor_roots.pop()
                  uf.union(v, root)
                  processed[v] = True
              else:
                  # v is a saddle - merges multiple components
                  # Sort roots by birth height (descending) - oldest survives
                  roots_by_birth = sorted(
                      neighbor_roots,
                      key=lambda r: vertex_heights[component_birth[r]],
                      reverse=True
                  )
                  
                  # Oldest (highest) component survives
                  survivor_root = roots_by_birth[0]
                  
                  # Other components die - create persistence pairs
                  for dying_root in roots_by_birth[1:]:
                      max_vertex = component_birth[dying_root]
                      pairs.append(PersistencePair(
                          maximum_idx=max_vertex,
                          saddle_idx=v,
                          maximum_height=vertex_heights[max_vertex],
                          saddle_height=h_v,
                          persistence=vertex_heights[max_vertex] - h_v
                      ))
                  
                  # Merge all into survivor
                  for root in roots_by_birth[1:]:
                      uf.union(survivor_root, root)
                      del component_birth[root]
                  
                  # v joins the merged component
                  uf.union(v, survivor_root)
                  processed[v] = True
                  
                  # Update survivor's root if changed
                  new_root = uf.find(survivor_root)
                  if new_root != survivor_root:
                      component_birth[new_root] = component_birth.pop(survivor_root)
      
      return pairs, vertex_heights
  ```

---

### Phase 4: Bubble Relevance Scoring

- [ ] **Step 4.1: Implement trapped region growing**

  Grow region from maximum down to saddle height:

  ```python
  def grow_trapped_region(
      mesh: trimesh.Trimesh,
      maximum_idx: int,
      saddle_height: float,
      vertex_heights: np.ndarray,
      face_neighbors: Dict[int, List[int]] = None
  ) -> Set[int]:
      """
      Grow region A_m^n from maximum m to saddle height n.
      
      Returns set of face indices in the trapped region.
      """
      if face_neighbors is None:
          face_neighbors = build_face_neighbors(mesh)
      
      # Get face heights
      face_vertex_heights = vertex_heights[mesh.faces]
      face_heights = face_vertex_heights.mean(axis=1)
      
      # Find seed face: face containing maximum vertex with highest centroid
      max_vertex_faces = np.where(np.any(mesh.faces == maximum_idx, axis=1))[0]
      if len(max_vertex_faces) == 0:
          return set()
      
      seed_face = max_vertex_faces[np.argmax(face_heights[max_vertex_faces])]
      
      # BFS to grow region above saddle height
      trapped_faces = set()
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
  ```

- [ ] **Step 4.2: Implement tiltable region computation**

  Compute faces that can drain with tilting:

  ```python
  def compute_tiltable_region(
      mesh: trimesh.Trimesh,
      direction: np.ndarray,
      trapped_faces: Set[int],
      saddle_idx: int,
      tilt_angle_deg: float = 10.0,
      face_neighbors: Dict[int, List[int]] = None
  ) -> Set[int]:
      """
      Compute region T of faces that can drain with tilting.
      
      Grows from saddle, including faces whose normal is within
      tilt_angle of the pouring direction (can flow out).
      
      Args:
          mesh: Input mesh
          direction: Pouring direction (unit vector)
          trapped_faces: Set of face indices in A_m^n
          saddle_idx: Vertex index of saddle point
          tilt_angle_deg: Maximum tilt angle in degrees
          
      Returns:
          Set of face indices that can drain (T)
      """
      if face_neighbors is None:
          face_neighbors = build_face_neighbors(mesh)
      
      tilt_cos = np.cos(np.radians(tilt_angle_deg))
      normals = mesh.face_normals
      
      # Find seed faces containing saddle vertex
      saddle_faces = set(np.where(np.any(mesh.faces == saddle_idx, axis=1))[0])
      seed_faces = saddle_faces & trapped_faces
      
      if not seed_faces:
          return set()
      
      # BFS from saddle faces, only through tiltable faces
      tiltable = set()
      queue = list(seed_faces)
      visited = set(seed_faces)
      
      while queue:
          face_idx = queue.pop(0)
          
          # Face is tiltable if normal aligns with direction (within tilt angle)
          # dot(normal, direction) > cos(90 - tilt_angle)  sin(tilt_angle)
          # For small angles: face can drain if it's somewhat upward-facing
          normal_dot = np.dot(normals[face_idx], direction)
          
          # Face can drain if tilting would let air escape
          # This means the face normal should be somewhat aligned with gravity
          if normal_dot > -tilt_cos:  # Not facing too far down
              tiltable.add(face_idx)
              
              # Propagate to neighbors in trapped region
              for neighbor in face_neighbors.get(face_idx, []):
                  if neighbor in trapped_faces and neighbor not in visited:
                      visited.add(neighbor)
                      queue.append(neighbor)
      
      return tiltable
  ```

- [ ] **Step 4.3: Compute relevance score for a pair**

  ```python
  def compute_pair_relevance(
      mesh: trimesh.Trimesh,
      pair: PersistencePair,
      direction: np.ndarray,
      vertex_heights: np.ndarray,
      tilt_angle_deg: float = 10.0,
      face_neighbors: Dict[int, List[int]] = None
  ) -> PersistencePair:
      """
      Compute relevance score for a persistence pair.
      
      Score = area of (A_m^n - T), the trapped region minus tiltable region.
      
      Returns:
          Updated PersistencePair with relevance_score and trapped_faces set
      """
      if face_neighbors is None:
          face_neighbors = build_face_neighbors(mesh)
      
      # Grow trapped region from maximum to saddle
      trapped = grow_trapped_region(
          mesh, pair.maximum_idx, pair.saddle_height,
          vertex_heights, face_neighbors
      )
      
      # Compute tiltable region
      tiltable = compute_tiltable_region(
          mesh, direction, trapped, pair.saddle_idx,
          tilt_angle_deg, face_neighbors
      )
      
      # Remaining trapped region
      final_trapped = trapped - tiltable
      
      # Compute area
      face_areas = mesh.area_faces
      score = sum(face_areas[f] for f in final_trapped)
      
      pair.trapped_faces = final_trapped
      pair.relevance_score = score
      
      return pair
  ```

---

### Phase 5: Direction Scoring and Selection

- [ ] **Step 5.1: Score a single direction**

  ```python
  def score_pouring_direction(
      mesh: trimesh.Trimesh,
      direction: np.ndarray,
      tilt_angle_deg: float = 10.0,
      area_threshold_mm2: float = 0.5,
      vertex_neighbors: Dict[int, Set[int]] = None,
      face_neighbors: Dict[int, List[int]] = None
  ) -> PouringDirectionResult:
      """
      Compute bubble-trapping score for a candidate pouring direction.
      
      Args:
          mesh: Input mesh
          direction: Candidate pouring direction (unit vector)
          tilt_angle_deg: Assumed tilt capability during casting
          area_threshold_mm2: Filter out pairs below this area
          
      Returns:
          PouringDirectionResult with pairs and total score
      """
      direction = direction / np.linalg.norm(direction)
      
      # Precompute adjacency
      if vertex_neighbors is None:
          vertex_neighbors = build_vertex_neighbors(mesh)
      if face_neighbors is None:
          face_neighbors = build_face_neighbors(mesh)
      
      # Get persistence pairs
      pairs, vertex_heights = compute_persistence_pairs(
          mesh, direction, vertex_neighbors
      )
      
      # Compute relevance for each pair
      relevant_pairs = []
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
          direction=direction,
          pairs=relevant_pairs,
          total_score=total_score,
          relevant_pair_count=len(relevant_pairs)
      )
  ```

- [ ] **Step 5.2: Evaluate all candidate directions**

  ```python
  def evaluate_candidate_directions(
      mesh: trimesh.Trimesh,
      n_directions: int = 64,
      tilt_angle_deg: float = 10.0,
      area_threshold_mm2: float = 0.5,
      directions: np.ndarray = None
  ) -> List[PouringDirectionResult]:
      """
      Evaluate multiple candidate directions and return sorted by score.
      
      Args:
          mesh: Input mesh
          n_directions: Number of directions to sample (if directions not provided)
          tilt_angle_deg: Assumed tilt capability
          area_threshold_mm2: Filter threshold
          directions: Optional pre-specified directions
          
      Returns:
          List of PouringDirectionResult, sorted by total_score (ascending)
      """
      if directions is None:
          directions = fibonacci_sphere(n_directions)
      
      # Precompute adjacency once
      vertex_neighbors = build_vertex_neighbors(mesh)
      face_neighbors = build_face_neighbors(mesh)
      
      results = []
      for direction in directions:
          result = score_pouring_direction(
              mesh, direction, tilt_angle_deg, area_threshold_mm2,
              vertex_neighbors, face_neighbors
          )
          results.append(result)
      
      # Sort by score (lower is better)
      results.sort(key=lambda r: r.total_score)
      
      return results
  ```

---

### Phase 6: Silicone and Resin Direction Selection

- [ ] **Step 6.1: Select aligned silicone directions**

  ```python
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
      
      Args:
          scored_results: Results sorted by score (ascending)
          max_alignment_angle_deg: Maximum angle between f1 and f2
          n_candidates: Number of top candidates to consider
          
      Returns:
          (f1, f2): Two direction vectors for silicone pouring
      """
      candidates = scored_results[:n_candidates]
      
      min_cos = np.cos(np.radians(max_alignment_angle_deg))
      
      best_pair = None
      best_combined_score = float('inf')
      
      for i, r1 in enumerate(candidates):
          for r2 in candidates[i+1:]:
              # Check alignment
              alignment = np.dot(r1.direction, r2.direction)
              
              if alignment >= min_cos:  # Nearly aligned
                  combined = r1.total_score + r2.total_score
                  
                  if combined < best_combined_score:
                      best_combined_score = combined
                      best_pair = (r1.direction, r2.direction)
      
      if best_pair is None:
          # Fallback: just use two best
          return (candidates[0].direction, candidates[1].direction)
      
      return best_pair
  ```

- [ ] **Step 6.2: Select resin direction in cone around bisector**

  ```python
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
      in a small cone around the bisector of f1 and f2.
      
      Args:
          mesh: Input mesh
          f1, f2: Silicone pouring directions
          cone_half_angle_deg: Half-angle of sampling cone (10 as per paper)
          n_samples_in_cone: Number of directions to sample in cone
          
      Returns:
          Best PouringDirectionResult for resin
      """
      # Compute bisector
      bisector = (f1 + f2) / 2
      bisector = bisector / np.linalg.norm(bisector)
      
      # Sample directions in cone around bisector
      cone_directions = sample_cone_directions(
          bisector, cone_half_angle_deg, n_samples_in_cone
      )
      
      # Include bisector itself
      all_directions = np.vstack([bisector, cone_directions])
      
      # Score all
      results = evaluate_candidate_directions(
          mesh,
          directions=all_directions,
          tilt_angle_deg=tilt_angle_deg,
          area_threshold_mm2=area_threshold_mm2
      )
      
      return results[0]  # Best scoring

  def sample_cone_directions(
      axis: np.ndarray,
      half_angle_deg: float,
      n_samples: int
  ) -> np.ndarray:
      """
      Sample directions uniformly in a cone around an axis.
      """
      half_angle_rad = np.radians(half_angle_deg)
      
      # Create orthonormal basis with axis as z
      if abs(axis[0]) < 0.9:
          perp = np.cross(axis, [1, 0, 0])
      else:
          perp = np.cross(axis, [0, 1, 0])
      perp = perp / np.linalg.norm(perp)
      perp2 = np.cross(axis, perp)
      
      directions = []
      for i in range(n_samples):
          # Random point in cone
          phi = 2 * np.pi * i / n_samples
          theta = half_angle_rad * np.sqrt(np.random.random())  # Uniform in solid angle
          
          # Convert to Cartesian in local frame
          x = np.sin(theta) * np.cos(phi)
          y = np.sin(theta) * np.sin(phi)
          z = np.cos(theta)
          
          # Transform to global frame
          direction = x * perp + y * perp2 + z * axis
          directions.append(direction)
      
      return np.array(directions)
  ```

---

### Phase 7: Main API and Integration

- [ ] **Step 7.1: Create main entry point**

  ```python
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
      
  def find_optimal_pouring_directions(
      mesh: trimesh.Trimesh,
      n_candidate_directions: int = 64,
      tilt_angle_deg: float = 10.0,
      area_threshold_mm2: float = 0.5,
      alignment_angle_deg: float = 30.0,
      resin_cone_angle_deg: float = 10.0
  ) -> OptimalPouringDirections:
      """
      Find optimal silicone and resin pouring directions.
      
      Main API function that:
      1. Samples candidate directions
      2. Scores each direction for bubble trapping
      3. Selects two nearly-aligned low-scoring silicone directions
      4. Selects optimal resin direction in cone around their bisector
      
      Args:
          mesh: Input part mesh
          n_candidate_directions: Number of directions to sample
          tilt_angle_deg: Assumed tilt capability during casting
          area_threshold_mm2: Ignore trapped regions smaller than this
          alignment_angle_deg: Max angle between silicone directions
          resin_cone_angle_deg: Cone angle for resin direction sampling
          
      Returns:
          OptimalPouringDirections with all results
      """
      # Step 1: Evaluate all candidate directions
      print(f"Evaluating {n_candidate_directions} candidate directions...")
      all_results = evaluate_candidate_directions(
          mesh, n_candidate_directions, tilt_angle_deg, area_threshold_mm2
      )
      
      # Step 2: Select silicone directions
      print("Selecting silicone pouring directions...")
      f1, f2 = select_silicone_directions(
          all_results, alignment_angle_deg
      )
      
      # Find scores for selected directions
      f1_result = next(r for r in all_results if np.allclose(r.direction, f1))
      f2_result = next(r for r in all_results if np.allclose(r.direction, f2))
      
      # Step 3: Select resin direction
      print("Selecting resin pouring direction...")
      resin_result = select_resin_direction(
          mesh, f1, f2, resin_cone_angle_deg,
          tilt_angle_deg=tilt_angle_deg,
          area_threshold_mm2=area_threshold_mm2
      )
      
      return OptimalPouringDirections(
          silicone_direction_1=f1,
          silicone_direction_2=f2,
          silicone_score_1=f1_result.total_score,
          silicone_score_2=f2_result.total_score,
          resin_direction=resin_result.direction,
          resin_score=resin_result.total_score,
          all_scored_directions=all_results
      )
  ```

- [ ] **Step 7.2: Add to module exports**

  Update `desktop_app/core/__init__.py`:

  ```python
  from .pouring_direction import (
      find_optimal_pouring_directions,
      OptimalPouringDirections,
      PouringDirectionResult,
      PersistencePair,
      evaluate_candidate_directions,
      score_pouring_direction
  )
  ```

---

### Phase 8: UI Integration

- [ ] **Step 8.1: Add pouring direction controls to main window**

  Add button and display in `desktop_app/ui/main_window.py`:

  ```python
  def _create_pouring_direction_ui(self):
      """Create UI elements for pouring direction optimization."""
      group = QGroupBox("Pouring Direction")
      layout = QVBoxLayout()
      
      # Parameters
      param_layout = QFormLayout()
      
      self.tilt_angle_spin = QDoubleSpinBox()
      self.tilt_angle_spin.setRange(0.0, 45.0)
      self.tilt_angle_spin.setValue(10.0)
      self.tilt_angle_spin.setSuffix("")
      param_layout.addRow("Tilt Angle:", self.tilt_angle_spin)
      
      self.area_threshold_spin = QDoubleSpinBox()
      self.area_threshold_spin.setRange(0.01, 10.0)
      self.area_threshold_spin.setValue(0.5)
      self.area_threshold_spin.setSuffix(" mm")
      param_layout.addRow("Area Threshold:", self.area_threshold_spin)
      
      layout.addLayout(param_layout)
      
      # Compute button
      self.compute_pouring_btn = QPushButton("Find Optimal Directions")
      self.compute_pouring_btn.clicked.connect(self._on_compute_pouring)
      layout.addWidget(self.compute_pouring_btn)
      
      # Results display
      self.pouring_results_label = QLabel("No results yet")
      layout.addWidget(self.pouring_results_label)
      
      group.setLayout(layout)
      return group
  ```

- [ ] **Step 8.2: Implement computation callback**

  ```python
  def _on_compute_pouring(self):
      """Handle pouring direction computation."""
      if self.mesh is None:
          QMessageBox.warning(self, "Warning", "Please load a mesh first.")
          return
      
      tilt_angle = self.tilt_angle_spin.value()
      area_threshold = self.area_threshold_spin.value()
      
      self.compute_pouring_btn.setEnabled(False)
      self.pouring_results_label.setText("Computing...")
      
      try:
          from core.pouring_direction import find_optimal_pouring_directions
          
          result = find_optimal_pouring_directions(
              self.mesh,
              tilt_angle_deg=tilt_angle,
              area_threshold_mm2=area_threshold
          )
          
          # Display results
          text = (
              f"Silicone Direction 1: [{result.silicone_direction_1[0]:.3f}, "
              f"{result.silicone_direction_1[1]:.3f}, {result.silicone_direction_1[2]:.3f}]\n"
              f"  Score: {result.silicone_score_1:.2f} mm\n\n"
              f"Silicone Direction 2: [{result.silicone_direction_2[0]:.3f}, "
              f"{result.silicone_direction_2[1]:.3f}, {result.silicone_direction_2[2]:.3f}]\n"
              f"  Score: {result.silicone_score_2:.2f} mm\n\n"
              f"Resin Direction: [{result.resin_direction[0]:.3f}, "
              f"{result.resin_direction[1]:.3f}, {result.resin_direction[2]:.3f}]\n"
              f"  Score: {result.resin_score:.2f} mm"
          )
          self.pouring_results_label.setText(text)
          
          # Visualize direction
          self.viewer.show_pouring_directions(result)
          
      except Exception as e:
          QMessageBox.critical(self, "Error", f"Computation failed: {str(e)}")
      finally:
          self.compute_pouring_btn.setEnabled(True)
  ```

---

### Phase 9: Visualization

- [ ] **Step 9.1: Add direction arrows to mesh viewer**

  Update `desktop_app/viewer/mesh_viewer.py`:

  ```python
  def show_pouring_directions(self, result: 'OptimalPouringDirections'):
      """Visualize pouring directions as arrows."""
      import pyvista as pv
      
      # Clear previous arrows
      self.plotter.remove_actor('silicone_arrow_1', reset_camera=False)
      self.plotter.remove_actor('silicone_arrow_2', reset_camera=False)
      self.plotter.remove_actor('resin_arrow', reset_camera=False)
      
      # Get mesh center and scale
      center = self.mesh.centroid
      scale = self.mesh.extents.max() * 0.5
      
      # Silicone direction 1 (blue)
      arrow1 = pv.Arrow(
          start=center - result.silicone_direction_1 * scale,
          direction=result.silicone_direction_1,
          scale=scale * 2
      )
      self.plotter.add_mesh(arrow1, color='blue', name='silicone_arrow_1')
      
      # Silicone direction 2 (cyan)
      arrow2 = pv.Arrow(
          start=center - result.silicone_direction_2 * scale,
          direction=result.silicone_direction_2,
          scale=scale * 2
      )
      self.plotter.add_mesh(arrow2, color='cyan', name='silicone_arrow_2')
      
      # Resin direction (red)
      arrow3 = pv.Arrow(
          start=center - result.resin_direction * scale,
          direction=result.resin_direction,
          scale=scale * 2
      )
      self.plotter.add_mesh(arrow3, color='red', name='resin_arrow')
  ```

- [ ] **Step 9.2: Add trapped region highlighting**

  ```python
  def highlight_trapped_regions(
      self, 
      pairs: List['PersistencePair'],
      mesh: 'trimesh.Trimesh'
  ):
      """Highlight faces where air bubbles would be trapped."""
      # Collect all trapped faces
      all_trapped = set()
      for pair in pairs:
          all_trapped.update(pair.trapped_faces)
      
      # Create face coloring
      face_colors = np.ones((len(mesh.faces), 4)) * [0.8, 0.8, 0.8, 1.0]  # Default gray
      for face_idx in all_trapped:
          face_colors[face_idx] = [1.0, 0.3, 0.3, 1.0]  # Red for trapped
      
      # Update mesh visualization
      # (Implementation depends on visualization library)
  ```

---

### Phase 10: Testing and Validation

- [ ] **Step 10.1: Create unit tests**

  Create `desktop_app/tests/test_pouring_direction.py`:

  ```python
  import pytest
  import numpy as np
  import trimesh
  from core.pouring_direction import (
      compute_persistence_pairs,
      score_pouring_direction,
      find_optimal_pouring_directions
  )
  
  def test_height_field_computation():
      """Test that height function is computed correctly."""
      # Create simple mesh (unit cube)
      mesh = trimesh.primitives.Box()
      direction = np.array([0, 0, 1])
      
      from core.mesh_analysis import compute_height_field
      vertex_heights, face_heights = compute_height_field(mesh, direction)
      
      # Verify heights match z-coordinates
      assert np.allclose(vertex_heights, mesh.vertices[:, 2])

  def test_critical_points_on_sphere():
      """Sphere should have exactly 2 critical points (top max, bottom min)."""
      mesh = trimesh.primitives.Sphere()
      direction = np.array([0, 0, 1])
      
      pairs, _ = compute_persistence_pairs(mesh, direction)
      
      # One persistence pair for the single maximum
      assert len(pairs) >= 0  # May vary based on mesh resolution

  def test_saddle_detection_on_torus():
      """Torus should have saddle points."""
      mesh = trimesh.primitives.Torus()
      direction = np.array([0, 0, 1])
      
      pairs, _ = compute_persistence_pairs(mesh, direction)
      
      # Torus should produce multiple persistence pairs
      assert len(pairs) > 0

  def test_direction_scoring():
      """Test that direction scoring produces valid results."""
      mesh = trimesh.primitives.Box()
      direction = np.array([0, 0, 1])
      
      result = score_pouring_direction(mesh, direction)
      
      assert result.total_score >= 0
      assert result.direction is not None

  def test_full_optimization():
      """Test end-to-end optimization."""
      mesh = trimesh.load("test_models/simple_part.stl")  # Need test model
      
      result = find_optimal_pouring_directions(mesh)
      
      assert result.silicone_direction_1 is not None
      assert result.silicone_direction_2 is not None
      assert result.resin_direction is not None
      
      # Directions should be unit vectors
      assert np.isclose(np.linalg.norm(result.silicone_direction_1), 1.0)
      assert np.isclose(np.linalg.norm(result.resin_direction), 1.0)
  ```

- [ ] **Step 10.2: Add integration tests with known geometries**

  Test with geometries that have known optimal directions:
  - Simple convex shapes (should have low scores in all directions)
  - Cup/bowl shapes (should prefer upward pouring)
  - Complex shapes with pockets (should identify problematic regions)

---

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_candidate_directions` | 64 | Number of Fibonacci sphere samples |
| `tilt_angle_deg` | 10.0 | Assumed mold tilting capability |
| `area_threshold_mm2` | 0.5 | Minimum trapped area to consider |
| `alignment_angle_deg` | 30.0 | Max angle between silicone directions |
| `resin_cone_angle_deg` | 10.0 | Cone angle for resin direction search |

---

## Dependencies

Existing:
- `trimesh` - Mesh operations
- `numpy` - Numerical computations
- `pyvista` - Visualization (optional)

No new dependencies required.

---

## File Structure

```
desktop_app/
 core/
    __init__.py              # Updated exports
    mesh_analysis.py         # Add height field, vertex neighbors
    pouring_direction.py     # NEW - Main implementation
    ...
 ui/
    main_window.py           # Add pouring direction UI
 viewer/
    mesh_viewer.py           # Add direction visualization
 tests/
     test_pouring_direction.py  # NEW - Unit tests
```

---

## References

1. Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2000). Topological persistence and simplification.
2. Milnor, J. (1963). Morse theory.
3. Biasotti, S., et al. (2013). Persistence-based shape segmentation.

---

## Checklist

- [ ] Phase 1: Height function computation
- [ ] Phase 2: Critical point detection
- [ ] Phase 3: Persistence pairing
- [ ] Phase 4: Bubble relevance scoring
- [ ] Phase 5: Direction scoring and selection
- [ ] Phase 6: Silicone and resin direction selection
- [ ] Phase 7: Main API
- [ ] Phase 8: UI integration
- [ ] Phase 9: Visualization
- [ ] Phase 10: Testing
