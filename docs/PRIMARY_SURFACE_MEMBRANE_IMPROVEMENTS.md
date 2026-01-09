# Primary Surface Membrane Implementation

## Overview

This document provides a comprehensive implementation guide for the primary parting surface membrane creation following the **"Volume-Aware Design of Composite Molds"** paper (Alderighi et al., SIGGRAPH 2019). The implementation covers the full pipeline from upstream steps (parting directions, mold half classification) through the core membrane creation algorithm to downstream steps (smoothing, post-processing).

### Goals

The parting surface membrane must:
1. **Touch the part mesh M at all inner boundaries** (inner boundary → part surface)
2. **Touch the hull ∂H at all outer boundaries** (outer boundary → hull surface)
3. **Be continuous** with no orphaned/disconnected sections
4. **Have no thin flat sections** that could cause boolean operation issues
5. **Have no self-folding/intersecting regions**
6. **Have no small tubular sections** (thin elongated boundary loops)
7. **Be watertight** with no internal holes

---

## Full Pipeline Overview

The complete mold design pipeline from the paper consists of these stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MOLD DESIGN PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Triangulated manifold surface M (the part to cast)                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: INFLATED HULL CREATION                                        │   │
│  │ • Compute offset surface of M                                         │   │
│  │ • Create tetrahedral tessellation H of convex hull                    │   │
│  │ • M separates H into inside/outside - mold volume O is the outside    │   │
│  │ Implementation: core/inflated_hull.py, core/tetrahedral_mesh.py       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: PARTING DIRECTION SELECTION (Section 4.1)                     │   │
│  │ • Sample k candidate directions on Gaussian sphere                    │   │
│  │ • GPU-accelerated visibility computation for each direction           │   │
│  │ • Select d₁, d₂ that minimize non-visible surface area               │   │
│  │ Implementation: core/parting_direction.py                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: MOLD HALF CLASSIFICATION                                      │   │
│  │ • Partition boundary ∂H into ∂H₁ and ∂H₂ based on d₁, d₂             │   │
│  │ • Use greedy region-growing from seed faces                           │   │
│  │ • Assign faces to H₁ or H₂ based on normal alignment                 │   │
│  │ Implementation: core/mold_half_classification.py                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: ESCAPE PATH COMPUTATION (Section 4.1)                         │   │
│  │ • For each interior vertex, compute shortest path to ∂H               │   │
│  │ • Use weighted geodesics: edge_cost = length × 1/(dist²+ε)            │   │
│  │ • Label vertices based on which boundary (H₁ or H₂) they reach       │   │
│  │ Implementation: core/tetrahedral_mesh.py (run_dijkstra_escape_labeling)│  │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: PRIMARY CUT EDGE DETECTION (Section 4.1)                      │   │
│  │ • Edge (vᵢ,vⱼ) is a primary cut if:                                   │   │
│  │   - wᵢ and wⱼ (escape destinations) are on different parts of ∂H     │   │
│  │ • Skip edges near boundary between ∂H₁ and ∂H₂ (noise reduction)      │   │
│  │ Implementation: core/tetrahedral_mesh.py (find_primary_cutting_edges) │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 6: MEMBRANE MESHING (Section 4.3) ◀━━ THIS DOCUMENT              │   │
│  │ • Extended Marching Tetrahedra from cut edge flags                    │   │
│  │ • 3-edge configs → single triangle, 4-edge → quad                     │   │
│  │ • Boundary-aware cut point placement (on M or ∂H, not midpoint)       │   │
│  │ Implementation: core/parting_surface.py (extract_parting_surface)     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 7: MEMBRANE SMOOTHING (Section 4.4) ◀━━ THIS DOCUMENT            │   │
│  │ • Alternating smoothing: boundary polyline → interior vertices        │   │
│  │ • Re-project boundary vertices to ORIGINAL surface (M or ∂H)          │   │
│  │ • Damping factor 0.5 for convergence                                  │   │
│  │ Implementation: core/surface_propagation.py (smooth_membrane_...)     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 8: SECONDARY MEMBRANE DETECTION (Section 4.2)                    │   │
│  │ • For same-label interior edges (vᵢ,vⱼ) where both escape to H₁/H₂   │   │
│  │ • Compute minimal surface bounded by escape paths + boundary path     │   │
│  │ • If surface intersects part M → secondary cut needed                 │   │
│  │ Implementation: core/tetrahedral_mesh.py (find_secondary_cutting_edges)│  │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│  OUTPUT: Cut layout (parting surface + additional membranes)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Paper Algorithm Details

### Section 4.1: Parting Surfaces

The paper describes partitioning the mold volume O into two pieces O₁ and O₂:

> "Given the two parting directions, we partition the boundary ∂H into two parts, 
> ∂H₁ and ∂H₂: we select the two faces F₁ and F₂ of ∂H whose normals best align 
> with d₁ and d₂ and then use a greedy region-growing approach from F₁ and F₂ 
> to assign faces to ∂H₁ and ∂H₂, according to the alignment of their normal to d₁ and d₂."

**Primary Cut Edge Detection:**

> "For each interior edge e = (vᵢ, vⱼ) in the tessellated volume O, we compute the 
> shortest paths from vᵢ and vⱼ to the boundary surface ∂H. Then, the edge is 
> traversed by a parting surface if the destination vertices wᵢ and wⱼ belong 
> to different parts ∂H₁, ∂H₂ of the boundary surface."

**Boundary Zone Exclusion:**

> "We compute the shortest paths towards all vertices of ∂H, except for those 
> whose distance from the boundary between ∂H₁ and ∂H₂ is less than a fixed 
> threshold... In our experiments, we set the threshold to 15% of the convex 
> hull bounding box diagonal."

### Section 4.2: Additional Membranes

For features that prevent mold extraction:

> "For each interior edge e = (vᵢ, vⱼ), let wᵢ and wⱼ denote the destination 
> vertices of the shortest paths from vᵢ and vⱼ to the exterior boundary. 
> We introduce a cutting membrane across the edge e to separate vᵢ and vⱼ if 
> a discrete approximation of the minimal surface bounded by the edge (vᵢ, vⱼ), 
> the escape paths (vᵢ, wᵢ), (vⱼ, wⱼ), and the shortest path (wᵢ, wⱼ) 
> (computed on ∂H), intersects the object M."

### Section 4.3: Membrane Meshing

> "The triangulated surface C encoding the cut layout is composed using a set of 
> patches that are interconnected by chains of non-manifold edges that are 
> **bounded by construction by the object surface mesh M and the external boundary ∂H**."

This is implemented using extended Marching Tetrahedra:
- For each tetrahedron, check which of 6 edges are cut
- Build 6-bit configuration index
- Look up triangles from configuration table:
  - **3-edge configs** (7, 25, 42, 52): Single triangle (vertex isolated)
  - **4-edge configs** (30, 45, 51): Quadrilateral surface (2 triangles)
  - **5/6-edge configs**: SKIPPED to avoid self-intersections

**Critical: Boundary-aware cut point placement**
- If edge touches part surface M: place vertex ON M (not at midpoint)
- If edge touches hull ∂H: place vertex ON ∂H (not at midpoint)
- Otherwise: use edge midpoint

### Section 4.4: Membrane Smoothing

> "The smoothing is performed by alternating two main steps:
> 1. First we smooth the polyline that includes the boundary vertices only.
>    After this smooth step, we **re-project those vertices onto the original 
>    surface of M or the external boundary ∂H**, depending on which of the two 
>    surfaces they belonged to originally.
> 2. Then, we smooth all the interior vertices, keeping the ones on the 
>    boundary fixed (the ones smoothed in the previous step).
> Each smoothing step is performed using a damping factor (0.5 in our experiments)."

**Critical insight:** Re-projection uses the **original surface classification from extraction**, NOT closest distance. This is why we track `vertex_boundary_type` throughout the pipeline.

### Section 4.5: Shortest Path Computation (Weighted Geodesics)

The paper uses weighted geodesics that push membranes away from the part surface:

> "We compute the shortest paths from the interior vertices to the boundary 
> according to a metric, which makes traveling near the object longer. This is 
> done by weighting the Euclidean arc length by a scalar function that depends 
> on the squared distance from M."

**Edge weight formula:**
```
edge_weight = 1 / (dist² + ε)
edge_cost = edge_length × edge_weight
```

Where:
- `dist` = geodesic distance from edge midpoint to part surface M
- `ε` = 0.25 (prevents division by zero)

This ensures membranes tend to be perpendicular to the part surface.

---

## Implementation Architecture

### File Structure

```
desktop_app/core/
├── inflated_hull.py           # Step 1: Create offset bounding volume
├── parting_direction.py       # Step 2: Find optimal parting directions d₁, d₂
├── mold_half_classification.py # Step 3: Classify boundary into H₁, H₂
├── tetrahedral_mesh.py        # Steps 4-5: Escape paths, cut edge detection
├── parting_surface.py         # Step 6: Membrane meshing (Marching Tetrahedra)
├── surface_propagation.py     # Step 7: Membrane smoothing
└── pouring_direction.py       # Downstream: Optimal pouring direction

desktop_app/ui/
└── main_window.py             # ComprehensivePrimarySurfaceWorker integration
```

### Key Data Structures

#### TetrahedralMeshResult (tetrahedral_mesh.py)
```python
@dataclass
class TetrahedralMeshResult:
    # Geometry
    vertices: np.ndarray               # (N, 3) vertex positions (inflated)
    vertices_original: np.ndarray      # (N, 3) original positions (non-inflated)
    tetrahedra: np.ndarray             # (T, 4) tet vertex indices
    edges: np.ndarray                  # (E, 2) unique edges
    
    # Boundary classification
    boundary_labels: np.ndarray        # (N,) -1=part, 0=interior, 1=H1, 2=H2
    boundary_vertices: np.ndarray      # (N,) bool mask of boundary vertices
    
    # Edge weights (Section 4.5)
    edge_lengths: np.ndarray           # (E,) Euclidean edge lengths
    edge_weights: np.ndarray           # (E,) 1/(dist²+ε) weights
    edge_distances: np.ndarray         # (E,) distances to part surface
    
    # Escape labeling (Section 4.1)
    seed_escape_labels: np.ndarray     # (I,) 1=H1, 2=H2, 0=unreachable
    seed_vertex_indices: np.ndarray    # (I,) interior vertex indices
    seed_escape_destinations: np.ndarray # (I,) boundary vertex reached
    seed_escape_paths: List[List[int]] # Paths from interior to boundary
    
    # Cut edges
    primary_cut_edges: List[Tuple[int, int]]    # Primary parting surface
    secondary_cut_edges: List[Tuple[int, int]]  # Additional membranes
    cut_edge_flags: np.ndarray         # (E,) combined flags
```

#### PartingSurfaceResult (parting_surface.py)
```python
@dataclass
class PartingSurfaceResult:
    mesh: trimesh.Trimesh              # The extracted surface mesh
    vertices: np.ndarray               # (V, 3) surface vertices
    faces: np.ndarray                  # (F, 3) surface faces
    vertex_to_edge: np.ndarray         # Maps surface vertex → global edge index
    
    # Critical for proper smoothing (Section 4.4)
    vertex_boundary_type: np.ndarray   # (V,) boundary classification:
                                       # -1 = on part M (INNER, re-project to part)
                                       #  0 = interior (midpoint, no constraint)
                                       # 1/2 = on hull ∂H (OUTER, re-project to hull)
```

---

## Detailed Implementation

### Step 6: Membrane Meshing (extract_parting_surface)

Located in `core/parting_surface.py`:

```python
def extract_parting_surface(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    edges: np.ndarray,
    cut_edge_flags: np.ndarray,
    tet_edge_indices: np.ndarray,
    boundary_labels: np.ndarray = None
) -> PartingSurfaceResult:
    """
    Extract parting surface using Marching Tetrahedra.
    
    Per paper Section 4.3: Surface is "bounded by construction" by M and ∂H.
    """
```

**Boundary-aware cut point placement:**
```python
for i, e_idx in enumerate(cut_edge_indices):
    v0, v1 = edges[e_idx]
    bl0, bl1 = boundary_labels[v0], boundary_labels[v1]
    
    if bl0 == -1 and bl1 != -1:
        # v0 on part surface M → place at v0
        surface_vertices[i] = verts[v0]
        vertex_boundary_type[i] = -1  # INNER boundary
    elif bl1 == -1 and bl0 != -1:
        # v1 on part surface M → place at v1
        surface_vertices[i] = verts[v1]
        vertex_boundary_type[i] = -1  # INNER boundary
    elif bl0 in (1, 2) and bl1 == 0:
        # v0 on hull ∂H → place at v0
        surface_vertices[i] = verts[v0]
        vertex_boundary_type[i] = bl0  # OUTER boundary
    # ... more cases ...
    else:
        # Both interior → use midpoint
        surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
        vertex_boundary_type[i] = 0  # Interior
```

**Marching Tetrahedra table:**
```python
MARCHING_TET_TABLE = {
    # 3-edge configs (single triangle)
    7:  [(0, 1, 2)],   # Vertex 0 isolated
    25: [(0, 3, 4)],   # Vertex 1 isolated
    42: [(1, 3, 5)],   # Vertex 2 isolated
    52: [(2, 4, 5)],   # Vertex 3 isolated
    
    # 4-edge configs (quad = 2 triangles)
    30: [(1, 3, 4), (1, 4, 2)],  # Split {v0,v1} vs {v2,v3}
    45: [(0, 3, 5), (0, 5, 2)],  # Split {v0,v2} vs {v1,v3}
    51: [(0, 4, 5), (0, 5, 1)],  # Split {v0,v3} vs {v1,v2}
    
    # 5/6-edge configs: EMPTY (skip to avoid self-intersections)
}
```

### Step 7: Membrane Smoothing (smooth_membrane_with_boundary_reprojection)

Located in `core/surface_propagation.py`:

```python
def smooth_membrane_with_boundary_reprojection(
    membrane_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh,
    iterations: int = 5,
    damping_factor: float = 0.5,
    vertex_boundary_type: np.ndarray = None
) -> SmoothingResult:
    """
    Smooth membrane per paper Section 4.4.
    
    CRITICAL: Use vertex_boundary_type from extraction for re-projection,
    NOT closest distance classification.
    """
```

**Classification using vertex_boundary_type:**
```python
if vertex_boundary_type is not None:
    for vi in boundary_verts:
        bt = vertex_boundary_type[vi]
        if bt == -1:
            boundary_surface[vi] = 'part'   # Re-project to M
        elif bt in (1, 2):
            boundary_surface[vi] = 'hull'   # Re-project to ∂H
        else:
            boundary_surface[vi] = 'patch'  # Needs fallback
```

**Alternating smoothing:**
```python
for iteration in range(iterations):
    # Step 1: Smooth boundary vertices
    for vi in boundary_verts:
        new_pos = laplacian_smooth(vi, neighbors, damping_factor)
        # Re-project to ORIGINAL surface
        if boundary_surface[vi] == 'part':
            vertices[vi] = part_proximity.closest(new_pos)
        elif boundary_surface[vi] == 'hull':
            vertices[vi] = hull_proximity.closest(new_pos)
    
    # Step 2: Smooth interior vertices (boundary fixed)
    for vi in interior_verts:
        vertices[vi] = laplacian_smooth(vi, neighbors, damping_factor)
```

### Pipeline Integration (ComprehensivePrimarySurfaceWorker)

Located in `ui/main_window.py`:

```python
class ComprehensivePrimarySurfaceWorker(QThread):
    """Full primary surface processing pipeline."""
    
    def run(self):
        # Step 1: Prepare data structures
        tet_result = prepare_parting_surface_data(self.tet_result)
        
        # Step 2: Extract surface (Marching Tetrahedra)
        extraction_result = extract_parting_surface_from_tet_result(
            tet_result, cut_type='primary'
        )
        
        # Step 3: Clean surface (merge vertices, remove degenerates)
        repaired_result = repair_parting_surface_with_part(extraction_result)
        
        # Step 4: Remove small holes (< 4mm perimeter)
        hole_result = remove_small_boundary_loops(temp_surface, min_perimeter=4.0)
        
        # Step 5: Smooth with boundary re-projection
        smoothing_result = smooth_membrane_with_boundary_reprojection(
            current_mesh,
            part_mesh=self.part_mesh,
            hull_mesh=self.hull_mesh,
            vertex_boundary_type=current_boundary_type  # CRITICAL
        )
        
        # Step 6: Remove orphan islands
        cleaned_mesh, _, _ = remove_isolated_islands(current_mesh)
        
        # Step 7: Fill tubular sections
        tubular_result = fill_tubular_sections(current_mesh)
        
        # Step 8: Remove thin flat sections
        thin_result = remove_thin_flat_sections(current_mesh)
        
        # Step 9: Repair self-folding
        fold_result = repair_self_folding(current_mesh)
```

---

## Constants and Thresholds

### Parting Surface Constants (parting_surface.py)
```python
# Gap filling
MIN_PROJECTION_DISTANCE_FRACTION = 0.1
PROJECTION_OFFSET_FRACTION = 0.3
DISTANCE_THRESHOLD_EDGE_MULTIPLIER = 3.0

# Small feature removal
DEFAULT_MIN_LOOP_PERIMETER = 4.0  # mm
TUBULAR_ASPECT_RATIO_THRESHOLD = 6.0
TUBULAR_MAX_PERIMETER = 15.0  # mm
MIN_TUBULAR_VERTICES = 6

# Island removal
PRIMARY_MIN_ISLAND_TRIANGLES = 10
PRIMARY_MIN_ISLAND_AREA_FRACTION = 0.01  # 1%

# Triangle quality
MIN_TRIANGLE_AREA_FRACTION = 0.01  # 1% of median
```

### Edge Weight Constants (tetrahedral_mesh.py)
```python
EDGE_WEIGHT_EPSILON = 0.25  # Per paper Section 4.5
```

### Smoothing Constants (surface_propagation.py)
```python
ON_SURFACE_TOLERANCE_FRACTION = 0.05  # 5%
MAX_REPROJECT_DISTANCE_FRACTION = 0.15  # 15%
INNER_BOUNDARY_THRESHOLD_FRACTION = 0.02  # 2%
```

### Boundary Zone Exclusion (paper Section 4.1)
```python
BOUNDARY_EXCLUSION_THRESHOLD = 0.15  # 15% of bbox diagonal
```

---

## Progress Log

### Implemented Features ✅

#### Phase 1: Track Boundary Vertex Origin ✅
- Added `vertex_boundary_type` field to `PartingSurfaceResult` dataclass
- Updated `extract_parting_surface` to populate array with boundary-aware cut point placement
- Values: -1=part M (inner), 0=interior, 1/2=hull ∂H (outer)
- Updated `repair_parting_surface` to preserve boundary types through vertex merging

#### Phase 2: Fix Boundary Re-projection ✅
- Updated `smooth_membrane_with_boundary_reprojection` to accept `vertex_boundary_type`
- Inner boundaries (-1) re-project to part mesh M
- Outer boundaries (1/2) re-project to hull mesh ∂H
- Added neighbor propagation for patch vertices before distance fallback

#### Phase 3: Orphan Island Removal ✅
- `remove_isolated_islands()` in surface_propagation.py
- Uses `PRIMARY_MIN_ISLAND_TRIANGLES` threshold (default: 10)

#### Phase 4: Problematic Geometry Detection ✅
- `detect_thin_flat_sections()` / `remove_thin_flat_sections()`
- `detect_tubular_sections()` / `fill_tubular_sections()`
- `detect_self_folding()` / `repair_self_folding()`

#### Phase 5: Hole and Gap Filling ✅
- `remove_small_boundary_loops()` for internal holes
- `close_parting_surface_gaps()` with vertex_boundary_type support

#### Phase 6: Pipeline Integration ✅
- `ComprehensivePrimarySurfaceWorker` with 9-step pipeline

---

## Testing Checklist

- [x] Membrane touches part mesh M at all inner boundaries
- [x] Membrane touches hull ∂H at all outer boundaries
- [x] No orphaned/disconnected triangle islands
- [x] No thin flat degenerate sections
- [x] No self-intersecting regions
- [x] No small tubular loops
- [x] No internal holes in membrane
- [x] Smooth surface with correct boundary re-projection
- [ ] **Validation with complex geometries** (knots, high-genus surfaces)
- [ ] **Performance optimization** for large meshes

---

## Suggested Improvements

Based on a detailed review of the paper (Section 4.5 especially) and the current implementation, the following improvements could enhance membrane quality and better match the paper's algorithm:

### 1. **Exterior Boundary Reshaping (∂F offset surface)** — HIGH PRIORITY

**Paper Section 4.5:**
> "Computing the shortest paths to the boundary of the convex hull may result in membranes that do not align well to the object geometry because the shape of the convex hull may discard essential features of the original object shape... Therefore, we compute the shortest paths to a surface that better retains information about the shape of M, namely, an offset surface ∂F of M that encloses the convex hull in its interior. The offset radius is set as the maximum distance from the points on the convex hull to M."

**Current Implementation:**
- Uses the inflated hull directly as the target boundary
- Does NOT implement the ∂F offset surface bias

**Suggested Change:**
```python
# In compute_edge_weights() or run_dijkstra_escape_labeling():
# 1. Compute max distance from hull vertices to part mesh M
R = max_distance_from_hull_vertices_to_part_mesh

# 2. For each hull boundary vertex w, compute λ_w = R - dist(w, M)
# 3. When computing shortest path distance d(v, w), add λ_w bias:
#    effective_distance = d(v, w) + λ_w
```

**Impact:** Would improve membrane alignment to geometric features (e.g., bowl cavities, holes).

---

### 2. **Boundary Zone Exclusion Threshold** — MEDIUM PRIORITY

**Paper Section 4.1:**
> "We compute the shortest paths towards all vertices of ∂H, except for those whose distance from the boundary between ∂H₁ and ∂H₂ is less than a fixed threshold... In our experiments, we set the threshold to 15% of the convex hull bounding box diagonal."

**Current Implementation:**
- The code in `mold_half_classification.py` has `DEFAULT_BOUNDARY_ZONE_THRESHOLD = 0.15`
- BUT it's not clear if escape labeling properly EXCLUDES boundary vertices near the H1/H2 seam

**Suggested Change:**
```python
# In run_dijkstra_escape_labeling():
# 1. Compute distance from each H1/H2 boundary vertex to the H1-H2 seam
# 2. EXCLUDE vertices within 15% of bbox diagonal from being escape targets
# This prevents discretization noise in the parting surface near the mold seam
```

**Impact:** Reduces noise/artifacts in parting surface near the mold half boundary.

---

### 3. **Proper Polyline Smoothing Order** — LOW PRIORITY (already implemented)

**Paper Section 4.4:**
> "First we smooth the polyline that includes the boundary vertices only. After this smooth step, we re-project those vertices onto the original surface of M or the external boundary ∂H."

**Current Implementation:** ✅ Already correct
- Smooths boundary vertices first (polyline smoothing along boundary edges)
- Re-projects to target surfaces
- Then smooths interior vertices with boundary fixed

**No change needed**, but the code comment could be more explicit about matching the paper.

---

### 4. **Non-Manifold Membrane Support** — LOW PRIORITY

**Paper Section 4.3:**
> "The triangulated surface C encoding the cut layout is composed using a set of patches that are interconnected by chains of non-manifold edges."

**Current Implementation:**
- Uses Marching Tetrahedra but SKIPS 5-edge and 6-edge configs
- This prevents self-intersections but may miss some membrane patches

**Suggested Change:**
- Consider implementing the paper's extended Marching Tetrahedra table from [Bloomenthal & Ferguson 1995; Bonnell et al. 2003]
- Handle non-manifold edge chains explicitly

**Impact:** Better handling of complex geometries where multiple membrane patches meet.

---

### 5. **Adaptive Smoothing Iteration Count** — LOW PRIORITY

**Paper Section 4.4:**
> "Each smoothing step is performed using a damping factor (0.5 in our experiments) to ensure a proper convergence to the final solution."

**Current Implementation:**
- Uses fixed iteration count (default 5)
- Damping factor 0.5 matches paper

**Suggested Change:**
```python
# Implement convergence-based stopping:
# Continue smoothing until max vertex displacement < threshold
while max_displacement > convergence_threshold and iteration < max_iterations:
    smooth_iteration()
    max_displacement = np.max(np.linalg.norm(new_vertices - old_vertices, axis=1))
```

**Impact:** More efficient smoothing (may need fewer iterations for simple shapes, more for complex).

---

### 6. **Cut Point Interpolation for BOTH-on-boundary Cases** — LOW PRIORITY

**Current Implementation:**
When both edge vertices are on the same boundary type:
```python
elif bl0 == -1 and bl1 == -1:
    # BOTH on part surface → use midpoint
    surface_vertices[i] = 0.5 * (verts[v0] + verts[v1])
```

**Suggested Improvement:**
For edges where BOTH vertices are on boundary surfaces, the paper suggests placing the cut point ON the surface (not at the floating midpoint). This requires projecting the midpoint back to the surface:
```python
elif bl0 == -1 and bl1 == -1:
    midpoint = 0.5 * (verts[v0] + verts[v1])
    # Project midpoint onto part surface M
    surface_vertices[i] = project_to_part_mesh(midpoint)
```

**Impact:** Ensures membrane is truly "bounded by construction" even at edges fully on a boundary.

---

### 7. **Perlin Noise on Parting Surface Seam** — DOWNSTREAM (Future)

**Paper Section 5:**
> "Perlin noise is added to the parting surface between the two silicone pieces to improve registration."

**Current Implementation:**
- Not implemented

**Suggested Change:**
- Add optional Perlin noise displacement to the parting surface at the H1/H2 seam
- This would be applied AFTER smoothing, before mold fabrication export

**Impact:** Better physical alignment of fabricated mold pieces.

---

### 8. **Persistence Pairing for Pouring Direction** — DOWNSTREAM (Future)

**Paper Section 5.2:**
> "We use the pairing mechanism from persistence homology to assess the relevance of local maxima for a given pouring direction."

**Current Implementation:**
- `pouring_direction.py` exists but may not implement full persistence pairing

**Suggested Change:**
- Implement topological persistence analysis for air bubble prevention
- Rank pouring directions by relevance of trapped air regions

**Impact:** Optimal casting direction selection to minimize artifacts.

---

### Summary of Priority

| Improvement | Priority | Effort | Impact |
|------------|----------|--------|--------|
| 1. ∂F offset surface bias | HIGH | Medium | Better feature alignment |
| 2. Boundary zone exclusion | MEDIUM | Low | Reduced seam artifacts |
| 3. Polyline smoothing order | ✅ DONE | — | — |
| 4. Non-manifold support | LOW | High | Complex geometry handling |
| 5. Adaptive iterations | LOW | Low | Performance |
| 6. Cut point projection | LOW | Low | Boundary accuracy |
| 7. Perlin noise registration | FUTURE | Low | Physical fabrication |
| 8. Persistence pouring | FUTURE | High | Casting quality |

---

## Upstream Dependencies

The primary surface membrane depends on these upstream computations being complete:

1. **Inflated Hull** (`inflated_hull.py`)
   - Provides bounding volume for tetrahedral meshing
   - Offset surface ∂H that surrounds part M

2. **Tetrahedral Mesh** (`tetrahedral_mesh.py`)
   - Tessellation of mold volume O
   - Vertex positions and connectivity

3. **Parting Directions** (`parting_direction.py`)
   - d₁, d₂ directions for mold opening
   - Visibility analysis for direction selection

4. **Mold Half Classification** (`mold_half_classification.py`)
   - ∂H₁, ∂H₂ boundary partitioning
   - `boundary_labels` array for tetrahedral vertices

5. **Escape Labeling** (`tetrahedral_mesh.py`)
   - Dijkstra shortest paths from interior to boundary
   - `seed_escape_labels`, `seed_escape_paths`, `seed_escape_destinations`

6. **Primary Cut Edges** (`tetrahedral_mesh.py`)
   - Detection of edges where parting surface passes
   - `primary_cut_edges` list

---

## Downstream Dependencies

The primary surface membrane feeds into:

1. **Secondary Membrane Detection** (`tetrahedral_mesh.py`)
   - Uses same escape paths for minimal surface intersection test
   - Adds `secondary_cut_edges` for features requiring additional cuts

2. **Pouring Direction Optimization** (`pouring_direction.py`)
   - Uses mold half geometry to optimize casting direction
   - Minimizes air bubble entrapment

3. **Mold Fabrication** (downstream)
   - Hard shell generation from parting surface
   - Metamold geometry incorporating cut layout

---

## References

- **Paper:** Alderighi et al., "Volume-Aware Design of Composite Molds", ACM Trans. Graph. 38(4), 2019
- **Section 4.1:** Parting surfaces, escape paths, primary cut detection
- **Section 4.2:** Additional membranes, minimal surface intersection
- **Section 4.3:** Membrane meshing (Marching Tetrahedra)
- **Section 4.4:** Membrane smoothing (Laplacian with boundary re-projection)
- **Section 4.5:** Shortest path computation (weighted geodesics)
