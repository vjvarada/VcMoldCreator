---
applyTo: 'desktop_app/core/**'
---

# Core Algorithm Implementation Guidelines

This file provides detailed guidance for working with the core algorithm modules in `desktop_app/core/`.

---

## Module Overview

| Module | Purpose | Paper Section |
|--------|---------|---------------|
| `stl_loader.py` | Load and validate STL files | Input |
| `mesh_analysis.py` | Mesh diagnostics and metrics | Input |
| `mesh_repair.py` | Fix non-manifold issues | Input |
| `mesh_decimation.py` | Reduce triangle count | Input |
| `inflated_hull.py` | Generate bounding volume | Section 3 |
| `parting_direction.py` | Find d₁, d₂ directions | Section 4.1 |
| `mold_half_classification.py` | Classify ∂H into H₁, H₂ | Section 4.1 |
| `tetrahedral_mesh.py` | Tet meshing, Dijkstra, cuts | Sections 4.1, 4.5 |
| `parting_surface.py` | Marching Tetrahedra | Section 4.3 |
| `surface_propagation.py` | Membrane smoothing | Section 4.4 |
| `pouring_direction.py` | Optimal pour directions | Section 5.2 |
| `csg_backend.py` | CSG operations (libigl/CGAL + manifold3d fallback) | Section 5 |
| `mold_fabrication.py` | Hard shell & metamold geometry | Section 5 |
| `resin_channels.py` | Resin inlet & air escape holes | Section 5 |

---

## tetrahedral_mesh.py

### Key Functions

#### `tetrahedralize_mesh(surface_mesh, edge_length_fac, optimize)`
Creates tetrahedral mesh using fTetWild.

#### `compute_edge_weights(tet_result, part_mesh)`
Computes weighted edge lengths using paper's formula:
```python
# Section 4.5: "weighting the Euclidean arc length by a scalar function 
# that depends on the squared distance from M"
weight = 1.0 / (dist_to_part ** 2 + EPSILON)

# Section 4.5: "We simply bias the metric by storing on the convex hull 
# vertices their distance from ∂F and adding this distance"
if vertex_on_boundary:
    lambda_w = R - dist_to_part  # R = max hull-to-part distance
    biased_dist = dist_to_part + max(lambda_w, 0)
    weight = 1.0 / (biased_dist ** 2 + EPSILON)
```

#### `run_dijkstra_escape_labeling(tet_result)`
Dijkstra from interior vertices to H₁/H₂ boundary.

**Important**: Boundary zone vertices (within 15% of bbox diagonal from H₁-H₂ seam) are NOT valid escape destinations.

#### `find_primary_cutting_edges(tet_result)`
Edge (vᵢ, vⱼ) is primary cut if:
- Both vertices are interior (escape_label != 0)
- wᵢ and wⱼ (escape destinations) are on different boundaries

#### `find_secondary_cutting_edges(tet_result, part_mesh)`
For same-label interior edges:
1. Compute minimal surface bounded by escape paths
2. If surface intersects part M → secondary cut needed

### Data Structure: TetrahedralMeshResult

```python
@dataclass
class TetrahedralMeshResult:
    # Geometry
    vertices: np.ndarray           # (N, 3) current positions
    vertices_original: np.ndarray  # (N, 3) original (uninflated)
    tetrahedra: np.ndarray         # (T, 4) tet vertex indices
    edges: np.ndarray              # (E, 2) unique edges
    
    # Classification
    boundary_labels: np.ndarray    # (N,) -1=part, 0=interior, 1=H1, 2=H2
    boundary_vertices: np.ndarray  # (N,) bool mask
    
    # Edge weights (Section 4.5)
    edge_lengths: np.ndarray       # (E,) Euclidean lengths
    edge_weights: np.ndarray       # (E,) 1/(d²+ε)
    weighted_edge_lengths: np.ndarray  # edge_lengths * edge_weights
    
    # Dijkstra results
    seed_escape_labels: np.ndarray      # (I,) 1=H1, 2=H2, 0=unreachable
    seed_vertex_indices: np.ndarray     # (I,) interior vertex indices
    seed_escape_destinations: np.ndarray # (I,) boundary vertex reached
    seed_escape_paths: List[List[int]]  # Paths to boundary
    
    # Cut edges
    primary_cut_edges: List[Tuple[int, int]]
    secondary_cut_edges: List[Tuple[int, int]]
    cut_edge_flags: np.ndarray     # (E,) bool
    
    # Marching Tets support
    edge_to_index: Dict[Tuple, int]
    tet_edge_indices: np.ndarray   # (T, 6) global edge indices per tet
```

---

## parting_surface.py

### Marching Tetrahedra Algorithm

The module implements extended Marching Tetrahedra per Nielson & Franke 1997.

#### Edge Numbering
```
Edge 0: (v0, v1)    Edge 3: (v1, v2)
Edge 1: (v0, v2)    Edge 4: (v1, v3)
Edge 2: (v0, v3)    Edge 5: (v2, v3)
```

#### Configuration Table (64 entries)

| Edges Cut | Example Config | Output | Vertex Types |
|-----------|----------------|--------|--------------|
| 0 | 0 | No surface | None |
| 3 | 7, 25, 42, 52 | 1 triangle | Mid-edge |
| 4 | 30, 45, 51 | 2 triangles | Mid-edge |
| 5 | 62, 61, 59, 55, 47, 31 | 5 triangles | Mid-edge + face |
| 6 | 63 | 12 triangles | Mid-edge + face + inner |

#### 5-Edge Configuration (Case 3)
One edge NOT cut means two vertices share same label.
```python
# Face opposite to same-label vertices has 3 cut edges
# Face vertex = centroid of 3 mid-edge points on that face
face_vertex = (m_ij + m_ik + m_jk) / 3

# Generate 5 triangles connecting mid-edges through face vertex
```

#### 6-Edge Configuration (Case 4)
All 4 vertices have different labels.
```python
# Step 1: Compute 4 face vertices
m_012 = (m_01 + m_02 + m_12) / 3  # Face (v0,v1,v2)
m_013 = (m_01 + m_03 + m_13) / 3  # Face (v0,v1,v3)
m_023 = (m_02 + m_03 + m_23) / 3  # Face (v0,v2,v3)
m_123 = (m_12 + m_13 + m_23) / 3  # Face (v1,v2,v3)

# Step 2: Compute inner vertex
m_t = (m_012 + m_013 + m_023 + m_123) / 4

# Step 3: Generate 12 triangles (3 per face through m_t)
```

### Boundary-Aware Cut Point Placement

```python
# If edge touches part (boundary_label == -1):
if bl0 == -1 and bl1 != -1:
    cut_point = verts[v0]  # Place on part surface
    vertex_boundary_type[i] = -1  # Track for re-projection

# If edge touches hull (boundary_label in (1,2)):
elif bl0 in (1, 2) and bl1 == 0:
    cut_point = verts[v0]  # Place on hull surface
    vertex_boundary_type[i] = bl0  # Track for re-projection

# Interior edges:
else:
    cut_point = (verts[v0] + verts[v1]) / 2  # Midpoint
    vertex_boundary_type[i] = 0  # Interior
```

### Key Functions

#### `extract_parting_surface(vertices, tetrahedra, edges, cut_edge_flags, ...)`
Main entry point. Returns `PartingSurfaceResult`.

#### `_get_5_edge_triangles(config, cut_edge_indices, vertices, edges, ...)`
Generates 5 triangles for 5-edge configurations using face vertex.

#### `_get_6_edge_triangles(cut_edge_indices, vertices, edges, ...)`
Generates 12 triangles for 6-edge configuration using inner vertex.

---

## surface_propagation.py

### Smoothing Algorithm (Section 4.4)

```python
for iteration in range(n_iterations):
    # Step 1: Smooth boundary vertices
    for vi in boundary_vertices:
        new_pos = laplacian_smooth(vi, neighbors, damping=0.5)
        
        # Re-project to ORIGINAL surface
        if vertex_boundary_type[vi] == -1:  # Inner boundary
            vertices[vi] = part_mesh.nearest_point(new_pos)
        elif vertex_boundary_type[vi] in (1, 2):  # Outer boundary
            vertices[vi] = hull_mesh.nearest_point(new_pos)
    
    # Step 2: Smooth interior vertices (boundary fixed)
    for vi in interior_vertices:
        vertices[vi] = laplacian_smooth(vi, neighbors, damping=0.5)
```

### Key Insight
The `vertex_boundary_type` array from extraction determines re-projection target. **Do NOT use closest-distance classification** - use the original boundary type.

### Gap Filling

After smoothing, boundary vertices are on surfaces but edges between them may "float":
```python
# Detect floating edges
for edge in boundary_edges:
    midpoint = (v0 + v1) / 2
    dist = part_mesh.nearest_distance(midpoint)
    if dist > threshold:
        # Edge is floating - fill with triangles
```

---

## parting_direction.py

### Visibility Computation

Uses Embree-accelerated raycasting:
```python
# For each candidate direction:
# 1. Compute which triangles face the direction (dot(normal, dir) > 0)
# 2. Ray cast from triangle centroids in direction
# 3. If no intersection → triangle is visible
```

### Direction Selection

1. Sample k directions on Fibonacci sphere
2. Score each direction by total visible area
3. Select d₁ = direction with maximum visible area
4. Select d₂ = direction with maximum visible area among those >135° from d₁

---

## pouring_direction.py

### Persistence Homology (Section 5.2)

#### Height Function
```python
f(v) = dot(v, direction)  # Project vertex onto pouring direction
```

#### Superlevel Set Filtration
Process vertices from highest to lowest:
- New component → birth (maximum)
- Components merge → death of younger (saddle)
- Record (maximum, saddle) pairs

#### Relevance Scoring
```python
# Trapped region = faces above saddle height, connected to maximum
# Tiltable region = faces that can drain with α-degree tilt
# Relevance = area(trapped - tiltable)
```

#### Direction Selection
1. Score all candidate directions
2. Select f₁, f₂ = two low-scoring, nearly-aligned directions for silicone
3. Select f_resin = lowest-scoring direction in 10° cone around bisector(f₁, f₂)

---

## Common Patterns

### Progress Reporting
```python
def compute_with_progress(data, progress_callback=None):
    total = len(data)
    for i, item in enumerate(data):
        process(item)
        if progress_callback:
            progress_callback(int(100 * i / total), f"Processing {i}/{total}")
```

### Error Handling
```python
def safe_operation(mesh):
    if mesh.vertices.shape[0] == 0:
        logger.error("Empty mesh provided")
        return None
    
    try:
        result = expensive_operation(mesh)
    except MemoryError:
        logger.error("Insufficient memory for %d vertices", len(mesh.vertices))
        return None
    
    return result
```

### Caching Expensive Computations
```python
# Many functions take optional precomputed data:
def compute_pairs(mesh, direction, vertex_neighbors=None):
    if vertex_neighbors is None:
        vertex_neighbors = build_vertex_neighbors(mesh)  # Expensive
    # Use vertex_neighbors...
```

---

## Testing

### Unit Test Template
```python
import pytest
import numpy as np
import trimesh

def test_marching_tet_3_edge():
    """Test 3-edge configuration produces single triangle."""
    # Create minimal test case
    vertices = np.array([[0,0,0], [1,0,0], [0.5,1,0], [0.5,0.5,1]])
    tetrahedra = np.array([[0,1,2,3]])
    
    # Set up cut flags for 3-edge config
    cut_flags = np.array([1,1,1,0,0,0], dtype=bool)
    
    result = extract_parting_surface(vertices, tetrahedra, ...)
    
    assert len(result.faces) == 1  # Single triangle
```

### Integration Test Template
```python
def test_full_pipeline_on_cube():
    """Test complete pipeline on simple cube."""
    mesh = trimesh.primitives.Box()
    
    # Run each step
    d1, d2 = find_parting_directions(mesh)
    hull = generate_inflated_hull(mesh)
    tet_result = tetrahedralize_hull(hull, mesh)
    classify_mold_halves(tet_result, d1, d2)
    compute_edge_weights(tet_result, mesh)
    run_dijkstra_escape_labeling(tet_result)
    find_primary_cutting_edges(tet_result)
    surface = extract_parting_surface(tet_result)
    
    # Verify results
    assert surface.mesh.is_watertight
    assert len(surface.faces) > 0
```
