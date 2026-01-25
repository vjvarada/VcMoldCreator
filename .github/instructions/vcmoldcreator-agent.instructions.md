---
applyTo: '**'
---

# VcMoldCreator Development Agent Instructions

## Project Overview

VcMoldCreator is a computational mold design application that automatically designs composite, two-piece silicone molds for casting highly complex 3D shapes. The application implements the algorithm from **"Volume-Aware Design of Composite Molds"** by Alderighi et al. (SIGGRAPH 2019).

### Core Value Proposition

The system enables casting of objects with:
- Highly complex geometric features (thin protrusions, deep undercuts)
- Non-zero genus topology (objects with holes)
- Knotted surfaces and intertwined pieces
- Shapes that cannot be cast with traditional rigid molds

### Architecture

The project has three main components:

1. **Desktop Application** (`desktop_app/`) - Python/PyQt6 desktop app with full pipeline
2. **Web Frontend** (`frontend/`) - TypeScript/React/Three.js web interface
3. **Backend** (`backend/`) - Python/FastAPI server for web deployment

### Documentation Resources

For detailed paper-to-code mappings and algorithm references, see:
- `docs/README.md` - Documentation index and paper guide
- `docs/PAPERS_TO_CODE_MAPPING.md` - Maps paper sections to code functions
- `docs/alderighi-2019-composite-molds/` - Primary reference paper (SIGGRAPH 2019)
- `docs/nielson-franke-1997-marching-tetrahedra/` - Marching Tetrahedra algorithm
- `docs/bloomenthal-1995-non-manifold/` - Non-manifold surface handling

---

## Core Algorithm Pipeline

The mold design algorithm follows these sequential steps:

### Step 1: Input Processing
- **Input**: Triangulated manifold surface mesh M (the part to cast)
- **Operations**: Load STL, validate mesh, optionally repair non-manifold issues
- **Files**: `core/stl_loader.py`, `core/mesh_repair.py`, `core/mesh_analysis.py`

### Step 2: Inflated Hull Creation
- **Purpose**: Create bounding volume for the mold
- **Algorithm**: Compute offset surface of M, create tetrahedral tessellation H of convex hull
- **Output**: Mold volume O (space between M and ∂H)
- **Files**: `core/inflated_hull.py`, `core/tetrahedral_mesh.py`

### Step 3: Parting Direction Selection (Paper Section 4.1)
- **Purpose**: Find optimal directions d₁, d₂ for mold opening
- **Algorithm**: 
  - Sample k candidate directions on Gaussian sphere (Fibonacci sampling)
  - GPU-accelerated visibility computation for each direction
  - Select d₁, d₂ that minimize non-visible surface area
- **Files**: `core/parting_direction.py`

### Step 4: Mold Half Classification
- **Purpose**: Partition hull boundary ∂H into ∂H₁ and ∂H₂
- **Algorithm**:
  - Select seed faces F₁, F₂ with normals best aligned to d₁, d₂
  - Greedy region-growing from seeds based on normal alignment
  - Create boundary zone (15% of bbox diagonal) between H₁/H₂
- **Files**: `core/mold_half_classification.py`

### Step 5: Escape Path Computation (Paper Section 4.5)
- **Purpose**: Find shortest paths from interior vertices to boundary
- **Algorithm**:
  - Weighted geodesics: `edge_cost = length × 1/(dist²+ε)`
  - Dijkstra from interior vertices to ∂H₁ or ∂H₂
  - Label vertices based on which boundary they reach
- **Key Formula**: Weight pushes membranes away from part surface
- **Files**: `core/tetrahedral_mesh.py`

### Step 6: Primary Cut Edge Detection (Paper Section 4.1)
- **Purpose**: Find edges where parting surface passes through
- **Algorithm**: Edge (vᵢ,vⱼ) is primary cut if wᵢ and wⱼ (escape destinations) are on different parts ∂H₁, ∂H₂
- **Files**: `core/tetrahedral_mesh.py`

### Step 7: Membrane Meshing (Paper Section 4.3)
- **Purpose**: Extract triangulated parting surface from cut edges
- **Algorithm**: Extended Marching Tetrahedra
  - 3-edge configs → single triangle
  - 4-edge configs → quad (2 triangles)
  - 5-edge configs → 5 triangles (face vertex)
  - 6-edge configs → 12 triangles (face + inner vertices)
- **References**: Nielson & Franke 1997, Bloomenthal & Ferguson 1995
- **Files**: `core/parting_surface.py`

### Step 8: Membrane Smoothing (Paper Section 4.4)
- **Purpose**: Smooth membrane while preserving boundaries
- **Algorithm**:
  1. Smooth boundary polyline vertices
  2. Re-project boundary vertices to original surface (M or ∂H)
  3. Smooth interior vertices with boundary fixed
  4. Damping factor 0.5 for convergence
- **Files**: `core/surface_propagation.py`

### Step 9: Secondary Membrane Detection (Paper Section 4.2)
- **Purpose**: Detect additional cuts needed for features preventing extraction
- **Algorithm**: For same-label edges, compute minimal surface bounded by escape paths; if intersects M, add secondary cut
- **Files**: `core/tetrahedral_mesh.py`

### Step 10: Pouring Direction Optimization (Paper Section 5.2)
- **Purpose**: Find optimal silicone/resin pouring directions to minimize air bubbles
- **Algorithm**: Persistence homology for height function maxima
  - Compute maximum-saddle pairs
  - Score based on trapped air region area accounting for tilt
  - Select aligned low-scoring directions for silicone
  - Select resin direction in cone around bisector
- **Files**: `core/pouring_direction.py`

---

## Key Mathematical Concepts

### Weighted Geodesics (Paper Section 4.5)
```
edge_weight = 1 / (dist² + ε)
edge_cost = edge_length × edge_weight
```
Where:
- `dist` = geodesic distance from edge midpoint to part surface M
- `ε` = 0.25 (prevents division by zero)

This ensures membranes tend to be perpendicular to the part surface.

### Exterior Boundary Reshape (Paper Figure 8)
```
λ_w = R - dist(w, M)  // distance from hull vertex to offset surface
R = max distance from hull vertices to part mesh
biased_distance = δ + λ_w  // add bias to computed distance
```

### Persistence Homology for Pouring
- Height function: f(v) = dot(v, direction)
- Superlevel sets: M_a = f⁻¹(a, +∞)
- Maximum-saddle pairs identify bubble-trapping features
- Relevance score = area of (A_m^n - T) accounting for tilt angle α

---

## Data Structures

### TetrahedralMeshResult
```python
@dataclass
class TetrahedralMeshResult:
    vertices: np.ndarray              # (N, 3) vertex positions
    tetrahedra: np.ndarray            # (T, 4) tet vertex indices
    edges: np.ndarray                 # (E, 2) unique edges
    boundary_labels: np.ndarray       # (N,) -1=part, 0=interior, 1=H1, 2=H2
    edge_weights: np.ndarray          # (E,) 1/(dist²+ε) weights
    seed_escape_labels: np.ndarray    # (I,) 1=H1, 2=H2, 0=unreachable
    primary_cut_edges: List[Tuple[int, int]]
    secondary_cut_edges: List[Tuple[int, int]]
    cut_edge_flags: np.ndarray        # (E,) bool
```

### PartingSurfaceResult
```python
@dataclass
class PartingSurfaceResult:
    mesh: trimesh.Trimesh
    vertices: np.ndarray              # (V, 3) surface vertices
    faces: np.ndarray                 # (F, 3) surface faces
    vertex_boundary_type: np.ndarray  # (V,) -1=part, 0=interior, 1/2=hull
```

---

## Implementation Guidelines

### When Working on Marching Tetrahedra (`parting_surface.py`)

1. **Edge Configuration Encoding**: 6 bits, one per edge (0-5)
   - Edge 0: (v0,v1), Edge 1: (v0,v2), Edge 2: (v0,v3)
   - Edge 3: (v1,v2), Edge 4: (v1,v3), Edge 5: (v2,v3)

2. **Valid Configurations for Binary Labeling**:
   - 0-edge: No surface (trivial)
   - 3-edge: One vertex isolated → 1 triangle
   - 4-edge: 2v2 split → 2 triangles (quad)
   - 5-edge: Requires face vertex → 5 triangles
   - 6-edge: Requires inner vertex → 12 triangles

3. **Boundary-Aware Cut Point Placement**:
   - If edge touches part M → place vertex ON part, track as inner boundary (-1)
   - If edge touches hull ∂H → place vertex ON hull, track as outer boundary (1/2)
   - Otherwise → use midpoint, track as interior (0)

### When Working on Smoothing (`surface_propagation.py`)

1. **Alternating Smoothing Order**:
   - First: Smooth boundary vertices only
   - Then: Re-project boundary vertices to ORIGINAL surface (use vertex_boundary_type)
   - Finally: Smooth interior vertices with boundary fixed

2. **Re-projection Rules**:
   - vertex_boundary_type == -1 → re-project to part mesh M
   - vertex_boundary_type in (1, 2) → re-project to hull mesh ∂H
   - vertex_boundary_type == 0 → interior, no constraint

### When Working on Dijkstra/Escape Paths (`tetrahedral_mesh.py`)

1. **Boundary Zone Exclusion**: Vertices within 15% of bbox diagonal from H₁/H₂ interface are NOT valid escape destinations

2. **Weight Computation**:
   ```python
   weights = 1.0 / (dist_to_part ** 2 + 0.25)
   weighted_edge_lengths = edge_lengths * weights
   ```

3. **Escape Labeling**: Run Dijkstra from interior vertices, target vertices are those with boundary_labels in (1, 2)

---

## Constants and Thresholds

| Constant | Value | Source | Purpose |
|----------|-------|--------|---------|
| `EDGE_WEIGHT_EPSILON` | 0.25 | Paper 4.5 | Prevents division by zero |
| `BOUNDARY_ZONE_THRESHOLD` | 0.15 | Paper 4.1 | 15% of bbox diagonal |
| `SMOOTHING_DAMPING` | 0.5 | Paper 4.4 | Laplacian smoothing damping |
| `TILT_ANGLE_DEFAULT` | 10.0° | Paper 5.2 | Assumed mold tilting capability |
| `AREA_THRESHOLD` | 0.5 mm² | Paper 5.2 | Min trapped area to consider |

---

## Dependencies

### Python (Desktop App)
- **trimesh**: Mesh loading, manipulation, ray intersection
- **numpy**: Numerical operations
- **PyQt6**: GUI framework
- **pyvista**: 3D visualization
- **pytetwild**: Tetrahedral meshing (fTetWild bindings)
- **embreex**: Fast ray tracing (optional but recommended)

### TypeScript (Frontend)
- **three.js**: 3D rendering
- **react**: UI framework
- **vite**: Build tool

---

## File Organization

```
desktop_app/
├── main.py                    # Application entry point
├── core/
│   ├── stl_loader.py         # STL file loading
│   ├── mesh_analysis.py      # Mesh analysis and diagnostics
│   ├── mesh_repair.py        # Mesh repair operations
│   ├── mesh_decimation.py    # Mesh simplification
│   ├── inflated_hull.py      # Hull generation
│   ├── tetrahedral_mesh.py   # Tet meshing, Dijkstra, cut edges
│   ├── parting_direction.py  # Visibility-based direction finding
│   ├── mold_half_classification.py  # H1/H2 classification
│   ├── parting_surface.py    # Marching tetrahedra, surface extraction
│   ├── surface_propagation.py # Smoothing, boundary repair
│   └── pouring_direction.py  # Persistence-based direction optimization
├── ui/
│   └── main_window.py        # Main application window
└── viewer/
    └── mesh_viewer.py        # 3D visualization widget
```

---

## Development Workflow

### When Adding New Features

1. **Understand the Paper**: Reference the specific section(s) of the Alderighi paper
2. **Check Dependencies**: Identify upstream steps that must complete first
3. **Data Structures**: Use existing dataclasses or add fields as needed
4. **Threading**: Long operations should use QThread with progress signals
5. **Visualization**: Add viewer support for debugging/validation
6. **Testing**: Add unit tests for algorithmic correctness

### When Debugging

1. **Visualize Intermediate Results**: Use PyVista to inspect meshes at each step
2. **Check Boundary Labels**: Most issues trace back to incorrect boundary classification
3. **Validate Configurations**: For Marching Tets, log skipped configurations
4. **Monitor Dijkstra**: Verify escape paths reach correct boundary halves

---

## References

### Primary Paper
- Alderighi et al., "Volume-Aware Design of Composite Molds", SIGGRAPH 2019

### Referenced Algorithms
- Nielson & Franke, "Computing the Separating Surface for Segmented Data", 1997
- Bloomenthal & Ferguson, "Polygonization of Non-Manifold Implicit Surfaces", 1995
- Gibson, "Constrained Elastic Surface Nets", 1998
- Edelsbrunner et al., "Topological Persistence and Simplification", 2000

---

## Coding Standards

1. **Type Hints**: Use Python type hints for all function signatures
2. **Dataclasses**: Use @dataclass for result structures
3. **Logging**: Use the logging module, not print statements
4. **Documentation**: Docstrings with Args, Returns, and algorithm references
5. **Constants**: Define at module level with descriptive names
6. **Error Handling**: Graceful degradation with informative messages
