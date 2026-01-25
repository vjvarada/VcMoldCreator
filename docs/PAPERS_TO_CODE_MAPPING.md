# Papers to Code Mapping

This document maps academic paper sections to their implementation in the codebase.
Use this to understand WHERE each algorithm is implemented.

---

## Alderighi et al. 2019 - Volume-Aware Design of Composite Molds

### Section 4.1 - Parting Surfaces and Direction Selection

| Paper Concept | Implementation | Function/Class |
|---------------|----------------|----------------|
| Fibonacci sphere sampling | `parting_direction.py` | `fibonacci_sphere()` |
| GPU-accelerated visibility | `parting_direction.py` | `compute_visibility_for_direction()` |
| Optimal direction pair (d₁, d₂) | `parting_direction.py` | `find_parting_directions()` |
| Region growing for ∂H₁/∂H₂ | `mold_half_classification.py` | `classify_mold_halves()` |
| Primary cut edge detection | `tetrahedral_mesh.py` | `find_primary_cutting_edges()` |

**Key Formulas:**
- Select d₁, d₂ minimizing non-visible surface area
- Ensure d₁ · d₂ < cos(135°) = -0.707

### Section 4.2 - Additional Membranes (Secondary Cuts)

| Paper Concept | Implementation | Function/Class |
|---------------|----------------|----------------|
| Secondary cut detection | `tetrahedral_mesh.py` | `find_secondary_cutting_edges()` |
| Minimal surface intersection test | `tetrahedral_mesh.py` | (within secondary detection) |

**Condition:** For same-label edges, if minimal surface bounded by escape paths intersects M → add secondary cut

### Section 4.3 - Membrane Meshing (Marching Tetrahedra)

| Paper Concept | Implementation | Function/Class |
|---------------|----------------|----------------|
| Extended Marching Tetrahedra | `parting_surface.py` | `extract_parting_surface()` |
| 3-edge configurations | `parting_surface.py` | `_generate_triangles_config_3()` |
| 4-edge configurations | `parting_surface.py` | `_generate_triangles_config_4()` |
| 5-edge configurations (face vertex) | `parting_surface.py` | `_generate_triangles_config_5()` |
| 6-edge configurations (inner vertex) | `parting_surface.py` | `_generate_triangles_config_6()` |
| Cut point placement | `parting_surface.py` | `_compute_cut_points()` |

**References Nielson & Franke 1997 for:**
- Mid-edge point: m_ij = (V_i + V_j) / 2
- Mid-face point: m_ijk = (m_ij + m_ik + m_jk) / 3
- Mid-tetrahedron: m_t = average of 4 face points

### Section 4.4 - Membrane Smoothing

| Paper Concept | Implementation | Function/Class |
|---------------|----------------|----------------|
| Boundary polyline smoothing | `surface_propagation.py` | `_smooth_boundary_vertices()` |
| Boundary re-projection | `surface_propagation.py` | `_reproject_boundary_vertices()` |
| Interior vertex smoothing | `surface_propagation.py` | `_smooth_interior_vertices()` |
| Complete smoothing loop | `surface_propagation.py` | `smooth_membrane_with_boundary_reprojection()` |

**Algorithm:**
```
for each iteration:
    smooth_boundary_vertices()
    reproject_boundary_vertices()  # to M or ∂H based on type
    smooth_interior_vertices()     # boundary fixed
damping = 0.5
```

### Section 4.5 - Weighted Geodesics (Escape Paths)

| Paper Concept | Implementation | Function/Class |
|---------------|----------------|----------------|
| Edge weight computation | `tetrahedral_mesh.py` | `compute_edge_weights()` |
| Exterior boundary bias (λ_w) | `tetrahedral_mesh.py` | (within edge weights) |
| Dijkstra from interior to ∂H | `tetrahedral_mesh.py` | `compute_escape_paths()` |
| Escape path labeling | `tetrahedral_mesh.py` | `compute_escape_labels()` |

**Key Formulas:**
```python
weight = 1 / (dist_to_part² + 0.25)  # ε = 0.25
λ_w = R - dist(w, M)                  # boundary bias
biased_dist = dist + max(λ_w, 0)
```

### Section 5.2 - Pouring Direction Optimization

| Paper Concept | Implementation | Function/Class |
|---------------|----------------|----------------|
| Height function h(v) | `pouring_direction.py` | `compute_height_function()` |
| Persistence pairing | `pouring_direction.py` | `compute_persistence_pairs()` |
| Trapped region identification | `pouring_direction.py` | `grow_trapped_region()` |
| Tiltable region computation | `pouring_direction.py` | `compute_tiltable_region()` |
| Direction scoring | `pouring_direction.py` | `score_pouring_direction()` |
| Optimal direction selection | `pouring_direction.py` | `find_optimal_pouring_directions()` |

**Algorithm:**
1. For each candidate direction f:
   - Compute height function h(v) = v · f
   - Find (maximum, saddle) persistence pairs
   - Score = Σ (trapped_area - tiltable_area)
2. Select f₁, f₂ (silicone) = lowest scoring, nearly aligned
3. Select f_resin = lowest scoring in cone around bisector(f₁, f₂)

---

## Nielson & Franke 1997 - Computing the Separating Surface

### Case Classification (5 Cases)

| Case | Edge Count | Implementation | Triangles Generated |
|------|------------|----------------|---------------------|
| Case 0 | 0 edges | Skip | 0 |
| Case 1 | 3 edges | `_generate_triangles_config_3()` | 1 |
| Case 2 | 4 edges | `_generate_triangles_config_4()` | 2 (quad) |
| Case 3 | 5 edges | `_generate_triangles_config_5()` | 5 |
| Case 4 | 6 edges | `_generate_triangles_config_6()` | 12 |

### Mid-Point Formulas

| Point Type | Formula | Usage |
|------------|---------|-------|
| Mid-edge m_ij | (V_i + V_j) / 2 | Cut point on edge |
| Mid-face m_ijk | (m_ij + m_ik + m_jk) / 3 | Face vertex (Case 3) |
| Mid-tet m_t | avg(4 face centroids) | Inner vertex (Case 4) |

---

## Bloomenthal & Ferguson 1995 - Non-Manifold Surfaces

### Face Vertex Computation

| Concept | When Needed | Implementation |
|---------|-------------|----------------|
| Face vertex | 3 regions meet on face (5-edge) | `_compute_face_vertex()` |
| Contour following | Locate foreign region | Part of face vertex logic |

**Algorithm:** Follow triangles on face until "foreign" region found → place vertex

### Inner Vertex Computation

| Concept | When Needed | Implementation |
|---------|-------------|----------------|
| Inner vertex | 4 regions meet in tet (6-edge) | `_compute_inner_vertex()` |

**Formula:** inner = average of face vertices for regions of interest

---

## Gibson 1998 - Constrained Elastic Surface Nets

### Relevant Concepts (Reference Only)

| Concept | Potential Application |
|---------|----------------------|
| Elastic relaxation | Alternative to Laplacian smoothing |
| Constraint preservation | Boundary-preserving smoothing |
| Quad-based surface | Alternative to triangle meshing |

*Note: Not directly implemented, but useful for understanding smoothing alternatives.*

---

## Data Flow Through Code

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW: PAPER → CODE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STL File                                                                   │
│       │                                                                      │
│       ▼                                                                      │
│   stl_loader.py → trimesh.Trimesh                                           │
│       │                                                                      │
│       ▼                                                                      │
│   parting_direction.py → (d₁, d₂, visibility_data)         [Paper §4.1]    │
│       │                                                                      │
│       ▼                                                                      │
│   inflated_hull.py → hull_mesh                                              │
│       │                                                                      │
│       ▼                                                                      │
│   tetrahedral_mesh.py → TetrahedralMeshResult               [Paper §4.5]    │
│       │                                                                      │
│       ├──→ compute_edge_weights()                           [Paper §4.5]    │
│       │                                                                      │
│       ├──→ compute_escape_paths()                           [Paper §4.5]    │
│       │                                                                      │
│       └──→ find_primary_cutting_edges()                     [Paper §4.1]    │
│               │                                                              │
│               ▼                                                              │
│   mold_half_classification.py → boundary_labels             [Paper §4.1]    │
│               │                                                              │
│               ▼                                                              │
│   parting_surface.py → PartingSurfaceResult                 [Paper §4.3]    │
│       │                                                     [N&F 1997]      │
│       │                                                     [B&F 1995]      │
│       ▼                                                                      │
│   surface_propagation.py → smoothed_membrane                [Paper §4.4]    │
│       │                                                                      │
│       ▼                                                                      │
│   pouring_direction.py → (f₁, f₂, f_resin)                  [Paper §5.2]    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Lookup: "I want to change X"

| If you want to change... | Look at... |
|--------------------------|------------|
| How directions are sampled | `parting_direction.py` → `fibonacci_sphere()` |
| How visibility is computed | `parting_direction.py` → `compute_visibility_for_direction()` |
| How the hull is created | `inflated_hull.py` |
| How edge weights are calculated | `tetrahedral_mesh.py` → `compute_edge_weights()` |
| How escape paths are found | `tetrahedral_mesh.py` → `compute_escape_paths()` |
| How cut edges are detected | `tetrahedral_mesh.py` → `find_primary_cutting_edges()` |
| How triangles are generated | `parting_surface.py` → `_generate_triangles_config_*()` |
| How smoothing works | `surface_propagation.py` → `smooth_membrane_with_boundary_reprojection()` |
| How pouring directions are scored | `pouring_direction.py` → `score_pouring_direction()` |
| How the UI pipeline works | `main_window.py` → look for `_on_*_clicked()` methods |
