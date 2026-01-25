# VcMoldCreator Documentation

This folder contains the research papers, implementation guides, and reference materials for the VcMoldCreator mold design system.

## Document Organization

### 📚 Research Papers (Reference)

These papers form the theoretical foundation of the implementation:

| Paper | Folder | Role in Project |
|-------|--------|-----------------|
| **Volume-Aware Design of Composite Molds** | `alderighi-2019-composite-molds/` | **PRIMARY** - Main algorithm paper (SIGGRAPH 2019) |
| Computing the Separating Surface for Segmented Data | `nielson-franke-1997-marching-tetrahedra/` | Marching Tetrahedra reference (Nielson & Franke 1997) |
| Polygonization of Non-Manifold Implicit Surfaces | `bloomenthal-1995-non-manifold/` | Face/inner vertex computation (Bloomenthal & Ferguson 1995) |
| Constrained Elastic Surface Nets | `gibson-1998-surface-nets/` | Alternative surface smoothing approach (Gibson 1998) |

### 📋 Implementation Guides

These documents describe how the papers are implemented:

| Document | Purpose |
|----------|---------|
| `PAPERS_TO_CODE_MAPPING.md` | **NEW** - Maps paper sections to code functions |
| `PRIMARY_SURFACE_MEMBRANE_IMPROVEMENTS.md` | Membrane extraction, smoothing, and gap filling |
| `POURING_DIRECTION_IMPLEMENTATION.md` | Persistence homology for bubble minimization |

### 📁 Other Folders

| Folder | Contents |
|--------|----------|
| `examples/` | Example STL files for testing (currently empty) |

---

## Paper Hierarchy and Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THEORETICAL FOUNDATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  PRIMARY PAPER: Volume-Aware Design of Composite Molds             │     │
│  │  Authors: Alderighi et al. (SIGGRAPH 2019)                         │     │
│  │                                                                     │     │
│  │  Provides:                                                          │     │
│  │  • Overall algorithm architecture                                   │     │
│  │  • Escape path computation (Section 4.5)                           │     │
│  │  • Parting direction selection (Section 4.1)                       │     │
│  │  • Membrane smoothing procedure (Section 4.4)                      │     │
│  │  • Pouring direction optimization (Section 5.2)                    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              │ References                                    │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SUPPORTING PAPERS (for Membrane Meshing - Section 4.3)             │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Nielson & Franke 1997                                        │   │    │
│  │  │  "Computing the Separating Surface for Segmented Data"        │   │    │
│  │  │                                                                │   │    │
│  │  │  Provides:                                                     │   │    │
│  │  │  • 5-case Marching Tetrahedra algorithm                       │   │    │
│  │  │  • Mid-edge, mid-face, mid-tetrahedron point formulas         │   │    │
│  │  │  • Case 3 (5-edge): face vertex triangulation                 │   │    │
│  │  │  • Case 4 (6-edge): inner vertex triangulation                │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Bloomenthal & Ferguson 1995                                  │   │    │
│  │  │  "Polygonization of Non-Manifold Implicit Surfaces"           │   │    │
│  │  │                                                                │   │    │
│  │  │  Provides:                                                     │   │    │
│  │  │  • Non-manifold surface handling (degree ≥3 edges)            │   │    │
│  │  │  • Face vertex computation via contour following              │   │    │
│  │  │  • Inner vertex for 4-region tetrahedra                       │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ALTERNATIVE APPROACHES (for reference/comparison)                   │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Gibson 1998                                                  │   │    │
│  │  │  "Constrained Elastic Surface Nets"                           │   │    │
│  │  │                                                                │   │    │
│  │  │  Provides:                                                     │   │    │
│  │  │  • Alternative to Marching Cubes/Tetrahedra                   │   │    │
│  │  │  • Elastic relaxation for smooth surfaces                     │   │    │
│  │  │  • Constraint preservation during smoothing                   │   │    │
│  │  │  NOTE: Not directly used, but useful reference                │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Which Paper for Which Problem?

| Problem | Primary Reference | Section |
|---------|------------------|---------|
| How to find parting directions d₁, d₂? | Alderighi 2019 | Section 4.1 |
| How to compute escape paths? | Alderighi 2019 | Section 4.5 |
| How to detect primary cut edges? | Alderighi 2019 | Section 4.1 |
| How to detect secondary cuts? | Alderighi 2019 | Section 4.2 |
| How to triangulate membrane (3/4-edge configs)? | Nielson & Franke 1997 | Cases 1-2 |
| How to handle 5-edge configurations? | Nielson & Franke 1997 | Case 3 |
| How to handle 6-edge configurations? | Nielson & Franke 1997 | Case 4 |
| How to compute face vertices? | Bloomenthal 1995 | Section 5 |
| How to compute inner vertices? | Bloomenthal 1995 | Section 5 |
| How to smooth membrane surface? | Alderighi 2019 | Section 4.4 |
| How to optimize pouring direction? | Alderighi 2019 | Section 5.2 |

---

## Key Algorithms by Paper

### Volume-Aware Design of Composite Molds (Alderighi 2019)

**Section 4.1 - Parting Surfaces:**
```
1. Sample k directions on Gaussian sphere (Fibonacci)
2. Compute visibility for each direction via GPU raycasting
3. Select d₁, d₂ minimizing non-visible area (>135° apart)
4. Partition ∂H into ∂H₁, ∂H₂ via region growing
5. Compute escape paths from interior to boundary
6. Mark edge as cut if endpoints escape to different halves
```

**Section 4.5 - Weighted Geodesics:**
```
edge_weight = 1 / (dist² + ε)    where ε = 0.25
edge_cost = edge_length × edge_weight

For boundary vertices:
  λ_w = R - dist(w, M)           where R = max hull-to-part distance
  biased_dist = dist + max(λ_w, 0)
  weight = 1 / (biased_dist² + ε)
```

**Section 4.4 - Membrane Smoothing:**
```
for each iteration:
    1. Smooth boundary vertices (polyline)
    2. Re-project boundary to original surface (M or ∂H)
    3. Smooth interior vertices (boundary fixed)
    
damping_factor = 0.5
```

**Section 5.2 - Pouring Direction:**
```
1. For direction f, compute height function h(v) = dot(v, f)
2. Find persistence pairs (maximum, saddle) via superlevel filtration
3. Score = sum of (trapped_area - tiltable_area) for relevant pairs
4. Select f₁, f₂ (silicone) = low-scoring, nearly-aligned
5. Select f_resin = lowest-scoring in cone around bisector(f₁, f₂)
```

### Nielson & Franke 1997 - Marching Tetrahedra

**Case Classification:**
```
Case 0: All 4 vertices same class → No surface
Case 1: 3+1 split (3 edges cut) → 1 triangle
Case 2: 2+2 split (4 edges cut) → 2 triangles (quad)
Case 3: 2+1+1 split (5 edges cut) → 5 triangles with face vertex
Case 4: 1+1+1+1 split (6 edges cut) → 12 triangles with inner vertex
```

**Mid-point Formulas:**
```
m_ij = (V_i + V_j) / 2                     // mid-edge
m_ijk = (m_ij + m_ik + m_jk) / 3           // mid-face (centroid of mid-edges)
m_t = (m_ijk + m_jkl + m_ikl + m_ijl) / 4  // mid-tetrahedron
```

### Bloomenthal & Ferguson 1995 - Non-Manifold Surfaces

**Face Vertex Location:**
```
When 3 regions meet on a face:
1. Start from edge vertex separating regions of interest
2. Follow face contour via small triangles
3. When "foreign" region encountered, place face vertex at triangle center
```

**Inner Vertex Location:**
```
When 4 regions meet in tetrahedron:
  inner_vertex = average of face vertices separating regions of interest
```

---

## Implementation Document Guide

### PRIMARY_SURFACE_MEMBRANE_IMPROVEMENTS.md

Covers:
- Full pipeline from upstream (import) to downstream (pouring)
- Data structure definitions (`TetrahedralMeshResult`, `PartingSurfaceResult`)
- Marching Tetrahedra implementation details
- Boundary-aware cut point placement
- Smoothing with re-projection
- Gap filling for floating edges
- 5-edge and 6-edge configuration handling

### POURING_DIRECTION_IMPLEMENTATION.md

Covers:
- Persistence homology theory
- Height function computation
- Critical point detection (maxima, saddles)
- Union-Find for component tracking
- Trapped region growing
- Tiltable region computation
- Direction scoring and selection
- UI integration

---

## Reading Order for New Developers

1. **Start here:** `alderighi-2019-composite-molds/Volume-Aware Design of Composite Molds.md`
   - Read Sections 1-3 for overview
   - Read Section 4 for core algorithm
   - Read Section 5 for fabrication

2. **For membrane meshing details:** `nielson-franke-1997-marching-tetrahedra/`
   - Focus on the 5-case algorithm
   - Study the mid-point formulas

3. **For non-manifold handling:** `bloomenthal-1995-non-manifold/`
   - Focus on face vertex and inner vertex computation
   - Understand when they're needed

4. **Implementation guides:**
   - `PRIMARY_SURFACE_MEMBRANE_IMPROVEMENTS.md` - How it's implemented
   - `POURING_DIRECTION_IMPLEMENTATION.md` - Pouring algorithm

---

## Citation Information

```bibtex
@article{alderighi2019volume,
  title={Volume-Aware Design of Composite Molds},
  author={Alderighi, Thomas and Malomo, Luigi and Giorgi, Daniela and 
          Bickel, Bernd and Cignoni, Paolo and Pietroni, Nico},
  journal={ACM Transactions on Graphics (TOG)},
  volume={38},
  number={4},
  pages={1--12},
  year={2019},
  publisher={ACM}
}

@inproceedings{nielson1997computing,
  title={Computing the separating surface for segmented data},
  author={Nielson, Gregory M and Franke, Richard},
  booktitle={Proceedings of Visualization'97},
  pages={229--233},
  year={1997},
  organization={IEEE}
}

@inproceedings{bloomenthal1995polygonization,
  title={Polygonization of non-manifold implicit surfaces},
  author={Bloomenthal, Jules and Ferguson, Keith},
  booktitle={Proceedings of SIGGRAPH'95},
  pages={309--316},
  year={1995},
  organization={ACM}
}

@inproceedings{gibson1998constrained,
  title={Constrained elastic surface nets: Generating smooth surfaces 
         from binary segmented data},
  author={Gibson, Sarah FF},
  booktitle={International Conference on Medical Image Computing and 
             Computer-Assisted Intervention},
  pages={888--898},
  year={1998},
  organization={Springer}
}
```
