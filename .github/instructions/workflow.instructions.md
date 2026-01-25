---
applyTo: '**'
---

# VcMoldCreator Development Workflow

## Pipeline Execution Order

The mold creation algorithm must execute steps in strict order due to data dependencies:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MOLD DESIGN PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: IMPORT & VALIDATION                                          │   │
│  │ Input: STL file                                                       │   │
│  │ Output: trimesh.Trimesh (validated manifold mesh M)                  │   │
│  │ Files: stl_loader.py, mesh_repair.py                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: PARTING DIRECTION                                            │   │
│  │ Input: Mesh M                                                         │   │
│  │ Output: d₁, d₂ (unit vectors), visibility data                       │   │
│  │ Algorithm: Fibonacci sphere sampling + GPU visibility                │   │
│  │ Files: parting_direction.py                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: INFLATED HULL                                                │   │
│  │ Input: Mesh M, offset distance                                        │   │
│  │ Output: Hull mesh H, offset surface ∂H                               │   │
│  │ Algorithm: Convex hull inflation                                     │   │
│  │ Files: inflated_hull.py                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: TETRAHEDRALIZE                                               │   │
│  │ Input: Hull mesh H, part mesh M                                      │   │
│  │ Output: TetrahedralMeshResult (vertices, tets, edges)                │   │
│  │ Algorithm: fTetWild via pytetwild                                    │   │
│  │ Files: tetrahedral_mesh.py                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: MOLD HALF CLASSIFICATION                                     │   │
│  │ Input: TetrahedralMeshResult, d₁, d₂                                 │   │
│  │ Output: boundary_labels (N,) with values -1, 0, 1, 2                 │   │
│  │ Algorithm: Region growing from seed faces                            │   │
│  │ Files: mold_half_classification.py                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 6: EDGE WEIGHTS                                                 │   │
│  │ Input: TetrahedralMeshResult, part mesh M                            │   │
│  │ Output: edge_weights, weighted_edge_lengths                          │   │
│  │ Algorithm: 1/(dist²+ε) with ∂F boundary bias                        │   │
│  │ Files: tetrahedral_mesh.py                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 7: DIJKSTRA ESCAPE PATHS                                        │   │
│  │ Input: Weighted graph, boundary_labels                               │   │
│  │ Output: seed_escape_labels, seed_escape_paths, primary_cut_edges     │   │
│  │ Algorithm: Dijkstra from interior vertices to H₁/H₂                  │   │
│  │ Files: tetrahedral_mesh.py                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 8: PRIMARY SURFACE EXTRACTION                                   │   │
│  │ Input: cut_edge_flags, tet_edge_indices                              │   │
│  │ Output: PartingSurfaceResult (mesh, vertex_boundary_type)            │   │
│  │ Algorithm: Extended Marching Tetrahedra (Nielson & Franke)           │   │
│  │ Files: parting_surface.py                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 9: MEMBRANE SMOOTHING                                           │   │
│  │ Input: Extracted surface, part mesh M, hull mesh ∂H                  │   │
│  │ Output: Smoothed membrane mesh                                       │   │
│  │ Algorithm: Laplacian smoothing with boundary re-projection           │   │
│  │ Files: surface_propagation.py                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 10: SECONDARY CUT DETECTION                                     │   │
│  │ Input: Escape paths, part mesh M                                     │   │
│  │ Output: secondary_cut_edges                                          │   │
│  │ Algorithm: Minimal surface intersection test                         │   │
│  │ Files: tetrahedral_mesh.py                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 11: SECONDARY SURFACE EXTRACTION                                │   │
│  │ Input: secondary_cut_edges                                           │   │
│  │ Output: Secondary membrane meshes                                    │   │
│  │ Algorithm: Same as primary surface                                   │   │
│  │ Files: parting_surface.py                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 12: POURING DIRECTION OPTIMIZATION                              │   │
│  │ Input: Mold half meshes                                              │   │
│  │ Output: f₁, f₂ (silicone), f_resin (resin pouring directions)       │   │
│  │ Algorithm: Persistence homology for bubble trapping                  │   │
│  │ Files: pouring_direction.py                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OUTPUT: Complete cut layout (parting surface + additional membranes)       │
│          + optimal pouring directions                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Dependencies

### Required Data for Each Step

| Step | Requires | Produces |
|------|----------|----------|
| Import | STL file path | `trimesh.Trimesh` (M) |
| Parting Direction | M | d₁, d₂, visibility_paint_data |
| Inflated Hull | M, offset | hull_mesh (H), InflatedHullResult |
| Tetrahedralize | H, M | TetrahedralMeshResult |
| Mold Halves | tet_result, d₁, d₂ | boundary_labels |
| Edge Weights | tet_result, M | edge_weights, weighted_edge_lengths |
| Dijkstra | weighted_graph, boundary_labels | seed_escape_labels, primary_cut_edges |
| Primary Surface | cut_edge_flags | PartingSurfaceResult |
| Smoothing | surface, M, H | smoothed_mesh |
| Secondary Cuts | escape_paths, M | secondary_cut_edges |
| Secondary Surface | secondary_cut_edges | secondary_membrane |
| Pouring | mold_halves | f₁, f₂, f_resin |

---

## Feature Implementation Checklist

### ✅ Implemented Features

#### Core Pipeline
- [x] STL loading with validation
- [x] Mesh repair for non-manifold issues
- [x] Mesh decimation for large models
- [x] Parting direction via visibility (Fibonacci + Embree)
- [x] Inflated hull generation
- [x] Tetrahedral meshing via fTetWild
- [x] Mold half classification (region growing)
- [x] Edge weight computation (1/(d²+ε) + ∂F bias)
- [x] Dijkstra escape path labeling
- [x] Primary cut edge detection
- [x] Marching Tetrahedra (0, 3, 4, 5, 6 edge configs)
- [x] Boundary-aware cut point placement
- [x] Membrane smoothing with re-projection
- [x] Floating edge gap filling
- [x] Small hole removal
- [x] Orphan island removal
- [x] Tubular section detection
- [x] Thin flat section removal
- [x] Self-folding repair

#### Pouring Direction
- [x] Persistence pairing (max-saddle)
- [x] Trapped region growing
- [x] Tiltable region computation
- [x] Direction scoring
- [x] Silicone direction selection
- [x] Resin direction selection (cone around bisector)

#### UI/Visualization
- [x] PyQt6 main window with step panels
- [x] PyVista 3D viewer
- [x] Progress tracking
- [x] Console logging panel
- [x] Visibility painting
- [x] Cut edge visualization

### 🔄 Partially Implemented

- [ ] Secondary membrane extraction (cut detection done, meshing WIP)
- [ ] Session save/load (partial .vcm format)

### ❌ Not Yet Implemented

#### Downstream Fabrication
- [ ] Hard shell generation
- [ ] Metamold geometry
- [ ] Air vent placement
- [ ] Perlin noise registration on seams
- [ ] Export for 3D printing

#### Advanced Features
- [ ] Multi-piece molds (>2 pieces)
- [ ] Adaptive tetrahedral resolution
- [ ] Real-time preview during computation

---

## Development Tasks By Priority

### High Priority (Core Functionality)
1. **Complete secondary surface extraction** - Connect secondary_cut_edges to marching tets
2. **Session persistence** - Save/load full computation state
3. **Batch processing** - Process multiple STL files

### Medium Priority (Quality Improvements)
4. **Performance optimization** - Profile and optimize hot paths
5. **GPU acceleration** - CUDA for Dijkstra and visibility
6. **Adaptive mesh resolution** - Variable tet density based on feature size

### Low Priority (Fabrication Pipeline)
7. **Hard shell export** - Generate printable shell geometry
8. **Metamold generation** - Create casting containers
9. **Air channel placement** - Add vents and pour spouts

---

## Testing Strategy

### Unit Tests
- Test each algorithm step in isolation
- Use synthetic test meshes (cubes, spheres, tori)
- Verify boundary conditions and edge cases

### Integration Tests
- Run full pipeline on reference models
- Compare output to known-good results
- Check mesh validity at each step

### Performance Tests
- Benchmark on models of varying complexity
- Profile memory usage
- Test parallel processing

### Visual Validation
- Inspect membrane continuity
- Verify boundary touching
- Check smoothing quality

---

## Debugging Workflow

### Common Issues and Solutions

#### "Membrane has holes"
1. Check cut_edge_flags for missing edges
2. Verify all vertices have escape labels
3. Look for skipped 5/6-edge configurations
4. Check boundary zone exclusion

#### "Membrane doesn't touch part"
1. Verify vertex_boundary_type == -1 for inner boundary
2. Check re-projection in smoothing step
3. Inspect cut point placement

#### "Dijkstra produces incorrect labels"
1. Verify boundary_labels assignment
2. Check edge weight computation
3. Ensure boundary zone vertices excluded
4. Inspect seed selection

#### "Surface self-intersects"
1. Check diagonal choice consistency in 4-edge configs
2. Verify winding order of triangles
3. Inspect smoothing for fold-over

### Visualization Debugging

```python
# Add to any step to visualize intermediate result:
import pyvista as pv

plotter = pv.Plotter()
plotter.add_mesh(mesh, color='white', opacity=0.5)
plotter.add_points(vertices[labels == 1], color='blue', point_size=5)
plotter.add_points(vertices[labels == 2], color='red', point_size=5)
plotter.show()
```

---

## Code Quality Standards

### Documentation
- Every function needs docstring with:
  - One-line summary
  - Args section with types
  - Returns section
  - Paper reference if applicable

### Error Handling
```python
# Good:
if mesh.vertices.shape[0] == 0:
    raise ValueError("Cannot process empty mesh")

# Also good:
try:
    result = compute_expensive_operation()
except MemoryError:
    logger.error("Insufficient memory for mesh with %d vertices", n_verts)
    return None
```

### Logging
```python
# Use appropriate levels:
logger.debug("Processing vertex %d of %d", i, n)  # Verbose iteration
logger.info("Computed %d cut edges", len(cut_edges))  # Milestone
logger.warning("Skipping 5-edge config %d", config)  # Potential issue
logger.error("Failed to extract surface: %s", str(e))  # Recoverable error
```

### Threading
```python
# Long operations must use QThread:
class ComputeWorker(QThread):
    progress = pyqtSignal(int, str)  # (percent, message)
    finished = pyqtSignal(object)  # result object
    error = pyqtSignal(str)  # error message
    
    def run(self):
        try:
            result = do_computation()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

---

## Reference Implementation Mapping

| Paper Section | Implementation File | Key Function |
|---------------|---------------------|--------------|
| 4.1 Parting Surface | tetrahedral_mesh.py | `find_primary_cutting_edges()` |
| 4.1 Direction Selection | parting_direction.py | `find_parting_directions()` |
| 4.2 Additional Membranes | tetrahedral_mesh.py | `find_secondary_cutting_edges()` |
| 4.3 Membrane Meshing | parting_surface.py | `extract_parting_surface()` |
| 4.4 Membrane Smoothing | surface_propagation.py | `smooth_membrane_with_boundary_reprojection()` |
| 4.5 Weighted Geodesics | tetrahedral_mesh.py | `compute_edge_weights()` |
| 5.2 Pouring Direction | pouring_direction.py | `find_optimal_pouring_directions()` |
