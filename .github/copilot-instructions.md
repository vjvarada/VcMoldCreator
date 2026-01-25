# VcMoldCreator - Repository Instructions for GitHub Copilot

## Project Summary

VcMoldCreator is a computational mold design application implementing the algorithm from **"Volume-Aware Design of Composite Molds"** (Alderighi et al., SIGGRAPH 2019). It automatically designs composite, two-piece silicone molds for casting complex 3D shapes with thin protrusions, deep undercuts, and non-zero genus topology.

## Project Structure

```
VcMoldCreator/
├── desktop_app/           # Main Python/PyQt6 desktop application
│   ├── main.py            # Application entry point
│   ├── core/              # Algorithm implementations
│   │   ├── stl_loader.py          # STL file loading
│   │   ├── mesh_repair.py         # Mesh repair operations
│   │   ├── inflated_hull.py       # Hull generation
│   │   ├── tetrahedral_mesh.py    # Tet meshing, Dijkstra, cut edges
│   │   ├── parting_direction.py   # Visibility-based direction finding
│   │   ├── mold_half_classification.py  # H1/H2 classification
│   │   ├── parting_surface.py     # Marching Tetrahedra surface extraction
│   │   ├── surface_propagation.py # Membrane smoothing
│   │   └── pouring_direction.py   # Persistence-based pouring optimization
│   ├── ui/main_window.py  # Main PyQt6 window (~9400 lines)
│   └── viewer/mesh_viewer.py  # PyVista 3D visualization
├── frontend/              # TypeScript/React/Three.js web interface
├── backend/               # Python/FastAPI server
└── docs/                  # Research papers and implementation guides
```

## Build & Run Commands

### Desktop Application (Primary)

```bash
cd desktop_app
pip install -r requirements.txt
python main.py
```

### Frontend (Web)

```bash
cd frontend
npm install
npm run dev
```

### Backend (API Server)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| PyQt6 | Desktop GUI framework |
| trimesh | Mesh loading, manipulation, ray intersection |
| numpy | Numerical operations |
| pyvista/pyvistaqt | 3D visualization |
| pytetwild | Tetrahedral meshing (fTetWild bindings) |
| manifold3d | CSG operations for mold cavity creation |
| meshlib | High-performance mesh repair |
| torch | Optional GPU acceleration |

## Algorithm Pipeline (10 Steps)

1. **Import** → Load STL, validate mesh
2. **Parting Direction** → Fibonacci sampling + GPU visibility
3. **Inflated Hull** → Create bounding volume
4. **Tetrahedralize** → fTetWild tetrahedral meshing
5. **Mold Halves** → Region growing classification (H₁/H₂)
6. **Edge Weights** → 1/(dist²+ε) computation
7. **Dijkstra** → Escape paths from interior to boundary
8. **Primary Surface** → Marching Tetrahedra (3/4/5/6-edge configs)
9. **Smoothing** → Laplacian with boundary re-projection
10. **Pouring Direction** → Persistence homology for bubble minimization

## Key Formulas

```python
# Weighted geodesics (Paper Section 4.5)
edge_weight = 1 / (dist_to_part² + 0.25)  # ε = 0.25

# Boundary bias
λ_w = R - dist(w, M)
biased_dist = dist + max(λ_w, 0)

# Smoothing damping
damping_factor = 0.5
```

## Coding Standards

- **Type hints** for all function signatures
- **Dataclasses** for result structures
- **Logging module** (not print statements)
- **Docstrings** with Args, Returns, and paper references
- **Threading**: Long operations must use QThread with progress signals

## Documentation References

- `docs/README.md` - Documentation index
- `docs/PAPERS_TO_CODE_MAPPING.md` - Paper sections → code functions
- `docs/alderighi-2019-composite-molds/` - Primary paper (SIGGRAPH 2019)
- `docs/nielson-franke-1997-marching-tetrahedra/` - Marching Tetrahedra
- `docs/bloomenthal-1995-non-manifold/` - Non-manifold surface handling

## Path-Specific Instructions

Additional context is available in `.github/instructions/`:
- `vcmoldcreator-agent.instructions.md` - Full algorithm details
- `workflow.instructions.md` - Pipeline workflow and debugging
- `core-algorithms.instructions.md` - Implementation specifics for `core/`
- `memory.instruction.md` - Project state and quick reference

## Testing

Run the desktop app and load an STL file to test the pipeline. Check the console for progress messages and errors at each step.

## Common Issues

| Issue | Solution |
|-------|----------|
| "Membrane has holes" | Check cut_edge_flags, verify 5/6-edge configs |
| "Membrane doesn't touch part" | Verify vertex_boundary_type == -1, check re-projection |
| "Dijkstra incorrect labels" | Verify boundary_labels assignment, check boundary zone exclusion |
