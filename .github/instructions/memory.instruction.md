---
applyTo: '**'
---

# VcMoldCreator Agent Memory

## Project State

### Last Updated
- Date: January 23, 2026
- Status: Active Development

### Current Implementation Status

#### Fully Implemented ✅
- Import/Validation pipeline
- Parting direction computation (Fibonacci + Embree)
- Inflated hull generation
- Tetrahedral meshing (fTetWild/pytetwild)
- Mold half classification
- Edge weight computation with ∂F bias
- Dijkstra escape path labeling
- Primary cut edge detection
- Marching Tetrahedra (all 64 configurations including 5/6-edge)
- Membrane smoothing with boundary re-projection
- Gap filling for floating edges
- Small hole removal
- Persistence-based pouring direction optimization

#### In Progress 🔄
- Secondary membrane extraction
- Session save/load

#### Not Started ❌
- Fabrication pipeline (shell, metamold)
- Multi-piece molds
- GPU acceleration for Dijkstra

---

## Known Issues

### Performance
- Large meshes (>100k triangles) can be slow in tetrahedral meshing
- Dijkstra on dense tet meshes benefits from C++ extension

### Algorithm
- Very thin features may not generate complete membranes
- Highly concave regions may have multiple valid parting surfaces

---

## User Preferences

### Code Style
- Use type hints everywhere
- Prefer dataclasses for results
- Document paper references in docstrings

### Development Environment
- Python 3.13
- Windows 11
- PyQt6 for desktop app
- VS Code as primary editor

### Virtual Environment (IMPORTANT!)
**ALWAYS activate the virtual environment before running any Python commands!**

```powershell
# Activate virtual environment FIRST before any Python command
cd c:\Users\VijayRaghavVarada\Documents\Github\VcMoldCreator
.\.venv\Scripts\Activate.ps1

# Then navigate and run Python commands
cd desktop_app
python main.py
```

The virtual environment is located at:
- Path: `c:\Users\VijayRaghavVarada\Documents\Github\VcMoldCreator\.venv\`
- Activation: `.\.venv\Scripts\Activate.ps1` (PowerShell) from project root

**Never run Python directly without activating the venv first!**

---

## Important Files to Know

### Core Algorithm
- `desktop_app/core/tetrahedral_mesh.py` - Main algorithm implementation
- `desktop_app/core/parting_surface.py` - Marching tetrahedra
- `desktop_app/core/surface_propagation.py` - Membrane smoothing
- `desktop_app/core/pouring_direction.py` - Bubble optimization

### UI
- `desktop_app/ui/main_window.py` - Main application window (9400+ lines)
- `desktop_app/viewer/mesh_viewer.py` - 3D visualization

### Documentation
- `docs/alderighi-2019-composite-molds/` - Primary reference paper (SIGGRAPH 2019)
- `docs/nielson-franke-1997-marching-tetrahedra/` - Marching Tetrahedra reference
- `docs/bloomenthal-1995-non-manifold/` - Non-manifold surface handling
- `docs/gibson-1998-surface-nets/` - Elastic surface nets (alternative reference)
- `docs/POURING_DIRECTION_IMPLEMENTATION.md` - Pouring algorithm design
- `docs/PRIMARY_SURFACE_MEMBRANE_IMPROVEMENTS.md` - Membrane improvements
- `docs/README.md` - Documentation index and paper guide

---

## Quick Reference

### Key Formulas
```
edge_weight = 1 / (dist_to_part² + 0.25)
λ_w = R - dist(w, M)  // boundary bias
biased_dist = δ + max(λ_w, 0)
```

### Key Constants
- EDGE_WEIGHT_EPSILON = 0.25
- BOUNDARY_ZONE_THRESHOLD = 0.15 (15%)
- SMOOTHING_DAMPING = 0.5
- TILT_ANGLE_DEFAULT = 10.0°

### Boundary Labels
- -1 = on part surface M (inner boundary)
- 0 = interior or boundary zone
- 1 = on hull H₁
- 2 = on hull H₂

---

## Recent Changes

### January 2026
- Implemented 5-edge and 6-edge Marching Tetrahedra configurations
- Added face vertex and inner vertex computation
- Completed persistence-based pouring direction optimization
- Added floating edge gap filling

---

## Notes

### Common Gotchas
1. `vertex_boundary_type` must be tracked through smoothing for correct re-projection
2. Boundary zone vertices (label=0) are NOT valid Dijkstra targets
3. Diagonal choice in 4-edge configs must be consistent across tetrahedra
4. Face vertices in 5-edge configs go at centroid of 3 mid-edge points

### Testing Tips
- Use `trimesh.primitives` for quick test meshes
- Visualize intermediate results with PyVista
- Check mesh validity with `trimesh.Trimesh.is_watertight`
