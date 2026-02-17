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
- Hard shell prism generation (aligned with pouring direction)
- CSG operations via `csg_backend.py` abstraction layer (libigl/CGAL primary, manifold3d fallback)
- Outer collar extension (parting surface extended outward)
- Shell splitting into two manifold halves using membrane

#### In Progress 🔄
- Secondary membrane extraction
- Session save/load
- Shell halves export for 3D printing

#### Not Started ❌
- Air vent placement
- Registration marks on shell halves
- Metamold geometry
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
- `desktop_app/core/csg_backend.py` - CSG abstraction layer (libigl/CGAL primary, manifold3d fallback)
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

### January 2026 - mold_fabrication.py Code Quality Refactoring (Session 6)
- **Extracted 3 shared helpers** to eliminate ~400 lines of duplicated code:
  - `_build_pouring_basis(direction)` — Gram-Schmidt orthonormal basis (was copy-pasted in 3 functions)
  - `_extrude_surface_to_volume(surface, direction, offset_pos, offset_neg)` — unified extrusion (was copy-pasted in 4 functions)
  - `_split_and_classify_halves(mesh, direction)` — split+sort+classify pattern (was copy-pasted in 3 functions)
  - `_find_boundary_edges_with_winding(faces)` — boundary edge extraction sub-helper
- **Converted 4 extrusion functions to thin delegators:**
  - `_create_cutting_volume_from_surface()` → delegates to `_extrude_surface_to_volume()`
  - `_create_half_space_from_membrane()` → delegates to `_extrude_surface_to_volume()`
  - `_create_cutting_blade_from_membrane()` → delegates to `_extrude_surface_to_volume()`
  - `_create_tall_cutting_blade()` → delegates to `_extrude_surface_to_volume()`
- **Renamed `MANIFOLD_AVAILABLE` → `CSG_AVAILABLE`** across mold_fabrication.py and resin_channels.py (8+4 occurrences)
- **Replaced 6 `import traceback; traceback.print_exc()`** blocks with `logger.exception(e)`
- **Removed dead code** after `_create_tall_cutting_blade` return statement
- **Net result:** 3224 → 2812 lines (−412 lines, or −13%)
- **All public API signatures unchanged** — no downstream breakage

### January 2026 - CSG Backend Migration & Metamold Pipeline (Session 5)
- **Metamold execution order change:** Union blade+part first, then single CSG subtract from prism
  - Added `create_metamold_with_combined_cut()` (~130 lines)
  - Rewrote `MetamoldWorker.run()` from 6-step to 2-step pipeline
  - Handles both split and unsplit edge cases
- **Created `desktop_app/core/csg_backend.py`** — CSG abstraction layer
  - Primary: `igl.copyleft.cgal.mesh_boolean()` (Zhou et al. 2016, exact arithmetic, 100% robust)
  - Fallback: `manifold3d` (floating point, may fail on edge cases)
  - Provides: `csg_difference()`, `csg_union()`, `csg_intersection()`, `csg_trim_by_plane()`
- **Migrated `mold_fabrication.py`** — Replaced all 9 manifold3d CSG call sites:
  - `create_shell_with_cavity()` → `csg_backend.csg_difference()`
  - `split_shell_with_membrane()` → `csg_backend.csg_difference()`
  - `split_shell_with_tall_blade()` → `csg_backend.csg_difference()`
  - `split_shell_along_parting_surface()` → `csg_backend.csg_intersection()` × 2
  - `create_part_with_thickened_secondary()` → `csg_backend.csg_union()`
  - `add_part_to_metamold_half()` → `csg_backend.csg_union()`
  - `trim_metamold_halves()` → `csg_backend.csg_trim_by_plane()` × 2
  - Removed `_trimesh_to_manifold()` and `_manifold_to_trimesh()` helpers
- **Migrated `resin_channels.py`** — Replaced all 5 manifold3d CSG call sites:
  - `create_resin_channels()` → loop of `csg_backend.csg_difference()`
  - `create_hard_shell_inlet()` → `csg_backend.csg_difference()`
  - `create_hard_shell_air_escapes()` → loop of `csg_backend.csg_difference()`
  - Removed duplicate `_trimesh_to_manifold()` and `_manifold_to_trimesh()` helpers
- **Updated `requirements.txt`** — Added `libigl>=2.5.0` as primary, kept `manifold3d>=2.3.0` as fallback
- **Motivation**: Alderighi 2019 Section 5 explicitly requires Zhou et al. 2016 exact arithmetic booleans; manifold3d used floating point and failed on complex geometries

### January 2026 - Static Analysis Refactoring (Session 4)
- **Bare except removal:** Fixed 11 bare `except:` → `except Exception:` across 4 files (mesh_viewer.py, main_window.py, parting_surface.py, mold_fabrication.py)
- **Print→logger conversion:** Converted ~100 debug print() calls in mesh_viewer.py to logger.debug() using lines-list pattern. Renamed `_print_tet_edge_info` → `_get_tet_edge_info_lines` (returns List[str])
- **Logger lazy formatting:** Converted 30 `logger.exception(f"...")` calls to `logger.exception("...: %s", e)` across main_window.py, main.py, mesh_repair.py
- **Dead code removal:**
  - Removed unused backward-compat aliases: `RegistrationNoiseResult`, `PerlinRegistrationResult` (registration_marks.py)
  - Removed unused convenience functions: `analyze_mesh()` (mesh_analysis.py), `repair_mesh()` (mesh_repair.py), `get_band_visualization_data()` (registration_marks.py), `get_mold_half_face_colors()` + `MoldHalfColors` class (mold_half_classification.py), `get_implementation_info()` (fast_algorithms_wrapper.py)
  - Removed unused `Tuple` import from registration_marks.py

### January 2026 - Unused Import Cleanup (Session 3)
- Cleaned unused imports across 6 core files

### January 2026 - Earlier
- Implemented 5-edge and 6-edge Marching Tetrahedra configurations
- Added face vertex and inner vertex computation
- Completed persistence-based pouring direction optimization
- Added floating edge gap filling
- **Hard Shell Pipeline:**
  - Created `create_hard_shell_prism()` in mold_fabrication.py
  - Implemented CSG subtraction using manifold3d
  - Added outer collar extension (parting surface extended beyond hull)
  - Implemented `split_shell_with_membrane()` using half-space volumes
  - Added display options for shell with cavity, prism, outer collar, shell halves
  - Shell halves shown in teal (half 1) and coral (half 2) colors

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
