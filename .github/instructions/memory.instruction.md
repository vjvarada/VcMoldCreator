---
applyTo: '**'
---

# VcMoldCreator Agent Memory

## Project State

### Last Updated
- Date: January 24, 2026
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
- CSG subtraction (prism - hull) using manifold3d
- Outer collar extension (parting surface extended outward)
- Shell splitting into two manifold halves using membrane

#### In Progress 🔄
- Secondary membrane extraction
- Session save/load
- Shell halves export for 3D printing
- Metamold hollowing + adaptive trim (implemented, needs real-model validation)

#### Not Started ❌
- Air vent placement
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

### CSG / Metamold (Fixed)
- **Deflation offset increased from 1 µm to 50 µm** (April 2026): The original 1 µm deflation was at the edge of numerical precision for manifold3d when mesh coordinates are in the 10s-100s mm range. This caused coplanar-face artifacts (bad triangles) at the part boundary inside the metamold. The 50 µm offset is still invisible in the final mold (below any 3D printer resolution) but provides 50x more numerical margin.

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
- `desktop_app/ui/main_window.py` - Main application window (13,000+ lines)
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

### March 2026 - Adaptive Base Trim Moved to Resin Step (Session 14, Phase 4)
- **Problem:** When hollowing metamold halves, the hollow cavity from the base and the hollow cavity from the part-cavity side could merge at the lowest points of internal geometry. The original 4mm trim threshold was too aggressive for thicker walls.
- **Solution:** Moved base trimming from `MetamoldWorker` (metamold step) to `ResinChannelsWorker` (resin channels step). The trim threshold adapts to hollowing parameters:
  - No hollowing: `TRIM_THRESHOLD_DEFAULT = 4.0` mm (unchanged behavior)
  - With hollowing: `max(4.0, 2 × wall_thickness + TRIM_HOLLOW_MIN_GAP)` where `TRIM_HOLLOW_MIN_GAP = 1.0` mm
  - Formula ensures two wall-thickness shells (base-side + cavity-side) never merge
- **Changes:**
  - `mold_fabrication.py`: Added `TRIM_THRESHOLD_DEFAULT = 4.0`, `TRIM_HOLLOW_MIN_GAP = 1.0`, `compute_adaptive_trim_threshold()` helper
  - `main_window.py` — MetamoldWorker: Removed `trim_metamold_halves` import and call. Halves emitted are now UNTRIMMED. Added comment explaining the move.
  - `main_window.py` — ResinChannelsWorker: Added `blade_mesh=None` parameter. Added trim step BEFORE hollowing using blade mesh (outer collar) as height reference. Stores `_trim_threshold`, `_trim_upper_saved`, `_trim_lower_saved`, `_trim_ms` for stats.
  - `main_window.py` — `_on_create_resin_channels()`: Passes `self._outer_collar_result.mesh` as `blade_mesh`
  - `main_window.py` — `_update_resin_channels_step_ui()`: Shows trim stats row + attaches trim metadata to `channel_result`
- **Execution order in ResinChannelsWorker:** Trim → Hollow → Drill channels
- **Adaptive threshold examples:** wall=2.5mm → 6.0mm, wall=5.0mm → 11.0mm, wall=1.0mm → 4.0mm

### March 2026 - Cleanup Validation Gate (Session 14, Phase 3)
- **Change:** With part deflation eliminating coplanar artifacts, the full `cleanup_csg_mesh()` is now gated behind a `_mesh_needs_cleanup()` validation check in MetamoldWorker Step 6. If the mesh has 0 NM edges and 0 zero-area faces, cleanup is skipped entirely.
- **File:** `main_window.py` — MetamoldWorker.run(), `_mesh_needs_cleanup()` helper function

### March 2026 - Part Deflation for CSG Overlap (Session 14, Phase 2)
- **Problem:** In `build_metamold_halves_manifold_space()`, the `half + part` union creates coplanar face shards because the cavity surface (from `prism - part`) is geometrically identical to the part surface. manifold3d's symbolic perturbation cannot always resolve exact coincident faces (GitHub manifold #1430, #1359).
- **Previous mitigations:** `_remove_coplanar_shard_components()` (Session 6), `ensure_manifold_mesh()` (Session 10). These are band-aids that remove artifacts after the fact.
- **Root-cause fix:** Deflate the part mesh by 1 µm (0.001 mm) before the cavity subtraction, then use the original (un-deflated) part for the union. This creates genuine geometric overlap at the cavity boundary, eliminating coincident faces entirely.
- **Implementation:** 
  - Added `CSG_PART_DEFLATION_MM = 0.001` constant in `mold_fabrication.py`
  - Added `_deflate_part_for_csg()` helper: moves each vertex inward along its vertex normal by `deflation_mm`. Preserves exact mesh topology (same face count, same connectivity). ~6ms for 1K faces, ~64ms for 20K faces.
  - Modified `build_metamold_halves_manifold_space()` pipeline:
    - Step 1: Deflate part via `_deflate_part_for_csg(part_mesh)`
    - Step 2: Convert deflated part AND original part to manifold3d separately
    - Step 3: `cavity = prism - deflated_part` (cavity is 1 µm larger)
    - Step 7: `half + original_part` (original is 1 µm larger than cavity → overlap)
  - Falls back to original part if deflation fails (graceful degradation)
- **Why vertex-normal offset, not MeshLib voxelization:** MeshLib `generalOffsetMesh` works but causes face-count explosion (1,280 → 274,088 for a sphere) due to marching cubes remeshing. Vertex-normal displacement preserves exact topology and is orders of magnitude faster. At 1 µm scale, self-intersection from convergent normals is completely negligible.
- **Why deflation not scaling:** Scaling from centroid would not affect concavities uniformly — holes and internal cavities might get no overlap. Vertex-normal offset shrinks every surface point inward by exactly the same amount.
- **Test results:**
  - Sphere (r=10): extent shrink = 0.002mm/axis (0.001 from each side) ✓
  - Cube (20mm): bounds shrunk correctly ✓  
  - Torus (R=15, r=5): concavity handled, volume reduction 0.04% ✓
  - All meshes: face count preserved exactly ✓
  - CSG integration: clean output, no shard artifacts ✓

### March 2026 - Metamold Hollowing (Session 14)
- **Goal:** Add optional hollowing of metamold halves to conserve 3D printing material, similar to Meshmixer's Hollow tool.
- **Research:** Evaluated 3 approaches: MeshLib `thickenMesh` (chosen), MeshLib `offsetMesh` + Boolean, OpenVDB direct (Blender Print3D Toolbox approach). Chose MeshLib since it's already installed and `thickenMesh` with negative offset is purpose-built for hollowing.
- **New function:** `hollow_metamold_half()` added to `mold_fabrication.py` (lines 2708-2815):
  - Converts trimesh → meshlib mesh via `mrmeshnumpy.meshFromFacesVerts()`
  - Auto-detects open vs closed mesh → selects `SignDetectionMode.Unsigned` or `.OpenVDB`
  - Uses `mm.suggestVoxelSize(mesh, 5000000.0)` for auto voxel sizing
  - Calls `mm.thickenMesh(mesh, -wall_thickness, params)` — negative offset = hollowing mode
  - Converts back with `process=False` to preserve topology
  - Logs volume savings percentage
  - Returns `(hollowed_mesh, elapsed_ms, success)` tuple
- **UI integration:** Added "Metamold Hollowing" group box to resin channels panel:
  - Checkbox: "Hollow out metamold halves" (default: off)
  - Spinbox: Wall thickness 0.5–10.0 mm (default: 2.5 mm)
  - Thickness spinbox disabled when checkbox unchecked
- **Worker integration:** Hollowing runs as first step in `ResinChannelsWorker.run()`, BEFORE channel drilling, so channels correctly penetrate thin walls
- **Stats display:** Hollowing result shown in resin channels stats box
- **Test results:**
  - Synthetic metamold (box with cavity): 16,800 → 3,677 mm³ (78% saved) in 346ms
  - Real metamold half1 (225K faces, open): 352ms, success
  - Real metamold half2 (226K faces, open): 341ms, success
  - Edge cases (empty mesh, tiny box): all pass
- **Key API:** `mrmeshpy.thickenMesh(mesh, offset, params)` — "in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode)"

### January 2026 - Metamold Code Cleanup (Session 11)
- **Goal:** Review and clean up all metamold creation/cleanup implementations accumulated over Sessions 6–10.
- **Dead code removed (4 functions, ~398 lines):**
  - `remove_overlapping_face_pairs()` (~387 lines) — superseded by `ensure_manifold_mesh()` in Session 10
  - `split_shell_along_parting_surface()` (~132 lines) — superseded by `split_shell_with_membrane()`, was already dead
  - `add_part_to_metamold_halves()` (~46 lines) — wrapper that called `add_part_to_metamold_half()` for both halves, unused since Session 7's fused pipeline
  - `add_part_to_metamold_half()` (~88 lines) — individual part union, unused since `build_metamold_halves_manifold_space()` does this in manifold space
- **Code quality fixes:**
  - Converted 4 `traceback.print_exc()` → `logger.exception()` in `build_metamold_halves_manifold_space`, `split_shell_with_membrane`, `thicken_surface_symmetric`, `create_part_with_thickened_secondary`
  - Cleaned MetamoldWorker imports in main_window.py (removed `create_shell_with_cavity`, `split_shell_with_membrane`, `add_part_to_metamold_halves`)
- **Bug fix during cleanup:** Two botched replacements from multi_replace_string_in_file left orphaned function bodies creating duplicate `def` definitions:
  - `def cleanup_csg_mesh(` had the orphaned body of `remove_overlapping_face_pairs` underneath it (lines 2477-2860)
  - `def thicken_surface_symmetric(` had the orphaned body of `split_shell_along_parting_surface` underneath it (lines 1814-1937)
  - Both repaired: file went from 4292 → 3629 lines (663 lines removed net)
- **Final state:** `mold_fabrication.py` has 29 functions (down from 33), zero syntax errors, zero IDE errors, all 14 public functions import correctly.

### January 2026 - Simplified Manifold Enforcement (Session 10)
- **Problem:** The iterative `remove_overlapping_face_pairs()` approach (391 lines, 3 rounds + final pass + T-junction handling) was complex and fragile. User reported the issue still persisted on some models despite testing clean on debug meshes.
- **Root Cause of complexity:** The old approach tried to detect and remove specific anti-parallel face pairs at NM edges, then fill holes with meshlib, then `process=True` vertex merge recreated NM edges, requiring iterative cycles. Each step could fail or introduce new artifacts.
- **New Solution:** Replaced with a simple 3-step `ensure_manifold_mesh()` function in `mold_fabrication.py`:
  1. **Import into meshlib** via `meshFromFacesVerts()` — automatically splits NM edges by duplicating vertices in the half-edge structure. Overlapping face patches become disconnected topological components.
  2. **Remove small components** via `getAllComponents()` + `deleteFaces()` — the split NM patches are small (typically <36 faces each, 484-699 components total).
  3. **Export with `process=False`** — critical: trimesh's `process=True` merges geometrically coincident vertices, which RE-CREATES the NM topology that meshlib just fixed.
- **Key insight:** `process=False` is essential. The old approach used `process=True` after meshlib, which caused the cycling problem. The new approach avoids vertex merging entirely.
- **Integration in `cleanup_csg_mesh()` selective mode:**
  - Step A: Remove zero-area faces (unchanged)
  - Step B: `ensure_manifold_mesh()` (NEW — replaces 391-line `remove_overlapping_face_pairs()`)
  - Step C: Per-vertex proximity needle edge collapse (unchanged)
  - Step D: `ensure_manifold_mesh()` again (NEW — catches NM edges from collapse)
- **Results:**
  - Half1: 313 NM → 0 NM, 4644 faces removed (699 components), 82ms
  - Half2: 177 NM → 0 NM + 3 NM from collapse → 0 NM, 4101 faces removed (484 components), 77+65ms
  - Total time: ~150ms per half (vs ~4000ms before)
  - `remove_overlapping_face_pairs()` is retained but no longer called from `cleanup_csg_mesh()` selective mode
- **meshlib functions used:** `meshFromFacesVerts` (auto-splits NM), `fixMultipleEdges` (safety net), `getAllComponents`, `FaceBitSet`, `deleteFaces`, `FillHoleParams`, `fillHole`, `packOptimally`, `getNumpyVerts`, `getNumpyFaces`

### January 2026 - Iterative Pair Removal with Final Pass (Session 9)
- **Problem:** After Session 8's `remove_overlapping_face_pairs()`, 25 NM edges remained in the final mesh. These were NEWLY CREATED by the `process=True` vertex merge after meshlib hole-filling — meshlib creates new vertices that coincide with existing vertices, and trimesh merges them, creating NM edges.
- **Root Cause 1 (cycling):** The original single-pass approach removed pairs → meshlib filled holes → `process=True` merged vertices → created ~12-13 new NM edges with anti-parallel face pairs. A naive iterative approach cycles: round N removes pairs at these edges, meshlib fills, vertex merge recreates the same NM edges at round N+1.
- **Root Cause 2 (threshold):** One stubborn NM edge had large flat faces (areas 2-4.5 mm²) at the metamold base plane. The anti-parallel pairs had dot=-1.0 but centroid distances of 1.5-2.1 mm — exceeding the 1.0 mm threshold.
- **Solution (3 changes to `remove_overlapping_face_pairs()` in `mold_fabrication.py`):**
  1. **Iterative loop (max 3 rounds):** Each round: detect NM edges → pair anti-parallel faces → remove → meshlib fill → `process=True` reload. Second round catches vertex-merge-created NM edges.
  2. **Cycle detection:** If NM edge count is unchanged from previous round, break early to the final pass (saves one unnecessary meshlib round).
  3. **Final lightweight pass:** After the loop, remaining stubborn NM edges are resolved by removing only ONE pair per edge (reducing multiplicity 4→2 = manifold). No meshlib or `process=True` needed — no holes created when 2 of 4 faces remain.
  4. **Relaxed threshold for near-perfect anti-parallel pairs:** `dot < -0.99` → pair regardless of centroid distance (clearly overlapping). `dot < -0.3 AND cdist < 1.0` → pair with proximity confirmation (as before).
- **Results:**
  - Half1: 597 NM → 0 NM (was 25 before fix). 234,068 → 228,551 faces, 790 pairs removed in 2 rounds + final pass.
  - Half2: 351 NM → 0 NM. 233,086 → 228,706 faces, 461 pairs removed in 2 rounds + final pass.
  - Time: ~4 seconds per half (same as before, cycle detection saves the wasted 3rd meshlib round).

### January 2026 - Overlapping Face Pair Removal (Session 8)
- **Root Cause:** manifold3d union of coincident surfaces (cavity + part) leaves overlapping anti-parallel face pairs at the intersecting surface. These double-wall faces create 600+ non-manifold edges (multiplicity 4 = 2 proper + 2 overlapping) and 880+ small fragment components.
- **Solution:** Added `remove_overlapping_face_pairs()` in `mold_fabrication.py`:
  - Builds edge→face mapping, finds NM edges (multiplicity > 2)
  - At each NM edge, greedily pairs anti-parallel faces (dot < -0.3, centroid dist < 1.0)
  - Removes BOTH faces of each pair + zero-area faces
  - Uses meshlib to fill resulting holes and remove small components (< 100 faces)
  - Returns mesh with process=True for downstream compatibility
- **Integration:** Replaced Step C in `cleanup_csg_mesh()` selective mode. Now the order is:
  - Step A: Remove zero-area faces
  - Step B: Overlapping pair removal (NEW — runs BEFORE needle collapse)
  - Step C: Per-vertex proximity needle edge collapse  
- **Results on real model (233k face metamold half):**
  - NM edges: 606 → 16 (97.4% reduction)
  - Components: 882 → 1
  - Boundary edges: 0 → 0
  - Faces: 233,388 → 223,382 (4.3% removed)
  - ~9 residual NM edges from meshlib hole-fill vertex merging (cosmetic, < 0.005% of edges)
- **Key insight:** Pair removal MUST run before needle edge collapse — the edge collapse alters mesh topology and creates additional NM edges that make pair detection less effective.

### January 2026 - Manifold-Space Pipeline (Session 7)
- **Problem:** manifold3d union creates 200k+ coincident faces when `cavity + part` is done by converting back to trimesh between operations (breaks provenance tracking).
- **Solution:** Added `build_metamold_halves_manifold_space()` in `mold_fabrication.py` — performs cavity creation, blade split, decompose, and part union all in one manifold3d session without round-tripping through trimesh.
- **manifold3d upgrade:** 3.3.2 → 3.4.0 (PR #1445 new symbolic perturbation, PR #1467 coincident face handling). Works perfectly on synthetic tests but real models still have issues due to `_trimesh_to_manifold` / `_repair_mesh_for_csg` breaking internal provenance.
- **Result:** Synthetic test: perfect (3,206 faces, 0 internal, watertight). Real model: same as before (provenance broken by mesh repair). Fixed by Session 8's pair removal approach.

### January 2026 - Coplanar Shard Cleanup (Session 6)
- **Root Cause:** When adding the part back to metamold halves (`half + part` union), 98.3% of part faces are exactly coincident with cavity faces (opposite normals, mean dot = -1.000). This triggers manifold3d's known coplanar face shard artifact (GitHub issues #1430, #1359).
- **Diagnosis:** Union produces 337/536 components with 4170 tiny degenerate shard triangles. `trim_by_plane()` also produces 3-5 component shards (4-12 faces, zero volume).
- **Fix Strategy Tested:** 6 approaches compared — inflate centroid (worse), normal offset (worse), keep largest component (BEST), meshlib fixDegeneracies (terrible — 4× face explosion), double manifold3d roundtrip (marginal), inflate+roundtrip (worse).
- **Solution:** Added `_remove_coplanar_shard_components()` helper in `mold_fabrication.py`:
  - Splits mesh into components, keeps only those with face count ≥ 1% of largest (minimum 100 faces)
  - Applied after part union in `build_metamold_halves_manifold_space()`
  - Applied after both `trim_by_plane()` calls in `trim_metamold_halves()`
- **meshlib Assessment:** Already correctly used for PRE-CSG repair. NOT suitable for post-CSG cleanup (fixMeshDegeneracies explodes manifold3d output from 134k→536k faces and breaks watertight).
- **Result:** Both halves now single-component, watertight after union AND trim.
- **Diagnostic reports:** Saved to `.tmp/csg_debug/` (metamold_coplanar_analysis.txt, fix_comparison.txt, union_diagnostic.txt)

### January 2026 - Collar Validation & Repair (Session 5)
- **Problem:** Collar vertices could end up outside the part mesh, causing the downstream CSG blade subtraction in `split_shell_with_membrane()` to fail (shell not split into two halves).
- **Solution:** Added post-collar validation + iterative repair system in `parting_surface.py`:
  - **Constants:** `COLLAR_VALIDATION_MAX_REPAIR_ITERATIONS=3`, `COLLAR_VALIDATION_DEPTH_ESCALATION=2.0`, `COLLAR_VALIDATION_MIN_COVERAGE=0.9`, `COLLAR_REPAIR_PENETRATION_MULTIPLIER=3.0`
  - **`FloatingEdgeFillingResult`:** Added 5 stats fields: `collar_vertices_total`, `collar_vertices_inside_part`, `collar_vertices_repaired`, `collar_coverage_fraction`, `repair_iterations_used`
  - **`_validate_collar_penetration()`:** Ray-segment intersection test (membrane vertex → collar vertex) to check if segment crosses part surface. Uses parity for signed distance estimation. Does NOT require watertight mesh.
  - **`_repair_collar_vertices()`:** 3-strategy repair: (1) adaptive placement with escalated depth, (2) 6 axis-aligned rays, (3) guaranteed fallback (nearest point + inward normal push).
  - **`_quick_segment_crosses_part()`:** Helper for fast segment-part intersection check.
  - **STEP 7b in `create_robust_collar_extension()`:** Iterative validate-repair loop integrated between STEP 6 (fan triangles) and STEP 7 (create result mesh). Logs coverage warnings/errors. Records statistics in result.
- **Tests:** All unit tests and edge cases pass (empty map, zero-length, on-surface, deeply-inside, sphere, multi-iteration repair).

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
