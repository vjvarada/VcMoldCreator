# Metamold Union-First CSG Design

## Document Purpose

This document records the analysis, rationale, and implementation details for
the **union-first** CSG approach used in metamold half creation.  It exists to
ensure that the problem we are solving — and *why* we are solving it this way —
is preserved for future iterations.

---

## Problem Statement

### What Happens During Metamold Half Creation

The metamold is a 3D-printed container used to cast silicone mold halves.  To
create two metamold halves (upper and lower), we:

1. Start with a **prism** — a solid volume aligned with the pouring direction.
2. Subtract the **part mesh** to create a cavity shaped like the object to cast.
3. Split the cavity into two halves using a **cutting blade** derived from the
   parting surface (primary membrane + outer collar extension).

The cutting blade is a thin watertight solid (~1 µm thick) created by
thickening the outer collar mesh along its vertex normals.

### The Core Failure Mode

For the prism to split into exactly **two** connected components, the cutting
blade must form a **complete cross-sectional barrier** through the annular void
between the part surface and the prism wall.  Visually:

```
    ┌──────────── PRISM WALL ────────────┐
    │                                      │
    │   ┌────── PART SURFACE ──────┐      │
    │   │                           │      │
    │   │    (solid part — no       │      │
    │   │     material here in      │      │
    │   │     the cavity)           │      │
    │   │                           │      │
    │   └───────────────────────────┘      │
    │           ▲                           │
    │           │                           │
    │     ANNULAR GAP                      │
    │     (this is the cavity that         │
    │      must be completely bisected     │
    │      by the blade)                   │
    │                                      │
    └──────────────────────────────────────┘
```

The blade (derived from the collar + membrane) spans from inside the part
surface outward to (and beyond) the prism wall.  If **any** gap exists between
the collar's inner boundary and the part surface, a topological bridge of
cavity material connects the two sides, and `manifold3d.decompose()` returns
**1 component** instead of 2.

### Why Gaps Occur

The collar extension (`create_robust_collar_extension()` in
`parting_surface.py`) pushes collar vertices inward toward the part surface by
`collar_depth` (~0.2 mm).  On complex geometry — sharp concavities, saddle
transitions, thin protrusions — some collar edges fail to reach the part
surface.  Reasons include:

- **Concave corners**: Fan arcs from adjacent boundary vertices may diverge
  instead of converging toward the part surface.
- **Self-intersection avoidance**: The repair step may remove collar triangles
  that self-intersect, creating intentional gaps for mesh validity.
- **Proximity tolerance**: The part surface itself may have local features
  (sharp creases, thin ridges) where the collar's linear push doesn't land on
  the actual surface.

Even a single missing triangle (a gap of < 0.1 mm²) at the collar-to-part
junction prevents a topologically clean split.

---

## Solution: Union-First Approach

### Key Insight

The part mesh already fills the entire interior of the annular gap.  If we
**union the blade with the (deflated) part mesh before subtracting from the
prism**, any gap between the collar inner boundary and the part surface becomes
irrelevant — the part mesh's own surface closes it.

### Old Pipeline (Sequential Subtraction)

```
    Step 1:  cavity  = prism − deflated_part
    Step 2:  cut     = cavity − blade
    Step 3:  halves  = cut.decompose()       ← FAILS if blade has gaps
    Step 4:  half_i  = component_i + part    ← union part back
```

In this pipeline, the blade must **perfectly** bisect the annular cavity.  Any
gap leaves a bridge.

### New Pipeline (Union-First)

```
    Step 1:  cutter  = blade ∪ deflated_part   ← union first
    Step 2:  validate(cutter)                   ← check watertight, log warnings
    Step 3:  halves  = prism − cutter           ← single subtraction
    Step 4:  components = halves.decompose()    ← robust: cutter is solid
    Step 5:  half_i  = component_i + part       ← union part back (original size)
```

### Why This Works

1. **Gaps auto-fill**: The collar only needs to get *close enough* to the part
   for boolean overlap.  The part surface geometry closes any remaining gap.
   Even if collar vertices miss the part by 0.05 mm, the blade's ±0.0005 mm
   thickness plus the part mesh's surface create a solid barrier.

2. **Validation window**: Before the destructive subtraction, we can inspect
   `blade ∪ deflated_part` for watertightness.  If it's not watertight, we
   know the split may fail and can log a diagnostic warning — instead of
   silently producing an unsplit component.

3. **Consistent with secondary membranes**: The function
   `create_part_with_thickened_secondary()` already does exactly this pattern:
   `part + thickened_secondary` as a union.  The union-first approach for the
   primary blade is the same principle.

### Deflation Interaction

The existing deflation trick (shrink part by 1 µm for cavity subtraction to
avoid coplanar face artifacts) is preserved:

- **Cutter uses deflated part**: `cutter = blade ∪ deflated_part`.  This makes
  the cutter fractionally larger on the part-surface side.
- **Final union uses original part**: `half_i = component_i + original_part`.
  Since the original part is ~1 µm larger than the cavity, the union has
  genuine geometric overlap → clean boolean result without coplanar shards.

### CSG Operation Count Comparison

| Step | Old Pipeline | New Pipeline |
|------|-------------|-------------|
| 1 | `prism − deflated_part` (subtraction) | `blade ∪ deflated_part` (union) |
| 2 | `cavity − blade` (subtraction) | `prism − cutter` (subtraction) |
| 3 | `decompose()` | `decompose()` |
| 4+5 | `component + part` × 2 (unions) | `component + part` × 2 (unions) |
| **Total** | 2 subtractions + 2 unions | 1 union + 1 subtraction + 2 unions |

Same number of CSG operations — no performance penalty.

---

## Implementation Notes

### `build_metamold_halves_manifold_space()`

The primary function modified.  The new pipeline is:

```python
# 1. Create blade from membrane
blade_trimesh = _create_cutting_blade_from_membrane(membrane_mesh, ...)

# 2. Deflate part mesh
deflated_part = _deflate_part_for_csg(part_mesh)

# 3. Union blade + deflated_part → cutter  (NEW STEP)
cutter_m = blade_m + cavity_part_m
# Validate: log component count and watertight status

# 4. Subtract cutter from prism  (replaces two sequential subtractions)
halves_m = prism_m - cutter_m

# 5. Decompose into components
components = halves_m.decompose()

# 6. Classify upper/lower, union with original part
```

### `split_shell_with_membrane()`

The hard-shell splitting function follows the same pattern when `prism_mesh`
and `subtractor_mesh` are provided:

```python
# Old:  (prism − subtractor) − blade
# New:  prism − (blade ∪ subtractor)
cutter_m = blade_m + subtractor_m
cut_m = prism_m - cutter_m
```

### Fallback Behavior

If the union-first approach produces fewer than 2 components (which would mean
the cutter doesn't span the prism — possible on extremely broken collar
geometry), the function logs detailed diagnostics and returns the single
component with `success=False`, same as the old pipeline.

---

## Edge Cases and Risks

### 1. Blade Doesn't Overlap Part

If the collar extension completely fails (no collar vertices penetrate the part
surface), the blade and deflated_part may not overlap geometrically.  In this
case, `blade ∪ deflated_part` produces two disconnected components — the blade
floating in the annular gap and the part as a separate solid.  The subtraction
from the prism would still work (it removes both), but the "gap" between them
becomes a thin bridge in the result.

**Mitigation**: This is the same failure mode as the old pipeline.  The
union-first approach doesn't make it worse, and the validation step makes it
detectable.

### 2. Combined Part Mesh (Secondary Membranes)

When `combined_part_mesh` is provided (part + thickened secondary surface), the
**cutter** still uses `deflated_part` (not the combined mesh).  The combined
mesh is only used for the final `component + part` union step.  This preserves
the correct cavity shape.

### 3. Manifold3d Coplanar Artifacts

The union `blade ∪ deflated_part` may produce coplanar face artifacts where the
blade surface coincides with the part surface (the collar pushes 0.2 mm into
the part — shared geometry).  This is mitigated by the fact that the blade is a
solid volume (not a surface), so the union resolves to a clean volumetric merge.

---

## Verification

After implementation, verify with:

1. **Synthetic test**: Box prism + sphere part + spanning disc blade.  Union
   first should produce 2 components.
2. **Gap test**: Same setup but with a blade that has a hole (doesn't fully
   span).  Old approach → 1 component.  Union-first → 2 components (part fills
   the gap).
3. **Real model test**: Run on a complex STL where the collar previously failed
   to split.  Check `decompose()` returns 2 components.

---

## References

- Alderighi et al., "Volume-Aware Design of Composite Molds", SIGGRAPH 2019,
  Section 5: Fabrication
- manifold3d GitHub issues #1430, #1359: Coplanar face shard artifacts
- `_deflate_part_for_csg()`: 1 µm deflation to avoid coincident faces
- `create_robust_collar_extension()`: Collar generation with gap-filling
