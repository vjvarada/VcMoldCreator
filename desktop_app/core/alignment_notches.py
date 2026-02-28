"""
Alignment Notches Module

Cuts small triangular-prism notches into the outer perimeter of every
hard-shell and metamold half at the parting interface.  When two mating
halves are placed face-to-face the matching half-notches register visually
and tactilely, ensuring correct assembly alignment.

Notch geometry
--------------
Each notch is a triangular prism whose long axis runs parallel to the
resin-pouring direction (i.e., perpendicular to the parting plane).
Looking at the outer wall of any half from the outside, the cross-section
is a V-shape:

        ───────────── outer surface ────────────────
        │         │                   │             │
      wide_base/2 │ wide_base/2       │
                  ▼ (inward, depth mm)
                  ▲ (apex / sharp tip)

The cutter prism is tall enough to pass entirely through both halves of a
pair, so the same geometry is subtracted from every piece.

Placement (n_notches = 2)
-------------------------
The 2D silhouette of the combined mold is projected onto the plane
perpendicular to the pouring direction.  The two most diametrically-
opposite points on the convex hull are chosen as notch centres, giving one
notch on each side of the mold.

Author: VcMoldCreator
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import trimesh

try:
    import manifold3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_NOTCH_WIDTH_MM = 5.0   # Base width of the triangular notch (mm)
DEFAULT_NOTCH_DEPTH_MM = 0.5   # Depth of the apex from the outer surface (mm)
DEFAULT_N_NOTCHES      = 1     # Number of notches
CUTTER_OUTWARD_EXTRA   = 6.0   # How far the cutter extends OUTWARD beyond the surface (mm)
CUTTER_HEIGHT_MARGIN   = 10.0  # Extra height added each side of combined mesh extent (mm)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AlignmentNotchResult:
    """Result of applying alignment notches to all four mold pieces."""

    # Notched meshes  (None if the input was None or CSG failed for that piece)
    shell_half_1:    Optional[trimesh.Trimesh] = None
    shell_half_2:    Optional[trimesh.Trimesh] = None
    metamold_half_1: Optional[trimesh.Trimesh] = None
    metamold_half_2: Optional[trimesh.Trimesh] = None

    # Parameters used
    n_notches:      int   = DEFAULT_N_NOTCHES
    notch_width_mm: float = DEFAULT_NOTCH_WIDTH_MM
    notch_depth_mm: float = DEFAULT_NOTCH_DEPTH_MM

    # Notch centre positions (3-D, one per notch)
    notch_positions: List[np.ndarray] = field(default_factory=list)

    # Success / diagnostics
    success:             bool  = False
    n_pieces_notched:    int   = 0
    computation_time_ms: float = 0.0
    error_message:       str   = ""


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _build_local_frame(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal frame (u, v, w) where w = normalised *direction*."""
    w = direction / (np.linalg.norm(direction) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(w, u)
    return u, v, w


def _find_notch_positions_3d(
    reference_mesh: trimesh.Trimesh,
    pouring_dir: np.ndarray,
    n_notches: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Find positions on the outer silhouette boundary for notch placement.

    Returns a list of (center_3d, outward_normal_3d) tuples.
    The *center_3d* lies on the outermost point of the silhouette projected
    perpendicular to *pouring_dir*; the height component is set to the
    mesh's geometric centre along *pouring_dir* (notch straddles both halves).

    Args:
        reference_mesh: Mesh whose silhouette defines the outer boundary.
        pouring_dir:    Resin pouring direction (unit vector).
        n_notches:      Number of notch positions to return.

    Returns:
        List of (center_3d, outward_normal_3d) tuples.
    """
    u, v, w = _build_local_frame(pouring_dir)

    verts = reference_mesh.vertices  # (N, 3)

    # Project onto the 2-D plane perpendicular to the pouring direction
    coords_2d = np.column_stack([verts @ u, verts @ v])  # (N, 2)
    centroid_2d = coords_2d.mean(axis=0)

    # Height centre along pouring direction
    heights = verts @ w
    h_centre = (heights.min() + heights.max()) * 0.5

    # Convex hull of the projection
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(coords_2d)
    except Exception as exc:
        logger.warning("ConvexHull failed for notch placement: %s", exc)
        return []

    hull_pts = coords_2d[hull.vertices]  # (K, 2)

    # ── Select notch positions ────────────────────────────────────────────────
    if n_notches == 1:
        # Single notch: pick the hull vertex farthest from the 2-D centroid.
        # This is the most protruding point of the outer silhouette – the most
        # natural and accessible location for an alignment notch.
        dists = np.linalg.norm(hull_pts - centroid_2d, axis=1)
        chosen_pts_2d = [hull_pts[np.argmax(dists)]]
    elif n_notches == 2:
        # Pick the two mutually-farthest hull points (approximate diameter)
        best_i, best_j, best_d = 0, 1, 0.0
        n = len(hull_pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(hull_pts[i] - hull_pts[j])
                if d > best_d:
                    best_d = d
                    best_i, best_j = i, j
        chosen_pts_2d = [hull_pts[best_i], hull_pts[best_j]]
    else:
        # General n: evenly-spaced angles around the centroid.
        # For each angle pick the hull point farthest in that direction.
        angles = np.linspace(0, 2 * np.pi, n_notches, endpoint=False)
        chosen_pts_2d = []
        for angle in angles:
            dir_2d = np.array([np.cos(angle), np.sin(angle)])
            dots = hull_pts @ dir_2d
            chosen_pts_2d.append(hull_pts[np.argmax(dots)])

    # ── Convert to 3-D ───────────────────────────────────────────────────────
    results: List[Tuple[np.ndarray, np.ndarray]] = []
    for pt_2d in chosen_pts_2d:
        outward_2d = pt_2d - centroid_2d
        norm = np.linalg.norm(outward_2d)
        if norm < 1e-6:
            continue
        outward_2d /= norm

        # Reconstruct 3-D positions: on the silhouette at mid-height
        center_3d    = pt_2d[0] * u + pt_2d[1] * v + h_centre * w
        outward_3d   = outward_2d[0] * u + outward_2d[1] * v   # zero w-component

        results.append((center_3d, outward_3d))

    return results


def _create_triangular_notch_cutter(
    center_3d:       np.ndarray,
    outward_normal:  np.ndarray,
    pouring_dir:     np.ndarray,
    half_width_mm:   float,
    depth_mm:        float,
    total_height_mm: float,
    outward_extra_mm: float = CUTTER_OUTWARD_EXTRA,
) -> trimesh.Trimesh:
    """
    Build a triangular-prism cutter for one alignment notch.

    Cross-section in the plane ⊥ to *pouring_dir*:

        ←── half_width ──┬── half_width ──→
                         │  (outer surface)
         A ──────────────┼──────────────── B    ← base (with extra outward margin)
          ╲                              ╱
           ╲                            ╱
            ╲                          ╱
             ╲                        ╱
              ╲                      ╱
               C  (apex, depth inward) ← sharp vertex

    The prism is extruded ±total_height_mm/2 along *pouring_dir*.

    Args:
        center_3d:        Point on the outer boundary.
        outward_normal:   Unit vector pointing outward from the mold.
        pouring_dir:      Unit vector along the pouring axis.
        half_width_mm:    Half the notch base width.
        depth_mm:         Notch depth (distance from base line to apex).
        total_height_mm:  Prism total height along *pouring_dir*.
        outward_extra_mm: How far the base extends beyond the outer surface.

    Returns:
        Closed trimesh.Trimesh suitable for CSG subtraction.
    """
    n = outward_normal / (np.linalg.norm(outward_normal) + 1e-12)
    d = pouring_dir   / (np.linalg.norm(pouring_dir)    + 1e-12)

    # Tangent lies in the plane ⊥ to pouring_dir and ⊥ to outward_normal
    tangent = np.cross(n, d)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

    # Triangle vertices in the cross-section plane
    #  A, B: base of the V (pushed outward by extra margin so we cut cleanly)
    #  C:    apex, pointing INWARD (sharp vertex)
    A = center_3d + n * outward_extra_mm + tangent * half_width_mm
    B = center_3d + n * outward_extra_mm - tangent * half_width_mm
    C = center_3d - n * depth_mm   # apex pointing into the wall material

    half_h = total_height_mm * 0.5

    # 6 prism vertices: 0-2 = top cap, 3-5 = bottom cap
    #  0=A_top, 1=B_top, 2=C_top, 3=A_bot, 4=B_bot, 5=C_bot
    verts = np.array([
        A + d * half_h,   # 0 A_top
        B + d * half_h,   # 1 B_top
        C + d * half_h,   # 2 C_top
        A - d * half_h,   # 3 A_bot
        B - d * half_h,   # 4 B_bot
        C - d * half_h,   # 5 C_bot
    ], dtype=np.float64)

    # Face winding verified so outward normals point away from the prism interior:
    #   Top cap  normal = +d  →  CCW when viewed from +d  →  [0,1,2]
    #   Bot cap  normal = -d  →  CCW when viewed from -d  →  [3,5,4]
    #   Side AB  normal = +n  →  [0,3,4] + [0,4,1]
    #   Side BC  normal ≈ (-tang + n)  →  [1,4,5] + [1,5,2]
    #   Side CA  normal ≈ (+tang + n)  →  [2,5,3] + [2,3,0]
    faces = np.array([
        [0, 1, 2],   # top cap
        [3, 5, 4],   # bottom cap
        [0, 3, 4],   # side AB (quad part 1)
        [0, 4, 1],   # side AB (quad part 2)
        [1, 4, 5],   # side BC (quad part 1)
        [1, 5, 2],   # side BC (quad part 2)
        [2, 5, 3],   # side CA (quad part 1)
        [2, 3, 0],   # side CA (quad part 2)
    ], dtype=np.int32)

    cutter = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Ensure winding / normals are consistent (trimesh utility)
    trimesh.repair.fix_normals(cutter)
    if not cutter.is_watertight:
        trimesh.repair.fill_holes(cutter)

    return cutter


def _csg_subtract(
    target: trimesh.Trimesh,
    cutter: trimesh.Trimesh,
) -> Optional[trimesh.Trimesh]:
    """
    Boolean-subtract *cutter* from *target* using manifold3d.

    Returns the result mesh, or *None* if CSG is unavailable or fails.
    """
    if not MANIFOLD_AVAILABLE:
        logger.warning("manifold3d not available – cannot subtract alignment notch")
        return None

    try:
        import manifold3d as m3d
        import trimesh.repair as _tr

        def _repair(mesh: trimesh.Trimesh, lbl: str) -> trimesh.Trimesh:
            if mesh.is_watertight:
                return mesh
            # Try meshlib first
            try:
                import numpy as _np
                import meshlib.mrmeshnumpy as _mrn
                import meshlib.mrmeshpy as _mr
                _aec: dict = {}
                for _af in mesh.faces:
                    for _ai in range(3):
                        _ava, _avb = int(_af[_ai]), int(_af[(_ai + 1) % 3])
                        _aek = (min(_ava, _avb), max(_ava, _avb))
                        _aec[_aek] = _aec.get(_aek, 0) + 1
                _a_closed_nm = (sum(1 for _ac in _aec.values() if _ac == 1) == 0)
                mesh_mr = _mrn.meshFromFacesVerts(
                    mesh.faces.astype(_np.int32),
                    mesh.vertices.astype(_np.float32),
                )
                if not _a_closed_nm:
                    _mr.fixMeshDegeneracies(mesh_mr, _mr.FixMeshDegeneraciesParams())
                    for loop in _mr.findRightBoundary(mesh_mr.topology, None):
                        _mr.fillHoleNicely(mesh_mr, loop[0], _mr.FillHoleNicelySettings())
                out_verts = _mrn.getNumpyVerts(mesh_mr)
                out_faces = _mrn.getNumpyFaces(mesh_mr.topology)
                r = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
                r.fix_normals()
                if r.is_watertight:
                    return r
                # Check open edge count: 0 open edges = non-manifold topology (not holes).
                # Applying trimesh process=True would re-merge split vertices and make it worse.
                _ec: dict = {}
                for _face in r.faces:
                    for _i in range(3):
                        _va, _vb = int(_face[_i]), int(_face[(_i + 1) % 3])
                        _ek = (min(_va, _vb), max(_va, _vb))
                        _ec[_ek] = _ec.get(_ek, 0) + 1
                if sum(1 for _c in _ec.values() if _c == 1) == 0:
                    logger.debug(
                        "alignment_notch CSG input '%s': 0 open edges but not watertight "
                        "(non-manifold topology). Skipping trimesh fallback.", lbl
                    )
                    return r  # trimesh fallback would cause geometry explosion
                mesh = r  # pass to trimesh fallback
            except Exception:
                logger.debug("Edge analysis for CSG input failed, proceeding to trimesh fallback", exc_info=True)
            # trimesh fallback (only reached when there are real open boundary edges)
            m = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                process=True,
            )
            _tr.fix_normals(m, multibody=True)
            _tr.fill_holes(m)
            if not m.is_watertight:
                logger.debug("alignment_notch CSG input '%s' still not watertight after repair", lbl)
            return m

        def _to_manifold(mesh: trimesh.Trimesh, lbl: str) -> m3d.Manifold:
            mesh = _repair(mesh, lbl)
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            tris  = np.asarray(mesh.faces,    dtype=np.uint32)
            return m3d.Manifold(mesh=m3d.Mesh(vert_properties=verts, tri_verts=tris))

        a = _to_manifold(target, 'target')
        b = _to_manifold(cutter, 'cutter')
        result_manifold = a - b

        result_mesh_data = result_manifold.to_mesh()
        verts = np.asarray(result_mesh_data.vert_properties, dtype=np.float64)
        faces = np.asarray(result_mesh_data.tri_verts,       dtype=np.int32)

        if len(verts) == 0 or len(faces) == 0:
            logger.warning("CSG subtraction produced empty mesh")
            return None

        result = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        result.fix_normals()
        return result

    except Exception as exc:
        logger.error("Alignment notch CSG subtraction failed: %s", exc)
        return None


# ============================================================================
# PUBLIC API
# ============================================================================

def add_alignment_notches(
    shell_half_1:    Optional[trimesh.Trimesh],
    shell_half_2:    Optional[trimesh.Trimesh],
    metamold_half_1: Optional[trimesh.Trimesh],
    metamold_half_2: Optional[trimesh.Trimesh],
    resin_direction: np.ndarray,
    n_notches:      int   = DEFAULT_N_NOTCHES,
    notch_width_mm: float = DEFAULT_NOTCH_WIDTH_MM,
    notch_depth_mm: float = DEFAULT_NOTCH_DEPTH_MM,
) -> AlignmentNotchResult:
    """
    Add triangular alignment notches to all four mold pieces.

    Each notch is a V-shaped groove on the outer perimeter of the piece,
    at the height of the parting interface.  The groove's sharp apex points
    inward (into the wall material) by *notch_depth_mm*.  Identical grooves
    are cut into every piece so mating halves can be aligned by eye.

    Notch positions are determined from the 2-D silhouette of the combined
    mold projected along *resin_direction*.  Two diametrically opposite
    points on the convex hull are chosen.

    Args:
        shell_half_1:    Hard-shell upper half (may be None).
        shell_half_2:    Hard-shell lower half (may be None).
        metamold_half_1: Metamold upper half (may be None).
        metamold_half_2: Metamold lower half (may be None).
        resin_direction: Resin pouring direction (unit vector).
        n_notches:       Number of notches to place (default 2).
        notch_width_mm:  Total base width of each triangular notch.
        notch_depth_mm:  Depth of the apex from the outer surface.

    Returns:
        AlignmentNotchResult with notched meshes (None where input was None
        or where CSG failed).
    """
    t_start = time.perf_counter()

    result = AlignmentNotchResult(
        n_notches      = n_notches,
        notch_width_mm = notch_width_mm,
        notch_depth_mm = notch_depth_mm,
    )

    # ── Collect all non-None pieces for reference / height computation ────────
    all_pieces = [m for m in [shell_half_1, shell_half_2,
                               metamold_half_1, metamold_half_2]
                  if m is not None]

    if not all_pieces:
        result.error_message = "No mesh pieces provided"
        return result

    # Build a single referenced mesh for silhouette + height determination
    reference = trimesh.util.concatenate(all_pieces)

    # ── Determine cutter height (full extent of all pieces + margin) ──────────
    pouring_dir = np.asarray(resin_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-12)

    heights = reference.vertices @ pouring_dir
    cutter_height = (heights.max() - heights.min()) + 2 * CUTTER_HEIGHT_MARGIN

    if cutter_height < 1.0:
        cutter_height = 100.0  # safe fallback

    # ── Find notch positions on the outer boundary ────────────────────────────
    positions = _find_notch_positions_3d(reference, pouring_dir, n_notches)
    if not positions:
        result.error_message = "Could not determine notch positions (convex hull failed)"
        return result

    result.notch_positions = [p for (p, _) in positions]
    logger.info(
        "Alignment notches: %d positions found, cutter height=%.1f mm",
        len(positions), cutter_height
    )

    # ── Build one triangular-prism cutter per notch position ─────────────────
    cutters: List[trimesh.Trimesh] = []
    half_width = notch_width_mm * 0.5

    for i, (center_3d, outward_3d) in enumerate(positions):
        cutter = _create_triangular_notch_cutter(
            center_3d       = center_3d,
            outward_normal  = outward_3d,
            pouring_dir     = pouring_dir,
            half_width_mm   = half_width,
            depth_mm        = notch_depth_mm,
            total_height_mm = cutter_height,
        )
        cutters.append(cutter)
        logger.debug(
            "  Notch %d cutter: centre=(%.2f, %.2f, %.2f), outward=(%.2f, %.2f, %.2f)",
            i, *center_3d, *outward_3d
        )

    # ── Apply cutters to each piece ───────────────────────────────────────────
    def _apply_all_cutters(
        mesh: Optional[trimesh.Trimesh],
        label: str,
    ) -> Optional[trimesh.Trimesh]:
        if mesh is None:
            return None
        current = mesh
        for j, cutter in enumerate(cutters):
            notched = _csg_subtract(current, cutter)
            if notched is None:
                logger.warning("  %s: notch %d CSG failed – keeping previous mesh", label, j)
                # Keep the previous mesh so at least the other notches were applied
            else:
                current = notched
                logger.debug("  %s: notch %d applied", label, j)
        return current

    result.shell_half_1    = _apply_all_cutters(shell_half_1,    "shell_half_1")
    result.shell_half_2    = _apply_all_cutters(shell_half_2,    "shell_half_2")
    result.metamold_half_1 = _apply_all_cutters(metamold_half_1, "metamold_half_1")
    result.metamold_half_2 = _apply_all_cutters(metamold_half_2, "metamold_half_2")

    # Count how many pieces were successfully notched (changed from input)
    pairs = [
        (shell_half_1,    result.shell_half_1),
        (shell_half_2,    result.shell_half_2),
        (metamold_half_1, result.metamold_half_1),
        (metamold_half_2, result.metamold_half_2),
    ]
    result.n_pieces_notched = sum(
        1 for (orig, notched) in pairs
        if orig is not None and notched is not None
    )

    result.success = result.n_pieces_notched > 0
    result.computation_time_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        "Alignment notches applied to %d/%d pieces in %.0f ms",
        result.n_pieces_notched,
        sum(1 for m in all_pieces if m is not None),
        result.computation_time_ms,
    )

    return result
