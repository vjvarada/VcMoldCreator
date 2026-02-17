"""
Mold Fabrication Module

Creates the hard shell geometry for composite mold fabrication.

From the paper (Section 5):
> "The hard shell is a prism aligned with the resin pouring direction, from which
> we then subtract the inner soft mold volume; the prism's flat base is orthogonal
> to the resin pouring direction and shaped like the silhouette of the convex hull."

This module implements:
1. Outer shell hull creation (offset from current hull by wall_thickness)
2. Outer collar extension (extend parting surface to outer shell boundary)
3. Prism creation from extended parting surface boundary
4. Shell half separation

Author: VcMoldCreator
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

import trimesh
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

# CSG operations via abstraction layer (libigl/CGAL preferred, manifold3d fallback)
from . import csg_backend
CSG_AVAILABLE = csg_backend.CSG_AVAILABLE


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_WALL_THICKNESS_MM = 5.0  # Default hard shell wall thickness
DEFAULT_COLLAR_SUBDIVISIONS = 4  # Fan subdivisions at corners


# ============================================================================
# SHARED HELPERS (DRY consolidation)
# ============================================================================

def _build_pouring_basis(
    pouring_direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal basis from a pouring direction via Gram-Schmidt.

    Args:
        pouring_direction: Raw direction vector (will be normalised).

    Returns:
        (pouring_dir, u_axis, v_axis, world_to_2d) where
        - pouring_dir is the normalised input,
        - u_axis, v_axis span the plane perpendicular to it,
        - world_to_2d is the (3, 3) rotation matrix that projects
          world coordinates to (u, v, height) space.
    """
    pouring_dir = np.asarray(pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)

    arbitrary = (
        np.array([1.0, 0.0, 0.0])
        if abs(pouring_dir[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )

    u_axis = arbitrary - np.dot(arbitrary, pouring_dir) * pouring_dir
    u_axis = u_axis / np.linalg.norm(u_axis)

    v_axis = np.cross(pouring_dir, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    world_to_2d = np.column_stack([u_axis, v_axis, pouring_dir]).T  # (3, 3)

    return pouring_dir, u_axis, v_axis, world_to_2d


def _find_boundary_edges_with_winding(
    faces: np.ndarray,
) -> Tuple[List[Tuple[int, int]], Dict]:
    """Find boundary edges and their original winding from a face array.

    A boundary edge appears in exactly one face.

    Args:
        faces: (F, 3) int face array.

    Returns:
        (boundary_edges, edge_to_face) where
        - boundary_edges is a list of (v0, v1) edge keys (min, max),
        - edge_to_face maps edge_key → (face_idx, orig_v0, orig_v1)
          preserving the oriented half-edge from the parent face.
    """
    edge_count: Dict[Tuple[int, int], int] = {}
    edge_to_face: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
            if edge_key not in edge_to_face:
                edge_to_face[edge_key] = (fi, v0, v1)

    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    return boundary_edges, edge_to_face


def _chain_edges_into_loops(
    boundary_edges: List[Tuple[int, int]],
) -> List[List[int]]:
    """Chain unordered boundary edges into ordered vertex loops.

    Args:
        boundary_edges: List of (v0, v1) undirected edge tuples.

    Returns:
        List of loops, where each loop is an ordered list of vertex indices.
    """
    if not boundary_edges:
        return []

    # Build adjacency: vertex -> set of connected vertices
    adj: Dict[int, List[int]] = {}
    for v0, v1 in boundary_edges:
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    visited_edges: set = set()
    loops: List[List[int]] = []

    for start_v0, start_v1 in boundary_edges:
        edge_key = (min(start_v0, start_v1), max(start_v0, start_v1))
        if edge_key in visited_edges:
            continue

        # Walk a new loop
        loop = [start_v0, start_v1]
        visited_edges.add(edge_key)
        current = start_v1
        prev = start_v0

        while True:
            neighbors = adj.get(current, [])
            next_v = None
            for nb in neighbors:
                ek = (min(current, nb), max(current, nb))
                if ek not in visited_edges:
                    next_v = nb
                    break
            if next_v is None or next_v == start_v0:
                break
            visited_edges.add((min(current, next_v), max(current, next_v)))
            loop.append(next_v)
            prev = current
            current = next_v

        loops.append(loop)

    return loops


def _fill_inner_boundary_loops(surface: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fill inner boundary loops of an open surface mesh.

    The membrane (outer collar) typically has two kinds of boundary:
    - An **outer** boundary where it meets the prism walls (largest loop).
    - One or more **inner** boundaries where it meets the part surface.

    This function fills every inner loop with a centroid-fan triangulation so
    that ``_extrude_surface_to_volume`` only creates side walls along the outer
    boundary.  The filled region doesn't need to be geometrically perfect —
    it just needs to seal the opening so the extruded volume is solid.

    Args:
        surface: Open surface mesh with at least two boundary loops.

    Returns:
        A new trimesh with inner loops filled.  If the mesh has ≤ 1 boundary
        loop the original mesh is returned unchanged.
    """
    faces = np.asarray(surface.faces, dtype=np.int64)
    boundary_edges, _ = _find_boundary_edges_with_winding(faces)

    if not boundary_edges:
        return surface  # Already closed

    loops = _chain_edges_into_loops(boundary_edges)

    if len(loops) <= 1:
        return surface  # Single boundary — nothing inner to fill

    # Identify the outer loop as the one with the largest perimeter
    verts = np.asarray(surface.vertices, dtype=np.float64)
    perimeters = []
    for loop in loops:
        loop_verts = verts[loop]
        diffs = np.diff(np.vstack([loop_verts, loop_verts[:1]]), axis=0)
        perimeters.append(np.sum(np.linalg.norm(diffs, axis=1)))
    outer_idx = int(np.argmax(perimeters))

    # Build new vertex/face arrays with centroid-fan fills for inner loops
    new_verts = list(verts)
    new_faces = list(faces)
    n_filled = 0

    for i, loop in enumerate(loops):
        if i == outer_idx:
            continue  # Keep outer boundary open
        if len(loop) < 3:
            continue

        centroid = verts[loop].mean(axis=0)
        ci = len(new_verts)
        new_verts.append(centroid)

        for j in range(len(loop)):
            v1 = loop[j]
            v2 = loop[(j + 1) % len(loop)]
            new_faces.append([ci, v1, v2])
        n_filled += 1

    if n_filled == 0:
        return surface

    filled = trimesh.Trimesh(
        vertices=np.array(new_verts, dtype=np.float64),
        faces=np.array(new_faces, dtype=np.int64),
        process=False,
    )
    logger.info("  Filled %d inner boundary loop(s) in membrane "
                "(%d -> %d verts, %d -> %d faces)",
                n_filled, len(verts), len(filled.vertices),
                len(faces), len(filled.faces))
    return filled


def _extrude_surface_to_volume(
    surface: trimesh.Trimesh,
    direction: np.ndarray,
    offset_pos: float,
    offset_neg: float,
) -> trimesh.Trimesh:
    """Extrude a surface mesh along *direction* to create a closed volume.

    The resulting volume has:
    - A **top** cap at ``vertices + direction * offset_pos``
    - A **bottom** cap at ``vertices - direction * offset_neg``
    - Side quads stitching the boundary edges.

    Every extrusion variant in the module (cutting blade, tall blade,
    half-space, thick slab) reduces to choosing ``offset_pos`` and
    ``offset_neg``.

    Args:
        surface: The input surface mesh.
        direction: Unit vector for extrusion axis.
        offset_pos: Distance to extrude in the *+direction* side.
        offset_neg: Distance to extrude in the *-direction* side.

    Returns:
        Watertight extruded volume as a trimesh.Trimesh.
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    n_verts = len(vertices)

    top_vertices = vertices + direction * offset_pos
    bottom_vertices = vertices - direction * offset_neg

    all_vertices = np.vstack([bottom_vertices, top_vertices])

    # Bottom faces — reversed winding so normals point outward (downward)
    bottom_faces = faces[:, ::-1]
    # Top faces — original winding, offset indices
    top_faces = faces + n_verts

    # Side faces from boundary edges
    boundary_edges, edge_to_face = _find_boundary_edges_with_winding(faces)

    side_faces = []
    for v0, v1 in boundary_edges:
        _, orig_v0, _orig_v1 = edge_to_face[(v0, v1)]
        b0, b1 = v0, v1
        t0, t1 = v0 + n_verts, v1 + n_verts

        if orig_v0 == v0:
            side_faces.append([b0, b1, t1])
            side_faces.append([b0, t1, t0])
        else:
            side_faces.append([b0, t0, t1])
            side_faces.append([b0, t1, b1])

    all_faces_list = [bottom_faces, top_faces]
    if side_faces:
        all_faces_list.append(np.array(side_faces, dtype=np.int64))
    all_faces = np.vstack(all_faces_list)

    volume = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)
    volume.fix_normals()
    return volume


def _split_and_classify_halves(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh],
           List[trimesh.Trimesh], bool]:
    """Split a mesh into connected components and classify the two largest.

    The two largest components (by face count) are classified as
    *upper* (higher centroid projection onto *direction*) and *lower*.

    Uses face-adjacency connected components (trimesh.split).  If the
    default ``process=True`` split finds only 1 component, retries with
    an unprocessed copy (``process=False``) to rule out vertex-merging
    artefacts from the CSG backend.

    Args:
        mesh: The mesh to split.
        direction: Unit vector defining the "upper" direction.

    Returns:
        (upper_half, lower_half, all_components, did_split) where
        - upper_half / lower_half are the two largest components
          (None if fewer than 2 components),
        - all_components is the full list,
        - did_split is True when ≥ 2 components were found.
    """
    # ----- diagnostic: mesh stats before split -----
    logger.info("  Split input: %d verts, %d faces, watertight=%s",
                len(mesh.vertices), len(mesh.faces), mesh.is_watertight)

    components = mesh.split(only_watertight=False)
    logger.info("  Found %d connected components (process=True split)",
                len(components))

    # ----- fallback: retry WITHOUT vertex merging -----
    if len(components) < 2:
        logger.warning("  Only 1 component found — retrying split with "
                       "process=False (no vertex merging)...")
        raw_mesh = trimesh.Trimesh(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            process=False,
        )
        components = raw_mesh.split(only_watertight=False)
        logger.info("  Found %d connected components (process=False split)",
                    len(components))

    # ----- diagnostic: log component sizes -----
    if len(components) < 2:
        proj = mesh.vertices @ direction
        logger.warning("  SPLIT FAILED — mesh remains 1 component")
        logger.warning("    Vertex extent along direction: %.3f to %.3f",
                       float(proj.min()), float(proj.max()))
        logger.warning("    Bounds min: %s", mesh.bounds[0])
        logger.warning("    Bounds max: %s", mesh.bounds[1])
        return None, None, components, False

    components = sorted(components, key=lambda m: len(m.faces), reverse=True)
    for i, c in enumerate(components[:5]):
        proj_c = float(np.dot(c.centroid, direction))
        logger.info("    Component %d: %d verts, %d faces, "
                    "centroid·dir=%.2f, watertight=%s",
                    i, len(c.vertices), len(c.faces), proj_c, c.is_watertight)

    comp1, comp2 = components[0], components[1]

    proj1 = float(np.dot(comp1.centroid, direction))
    proj2 = float(np.dot(comp2.centroid, direction))

    if proj1 > proj2:
        upper, lower = comp1, comp2
    else:
        upper, lower = comp2, comp1

    if len(components) > 2:
        logger.warning("  %d additional fragments discarded", len(components) - 2)

    return upper, lower, components, True


def _log_blade_cut_diagnostics(
    shell: trimesh.Trimesh,
    membrane: trimesh.Trimesh,
    blade: trimesh.Trimesh,
    direction: np.ndarray,
) -> None:
    """Log detailed mesh diagnostics before a blade-cut CSG operation.

    Logs membrane/blade watertightness, boundary edge counts, bounding-box
    overlap along the pouring direction, and pre-cut shell connectivity.
    Only produces useful output when the logger is at DEBUG level.
    """
    logger.debug("Membrane: %d verts, %d faces, watertight=%s",
                 len(membrane.vertices), len(membrane.faces), membrane.is_watertight)
    membrane_boundary, _ = _find_boundary_edges_with_winding(np.asarray(membrane.faces))
    logger.debug("  Membrane boundary edges: %d", len(membrane_boundary))

    logger.debug("Blade: watertight=%s", blade.is_watertight)
    blade_boundary, _ = _find_boundary_edges_with_winding(np.asarray(blade.faces))
    if blade_boundary:
        logger.warning("Blade has %d open edges - not watertight!", len(blade_boundary))

    logger.debug("Bounds - Shell: %s to %s", shell.bounds[0], shell.bounds[1])
    logger.debug("Bounds - Membrane: %s to %s", membrane.bounds[0], membrane.bounds[1])
    logger.debug("Bounds - Blade: %s to %s", blade.bounds[0], blade.bounds[1])

    shell_proj = shell.vertices @ direction
    blade_proj = blade.vertices @ direction
    s_min, s_max = float(shell_proj.min()), float(shell_proj.max())
    b_min, b_max = float(blade_proj.min()), float(blade_proj.max())
    logger.debug("Shell extent along pouring dir: [%.2f, %.2f]", s_min, s_max)
    logger.debug("Blade extent along pouring dir: [%.2f, %.2f]", b_min, b_max)
    if b_max < s_min or b_min > s_max:
        logger.warning("Blade does NOT overlap with shell along pouring direction - cut may fail")

    components = shell.split(only_watertight=False)
    logger.debug("Shell has %d connected components BEFORE cutting", len(components))
    if len(components) > 1:
        for i, comp in enumerate(
            sorted(components, key=lambda m: len(m.faces), reverse=True)[:5]
        ):
            logger.debug("  Component %d: %d verts, %d faces",
                         i + 1, len(comp.vertices), len(comp.faces))


# ============================================================================
# SPLIT FAILURE DIAGNOSTICS
# ============================================================================

@dataclass
class SplitDiagnostics:
    """Diagnostic information when the metamold prism fails to split into two halves.

    Holds spatial coverage analysis, bridge-face detection results, and
    optional references to exported debug meshes.  Created by
    :func:`diagnose_split_failure`.
    """

    # --- Spatial coverage ---
    prism_bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prism_bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cutter_bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cutter_bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Coverage in the plane perpendicular to pouring direction (u, v)
    prism_span_u: float = 0.0  # prism extent along u-axis
    prism_span_v: float = 0.0  # prism extent along v-axis
    cutter_span_u: float = 0.0  # cutter extent along u-axis
    cutter_span_v: float = 0.0  # cutter extent along v-axis
    coverage_u_pct: float = 0.0  # cutter_span_u / prism_span_u * 100
    coverage_v_pct: float = 0.0  # cutter_span_v / prism_span_v * 100

    # Whether the cutter fully extends beyond the prism on each lateral side
    cutter_exceeds_u_neg: bool = False
    cutter_exceeds_u_pos: bool = False
    cutter_exceeds_v_neg: bool = False
    cutter_exceeds_v_pos: bool = False
    full_lateral_coverage: bool = False  # True if all four are True

    # --- Bridge face detection ---
    # Faces in the CSG result that lie near the membrane plane and keep
    # the two halves connected.
    n_bridge_faces: int = 0
    bridge_mesh: Optional[trimesh.Trimesh] = None  # submesh of bridge faces
    bridge_centroid: Optional[np.ndarray] = None  # average position of bridges
    bridge_bounds_min: Optional[np.ndarray] = None
    bridge_bounds_max: Optional[np.ndarray] = None

    # --- Gap analysis ---
    # Regions where the prism extends laterally but the cutter does not
    n_gap_sample_points: int = 0
    n_gaps_found: int = 0
    gap_locations: Optional[np.ndarray] = None  # (N, 3) positions of gaps

    # --- Exported files ---
    export_dir: Optional[str] = None
    exported_files: List[str] = field(default_factory=list)

    # Summary message for user display
    summary: str = ""


def diagnose_split_failure(
    prism_mesh: trimesh.Trimesh,
    combined_cutter: trimesh.Trimesh,
    csg_result: trimesh.Trimesh,
    membrane: trimesh.Trimesh,
    direction: np.ndarray,
    blade: Optional[trimesh.Trimesh] = None,
    export_dir: Optional[str] = None,
) -> SplitDiagnostics:
    """Diagnose why a prism failed to split into two halves after CSG subtraction.

    Performs three analyses and returns a :class:`SplitDiagnostics` object:

    1. **Spatial coverage** — checks whether the combined cutter extends beyond
       the prism on all four lateral sides (perpendicular to the pouring
       direction).  If the cutter is narrower than the prism anywhere, the
       blade left a gap.

    2. **Bridge-face detection** — finds faces in the CSG result that sit near
       the membrane plane.  These are the faces that keep the mesh as a single
       connected component rather than two halves.  A sub-mesh of just the
       bridge faces is built so it can be visualised in the viewer.

    3. **Gap ray-casting** — casts a grid of rays through the prism along the
       pouring direction and checks which rays do NOT intersect the combined
       cutter.  Gap locations show exactly where the blade fails to cut.

    Optionally exports all diagnostic meshes to *export_dir* as STL files
    for external inspection.

    Args:
        prism_mesh: The metamold prism before CSG.
        combined_cutter: The blade ∪ part mesh used for subtraction.
        csg_result: The CSG result (prism − combined_cutter).
        membrane: The parting surface / outer collar mesh.
        direction: Pouring direction unit vector.
        blade: Optional separate blade mesh (for export only).
        export_dir: Directory to write debug STL files.  ``None`` to skip.

    Returns:
        :class:`SplitDiagnostics` with all analysis results.
    """
    import os

    diag = SplitDiagnostics()
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Build a 2D projection basis
    pouring_dir, u_axis, v_axis, world_to_2d = _build_pouring_basis(direction)

    # ------------------------------------------------------------------
    # 1. Spatial coverage analysis
    # ------------------------------------------------------------------
    logger.info("=== SPLIT FAILURE DIAGNOSTICS ===")

    diag.prism_bbox_min = prism_mesh.bounds[0].copy()
    diag.prism_bbox_max = prism_mesh.bounds[1].copy()
    diag.cutter_bbox_min = combined_cutter.bounds[0].copy()
    diag.cutter_bbox_max = combined_cutter.bounds[1].copy()

    # Project vertices to (u, v, h) space
    prism_uv = prism_mesh.vertices @ world_to_2d.T   # (N, 3): columns = u, v, h
    cutter_uv = combined_cutter.vertices @ world_to_2d.T

    p_u_min, p_u_max = float(prism_uv[:, 0].min()), float(prism_uv[:, 0].max())
    p_v_min, p_v_max = float(prism_uv[:, 1].min()), float(prism_uv[:, 1].max())
    c_u_min, c_u_max = float(cutter_uv[:, 0].min()), float(cutter_uv[:, 0].max())
    c_v_min, c_v_max = float(cutter_uv[:, 1].min()), float(cutter_uv[:, 1].max())

    diag.prism_span_u = p_u_max - p_u_min
    diag.prism_span_v = p_v_max - p_v_min
    diag.cutter_span_u = c_u_max - c_u_min
    diag.cutter_span_v = c_v_max - c_v_min

    diag.coverage_u_pct = (diag.cutter_span_u / (diag.prism_span_u + 1e-10)) * 100
    diag.coverage_v_pct = (diag.cutter_span_v / (diag.prism_span_v + 1e-10)) * 100

    eps = 0.01  # 10 µm tolerance for "exceeds"
    diag.cutter_exceeds_u_neg = c_u_min <= p_u_min + eps
    diag.cutter_exceeds_u_pos = c_u_max >= p_u_max - eps
    diag.cutter_exceeds_v_neg = c_v_min <= p_v_min + eps
    diag.cutter_exceeds_v_pos = c_v_max >= p_v_max - eps
    diag.full_lateral_coverage = all([
        diag.cutter_exceeds_u_neg, diag.cutter_exceeds_u_pos,
        diag.cutter_exceeds_v_neg, diag.cutter_exceeds_v_pos,
    ])

    logger.info("Spatial coverage (projected onto plane perp. to pouring dir):")
    logger.info("  Prism  span: u=[%.2f, %.2f] (%.2fmm)  v=[%.2f, %.2f] (%.2fmm)",
                p_u_min, p_u_max, diag.prism_span_u,
                p_v_min, p_v_max, diag.prism_span_v)
    logger.info("  Cutter span: u=[%.2f, %.2f] (%.2fmm)  v=[%.2f, %.2f] (%.2fmm)",
                c_u_min, c_u_max, diag.cutter_span_u,
                c_v_min, c_v_max, diag.cutter_span_v)
    logger.info("  Coverage: u=%.1f%%  v=%.1f%%", diag.coverage_u_pct, diag.coverage_v_pct)
    logger.info("  Cutter exceeds prism edges:  u-=[%s]  u+=[%s]  v-=[%s]  v+=[%s]",
                diag.cutter_exceeds_u_neg, diag.cutter_exceeds_u_pos,
                diag.cutter_exceeds_v_neg, diag.cutter_exceeds_v_pos)

    if not diag.full_lateral_coverage:
        gaps = []
        if not diag.cutter_exceeds_u_neg:
            gaps.append("u- (cutter u_min=%.2f > prism u_min=%.2f, gap=%.2fmm)" %
                        (c_u_min, p_u_min, c_u_min - p_u_min))
        if not diag.cutter_exceeds_u_pos:
            gaps.append("u+ (cutter u_max=%.2f < prism u_max=%.2f, gap=%.2fmm)" %
                        (c_u_max, p_u_max, p_u_max - c_u_max))
        if not diag.cutter_exceeds_v_neg:
            gaps.append("v- (cutter v_min=%.2f > prism v_min=%.2f, gap=%.2fmm)" %
                        (c_v_min, p_v_min, c_v_min - p_v_min))
        if not diag.cutter_exceeds_v_pos:
            gaps.append("v+ (cutter v_max=%.2f < prism v_max=%.2f, gap=%.2fmm)" %
                        (c_v_max, p_v_max, p_v_max - c_v_max))
        logger.warning("  GAPS DETECTED on sides: %s", "; ".join(gaps))

    # ------------------------------------------------------------------
    # 2. Bridge face detection
    # ------------------------------------------------------------------
    # Find faces in the CSG result whose centroids lie close to the membrane
    # plane.  These faces form the "bridge" keeping the halves connected.
    membrane_heights = membrane.vertices @ direction
    membrane_h_min = float(membrane_heights.min())
    membrane_h_max = float(membrane_heights.max())
    membrane_h_mid = (membrane_h_min + membrane_h_max) / 2.0
    membrane_thickness = membrane_h_max - membrane_h_min

    # Tolerance: faces within ±(membrane_thickness + 1mm) of the membrane
    bridge_tol = max(membrane_thickness * 2.0, 1.0)

    result_verts = np.asarray(csg_result.vertices, dtype=np.float64)
    result_faces = np.asarray(csg_result.faces, dtype=np.int64)

    face_centroids = result_verts[result_faces].mean(axis=1)  # (F, 3)
    face_heights = face_centroids @ direction  # (F,)

    near_membrane = np.abs(face_heights - membrane_h_mid) < bridge_tol
    near_face_indices = np.where(near_membrane)[0]

    logger.info("Bridge face detection:")
    logger.info("  Membrane height range: [%.2f, %.2f] (mid=%.2f, thickness=%.3f)",
                membrane_h_min, membrane_h_max, membrane_h_mid, membrane_thickness)
    logger.info("  Bridge tolerance: ±%.2fmm from membrane mid-plane", bridge_tol)
    logger.info("  Faces near membrane plane: %d / %d total",
                len(near_face_indices), len(result_faces))

    if len(near_face_indices) > 0:
        # Build a sub-mesh of just the bridge faces
        bridge_faces_raw = result_faces[near_face_indices]
        # Remap vertex indices to create a compact sub-mesh
        unique_verts, inverse = np.unique(bridge_faces_raw.ravel(), return_inverse=True)
        bridge_verts = result_verts[unique_verts]
        bridge_faces_remapped = inverse.reshape(-1, 3)

        diag.bridge_mesh = trimesh.Trimesh(
            vertices=bridge_verts, faces=bridge_faces_remapped, process=False)
        diag.n_bridge_faces = len(near_face_indices)
        diag.bridge_centroid = bridge_verts.mean(axis=0)
        diag.bridge_bounds_min = bridge_verts.min(axis=0)
        diag.bridge_bounds_max = bridge_verts.max(axis=0)

        # Project bridge faces to (u, v) to see WHERE the bridges are
        bridge_uv = bridge_verts @ world_to_2d.T
        logger.info("  Bridge faces bounds (world): [%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f]",
                    *diag.bridge_bounds_min, *diag.bridge_bounds_max)
        logger.info("  Bridge faces bounds (u,v):   u=[%.2f, %.2f]  v=[%.2f, %.2f]",
                    bridge_uv[:, 0].min(), bridge_uv[:, 0].max(),
                    bridge_uv[:, 1].min(), bridge_uv[:, 1].max())

        # Check if removing bridge faces disconnects the mesh
        remaining_mask = np.ones(len(result_faces), dtype=bool)
        remaining_mask[near_face_indices] = False
        remaining_faces = result_faces[remaining_mask]
        if len(remaining_faces) > 0:
            remaining_mesh = trimesh.Trimesh(
                vertices=result_verts, faces=remaining_faces, process=False)
            remaining_components = remaining_mesh.split(only_watertight=False)
            logger.info("  After removing bridge faces: %d components (need ≥2 for split)",
                        len(remaining_components))
            if len(remaining_components) >= 2:
                sizes = sorted([len(c.faces) for c in remaining_components], reverse=True)
                logger.info("  Component sizes: %s", sizes[:5])
                logger.info("  → Removing bridge faces WOULD split the mesh!")
                logger.info("  → The blade/cutter has gaps near these bridge locations")
            else:
                logger.info("  → Removing bridge faces still leaves 1 component")
                logger.info("  → The connection may be further from the membrane plane")
                # Try increasing tolerance
                for mult in [4.0, 8.0, 16.0]:
                    wider_tol = max(membrane_thickness * mult, mult)
                    wider_mask = np.abs(face_heights - membrane_h_mid) < wider_tol
                    wider_remaining = result_faces[~wider_mask]
                    if len(wider_remaining) > 0:
                        wider_mesh = trimesh.Trimesh(
                            vertices=result_verts, faces=wider_remaining, process=False)
                        n_comp = len(wider_mesh.split(only_watertight=False))
                        logger.info("    tol=±%.1fmm → removing %d faces → %d components",
                                    wider_tol, int(wider_mask.sum()), n_comp)
                        if n_comp >= 2:
                            logger.info("    → Connection found within ±%.1fmm of membrane", wider_tol)
                            break

    # ------------------------------------------------------------------
    # 3. Gap ray-casting analysis
    # ------------------------------------------------------------------
    # Cast rays parallel to pouring direction through a grid, check which
    # go through the prism but NOT through the cutter.
    logger.info("Gap ray-casting analysis:")
    grid_res = 30  # 30x30 grid
    u_samples = np.linspace(p_u_min + 0.5, p_u_max - 0.5, grid_res)
    v_samples = np.linspace(p_v_min + 0.5, p_v_max - 0.5, grid_res)
    uu, vv = np.meshgrid(u_samples, v_samples)
    grid_points_uv = np.column_stack([uu.ravel(), vv.ravel()])  # (N, 2)

    # Convert grid points from (u, v) back to 3D world at the membrane mid-height
    grid_3d = (grid_points_uv[:, 0:1] * u_axis[np.newaxis, :] +
               grid_points_uv[:, 1:2] * v_axis[np.newaxis, :] +
               membrane_h_mid * direction[np.newaxis, :])

    # Ray origins: start well above the membrane plane
    ray_height_offset = max(diag.prism_span_u, diag.prism_span_v) * 2
    ray_origins = grid_3d + direction * ray_height_offset
    ray_directions = np.tile(-direction, (len(ray_origins), 1))  # shoot downward

    # Test rays against prism
    try:
        prism_hits = prism_mesh.ray.intersects_any(ray_origins, ray_directions)
    except Exception:
        prism_hits = np.zeros(len(ray_origins), dtype=bool)

    # Test rays against combined cutter
    try:
        cutter_hits = combined_cutter.ray.intersects_any(ray_origins, ray_directions)
    except Exception:
        cutter_hits = np.zeros(len(ray_origins), dtype=bool)

    # Gap = hits prism but NOT cutter
    gap_mask = prism_hits & ~cutter_hits
    diag.n_gap_sample_points = int(prism_hits.sum())
    diag.n_gaps_found = int(gap_mask.sum())

    if diag.n_gaps_found > 0:
        diag.gap_locations = grid_3d[gap_mask]
        logger.warning("  %d / %d rays through prism do NOT hit the cutter (%.1f%% gap coverage)",
                       diag.n_gaps_found, diag.n_gap_sample_points,
                       diag.n_gaps_found / max(diag.n_gap_sample_points, 1) * 100)
        gap_uv = grid_points_uv[gap_mask]
        logger.warning("  Gap region (u,v): u=[%.2f, %.2f]  v=[%.2f, %.2f]",
                       gap_uv[:, 0].min(), gap_uv[:, 0].max(),
                       gap_uv[:, 1].min(), gap_uv[:, 1].max())
    else:
        logger.info("  All %d rays through prism also hit the cutter — no lateral gaps detected",
                    diag.n_gap_sample_points)
        if diag.n_bridge_faces > 0:
            logger.info("  → Bridges exist but no ray gaps — the cutter may be coplanar")
            logger.info("    with prism faces, causing CSG degeneracy (try thicker blade)")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    parts = []
    if not diag.full_lateral_coverage:
        parts.append("Cutter does NOT extend beyond prism on all sides — "
                     "the blade/collar is too narrow")
    if diag.n_gaps_found > 0:
        parts.append("%d gap locations where prism exists but cutter does not" %
                     diag.n_gaps_found)
    if diag.n_bridge_faces > 0:
        parts.append("%d bridge faces near the membrane keep halves connected" %
                     diag.n_bridge_faces)
    if not parts:
        parts.append("No obvious geometric gaps — may be a CSG precision issue "
                     "(try increasing blade_thickness)")
    diag.summary = "; ".join(parts)
    logger.info("DIAGNOSIS: %s", diag.summary)

    # ------------------------------------------------------------------
    # 5. Export diagnostic meshes
    # ------------------------------------------------------------------
    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)
        diag.export_dir = export_dir
        _export_diag = [
            ("debug_prism.stl", prism_mesh),
            ("debug_combined_cutter.stl", combined_cutter),
            ("debug_csg_result.stl", csg_result),
            ("debug_membrane.stl", membrane),
        ]
        if blade is not None:
            _export_diag.append(("debug_blade.stl", blade))
        if diag.bridge_mesh is not None:
            _export_diag.append(("debug_bridge_faces.stl", diag.bridge_mesh))
        if diag.gap_locations is not None and len(diag.gap_locations) > 0:
            # Create small spheres at each gap location for visualization
            gap_spheres = []
            sphere_r = max(diag.prism_span_u, diag.prism_span_v) * 0.01
            for pt in diag.gap_locations[:200]:  # limit to 200 spheres
                s = trimesh.creation.icosphere(subdivisions=1, radius=sphere_r)
                s.apply_translation(pt)
                gap_spheres.append(s)
            if gap_spheres:
                gap_mesh = trimesh.util.concatenate(gap_spheres)
                _export_diag.append(("debug_gap_locations.stl", gap_mesh))

        for fname, mesh in _export_diag:
            fpath = os.path.join(export_dir, fname)
            try:
                mesh.export(fpath)
                diag.exported_files.append(fpath)
                logger.info("  Exported: %s (%d faces)", fpath, len(mesh.faces))
            except Exception as e:
                logger.warning("  Failed to export %s: %s", fpath, e)

    logger.info("=== END SPLIT FAILURE DIAGNOSTICS ===")
    return diag


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OuterShellHullResult:
    """Result of creating the outer shell hull (offset from inner hull)."""
    
    # The outer shell hull mesh
    mesh: trimesh.Trimesh
    
    # The original inner hull for reference
    inner_hull: trimesh.Trimesh
    
    # Wall thickness used
    wall_thickness: float
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass
class OuterCollarResult:
    """Result of extending parting surface outward to shell boundary."""
    
    # The extended parting surface mesh (with outer collar)
    mesh: trimesh.Trimesh
    
    # Vertex positions
    vertices: np.ndarray
    faces: np.ndarray
    
    # Original parting surface vertex count (before extension)
    original_vertex_count: int = 0
    original_face_count: int = 0
    
    # New vertices/faces added for the collar
    collar_vertex_count: int = 0
    collar_face_count: int = 0
    
    # Extension distance used (2x wall thickness to fully cut prism)
    extension_distance: float = 0.0
    
    # Which vertices are on the outer boundary (after extension)
    outer_boundary_vertices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    
    # The outer boundary loop (ordered vertex indices)
    outer_boundary_loop: List[int] = field(default_factory=list)
    
    # Statistics
    n_outer_boundary_edges_processed: int = 0
    n_corner_fans_created: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass 
class HardShellHalfResult:
    """Result of generating one half of the hard shell."""
    
    # The shell half mesh
    mesh: trimesh.Trimesh
    
    # Which mold half this belongs to (1 or 2)
    mold_half: int
    
    # The prism base polygon (2D silhouette)
    prism_base_vertices: np.ndarray  # (N, 2)
    
    # The pouring direction used
    pouring_direction: np.ndarray  # (3,)
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass
class HardShellPrismResult:
    """Result of creating the hard shell prism.
    
    The prism is aligned with the resin pouring direction, with flat bases
    orthogonal to that direction, shaped like the silhouette of the convex hull.
    (per paper Section 5)
    """
    
    # The full prism mesh (before subtracting mold cavity)
    prism_mesh: trimesh.Trimesh
    
    # The 2D silhouette polygon (convex hull projection)
    silhouette_2d: np.ndarray  # (N, 2) ordered vertices of 2D convex hull
    
    # The 3D silhouette polygon (on the base plane)
    silhouette_3d: np.ndarray  # (N, 3) 3D positions on base plane
    
    # The base plane center and normal
    base_plane_center: np.ndarray  # (3,)
    base_plane_normal: np.ndarray  # (3,) = pouring direction
    
    # Prism height (total extrusion distance)
    prism_height: float
    
    # The transformation matrix from world to 2D projection plane
    world_to_2d: np.ndarray  # (3, 3) rotation matrix
    
    # The pouring direction used
    pouring_direction: np.ndarray  # (3,)
    
    # Wall thickness added around the silhouette
    wall_thickness: float
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


@dataclass
class HardShellSplitResult:
    """Result of CSG operations to create two hard shell halves.
    
    The workflow is:
    1. Subtract hull from prism to create cavity
    2. Split along parting surface (with extended collar) into two halves
    """
    
    # The two shell halves (H1 and H2)
    shell_half_1: Optional[trimesh.Trimesh] = None  # Upper half (aligned with d1)
    shell_half_2: Optional[trimesh.Trimesh] = None  # Lower half (aligned with d2)
    
    # Intermediate result: prism with cavity (before splitting)
    shell_with_cavity: Optional[trimesh.Trimesh] = None
    
    # The pouring direction used
    pouring_direction: Optional[np.ndarray] = None
    
    # Statistics for each half
    half_1_vertex_count: int = 0
    half_1_face_count: int = 0
    half_2_vertex_count: int = 0
    half_2_face_count: int = 0
    
    # Computation times for each step
    cavity_time_ms: float = 0.0
    split_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Success flags
    cavity_success: bool = False
    split_success: bool = False


@dataclass
class MetamoldPrismResult:
    """Result of creating the metamold prism.
    
    The metamold is a mold used to cast the silicone soft mold itself.
    It uses the same silhouette/direction as the hard shell (hull + resin direction)
    but the height is based on the PART MESH extent, not the parting surface.
    
    The prism is aligned with the resin pouring direction (same as hard shell), with:
    - Silhouette derived from the HULL projection (same as hard shell)
    - Height based on PART MESH extent + configurable offsets above/below
    """
    
    # The prism mesh (before subtracting any cavities)
    prism_mesh: trimesh.Trimesh
    
    # The 2D silhouette polygon (convex hull of parting surface projection)
    silhouette_2d: np.ndarray  # (N, 2) ordered vertices of 2D convex hull
    
    # The 3D silhouette polygon (on the base plane)
    silhouette_3d: np.ndarray  # (N, 3) 3D positions on base plane
    
    # The base plane center and normal
    base_plane_center: np.ndarray  # (3,)
    base_plane_normal: np.ndarray  # (3,) = silicone pouring direction
    
    # Prism dimensions
    prism_height: float  # Total extrusion distance
    height_above_surface: float  # Distance from parting surface max to top
    height_below_surface: float  # Distance from parting surface min to bottom
    
    # The transformation matrix from world to 2D projection plane
    world_to_2d: np.ndarray  # (3, 3) rotation matrix
    
    # The silicone pouring direction used
    pouring_direction: np.ndarray  # (3,)
    
    # Wall thickness (horizontal offset from silhouette)
    wall_thickness: float
    
    # Reference heights from part mesh
    part_mesh_min_height: float  # Min projection along pouring direction
    part_mesh_max_height: float  # Max projection along pouring direction
    
    # Statistics
    vertex_count: int = 0
    face_count: int = 0
    
    # Computation time
    computation_time_ms: float = 0.0


# ============================================================================
# METAMOLD PRISM CREATION
# ============================================================================

def create_metamold_prism(
    hull_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    resin_pouring_direction: np.ndarray,
    wall_thickness: float = 5.0,
    margin: float = 0.0,
    height_above: float = 2.0,
    height_below: float = 2.0,
    parting_surface: trimesh.Trimesh = None
) -> MetamoldPrismResult:
    """
    Create a metamold prism aligned with the resin pouring direction.
    
    The metamold is used to cast the silicone soft mold. It uses the EXACT SAME
    silhouette and extrusion direction as the hard shell prism:
    
    - Silhouette: 2D convex hull of HULL vertices projected onto a plane
                  perpendicular to the RESIN pouring direction (same as hard shell)
    - Extrusion: Along RESIN pouring direction (same as hard shell)
    - Wall offset: wall_thickness + margin (same as hard shell)
    - Height: From (part_min - height_below) to (part_max + height_above)
              where min/max are the PART MESH extent along pouring direction
              IMPORTANT: Also extends to include parting_surface if provided,
              ensuring the cutting blade can pass through the prism walls.
    
    This ensures the metamold has the EXACT same footprint as the hard shell and is
    tall enough to enclose the entire part with 2mm margin on each side.
    
    Args:
        hull_mesh: The inflated hull mesh (used for silhouette, same as hard shell)
        part_mesh: The original part mesh (used for height reference)
        resin_pouring_direction: Unit vector for RESIN pouring direction (same as hard shell)
        wall_thickness: How thick the hard shell wall should be (mm)
        margin: Additional margin beyond the hull bounds (mm) - same as hard shell
        height_above: Distance above part mesh max to top of prism (mm)
        height_below: Distance below part mesh min to bottom of prism (mm)
        parting_surface: Optional parting surface mesh - if provided, prism height
                        will be extended to include it (ensuring blade cuts through)
        
    Returns:
        MetamoldPrismResult with the prism mesh and metadata
    """
    start_time = time.time()
    
    logger.info(f"Creating metamold prism aligned with resin direction: {resin_pouring_direction}")
    logger.info(f"  Wall thickness: {wall_thickness}mm, margin: {margin}mm, height_above: {height_above}mm, height_below: {height_below}mm")
    
    # Normalize pouring direction (same as hard shell)
    pouring_dir = np.array(resin_pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    # Use hull vertices for silhouette (same as hard shell)
    hull_vertices = np.array(hull_mesh.vertices, dtype=np.float64)
    # Use part mesh vertices for height extent
    part_vertices = np.array(part_mesh.vertices, dtype=np.float64)
    
    # =========================================================================
    # Step 1: Build orthonormal basis for the projection plane (same as hard shell)
    # =========================================================================
    pouring_dir, _u, _v, world_to_2d = _build_pouring_basis(pouring_dir)
    
    # =========================================================================
    # Step 2: Project part mesh vertices to get height extent
    # =========================================================================
    
    part_transformed = part_vertices @ world_to_2d.T
    part_heights = part_transformed[:, 2]  # projection along pouring direction
    
    # Get height extent of the part mesh
    part_min_height = np.min(part_heights)
    part_max_height = np.max(part_heights)
    
    logger.info(f"Part mesh height extent: {part_min_height:.2f} to {part_max_height:.2f}")
    
    # =========================================================================
    # Step 3: Project HULL vertices to get 2D silhouette (SAME as hard shell)
    # =========================================================================
    
    hull_transformed = hull_vertices @ world_to_2d.T
    projected_2d = hull_transformed[:, :2]  # (N, 2)
    
    try:
        hull_2d = ConvexHull(projected_2d)
        silhouette_indices = hull_2d.vertices
        silhouette_2d = projected_2d[silhouette_indices]
    except Exception as e:
        logger.error(f"Failed to compute 2D convex hull: {e}")
        # Fallback: use bounding box
        min_2d = np.min(projected_2d, axis=0)
        max_2d = np.max(projected_2d, axis=0)
        silhouette_2d = np.array([
            [min_2d[0], min_2d[1]],
            [max_2d[0], min_2d[1]],
            [max_2d[0], max_2d[1]],
            [min_2d[0], max_2d[1]],
        ], dtype=np.float64)
    
    logger.info(f"2D silhouette from hull has {len(silhouette_2d)} vertices")
    
    # =========================================================================
    # Step 4: Offset the silhouette outward by wall_thickness + margin (SAME as hard shell)
    # =========================================================================
    
    total_offset = wall_thickness + margin
    silhouette_2d_offset = _offset_polygon_2d(silhouette_2d, total_offset)
    
    logger.info(f"Offset silhouette has {len(silhouette_2d_offset)} vertices (offset={total_offset}mm)")
    
    # =========================================================================
    # Step 5: Determine prism height (based on part mesh + offsets)
    #         Also extend to include parting surface if provided
    # =========================================================================
    
    min_height = part_min_height - height_below
    max_height = part_max_height + height_above
    
    # If parting surface is provided, extend bounds to include it
    # This ensures the cutting blade (at parting surface level) passes through walls
    if parting_surface is not None:
        ps_vertices = np.array(parting_surface.vertices, dtype=np.float64)
        ps_transformed = ps_vertices @ world_to_2d.T
        ps_heights = ps_transformed[:, 2]  # projection along pouring direction
        ps_min_height = np.min(ps_heights)
        ps_max_height = np.max(ps_heights)
        
        logger.info(f"Parting surface height extent: {ps_min_height:.2f} to {ps_max_height:.2f}")
        logger.info(f"Part mesh height extent: {part_min_height:.2f} to {part_max_height:.2f}")
        
        # Extend prism bounds to include parting surface with margin
        parting_margin = 2.0  # 2mm margin around parting surface
        if ps_min_height - parting_margin < min_height:
            old_min = min_height
            min_height = ps_min_height - parting_margin
            logger.info(f"Extended prism bottom from {old_min:.2f} to {min_height:.2f} to include parting surface")
        
        if ps_max_height + parting_margin > max_height:
            old_max = max_height
            max_height = ps_max_height + parting_margin
            logger.info(f"Extended prism top from {old_max:.2f} to {max_height:.2f} to include parting surface")
    
    prism_height = max_height - min_height
    
    logger.info(f"Prism height: {prism_height:.2f} (from {min_height:.2f} to {max_height:.2f})")
    
    # =========================================================================
    # Step 6: Create prism mesh by extruding the silhouette
    # =========================================================================
    
    prism_mesh, silhouette_3d_bottom, silhouette_3d_top = _extrude_polygon_to_prism(
        silhouette_2d_offset,
        world_to_2d,
        min_height,
        max_height
    )
    
    # Compute base plane center (centroid of bottom silhouette)
    base_plane_center = np.mean(silhouette_3d_bottom, axis=0)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    result = MetamoldPrismResult(
        prism_mesh=prism_mesh,
        silhouette_2d=silhouette_2d_offset,
        silhouette_3d=silhouette_3d_bottom,
        base_plane_center=base_plane_center,
        base_plane_normal=pouring_dir,
        prism_height=prism_height,
        height_above_surface=height_above,
        height_below_surface=height_below,
        world_to_2d=world_to_2d,
        pouring_direction=pouring_dir,
        wall_thickness=wall_thickness,
        part_mesh_min_height=part_min_height,
        part_mesh_max_height=part_max_height,
        vertex_count=len(prism_mesh.vertices),
        face_count=len(prism_mesh.faces),
        computation_time_ms=elapsed_ms
    )
    
    logger.info(f"Metamold prism created: {result.vertex_count} vertices, "
                f"{result.face_count} faces in {elapsed_ms:.1f}ms")
    
    return result


# ============================================================================
# HARD SHELL PRISM CREATION (Paper Section 5)
# ============================================================================

def create_hard_shell_prism(
    hull_mesh: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    wall_thickness: float = DEFAULT_WALL_THICKNESS_MM,
    margin: float = 2.0
) -> HardShellPrismResult:
    """
    Create the hard shell prism aligned with the pouring direction.
    
    From the paper (Section 5):
    > "The hard shell is a prism aligned with the resin pouring direction, from which
    > we then subtract the inner soft mold volume; the prism's flat base is orthogonal
    > to the resin pouring direction and shaped like the silhouette of the convex hull."
    
    Algorithm:
    1. Define a plane perpendicular to the pouring direction
    2. Project all hull vertices onto this plane
    3. Compute the 2D convex hull of the projected points (silhouette)
    4. Offset the silhouette outward by wall_thickness
    5. Extrude the silhouette along the pouring direction to create a prism
    
    Args:
        hull_mesh: The inflated hull mesh (∂H)
        pouring_direction: Unit vector for resin pouring direction (f_resin)
        wall_thickness: How thick the hard shell wall should be (mm)
        margin: Additional margin beyond the hull bounds (mm)
        
    Returns:
        HardShellPrismResult with the prism mesh and metadata
    """
    start_time = time.time()
    
    logger.info(f"Creating hard shell prism aligned with pouring direction: {pouring_direction}")
    
    # Normalize pouring direction
    pouring_dir = np.array(pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    vertices = np.array(hull_mesh.vertices, dtype=np.float64)
    
    # =========================================================================
    # Step 1: Build orthonormal basis for the projection plane
    # =========================================================================
    pouring_dir, _u, _v, world_to_2d = _build_pouring_basis(pouring_dir)
    
    # =========================================================================
    # Step 2: Project vertices onto the plane and compute 2D silhouette
    # =========================================================================
    
    # Project vertices: get (u, v) coordinates  
    vertices_transformed = vertices @ world_to_2d.T
    projected_2d = vertices_transformed[:, :2]  # (N, 2)
    heights = vertices_transformed[:, 2]  # projection along pouring direction
    
    # Compute 2D convex hull (silhouette)
    try:
        hull_2d = ConvexHull(projected_2d)
        silhouette_indices = hull_2d.vertices
        silhouette_2d = projected_2d[silhouette_indices]
    except Exception as e:
        logger.error(f"Failed to compute 2D convex hull: {e}")
        # Fallback: use bounding box
        min_2d = np.min(projected_2d, axis=0)
        max_2d = np.max(projected_2d, axis=0)
        silhouette_2d = np.array([
            [min_2d[0], min_2d[1]],
            [max_2d[0], min_2d[1]],
            [max_2d[0], max_2d[1]],
            [min_2d[0], max_2d[1]],
        ], dtype=np.float64)
    
    logger.info(f"2D silhouette has {len(silhouette_2d)} vertices")
    
    # =========================================================================
    # Step 3: Offset the silhouette outward by wall_thickness + margin
    # =========================================================================
    
    total_offset = wall_thickness + margin
    silhouette_2d_offset = _offset_polygon_2d(silhouette_2d, total_offset)
    
    logger.info(f"Offset silhouette has {len(silhouette_2d_offset)} vertices")
    
    # =========================================================================
    # Step 4: Determine prism height (extrusion distance)
    # =========================================================================
    
    min_height = np.min(heights) - total_offset
    max_height = np.max(heights) + total_offset
    prism_height = max_height - min_height
    
    logger.info(f"Prism height: {prism_height:.2f} (from {min_height:.2f} to {max_height:.2f})")
    
    # =========================================================================
    # Step 5: Create prism mesh by extruding the silhouette
    # =========================================================================
    
    prism_mesh, silhouette_3d_bottom, silhouette_3d_top = _extrude_polygon_to_prism(
        silhouette_2d_offset,
        world_to_2d,
        min_height,
        max_height
    )
    
    # Compute base plane center (centroid of bottom silhouette)
    base_plane_center = np.mean(silhouette_3d_bottom, axis=0)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    result = HardShellPrismResult(
        prism_mesh=prism_mesh,
        silhouette_2d=silhouette_2d_offset,
        silhouette_3d=silhouette_3d_bottom,
        base_plane_center=base_plane_center,
        base_plane_normal=pouring_dir,
        prism_height=prism_height,
        world_to_2d=world_to_2d,
        pouring_direction=pouring_dir,
        wall_thickness=wall_thickness,
        vertex_count=len(prism_mesh.vertices),
        face_count=len(prism_mesh.faces),
        computation_time_ms=elapsed_ms
    )
    
    logger.info(f"Hard shell prism created: {result.vertex_count} vertices, "
                f"{result.face_count} faces in {elapsed_ms:.1f}ms")
    
    return result


# ============================================================================
# CSG OPERATIONS FOR HARD SHELL (Cavity + Split)
# ============================================================================

# NOTE: _trimesh_to_manifold / _manifold_to_trimesh removed.
# All CSG operations now go through csg_backend (libigl/CGAL preferred).


def create_shell_with_cavity(
    prism_mesh: trimesh.Trimesh,
    hull_mesh: trimesh.Trimesh
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Subtract the hull (mold cavity) from the prism to create the shell.
    
    This implements: shell = prism - hull
    
    Uses robust CSG boolean operations [Zhou et al. 2016] via csg_backend.
    
    Args:
        prism_mesh: The hard shell prism mesh
        hull_mesh: The inflated hull mesh (defines the cavity)
        
    Returns:
        Tuple of (shell_with_cavity, computation_time_ms, success)
    """
    start_time = time.time()
    
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available - cannot perform CSG operations")
        return None, 0.0, False
    
    try:
        logger.info("Creating shell with cavity (prism - hull)...")
        
        shell_mesh = csg_backend.csg_difference(prism_mesh, hull_mesh)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell with cavity created: {len(shell_mesh.vertices)} vertices, "
                   f"{len(shell_mesh.faces)} faces in {elapsed_ms:.1f}ms")
        
        return shell_mesh, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to create shell with cavity: {e}")
        return None, elapsed_ms, False


def _create_cutting_volume_from_surface(
    parting_surface: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    thickness: float = 100.0
) -> trimesh.Trimesh:
    """Create a thick volume slab from the parting surface for CSG splitting.

    Delegates to :func:`_extrude_surface_to_volume` with symmetric offsets.

    Args:
        parting_surface: The parting surface mesh (with extended collar).
        pouring_direction: The pouring direction (extrusion axis).
        thickness: Total slab thickness (extends thickness/2 each side).

    Returns:
        Watertight volume mesh.
    """
    half = thickness / 2.0
    return _extrude_surface_to_volume(parting_surface, pouring_direction, half, half)


def _create_half_space_from_membrane(
    membrane: trimesh.Trimesh,
    direction: np.ndarray,
    extrusion_distance: float = 500.0
) -> trimesh.Trimesh:
    """Create a one-sided half-space volume bounded by the membrane.

    Delegates to :func:`_extrude_surface_to_volume` with asymmetric offsets
    (offset_pos = extrusion_distance, offset_neg = 0).

    Args:
        membrane: The membrane mesh.
        direction: Extrusion direction (defines "which side").
        extrusion_distance: How far to extrude.

    Returns:
        Watertight half-space volume mesh.
    """
    return _extrude_surface_to_volume(membrane, direction, extrusion_distance, 0.0)


def _create_cutting_blade_from_membrane(
    membrane: trimesh.Trimesh,
    direction: np.ndarray,
    thickness: float = 0.0001
) -> trimesh.Trimesh:
    """Create a thin symmetric blade volume from the membrane for CSG cutting.

    Delegates to :func:`_extrude_surface_to_volume` with symmetric micro-offsets.

    Args:
        membrane: The membrane mesh.
        direction: Unit vector perpendicular to the membrane.
        thickness: Total blade thickness (default 0.1 µm).

    Returns:
        Watertight blade volume mesh.
    """
    half = thickness / 2.0
    return _extrude_surface_to_volume(membrane, direction, half, half)


def _create_tall_cutting_blade(
    membrane: trimesh.Trimesh,
    direction: np.ndarray,
    extrusion_distance: float = 100.0,
    blade_thickness: float = 1.0
) -> trimesh.Trimesh:
    """Create a tall cutting blade extending far above and below the membrane.

    Delegates to :func:`_extrude_surface_to_volume` with large symmetric
    offsets plus a micro-gap for CSG subtraction.

    Args:
        membrane: The membrane mesh (parting surface + outer collar).
        direction: Pouring direction (extrusion axis).
        extrusion_distance: How far to extrude each side (default 100 mm).
        blade_thickness: Gap thickness for CSG subtraction (default 0.1 µm).

    Returns:
        Tall prism-shaped blade mesh.
    """
    half_gap = blade_thickness / 2.0
    offset = extrusion_distance + half_gap
    return _extrude_surface_to_volume(membrane, direction, offset, offset)


def extend_membrane_height(
    membrane: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    target_min_height: float,
    target_max_height: float
) -> trimesh.Trimesh:
    """
    Extend a membrane mesh vertically to match target height bounds.
    
    This is needed when using a cutting membrane (designed for one prism) to split
    another prism with different height bounds (e.g., using hard shell's outer collar
    to split the metamold).
    
    Algorithm:
    1. Convert membrane to aligned coordinate system (z = pouring direction)
    2. Find current height bounds of the membrane
    3. For each boundary edge at min_height, extend down to target_min_height
    4. For each boundary edge at max_height, extend up to target_max_height
    5. Create collar faces connecting original boundary to extended boundary
    
    Args:
        membrane: The cutting membrane mesh
        pouring_direction: Unit vector defining the "up" direction
        target_min_height: Target minimum height (along pouring direction)
        target_max_height: Target maximum height (along pouring direction)
        
    Returns:
        Extended membrane trimesh
    """
    if membrane is None or len(membrane.vertices) == 0:
        return membrane
    
    vertices = np.array(membrane.vertices, dtype=np.float64)
    faces = np.array(membrane.faces, dtype=np.int64)
    
    # Normalize pouring direction
    direction = np.array(pouring_direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Project all vertices onto pouring direction
    heights = np.dot(vertices, direction)
    current_min_height = heights.min()
    current_max_height = heights.max()
    
    logger.info(f"Extending membrane height from [{current_min_height:.2f}, {current_max_height:.2f}] "
               f"to [{target_min_height:.2f}, {target_max_height:.2f}]")
    
    # Check if extension is needed
    extension_below = target_min_height - current_min_height  # Should be negative if we need to go lower
    extension_above = target_max_height - current_max_height  # Should be positive if we need to go higher
    
    # Tolerance for identifying boundary vertices
    height_tol = (current_max_height - current_min_height) * 0.05  # 5% of height range
    
    if extension_below >= -0.1 and extension_above <= 0.1:
        logger.info("No significant height extension needed")
        return membrane
    
    # Find boundary edges
    boundary_edges, edge_to_face = _find_boundary_edges_with_winding(faces)
    
    if not boundary_edges:
        logger.warning("No boundary edges found - cannot extend membrane height")
        return membrane
    
    # Categorize boundary edges by height
    bottom_edges = []  # Edges near current min_height
    top_edges = []     # Edges near current max_height
    
    for v0, v1 in boundary_edges:
        h0 = heights[v0]
        h1 = heights[v1]
        avg_height = (h0 + h1) / 2
        
        # Check if edge is near bottom
        if avg_height < current_min_height + height_tol:
            bottom_edges.append((v0, v1))
        # Check if edge is near top
        elif avg_height > current_max_height - height_tol:
            top_edges.append((v0, v1))
    
    logger.info(f"Found {len(bottom_edges)} bottom edges, {len(top_edges)} top edges")
    
    # Extend vertices and create collar faces
    new_vertices = list(vertices)
    new_faces = list(faces)
    
    # Maps: original_vertex_index -> extended_vertex_index
    bottom_extension_map = {}
    top_extension_map = {}
    
    # Extend bottom boundary if needed
    if extension_below < -0.1 and len(bottom_edges) > 0:
        for v0, v1 in bottom_edges:
            for vi in [v0, v1]:
                if vi not in bottom_extension_map:
                    # Create extended vertex
                    old_pos = vertices[vi]
                    new_pos = old_pos + direction * extension_below  # extension_below is negative
                    new_idx = len(new_vertices)
                    new_vertices.append(new_pos)
                    bottom_extension_map[vi] = new_idx
        
        # Create collar faces for bottom
        for v0, v1 in bottom_edges:
            ext_v0 = bottom_extension_map.get(v0)
            ext_v1 = bottom_extension_map.get(v1)
            if ext_v0 is not None and ext_v1 is not None:
                # Get winding from original face
                edge_key = (min(v0, v1), max(v0, v1))
                face_info = edge_to_face.get(edge_key)
                if face_info is not None:
                    fi, orig_v0, orig_v1 = face_info
                    # Create quad connecting original to extended (downward)
                    # Winding should be consistent with face orientation
                    if orig_v0 == v0:
                        # Edge goes v0->v1 in the face
                        new_faces.append([v0, ext_v0, ext_v1])
                        new_faces.append([v0, ext_v1, v1])
                    else:
                        # Edge goes v1->v0 in the face
                        new_faces.append([v0, v1, ext_v1])
                        new_faces.append([v0, ext_v1, ext_v0])
    
    # Extend top boundary if needed
    if extension_above > 0.1 and len(top_edges) > 0:
        for v0, v1 in top_edges:
            for vi in [v0, v1]:
                if vi not in top_extension_map:
                    # Create extended vertex
                    old_pos = vertices[vi]
                    new_pos = old_pos + direction * extension_above  # extension_above is positive
                    new_idx = len(new_vertices)
                    new_vertices.append(new_pos)
                    top_extension_map[vi] = new_idx
        
        # Create collar faces for top
        for v0, v1 in top_edges:
            ext_v0 = top_extension_map.get(v0)
            ext_v1 = top_extension_map.get(v1)
            if ext_v0 is not None and ext_v1 is not None:
                # Get winding from original face
                edge_key = (min(v0, v1), max(v0, v1))
                face_info = edge_to_face.get(edge_key)
                if face_info is not None:
                    fi, orig_v0, orig_v1 = face_info
                    # Create quad connecting original to extended (upward)
                    if orig_v0 == v0:
                        new_faces.append([v0, v1, ext_v1])
                        new_faces.append([v0, ext_v1, ext_v0])
                    else:
                        new_faces.append([v0, ext_v0, ext_v1])
                        new_faces.append([v0, ext_v1, v1])
    
    # Create extended mesh
    extended_mesh = trimesh.Trimesh(
        vertices=np.array(new_vertices, dtype=np.float64),
        faces=np.array(new_faces, dtype=np.int64),
        process=True
    )
    extended_mesh.fix_normals()
    
    logger.info(f"Extended membrane: {len(vertices)} -> {len(new_vertices)} vertices, "
               f"{len(faces)} -> {len(new_faces)} faces")
    
    return extended_mesh


def split_shell_with_membrane(
    shell_with_cavity: trimesh.Trimesh,
    membrane: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    blade_thickness: float = 1.0
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool,
           Optional[trimesh.Trimesh]]:
    """
    Split the shell into two manifold halves using the membrane as a cutting blade.
    
    This creates a very thin volume (blade) from the membrane and subtracts it
    from the shell, leaving a tiny gap that separates the shell into two 
    disconnected manifold components.
    
    Algorithm:
    1. Create thin "blade" by extruding membrane ±thickness/2 in pouring direction
    2. Subtract blade from shell: cut_shell = shell - blade
    3. Split result into connected components
    4. Return the two largest components as half_1 (upper) and half_2 (lower)
    
    Args:
        shell_with_cavity: The shell mesh (prism - hull)
        membrane: The cutting membrane (outer collar extended parting surface)
        pouring_direction: Unit vector defining the "upper" direction
        blade_thickness: Thickness of the cutting blade (default: 0.0001mm = 0.1 micron)
                        The blade is centered on the membrane (±thickness/2 each side)
        
    Returns:
        Tuple of (shell_half_1, shell_half_2, computation_time_ms, success, blade)
        shell_half_1 is the half in the positive pouring direction (upper)
        shell_half_2 is the half in the negative pouring direction (lower)
        blade is the cutting blade mesh (for reuse in metamold step)
    """
    start_time = time.time()
    
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available - cannot perform CSG operations")
        return None, None, 0.0, False, None
    
    try:
        logger.info(f"Splitting shell using thin blade subtraction (thickness={blade_thickness:.6f}mm)...")
        
        # Normalize direction
        direction = np.array(pouring_direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Step 1: Create thin cutting blade from membrane
        logger.info("Creating cutting blade from membrane...")
        blade = _create_cutting_blade_from_membrane(membrane, direction, blade_thickness)
        logger.info(f"Blade: {len(blade.vertices)} verts, {len(blade.faces)} faces")
        
        # Detailed diagnostics (debug level only)
        if logger.isEnabledFor(logging.DEBUG):
            _log_blade_cut_diagnostics(shell_with_cavity, membrane, blade, direction)
        
        # Step 2: CSG subtraction - cut the shell with the blade
        logger.info("Performing CSG subtraction: shell - blade...")
        cut_shell = csg_backend.csg_difference(shell_with_cavity, blade)
        logger.info(f"Cut shell: {len(cut_shell.vertices)} verts, {len(cut_shell.faces)} faces")
        
        # Step 3: Split into connected components and classify halves
        upper, lower, components, did_split = _split_and_classify_halves(cut_shell, direction)
        
        if not did_split:
            logger.warning("Failed to split shell into two components - blade may not have cut through")
            elapsed_ms = (time.time() - start_time) * 1000
            if len(components) == 1:
                return components[0], None, elapsed_ms, False, blade
            return None, None, elapsed_ms, False, blade
        
        shell_half_1, shell_half_2 = upper, lower
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell split complete in {elapsed_ms:.1f}ms:")
        logger.info(f"  Half 1 (upper): {len(shell_half_1.vertices)} verts, {len(shell_half_1.faces)} faces")
        logger.info(f"  Half 2 (lower): {len(shell_half_2.vertices)} verts, {len(shell_half_2.faces)} faces")
        
        return shell_half_1, shell_half_2, elapsed_ms, True, blade
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to split shell with membrane: %s", e)
        logger.exception(e)
        return None, None, elapsed_ms, False, None


# ============================================================================
# COMBINED CUTTER VALIDATION & REPAIR
# ============================================================================

def _count_non_manifold_edges(mesh: trimesh.Trimesh) -> int:
    """Count edges shared by != 2 faces (boundary or non-manifold junctions).

    An edge shared by 1 face is an open boundary.
    An edge shared by 3+ faces is a non-manifold junction.
    Both are counted.

    Args:
        mesh: Triangle mesh to inspect.

    Returns:
        Number of edges with face-count != 2.
    """
    from collections import Counter
    edge_counts: Counter = Counter()
    for f in mesh.faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            if a != b:  # skip degenerate edges
                edge_counts[tuple(sorted((a, b)))] += 1
    return sum(1 for c in edge_counts.values() if c != 2)


def _validate_and_repair_cutter(
    mesh: trimesh.Trimesh,
    label: str = "combined_cutter",
) -> Tuple[trimesh.Trimesh, bool, List[str]]:
    """Validate a CSG-produced mesh and attempt comprehensive repair if needed.

    The repair pipeline is structured to avoid one step undoing another:

    **Pass 1 — MeshLib deep repair** (if available):
        Handles degenerate faces, self-intersections, holes, and normals
        in a single coordinated pass.

    **Pass 2 — Trimesh cleanup** (catches anything MeshLib missed):
        1. Remove degenerate (zero-area) faces.
        2. Remove duplicate faces.
        3. Merge close vertices.
        4. Fix normals (consistent winding).
        5. Fill holes (second pass, in case Pass 1 opened new ones).

    **Validation**:
        * Watertight check (no open boundaries).
        * Manifold edge check (every edge shared by exactly 2 faces).

    Args:
        mesh: The mesh to validate / repair.
        label: Human-readable label for log messages.

    Returns:
        (repaired_mesh, is_valid, repair_log)
        - repaired_mesh: The (possibly repaired) mesh.
        - is_valid: True if the mesh passes all manifold / watertight checks.
        - repair_log: List of human-readable strings describing each step.
    """
    log: List[str] = []
    m = mesh.copy()

    v0, f0 = len(m.vertices), len(m.faces)
    log.append(f"{label}: input {v0} verts, {f0} faces, "
               f"watertight={m.is_watertight}")

    # ================================================================== #
    # PASS 1 — MeshLib deep repair (degenerates, self-ix, holes, normals)
    # ================================================================== #
    try:
        from core.mesh_repair import MeshRepairer
        repairer = MeshRepairer(m)
        repair_result = repairer.repair(
            merge_vertices=True,
            remove_degenerate=True,
            remove_duplicates=True,
            fix_normals=True,
            fill_holes=True,
            fix_self_intersections=True,
            use_convex_hull_fallback=False,
        )
        if repair_result.was_repaired:
            m = repair_result.mesh
            for step in repair_result.repair_steps:
                log.append(f"  [MeshRepairer] {step}")
        else:
            log.append("  [MeshRepairer] No repairs needed")
    except Exception as e:
        log.append(f"  [MeshRepairer] Skipped: {e}")

    # ================================================================== #
    # PASS 2 — Trimesh cleanup (catches anything MeshLib missed / opened)
    # ================================================================== #

    # 2a. Remove degenerate (zero-area) faces
    if hasattr(m, 'area_faces') and len(m.faces) > 0:
        degen_mask = m.area_faces < 1e-12
        n_degen = int(degen_mask.sum())
        if n_degen > 0:
            m.update_faces(~degen_mask)
            m.remove_unreferenced_vertices()
            log.append(f"  Removed {n_degen} degenerate (zero-area) faces")

    # 2b. Remove duplicate faces
    if len(m.faces) > 0:
        sorted_faces = np.sort(m.faces, axis=1)
        _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
        n_dup = len(m.faces) - len(unique_idx)
        if n_dup > 0:
            m = trimesh.Trimesh(
                vertices=m.vertices.copy(),
                faces=m.faces[unique_idx],
                process=False,
            )
            log.append(f"  Removed {n_dup} duplicate faces")

    # 2c. Merge close vertices
    pre_merge = len(m.vertices)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    n_merged = pre_merge - len(m.vertices)
    if n_merged > 0:
        log.append(f"  Merged {n_merged} duplicate/close vertices")

    # 2d. Fix normals (consistent winding)
    m.fix_normals()

    # 2e. Fill holes — second pass to seal any boundaries opened by
    #     MeshLib's degenerate-face removal.
    if hasattr(m, 'fill_holes'):
        try:
            m.fill_holes()
            log.append("  Filled holes (post-repair pass)")
        except Exception as e:
            log.append(f"  Hole filling skipped: {e}")

    # ================================================================== #
    # VALIDATION
    # ================================================================== #
    is_watertight = bool(m.is_watertight)
    is_volume = bool(m.is_volume) if hasattr(m, 'is_volume') else is_watertight
    non_manifold_edges = _count_non_manifold_edges(m)
    is_manifold = is_watertight and (non_manifold_edges == 0)

    log.append(f"  Final: {len(m.vertices)} verts, {len(m.faces)} faces")
    log.append(f"  Watertight: {is_watertight}, Volume: {is_volume}, "
               f"Non-manifold edges: {non_manifold_edges}, "
               f"Manifold: {is_manifold}")

    for line in log:
        logger.info(line)

    return m, is_manifold, log


def create_metamold_with_combined_cut(
    prism_mesh: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh,
    membrane: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    blade_thickness: float = 1.0,
    secondary_mesh: Optional[trimesh.Trimesh] = None,
    secondary_thickness: float = 0.2,
    cutting_blade: Optional[trimesh.Trimesh] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh],
           Optional[trimesh.Trimesh], float, bool, Optional['SplitDiagnostics']]:
    """Create metamold halves using CSG cavity + half-space intersection.

    Uses a robust two-step approach:
        1. Create cavity:  cavity = prism − part
        2. Create upper half-space volume from membrane (extruded in +direction)
        3. Create lower half-space volume from membrane (extruded in −direction)
        4. half_1 = cavity ∩ upper_halfspace
        5. half_2 = cavity ∩ lower_halfspace

    This eliminates the unreliable blade-subtraction + connected-component
    splitting approach.  Each half is directly computed via CSG intersection,
    so no component detection or vertex-merging issues arise.

    A small gap (``blade_thickness``) is introduced at the parting line by
    shifting each half-space inward by half the thickness.

    Uses robust exact arithmetic CSG operations (Zhou et al. 2016 via CGAL).

    Args:
        prism_mesh: The metamold prism mesh.
        part_mesh: The original part mesh to cast.
        membrane: The parting surface mesh (including outer collar extension).
        pouring_direction: Unit vector for the resin pouring direction.
        blade_thickness: Gap at the parting line (mm).  Each half-space is
            shifted inward by ``blade_thickness / 2``.
        secondary_mesh: Optional secondary membrane (unused for now).
        secondary_thickness: Thickness for secondary membrane (unused for now).
        cutting_blade: Ignored — kept for API compatibility.

    Returns:
        (half_1, half_2, shell_with_cavity, computation_time_ms,
         split_success, diagnostics)

        - half_1: Upper half (positive pouring direction), or None on failure.
        - half_2: Lower half (negative pouring direction), or None on failure.
        - shell_with_cavity: The cavity result (prism − part), always returned
          when the first boolean succeeds.
        - computation_time_ms: Total computation time.
        - split_success: True if both intersections produced valid geometry.
        - diagnostics: ``None`` (reserved for future use).
    """
    start_time = time.time()

    if not CSG_AVAILABLE:
        logger.error("CSG backend not available — cannot perform CSG operations")
        return None, None, None, 0.0, False, None

    try:
        # Normalize direction
        direction = np.array(pouring_direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # ------------------------------------------------------------------
        # Step 1: Create cavity — cavity = prism − part
        # ------------------------------------------------------------------
        logger.info("Creating metamold cavity (prism − part)...")
        shell_with_cavity = csg_backend.csg_difference(prism_mesh, part_mesh)
        logger.info("  Cavity: %d verts, %d faces",
                     len(shell_with_cavity.vertices),
                     len(shell_with_cavity.faces))

        # ------------------------------------------------------------------
        # Step 2: Build upper and lower half-space volumes from membrane
        # ------------------------------------------------------------------
        half_gap = blade_thickness / 2.0
        hs_distance = 500.0  # large enough to contain any prism

        logger.info("Building half-space volumes from membrane "
                     "(gap=%.3fmm, hs_distance=%.0fmm)...", blade_thickness,
                     hs_distance)

        # Shift membrane vertices inward by half_gap for each side
        membrane_verts = np.asarray(membrane.vertices, dtype=np.float64)
        membrane_faces = np.asarray(membrane.faces, dtype=np.int64)

        # Upper half-space: membrane shifted UP by half_gap, extruded in +dir
        upper_membrane = trimesh.Trimesh(
            vertices=membrane_verts + direction * half_gap,
            faces=membrane_faces,
            process=False,
        )
        # Fill inner boundary loops so the half-space is a true solid slab,
        # not an annular ring with part-shaped holes punched through it.
        upper_membrane = _fill_inner_boundary_loops(upper_membrane)
        upper_hs = _extrude_surface_to_volume(
            upper_membrane, direction,
            offset_pos=hs_distance, offset_neg=0.0)

        # Lower half-space: membrane shifted DOWN by half_gap, extruded in −dir
        lower_membrane = trimesh.Trimesh(
            vertices=membrane_verts - direction * half_gap,
            faces=membrane_faces,
            process=False,
        )
        lower_membrane = _fill_inner_boundary_loops(lower_membrane)
        lower_hs = _extrude_surface_to_volume(
            lower_membrane, direction,
            offset_pos=0.0, offset_neg=hs_distance)

        logger.info("  Upper HS: %d verts, %d faces, watertight=%s",
                     len(upper_hs.vertices), len(upper_hs.faces),
                     upper_hs.is_watertight)
        logger.info("  Lower HS: %d verts, %d faces, watertight=%s",
                     len(lower_hs.vertices), len(lower_hs.faces),
                     lower_hs.is_watertight)

        # --- Diagnostic: bounds comparison ---
        c_bounds = shell_with_cavity.bounds
        u_bounds = upper_hs.bounds
        l_bounds = lower_hs.bounds
        logger.info("  Cavity bounds: min=%s  max=%s",
                     c_bounds[0], c_bounds[1])
        logger.info("  Upper HS bounds: min=%s  max=%s",
                     u_bounds[0], u_bounds[1])
        logger.info("  Lower HS bounds: min=%s  max=%s",
                     l_bounds[0], l_bounds[1])

        # ------------------------------------------------------------------
        # Step 3: Intersect cavity with each half-space
        # ------------------------------------------------------------------
        logger.info("Computing upper half: cavity ∩ upper_HS...")
        half_1 = csg_backend.csg_intersection(shell_with_cavity, upper_hs)
        logger.info("  Upper half: %d verts, %d faces",
                     len(half_1.vertices), len(half_1.faces))

        logger.info("Computing lower half: cavity ∩ lower_HS...")
        half_2 = csg_backend.csg_intersection(shell_with_cavity, lower_hs)
        logger.info("  Lower half: %d verts, %d faces",
                     len(half_2.vertices), len(half_2.faces))

        # ------------------------------------------------------------------
        # Step 4: Validate results
        # ------------------------------------------------------------------
        h1_valid = half_1 is not None and len(half_1.vertices) > 4
        h2_valid = half_2 is not None and len(half_2.vertices) > 4
        did_split = h1_valid and h2_valid

        if not did_split:
            logger.warning("Half-space intersection failed to produce "
                           "two valid halves (h1=%s, h2=%s)",
                           h1_valid, h2_valid)
            elapsed_ms = (time.time() - start_time) * 1000
            return None, None, shell_with_cavity, elapsed_ms, False, None

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info("Metamold half-space split complete in %.1fms:", elapsed_ms)
        logger.info("  Half 1 (upper): %d verts, %d faces",
                     len(half_1.vertices), len(half_1.faces))
        logger.info("  Half 2 (lower): %d verts, %d faces",
                     len(half_2.vertices), len(half_2.faces))

        return half_1, half_2, shell_with_cavity, elapsed_ms, True, None

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to create metamold with half-space split: %s", e)
        logger.exception(e)
        return None, None, None, elapsed_ms, False, None


def split_shell_with_tall_blade(
    shell: trimesh.Trimesh,
    membrane: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    extrusion_distance: float = 200.0,
    blade_gap: float = 0.0001
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool]:
    """
    Split a shell using a tall cutting blade that extends far above and below the membrane.
    
    This is designed for splitting the metamold, where the shell may have different
    height bounds than the membrane (parting surface + outer collar from hard shell step).
    The tall blade ensures the cut extends through the entire shell regardless of height.
    
    Algorithm:
    1. Create tall blade by extruding membrane ±extrusion_distance in pouring direction
    2. Subtract blade from shell: cut_shell = shell - blade
    3. Split result into connected components
    4. Return the two largest components as half_1 (upper) and half_2 (lower)
    
    Args:
        shell: The shell mesh to split (e.g., metamold with cavity)
        membrane: The cutting membrane (parting surface + outer collar from hard shell)
        pouring_direction: Unit vector defining the "upper" direction
        extrusion_distance: How far to extrude blade above/below membrane (default: 200mm)
        blade_gap: The thin gap created by the cut (default: 0.1 micron)
        
    Returns:
        Tuple of (shell_half_1, shell_half_2, computation_time_ms, success)
        shell_half_1 is the half in the positive pouring direction (upper)
        shell_half_2 is the half in the negative pouring direction (lower)
    """
    start_time = time.time()
    
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available - cannot perform CSG operations")
        return None, None, 0.0, False
    
    try:
        logger.info(f"Splitting shell using tall blade (extrusion={extrusion_distance}mm, gap={blade_gap:.6f}mm)...")
        
        # Normalize direction
        direction = np.array(pouring_direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Step 1: Create tall cutting blade from membrane
        logger.info("Creating tall cutting blade from membrane...")
        blade = _create_tall_cutting_blade(membrane, direction, extrusion_distance, blade_gap)
        logger.info(f"Tall blade: {len(blade.vertices)} verts, {len(blade.faces)} faces")
        
        # Step 2: CSG subtraction - cut the shell with the blade
        logger.info("Performing CSG subtraction: shell - tall_blade...")
        cut_shell = csg_backend.csg_difference(shell, blade)
        logger.info(f"Cut shell: {len(cut_shell.vertices)} verts, {len(cut_shell.faces)} faces")
        
        # Step 3: Split into connected components and classify halves
        upper, lower, components, did_split = _split_and_classify_halves(cut_shell, direction)
        
        if not did_split:
            logger.warning("Failed to split shell into two components - blade may not have cut through")
            elapsed_ms = (time.time() - start_time) * 1000
            if len(components) == 1:
                return components[0], None, elapsed_ms, False
            return None, None, elapsed_ms, False
        
        shell_half_1, shell_half_2 = upper, lower
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell split complete in {elapsed_ms:.1f}ms:")
        logger.info(f"  Half 1 (upper): {len(shell_half_1.vertices)} verts, {len(shell_half_1.faces)} faces")
        logger.info(f"  Half 2 (lower): {len(shell_half_2.vertices)} verts, {len(shell_half_2.faces)} faces")
        
        return shell_half_1, shell_half_2, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to split shell with tall blade: %s", e)
        logger.exception(e)
        return None, None, elapsed_ms, False


def split_shell_along_parting_surface(
    shell_with_cavity: trimesh.Trimesh,
    parting_surface: trimesh.Trimesh,
    pouring_direction: np.ndarray
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool]:
    """
    Split the shell into two halves along the parting surface.
    
    Uses the parting surface (with extended collar) as a cutting plane.
    
    Args:
        shell_with_cavity: The shell mesh with cavity already subtracted
        parting_surface: The parting surface mesh (with extended collar)
        pouring_direction: The pouring direction (used to determine which half is which)
        
    Returns:
        Tuple of (shell_half_1, shell_half_2, computation_time_ms, success)
        shell_half_1 is the positive pouring direction half
        shell_half_2 is the negative pouring direction half
    """
    start_time = time.time()
    
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available - cannot perform CSG operations")
        return None, None, 0.0, False
    
    try:
        logger.info("Splitting shell along parting surface...")
        
        # Create a thick volume from the parting surface for CSG
        # The volume extends far in both directions to ensure complete intersection
        cutting_volume = _create_cutting_volume_from_surface(
            parting_surface,
            pouring_direction,
            thickness=1000.0  # Large enough to extend beyond the shell
        )
        
        # Create large half-space boxes on each side of the parting surface
        # and intersect with the shell to produce two halves.
        shell_bounds = shell_with_cavity.bounds
        shell_size = np.linalg.norm(shell_bounds[1] - shell_bounds[0])
        box_size = shell_size * 3  # Make box much larger than shell
        
        direction = np.array(pouring_direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        half_box_size = box_size
        parting_center = parting_surface.centroid
        
        box_center_1 = parting_center + direction * (half_box_size / 2.0 + 0.1)
        box_1 = trimesh.creation.box(
            extents=[box_size, box_size, box_size],
            transform=trimesh.transformations.translation_matrix(box_center_1)
        )
        
        box_center_2 = parting_center - direction * (half_box_size / 2.0 + 0.1)
        box_2 = trimesh.creation.box(
            extents=[box_size, box_size, box_size],
            transform=trimesh.transformations.translation_matrix(box_center_2)
        )
        
        # Intersect shell with each half-space box
        shell_half_1 = csg_backend.csg_intersection(shell_with_cavity, box_1)
        shell_half_2 = csg_backend.csg_intersection(shell_with_cavity, box_2)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Shell split complete in {elapsed_ms:.1f}ms:")
        logger.info(f"  Half 1: {len(shell_half_1.vertices)} verts, {len(shell_half_1.faces)} faces")
        logger.info(f"  Half 2: {len(shell_half_2.vertices)} verts, {len(shell_half_2.faces)} faces")
        
        return shell_half_1, shell_half_2, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to split shell: %s", e)
        logger.exception(e)
        return None, None, elapsed_ms, False


# ============================================================================
# SURFACE THICKENING (for secondary membranes)
# ============================================================================

def thicken_surface_symmetric(
    surface_mesh: trimesh.Trimesh,
    thickness: float = 0.2
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Thicken a surface mesh symmetrically by offsetting it in both normal directions.
    
    Creates a watertight solid by:
    1. Computing per-vertex normals from the surface
    2. Offsetting vertices by +thickness/2 and -thickness/2 along normals
    3. Creating side faces to close the boundary
    
    The result is a manifold solid "slab" centered on the original surface.
    
    Args:
        surface_mesh: The input surface mesh to thicken
        thickness: Total thickness (will extend thickness/2 on each side)
        
    Returns:
        Tuple of (thickened_mesh, computation_time_ms, success)
    """
    start_time = time.time()
    
    if surface_mesh is None or len(surface_mesh.vertices) == 0:
        logger.error("Surface mesh is empty or None")
        return None, 0.0, False
    
    if thickness <= 0:
        logger.error("Thickness must be positive")
        return None, 0.0, False
    
    try:
        logger.info(f"Thickening surface symmetrically by {thickness}mm...")
        logger.info(f"  Input: {len(surface_mesh.vertices)} verts, {len(surface_mesh.faces)} faces")
        
        vertices = np.asarray(surface_mesh.vertices, dtype=np.float64)
        faces = np.asarray(surface_mesh.faces, dtype=np.int64)
        n_verts = len(vertices)
        n_faces = len(faces)
        
        # Compute per-vertex normals
        # Use trimesh's vertex normals (weighted by face area)
        vertex_normals = np.asarray(surface_mesh.vertex_normals, dtype=np.float64)
        
        # Normalize the vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0  # Avoid division by zero
        vertex_normals = vertex_normals / norms
        
        half_thickness = thickness / 2.0
        
        # Create offset vertices
        # Top surface: offset in positive normal direction
        top_vertices = vertices + vertex_normals * half_thickness
        # Bottom surface: offset in negative normal direction
        bottom_vertices = vertices - vertex_normals * half_thickness
        
        # Combined vertices: [bottom, top]
        all_vertices = np.vstack([bottom_vertices, top_vertices])
        
        # Create faces:
        # 1. Bottom faces - reversed winding (normals point away from slab in -normal direction)
        bottom_faces = faces[:, ::-1].copy()
        
        # 2. Top faces - original winding (normals point away from slab in +normal direction), offset by n_verts
        top_faces = faces + n_verts
        
        # 3. Side faces connecting boundary edges
        boundary_edges, edge_to_face = _find_boundary_edges_with_winding(faces)
        
        side_faces = []
        for edge_key in boundary_edges:
            _, v0_orig, v1_orig = edge_to_face[edge_key]
            
            # Bottom vertices
            b0, b1 = v0_orig, v1_orig
            # Top vertices (offset by n_verts)
            t0, t1 = v0_orig + n_verts, v1_orig + n_verts
            
            # Create quad (2 triangles) with outward-facing normals
            # Winding consistent with reversed bottom (boundary b1->b0) and original top (t0->t1)
            side_faces.append([b0, b1, t1])
            side_faces.append([b0, t1, t0])
        
        # Combine all faces
        all_faces = np.vstack([
            bottom_faces,
            top_faces,
            np.array(side_faces, dtype=np.int64) if side_faces else np.empty((0, 3), dtype=np.int64)
        ])
        
        # Create the thickened mesh
        thickened_mesh = trimesh.Trimesh(
            vertices=all_vertices,
            faces=all_faces,
            process=True
        )
        
        # Try to fix any winding issues
        thickened_mesh.fix_normals()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if thickened_mesh.is_watertight:
            logger.info(f"Surface thickened: {len(thickened_mesh.vertices)} verts, "
                       f"{len(thickened_mesh.faces)} faces (watertight) in {elapsed_ms:.1f}ms")
        else:
            logger.warning(f"Surface thickened: {len(thickened_mesh.vertices)} verts, "
                          f"{len(thickened_mesh.faces)} faces (NOT watertight) in {elapsed_ms:.1f}ms")
        
        return thickened_mesh, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to thicken surface: %s", e)
        logger.exception(e)
        return None, elapsed_ms, False


def create_part_with_thickened_secondary(
    part_mesh: trimesh.Trimesh,
    secondary_surface: trimesh.Trimesh,
    secondary_thickness: float = 0.2
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Create a combined mesh of part + thickened secondary surface.
    
    This is used for metamold creation where we want the secondary membrane
    features to be part of the mold cavity.
    
    Args:
        part_mesh: The original part mesh
        secondary_surface: The smoothed secondary parting surface
        secondary_thickness: Thickness for the secondary surface (symmetric)
        
    Returns:
        Tuple of (combined_mesh, computation_time_ms, success)
    """
    start_time = time.time()
    
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available - cannot perform CSG operations")
        return None, 0.0, False
    
    if part_mesh is None or len(part_mesh.vertices) == 0:
        logger.warning("Part mesh is empty - returning None")
        return None, 0.0, False
    
    # If no secondary surface, just return the part mesh
    if secondary_surface is None or len(secondary_surface.vertices) == 0:
        logger.info("No secondary surface provided - using part mesh only")
        elapsed_ms = (time.time() - start_time) * 1000
        return part_mesh.copy(), elapsed_ms, True
    
    try:
        logger.info(f"Creating part mesh with thickened secondary surface...")
        logger.info(f"  Part: {len(part_mesh.vertices)} verts, {len(part_mesh.faces)} faces")
        logger.info(f"  Secondary: {len(secondary_surface.vertices)} verts, {len(secondary_surface.faces)} faces")
        logger.info(f"  Secondary thickness: {secondary_thickness}mm")
        
        # Step 1: Thicken the secondary surface
        thickened_secondary, thicken_time, thicken_success = thicken_surface_symmetric(
            secondary_surface, secondary_thickness
        )
        
        if not thicken_success or thickened_secondary is None:
            logger.warning("Failed to thicken secondary surface - using part mesh only")
            elapsed_ms = (time.time() - start_time) * 1000
            return part_mesh.copy(), elapsed_ms, True
        
        # Step 2: Boolean union part + thickened secondary
        logger.info("Performing boolean union: part + thickened_secondary...")
        combined_mesh = csg_backend.csg_union(part_mesh, thickened_secondary)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if combined_mesh.is_watertight:
            logger.info(f"Combined mesh created: {len(combined_mesh.vertices)} verts, "
                       f"{len(combined_mesh.faces)} faces (watertight) in {elapsed_ms:.1f}ms")
        else:
            logger.warning(f"Combined mesh created: {len(combined_mesh.vertices)} verts, "
                          f"{len(combined_mesh.faces)} faces (NOT watertight) in {elapsed_ms:.1f}ms")
        
        return combined_mesh, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to create part with thickened secondary: %s", e)
        logger.exception(e)
        return None, elapsed_ms, False


# ============================================================================
# METAMOLD: ADD PART MESH BACK TO HALVES
# ============================================================================

def add_part_to_metamold_half(
    metamold_half: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh
) -> Tuple[Optional[trimesh.Trimesh], float, bool]:
    """
    Boolean ADD (union) the part mesh to a metamold half.
    
    This operation adds the part mesh back into the metamold half, creating
    a solid where the part would be placed. This is useful for creating
    metamold halves that can be used to cast silicone around an actual part.
    
    The result is a manifold mesh thanks to the robust CSG backend (libigl/CGAL or manifold3d).
    
    Args:
        metamold_half: One half of the split metamold (with cavity)
        part_mesh: The original part mesh to add back
        
    Returns:
        Tuple of (result_mesh, computation_time_ms, success)
        result_mesh is the metamold half with the part added as solid geometry
    """
    start_time = time.time()
    
    if not CSG_AVAILABLE:
        logger.error("CSG backend not available - cannot perform CSG operations")
        return None, 0.0, False
    
    if metamold_half is None or len(metamold_half.vertices) == 0:
        logger.error("Metamold half is empty or None")
        return None, 0.0, False
    
    if part_mesh is None or len(part_mesh.vertices) == 0:
        logger.error("Part mesh is empty or None")
        return None, 0.0, False
    
    try:
        logger.info(f"Adding part mesh to metamold half (boolean union)...")
        logger.info(f"  Metamold half: {len(metamold_half.vertices)} verts, {len(metamold_half.faces)} faces")
        logger.info(f"  Part mesh: {len(part_mesh.vertices)} verts, {len(part_mesh.faces)} faces")
        
        # Perform union: result = half + part
        result_mesh = csg_backend.csg_union(metamold_half, part_mesh)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Verify manifold output
        if result_mesh.is_watertight:
            logger.info(f"Part added to metamold half: {len(result_mesh.vertices)} verts, "
                       f"{len(result_mesh.faces)} faces (watertight) in {elapsed_ms:.1f}ms")
        else:
            logger.warning(f"Part added to metamold half: {len(result_mesh.vertices)} verts, "
                          f"{len(result_mesh.faces)} faces (NOT watertight) in {elapsed_ms:.1f}ms")
        
        return result_mesh, elapsed_ms, True
        
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to add part to metamold half: %s", e)
        logger.exception(e)
        return None, elapsed_ms, False


def add_part_to_metamold_halves(
    metamold_half_1: trimesh.Trimesh,
    metamold_half_2: trimesh.Trimesh,
    part_mesh: trimesh.Trimesh
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh], float, bool]:
    """
    Boolean ADD (union) the part mesh to both metamold halves.
    
    This is a convenience function that calls add_part_to_metamold_half for both halves.
    
    Args:
        metamold_half_1: First metamold half (upper, positive pouring direction)
        metamold_half_2: Second metamold half (lower, negative pouring direction)
        part_mesh: The original part mesh to add back
        
    Returns:
        Tuple of (half_1_with_part, half_2_with_part, total_computation_time_ms, success)
    """
    start_time = time.time()
    
    logger.info("Adding part mesh to both metamold halves...")
    
    # Process half 1
    half_1_with_part, time_1, success_1 = add_part_to_metamold_half(metamold_half_1, part_mesh)
    
    if not success_1:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to add part to metamold half 1")
        return None, None, elapsed_ms, False
    
    # Process half 2
    half_2_with_part, time_2, success_2 = add_part_to_metamold_half(metamold_half_2, part_mesh)
    
    if not success_2:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("Failed to add part to metamold half 2")
        return half_1_with_part, None, elapsed_ms, False
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Part added to both metamold halves in {elapsed_ms:.1f}ms")
    logger.info(f"  Half 1 with part: {len(half_1_with_part.vertices)} verts, {len(half_1_with_part.faces)} faces")
    logger.info(f"  Half 2 with part: {len(half_2_with_part.vertices)} verts, {len(half_2_with_part.faces)} faces")
    
    return half_1_with_part, half_2_with_part, elapsed_ms, True


def trim_metamold_halves(
    upper_half: trimesh.Trimesh,
    lower_half: trimesh.Trimesh,
    parting_surface: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    trim_threshold: float = 4.0
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, float, float, float]:
    """
    Trim each metamold half's BASE to save 3D printing material.
    
    The upper half (positive pouring direction side) has its TOP trimmed.
    The lower half (negative pouring direction side) has its BOTTOM trimmed.
    Each half is trimmed independently so one never cuts into the other.
    
    Must be called AFTER all CSG operations (cavity, split, part union)
    so both prism walls and part geometry are cleanly clipped.
    
    The caller must pass the halves in the correct order as returned by
    ``split_shell_with_membrane``:
        half_1 → upper_half (positive pouring direction)
        half_2 → lower_half (negative pouring direction)
    
    Args:
        upper_half: The metamold half on the positive pouring-direction side.
        lower_half: The metamold half on the negative pouring-direction side.
        parting_surface: The parting surface mesh (height reference).
        pouring_direction: Unit vector for the resin pouring direction.
        trim_threshold: Keep at least this much material (mm) between the
            parting surface and the trimmed base.  Set ≤ 0 to disable.
    
    Returns:
        (trimmed_upper, trimmed_lower, upper_saved_mm, lower_saved_mm,
         computation_time_ms).
    """
    start_time = time.time()
    
    if not CSG_AVAILABLE:
        logger.warning("CSG backend not available – cannot trim metamold halves")
        return upper_half, lower_half, 0.0, 0.0, 0.0
    
    if trim_threshold <= 0:
        logger.info("Metamold trim disabled (threshold=%s)", trim_threshold)
        elapsed_ms = (time.time() - start_time) * 1000
        return upper_half, lower_half, 0.0, 0.0, elapsed_ms
    
    pouring_dir = np.asarray(pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    # ------------------------------------------------------------------
    # Parting surface height extent along the pouring direction
    # ------------------------------------------------------------------
    ps_verts = np.asarray(parting_surface.vertices, dtype=np.float64)
    ps_heights = ps_verts @ pouring_dir
    ps_min_height = float(np.min(ps_heights))
    ps_max_height = float(np.max(ps_heights))
    
    logger.info(f"Trimming metamold halves (threshold={trim_threshold:.1f}mm)")
    logger.info(f"  Parting surface height: {ps_min_height:.2f} to {ps_max_height:.2f}")
    
    upper_saved = 0.0
    lower_saved = 0.0
    
    # ------------------------------------------------------------------
    # Trim the UPPER half's TOP (its base, away from PS)
    # Keep everything BELOW (ps_max + threshold)
    # ------------------------------------------------------------------
    trimmed_upper = upper_half
    if upper_half is not None and len(upper_half.vertices) > 0:
        try:
            uh = np.asarray(upper_half.vertices, dtype=np.float64) @ pouring_dir
            half_max = float(np.max(uh))
            trim_at = ps_max_height + trim_threshold
            gap = half_max - trim_at
            
            logger.info(f"  Upper half extent: {float(np.min(uh)):.2f} to {half_max:.2f}")
            
            if gap > 0.01:
                # Keep everything BELOW trim_at (i.e. dot(v, pouring_dir) <= trim_at)
                # csg_trim_by_plane uses manifold3d convention: keeps dot(v, normal) >= offset
                # So we negate: normal = -pouring_dir, offset = -trim_at
                trimmed_upper = csg_backend.csg_trim_by_plane(
                    upper_half, (-pouring_dir).tolist(), float(-trim_at)
                )
                upper_saved = gap
                logger.info(f"    Trimmed upper top by {gap:.2f}mm (cut at h={trim_at:.2f})")
            else:
                logger.info(f"    Upper half: no trim needed (gap={gap:.2f}mm)")
        except Exception as e:
            logger.error(f"Failed to trim upper half: {e}")
    
    # ------------------------------------------------------------------
    # Trim the LOWER half's BOTTOM (its base, away from PS)
    # Keep everything ABOVE (ps_min - threshold)
    # ------------------------------------------------------------------
    trimmed_lower = lower_half
    if lower_half is not None and len(lower_half.vertices) > 0:
        try:
            lh = np.asarray(lower_half.vertices, dtype=np.float64) @ pouring_dir
            half_min = float(np.min(lh))
            trim_at = ps_min_height - trim_threshold
            gap = trim_at - half_min
            
            logger.info(f"  Lower half extent: {half_min:.2f} to {float(np.max(lh)):.2f}")
            
            if gap > 0.01:
                # Keep everything ABOVE trim_at (i.e. dot(v, pouring_dir) >= trim_at)
                # csg_trim_by_plane: normal = pouring_dir, offset = trim_at
                trimmed_lower = csg_backend.csg_trim_by_plane(
                    lower_half, pouring_dir.tolist(), float(trim_at)
                )
                lower_saved = gap
                logger.info(f"    Trimmed lower bottom by {gap:.2f}mm (cut at h={trim_at:.2f})")
            else:
                logger.info(f"    Lower half: no trim needed (gap={gap:.2f}mm)")
        except Exception as e:
            logger.error(f"Failed to trim lower half: {e}")
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Metamold trim complete in {elapsed_ms:.1f}ms "
               f"(upper saved: {upper_saved:.2f}mm, lower saved: {lower_saved:.2f}mm)")
    
    return trimmed_upper, trimmed_lower, upper_saved, lower_saved, elapsed_ms


def create_split_hard_shell(
    prism_result: HardShellPrismResult,
    hull_mesh: trimesh.Trimesh,
    parting_surface_with_collar: trimesh.Trimesh,
    pouring_direction: np.ndarray
) -> HardShellSplitResult:
    """
    Complete CSG pipeline: create shell with cavity and split into two halves.
    
    Pipeline:
    1. Subtract hull from prism to create cavity
    2. Split along parting surface into two halves
    
    Args:
        prism_result: The hard shell prism result
        hull_mesh: The inflated hull mesh (defines mold cavity)
        parting_surface_with_collar: The parting surface with extended collar
        pouring_direction: The resin pouring direction
        
    Returns:
        HardShellSplitResult with both shell halves
    """
    result = HardShellSplitResult(
        pouring_direction=np.array(pouring_direction)
    )
    
    total_start = time.time()
    
    # Step 1: Create shell with cavity
    shell_with_cavity, cavity_time, cavity_success = create_shell_with_cavity(
        prism_result.prism_mesh,
        hull_mesh
    )
    
    result.shell_with_cavity = shell_with_cavity
    result.cavity_time_ms = cavity_time
    result.cavity_success = cavity_success
    
    if not cavity_success or shell_with_cavity is None:
        result.total_time_ms = (time.time() - total_start) * 1000
        return result
    
    # Step 2: Split along parting surface
    shell_half_1, shell_half_2, split_time, split_success = split_shell_along_parting_surface(
        shell_with_cavity,
        parting_surface_with_collar,
        pouring_direction
    )
    
    result.shell_half_1 = shell_half_1
    result.shell_half_2 = shell_half_2
    result.split_time_ms = split_time
    result.split_success = split_success
    
    if shell_half_1 is not None:
        result.half_1_vertex_count = len(shell_half_1.vertices)
        result.half_1_face_count = len(shell_half_1.faces)
    
    if shell_half_2 is not None:
        result.half_2_vertex_count = len(shell_half_2.vertices)
        result.half_2_face_count = len(shell_half_2.faces)
    
    result.total_time_ms = (time.time() - total_start) * 1000
    
    return result


def _offset_polygon_2d(polygon: np.ndarray, offset: float) -> np.ndarray:
    """
    Offset a 2D convex polygon outward by a given distance.
    
    Uses the following approach for convex polygons:
    - Compute the outward normal for each edge
    - Move each vertex along the bisector of adjacent edge normals
    - Scale by offset / cos(half_angle) to maintain proper offset distance
    
    Args:
        polygon: (N, 2) array of 2D vertices in CCW order
        offset: Distance to offset outward
        
    Returns:
        (N, 2) array of offset polygon vertices
    """
    n = len(polygon)
    if n < 3:
        return polygon.copy()
    
    # Ensure CCW ordering
    # Compute signed area
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += polygon[i, 0] * polygon[j, 1]
        signed_area -= polygon[j, 0] * polygon[i, 1]
    
    if signed_area < 0:
        # CW ordering, reverse
        polygon = polygon[::-1].copy()
    
    # Compute edge normals (outward for CCW polygon)
    edge_normals = []
    for i in range(n):
        j = (i + 1) % n
        edge = polygon[j] - polygon[i]
        # Normal: perpendicular to edge, pointing outward (right side for CCW)
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        edge_normals.append(normal)
    
    # Offset each vertex
    offset_polygon = []
    for i in range(n):
        prev_i = (i - 1) % n
        
        # Bisector of prev and current edge normals
        n_prev = edge_normals[prev_i]
        n_curr = edge_normals[i]
        
        bisector = n_prev + n_curr
        bisector_len = np.linalg.norm(bisector)
        
        if bisector_len > 1e-10:
            bisector = bisector / bisector_len
            
            # cos(half_angle) = dot(n_prev, bisector) = dot(n_curr, bisector)
            cos_half = np.dot(n_prev, bisector)
            
            # Offset distance along bisector
            if cos_half > 0.1:  # Avoid very sharp corners
                offset_dist = offset / cos_half
            else:
                offset_dist = offset * 5  # Cap at 5x for sharp corners
            
            new_vertex = polygon[i] + bisector * offset_dist
        else:
            # Normals are opposite (happens at very sharp corners)
            new_vertex = polygon[i] + n_prev * offset
        
        offset_polygon.append(new_vertex)
    
    return np.array(offset_polygon, dtype=np.float64)


def _ray_2d_convex_polygon_exit(
    origin: np.ndarray,
    direction: np.ndarray,
    polygon: np.ndarray
) -> Optional[float]:
    """
    Find the distance from a point inside a convex polygon to its boundary along a ray.
    
    For a point inside a convex polygon, casts a ray along the given direction and
    finds the exit distance (closest forward intersection with a polygon edge).
    
    This is used by the adaptive collar extension to determine exactly how far each
    boundary vertex needs to extend to pass through the prism's offset silhouette.
    
    Args:
        origin: 2D point (2,) - assumed to be inside the polygon
        direction: 2D unit direction vector (2,)
        polygon: (N, 2) convex polygon vertices
        
    Returns:
        Distance along ray to exit the polygon, or None if no intersection found.
    """
    n = len(polygon)
    min_positive_t = None
    
    for i in range(n):
        j = (i + 1) % n
        
        # Edge from polygon[i] to polygon[j]
        p0 = polygon[i]
        edge = polygon[j] - p0
        
        # Solve: origin + t * direction = p0 + s * edge
        # t * direction - s * edge = p0 - origin
        # Using Cramer's rule:
        # denom = direction[0] * (-edge[1]) - direction[1] * (-edge[0])
        denom = direction[1] * edge[0] - direction[0] * edge[1]
        
        if abs(denom) < 1e-12:
            continue  # Ray parallel to edge
        
        diff = p0 - origin
        t = (diff[1] * edge[0] - diff[0] * edge[1]) / denom
        s = (diff[1] * direction[0] - diff[0] * direction[1]) / denom
        
        # Valid intersection: t > 0 (forward along ray) and 0 <= s <= 1 (on edge segment)
        if t > 1e-8 and -1e-8 <= s <= 1.0 + 1e-8:
            if min_positive_t is None or t < min_positive_t:
                min_positive_t = t
    
    return min_positive_t


def _extrude_polygon_to_prism(
    polygon_2d: np.ndarray,
    world_to_2d: np.ndarray,
    min_height: float,
    max_height: float
) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """
    Extrude a 2D polygon along the normal direction to create a prism.
    
    Args:
        polygon_2d: (N, 2) array of 2D polygon vertices
        world_to_2d: (3, 3) rotation matrix from world to 2D plane
        min_height: Bottom extrusion distance (along the third axis after rotation)
        max_height: Top extrusion distance
        
    Returns:
        Tuple of:
        - trimesh.Trimesh: The prism mesh
        - np.ndarray: (N, 3) bottom polygon vertices in 3D
        - np.ndarray: (N, 3) top polygon vertices in 3D
    """
    n = len(polygon_2d)
    
    # Transform back to 3D world coordinates
    # world_to_2d.T is the inverse (since it's orthonormal)
    two_d_to_world = world_to_2d.T
    
    # Create bottom and top polygons in 3D
    bottom_vertices = []
    top_vertices = []
    
    for i in range(n):
        # 2D to 3D: [u, v, height] -> world
        pt_bottom = np.array([polygon_2d[i, 0], polygon_2d[i, 1], min_height])
        pt_top = np.array([polygon_2d[i, 0], polygon_2d[i, 1], max_height])
        
        bottom_3d = pt_bottom @ two_d_to_world.T
        top_3d = pt_top @ two_d_to_world.T
        
        bottom_vertices.append(bottom_3d)
        top_vertices.append(top_3d)
    
    bottom_vertices = np.array(bottom_vertices, dtype=np.float64)
    top_vertices = np.array(top_vertices, dtype=np.float64)
    
    # Build mesh vertices: bottom ring + top ring
    # Indices: 0..n-1 = bottom, n..2n-1 = top
    all_vertices = np.vstack([bottom_vertices, top_vertices])
    
    faces = []
    
    # Side faces: quads as 2 triangles each
    for i in range(n):
        j = (i + 1) % n
        
        b0 = i          # bottom vertex i
        b1 = j          # bottom vertex i+1
        t0 = i + n      # top vertex i
        t1 = j + n      # top vertex i+1
        
        # Quad: (b0, b1, t1, t0) -> triangles with outward normals
        # CCW when viewed from outside
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    
    # Bottom cap: fan triangulation from center
    # Add center vertex
    bottom_center = np.mean(bottom_vertices, axis=0)
    bottom_center_idx = len(all_vertices)
    all_vertices = np.vstack([all_vertices, bottom_center.reshape(1, 3)])
    
    for i in range(n):
        j = (i + 1) % n
        # Bottom face: normal points down (reversed winding)
        faces.append([bottom_center_idx, j, i])
    
    # Top cap: fan triangulation from center
    top_center = np.mean(top_vertices, axis=0)
    top_center_idx = len(all_vertices)
    all_vertices = np.vstack([all_vertices, top_center.reshape(1, 3)])
    
    for i in range(n):
        j = (i + 1) % n
        # Top face: normal points up
        faces.append([top_center_idx, i + n, j + n])
    
    faces = np.array(faces, dtype=np.int64)
    
    prism_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=faces,
        process=True
    )
    prism_mesh.fix_normals()
    
    return prism_mesh, bottom_vertices, top_vertices


# ============================================================================
# OUTER SHELL HULL CREATION
# ============================================================================

def create_outer_shell_hull(
    inner_hull: trimesh.Trimesh,
    wall_thickness: float = DEFAULT_WALL_THICKNESS_MM
) -> OuterShellHullResult:
    """
    Create the outer shell hull by offsetting the inner hull outward.
    
    The outer shell hull defines the exterior boundary of the hard shell.
    The space between inner hull and outer shell hull is where the hard
    plastic shell wall will be.
    
    Args:
        inner_hull: The current inflated hull mesh (∂H)
        wall_thickness: How thick the hard shell wall should be (mm)
        
    Returns:
        OuterShellHullResult with the offset hull
    """
    start_time = time.time()
    
    logger.info(f"Creating outer shell hull with wall_thickness={wall_thickness}mm")
    
    # Get vertices and compute smooth normals
    vertices = np.array(inner_hull.vertices, dtype=np.float64)
    faces = np.array(inner_hull.faces, dtype=np.int32)
    
    # Compute area-weighted smooth vertex normals
    smooth_normals = _compute_smooth_vertex_normals(vertices, faces)
    
    # Offset vertices outward along normals
    outer_vertices = vertices + smooth_normals * wall_thickness
    
    # Create mesh from offset vertices
    outer_hull_mesh = trimesh.Trimesh(
        vertices=outer_vertices,
        faces=faces.copy(),
        process=True
    )
    
    # Fix normals to point outward
    outer_hull_mesh.fix_normals()
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Outer shell hull created: {len(outer_hull_mesh.vertices)} vertices, "
                f"{len(outer_hull_mesh.faces)} faces in {elapsed_ms:.1f}ms")
    
    return OuterShellHullResult(
        mesh=outer_hull_mesh,
        inner_hull=inner_hull,
        wall_thickness=wall_thickness,
        vertex_count=len(outer_hull_mesh.vertices),
        face_count=len(outer_hull_mesh.faces),
        computation_time_ms=elapsed_ms
    )


def _compute_smooth_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute area-weighted smooth vertex normals.
    
    Each vertex normal is the normalized sum of face normals weighted by face area.
    """
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute face normals via cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    
    # Face normal length = 2 * area
    face_normal_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    areas = face_normal_lengths / 2.0
    
    # Normalize face normals
    face_normals_normalized = face_normals / (face_normal_lengths + 1e-10)
    
    # Weight by area
    weighted_normals = face_normals_normalized * areas
    
    # Accumulate at each vertex
    normal_accum = np.zeros((num_vertices, 3), dtype=np.float64)
    np.add.at(normal_accum, faces[:, 0], weighted_normals)
    np.add.at(normal_accum, faces[:, 1], weighted_normals)
    np.add.at(normal_accum, faces[:, 2], weighted_normals)
    
    # Normalize
    norms = np.linalg.norm(normal_accum, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    smooth_normals = normal_accum / norms
    
    return smooth_normals


# ============================================================================
# OUTER COLLAR EXTENSION
# ============================================================================


def _classify_boundary_edges_by_type(
    boundary_edges: List[Tuple[int, int]],
    vertex_boundary_type: Optional[np.ndarray],
    n_orig_verts: int,
    vertices_arr: np.ndarray,
    inner_hull: trimesh.Trimesh,
) -> Tuple[
    List[Tuple[int, int]],  # outer_boundary_edges
    List[Tuple[int, int]],  # inner_boundary_edges
    Dict[int, List],        # vertex_to_boundary_edges
    List[int],              # corner_vertices
    List[int],              # endpoint_vertices
]:
    """Classify mesh boundary edges as inner (part) or outer (hull).

    Outer edges have at least one vertex with ``boundary_type`` in {1, 2}
    (hull boundary).  Inner edges have at least one vertex with
    ``boundary_type == -1`` (part surface).  When *vertex_boundary_type* is
    unavailable, falls back to proximity testing against *inner_hull*
    (< 1 mm threshold).

    Also builds per-vertex adjacency for the outer boundary and identifies
    corner vertices (2+ incident outer boundary edges) and endpoints (1 edge).

    Returns:
        (outer_edges, inner_edges, vertex_to_boundary_edges,
         corner_vertices, endpoint_vertices)
    """
    has_bt = (vertex_boundary_type is not None
              and len(vertex_boundary_type) >= n_orig_verts)

    outer_boundary_edges: List[Tuple[int, int]] = []
    inner_boundary_edges: List[Tuple[int, int]] = []

    for v0, v1 in boundary_edges:
        is_outer = False

        if has_bt:
            bt0 = vertex_boundary_type[v0] if v0 < len(vertex_boundary_type) else 0
            bt1 = vertex_boundary_type[v1] if v1 < len(vertex_boundary_type) else 0

            if bt0 in (1, 2) or bt1 in (1, 2):
                is_outer = True
            elif bt0 == -1 or bt1 == -1:
                is_outer = False
            else:
                is_outer = True  # Both interior — default to outer
        else:
            try:
                _, dists, _ = trimesh.proximity.closest_point(
                    inner_hull, vertices_arr[[v0, v1]])
                is_outer = float(np.min(dists)) < 1.0
            except Exception:
                is_outer = True

        (outer_boundary_edges if is_outer else inner_boundary_edges).append((v0, v1))

    # Build vertex adjacency for outer boundary
    vertex_to_boundary_edges: Dict[int, List] = {}
    for v0, v1 in outer_boundary_edges:
        vertex_to_boundary_edges.setdefault(v0, []).append(((v0, v1), v0, v1))
        vertex_to_boundary_edges.setdefault(v1, []).append(((v0, v1), v1, v0))

    corner_vertices = [
        vi for vi, edges in vertex_to_boundary_edges.items() if len(edges) >= 2
    ]
    endpoint_vertices = [
        vi for vi, edges in vertex_to_boundary_edges.items() if len(edges) == 1
    ]

    return (outer_boundary_edges, inner_boundary_edges,
            vertex_to_boundary_edges, corner_vertices, endpoint_vertices)


def _compute_collar_extension_dir(
    vi_pos: np.ndarray,
    inner_hull: trimesh.Trimesh,
    face_normal: np.ndarray,
    pouring_dir: np.ndarray,
) -> np.ndarray:
    """Compute the outward extension direction for a collar vertex.

    The direction is the hull surface normal at the closest point,
    projected onto the plane perpendicular to *pouring_dir*.  Falls
    back to *face_normal* and then an arbitrary perpendicular when
    the projection degenerates.

    Returns:
        Unit vector in the plane perpendicular to *pouring_dir*.
    """
    try:
        _, _, closest_faces = trimesh.proximity.closest_point(
            inner_hull, [vi_pos])
        hull_normal = inner_hull.face_normals[closest_faces[0]]
    except Exception:
        hull_normal = face_normal

    # Try hull normal projection
    perp = hull_normal - np.dot(hull_normal, pouring_dir) * pouring_dir
    if np.linalg.norm(perp) > 1e-6:
        return perp / np.linalg.norm(perp)

    # Try face normal projection
    perp = face_normal - np.dot(face_normal, pouring_dir) * pouring_dir
    if np.linalg.norm(perp) > 1e-6:
        return perp / np.linalg.norm(perp)

    # Absolute fallback: arbitrary perpendicular
    d = np.cross(pouring_dir, [1, 0, 0])
    if np.linalg.norm(d) < 1e-6:
        d = np.cross(pouring_dir, [0, 1, 0])
    return d / np.linalg.norm(d)


def create_outer_collar_extension(
    parting_surface: trimesh.Trimesh,
    inner_hull: trimesh.Trimesh,
    vertex_boundary_type: np.ndarray,
    pouring_direction: np.ndarray,
    extension_distance: float = 10.0,
    collar_subdivisions: int = DEFAULT_COLLAR_SUBDIVISIONS,
    wall_thickness: Optional[float] = None,
    prism_margin: float = 0.0,
    safety_margin: float = 2.0
) -> OuterCollarResult:
    """
    Extend the parting surface outward to pass through the hard shell prism boundary.
    
    This creates a "collar" around the outer boundary of the parting surface,
    ensuring it fully passes through the hard shell prism for proper CSG cutting.
    
    When wall_thickness is provided, uses adaptive per-vertex extension distances
    computed by ray-casting against the prism's 2D offset silhouette. This
    guarantees each collar vertex exits the prism boundary, even at sharp corners
    where a fixed multiplier would be insufficient.
    
    The extension is performed perpendicular to the pouring direction to 
    maintain the prism-like structure of the hard shell.
    
    Algorithm:
    1. Find outer boundary edges of parting surface (edges on hull, type 1 or 2)
    2. If wall_thickness provided, compute the prism's 2D offset silhouette
    3. For each outer boundary vertex:
       - Compute extension direction (perpendicular to pouring, outward from hull)
       - If adaptive: ray-cast in 2D to find exit distance from offset silhouette
       - Extend by max(computed_distance + safety_margin, extension_distance)
       - Create collar quad connecting parting surface to extended boundary
    4. Handle corners with fan triangles
    
    Args:
        parting_surface: The smoothed parting surface mesh
        inner_hull: The current hull mesh (∂H)
        vertex_boundary_type: Array with -1=part, 0=interior, 1/2=hull
        pouring_direction: Unit vector for resin pouring direction
        extension_distance: Minimum/fallback distance to extend outward (mm)
        collar_subdivisions: Number of fan subdivisions at corners
        wall_thickness: If provided, enables adaptive extension by computing
                       the prism's offset silhouette (same as create_hard_shell_prism)
        prism_margin: Additional margin used by the prism beyond wall_thickness (mm)
        safety_margin: Extra distance beyond the prism boundary for each vertex (mm)
        
    Returns:
        OuterCollarResult with the extended parting surface
    """
    start_time = time.time()
    
    logger.info(f"Creating outer collar extension for parting surface (extension={extension_distance}mm)")
    
    result = OuterCollarResult(
        mesh=parting_surface,
        vertices=np.array(parting_surface.vertices),
        faces=np.array(parting_surface.faces),
        original_vertex_count=len(parting_surface.vertices),
        original_face_count=len(parting_surface.faces),
        extension_distance=extension_distance
    )
    
    if parting_surface is None:
        logger.warning("No parting surface provided")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    # Convert to working arrays
    vertices = list(parting_surface.vertices)
    faces = list(parting_surface.faces)
    n_orig_verts = len(vertices)
    n_orig_faces = len(faces)
    
    vertices_arr = np.array(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int64)
    
    # Normalize pouring direction
    pouring_dir = np.array(pouring_direction, dtype=np.float64)
    pouring_dir = pouring_dir / (np.linalg.norm(pouring_dir) + 1e-10)
    
    # =========================================================================
    # ADAPTIVE EXTENSION: Compute prism's 2D offset silhouette for per-vertex distances
    # =========================================================================
    
    offset_silhouette_2d = None
    _collar_u_axis = None
    _collar_v_axis = None
    
    if wall_thickness is not None:
        # Build orthonormal basis for the projection plane (same as create_hard_shell_prism)
        _, _collar_u_axis, _collar_v_axis, _collar_world_to_2d = _build_pouring_basis(pouring_dir)
        
        # Project hull vertices to 2D and compute silhouette (same algorithm as prism)
        hull_vertices_3d = np.array(inner_hull.vertices, dtype=np.float64)
        hull_transformed = hull_vertices_3d @ _collar_world_to_2d.T
        projected_2d = hull_transformed[:, :2]
        
        try:
            hull_2d = ConvexHull(projected_2d)
            silhouette_indices = hull_2d.vertices
            silhouette_2d = projected_2d[silhouette_indices]
            
            # Offset by the same amount as the prism: wall_thickness + prism_margin
            total_offset = wall_thickness + prism_margin
            offset_silhouette_2d = _offset_polygon_2d(silhouette_2d, total_offset)
            
            logger.info(f"Adaptive extension: computed offset silhouette "
                       f"({len(offset_silhouette_2d)} vertices, offset={total_offset:.1f}mm, "
                       f"safety_margin={safety_margin:.1f}mm)")
        except Exception as e:
            logger.warning(f"Failed to compute offset silhouette for adaptive extension: {e}")
            logger.warning("Falling back to fixed extension_distance")
            offset_silhouette_2d = None
    
    # =========================================================================
    # STEP 1: Find all mesh boundary edges and classify as inner/outer
    # =========================================================================
    
    boundary_edges, edge_to_face = _find_boundary_edges_with_winding(faces_arr)
    
    if len(boundary_edges) == 0:
        logger.info("No boundary edges found in parting surface")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    logger.info(f"Found {len(boundary_edges)} mesh boundary edges")
    
    # =========================================================================
    # STEP 2: Classify boundary edges as inner/outer and build adjacency
    # =========================================================================
    
    (outer_boundary_edges, inner_boundary_edges,
     vertex_to_boundary_edges, corner_vertices, endpoint_vertices
    ) = _classify_boundary_edges_by_type(
        boundary_edges, vertex_boundary_type, n_orig_verts,
        vertices_arr, inner_hull,
    )
    
    logger.info(f"Outer boundary edges: {len(outer_boundary_edges)}, inner: {len(inner_boundary_edges)}")
    
    if len(outer_boundary_edges) == 0:
        logger.info("No outer boundary edges to extend")
        result.computation_time_ms = (time.time() - start_time) * 1000
        return result
    
    logger.info(f"Corner vertices: {len(corner_vertices)}, endpoints: {len(endpoint_vertices)}")
    
    # =========================================================================
    # STEP 3: Create collar vertices for outer boundary
    # =========================================================================
    
    # For each outer boundary vertex, create a collar vertex on the outer shell hull
    # The collar direction is PERPENDICULAR to pouring direction
    
    membrane_face_normals = parting_surface.face_normals
    inner_hull_normals = inner_hull.vertex_normals if hasattr(inner_hull, 'vertex_normals') else None
    
    # Map: original vertex index -> collar vertex index
    vert_to_collar: Dict[int, int] = {}
    
    # For corner vertices with multiple edges, we may need multiple collar vertices
    # edge_key -> {vertex: collar_idx}
    edge_endpoint_collar: Dict[Tuple[int, int], Dict[int, int]] = {}
    
    collar_vertices_created = 0
    _adaptive_extensions = []  # Track per-vertex computed extensions for diagnostics
    
    for v0, v1 in outer_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        if edge_key not in edge_endpoint_collar:
            edge_endpoint_collar[edge_key] = {}
        
        # Get face info for normal
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi = face_info[0]
        if fi < len(membrane_face_normals):
            face_normal = membrane_face_normals[fi]
        else:
            face_normal = np.array([0, 0, 1])
        
        for vi in [v0, v1]:
            if vi in edge_endpoint_collar[edge_key]:
                continue
            
            vi_pos = vertices_arr[vi]
            
            # Compute extension direction (perpendicular to pouring, outward from hull)
            extension_dir = _compute_collar_extension_dir(
                vi_pos, inner_hull, face_normal, pouring_dir)
            
            # ==================================================================
            # Compute extension distance (adaptive or fixed)
            # ==================================================================
            
            actual_extension = extension_distance  # Default fallback
            
            if offset_silhouette_2d is not None and _collar_u_axis is not None:
                # Project vertex position and extension direction to 2D
                vi_2d = np.array([
                    np.dot(vi_pos, _collar_u_axis),
                    np.dot(vi_pos, _collar_v_axis)
                ])
                ext_2d = np.array([
                    np.dot(extension_dir, _collar_u_axis),
                    np.dot(extension_dir, _collar_v_axis)
                ])
                ext_2d_norm = np.linalg.norm(ext_2d)
                
                if ext_2d_norm > 1e-8:
                    ext_2d_unit = ext_2d / ext_2d_norm
                    exit_dist = _ray_2d_convex_polygon_exit(
                        vi_2d, ext_2d_unit, offset_silhouette_2d
                    )
                    
                    if exit_dist is not None:
                        # Convert 2D exit distance to 3D extension distance
                        # (accounts for ext_dir not being perfectly unit in 2D)
                        required_extension = (exit_dist + safety_margin) / ext_2d_norm
                        actual_extension = max(required_extension, extension_distance)
            
            collar_pt = vi_pos + extension_dir * actual_extension
            _adaptive_extensions.append(actual_extension)
            
            # Create collar vertex
            collar_idx = len(vertices)
            vertices.append(collar_pt.copy())
            edge_endpoint_collar[edge_key][vi] = collar_idx
            
            # Also store in simple map for non-corner vertices
            if vi not in vert_to_collar:
                vert_to_collar[vi] = collar_idx
            
            collar_vertices_created += 1
    
    logger.info(f"Created {collar_vertices_created} collar vertices")
    
    # Log adaptive extension diagnostics
    if _adaptive_extensions and offset_silhouette_2d is not None:
        ext_arr = np.array(_adaptive_extensions)
        n_above_default = int(np.sum(ext_arr > extension_distance + 0.01))
        logger.info(f"Adaptive extension stats: min={ext_arr.min():.2f}mm, max={ext_arr.max():.2f}mm, "
                   f"mean={ext_arr.mean():.2f}mm, {n_above_default}/{len(ext_arr)} exceeded "
                   f"default ({extension_distance:.1f}mm)")
    
    # =========================================================================
    # STEP 5: Create quad collars for each edge
    # =========================================================================
    
    collar_faces_created = 0
    
    for v0, v1 in outer_boundary_edges:
        edge_key = (min(v0, v1), max(v0, v1))
        
        if edge_key not in edge_endpoint_collar:
            continue
        
        c0 = edge_endpoint_collar[edge_key].get(v0)
        c1 = edge_endpoint_collar[edge_key].get(v1)
        
        if c0 is None or c1 is None:
            continue
        
        # Get face to determine winding order
        face_info = edge_to_face.get(edge_key)
        if face_info is None:
            continue
        
        fi, orig_v0, _orig_v1 = face_info
        
        # Determine edge direction in the original face
        edge_forward = (orig_v0 == v0)
        
        # Create quad (2 triangles) connecting edge to collar
        # Winding: if edge is v0->v1 in face, collar should be c1->c0 for consistent normal
        if edge_forward:
            # Edge goes v0 -> v1 in face, so outward normal wants:
            # Triangle 1: v0, c0, v1
            # Triangle 2: v1, c0, c1
            faces.append([v0, c0, v1])
            faces.append([v1, c0, c1])
        else:
            # Edge goes v1 -> v0 in face
            # Triangle 1: v0, v1, c0
            # Triangle 2: v1, c1, c0
            faces.append([v0, v1, c0])
            faces.append([v1, c1, c0])
        
        collar_faces_created += 2
    
    logger.info(f"Created {collar_faces_created} collar faces")
    
    # =========================================================================
    # STEP 6: Handle corners with fan triangles
    # =========================================================================
    
    corner_fans_created = 0
    
    for vi in corner_vertices:
        edges = vertex_to_boundary_edges.get(vi, [])
        if len(edges) < 2:
            continue
        
        # Get collar vertices for each edge at this corner
        collar_verts = []
        for edge_tuple, this_v, other_v in edges:
            edge_key = (min(edge_tuple[0], edge_tuple[1]), max(edge_tuple[0], edge_tuple[1]))
            collar_map = edge_endpoint_collar.get(edge_key, {})
            collar_idx = collar_map.get(vi)
            if collar_idx is not None:
                collar_verts.append((collar_idx, other_v))
        
        if len(collar_verts) < 2:
            continue
        
        # Order collar vertices angularly around the corner
        vi_pos = vertices_arr[vi]
        
        # Compute angles for each collar vertex
        angles = []
        for collar_idx, other_v in collar_verts:
            collar_pos = np.array(vertices[collar_idx])
            vec = collar_pos - vi_pos
            # Project onto plane perpendicular to pouring direction
            vec_perp = vec - np.dot(vec, pouring_dir) * pouring_dir
            angle = np.arctan2(
                np.dot(vec_perp, np.cross(pouring_dir, [1, 0, 0])),
                np.dot(vec_perp, [1, 0, 0])
            )
            angles.append((angle, collar_idx))
        
        # Sort by angle
        angles.sort(key=lambda x: x[0])
        sorted_collars = [c for _, c in angles]
        
        # Create fan triangles connecting consecutive collar vertices
        for i in range(len(sorted_collars)):
            c_curr = sorted_collars[i]
            c_next = sorted_collars[(i + 1) % len(sorted_collars)]
            
            if c_curr != c_next:
                # Fan triangle: corner vertex to two consecutive collar vertices
                faces.append([vi, c_curr, c_next])
                corner_fans_created += 1
    
    logger.info(f"Created {corner_fans_created} corner fan triangles")
    
    # =========================================================================
    # STEP 7: Build result mesh
    # =========================================================================
    
    vertices_arr_final = np.array(vertices, dtype=np.float64)
    faces_arr_final = np.array(faces, dtype=np.int64)
    
    extended_mesh = trimesh.Trimesh(
        vertices=vertices_arr_final,
        faces=faces_arr_final,
        process=False  # Don't merge vertices
    )
    
    # Identify outer boundary vertices (the collar vertices)
    outer_boundary_verts = np.array(list(set(vert_to_collar.values())), dtype=np.int64)
    
    # Try to order the outer boundary as a loop
    outer_boundary_loop = _extract_boundary_loop(faces_arr_final, outer_boundary_verts, vertices_arr_final)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    result = OuterCollarResult(
        mesh=extended_mesh,
        vertices=vertices_arr_final,
        faces=faces_arr_final,
        original_vertex_count=n_orig_verts,
        original_face_count=n_orig_faces,
        collar_vertex_count=collar_vertices_created,
        collar_face_count=collar_faces_created + corner_fans_created,
        extension_distance=extension_distance,
        outer_boundary_vertices=outer_boundary_verts,
        outer_boundary_loop=outer_boundary_loop,
        n_outer_boundary_edges_processed=len(outer_boundary_edges),
        n_corner_fans_created=corner_fans_created,
        computation_time_ms=elapsed_ms
    )
    
    logger.info(f"Outer collar extension complete: {len(vertices)} vertices, {len(faces)} faces "
                f"(+{collar_vertices_created} verts, +{collar_faces_created + corner_fans_created} faces, "
                f"extended {extension_distance}mm) in {elapsed_ms:.1f}ms")
    
    return result


def _extract_boundary_loop(
    faces: np.ndarray,
    boundary_verts: np.ndarray,
    vertices: np.ndarray
) -> List[int]:
    """
    Extract the outer boundary as an ordered loop of vertex indices.
    
    This attempts to order the boundary vertices into a continuous loop
    by following boundary edges.
    """
    if len(boundary_verts) == 0:
        return []
    
    boundary_set = set(boundary_verts)
    
    # Build edge connectivity for boundary vertices only
    # Find edges where both vertices are in boundary_set
    edge_count = {}
    for face in faces:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            if v0 in boundary_set and v1 in boundary_set:
                edge_key = (min(v0, v1), max(v0, v1))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
    
    # True boundary edges are used once
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    
    if len(boundary_edges) == 0:
        # No true boundary - return unordered
        return list(boundary_verts)
    
    # Build adjacency
    adjacency: Dict[int, List[int]] = {}
    for v0, v1 in boundary_edges:
        if v0 not in adjacency:
            adjacency[v0] = []
        if v1 not in adjacency:
            adjacency[v1] = []
        adjacency[v0].append(v1)
        adjacency[v1].append(v0)
    
    # Walk the loop
    loop = []
    visited = set()
    
    # Start from first boundary vertex
    start = boundary_edges[0][0]
    current = start
    
    while current not in visited:
        visited.add(current)
        loop.append(current)
        
        # Find next unvisited neighbor
        neighbors = adjacency.get(current, [])
        next_v = None
        for n in neighbors:
            if n not in visited:
                next_v = n
                break
        
        if next_v is None:
            break
        current = next_v
    
    return loop


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_wall_thickness_from_bbox(mesh: trimesh.Trimesh, fraction: float = 0.02) -> float:
    """
    Compute a reasonable wall thickness based on mesh size.
    
    Args:
        mesh: Input mesh
        fraction: Fraction of bounding box diagonal (default 2%)
        
    Returns:
        Wall thickness in mesh units
    """
    bounds = mesh.bounds
    diagonal = np.linalg.norm(bounds[1] - bounds[0])
    return diagonal * fraction
