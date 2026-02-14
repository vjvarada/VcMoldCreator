"""
Mesh Repair Module using MeshLib

Provides comprehensive mesh repair functionality using MeshLib's powerful algorithms.
MeshLib offers up to 10x faster execution compared to alternative SDKs.

Repair strategies:
1. Vertex merging - merge duplicate/close vertices (uniteCloseVertices)
2. Face repair - fix degenerate faces and short edges (fixMeshDegeneracies)
3. Hole filling - fill holes with high-quality triangulation (fillHole/fillHoleNicely)
4. Self-intersection repair - fix self-intersecting geometry (SelfIntersections.fix)
5. Normal fixing - ensure consistent face winding

Design Principles:
- Strategy pattern for repair operations
- Clear separation between MeshLib and fallback implementations
- Named constants for all thresholds
- Comprehensive logging at each step
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol, Callable

import numpy as np
import trimesh

# Try to import MeshLib
try:
    import meshlib.mrmeshpy as mrmesh
    import meshlib.mrmeshnumpy as mrnumpy
    MESHLIB_AVAILABLE = True
except ImportError:
    MESHLIB_AVAILABLE = False
    mrmesh = None
    mrnumpy = None

from core.mesh_analysis import MeshAnalyzer, MeshDiagnostics

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Vertex merging threshold (0.0 = exact duplicates only)
VERTEX_MERGE_TOLERANCE = 0.0

# Degeneracy fix thresholds (relative to mesh bounding box diagonal)
DEGENERACY_MAX_DEVIATION_FACTOR = 1e-5
DEGENERACY_TINY_EDGE_FACTOR = 1e-4
DEGENERACY_CRITICAL_ASPECT_RATIO = 1e4

# Hole filling parameters
HOLE_FILL_MAX_EDGE_LEN_FACTOR = 2.0
HOLE_FILL_MAX_SPLITS = 1000

# Self-intersection fix parameters
SELF_INTERSECT_RELAX_ITERATIONS = 3
SELF_INTERSECT_MAX_EXPAND = 3


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class RepairMethod(Enum):
    """Repair methods used during mesh repair."""
    NONE = "none"
    VERTEX_MERGE = "vertex merge"
    FACE_CLEANUP = "face cleanup"
    HOLE_FILLING = "hole filling"
    SELF_INTERSECTION_FIX = "self-intersection fix"
    DEGENERACY_FIX = "degeneracy fix"
    NORMAL_FIX = "normal fix"
    CONVEX_HULL = "convex hull (mesh was not repairable)"
    BOUNDING_BOX = "bounding box (severe mesh issues)"
    COMBINED = "combined repairs"


@dataclass
class MeshRepairResult:
    """
    Result of mesh repair operation.
    
    Mirrors MeshRepairResult from the frontend.
    
    Attributes:
        mesh: The repaired mesh
        diagnostics: Post-repair mesh diagnostics
        was_repaired: Whether any repairs were applied
        repair_method: Description of repair method(s) used
        repair_steps: Detailed log of repair operations
        original_vertex_count: Vertex count before repair
        original_face_count: Face count before repair
    """
    mesh: trimesh.Trimesh
    diagnostics: MeshDiagnostics
    was_repaired: bool
    repair_method: str
    repair_steps: List[str]
    original_vertex_count: int
    original_face_count: int


@dataclass
class RepairContext:
    """
    Mutable context passed through repair operations.
    
    Tracks state during the repair pipeline.
    """
    mesh: trimesh.Trimesh
    mr_mesh: Optional['mrmesh.Mesh']
    repair_steps: List[str]
    methods_used: List[str]
    was_repaired: bool
    
    def add_step(self, message: str) -> None:
        """Add a repair step to the log."""
        self.repair_steps.append(message)
        logger.info(message)
    
    def mark_repaired(self, method: str) -> None:
        """Mark that a repair was applied."""
        self.was_repaired = True
        if method not in self.methods_used:
            self.methods_used.append(method)


# =============================================================================
# MESHLIB CONVERSION UTILITIES
# =============================================================================

def trimesh_to_meshlib(mesh: trimesh.Trimesh) -> 'mrmesh.Mesh':
    """
    Convert a trimesh mesh to MeshLib mesh.
    
    Args:
        mesh: Trimesh mesh object
        
    Returns:
        MeshLib Mesh object
        
    Raises:
        ImportError: If MeshLib is not installed
    """
    if not MESHLIB_AVAILABLE:
        raise ImportError("MeshLib is not installed. Install with: pip install meshlib")
    
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    
    return mrnumpy.meshFromFacesVerts(faces, vertices)


def meshlib_to_trimesh(mr_mesh: 'mrmesh.Mesh') -> trimesh.Trimesh:
    """
    Convert a MeshLib mesh back to trimesh.
    
    Args:
        mr_mesh: MeshLib Mesh object
        
    Returns:
        Trimesh mesh object
        
    Raises:
        ImportError: If MeshLib is not installed
    """
    if not MESHLIB_AVAILABLE:
        raise ImportError("MeshLib is not installed. Install with: pip install meshlib")
    
    vertices = mrnumpy.getNumpyVerts(mr_mesh)
    faces = mrnumpy.getNumpyFaces(mr_mesh.topology)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# =============================================================================
# REPAIR OPERATIONS (MESHLIB)
# =============================================================================

def _merge_vertices_meshlib(ctx: RepairContext) -> bool:
    """
    Merge close vertices using MeshLib.
    
    Args:
        ctx: Repair context with MeshLib mesh
        
    Returns:
        True if any vertices were merged
    """
    if ctx.mr_mesh is None:
        return False
    
    try:
        original_count = mrnumpy.getNumpyVerts(ctx.mr_mesh).shape[0]
        mrmesh.MeshBuilder.uniteCloseVertices(ctx.mr_mesh, VERTEX_MERGE_TOLERANCE)
        new_count = mrnumpy.getNumpyVerts(ctx.mr_mesh).shape[0]
        
        if new_count < original_count:
            merged = original_count - new_count
            ctx.add_step(f"Merged {merged} duplicate vertices (MeshLib)")
            return True
        
        logger.debug("No duplicate vertices found")
        return False
        
    except Exception as e:
        logger.warning("MeshLib vertex merge failed: %s", e)
        ctx.add_step(f"Vertex merge error: {e}")
        return False


def _fix_degeneracies_meshlib(ctx: RepairContext) -> bool:
    """
    Fix degenerate faces and short edges using MeshLib.
    
    Args:
        ctx: Repair context with MeshLib mesh
        
    Returns:
        True if any degeneracies were fixed
    """
    if ctx.mr_mesh is None:
        return False
    
    try:
        original_face_count = mrnumpy.getNumpyFaces(ctx.mr_mesh.topology).shape[0]
        
        # Compute tolerance based on mesh size
        bbox = ctx.mr_mesh.computeBoundingBox()
        diagonal = bbox.diagonal()
        
        params = mrmesh.FixMeshDegeneraciesParams()
        params.maxDeviation = DEGENERACY_MAX_DEVIATION_FACTOR * diagonal
        params.tinyEdgeLength = DEGENERACY_TINY_EDGE_FACTOR * diagonal
        params.criticalTriAspectRatio = DEGENERACY_CRITICAL_ASPECT_RATIO
        params.mode = mrmesh.FixMeshDegeneraciesParams.Mode.Remesh
        
        mrmesh.fixMeshDegeneracies(ctx.mr_mesh, params)
        
        new_face_count = mrnumpy.getNumpyFaces(ctx.mr_mesh.topology).shape[0]
        
        if new_face_count != original_face_count:
            diff = original_face_count - new_face_count
            if diff > 0:
                ctx.add_step(f"Fixed degeneracies: removed {diff} degenerate faces (MeshLib)")
            else:
                ctx.add_step(f"Fixed degeneracies: added {-diff} faces during remesh (MeshLib)")
            return True
        
        logger.debug("No degeneracies found")
        return False
        
    except Exception as e:
        logger.warning("MeshLib degeneracy fix failed: %s", e)
        ctx.add_step(f"Degeneracy fix error: {e}")
        return False


def _fix_self_intersections_meshlib(ctx: RepairContext) -> bool:
    """
    Fix self-intersecting geometry using MeshLib.
    
    Args:
        ctx: Repair context with MeshLib mesh
        
    Returns:
        True if any self-intersections were fixed
    """
    if ctx.mr_mesh is None:
        return False
    
    try:
        self_intersections = mrmesh.SelfIntersections.getFaces(ctx.mr_mesh)
        intersection_count = self_intersections.count()
        
        if intersection_count == 0:
            logger.debug("No self-intersections found")
            return False
        
        logger.info("Found %d self-intersecting faces, fixing...", intersection_count)
        
        settings = mrmesh.SelfIntersections.Settings()
        settings.method = mrmesh.SelfIntersections.Settings.Method.CutAndFill
        settings.relaxIterations = SELF_INTERSECT_RELAX_ITERATIONS
        settings.maxExpand = SELF_INTERSECT_MAX_EXPAND
        
        mrmesh.SelfIntersections.fix(ctx.mr_mesh, settings)
        
        remaining = mrmesh.SelfIntersections.getFaces(ctx.mr_mesh).count()
        fixed = intersection_count - remaining
        
        if fixed > 0:
            ctx.add_step(f"Fixed {fixed} self-intersecting faces (MeshLib)")
            if remaining > 0:
                ctx.add_step(f"Note: {remaining} self-intersections could not be fixed")
            return True
        
        if remaining > 0:
            ctx.add_step(f"Could not fix {remaining} self-intersections")
        
        return False
        
    except Exception as e:
        logger.warning("MeshLib self-intersection fix failed: %s", e)
        ctx.add_step(f"Self-intersection fix error: {e}")
        return False


def _fill_holes_meshlib(ctx: RepairContext) -> bool:
    """
    Fill holes in the mesh using MeshLib.
    
    Args:
        ctx: Repair context with MeshLib mesh
        
    Returns:
        True if any holes were filled
    """
    if ctx.mr_mesh is None:
        return False
    
    try:
        hole_edges = ctx.mr_mesh.topology.findHoleRepresentiveEdges()
        num_holes = hole_edges.size()
        
        if num_holes == 0:
            logger.debug("No holes found")
            return False
        
        logger.info("Found %d holes, filling...", num_holes)
        
        holes_filled = 0
        for edge in hole_edges:
            if _fill_single_hole(ctx.mr_mesh, edge):
                holes_filled += 1
        
        if holes_filled > 0:
            ctx.add_step(f"Filled {holes_filled} of {num_holes} holes (MeshLib)")
            return True
        
        ctx.add_step(f"Could not fill any of {num_holes} holes")
        return False
        
    except Exception as e:
        logger.warning("MeshLib hole filling failed: %s", e)
        ctx.add_step(f"Hole filling error: {e}")
        return False


def _fill_single_hole(mr_mesh: 'mrmesh.Mesh', edge) -> bool:
    """
    Attempt to fill a single hole using nice fill first, then simple fill.
    
    Args:
        mr_mesh: MeshLib mesh
        edge: Edge identifying the hole
        
    Returns:
        True if hole was filled successfully
    """
    # Try fillHoleNicely first for better quality
    try:
        settings = mrmesh.FillHoleNicelySettings()
        settings.triangulateParams.metric = mrmesh.getUniversalMetric(mr_mesh)
        settings.triangulateParams.multipleEdgesResolveMode = (
            mrmesh.FillHoleParams.MultipleEdgesResolveMode.Strong
        )
        settings.maxEdgeLen = mr_mesh.averageEdgeLength() * HOLE_FILL_MAX_EDGE_LEN_FACTOR
        settings.maxEdgeSplits = HOLE_FILL_MAX_SPLITS
        settings.triangulateOnly = False
        settings.smoothCurvature = True
        
        mrmesh.fillHoleNicely(mr_mesh, edge, settings)
        return True
        
    except Exception:
        pass
    
    # Fallback to simple fillHole
    try:
        params = mrmesh.FillHoleParams()
        params.metric = mrmesh.getUniversalMetric(mr_mesh)
        mrmesh.fillHole(mr_mesh, edge, params)
        return True
        
    except Exception as e:
        logger.debug("Could not fill hole: %s", e)
        return False


# =============================================================================
# REPAIR OPERATIONS (TRIMESH FALLBACK)
# =============================================================================

def _merge_vertices_trimesh(ctx: RepairContext) -> bool:
    """Merge close vertices using trimesh."""
    try:
        original_count = len(ctx.mesh.vertices)
        ctx.mesh.merge_vertices(merge_tex=True, merge_norm=True)
        new_count = len(ctx.mesh.vertices)
        
        if new_count < original_count:
            merged = original_count - new_count
            ctx.add_step(f"Merged {merged} duplicate vertices")
            return True
        return False
        
    except Exception as e:
        ctx.add_step(f"Vertex merge failed: {e}")
        return False


def _remove_degenerate_trimesh(ctx: RepairContext) -> bool:
    """Remove degenerate faces using trimesh."""
    try:
        original_count = len(ctx.mesh.faces)
        nondegenerate = ctx.mesh.nondegenerate_faces()
        
        if nondegenerate.sum() < original_count:
            ctx.mesh.update_faces(nondegenerate)
            removed = original_count - len(ctx.mesh.faces)
            ctx.add_step(f"Removed {removed} degenerate faces")
            return True
        return False
        
    except Exception as e:
        ctx.add_step(f"Degenerate face removal failed: {e}")
        return False


def _remove_duplicates_trimesh(ctx: RepairContext) -> bool:
    """Remove duplicate faces using trimesh."""
    try:
        original_count = len(ctx.mesh.faces)
        unique_mask = ctx.mesh.unique_faces()
        
        if unique_mask.sum() < original_count:
            ctx.mesh.update_faces(unique_mask)
            removed = original_count - len(ctx.mesh.faces)
            ctx.add_step(f"Removed {removed} duplicate faces")
            return True
        return False
        
    except Exception as e:
        ctx.add_step(f"Duplicate face removal failed: {e}")
        return False


def _fill_holes_trimesh(ctx: RepairContext) -> bool:
    """Fill holes using trimesh."""
    try:
        if not ctx.mesh.is_watertight:
            ctx.mesh.fill_holes()
            if ctx.mesh.is_watertight:
                ctx.add_step("Filled holes to make mesh watertight")
                return True
        return False
        
    except Exception as e:
        ctx.add_step(f"Hole filling failed: {e}")
        return False


def _fix_normals(ctx: RepairContext) -> bool:
    """Fix face normals to have consistent winding."""
    try:
        if ctx.mesh.is_watertight and ctx.mesh.volume < 0:
            ctx.mesh.invert()
            ctx.add_step("Inverted mesh (was inside-out)")
            return True
        
        ctx.mesh.fix_normals()
        ctx.add_step("Fixed face normals for consistent winding")
        return True
        
    except Exception as e:
        ctx.add_step(f"Normal fix failed: {e}")
        return False


def _create_bounding_box_mesh(original_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Create a box mesh from the original bounding box."""
    bounds = original_mesh.bounds
    extents = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    
    return box


def _try_convex_hull_fallback(ctx: RepairContext, original_mesh: trimesh.Trimesh) -> bool:
    """Try convex hull as a fallback repair method."""
    try:
        hull = ctx.mesh.convex_hull
        if hull is not None and len(hull.faces) > 0:
            ctx.mesh = hull
            ctx.add_step("Created convex hull as fallback")
            ctx.methods_used = [RepairMethod.CONVEX_HULL.value]
            return True
    except Exception as e:
        ctx.add_step(f"Convex hull failed: {e}")
    
    # Last resort: bounding box
    try:
        ctx.mesh = _create_bounding_box_mesh(original_mesh)
        ctx.add_step("Created bounding box as last resort")
        ctx.methods_used = [RepairMethod.BOUNDING_BOX.value]
        return True
    except Exception as e:
        ctx.add_step(f"Bounding box creation failed: {e}")
    
    return False


# =============================================================================
# MAIN REPAIR PIPELINES
# =============================================================================

def _run_meshlib_repair(
    ctx: RepairContext,
    merge_vertices: bool,
    remove_degenerate: bool,
    fix_self_intersections: bool,
    fill_holes: bool,
    fix_normals: bool
) -> None:
    """
    Run MeshLib-based repair pipeline.
    
    Args:
        ctx: Repair context
        merge_vertices: Whether to merge close vertices
        remove_degenerate: Whether to fix degenerate faces
        fix_self_intersections: Whether to fix self-intersections
        fill_holes: Whether to fill holes
        fix_normals: Whether to fix normals
    """
    logger.info("Converting mesh to MeshLib format...")
    ctx.mr_mesh = trimesh_to_meshlib(ctx.mesh)
    
    initial_verts = mrnumpy.getNumpyVerts(ctx.mr_mesh).shape[0]
    initial_faces = mrnumpy.getNumpyFaces(ctx.mr_mesh.topology).shape[0]
    logger.info("MeshLib mesh: %d vertices, %d faces", initial_verts, initial_faces)
    
    # Run repair operations
    if merge_vertices and _merge_vertices_meshlib(ctx):
        ctx.mark_repaired("vertex merge")
    
    if remove_degenerate and _fix_degeneracies_meshlib(ctx):
        ctx.mark_repaired("degeneracy fix")
    
    if fix_self_intersections and _fix_self_intersections_meshlib(ctx):
        ctx.mark_repaired("self-intersection fix")
    
    if fill_holes and _fill_holes_meshlib(ctx):
        ctx.mark_repaired("hole filling")
    
    # Convert back to trimesh
    logger.info("Converting mesh back to trimesh format...")
    ctx.mesh = meshlib_to_trimesh(ctx.mr_mesh)
    
    # Fix normals (using trimesh)
    if fix_normals and _fix_normals(ctx):
        ctx.mark_repaired("normal fix")


def _run_trimesh_repair(
    ctx: RepairContext,
    merge_vertices: bool,
    remove_degenerate: bool,
    remove_duplicates: bool,
    fix_normals: bool,
    fill_holes: bool
) -> None:
    """
    Run trimesh-based fallback repair pipeline.
    
    Args:
        ctx: Repair context
        merge_vertices: Whether to merge close vertices
        remove_degenerate: Whether to remove degenerate faces
        remove_duplicates: Whether to remove duplicate faces  
        fix_normals: Whether to fix normals
        fill_holes: Whether to fill holes
    """
    ctx.add_step("Using fallback trimesh repair (MeshLib unavailable)")
    
    if merge_vertices and _merge_vertices_trimesh(ctx):
        ctx.mark_repaired("vertex merge")
    
    if remove_degenerate and _remove_degenerate_trimesh(ctx):
        ctx.mark_repaired("degenerate removal")
    
    if remove_duplicates and _remove_duplicates_trimesh(ctx):
        ctx.mark_repaired("duplicate removal")
    
    if fix_normals and _fix_normals(ctx):
        ctx.mark_repaired("normal fix")
    
    if fill_holes and _fill_holes_trimesh(ctx):
        ctx.mark_repaired("hole filling")


def _build_result(
    ctx: RepairContext,
    original_vertex_count: int,
    original_face_count: int
) -> MeshRepairResult:
    """Build the final MeshRepairResult from context."""
    analyzer = MeshAnalyzer(ctx.mesh)
    diagnostics = analyzer.analyze()
    
    # Determine repair method string
    if not ctx.was_repaired:
        repair_method = RepairMethod.NONE.value
    elif len(ctx.methods_used) == 1:
        repair_method = ctx.methods_used[0]
    else:
        repair_method = f"combined ({', '.join(ctx.methods_used)})"
    
    if ctx.was_repaired:
        diagnostics.issues.insert(0, f"Repaired using: {repair_method}")
    
    return MeshRepairResult(
        mesh=ctx.mesh,
        diagnostics=diagnostics,
        was_repaired=ctx.was_repaired,
        repair_method=repair_method,
        repair_steps=ctx.repair_steps.copy(),
        original_vertex_count=original_vertex_count,
        original_face_count=original_face_count
    )


# =============================================================================
# PUBLIC API
# =============================================================================

class MeshRepairer:
    """
    Mesh repair using MeshLib's powerful algorithms.
    
    Attempts to create a valid, watertight mesh from potentially broken input.
    Uses MeshLib for high-performance mesh repair operations with automatic
    fallback to trimesh operations if MeshLib is unavailable.
    
    Example:
        >>> repairer = MeshRepairer(mesh)
        >>> result = repairer.repair(fill_holes=True)
        >>> if result.was_repaired:
        ...     print(f"Applied: {result.repair_method}")
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize repairer with a mesh.
        
        Args:
            mesh: The trimesh mesh to repair
        """
        self.original_mesh = mesh
        self.mesh = mesh.copy()
    
    def repair(
        self,
        merge_vertices: bool = True,
        remove_degenerate: bool = True,
        remove_duplicates: bool = True,
        fix_normals: bool = True,
        fill_holes: bool = True,
        fix_self_intersections: bool = True,
        use_convex_hull_fallback: bool = False
    ) -> MeshRepairResult:
        """
        Repair the mesh using available algorithms.
        
        Attempts MeshLib-based repair first, falling back to basic trimesh
        operations if MeshLib is unavailable or fails.
        
        Args:
            merge_vertices: Merge vertices that are very close together
            remove_degenerate: Remove degenerate faces and short edges
            remove_duplicates: Remove duplicate faces
            fix_normals: Fix inconsistent face winding
            fill_holes: Attempt to fill holes
            fix_self_intersections: Fix self-intersecting geometry
            use_convex_hull_fallback: Use convex hull if repair fails
            
        Returns:
            MeshRepairResult with the repaired mesh and diagnostics
        """
        original_vertex_count = len(self.original_mesh.vertices)
        original_face_count = len(self.original_mesh.faces)
        
        ctx = RepairContext(
            mesh=self.mesh.copy(),
            mr_mesh=None,
            repair_steps=[],
            methods_used=[],
            was_repaired=False
        )
        
        # Try MeshLib first if available
        if MESHLIB_AVAILABLE:
            try:
                _run_meshlib_repair(
                    ctx,
                    merge_vertices=merge_vertices,
                    remove_degenerate=remove_degenerate,
                    fix_self_intersections=fix_self_intersections,
                    fill_holes=fill_holes,
                    fix_normals=fix_normals
                )
            except Exception as e:
                logger.exception("MeshLib repair failed: %s", e)
                ctx.add_step(f"MeshLib repair error: {e}")
                logger.info("Falling back to basic trimesh repair...")
                
                # Reset context for fallback
                ctx.mesh = self.mesh.copy()
                ctx.mr_mesh = None
                ctx.methods_used = []
                ctx.was_repaired = False
                
                _run_trimesh_repair(
                    ctx,
                    merge_vertices=merge_vertices,
                    remove_degenerate=remove_degenerate,
                    remove_duplicates=remove_duplicates,
                    fix_normals=fix_normals,
                    fill_holes=fill_holes
                )
        else:
            logger.warning("MeshLib not available, using trimesh repair")
            _run_trimesh_repair(
                ctx,
                merge_vertices=merge_vertices,
                remove_degenerate=remove_degenerate,
                remove_duplicates=remove_duplicates,
                fix_normals=fix_normals,
                fill_holes=fill_holes
            )
        
        # Build result and check if convex hull fallback is needed
        result = _build_result(ctx, original_vertex_count, original_face_count)
        
        if not result.diagnostics.is_watertight and use_convex_hull_fallback:
            ctx.add_step("Mesh still not watertight, attempting convex hull fallback")
            if _try_convex_hull_fallback(ctx, self.original_mesh):
                ctx.was_repaired = True
                result = _build_result(ctx, original_vertex_count, original_face_count)
        
        # Update instance mesh reference
        self.mesh = ctx.mesh
        
        return result


def is_meshlib_available() -> bool:
    """Check if MeshLib is available."""
    return MESHLIB_AVAILABLE


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Keep old function names as aliases for backward compatibility
_trimesh_to_meshlib = trimesh_to_meshlib
_meshlib_to_trimesh = meshlib_to_trimesh
