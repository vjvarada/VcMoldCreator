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
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import logging
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
    """
    mesh: trimesh.Trimesh
    diagnostics: MeshDiagnostics
    was_repaired: bool
    repair_method: str
    repair_steps: List[str]
    original_vertex_count: int
    original_face_count: int


def _trimesh_to_meshlib(mesh: trimesh.Trimesh) -> 'mrmesh.Mesh':
    """
    Convert a trimesh mesh to MeshLib mesh.
    
    Args:
        mesh: Trimesh mesh object
        
    Returns:
        MeshLib Mesh object
    """
    if not MESHLIB_AVAILABLE:
        raise ImportError("MeshLib is not installed. Install with: pip install meshlib")
    
    # Get vertices and faces as numpy arrays
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    
    # Create MeshLib mesh using mrmeshnumpy
    mr_mesh = mrnumpy.meshFromFacesVerts(faces, vertices)
    return mr_mesh


def _meshlib_to_trimesh(mr_mesh: 'mrmesh.Mesh') -> trimesh.Trimesh:
    """
    Convert a MeshLib mesh back to trimesh.
    
    Args:
        mr_mesh: MeshLib Mesh object
        
    Returns:
        Trimesh mesh object
    """
    if not MESHLIB_AVAILABLE:
        raise ImportError("MeshLib is not installed. Install with: pip install meshlib")
    
    # Get numpy arrays using mrmeshnumpy
    vertices = mrnumpy.getNumpyVerts(mr_mesh)
    faces = mrnumpy.getNumpyFaces(mr_mesh.topology)
    
    # Create trimesh
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class MeshRepairer:
    """
    Mesh repair using MeshLib's powerful algorithms.
    
    Attempts to create a valid, watertight mesh from potentially broken input.
    Uses MeshLib for high-performance mesh repair operations.
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize repairer with a mesh.
        
        Args:
            mesh: The trimesh mesh to repair
        """
        self.original_mesh = mesh
        self.mesh = mesh.copy()
        self._repair_steps: List[str] = []
        self._mr_mesh: Optional['mrmesh.Mesh'] = None
    
    def repair(self, 
               merge_vertices: bool = True,
               remove_degenerate: bool = True,
               remove_duplicates: bool = True,
               fix_normals: bool = True,
               fill_holes: bool = True,
               fix_self_intersections: bool = True,
               use_convex_hull_fallback: bool = False) -> MeshRepairResult:
        """
        Repair the mesh using MeshLib's algorithms.
        
        Args:
            merge_vertices: Merge vertices that are very close together
            remove_degenerate: Remove degenerate faces and short edges
            remove_duplicates: Remove duplicate faces (handled by MeshLib during conversion)
            fix_normals: Fix inconsistent face winding
            fill_holes: Attempt to fill holes
            fix_self_intersections: Fix self-intersecting geometry
            use_convex_hull_fallback: Use convex hull if repair fails (default False)
            
        Returns:
            MeshRepairResult with the repaired mesh and diagnostics
        """
        self._repair_steps = []
        original_vertex_count = len(self.original_mesh.vertices)
        original_face_count = len(self.original_mesh.faces)
        
        was_repaired = False
        repair_methods_used = []
        
        # Check if MeshLib is available
        if not MESHLIB_AVAILABLE:
            logger.warning("MeshLib not available, falling back to basic trimesh repair")
            return self._fallback_repair(
                merge_vertices, remove_degenerate, remove_duplicates,
                fix_normals, fill_holes, use_convex_hull_fallback
            )
        
        try:
            # Convert to MeshLib format
            logger.info("Converting mesh to MeshLib format...")
            self._mr_mesh = _trimesh_to_meshlib(self.mesh)
            initial_verts = mrnumpy.getNumpyVerts(self._mr_mesh).shape[0]
            initial_faces = mrnumpy.getNumpyFaces(self._mr_mesh.topology).shape[0]
            logger.info(f"MeshLib mesh created: {initial_verts} vertices, {initial_faces} faces")
            
            # Step 1: Merge close vertices
            if merge_vertices:
                result = self._merge_vertices_meshlib()
                if result:
                    was_repaired = True
                    repair_methods_used.append("vertex merge")
            
            # Step 2: Fix degeneracies (includes removing degenerate faces and short edges)
            if remove_degenerate:
                result = self._fix_degeneracies_meshlib()
                if result:
                    was_repaired = True
                    repair_methods_used.append("degeneracy fix")
            
            # Step 3: Fix self-intersections
            if fix_self_intersections:
                result = self._fix_self_intersections_meshlib()
                if result:
                    was_repaired = True
                    repair_methods_used.append("self-intersection fix")
            
            # Step 4: Fill holes
            if fill_holes:
                result = self._fill_holes_meshlib()
                if result:
                    was_repaired = True
                    repair_methods_used.append("hole filling")
            
            # Convert back to trimesh
            logger.info("Converting mesh back to trimesh format...")
            self.mesh = _meshlib_to_trimesh(self._mr_mesh)
            
            # Step 5: Fix normals (using trimesh as it's simpler)
            if fix_normals:
                result = self._fix_normals()
                if result:
                    was_repaired = True
                    repair_methods_used.append("normal fix")
            
        except Exception as e:
            logger.exception("MeshLib repair failed: %s", e)
            self._repair_steps.append(f"MeshLib repair error: {e}")
            # Fall back to basic trimesh repair
            logger.info("Falling back to basic trimesh repair...")
            return self._fallback_repair(
                merge_vertices, remove_degenerate, remove_duplicates,
                fix_normals, fill_holes, use_convex_hull_fallback
            )
        
        # Check if mesh is now valid
        analyzer = MeshAnalyzer(self.mesh)
        diagnostics = analyzer.analyze()
        
        # If still not watertight, try convex hull as fallback
        if not diagnostics.is_watertight and use_convex_hull_fallback:
            self._repair_steps.append("Mesh still not watertight, attempting convex hull fallback")
            
            try:
                hull = self.mesh.convex_hull
                if hull is not None and len(hull.faces) > 0:
                    self.mesh = hull
                    was_repaired = True
                    repair_methods_used = [RepairMethod.CONVEX_HULL.value]
                    self._repair_steps.append("Created convex hull as fallback")
                    
                    # Re-analyze
                    analyzer = MeshAnalyzer(self.mesh)
                    diagnostics = analyzer.analyze()
            except Exception as e:
                self._repair_steps.append(f"Convex hull failed: {e}")
                
                # Last resort: bounding box
                try:
                    self.mesh = self._create_bounding_box_mesh()
                    was_repaired = True
                    repair_methods_used = [RepairMethod.BOUNDING_BOX.value]
                    self._repair_steps.append("Created bounding box as last resort")
                    
                    analyzer = MeshAnalyzer(self.mesh)
                    diagnostics = analyzer.analyze()
                except Exception as e2:
                    self._repair_steps.append(f"Bounding box creation failed: {e2}")
        
        # Determine repair method string
        if not was_repaired:
            repair_method = RepairMethod.NONE.value
        elif len(repair_methods_used) == 1:
            repair_method = repair_methods_used[0]
        else:
            repair_method = f"combined ({', '.join(repair_methods_used)})"
        
        # Add repair info to diagnostics issues
        if was_repaired:
            diagnostics.issues.insert(0, f"Repaired using: {repair_method}")
        
        return MeshRepairResult(
            mesh=self.mesh,
            diagnostics=diagnostics,
            was_repaired=was_repaired,
            repair_method=repair_method,
            repair_steps=self._repair_steps.copy(),
            original_vertex_count=original_vertex_count,
            original_face_count=original_face_count
        )
    
    def _merge_vertices_meshlib(self) -> bool:
        """
        Merge close vertices using MeshLib.
        
        Returns:
            True if any vertices were merged
        """
        if self._mr_mesh is None:
            return False
        
        try:
            original_count = mrnumpy.getNumpyVerts(self._mr_mesh).shape[0]
            
            # Use MeshLib's uniteCloseVertices
            mrmesh.MeshBuilder.uniteCloseVertices(self._mr_mesh, 0.0)
            
            new_count = mrnumpy.getNumpyVerts(self._mr_mesh).shape[0]
            if new_count < original_count:
                merged = original_count - new_count
                self._repair_steps.append(f"Merged {merged} duplicate vertices (MeshLib)")
                logger.info(f"Merged {merged} vertices")
                return True
            else:
                logger.info("No duplicate vertices found")
                
        except Exception as e:
            logger.warning(f"MeshLib vertex merge failed: {e}")
            self._repair_steps.append(f"Vertex merge error: {e}")
        
        return False
    
    def _fix_degeneracies_meshlib(self) -> bool:
        """
        Fix degenerate faces and short edges using MeshLib.
        
        Returns:
            True if any degeneracies were fixed
        """
        if self._mr_mesh is None:
            return False
        
        try:
            # Check for degenerate faces first
            original_face_count = mrnumpy.getNumpyFaces(self._mr_mesh.topology).shape[0]
            
            # Get bounding box diagonal for tolerance calculation
            bbox = self._mr_mesh.computeBoundingBox()
            diagonal = bbox.diagonal()
            
            # Set up parameters for fixing degeneracies
            params = mrmesh.FixMeshDegeneraciesParams()
            params.maxDeviation = 1e-5 * diagonal  # Small tolerance relative to mesh size
            params.tinyEdgeLength = 1e-4 * diagonal  # Very small edges to collapse
            params.criticalTriAspectRatio = 1e4  # Very thin triangles
            params.mode = mrmesh.FixMeshDegeneraciesParams.Mode.Remesh  # Use remesh mode for better results
            
            # Fix degeneracies
            mrmesh.fixMeshDegeneracies(self._mr_mesh, params)
            
            new_face_count = mrnumpy.getNumpyFaces(self._mr_mesh.topology).shape[0]
            
            if new_face_count != original_face_count:
                diff = original_face_count - new_face_count
                if diff > 0:
                    self._repair_steps.append(f"Fixed degeneracies: removed {diff} degenerate faces (MeshLib)")
                else:
                    self._repair_steps.append(f"Fixed degeneracies: added {-diff} faces during remesh (MeshLib)")
                logger.info(f"Fixed degeneracies: {original_face_count} -> {new_face_count} faces")
                return True
            else:
                logger.info("No degeneracies found")
                
        except Exception as e:
            logger.warning(f"MeshLib degeneracy fix failed: {e}")
            self._repair_steps.append(f"Degeneracy fix error: {e}")
        
        return False
    
    def _fix_self_intersections_meshlib(self) -> bool:
        """
        Fix self-intersecting geometry using MeshLib.
        
        Returns:
            True if any self-intersections were fixed
        """
        if self._mr_mesh is None:
            return False
        
        try:
            # Check for self-intersections first
            self_intersections = mrmesh.SelfIntersections.getFaces(self._mr_mesh)
            intersection_count = self_intersections.count()
            
            if intersection_count == 0:
                logger.info("No self-intersections found")
                return False
            
            logger.info(f"Found {intersection_count} self-intersecting faces, fixing...")
            
            # Set up fix parameters
            settings = mrmesh.SelfIntersections.Settings()
            settings.method = mrmesh.SelfIntersections.Settings.Method.CutAndFill
            settings.relaxIterations = 3
            settings.maxExpand = 3
            
            # Fix self-intersections
            mrmesh.SelfIntersections.fix(self._mr_mesh, settings)
            
            # Verify fix
            remaining = mrmesh.SelfIntersections.getFaces(self._mr_mesh).count()
            
            fixed = intersection_count - remaining
            if fixed > 0:
                self._repair_steps.append(f"Fixed {fixed} self-intersecting faces (MeshLib)")
                logger.info(f"Fixed {fixed} self-intersections, {remaining} remaining")
                return True
            elif remaining > 0:
                self._repair_steps.append(f"Could not fix all self-intersections: {remaining} remaining")
                logger.warning(f"Could not fix all self-intersections: {remaining} remaining")
                
        except Exception as e:
            logger.warning(f"MeshLib self-intersection fix failed: {e}")
            self._repair_steps.append(f"Self-intersection fix error: {e}")
        
        return False
    
    def _fill_holes_meshlib(self) -> bool:
        """
        Fill holes in the mesh using MeshLib.
        
        Returns:
            True if any holes were filled
        """
        if self._mr_mesh is None:
            return False
        
        try:
            # Find holes
            hole_edges = self._mr_mesh.topology.findHoleRepresentiveEdges()
            num_holes = hole_edges.size()
            
            if num_holes == 0:
                logger.info("No holes found")
                return False
            
            logger.info(f"Found {num_holes} holes, filling...")
            
            # Fill each hole - iterate directly over hole_edges
            holes_filled = 0
            for i, edge in enumerate(hole_edges):
                try:
                    # Use fillHoleNicely for better quality
                    settings = mrmesh.FillHoleNicelySettings()
                    settings.triangulateParams.metric = mrmesh.getUniversalMetric(self._mr_mesh)
                    settings.triangulateParams.multipleEdgesResolveMode = mrmesh.FillHoleParams.MultipleEdgesResolveMode.Strong
                    settings.maxEdgeLen = self._mr_mesh.averageEdgeLength() * 2.0
                    settings.maxEdgeSplits = 1000
                    settings.triangulateOnly = False
                    settings.smoothCurvature = True
                    
                    mrmesh.fillHoleNicely(self._mr_mesh, edge, settings)
                    holes_filled += 1
                    
                except Exception as hole_error:
                    # Try simpler fillHole if fillHoleNicely fails
                    try:
                        params = mrmesh.FillHoleParams()
                        params.metric = mrmesh.getUniversalMetric(self._mr_mesh)
                        mrmesh.fillHole(self._mr_mesh, edge, params)
                        holes_filled += 1
                    except Exception as e2:
                        logger.warning(f"Could not fill hole {i}: {e2}")
            
            if holes_filled > 0:
                self._repair_steps.append(f"Filled {holes_filled} holes (MeshLib)")
                logger.info(f"Filled {holes_filled} of {num_holes} holes")
                return True
            else:
                self._repair_steps.append(f"Could not fill any of {num_holes} holes")
                logger.warning(f"Could not fill any of {num_holes} holes")
                
        except Exception as e:
            logger.warning(f"MeshLib hole filling failed: {e}")
            self._repair_steps.append(f"Hole filling error: {e}")
        
        return False
    
    def _fix_normals(self) -> bool:
        """
        Fix face normals to have consistent winding.
        
        Returns:
            True if normals were fixed
        """
        try:
            # Check if mesh volume is negative (inside-out)
            if self.mesh.is_watertight and self.mesh.volume < 0:
                self.mesh.invert()
                self._repair_steps.append("Inverted mesh (was inside-out)")
                return True
            
            # Fix winding for consistent normals
            self.mesh.fix_normals()
            self._repair_steps.append("Fixed face normals for consistent winding")
            return True
            
        except Exception as e:
            self._repair_steps.append(f"Normal fix failed: {e}")
        
        return False
    
    def _fallback_repair(self, merge_vertices: bool, remove_degenerate: bool,
                         remove_duplicates: bool, fix_normals: bool,
                         fill_holes: bool, use_convex_hull_fallback: bool) -> MeshRepairResult:
        """
        Fallback repair using basic trimesh operations when MeshLib is unavailable.
        """
        original_vertex_count = len(self.original_mesh.vertices)
        original_face_count = len(self.original_mesh.faces)
        
        was_repaired = False
        repair_methods_used = []
        
        self._repair_steps.append("Using fallback trimesh repair (MeshLib unavailable)")
        
        # Basic trimesh repairs
        if merge_vertices:
            try:
                original_count = len(self.mesh.vertices)
                self.mesh.merge_vertices(merge_tex=True, merge_norm=True)
                new_count = len(self.mesh.vertices)
                if new_count < original_count:
                    merged = original_count - new_count
                    self._repair_steps.append(f"Merged {merged} duplicate vertices")
                    was_repaired = True
                    repair_methods_used.append("vertex merge")
            except Exception as e:
                self._repair_steps.append(f"Vertex merge failed: {e}")
        
        if remove_degenerate:
            try:
                original_count = len(self.mesh.faces)
                nondegenerate = self.mesh.nondegenerate_faces()
                if nondegenerate.sum() < original_count:
                    self.mesh.update_faces(nondegenerate)
                    removed = original_count - len(self.mesh.faces)
                    self._repair_steps.append(f"Removed {removed} degenerate faces")
                    was_repaired = True
                    repair_methods_used.append("degenerate removal")
            except Exception as e:
                self._repair_steps.append(f"Degenerate face removal failed: {e}")
        
        if remove_duplicates:
            try:
                original_count = len(self.mesh.faces)
                unique_mask = self.mesh.unique_faces()
                if unique_mask.sum() < original_count:
                    self.mesh.update_faces(unique_mask)
                    removed = original_count - len(self.mesh.faces)
                    self._repair_steps.append(f"Removed {removed} duplicate faces")
                    was_repaired = True
                    repair_methods_used.append("duplicate removal")
            except Exception as e:
                self._repair_steps.append(f"Duplicate face removal failed: {e}")
        
        if fix_normals:
            result = self._fix_normals()
            if result:
                was_repaired = True
                repair_methods_used.append("normal fix")
        
        if fill_holes:
            try:
                if not self.mesh.is_watertight:
                    self.mesh.fill_holes()
                    if self.mesh.is_watertight:
                        self._repair_steps.append("Filled holes to make mesh watertight")
                        was_repaired = True
                        repair_methods_used.append("hole filling")
            except Exception as e:
                self._repair_steps.append(f"Hole filling failed: {e}")
        
        # Check if mesh is now valid
        analyzer = MeshAnalyzer(self.mesh)
        diagnostics = analyzer.analyze()
        
        # Convex hull fallback
        if not diagnostics.is_watertight and use_convex_hull_fallback:
            try:
                hull = self.mesh.convex_hull
                if hull is not None and len(hull.faces) > 0:
                    self.mesh = hull
                    was_repaired = True
                    repair_methods_used = [RepairMethod.CONVEX_HULL.value]
                    self._repair_steps.append("Created convex hull as fallback")
                    analyzer = MeshAnalyzer(self.mesh)
                    diagnostics = analyzer.analyze()
            except Exception as e:
                self._repair_steps.append(f"Convex hull failed: {e}")
        
        # Determine repair method string
        if not was_repaired:
            repair_method = RepairMethod.NONE.value
        elif len(repair_methods_used) == 1:
            repair_method = repair_methods_used[0]
        else:
            repair_method = f"combined ({', '.join(repair_methods_used)})"
        
        if was_repaired:
            diagnostics.issues.insert(0, f"Repaired using: {repair_method}")
        
        return MeshRepairResult(
            mesh=self.mesh,
            diagnostics=diagnostics,
            was_repaired=was_repaired,
            repair_method=repair_method,
            repair_steps=self._repair_steps.copy(),
            original_vertex_count=original_vertex_count,
            original_face_count=original_face_count
        )
    
    def _create_bounding_box_mesh(self) -> trimesh.Trimesh:
        """
        Create a mesh from the bounding box as a last resort.
        
        Returns:
            A box mesh matching the original bounding box
        """
        bounds = self.original_mesh.bounds
        extents = bounds[1] - bounds[0]
        center = (bounds[0] + bounds[1]) / 2
        
        # Create a box
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(center)
        
        return box


def is_meshlib_available() -> bool:
    """Check if MeshLib is available."""
    return MESHLIB_AVAILABLE
