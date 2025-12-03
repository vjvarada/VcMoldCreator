"""
Mesh Repair Module

Provides mesh repair functionality similar to the frontend's meshRepairManifold.ts.

Repair strategies:
1. Vertex merging - merge duplicate/close vertices
2. Face repair - fix degenerate and duplicate faces
3. Hole filling - fill small holes in the mesh
4. Normal fixing - ensure consistent face winding
5. Convex hull fallback - if mesh cannot be repaired
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np
import trimesh

from core.mesh_analysis import MeshAnalyzer, MeshDiagnostics


class RepairMethod(Enum):
    """Repair methods used during mesh repair."""
    NONE = "none"
    VERTEX_MERGE = "vertex merge"
    FACE_CLEANUP = "face cleanup"
    HOLE_FILLING = "hole filling"
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


class MeshRepairer:
    """
    Mesh repair using trimesh operations.
    
    Attempts to create a valid, watertight mesh from potentially broken input.
    Similar to the Manifold-based repair in the frontend.
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
    
    def repair(self, 
               merge_vertices: bool = True,
               remove_degenerate: bool = True,
               remove_duplicates: bool = True,
               fix_normals: bool = True,
               fill_holes: bool = True,
               use_convex_hull_fallback: bool = True) -> MeshRepairResult:
        """
        Repair the mesh using various strategies.
        
        Args:
            merge_vertices: Merge vertices that are very close together
            remove_degenerate: Remove zero-area faces
            remove_duplicates: Remove duplicate faces
            fix_normals: Fix inconsistent face winding
            fill_holes: Attempt to fill holes
            use_convex_hull_fallback: Use convex hull if repair fails
            
        Returns:
            MeshRepairResult with the repaired mesh and diagnostics
        """
        self._repair_steps = []
        original_vertex_count = len(self.original_mesh.vertices)
        original_face_count = len(self.original_mesh.faces)
        
        was_repaired = False
        repair_methods_used = []
        
        # Step 1: Merge close vertices
        if merge_vertices:
            result = self._merge_vertices()
            if result:
                was_repaired = True
                repair_methods_used.append("vertex merge")
        
        # Step 2: Remove degenerate faces
        if remove_degenerate:
            result = self._remove_degenerate_faces()
            if result:
                was_repaired = True
                repair_methods_used.append("degenerate removal")
        
        # Step 3: Remove duplicate faces
        if remove_duplicates:
            result = self._remove_duplicate_faces()
            if result:
                was_repaired = True
                repair_methods_used.append("duplicate removal")
        
        # Step 4: Fix normals (consistent winding)
        if fix_normals:
            result = self._fix_normals()
            if result:
                was_repaired = True
                repair_methods_used.append("normal fix")
        
        # Step 5: Fill holes
        if fill_holes:
            result = self._fill_holes()
            if result:
                was_repaired = True
                repair_methods_used.append("hole filling")
        
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
    
    def _merge_vertices(self, tolerance: float = 1e-8) -> bool:
        """
        Merge vertices that are very close together.
        
        Returns:
            True if any vertices were merged
        """
        original_count = len(self.mesh.vertices)
        
        try:
            # Use trimesh's merge_vertices
            self.mesh.merge_vertices(merge_tex=True, merge_norm=True)
            
            new_count = len(self.mesh.vertices)
            if new_count < original_count:
                merged = original_count - new_count
                self._repair_steps.append(f"Merged {merged} duplicate vertices")
                return True
        except Exception as e:
            self._repair_steps.append(f"Vertex merge failed: {e}")
        
        return False
    
    def _remove_degenerate_faces(self) -> bool:
        """
        Remove faces with zero or near-zero area.
        
        Returns:
            True if any faces were removed
        """
        original_count = len(self.mesh.faces)
        
        try:
            # Remove degenerate faces (zero area) using nondegenerate mask
            # In newer trimesh, use nondegenerate faces property
            nondegenerate = self.mesh.nondegenerate_faces()
            if nondegenerate.sum() < original_count:
                self.mesh.update_faces(nondegenerate)
                removed = original_count - len(self.mesh.faces)
                self._repair_steps.append(f"Removed {removed} degenerate faces")
                return True
        except Exception as e:
            self._repair_steps.append(f"Degenerate face removal failed: {e}")
        
        return False
    
    def _remove_duplicate_faces(self) -> bool:
        """
        Remove duplicate faces.
        
        Returns:
            True if any faces were removed
        """
        original_count = len(self.mesh.faces)
        
        try:
            # Remove duplicate faces using unique_faces
            unique_mask = self.mesh.unique_faces()
            if unique_mask.sum() < original_count:
                self.mesh.update_faces(unique_mask)
                removed = original_count - len(self.mesh.faces)
                self._repair_steps.append(f"Removed {removed} duplicate faces")
                return True
        except Exception as e:
            self._repair_steps.append(f"Duplicate face removal failed: {e}")
        
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
    
    def _fill_holes(self) -> bool:
        """
        Fill holes in the mesh.
        
        Returns:
            True if any holes were filled
        """
        try:
            # Check if there are holes (boundary edges)
            if not self.mesh.is_watertight:
                # trimesh's fill_holes method
                filled = self.mesh.fill_holes()
                
                if self.mesh.is_watertight:
                    self._repair_steps.append("Filled holes to make mesh watertight")
                    return True
                else:
                    self._repair_steps.append("Attempted hole filling but mesh still has holes")
        except Exception as e:
            self._repair_steps.append(f"Hole filling failed: {e}")
        
        return False
    
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


def repair_mesh(mesh: trimesh.Trimesh, **kwargs) -> MeshRepairResult:
    """
    Convenience function to repair a mesh.
    
    Args:
        mesh: The trimesh mesh to repair
        **kwargs: Additional arguments passed to MeshRepairer.repair()
        
    Returns:
        MeshRepairResult with the repaired mesh and diagnostics
    """
    repairer = MeshRepairer(mesh)
    return repairer.repair(**kwargs)
