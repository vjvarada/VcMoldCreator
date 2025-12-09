"""
Mesh Analysis Module

Provides comprehensive mesh diagnostics and analysis similar to the
frontend's meshRepairManifold.ts functionality.

Analyzes:
- Vertex and face counts
- Manifold status
- Genus (topological holes/handles)
- Volume and surface area
- Bounding box dimensions
- Mesh quality issues
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import trimesh


@dataclass
class BoundingBox:
    """3D bounding box representation."""
    min_point: np.ndarray  # [x, y, z]
    max_point: np.ndarray  # [x, y, z]
    
    @property
    def size(self) -> np.ndarray:
        """Get the size (dimensions) of the bounding box."""
        return self.max_point - self.min_point
    
    @property
    def center(self) -> np.ndarray:
        """Get the center point of the bounding box."""
        return (self.min_point + self.max_point) / 2
    
    @property
    def diagonal(self) -> float:
        """Get the diagonal length of the bounding box."""
        return float(np.linalg.norm(self.size))
    
    def __str__(self) -> str:
        size = self.size
        return f"Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"


@dataclass
class MeshDiagnostics:
    """
    Comprehensive mesh diagnostics.
    
    Mirrors the MeshDiagnostics interface from the frontend.
    """
    vertex_count: int
    face_count: int
    is_manifold: bool
    is_watertight: bool
    genus: int
    volume: float
    surface_area: float
    bounding_box: BoundingBox
    euler_number: int
    issues: List[str] = field(default_factory=list)
    
    # Additional metrics
    edge_count: int = 0
    has_degenerate_faces: bool = False
    has_duplicate_faces: bool = False
    has_flipped_normals: bool = False
    
    def format(self) -> str:
        """Format diagnostics for display (similar to formatDiagnostics in frontend)."""
        lines = [
            f"Vertices: {self.vertex_count:,}",
            f"Faces: {self.face_count:,}",
            f"Edges: {self.edge_count:,}",
            "",
            f"Status: {'✓ Valid Manifold' if self.is_manifold else '✗ Non-Manifold'}",
            f"Watertight: {'✓ Yes' if self.is_watertight else '✗ No'}",
        ]
        
        if self.genus >= 0:
            genus_desc = ""
            if self.genus == 0:
                genus_desc = "(sphere-like)"
            elif self.genus == 1:
                genus_desc = "(torus-like)"
            else:
                genus_desc = f"({self.genus} holes/handles)"
            lines.append(f"Genus: {self.genus} {genus_desc}")
        
        lines.append(f"Euler Number: {self.euler_number}")
        lines.append("")
        
        if self.volume > 0:
            lines.append(f"Volume: {self.volume:,.2f}")
        lines.append(f"Surface Area: {self.surface_area:,.2f}")
        
        lines.append("")
        lines.append(f"Bounding Box: {self.bounding_box}")
        
        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  • {issue}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert diagnostics to dictionary."""
        return {
            'vertex_count': self.vertex_count,
            'face_count': self.face_count,
            'edge_count': self.edge_count,
            'is_manifold': self.is_manifold,
            'is_watertight': self.is_watertight,
            'genus': self.genus,
            'euler_number': self.euler_number,
            'volume': self.volume,
            'surface_area': self.surface_area,
            'bounding_box': {
                'min': self.bounding_box.min_point.tolist(),
                'max': self.bounding_box.max_point.tolist(),
                'size': self.bounding_box.size.tolist(),
            },
            'issues': self.issues,
        }


class MeshAnalyzer:
    """
    Mesh analysis and diagnostics.
    
    Provides comprehensive analysis of mesh properties, quality, and issues.
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize analyzer with a mesh.
        
        Args:
            mesh: The trimesh mesh to analyze
        """
        self.mesh = mesh
        self._diagnostics: Optional[MeshDiagnostics] = None
    
    def analyze(self) -> MeshDiagnostics:
        """
        Perform comprehensive mesh analysis.
        
        Returns:
            MeshDiagnostics containing all analysis results
        """
        mesh = self.mesh
        issues: List[str] = []
        
        # Basic counts
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.faces)
        edge_count = len(mesh.edges_unique) if hasattr(mesh, 'edges_unique') else 0
        
        # Bounding box
        bounds = mesh.bounds
        bounding_box = BoundingBox(
            min_point=bounds[0].copy(),
            max_point=bounds[1].copy()
        )
        
        # Manifold and watertight checks
        is_manifold = mesh.is_watertight  # In trimesh, watertight implies manifold
        is_watertight = mesh.is_watertight
        
        if not is_manifold:
            issues.append("Mesh is not manifold (has boundary edges or non-manifold edges)")
        
        if not is_watertight:
            issues.append("Mesh is not watertight (has holes)")
        
        # Euler characteristic and genus
        # Euler characteristic: V - E + F = 2 - 2g (for closed surfaces)
        # where g is the genus
        euler_number = mesh.euler_number
        
        # For a closed manifold: genus = (2 - euler) / 2
        if is_watertight:
            genus = (2 - euler_number) // 2
        else:
            genus = -1  # Cannot determine for non-closed meshes
            issues.append("Genus cannot be determined for non-watertight mesh")
        
        # Volume and surface area
        if is_watertight:
            volume = float(mesh.volume) if mesh.volume > 0 else 0.0
        else:
            volume = 0.0
            issues.append("Volume cannot be computed for non-watertight mesh")
        
        surface_area = float(mesh.area)
        
        # Check for degenerate faces (zero area)
        face_areas = mesh.area_faces
        degenerate_count = np.sum(face_areas < 1e-10)
        has_degenerate_faces = degenerate_count > 0
        if has_degenerate_faces:
            issues.append(f"Found {degenerate_count} degenerate (zero-area) faces")
        
        # Check for duplicate faces
        has_duplicate_faces = False
        try:
            # trimesh doesn't have a direct method, but we can check face count before/after
            unique_faces = np.unique(np.sort(mesh.faces, axis=1), axis=0)
            if len(unique_faces) < len(mesh.faces):
                has_duplicate_faces = True
                dup_count = len(mesh.faces) - len(unique_faces)
                issues.append(f"Found {dup_count} duplicate faces")
        except Exception:
            pass
        
        # Check for inverted/flipped normals
        has_flipped_normals = False
        if is_watertight and volume < 0:
            has_flipped_normals = True
            issues.append("Mesh has inverted normals (inside-out)")
        
        # Check for non-manifold vertices
        # A vertex is non-manifold if it's shared by faces that don't form a fan
        # trimesh handles this internally, but we can check edge manifoldness
        
        self._diagnostics = MeshDiagnostics(
            vertex_count=vertex_count,
            face_count=face_count,
            edge_count=edge_count,
            is_manifold=is_manifold,
            is_watertight=is_watertight,
            genus=genus,
            euler_number=euler_number,
            volume=abs(volume),  # Use absolute value in case normals are flipped
            surface_area=surface_area,
            bounding_box=bounding_box,
            has_degenerate_faces=has_degenerate_faces,
            has_duplicate_faces=has_duplicate_faces,
            has_flipped_normals=has_flipped_normals,
            issues=issues
        )
        
        return self._diagnostics
    
    @property
    def diagnostics(self) -> Optional[MeshDiagnostics]:
        """Get cached diagnostics (call analyze() first)."""
        return self._diagnostics
    
    def get_mesh_quality_score(self) -> float:
        """
        Compute an overall mesh quality score (0-100).
        
        Returns:
            Quality score from 0 (poor) to 100 (excellent)
        """
        if self._diagnostics is None:
            self.analyze()
        
        diag = self._diagnostics
        score = 100.0
        
        # Penalize for non-manifold
        if not diag.is_manifold:
            score -= 30
        
        # Penalize for not watertight
        if not diag.is_watertight:
            score -= 20
        
        # Penalize for degenerate faces
        if diag.has_degenerate_faces:
            score -= 10
        
        # Penalize for duplicate faces
        if diag.has_duplicate_faces:
            score -= 10
        
        # Penalize for flipped normals
        if diag.has_flipped_normals:
            score -= 10
        
        # Penalize for high genus (complex topology)
        if diag.genus > 0:
            score -= min(5 * diag.genus, 20)
        
        return max(0.0, score)


def analyze_mesh(mesh: trimesh.Trimesh) -> MeshDiagnostics:
    """
    Convenience function to analyze a mesh.
    
    Args:
        mesh: The trimesh mesh to analyze
        
    Returns:
        MeshDiagnostics containing all analysis results
    """
    analyzer = MeshAnalyzer(mesh)
    return analyzer.analyze()
