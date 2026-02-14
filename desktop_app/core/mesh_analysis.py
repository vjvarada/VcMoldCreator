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
- Height fields for direction analysis
- Vertex/face adjacency structures

Design Principles:
- Immutable data structures for diagnostics
- Clear separation between analysis and utility functions
- Consistent type hints and error handling
- Efficient adjacency computation using half-edges
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import trimesh


logger = logging.getLogger(__name__)

# Constants for analysis thresholds
DEGENERATE_FACE_AREA_THRESHOLD = 1e-10
QUALITY_PENALTY_NON_MANIFOLD = 30
QUALITY_PENALTY_NOT_WATERTIGHT = 20
QUALITY_PENALTY_DEGENERATE = 10
QUALITY_PENALTY_DUPLICATE = 10
QUALITY_PENALTY_FLIPPED = 10
QUALITY_PENALTY_PER_GENUS = 5
QUALITY_MAX_GENUS_PENALTY = 20


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable 3D bounding box representation.
    
    Attributes:
        min_point: Minimum corner coordinates [x, y, z]
        max_point: Maximum corner coordinates [x, y, z]
    """
    min_point: np.ndarray
    max_point: np.ndarray
    
    def __post_init__(self):
        # Ensure arrays are immutable views
        object.__setattr__(self, 'min_point', np.asarray(self.min_point).copy())
        object.__setattr__(self, 'max_point', np.asarray(self.max_point).copy())
        self.min_point.flags.writeable = False
        self.max_point.flags.writeable = False
    
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
    
    def __hash__(self):
        return hash((tuple(self.min_point), tuple(self.max_point)))


@dataclass
class MeshDiagnostics:
    """
    Comprehensive mesh diagnostics.
    
    Mirrors the MeshDiagnostics interface from the frontend.
    
    Attributes:
        vertex_count: Number of vertices in the mesh
        face_count: Number of faces in the mesh
        is_manifold: Whether the mesh is manifold (no non-manifold edges/vertices)
        is_watertight: Whether the mesh is watertight (no boundary edges)
        genus: Topological genus (number of handles), -1 if not computable
        volume: Mesh volume (0 if not watertight)
        surface_area: Total surface area
        bounding_box: Axis-aligned bounding box
        euler_number: Euler characteristic (V - E + F)
        issues: List of identified issues
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
            genus_desc = self._get_genus_description()
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
    
    def _get_genus_description(self) -> str:
        """Get human-readable description of genus value."""
        if self.genus == 0:
            return "(sphere-like)"
        elif self.genus == 1:
            return "(torus-like)"
        else:
            return f"({self.genus} holes/handles)"
    
    def to_dict(self) -> dict:
        """Convert diagnostics to dictionary for serialization."""
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


# =============================================================================
# MESH ANALYZER
# =============================================================================

class MeshAnalyzer:
    """
    Mesh analysis and diagnostics.
    
    Provides comprehensive analysis of mesh properties, quality, and issues.
    Results are cached after first analysis.
    
    Example:
        >>> analyzer = MeshAnalyzer(mesh)
        >>> diagnostics = analyzer.analyze()
        >>> print(diagnostics.format())
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
        bounding_box = BoundingBox(
            min_point=mesh.bounds[0],
            max_point=mesh.bounds[1]
        )
        
        # Manifold and watertight checks
        is_manifold = mesh.is_watertight  # In trimesh, watertight implies manifold
        is_watertight = mesh.is_watertight
        
        if not is_manifold:
            issues.append("Mesh is not manifold (has boundary edges or non-manifold edges)")
        
        if not is_watertight:
            issues.append("Mesh is not watertight (has holes)")
        
        # Euler characteristic and genus
        euler_number = mesh.euler_number
        genus = self._compute_genus(euler_number, is_watertight, issues)
        
        # Volume and surface area
        volume = self._compute_volume(mesh, is_watertight, issues)
        surface_area = float(mesh.area)
        
        # Quality checks
        has_degenerate_faces = self._check_degenerate_faces(mesh, issues)
        has_duplicate_faces = self._check_duplicate_faces(mesh, issues)
        has_flipped_normals = self._check_flipped_normals(mesh, is_watertight, volume, issues)
        
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
    
    @staticmethod
    def _compute_genus(euler_number: int, is_watertight: bool, issues: List[str]) -> int:
        """Compute genus from Euler characteristic."""
        if is_watertight:
            # For a closed manifold: genus = (2 - euler) / 2
            return (2 - euler_number) // 2
        else:
            issues.append("Genus cannot be determined for non-watertight mesh")
            return -1
    
    @staticmethod
    def _compute_volume(mesh: trimesh.Trimesh, is_watertight: bool, issues: List[str]) -> float:
        """Compute mesh volume if possible."""
        if is_watertight:
            vol = float(mesh.volume) if mesh.volume > 0 else 0.0
            return vol
        else:
            issues.append("Volume cannot be computed for non-watertight mesh")
            return 0.0
    
    @staticmethod
    def _check_degenerate_faces(mesh: trimesh.Trimesh, issues: List[str]) -> bool:
        """Check for degenerate (zero-area) faces."""
        face_areas = mesh.area_faces
        degenerate_count = int(np.sum(face_areas < DEGENERATE_FACE_AREA_THRESHOLD))
        if degenerate_count > 0:
            issues.append(f"Found {degenerate_count} degenerate (zero-area) faces")
            return True
        return False
    
    @staticmethod
    def _check_duplicate_faces(mesh: trimesh.Trimesh, issues: List[str]) -> bool:
        """Check for duplicate faces."""
        try:
            unique_faces = np.unique(np.sort(mesh.faces, axis=1), axis=0)
            if len(unique_faces) < len(mesh.faces):
                dup_count = len(mesh.faces) - len(unique_faces)
                issues.append(f"Found {dup_count} duplicate faces")
                return True
        except Exception as e:
            logger.debug("Could not check for duplicate faces: %s", e)
        return False
    
    @staticmethod
    def _check_flipped_normals(
        mesh: trimesh.Trimesh, 
        is_watertight: bool, 
        volume: float, 
        issues: List[str]
    ) -> bool:
        """Check for inverted/flipped normals."""
        if is_watertight and volume < 0:
            issues.append("Mesh has inverted normals (inside-out)")
            return True
        return False
    
    @property
    def diagnostics(self) -> Optional[MeshDiagnostics]:
        """Get cached diagnostics (call analyze() first)."""
        return self._diagnostics
    
    def get_mesh_quality_score(self) -> float:
        """
        Compute an overall mesh quality score (0-100).
        
        Higher scores indicate better quality meshes suitable for
        downstream processing.
        
        Returns:
            Quality score from 0 (poor) to 100 (excellent)
        """
        if self._diagnostics is None:
            self.analyze()
        
        diag = self._diagnostics
        score = 100.0
        
        if not diag.is_manifold:
            score -= QUALITY_PENALTY_NON_MANIFOLD
        
        if not diag.is_watertight:
            score -= QUALITY_PENALTY_NOT_WATERTIGHT
        
        if diag.has_degenerate_faces:
            score -= QUALITY_PENALTY_DEGENERATE
        
        if diag.has_duplicate_faces:
            score -= QUALITY_PENALTY_DUPLICATE
        
        if diag.has_flipped_normals:
            score -= QUALITY_PENALTY_FLIPPED
        
        if diag.genus > 0:
            genus_penalty = min(QUALITY_PENALTY_PER_GENUS * diag.genus, QUALITY_MAX_GENUS_PENALTY)
            score -= genus_penalty
        
        return max(0.0, score)


# =============================================================================
# HEIGHT FIELD AND ADJACENCY FUNCTIONS
# =============================================================================

def compute_height_field(
    mesh: trimesh.Trimesh, 
    direction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute height function h(v) = dot(v, direction) for all vertices.
    
    The height function projects each vertex position onto a direction vector,
    creating a scalar field used for persistence homology analysis in
    pouring direction optimization (Paper Section 5.2).
    
    Args:
        mesh: Input trimesh mesh
        direction: Unit vector representing the "up" direction (pouring direction)
        
    Returns:
        Tuple containing:
        - vertex_heights: (N,) array of vertex height values
        - face_heights: (M,) array of face height values (average of vertex heights)
        
    Example:
        >>> heights, face_heights = compute_height_field(mesh, np.array([0, 0, 1]))
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        raise ValueError("Direction vector cannot be zero")
    direction = direction / norm
    
    # Compute vertex heights via dot product
    vertex_heights = mesh.vertices @ direction
    
    # Compute face heights as average of vertex heights
    face_vertex_heights = vertex_heights[mesh.faces]  # (M, 3)
    face_heights = np.mean(face_vertex_heights, axis=1)  # (M,)
    
    return vertex_heights, face_heights


def build_vertex_neighbors(mesh: trimesh.Trimesh) -> Dict[int, Set[int]]:
    """
    Build vertex adjacency map: vertex_index -> set of neighboring vertex indices.
    
    Two vertices are neighbors if they share an edge in the mesh.
    Uses edge connectivity from trimesh for efficient O(E) construction.
    
    Args:
        mesh: Input trimesh mesh
        
    Returns:
        Dictionary mapping each vertex index to a set of its neighbor vertex indices.
        Uses Set for O(1) membership testing.
        
    Example:
        >>> neighbors = build_vertex_neighbors(mesh)
        >>> adjacent_to_0 = neighbors.get(0, set())
    """
    neighbors: Dict[int, Set[int]] = defaultdict(set)
    
    for v0, v1 in mesh.edges:
        neighbors[v0].add(v1)
        neighbors[v1].add(v0)
    
    return dict(neighbors)


def build_face_neighbors(mesh: trimesh.Trimesh) -> Dict[int, List[int]]:
    """
    Build face adjacency map: face_index -> list of neighboring face indices.
    
    Two faces are neighbors if they share an edge. Uses trimesh's optimized
    face_adjacency computation which is O(F) using half-edge structure.
    
    Args:
        mesh: Input trimesh mesh
        
    Returns:
        Dictionary mapping each face index to a list of its neighbor face indices.
        Uses List to preserve ordering and allow duplicates if needed.
        
    Example:
        >>> face_neighbors = build_face_neighbors(mesh)
        >>> adjacent_faces = face_neighbors.get(0, [])
    """
    n_faces = len(mesh.faces)
    neighbors: Dict[int, List[int]] = {i: [] for i in range(n_faces)}
    
    # trimesh.face_adjacency is an (n, 2) array of pairs of adjacent face indices
    for a, b in mesh.face_adjacency:
        neighbors[a].append(b)
        neighbors[b].append(a)
    
    return neighbors

