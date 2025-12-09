"""
Mesh Decimation Module

Provides mesh decimation functionality to reduce triangle count for meshes
that have too many triangles for efficient downstream processing.

Uses Quadric Edge Collapse Decimation (simplify_quadric_decimation) from trimesh.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging
import trimesh

logger = logging.getLogger(__name__)


class DecimationQuality(Enum):
    """Preset quality levels for decimation."""
    HIGH = "high"          # 75% of original triangles
    MEDIUM = "medium"      # 50% of original triangles
    LOW = "low"            # 25% of original triangles
    CUSTOM = "custom"      # User-specified target


# Thresholds for determining if decimation is needed
# These are based on typical downstream processing requirements
TRIANGLE_COUNT_THRESHOLDS = {
    'recommended': 500_000,    # Recommended max for smooth real-time processing
    'warning': 750_000,        # Warning threshold - may slow down operations
    'critical': 1_000_000,     # Critical threshold - likely to cause issues
}


@dataclass
class DecimationResult:
    """Result of mesh decimation operation."""
    mesh: trimesh.Trimesh
    was_decimated: bool
    original_triangle_count: int
    final_triangle_count: int
    reduction_percentage: float
    target_triangle_count: int
    quality_setting: str
    error_message: Optional[str] = None


def get_decimation_recommendation(triangle_count: int) -> dict:
    """
    Get a recommendation for whether decimation is needed.
    
    Args:
        triangle_count: Number of triangles in the mesh
        
    Returns:
        Dictionary with recommendation details
    """
    if triangle_count <= TRIANGLE_COUNT_THRESHOLDS['recommended']:
        return {
            'needs_decimation': False,
            'severity': 'none',
            'message': f'Mesh has {triangle_count:,} triangles - optimal for processing.',
            'suggested_target': triangle_count
        }
    elif triangle_count <= TRIANGLE_COUNT_THRESHOLDS['warning']:
        return {
            'needs_decimation': True,
            'severity': 'warning',
            'message': f'Mesh has {triangle_count:,} triangles. Consider decimating to ~{TRIANGLE_COUNT_THRESHOLDS["recommended"]:,} for faster processing.',
            'suggested_target': TRIANGLE_COUNT_THRESHOLDS['recommended']
        }
    elif triangle_count <= TRIANGLE_COUNT_THRESHOLDS['critical']:
        return {
            'needs_decimation': True,
            'severity': 'high',
            'message': f'Mesh has {triangle_count:,} triangles - may cause slow processing. Decimation recommended.',
            'suggested_target': TRIANGLE_COUNT_THRESHOLDS['recommended']
        }
    else:
        return {
            'needs_decimation': True,
            'severity': 'critical',
            'message': f'Mesh has {triangle_count:,} triangles - will likely cause processing issues. Decimation strongly recommended.',
            'suggested_target': TRIANGLE_COUNT_THRESHOLDS['recommended']
        }


class MeshDecimator:
    """
    Mesh decimation using trimesh's quadric edge collapse.
    
    Reduces triangle count while attempting to preserve mesh quality.
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize decimator with a mesh.
        
        Args:
            mesh: The trimesh mesh to decimate
        """
        self.original_mesh = mesh
        self.mesh = mesh.copy()
    
    def get_triangle_count(self) -> int:
        """Get the current triangle count."""
        return len(self.mesh.faces)
    
    def get_recommendation(self) -> dict:
        """Get decimation recommendation for this mesh."""
        return get_decimation_recommendation(self.get_triangle_count())
    
    def decimate(self, 
                 quality: DecimationQuality = DecimationQuality.MEDIUM,
                 target_triangles: Optional[int] = None) -> DecimationResult:
        """
        Decimate the mesh to reduce triangle count.
        
        Args:
            quality: Preset quality level (ignored if target_triangles is set)
            target_triangles: Specific target triangle count (overrides quality)
            
        Returns:
            DecimationResult with the decimated mesh and statistics
        """
        original_count = len(self.original_mesh.faces)
        
        # Determine target triangle count
        if target_triangles is not None:
            target = max(100, min(target_triangles, original_count))
            quality_str = f"custom ({target:,} target)"
        else:
            ratios = {
                DecimationQuality.HIGH: 0.75,
                DecimationQuality.MEDIUM: 0.50,
                DecimationQuality.LOW: 0.25,
            }
            ratio = ratios.get(quality, 0.50)
            target = max(100, int(original_count * ratio))
            quality_str = quality.value
        
        logger.info(f"Decimating mesh from {original_count:,} to ~{target:,} triangles ({quality_str})")
        
        try:
            # Perform decimation using quadric edge collapse
            # Use face_count parameter to specify target number of faces
            decimated = self.original_mesh.simplify_quadric_decimation(
                face_count=target,
                aggression=5  # Balance between speed and quality (1-10)
            )
            
            final_count = len(decimated.faces)
            reduction = ((original_count - final_count) / original_count) * 100
            
            logger.info(f"Decimation complete: {original_count:,} -> {final_count:,} triangles ({reduction:.1f}% reduction)")
            
            return DecimationResult(
                mesh=decimated,
                was_decimated=True,
                original_triangle_count=original_count,
                final_triangle_count=final_count,
                reduction_percentage=reduction,
                target_triangle_count=target,
                quality_setting=quality_str
            )
            
        except Exception as e:
            error_msg = f"Decimation failed: {str(e)}"
            logger.error(error_msg)
            
            return DecimationResult(
                mesh=self.original_mesh.copy(),
                was_decimated=False,
                original_triangle_count=original_count,
                final_triangle_count=original_count,
                reduction_percentage=0,
                target_triangle_count=target,
                quality_setting=quality_str,
                error_message=error_msg
            )
    
    def preview_decimation(self, target_triangles: int) -> dict:
        """
        Preview what decimation would do without actually performing it.
        
        Args:
            target_triangles: Target triangle count
            
        Returns:
            Dictionary with preview information
        """
        original_count = len(self.original_mesh.faces)
        target = max(100, min(target_triangles, original_count))
        expected_reduction = ((original_count - target) / original_count) * 100
        
        return {
            'original_triangles': original_count,
            'target_triangles': target,
            'expected_reduction_percentage': expected_reduction,
            'original_vertices': len(self.original_mesh.vertices),
            'estimated_vertices': int(len(self.original_mesh.vertices) * (target / original_count))
        }
