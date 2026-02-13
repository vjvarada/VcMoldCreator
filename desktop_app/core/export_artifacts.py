"""
Export Artifacts Module

Exports the final fabrication artifacts (hard shells, metamolds, pouring plug)
to a .tmp folder for downstream 3D printing/fabrication use.

The .tmp folder is untracked by git and provides a clean export location
for the 5 STL files plus a metadata summary text file.

Artifacts exported:
    1. hard_shell_half_1.stl - Upper hard shell half (with through-holes if drilled)
    2. hard_shell_half_2.stl - Lower hard shell half (with through-holes if drilled)
    3. metamold_half_1.stl   - Upper metamold half (with resin channels if created)
    4. metamold_half_2.stl   - Lower metamold half (with resin channels if created)
    5. pouring_plug.stl      - Resin pouring plug

Metadata file:
    metadata.txt - Volume estimates for silicone and resin
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExportArtifact:
    """A single artifact to export."""
    filename: str
    mesh: Optional[trimesh.Trimesh]
    description: str


@dataclass
class VolumeMetadata:
    """Volume estimates for fabrication materials."""
    # Hull volume (boundary volume of the mold)
    hull_volume_mm3: float = 0.0
    # Part mesh volume
    part_volume_mm3: float = 0.0
    # Pouring plug volume
    plug_volume_mm3: float = 0.0
    # Silicone needed = hull_volume - part_volume
    silicone_volume_mm3: float = 0.0
    # Resin needed = part_volume + plug_volume
    resin_volume_mm3: float = 0.0


@dataclass
class ExportResult:
    """Result of the export operation."""
    success: bool = False
    export_dir: str = ""
    exported_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    metadata_file: str = ""
    volume_metadata: Optional[VolumeMetadata] = None
    error_message: str = ""


# ============================================================================
# VOLUME COMPUTATION
# ============================================================================

def compute_volume_metadata(
    hull_mesh: Optional[trimesh.Trimesh],
    part_mesh: Optional[trimesh.Trimesh],
    plug_mesh: Optional[trimesh.Trimesh],
) -> VolumeMetadata:
    """Compute volume estimates for silicone and resin.

    Args:
        hull_mesh: The inflated hull mesh (bounding volume).
        part_mesh: The input part mesh.
        plug_mesh: The resin pouring plug mesh.

    Returns:
        VolumeMetadata with computed volumes.

    Notes:
        Silicone volume = hull_volume - part_volume
        Resin volume = part_volume + plug_volume
        All volumes in mm³ (assuming mesh units are mm).
    """
    metadata = VolumeMetadata()

    if hull_mesh is not None:
        try:
            if hull_mesh.is_watertight:
                metadata.hull_volume_mm3 = abs(float(hull_mesh.volume))
            else:
                # Use convex hull volume as approximation
                metadata.hull_volume_mm3 = abs(float(hull_mesh.convex_hull.volume))
                logger.warning("Hull mesh is not watertight; using convex hull volume as approximation")
        except Exception as e:
            logger.warning("Could not compute hull volume: %s", e)

    if part_mesh is not None:
        try:
            if part_mesh.is_watertight:
                metadata.part_volume_mm3 = abs(float(part_mesh.volume))
            else:
                metadata.part_volume_mm3 = abs(float(part_mesh.convex_hull.volume))
                logger.warning("Part mesh is not watertight; using convex hull volume as approximation")
        except Exception as e:
            logger.warning("Could not compute part volume: %s", e)

    if plug_mesh is not None:
        try:
            if plug_mesh.is_watertight:
                metadata.plug_volume_mm3 = abs(float(plug_mesh.volume))
            else:
                metadata.plug_volume_mm3 = abs(float(plug_mesh.convex_hull.volume))
                logger.warning("Plug mesh is not watertight; using convex hull volume as approximation")
        except Exception as e:
            logger.warning("Could not compute plug volume: %s", e)

    # Derived volumes
    metadata.silicone_volume_mm3 = max(
        metadata.hull_volume_mm3 - metadata.part_volume_mm3, 0.0
    )
    metadata.resin_volume_mm3 = metadata.part_volume_mm3 + metadata.plug_volume_mm3

    return metadata


# ============================================================================
# METADATA FILE GENERATION
# ============================================================================

def _format_volume(volume_mm3: float) -> str:
    """Format a volume in mm³ with mL and cm³ conversions."""
    volume_cm3 = volume_mm3 / 1000.0  # 1 cm³ = 1000 mm³
    volume_ml = volume_cm3             # 1 mL = 1 cm³
    return f"{volume_mm3:,.1f} mm³  ({volume_cm3:,.2f} cm³ / {volume_ml:,.2f} mL)"


def generate_metadata_text(
    volume_metadata: VolumeMetadata,
    model_name: str = "Unknown",
    exported_files: Optional[List[str]] = None,
) -> str:
    """Generate a metadata text file content.

    Args:
        volume_metadata: Computed volume data.
        model_name: Name of the input model.
        exported_files: List of exported file names.

    Returns:
        Formatted metadata string.
    """
    lines = [
        "=" * 60,
        "VcMoldCreator - Export Metadata",
        "=" * 60,
        "",
        f"Model:      {model_name}",
        f"Exported:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 60,
        "VOLUME ESTIMATES",
        "-" * 60,
        "",
        f"Hull (bounding) volume:  {_format_volume(volume_metadata.hull_volume_mm3)}",
        f"Part mesh volume:        {_format_volume(volume_metadata.part_volume_mm3)}",
        f"Pouring plug volume:     {_format_volume(volume_metadata.plug_volume_mm3)}",
        "",
        "-" * 60,
        "MATERIAL REQUIREMENTS",
        "-" * 60,
        "",
        f"Silicone needed:  {_format_volume(volume_metadata.silicone_volume_mm3)}",
        "  (= hull volume - part volume)",
        "",
        f"Resin needed:     {_format_volume(volume_metadata.resin_volume_mm3)}",
        "  (= part volume + plug volume)",
        "",
    ]

    if exported_files:
        lines.extend([
            "-" * 60,
            "EXPORTED FILES",
            "-" * 60,
            "",
        ])
        for f in exported_files:
            lines.append(f"  • {f}")
        lines.append("")

    lines.extend([
        "-" * 60,
        "NOTES",
        "-" * 60,
        "",
        "• Hard shell halves are 3D-printed rigid containers.",
        "• Metamold halves are 3D-printed containers for casting silicone.",
        "• Pouring plug seals the resin inlet during silicone casting.",
        "• Silicone volume is approximate (hull minus part cavity).",
        "• Add ~10-15% extra silicone to account for spillage and trim.",
        "• Resin volume includes the plug; add ~5% for overflow.",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


# ============================================================================
# EXPORT FUNCTION
# ============================================================================

def export_artifacts(
    export_dir: str,
    shell_half_1: Optional[trimesh.Trimesh],
    shell_half_2: Optional[trimesh.Trimesh],
    metamold_half_1: Optional[trimesh.Trimesh],
    metamold_half_2: Optional[trimesh.Trimesh],
    plug_mesh: Optional[trimesh.Trimesh],
    hull_mesh: Optional[trimesh.Trimesh] = None,
    part_mesh: Optional[trimesh.Trimesh] = None,
    model_name: str = "Unknown",
) -> ExportResult:
    """Export all fabrication artifacts to the specified directory.

    Exports 5 STL files (hard shells, metamolds, pouring plug) and a
    metadata.txt with volume estimates.

    Args:
        export_dir: Absolute path to the export directory.
        shell_half_1: Upper hard shell half mesh.
        shell_half_2: Lower hard shell half mesh.
        metamold_half_1: Upper metamold half mesh (with channels).
        metamold_half_2: Lower metamold half mesh (with channels).
        plug_mesh: Resin pouring plug mesh.
        hull_mesh: Inflated hull mesh (for volume computation).
        part_mesh: Input part mesh (for volume computation).
        model_name: Name of the input model for metadata.

    Returns:
        ExportResult with status and file listing.
    """
    result = ExportResult()

    try:
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        result.export_dir = export_dir
        logger.info("Export directory: %s", export_dir)

        # Define artifacts to export
        artifacts = [
            ExportArtifact("hard_shell_half_1.stl", shell_half_1, "Upper hard shell half"),
            ExportArtifact("hard_shell_half_2.stl", shell_half_2, "Lower hard shell half"),
            ExportArtifact("metamold_half_1.stl", metamold_half_1, "Upper metamold half"),
            ExportArtifact("metamold_half_2.stl", metamold_half_2, "Lower metamold half"),
            ExportArtifact("pouring_plug.stl", plug_mesh, "Resin pouring plug"),
        ]

        # Export each artifact
        for artifact in artifacts:
            if artifact.mesh is not None:
                filepath = os.path.join(export_dir, artifact.filename)
                artifact.mesh.export(filepath, file_type='stl')
                result.exported_files.append(artifact.filename)
                logger.info("Exported %s: %s (%d verts, %d faces)",
                           artifact.description, artifact.filename,
                           len(artifact.mesh.vertices), len(artifact.mesh.faces))
            else:
                result.skipped_files.append(artifact.filename)
                logger.warning("Skipped %s: mesh not available", artifact.description)

        # Compute volume metadata
        volume_metadata = compute_volume_metadata(hull_mesh, part_mesh, plug_mesh)
        result.volume_metadata = volume_metadata

        # Generate and write metadata file
        metadata_text = generate_metadata_text(
            volume_metadata,
            model_name=model_name,
            exported_files=result.exported_files,
        )
        metadata_path = os.path.join(export_dir, "metadata.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(metadata_text)
        result.metadata_file = "metadata.txt"
        logger.info("Metadata written to %s", metadata_path)

        result.success = len(result.exported_files) > 0
        if result.skipped_files:
            logger.warning("Skipped artifacts: %s", ", ".join(result.skipped_files))

    except Exception as e:
        logger.exception("Export failed: %s", e)
        result.error_message = str(e)
        result.success = False

    return result
