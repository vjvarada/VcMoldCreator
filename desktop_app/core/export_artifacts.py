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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import trimesh
import trimesh.repair as _trimesh_repair

logger = logging.getLogger(__name__)


def _repair_mesh_for_export(mesh: Optional[trimesh.Trimesh], label: str) -> Optional[trimesh.Trimesh]:
    """Attempt to repair a mesh before export so the STL is as watertight as possible.

    Uses meshlib as the primary engine (most robust for complex topology),
    with trimesh as a fallback.

    Args:
        mesh: Input trimesh, or None.
        label: Human-readable name for log messages.

    Returns:
        Repaired mesh (or original if it was already watertight / None).
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh

    if mesh.is_watertight:
        logger.info("Export mesh '%s': already watertight (%d verts, %d faces)",
                    label, len(mesh.vertices), len(mesh.faces))
        return mesh

    def _count_open_edges(m: trimesh.Trimesh) -> int:
        ec: dict = {}
        for face in m.faces:
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                ek = (min(v0, v1), max(v0, v1))
                ec[ek] = ec.get(ek, 0) + 1
        return sum(1 for c in ec.values() if c == 1)

    open_before = _count_open_edges(mesh)
    logger.warning(
        "Export mesh '%s' is NOT watertight before export: %d open boundary edges "
        "(%d verts, %d faces). Attempting repair...",
        label, open_before, len(mesh.vertices), len(mesh.faces)
    )

    # ------------------------------------------------------------------
    # meshlib primary repair
    # ------------------------------------------------------------------
    try:
        import numpy as _np
        import meshlib.mrmeshnumpy as _mrn
        import meshlib.mrmeshpy as _mr

        _eec: dict = {}
        for _ef in mesh.faces:
            for _ei in range(3):
                _eva, _evb = int(_ef[_ei]), int(_ef[(_ei + 1) % 3])
                _eek = (min(_eva, _evb), max(_eva, _evb))
                _eec[_eek] = _eec.get(_eek, 0) + 1
        _e_closed_nm = (sum(1 for _ec_v in _eec.values() if _ec_v == 1) == 0)
        mesh_mr = _mrn.meshFromFacesVerts(
            mesh.faces.astype(_np.int32),
            mesh.vertices.astype(_np.float32),
        )
        if not _e_closed_nm:
            _mr.fixMeshDegeneracies(mesh_mr, _mr.FixMeshDegeneraciesParams())
            for loop in _mr.findRightBoundary(mesh_mr.topology, None):
                _mr.fillHoleNicely(mesh_mr, loop[0], _mr.FillHoleNicelySettings())

        out_verts = _mrn.getNumpyVerts(mesh_mr)
        out_faces = _mrn.getNumpyFaces(mesh_mr.topology)
        result = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
        result.fix_normals()

        if result.is_watertight:
            logger.info(
                "Export mesh '%s': repaired via meshlib → watertight (%d verts, %d faces)",
                label, len(result.vertices), len(result.faces)
            )
            return result

        open_after_ml = _count_open_edges(result)
        if open_after_ml == 0:
            # Non-manifold topology (0 open edges, no holes to fill).
            # Trimesh process=True would re-merge the vertices that meshlib just split,
            # recreating the non-manifold edges and causing geometry explosion.  Return as-is.
            logger.warning(
                "Export mesh '%s': 0 open edges but not watertight (non-manifold topology). "
                "Skipping trimesh fallback to avoid geometry explosion.",
                label,
            )
            return result
        logger.warning(
            "Export mesh '%s': meshlib repair left %d open edges. Falling back to trimesh.",
            label, open_after_ml
        )
        mesh = result
    except Exception as exc:
        logger.warning("Export mesh '%s': meshlib repair failed (%s). Falling back to trimesh.",
                       label, exc)

    # ------------------------------------------------------------------
    # trimesh fallback (only reached when there are real open boundary edges)
    # ------------------------------------------------------------------
    m = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=True,
    )
    _trimesh_repair.fix_normals(m, multibody=True)
    _trimesh_repair.fill_holes(m)

    if m.is_watertight:
        logger.info(
            "Export mesh '%s': repaired via trimesh fallback → watertight (%d verts, %d faces)",
            label, len(m.vertices), len(m.faces)
        )
        return m

    open_after = _count_open_edges(m)
    logger.warning(
        "Export mesh '%s': still NOT watertight after repair (%d open boundary edges remain). "
        "STL will have open faces.",
        label, open_after
    )
    return m


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

def _safe_mesh_volume(mesh: Optional[trimesh.Trimesh], label: str) -> float:
    """Compute the volume of a mesh, falling back to convex hull if not watertight.

    Args:
        mesh: The mesh to compute volume for, or None.
        label: Human-readable label for log messages (e.g. 'hull', 'part').

    Returns:
        Absolute volume in mesh units (mm³), or 0.0 if mesh is None or fails.
    """
    if mesh is None:
        return 0.0
    try:
        if mesh.is_watertight:
            return abs(float(mesh.volume))
        logger.warning("%s mesh is not watertight; using convex hull volume as approximation", label)
        return abs(float(mesh.convex_hull.volume))
    except Exception as e:
        logger.warning("Could not compute %s volume: %s", label, e)
        return 0.0


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
    metadata.hull_volume_mm3 = _safe_mesh_volume(hull_mesh, "Hull")
    metadata.part_volume_mm3 = _safe_mesh_volume(part_mesh, "Part")
    metadata.plug_volume_mm3 = _safe_mesh_volume(plug_mesh, "Plug")

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
    clamp_mesh: Optional[trimesh.Trimesh] = None,
) -> ExportResult:
    """Export all fabrication artifacts to the specified directory.

    Exports STL files (hard shells, metamolds, pouring plug, clamp) and a
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
        clamp_mesh: Metamold clamp mesh.

    Returns:
        ExportResult with status and file listing.
    """
    result = ExportResult()

    try:
        # Create export directory
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        result.export_dir = export_dir
        logger.info("Export directory: %s", export_dir)

        # Define artifacts to export
        artifacts = [
            ExportArtifact("hard_shell_half_1.stl", shell_half_1, "Upper hard shell half"),
            ExportArtifact("hard_shell_half_2.stl", shell_half_2, "Lower hard shell half"),
            ExportArtifact("metamold_half_1.stl", metamold_half_1, "Upper metamold half"),
            ExportArtifact("metamold_half_2.stl", metamold_half_2, "Lower metamold half"),
            ExportArtifact("pouring_plug.stl", plug_mesh, "Resin pouring plug"),
            ExportArtifact("metamold_clamp.stl", clamp_mesh, "Metamold clamp"),
        ]

        # Export each artifact
        for artifact in artifacts:
            if artifact.mesh is not None:
                filepath = export_path / artifact.filename
                # Repair before export to maximise watertightness
                repaired = _repair_mesh_for_export(artifact.mesh, artifact.filename)
                repaired.export(str(filepath), file_type='stl')
                result.exported_files.append(artifact.filename)
                logger.info("Exported %s: %s (%d verts, %d faces, watertight=%s)",
                           artifact.description, artifact.filename,
                           len(repaired.vertices), len(repaired.faces),
                           repaired.is_watertight)
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
        metadata_path = export_path / "metadata.txt"
        metadata_path.write_text(metadata_text, encoding='utf-8')
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
