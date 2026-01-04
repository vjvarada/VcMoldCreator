# Metamold Implementation Plan

## Overview

A **metamold** is a 3D printed mold used to cast silicone mold pieces. The metamold shape incorporates the geometric details of the parting surface plus any secondary membranes as solid shells. Once silicone is cast into the metamold, the resulting silicone pieces form the final multi-piece mold that can be assembled to cast the target object in resin.

> **Pipeline Summary**:
> ```
> 3D Print Metamold → Cast Silicone → Assemble Silicone Mold Pieces → Cast Resin → Final Part
> ```

### Prerequisites (Already Implemented)

| Component | Status | Location |
|-----------|--------|----------|
| Parting Surface | ✅ Implemented | `core/parting_surface.py` |
| Secondary Membranes | ✅ Implemented | Part of parting surface pipeline |
| Membrane Solidification | ✅ Implemented | `core/membrane_solidifier.py` |
| Pouring Direction Optimization | ✅ Implemented | `core/pouring_direction.py` |
| Mold Half Classification | ✅ Implemented | `core/mold_half_classification.py` |

---

## Design Goals

The metamold design must satisfy several requirements:

1. **Minimize Material Costs**: Reduce the amount of silicone required to fill each metamold
2. **Prevent Air Trapping**: Ensure casting directions avoid bubble formation
3. **Prevent Excessive Pressure**: Minimize metamold height to reduce hydrostatic pressure
4. **Practical Assembly**: Generate mold pieces that form a box shape for easy use

### Box-Shaped Mold Rationale

The final assembled silicone mold should form a **box** because:
- Easy to use for resin casting (flat base)
- Easy to cast the metamolds themselves (flat surfaces)
- Pouring direction should be orthogonal to one box face
- Simple registration and alignment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        METAMOLD CREATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Inputs     │ -> │  Direction   │ -> │  Box Layout  │ -> │  Metamold  │ │
│  │  P1, P2, PS  │    │  Selection   │    │  Generation  │    │  Details   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         v                   v                   v                   v        │
│  • Mold pieces        • Silicone dirs     • Bounding box      • Air vents   │
│  • Parting surface    • Resin dir         • Minimal height    • Sealing dam │
│  • Membranes          • Box-compatible    • Piece placement   • Reg. pegs   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 0: Membrane Solidification (Pre-processing)

Before membranes can be used as cutting geometry, they must be converted from flat surfaces to solid 3D volumes. This is implemented in `core/membrane_solidifier.py`.

#### Why Solidification is Needed

Secondary membranes (and the parting surface) start as **flat surface meshes**. For CSG operations (splitting the mold cavity), we need **solid volumes**:

```
┌─────────────────────────────────────────────────────────────────────┐
│           MEMBRANE SOLIDIFICATION                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   BEFORE: Flat surface (no volume)                                  │
│   ════════════════════════════════                                  │
│                                                                      │
│                    ╔═══════════════════╗                            │
│                    ║   Surface Mesh    ║  ← Zero thickness          │
│                    ╚═══════════════════╝    Can't use for CSG       │
│                                                                      │
│   AFTER: Solid volume with thickness                                │
│   ═══════════════════════════════════                               │
│                                                                      │
│                    ┌───────────────────┐                            │
│                    │                   │  ← Top offset surface      │
│                    │   Solid Volume    │                            │
│                    │                   │  ← Bottom offset surface   │
│                    └───────────────────┘                            │
│                    │                   │  ← Side faces (boundary)   │
│                                                                      │
│   Properties:                                                        │
│     • Watertight (manifold)                                         │
│     • No self-intersections                                         │
│     • Ready for CSG subtract/intersect                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Solidification Algorithm

```python
from core.membrane_solidifier import solidify_membrane, solidify_membranes

# Solidify a single membrane
result = solidify_membrane(
    surface_mesh=membrane,      # Input: flat surface mesh
    thickness=2.0,              # Total thickness in mesh units (mm)
    auto_reduce_thickness=True, # Reduce if self-intersection risk
    use_manifold_repair=True    # Use manifold3d for cleanup
)

# Result contains:
# - result.solid_mesh: The solidified trimesh.Trimesh
# - result.is_manifold: Boolean
# - result.is_watertight: Boolean  
# - result.has_self_intersections: Boolean
# - result.thickness_used: Actual thickness (may be reduced)
# - result.warnings: List of warnings

# Solidify multiple membranes at once
results = solidify_membranes(
    membranes=[membrane1, membrane2],
    thickness=2.0
)
```

#### Key Features

| Feature | Description |
|---------|-------------|
| **Offset Surface Creation** | Moves vertices along normals by ±thickness/2 |
| **Side Face Generation** | Connects boundary edges of top and bottom surfaces |
| **Auto Thickness Reduction** | Detects high curvature areas and reduces thickness to avoid self-intersections |
| **Manifold Repair** | Uses manifold3d to repair any non-manifold geometry |
| **Boundary Loop Detection** | Correctly orders boundary edges for side face creation |

#### Integration with Pipeline

```python
# After parting surface creation
from core.parting_surface import create_parting_surface
from core.membrane_solidifier import solidify_membrane, solidify_membranes

# Get parting surface and membranes
parting_result = create_parting_surface(mesh, parting_direction, ...)

# Solidify for use as cutting geometry
solidified_parting = solidify_membrane(
    parting_result.parting_surface, 
    thickness=2.0
)

solidified_membranes = solidify_membranes(
    parting_result.secondary_membranes,  # If any
    thickness=2.0
)

# Now ready for mold piece splitting
cutting_volumes = [solidified_parting.solid_mesh] + \
                  [r.solid_mesh for r in solidified_membranes]
```

---

### Phase 1: Input Preparation

Collect all required geometry from previous pipeline stages.

**Inputs Required**:
- `P1`, `P2`: Silicone mold piece meshes (from mold cavity splitting)
- `parting_surface`: The parting surface mesh
- `membranes`: Secondary membrane meshes (if any)
- `original_mesh`: The target object mesh

```python
@dataclass
class MetamoldInputs:
    """Inputs required for metamold generation."""
    mold_piece_1: trimesh.Trimesh      # P1 - first silicone mold piece
    mold_piece_2: trimesh.Trimesh      # P2 - second silicone mold piece  
    parting_surface: trimesh.Trimesh   # The parting surface
    membranes: List[trimesh.Trimesh]   # Secondary membranes (may be empty)
    original_mesh: trimesh.Trimesh     # Target object
    parting_direction: np.ndarray      # The parting/demolding direction
```

---

### Phase 2: Pouring Direction Selection

Determine optimal directions for both silicone and resin casting. This phase uses the already-implemented `pouring_direction.py` module.

#### Step 2.1: Compute Candidate Directions for Each Mold Piece

For each mold piece `Pi`:
1. Compute the best-fitting plane to the parting line (boundary of parting surface)
2. Sample candidate directions in a cone around the plane normal
3. Filter directions using persistence homology to minimize air trapping

```python
def compute_parting_line_plane(parting_surface: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the best-fitting plane to the parting line.
    
    Returns:
        centroid: Center point of the parting line
        normal: Normal vector of the best-fitting plane
    """
    # Extract boundary edges of parting surface
    boundary_vertices = get_boundary_vertices(parting_surface)
    
    # Fit plane using PCA or SVD
    centroid = np.mean(boundary_vertices, axis=0)
    centered = boundary_vertices - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # Smallest singular value direction = normal
    
    return centroid, normal
```

#### Step 2.2: Find Box-Compatible Direction Tuples

Search for direction combinations that are either:
- **Mutually orthogonal**: Directions at ~90° to each other
- **Mutually parallel**: Directions aligned (same or opposite)

This ensures the mold pieces can be assembled into a box.

```python
def find_box_compatible_directions(
    candidate_sets: List[List[np.ndarray]],
    orthogonality_tolerance: float = 10.0,  # degrees
    parallelism_tolerance: float = 10.0     # degrees
) -> List[Tuple[np.ndarray, ...]]:
    """
    Find q-tuples of directions that are box-compatible.
    
    Box-compatible means directions are approximately:
    - Mutually orthogonal, OR
    - Mutually parallel (same or opposite)
    
    Args:
        candidate_sets: List of candidate direction sets, one per mold piece
        orthogonality_tolerance: Max deviation from 90° for orthogonal
        parallelism_tolerance: Max deviation from 0°/180° for parallel
        
    Returns:
        List of box-compatible direction tuples
    """
    pass  # Implementation
```

#### Step 2.3: Select Optimal Direction Tuple

Among box-compatible tuples, choose the one that:
1. **Minimizes box height** (reduces silicone volume and casting pressure)
2. **Minimizes air bubble risk** (lowest combined pouring score)

```python
def select_optimal_directions(
    box_compatible_tuples: List[Tuple[np.ndarray, ...]],
    mold_pieces: List[trimesh.Trimesh]
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Select the best direction tuple for silicone and resin casting.
    
    Returns:
        silicone_directions: Optimal direction for each mold piece
        resin_direction: Optimal direction for resin casting
    """
    pass  # Implementation
```

---

### Phase 3: Bounding Box Generation

Generate the bounding box that will contain all mold pieces.

#### Step 3.1: Compute Oriented Bounding Box

```python
def compute_mold_bounding_box(
    mold_pieces: List[trimesh.Trimesh],
    silicone_directions: List[np.ndarray],
    wall_thickness: float = 5.0,      # mm
    base_thickness: float = 10.0      # mm
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Compute the bounding box for the assembled mold.
    
    The box is oriented so that:
    - One face is orthogonal to the resin pouring direction
    - Height is minimized to reduce silicone volume
    
    Args:
        mold_pieces: List of mold piece meshes
        silicone_directions: Pouring directions for each piece
        wall_thickness: Thickness of box walls
        base_thickness: Thickness of box base
        
    Returns:
        box_mesh: The bounding box as a mesh
        box_transform: Transform from world to box coordinates
    """
    pass  # Implementation
```

#### Step 3.2: Minimize Box Height

When direction tuples leave a degree of freedom (e.g., parallel opposite directions), choose the remaining box axis to minimize volume.

```python
def minimize_box_volume(
    mold_pieces: List[trimesh.Trimesh],
    fixed_axes: List[np.ndarray]
) -> np.ndarray:
    """
    Determine the remaining box axis that minimizes total volume.
    
    Args:
        mold_pieces: Mold piece meshes
        fixed_axes: Already determined box axes
        
    Returns:
        remaining_axis: The third box axis that minimizes volume
    """
    pass  # Implementation
```

---

### Phase 4: Metamold Shell Generation

Create the metamold geometry by offsetting the parting surface and membranes.

#### Step 4.1: Create Shell from Parting Surface

The parting surface becomes a solid shell in the metamold.

```python
def create_parting_surface_shell(
    parting_surface: trimesh.Trimesh,
    shell_thickness: float = 2.0  # mm - thickness of the solid shell
) -> trimesh.Trimesh:
    """
    Convert the parting surface into a solid shell for 3D printing.
    
    The shell is created by offsetting the surface in both directions
    by half the shell thickness, then connecting the boundaries.
    
    Args:
        parting_surface: The parting surface mesh
        shell_thickness: Thickness of the resulting shell
        
    Returns:
        shell_mesh: Solid shell mesh suitable for 3D printing
    """
    pass  # Implementation
```

#### Step 4.2: Create Shell from Membranes

Secondary membranes also become solid shells.

```python
def create_membrane_shells(
    membranes: List[trimesh.Trimesh],
    shell_thickness: float = 2.0  # mm
) -> List[trimesh.Trimesh]:
    """
    Convert secondary membranes into solid shells.
    
    Args:
        membranes: List of membrane surface meshes
        shell_thickness: Thickness of each shell
        
    Returns:
        shell_meshes: List of solid shell meshes
    """
    pass  # Implementation
```

#### Step 4.3: Combine Shells with Bounding Box

```python
def create_metamold_geometry(
    bounding_box: trimesh.Trimesh,
    parting_shell: trimesh.Trimesh,
    membrane_shells: List[trimesh.Trimesh],
    mold_piece_index: int  # Which mold piece this metamold is for
) -> trimesh.Trimesh:
    """
    Combine all components into the final metamold geometry.
    
    The metamold is the bounding box with the shells incorporated
    as solid walls that will form the parting surface cavity.
    
    Args:
        bounding_box: The outer bounding box
        parting_shell: The parting surface shell
        membrane_shells: Secondary membrane shells
        mold_piece_index: 0 for P1's metamold, 1 for P2's metamold
        
    Returns:
        metamold_mesh: Complete metamold geometry
    """
    pass  # Implementation
```

---

### Phase 5: Air Vent Generation

Add vents to allow air to escape during silicone casting.

#### Step 5.1: Identify Local Maxima

Using the pouring direction, identify local maxima where air would be trapped.

```python
def find_air_trap_locations(
    mold_piece: trimesh.Trimesh,
    pouring_direction: np.ndarray,
    tilt_angle_deg: float = 10.0,
    area_threshold_mm2: float = 0.5
) -> List[AirTrapLocation]:
    """
    Find locations where air would be trapped during casting.
    
    Uses persistence homology to identify relevant local maxima
    (already implemented in pouring_direction.py).
    
    Args:
        mold_piece: The mold piece mesh
        pouring_direction: Direction silicone will be poured
        tilt_angle_deg: Allowed mold tilt during casting
        area_threshold_mm2: Minimum trapped area to consider
        
    Returns:
        trap_locations: List of air trap locations with metadata
    """
    # Reuse compute_persistence_pairs from pouring_direction.py
    pass
```

#### Step 5.2: Generate Vent Pegs

Small pegs are placed at each local maximum as anchor points for 3D printed pipes.

```python
@dataclass
class VentPeg:
    """A vent peg for air escape."""
    position: np.ndarray      # Location on metamold surface
    direction: np.ndarray     # Direction the vent points (usually up)
    diameter: float           # Peg diameter in mm
    is_pour_hole: bool        # True if this is the main pour hole
    
def generate_vent_pegs(
    metamold: trimesh.Trimesh,
    air_trap_locations: List[AirTrapLocation],
    pouring_direction: np.ndarray,
    small_vent_diameter: float = 3.0,   # mm
    pour_hole_diameter: float = 10.0    # mm
) -> List[VentPeg]:
    """
    Generate vent pegs at air trap locations.
    
    - Small pegs at each local maximum for air vents
    - Larger peg at global maximum for pouring hole
    
    Args:
        metamold: The metamold mesh
        air_trap_locations: Locations where air would be trapped
        pouring_direction: Silicone pouring direction
        small_vent_diameter: Diameter of small air vents
        pour_hole_diameter: Diameter of main pour hole
        
    Returns:
        vent_pegs: List of vent peg specifications
    """
    pass  # Implementation
```

#### Step 5.3: Create Vent Pipe Geometry

```python
def create_vent_pipes(
    vent_pegs: List[VentPeg],
    pipe_length: float = 20.0,  # mm - length of vent pipe
    wall_thickness: float = 1.0  # mm - pipe wall thickness
) -> List[trimesh.Trimesh]:
    """
    Create 3D printable vent pipe geometry.
    
    These pipes attach to the pegs on the metamold to form
    complete air escape channels.
    
    Returns:
        pipe_meshes: List of pipe meshes to 3D print separately
    """
    pass  # Implementation
```

---

### Phase 6: Sealing Dam Generation

Add plug-and-slot sealing structures around the parting surface.

#### Step 6.1: Generate Dam Profile

The sealing dam is a closed loop around the object on the parting surface.

```python
@dataclass
class SealingDam:
    """Sealing dam structure for mold piece interface."""
    plug_mesh: trimesh.Trimesh   # The protruding plug part
    slot_mesh: trimesh.Trimesh   # The receiving slot part
    loop_path: np.ndarray        # (N, 3) path around object
    
def generate_sealing_dam(
    parting_surface: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh,
    dam_width: float = 3.0,       # mm - width of dam
    dam_height: float = 2.0,      # mm - height of plug
    dam_clearance: float = 0.2    # mm - clearance for fit
) -> SealingDam:
    """
    Generate sealing dam geometry.
    
    Creates a plug-and-slot structure that:
    - Surrounds the object in a closed loop
    - Sits on the parting surface
    - Ensures watertight seal between mold pieces
    
    Args:
        parting_surface: The parting surface mesh
        original_mesh: The target object mesh
        dam_width: Width of the dam profile
        dam_height: Height of the plug portion
        dam_clearance: Gap for easy assembly
        
    Returns:
        sealing_dam: The dam geometry for both mold pieces
    """
    pass  # Implementation
```

---

### Phase 7: Registration Peg Generation

Add pegs and holes to ensure perfect alignment of mold pieces.

#### Step 7.1: Identify Registration Locations

Registration features are placed on the parting surface, especially at membrane intersections.

```python
def find_registration_locations(
    parting_surface: trimesh.Trimesh,
    membranes: List[trimesh.Trimesh],
    min_spacing: float = 20.0  # mm - minimum distance between pegs
) -> List[np.ndarray]:
    """
    Find optimal locations for registration pegs.
    
    Pegs should be placed:
    - At membrane intersections with parting surface
    - Along parting surface edges
    - With sufficient spacing for stability
    
    Returns:
        peg_locations: List of (position, normal) tuples
    """
    pass  # Implementation
```

#### Step 7.2: Generate Peg and Hole Geometry

```python
@dataclass
class RegistrationFeature:
    """A registration peg or hole."""
    position: np.ndarray
    normal: np.ndarray
    is_peg: bool          # True = peg, False = hole
    diameter: float       # mm
    depth: float          # mm
    
def generate_registration_features(
    peg_locations: List[Tuple[np.ndarray, np.ndarray]],
    peg_diameter: float = 4.0,    # mm
    peg_depth: float = 5.0,       # mm
    hole_clearance: float = 0.2   # mm - extra space for fit
) -> Tuple[List[RegistrationFeature], List[RegistrationFeature]]:
    """
    Generate registration pegs and corresponding holes.
    
    Returns:
        pegs_for_p1: Registration pegs to add to P1's metamold
        holes_for_p2: Corresponding holes for P2's metamold
    """
    pass  # Implementation
```

---

### Phase 8: Final Assembly

Combine all components into the final metamold meshes.

```python
@dataclass
class MetamoldResult:
    """Result of metamold generation."""
    metamold_1: trimesh.Trimesh           # Metamold for casting P1
    metamold_2: trimesh.Trimesh           # Metamold for casting P2
    vent_pipes: List[trimesh.Trimesh]     # Separate vent pipe parts
    
    # Metadata
    silicone_direction_1: np.ndarray      # Pour direction for M1
    silicone_direction_2: np.ndarray      # Pour direction for M2
    resin_direction: np.ndarray           # Pour direction for final casting
    
    estimated_silicone_volume_1: float    # ml
    estimated_silicone_volume_2: float    # ml
    
    # Air trap info
    num_vents_1: int
    num_vents_2: int
    
def create_metamolds(
    inputs: MetamoldInputs,
    shell_thickness: float = 2.0,
    wall_thickness: float = 5.0,
    dam_width: float = 3.0,
    peg_diameter: float = 4.0,
    vent_diameter: float = 3.0,
    pour_hole_diameter: float = 10.0
) -> MetamoldResult:
    """
    Main function to create complete metamold geometry.
    
    This is the primary API that combines all phases.
    """
    # Phase 2: Compute optimal pouring directions
    directions = compute_optimal_directions(inputs)
    
    # Phase 3: Generate bounding box
    box = compute_mold_bounding_box(...)
    
    # Phase 4: Create shells
    parting_shell = create_parting_surface_shell(...)
    membrane_shells = create_membrane_shells(...)
    
    # Phase 5: Generate vents
    vents_1 = generate_vent_pegs(...)
    vents_2 = generate_vent_pegs(...)
    
    # Phase 6: Generate sealing dam
    sealing_dam = generate_sealing_dam(...)
    
    # Phase 7: Generate registration features
    reg_pegs, reg_holes = generate_registration_features(...)
    
    # Phase 8: Assemble final metamolds
    metamold_1 = assemble_metamold(box, parting_shell, membrane_shells, 
                                    vents_1, sealing_dam.plug_mesh, reg_pegs, ...)
    metamold_2 = assemble_metamold(box, parting_shell, membrane_shells,
                                    vents_2, sealing_dam.slot_mesh, reg_holes, ...)
    
    return MetamoldResult(...)
```

---

## File Structure

```
desktop_app/
├── core/
│   ├── membrane_solidifier.py   # NEW - Convert surfaces to solid volumes
│   ├── metamold.py              # NEW - Main metamold generation
│   ├── metamold_box.py          # NEW - Bounding box computation
│   ├── metamold_shells.py       # NEW - Shell generation from surfaces
│   ├── metamold_vents.py        # NEW - Air vent generation
│   ├── metamold_sealing.py      # NEW - Sealing dam generation
│   ├── metamold_registration.py # NEW - Registration peg generation
│   ├── pouring_direction.py     # EXISTING - Reuse for direction selection
│   └── parting_surface.py       # EXISTING - Provides parting surface
├── ui/
│   └── main_window.py           # Add metamold step UI
└── viewer/
    └── mesh_viewer.py           # Add metamold visualization
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Previous Pipeline Steps                                                     │
│  ══════════════════════                                                      │
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ STL Import   │ --> │ Parting Dir  │ --> │ Parting      │                 │
│  │              │     │ Selection    │     │ Surface      │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                   │                          │
│                                                   v                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ Inflated     │ --> │ Mold Cavity  │ --> │ Mold Piece   │                 │
│  │ Hull         │     │ (CSG)        │     │ Split        │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                   │                          │
│                                                   v                          │
│  Metamold Pipeline (This Document)                                           │
│  ═════════════════════════════════                                           │
│                                                   │                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ P1, P2       │ --> │ Direction    │ --> │ Box Layout   │                 │
│  │ Parting Surf │     │ Selection    │     │ Generation   │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                                         │                          │
│         v                                         v                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ Shell        │ --> │ Vents +      │ --> │ Final        │                 │
│  │ Generation   │     │ Sealing +    │     │ Metamolds    │                 │
│  │              │     │ Registration │     │ M1, M2       │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                   │                          │
│                                                   v                          │
│                                            ┌──────────────┐                 │
│                                            │ 3D Print     │                 │
│                                            │ Export       │                 │
│                                            └──────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Algorithms

### Algorithm 1: Box-Compatible Direction Search

```
INPUT: Candidate direction sets C1, C2, ..., Cq for each mold piece
OUTPUT: Box-compatible q-tuple (c1, c2, ..., cq)

1. For each combination (c1, c2, ..., cq) where ci ∈ Ci:
   2. Compute pairwise angles between all directions
   3. Check if all pairs are either:
      - Orthogonal (|angle - 90°| < tolerance), OR
      - Parallel (angle < tolerance OR |angle - 180°| < tolerance)
   4. If box-compatible, add to candidate list
   
5. For each box-compatible tuple:
   6. Compute resulting box dimensions
   7. Score = box_height * α + total_bubble_risk * β
   
8. Return tuple with minimum score
```

### Algorithm 2: Shell Generation from Surface

```
INPUT: Surface mesh S, thickness t
OUTPUT: Solid shell mesh

1. Compute vertex normals for S
2. Create offset surface S+ by moving vertices along normals by +t/2
3. Create offset surface S- by moving vertices along normals by -t/2
4. Identify boundary edges of S
5. Create side faces connecting S+ boundary to S- boundary
6. Combine S+, S-, and side faces into watertight mesh
7. Return shell mesh
```

### Algorithm 3: Sealing Dam Generation

```
INPUT: Parting surface PS, object mesh O
OUTPUT: Plug mesh, Slot mesh

1. Find intersection curve C between PS and O
2. Offset C outward on PS by dam_width/2 to get outer boundary
3. Create dam profile (rectangular cross-section)
4. Sweep profile along C to create dam geometry
5. Split into plug (upper half) and slot (lower half)
6. Add clearance to slot for assembly fit
7. Return plug and slot meshes
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shell_thickness` | 2.0 mm | Thickness of parting surface shell |
| `wall_thickness` | 5.0 mm | Thickness of bounding box walls |
| `base_thickness` | 10.0 mm | Thickness of bounding box base |
| `dam_width` | 3.0 mm | Width of sealing dam |
| `dam_height` | 2.0 mm | Height of sealing dam plug |
| `dam_clearance` | 0.2 mm | Assembly clearance for dam |
| `peg_diameter` | 4.0 mm | Diameter of registration pegs |
| `peg_depth` | 5.0 mm | Depth of registration pegs |
| `peg_clearance` | 0.2 mm | Assembly clearance for pegs |
| `vent_diameter` | 3.0 mm | Diameter of air vent pegs |
| `pour_hole_diameter` | 10.0 mm | Diameter of main pour hole |
| `pipe_length` | 20.0 mm | Length of vent pipes |

---

## Integration with Existing Code

### Using Pouring Direction Module

The existing `pouring_direction.py` module provides:

```python
from core.pouring_direction import (
    find_optimal_pouring_for_mold_pieces,
    compute_persistence_pairs,
    compute_direction_score,
    OptimalPouringDirections
)

# Get optimal directions for both mold pieces
result = find_optimal_pouring_for_mold_pieces(
    mold_piece_1=P1,
    mold_piece_2=P2,
    n_candidate_directions=64,
    tilt_angle_deg=10.0,
    area_threshold_mm2=0.5,
    alignment_angle_deg=30.0,
    resin_cone_angle_deg=10.0
)

# Use result.silicone_direction_1, silicone_direction_2, resin_direction
```

### Using Parting Surface Module

The existing `parting_surface.py` provides:

```python
from core.parting_surface import (
    create_parting_surface,
    PartingSurfaceResult
)

# Get parting surface and membranes from earlier step
parting_result: PartingSurfaceResult = ...
parting_surface = parting_result.parting_surface
membranes = parting_result.membranes  # If available
```

---

## Checklist

- [ ] Phase 1: Input preparation and data structures
- [ ] Phase 2: Pouring direction selection (integrate existing module)
- [ ] Phase 3: Bounding box generation
- [ ] Phase 4: Shell generation from surfaces
- [ ] Phase 5: Air vent generation
- [ ] Phase 6: Sealing dam generation
- [ ] Phase 7: Registration peg generation
- [ ] Phase 8: Final assembly and export
- [ ] Phase 9: UI integration
- [ ] Phase 10: Visualization
- [ ] Phase 11: Testing

---

## References

1. Alderighi, T., et al. (2018). Metamolds: Computational Design of Silicone Molds. ACM Trans. Graph. 37(4).
2. Malomo, L., et al. (2016). FlexMolds: Automatic Design of Flexible Shells for Molding. ACM Trans. Graph.
3. Existing implementation: `core/pouring_direction.py` - Persistence homology for bubble detection

---

## Notes

### Dependency on Mold Piece Generation

This metamold pipeline **requires** the mold pieces (P1, P2) to be generated first. The current pipeline status:

| Step | Status | Notes |
|------|--------|-------|
| Parting Direction | ✅ Complete | `parting_direction.py` |
| Parting Surface | ✅ Complete | `parting_surface.py` |
| Inflated Hull | ✅ Complete | `inflated_hull.py` |
| Mold Cavity (CSG) | ✅ Complete | `mold_cavity.py` - creates single cavity mesh |
| Mold Half Classification | ✅ Complete | `mold_half_classification.py` - **labels** faces as H1/H2 |
| Membrane Solidification | ✅ Complete | `membrane_solidifier.py` - converts flat surfaces to solid volumes |
| **Mold Piece Split** | ❌ **NOT IMPLEMENTED** | Need to **cut** cavity into P1, P2 meshes |
| **Metamold** | ⏳ Pending | This document |

### What "Mold Piece Split" Needs to Do

The `mold_half_classification.py` module only **classifies** which triangles belong to which mold half (H1 or H2). It does NOT create separate meshes. We need a new module that uses the **parting surface as a cutting tool** to split both boundaries.

**KEY INSIGHT**: The parting surface is the geometry that divides BOTH:
1. The **outer boundary** (inflated hull)
2. The **inner boundary** (original part mesh)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PARTING SURFACE AS CUTTING GEOMETRY                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                        Inflated Hull (outer)                             │
│                    ┌─────────────────────────┐                          │
│                   ╱                           ╲                          │
│                  ╱                             ╲                         │
│                 ╱     ┌───────────────────┐     ╲                        │
│                │      │                   │      │                       │
│                │      │   Original Part   │      │                       │
│                │      │     (inner)       │      │                       │
│                │      │                   │      │                       │
│                │      └───────────────────┘      │                       │
│                 ╲                               ╱                         │
│                  ╲                             ╱                          │
│                   ╲___________________________╱                           │
│                                                                          │
│                              ║                                           │
│                              ║  ← Parting Surface cuts through BOTH      │
│                              ║    the hull AND the part mesh             │
│                              ║                                           │
│                              ▼                                           │
│                                                                          │
│        ┌─────────────┐                    ┌─────────────┐               │
│        │     P1      │                    │     P2      │               │
│        │  ┌───────┐  │                    │  ┌───────┐  │               │
│        │  │ part  │  │                    │  │ part  │  │               │
│        │  │ half  │  │                    │  │ half  │  │               │
│        │  └───────┘  │                    │  └───────┘  │               │
│        │  hull half  │                    │  hull half  │               │
│        └──────┬──────┘                    └──────┬──────┘               │
│               │                                  │                       │
│               └──────────┐      ┌────────────────┘                       │
│                          │      │                                        │
│                          ▼      ▼                                        │
│                    Parting surface caps                                  │
│                    (close the open edges)                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Each mold piece P1/P2 contains:**

| Component | Source | How it's created |
|-----------|--------|------------------|
| **Outer surface** | Inflated hull | Cut by parting surface → one half |
| **Inner surface** | Original part mesh | Cut by parting surface → one half |
| **Cap surface** | Parting surface | Closes the open boundary |

```python
# What we need (NOT YET IMPLEMENTED):
P1, P2 = split_cavity_into_mold_pieces(
    hull_mesh=inflated_hull,           # Outer boundary
    part_mesh=original_mesh,           # Inner boundary  
    parting_surface=parting_surface,   # The cutting geometry
)

# The parting surface cuts through both meshes, creating:
# P1 = hull_half_1 + part_half_1 + parting_surface_cap
# P2 = hull_half_2 + part_half_2 + parting_surface_cap (flipped)
```

### The Cutting Operation

The parting surface acts as a **cutting plane/surface** that:

1. **Cuts the hull**: Splits the inflated hull into two pieces (H1 and H2 regions)
2. **Cuts the part**: Splits the original part mesh into two halves
3. **Becomes the cap**: The parting surface itself becomes the interface where P1 and P2 meet

This is essentially a **CSG split operation** using the parting surface as the cutting tool.

### Implementation Priority

Before metamold can be implemented, we need:

```
┌─────────────────────────────────────────────────────────────────────┐
│  REQUIRED BEFORE METAMOLD                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  NEW MODULE: core/mold_piece_split.py                               │
│                                                                      │
│  Inputs:                                                             │
│    • hull_mesh (inflated hull from inflated_hull.py)                │
│    • part_mesh (original input part)                                │
│    • primary_parting_surface (from parting_surface.py)              │
│    • secondary_membranes (list, may be empty)                       │
│                                                                      │
│  Outputs:                                                            │
│    • mold_pieces: List[trimesh.Trimesh] (2 or more pieces)          │
│                                                                      │
│  Algorithm (Unified Cutting Approach):                               │
│    All cutting surfaces (primary + secondary) are used TOGETHER     │
│    to split the geometry in a single operation.                     │
│                                                                      │
│    1. Combine all cutting surfaces:                                  │
│       all_surfaces = [primary_parting] + secondary_membranes        │
│                                                                      │
│    2. These surfaces divide 3D space into regions:                  │
│       - 2 surfaces → up to 4 regions (but typically 2-3 pieces)     │
│       - Each region bounded by some subset of the surfaces          │
│                                                                      │
│    3. For each region that intersects the cavity:                   │
│       - hull_portion = hull ∩ region                                │
│       - part_portion = part ∩ region                                │
│       - caps = surfaces bounding this region                        │
│                                                                      │
│    4. Assemble each mold piece:                                     │
│       P_i = hull_portion_i ∪ part_portion_i ∪ caps_i                │
│                                                                      │
│    5. Ensure all pieces are watertight                              │
│                                                                      │
│  Why Unified Cutting?                                                │
│    • Handles 2-piece molds (primary only)                           │
│    • Handles 3+ piece molds (primary + secondary)                   │
│    • Handles internal keys/flaps (secondary within a region)        │
│    • Same logic regardless of complexity                            │
│    • Cleaner watertight guarantee                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Handling Secondary Membranes

Secondary membranes serve two purposes:

1. **Create additional mold pieces** (3+ piece mold for complex geometry)
2. **Create internal flexibility** (keys/flaps for undercuts)

The unified cutting approach handles both:

```
┌─────────────────────────────────────────────────────────────────────────┐
│           SECONDARY MEMBRANE SCENARIOS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SCENARIO A: Additional Mold Piece (e.g., 3-piece mold)                 │
│  ─────────────────────────────────────────────────────                  │
│                                                                          │
│        Primary ═══════════════════════════                              │
│                      │                                                   │
│           P1         │         P2                                        │
│        ┌─────┐       │       ┌─────┐                                    │
│        │     │       │       │     │                                    │
│        │     │═══════│═══════│     │  ← Secondary                       │
│        │     │       │    ╔══│     │                                    │
│        └─────┘       │    ║  └─────┘                                    │
│                      │    ║                                              │
│                      │    ╠═══════╗                                      │
│                      │           P3                                      │
│                                                                          │
│        Result: 3 mold pieces, each with hull + part + caps              │
│                                                                          │
│  SCENARIO B: Internal Key (undercut handling)                           │
│  ────────────────────────────────────────────                           │
│                                                                          │
│        Primary ═══════════════════════════                              │
│                      │                                                   │
│           P1         │         P2                                        │
│        ┌─────┐       │       ┌─────┐                                    │
│        │  ┌──┼───────│───────┤     │                                    │
│        │  │  │ under │       │     │  ← Secondary creates               │
│        │  │  │ cut   │       │     │    a "flap" in P1                  │
│        │  │  │───────│───────┤     │    that can flex                   │
│        │  └──┘       │       │     │                                    │
│        └─────┘       │       └─────┘                                    │
│                                                                          │
│        Result: 2 mold pieces, P1 has internal membrane                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### CSG Split Approach

The parting surface can be converted to a cutting volume by:
1. Thickening the parting surface slightly (or using it as a splitting plane)
2. Using CSG operations to isolate geometry on each side

```python
# Conceptual approach using manifold3d or trimesh CSG:

def split_by_parting_surface(mesh, parting_surface, direction):
    """
    Split a mesh into two halves using the parting surface.
    
    Args:
        mesh: The mesh to split (hull or part)
        parting_surface: The cutting surface
        direction: Normal direction of parting surface (determines which side is which)
    
    Returns:
        mesh_half_1, mesh_half_2: The two halves
    """
    # Option 1: Create half-space volumes and intersect
    # Option 2: Use plane-based splitting if parting surface is planar
    # Option 3: Use the parting surface to define a cutting volume
    pass
```

### Shell Generation Complexity

Converting a surface mesh to a solid shell is non-trivial:
- Need to handle boundary edges properly
- Must maintain watertightness for 3D printing
- Consider using `trimesh.creation.extrude_polygon` for simple cases
- May need more sophisticated approach for complex parting surfaces

### Print Orientation

The metamold should be 3D printed with:
- Pour hole facing up (to avoid supports in critical areas)
- Shell walls vertical where possible (for print quality)
- Consider splitting large metamolds into printable pieces
