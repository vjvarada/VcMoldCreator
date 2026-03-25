<div align="center">

# VcMoldCreator

### Automatic Composite Mold Design for Complex 3D Shapes

*A computational tool that automatically designs two-piece composite silicone molds for casting objects of unprecedented geometric complexity — implementing the algorithm from **"Volume-Aware Design of Composite Molds"** (Alderighi et al., SIGGRAPH 2019).*

<br>

<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0001-05.png" width="30%" alt="Cut layout with parting surface (blue) and additional membranes (red)">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0001-06.png" width="30%" alt="Composite mold assembled">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0001-07.png" width="30%" alt="Final cast object">

*Left: Cut layout — parting surface (blue) and additional membranes (red). Center: Assembled composite mold. Right: Cast object.*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-SIGGRAPH%202019-red.svg)](https://doi.org/10.1145/3306346.3322981)

</div>

---

## What Is This?

**VcMoldCreator** is a desktop application that takes a 3D mesh (STL file) as input and automatically computes everything needed to fabricate a composite, two-piece silicone mold for casting that shape. The system handles objects that are impossible to cast with traditional rigid molds:

- **Thin protruding features** (tentacles, horns, fingers)
- **Deep undercuts** and concavities
- **Non-zero genus topology** (objects with holes, like a torus)
- **Knotted surfaces** and intertwined pieces
- **Multi-component entangled objects** (a cage around a knot)

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0001-04.png" width="85%" alt="Complex entangled objects that can be cast with our technique">

*Objects of extreme complexity — including entangled multi-piece assemblies — can be cast using automatically designed composite molds.*
</div>

---

## How It Works

The core idea is **volumetric analysis of escape paths**. Instead of analyzing just the surface of the object (like prior methods), VcMoldCreator tessellates the entire mold volume surrounding the object and analyzes the shortest paths that each volume element would take when removing the mold.

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0003-01.png" width="60%" alt="Escape path concept">

*The escape path concept: for each point in the mold volume, the escape path is the shortest walk toward the exterior. The behavior of these paths determines where to place cuts in the mold.*
</div>

### The Composite Mold

Each mold piece consists of two parts:
- **A hard plastic shell** — 3D printed, keeps the mold rigid during casting
- **A flexible silicone part** — cast inside the shell, conforms to the object's complex geometry

The thin silicone part can flex to release undercuts, while the hard shell prevents deformation under casting pressure.

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0004-05.png" width="42%" alt="Mold volume schema">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0004-06.png" width="42%" alt="Parting directions">

*Left: The mold volume O lies between the object surface M and the exterior hull ∂H. Right: Two parting directions d₁, d₂ split the hull boundary, defining how the mold opens.*
</div>

---

## Algorithm Pipeline

The application implements a 12-step computational pipeline:

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0003-11.png" width="90%" alt="Fabrication pipeline overview">

*The full fabrication pipeline: from cut layout design (a,b) through 3D printing the hard shell (c), creating metamolds (d), casting silicone (e), to the final assembled composite mold with cast object (f).*
</div>

### Step-by-Step Breakdown

| Step | Operation | Description |
|:----:|-----------|-------------|
| 1 | **Import & Validation** | Load STL, validate manifold mesh, optionally repair |
| 2 | **Parting Direction** | Sample k directions on Gaussian sphere, GPU-accelerated visibility to find optimal d₁, d₂ |
| 3 | **Inflated Hull** | Create bounding volume (convex hull offset) around the object |
| 4 | **Tetrahedralization** | Tessellate the mold volume using fTetWild (robust tetrahedral meshing) |
| 5 | **Mold Half Classification** | Region-growing from seed faces to partition hull into ∂H₁ and ∂H₂ |
| 6 | **Edge Weights** | Compute weighted geodesics: `w = 1/(dist² + ε)` to push membranes perpendicular to surface |
| 7 | **Dijkstra Escape Paths** | Shortest paths from interior vertices to hull boundary — labels determine mold piece assignment |
| 8 | **Primary Surface Extraction** | Extended Marching Tetrahedra to extract the parting surface from cut edges |
| 9 | **Membrane Smoothing** | Laplacian smoothing with boundary re-projection to M and ∂H |
| 10 | **Secondary Membranes** | Detect features that prevent extraction; add additional cuts |
| 11 | **Pouring Direction** | Persistence homology to minimize trapped air bubbles |
| 12 | **Mold Fabrication** | Generate hard shell, metamold geometry, and export for 3D printing |

### Membrane Computation

The algorithm detects where to place cutting membranes by analyzing how escape paths diverge around object features:

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0005-01.png" width="30%" alt="Parting surface cut">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0005-02.png" width="60%" alt="Additional membrane detection">

*Left: A **parting surface** cut separates vertices whose escape paths reach different hull halves. Center: An **additional membrane** is needed when the minimal surface between escape paths intersects the object. Right: No membrane needed when the surface doesn't intersect.*
</div>

### Weighted Geodesics & Membrane Quality

Plain geodesics produce poorly shaped membranes that travel tangentially to the surface. Weighted geodesics with exterior boundary reshaping produce clean, perpendicular membranes:

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0006-01.png" width="85%" alt="Effects of weighting and reshaping">

*The effect of weighted geodesics and boundary reshaping: (a) plain geodesics produce tangent membranes; (b) boundary reshape alone; (c) distance weighting alone; (d) both combined produce optimal results.*
</div>

### Membrane Smoothing

The raw membranes extracted from the tetrahedral mesh are noisy. Laplacian smoothing with boundary re-projection produces clean, printable surfaces:

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0005-12.png" width="42%" alt="Noisy membranes">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0005-13.png" width="42%" alt="Smoothed membranes">

*Left: Raw membranes from Marching Tetrahedra. Right: After Laplacian smoothing with boundary re-projection.*
</div>

### Pouring Direction Optimization

Air bubbles get trapped at local maxima during casting. The algorithm uses persistence homology to find pouring directions that minimize trapped air:

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0008-01.png" width="75%" alt="Persistence pairing for air bubble minimization">

*Maximum-saddle pairs identify air bubble trapping regions (red). Tilting the mold allows some air to escape (green), but trapped regions remain. The algorithm selects directions that minimize the total trapped area.*
</div>

---

## Fabrication Pipeline

Once the computational design is complete, the physical fabrication follows these steps:

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0007-01.png" width="45%" alt="Fabrication pipeline left">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0007-02.png" width="45%" alt="Fabrication pipeline right">

*The fabrication and assembly pipeline: 3D print the hard shell and metamolds → assemble casting containers → pour silicone → assemble composite mold → pour resin to cast the final object.*
</div>

1. **3D Print** the hard plastic shell (2 pieces) and metamolds (2 pieces)
2. **Assemble** each shell piece with its corresponding metamold
3. **Pour silicone** into the containers to cast the flexible mold parts
4. **Assemble** the composite mold (hard shell + silicone parts)
5. **Cast** the final object by pouring resin into the assembled mold

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0007-04.png" width="42%" alt="Air vents in hard shell">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0007-05.png" width="42%" alt="Air pipes in metamold">

*Air vents and pipes are incorporated into the hard shell and metamold geometry to allow casting liquid to flow in and air to escape.*
</div>

---

## Results

The algorithm successfully handles objects of extreme geometric and topological complexity:

### Complex Geometric Features

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0010-01.png" width="22%" alt="Medusa">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0010-02.png" width="22%" alt="Laocoonte">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0010-03.png" width="22%" alt="Dragon Head">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0010-04.png" width="22%" alt="Dragon">

*Successful casting of geometrically complex shapes: Medusa, Laocoonte, Dragon Head, and Dragon models.*
</div>

### Non-Zero Genus, Knots & Entangled Pieces

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0011-01.png" width="45%" alt="Wheel of life and knot">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0011-02.png" width="45%" alt="Caged knot">

*Left: Wheel of Life (non-zero genus) and trefoil knot. Right: Caged knot — two entangled pieces cast simultaneously.*
</div>

### Smart Membrane Design

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0011-09.png" width="30%" alt="Membrane closeup 1">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0011-10.png" width="30%" alt="Membrane closeup 2">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0011-11.png" width="30%" alt="Ant membranes">

*Close-ups of additional membranes on the Medusa model (left, center) and full membrane layout on an ant model (right). Membranes intelligently follow object features and intersect freely.*
</div>

### Reusable Molds — Multiple Casts

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0011-04.png" width="60%" alt="Multiple casts from the same mold">

*The composite molds are reusable — multiple copies can be cast efficiently with no damage to the mold.*
</div>

---

## Installation

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/VcMoldCreator.git
cd VcMoldCreator

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
cd desktop_app
pip install -r requirements.txt
```

### Run the Application

```bash
cd desktop_app
python main.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| **PyQt6** | Desktop GUI framework |
| **trimesh** | Mesh loading, manipulation, ray intersection |
| **numpy / scipy** | Numerical operations |
| **pyvista / pyvistaqt** | 3D visualization with PyQt integration |
| **pytetwild** | Tetrahedral meshing (fTetWild bindings) |
| **manifold3d** | CSG operations for mold cavity creation |
| **meshlib** | High-performance mesh repair |
| **potpourri3d** | Geodesic computation (Heat Method) |
| **torch** | Optional GPU acceleration for distance computation |
| **python-fcl** | Collision detection for secondary cuts |

---

## Project Structure

```
VcMoldCreator/
├── README.md                          # This file
├── desktop_app/                       # Main Python/PyQt6 desktop application
│   ├── main.py                        # Application entry point
│   ├── requirements.txt               # Python dependencies
│   ├── core/                          # Algorithm implementations
│   │   ├── stl_loader.py              # STL file loading & validation
│   │   ├── mesh_repair.py             # Non-manifold mesh repair
│   │   ├── mesh_decimation.py         # Mesh simplification for large models
│   │   ├── mesh_analysis.py           # Mesh diagnostics
│   │   ├── inflated_hull.py           # Convex hull inflation (bounding volume)
│   │   ├── parting_direction.py       # Visibility-based direction selection
│   │   ├── tetrahedral_mesh.py        # Tet meshing, Dijkstra, cut edges
│   │   ├── mold_half_classification.py # H₁/H₂ region-growing classification
│   │   ├── parting_surface.py         # Extended Marching Tetrahedra
│   │   ├── surface_propagation.py     # Laplacian smoothing with re-projection
│   │   ├── secondary_membrane.py      # Secondary cut detection
│   │   ├── pouring_direction.py       # Persistence-based pouring optimization
│   │   ├── mold_fabrication.py        # Hard shell & metamold generation
│   │   ├── export_artifacts.py        # STL/OBJ export
│   │   ├── alignment_notches.py       # Mold alignment features
│   │   ├── registration_marks.py      # Perlin noise registration on seams
│   │   └── resin_channels.py          # Resin pour channels & air vents
│   ├── ui/
│   │   └── main_window.py             # Main PyQt6 application window
│   └── viewer/
│       └── mesh_viewer.py             # PyVista 3D visualization widget
└── docs/                              # Research papers & implementation guides
    ├── README.md                      # Documentation index
    ├── PAPERS_TO_CODE_MAPPING.md      # Paper sections → code functions
    ├── alderighi-2019-composite-molds/ # Primary reference paper
    ├── nielson-franke-1997-marching-tetrahedra/  # Marching Tetrahedra
    ├── bloomenthal-1995-non-manifold/  # Non-manifold surface handling
    └── gibson-1998-surface-nets/       # Surface Nets algorithm
```

---

## Key Algorithms

### Weighted Geodesics (Paper §4.5)

Membranes are pushed perpendicular to the object surface by weighting edge lengths:

$$w_e = \frac{1}{d_e^2 + \varepsilon}, \quad \text{cost}(e) = l(e) \cdot w_e$$

where $d_e$ is the geodesic distance from the edge midpoint to the object surface $M$, and $\varepsilon = 0.25$.

### Exterior Boundary Reshape (Paper §4.5, Figure 8)

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0006-03.png" width="50%" alt="Boundary reshape">

*The distance from interior vertex v to boundary point w is biased by λ_w, the distance from w to the offset surface ∂F.*
</div>

$$\lambda_w = R - \text{dist}(w, M), \quad \text{biased\_dist}(v, w) = d(v, w) + \lambda_w$$

### Extended Marching Tetrahedra (Paper §4.3)

Surfaces are extracted from the tetrahedral mesh using 6-bit edge configuration encoding:

| Cut Edges | Configuration | Triangles Generated |
|:---------:|:-------------:|:-------------------:|
| 0 | No surface | 0 |
| 3 | One vertex isolated | 1 |
| 4 | 2v2 split | 2 (quad) |
| 5 | Face vertex required | 5 |
| 6 | Inner vertex required | 12 |

---

## Comparison with Prior Work

VcMoldCreator's volumetric approach handles cases where surface-based methods (like Metamolds) fail:

<div align="center">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0012-09.png" width="18%" alt="Comparison 1">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0012-10.png" width="18%" alt="Comparison 2">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0012-11.png" width="18%" alt="Comparison 3">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0012-12.png" width="18%" alt="Comparison 4">
<img src="docs/alderighi-2019-composite-molds/images/Volume-Aware%20Design%20of%20Composite%20Molds.pdf-0012-13.png" width="18%" alt="Comparison 5">

*Failure cases of surface-based Metamolds (left in each pair) vs. our volumetric approach (right). Complex knots, hindering membranes, and invalid segmentations are all handled correctly.*
</div>

---

## References

**Primary Paper:**

> Thomas Alderighi, Luigi Malomo, Daniela Giorgi, Bernd Bickel, Paolo Cignoni, and Nico Pietroni. 2019. **Volume-Aware Design of Composite Molds.** *ACM Trans. Graph.* 38, 4, Article 110. [DOI: 10.1145/3306346.3322981](https://doi.org/10.1145/3306346.3322981)

**Referenced Algorithms:**

- Nielson & Franke, *"Computing the Separating Surface for Segmented Data"*, 1997 — Marching Tetrahedra
- Bloomenthal & Ferguson, *"Polygonization of Non-Manifold Implicit Surfaces"*, 1995 — Non-manifold handling
- Gibson, *"Constrained Elastic Surface Nets"*, 1998 — Surface Nets
- Edelsbrunner et al., *"Topological Persistence and Simplification"*, 2000 — Persistence homology
- Alderighi et al., *"Metamolds: Computational Design of Silicone Molds"*, 2018 — Predecessor method

---

## License

This project is for academic and research purposes. See [LICENSE](LICENSE) for details.

---

<div align="center">

*VcMoldCreator — Turning impossible molds into reality, one tetrahedron at a time.*

</div>