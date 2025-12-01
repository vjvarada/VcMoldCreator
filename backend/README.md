# Mold Creator Backend

Python backend for mold creation and tetrahedralization using fTetWild.

## Features

- **Tetrahedralization**: Convert surface meshes to tetrahedral meshes using [fTetWild](https://github.com/wildmeshing/fTetWild) via [pytetwild](https://github.com/pyvista/pytetwild)
- Supports STL, OBJ, PLY, and OFF file formats
- Export to MSH format for FEM analysis

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running

```bash
python app.py
```

The server will start on http://localhost:8000

## API Endpoints

### Health Check
- `GET /health` - Returns `{"status": "ok"}`

### Tetrahedralization

#### POST /tetrahedralize
Upload a surface mesh file and get the tetrahedral mesh data as JSON.

**Parameters:**
- `file`: Surface mesh file (STL, OBJ, PLY, or OFF)
- `edge_length_fac` (optional): Edge length factor relative to bounding box diagonal (default: 0.05)
- `optimize` (optional): Whether to optimize mesh quality (default: true)

**Example:**
```bash
curl -X POST "http://localhost:8000/tetrahedralize" \
  -F "file=@model.stl" \
  -F "edge_length_fac=0.05" \
  -F "optimize=true"
```

**Response:**
```json
{
  "success": true,
  "message": "Tetrahedralization completed successfully",
  "input_stats": {
    "num_vertices": 100,
    "num_faces": 196
  },
  "output_stats": {
    "num_vertices": 500,
    "num_tetrahedra": 2000
  },
  "vertices": [[x, y, z], ...],
  "tetrahedra": [[v0, v1, v2, v3], ...]
}
```

#### POST /tetrahedralize/msh
Upload a surface mesh and download the result as a MSH file.

**Example:**
```bash
curl -X POST "http://localhost:8000/tetrahedralize/msh" \
  -F "file=@model.stl" \
  -o model_tet.msh
```

#### POST /tetrahedralize/arrays
Tetrahedralize from raw vertex and face arrays (JSON body).

**Body:**
```json
{
  "vertices": [[0, 0, 0], [1, 0, 0], ...],
  "faces": [[0, 1, 2], [1, 2, 3], ...],
  "edge_length_fac": 0.05,
  "optimize": true
}
```

## Dependencies

- **FastAPI**: Web framework
- **pytetwild**: Python bindings for fTetWild tetrahedral meshing
- **trimesh**: Mesh loading and processing
- **meshio**: Mesh I/O for various formats
- **pyvista**: 3D visualization and mesh manipulation
- **numpy**: Numerical computing
