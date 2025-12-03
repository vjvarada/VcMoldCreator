# VcMoldCreator Desktop Application

A Python/PyQt6 desktop application for mold analysis and parting surface computation.

## Features

- **STL File Loading**: Load and display STL files with drag-and-drop support
- **Mesh Analysis**: Analyze mesh properties (vertices, faces, bounding box, etc.)
- **Mesh Repair**: Automatic repair of non-manifold meshes using trimesh/manifold3d
- **3D Visualization**: Interactive 3D viewer using PyVistaQt

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Project Structure

```
desktop_app/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── ui/
│   ├── __init__.py
│   ├── main_window.py     # Main application window
│   └── file_upload.py     # File upload widget
├── core/
│   ├── __init__.py
│   ├── stl_loader.py      # STL file loading
│   ├── mesh_analysis.py   # Mesh analysis and diagnostics
│   └── mesh_repair.py     # Mesh repair operations
└── viewer/
    ├── __init__.py
    └── mesh_viewer.py     # 3D mesh visualization
```

## Dependencies

- **PyQt6**: GUI framework
- **trimesh**: Mesh loading, analysis, and manipulation
- **numpy**: Numerical operations
- **PyVista**: 3D visualization
- **PyVistaQt**: PyQt integration for PyVista

## Development

This application mirrors the functionality of the web-based frontend, reimplemented in Python for desktop use.
