"""
Mesh Viewer Widget

PyQt6 widget for 3D mesh visualization using PyVista.
Provides interactive 3D viewing of STL meshes.
"""

from typing import Optional
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

import trimesh


class MeshViewer(QWidget):
    """
    3D mesh viewer using PyVista.
    
    Provides interactive 3D visualization of trimesh meshes with:
    - Orbit controls (rotate, pan, zoom)
    - Lighting
    - Mesh coloring
    """
    
    # Colors matching the frontend
    MESH_COLOR = "#6699ff"  # Blue-ish for main mesh
    BACKGROUND_COLOR = "#1a1a1a"  # Dark background
    EDGE_COLOR = "#333333"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mesh: Optional[trimesh.Trimesh] = None
        self._pv_mesh: Optional[pv.PolyData] = None
        self._actor = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the viewer UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if not PYVISTA_AVAILABLE:
            # Fallback if PyVista is not installed
            label = QLabel("PyVista not installed.\nInstall with: pip install pyvista pyvistaqt")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("""
                color: #ff6666;
                font-size: 14px;
                padding: 20px;
            """)
            layout.addWidget(label)
            return
        
        # Create PyVista Qt interactor
        self.plotter = QtInteractor(self)
        self.plotter.set_background(self.BACKGROUND_COLOR)
        
        # Enable anti-aliasing
        self.plotter.enable_anti_aliasing()
        
        # Set up lighting for better visualization
        self._setup_lighting()
        
        # Add coordinate axes
        self.plotter.add_axes()
        
        # Add to layout
        layout.addWidget(self.plotter.interactor)
    
    def _setup_lighting(self):
        """Set up professional lighting for better mesh visualization."""
        # Remove default lights
        self.plotter.remove_all_lights()
        
        # Add a 3-point lighting setup for professional look
        # Key light (main light) - bright, from upper right front
        key_light = pv.Light(
            position=(1, 1, 1),
            focal_point=(0, 0, 0),
            color='white',
            intensity=1.0,
        )
        key_light.positional = False  # Directional light
        self.plotter.add_light(key_light)
        
        # Fill light - softer, from left side to reduce harsh shadows
        fill_light = pv.Light(
            position=(-1, 0.5, 0.5),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.4,
        )
        fill_light.positional = False
        self.plotter.add_light(fill_light)
        
        # Back/rim light - from behind to add edge definition
        back_light = pv.Light(
            position=(0, -1, 0.5),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.3,
        )
        back_light.positional = False
        self.plotter.add_light(back_light)
        
        # Top light for overall ambient
        top_light = pv.Light(
            position=(0, 0, 2),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.2,
        )
        top_light.positional = False
        self.plotter.add_light(top_light)
    
    def set_mesh(self, mesh: trimesh.Trimesh, color: Optional[str] = None):
        """
        Set the mesh to display.
        
        Args:
            mesh: The trimesh mesh to display
            color: Optional color for the mesh (hex string)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._mesh = mesh
        
        # Convert trimesh to PyVista
        self._pv_mesh = self._trimesh_to_pyvista(mesh)
        
        # Clear previous mesh
        self.plotter.clear()
        
        # Re-setup lighting after clear
        self._setup_lighting()
        
        # Add the mesh with improved material properties
        self._actor = self.plotter.add_mesh(
            self._pv_mesh,
            color=color or self.MESH_COLOR,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
            specular=0.8,           # Higher specular for more shine
            specular_power=50,       # Tighter specular highlights
            ambient=0.2,             # Some ambient to prevent pure black shadows
            diffuse=0.8,             # Good diffuse reflection
            metallic=0.1,            # Slight metallic look
            roughness=0.3,           # Some roughness for realistic look
            pbr=True,                # Use physically based rendering if available
        )
        
        # Add axes
        self.plotter.add_axes()
        
        # Reset camera to fit mesh
        self.plotter.reset_camera()
        
        # Update the view
        self.plotter.update()
    
    def _trimesh_to_pyvista(self, mesh: trimesh.Trimesh) -> 'pv.PolyData':
        """
        Convert a trimesh mesh to PyVista PolyData.
        
        Args:
            mesh: The trimesh mesh to convert
            
        Returns:
            PyVista PolyData mesh
        """
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # PyVista expects faces in format: [n, v0, v1, v2, n, v0, v1, v2, ...]
        # where n is the number of vertices per face (3 for triangles)
        n_faces = len(faces)
        pv_faces = np.column_stack([
            np.full(n_faces, 3),  # All triangles have 3 vertices
            faces
        ]).ravel()
        
        # Create PyVista mesh
        pv_mesh = pv.PolyData(vertices, pv_faces)
        
        # Compute normals for better rendering
        pv_mesh.compute_normals(inplace=True)
        
        return pv_mesh
    
    def clear(self):
        """Clear the viewer."""
        if not PYVISTA_AVAILABLE:
            return
        
        self._mesh = None
        self._pv_mesh = None
        self._actor = None
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.update()
    
    def reset_camera(self):
        """Reset camera to fit the mesh."""
        if not PYVISTA_AVAILABLE:
            return
        
        self.plotter.reset_camera()
    
    def set_view(self, view: str):
        """
        Set a predefined view.
        
        Args:
            view: One of 'xy', 'xz', 'yz', 'iso'
        """
        if not PYVISTA_AVAILABLE:
            return
        
        if view == 'xy':
            self.plotter.view_xy()
        elif view == 'xz':
            self.plotter.view_xz()
        elif view == 'yz':
            self.plotter.view_yz()
        elif view == 'iso':
            self.plotter.view_isometric()
        
        self.plotter.reset_camera()
    
    def show_edges(self, show: bool = True):
        """Toggle edge visibility."""
        if not PYVISTA_AVAILABLE or self._actor is None:
            return
        
        # Need to re-add the mesh with edges
        if self._pv_mesh is not None:
            self.plotter.clear()
            self._setup_lighting()
            self._actor = self.plotter.add_mesh(
                self._pv_mesh,
                color=self.MESH_COLOR,
                smooth_shading=True,
                show_edges=show,
                edge_color=self.EDGE_COLOR,
                opacity=1.0,
                specular=0.8,
                specular_power=50,
                ambient=0.2,
                diffuse=0.8,
                pbr=True,
            )
            self.plotter.add_axes()
            self.plotter.update()
    
    def set_mesh_color(self, color: str):
        """
        Set the mesh color.
        
        Args:
            color: Hex color string (e.g., '#ff0000')
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        self.plotter.clear()
        self._setup_lighting()
        self._actor = self.plotter.add_mesh(
            self._pv_mesh,
            color=color,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
            specular=0.8,
            specular_power=50,
            ambient=0.2,
            diffuse=0.8,
            pbr=True,
        )
        self.plotter.add_axes()
        self.plotter.update()
    
    @property
    def mesh(self) -> Optional[trimesh.Trimesh]:
        """Get the current mesh."""
        return self._mesh
