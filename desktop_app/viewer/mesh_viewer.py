"""
Mesh Viewer Widget

PyQt6 widget for 3D mesh visualization using PyVista.
Provides interactive 3D viewing of STL meshes with:
- Orthographic projection (no perspective distortion)
- Lighting and colors matching the React/Three.js frontend
- Efficient rendering for large meshes
"""

import logging
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

logger = logging.getLogger(__name__)


class MeshViewer(QWidget):
    """
    3D mesh viewer using PyVista with orthographic projection.
    
    Provides interactive 3D visualization of trimesh meshes with:
    - Orthographic camera (no perspective distortion)
    - Colors and lighting matching the React/Three.js frontend
    - Grid plane matching the frontend style
    - Scale rulers on X and Y axes showing dimensions
    - Orbit controls (rotate, pan, zoom)
    """
    
    # Colors matching React frontend (ThreeViewer.tsx + partingDirection.ts)
    MESH_COLOR = "#00aaff"       # Light blue - COLORS.MESH_DEFAULT
    BACKGROUND_COLOR = "#1a1a1a" # Dark background - BACKGROUND_COLOR
    GRID_COLOR_CENTER = "#444444" # Grid center line
    GRID_COLOR_LINES = "#333333"  # Grid other lines
    EDGE_COLOR = "#404040"
    SCALE_COLOR = "#ffcc00"      # Yellow/gold for scale rulers
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mesh: Optional[trimesh.Trimesh] = None
        self._pv_mesh: Optional[pv.PolyData] = None
        self._actor = None
        self._grid_actor = None
        self._scale_actors = []  # List of scale ruler actors
        self._show_edges = False
        self._current_color = self.MESH_COLOR
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
        
        # === GPU ACCELERATION AND PERFORMANCE OPTIMIZATION ===
        
        # Use faster anti-aliasing (FXAA is GPU-accelerated and faster than SSAA)
        self.plotter.enable_anti_aliasing('fxaa')
        
        # Enable depth peeling for better transparency handling (GPU accelerated)
        self.plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        
        # Set interaction style for smoother orbiting
        self.plotter.interactor.SetInteractorStyle(None)  # Reset first
        
        # Use trackball camera style for smoother rotation
        try:
            import vtkmodules.vtkInteractionStyle as vtk_style
            style = vtk_style.vtkInteractorStyleTrackballCamera()
            self.plotter.interactor.SetInteractorStyle(style)
        except ImportError:
            pass  # Fall back to default
        
        # Optimize render window for performance
        render_window = self.plotter.render_window
        if render_window:
            # Enable GPU acceleration features
            render_window.SetMultiSamples(0)  # Disable multisampling (using FXAA instead)
            render_window.SetPointSmoothing(False)  # Faster point rendering
            render_window.SetLineSmoothing(False)   # Faster line rendering
            
            # Set desired update rate for smooth interaction
            self.plotter.interactor.SetDesiredUpdateRate(60.0)  # 60 FPS target during interaction
            self.plotter.interactor.SetStillUpdateRate(0.01)    # High quality when still
        
        # Set orthographic projection (parallel projection, no perspective)
        self.plotter.enable_parallel_projection()
        
        # Set up soft, even lighting (optimized for performance)
        self._setup_lighting()
        
        # Add coordinate axes
        self.plotter.add_axes(
            line_width=2,
            color='white',
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
        
        # Add to layout
        layout.addWidget(self.plotter.interactor)
        
        # Log GPU/renderer info
        self._log_renderer_info()
        
        logger.info("Mesh viewer initialized with orthographic projection")
    
    def _log_renderer_info(self):
        """Log GPU/renderer information for debugging performance."""
        try:
            render_window = self.plotter.render_window
            if render_window:
                render_window.Render()  # Initialize rendering context
                info = render_window.ReportCapabilities()
                # Extract key info
                lines = info.split('\n') if info else []
                for line in lines[:10]:  # First 10 lines usually have GPU info
                    if any(kw in line.lower() for kw in ['opengl', 'renderer', 'vendor', 'gpu']):
                        logger.info(f"Renderer: {line.strip()}")
        except Exception as e:
            logger.debug(f"Could not get renderer info: {e}")
    
    def _setup_lighting(self):
        """
        Set up lighting matching the React/Three.js frontend.
        
        Three.js setup:
        - AmbientLight: intensity 0.6
        - DirectionalLight 1: intensity 0.8, position (200, 200, 200)
        - DirectionalLight 2: intensity 0.4, position (-200, -200, -200)
        """
        if not PYVISTA_AVAILABLE:
            return
            
        # Remove default lights
        self.plotter.remove_all_lights()
        
        # Ambient light (simulated with a very soft omnidirectional light)
        # PyVista doesn't have a true ambient light, so we use the renderer's ambient
        
        # Main directional light - matching Three.js light1
        light1 = pv.Light(
            position=(1, 1, 1),  # Normalized direction
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.8,
        )
        light1.positional = False  # Directional light
        self.plotter.add_light(light1)
        
        # Fill directional light - matching Three.js light2
        light2 = pv.Light(
            position=(-1, -1, -1),  # Normalized direction
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.4,
        )
        light2.positional = False  # Directional light
        self.plotter.add_light(light2)
    
    def _add_grid_plane(self, mesh_bounds=None):
        """
        Add a grid plane matching the React/Three.js frontend.
        
        Grid uses GRID_COLOR_CENTER for center lines and GRID_COLOR_LINES for others.
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove existing grid if any
        if hasattr(self, '_grid_actor') and self._grid_actor is not None:
            self.plotter.remove_actor(self._grid_actor)
            self._grid_actor = None
        
        # Calculate grid size based on mesh bounds or use default
        if mesh_bounds is not None:
            max_extent = max(
                mesh_bounds[1] - mesh_bounds[0],  # X extent
                mesh_bounds[3] - mesh_bounds[2],  # Y extent
                mesh_bounds[5] - mesh_bounds[4],  # Z extent
            )
            grid_size = max_extent * 2  # Grid is 2x the mesh size
        else:
            grid_size = 200  # Default size
        
        # Create grid lines manually for better control over colors
        # PyVista doesn't have a direct GridHelper like Three.js
        n_cells = 20  # Number of grid divisions
        cell_size = grid_size / n_cells
        half_size = grid_size / 2
        
        # Create grid as a plane at Z=0 (or at mesh bottom)
        z_pos = mesh_bounds[4] if mesh_bounds else 0  # Bottom of mesh
        
        # Create the grid using pyvista's built-in method
        grid_mesh = pv.Plane(
            center=(0, 0, z_pos),
            direction=(0, 0, 1),
            i_size=grid_size,
            j_size=grid_size,
            i_resolution=n_cells,
            j_resolution=n_cells,
        )
        
        # Extract edges for wireframe grid
        edges = grid_mesh.extract_all_edges()
        
        self._grid_actor = self.plotter.add_mesh(
            edges,
            color=self.GRID_COLOR_LINES,
            line_width=1,
            opacity=0.5,
            render_lines_as_tubes=False,
        )
    
    def _add_scale_rulers(self, mesh_bounds):
        """
        Add scale rulers on X and Y axes showing the object dimensions.
        
        Mimics the React/Three.js frontend scale rulers with:
        - Main ruler line
        - Tick marks (major and minor)
        - End caps
        - Dimension labels
        """
        if not PYVISTA_AVAILABLE or mesh_bounds is None:
            return
        
        # Remove existing scale rulers
        for actor in self._scale_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self._scale_actors = []
        
        # Get mesh dimensions
        size_x = mesh_bounds[1] - mesh_bounds[0]
        size_y = mesh_bounds[3] - mesh_bounds[2]
        size_z = mesh_bounds[5] - mesh_bounds[4]
        
        # Center of mesh
        center_x = (mesh_bounds[0] + mesh_bounds[1]) / 2
        center_y = (mesh_bounds[2] + mesh_bounds[3]) / 2
        z_pos = mesh_bounds[4]  # Bottom of mesh
        
        # Offset for rulers (outside the mesh)
        max_dim = max(size_x, size_y, size_z)
        ruler_offset = max(2, max_dim * 0.08)
        
        # Create X ruler (along X axis, at -Y edge)
        x_start = np.array([mesh_bounds[0], mesh_bounds[2] - ruler_offset, z_pos])
        self._create_ruler(x_start, size_x, 'X', direction='x')
        
        # Create Y ruler (along Y axis, at -X edge)
        y_start = np.array([mesh_bounds[0] - ruler_offset, mesh_bounds[2], z_pos])
        self._create_ruler(y_start, size_y, 'Y', direction='y')
    
    def _create_ruler(self, start_pos: np.ndarray, length_mm: float, label: str, direction: str = 'x'):
        """
        Create a single scale ruler with tick marks and labels.
        
        Args:
            start_pos: Starting position of the ruler
            length_mm: Length of the ruler in mm
            label: Axis label ('X' or 'Y')
            direction: Direction of the ruler ('x' or 'y')
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Direction vectors
        if direction == 'x':
            dir_vec = np.array([1, 0, 0])
            perp_vec = np.array([0, -1, 0])  # Perpendicular for tick marks
        else:
            dir_vec = np.array([0, 1, 0])
            perp_vec = np.array([-1, 0, 0])
        
        end_pos = start_pos + dir_vec * length_mm
        
        # Scale for tick marks based on ruler length
        ruler_scale = max(1, length_mm / 50)
        
        # Determine tick intervals based on length
        if length_mm > 100:
            tick_interval = 10
            major_tick_interval = 50
        elif length_mm > 20:
            tick_interval = 5
            major_tick_interval = 10
        else:
            tick_interval = 1
            major_tick_interval = 5
        
        # Main ruler line
        main_line = pv.Line(start_pos, end_pos)
        actor = self.plotter.add_mesh(
            main_line,
            color=self.SCALE_COLOR,
            line_width=2,
            opacity=0.8,
        )
        self._scale_actors.append(actor)
        
        # Create tick marks
        tick_points = []
        for mm in np.arange(0, length_mm + tick_interval, tick_interval):
            if mm > length_mm:
                mm = length_mm
            
            tick_pos = start_pos + dir_vec * mm
            is_major = (mm % major_tick_interval == 0) or mm == 0 or abs(mm - length_mm) < 0.1
            tick_length = ruler_scale * 2 if is_major else ruler_scale * 1
            
            tick_start = tick_pos + perp_vec * (-tick_length / 2)
            tick_end = tick_pos + perp_vec * (tick_length / 2)
            tick_points.append([tick_start, tick_end])
        
        # Add tick marks as line segments
        for tick_start, tick_end in tick_points:
            tick_line = pv.Line(tick_start, tick_end)
            actor = self.plotter.add_mesh(
                tick_line,
                color=self.SCALE_COLOR,
                line_width=1,
                opacity=0.7,
            )
            self._scale_actors.append(actor)
        
        # End caps
        cap_height = ruler_scale * 3
        for pos in [start_pos, end_pos]:
            cap_start = pos + perp_vec * (-cap_height / 2)
            cap_end = pos + perp_vec * (cap_height / 2)
            cap_line = pv.Line(cap_start, cap_end)
            actor = self.plotter.add_mesh(
                cap_line,
                color=self.SCALE_COLOR,
                line_width=2,
                opacity=0.8,
            )
            self._scale_actors.append(actor)
        
        # Add dimension label at the end
        label_offset = perp_vec * (ruler_scale * 5)
        label_pos = end_pos + label_offset
        
        dimension_text = f"{length_mm:.1f} mm"
        self.plotter.add_point_labels(
            [label_pos],
            [dimension_text],
            font_size=12,
            text_color=self.SCALE_COLOR,
            font_family='arial',
            bold=False,
            show_points=False,
            always_visible=True,
            shape=None,
            fill_shape=False,
        )
        
        # Add axis label at the end (max position)
        axis_label_offset = perp_vec * (-ruler_scale * 5)
        axis_label_pos = end_pos + axis_label_offset
        
        self.plotter.add_point_labels(
            [axis_label_pos],
            [label],
            font_size=14,
            text_color=self.SCALE_COLOR,
            font_family='arial',
            bold=True,
            show_points=False,
            always_visible=True,
            shape=None,
            fill_shape=False,
        )
    
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
        self._current_color = color or self.MESH_COLOR
        
        logger.info(f"Setting mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Convert trimesh to PyVista
        self._pv_mesh = self._trimesh_to_pyvista(mesh)
        
        # Render the mesh
        self._render_mesh()
        
        # Reset camera to fit mesh with orthographic view
        self._fit_camera_to_mesh()
        
        logger.info("Mesh rendered successfully")
    
    def _render_mesh(self):
        """Render the current mesh with settings matching Three.js MeshPhongMaterial."""
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Clear previous content
        self.plotter.clear()
        
        # Re-setup lighting after clear
        self._setup_lighting()
        
        # Add the mesh with material matching Three.js MeshPhongMaterial:
        # color: COLORS.MESH_DEFAULT (0x00aaff)
        # specular: 0x333333
        # shininess: 60
        self._actor = self.plotter.add_mesh(
            self._pv_mesh,
            color=self._current_color,
            smooth_shading=True,       # Smooth shading like Phong
            show_edges=self._show_edges,
            edge_color=self.EDGE_COLOR,
            line_width=1,
            opacity=1.0,
            # Material properties matching Three.js MeshPhongMaterial
            ambient=0.6,               # Matching Three.js AmbientLight intensity
            diffuse=0.8,               # Good diffuse response
            specular=0.2,              # specular: 0x333333 ≈ 0.2 intensity
            specular_power=60,         # shininess: 60
            render_points_as_spheres=False,
            render_lines_as_tubes=False,
        )
        
        # Set Phong interpolation for proper specular highlights
        if self._actor is not None:
            prop = self._actor.GetProperty()
            if prop:
                prop.SetInterpolationToPhong()
        
        # Add grid plane (like Three.js GridHelper)
        if self._pv_mesh is not None:
            self._add_grid_plane(self._pv_mesh.bounds)
            
            # Add scale rulers on X and Y axes
            self._add_scale_rulers(self._pv_mesh.bounds)
        
        # Add axes
        self.plotter.add_axes(
            line_width=2,
            color='white',
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
        
        # Ensure orthographic projection is maintained
        self.plotter.enable_parallel_projection()
        
        # Update the view
        self.plotter.update()
    
    def _fit_camera_to_mesh(self):
        """Fit the orthographic camera to show the entire mesh."""
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Get mesh bounds
        bounds = self._pv_mesh.bounds
        
        # Calculate center and size
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
        
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]
        max_size = max(size_x, size_y, size_z)
        
        # Set camera for isometric-like view
        distance = max_size * 2.5
        
        # Position camera at isometric angle
        cam_pos = [
            center[0] + distance * 0.7,
            center[1] - distance * 0.7,
            center[2] + distance * 0.5,
        ]
        
        self.plotter.camera_position = [
            cam_pos,           # Camera position
            center,            # Focal point
            (0, 0, 1),         # View up vector (Z-up)
        ]
        
        # Set parallel scale for orthographic projection
        self.plotter.camera.parallel_scale = max_size * 0.7
        
        self.plotter.reset_camera()
    
    def _trimesh_to_pyvista(self, mesh: trimesh.Trimesh) -> 'pv.PolyData':
        """
        Convert a trimesh mesh to PyVista PolyData.
        
        Args:
            mesh: The trimesh mesh to convert
            
        Returns:
            PyVista PolyData mesh
        """
        # Get vertices and faces with explicit dtype for efficiency
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        
        # PyVista expects faces in format: [n, v0, v1, v2, n, v0, v1, v2, ...]
        n_faces = len(faces)
        pv_faces = np.column_stack([
            np.full(n_faces, 3, dtype=np.int32),
            faces
        ]).ravel()
        
        # Create PyVista mesh
        pv_mesh = pv.PolyData(vertices, pv_faces)
        
        # Compute normals for smooth shading
        pv_mesh.compute_normals(inplace=True, cell_normals=True, point_normals=True)
        
        return pv_mesh
    
    def clear(self):
        """Clear the viewer."""
        if not PYVISTA_AVAILABLE:
            return
        
        self._mesh = None
        self._pv_mesh = None
        self._actor = None
        self._grid_actor = None
        self._scale_actors = []
        self.plotter.clear()
        self._setup_lighting()
        self.plotter.add_axes(
            line_width=2,
            color='white',
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
        self.plotter.enable_parallel_projection()
        self.plotter.update()
    
    def reset_camera(self):
        """Reset camera to fit the mesh."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._pv_mesh is not None:
            self._fit_camera_to_mesh()
        else:
            self.plotter.reset_camera()
    
    def set_view(self, view: str):
        """
        Set a predefined orthographic view.
        
        Args:
            view: One of 'front', 'back', 'left', 'right', 'top', 'bottom', 'iso'
        """
        if not PYVISTA_AVAILABLE:
            return
        
        if view == 'front':
            self.plotter.view_yz()
        elif view == 'back':
            self.plotter.view_yz(negative=True)
        elif view == 'left':
            self.plotter.view_xz(negative=True)
        elif view == 'right':
            self.plotter.view_xz()
        elif view == 'top':
            self.plotter.view_xy()
        elif view == 'bottom':
            self.plotter.view_xy(negative=True)
        elif view == 'iso':
            self.plotter.view_isometric()
        
        # Ensure orthographic projection
        self.plotter.enable_parallel_projection()
        self.plotter.reset_camera()
    
    def show_edges(self, show: bool = True):
        """Toggle edge visibility."""
        if not PYVISTA_AVAILABLE:
            return
        
        self._show_edges = show
        
        if self._pv_mesh is not None:
            self._render_mesh()
            self._fit_camera_to_mesh()
    
    def set_mesh_visible(self, visible: bool):
        """
        Set mesh visibility (show/hide).
        
        Args:
            visible: True to show mesh, False to hide
        """
        if not PYVISTA_AVAILABLE:
            return
        
        if self._actor is not None:
            self._actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Mesh visibility set to: {visible}")
    
    def set_mesh_color(self, color: str):
        """
        Set the mesh color.
        
        Args:
            color: Hex color string (e.g., '#ff0000')
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._current_color = color
        
        if self._pv_mesh is not None:
            self._render_mesh()
            self._fit_camera_to_mesh()
    
    def set_background_color(self, color: str):
        """Set the background color."""
        if not PYVISTA_AVAILABLE:
            return
        
        self.plotter.set_background(color)
        self.plotter.update()
    
    @property
    def mesh(self) -> Optional[trimesh.Trimesh]:
        """Get the current mesh."""
        return self._mesh
    
    @property
    def is_orthographic(self) -> bool:
        """Check if using orthographic projection."""
        if not PYVISTA_AVAILABLE:
            return True
        return self.plotter.camera.parallel_projection
