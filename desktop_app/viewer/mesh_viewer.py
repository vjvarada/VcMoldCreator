"""
Mesh Viewer Widget

PyQt6 widget for 3D mesh visualization using PyVista.
Provides interactive 3D viewing of STL meshes with:
- Orthographic projection (no perspective distortion)
- Lighting and colors matching the React/Three.js frontend
- Efficient rendering for large meshes
- Parting direction visualization (arrows and visibility painting)
"""

import logging
from typing import Optional, List
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
    - Parting direction arrows
    - Visibility painting for mold analysis
    - Standard shading with feature edges
    """
    
    # Colors for light theme
    MESH_COLOR = "#00aaff"       # Light blue - mesh color
    BACKGROUND_COLOR = "#f0f0f0" # Light gray background
    GRID_COLOR_CENTER = "#aaaaaa" # Grid center line (darker for visibility)
    GRID_COLOR_LINES = "#cccccc"  # Grid other lines
    EDGE_COLOR = "#000000"        # Black for feature/silhouette edges (cel shading)
    SCALE_COLOR = "#333333"      # Dark gray for scale rulers (visible on light bg)
    
    # Parting direction colors (matching React frontend)
    PARTING_D1_COLOR = "#00ff00"  # Green - Primary direction
    PARTING_D2_COLOR = "#ff6600"  # Orange - Secondary direction
    
    # Pouring direction colors (distinct from parting)
    POURING_S1_COLOR = "#00dddd"  # Cyan - Silicone direction 1
    POURING_S2_COLOR = "#dd44ff"  # Magenta - Silicone direction 2
    POURING_RESIN_COLOR = "#ff6666"  # Red - Resin direction
    
    # Inflated hull colors (matching React frontend)
    HULL_COLOR = "#9966ff"        # Purple - Inflated hull
    ORIGINAL_HULL_COLOR = "#666666"  # Gray - Original hull wireframe
    HULL_OPACITY = 0.3            # Transparency for hull
    
    # Feature edge detection angle (degrees) - edges sharper than this are shown
    # This is the dihedral angle threshold - faces meeting at angles > (180 - this) are detected
    # 45° means edges where faces meet at 135°+ (i.e., 45° corners and sharper)
    # This catches chamfers, bevels, and hard edges while ignoring smooth curves
    FEATURE_EDGE_ANGLE = 45.0  # Detect 45° corners and sharper
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mesh: Optional[trimesh.Trimesh] = None
        self._pv_mesh: Optional[pv.PolyData] = None
        self._actor = None
        self._grid_actor = None
        self._scale_actors = []  # List of scale ruler actors
        self._show_edges = False  # Feature edges off by default (no cel shading)
        self._current_color = self.MESH_COLOR
        self._feature_edges_actor = None  # Separate actor for feature edges
        self._silhouette_actor = None     # Dynamic silhouette edges (view-dependent)
        
        # Parting direction visualization
        self._arrow_actors: List = []  # List of arrow actors
        self._parting_d1: Optional[np.ndarray] = None
        self._parting_d2: Optional[np.ndarray] = None
        self._visibility_paint_active = False
        self._visibility_paint_showing = False  # Whether paint is currently displayed
        self._stored_face_colors: Optional[np.ndarray] = None  # Stored face colors for toggling
        self._original_scalars = None  # Store original mesh colors
        
        # Pouring direction visualization
        self._pouring_arrow_actors: List = []  # List of pouring direction arrow actors
        self._pouring_s1: Optional[np.ndarray] = None
        self._pouring_s2: Optional[np.ndarray] = None
        self._pouring_resin: Optional[np.ndarray] = None
        self._resin_maxima_actor = None  # Spheres at bubble-trapping maxima (yellow)
        self._resin_maxima_visible = True
        self._resin_global_max_actor = None  # Sphere at global maximum (red)
        self._resin_global_max_visible = True
        
        # H1/H2 split mesh visualization for pouring direction analysis
        self._h1_split_mesh_actor = None
        self._h2_split_mesh_actor = None
        self._split_mesh_visible = True
        
        # Inflated hull visualization
        self._hull_mesh: Optional[trimesh.Trimesh] = None
        self._hull_actor = None
        self._original_hull_actor = None
        self._hull_visible = True
        self._original_hull_visible = False  # Wireframe of original hull
        
        # Mold halves classification visualization (painted on tet boundary mesh)
        self._mold_halves_actor = None  # Seed edges actor (unused, kept for compatibility)
        self._mold_halves_visible = True
        self._mold_halves_edges_actor = None  # Outer boundary edges (H1/H2/boundary)
        self._outer_boundary_visible = True  # Visibility for outer boundary
        self._inner_boundary_actor = None  # Inner boundary mesh (adjacent to part)
        self._inner_boundary_visible = True  # Visibility for inner boundary
        
        # Part surface reference visualization (for debugging)
        self._part_surface_actor = None
        self._part_surface_visible = True
        
        # Tetrahedral mesh visualization
        self._tet_edges_actor = None  # Combined edges (when not using split view)
        self._tet_interior_actor = None  # Interior edges (weight colored)
        self._tet_boundary_actor = None  # Boundary edges (mold-half colored)
        self._tet_visible = True
        self._tet_interior_visible = True
        self._tet_boundary_visible = True
        
        # Dijkstra result visualization (tetrahedra colored by escape label)
        self._dijkstra_actor = None  # H1/H2 edges (green/orange)
        self._dijkstra_primary_cuts_actor = None  # Primary cuts only (yellow)
        self._dijkstra_visible = True
        self._dijkstra_show_h1h2 = True  # Toggle for H1/H2 edges visibility
        
        # Secondary cuts visualization (edges in red)
        self._secondary_cuts_actor = None
        self._secondary_cuts_visible = True
        
        # Primary parting surface visualization (blue)
        self._parting_surface_mesh: Optional[trimesh.Trimesh] = None
        self._parting_surface_actor = None
        self._parting_surface_visible = True
        
        # Secondary parting surface visualization (red)
        self._secondary_parting_surface_mesh: Optional[trimesh.Trimesh] = None
        self._secondary_parting_surface_actor = None
        self._secondary_parting_surface_visible = True
        
        # Inflated boundary mesh visualization
        self._inflated_boundary_actor = None
        self._inflated_boundary_visible = True
        
        # R distance line visualization (max hull-to-part distance)
        self._r_line_actor = None
        self._r_point_actors = []  # Spheres at endpoints
        self._r_label_actor = None
        
        # Edge debug mode for parting surface
        self._edge_debug_mode = False
        self._boundary_edges_actor = None  # Highlight boundary edges
        self._selected_edge_actor = None  # Currently selected edge
        self._boundary_edges_data = None  # Store edge data for picking
        self._part_mesh_ref = None  # Reference to part mesh for distance calculations
        self._tet_result_ref = None  # Reference to tet result for escape label analysis
        self._parting_surface_result_ref = None  # Reference to parting surface result for vertex_to_edge mapping
        
        # Triangle debug mode for membrane analysis
        self._triangle_debug_mode = False
        self._selected_triangle_actor = None  # Currently selected triangle highlight
        self._triangle_debug_mesh_ref = None  # Reference to the mesh being debugged
        self._triangle_debug_boundary_type_ref = None  # Vertex boundary types for debug
        
        # Feature debug visualization (sharp edges/corners)
        self._feature_debug_sharp_edges_actor = None  # Lines showing sharp edges on target mesh
        self._feature_debug_corners_actor = None  # Spheres at convex corner vertices (red)
        self._feature_debug_concave_corners_actor = None  # Spheres at concave corner vertices (purple)
        self._feature_debug_sharp_edge_verts_actor = None  # Spheres at convex sharp edge vertices (cyan)
        self._feature_debug_concave_edge_verts_actor = None  # Spheres at concave sharp edge vertices (orange)
        self._feature_debug_membrane_smooth_actor = None  # Membrane boundary verts -> smooth (green)
        self._feature_debug_membrane_edge_actor = None  # Membrane boundary verts -> convex sharp edge (cyan)
        self._feature_debug_membrane_corner_actor = None  # Membrane boundary verts -> convex corner (red)
        self._feature_debug_membrane_concave_edge_actor = None  # Membrane boundary verts -> concave edge (orange)
        self._feature_debug_membrane_concave_corner_actor = None  # Membrane boundary verts -> concave corner (magenta, FIXED)
        self._feature_debug_restored_corners_actor = None  # Restored corner vertices after smoothing (blue)
        self._feature_debug_visible = True
        
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
        
        # Use trackball camera style with remapped controls:
        # - Left click: Rotate (default)
        # - Middle click: Zoom (remapped from pan)
        # - Right click: Pan (remapped from zoom)
        try:
            import vtkmodules.vtkInteractionStyle as vtk_style
            style = vtk_style.vtkInteractorStyleTrackballCamera()
            self.plotter.interactor.SetInteractorStyle(style)
            
            # Remap mouse buttons using VTK's observer pattern
            # Store reference to style for the callbacks
            self._vtk_style = style
            
            def on_middle_button_down(obj, event):
                # Middle button does zoom/dolly instead of pan
                style.StartDolly()
            
            def on_middle_button_up(obj, event):
                style.EndDolly()
            
            def on_right_button_down(obj, event):
                # Right button does pan instead of zoom/dolly
                style.StartPan()
            
            def on_right_button_up(obj, event):
                style.EndPan()
            
            # Remove default middle and right button behavior and add our own
            style.AddObserver("MiddleButtonPressEvent", on_middle_button_down)
            style.AddObserver("MiddleButtonReleaseEvent", on_middle_button_up)
            style.AddObserver("RightButtonPressEvent", on_right_button_down)
            style.AddObserver("RightButtonReleaseEvent", on_right_button_up)
            
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
            color='#333333',  # Dark gray for light theme
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
        Set up soft, even lighting for good model visibility.
        Uses balanced fill lights from all directions for minimal shadows.
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove default lights
        self.plotter.remove_all_lights()
        
        # Use balanced lighting from multiple directions for soft shadows
        # All lights at higher intensity for brighter scene
        
        # Front-top-right
        light1 = pv.Light(
            position=(10, 10, 10),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.8,
        )
        light1.positional = False
        self.plotter.add_light(light1)
        
        # Back-top-left
        light2 = pv.Light(
            position=(-10, 10, -10),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.8,
        )
        light2.positional = False
        self.plotter.add_light(light2)
        
        # Front-bottom-left
        light3 = pv.Light(
            position=(-10, -5, 10),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.8,
        )
        light3.positional = False
        self.plotter.add_light(light3)
        
        # Back-bottom-right
        light4 = pv.Light(
            position=(10, -5, -10),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.8,
        )
        light4.positional = False
        self.plotter.add_light(light4)
        
        # Top light
        light5 = pv.Light(
            position=(0, 15, 0),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.7,
        )
        light5.positional = False
        self.plotter.add_light(light5)
        
        # Bottom fill
        light6 = pv.Light(
            position=(0, -10, 0),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.7,
        )
        light6.positional = False
        self.plotter.add_light(light6)
    
    def _add_grid_plane(self, mesh_bounds=None):
        """
        Add a grid plane matching the React/Three.js frontend.
        
        Grid is centered around the mesh and uses GRID_COLOR_LINES for lines.
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove existing grid if any
        if hasattr(self, '_grid_actor') and self._grid_actor is not None:
            self.plotter.remove_actor(self._grid_actor)
            self._grid_actor = None
        
        # Calculate grid size and center based on mesh bounds
        if mesh_bounds is not None:
            max_extent = max(
                mesh_bounds[1] - mesh_bounds[0],  # X extent
                mesh_bounds[3] - mesh_bounds[2],  # Y extent
                mesh_bounds[5] - mesh_bounds[4],  # Z extent
            )
            grid_size = max_extent * 2  # Grid is 2x the mesh size
            
            # Center the grid around the mesh center (X, Y)
            center_x = (mesh_bounds[0] + mesh_bounds[1]) / 2
            center_y = (mesh_bounds[2] + mesh_bounds[3]) / 2
            z_pos = mesh_bounds[4]  # Bottom of mesh
        else:
            grid_size = 200  # Default size
            center_x = 0
            center_y = 0
            z_pos = 0
        
        # Number of grid divisions
        n_cells = 20
        
        # Create the grid centered around the mesh
        grid_mesh = pv.Plane(
            center=(center_x, center_y, z_pos),
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
            line_width=1.5,
            opacity=0.6,
            render_lines_as_tubes=False,
            ambient=1.0,  # Full ambient for unshaded lines
            diffuse=0.0,
            specular=0.0,
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
        
        # Dynamically determine tick intervals to limit total tick count
        # Target ~10-20 ticks maximum regardless of mesh size
        MAX_TICKS = 20
        
        # Calculate appropriate tick interval based on length
        if length_mm <= 0:
            return
        
        # Find a "nice" tick interval that gives us roughly MAX_TICKS ticks
        raw_interval = length_mm / MAX_TICKS
        
        # Round to a nice number (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, etc.)
        magnitude = 10 ** np.floor(np.log10(raw_interval)) if raw_interval > 0 else 1
        normalized = raw_interval / magnitude
        
        if normalized <= 1:
            nice_interval = 1 * magnitude
        elif normalized <= 2:
            nice_interval = 2 * magnitude
        elif normalized <= 5:
            nice_interval = 5 * magnitude
        else:
            nice_interval = 10 * magnitude
        
        tick_interval = max(1, nice_interval)
        major_tick_interval = tick_interval * 5  # Every 5th tick is major
        
        # Main ruler line
        main_line = pv.Line(start_pos, end_pos)
        actor = self.plotter.add_mesh(
            main_line,
            color=self.SCALE_COLOR,
            line_width=2,
            opacity=0.8,
        )
        self._scale_actors.append(actor)
        
        # Collect all tick mark points to create a single combined mesh
        all_tick_points = []
        all_tick_lines = []
        
        for mm in np.arange(0, length_mm + tick_interval, tick_interval):
            if mm > length_mm:
                mm = length_mm
            
            tick_pos = start_pos + dir_vec * mm
            is_major = (mm % major_tick_interval < 0.1) or mm == 0 or abs(mm - length_mm) < 0.1
            tick_length = ruler_scale * 2 if is_major else ruler_scale * 1
            
            tick_start = tick_pos + perp_vec * (-tick_length / 2)
            tick_end = tick_pos + perp_vec * (tick_length / 2)
            all_tick_points.extend([tick_start, tick_end])
            all_tick_lines.append([len(all_tick_points) - 2, len(all_tick_points) - 1])
        
        # Create a single combined mesh for all tick marks
        if all_tick_points:
            points = np.array(all_tick_points)
            lines = np.array([[2, line[0], line[1]] for line in all_tick_lines]).ravel()
            tick_mesh = pv.PolyData(points, lines=lines)
            
            actor = self.plotter.add_mesh(
                tick_mesh,
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
        """Render the current mesh with fixture-view style shading."""
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Clear previous content
        self.plotter.clear()
        self._feature_edges_actor = None
        self._silhouette_actor = None
        self._silhouette_filter = None
        # Hull actor is invalidated by clear(), reset reference
        self._hull_actor = None
        self._original_hull_actor = None
        
        # Set up lighting after clear
        self._setup_lighting()
        
        # Add the mesh with fixture-view style material properties
        # roughness=0.6, metalness=0.1 matches the DEFAULT_MATERIAL_CONFIG
        self._actor = self.plotter.add_mesh(
            self._pv_mesh,
            color=self._current_color,
            pbr=True,
            metallic=0.1,              # Low metallic for plastic/resin look
            roughness=0.6,             # Higher roughness = less shiny
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
        )
        
        # Add grid plane (like Three.js GridHelper)
        if self._pv_mesh is not None:
            self._add_grid_plane(self._pv_mesh.bounds)
            
            # Add scale rulers on X and Y axes
            self._add_scale_rulers(self._pv_mesh.bounds)
        
        # Add axes
        self.plotter.add_axes(
            line_width=2,
            color='#333333',  # Dark gray for light theme
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
        
        # Ensure orthographic projection is maintained
        self.plotter.enable_parallel_projection()
        
        # Re-add parting direction arrows if they exist (they get cleared by plotter.clear())
        if self._parting_d1 is not None and self._parting_d2 is not None:
            # Clear the stale actor references
            self._arrow_actors = []
            # Re-add arrows
            self.add_parting_direction_arrows(self._parting_d1, self._parting_d2)
        
        # Re-add pouring direction arrows if they exist (they get cleared by plotter.clear())
        if self._pouring_s1 is not None and self._pouring_s2 is not None and self._pouring_resin is not None:
            # Clear the stale actor references
            self._pouring_arrow_actors = []
            # Re-add arrows
            self.add_pouring_direction_arrows(self._pouring_s1, self._pouring_s2, self._pouring_resin)
        
        # Re-add hull actor if it exists (gets cleared by plotter.clear())
        self._re_add_hull_actor()
        
        # Update the view
        self.plotter.update()
    
    def _re_add_hull_actor(self):
        """
        Re-add hull actor after plotter.clear().
        
        When _render_mesh() is called (e.g., loading new mesh, color change),
        plotter.clear() removes all actors including hull. This method
        re-adds it from the stored mesh data if available.
        
        NOTE: The new _update_mesh_display() method avoids clearing the scene,
        so hull actors are preserved when toggling visibility paint.
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Re-add hull mesh if it exists
        if self._hull_mesh is not None:
            pv_hull = self._trimesh_to_pyvista(self._hull_mesh)
            self._hull_actor = self.plotter.add_mesh(
                pv_hull,
                color=self.HULL_COLOR,
                opacity=self.HULL_OPACITY,
                pbr=True,
                metallic=0.1,
                roughness=0.6,
                smooth_shading=True,
                show_edges=False,
                style='surface',
            )
            if self._hull_actor is not None:
                # Restore visibility state
                self._hull_actor.SetVisibility(self._hull_visible)
    
    def _add_feature_edges(self):
        """
        Add feature edges for cel shading effect.
        
        For clean cel-shading without internal lines on curved surfaces:
        - Only show VERY sharp feature edges (near 90° corners)
        - Show boundary edges (open edges on non-watertight meshes)
        - Skip dynamic silhouette (it creates unwanted lines on curves)
        
        This gives clean outlines on engineering models without
        drawing internal contour lines on cylinders/spheres.
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        if not self._show_edges:
            return
        
        try:
            # Extract ONLY very sharp feature edges and boundary edges
            # Use a very high angle (85°+) to only catch true corners
            # This avoids detecting edges on curved surfaces
            feature_edges = self._pv_mesh.extract_feature_edges(
                feature_angle=self.FEATURE_EDGE_ANGLE,  # Very high - only sharp corners
                boundary_edges=True,      # Include boundary/open edges
                non_manifold_edges=True,  # Include non-manifold edges
                feature_edges=True,       # Include sharp feature edges
                manifold_edges=False,     # Don't include smooth manifold edges
            )
            
            if feature_edges.n_points > 0:
                # Render feature edges as black lines
                self._feature_edges_actor = self.plotter.add_mesh(
                    feature_edges,
                    color=self.EDGE_COLOR,
                    line_width=2.0,         # Good thickness for outline
                    opacity=1.0,
                    render_lines_as_tubes=False,
                    ambient=1.0,            # Full ambient (unlit)
                    diffuse=0.0,
                    specular=0.0,
                )
                logger.debug(f"Added {feature_edges.n_lines} feature edges")
        except Exception as e:
            logger.warning(f"Could not extract feature edges: {e}")
    
    def _add_silhouette_edges(self):
        """
        Add dynamic silhouette edges that update with camera movement.
        
        Silhouette edges are the TRUE outline of the object from the current
        viewpoint - essential for organic shapes that lack sharp features.
        Uses VTK's vtkPolyDataSilhouette filter with contour edges only
        (no border edges to avoid internal line artifacts).
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        try:
            import vtkmodules.vtkFiltersHybrid as vtk_hybrid
            
            # Create silhouette filter
            silhouette = vtk_hybrid.vtkPolyDataSilhouette()
            silhouette.SetInputData(self._pv_mesh)
            
            # Connect to renderer's active camera for dynamic updates
            renderer = self.plotter.renderer
            silhouette.SetCamera(renderer.GetActiveCamera())
            
            # IMPORTANT: Only use contour edges (true silhouette outline)
            # BorderEdgesOff prevents internal edge artifacts
            silhouette.SetEnableFeatureAngle(False)  # We handle features separately
            silhouette.BorderEdgesOff()  # Disable border edges - these cause internal lines
            silhouette.PieceInvariantOn()
            
            # Update the filter
            silhouette.Update()
            
            # Get the silhouette output
            silhouette_output = silhouette.GetOutput()
            
            if silhouette_output and silhouette_output.GetNumberOfLines() > 0:
                # Wrap in PyVista and add to plotter
                silhouette_mesh = pv.wrap(silhouette_output)
                
                self._silhouette_actor = self.plotter.add_mesh(
                    silhouette_mesh,
                    color=self.EDGE_COLOR,
                    line_width=2.5,         # Thicker for main outline
                    opacity=1.0,
                    render_lines_as_tubes=False,
                    ambient=1.0,
                    diffuse=0.0,
                    specular=0.0,
                )
                
                # Store the silhouette filter for dynamic updates
                self._silhouette_filter = silhouette
                
                logger.debug(f"Added silhouette edges (contour only)")
        except ImportError:
            logger.debug("VTK silhouette filter not available, using feature edges only")
        except Exception as e:
            logger.warning(f"Could not add silhouette edges: {e}")
    
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
        """Clear the viewer and reset all state."""
        if not PYVISTA_AVAILABLE:
            return
        
        # Reset main mesh state
        self._mesh = None
        self._pv_mesh = None
        self._actor = None
        self._grid_actor = None
        self._scale_actors = []
        self._feature_edges_actor = None
        self._silhouette_actor = None
        self._silhouette_filter = None
        
        # Reset parting direction state
        self._parting_d1 = None
        self._parting_d2 = None
        self._arrow_actors = []
        self._visibility_paint_active = False
        self._visibility_paint_showing = False
        self._stored_face_colors = None
        self._original_scalars = None
        
        # Reset pouring direction state
        self._pouring_s1 = None
        self._pouring_s2 = None
        self._pouring_resin = None
        self._pouring_arrow_actors = []
        
        # Reset hull state
        self._hull_mesh = None
        self._hull_actor = None
        self._original_hull_actor = None
        self._hull_visible = True
        self._original_hull_visible = False
        
        # Reset mold halves state
        self._mold_halves_actor = None
        self._mold_halves_visible = True
        self._mold_halves_edges_actor = None
        self._inner_boundary_actor = None
        self._inner_boundary_visible = True
        
        # Reset part surface reference state
        self._part_surface_actor = None
        self._part_surface_visible = True
        
        # Reset tetrahedral mesh state
        self._tet_edges_actor = None
        self._tet_interior_actor = None
        self._tet_boundary_actor = None
        self._tet_visible = True
        self._tet_interior_visible = True
        self._tet_boundary_visible = True
        
        # Reset Dijkstra visualization state
        self._dijkstra_actor = None
        self._dijkstra_visible = True
        
        # Reset secondary cuts visualization state
        self._secondary_cuts_actor = None
        self._secondary_cuts_visible = True
        
        # Reset inflated boundary mesh state
        self._inflated_boundary_actor = None
        self._inflated_boundary_visible = True
        
        # Reset R distance line state
        self._r_line_actor = None
        self._r_point_actors = []
        self._r_label_actor = None
        
        self.plotter.clear()
        self._setup_lighting()
        self.plotter.add_axes(
            line_width=2,
            color='#333333',  # Dark gray for light theme
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
        """Toggle feature/silhouette edge visibility for cel shading outline."""
        if not PYVISTA_AVAILABLE:
            return
        
        self._show_edges = show
        
        # Toggle feature edges visibility
        if self._feature_edges_actor is not None:
            self._feature_edges_actor.SetVisibility(show)
        
        # Toggle silhouette edges visibility
        if self._silhouette_actor is not None:
            self._silhouette_actor.SetVisibility(show)
        
        if show and self._pv_mesh is not None:
            # Re-add edges if they don't exist
            if self._feature_edges_actor is None and self._silhouette_actor is None:
                self._add_feature_edges()
        
        self.plotter.update()
    
    def set_mesh_visible(self, visible: bool):
        """
        Set mesh visibility (show/hide).
        
        Args:
            visible: True to show mesh, False to hide
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Toggle main mesh actor
        if self._actor is not None:
            self._actor.SetVisibility(visible)
        
        # Toggle feature edges (silhouette) - they should follow mesh visibility
        if self._feature_edges_actor is not None:
            self._feature_edges_actor.SetVisibility(visible and self._show_edges)
        
        if self._silhouette_actor is not None:
            self._silhouette_actor.SetVisibility(visible and self._show_edges)
        
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
    
    # ========================================================================
    # PARTING DIRECTION VISUALIZATION
    # ========================================================================
    
    def add_parting_direction_arrows(
        self,
        d1: np.ndarray,
        d2: np.ndarray,
        arrow_length: Optional[float] = None
    ):
        """
        Add parting direction arrows to the viewer.
        
        Creates two arrows at the mesh center showing the optimal
        parting directions for a two-piece mold.
        
        Args:
            d1: Primary direction (3,) unit vector - shown in green
            d2: Secondary direction (3,) unit vector - shown in orange
            arrow_length: Optional arrow length (default: based on mesh size)
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Store directions for later use
        self._parting_d1 = d1.copy()
        self._parting_d2 = d2.copy()
        
        # Remove existing arrows
        self.remove_parting_direction_arrows()
        
        # Calculate arrow dimensions based on mesh bounds
        bounds = self._pv_mesh.bounds
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        
        size = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ])
        max_size = np.max(size)
        min_size = np.min(size[size > 0]) if np.any(size > 0) else max_size
        
        # Scale arrow proportionally to mesh size
        if arrow_length is None:
            arrow_length = max_size * 0.8  # Slightly smaller than before
        
        # Scale arrow thickness based on bounding box - thinner arrows
        # Use a factor that makes arrows visible but not overwhelming
        thickness_factor = max_size / 100.0  # Base scale factor
        shaft_radius = max(0.005, min(0.015, thickness_factor * 0.5))  # Thin shaft
        tip_radius = shaft_radius * 2.5  # Tip is 2.5x shaft width
        tip_length = 0.15  # 15% of arrow length for the tip
        
        # Create D1 arrow (green) - Primary direction
        arrow1 = pv.Arrow(
            start=center,
            direction=d1,
            scale=arrow_length,
            tip_length=tip_length,
            tip_radius=tip_radius,
            shaft_radius=shaft_radius,
            tip_resolution=16,
            shaft_resolution=16,
        )
        
        # Add D1 arrow with PBR shading
        actor1 = self.plotter.add_mesh(
            arrow1,
            color=self.PARTING_D1_COLOR,
            opacity=1.0,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
        )
        self._arrow_actors.append(actor1)
        
        # Create D2 arrow (orange) - Secondary direction
        arrow2 = pv.Arrow(
            start=center,
            direction=d2,
            scale=arrow_length,
            tip_length=tip_length,
            tip_radius=tip_radius,
            shaft_radius=shaft_radius,
            tip_resolution=16,
            shaft_resolution=16,
        )
        
        # Add D2 arrow with PBR shading
        actor2 = self.plotter.add_mesh(
            arrow2,
            color=self.PARTING_D2_COLOR,
            opacity=1.0,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
        )
        self._arrow_actors.append(actor2)
        
        # Add labels at arrow tips
        d1_tip = center + d1 * arrow_length * 1.05
        d2_tip = center + d2 * arrow_length * 1.05
        
        self.plotter.add_point_labels(
            [d1_tip],
            ['D1'],
            font_size=12,
            text_color=self.PARTING_D1_COLOR,
            font_family='arial',
            bold=True,
            show_points=False,
            always_visible=True,
            shape=None,
            fill_shape=False,
        )
        
        self.plotter.add_point_labels(
            [d2_tip],
            ['D2'],
            font_size=12,
            text_color=self.PARTING_D2_COLOR,
            font_family='arial',
            bold=True,
            show_points=False,
            always_visible=True,
            shape=None,
            fill_shape=False,
        )
        
        self.plotter.update()
        logger.info(f"Added parting direction arrows: D1={d1}, D2={d2}")
    
    def remove_parting_direction_arrows(self):
        """Remove parting direction arrows from the viewer."""
        if not PYVISTA_AVAILABLE:
            return
        
        for actor in self._arrow_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception as e:
                logger.debug(f"Could not remove arrow actor: {e}")
        
        self._arrow_actors = []
        self.plotter.update()
    
    def set_parting_arrows_visible(self, visible: bool):
        """
        Set visibility of parting direction arrows.
        
        Args:
            visible: True to show arrows, False to hide
        """
        if not PYVISTA_AVAILABLE:
            return
        
        for actor in self._arrow_actors:
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass
        
        self.plotter.update()
    
    # ========================================================================
    # POURING DIRECTION VISUALIZATION
    # ========================================================================
    
    def add_pouring_direction_arrows(
        self,
        silicone_1: np.ndarray,
        silicone_2: np.ndarray,
        resin: np.ndarray,
        arrow_length: Optional[float] = None
    ):
        """
        Add pouring direction arrows to the viewer.
        
        Creates three arrows at the mesh center showing the optimal
        pouring directions for silicone molds and resin casting.
        Uses distinct colors from parting directions:
        - Silicone 1: Cyan
        - Silicone 2: Magenta
        - Resin: Red
        
        Args:
            silicone_1: First silicone pouring direction (3,) unit vector
            silicone_2: Second silicone pouring direction (3,) unit vector
            resin: Resin pouring direction (3,) unit vector
            arrow_length: Optional arrow length (default: based on mesh size)
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Store directions for later use
        self._pouring_s1 = silicone_1.copy()
        self._pouring_s2 = silicone_2.copy()
        self._pouring_resin = resin.copy()
        
        # Remove existing pouring arrows
        self.remove_pouring_direction_arrows()
        
        # Calculate arrow dimensions based on mesh bounds
        bounds = self._pv_mesh.bounds
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        
        size = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ])
        max_size = np.max(size)
        
        # Scale arrow proportionally to mesh size (slightly smaller than parting arrows)
        if arrow_length is None:
            arrow_length = max_size * 0.6
        
        # Scale arrow thickness based on bounding box
        thickness_factor = max_size / 100.0
        shaft_radius = max(0.004, min(0.012, thickness_factor * 0.4))
        tip_radius = shaft_radius * 2.5
        tip_length = 0.15
        
        # Offset arrows slightly from center to avoid overlap with parting arrows
        offset = max_size * 0.05
        
        # Create Silicone 1 arrow (Cyan)
        arrow1 = pv.Arrow(
            start=center + np.array([offset, 0, 0]),
            direction=silicone_1,
            scale=arrow_length,
            tip_length=tip_length,
            tip_radius=tip_radius,
            shaft_radius=shaft_radius,
            tip_resolution=16,
            shaft_resolution=16,
        )
        
        actor1 = self.plotter.add_mesh(
            arrow1,
            color=self.POURING_S1_COLOR,
            opacity=1.0,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
        )
        self._pouring_arrow_actors.append(actor1)
        
        # Create Silicone 2 arrow (Magenta)
        arrow2 = pv.Arrow(
            start=center - np.array([offset, 0, 0]),
            direction=silicone_2,
            scale=arrow_length,
            tip_length=tip_length,
            tip_radius=tip_radius,
            shaft_radius=shaft_radius,
            tip_resolution=16,
            shaft_resolution=16,
        )
        
        actor2 = self.plotter.add_mesh(
            arrow2,
            color=self.POURING_S2_COLOR,
            opacity=1.0,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
        )
        self._pouring_arrow_actors.append(actor2)
        
        # Create Resin arrow (Red)
        arrow3 = pv.Arrow(
            start=center,
            direction=resin,
            scale=arrow_length,
            tip_length=tip_length,
            tip_radius=tip_radius,
            shaft_radius=shaft_radius,
            tip_resolution=16,
            shaft_resolution=16,
        )
        
        actor3 = self.plotter.add_mesh(
            arrow3,
            color=self.POURING_RESIN_COLOR,
            opacity=1.0,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
        )
        self._pouring_arrow_actors.append(actor3)
        
        # Add labels at arrow tips
        s1_tip = center + np.array([offset, 0, 0]) + silicone_1 * arrow_length * 1.05
        s2_tip = center - np.array([offset, 0, 0]) + silicone_2 * arrow_length * 1.05
        resin_tip = center + resin * arrow_length * 1.05
        
        self.plotter.add_point_labels(
            [s1_tip],
            ['S1'],
            font_size=10,
            text_color=self.POURING_S1_COLOR,
            font_family='arial',
            bold=True,
            show_points=False,
            always_visible=True,
            shape=None,
            fill_shape=False,
        )
        
        self.plotter.add_point_labels(
            [s2_tip],
            ['S2'],
            font_size=10,
            text_color=self.POURING_S2_COLOR,
            font_family='arial',
            bold=True,
            show_points=False,
            always_visible=True,
            shape=None,
            fill_shape=False,
        )
        
        self.plotter.add_point_labels(
            [resin_tip],
            ['R'],
            font_size=10,
            text_color=self.POURING_RESIN_COLOR,
            font_family='arial',
            bold=True,
            show_points=False,
            always_visible=True,
            shape=None,
            fill_shape=False,
        )
        
        self.plotter.update()
        logger.info(f"Added pouring direction arrows: S1={silicone_1}, S2={silicone_2}, Resin={resin}")
    
    def remove_pouring_direction_arrows(self):
        """Remove pouring direction arrows from the viewer."""
        if not PYVISTA_AVAILABLE:
            return
        
        for actor in self._pouring_arrow_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception as e:
                logger.debug(f"Could not remove pouring arrow actor: {e}")
        
        self._pouring_arrow_actors = []
        self.plotter.update()
    
    def add_resin_maxima_points(
        self,
        positions: np.ndarray,
        sphere_radius: Optional[float] = None
    ):
        """
        Add yellow spheres at the bubble-trapping maxima positions.
        
        These are the local maxima from the persistence analysis that
        identify where air bubbles would accumulate during resin casting.
        
        Args:
            positions: (N, 3) array of 3D positions of maxima
            sphere_radius: Optional radius for spheres (default: based on mesh size)
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        if positions is None or len(positions) == 0:
            logger.info("No resin maxima positions to display")
            return
        
        # Remove existing maxima visualization
        self.remove_resin_maxima_points()
        
        # Calculate sphere radius based on mesh bounds
        bounds = self._pv_mesh.bounds
        size = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ])
        max_size = np.max(size)
        
        if sphere_radius is None:
            # Scale sphere radius to be visible but not too large
            sphere_radius = max_size * 0.015  # 1.5% of mesh size
        
        # Create spheres at each maximum position
        spheres = pv.PolyData()
        for pos in positions:
            sphere = pv.Sphere(radius=sphere_radius, center=pos)
            spheres = spheres.merge(sphere)
        
        # Add to plotter with yellow color
        actor = self.plotter.add_mesh(
            spheres,
            color='yellow',
            opacity=1.0,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
        )
        
        self._resin_maxima_actor = actor
        self._resin_maxima_visible = True
        self.plotter.update()
        
        logger.info(f"Added {len(positions)} yellow spheres at resin bubble-trapping maxima")
    
    def remove_resin_maxima_points(self):
        """Remove resin maxima spheres from the viewer."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._resin_maxima_actor is not None:
            try:
                self.plotter.remove_actor(self._resin_maxima_actor)
            except Exception as e:
                logger.debug(f"Could not remove resin maxima actor: {e}")
            self._resin_maxima_actor = None
        
        self.plotter.update()
    
    def set_resin_maxima_visible(self, visible: bool):
        """
        Set visibility of resin maxima spheres.
        
        Args:
            visible: True to show spheres, False to hide
        """
        if not PYVISTA_AVAILABLE or self._resin_maxima_actor is None:
            return
        
        try:
            self._resin_maxima_actor.SetVisibility(visible)
            self._resin_maxima_visible = visible
        except Exception:
            pass
        
        self.plotter.update()
    
    def add_resin_global_maximum_point(
        self,
        position: np.ndarray,
        sphere_radius: Optional[float] = None
    ):
        """
        Add a red sphere at the global maximum position.
        
        The global maximum is the highest local maximum in the resin
        pouring direction - the most critical bubble trap location.
        
        Args:
            position: (3,) array of 3D position of the global maximum
            sphere_radius: Optional radius for sphere (default: based on mesh size)
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        if position is None:
            logger.info("No global maximum position to display")
            return
        
        # Remove existing global max visualization
        self.remove_resin_global_maximum_point()
        
        # Calculate sphere radius based on mesh bounds
        bounds = self._pv_mesh.bounds
        size = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ])
        max_size = np.max(size)
        
        if sphere_radius is None:
            # Slightly larger than local maxima spheres for emphasis
            sphere_radius = max_size * 0.022  # 2.2% of mesh size
        
        # Create sphere at global maximum position
        sphere = pv.Sphere(radius=sphere_radius, center=position)
        
        # Add to plotter with red color
        actor = self.plotter.add_mesh(
            sphere,
            color='red',
            opacity=1.0,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
        )
        
        self._resin_global_max_actor = actor
        self._resin_global_max_visible = True
        self.plotter.update()
        
        logger.info(f"Added red sphere at resin global maximum: {position}")
    
    def remove_resin_global_maximum_point(self):
        """Remove resin global maximum sphere from the viewer."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._resin_global_max_actor is not None:
            try:
                self.plotter.remove_actor(self._resin_global_max_actor)
            except Exception as e:
                logger.debug(f"Could not remove resin global max actor: {e}")
            self._resin_global_max_actor = None
        
        self.plotter.update()
    
    def set_resin_global_maximum_visible(self, visible: bool):
        """
        Set visibility of resin global maximum sphere.
        
        Args:
            visible: True to show sphere, False to hide
        """
        if not PYVISTA_AVAILABLE or self._resin_global_max_actor is None:
            return
        
        try:
            self._resin_global_max_actor.SetVisibility(visible)
            self._resin_global_max_visible = visible
        except Exception:
            pass
        
        self.plotter.update()
    
    def set_pouring_arrows_visible(self, visible: bool):
        """
        Set visibility of pouring direction arrows.
        
        Args:
            visible: True to show arrows, False to hide
        """
        if not PYVISTA_AVAILABLE:
            return
        
        for actor in self._pouring_arrow_actors:
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass
        
        self.plotter.update()
    
    def show_split_meshes_for_pouring(
        self,
        h1_mesh: Optional[trimesh.Trimesh],
        h2_mesh: Optional[trimesh.Trimesh]
    ):
        """
        Display the H1 and H2 split meshes used for pouring direction analysis.
        
        This visualization helps verify that the mesh is correctly split by mold half
        before computing silicone pouring directions.
        
        Args:
            h1_mesh: The H1 mold half portion of the part surface (cyan)
            h2_mesh: The H2 mold half portion of the part surface (magenta)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove existing split mesh actors
        self.remove_split_meshes_for_pouring()
        
        # Hide the main mesh to show split meshes clearly
        if self._actor is not None:
            self._actor.SetVisibility(False)
        
        # Add H1 mesh (cyan - same as S1 arrow color)
        if h1_mesh is not None and len(h1_mesh.faces) > 0:
            h1_pv = self._trimesh_to_pyvista(h1_mesh)
            self._h1_split_mesh_actor = self.plotter.add_mesh(
                h1_pv,
                color=self.POURING_S1_COLOR,  # Cyan
                opacity=0.9,
                pbr=True,
                metallic=0.1,
                roughness=0.6,
                smooth_shading=True,
                show_edges=True,
                edge_color='#006666',
                line_width=0.5,
            )
            logger.info(f"Added H1 split mesh: {len(h1_mesh.faces)} faces (cyan)")
        
        # Add H2 mesh (magenta - same as S2 arrow color)
        if h2_mesh is not None and len(h2_mesh.faces) > 0:
            h2_pv = self._trimesh_to_pyvista(h2_mesh)
            self._h2_split_mesh_actor = self.plotter.add_mesh(
                h2_pv,
                color=self.POURING_S2_COLOR,  # Magenta
                opacity=0.9,
                pbr=True,
                metallic=0.1,
                roughness=0.6,
                smooth_shading=True,
                show_edges=True,
                edge_color='#660066',
                line_width=0.5,
            )
            logger.info(f"Added H2 split mesh: {len(h2_mesh.faces)} faces (magenta)")
        
        self._split_mesh_visible = True
        self.plotter.update()
    
    def remove_split_meshes_for_pouring(self):
        """Remove the H1/H2 split mesh visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._h1_split_mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._h1_split_mesh_actor)
            except Exception:
                pass
            self._h1_split_mesh_actor = None
        
        if self._h2_split_mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._h2_split_mesh_actor)
            except Exception:
                pass
            self._h2_split_mesh_actor = None
        
        # Restore main mesh visibility
        if self._actor is not None:
            self._actor.SetVisibility(True)
        
        self._split_mesh_visible = False
        self.plotter.update()
    
    def set_split_meshes_visible(self, visible: bool):
        """
        Toggle visibility of the H1/H2 split meshes.
        
        Args:
            visible: True to show split meshes (hides main mesh), 
                    False to hide split meshes (shows main mesh)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._split_mesh_visible = visible
        
        if self._h1_split_mesh_actor is not None:
            self._h1_split_mesh_actor.SetVisibility(visible)
        if self._h2_split_mesh_actor is not None:
            self._h2_split_mesh_actor.SetVisibility(visible)
        
        # Toggle main mesh visibility opposite to split meshes
        if self._actor is not None:
            self._actor.SetVisibility(not visible)
        
        self.plotter.update()

    # =========================================================================
    # FEATURE DEBUG VISUALIZATION (Sharp Edges / Corners)
    # =========================================================================
    
    def show_feature_debug_visualization(
        self,
        debug_data,  # FeatureDebugVisualization from surface_propagation
        sphere_radius: Optional[float] = None
    ):
        """
        Show debug visualization for sharp edge/corner feature classification.
        
        This displays:
        1. Sharp edges on the target mesh as thick lines
        2. Corner vertices as red spheres
        3. Sharp edge vertices as blue spheres
        4. Membrane boundary vertices colored by their projection type:
           - Green: will use smooth surface projection
           - Blue: will use sharp edge projection
           - Red: corner (will be kept fixed)
        
        Args:
            debug_data: FeatureDebugVisualization from get_feature_debug_visualization()
            sphere_radius: Optional radius for marker spheres
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove existing debug visualization
        self.remove_feature_debug_visualization()
        
        if debug_data is None:
            return
        
        # Calculate sphere radius based on mesh bounds
        # Make spheres just slightly thicker than the edge lines
        if sphere_radius is None and self._pv_mesh is not None:
            bounds = self._pv_mesh.bounds
            size = np.array([
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ])
            max_size = np.max(size)
            sphere_radius = max_size * 0.0015  # 0.15% of mesh size - just slightly larger than edge lines
        elif sphere_radius is None:
            sphere_radius = 0.1  # Fallback (smaller)
        
        # === 1. Draw sharp edges as thick lines (use efficient line construction) ===
        if debug_data.target_sharp_edges and debug_data.target_mesh_vertices is not None:
            # Build lines efficiently using cells array
            n_edges = len(debug_data.target_sharp_edges)
            if n_edges > 0:
                # Collect unique vertices used in edges
                edge_array = np.array(debug_data.target_sharp_edges)
                points = debug_data.target_mesh_vertices[edge_array.flatten()].reshape(-1, 3)
                
                # Build cells: [2, 0, 1, 2, 2, 3, ...] for line segments
                cells = np.zeros((n_edges, 3), dtype=np.int64)
                cells[:, 0] = 2  # Each line has 2 points
                cells[:, 1] = np.arange(0, n_edges * 2, 2)  # Start indices
                cells[:, 2] = np.arange(1, n_edges * 2 + 1, 2)  # End indices
                
                lines = pv.PolyData(points, lines=cells.flatten())
                
                self._feature_debug_sharp_edges_actor = self.plotter.add_mesh(
                    lines,
                    color='yellow',
                    line_width=4,
                    render_lines_as_tubes=True,
                )
                logger.info(f"Added {n_edges} sharp edge lines")
        
        # === 2. Draw corner vertices using glyphs ===
        # Feature types: 2 = convex corner, 4 = concave corner
        if debug_data.target_feature_types is not None and debug_data.target_mesh_vertices is not None:
            # Convex corners (red) - can be smoothed
            convex_corner_mask = debug_data.target_feature_types == 2
            convex_corner_positions = debug_data.target_mesh_vertices[convex_corner_mask]
            
            if len(convex_corner_positions) > 0:
                convex_cloud = pv.PolyData(convex_corner_positions)
                convex_glyphs = convex_cloud.glyph(
                    geom=pv.Sphere(radius=sphere_radius * 1.5),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_corners_actor = self.plotter.add_mesh(
                    convex_glyphs,
                    color='red',
                    opacity=1.0,
                )
                logger.info(f"Added {len(convex_corner_positions)} CONVEX corner spheres (red)")
            
            # Concave corners (purple) - kept fixed during smoothing
            concave_corner_mask = debug_data.target_feature_types == 4
            concave_corner_positions = debug_data.target_mesh_vertices[concave_corner_mask]
            
            if len(concave_corner_positions) > 0:
                concave_cloud = pv.PolyData(concave_corner_positions)
                concave_glyphs = concave_cloud.glyph(
                    geom=pv.Sphere(radius=sphere_radius * 1.8),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_concave_corners_actor = self.plotter.add_mesh(
                    concave_glyphs,
                    color='purple',
                    opacity=1.0,
                )
                logger.info(f"Added {len(concave_corner_positions)} CONCAVE corner spheres (purple)")
        
        # === 3. Draw sharp edge vertices using glyphs ===
        # Feature types: 1 = convex sharp edge, 3 = concave sharp edge
        if debug_data.target_feature_types is not None and debug_data.target_mesh_vertices is not None:
            # Convex sharp edges (cyan)
            convex_edge_mask = debug_data.target_feature_types == 1
            convex_edge_positions = debug_data.target_mesh_vertices[convex_edge_mask]
            
            if len(convex_edge_positions) > 0:
                convex_edge_cloud = pv.PolyData(convex_edge_positions)
                convex_edge_glyphs = convex_edge_cloud.glyph(
                    geom=pv.Sphere(radius=sphere_radius),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_sharp_edge_verts_actor = self.plotter.add_mesh(
                    convex_edge_glyphs,
                    color='cyan',
                    opacity=0.8,
                )
                logger.info(f"Added {len(convex_edge_positions)} CONVEX sharp edge spheres (cyan)")
            
            # Concave sharp edges (orange)
            concave_edge_mask = debug_data.target_feature_types == 3
            concave_edge_positions = debug_data.target_mesh_vertices[concave_edge_mask]
            
            if len(concave_edge_positions) > 0:
                concave_edge_cloud = pv.PolyData(concave_edge_positions)
                concave_edge_glyphs = concave_edge_cloud.glyph(
                    geom=pv.Sphere(radius=sphere_radius),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_concave_edge_verts_actor = self.plotter.add_mesh(
                    concave_edge_glyphs,
                    color='orange',
                    opacity=0.8,
                )
                logger.info(f"Added {len(concave_edge_positions)} CONCAVE sharp edge spheres (orange)")
        
        # === 4. Draw membrane boundary vertices by projection type (using glyphs) ===
        # Feature types: 0=smooth, 1=sharp_convex, 2=corner_convex, 3=sharp_concave, 4=corner_concave
        if debug_data.membrane_boundary_positions is not None and debug_data.membrane_boundary_projected_types is not None:
            positions = debug_data.membrane_boundary_positions
            types = debug_data.membrane_boundary_projected_types
            
            membrane_radius = sphere_radius * 1.2
            
            # Smooth (green) - normal projection
            smooth_mask = types == 0
            if np.sum(smooth_mask) > 0:
                smooth_positions = positions[smooth_mask]
                smooth_cloud = pv.PolyData(smooth_positions)
                smooth_glyphs = smooth_cloud.glyph(
                    geom=pv.Sphere(radius=membrane_radius),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_membrane_smooth_actor = self.plotter.add_mesh(
                    smooth_glyphs,
                    color='lime',
                    opacity=0.9,
                )
                logger.info(f"Added {np.sum(smooth_mask)} membrane smooth spheres (green)")
            
            # Sharp edge convex (cyan) - edge projection, can be smoothed
            convex_edge_mask = types == 1
            if np.sum(convex_edge_mask) > 0:
                edge_positions = positions[convex_edge_mask]
                edge_cloud = pv.PolyData(edge_positions)
                edge_glyphs = edge_cloud.glyph(
                    geom=pv.Sphere(radius=membrane_radius),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_membrane_edge_actor = self.plotter.add_mesh(
                    edge_glyphs,
                    color='cyan',
                    opacity=0.9,
                )
                logger.info(f"Added {np.sum(convex_edge_mask)} membrane CONVEX edge spheres (cyan)")
            
            # Corner convex (red) - can be smoothed
            convex_corner_mask = types == 2
            if np.sum(convex_corner_mask) > 0:
                corner_positions = positions[convex_corner_mask]
                corner_cloud = pv.PolyData(corner_positions)
                corner_glyphs = corner_cloud.glyph(
                    geom=pv.Sphere(radius=membrane_radius * 1.3),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_membrane_corner_actor = self.plotter.add_mesh(
                    corner_glyphs,
                    color='red',
                    opacity=0.9,
                )
                logger.info(f"Added {np.sum(convex_corner_mask)} membrane CONVEX corner spheres (red)")
            
            # Sharp edge concave (orange) - edge projection
            concave_edge_mask = types == 3
            if np.sum(concave_edge_mask) > 0:
                edge_positions = positions[concave_edge_mask]
                edge_cloud = pv.PolyData(edge_positions)
                edge_glyphs = edge_cloud.glyph(
                    geom=pv.Sphere(radius=membrane_radius),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_membrane_concave_edge_actor = self.plotter.add_mesh(
                    edge_glyphs,
                    color='orange',
                    opacity=0.9,
                )
                logger.info(f"Added {np.sum(concave_edge_mask)} membrane CONCAVE edge spheres (orange)")
            
            # Corner concave (magenta) - FIXED, no smoothing allowed
            # Also includes isolated triangle tip vertices (type 5) which are treated the same
            concave_corner_mask = (types == 4) | (types == 5)
            n_concave_corners = np.sum(types == 4)
            n_isolated_tips = np.sum(types == 5)
            if np.sum(concave_corner_mask) > 0:
                corner_positions = positions[concave_corner_mask]
                corner_cloud = pv.PolyData(corner_positions)
                corner_glyphs = corner_cloud.glyph(
                    geom=pv.Sphere(radius=membrane_radius * 1.5),
                    scale=False,
                    orient=False
                )
                
                self._feature_debug_membrane_concave_corner_actor = self.plotter.add_mesh(
                    corner_glyphs,
                    color='magenta',
                    opacity=1.0,
                )
                logger.info(f"Added {np.sum(concave_corner_mask)} membrane FIXED spheres (magenta) - "
                           f"{n_concave_corners} concave corners + {n_isolated_tips} isolated triangle tips")
        
        # === 4. Draw restored corner vertices (blue spheres) ===
        # These are concave corner vertices that were snapped back to original positions after smoothing
        if debug_data.restored_corner_positions is not None and len(debug_data.restored_corner_positions) > 0:
            restored_positions = debug_data.restored_corner_positions
            restored_cloud = pv.PolyData(restored_positions)
            # Use slightly larger spheres so they're visible over magenta
            restored_radius = membrane_radius * 1.8 if 'membrane_radius' in dir() else sphere_radius * 1.2
            restored_glyphs = restored_cloud.glyph(
                geom=pv.Sphere(radius=restored_radius),
                scale=False,
                orient=False
            )
            
            self._feature_debug_restored_corners_actor = self.plotter.add_mesh(
                restored_glyphs,
                color='blue',
                opacity=1.0,
            )
            logger.info(f"Added {len(restored_positions)} restored corner spheres (blue)")
        
        self._feature_debug_visible = True
        self.plotter.update()
    
    def remove_feature_debug_visualization(self):
        """Remove all feature debug visualization actors."""
        if not PYVISTA_AVAILABLE:
            return
        
        actors = [
            self._feature_debug_sharp_edges_actor,
            self._feature_debug_corners_actor,
            self._feature_debug_concave_corners_actor,
            self._feature_debug_sharp_edge_verts_actor,
            self._feature_debug_concave_edge_verts_actor,
            self._feature_debug_membrane_smooth_actor,
            self._feature_debug_membrane_edge_actor,
            self._feature_debug_membrane_corner_actor,
            self._feature_debug_membrane_concave_edge_actor,
            self._feature_debug_membrane_concave_corner_actor,
            self._feature_debug_restored_corners_actor,
        ]
        
        for actor in actors:
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        
        self._feature_debug_sharp_edges_actor = None
        self._feature_debug_corners_actor = None
        self._feature_debug_concave_corners_actor = None
        self._feature_debug_sharp_edge_verts_actor = None
        self._feature_debug_concave_edge_verts_actor = None
        self._feature_debug_membrane_smooth_actor = None
        self._feature_debug_membrane_edge_actor = None
        self._feature_debug_membrane_corner_actor = None
        self._feature_debug_membrane_concave_edge_actor = None
        self._feature_debug_membrane_concave_corner_actor = None
        self._feature_debug_restored_corners_actor = None
        
        self.plotter.update()
    
    def set_feature_debug_visible(self, visible: bool):
        """Toggle visibility of feature debug visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        self._feature_debug_visible = visible
        
        actors = [
            self._feature_debug_sharp_edges_actor,
            self._feature_debug_corners_actor,
            self._feature_debug_sharp_edge_verts_actor,
            self._feature_debug_membrane_smooth_actor,
            self._feature_debug_membrane_edge_actor,
            self._feature_debug_membrane_corner_actor,
        ]
        
        for actor in actors:
            if actor is not None:
                try:
                    actor.SetVisibility(visible)
                except Exception:
                    pass
        
        self.plotter.update()
    
    def set_feature_sharp_edges_visible(self, visible: bool):
        """Toggle visibility of sharp edges (yellow lines)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_sharp_edges_actor is not None:
            try:
                self._feature_debug_sharp_edges_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_convex_corners_visible(self, visible: bool):
        """Toggle visibility of convex corners (red spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_corners_actor is not None:
            try:
                self._feature_debug_corners_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_concave_corners_visible(self, visible: bool):
        """Toggle visibility of concave corners (purple spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_concave_corners_actor is not None:
            try:
                self._feature_debug_concave_corners_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_convex_edge_verts_visible(self, visible: bool):
        """Toggle visibility of convex sharp edge vertices (cyan spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        # This includes both target mesh convex edge verts and membrane convex edge verts
        actors = [
            self._feature_debug_sharp_edge_verts_actor,
            self._feature_debug_membrane_edge_actor,
        ]
        for actor in actors:
            if actor is not None:
                try:
                    actor.SetVisibility(visible)
                except Exception:
                    pass
        self.plotter.update()
    
    def set_feature_concave_edge_verts_visible(self, visible: bool):
        """Toggle visibility of concave sharp edge vertices (orange spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        # This includes both target mesh concave edge verts and membrane concave edge verts
        actors = [
            self._feature_debug_concave_edge_verts_actor,
            self._feature_debug_membrane_concave_edge_actor,
        ]
        for actor in actors:
            if actor is not None:
                try:
                    actor.SetVisibility(visible)
                except Exception:
                    pass
        self.plotter.update()
    
    def set_feature_membrane_smooth_visible(self, visible: bool):
        """Toggle visibility of membrane smooth vertices (lime spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_membrane_smooth_actor is not None:
            try:
                self._feature_debug_membrane_smooth_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_membrane_convex_edge_visible(self, visible: bool):
        """Toggle visibility of membrane convex edge vertices (cyan spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_membrane_edge_actor is not None:
            try:
                self._feature_debug_membrane_edge_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_membrane_convex_corner_visible(self, visible: bool):
        """Toggle visibility of membrane convex corner vertices (red spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_membrane_corner_actor is not None:
            try:
                self._feature_debug_membrane_corner_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_membrane_concave_edge_visible(self, visible: bool):
        """Toggle visibility of membrane concave edge vertices (orange spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_membrane_concave_edge_actor is not None:
            try:
                self._feature_debug_membrane_concave_edge_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_membrane_concave_corner_visible(self, visible: bool):
        """Toggle visibility of membrane concave corner vertices (magenta spheres - FIXED)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_membrane_concave_corner_actor is not None:
            try:
                self._feature_debug_membrane_concave_corner_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass
    
    def set_feature_restored_corners_visible(self, visible: bool):
        """Toggle visibility of restored corner vertices (blue spheres)."""
        if not PYVISTA_AVAILABLE:
            return
        if self._feature_debug_restored_corners_actor is not None:
            try:
                self._feature_debug_restored_corners_actor.SetVisibility(visible)
                self.plotter.update()
            except Exception:
                pass

    def apply_visibility_paint(
        self,
        face_colors: np.ndarray,
    ):
        """
        Apply visibility-based face colors to the mesh.
        
        Colors the mesh triangles based on which parting direction
        can see them (D1=green, D2=orange, overlap=yellow, neutral=gray).
        
        Args:
            face_colors: Array of shape (n_faces, 4) with RGBA colors (0-255)
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        logger.info("Applying visibility paint to mesh")
        
        # Convert colors to 0-255 range if needed
        if face_colors.max() <= 1.0:
            face_colors = (face_colors * 255).astype(np.uint8)
        
        # Store the face colors for later toggling
        self._stored_face_colors = face_colors.copy()
        self._visibility_paint_active = True
        self._visibility_paint_showing = True
        
        # Apply colors as cell data
        self._pv_mesh.cell_data['colors'] = face_colors
        
        # Update the mesh actor to use the new colors
        self._update_mesh_display(use_colors=True)
        
        logger.debug(f"Applied visibility paint to {len(face_colors)} faces")
    
    def _update_mesh_display(self, use_colors: bool = False):
        """
        Update mesh display without clearing the entire scene.
        
        This method efficiently updates only the mesh actor, preserving
        hull and arrow actors.
        
        Args:
            use_colors: If True, display with per-cell colors; if False, solid color
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Remove only the main mesh actor (not hull/arrows)
        if self._actor is not None:
            try:
                self.plotter.remove_actor(self._actor)
            except Exception:
                pass
            self._actor = None
        
        # Remove feature edges (they're tied to mesh appearance)
        if self._feature_edges_actor is not None:
            try:
                self.plotter.remove_actor(self._feature_edges_actor)
            except Exception:
                pass
            self._feature_edges_actor = None
        
        # Add mesh with appropriate coloring using PBR
        if use_colors and 'colors' in self._pv_mesh.cell_data:
            self._actor = self.plotter.add_mesh(
                self._pv_mesh,
                scalars='colors',
                rgb=True,
                pbr=True,
                metallic=0.1,
                roughness=0.6,
                smooth_shading=True,
                show_edges=False,
                opacity=1.0,
            )
            # Skip feature edges when showing per-cell colors (avoid cell-shaded look)
        else:
            self._actor = self.plotter.add_mesh(
                self._pv_mesh,
                color=self._current_color,
                pbr=True,
                metallic=0.1,
                roughness=0.6,
                smooth_shading=True,
                show_edges=False,
                opacity=1.0,
            )
            # Re-add feature edges for solid color mesh
            self._add_feature_edges()
        
        self.plotter.update()
    
    def set_visibility_paint_visible(self, visible: bool):
        """
        Toggle visibility paint display without full re-render.
        
        This efficiently toggles between colored (parting analysis) and
        solid color mesh display without affecting other scene elements.
        
        Args:
            visible: True to show parting colors, False to show solid color
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        if not self._visibility_paint_active:
            # No paint data available
            return
        
        if visible == self._visibility_paint_showing:
            # Already in the desired state
            return
        
        self._visibility_paint_showing = visible
        
        if visible:
            # Restore the stored colors
            if self._stored_face_colors is not None:
                self._pv_mesh.cell_data['colors'] = self._stored_face_colors
                self._update_mesh_display(use_colors=True)
        else:
            # Show solid color (remove colors from cell data display)
            self._update_mesh_display(use_colors=False)
        
        logger.debug(f"Visibility paint display set to: {visible}")
    
    def _render_mesh_with_colors(self):
        """
        Full re-render with per-cell colors (fallback method).
        
        NOTE: This method clears the entire scene and re-adds all actors.
        Prefer using _update_mesh_display() for toggling visibility paint
        as it's more efficient and doesn't affect hull actors.
        
        This method is kept for edge cases where a full re-render is needed.
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Clear previous content
        self.plotter.clear()
        self._feature_edges_actor = None
        self._silhouette_actor = None
        self._silhouette_filter = None
        # Hull actor is invalidated by clear(), reset reference
        self._hull_actor = None
        self._original_hull_actor = None
        
        # Re-setup lighting after clear
        self._setup_lighting()
        
        # Add mesh with cell colors using PBR shading
        self._actor = self.plotter.add_mesh(
            self._pv_mesh,
            scalars='colors',
            rgb=True,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
        )
        
        # Skip feature edges when showing per-cell colors (avoid cell-shaded look)
        
        # Re-add grid and scale rulers
        if self._pv_mesh is not None:
            self._add_grid_plane(self._pv_mesh.bounds)
            self._add_scale_rulers(self._pv_mesh.bounds)
        
        # Re-add parting direction arrows if they exist
        if self._parting_d1 is not None and self._parting_d2 is not None:
            # Store current arrows
            old_actors = self._arrow_actors
            self._arrow_actors = []
            
            # Re-add arrows
            self.add_parting_direction_arrows(self._parting_d1, self._parting_d2)
            
            # Clean up old actors (they were already cleared)
            old_actors.clear()
        
        # Add axes
        self.plotter.add_axes(
            line_width=2,
            color='#333333',  # Dark gray for light theme
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
        
        # Ensure orthographic projection
        self.plotter.enable_parallel_projection()
        
        # Re-add hull actor if it exists (gets cleared by plotter.clear())
        self._re_add_hull_actor()
        
        self.plotter.update()
    
    def remove_visibility_paint(self):
        """Remove visibility paint and restore original mesh appearance."""
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        if not self._visibility_paint_active:
            return
        
        logger.info("Removing visibility paint from mesh")
        
        self._visibility_paint_active = False
        self._visibility_paint_showing = False
        self._stored_face_colors = None
        
        # Remove the colors cell data
        if 'colors' in self._pv_mesh.cell_data:
            del self._pv_mesh.cell_data['colors']
        
        # Update mesh display to solid color (efficient - doesn't clear scene)
        self._update_mesh_display(use_colors=False)
    
    @property
    def parting_d1(self) -> Optional[np.ndarray]:
        """Get the primary parting direction."""
        return self._parting_d1
    
    @property
    def parting_d2(self) -> Optional[np.ndarray]:
        """Get the secondary parting direction."""
        return self._parting_d2
    
    @property
    def has_parting_directions(self) -> bool:
        """Check if parting directions have been computed."""
        return self._parting_d1 is not None and self._parting_d2 is not None
    
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

    # =========================================================================
    # INFLATED HULL VISUALIZATION
    # =========================================================================
    
    def set_hull_mesh(
        self, 
        hull_mesh: trimesh.Trimesh, 
        original_hull: Optional[trimesh.Trimesh] = None
    ):
        """
        Set and display the inflated hull mesh.
        
        Args:
            hull_mesh: The inflated convex hull mesh
            original_hull: Optional original (non-inflated) hull for reference
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._hull_mesh = hull_mesh
        
        # Convert hull to PyVista
        pv_hull = self._trimesh_to_pyvista(hull_mesh)
        
        # Remove existing hull actors (but don't clear mesh data)
        if self._hull_actor is not None:
            try:
                self.plotter.remove_actor(self._hull_actor)
            except Exception:
                pass
            self._hull_actor = None
        
        if self._original_hull_actor is not None:
            try:
                self.plotter.remove_actor(self._original_hull_actor)
            except Exception:
                pass
            self._original_hull_actor = None
        
        # Add inflated hull mesh (semi-transparent) with PBR
        self._hull_actor = self.plotter.add_mesh(
            pv_hull,
            color=self.HULL_COLOR,
            opacity=self.HULL_OPACITY,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            smooth_shading=True,
            show_edges=False,
            style='surface',
        )
        
        # Optionally add original hull wireframe
        if original_hull is not None and len(original_hull.vertices) > 0:
            pv_orig_hull = self._trimesh_to_pyvista(original_hull)
            self._original_hull_actor = self.plotter.add_mesh(
                pv_orig_hull,
                color=self.ORIGINAL_HULL_COLOR,
                opacity=0.2,
                style='wireframe',
                line_width=1,
                render_lines_as_tubes=False,
            )
            # Hide by default
            if self._original_hull_actor is not None:
                self._original_hull_actor.SetVisibility(self._original_hull_visible)
        
        self.plotter.update()
        logger.info(f"Hull mesh displayed: {len(hull_mesh.vertices)} vertices, {len(hull_mesh.faces)} faces")
    
    def clear_hull(self):
        """Remove hull visualization from the scene."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._hull_actor is not None:
            try:
                self.plotter.remove_actor(self._hull_actor)
            except Exception:
                pass
            self._hull_actor = None
        
        if self._original_hull_actor is not None:
            try:
                self.plotter.remove_actor(self._original_hull_actor)
            except Exception:
                pass
            self._original_hull_actor = None
        
        self._hull_mesh = None
        self.plotter.update()
        logger.info("Hull mesh cleared")
    
    def set_hull_visible(self, visible: bool):
        """
        Set visibility of the inflated hull.
        
        Args:
            visible: True to show hull, False to hide
        """
        self._hull_visible = visible
        if self._hull_actor is not None:
            self._hull_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Hull visibility set to {visible}")
        else:
            logger.debug(f"Hull visibility requested ({visible}) but no actor exists (mesh exists: {self._hull_mesh is not None})")
    
    def set_original_hull_visible(self, visible: bool):
        """
        Set visibility of the original hull wireframe.
        
        Args:
            visible: True to show original hull, False to hide
        """
        self._original_hull_visible = visible
        if self._original_hull_actor is not None:
            self._original_hull_actor.SetVisibility(visible)
            self.plotter.update()
    
    def set_hull_opacity(self, opacity: float):
        """
        Set the opacity of the inflated hull.
        
        Args:
            opacity: Opacity value from 0.0 (transparent) to 1.0 (opaque)
        """
        if self._hull_actor is not None:
            prop = self._hull_actor.GetProperty()
            if prop:
                prop.SetOpacity(opacity)
            self.plotter.update()
    
    @property
    def hull_mesh(self) -> Optional[trimesh.Trimesh]:
        """Get the current hull mesh."""
        return self._hull_mesh
    
    @property
    def has_hull(self) -> bool:
        """Check if a hull has been computed."""
        return self._hull_mesh is not None

    # =========================================================================
    # TETRAHEDRAL MESH MOLD HALF CLASSIFICATION
    # =========================================================================
    
    def apply_tet_mesh_classification(
        self,
        tet_vertices: np.ndarray,
        tet_edges: np.ndarray,
        boundary_mesh: 'trimesh.Trimesh',
        classification_result,
        part_mesh: 'trimesh.Trimesh' = None,
        seed_distance_threshold: float = None  # Deprecated, kept for API compatibility
    ):
        """
        Apply mold half classification to the OUTER HULL boundary of the tetrahedral mesh.
        
        This method visualizes ONLY the outer hull boundary (H1/H2/boundary zone),
        excluding the inner boundary (part surface). This ensures the classification
        visualization accurately represents the mold half partition of the hull.
        
        Colors:
        - H₁ (Green): Edges on H1 outer hull triangles
        - H₂ (Orange): Edges on H2 outer hull triangles  
        - Boundary Zone (Gray): Interface between H₁ and H₂ on the hull
        
        Inner boundary (part surface) edges are NOT displayed, as they are not
        part of the mold half classification.
        
        Args:
            tet_vertices: Full tetrahedral mesh vertices (N x 3) - not used, kept for API compatibility
            tet_edges: Tetrahedral mesh edges (E x 2) - not used, kept for API compatibility
            boundary_mesh: The tetrahedral boundary mesh
            classification_result: MoldHalfClassificationResult
            part_mesh: Not used, kept for API compatibility
            seed_distance_threshold: Not used, kept for API compatibility
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        import time
        start_time = time.time()
        
        logger.info("Applying tet mesh classification")
        
        # Remove existing mold halves actors
        if self._mold_halves_actor is not None:
            try:
                self.plotter.remove_actor(self._mold_halves_actor)
            except Exception:
                pass
            self._mold_halves_actor = None
        
        if self._mold_halves_edges_actor is not None:
            try:
                self.plotter.remove_actor(self._mold_halves_edges_actor)
            except Exception:
                pass
            self._mold_halves_edges_actor = None
        
        # Remove inner boundary actor if exists
        if self._inner_boundary_actor is not None:
            try:
                self.plotter.remove_actor(self._inner_boundary_actor)
            except Exception:
                pass
            self._inner_boundary_actor = None
        
        # Remove part surface reference if exists
        if self._part_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._part_surface_actor)
            except Exception:
                pass
            self._part_surface_actor = None
        
        # Hide the tetrahedral mesh visualization from the tetrahedralize step
        # (don't remove it - user can toggle it back on via display options)
        self.set_tetrahedral_mesh_visible(False)
        
        # NOTE: Seed vertex identification (vertices close to part) has been removed
        # from visualization step as it was expensive and not used for display.
        # Seed identification is done during Dijkstra computation instead.
        
        # =====================================================================
        # STEP 1: Classify OUTER hull boundary for H1/H2/boundary visualization
        # Only include faces that are on the outer hull (H1, H2, or boundary zone)
        # Exclude inner boundary faces (part surface)
        # =====================================================================
        
        boundary_vertices = np.asarray(boundary_mesh.vertices)
        boundary_faces = np.asarray(boundary_mesh.faces)
        n_boundary_verts = len(boundary_vertices)
        
        # Get classification sets
        h1_set = classification_result.h1_triangles
        h2_set = classification_result.h2_triangles
        boundary_set = classification_result.boundary_zone_triangles
        inner_set = classification_result.inner_boundary_triangles
        
        # Log detailed classification stats
        logger.info(f"Classification result sets: H1={len(h1_set)}, H2={len(h2_set)}, "
                   f"boundary_zone={len(boundary_set)}, inner={len(inner_set)}")
        logger.info(f"Total boundary faces: {len(boundary_faces)}")
        
        # Outer hull faces = H1 + H2 + boundary zone (excludes inner boundary)
        outer_hull_faces = h1_set | h2_set | boundary_set
        n_outer_faces = len(outer_hull_faces)
        n_inner_faces = len(inner_set)
        
        # Verify sets don't overlap and cover all faces
        all_classified = h1_set | h2_set | boundary_set | inner_set
        n_unclassified = len(boundary_faces) - len(all_classified)
        if n_unclassified > 0:
            logger.warning(f"WARNING: {n_unclassified} boundary faces are not classified!")
        
        logger.info(f"Outer hull faces: {n_outer_faces}, Inner boundary faces: {n_inner_faces}")
        
        # Assign vertex labels based on adjacent OUTER triangles only
        # VECTORIZED for performance
        vertex_h1_count = np.zeros(n_boundary_verts, dtype=np.int32)
        vertex_h2_count = np.zeros(n_boundary_verts, dtype=np.int32)
        vertex_boundary_count = np.zeros(n_boundary_verts, dtype=np.int32)
        
        # Create face label array (vectorized lookup)
        n_faces = len(boundary_faces)
        face_labels = np.zeros(n_faces, dtype=np.int8)  # 0=inner/unknown, 1=H1, 2=H2, 3=boundary
        
        # Convert sets to arrays for vectorized assignment
        if len(h1_set) > 0:
            h1_indices = np.array(list(h1_set), dtype=np.int64)
            h1_indices = h1_indices[h1_indices < n_faces]  # Bounds check
            face_labels[h1_indices] = 1
        if len(h2_set) > 0:
            h2_indices = np.array(list(h2_set), dtype=np.int64)
            h2_indices = h2_indices[h2_indices < n_faces]
            face_labels[h2_indices] = 2
        if len(boundary_set) > 0:
            boundary_indices = np.array(list(boundary_set), dtype=np.int64)
            boundary_indices = boundary_indices[boundary_indices < n_faces]
            face_labels[boundary_indices] = 3
        
        # Vectorized vertex counting using np.add.at
        # H1 faces
        h1_mask = face_labels == 1
        h1_faces = boundary_faces[h1_mask]
        if len(h1_faces) > 0:
            np.add.at(vertex_h1_count, h1_faces[:, 0], 1)
            np.add.at(vertex_h1_count, h1_faces[:, 1], 1)
            np.add.at(vertex_h1_count, h1_faces[:, 2], 1)
        
        # H2 faces
        h2_mask = face_labels == 2
        h2_faces = boundary_faces[h2_mask]
        if len(h2_faces) > 0:
            np.add.at(vertex_h2_count, h2_faces[:, 0], 1)
            np.add.at(vertex_h2_count, h2_faces[:, 1], 1)
            np.add.at(vertex_h2_count, h2_faces[:, 2], 1)
        
        # Boundary zone faces
        bz_mask = face_labels == 3
        bz_faces = boundary_faces[bz_mask]
        if len(bz_faces) > 0:
            np.add.at(vertex_boundary_count, bz_faces[:, 0], 1)
            np.add.at(vertex_boundary_count, bz_faces[:, 1], 1)
            np.add.at(vertex_boundary_count, bz_faces[:, 2], 1)
        
        # Assign boundary vertex labels - VECTORIZED
        boundary_vertex_labels = np.zeros(n_boundary_verts, dtype=np.int8)
        
        # H1 wins: h1 > h2 AND h1 > boundary
        h1_wins = (vertex_h1_count > vertex_h2_count) & (vertex_h1_count > vertex_boundary_count)
        boundary_vertex_labels[h1_wins] = 1
        
        # H2 wins: h2 > h1 AND h2 > boundary (and not already H1)
        h2_wins = (vertex_h2_count > vertex_h1_count) & (vertex_h2_count > vertex_boundary_count) & ~h1_wins
        boundary_vertex_labels[h2_wins] = 2
        
        # Boundary zone: boundary > 0 (and not H1 or H2)
        bz_wins = (vertex_boundary_count > 0) & ~h1_wins & ~h2_wins
        boundary_vertex_labels[bz_wins] = 3
        
        # Fallback: any H1 count -> H1, any H2 count -> H2
        h1_fallback = (vertex_h1_count > 0) & (boundary_vertex_labels == 0)
        boundary_vertex_labels[h1_fallback] = 1
        h2_fallback = (vertex_h2_count > 0) & (boundary_vertex_labels == 0)
        boundary_vertex_labels[h2_fallback] = 2
        
        # Build edges ONLY from outer hull faces (H1, H2, boundary zone)
        # This excludes edges from the inner boundary (part surface)
        # VECTORIZED for performance - process all outer faces at once
        outer_face_indices = np.array(list(outer_hull_faces), dtype=np.int64)
        if len(outer_face_indices) > 0:
            outer_faces = boundary_faces[outer_face_indices]  # (n_outer, 3)
            # Extract all 3 edges from each face
            edge01 = np.sort(outer_faces[:, [0, 1]], axis=1)
            edge12 = np.sort(outer_faces[:, [1, 2]], axis=1)
            edge20 = np.sort(outer_faces[:, [2, 0]], axis=1)
            all_edges = np.vstack([edge01, edge12, edge20])  # (3*n_outer, 2)
            # Remove duplicates using numpy unique
            boundary_edges = np.unique(all_edges, axis=0)
            n_boundary_edges = len(boundary_edges)
        else:
            boundary_edges = np.empty((0, 2), dtype=np.int64)
            n_boundary_edges = 0
        
        logger.info(f"Outer hull edges: {n_boundary_edges}")
        
        # =====================================================================
        # STEP 2b: Assign colors to outer hull boundary edges only
        # =====================================================================
        
        # Assign edge colors based on endpoint labels - VECTORIZED
        edge_colors = np.full((n_boundary_edges, 3), 158, dtype=np.uint8)  # Default gray
        
        if n_boundary_edges > 0:
            # Get labels for both endpoints
            l0 = boundary_vertex_labels[boundary_edges[:, 0]]
            l1 = boundary_vertex_labels[boundary_edges[:, 1]]
            
            # Both H1 -> Green
            both_h1 = (l0 == 1) & (l1 == 1)
            edge_colors[both_h1] = [76, 175, 80]
            
            # Both H2 -> Orange
            both_h2 = (l0 == 2) & (l1 == 2)
            edge_colors[both_h2] = [255, 152, 0]
            
            # At least one H1 (but not both H2) -> Green
            has_h1 = ((l0 == 1) | (l1 == 1)) & ~both_h2
            edge_colors[has_h1] = [76, 175, 80]
            
            # At least one H2 (but not both H1) -> Orange
            has_h2 = ((l0 == 2) | (l1 == 2)) & ~both_h1 & ~has_h1
            edge_colors[has_h2] = [255, 152, 0]
            
            # Rest stay gray (boundary zone, unclassified, etc.)
        
        n_h1_verts = np.sum(boundary_vertex_labels == 1)
        n_h2_verts = np.sum(boundary_vertex_labels == 2)
        n_boundary_zone_verts = np.sum(boundary_vertex_labels == 3)
        logger.info(f"Outer hull vertex labels: H1={n_h1_verts}, H2={n_h2_verts}, boundary_zone={n_boundary_zone_verts}")
        
        # =====================================================================
        # STEP 3: Create visualization
        # =====================================================================
        
        # Create boundary edge mesh with H1/H2/boundary colors
        if n_boundary_edges > 0:
            cells = np.empty((n_boundary_edges, 3), dtype=np.int64)
            cells[:, 0] = 2
            cells[:, 1] = boundary_edges[:, 0]
            cells[:, 2] = boundary_edges[:, 1]
            cell_types = np.full(n_boundary_edges, 3, dtype=np.uint8)  # VTK_LINE
            
            edge_grid = pv.UnstructuredGrid(cells.ravel(), cell_types, boundary_vertices)
            edge_grid.cell_data['colors'] = edge_colors
            
            # Add boundary edges
            self._mold_halves_edges_actor = self.plotter.add_mesh(
                edge_grid,
                scalars='colors',
                rgb=True,
                line_width=3,
                show_edges=False,
            )
        else:
            logger.info("No boundary edges to display")
        
        # Set visibility
        if self._mold_halves_edges_actor is not None:
            self._mold_halves_edges_actor.SetVisibility(self._outer_boundary_visible)
        
        # =====================================================================
        # STEP 4: Create inner boundary visualization (blue edges)
        # This shows the mold cavity edges adjacent to the part
        # =====================================================================
        
        if n_inner_faces > 0:
            # Build edges from inner boundary faces - VECTORIZED
            inner_face_indices = np.array(list(inner_set), dtype=np.int64)
            inner_faces = boundary_faces[inner_face_indices]  # (n_inner, 3)
            # Extract all 3 edges from each face
            edge01 = np.sort(inner_faces[:, [0, 1]], axis=1)
            edge12 = np.sort(inner_faces[:, [1, 2]], axis=1)
            edge20 = np.sort(inner_faces[:, [2, 0]], axis=1)
            all_inner_edges = np.vstack([edge01, edge12, edge20])
            # Remove duplicates
            inner_edges = np.unique(all_inner_edges, axis=0)
            n_inner_edges = len(inner_edges)
            
            if n_inner_edges > 0:
                # Create edge cells for inner boundary
                inner_cells = np.empty((n_inner_edges, 3), dtype=np.int64)
                inner_cells[:, 0] = 2
                inner_cells[:, 1] = inner_edges[:, 0]
                inner_cells[:, 2] = inner_edges[:, 1]
                inner_cell_types = np.full(n_inner_edges, 3, dtype=np.uint8)  # VTK_LINE
                
                inner_edge_grid = pv.UnstructuredGrid(inner_cells.ravel(), inner_cell_types, boundary_vertices)
                
                # Add inner boundary edges as blue lines
                self._inner_boundary_actor = self.plotter.add_mesh(
                    inner_edge_grid,
                    color='#2196F3',  # Blue color
                    line_width=3,
                    show_edges=False,
                )
                
                if self._inner_boundary_actor is not None:
                    self._inner_boundary_actor.SetVisibility(self._inner_boundary_visible)
                
                logger.info(f"Added inner boundary visualization: {n_inner_edges} edges (blue)")
            else:
                logger.info("No inner boundary edges to display")
        else:
            logger.info("No inner boundary faces to display")
        
        self.plotter.update()
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Applied tet classification in {elapsed:.0f}ms: {n_boundary_edges} outer hull edges, {n_inner_faces} inner boundary faces")
    
    def show_part_surface_reference(self, part_mesh: 'trimesh.Trimesh'):
        """
        Show the part mesh surface as a semi-transparent reference.
        
        This helps debug why inner boundary detection might be failing by
        showing where the part surface actually is.
        
        Args:
            part_mesh: The original part mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        logger.info(f"Showing part surface reference: {len(part_mesh.vertices)} vertices")
        
        # Remove existing part surface actor
        if self._part_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._part_surface_actor)
            except Exception:
                pass
            self._part_surface_actor = None
        
        # Convert to PyVista
        pv_mesh = self._trimesh_to_pyvista(part_mesh)
        
        # Add as semi-transparent cyan surface
        self._part_surface_actor = self.plotter.add_mesh(
            pv_mesh,
            color='cyan',
            opacity=0.3,
            show_edges=True,
            edge_color='darkblue',
            line_width=1,
        )
        
        if self._part_surface_actor is not None:
            self._part_surface_actor.SetVisibility(self._part_surface_visible)
        
        self.plotter.update()
    
    def set_part_surface_visible(self, visible: bool):
        """Set visibility of part surface reference."""
        self._part_surface_visible = visible
        if self._part_surface_actor is not None:
            self._part_surface_actor.SetVisibility(visible)
            self.plotter.update()
    
    def set_mold_halves_visible(self, visible: bool):
        """Set visibility of mold halves classification (both outer and inner boundary)."""
        self._mold_halves_visible = visible
        if self._mold_halves_edges_actor is not None:
            self._mold_halves_edges_actor.SetVisibility(visible)
        if self._inner_boundary_actor is not None:
            self._inner_boundary_actor.SetVisibility(visible)
        self.plotter.update()
    
    def set_outer_boundary_visible(self, visible: bool):
        """Set visibility of outer boundary edges (H1/H2/boundary zone)."""
        self._outer_boundary_visible = visible
        if self._mold_halves_edges_actor is not None:
            self._mold_halves_edges_actor.SetVisibility(visible)
        self.plotter.update()
    
    def set_inner_boundary_visible(self, visible: bool):
        """Set visibility of inner boundary surface (adjacent to part, blue)."""
        self._inner_boundary_visible = visible
        if self._inner_boundary_actor is not None:
            self._inner_boundary_actor.SetVisibility(visible)
        self.plotter.update()
    
    @property
    def has_mold_halves(self) -> bool:
        """Check if mold halves classification is displayed."""
        return self._mold_halves_edges_actor is not None or self._inner_boundary_actor is not None

    # ========================================================================
    # TETRAHEDRAL MESH VISUALIZATION
    # ========================================================================
    
    def set_tetrahedral_mesh(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_weights: Optional[np.ndarray] = None,
        edge_boundary_labels: Optional[np.ndarray] = None,
        colormap: str = 'coolwarm'
    ):
        """
        Display tetrahedral mesh edges with coloring based on weights and boundary labels.
        
        - Interior edges: colored by weight (coolwarm colormap)
        - H1 boundary edges: green
        - H2 boundary edges: orange  
        - Inner boundary edges: dark gray
        - Mixed boundary edges: light gray
        
        Args:
            vertices: (N, 3) vertex positions
            edges: (E, 2) edge vertex indices
            edge_weights: (E,) weights for coloring interior edges (optional)
            edge_boundary_labels: (E,) boundary labels: 0=interior, 1=H1, 2=H2, -1=inner, -2=mixed
            colormap: Matplotlib colormap name for weight visualization
        """
        if not PYVISTA_AVAILABLE:
            return
        
        logger.info(f"Setting tetrahedral mesh: {len(vertices)} verts, {len(edges)} edges")
        
        # Remove existing tet mesh actors
        self.clear_tetrahedral_mesh()
        
        # Reset visibility flags to True for newly created edges
        self._tet_interior_visible = True
        self._tet_boundary_visible = True
        
        n_edges = len(edges)
        
        # If we have boundary labels, separate boundary and interior edges
        if edge_boundary_labels is not None:
            # Boundary edges: labels != 0
            boundary_mask = edge_boundary_labels != 0
            interior_mask = ~boundary_mask
            
            n_boundary = np.sum(boundary_mask)
            n_interior = np.sum(interior_mask)
            logger.info(f"  Interior edges: {n_interior}, Boundary edges: {n_boundary}")
            
            # === INTERIOR EDGES: Colored by weight ===
            if n_interior > 0:
                interior_edges = edges[interior_mask]
                interior_weights = edge_weights[interior_mask] if edge_weights is not None else None
                
                # Build PyVista PolyData for interior edges
                interior_mesh = pv.PolyData()
                interior_mesh.points = vertices.astype(np.float32)
                
                # Create cells array: for lines, each cell is [2, v0, v1]
                cells = np.column_stack([
                    np.full(n_interior, 2, dtype=np.int64),
                    interior_edges[:, 0],
                    interior_edges[:, 1]
                ]).ravel()
                
                interior_mesh.lines = cells
                
                logger.debug(f"Interior mesh: {interior_mesh.n_cells} cells, interior_edges: {len(interior_edges)}")
                
                if interior_weights is not None:
                    logger.debug(f"Interior weights: {len(interior_weights)} weights")
                    if len(interior_weights) == n_interior:
                        # Apply log scale to weights for better visualization
                        # Since weights = 1/(dist² + ε), high weight = close to part
                        # Log scale spreads out the values for better color differentiation
                        log_weights = np.log10(interior_weights + 1e-10)
                        
                        # Normalize to 0-1 range for better colormap usage
                        w_min, w_max = log_weights.min(), log_weights.max()
                        if w_max > w_min:
                            normalized_weights = (log_weights - w_min) / (w_max - w_min)
                        else:
                            normalized_weights = np.zeros_like(log_weights)
                        
                        logger.debug(f"Weight range: [{interior_weights.min():.4f}, {interior_weights.max():.4f}]")
                        logger.debug(f"Log weight range: [{w_min:.4f}, {w_max:.4f}]")
                        
                        # Weights match number of interior edges - assign to cells
                        interior_mesh.cell_data['weight'] = normalized_weights.astype(np.float32)
                        self._tet_interior_actor = self.plotter.add_mesh(
                            interior_mesh,
                            scalars='weight',
                            cmap='plasma',  # plasma has better perceptual uniformity than coolwarm
                            clim=[0, 1],  # Fixed range since we normalized
                            line_width=1.5,
                            render_lines_as_tubes=False,
                            show_scalar_bar=True,
                            scalar_bar_args={
                                'title': 'Edge Weight (log)',
                                'title_font_size': 12,
                                'label_font_size': 10,
                                'n_labels': 5,
                                'position_x': 0.85,
                                'position_y': 0.1,
                                'width': 0.1,
                                'height': 0.3,
                            }
                        )
                    else:
                        logger.warning(f"Interior weights length mismatch: {len(interior_weights)} vs {n_interior} edges")
                        self._tet_interior_actor = self.plotter.add_mesh(
                            interior_mesh,
                            color='#888888',  # Gray default
                            line_width=1.5,
                            render_lines_as_tubes=False,
                        )
                else:
                    logger.warning("No interior weights provided")
                    self._tet_interior_actor = self.plotter.add_mesh(
                        interior_mesh,
                        color='#888888',  # Gray default
                        line_width=1.5,
                        render_lines_as_tubes=False,
                    )
            
            # === BOUNDARY EDGES: H1/H2 only (inner boundary already shown in mold halves step) ===
            if n_boundary > 0:
                boundary_edges = edges[boundary_mask]
                boundary_labels = edge_boundary_labels[boundary_mask]
                
                # Only show H1 and H2 edges (skip inner boundary and mixed - already in mold halves step)
                h1h2_mask = (boundary_labels == 1) | (boundary_labels == 2)
                
                if np.any(h1h2_mask):
                    h1h2_edges = boundary_edges[h1h2_mask]
                    h1h2_labels = boundary_labels[h1h2_mask]
                    n_h1h2 = len(h1h2_edges)
                    
                    # Create RGBA colors for H1/H2 edges
                    boundary_colors = np.zeros((n_h1h2, 4), dtype=np.uint8)
                    
                    # H1 edges (label=1): Green
                    h1_mask = h1h2_labels == 1
                    boundary_colors[h1_mask] = [76, 175, 80, 255]  # #4CAF50
                    
                    # H2 edges (label=2): Orange
                    h2_mask = h1h2_labels == 2
                    boundary_colors[h2_mask] = [255, 152, 0, 255]  # #FF9800
                    
                    # Build PyVista PolyData for boundary edges
                    boundary_mesh = pv.PolyData()
                    boundary_mesh.points = vertices.astype(np.float32)
                    
                    # Create cells array: for lines, each cell is [2, v0, v1]
                    cells = np.column_stack([
                        np.full(n_h1h2, 2, dtype=np.int64),
                        h1h2_edges[:, 0],
                        h1h2_edges[:, 1]
                    ]).ravel()
                    
                    # Set lines using the lines property with proper format
                    boundary_mesh.lines = cells
                    
                    logger.debug(f"Boundary mesh (H1/H2 only): {boundary_mesh.n_cells} cells, {len(boundary_colors)} colors")
                    
                    # Verify cell count matches colors
                    if boundary_mesh.n_cells == len(boundary_colors):
                        boundary_mesh.cell_data['colors'] = boundary_colors
                        
                        self._tet_boundary_actor = self.plotter.add_mesh(
                            boundary_mesh,
                            scalars='colors',
                            rgb=True,
                            line_width=2.5,  # Slightly thicker for boundary
                            render_lines_as_tubes=False,
                            show_scalar_bar=False,
                        )
                    else:
                        logger.warning(
                            f"Cell count mismatch: {boundary_mesh.n_cells} cells vs {len(boundary_colors)} colors. "
                            "Showing boundary edges without mold half coloring."
                        )
                        self._tet_boundary_actor = self.plotter.add_mesh(
                            boundary_mesh,
                            color='#888888',
                            line_width=2.5,
                            render_lines_as_tubes=False,
                        )
        else:
            # No boundary labels - show all edges colored by weight (original behavior)
            mesh = pv.PolyData()
            mesh.points = vertices.astype(np.float32)
            
            # Create cells array for all edges
            cells = np.column_stack([
                np.full(n_edges, 2, dtype=np.int64),
                edges[:, 0],
                edges[:, 1]
            ]).ravel()
            
            mesh.lines = cells
            
            logger.debug(f"All edges mesh: {mesh.n_cells} cells")
            
            if edge_weights is not None and len(edge_weights) == mesh.n_cells:
                mesh.cell_data['weight'] = edge_weights.astype(np.float32)
                self._tet_interior_actor = self.plotter.add_mesh(
                    mesh,
                    scalars='weight',
                    cmap=colormap,
                    line_width=1.5,
                    render_lines_as_tubes=False,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        'title': 'Edge Weight',
                        'title_font_size': 12,
                        'label_font_size': 10,
                        'n_labels': 5,
                        'position_x': 0.85,
                        'position_y': 0.1,
                        'width': 0.1,
                        'height': 0.3,
                    }
                )
            else:
                self._tet_interior_actor = self.plotter.add_mesh(
                    mesh,
                    color='#ffaa00',  # Orange default
                    line_width=1.5,
                    render_lines_as_tubes=False,
                )
        
        # Set visibility
        self._apply_tet_visibility()
        
        self.plotter.update()
        logger.info(f"Tetrahedral mesh edges displayed")
    
    def _apply_tet_visibility(self):
        """Apply current visibility settings to tetrahedral mesh actors."""
        if self._tet_interior_actor is not None:
            self._tet_interior_actor.SetVisibility(self._tet_interior_visible)
        if self._tet_boundary_actor is not None:
            self._tet_boundary_actor.SetVisibility(self._tet_boundary_visible)
    
    def clear_tetrahedral_mesh(self):
        """Remove tetrahedral mesh visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._tet_interior_actor is not None:
            try:
                self.plotter.remove_actor(self._tet_interior_actor)
            except Exception:
                pass
            self._tet_interior_actor = None
        
        if self._tet_boundary_actor is not None:
            try:
                self.plotter.remove_actor(self._tet_boundary_actor)
            except Exception:
                pass
            self._tet_boundary_actor = None
        
        # Also clear old single actor if exists
        if self._tet_edges_actor is not None:
            try:
                self.plotter.remove_actor(self._tet_edges_actor)
            except Exception:
                pass
            self._tet_edges_actor = None
    
    def set_tetrahedral_mesh_visible(self, visible: bool):
        """Set visibility of all tetrahedral mesh edges."""
        self._tet_interior_visible = visible
        self._tet_boundary_visible = visible
        self._tet_visible = visible
        self._apply_tet_visibility()
        if self._tet_edges_actor is not None:
            self._tet_edges_actor.SetVisibility(visible)
        self.plotter.update()
    
    def set_tet_interior_visible(self, visible: bool):
        """Set visibility of interior tetrahedral edges (weight-colored)."""
        self._tet_interior_visible = visible
        if self._tet_interior_actor is not None:
            self._tet_interior_actor.SetVisibility(visible)
            self.plotter.update()
    
    def set_tet_boundary_visible(self, visible: bool):
        """Set visibility of boundary tetrahedral edges (mold-half colored)."""
        self._tet_boundary_visible = visible
        if self._tet_boundary_actor is not None:
            self._tet_boundary_actor.SetVisibility(visible)
            self.plotter.update()
    
    @property
    def has_tetrahedral_mesh(self) -> bool:
        """Check if tetrahedral mesh is displayed."""
        return (self._tet_edges_actor is not None or 
                self._tet_interior_actor is not None or 
                self._tet_boundary_actor is not None)

    # ========================================================================
    # INFLATED BOUNDARY MESH VISUALIZATION
    # ========================================================================
    
    def set_inflated_boundary_mesh(self, boundary_mesh: trimesh.Trimesh):
        """
        Display the inflated boundary mesh as a semi-transparent surface.
        
        Args:
            boundary_mesh: The inflated boundary surface mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        logger.info(f"Setting inflated boundary mesh: {len(boundary_mesh.vertices)} verts, {len(boundary_mesh.faces)} faces")
        
        # Remove existing inflated boundary actor
        self.clear_inflated_boundary_mesh()
        
        # Convert to PyVista
        vertices = np.asarray(boundary_mesh.vertices, dtype=np.float32)
        faces = np.asarray(boundary_mesh.faces, dtype=np.int32)
        
        # PyVista face format: [n_points, idx0, idx1, idx2, ...]
        pv_faces = np.column_stack([
            np.full(len(faces), 3, dtype=np.int32),
            faces
        ]).ravel()
        
        pv_mesh = pv.PolyData(vertices, pv_faces)
        
        # Use a distinct color for inflated boundary - golden/amber
        inflated_color = "#ffaa00"  # Golden amber
        
        # Add with PBR shading
        self._inflated_boundary_actor = self.plotter.add_mesh(
            pv_mesh,
            color=inflated_color,
            opacity=0.6,
            pbr=True,
            metallic=0.1,
            roughness=0.6,
            show_edges=False,
            smooth_shading=True,
        )
        
        if self._inflated_boundary_actor is not None:
            self._inflated_boundary_actor.SetVisibility(self._inflated_boundary_visible)
        
        self.plotter.update()
        logger.info("Inflated boundary mesh displayed")
    
    def clear_inflated_boundary_mesh(self):
        """Remove inflated boundary mesh visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._inflated_boundary_actor is not None:
            try:
                self.plotter.remove_actor(self._inflated_boundary_actor)
            except Exception:
                pass
            self._inflated_boundary_actor = None
    
    def set_inflated_boundary_visible(self, visible: bool):
        """Set visibility of inflated boundary mesh."""
        self._inflated_boundary_visible = visible
        if self._inflated_boundary_actor is not None:
            self._inflated_boundary_actor.SetVisibility(visible)
            self.plotter.update()
    
    @property
    def has_inflated_boundary(self) -> bool:
        """Check if inflated boundary mesh is displayed."""
        return self._inflated_boundary_actor is not None

    # ========================================================================
    # R DISTANCE LINE VISUALIZATION
    # ========================================================================
    
    def set_r_distance_line(
        self,
        hull_point: np.ndarray,
        part_point: np.ndarray,
        r_value: float
    ):
        """
        Display the R distance line (max hull-to-part distance).
        
        Shows a simple red line between the hull point and the closest point on the part,
        using the same edge-based visualization style as mold halves.
        
        Args:
            hull_point: (3,) position on hull surface (maximum distance point)
            part_point: (3,) closest point on part surface
            r_value: The R distance value
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        logger.info(f"Setting R distance line: R={r_value:.4f}")
        
        # Clear existing R visualization
        self.clear_r_distance_line()
        
        # Create a simple line using UnstructuredGrid with VTK_LINE (same as mold halves)
        vertices = np.array([hull_point, part_point])
        
        # Create cells array: [2, 0, 1] for a single line
        cells = np.array([[2, 0, 1]], dtype=np.int64)
        cell_types = np.array([3], dtype=np.uint8)  # VTK_LINE = 3
        
        line_grid = pv.UnstructuredGrid(cells.ravel(), cell_types, vertices)
        
        # Add the line with red color
        self._r_line_actor = self.plotter.add_mesh(
            line_grid,
            color='#ff3333',  # Bright red
            line_width=4,
            show_edges=False,
        )
        
        # Add label at midpoint
        midpoint = (hull_point + part_point) / 2
        # Offset label slightly to avoid overlapping with line
        offset = np.array([r_value * 0.1, r_value * 0.1, 0])
        label_pos = midpoint + offset
        
        self._r_label_actor = self.plotter.add_point_labels(
            [label_pos],
            [f"R = {r_value:.3f}"],
            font_size=14,
            text_color='red',
            point_color='red',
            point_size=0,  # Don't show point
            shape_opacity=0.7,
            always_visible=True,
        )
        
        self.plotter.update()
        logger.info(f"R distance line displayed: hull={hull_point}, part={part_point}")
    
    def clear_r_distance_line(self):
        """Remove R distance line visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove line actor
        if self._r_line_actor is not None:
            try:
                self.plotter.remove_actor(self._r_line_actor)
            except Exception:
                pass
            self._r_line_actor = None
        
        # Remove point actors
        for actor in self._r_point_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._r_point_actors = []
        
        # Remove label actor
        if self._r_label_actor is not None:
            try:
                self.plotter.remove_actor(self._r_label_actor)
            except Exception:
                pass
            self._r_label_actor = None
        
        self.plotter.update()
    
    @property
    def has_r_distance_line(self) -> bool:
        """Check if R distance line is displayed."""
        return self._r_line_actor is not None

    # ========================================================================
    # DIJKSTRA RESULT VISUALIZATION
    # ========================================================================
    
    def set_dijkstra_result(
        self,
        vertices: np.ndarray,
        interior_vertex_indices: np.ndarray,
        interior_escape_labels: np.ndarray,
        boundary_mesh: 'trimesh.Trimesh' = None,
        interior_distances: np.ndarray = None,
        tet_edges: np.ndarray = None,
        boundary_labels: np.ndarray = None
    ):
        """
        Display Dijkstra results as colored edges for ALL interior vertices.
        
        Shows all interior edges (edges where at least one vertex is interior)
        colored by which boundary (H1 or H2) each vertex walks to via the shortest weighted path.
        
        Edge colors:
        - Green: both endpoints walk to H1, or interior connects to H1 boundary
        - Orange: both endpoints walk to H2, or interior connects to H2 boundary  
        - Yellow: endpoints walk to opposite boundaries (parting surface passes through)
        - Gray: one or both endpoints unreachable
        
        Args:
            vertices: (N, 3) vertex positions (use original for proper geometry)
            interior_vertex_indices: (I,) indices of interior vertices in the tet mesh
            interior_escape_labels: (I,) labels: 1=walks to H1, 2=walks to H2, 0=unreachable
            boundary_mesh: The tetrahedral boundary mesh (fallback for edge extraction)
            interior_distances: (I,) optional distances to boundary
            tet_edges: (E, 2) tetrahedral mesh edges - used to find all interior edges
            boundary_labels: (N,) vertex boundary labels: 1=H1, 2=H2, -1=inner, 0=interior
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove existing actor
        self.clear_dijkstra_result()
        
        n_interior = len(interior_vertex_indices)
        if n_interior == 0:
            logger.warning("No interior vertices to visualize")
            return
            
        logger.info(f"Setting Dijkstra result: {n_interior} interior vertices")
        
        # Ensure interior_vertex_indices is a 1D numpy array of integers
        interior_vertex_indices = np.asarray(interior_vertex_indices).flatten().astype(int)
        interior_escape_labels = np.asarray(interior_escape_labels).flatten()
        
        # Create a mapping from tet vertex index to escape label
        # Initialize all vertices as -1 (not interior / on boundary)
        vertex_escape_labels = np.full(len(vertices), -1, dtype=np.int8)
        for i, vert_idx in enumerate(interior_vertex_indices):
            vertex_escape_labels[vert_idx] = interior_escape_labels[i]
        
        # Get interior vertex set for fast lookup
        interior_set = set(interior_vertex_indices.tolist())
        
        # Build H1/H2 boundary sets if boundary_labels provided
        h1_boundary_set = set()
        h2_boundary_set = set()
        if boundary_labels is not None:
            h1_boundary_set = set(np.where(boundary_labels == 1)[0].tolist())
            h2_boundary_set = set(np.where(boundary_labels == 2)[0].tolist())
            logger.info(f"Boundary sets: H1={len(h1_boundary_set)}, H2={len(h2_boundary_set)}")
        
        # Build edges connecting interior vertices (including edges to boundary)
        # Use tetrahedral mesh edges if provided, otherwise fall back to boundary mesh
        interior_edge_set = set()  # Edges where BOTH vertices are interior
        boundary_edge_list = []  # Edges connecting interior to H1/H2 boundary
        
        if tet_edges is not None:
            for v0, v1 in tet_edges:
                v0_interior = v0 in interior_set
                v1_interior = v1 in interior_set
                v0_h1 = v0 in h1_boundary_set
                v0_h2 = v0 in h2_boundary_set
                v1_h1 = v1 in h1_boundary_set
                v1_h2 = v1 in h2_boundary_set
                
                if v0_interior and v1_interior:
                    # Both interior - add to interior edges
                    interior_edge_set.add((min(v0, v1), max(v0, v1)))
                elif v0_interior and (v1_h1 or v1_h2):
                    # v0 interior, v1 on H1/H2 boundary
                    boundary_edge_list.append((v0, v1, 1 if v1_h1 else 2))
                elif v1_interior and (v0_h1 or v0_h2):
                    # v1 interior, v0 on H1/H2 boundary
                    boundary_edge_list.append((v1, v0, 1 if v0_h1 else 2))
            
            logger.info(f"Found {len(interior_edge_set)} interior-interior edges, {len(boundary_edge_list)} interior-boundary edges")
        elif boundary_mesh is not None:
            # Fallback: use boundary mesh faces
            from scipy.spatial import cKDTree
            
            # Map boundary mesh vertices to tet vertices
            boundary_verts = np.asarray(boundary_mesh.vertices)
            tet_tree = cKDTree(vertices)
            _, boundary_to_tet = tet_tree.query(boundary_verts, k=1)
            
            # Find edges from boundary mesh faces where BOTH vertices are interior
            for face in boundary_mesh.faces:
                tet_v0 = boundary_to_tet[face[0]]
                tet_v1 = boundary_to_tet[face[1]]
                tet_v2 = boundary_to_tet[face[2]]
                
                # Check each edge of this face
                for va, vb in [(tet_v0, tet_v1), (tet_v1, tet_v2), (tet_v2, tet_v0)]:
                    if va in interior_set and vb in interior_set:
                        interior_edge_set.add((min(va, vb), max(va, vb)))
        
        interior_edges = np.array(list(interior_edge_set)) if interior_edge_set else np.empty((0, 2), dtype=np.int64)
        
        n_interior_edges = len(interior_edges)
        n_boundary_edges = len(boundary_edge_list)
        logger.info(f"Total edges to visualize: {n_interior_edges} interior + {n_boundary_edges} boundary-connected")
        
        h1h2_edges = []  # Green and orange edges
        primary_cut_edges = []  # Yellow edges
        h1h2_colors = []
        
        # Process interior-interior edges
        for v0, v1 in interior_edges:
            l0 = vertex_escape_labels[v0]
            l1 = vertex_escape_labels[v1]
            
            # Classify edge
            if (l0 == 1 and l1 == 2) or (l0 == 2 and l1 == 1):
                # Yellow - primary cut (interface between H1 and H2)
                primary_cut_edges.append((v0, v1))
            elif l0 == 1 and l1 == 1:
                h1h2_edges.append((v0, v1))
                h1h2_colors.append([76, 175, 80])  # Green - both walk to H1
            elif l0 == 2 and l1 == 2:
                h1h2_edges.append((v0, v1))
                h1h2_colors.append([255, 152, 0])  # Orange - both walk to H2
            elif l0 == 1 or l1 == 1:
                h1h2_edges.append((v0, v1))
                h1h2_colors.append([76, 175, 80])  # Green - at least one walks to H1
            elif l0 == 2 or l1 == 2:
                h1h2_edges.append((v0, v1))
                h1h2_colors.append([255, 152, 0])  # Orange - at least one walks to H2
            else:
                h1h2_edges.append((v0, v1))
                h1h2_colors.append([150, 150, 150])  # Gray - unreachable
        
        # Process interior-to-boundary edges
        for interior_v, boundary_v, boundary_type in boundary_edge_list:
            escape_label = vertex_escape_labels[interior_v]
            
            if boundary_type == 1:
                # Edge connects to H1 boundary
                h1h2_edges.append((interior_v, boundary_v))
                h1h2_colors.append([76, 175, 80])  # Green
            else:
                # Edge connects to H2 boundary
                h1h2_edges.append((interior_v, boundary_v))
                h1h2_colors.append([255, 152, 0])  # Orange
        
        n_edges = len(h1h2_edges) + len(primary_cut_edges)
        
        if n_edges > 0:
            # Create actor for H1/H2 edges (green/orange/gray)
            if len(h1h2_edges) > 0:
                h1h2_edges_arr = np.array(h1h2_edges)
                h1h2_colors_arr = np.array(h1h2_colors, dtype=np.uint8)
                
                cells = np.empty((len(h1h2_edges), 3), dtype=np.int64)
                cells[:, 0] = 2
                cells[:, 1] = h1h2_edges_arr[:, 0]
                cells[:, 2] = h1h2_edges_arr[:, 1]
                cells = cells.ravel()
                
                cell_types = np.full(len(h1h2_edges), 3, dtype=np.uint8)
                edge_grid = pv.UnstructuredGrid(cells, cell_types, vertices.astype(np.float64))
                edge_grid.cell_data['colors'] = h1h2_colors_arr
                
                self._dijkstra_actor = self.plotter.add_mesh(
                    edge_grid,
                    scalars='colors',
                    rgb=True,
                    line_width=4,
                    show_edges=False,
                    show_scalar_bar=False
                )
            
            # Create separate actor for primary cut edges (yellow)
            if len(primary_cut_edges) > 0:
                primary_edges_arr = np.array(primary_cut_edges)
                primary_colors = np.full((len(primary_cut_edges), 3), [255, 255, 0], dtype=np.uint8)
                
                cells = np.empty((len(primary_cut_edges), 3), dtype=np.int64)
                cells[:, 0] = 2
                cells[:, 1] = primary_edges_arr[:, 0]
                cells[:, 2] = primary_edges_arr[:, 1]
                cells = cells.ravel()
                
                cell_types = np.full(len(primary_cut_edges), 3, dtype=np.uint8)
                edge_grid = pv.UnstructuredGrid(cells, cell_types, vertices.astype(np.float64))
                edge_grid.cell_data['colors'] = primary_colors
                
                self._dijkstra_primary_cuts_actor = self.plotter.add_mesh(
                    edge_grid,
                    scalars='colors',
                    rgb=True,
                    line_width=5,  # Slightly thicker for emphasis
                    show_edges=False,
                    show_scalar_bar=False
                )
            
            logger.info(f"Created {len(h1h2_edges)} H1/H2 edges, {len(primary_cut_edges)} primary cut edges")
        else:
            # Fallback: show interior points if no edges
            interior_positions = vertices[interior_vertex_indices]
            point_cloud = pv.PolyData(interior_positions.astype(np.float64))
            
            colors = np.zeros((n_interior, 3), dtype=np.uint8)
            colors[interior_escape_labels == 1] = [76, 175, 80]  # Green
            colors[interior_escape_labels == 2] = [255, 152, 0]  # Orange
            colors[interior_escape_labels == 0] = [150, 150, 150]  # Gray
            
            point_cloud.point_data['colors'] = colors
            
            self._dijkstra_actor = self.plotter.add_mesh(
                point_cloud,
                scalars='colors',
                rgb=True,
                point_size=8.0,
                render_points_as_spheres=True,
                show_scalar_bar=False
            )
        
        # Set visibility based on current state
        if self._dijkstra_actor is not None:
            self._dijkstra_actor.SetVisibility(self._dijkstra_visible and self._dijkstra_show_h1h2)
        if self._dijkstra_primary_cuts_actor is not None:
            self._dijkstra_primary_cuts_actor.SetVisibility(self._dijkstra_visible)
        
        self.plotter.update()
        
        n_h1 = np.sum(interior_escape_labels == 1)
        n_h2 = np.sum(interior_escape_labels == 2)
        n_unreached = np.sum(interior_escape_labels == 0)
        
        # Count yellow (parting) edges
        n_yellow = len(primary_cut_edges) if n_edges > 0 else 0
        
        logger.info(f"Dijkstra visualization: {n_h1} verts→H1 (green), {n_h2} verts→H2 (orange), {n_yellow} parting edges (yellow), {n_unreached} unreached")
    
    def clear_dijkstra_result(self):
        """Remove Dijkstra result visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._dijkstra_actor is not None:
            try:
                self.plotter.remove_actor(self._dijkstra_actor)
            except Exception:
                pass
            self._dijkstra_actor = None
        
        if self._dijkstra_primary_cuts_actor is not None:
            try:
                self.plotter.remove_actor(self._dijkstra_primary_cuts_actor)
            except Exception:
                pass
            self._dijkstra_primary_cuts_actor = None
    
    def set_dijkstra_visible(self, visible: bool):
        """Set visibility of Dijkstra result (both H1/H2 and primary cuts)."""
        self._dijkstra_visible = visible
        
        if self._dijkstra_actor is not None:
            self._dijkstra_actor.SetVisibility(visible and self._dijkstra_show_h1h2)
        if self._dijkstra_primary_cuts_actor is not None:
            self._dijkstra_primary_cuts_actor.SetVisibility(visible)
        self.plotter.update()
    
    def set_dijkstra_show_h1h2(self, show: bool):
        """Set visibility of H1/H2 edges only (green/orange). Primary cuts (yellow) always shown when dijkstra visible."""
        self._dijkstra_show_h1h2 = show
        
        if self._dijkstra_actor is not None:
            self._dijkstra_actor.SetVisibility(self._dijkstra_visible and show)
            self.plotter.update()
    
    @property
    def dijkstra_visible(self) -> bool:
        """Check if Dijkstra result is visible."""
        return self._dijkstra_visible
    
    @property
    def dijkstra_show_h1h2(self) -> bool:
        """Check if H1/H2 edges are visible."""
        return self._dijkstra_show_h1h2
    
    @property
    def has_dijkstra(self) -> bool:
        """Check if Dijkstra result exists."""
        return self._dijkstra_actor is not None or self._dijkstra_primary_cuts_actor is not None

    # =========================================================================
    # SECONDARY CUTS VISUALIZATION
    # =========================================================================
    
    def set_secondary_cuts(
        self,
        vertices: np.ndarray,
        secondary_cut_edges: list
    ):
        """
        Display secondary cutting edges in red.
        
        Args:
            vertices: (N, 3) vertex positions
            secondary_cut_edges: List of (vi, vj) tuples representing secondary cut edges
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove existing actor
        self.clear_secondary_cuts()
        
        n_edges = len(secondary_cut_edges)
        if n_edges == 0:
            logger.info("No secondary cutting edges to visualize")
            return
        
        logger.info(f"Setting secondary cuts visualization: {n_edges} edges")
        
        # Convert to numpy array
        edges = np.array(secondary_cut_edges, dtype=np.int64)
        
        # All edges are red
        edge_colors = np.full((n_edges, 3), [255, 0, 0], dtype=np.uint8)
        
        # Build cells array for lines: [2, p0, p1, 2, p0, p1, ...]
        cells = np.empty((n_edges, 3), dtype=np.int64)
        cells[:, 0] = 2  # Each line has 2 points
        cells[:, 1] = edges[:, 0]
        cells[:, 2] = edges[:, 1]
        cells = cells.ravel()
        
        # Cell types: VTK_LINE = 3
        cell_types = np.full(n_edges, 3, dtype=np.uint8)
        
        # Create unstructured grid for lines
        edge_grid = pv.UnstructuredGrid(cells, cell_types, vertices.astype(np.float64))
        edge_grid.cell_data['colors'] = edge_colors
        
        # Add edges as red lines - thicker than Dijkstra edges
        self._secondary_cuts_actor = self.plotter.add_mesh(
            edge_grid,
            scalars='colors',
            rgb=True,
            line_width=5,
            show_edges=False,
            show_scalar_bar=False
        )
        
        if self._secondary_cuts_actor is not None:
            self._secondary_cuts_actor.SetVisibility(self._secondary_cuts_visible)
        
        self.plotter.update()
        logger.info(f"Secondary cuts visualization: {n_edges} edges shown in red")
    
    def clear_secondary_cuts(self):
        """Remove secondary cuts visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._secondary_cuts_actor is not None:
            try:
                self.plotter.remove_actor(self._secondary_cuts_actor)
                self.plotter.update()  # Force visual update after removal
            except Exception:
                pass
            self._secondary_cuts_actor = None
    
    def set_secondary_cuts_visible(self, visible: bool):
        """Set visibility of secondary cuts."""
        self._secondary_cuts_visible = visible
        
        if self._secondary_cuts_actor is not None:
            self._secondary_cuts_actor.SetVisibility(visible)
            self.plotter.update()
    
    @property
    def secondary_cuts_visible(self) -> bool:
        """Check if secondary cuts are visible."""
        return self._secondary_cuts_visible
    
    @property
    def has_secondary_cuts(self) -> bool:
        """Check if secondary cuts exist."""
        return self._secondary_cuts_actor is not None

    # =========================================================================
    # PARTING SURFACE VISUALIZATION
    # =========================================================================
    
    # Color for primary parting surface (blue, semi-transparent)
    PARTING_SURFACE_COLOR = "#3399ff"  # Blue
    PARTING_SURFACE_OPACITY = 0.7
    
    # Color for gap-fill triangles (yellow)
    GAP_FILL_COLOR = "#ffcc00"  # Yellow
    GAP_FILL_OPACITY = 0.8
    
    # Color for secondary parting surface (red, semi-transparent)
    SECONDARY_PARTING_SURFACE_COLOR = "#ff4444"  # Red
    SECONDARY_PARTING_SURFACE_OPACITY = 0.7
    
    def set_parting_surface(self, parting_mesh: trimesh.Trimesh, fill_face_indices: np.ndarray = None):
        """
        Set and display the parting surface mesh.
        
        Args:
            parting_mesh: The parting surface mesh (triangulated surface separating mold halves)
            fill_face_indices: Optional array of face indices that are gap-fill triangles (shown in yellow)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._parting_surface_mesh = parting_mesh
        
        # Remove existing actors
        if self._parting_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._parting_surface_actor)
            except Exception:
                pass
            self._parting_surface_actor = None
        
        # Remove existing gap-fill actor if any
        if hasattr(self, '_gap_fill_actor') and self._gap_fill_actor is not None:
            try:
                self.plotter.remove_actor(self._gap_fill_actor)
            except Exception:
                pass
            self._gap_fill_actor = None
        
        # If we have fill face indices, split into two meshes
        if fill_face_indices is not None and len(fill_face_indices) > 0:
            # Create mask for main surface (non-fill faces)
            all_faces = np.arange(len(parting_mesh.faces))
            fill_set = set(fill_face_indices.tolist())
            main_face_mask = np.array([i not in fill_set for i in all_faces])
            fill_face_mask = np.array([i in fill_set for i in all_faces])
            
            # Main parting surface (blue)
            if np.any(main_face_mask):
                main_mesh = trimesh.Trimesh(
                    vertices=parting_mesh.vertices,
                    faces=parting_mesh.faces[main_face_mask],
                    process=False
                )
                pv_main = self._trimesh_to_pyvista(main_mesh)
                self._parting_surface_actor = self.plotter.add_mesh(
                    pv_main,
                    color=self.PARTING_SURFACE_COLOR,
                    opacity=self.PARTING_SURFACE_OPACITY,
                    pbr=True,
                    metallic=0.1,
                    roughness=0.6,
                    smooth_shading=True,
                    show_edges=True,
                    edge_color='#1166cc',
                    line_width=1,
                    style='surface',
                )
            
            # Gap-fill triangles (yellow)
            if np.any(fill_face_mask):
                fill_mesh = trimesh.Trimesh(
                    vertices=parting_mesh.vertices,
                    faces=parting_mesh.faces[fill_face_mask],
                    process=False
                )
                pv_fill = self._trimesh_to_pyvista(fill_mesh)
                self._gap_fill_actor = self.plotter.add_mesh(
                    pv_fill,
                    color=self.GAP_FILL_COLOR,
                    opacity=self.GAP_FILL_OPACITY,
                    pbr=True,
                    metallic=0.1,
                    roughness=0.6,
                    smooth_shading=True,
                    show_edges=True,
                    edge_color='#cc9900',  # Darker yellow for edges
                    line_width=1,
                    style='surface',
                )
                logger.info(f"Gap-fill triangles displayed in yellow: {len(fill_face_indices)} faces")
        else:
            # No fill faces - show entire mesh in blue
            pv_surface = self._trimesh_to_pyvista(parting_mesh)
            self._parting_surface_actor = self.plotter.add_mesh(
                pv_surface,
                color=self.PARTING_SURFACE_COLOR,
                opacity=self.PARTING_SURFACE_OPACITY,
                pbr=True,
                metallic=0.1,
                roughness=0.6,
                smooth_shading=True,
                show_edges=True,
                edge_color='#1166cc',
                line_width=1,
                style='surface',
            )
        
        self.plotter.update()
        logger.info(f"Parting surface displayed: {len(parting_mesh.vertices)} vertices, {len(parting_mesh.faces)} faces")
    
    def clear_parting_surface(self):
        """Remove parting surface visualization from the scene."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._parting_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._parting_surface_actor)
            except Exception:
                pass
            self._parting_surface_actor = None
        
        # Also clear gap-fill actor
        if hasattr(self, '_gap_fill_actor') and self._gap_fill_actor is not None:
            try:
                self.plotter.remove_actor(self._gap_fill_actor)
            except Exception:
                pass
            self._gap_fill_actor = None
        
        self._parting_surface_mesh = None
        self.plotter.update()
        logger.info("Parting surface cleared")
    
    def set_parting_surface_visible(self, visible: bool):
        """
        Set visibility of the parting surface.
        
        Args:
            visible: True to show parting surface, False to hide
        """
        self._parting_surface_visible = visible
        if self._parting_surface_actor is not None:
            self._parting_surface_actor.SetVisibility(visible)
        # Also set visibility for gap-fill actor
        if hasattr(self, '_gap_fill_actor') and self._gap_fill_actor is not None:
            self._gap_fill_actor.SetVisibility(visible)
        self.plotter.update()
        logger.debug(f"Parting surface visibility set to {visible}")
    
    def set_parting_surface_opacity(self, opacity: float):
        """
        Set the opacity of the parting surface.
        
        Args:
            opacity: Opacity value from 0.0 (transparent) to 1.0 (opaque)
        """
        if self._parting_surface_actor is not None:
            prop = self._parting_surface_actor.GetProperty()
            if prop:
                prop.SetOpacity(opacity)
        # Also set opacity for gap-fill actor
        if hasattr(self, '_gap_fill_actor') and self._gap_fill_actor is not None:
            prop = self._gap_fill_actor.GetProperty()
            if prop:
                prop.SetOpacity(opacity)
        self.plotter.update()
    
    @property
    def parting_surface_visible(self) -> bool:
        """Check if parting surface is visible."""
        return self._parting_surface_visible
    
    @property
    def has_parting_surface(self) -> bool:
        """Check if parting surface exists."""
        return self._parting_surface_mesh is not None

    # =========================================================================
    # SECONDARY PARTING SURFACE VISUALIZATION
    # =========================================================================
    
    # Color for secondary surface gap-fill triangles (orange)
    SECONDARY_GAP_FILL_COLOR = "#ff9900"  # Orange
    SECONDARY_GAP_FILL_OPACITY = 0.8
    
    def set_secondary_parting_surface(self, parting_mesh: trimesh.Trimesh, fill_face_indices: np.ndarray = None):
        """
        Set and display the secondary parting surface mesh.
        
        Args:
            parting_mesh: The secondary parting surface mesh (from secondary cut edges)
            fill_face_indices: Optional array of face indices that are gap-fill triangles (shown in orange)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._secondary_parting_surface_mesh = parting_mesh
        
        # Remove existing actor
        if self._secondary_parting_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._secondary_parting_surface_actor)
            except Exception:
                pass
            self._secondary_parting_surface_actor = None
        
        # Remove existing secondary gap-fill actor if any
        if hasattr(self, '_secondary_gap_fill_actor') and self._secondary_gap_fill_actor is not None:
            try:
                self.plotter.remove_actor(self._secondary_gap_fill_actor)
            except Exception:
                pass
            self._secondary_gap_fill_actor = None
        
        # If we have fill face indices, split into two meshes
        if fill_face_indices is not None and len(fill_face_indices) > 0:
            # Create mask for main surface (non-fill faces)
            all_faces = np.arange(len(parting_mesh.faces))
            fill_set = set(fill_face_indices.tolist())
            main_face_mask = np.array([i not in fill_set for i in all_faces])
            fill_face_mask = np.array([i in fill_set for i in all_faces])
            
            # Main secondary surface (red)
            if np.any(main_face_mask):
                main_mesh = trimesh.Trimesh(
                    vertices=parting_mesh.vertices,
                    faces=parting_mesh.faces[main_face_mask],
                    process=False
                )
                pv_main = self._trimesh_to_pyvista(main_mesh)
                self._secondary_parting_surface_actor = self.plotter.add_mesh(
                    pv_main,
                    color=self.SECONDARY_PARTING_SURFACE_COLOR,
                    opacity=self.SECONDARY_PARTING_SURFACE_OPACITY,
                    pbr=True,
                    metallic=0.1,
                    roughness=0.6,
                    smooth_shading=True,
                    show_edges=True,
                    edge_color='#cc2222',
                    line_width=1,
                    style='surface',
                )
            
            # Gap-fill triangles (orange)
            if np.any(fill_face_mask):
                fill_mesh = trimesh.Trimesh(
                    vertices=parting_mesh.vertices,
                    faces=parting_mesh.faces[fill_face_mask],
                    process=False
                )
                pv_fill = self._trimesh_to_pyvista(fill_mesh)
                self._secondary_gap_fill_actor = self.plotter.add_mesh(
                    pv_fill,
                    color=self.SECONDARY_GAP_FILL_COLOR,
                    opacity=self.SECONDARY_GAP_FILL_OPACITY,
                    pbr=True,
                    metallic=0.1,
                    roughness=0.6,
                    smooth_shading=True,
                    show_edges=True,
                    edge_color='#cc7700',  # Darker orange for edges
                    line_width=1,
                    style='surface',
                )
                logger.info(f"Secondary gap-fill triangles displayed in orange: {len(fill_face_indices)} faces")
        else:
            # No fill faces - show entire mesh in red
            pv_surface = self._trimesh_to_pyvista(parting_mesh)
            # Add secondary parting surface mesh (semi-transparent red)
            self._secondary_parting_surface_actor = self.plotter.add_mesh(
                pv_surface,
                color=self.SECONDARY_PARTING_SURFACE_COLOR,
                opacity=self.SECONDARY_PARTING_SURFACE_OPACITY,
                pbr=True,
                metallic=0.1,
                roughness=0.6,
                smooth_shading=True,
                show_edges=True,
                edge_color='#cc2222',  # Darker red for edges
                line_width=1,
                style='surface',
            )
        
        self.plotter.update()
        logger.info(f"Secondary parting surface displayed: {len(parting_mesh.vertices)} vertices, {len(parting_mesh.faces)} faces")
    
    def clear_secondary_parting_surface(self):
        """Remove secondary parting surface visualization from the scene."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._secondary_parting_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._secondary_parting_surface_actor)
            except Exception:
                pass
            self._secondary_parting_surface_actor = None
        
        # Also clear secondary gap-fill actor
        if hasattr(self, '_secondary_gap_fill_actor') and self._secondary_gap_fill_actor is not None:
            try:
                self.plotter.remove_actor(self._secondary_gap_fill_actor)
            except Exception:
                pass
            self._secondary_gap_fill_actor = None
        
        self._secondary_parting_surface_mesh = None
        self.plotter.update()
        logger.info("Secondary parting surface cleared")
    
    def set_secondary_parting_surface_visible(self, visible: bool):
        """
        Set visibility of the secondary parting surface.
        
        Args:
            visible: True to show secondary parting surface, False to hide
        """
        self._secondary_parting_surface_visible = visible
        if self._secondary_parting_surface_actor is not None:
            self._secondary_parting_surface_actor.SetVisibility(visible)
        # Also set visibility for secondary gap-fill actor
        if hasattr(self, '_secondary_gap_fill_actor') and self._secondary_gap_fill_actor is not None:
            self._secondary_gap_fill_actor.SetVisibility(visible)
        self.plotter.update()
        logger.debug(f"Secondary parting surface visibility set to {visible}")
    
    def set_secondary_parting_surface_opacity(self, opacity: float):
        """
        Set the opacity of the secondary parting surface.
        
        Args:
            opacity: Opacity value from 0.0 (transparent) to 1.0 (opaque)
        """
        if self._secondary_parting_surface_actor is not None:
            prop = self._secondary_parting_surface_actor.GetProperty()
            if prop:
                prop.SetOpacity(opacity)
        # Also set opacity for secondary gap-fill actor
        if hasattr(self, '_secondary_gap_fill_actor') and self._secondary_gap_fill_actor is not None:
            prop = self._secondary_gap_fill_actor.GetProperty()
            if prop:
                prop.SetOpacity(opacity)
        self.plotter.update()
    
    @property
    def secondary_parting_surface_visible(self) -> bool:
        """Check if secondary parting surface is visible."""
        return self._secondary_parting_surface_visible
    
    @property
    def has_secondary_parting_surface(self) -> bool:
        """Check if secondary parting surface exists."""
        return self._secondary_parting_surface_mesh is not None

    # =========================================================================
    # EDGE DEBUG MODE FOR PARTING SURFACE
    # =========================================================================
    
    def set_part_mesh_reference(self, part_mesh: trimesh.Trimesh):
        """
        Store reference to the part mesh for edge distance calculations.
        
        Args:
            part_mesh: The original part mesh (for computing distances to parting surface edges)
        """
        self._part_mesh_ref = part_mesh
        logger.debug("Part mesh reference stored for edge debug mode")
    
    def set_tet_result_reference(self, tet_result):
        """
        Store reference to the tetrahedral mesh result for escape label analysis.
        
        Args:
            tet_result: TetrahedralMeshResult with Dijkstra escape labels
        """
        self._tet_result_ref = tet_result
        logger.debug("Tet result reference stored for edge debug mode")
    
    def set_parting_surface_result_reference(self, parting_surface_result):
        """
        Store reference to the parting surface result for vertex_to_edge mapping.
        
        Args:
            parting_surface_result: PartingSurfaceResult with vertex_to_edge mapping
        """
        self._parting_surface_result_ref = parting_surface_result
        logger.debug("Parting surface result reference stored for edge debug mode")
    
    def enable_edge_debug_mode(self, enabled: bool = True):
        """
        Enable or disable edge debug mode for the parting surface.
        
        When enabled:
        - Boundary edges of the parting surface are highlighted in yellow
        - Click on an edge to see debug info in the terminal
        - The edge closest to click is selected and highlighted in red
        
        Args:
            enabled: True to enable, False to disable
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._edge_debug_mode = enabled
        
        if enabled:
            if self._parting_surface_mesh is None:
                logger.warning("No parting surface mesh available for edge debug mode")
                return
            
            # Extract and visualize boundary edges
            self._extract_and_show_boundary_edges()
            
            # Enable point picking for edge selection
            self._enable_edge_picking()
            
            logger.info("Edge debug mode ENABLED - click on boundary edges to inspect")
        else:
            # Disable picking and clear visualization
            self._disable_edge_picking()
            self._clear_boundary_edges_visualization()
            logger.info("Edge debug mode DISABLED")
    
    def _extract_and_show_boundary_edges(self):
        """Extract boundary edges from parting surface and visualize them."""
        if self._parting_surface_mesh is None:
            return
        
        mesh = self._parting_surface_mesh
        
        # Find boundary edges (edges that appear in only one face)
        # In trimesh, edges_unique gives unique edges, edges_unique_inverse maps faces to edges
        edges = mesh.edges_unique
        
        # Count how many faces each edge belongs to
        edge_face_counts = np.zeros(len(edges), dtype=np.int32)
        for face in mesh.faces:
            # Get edges for this face
            for i in range(3):
                v0, v1 = face[i], face[(i+1) % 3]
                # Find this edge in edges_unique
                edge_key = tuple(sorted([v0, v1]))
                # Search for matching edge
                for e_idx, e in enumerate(edges):
                    if tuple(sorted(e)) == edge_key:
                        edge_face_counts[e_idx] += 1
                        break
        
        # Boundary edges have count == 1
        boundary_mask = edge_face_counts == 1
        boundary_edges = edges[boundary_mask]
        
        if len(boundary_edges) == 0:
            logger.info("No boundary edges found (mesh is watertight)")
            return
        
        logger.info(f"Found {len(boundary_edges)} boundary edges on parting surface")
        
        # Store boundary edge data for picking
        vertices = mesh.vertices
        self._boundary_edges_data = {
            'edges': boundary_edges,
            'vertices': vertices,
            'midpoints': np.array([(vertices[e[0]] + vertices[e[1]]) / 2 for e in boundary_edges]),
            'lengths': np.array([np.linalg.norm(vertices[e[1]] - vertices[e[0]]) for e in boundary_edges]),
        }
        
        # Compute distances to part mesh if available
        if self._part_mesh_ref is not None:
            self._compute_edge_to_part_distances()
        
        # Visualize boundary edges
        self._visualize_boundary_edges()
    
    def _compute_edge_to_part_distances(self):
        """Compute distance from each boundary edge to the part mesh."""
        if self._boundary_edges_data is None or self._part_mesh_ref is None:
            return
        
        from scipy.spatial import KDTree
        
        # Build KDTree from part mesh vertices
        part_vertices = np.array(self._part_mesh_ref.vertices)
        part_tree = KDTree(part_vertices)
        
        midpoints = self._boundary_edges_data['midpoints']
        edges = self._boundary_edges_data['edges']
        vertices = self._boundary_edges_data['vertices']
        
        # For each boundary edge, find distance to nearest part vertex
        # and also sample along the edge
        distances = []
        nearest_part_points = []
        nearest_part_vertex_indices = []
        
        for i, edge in enumerate(edges):
            v0, v1 = vertices[edge[0]], vertices[edge[1]]
            midpoint = midpoints[i]
            
            # Sample edge at multiple points
            sample_points = np.array([
                v0,
                (v0 + midpoint) / 2,
                midpoint,
                (midpoint + v1) / 2,
                v1
            ])
            
            # Find minimum distance across all sample points
            dists, indices = part_tree.query(sample_points)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            
            distances.append(min_dist)
            nearest_part_points.append(part_vertices[indices[min_idx]])
            nearest_part_vertex_indices.append(indices[min_idx])
        
        self._boundary_edges_data['part_distances'] = np.array(distances)
        self._boundary_edges_data['nearest_part_points'] = np.array(nearest_part_points)
        self._boundary_edges_data['nearest_part_vertex_indices'] = np.array(nearest_part_vertex_indices)
        
        # Log statistics
        dist_arr = np.array(distances)
        logger.info(f"Edge-to-part distances: min={dist_arr.min():.4f}, max={dist_arr.max():.4f}, "
                   f"mean={dist_arr.mean():.4f}, median={np.median(dist_arr):.4f}")
    
    def _visualize_boundary_edges(self):
        """Visualize boundary edges with color coding based on distance."""
        if self._boundary_edges_data is None:
            return
        
        # Clear existing visualization
        if self._boundary_edges_actor is not None:
            try:
                self.plotter.remove_actor(self._boundary_edges_actor)
            except Exception:
                pass
            self._boundary_edges_actor = None
        
        edges = self._boundary_edges_data['edges']
        vertices = self._boundary_edges_data['vertices']
        
        # Build line segments for PyVista
        n_edges = len(edges)
        lines = np.zeros((n_edges, 3), dtype=np.int64)
        lines[:, 0] = 2  # Each line has 2 points
        lines[:, 1] = np.arange(n_edges) * 2      # First point index
        lines[:, 2] = np.arange(n_edges) * 2 + 1  # Second point index
        
        # Flatten points: each edge contributes 2 vertices
        points = np.zeros((n_edges * 2, 3))
        for i, edge in enumerate(edges):
            points[i * 2] = vertices[edge[0]]
            points[i * 2 + 1] = vertices[edge[1]]
        
        # Create PyVista PolyData with lines
        edge_mesh = pv.PolyData(points, lines=lines.flatten())
        
        # Color by distance if available
        if 'part_distances' in self._boundary_edges_data:
            distances = self._boundary_edges_data['part_distances']
            # Assign same distance to both vertices of each edge
            scalars = np.repeat(distances, 2)
            edge_mesh['distance'] = scalars
            
            self._boundary_edges_actor = self.plotter.add_mesh(
                edge_mesh,
                scalars='distance',
                cmap='coolwarm',  # Blue (close) to Red (far)
                line_width=4,
                render_lines_as_tubes=True,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Distance to Part',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.1,
                    'width': 0.1,
                    'height': 0.3,
                }
            )
        else:
            # No distance data - use yellow
            self._boundary_edges_actor = self.plotter.add_mesh(
                edge_mesh,
                color='yellow',
                line_width=4,
                render_lines_as_tubes=True,
            )
        
        self.plotter.update()
        logger.info(f"Visualized {n_edges} boundary edges")
    
    def _enable_edge_picking(self):
        """Enable point picking for edge selection."""
        try:
            # Use point picking to select nearest edge
            self.plotter.enable_point_picking(
                callback=self._on_edge_picked,
                show_message=True,
                color='red',
                point_size=15,
                show_point=True,
                tolerance=0.025,  # Tolerance for picking
            )
            logger.debug("Edge picking enabled")
        except Exception as e:
            logger.error(f"Failed to enable edge picking: {e}")
    
    def _disable_edge_picking(self):
        """Disable point picking."""
        try:
            self.plotter.disable_picking()
            logger.debug("Edge picking disabled")
        except Exception as e:
            logger.debug(f"Could not disable picking: {e}")
    
    def _on_edge_picked(self, point):
        """Handle point pick event - find nearest boundary edge and show debug info."""
        if point is None or self._boundary_edges_data is None:
            return
        
        picked_point = np.array(point)
        midpoints = self._boundary_edges_data['midpoints']
        
        # Find nearest edge by midpoint distance
        distances_to_pick = np.linalg.norm(midpoints - picked_point, axis=1)
        nearest_idx = np.argmin(distances_to_pick)
        
        # Get edge data (parting surface edge)
        ps_edge = self._boundary_edges_data['edges'][nearest_idx]
        ps_vertices = self._boundary_edges_data['vertices']
        v0, v1 = ps_vertices[ps_edge[0]], ps_vertices[ps_edge[1]]
        midpoint = midpoints[nearest_idx]
        length = self._boundary_edges_data['lengths'][nearest_idx]
        
        # Highlight selected edge
        self._highlight_selected_edge(v0, v1)
        
        # Print debug info
        print("\n" + "="*70)
        print("🔍 PARTING SURFACE BOUNDARY EDGE DEBUG")
        print("="*70)
        print(f"Boundary edge index: {nearest_idx}")
        print(f"PS vertex indices: {ps_edge[0]} → {ps_edge[1]}")
        print(f"Vertex 0: [{v0[0]:.4f}, {v0[1]:.4f}, {v0[2]:.4f}]")
        print(f"Vertex 1: [{v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}]")
        print(f"Midpoint: [{midpoint[0]:.4f}, {midpoint[1]:.4f}, {midpoint[2]:.4f}]")
        print(f"Length: {length:.4f}")
        
        # Try to trace back to tetrahedral mesh edges
        if self._parting_surface_result_ref is not None and self._tet_result_ref is not None:
            self._print_tet_edge_info(ps_edge, ps_vertices)
        
        if 'part_distances' in self._boundary_edges_data:
            dist = self._boundary_edges_data['part_distances'][nearest_idx]
            print(f"\n📏 DISTANCE TO PART MESH")
            print(f"   Minimum distance: {dist:.4f}")
            
            if 'nearest_part_points' in self._boundary_edges_data:
                nearest_pt = self._boundary_edges_data['nearest_part_points'][nearest_idx]
                nearest_idx_part = self._boundary_edges_data['nearest_part_vertex_indices'][nearest_idx]
                print(f"   Nearest part point: [{nearest_pt[0]:.4f}, {nearest_pt[1]:.4f}, {nearest_pt[2]:.4f}]")
                print(f"   Nearest part vertex index: {nearest_idx_part}")
                
                # Check if edge should be connected (distance very small)
                if dist < 0.1:
                    print(f"   ✅ Edge is CLOSE to part (dist < 0.1)")
                elif dist < 1.0:
                    print(f"   ⚠️  Edge is NEAR part (0.1 < dist < 1.0)")
                else:
                    print(f"   ❌ Edge is FAR from part (dist >= 1.0)")
        
        # Additional analysis: check edge direction
        edge_direction = v1 - v0
        edge_direction_normalized = edge_direction / np.linalg.norm(edge_direction)
        print(f"\n📐 EDGE GEOMETRY")
        print(f"   Direction: [{edge_direction_normalized[0]:.4f}, {edge_direction_normalized[1]:.4f}, {edge_direction_normalized[2]:.4f}]")
        
        # Check if edge is mostly horizontal or vertical
        z_component = abs(edge_direction_normalized[2])
        if z_component > 0.9:
            print(f"   Orientation: VERTICAL (Z-aligned)")
        elif z_component < 0.1:
            print(f"   Orientation: HORIZONTAL (XY-plane)")
        else:
            print(f"   Orientation: DIAGONAL (mixed)")
        
        print("="*70 + "\n")
        
        # Also log to the logger
        logger.info(f"Selected edge {nearest_idx}: vertices {ps_edge}, length={length:.4f}, "
                   f"part_dist={self._boundary_edges_data.get('part_distances', [0])[nearest_idx] if 'part_distances' in self._boundary_edges_data else 'N/A':.4f}")
    
    def _print_tet_edge_info(self, ps_edge, ps_vertices):
        """Print tetrahedral mesh edge info for the parting surface edge vertices."""
        ps_result = self._parting_surface_result_ref
        tet_result = self._tet_result_ref
        
        # Check if vertex_to_edge is available
        vertex_to_edge = getattr(ps_result, 'vertex_to_edge', None)
        if vertex_to_edge is None:
            print("\n⚠️  No vertex_to_edge mapping available (surface may have been repaired/smoothed)")
            return
        
        print(f"\n🔬 TETRAHEDRAL MESH ANALYSIS")
        
        # Each parting surface vertex corresponds to the midpoint of a tet mesh edge
        # vertex_to_edge[ps_vertex] = tet_edge_index
        for i, ps_v_idx in enumerate(ps_edge):
            if ps_v_idx < len(vertex_to_edge):
                tet_edge_idx = vertex_to_edge[ps_v_idx]
                tet_edge = tet_result.edges[tet_edge_idx]
                tet_v0_idx, tet_v1_idx = int(tet_edge[0]), int(tet_edge[1])
                
                print(f"\n   PS Vertex {ps_v_idx} → Tet Edge {tet_edge_idx}: ({tet_v0_idx}, {tet_v1_idx})")
                
                # Get tet vertex positions
                tet_verts = tet_result.vertices_original if tet_result.vertices_original is not None else tet_result.vertices
                tv0 = tet_verts[tet_v0_idx]
                tv1 = tet_verts[tet_v1_idx]
                print(f"      Tet V{tet_v0_idx}: [{tv0[0]:.4f}, {tv0[1]:.4f}, {tv0[2]:.4f}]")
                print(f"      Tet V{tet_v1_idx}: [{tv1[0]:.4f}, {tv1[1]:.4f}, {tv1[2]:.4f}]")
                
                # Get boundary labels for these vertices
                if tet_result.boundary_labels is not None:
                    bl0 = tet_result.boundary_labels[tet_v0_idx]
                    bl1 = tet_result.boundary_labels[tet_v1_idx]
                    label_names = {-1: "INNER_BOUNDARY (part)", 0: "INTERIOR", 1: "H1", 2: "H2"}
                    print(f"      Boundary labels: V{tet_v0_idx}={label_names.get(bl0, bl0)}, V{tet_v1_idx}={label_names.get(bl1, bl1)}")
                
                # Get escape labels if available
                if tet_result.seed_vertex_indices is not None and tet_result.seed_escape_labels is not None:
                    seed_indices = tet_result.seed_vertex_indices
                    escape_labels = tet_result.seed_escape_labels
                    
                    # Find if these vertices are in the seed list
                    escape_names = {0: "UNREACHABLE", 1: "→H1", 2: "→H2"}
                    
                    for v_idx in [tet_v0_idx, tet_v1_idx]:
                        if v_idx in seed_indices:
                            seed_pos = np.where(seed_indices == v_idx)[0][0]
                            el = escape_labels[seed_pos]
                            print(f"      Escape label V{v_idx}: {escape_names.get(el, el)}")
                        else:
                            # Vertex not in seed list - it's on H1 or H2 boundary
                            if tet_result.boundary_labels is not None:
                                bl = tet_result.boundary_labels[v_idx]
                                if bl == 1:
                                    print(f"      Escape label V{v_idx}: ON_H1 (boundary)")
                                elif bl == 2:
                                    print(f"      Escape label V{v_idx}: ON_H2 (boundary)")
                                else:
                                    print(f"      Escape label V{v_idx}: NOT_IN_SEEDS (bl={bl})")
            else:
                print(f"\n   PS Vertex {ps_v_idx}: No mapping available")
    
    def _highlight_selected_edge(self, v0: np.ndarray, v1: np.ndarray):
        """Highlight the selected edge in red."""
        # Clear previous selection
        if self._selected_edge_actor is not None:
            try:
                self.plotter.remove_actor(self._selected_edge_actor)
            except Exception:
                pass
            self._selected_edge_actor = None
        
        # Create line for selected edge
        points = np.array([v0, v1])
        lines = np.array([2, 0, 1])  # 2 points, indices 0 and 1
        selected_edge = pv.PolyData(points, lines=lines)
        
        self._selected_edge_actor = self.plotter.add_mesh(
            selected_edge,
            color='magenta',
            line_width=8,
            render_lines_as_tubes=True,
        )
        
        self.plotter.update()
    
    def _clear_boundary_edges_visualization(self):
        """Clear all boundary edge visualization."""
        if self._boundary_edges_actor is not None:
            try:
                self.plotter.remove_actor(self._boundary_edges_actor)
            except Exception:
                pass
            self._boundary_edges_actor = None
        
        if self._selected_edge_actor is not None:
            try:
                self.plotter.remove_actor(self._selected_edge_actor)
            except Exception:
                pass
            self._selected_edge_actor = None
        
        self._boundary_edges_data = None
        self.plotter.update()
    
    @property
    def edge_debug_mode(self) -> bool:
        """Check if edge debug mode is enabled."""
        return self._edge_debug_mode

    # =========================================================================
    # TRIANGLE DEBUG MODE - For analyzing bad triangles in membranes
    # =========================================================================
    
    def set_triangle_debug_mode(self, enabled: bool, mesh: trimesh.Trimesh = None, 
                                 vertex_boundary_type: np.ndarray = None,
                                 part_mesh: trimesh.Trimesh = None):
        """
        Enable/disable triangle debug mode for membrane analysis.
        
        When enabled, clicking on a triangle will log detailed information about it
        including vertex positions, edge lengths, aspect ratio, neighbors, and more.
        
        Args:
            enabled: Whether to enable triangle debug mode
            mesh: The membrane mesh to debug (uses current parting surface if None)
            vertex_boundary_type: Optional array of boundary types for vertices
            part_mesh: Optional part mesh for distance calculations
        """
        self._triangle_debug_mode = enabled
        
        if enabled:
            # Store references
            if mesh is not None:
                self._triangle_debug_mesh_ref = mesh
            elif self._parting_surface_mesh is not None:
                self._triangle_debug_mesh_ref = self._parting_surface_mesh
            else:
                logger.warning("No mesh available for triangle debug mode")
                self._triangle_debug_mode = False
                return
            
            self._triangle_debug_boundary_type_ref = vertex_boundary_type
            if part_mesh is not None:
                self._part_mesh_ref = part_mesh
            
            # Enable cell picking
            self._enable_triangle_picking()
            logger.info("🔍 Triangle debug mode ENABLED - Click on triangles to analyze them")
            print("\n" + "="*70)
            print("🔍 TRIANGLE DEBUG MODE ENABLED")
            print("="*70)
            print("Click on any triangle in the membrane to see detailed analysis.")
            print("This will help identify problematic triangles.")
            print("="*70 + "\n")
        else:
            # Disable picking and clear visualization
            self._disable_triangle_picking()
            self._clear_selected_triangle()
            self._triangle_debug_mesh_ref = None
            self._triangle_debug_boundary_type_ref = None
            logger.info("Triangle debug mode DISABLED")
    
    def _enable_triangle_picking(self):
        """Enable point picking for triangle selection (more reliable than cell picking)."""
        try:
            # Use point picking - we'll find the nearest triangle ourselves
            # This is more reliable than cell picking which can pick from any mesh
            self.plotter.enable_point_picking(
                callback=self._on_point_picked_for_triangle,
                show_message=True,
                color='yellow',
                point_size=15,
                show_point=True,
                tolerance=0.025,
                pickable_window=True,  # Pick anywhere in window
            )
            logger.debug("Triangle picking enabled (point-based)")
            print("🎯 Click anywhere on the membrane to select a triangle")
        except Exception as e:
            logger.error(f"Failed to enable triangle picking: {e}")
    
    def _on_point_picked_for_triangle(self, point):
        """Handle point pick and find nearest triangle in the debug mesh."""
        if point is None or self._triangle_debug_mesh_ref is None:
            logger.debug("No point or no mesh reference")
            return
        
        picked_point = np.array(point)
        mesh = self._triangle_debug_mesh_ref
        
        # Find the nearest triangle by centroid distance
        centroids = mesh.triangles_center
        distances = np.linalg.norm(centroids - picked_point, axis=1)
        cell_id = int(np.argmin(distances))
        
        min_dist = distances[cell_id]
        logger.debug(f"Picked point {picked_point}, nearest triangle {cell_id} at distance {min_dist:.4f}")
        
        # Call the triangle analysis with this cell ID
        self._analyze_triangle(cell_id)
    
    def _disable_triangle_picking(self):
        """Disable cell picking."""
        try:
            self.plotter.disable_picking()
            logger.debug("Triangle picking disabled")
        except Exception as e:
            logger.debug(f"Could not disable picking: {e}")
    
    def _clear_selected_triangle(self):
        """Clear the selected triangle highlight."""
        if self._selected_triangle_actor is not None:
            try:
                self.plotter.remove_actor(self._selected_triangle_actor)
            except Exception:
                pass
            self._selected_triangle_actor = None
    
    def _compute_triangle_aspect_ratio(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute aspect ratio of a triangle (0 = degenerate, 1 = equilateral)."""
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)
        
        if a < 1e-10 or b < 1e-10 or c < 1e-10:
            return 0.0
        
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            return 0.0
        area = np.sqrt(area_sq)
        
        ratio = (4 * area * area) / (s * a * b * c)
        return min(1.0, ratio * 2)
    
    def _find_neighbor_faces(self, mesh: trimesh.Trimesh, face_idx: int) -> List[int]:
        """Find indices of faces that share an edge with the given face."""
        if mesh is None or face_idx >= len(mesh.faces):
            return []
        
        target_face = mesh.faces[face_idx]
        target_edges = set()
        for i in range(3):
            v0, v1 = target_face[i], target_face[(i + 1) % 3]
            target_edges.add((min(v0, v1), max(v0, v1)))
        
        neighbors = []
        for fi, face in enumerate(mesh.faces):
            if fi == face_idx:
                continue
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                if edge in target_edges:
                    neighbors.append(fi)
                    break
        
        return neighbors
    
    def _on_triangle_picked(self, picked_cell):
        """Handle cell pick event - analyze the picked triangle."""
        if picked_cell is None or self._triangle_debug_mesh_ref is None:
            return
        
        mesh = self._triangle_debug_mesh_ref
        
        # Get picked cell info
        try:
            # picked_cell is a pyvista object with cell info
            if hasattr(picked_cell, 'cell_data'):
                # Get cell ID from the picked result
                cell_id = picked_cell.get('original_cell_ids', [0])[0] if isinstance(picked_cell, dict) else 0
            else:
                # Try to find the cell by position
                cell_id = self._find_nearest_triangle(picked_cell)
        except Exception:
            cell_id = self._find_nearest_triangle(picked_cell)
        
        if cell_id is None or cell_id >= len(mesh.faces):
            logger.warning(f"Invalid cell ID: {cell_id}")
            return
        
        self._analyze_triangle(cell_id)
    
    def _analyze_triangle(self, cell_id: int):
        """Analyze and display detailed info about a triangle."""
        mesh = self._triangle_debug_mesh_ref
        
        if mesh is None or cell_id >= len(mesh.faces):
            logger.warning(f"Invalid cell ID: {cell_id}")
            return
        
        # Get triangle data
        face = mesh.faces[cell_id]
        v0_idx, v1_idx, v2_idx = face[0], face[1], face[2]
        v0 = mesh.vertices[v0_idx]
        v1 = mesh.vertices[v1_idx]
        v2 = mesh.vertices[v2_idx]
        
        # Compute properties
        edge_a = np.linalg.norm(v1 - v0)
        edge_b = np.linalg.norm(v2 - v1)
        edge_c = np.linalg.norm(v0 - v2)
        
        centroid = (v0 + v1 + v2) / 3
        normal = np.cross(v1 - v0, v2 - v0)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-10:
            normal = normal / normal_len
        
        area = normal_len / 2
        aspect_ratio = self._compute_triangle_aspect_ratio(v0, v1, v2)
        
        # Find neighbors
        neighbors = self._find_neighbor_faces(mesh, cell_id)
        
        # Check if it's a boundary triangle (has edge with only one face)
        is_boundary = len(neighbors) < 3
        
        # Highlight the selected triangle
        self._highlight_selected_triangle(v0, v1, v2)
        
        # Print detailed debug info
        print("\n" + "="*70)
        print("🔺 TRIANGLE DEBUG ANALYSIS")
        print("="*70)
        print(f"Face Index: {cell_id}")
        print(f"Vertex Indices: [{v0_idx}, {v1_idx}, {v2_idx}]")
        print(f"\n📍 VERTEX POSITIONS:")
        print(f"   V0 [{v0_idx}]: [{v0[0]:.6f}, {v0[1]:.6f}, {v0[2]:.6f}]")
        print(f"   V1 [{v1_idx}]: [{v1[0]:.6f}, {v1[1]:.6f}, {v1[2]:.6f}]")
        print(f"   V2 [{v2_idx}]: [{v2[0]:.6f}, {v2[1]:.6f}, {v2[2]:.6f}]")
        
        print(f"\n📐 EDGE LENGTHS:")
        print(f"   Edge V0→V1: {edge_a:.6f}")
        print(f"   Edge V1→V2: {edge_b:.6f}")
        print(f"   Edge V2→V0: {edge_c:.6f}")
        print(f"   Min edge: {min(edge_a, edge_b, edge_c):.6f}")
        print(f"   Max edge: {max(edge_a, edge_b, edge_c):.6f}")
        print(f"   Edge ratio (max/min): {max(edge_a, edge_b, edge_c) / max(min(edge_a, edge_b, edge_c), 1e-10):.2f}")
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"   Area: {area:.8f}")
        print(f"   Aspect Ratio: {aspect_ratio:.4f} (1.0 = equilateral, 0.0 = degenerate)")
        print(f"   Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
        print(f"   Normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
        
        # Quality assessment
        quality_status = "✅ GOOD" if aspect_ratio > 0.3 else ("⚠️ POOR" if aspect_ratio > 0.1 else "❌ BAD")
        print(f"   Quality: {quality_status}")
        
        print(f"\n🔗 CONNECTIVITY:")
        print(f"   Neighbor faces: {neighbors}")
        print(f"   Number of neighbors: {len(neighbors)}")
        print(f"   Is boundary triangle: {'YES' if is_boundary else 'NO'}")
        
        # Check which edges of this triangle are actually mesh boundary edges
        print(f"\n🔺 MESH BOUNDARY EDGE CHECK:")
        edges = [(v0_idx, v1_idx), (v1_idx, v2_idx), (v2_idx, v0_idx)]
        edge_names = ["V0→V1", "V1→V2", "V2→V0"]
        
        # Build edge-to-face count for entire mesh
        edge_face_count = {}
        for fi, face in enumerate(mesh.faces):
            for i in range(3):
                e0, e1 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = (min(e0, e1), max(e0, e1))
                edge_face_count[edge_key] = edge_face_count.get(edge_key, 0) + 1
        
        for (e0, e1), edge_name in zip(edges, edge_names):
            edge_key = (min(e0, e1), max(e0, e1))
            face_count = edge_face_count.get(edge_key, 0)
            is_mesh_boundary = (face_count == 1)
            status = "🔴 MESH BOUNDARY (1 face)" if is_mesh_boundary else f"⚪ Interior ({face_count} faces)"
            print(f"   {edge_name} [{e0}→{e1}]: {status}")
        
        # For each vertex, check if it has ANY mesh boundary edges
        print(f"\n🔍 VERTEX BOUNDARY EDGE CHECK:")
        for vi, v_idx in enumerate([v0_idx, v1_idx, v2_idx]):
            # Find all edges containing this vertex
            vertex_boundary_edges = []
            for (e0, e1), count in edge_face_count.items():
                if count == 1 and (e0 == v_idx or e1 == v_idx):
                    other = e1 if e0 == v_idx else e0
                    vertex_boundary_edges.append(other)
            
            if len(vertex_boundary_edges) == 0:
                print(f"   V{vi} [{v_idx}]: ❌ NO mesh boundary edges (interior vertex)")
            elif len(vertex_boundary_edges) == 1:
                print(f"   V{vi} [{v_idx}]: ⚠️  1 mesh boundary edge (to vertex {vertex_boundary_edges[0]}) - NO FAN POSSIBLE")
            else:
                print(f"   V{vi} [{v_idx}]: ✅ {len(vertex_boundary_edges)} mesh boundary edges (to vertices {vertex_boundary_edges}) - fan should exist")
        
        # Vertex boundary types if available
        if self._triangle_debug_boundary_type_ref is not None:
            bt = self._triangle_debug_boundary_type_ref
            print(f"\n🏷️ VERTEX BOUNDARY TYPES:")
            for vi, v_idx in enumerate([v0_idx, v1_idx, v2_idx]):
                if v_idx < len(bt):
                    btype = bt[v_idx]
                    btype_name = {-1: "INNER (part)", 0: "INTERIOR", 1: "OUTER (H1)", 2: "OUTER (H2)"}.get(btype, f"UNKNOWN ({btype})")
                    print(f"   V{vi} [{v_idx}]: {btype_name}")
        
        # Distance to part if available
        if self._part_mesh_ref is not None:
            try:
                pts = np.array([v0, v1, v2, centroid])
                closest_pts, distances, closest_faces = trimesh.proximity.closest_point(self._part_mesh_ref, pts)
                print(f"\n📏 DISTANCE TO PART MESH:")
                print(f"   V0 distance: {distances[0]:.6f}")
                print(f"   V1 distance: {distances[1]:.6f}")
                print(f"   V2 distance: {distances[2]:.6f}")
                print(f"   Centroid distance: {distances[3]:.6f}")
                
                # ========================================================
                # COLLAR EXTENSION DEBUG INFO
                # ========================================================
                print(f"\n🔧 COLLAR EXTENSION DEBUG:")
                part_face_normals = self._part_mesh_ref.face_normals
                
                for vi, (v_idx, v, closest_pt, closest_face, dist) in enumerate(zip(
                    [v0_idx, v1_idx, v2_idx], [v0, v1, v2], closest_pts[:3], closest_faces[:3], distances[:3]
                )):
                    print(f"\n   --- Vertex V{vi} [{v_idx}] ---")
                    print(f"   Position: [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]")
                    print(f"   Closest pt on part: [{closest_pt[0]:.6f}, {closest_pt[1]:.6f}, {closest_pt[2]:.6f}]")
                    print(f"   Distance to part: {dist:.6f}")
                    
                    # Part normal at closest point
                    if closest_face < len(part_face_normals):
                        part_normal = part_face_normals[closest_face]
                        print(f"   Part normal at closest: [{part_normal[0]:.4f}, {part_normal[1]:.4f}, {part_normal[2]:.4f}]")
                        
                        # Into-part direction
                        into_part = -part_normal
                        print(f"   Into-part direction: [{into_part[0]:.4f}, {into_part[1]:.4f}, {into_part[2]:.4f}]")
                        
                        # Collar point (closest + 2mm into part)
                        collar_depth = 2.0
                        collar_pt = closest_pt + collar_depth * into_part
                        print(f"   Collar point (2mm depth): [{collar_pt[0]:.6f}, {collar_pt[1]:.6f}, {collar_pt[2]:.6f}]")
                        
                        # Check if collar point is inside part
                        try:
                            inside = self._part_mesh_ref.contains([collar_pt])[0]
                            inside_status = "✅ INSIDE" if inside else "❌ OUTSIDE"
                            print(f"   Collar point containment: {inside_status}")
                            
                            # Try opposite direction
                            alt_collar_pt = closest_pt - collar_depth * into_part
                            alt_inside = self._part_mesh_ref.contains([alt_collar_pt])[0]
                            alt_status = "✅ INSIDE" if alt_inside else "❌ OUTSIDE"
                            print(f"   Alt collar (opposite dir): {alt_status}")
                        except Exception as ce:
                            print(f"   Containment check failed: {ce}")
                        
                        # Direction from vertex to closest point
                        to_part_dir = closest_pt - v
                        to_part_len = np.linalg.norm(to_part_dir)
                        if to_part_len > 1e-8:
                            to_part_dir_unit = to_part_dir / to_part_len
                            print(f"   Dir vertex→closest: [{to_part_dir_unit[0]:.4f}, {to_part_dir_unit[1]:.4f}, {to_part_dir_unit[2]:.4f}]")
                            
                            # Angle between into_part and to_part_dir
                            angle = np.degrees(np.arccos(np.clip(np.dot(into_part, to_part_dir_unit), -1, 1)))
                            print(f"   Angle (into_part vs vertex→closest): {angle:.1f}°")
                
                # ========================================================
                # INNER BOUNDARY EDGE ANALYSIS
                # ========================================================
                if self._triangle_debug_boundary_type_ref is not None:
                    bt = self._triangle_debug_boundary_type_ref
                    print(f"\n🔲 INNER BOUNDARY EDGE ANALYSIS:")
                    
                    edges = [(v0_idx, v1_idx, v0, v1), (v1_idx, v2_idx, v1, v2), (v2_idx, v0_idx, v2, v0)]
                    edge_names = ["V0→V1", "V1→V2", "V2→V0"]
                    
                    for (idx_a, idx_b, va, vb), edge_name in zip(edges, edge_names):
                        bt_a = bt[idx_a] if idx_a < len(bt) else 0
                        bt_b = bt[idx_b] if idx_b < len(bt) else 0
                        
                        # Check if edge touches inner boundary (part)
                        is_inner_edge = (bt_a == -1 or bt_b == -1)
                        edge_type = "INNER BOUNDARY" if is_inner_edge else "INTERIOR"
                        
                        # Edge midpoint and direction
                        edge_mid = (va + vb) / 2
                        edge_dir = vb - va
                        edge_len = np.linalg.norm(edge_dir)
                        
                        print(f"\n   --- Edge {edge_name} ({edge_type}) ---")
                        print(f"   Vertices: [{idx_a}] bt={bt_a} → [{idx_b}] bt={bt_b}")
                        print(f"   Edge length: {edge_len:.6f}")
                        print(f"   Edge midpoint: [{edge_mid[0]:.6f}, {edge_mid[1]:.6f}, {edge_mid[2]:.6f}]")
                        
                        if is_inner_edge:
                            # This edge needs collar - compute collar direction for midpoint
                            mid_closest, mid_dist, mid_face = trimesh.proximity.closest_point(
                                self._part_mesh_ref, [edge_mid]
                            )
                            mid_closest = mid_closest[0]
                            mid_dist = mid_dist[0]
                            mid_face = mid_face[0]
                            
                            print(f"   Closest pt (midpoint): [{mid_closest[0]:.6f}, {mid_closest[1]:.6f}, {mid_closest[2]:.6f}]")
                            print(f"   Distance to part: {mid_dist:.6f}")
                            
                            if mid_face < len(part_face_normals):
                                mid_part_normal = part_face_normals[mid_face]
                                mid_into_part = -mid_part_normal
                                collar_pt = mid_closest + collar_depth * mid_into_part
                                
                                try:
                                    inside = self._part_mesh_ref.contains([collar_pt])[0]
                                    status = "✅ INSIDE" if inside else "❌ OUTSIDE"
                                    print(f"   Collar at midpoint: {status}")
                                except Exception:
                                    pass
                                
            except Exception as e:
                print(f"\n📏 DISTANCE TO PART: Error computing - {e}")
        
        # Analyze potential issues
        print(f"\n🔎 POTENTIAL ISSUES:")
        issues_found = False
        
        if aspect_ratio < 0.1:
            print(f"   ❌ Very thin/degenerate triangle (aspect ratio {aspect_ratio:.4f} < 0.1)")
            issues_found = True
        elif aspect_ratio < 0.2:
            print(f"   ⚠️ Thin triangle (aspect ratio {aspect_ratio:.4f} < 0.2)")
            issues_found = True
        
        edge_ratio = max(edge_a, edge_b, edge_c) / max(min(edge_a, edge_b, edge_c), 1e-10)
        if edge_ratio > 10:
            print(f"   ❌ Extreme edge length ratio ({edge_ratio:.1f}:1)")
            issues_found = True
        elif edge_ratio > 5:
            print(f"   ⚠️ High edge length ratio ({edge_ratio:.1f}:1)")
            issues_found = True
        
        if area < 1e-8:
            print(f"   ❌ Near-zero area ({area:.2e})")
            issues_found = True
        
        if is_boundary and len(neighbors) == 0:
            print(f"   ❌ Isolated triangle (no neighbors)")
            issues_found = True
        
        if not issues_found:
            print(f"   ✅ No obvious issues detected")
        
        print("="*70 + "\n")
        
        # Also log to logger
        logger.info(f"Triangle {cell_id} analyzed: aspect={aspect_ratio:.4f}, area={area:.6f}, "
                   f"edges=[{edge_a:.4f}, {edge_b:.4f}, {edge_c:.4f}], neighbors={len(neighbors)}")
    
    def _find_nearest_triangle(self, picked_info) -> Optional[int]:
        """Find the nearest triangle to a picked point."""
        if self._triangle_debug_mesh_ref is None:
            return None
        
        mesh = self._triangle_debug_mesh_ref
        
        try:
            # Get the picked point coordinates
            if hasattr(picked_info, 'points') and len(picked_info.points) > 0:
                picked_point = np.array(picked_info.points[0])
            elif hasattr(picked_info, 'center'):
                picked_point = np.array(picked_info.center)
            elif isinstance(picked_info, np.ndarray):
                picked_point = picked_info
            else:
                logger.warning(f"Could not extract point from picked_info: {type(picked_info)}")
                return None
            
            # Compute centroids of all triangles
            centroids = mesh.triangles_center
            
            # Find nearest centroid
            distances = np.linalg.norm(centroids - picked_point, axis=1)
            nearest_idx = np.argmin(distances)
            
            return int(nearest_idx)
            
        except Exception as e:
            logger.error(f"Error finding nearest triangle: {e}")
            return None
    
    def _highlight_selected_triangle(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
        """Highlight the selected triangle."""
        # Clear previous selection
        self._clear_selected_triangle()
        
        try:
            # Create a mesh for just this triangle
            vertices = np.array([v0, v1, v2])
            faces = np.array([[3, 0, 1, 2]])  # PyVista format: count followed by indices
            
            tri_mesh = pv.PolyData(vertices, faces)
            
            self._selected_triangle_actor = self.plotter.add_mesh(
                tri_mesh,
                color='magenta',
                opacity=0.8,
                style='surface',
                show_edges=True,
                edge_color='yellow',
                line_width=4,
            )
            
            # Also add spheres at vertices for visibility
            for i, v in enumerate([v0, v1, v2]):
                sphere = pv.Sphere(radius=0.5, center=v)
                self.plotter.add_mesh(sphere, color=['red', 'green', 'blue'][i], opacity=0.8)
            
            self.plotter.update()
            
        except Exception as e:
            logger.error(f"Error highlighting triangle: {e}")
    
    @property
    def triangle_debug_mode(self) -> bool:
        """Check if triangle debug mode is enabled."""
        return self._triangle_debug_mode

    # =========================================================================
    # OUTER SHELL HULL AND OUTER COLLAR VISUALIZATION
    # =========================================================================
    
    def show_outer_shell_hull(self, mesh: 'trimesh.Trimesh'):
        """
        Display the outer shell hull (offset hull for hard plastic shell).
        
        Args:
            mesh: The outer shell hull mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_outer_shell_hull()
        
        # Store reference
        self._outer_shell_hull_mesh = mesh
        
        # Convert to PyVista
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert outer shell hull to PyVista: {e}")
            return
        
        # Add to scene with distinct color (purple, semi-transparent)
        self._outer_shell_hull_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#9370DB',  # Medium purple
            opacity=0.25,
            show_edges=True,
            edge_color='#6A5ACD',  # Slate blue
            line_width=0.5,
        )
        
        self._outer_shell_hull_visible = True
        self.plotter.update()
        logger.info(f"Outer shell hull displayed: {len(vertices)} vertices, {len(faces)} faces")
    
    def remove_outer_shell_hull(self):
        """Remove the outer shell hull mesh from the display."""
        if hasattr(self, '_outer_shell_hull_actor') and self._outer_shell_hull_actor is not None:
            try:
                self.plotter.remove_actor(self._outer_shell_hull_actor)
            except Exception:
                pass
            self._outer_shell_hull_actor = None
        
        if hasattr(self, '_outer_shell_hull_mesh'):
            self._outer_shell_hull_mesh = None
        
        self.plotter.update()
    
    def set_outer_shell_hull_visible(self, visible: bool):
        """
        Set visibility of the outer shell hull.
        
        Args:
            visible: True to show, False to hide
        """
        self._outer_shell_hull_visible = visible
        if hasattr(self, '_outer_shell_hull_actor') and self._outer_shell_hull_actor is not None:
            self._outer_shell_hull_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Outer shell hull visibility set to {visible}")
    
    def show_outer_collar_surface(self, mesh: 'trimesh.Trimesh'):
        """
        Display the extended parting surface (with outer collar).
        
        Args:
            mesh: The parting surface mesh extended with outer collar
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_outer_collar_surface()
        
        # Store reference
        self._outer_collar_mesh = mesh
        
        # Convert to PyVista
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert outer collar surface to PyVista: {e}")
            return
        
        # Add to scene with distinct color (teal/cyan for outer collar)
        self._outer_collar_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#20B2AA',  # Light sea green
            opacity=0.6,
            show_edges=True,
            edge_color='#008B8B',  # Dark cyan
            line_width=0.8,
        )
        
        self._outer_collar_visible = True
        self.plotter.update()
        logger.info(f"Outer collar surface displayed: {len(vertices)} vertices, {len(faces)} faces")
    
    def remove_outer_collar_surface(self):
        """Remove the outer collar surface from the display."""
        if hasattr(self, '_outer_collar_actor') and self._outer_collar_actor is not None:
            try:
                self.plotter.remove_actor(self._outer_collar_actor)
            except Exception:
                pass
            self._outer_collar_actor = None
        
        if hasattr(self, '_outer_collar_mesh'):
            self._outer_collar_mesh = None
        
        self.plotter.update()
    
    def set_outer_collar_surface_visible(self, visible: bool):
        """
        Set visibility of the outer collar surface.
        
        Args:
            visible: True to show, False to hide
        """
        self._outer_collar_visible = visible
        if hasattr(self, '_outer_collar_actor') and self._outer_collar_actor is not None:
            self._outer_collar_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Outer collar surface visibility set to {visible}")

    # =========================================================================
    # HARD SHELL PRISM VISUALIZATION
    # =========================================================================
    
    def show_hard_shell_prism(self, mesh: 'trimesh.Trimesh'):
        """
        Display the hard shell prism (aligned with pouring direction per paper Section 5).
        
        The prism has a flat base orthogonal to the pouring direction, shaped like
        the silhouette of the convex hull.
        
        Args:
            mesh: The hard shell prism mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_hard_shell_prism()
        
        # Store reference
        self._hard_shell_prism_mesh = mesh
        
        # Convert to PyVista
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert hard shell prism to PyVista: {e}")
            return
        
        # Add to scene with distinct color (orange, semi-transparent for prism)
        self._hard_shell_prism_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#FF8C00',  # Dark orange
            opacity=0.2,
            show_edges=True,
            edge_color='#FF4500',  # Orange red
            line_width=1.0,
        )
        
        self._hard_shell_prism_visible = True
        self.plotter.update()
        logger.info(f"Hard shell prism displayed: {len(vertices)} vertices, {len(faces)} faces")
    
    def remove_hard_shell_prism(self):
        """Remove the hard shell prism mesh from the display."""
        if hasattr(self, '_hard_shell_prism_actor') and self._hard_shell_prism_actor is not None:
            try:
                self.plotter.remove_actor(self._hard_shell_prism_actor)
            except Exception:
                pass
            self._hard_shell_prism_actor = None
        
        if hasattr(self, '_hard_shell_prism_mesh'):
            self._hard_shell_prism_mesh = None
        
        self.plotter.update()
    
    def set_hard_shell_prism_visible(self, visible: bool):
        """
        Set visibility of the hard shell prism.
        
        Args:
            visible: True to show, False to hide
        """
        self._hard_shell_prism_visible = visible
        if hasattr(self, '_hard_shell_prism_actor') and self._hard_shell_prism_actor is not None:
            self._hard_shell_prism_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Hard shell prism visibility set to {visible}")

    # =========================================================================
    # SHELL WITH CAVITY VISUALIZATION (CSG Result: prism - hull)
    # =========================================================================
    
    def show_shell_with_cavity(self, shell_mesh: 'trimesh.Trimesh'):
        """
        Display the hard shell with cavity (result of prism - hull CSG).
        
        Displayed in cyan to distinguish from the prism.
        
        Args:
            shell_mesh: The shell mesh with cavity subtracted
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_shell_with_cavity()
        
        if shell_mesh is None or len(shell_mesh.vertices) == 0:
            logger.warning("No valid shell mesh to display")
            return
        
        self._shell_with_cavity_mesh = shell_mesh
        
        try:
            vertices = np.asarray(shell_mesh.vertices)
            faces = np.asarray(shell_mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert shell mesh to PyVista: {e}")
            return
        
        self._shell_with_cavity_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#00CED1',  # Dark Turquoise
            opacity=0.5,  # More translucent to see inner details
            show_edges=True,
            edge_color='#008B8B',  # Dark Cyan
            line_width=0.5,
        )
        self._shell_with_cavity_visible = True
        logger.info(f"Shell with cavity displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()

    def remove_shell_with_cavity(self):
        """Remove the shell with cavity from the display."""
        if hasattr(self, '_shell_with_cavity_actor') and self._shell_with_cavity_actor is not None:
            try:
                self.plotter.remove_actor(self._shell_with_cavity_actor)
            except Exception:
                pass
            self._shell_with_cavity_actor = None
        
        if hasattr(self, '_shell_with_cavity_mesh'):
            self._shell_with_cavity_mesh = None
        
        self.plotter.update()

    def set_shell_with_cavity_visible(self, visible: bool):
        """
        Set visibility of the shell with cavity.
        
        Args:
            visible: True to show, False to hide
        """
        self._shell_with_cavity_visible = visible
        if hasattr(self, '_shell_with_cavity_actor') and self._shell_with_cavity_actor is not None:
            self._shell_with_cavity_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Shell with cavity visibility set to {visible}")

    # =========================================================================
    # OUTER COLLAR EXTENSION VISUALIZATION
    # =========================================================================
    
    def show_outer_collar(self, mesh: 'trimesh.Trimesh'):
        """
        Display the outer collar extension (parting surface extended outward).
        
        This is the parting surface with an extended "collar" that reaches
        beyond the hull boundary to fully cut through the hard shell prism.
        
        Args:
            mesh: The extended parting surface mesh with outer collar
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_outer_collar()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid outer collar mesh to display")
            return
        
        self._outer_collar_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert outer collar mesh to PyVista: {e}")
            return
        
        # Display in magenta to distinguish from parting surface (blue) and shell (cyan)
        self._outer_collar_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#FF00FF',  # Magenta
            opacity=0.6,
            show_edges=True,
            edge_color='#CC00CC',  # Darker magenta
            line_width=1.0,
        )
        self._outer_collar_visible = True
        logger.info(f"Outer collar displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_outer_collar(self):
        """Remove the outer collar extension mesh from the display."""
        if hasattr(self, '_outer_collar_actor') and self._outer_collar_actor is not None:
            try:
                self.plotter.remove_actor(self._outer_collar_actor)
            except Exception:
                pass
            self._outer_collar_actor = None
        
        if hasattr(self, '_outer_collar_mesh'):
            self._outer_collar_mesh = None
        
        self.plotter.update()
    
    def set_outer_collar_visible(self, visible: bool):
        """
        Set visibility of the outer collar extension.
        
        Args:
            visible: True to show, False to hide
        """
        self._outer_collar_visible = visible
        if hasattr(self, '_outer_collar_actor') and self._outer_collar_actor is not None:
            self._outer_collar_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Outer collar visibility set to {visible}")

    # =========================================================================
    # SHELL HALVES VISUALIZATION (SPLIT BY MEMBRANE)
    # =========================================================================
    
    def show_shell_half_1(self, mesh: 'trimesh.Trimesh'):
        """
        Display shell half 1 (upper half split by membrane).
        
        This is the upper portion of the hard shell cut by the parting membrane.
        Displayed in teal color.
        
        Args:
            mesh: The shell half 1 mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_shell_half_1()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid shell half 1 mesh to display")
            return
        
        self._shell_half_1_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert shell half 1 mesh to PyVista: {e}")
            return
        
        # Display in teal to represent "top" mold half
        self._shell_half_1_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#20B2AA',  # Light Sea Green / Teal
            opacity=0.6,
            show_edges=True,
            edge_color='#008080',  # Teal
            line_width=0.5,
        )
        self._shell_half_1_visible = True
        logger.info(f"Shell half 1 displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_shell_half_1(self):
        """Remove shell half 1 from the display."""
        if hasattr(self, '_shell_half_1_actor') and self._shell_half_1_actor is not None:
            try:
                self.plotter.remove_actor(self._shell_half_1_actor)
            except Exception:
                pass
            self._shell_half_1_actor = None
        
        if hasattr(self, '_shell_half_1_mesh'):
            self._shell_half_1_mesh = None
        
        self.plotter.update()
    
    def set_shell_half_1_visible(self, visible: bool):
        """
        Set visibility of shell half 1.
        
        Args:
            visible: True to show, False to hide
        """
        self._shell_half_1_visible = visible
        if hasattr(self, '_shell_half_1_actor') and self._shell_half_1_actor is not None:
            self._shell_half_1_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Shell half 1 visibility set to {visible}")
    
    def show_shell_half_2(self, mesh: 'trimesh.Trimesh'):
        """
        Display shell half 2 (lower half split by membrane).
        
        This is the lower portion of the hard shell cut by the parting membrane.
        Displayed in coral color to distinguish from half 1.
        
        Args:
            mesh: The shell half 2 mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_shell_half_2()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid shell half 2 mesh to display")
            return
        
        self._shell_half_2_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert shell half 2 mesh to PyVista: {e}")
            return
        
        # Display in coral to represent "bottom" mold half (contrasts with teal)
        self._shell_half_2_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#F08080',  # Light Coral
            opacity=0.6,
            show_edges=True,
            edge_color='#CD5C5C',  # Indian Red
            line_width=0.5,
        )
        self._shell_half_2_visible = True
        logger.info(f"Shell half 2 displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_shell_half_2(self):
        """Remove shell half 2 from the display."""
        if hasattr(self, '_shell_half_2_actor') and self._shell_half_2_actor is not None:
            try:
                self.plotter.remove_actor(self._shell_half_2_actor)
            except Exception:
                pass
            self._shell_half_2_actor = None
        
        if hasattr(self, '_shell_half_2_mesh'):
            self._shell_half_2_mesh = None
        
        self.plotter.update()
    
    def set_shell_half_2_visible(self, visible: bool):
        """
        Set visibility of shell half 2.
        
        Args:
            visible: True to show, False to hide
        """
        self._shell_half_2_visible = visible
        if hasattr(self, '_shell_half_2_actor') and self._shell_half_2_actor is not None:
            self._shell_half_2_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Shell half 2 visibility set to {visible}")

    # =========================================================================
    # METAMOLD PRISM VISUALIZATION
    # =========================================================================
    
    def show_metamold_prism(self, mesh: 'trimesh.Trimesh'):
        """
        Display the metamold prism for casting silicone mold halves.
        
        The prism is aligned with the silicone pouring direction and uses
        the parting surface silhouette as its base shape.
        
        Args:
            mesh: The metamold prism mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_metamold_prism()
        
        # Store reference
        self._metamold_prism_mesh = mesh
        
        # Convert to PyVista
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert metamold prism to PyVista: {e}")
            return
        
        # Add to scene with distinct color (green, semi-transparent)
        self._metamold_prism_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#4CAF50',  # Material Green
            opacity=0.25,
            show_edges=True,
            edge_color='#2E7D32',  # Dark Green
            line_width=1.0,
        )
        
        self._metamold_prism_visible = True
        self.plotter.update()
        logger.info(f"Metamold prism displayed: {len(vertices)} vertices, {len(faces)} faces")
    
    def remove_metamold_prism(self):
        """Remove the metamold prism mesh from the display."""
        if hasattr(self, '_metamold_prism_actor') and self._metamold_prism_actor is not None:
            try:
                self.plotter.remove_actor(self._metamold_prism_actor)
            except Exception:
                pass
            self._metamold_prism_actor = None
        
        if hasattr(self, '_metamold_prism_mesh'):
            self._metamold_prism_mesh = None
        
        self.plotter.update()
    
    def set_metamold_prism_visible(self, visible: bool):
        """
        Set visibility of the metamold prism.
        
        Args:
            visible: True to show, False to hide
        """
        self._metamold_prism_visible = visible
        if hasattr(self, '_metamold_prism_actor') and self._metamold_prism_actor is not None:
            self._metamold_prism_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Metamold prism visibility set to {visible}")

    # =========================================================================
    # METAMOLD WITH CAVITY VISUALIZATION
    # =========================================================================
    
    def show_metamold_with_cavity(self, mesh: 'trimesh.Trimesh'):
        """
        Display the metamold with cavity (prism - part mesh).
        
        Args:
            mesh: The metamold with cavity mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_metamold_with_cavity()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid metamold with cavity mesh to display")
            return
        
        self._metamold_cavity_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert metamold with cavity to PyVista: {e}")
            return
        
        # Display in light green (different from prism)
        self._metamold_cavity_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#81C784',  # Light Green
            opacity=0.4,
            show_edges=True,
            edge_color='#388E3C',  # Green
            line_width=0.5,
        )
        self._metamold_cavity_visible = True
        logger.info(f"Metamold with cavity displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_metamold_with_cavity(self):
        """Remove metamold with cavity from the display."""
        if hasattr(self, '_metamold_cavity_actor') and self._metamold_cavity_actor is not None:
            try:
                self.plotter.remove_actor(self._metamold_cavity_actor)
            except Exception:
                pass
            self._metamold_cavity_actor = None
        
        if hasattr(self, '_metamold_cavity_mesh'):
            self._metamold_cavity_mesh = None
        
        self.plotter.update()
    
    def set_metamold_with_cavity_visible(self, visible: bool):
        """
        Set visibility of metamold with cavity.
        
        Args:
            visible: True to show, False to hide
        """
        self._metamold_cavity_visible = visible
        if hasattr(self, '_metamold_cavity_actor') and self._metamold_cavity_actor is not None:
            self._metamold_cavity_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Metamold with cavity visibility set to {visible}")

    # =========================================================================
    # METAMOLD HALVES VISUALIZATION
    # =========================================================================
    
    def show_metamold_half_1(self, mesh: 'trimesh.Trimesh'):
        """
        Display metamold half 1 (upper half split by membrane).
        
        This is the upper portion of the metamold cut by the parting membrane.
        Displayed in green color.
        
        Args:
            mesh: The metamold half 1 mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_metamold_half_1()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid metamold half 1 mesh to display")
            return
        
        self._metamold_half_1_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert metamold half 1 mesh to PyVista: {e}")
            return
        
        # Display in distinct green
        self._metamold_half_1_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#66BB6A',  # Green 400
            opacity=0.6,
            show_edges=True,
            edge_color='#43A047',  # Green 600
            line_width=0.5,
        )
        self._metamold_half_1_visible = True
        logger.info(f"Metamold half 1 displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_metamold_half_1(self):
        """Remove metamold half 1 from the display."""
        if hasattr(self, '_metamold_half_1_actor') and self._metamold_half_1_actor is not None:
            try:
                self.plotter.remove_actor(self._metamold_half_1_actor)
            except Exception:
                pass
            self._metamold_half_1_actor = None
        
        if hasattr(self, '_metamold_half_1_mesh'):
            self._metamold_half_1_mesh = None
        
        self.plotter.update()
    
    def set_metamold_half_1_visible(self, visible: bool):
        """
        Set visibility of metamold half 1.
        
        Args:
            visible: True to show, False to hide
        """
        self._metamold_half_1_visible = visible
        if hasattr(self, '_metamold_half_1_actor') and self._metamold_half_1_actor is not None:
            self._metamold_half_1_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Metamold half 1 visibility set to {visible}")
    
    def show_metamold_half_2(self, mesh: 'trimesh.Trimesh'):
        """
        Display metamold half 2 (lower half split by membrane).
        
        This is the lower portion of the metamold cut by the parting membrane.
        Displayed in purple color to distinguish from half 1.
        
        Args:
            mesh: The metamold half 2 mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_metamold_half_2()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid metamold half 2 mesh to display")
            return
        
        self._metamold_half_2_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert metamold half 2 mesh to PyVista: {e}")
            return
        
        # Display in purple (contrasts with green)
        self._metamold_half_2_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#AB47BC',  # Purple 400
            opacity=0.6,
            show_edges=True,
            edge_color='#8E24AA',  # Purple 600
            line_width=0.5,
        )
        self._metamold_half_2_visible = True
        logger.info(f"Metamold half 2 displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_metamold_half_2(self):
        """Remove metamold half 2 from the display."""
        if hasattr(self, '_metamold_half_2_actor') and self._metamold_half_2_actor is not None:
            try:
                self.plotter.remove_actor(self._metamold_half_2_actor)
            except Exception:
                pass
            self._metamold_half_2_actor = None
        
        if hasattr(self, '_metamold_half_2_mesh'):
            self._metamold_half_2_mesh = None
        
        self.plotter.update()
    
    def set_metamold_half_2_visible(self, visible: bool):
        """
        Set visibility of metamold half 2.
        
        Args:
            visible: True to show, False to hide
        """
        self._metamold_half_2_visible = visible
        if hasattr(self, '_metamold_half_2_actor') and self._metamold_half_2_actor is not None:
            self._metamold_half_2_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Metamold half 2 visibility set to {visible}")

    # =========================================================================
    # METAMOLD HALVES WITH PART ADDED
    # =========================================================================

    def show_metamold_half_1_with_part(self, mesh: 'trimesh.Trimesh'):
        """
        Display metamold half 1 with part mesh added (boolean union).
        
        This is the upper portion of the metamold with the part mesh 
        added back as solid geometry.
        Displayed in cyan color to distinguish from other halves.
        
        Args:
            mesh: The metamold half 1 with part added mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_metamold_half_1_with_part()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid metamold half 1 with part mesh to display")
            return
        
        self._metamold_half_1_with_part_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert metamold half 1 with part mesh to PyVista: {e}")
            return
        
        # Display in cyan (bright, easy to see)
        self._metamold_half_1_with_part_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#00BCD4',  # Cyan 500
            opacity=0.7,
            show_edges=True,
            edge_color='#0097A7',  # Cyan 700
            line_width=0.5,
        )
        self._metamold_half_1_with_part_visible = True
        logger.info(f"Metamold half 1 with part displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_metamold_half_1_with_part(self):
        """Remove metamold half 1 with part from the display."""
        if hasattr(self, '_metamold_half_1_with_part_actor') and self._metamold_half_1_with_part_actor is not None:
            try:
                self.plotter.remove_actor(self._metamold_half_1_with_part_actor)
            except Exception:
                pass
            self._metamold_half_1_with_part_actor = None
        
        if hasattr(self, '_metamold_half_1_with_part_mesh'):
            self._metamold_half_1_with_part_mesh = None
        
        self.plotter.update()
    
    def set_metamold_half_1_with_part_visible(self, visible: bool):
        """
        Set visibility of metamold half 1 with part.
        
        Args:
            visible: True to show, False to hide
        """
        self._metamold_half_1_with_part_visible = visible
        if hasattr(self, '_metamold_half_1_with_part_actor') and self._metamold_half_1_with_part_actor is not None:
            self._metamold_half_1_with_part_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Metamold half 1 with part visibility set to {visible}")

    def show_metamold_half_2_with_part(self, mesh: 'trimesh.Trimesh'):
        """
        Display metamold half 2 with part mesh added (boolean union).
        
        This is the lower portion of the metamold with the part mesh 
        added back as solid geometry.
        Displayed in orange color to distinguish from other halves.
        
        Args:
            mesh: The metamold half 2 with part added mesh
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Remove existing actor
        self.remove_metamold_half_2_with_part()
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("No valid metamold half 2 with part mesh to display")
            return
        
        self._metamold_half_2_with_part_mesh = mesh
        
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        except Exception as e:
            logger.error(f"Failed to convert metamold half 2 with part mesh to PyVista: {e}")
            return
        
        # Display in orange (warm color, contrasts with cyan)
        self._metamold_half_2_with_part_actor = self.plotter.add_mesh(
            pv_mesh,
            color='#FF9800',  # Orange 500
            opacity=0.7,
            show_edges=True,
            edge_color='#F57C00',  # Orange 700
            line_width=0.5,
        )
        self._metamold_half_2_with_part_visible = True
        logger.info(f"Metamold half 2 with part displayed: {len(vertices)} vertices, {len(faces)} faces")
        
        self.plotter.update()
    
    def remove_metamold_half_2_with_part(self):
        """Remove metamold half 2 with part from the display."""
        if hasattr(self, '_metamold_half_2_with_part_actor') and self._metamold_half_2_with_part_actor is not None:
            try:
                self.plotter.remove_actor(self._metamold_half_2_with_part_actor)
            except Exception:
                pass
            self._metamold_half_2_with_part_actor = None
        
        if hasattr(self, '_metamold_half_2_with_part_mesh'):
            self._metamold_half_2_with_part_mesh = None
        
        self.plotter.update()
    
    def set_metamold_half_2_with_part_visible(self, visible: bool):
        """
        Set visibility of metamold half 2 with part.
        
        Args:
            visible: True to show, False to hide
        """
        self._metamold_half_2_with_part_visible = visible
        if hasattr(self, '_metamold_half_2_with_part_actor') and self._metamold_half_2_with_part_actor is not None:
            self._metamold_half_2_with_part_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Metamold half 2 with part visibility set to {visible}")
