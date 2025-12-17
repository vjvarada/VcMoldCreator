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
    - Cell/toon shading with feature edges (not triangle edges)
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
    
    # Inflated hull colors (matching React frontend)
    HULL_COLOR = "#9966ff"        # Purple - Inflated hull
    ORIGINAL_HULL_COLOR = "#666666"  # Gray - Original hull wireframe
    HULL_OPACITY = 0.3            # Transparency for hull
    
    # Mold cavity colors (matching React frontend)
    CAVITY_COLOR = "#00ffaa"      # Teal/Cyan - Mold cavity
    CAVITY_OPACITY = 0.5          # Semi-transparent
    
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
        self._show_edges = True  # Feature edges on by default for cel shading
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
        
        # Inflated hull visualization
        self._hull_mesh: Optional[trimesh.Trimesh] = None
        self._hull_actor = None
        self._original_hull_actor = None
        self._hull_visible = True
        self._original_hull_visible = False  # Wireframe of original hull
        
        # Mold cavity visualization
        self._cavity_mesh: Optional[trimesh.Trimesh] = None
        self._cavity_actor = None
        self._cavity_visible = True
        
        # Mold halves classification visualization (painted on cavity)
        self._mold_halves_actor = None  # Inner boundary (seed) edges
        self._mold_halves_visible = True
        self._mold_halves_edges_actor = None  # Outer boundary edges (H1/H2/boundary)
        self._outer_boundary_visible = True  # Visibility for outer boundary
        self._inner_boundary_visible = True  # Visibility for inner boundary (seeds)
        
        # Inner boundary (seed) vertices/edges visualization  
        self._inner_boundary_vertices_actor = None
        self._inner_boundary_edges_actor = None
        self._inner_boundary_visible = True
        
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
        self._edge_weights_visible = True  # Show/hide edge weight coloring
        
        # Original (non-inflated) tetrahedral mesh visualization
        self._tet_original_actor = None
        self._tet_original_visible = False
        
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
        Set up lighting for cell/toon shading style.
        
        Uses simplified lighting for a flatter, more stylized look:
        - Single strong directional light for clear shadows
        - High ambient to reduce harsh shadows
        """
        if not PYVISTA_AVAILABLE:
            return
            
        # Remove default lights
        self.plotter.remove_all_lights()
        
        # Single main directional light for cell shading
        # Positioned to create clear, readable shadows
        light1 = pv.Light(
            position=(1, 1, 1.5),  # Slightly from above
            focal_point=(0, 0, 0),
            color='white',
            intensity=1.0,  # Strong main light
        )
        light1.positional = False  # Directional light
        self.plotter.add_light(light1)
        
        # Subtle fill light to soften shadows (optional for cell shading)
        light2 = pv.Light(
            position=(-0.5, -0.5, 0.5),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.3,  # Subtle fill
        )
        light2.positional = False
        self.plotter.add_light(light2)
    
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
        """Render the current mesh with cell/toon shading style."""
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Clear previous content
        self.plotter.clear()
        self._feature_edges_actor = None
        self._silhouette_actor = None
        self._silhouette_filter = None
        # Hull and cavity actors are invalidated by clear(), reset references
        self._hull_actor = None
        self._cavity_actor = None
        self._original_hull_actor = None
        
        # Re-setup lighting after clear
        self._setup_lighting()
        
        # Add the mesh with cell/toon shading style:
        # - Flat shading for sharp facets
        # - High ambient, moderate diffuse, no specular
        # - NO triangle edges (we add feature edges separately)
        self._actor = self.plotter.add_mesh(
            self._pv_mesh,
            color=self._current_color,
            smooth_shading=False,      # Flat shading for cell/toon look
            show_edges=False,          # Don't show triangle edges
            opacity=1.0,
            # Cell shading material properties
            ambient=0.5,               # High ambient for flatter look
            diffuse=0.5,               # Moderate diffuse
            specular=0.0,              # No specular for toon style
            specular_power=1,
            render_points_as_spheres=False,
            render_lines_as_tubes=False,
        )
        
        # Set flat interpolation for cell shading effect
        if self._actor is not None:
            prop = self._actor.GetProperty()
            if prop:
                prop.SetInterpolationToFlat()
        
        # Add feature edges (silhouette + crease edges) for cel shading outline
        self._add_feature_edges()
        
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
        
        # Re-add hull and cavity actors if they exist (they get cleared by plotter.clear())
        self._re_add_hull_and_cavity_actors()
        
        # Update the view
        self.plotter.update()
    
    def _re_add_hull_and_cavity_actors(self):
        """
        Re-add hull and cavity actors after plotter.clear().
        
        When _render_mesh() is called (e.g., loading new mesh, color change),
        plotter.clear() removes all actors including hull and cavity. This method
        re-adds them from the stored mesh data if available.
        
        NOTE: The new _update_mesh_display() method avoids clearing the scene,
        so hull/cavity actors are preserved when toggling visibility paint.
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
                smooth_shading=True,
                show_edges=False,
                style='surface',
                render_points_as_spheres=False,
            )
            if self._hull_actor is not None:
                prop = self._hull_actor.GetProperty()
                if prop:
                    prop.SetInterpolationToPhong()
                # Restore visibility state
                self._hull_actor.SetVisibility(self._hull_visible)
        
        # Re-add cavity mesh if it exists
        if self._cavity_mesh is not None:
            pv_cavity = self._trimesh_to_pyvista(self._cavity_mesh)
            self._cavity_actor = self.plotter.add_mesh(
                pv_cavity,
                color=self.CAVITY_COLOR,
                opacity=self.CAVITY_OPACITY,
                smooth_shading=True,
                show_edges=False,
                style='surface',
                render_points_as_spheres=False,
            )
            if self._cavity_actor is not None:
                prop = self._cavity_actor.GetProperty()
                if prop:
                    prop.SetInterpolationToPhong()
                # Restore visibility state
                self._cavity_actor.SetVisibility(self._cavity_visible)
    
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
        
        # Reset hull state
        self._hull_mesh = None
        self._hull_actor = None
        self._original_hull_actor = None
        self._hull_visible = True
        self._original_hull_visible = False
        
        # Reset cavity state
        self._cavity_mesh = None
        self._cavity_actor = None
        self._cavity_visible = True
        
        # Reset mold halves state
        self._mold_halves_actor = None
        self._mold_halves_visible = True
        self._mold_halves_edges_actor = None
        
        # Reset inner boundary visualization state
        self._inner_boundary_vertices_actor = None
        self._inner_boundary_edges_actor = None
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
        self._edge_weights_visible = True
        
        # Reset original tet mesh state
        self._tet_original_actor = None
        self._tet_original_visible = False
        
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
        parting directions for a two-piece mold. Uses cell/toon shading
        for a cleaner, more stylized look.
        
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
        
        # Apply cell/toon shading style - flat shading with edge emphasis
        actor1 = self.plotter.add_mesh(
            arrow1,
            color=self.PARTING_D1_COLOR,
            opacity=1.0,
            smooth_shading=False,  # Flat shading for cell/toon look
            show_edges=False,
            ambient=0.4,      # Higher ambient for flatter look
            diffuse=0.6,      # Lower diffuse
            specular=0.0,     # No specular for toon style
        )
        # Set flat interpolation for cell shading effect
        if actor1 is not None:
            prop = actor1.GetProperty()
            if prop:
                prop.SetInterpolationToFlat()
                prop.SetEdgeVisibility(True)
                prop.SetEdgeColor(0.0, 0.6, 0.0)  # Darker green edge
                prop.SetLineWidth(1.5)
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
        
        # Apply cell/toon shading style
        actor2 = self.plotter.add_mesh(
            arrow2,
            color=self.PARTING_D2_COLOR,
            opacity=1.0,
            smooth_shading=False,  # Flat shading for cell/toon look
            show_edges=False,
            ambient=0.4,
            diffuse=0.6,
            specular=0.0,
        )
        if actor2 is not None:
            prop = actor2.GetProperty()
            if prop:
                prop.SetInterpolationToFlat()
                prop.SetEdgeVisibility(True)
                prop.SetEdgeColor(0.8, 0.3, 0.0)  # Darker orange edge
                prop.SetLineWidth(1.5)
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
        hull, cavity, and arrow actors.
        
        Args:
            use_colors: If True, display with per-cell colors; if False, solid color
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Remove only the main mesh actor (not hull/cavity/arrows)
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
        
        # Add mesh with appropriate coloring
        if use_colors and 'colors' in self._pv_mesh.cell_data:
            self._actor = self.plotter.add_mesh(
                self._pv_mesh,
                scalars='colors',
                rgb=True,
                smooth_shading=False,
                show_edges=False,
                opacity=1.0,
                ambient=0.5,
                diffuse=0.5,
                specular=0.0,
                specular_power=1,
            )
        else:
            self._actor = self.plotter.add_mesh(
                self._pv_mesh,
                color=self._current_color,
                smooth_shading=False,
                show_edges=False,
                opacity=1.0,
                ambient=0.5,
                diffuse=0.5,
                specular=0.0,
                specular_power=1,
                render_points_as_spheres=False,
                render_lines_as_tubes=False,
            )
        
        # Set flat interpolation for cell shading effect
        if self._actor is not None:
            prop = self._actor.GetProperty()
            if prop:
                prop.SetInterpolationToFlat()
        
        # Re-add feature edges
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
        as it's more efficient and doesn't affect hull/cavity actors.
        
        This method is kept for edge cases where a full re-render is needed.
        """
        if not PYVISTA_AVAILABLE or self._pv_mesh is None:
            return
        
        # Clear previous content
        self.plotter.clear()
        self._feature_edges_actor = None
        self._silhouette_actor = None
        self._silhouette_filter = None
        # Hull and cavity actors are invalidated by clear(), reset references
        self._hull_actor = None
        self._cavity_actor = None
        self._original_hull_actor = None
        
        # Re-setup lighting after clear
        self._setup_lighting()
        
        # Add mesh with cell colors and cell/toon shading
        # NO triangle edges - we add feature edges separately
        self._actor = self.plotter.add_mesh(
            self._pv_mesh,
            scalars='colors',
            rgb=True,
            smooth_shading=False,      # Flat shading for cell/toon look
            show_edges=False,          # Don't show triangle edges
            opacity=1.0,
            ambient=0.5,               # High ambient for flatter look
            diffuse=0.5,               # Moderate diffuse
            specular=0.0,              # No specular for toon style
            specular_power=1,
        )
        
        # Set flat interpolation for cell shading
        if self._actor is not None:
            prop = self._actor.GetProperty()
            if prop:
                prop.SetInterpolationToFlat()
        
        # Add feature edges for cel shading outline
        self._add_feature_edges()
        
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
        
        # Re-add hull and cavity actors if they exist (they get cleared by plotter.clear())
        self._re_add_hull_and_cavity_actors()
        
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
        
        # Add inflated hull mesh (semi-transparent)
        self._hull_actor = self.plotter.add_mesh(
            pv_hull,
            color=self.HULL_COLOR,
            opacity=self.HULL_OPACITY,
            smooth_shading=True,
            show_edges=False,
            style='surface',
            render_points_as_spheres=False,
        )
        
        # Make hull render behind the main mesh
        if self._hull_actor is not None:
            prop = self._hull_actor.GetProperty()
            if prop:
                prop.SetInterpolationToPhong()
        
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
    # MOLD CAVITY VISUALIZATION
    # =========================================================================
    
    def set_cavity_mesh(self, cavity_mesh: trimesh.Trimesh):
        """
        Set and display the mold cavity mesh.
        
        Args:
            cavity_mesh: The mold cavity mesh (result of hull - original)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._cavity_mesh = cavity_mesh
        
        # Convert cavity to PyVista
        pv_cavity = self._trimesh_to_pyvista(cavity_mesh)
        
        # Remove existing cavity actor (but don't clear mesh data)
        if self._cavity_actor is not None:
            try:
                self.plotter.remove_actor(self._cavity_actor)
            except Exception:
                pass
            self._cavity_actor = None
        
        # Add cavity mesh (semi-transparent)
        self._cavity_actor = self.plotter.add_mesh(
            pv_cavity,
            color=self.CAVITY_COLOR,
            opacity=self.CAVITY_OPACITY,
            smooth_shading=True,
            show_edges=False,
            style='surface',
            render_points_as_spheres=False,
        )
        
        # Set up rendering properties
        if self._cavity_actor is not None:
            prop = self._cavity_actor.GetProperty()
            if prop:
                prop.SetInterpolationToPhong()
        
        self.plotter.update()
        logger.info(f"Cavity mesh displayed: {len(cavity_mesh.vertices)} vertices, {len(cavity_mesh.faces)} faces")
    
    def clear_cavity(self):
        """Remove cavity visualization from the scene."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._cavity_actor is not None:
            try:
                self.plotter.remove_actor(self._cavity_actor)
            except Exception:
                pass
            self._cavity_actor = None
        
        self._cavity_mesh = None
        self.plotter.update()
        logger.info("Cavity mesh cleared")
    
    def set_cavity_visible(self, visible: bool):
        """
        Set visibility of the mold cavity.
        
        Args:
            visible: True to show cavity, False to hide
        """
        self._cavity_visible = visible
        if self._cavity_actor is not None:
            self._cavity_actor.SetVisibility(visible)
            self.plotter.update()
            logger.debug(f"Cavity visibility set to {visible}")
        else:
            logger.debug(f"Cavity visibility requested ({visible}) but no actor exists (mesh exists: {self._cavity_mesh is not None})")
    
    def set_cavity_opacity(self, opacity: float):
        """
        Set the opacity of the mold cavity.
        
        Args:
            opacity: Opacity value from 0.0 (transparent) to 1.0 (opaque)
        """
        if self._cavity_actor is not None:
            prop = self._cavity_actor.GetProperty()
            if prop:
                prop.SetOpacity(opacity)
            self.plotter.update()
    
    def apply_tet_mesh_classification(
        self,
        tet_vertices: np.ndarray,
        tet_edges: np.ndarray,
        boundary_mesh: 'trimesh.Trimesh',
        classification_result,
        part_mesh: 'trimesh.Trimesh',
        seed_distance_threshold: float = None
    ):
        """
        Apply mold half classification to tetrahedral mesh with seed vertices.
        
        This method:
        1. Colors boundary mesh edges by H1/H2/boundary classification
        2. Identifies and highlights seed vertices (close to part surface) in blue
        
        Colors:
        - H₁ (Green): Edges on H1 triangles
        - H₂ (Orange): Edges on H2 triangles  
        - Boundary Zone (Gray): Interface between H₁ and H₂
        - Seed (Blue): Tetrahedral vertices close to part surface
        
        Args:
            tet_vertices: Full tetrahedral mesh vertices (N x 3)
            tet_edges: Tetrahedral mesh edges (E x 2)
            boundary_mesh: The tetrahedral boundary mesh
            classification_result: MoldHalfClassificationResult
            part_mesh: Original part mesh (for identifying seed vertices)
            seed_distance_threshold: Distance threshold for seed vertices (auto if None)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        logger.info("Applying tet mesh classification with seed vertices")
        
        # Remove existing actors
        if self._cavity_actor is not None:
            try:
                self.plotter.remove_actor(self._cavity_actor)
            except Exception:
                pass
            self._cavity_actor = None
        
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
        
        # Remove part surface reference if exists
        if self._part_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._part_surface_actor)
            except Exception:
                pass
            self._part_surface_actor = None
        
        # =====================================================================
        # STEP 1: Identify seed vertices (tetrahedral vertices close to part)
        # =====================================================================
        
        # Compute distance from all tetrahedral vertices to part surface
        _, tet_to_part_distances, _ = part_mesh.nearest.on_surface(tet_vertices)
        
        # Auto-compute threshold if not provided
        if seed_distance_threshold is None:
            # Use a small fraction of the mesh size
            bounds = part_mesh.bounds
            mesh_size = np.linalg.norm(bounds[1] - bounds[0])
            seed_distance_threshold = mesh_size * 0.02  # 2% of mesh size
        
        # Find seed vertices (those very close to part surface)
        seed_mask = tet_to_part_distances < seed_distance_threshold
        n_seeds = np.sum(seed_mask)
        seed_indices = np.where(seed_mask)[0]
        
        logger.info(f"Found {n_seeds} seed vertices (distance < {seed_distance_threshold:.4f})")
        logger.info(f"  Distance to part: min={tet_to_part_distances.min():.4f}, max={tet_to_part_distances.max():.4f}")
        
        # =====================================================================
        # STEP 2: Classify boundary mesh for H1/H2/boundary visualization
        # =====================================================================
        
        boundary_vertices = np.asarray(boundary_mesh.vertices)
        boundary_faces = np.asarray(boundary_mesh.faces)
        n_boundary_verts = len(boundary_vertices)
        
        # Get classification sets
        h1_set = classification_result.h1_triangles
        h2_set = classification_result.h2_triangles
        boundary_set = classification_result.boundary_zone_triangles
        
        # Assign vertex labels based on adjacent triangles
        vertex_h1_count = np.zeros(n_boundary_verts, dtype=np.int32)
        vertex_h2_count = np.zeros(n_boundary_verts, dtype=np.int32)
        vertex_boundary_count = np.zeros(n_boundary_verts, dtype=np.int32)
        
        for face_idx, face in enumerate(boundary_faces):
            v0, v1, v2 = face
            if face_idx in h1_set:
                vertex_h1_count[v0] += 1
                vertex_h1_count[v1] += 1
                vertex_h1_count[v2] += 1
            elif face_idx in h2_set:
                vertex_h2_count[v0] += 1
                vertex_h2_count[v1] += 1
                vertex_h2_count[v2] += 1
            elif face_idx in boundary_set:
                vertex_boundary_count[v0] += 1
                vertex_boundary_count[v1] += 1
                vertex_boundary_count[v2] += 1
        
        # Assign boundary vertex labels
        boundary_vertex_labels = np.zeros(n_boundary_verts, dtype=np.int8)
        for i in range(n_boundary_verts):
            if vertex_h1_count[i] > vertex_h2_count[i] and vertex_h1_count[i] > vertex_boundary_count[i]:
                boundary_vertex_labels[i] = 1  # H1
            elif vertex_h2_count[i] > vertex_h1_count[i] and vertex_h2_count[i] > vertex_boundary_count[i]:
                boundary_vertex_labels[i] = 2  # H2
            elif vertex_boundary_count[i] > 0:
                boundary_vertex_labels[i] = 3  # Boundary
            elif vertex_h1_count[i] > 0:
                boundary_vertex_labels[i] = 1
            elif vertex_h2_count[i] > 0:
                boundary_vertex_labels[i] = 2
        
        # Build boundary edges
        boundary_edge_set = set()
        for face in boundary_faces:
            v0, v1, v2 = face
            boundary_edge_set.add((min(v0, v1), max(v0, v1)))
            boundary_edge_set.add((min(v1, v2), max(v1, v2)))
            boundary_edge_set.add((min(v2, v0), max(v2, v0)))
        
        boundary_edges = np.array(list(boundary_edge_set))
        n_boundary_edges = len(boundary_edges)
        
        # =====================================================================
        # STEP 2b: Identify inner boundary (seed) edges FIRST
        # =====================================================================
        # We need to identify seed edges before assigning colors so we can
        # exclude them from the outer boundary visualization
        
        _, boundary_to_part_distances, _ = part_mesh.nearest.on_surface(boundary_vertices)
        boundary_seed_mask = boundary_to_part_distances < seed_distance_threshold
        n_boundary_seeds = np.sum(boundary_seed_mask)
        logger.info(f"Found {n_boundary_seeds} seed vertices on boundary mesh surface")
        
        # Find which edges are seed edges (both endpoints are seeds)
        is_seed_edge = np.zeros(n_boundary_edges, dtype=bool)
        for i, (v0, v1) in enumerate(boundary_edges):
            if boundary_seed_mask[v0] and boundary_seed_mask[v1]:
                is_seed_edge[i] = True
        
        n_seed_edges = np.sum(is_seed_edge)
        logger.info(f"Found {n_seed_edges} seed edges on boundary surface")
        
        # =====================================================================
        # STEP 2c: Assign colors to NON-SEED edges only (outer boundary)
        # =====================================================================
        
        # Get non-seed edges for outer boundary visualization
        outer_edge_mask = ~is_seed_edge
        outer_edges = boundary_edges[outer_edge_mask]
        n_outer_edges = len(outer_edges)
        
        # Assign edge colors based on endpoint labels (only for outer edges)
        outer_edge_colors = np.zeros((n_outer_edges, 3), dtype=np.uint8)
        for i, (v0, v1) in enumerate(outer_edges):
            l0, l1 = boundary_vertex_labels[v0], boundary_vertex_labels[v1]
            
            if l0 == 1 and l1 == 1:
                outer_edge_colors[i] = [76, 175, 80]  # Green for H1
            elif l0 == 2 and l1 == 2:
                outer_edge_colors[i] = [255, 152, 0]  # Orange for H2
            elif l0 == 3 and l1 == 3:
                outer_edge_colors[i] = [158, 158, 158]  # Gray for boundary
            elif l0 in [1, 2] and l1 in [1, 2]:
                outer_edge_colors[i] = [158, 158, 158]  # Gray for H1-H2 interface
            elif l0 == 1 or l1 == 1:
                outer_edge_colors[i] = [76, 175, 80]  # Green
            elif l0 == 2 or l1 == 2:
                outer_edge_colors[i] = [255, 152, 0]  # Orange
            else:
                outer_edge_colors[i] = [158, 158, 158]  # Gray for unclassified
        
        n_h1_verts = np.sum(boundary_vertex_labels == 1)
        n_h2_verts = np.sum(boundary_vertex_labels == 2)
        n_boundary_zone_verts = np.sum(boundary_vertex_labels == 3)
        logger.info(f"Boundary vertex labels: H1={n_h1_verts}, H2={n_h2_verts}, boundary={n_boundary_zone_verts}")
        
        # =====================================================================
        # STEP 3: Create visualizations
        # =====================================================================
        
        # Create OUTER boundary edge mesh (excludes inner boundary/seed edges)
        if n_outer_edges > 0:
            cells = np.empty((n_outer_edges, 3), dtype=np.int64)
            cells[:, 0] = 2
            cells[:, 1] = outer_edges[:, 0]
            cells[:, 2] = outer_edges[:, 1]
            cell_types = np.full(n_outer_edges, 3, dtype=np.uint8)  # VTK_LINE
            
            edge_grid = pv.UnstructuredGrid(cells.ravel(), cell_types, boundary_vertices)
            edge_grid.cell_data['colors'] = outer_edge_colors
            
            # Add boundary edges
            self._mold_halves_edges_actor = self.plotter.add_mesh(
                edge_grid,
                scalars='colors',
                rgb=True,
                line_width=3,
                show_edges=False,
            )
        else:
            logger.info("No outer boundary edges to display")
        
        # Create INNER boundary (seed) edges visualization (blue)
        # We already identified seed edges above
        if n_seed_edges > 0:
            seed_edges = boundary_edges[is_seed_edge]
            
            # Create UnstructuredGrid with VTK_LINE cells for seed edges
            seed_cells = np.empty((n_seed_edges, 3), dtype=np.int64)
            seed_cells[:, 0] = 2  # Each line has 2 points
            seed_cells[:, 1] = seed_edges[:, 0]
            seed_cells[:, 2] = seed_edges[:, 1]
            seed_cell_types = np.full(n_seed_edges, 3, dtype=np.uint8)  # VTK_LINE = 3
            
            # Use boundary_vertices since edges reference boundary mesh
            seed_edge_grid = pv.UnstructuredGrid(seed_cells.ravel(), seed_cell_types, boundary_vertices)
            
            self._mold_halves_actor = self.plotter.add_mesh(
                seed_edge_grid,
                color='blue',
                line_width=4,
                show_edges=False,
            )
            logger.info(f"Visualizing {n_seed_edges} seed edges as blue lines on boundary surface")
        else:
            logger.warning("No seed edges found on boundary surface!")
        
        # Set visibility based on individual flags
        if self._mold_halves_edges_actor is not None:
            self._mold_halves_edges_actor.SetVisibility(self._outer_boundary_visible)
        if self._mold_halves_actor is not None:
            self._mold_halves_actor.SetVisibility(self._inner_boundary_visible)
        
        self.plotter.update()
        logger.info(f"Applied tet classification: {n_outer_edges} outer edges, {n_seed_edges} seed edges")

    def apply_cavity_classification_paint(self, face_colors: np.ndarray):
        """
        Apply mold half classification colors to the cavity mesh.
        
        Colors the cavity triangles based on their mold half classification:
        - H₁ (Green): Triangles pulled by D1
        - H₂ (Orange): Triangles pulled by D2
        - Boundary Zone (Gray): Interface between H₁ and H₂
        - Inner (Dark Gray): Part surface triangles
        
        Args:
            face_colors: Array of shape (n_faces, 4) with RGBA colors (0-255)
        """
        # This method is kept for backwards compatibility but we now use
        # apply_cavity_classification_vertices_edges for better visualization
        pass
    
    def apply_cavity_classification_vertices_edges(
        self,
        boundary_mesh: 'trimesh.Trimesh',
        classification_result
    ):
        """
        Apply mold half classification colors to vertices and edges (not faces).
        
        Colors vertices and edges based on their mold half classification:
        - H₁ (Green): Vertices/edges on H1 triangles
        - H₂ (Orange): Vertices/edges on H2 triangles  
        - Boundary Zone (Gray): Interface between H₁ and H₂
        - Inner/Seed (Blue): Part surface vertices (Dijkstra seeds)
        
        Args:
            boundary_mesh: The tetrahedral boundary mesh
            classification_result: MoldHalfClassificationResult
            part_mesh: Original part mesh (optional, for identifying seed vertices)
            seed_distance_threshold: Distance threshold for seed vertices (optional)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        logger.info("Applying mold half classification to vertices and edges")
        
        # Remove existing actors
        if self._cavity_actor is not None:
            try:
                self.plotter.remove_actor(self._cavity_actor)
            except Exception:
                pass
            self._cavity_actor = None
        
        if self._mold_halves_actor is not None:
            try:
                self.plotter.remove_actor(self._mold_halves_actor)
            except Exception:
                pass
            self._mold_halves_actor = None
        
        # Get mesh data
        vertices = np.asarray(boundary_mesh.vertices)
        faces = np.asarray(boundary_mesh.faces)
        n_verts = len(vertices)
        
        # Get classification sets
        h1_set = classification_result.h1_triangles
        h2_set = classification_result.h2_triangles
        boundary_set = classification_result.boundary_zone_triangles
        inner_set = classification_result.inner_boundary_triangles
        
        # Assign vertex labels based on adjacent triangles (majority vote)
        # Priority: inner > h1/h2 > boundary
        vertex_h1_count = np.zeros(n_verts, dtype=np.int32)
        vertex_h2_count = np.zeros(n_verts, dtype=np.int32)
        vertex_boundary_count = np.zeros(n_verts, dtype=np.int32)
        vertex_inner_count = np.zeros(n_verts, dtype=np.int32)
        
        for face_idx, face in enumerate(faces):
            v0, v1, v2 = face
            if face_idx in inner_set:
                vertex_inner_count[v0] += 1
                vertex_inner_count[v1] += 1
                vertex_inner_count[v2] += 1
            elif face_idx in h1_set:
                vertex_h1_count[v0] += 1
                vertex_h1_count[v1] += 1
                vertex_h1_count[v2] += 1
            elif face_idx in h2_set:
                vertex_h2_count[v0] += 1
                vertex_h2_count[v1] += 1
                vertex_h2_count[v2] += 1
            elif face_idx in boundary_set:
                vertex_boundary_count[v0] += 1
                vertex_boundary_count[v1] += 1
                vertex_boundary_count[v2] += 1
        
        # Assign vertex colors based on majority
        # Colors: H1=green, H2=orange, boundary=gray, inner=blue
        vertex_colors = np.zeros((n_verts, 3), dtype=np.uint8)
        vertex_labels = np.zeros(n_verts, dtype=np.int8)  # 0=unclassified, 1=H1, 2=H2, 3=boundary, -1=inner
        
        for i in range(n_verts):
            if vertex_inner_count[i] > 0:
                vertex_colors[i] = [0, 100, 255]  # Blue for inner/seed
                vertex_labels[i] = -1
            elif vertex_h1_count[i] > vertex_h2_count[i] and vertex_h1_count[i] > vertex_boundary_count[i]:
                vertex_colors[i] = [76, 175, 80]  # Green for H1
                vertex_labels[i] = 1
            elif vertex_h2_count[i] > vertex_h1_count[i] and vertex_h2_count[i] > vertex_boundary_count[i]:
                vertex_colors[i] = [255, 152, 0]  # Orange for H2
                vertex_labels[i] = 2
            elif vertex_boundary_count[i] > 0:
                vertex_colors[i] = [158, 158, 158]  # Gray for boundary
                vertex_labels[i] = 3
            elif vertex_h1_count[i] > 0:
                vertex_colors[i] = [76, 175, 80]  # Green for H1
                vertex_labels[i] = 1
            elif vertex_h2_count[i] > 0:
                vertex_colors[i] = [255, 152, 0]  # Orange for H2
                vertex_labels[i] = 2
            else:
                vertex_colors[i] = [100, 100, 100]  # Dark gray for unclassified
        
        # Count vertex labels
        n_inner = np.sum(vertex_labels == -1)
        n_h1 = np.sum(vertex_labels == 1)
        n_h2 = np.sum(vertex_labels == 2)
        n_boundary = np.sum(vertex_labels == 3)
        logger.info(f"Vertex labels: H1={n_h1}, H2={n_h2}, boundary={n_boundary}, inner/seed={n_inner}")
        
        # Build edges from all triangles
        edge_set = set()
        for face in faces:
            v0, v1, v2 = face
            edge_set.add((min(v0, v1), max(v0, v1)))
            edge_set.add((min(v1, v2), max(v1, v2)))
            edge_set.add((min(v2, v0), max(v2, v0)))
        
        edges = np.array(list(edge_set))
        n_edges = len(edges)
        logger.info(f"Total edges: {n_edges}")
        
        # Assign edge colors based on endpoint vertex labels
        # If both endpoints are same label, use that color
        # If mixed, use the "higher priority" color (inner > h1/h2 > boundary)
        edge_colors = np.zeros((n_edges, 3), dtype=np.uint8)
        
        for i, (v0, v1) in enumerate(edges):
            l0, l1 = vertex_labels[v0], vertex_labels[v1]
            
            # Priority: inner (-1) > H1 (1) = H2 (2) > boundary (3)
            if l0 == -1 or l1 == -1:
                edge_colors[i] = [0, 100, 255]  # Blue
            elif l0 == 1 and l1 == 1:
                edge_colors[i] = [76, 175, 80]  # Green
            elif l0 == 2 and l1 == 2:
                edge_colors[i] = [255, 152, 0]  # Orange
            elif l0 == 3 and l1 == 3:
                edge_colors[i] = [158, 158, 158]  # Gray
            elif l0 in [1, 2] and l1 in [1, 2]:
                # H1-H2 interface edge - use boundary color
                edge_colors[i] = [158, 158, 158]  # Gray
            elif l0 == 1 or l1 == 1:
                edge_colors[i] = [76, 175, 80]  # Green (one endpoint is H1)
            elif l0 == 2 or l1 == 2:
                edge_colors[i] = [255, 152, 0]  # Orange (one endpoint is H2)
            else:
                edge_colors[i] = [100, 100, 100]  # Dark gray
        
        # Create edge mesh using PyVista's from_irregular_faces for lines
        # Build cells array: [2, p0, p1, 2, p0, p1, ...]
        n_edges = len(edges)
        cells = np.empty((n_edges, 3), dtype=np.int64)
        cells[:, 0] = 2  # Each line has 2 points
        cells[:, 1] = edges[:, 0]
        cells[:, 2] = edges[:, 1]
        cells = cells.ravel()
        
        # Cell types: VTK_LINE = 3
        cell_types = np.full(n_edges, 3, dtype=np.uint8)
        
        # Create unstructured grid for lines with cell colors
        edge_grid = pv.UnstructuredGrid(cells, cell_types, vertices)
        edge_grid.cell_data['colors'] = edge_colors
        
        # Remove existing edges actor
        if self._mold_halves_edges_actor is not None:
            try:
                self.plotter.remove_actor(self._mold_halves_edges_actor)
            except Exception:
                pass
            self._mold_halves_edges_actor = None
        
        # Add edges as colored lines using UnstructuredGrid
        self._mold_halves_edges_actor = self.plotter.add_mesh(
            edge_grid,
            scalars='colors',
            rgb=True,
            line_width=4,
            show_edges=False,
        )
        
        # Set visibility
        if self._mold_halves_edges_actor is not None:
            self._mold_halves_edges_actor.SetVisibility(self._mold_halves_visible)
        
        self.plotter.update()
        logger.info(f"Applied classification to {n_verts} vertices, {n_edges} edges")
    
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
        if self._mold_halves_actor is not None:
            self._mold_halves_actor.SetVisibility(visible)
        self.plotter.update()
    
    def set_outer_boundary_visible(self, visible: bool):
        """Set visibility of outer boundary edges (H1/H2/boundary zone)."""
        self._outer_boundary_visible = visible
        if self._mold_halves_edges_actor is not None:
            self._mold_halves_edges_actor.SetVisibility(visible)
        self.plotter.update()
    
    def set_inner_boundary_visible(self, visible: bool):
        """Set visibility of inner boundary edges (seed edges close to part)."""
        self._inner_boundary_visible = visible
        logger.debug(f"set_inner_boundary_visible({visible})")
        
        # Control all inner boundary actors
        if self._mold_halves_actor is not None:
            self._mold_halves_actor.SetVisibility(visible)
            logger.debug(f"_mold_halves_actor visibility set to {visible}")
        
        if self._inner_boundary_vertices_actor is not None:
            self._inner_boundary_vertices_actor.SetVisibility(visible)
            logger.debug(f"_inner_boundary_vertices_actor visibility set to {visible}")
        
        if self._inner_boundary_edges_actor is not None:
            self._inner_boundary_edges_actor.SetVisibility(visible)
            logger.debug(f"_inner_boundary_edges_actor visibility set to {visible}")
        
        self.plotter.update()
    
    def set_inner_boundary_visualization(
        self,
        boundary_mesh: 'trimesh.Trimesh',
        classification_result
    ):
        """
        Visualize inner boundary (seed) vertices and edges in blue.
        
        This shows vertices that are on the inner boundary (close to part surface)
        and the edges connecting them. These are the seed vertices for Dijkstra.
        
        Args:
            boundary_mesh: The tetrahedral boundary mesh
            classification_result: MoldHalfClassificationResult with inner_boundary_triangles
        """
        if not PYVISTA_AVAILABLE:
            return
        
        import pyvista as pv
        
        # Get inner boundary triangle indices
        inner_triangles = classification_result.inner_boundary_triangles
        
        if len(inner_triangles) == 0:
            logger.warning("No inner boundary triangles to visualize")
            return
        
        logger.info(f"Visualizing {len(inner_triangles)} inner boundary triangles as seed vertices/edges")
        
        # Get all vertices that belong to inner boundary triangles
        faces = np.asarray(boundary_mesh.faces)
        inner_tri_array = np.array(list(inner_triangles))
        inner_faces = faces[inner_tri_array]  # Shape: (n_inner, 3)
        
        # Get unique vertex indices
        inner_vertex_indices = np.unique(inner_faces.flatten())
        inner_vertices = np.asarray(boundary_mesh.vertices)[inner_vertex_indices]
        
        logger.info(f"  Inner boundary: {len(inner_vertex_indices)} vertices")
        
        # Build edges from inner triangles
        # Each triangle has 3 edges: (v0,v1), (v1,v2), (v2,v0)
        edge_set = set()
        for face in inner_faces:
            v0, v1, v2 = face
            edge_set.add((min(v0, v1), max(v0, v1)))
            edge_set.add((min(v1, v2), max(v1, v2)))
            edge_set.add((min(v2, v0), max(v2, v0)))
        
        edges = np.array(list(edge_set))
        logger.info(f"  Inner boundary: {len(edges)} edges")
        
        # Remove existing inner boundary actors
        if self._inner_boundary_vertices_actor is not None:
            try:
                self.plotter.remove_actor(self._inner_boundary_vertices_actor)
            except Exception:
                pass
            self._inner_boundary_vertices_actor = None
        
        if self._inner_boundary_edges_actor is not None:
            try:
                self.plotter.remove_actor(self._inner_boundary_edges_actor)
            except Exception:
                pass
            self._inner_boundary_edges_actor = None
        
        # Create point cloud for inner boundary vertices
        inner_points = pv.PolyData(inner_vertices)
        
        # Add vertices as blue points
        self._inner_boundary_vertices_actor = self.plotter.add_mesh(
            inner_points,
            color='blue',
            point_size=8,
            render_points_as_spheres=True,
            style='points',
        )
        
        # Create line mesh for inner boundary edges
        if len(edges) > 0:
            all_vertices = np.asarray(boundary_mesh.vertices)
            
            # Build lines array for PyVista
            # Format: [n_pts, idx0, idx1, n_pts, idx0, idx1, ...]
            lines = np.zeros((len(edges), 3), dtype=np.int64)
            lines[:, 0] = 2  # Each line has 2 points
            lines[:, 1] = edges[:, 0]
            lines[:, 2] = edges[:, 1]
            
            edge_mesh = pv.PolyData(all_vertices, lines=lines.flatten())
            
            # Add edges as blue lines
            self._inner_boundary_edges_actor = self.plotter.add_mesh(
                edge_mesh,
                color='blue',
                line_width=3,
                style='wireframe',
            )
        
        # Set visibility
        if self._inner_boundary_vertices_actor is not None:
            self._inner_boundary_vertices_actor.SetVisibility(self._inner_boundary_visible)
        if self._inner_boundary_edges_actor is not None:
            self._inner_boundary_edges_actor.SetVisibility(self._inner_boundary_visible)
        
        self.plotter.update()
        logger.info(f"Inner boundary visualization complete: {len(inner_vertex_indices)} vertices, {len(edges)} edges")
    
    @property
    def has_mold_halves(self) -> bool:
        """Check if mold halves classification is displayed."""
        return self._mold_halves_actor is not None
    
    @property
    def cavity_mesh(self) -> Optional[trimesh.Trimesh]:
        """Get the current cavity mesh."""
        return self._cavity_mesh
    
    @property
    def has_cavity(self) -> bool:
        """Check if a cavity has been computed."""
        return self._cavity_mesh is not None

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
            
            # === BOUNDARY EDGES: Colored by mold half ===
            if n_boundary > 0:
                boundary_edges = edges[boundary_mask]
                boundary_labels = edge_boundary_labels[boundary_mask]
                
                # Create RGBA colors for each boundary edge
                # H1 = green, H2 = orange, inner = dark gray, mixed = light gray
                boundary_colors = np.zeros((n_boundary, 4), dtype=np.uint8)
                
                # H1 edges (label=1): Green
                h1_mask = boundary_labels == 1
                boundary_colors[h1_mask] = [76, 175, 80, 255]  # #4CAF50
                
                # H2 edges (label=2): Orange
                h2_mask = boundary_labels == 2
                boundary_colors[h2_mask] = [255, 152, 0, 255]  # #FF9800
                
                # Inner boundary edges (label=-1): Dark gray
                inner_mask = boundary_labels == -1
                boundary_colors[inner_mask] = [80, 80, 80, 255]  # #505050
                
                # Mixed boundary edges (label=-2): Light gray
                mixed_mask = boundary_labels == -2
                boundary_colors[mixed_mask] = [180, 180, 180, 255]  # #B4B4B4
                
                # Unclassified (label=0 but still in boundary mask somehow): Gray
                unclass_mask = boundary_labels == 0
                boundary_colors[unclass_mask] = [150, 150, 150, 255]
                
                # Build PyVista PolyData for boundary edges
                # Use the new cells format for PyVista
                boundary_mesh = pv.PolyData()
                boundary_mesh.points = vertices.astype(np.float32)
                
                # Create cells array: for lines, each cell is [2, v0, v1]
                cells = np.column_stack([
                    np.full(n_boundary, 2, dtype=np.int64),
                    boundary_edges[:, 0],
                    boundary_edges[:, 1]
                ]).ravel()
                
                # Set lines using the lines property with proper format
                boundary_mesh.lines = cells
                
                logger.debug(f"Boundary mesh: {boundary_mesh.n_cells} cells, {len(boundary_colors)} colors")
                
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
        
        # Add with cell shading style (flat interpolation, low specular)
        self._inflated_boundary_actor = self.plotter.add_mesh(
            pv_mesh,
            color=inflated_color,
            opacity=0.6,
            show_edges=False,
            smooth_shading=False,  # Flat shading for cell/toon style
            ambient=0.4,
            diffuse=0.6,
            specular=0.0,
        )
        
        # Configure actor properties for cell shading
        if self._inflated_boundary_actor is not None:
            prop = self._inflated_boundary_actor.GetProperty()
            prop.SetInterpolationToFlat()
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
    # EDGE WEIGHTS VISIBILITY
    # ========================================================================
    
    def set_edge_weights_visible(self, visible: bool):
        """Set visibility of edge weight visualization (interior and boundary edges)."""
        self._edge_weights_visible = visible
        
        if self._tet_interior_actor is not None:
            self._tet_interior_actor.SetVisibility(visible)
        if self._tet_boundary_actor is not None:
            self._tet_boundary_actor.SetVisibility(visible)
        
        self.plotter.update()
    
    @property
    def edge_weights_visible(self) -> bool:
        """Check if edge weights are visible."""
        return self._edge_weights_visible
    
    @property
    def has_edge_weights(self) -> bool:
        """Check if edge weight visualization exists."""
        return self._tet_interior_actor is not None or self._tet_boundary_actor is not None

    # ========================================================================
    # ORIGINAL TETRAHEDRAL MESH VISUALIZATION
    # ========================================================================
    
    def set_original_tetrahedral_mesh(
        self,
        vertices: np.ndarray,
        edges: np.ndarray
    ):
        """
        Display the original (non-inflated) tetrahedral mesh edges.
        
        Args:
            vertices: (N, 3) original vertex positions
            edges: (E, 2) edge vertex indices
        """
        if not PYVISTA_AVAILABLE:
            return
        
        # Remove existing actor
        self.clear_original_tetrahedral_mesh()
        
        n_edges = len(edges)
        logger.info(f"Setting original tetrahedral mesh: {len(vertices)} verts, {n_edges} edges")
        
        # Build PyVista PolyData for edges
        mesh = pv.PolyData()
        mesh.points = vertices.astype(np.float32)
        
        # Create cells array: for lines, each cell is [2, v0, v1]
        cells = np.column_stack([
            np.full(n_edges, 2, dtype=np.int64),
            edges[:, 0],
            edges[:, 1]
        ]).ravel()
        
        mesh.lines = cells
        
        self._tet_original_actor = self.plotter.add_mesh(
            mesh,
            color='#666666',  # Gray for original
            line_width=1.0,
            render_lines_as_tubes=False,
            opacity=0.5
        )
        
        # Initially hidden
        self._tet_original_actor.SetVisibility(self._tet_original_visible)
        
        self.plotter.update()
    
    def clear_original_tetrahedral_mesh(self):
        """Remove original tetrahedral mesh visualization."""
        if not PYVISTA_AVAILABLE:
            return
        
        if self._tet_original_actor is not None:
            try:
                self.plotter.remove_actor(self._tet_original_actor)
            except Exception:
                pass
            self._tet_original_actor = None
    
    def set_original_tet_visible(self, visible: bool):
        """Set visibility of original tetrahedral mesh."""
        self._tet_original_visible = visible
        
        if self._tet_original_actor is not None:
            self._tet_original_actor.SetVisibility(visible)
            self.plotter.update()
    
    @property
    def original_tet_visible(self) -> bool:
        """Check if original tet mesh is visible."""
        return self._tet_original_visible
    
    @property
    def has_original_tet(self) -> bool:
        """Check if original tet mesh exists."""
        return self._tet_original_actor is not None

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
        tet_edges: np.ndarray = None
    ):
        """
        Display Dijkstra results as colored edges for ALL interior vertices.
        
        Shows all interior edges (edges where both vertices are interior - not on H1/H2)
        colored by which boundary (H1 or H2) each vertex walks to via the shortest weighted path.
        
        Edge colors:
        - Green: both endpoints walk to H1
        - Orange: both endpoints walk to H2  
        - Yellow: endpoints walk to opposite boundaries (parting surface passes through)
        - Gray: one or both endpoints unreachable
        
        Args:
            vertices: (N, 3) vertex positions (use original for proper geometry)
            interior_vertex_indices: (I,) indices of interior vertices in the tet mesh
            interior_escape_labels: (I,) labels: 1=walks to H1, 2=walks to H2, 0=unreachable
            boundary_mesh: The tetrahedral boundary mesh (fallback for edge extraction)
            interior_distances: (I,) optional distances to boundary
            tet_edges: (E, 2) tetrahedral mesh edges - used to find all interior edges
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
        
        # Create a mapping from tet vertex index to escape label
        # Initialize all vertices as -1 (not interior / on boundary)
        vertex_escape_labels = np.full(len(vertices), -1, dtype=np.int8)
        for i, vert_idx in enumerate(interior_vertex_indices):
            vertex_escape_labels[vert_idx] = interior_escape_labels[i]
        
        # Get interior vertex set for fast lookup
        interior_set = set(interior_vertex_indices)
        
        # Build edges connecting interior vertices
        # Use tetrahedral mesh edges if provided, otherwise fall back to boundary mesh
        edge_set = set()
        
        if tet_edges is not None:
            # Use all tet edges where BOTH vertices are interior
            for v0, v1 in tet_edges:
                if v0 in interior_set and v1 in interior_set:
                    edge_set.add((min(v0, v1), max(v0, v1)))
            logger.info(f"Found {len(edge_set)} interior edges from {len(tet_edges)} tet edges")
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
                        edge_set.add((min(va, vb), max(va, vb)))
        
        edges = np.array(list(edge_set)) if edge_set else np.empty((0, 2), dtype=np.int64)
        
        n_edges = len(edges)
        logger.info(f"Found {n_edges} edges between interior vertices")
        
        if n_edges > 0:
            # Separate edges into H1/H2 edges and primary cut (yellow) edges
            h1h2_edges = []  # Green and orange edges
            primary_cut_edges = []  # Yellow edges
            h1h2_colors = []
            
            for i, (v0, v1) in enumerate(edges):
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
    
    # Color for secondary parting surface (red, semi-transparent)
    SECONDARY_PARTING_SURFACE_COLOR = "#ff4444"  # Red
    SECONDARY_PARTING_SURFACE_OPACITY = 0.7
    
    def set_parting_surface(self, parting_mesh: trimesh.Trimesh):
        """
        Set and display the parting surface mesh.
        
        Args:
            parting_mesh: The parting surface mesh (triangulated surface separating mold halves)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._parting_surface_mesh = parting_mesh
        
        # Convert to PyVista
        pv_surface = self._trimesh_to_pyvista(parting_mesh)
        
        # Remove existing actor
        if self._parting_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._parting_surface_actor)
            except Exception:
                pass
            self._parting_surface_actor = None
        
        # Add parting surface mesh (semi-transparent blue)
        self._parting_surface_actor = self.plotter.add_mesh(
            pv_surface,
            color=self.PARTING_SURFACE_COLOR,
            opacity=self.PARTING_SURFACE_OPACITY,
            smooth_shading=True,
            show_edges=True,
            edge_color='#1166cc',  # Darker blue for edges
            line_width=1,
            style='surface',
            render_points_as_spheres=False,
        )
        
        # Set up rendering properties for better visibility
        if self._parting_surface_actor is not None:
            prop = self._parting_surface_actor.GetProperty()
            if prop:
                prop.SetInterpolationToPhong()
                # Make it slightly emissive so it's visible even in shadows
                prop.SetAmbient(0.3)
                prop.SetDiffuse(0.7)
        
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
    
    def set_secondary_parting_surface(self, parting_mesh: trimesh.Trimesh):
        """
        Set and display the secondary parting surface mesh.
        
        Args:
            parting_mesh: The secondary parting surface mesh (from secondary cut edges)
        """
        if not PYVISTA_AVAILABLE:
            return
        
        self._secondary_parting_surface_mesh = parting_mesh
        
        # Convert to PyVista
        pv_surface = self._trimesh_to_pyvista(parting_mesh)
        
        # Remove existing actor
        if self._secondary_parting_surface_actor is not None:
            try:
                self.plotter.remove_actor(self._secondary_parting_surface_actor)
            except Exception:
                pass
            self._secondary_parting_surface_actor = None
        
        # Add secondary parting surface mesh (semi-transparent red)
        self._secondary_parting_surface_actor = self.plotter.add_mesh(
            pv_surface,
            color=self.SECONDARY_PARTING_SURFACE_COLOR,
            opacity=self.SECONDARY_PARTING_SURFACE_OPACITY,
            smooth_shading=True,
            show_edges=True,
            edge_color='#cc2222',  # Darker red for edges
            line_width=1,
            style='surface',
            render_points_as_spheres=False,
        )
        
        # Set up rendering properties for better visibility
        if self._secondary_parting_surface_actor is not None:
            prop = self._secondary_parting_surface_actor.GetProperty()
            if prop:
                prop.SetInterpolationToPhong()
                prop.SetAmbient(0.3)
                prop.SetDiffuse(0.7)
        
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
            self.plotter.update()
    
    @property
    def secondary_parting_surface_visible(self) -> bool:
        """Check if secondary parting surface is visible."""
        return self._secondary_parting_surface_visible
    
    @property
    def has_secondary_parting_surface(self) -> bool:
        """Check if secondary parting surface exists."""
        return self._secondary_parting_surface_mesh is not None
