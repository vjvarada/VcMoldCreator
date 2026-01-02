"""
Main Window

Main application window for the VcMoldCreator desktop app.
Designed to match the React frontend's UI/UX with Shards Dashboard theme.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import trimesh

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QFrame, QLabel, QProgressBar, QPushButton, 
    QMessageBox, QFileDialog, QScrollArea, QCheckBox,
    QPlainTextEdit, QSplitter, QDialog, QSlider, QSpinBox,
    QRadioButton, QButtonGroup, QGroupBox, QSizePolicy,
    QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QTextCharFormat, QColor, QFont

from core import (
    STLLoader, LoadResult,
    MeshAnalyzer, MeshDiagnostics,
    MeshRepairer, MeshRepairResult
)
from core.mesh_decimation import (
    MeshDecimator, DecimationQuality, DecimationResult,
    get_decimation_recommendation, TRIANGLE_COUNT_THRESHOLDS
)
from core.parting_direction import (
    find_parting_directions,
    compute_visibility_paint,
    get_face_colors,
    PartingDirectionResult,
    VisibilityPaintData,
)
from core.inflated_hull import (
    generate_inflated_hull,
    compute_default_offset,
    InflatedHullResult,
)
from core.mold_cavity import (
    create_mold_cavity,
    MoldCavityResult,
)
from core.mold_half_classification import (
    classify_mold_halves,
    get_mold_half_face_colors,
    MoldHalfClassificationResult,
)
from core.tetrahedral_mesh import (
    TetrahedralMeshResult,
    PYTETWILD_AVAILABLE,
)
from viewer import MeshViewer

logger = logging.getLogger(__name__)


# ============================================================================
# STEP DEFINITIONS (matching React frontend)
# ============================================================================

class Step(Enum):
    IMPORT = 'import'
    PARTING = 'parting'
    HULL = 'hull'
    CAVITY = 'cavity'
    TETRAHEDRALIZE = 'tetrahedralize'  # Now comes before mold halves
    MOLD_HALVES = 'mold-halves'        # Now operates on tet boundary
    EDGE_WEIGHTS = 'edge-weights'
    DIJKSTRA = 'dijkstra'              # Dijkstra walk to find parting surface
    SECONDARY_CUTS = 'secondary-cuts'  # Secondary cutting edges detection
    PARTING_SURFACE = 'parting-surface'  # Primary parting surface: generation + repair + smoothing
    SECONDARY_SURFACE = 'secondary-surface'  # Secondary surface: generation + propagation + smoothing


STEPS = [
    {'id': Step.IMPORT, 'icon': '📁', 'title': 'Import STL', 'description': 'Load a 3D model file in STL format for mold analysis'},
    {'id': Step.PARTING, 'icon': '🔀', 'title': 'Parting Direction', 'description': 'Compute optimal parting directions for mold separation'},
    {'id': Step.HULL, 'icon': '📦', 'title': 'Bounding Hull', 'description': 'Generate inflated convex hull for mold cavity creation'},
    {'id': Step.CAVITY, 'icon': '🕳️', 'title': 'Mold Cavity', 'description': 'Create mold cavity by subtracting the original mesh from the hull'},
    {'id': Step.TETRAHEDRALIZE, 'icon': '🔷', 'title': 'Tetrahedralize', 'description': 'Generate tetrahedral mesh of mold cavity volume'},
    {'id': Step.MOLD_HALVES, 'icon': '🎨', 'title': 'Mold Halves', 'description': 'Classify tetrahedral boundary into H₁ and H₂ mold halves'},
    {'id': Step.EDGE_WEIGHTS, 'icon': '⚖️', 'title': 'Edge Weights', 'description': 'Compute edge weights based on distance to part surface'},
    {'id': Step.DIJKSTRA, 'icon': '🛤️', 'title': 'Dijkstra Walk', 'description': 'Find shortest paths from part surface to mold halves'},
    {'id': Step.SECONDARY_CUTS, 'icon': '✂️', 'title': 'Secondary Cuts', 'description': 'Detect secondary cutting edges where membrane intersects part'},
    {'id': Step.PARTING_SURFACE, 'icon': '🔲', 'title': 'Primary Surface', 'description': 'Generate, repair and smooth the primary parting surface'},
    {'id': Step.SECONDARY_SURFACE, 'icon': '🔴', 'title': 'Secondary Surface', 'description': 'Generate, propagate and smooth secondary parting surfaces'},
]


# ============================================================================
# COLORS (matching React frontend - Shards Dashboard Theme)
# ============================================================================

class Colors:
    # Primary colors
    PRIMARY = '#007bff'
    SUCCESS = '#17c671'
    WARNING = '#ffb400'
    DANGER = '#c4183c'
    INFO = '#00b8d8'
    
    # Neutral colors
    DARK = '#212529'
    GRAY_DARK = '#5A6169'
    GRAY = '#868e96'
    GRAY_LIGHT = '#becad6'
    LIGHT = '#f5f6f8'
    WHITE = '#ffffff'
    
    # Viewer colors
    VIEWER_BG = '#1a1d21'
    BORDER = '#e1e5eb'
    
    # Parting direction colors (matching React frontend)
    PARTING_D1 = '#00ff00'  # Green - Primary direction
    PARTING_D2 = '#ff6600'  # Orange - Secondary direction


# ============================================================================
# QT LOGGING HANDLER
# ============================================================================

class LogSignalEmitter(QObject):
    """Signal emitter for logging handler."""
    log_message = pyqtSignal(str, int)  # message, level


class QtLogHandler(logging.Handler):
    """Custom logging handler that emits Qt signals for log messages."""
    
    def __init__(self):
        super().__init__()
        self.emitter = LogSignalEmitter()
        self.setFormatter(logging.Formatter('%(message)s'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.emitter.log_message.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)


class ConsoleLogPanel(QFrame):
    """Console log panel for displaying operation logs."""
    
    # Colors matching 3D viewer theme
    PANEL_BG = "#1a1a1a"
    PANEL_BORDER = "#333333"
    TEXT_COLOR = "#aaaaaa"
    
    # Log level colors
    LEVEL_COLORS = {
        logging.DEBUG: "#888888",
        logging.INFO: "#aaaaaa",
        logging.WARNING: "#ffb400",
        logging.ERROR: "#c4183c",
        logging.CRITICAL: "#ff0000",
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("consoleLogPanel")
        
        self.setStyleSheet(f"""
            #consoleLogPanel {{
                background-color: {self.PANEL_BG};
                border-top: 1px solid {self.PANEL_BORDER};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header bar
        header = QFrame()
        header.setFixedHeight(24)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: #252525;
                border-bottom: 1px solid {self.PANEL_BORDER};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        title = QLabel("Console")
        title.setStyleSheet(f"""
            font-size: 11px;
            font-weight: bold;
            color: {self.TEXT_COLOR};
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.TEXT_COLOR};
                font-size: 10px;
                padding: 2px 8px;
            }}
            QPushButton:hover {{
                color: #ffffff;
            }}
        """)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_log)
        header_layout.addWidget(clear_btn)
        
        layout.addWidget(header)
        
        # Log text area
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(500)  # Limit lines to prevent memory issues
        self.log_text.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {self.PANEL_BG};
                color: {self.TEXT_COLOR};
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 6px;
            }}
            QPlainTextEdit QScrollBar:vertical {{
                background: #252525;
                width: 8px;
                margin: 0;
            }}
            QPlainTextEdit QScrollBar::handle:vertical {{
                background: #444444;
                border-radius: 4px;
                min-height: 20px;
            }}
            QPlainTextEdit QScrollBar::add-line:vertical,
            QPlainTextEdit QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        layout.addWidget(self.log_text)
        
        # Setup log handler
        self._setup_logging()
        
        # Initial message
        self._append_log("Console initialized. Waiting for operations...", logging.INFO)
    
    def _setup_logging(self):
        """Setup the Qt logging handler."""
        self.log_handler = QtLogHandler()
        self.log_handler.emitter.log_message.connect(self._append_log)
        
        # Add handler to root logger and our app loggers
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        # Also add to specific loggers
        for logger_name in ['__main__', 'core', 'viewer', 'ui']:
            log = logging.getLogger(logger_name)
            log.addHandler(self.log_handler)
    
    def _append_log(self, message: str, level: int):
        """Append a log message to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_name = logging.getLevelName(level)
        color = self.LEVEL_COLORS.get(level, self.TEXT_COLOR)
        
        # Format with HTML for colors
        formatted = f'<span style="color: #666666;">[{timestamp}]</span> <span style="color: {color};">[{level_name}]</span> {message}'
        
        # Use appendHtml to add colored text
        self.log_text.appendHtml(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _clear_log(self):
        """Clear the log console."""
        self.log_text.clear()
        self._append_log("Console cleared.", logging.INFO)
    
    def log(self, message: str, level: int = logging.INFO):
        """Manually log a message to the console."""
        self._append_log(message, level)


# ============================================================================
# MESH DECIMATION DIALOG
# ============================================================================

class MeshDecimationDialog(QDialog):
    """
    Dialog for mesh decimation options.
    
    Shows when a mesh has too many triangles and allows user to choose
    whether to decimate the mesh or use it as-is.
    """
    
    def __init__(self, triangle_count: int, recommendation: dict, parent=None):
        super().__init__(parent)
        self.triangle_count = triangle_count
        self.recommendation = recommendation
        self._selected_target = recommendation.get('suggested_target', 100000)
        self._should_decimate = False
        
        self._setup_dialog()
    
    @property
    def should_decimate(self) -> bool:
        """Return True if user chose to decimate the mesh."""
        return self._should_decimate
    
    @property
    def target_triangles(self) -> int:
        """Return the target triangle count if decimation was chosen."""
        return self._selected_target
    
    def _setup_dialog(self):
        self.setWindowTitle("Mesh Optimization")
        self.setModal(True)
        self.setMinimumWidth(450)
        
        # Dialog styling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.LIGHT};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: {Colors.WHITE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {Colors.DARK};
            }}
            QLabel {{
                color: {Colors.DARK};
            }}
            QRadioButton {{
                color: {Colors.DARK};
                spacing: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Warning Icon and Title
        header_layout = QHBoxLayout()
        
        severity = self.recommendation.get('severity', 'warning')
        if severity == 'critical':
            icon = "⚠️"
            title_color = Colors.DANGER
        elif severity == 'high':
            icon = "⚡"
            title_color = Colors.WARNING
        else:
            icon = "💡"
            title_color = Colors.INFO
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 32px;")
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("High Triangle Count Detected")
        title_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {title_color};
        """)
        header_layout.addWidget(title_label, 1)
        layout.addLayout(header_layout)
        
        # Message
        message = self.recommendation.get('message', 'Mesh may benefit from optimization.')
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"color: {Colors.GRAY_DARK}; font-size: 13px; line-height: 1.4;")
        layout.addWidget(message_label)
        
        # Current Stats
        stats_group = QGroupBox("Current Mesh Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        stats_text = f"""
        <table style="width:100%;">
            <tr><td style="color: {Colors.GRAY};">Triangles:</td><td style="color: {Colors.DARK}; font-weight: bold;">{self.triangle_count:,}</td></tr>
            <tr><td style="color: {Colors.GRAY};">Recommended Max:</td><td style="color: {Colors.SUCCESS};">{TRIANGLE_COUNT_THRESHOLDS['recommended']:,}</td></tr>
            <tr><td style="color: {Colors.GRAY};">Over Limit By:</td><td style="color: {Colors.DANGER};">{self.triangle_count - TRIANGLE_COUNT_THRESHOLDS['recommended']:,} ({((self.triangle_count / TRIANGLE_COUNT_THRESHOLDS['recommended']) - 1) * 100:.0f}% more)</td></tr>
        </table>
        """
        stats_label = QLabel(stats_text)
        stats_label.setTextFormat(Qt.TextFormat.RichText)
        stats_layout.addWidget(stats_label)
        layout.addWidget(stats_group)
        
        # Options Group
        options_group = QGroupBox("What would you like to do?")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(12)
        
        self.button_group = QButtonGroup(self)
        
        # Option 1: Optimize (Decimate)
        self.optimize_radio = QRadioButton("Optimize mesh (recommended)")
        self.optimize_radio.setChecked(True)
        self.optimize_radio.toggled.connect(self._on_option_changed)
        self.button_group.addButton(self.optimize_radio)
        options_layout.addWidget(self.optimize_radio)
        
        # Target triangle slider
        self.target_container = QWidget()
        target_layout = QVBoxLayout(self.target_container)
        target_layout.setContentsMargins(28, 0, 0, 0)
        
        target_label = QLabel("Target triangle count:")
        target_label.setStyleSheet(f"color: {Colors.GRAY_DARK}; font-size: 12px;")
        target_layout.addWidget(target_label)
        
        slider_layout = QHBoxLayout()
        
        self.target_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_slider.setMinimum(10000)
        self.target_slider.setMaximum(min(self.triangle_count, 500000))
        self.target_slider.setValue(self._selected_target)
        self.target_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.target_slider.setTickInterval(50000)
        self.target_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.target_slider, 1)
        
        self.target_spinbox = QSpinBox()
        self.target_spinbox.setMinimum(10000)
        self.target_spinbox.setMaximum(min(self.triangle_count, 500000))
        self.target_spinbox.setValue(self._selected_target)
        self.target_spinbox.setSingleStep(10000)
        self.target_spinbox.setFixedWidth(100)
        self.target_spinbox.valueChanged.connect(self._on_spinbox_changed)
        slider_layout.addWidget(self.target_spinbox)
        
        target_layout.addLayout(slider_layout)
        
        # Reduction preview
        self.reduction_label = QLabel()
        self._update_reduction_label()
        target_layout.addWidget(self.reduction_label)
        
        options_layout.addWidget(self.target_container)
        
        # Option 2: Use as-is
        self.keep_radio = QRadioButton("Use original mesh (may slow down processing)")
        self.button_group.addButton(self.keep_radio)
        options_layout.addWidget(self.keep_radio)
        
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(100)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.WHITE};
                color: {Colors.GRAY_DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.LIGHT};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedWidth(100)
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
        """)
        self.apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
    
    def _on_option_changed(self, checked: bool):
        """Handle option radio button change."""
        self.target_container.setEnabled(checked)
        self.target_container.setVisible(checked)
        self.adjustSize()
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        self._selected_target = value
        self.target_spinbox.blockSignals(True)
        self.target_spinbox.setValue(value)
        self.target_spinbox.blockSignals(False)
        self._update_reduction_label()
    
    def _on_spinbox_changed(self, value: int):
        """Handle spinbox value change."""
        self._selected_target = value
        self.target_slider.blockSignals(True)
        self.target_slider.setValue(value)
        self.target_slider.blockSignals(False)
        self._update_reduction_label()
    
    def _update_reduction_label(self):
        """Update the reduction percentage label."""
        reduction = ((self.triangle_count - self._selected_target) / self.triangle_count) * 100
        self.reduction_label.setText(
            f"<span style='color: {Colors.GRAY};'>Expected reduction: </span>"
            f"<span style='color: {Colors.SUCCESS}; font-weight: bold;'>{reduction:.1f}%</span>"
        )
        self.reduction_label.setTextFormat(Qt.TextFormat.RichText)
    
    def _on_apply(self):
        """Handle apply button click."""
        self._should_decimate = self.optimize_radio.isChecked()
        self._selected_target = self.target_spinbox.value()
        self.accept()


# ============================================================================
# WORKER THREAD
# ============================================================================

class MeshLoadWorker(QThread):
    """Background worker for loading and processing meshes."""
    
    progress = pyqtSignal(str)
    load_complete = pyqtSignal(object)
    analysis_complete = pyqtSignal(object)
    repair_complete = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, file_path: str, scale_factor: float = 1.0):
        super().__init__()
        self.file_path = file_path
        self.scale_factor = scale_factor
        self._should_repair = True
    
    def run(self):
        try:
            logger.info(f"Loading STL file: {self.file_path}")
            self.progress.emit("Loading STL file...")
            loader = STLLoader()
            result = loader.load(self.file_path)
            
            if not result.success:
                logger.error(f"Failed to load STL: {result.error_message}")
                self.error.emit(f"Failed to load STL: {result.error_message}")
                return
            
            # Apply scale factor if not 1.0
            if self.scale_factor != 1.0:
                logger.info(f"Applying scale factor: {self.scale_factor}")
                self.progress.emit(f"Scaling mesh by {self.scale_factor}x...")
                result.mesh.vertices *= self.scale_factor
            
            logger.info(f"STL loaded: {result.mesh.vertices.shape[0]} vertices, {result.mesh.faces.shape[0]} faces")
            self.load_complete.emit(result)
            self.progress.emit(f"Loaded in {result.load_time_ms:.1f}ms")
            
            # === PHASE 1: Initial analysis ===
            logger.info("Analyzing mesh...")
            self.progress.emit("Analyzing mesh quality...")
            analyzer = MeshAnalyzer(result.mesh)
            diagnostics = analyzer.analyze()
            logger.info(f"Analysis complete: manifold={diagnostics.is_manifold}, watertight={diagnostics.is_watertight}")
            
            # === PHASE 2: Always run mesh cleanup (even on watertight meshes) ===
            # This ensures consistent mesh quality for downstream operations
            if self._should_repair:
                logger.info("Running mesh cleanup...")
                self.progress.emit("Cleaning up mesh (merging vertices, removing degenerates)...")
                
                repairer = MeshRepairer(result.mesh)
                # Always run cleanup operations but don't use convex hull fallback
                repair_result = repairer.repair(
                    merge_vertices=True,
                    remove_degenerate=True,
                    remove_duplicates=True,
                    fix_normals=True,
                    fill_holes=not diagnostics.is_watertight,  # Only fill holes if needed
                    use_convex_hull_fallback=False  # Never destroy original geometry
                )
                
                # Log what was done
                if repair_result.was_repaired:
                    logger.info(f"Mesh cleaned: {repair_result.repair_method}")
                    for step in repair_result.repair_steps:
                        logger.info(f"  - {step}")
                    
                    # Build summary message
                    summary_parts = []
                    for step in repair_result.repair_steps:
                        if "Merged" in step:
                            summary_parts.append(step)
                        elif "Removed" in step:
                            summary_parts.append(step)
                        elif "Filled" in step:
                            summary_parts.append(step)
                        elif "Fixed" in step and "normal" in step.lower():
                            summary_parts.append("Fixed normals")
                    
                    if summary_parts:
                        self.progress.emit(f"Cleanup: {'; '.join(summary_parts[:3])}")
                    else:
                        self.progress.emit("Mesh cleanup complete")
                else:
                    logger.info("Mesh is clean, no repairs needed")
                    self.progress.emit("Mesh is clean, no repairs needed")
                
                # Emit repair complete (updates mesh in viewer if needed)
                self.repair_complete.emit(repair_result)
                
                # Re-analyze after repair
                if repair_result.was_repaired:
                    analyzer = MeshAnalyzer(repair_result.mesh)
                    diagnostics = analyzer.analyze()
            
            # Emit analysis complete (with potentially updated diagnostics)
            self.analysis_complete.emit(diagnostics)
            self.progress.emit("Mesh processing complete")
                
        except Exception as e:
            logger.exception(f"Error in mesh processing: {e}")
            self.error.emit(str(e))


class PartingDirectionWorker(QThread):
    """Background worker for computing parting directions."""
    
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int, int)  # current, total
    complete = pyqtSignal(object)  # PartingDirectionResult
    error = pyqtSignal(str)
    
    def __init__(self, mesh, k: int = 128, num_workers: int = None):
        """
        Initialize parting direction worker.
        
        Args:
            mesh: The trimesh mesh to analyze
            k: Number of candidate directions to sample (default: 128)
            num_workers: Number of parallel workers (default: auto)
        """
        super().__init__()
        self.mesh = mesh
        self.k = k
        self.num_workers = num_workers
    
    def run(self):
        try:
            logger.info(f"Computing parting directions with k={self.k}...")
            self.progress.emit(f"Sampling {self.k} candidate directions...")
            
            def progress_callback(current, total):
                self.progress_value.emit(current, total)
                if current % 10 == 0 or current == total:
                    self.progress.emit(f"Testing direction {current}/{total}...")
            
            result = find_parting_directions(
                self.mesh,
                k=self.k,
                num_workers=self.num_workers,
                progress_callback=progress_callback
            )
            
            self.progress.emit(f"Found optimal parting directions in {result.computation_time_ms:.0f}ms")
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error computing parting directions: {e}")
            self.error.emit(str(e))


class VisibilityPaintWorker(QThread):
    """Background worker for computing visibility paint."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object, object)  # VisibilityPaintData, face_colors
    error = pyqtSignal(str)
    
    def __init__(self, mesh, d1: np.ndarray, d2: np.ndarray, show_d1: bool = True, show_d2: bool = True):
        """
        Initialize visibility paint worker.
        
        Args:
            mesh: The trimesh mesh
            d1: Primary parting direction
            d2: Secondary parting direction
            show_d1: Whether to show D1 visibility
            show_d2: Whether to show D2 visibility
        """
        super().__init__()
        self.mesh = mesh
        self.d1 = d1
        self.d2 = d2
        self.show_d1 = show_d1
        self.show_d2 = show_d2
    
    def run(self):
        try:
            logger.info("Computing visibility paint...")
            self.progress.emit("Computing triangle visibility...")
            
            paint_data = compute_visibility_paint(
                self.mesh,
                self.d1,
                self.d2,
                show_d1=self.show_d1,
                show_d2=self.show_d2
            )
            
            # Convert to face colors
            face_colors = get_face_colors(paint_data)
            
            self.progress.emit("Visibility paint computed")
            self.complete.emit(paint_data, face_colors)
            
        except Exception as e:
            logger.exception(f"Error computing visibility paint: {e}")
            self.error.emit(str(e))


class HullWorker(QThread):
    """Background worker for computing inflated convex hull."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # InflatedHullResult
    error = pyqtSignal(str)
    
    def __init__(self, mesh, offset: float, subdivisions: int = 2):
        """
        Initialize hull worker.
        
        Args:
            mesh: The trimesh mesh to create hull around
            offset: Distance to inflate outward
            subdivisions: Number of subdivision iterations (default: 2)
        """
        super().__init__()
        self.mesh = mesh
        self.offset = offset
        self.subdivisions = subdivisions
    
    def run(self):
        try:
            logger.info(f"Computing inflated hull with offset={self.offset}, subdivisions={self.subdivisions}...")
            self.progress.emit(f"Computing smooth vertex normals...")
            
            result = generate_inflated_hull(
                self.mesh,
                offset=self.offset,
                subdivisions=self.subdivisions
            )
            
            self.progress.emit(f"Hull generated: {result.vertex_count} vertices, {result.face_count} faces")
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error computing inflated hull: {e}")
            self.error.emit(str(e))


class CavityWorker(QThread):
    """Background worker for computing mold cavity (CSG subtraction)."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # MoldCavityResult
    error = pyqtSignal(str)
    
    def __init__(self, hull_mesh, original_mesh):
        """
        Initialize cavity worker.
        
        Args:
            hull_mesh: The inflated hull mesh
            original_mesh: The original mesh to subtract from hull
        """
        super().__init__()
        self.hull_mesh = hull_mesh
        self.original_mesh = original_mesh
    
    def run(self):
        try:
            logger.info("Computing mold cavity (CSG subtraction)...")
            self.progress.emit("Performing CSG subtraction (hull - original)...")
            
            result = create_mold_cavity(
                self.hull_mesh,
                self.original_mesh
            )
            
            if result.success:
                self.progress.emit(f"Cavity created: {result.validation.vertex_count} vertices, {result.validation.face_count} faces")
                self.complete.emit(result)
            else:
                self.error.emit(result.error_message or "CSG subtraction failed")
            
        except Exception as e:
            logger.exception(f"Error computing mold cavity: {e}")
            self.error.emit(str(e))


class MoldHalvesWorker(QThread):
    """Background worker for classifying mold halves (H₁ and H₂)."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # MoldHalfClassificationResult
    error = pyqtSignal(str)
    
    def __init__(
        self,
        cavity_mesh,
        hull_mesh,
        d1: np.ndarray,
        d2: np.ndarray,
        boundary_zone_threshold: float = 0.15,
        part_mesh=None,
        use_proximity_method: bool = False,
        tet_vertices: np.ndarray = None,
        tet_boundary_labels: np.ndarray = None
    ):
        """
        Initialize mold halves classification worker.
        
        Args:
            cavity_mesh: The mold cavity mesh (or tetrahedral boundary mesh)
            hull_mesh: The hull mesh (used to identify outer boundary)
            d1: Primary parting direction
            d2: Secondary parting direction
            boundary_zone_threshold: Threshold for boundary zone (0-1)
            part_mesh: Original part mesh (for proximity-based classification)
            use_proximity_method: Use proximity-based outer boundary detection
            tet_vertices: (N, 3) tet mesh vertices (enables fast label-based detection)
            tet_boundary_labels: (N,) boundary labels from tet mesh
        """
        super().__init__()
        self.cavity_mesh = cavity_mesh
        self.hull_mesh = hull_mesh
        self.d1 = d1
        self.d2 = d2
        self.boundary_zone_threshold = boundary_zone_threshold
        self.part_mesh = part_mesh
        self.use_proximity_method = use_proximity_method
        self.tet_vertices = tet_vertices
        self.tet_boundary_labels = tet_boundary_labels
    
    def run(self):
        try:
            logger.info("Classifying mold halves...")
            self.progress.emit("Classifying mold halves...")
            
            result = classify_mold_halves(
                self.cavity_mesh,
                self.hull_mesh,
                self.d1,
                self.d2,
                self.boundary_zone_threshold,
                part_mesh=self.part_mesh,
                use_proximity_method=self.use_proximity_method,
                tet_vertices=self.tet_vertices,
                tet_boundary_labels=self.tet_boundary_labels
            )
            
            self.progress.emit(
                f"Classification complete: H₁={len(result.h1_triangles)}, "
                f"H₂={len(result.h2_triangles)}"
            )
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error classifying mold halves: {e}")
            self.error.emit(str(e))


def _run_tetrahedralization_subprocess(vertices, faces, edge_length_fac, optimize):
    """
    Run tetrahedralization in a subprocess to avoid GIL blocking.
    
    This function is called in a separate process via multiprocessing.
    Returns tuple of (tet_vertices, tetrahedra) or raises exception.
    """
    import pytetwild
    tet_vertices, tetrahedra = pytetwild.tetrahedralize(
        vertices, faces,
        optimize=optimize,
        edge_length_fac=edge_length_fac
    )
    return tet_vertices, tetrahedra


class TetrahedralizeWorker(QThread):
    """Background worker for generating tetrahedral mesh of mold volume.
    
    Runs pytetwild in a separate PROCESS (not just thread) to avoid GIL blocking.
    This keeps the UI fully responsive during tetrahedralization.
    """
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # TetrahedralMeshResult
    error = pyqtSignal(str)
    
    def __init__(
        self, 
        cavity_mesh, 
        part_mesh=None,
        edge_length_fac: float = 0.05,
        optimize: bool = True
    ):
        """
        Initialize tetrahedralization worker.
        
        Args:
            cavity_mesh: The mold cavity mesh
            part_mesh: The original part mesh (for filtering tetrahedra inside part)
            edge_length_fac: Target edge length as fraction of bbox diagonal
            optimize: Whether to optimize tet quality
        """
        super().__init__()
        self.cavity_mesh = cavity_mesh
        self.part_mesh = part_mesh
        self.edge_length_fac = edge_length_fac
        self.optimize = optimize
        self._cancelled = False
        self._process = None
    
    def cancel(self):
        """Request cancellation of the operation."""
        self._cancelled = True
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
    
    def run(self):
        import time
        import multiprocessing as mp
        import concurrent.futures
        from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
        from core.tetrahedral_mesh import extract_edges, compute_edge_lengths, extract_boundary_surface, TetrahedralMeshResult
        
        try:
            if not PYTETWILD_AVAILABLE:
                self.error.emit("pytetwild is not installed. Install with: pip install pytetwild")
                return
            
            logger.info("Generating tetrahedral mesh in subprocess...")
            self.progress.emit("Starting tetrahedralization (subprocess)...")
            
            # Get mesh data as numpy arrays
            vertices = np.asarray(self.cavity_mesh.vertices, dtype=np.float64)
            faces = np.asarray(self.cavity_mesh.faces, dtype=np.int32)
            
            start_time = time.time()
            
            self.progress.emit(f"Tetrahedralizing {len(vertices)} vertices, {len(faces)} faces...")
            
            # Run tetrahedralization in a separate process to avoid GIL blocking
            # Use spawn context to ensure clean process on Windows
            ctx = mp.get_context('spawn')
            
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
                future = executor.submit(
                    _run_tetrahedralization_subprocess,
                    vertices, faces,
                    self.edge_length_fac, self.optimize
                )
                
                # Poll for completion while checking for cancellation
                while not future.done():
                    if self._cancelled:
                        future.cancel()
                        self.error.emit("Cancelled by user")
                        return
                    
                    # Update progress with elapsed time
                    elapsed = time.time() - start_time
                    self.progress.emit(f"Tetrahedralizing... ({elapsed:.1f}s elapsed)")
                    
                    # Sleep briefly to allow UI updates
                    time.sleep(0.5)
                
                # Get result - handle BrokenProcessPool specifically
                try:
                    tet_vertices, tetrahedra = future.result()
                except concurrent.futures.process.BrokenProcessPool as e:
                    logger.error(f"TetGen process crashed: {e}")
                    self.error.emit(
                        "Tetrahedralization failed: TetGen process crashed.\n"
                        "This may be caused by problematic mesh geometry.\n"
                        "Try: 1) Repair mesh, 2) Simplify mesh, 3) Adjust edge length factor."
                    )
                    return
            
            tet_time = (time.time() - start_time) * 1000
            
            if self._cancelled:
                self.error.emit("Cancelled by user")
                return
            
            # Filter out tetrahedra that are inside the part mesh
            if self.part_mesh is not None:
                self.progress.emit("Filtering tetrahedra inside part mesh...")
                from core.tetrahedral_mesh import filter_tetrahedra_outside_part
                tet_vertices, tetrahedra = filter_tetrahedra_outside_part(
                    tet_vertices, tetrahedra, self.part_mesh
                )
                
                if len(tetrahedra) == 0:
                    self.error.emit("All tetrahedra were inside the part mesh. Check mesh quality.")
                    return
            
            self.progress.emit("Extracting edges...")
            
            # Extract edges (this is fast, can run in thread)
            edges = extract_edges(tetrahedra)
            edge_lengths = compute_edge_lengths(tet_vertices, edges)
            
            self.progress.emit("Extracting boundary surface...")
            
            # Extract boundary surface from tetrahedral mesh
            boundary_mesh = extract_boundary_surface(tet_vertices, tetrahedra)
            
            # Build result
            result = TetrahedralMeshResult(
                vertices=tet_vertices,
                tetrahedra=tetrahedra,
                edges=edges,
                edge_lengths=edge_lengths,
                boundary_mesh=boundary_mesh,
                num_vertices=len(tet_vertices),
                num_tetrahedra=len(tetrahedra),
                num_edges=len(edges),
                num_boundary_faces=len(boundary_mesh.faces),
                tetrahedralize_time_ms=tet_time
            )
            
            self.progress.emit(
                f"Complete: {result.num_vertices} vertices, "
                f"{result.num_tetrahedra} tetrahedra, {result.num_edges} edges"
            )
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error generating tetrahedral mesh: {e}")
            self.error.emit(str(e))


class EdgeWeightsWorker(QThread):
    """Background worker for computing edge weights based on distance to part surface."""
    
    progress = pyqtSignal(str)
    r_distance_computed = pyqtSignal(object)  # Dict with hull_point, part_point, r_value
    mesh_inflated = pyqtSignal(object)  # Updated TetrahedralMeshResult after inflation
    boundary_classified = pyqtSignal(object)  # MoldHalfClassificationResult for the tet boundary mesh
    complete = pyqtSignal(object)  # Updated TetrahedralMeshResult
    error = pyqtSignal(str)
    
    def __init__(
        self, 
        tet_result,  # TetrahedralMeshResult from previous step
        part_mesh,
        cavity_mesh,
        hull_mesh,
        classification_result,
        d1=None,  # Parting direction 1 (for classifying tet boundary mesh)
        d2=None   # Parting direction 2 (for classifying tet boundary mesh)
    ):
        """
        Initialize edge weights worker.
        
        Args:
            tet_result: TetrahedralMeshResult from tetrahedralization step
            part_mesh: Original part mesh (for distance computation)
            cavity_mesh: Cavity mesh (for proximity-based boundary detection)
            hull_mesh: Hull mesh (for outer boundary identification)
            classification_result: Mold half classification result (used for d1/d2 if not provided)
            d1: Parting direction 1 (optional, will use default if not provided)
            d2: Parting direction 2 (optional, will use default if not provided)
        """
        super().__init__()
        self.tet_result = tet_result
        self.part_mesh = part_mesh
        self.cavity_mesh = cavity_mesh
        self.hull_mesh = hull_mesh
        self.classification_result = classification_result
        self.d1 = d1
        self.d2 = d2
    
    def run(self):
        try:
            from core.tetrahedral_mesh import (
                compute_distances_to_mesh,
                compute_distances_and_closest_points_gpu,
                identify_boundary_vertices,
                label_boundary_from_classification,
                CUDA_AVAILABLE
            )
            import time
            
            logger.info("Computing edge weights...")
            start_time = time.time()
            
            result = self.tet_result
            vertices = result.vertices
            edges = result.edges
            edge_lengths = result.edge_lengths
            
            # Step 1: Compute R - maximum distance from tetrahedral boundary surface to part surface
            self.progress.emit("Computing R (max boundary-to-part distance)...")
            
            # Get vertices from tetrahedral boundary surface
            boundary_mesh = result.boundary_mesh
            if boundary_mesh is None:
                self.error.emit("Tetrahedral boundary mesh not available")
                return
            
            boundary_vertices = np.asarray(boundary_mesh.vertices, dtype=np.float64)
            logger.info(f"Computing R using {len(boundary_vertices)} boundary vertices")
            
            # Compute distances from all boundary vertices to part surface
            # Use GPU if available for faster computation
            if CUDA_AVAILABLE:
                self.progress.emit(f"Computing R using GPU (CUDA) for {len(boundary_vertices)} vertices...")
                boundary_to_part_distances, closest_points = compute_distances_and_closest_points_gpu(
                    boundary_vertices, self.part_mesh
                )
            else:
                self.progress.emit(f"Computing R using CPU for {len(boundary_vertices)} vertices...")
                closest_points, boundary_to_part_distances, _ = self.part_mesh.nearest.on_surface(boundary_vertices)
            
            # Find the maximum distance
            max_idx = np.argmax(boundary_to_part_distances)
            r_value = float(boundary_to_part_distances[max_idx])
            boundary_point = boundary_vertices[max_idx].copy()
            part_point = closest_points[max_idx].copy()
            
            r_time = (time.time() - start_time) * 1000
            logger.info(f"R computed in {r_time:.0f}ms: {r_value:.4f} (max boundary-to-part distance)")
            logger.info(f"  Boundary point: {boundary_point}")
            logger.info(f"  Part point: {part_point}")
            
            # Emit R distance data for visualization
            r_data = {
                'hull_point': boundary_point,  # Keep key name for compatibility
                'part_point': part_point,
                'r_value': r_value
            }
            self.r_distance_computed.emit(r_data)
            
            # Store R in result for later use
            result.r_value = r_value
            result.r_hull_point = boundary_point  # Keep attribute name for compatibility
            result.r_part_point = part_point
            
            # Step 2: Classify the ORIGINAL boundary mesh BEFORE inflation
            # This is critical - after inflation, outer boundary triangles may be 
            # misclassified because their centroids move away from hull.
            self.progress.emit("Classifying boundary mesh (before inflation)...")
            
            from core.mold_half_classification import classify_mold_halves
            
            # Get parting directions (d1, d2) - either from constructor or use defaults
            if self.d1 is not None and self.d2 is not None:
                d1 = self.d1
                d2 = self.d2
            else:
                # Default parting directions (along Z-axis)
                d1 = np.array([0.0, 0.0, 1.0])
                d2 = np.array([0.0, 0.0, -1.0])
                logger.warning("Using default parting directions (Z-axis)")
            
            # Classify the ORIGINAL boundary mesh using proximity method
            # Triangle indices stay the same after inflation, so this classification
            # can be applied to the inflated mesh
            original_boundary_mesh = result.boundary_mesh
            tet_boundary_classification = classify_mold_halves(
                original_boundary_mesh,  # The ORIGINAL tetrahedral boundary mesh
                self.hull_mesh,
                d1, d2,
                boundary_zone_threshold=0.15,
                part_mesh=self.part_mesh,
                use_proximity_method=True  # Important: use proximity for tet boundary mesh
            )
            
            classify_time = (time.time() - start_time) * 1000 - r_time
            logger.info(f"Boundary mesh classified in {classify_time:.0f}ms")
            logger.info(
                f"  H1={len(tet_boundary_classification.h1_triangles)}, "
                f"H2={len(tet_boundary_classification.h2_triangles)}, "
                f"inner={len(tet_boundary_classification.inner_boundary_triangles)}"
            )
            
            # Step 3: Inflate boundary vertices outward by R - distance_to_part
            self.progress.emit("Inflating boundary vertices...")
            
            from core.tetrahedral_mesh import inflate_boundary_vertices
            
            # inner_boundary_threshold=None means auto-compute (2% of mesh size)
            # This ensures consistency with seed vertex identification in visualization
            inflated_result = inflate_boundary_vertices(
                result,
                self.part_mesh,
                r_value,
                use_gpu=CUDA_AVAILABLE,
                inner_boundary_threshold=None  # Auto-compute same as visualization
            )
            
            inflate_time = (time.time() - start_time) * 1000 - r_time - classify_time
            logger.info(f"Boundary inflation complete in {inflate_time:.0f}ms")
            
            # Emit inflated mesh for visualization
            self.mesh_inflated.emit(inflated_result)
            
            # Update result reference
            result = inflated_result
            
            # Emit classification for visualization
            self.boundary_classified.emit(tet_boundary_classification)
            
            # Step 4: Label boundary vertices from the tetrahedral boundary mesh classification
            # Use the INFLATED boundary mesh for vertex labeling (same face indices as original)
            self.progress.emit("Labeling boundary vertices...")
            
            inflated_boundary_mesh = result.boundary_mesh
            
            boundary_mask = result.boundary_vertices
            if boundary_mask is None:
                # Recompute boundary mask if not available
                from core.tetrahedral_mesh import identify_boundary_vertices
                boundary_mask = identify_boundary_vertices(result.vertices, inflated_boundary_mesh)
                result.boundary_vertices = boundary_mask
            
            result.boundary_labels = label_boundary_from_classification(
                result.vertices,
                boundary_mask,
                inflated_boundary_mesh,  # Use the inflated boundary mesh (same face indices)
                tet_boundary_classification,  # Classification done on original boundary mesh
                self.hull_mesh  # Not used, kept for compatibility
            )
            
            n_h1 = np.sum(result.boundary_labels == 1)
            n_h2 = np.sum(result.boundary_labels == 2)
            n_inner = np.sum(result.boundary_labels == -1)
            label_time = (time.time() - start_time) * 1000 - r_time - inflate_time - classify_time
            logger.info(f"Boundary labels computed in {label_time:.0f}ms: H1={n_h1}, H2={n_h2}, inner(seeds)={n_inner}")
            
            # Step 5: Compute edge weights on the inflated mesh
            self.progress.emit("Computing edge weights on inflated mesh...")
            
            from core.tetrahedral_mesh import compute_edge_weights_simple, compute_edge_boundary_labels
            
            edge_weights, edge_dist_to_part = compute_edge_weights_simple(
                result,
                self.part_mesh,
                epsilon=0.25,
                use_gpu=CUDA_AVAILABLE
            )
            
            result.edge_weights = edge_weights
            result.edge_dist_to_part = edge_dist_to_part
            result.weighted_edge_lengths = result.edge_lengths * edge_weights
            
            edge_weight_time = (time.time() - start_time) * 1000 - r_time - inflate_time - classify_time - label_time
            logger.info(f"Edge weights computed in {edge_weight_time:.0f}ms")
            
            # Step 6: Compute edge boundary labels for visualization
            self.progress.emit("Computing edge boundary labels...")
            
            result.edge_boundary_labels = compute_edge_boundary_labels(result)
            
            total_time = (time.time() - start_time) * 1000
            self.progress.emit(f"Complete in {total_time:.0f}ms. R={r_value:.4f}")
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error computing edge weights: {e}")
            self.error.emit(str(e))


class DijkstraWorker(QThread):
    """Background worker for running Dijkstra escape labeling on seed vertices."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # Updated TetrahedralMeshResult
    error = pyqtSignal(str)
    
    def __init__(self, tet_result):
        """
        Initialize Dijkstra worker.
        
        Args:
            tet_result: TetrahedralMeshResult with edge_weights and boundary_labels computed
        """
        super().__init__()
        self.tet_result = tet_result
    
    def run(self):
        try:
            from core.tetrahedral_mesh import run_dijkstra_escape_labeling
            import time
            
            logger.info("Running Dijkstra escape labeling...")
            start_time = time.time()
            
            self.progress.emit("Running Dijkstra from seed vertices to H1/H2...")
            
            (seed_escape_labels, seed_vertex_indices, seed_distances, 
             seed_escape_destinations, seed_escape_paths) = run_dijkstra_escape_labeling(
                self.tet_result,
                use_weighted_edges=True
            )
            
            # Store results in tet_result
            self.tet_result.seed_escape_labels = seed_escape_labels
            self.tet_result.seed_vertex_indices = seed_vertex_indices
            self.tet_result.seed_distances = seed_distances
            self.tet_result.seed_escape_destinations = seed_escape_destinations
            self.tet_result.seed_escape_paths = seed_escape_paths
            
            # Compute primary cut edges (edges where one vertex → H1, other → H2)
            self.progress.emit("Computing primary cut edges...")
            primary_cut_edges = self._compute_primary_cut_edges(
                self.tet_result.edges,
                seed_vertex_indices,
                seed_escape_labels
            )
            self.tet_result.primary_cut_edges = primary_cut_edges
            
            elapsed = (time.time() - start_time) * 1000
            
            n_h1 = np.sum(seed_escape_labels == 1)
            n_h2 = np.sum(seed_escape_labels == 2)
            n_unreached = np.sum(seed_escape_labels == 0)
            
            self.progress.emit(f"Complete in {elapsed:.0f}ms: {n_h1} seeds→H1, {n_h2} seeds→H2, {len(primary_cut_edges)} primary cuts")
            self.complete.emit(self.tet_result)
            
        except Exception as e:
            logger.exception(f"Error in Dijkstra: {e}")
            self.error.emit(str(e))
    
    def _compute_primary_cut_edges(
        self,
        edges: np.ndarray,
        interior_vertex_indices: np.ndarray,
        interior_escape_labels: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Compute primary cut edges - edges where one vertex escapes to H1 and other to H2.
        
        These are the edges the parting surface passes through.
        
        Args:
            edges: (E, 2) all tetrahedral mesh edges
            interior_vertex_indices: (I,) indices of interior vertices
            interior_escape_labels: (I,) escape labels (1=H1, 2=H2, 0=unreachable)
        
        Returns:
            List of (vi, vj) tuples for primary cut edges
        """
        # Build vertex -> escape label mapping
        n_verts = np.max(edges) + 1 if len(edges) > 0 else 0
        vertex_labels = np.full(n_verts, -1, dtype=np.int8)  # -1 = boundary or not interior
        
        for i, vert_idx in enumerate(interior_vertex_indices):
            vertex_labels[vert_idx] = interior_escape_labels[i]
        
        # Find edges where one endpoint → H1 and other → H2
        primary_cuts = []
        for v0, v1 in edges:
            l0 = vertex_labels[v0]
            l1 = vertex_labels[v1]
            
            # Primary cut: one goes to H1 (1), other goes to H2 (2)
            if (l0 == 1 and l1 == 2) or (l0 == 2 and l1 == 1):
                primary_cuts.append((int(v0), int(v1)))
        
        logger.info(f"Found {len(primary_cuts)} primary cut edges")
        return primary_cuts


class SecondaryCutsWorker(QThread):
    """Background worker for detecting secondary cutting edges."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # Updated TetrahedralMeshResult
    error = pyqtSignal(str)
    
    def __init__(self, tet_result, part_mesh, min_intersection_count: int = 20, use_gpu: bool = True):
        """
        Initialize secondary cuts worker.
        
        Args:
            tet_result: TetrahedralMeshResult with Dijkstra results (paths, destinations)
            part_mesh: The original part mesh to check intersection against
            min_intersection_count: Minimum number of segment-triangle intersections required (1-50)
            use_gpu: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.tet_result = tet_result
        self.part_mesh = part_mesh
        self.min_intersection_count = min_intersection_count
        self.use_gpu = use_gpu
    
    def run(self):
        try:
            from core.tetrahedral_mesh import find_secondary_cutting_edges, CUDA_AVAILABLE
            import time
            
            gpu_status = "GPU" if (self.use_gpu and CUDA_AVAILABLE) else "CPU"
            logger.info(f"Finding secondary cutting edges (min_intersections={self.min_intersection_count}, {gpu_status})...")
            start_time = time.time()
            
            self.progress.emit(f"Detecting secondary cuts ({gpu_status}, min_intersections={self.min_intersection_count})...")
            
            secondary_cuts = find_secondary_cutting_edges(
                self.tet_result,
                self.part_mesh,
                min_intersection_count=self.min_intersection_count,
                use_gpu=self.use_gpu
            )
            
            # Store results in tet_result
            self.tet_result.secondary_cut_edges = secondary_cuts
            
            elapsed = (time.time() - start_time) * 1000
            
            self.progress.emit(f"Complete in {elapsed:.0f}ms: {len(secondary_cuts)} secondary cuts found")
            self.complete.emit(self.tet_result)
            
        except Exception as e:
            logger.exception(f"Error in secondary cuts detection: {e}")
            self.error.emit(str(e))


@dataclass
class ComprehensiveSurfaceResult:
    """Result of comprehensive primary surface processing (generation + repair + gap fill + smoothing)."""
    # From extraction
    mesh: Optional[trimesh.Trimesh] = None
    num_vertices: int = 0
    num_faces: int = 0
    num_tets_contributing: int = 0
    num_tets_processed: int = 0
    
    # Mapping from surface vertex index to tet mesh edge index (for debugging)
    vertex_to_edge: Optional[np.ndarray] = None
    
    # From gap filling
    gaps_detected: int = 0
    gaps_filled: int = 0
    fill_faces_added: int = 0
    gap_fill_time_ms: float = 0.0
    # Indices of gap-fill faces (for visualization in different color)
    fill_face_indices: Optional[np.ndarray] = None
    
    # From smoothing
    boundary_vertices: int = 0
    interior_vertices: int = 0
    smooth_iterations: int = 0
    damping_factor: float = 0.5
    
    # Timing
    extraction_time_ms: float = 0.0
    repair_time_ms: float = 0.0
    smoothing_time_ms: float = 0.0
    total_time_ms: float = 0.0


class ComprehensivePrimarySurfaceWorker(QThread):
    """
    Background worker for primary surface processing.
    
    Performs:
    1. Extract surface using Marching Tetrahedra
    2. Clean surface (merge vertices, remove degenerates)
    3. Smooth surface with boundary re-projection
    """
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # ComprehensiveSurfaceResult
    error = pyqtSignal(str)
    
    def __init__(self, tet_result, part_mesh=None, hull_mesh=None,
                 smooth_iterations: int = 5, damping_factor: float = 0.5):
        """
        Initialize primary surface worker.
        
        Args:
            tet_result: TetrahedralMeshResult with primary cut edges
            part_mesh: Original part mesh for boundary re-projection
            hull_mesh: Hull mesh for boundary re-projection
            smooth_iterations: Number of smoothing iterations
            damping_factor: Smoothing damping factor
        """
        super().__init__()
        self.tet_result = tet_result
        self.part_mesh = part_mesh
        self.hull_mesh = hull_mesh
        self.smooth_iterations = smooth_iterations
        self.damping_factor = damping_factor
    
    def run(self):
        try:
            from core.parting_surface import (
                extract_parting_surface_from_tet_result,
                repair_parting_surface_with_part,
                close_parting_surface_gaps
            )
            from core.surface_propagation import smooth_membrane_with_boundary_reprojection
            from core.tetrahedral_mesh import prepare_parting_surface_data
            import time
            
            result = ComprehensiveSurfaceResult()
            result.smooth_iterations = self.smooth_iterations
            result.damping_factor = self.damping_factor
            
            total_start = time.time()
            
            # === Step 1: Prepare data structures if needed ===
            if self.tet_result.tet_edge_indices is None:
                self.progress.emit("Building edge index maps...")
                self.tet_result = prepare_parting_surface_data(self.tet_result)
            
            # === Step 2: Extract surface using Marching Tetrahedra ===
            self.progress.emit("Extracting primary surface with Marching Tetrahedra...")
            extraction_start = time.time()
            
            extraction_result = extract_parting_surface_from_tet_result(
                self.tet_result,
                use_original_vertices=True,
                prepare_data=False,
                cut_type='primary',
                extend_to_primary=False
            )
            
            result.extraction_time_ms = (time.time() - extraction_start) * 1000
            result.num_tets_processed = extraction_result.num_tets_processed
            result.num_tets_contributing = extraction_result.num_tets_contributing
            
            # Store vertex_to_edge mapping for debugging
            result.vertex_to_edge = extraction_result.vertex_to_edge
            
            if extraction_result.mesh is None or extraction_result.num_faces == 0:
                self.progress.emit("No primary surface generated")
                result.total_time_ms = (time.time() - total_start) * 1000
                self.complete.emit(result)
                return
            
            self.progress.emit(f"Extracted: {extraction_result.num_vertices:,} verts, {extraction_result.num_faces:,} faces")
            
            # === Step 3: Clean surface (merge vertices, remove degenerates) ===
            self.progress.emit("Cleaning surface...")
            repair_start = time.time()
            
            repaired_result = repair_parting_surface_with_part(
                extraction_result,
                self.part_mesh
            )
            
            result.repair_time_ms = (time.time() - repair_start) * 1000
            
            if repaired_result.mesh is None:
                self.progress.emit("Surface cleaning failed")
                result.mesh = extraction_result.mesh
                result.num_vertices = extraction_result.num_vertices
                result.num_faces = extraction_result.num_faces
                result.total_time_ms = (time.time() - total_start) * 1000
                self.complete.emit(result)
                return
            
            self.progress.emit(f"Cleaned: {repaired_result.num_vertices:,} verts, {repaired_result.num_faces:,} faces")
            
            # Current mesh after cleaning
            current_mesh = repaired_result.mesh
            
            # Track gap-fill info for smoothing exclusion
            excluded_vertices = None
            fill_face_start_idx = None
            
            # === Step 3.5: Close gaps between parting surface and part (BEFORE smoothing) ===
            # Gap-fill adds triangles to connect parting surface boundary to the part mesh.
            # These fill triangles will NOT be smoothed - only the original parting surface is.
            if self.part_mesh is not None:
                self.progress.emit("Closing gaps between parting surface and part...")
                gap_start = time.time()
                
                gap_result = close_parting_surface_gaps(
                    repaired_result,
                    self.part_mesh,
                    distance_threshold=None,  # Auto-compute
                    method='smart_curtain'
                )
                
                result.gap_fill_time_ms = (time.time() - gap_start) * 1000
                result.gaps_detected = gap_result.boundary_edges_near_part
                result.gaps_filled = gap_result.boundary_loops_found
                result.fill_faces_added = gap_result.fill_faces_added
                
                if gap_result.mesh is not None and gap_result.fill_faces_added > 0:
                    # Record where fill faces start (for visualization)
                    fill_face_start_idx = len(current_mesh.faces)
                    result.fill_face_indices = np.arange(
                        fill_face_start_idx,
                        fill_face_start_idx + gap_result.fill_faces_added
                    )
                    current_mesh = gap_result.mesh
                    
                    # Collect vertices to exclude from smoothing (fill vertices)
                    # part_constrained_vertices are the new projected vertices
                    # fill_boundary_vertices are the original boundary used in fill
                    excluded_verts_set = set()
                    if gap_result.part_constrained_vertices is not None:
                        excluded_verts_set.update(gap_result.part_constrained_vertices.tolist())
                    if gap_result.fill_boundary_vertices is not None:
                        excluded_verts_set.update(gap_result.fill_boundary_vertices.tolist())
                    if excluded_verts_set:
                        excluded_vertices = np.array(list(excluded_verts_set), dtype=np.int64)
                    
                    self.progress.emit(f"Gap fill: {gap_result.fill_faces_added} faces added, "
                                      f"{gap_result.boundary_loops_found} loops closed")
                else:
                    self.progress.emit("No gaps to fill (or already closed)")
            
            # === Step 4: Smooth surface with boundary re-projection (AFTER gap fill) ===
            # Smooth only the original parting surface, NOT the gap-fill triangles.
            if self.smooth_iterations > 0:
                self.progress.emit(f"Smoothing surface ({self.smooth_iterations} iterations, λ={self.damping_factor})...")
                
                # Log which meshes are available
                if self.hull_mesh is not None:
                    self.progress.emit(f"Hull mesh available: {len(self.hull_mesh.faces)} faces")
                else:
                    self.progress.emit("WARNING: No hull mesh for boundary re-projection!")
                
                smoothing_start = time.time()
                
                smoothing_result = smooth_membrane_with_boundary_reprojection(
                    current_mesh,
                    part_mesh=self.part_mesh,
                    hull_mesh=self.hull_mesh,
                    primary_mesh=None,  # No primary for primary surface smoothing
                    iterations=self.smooth_iterations,
                    damping_factor=self.damping_factor,
                    excluded_vertices=excluded_vertices  # Don't smooth gap-fill vertices
                )
                
                result.smoothing_time_ms = (time.time() - smoothing_start) * 1000
                result.boundary_vertices = smoothing_result.boundary_vertices
                result.interior_vertices = smoothing_result.interior_vertices
                
                if smoothing_result.mesh is not None:
                    current_mesh = smoothing_result.mesh
                else:
                    self.progress.emit("Smoothing returned no mesh, using unsmoothed")
            
            # === Step 5: Weld gap-fill triangles to smoothed parting surface ===
            # After smoothing, merge vertices at the boundary between gap-fill and parting surface
            if result.fill_face_indices is not None and len(result.fill_face_indices) > 0:
                self.progress.emit("Welding gap-fill triangles to parting surface...")
                try:
                    # Use trimesh's merge_vertices to weld close vertices
                    pre_weld_verts = len(current_mesh.vertices)
                    current_mesh.merge_vertices(merge_tex=False, merge_norm=False)
                    post_weld_verts = len(current_mesh.vertices)
                    merged_count = pre_weld_verts - post_weld_verts
                    
                    if merged_count > 0:
                        self.progress.emit(f"Welded {merged_count} vertices")
                        # Update fill_face_indices since faces may have been renumbered
                        # After merge, we can't reliably track fill faces, so clear the indices
                        # (They're already welded into the mesh now)
                        result.fill_face_indices = None
                    else:
                        self.progress.emit("No vertices needed welding")
                except Exception as e:
                    logger.warning(f"Weld step failed: {e}")
                    self.progress.emit(f"Weld warning: {e}")
            
            # Set final result
            result.mesh = current_mesh
            result.num_vertices = len(current_mesh.vertices)
            result.num_faces = len(current_mesh.faces)
            
            result.total_time_ms = (time.time() - total_start) * 1000
            
            # Build timing summary
            timing_parts = [f"extract: {result.extraction_time_ms:.0f}ms"]
            timing_parts.append(f"repair: {result.repair_time_ms:.0f}ms")
            if result.gap_fill_time_ms > 0:
                timing_parts.append(f"gap-fill: {result.gap_fill_time_ms:.0f}ms")
            if self.smooth_iterations > 0:
                timing_parts.append(f"smooth: {result.smoothing_time_ms:.0f}ms")
            
            self.progress.emit(
                f"Complete: {result.num_vertices:,} verts, {result.num_faces:,} faces "
                f"({', '.join(timing_parts)})"
            )
            
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error in comprehensive primary surface processing: {e}")
            self.error.emit(str(e))


@dataclass
class ComprehensiveSecondarySurfaceResult:
    """Result of comprehensive secondary surface processing (generation + propagation + smoothing)."""
    # Final mesh
    mesh: Optional[trimesh.Trimesh] = None
    num_vertices: int = 0
    num_faces: int = 0
    
    # From extraction
    extraction_vertices: int = 0
    extraction_faces: int = 0
    num_tets_contributing: int = 0
    
    # From propagation
    islands_removed: int = 0
    triangles_removed: int = 0
    boundary_vertices_extended: int = 0
    new_faces_added: int = 0
    
    # From smoothing
    boundary_vertices: int = 0
    interior_vertices: int = 0
    smooth_iterations: int = 0
    damping_factor: float = 0.5
    
    # Timing
    extraction_time_ms: float = 0.0
    propagation_time_ms: float = 0.0
    smoothing_time_ms: float = 0.0
    total_time_ms: float = 0.0


class ComprehensiveSecondarySurfaceWorker(QThread):
    """
    Background worker for comprehensive secondary surface processing.
    
    Combines all secondary surface tasks in one step:
    1. Extract secondary surface using Marching Tetrahedra
    2. Propagate - remove islands and extend boundaries to primary/part
    3. Smooth surface with boundary re-projection
    """
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # ComprehensiveSecondarySurfaceResult
    error = pyqtSignal(str)
    
    def __init__(self, tet_result, primary_mesh=None, part_mesh=None, hull_mesh=None,
                 min_island_triangles: int = 3, max_propagation_distance: float = 10.0,
                 smooth_iterations: int = 5, damping_factor: float = 0.5):
        """
        Initialize comprehensive secondary surface worker.
        
        Args:
            tet_result: TetrahedralMeshResult with secondary cut edges
            primary_mesh: Smoothed primary parting surface for boundary re-projection
            part_mesh: Original part mesh for bridging and boundary re-projection
            hull_mesh: Hull mesh for boundary re-projection
            min_island_triangles: Minimum triangles in island to keep
            max_propagation_distance: Maximum distance to propagate boundaries
            smooth_iterations: Number of smoothing iterations
            damping_factor: Smoothing damping factor
        """
        super().__init__()
        self.tet_result = tet_result
        self.primary_mesh = primary_mesh
        self.part_mesh = part_mesh
        self.hull_mesh = hull_mesh
        self.min_island_triangles = min_island_triangles
        self.max_propagation_distance = max_propagation_distance
        self.smooth_iterations = smooth_iterations
        self.damping_factor = damping_factor
    
    def run(self):
        try:
            from core.parting_surface import (
                extract_parting_surface_from_tet_result,
                repair_parting_surface_with_part
            )
            from core.surface_propagation import (
                propagate_secondary_surface,
                smooth_membrane_with_boundary_reprojection
            )
            from core.tetrahedral_mesh import prepare_parting_surface_data
            import time
            
            result = ComprehensiveSecondarySurfaceResult()
            result.smooth_iterations = self.smooth_iterations
            result.damping_factor = self.damping_factor
            
            total_start = time.time()
            
            # Check prerequisites
            if self.tet_result.secondary_cut_edges is None or len(self.tet_result.secondary_cut_edges) == 0:
                self.progress.emit("No secondary cut edges available")
                result.total_time_ms = (time.time() - total_start) * 1000
                self.complete.emit(result)
                return
            
            # === Step 1: Prepare data structures if needed ===
            if self.tet_result.tet_edge_indices is None:
                self.progress.emit("Building edge index maps...")
                self.tet_result = prepare_parting_surface_data(self.tet_result)
            
            # === Step 2: Extract secondary surface using Marching Tetrahedra ===
            self.progress.emit("Extracting secondary surface with Marching Tetrahedra...")
            extraction_start = time.time()
            
            extraction_result = extract_parting_surface_from_tet_result(
                self.tet_result,
                use_original_vertices=True,
                prepare_data=False,
                cut_type='secondary',
                extend_to_primary=True  # Include connections to primary
            )
            
            result.extraction_time_ms = (time.time() - extraction_start) * 1000
            result.num_tets_contributing = extraction_result.num_tets_contributing
            result.extraction_vertices = extraction_result.num_vertices
            result.extraction_faces = extraction_result.num_faces
            
            if extraction_result.mesh is None or extraction_result.num_faces == 0:
                self.progress.emit("No secondary surface generated")
                result.total_time_ms = (time.time() - total_start) * 1000
                self.complete.emit(result)
                return
            
            self.progress.emit(f"Extracted: {extraction_result.num_vertices:,} verts, {extraction_result.num_faces:,} faces")
            
            # === Step 3: Clean surface (merge vertices, remove degenerates) ===
            self.progress.emit("Cleaning secondary surface...")
            repair_result = repair_parting_surface_with_part(
                extraction_result,
                self.part_mesh
            )
            
            secondary_mesh = repair_result.mesh if repair_result.mesh is not None else extraction_result.mesh
            
            # === Step 4: Propagate - remove islands and extend boundaries ===
            self.progress.emit("Propagating surface (removing islands, extending boundaries)...")
            propagation_start = time.time()
            
            propagation_result = propagate_secondary_surface(
                secondary_mesh=secondary_mesh,
                primary_mesh=self.primary_mesh,
                part_mesh=self.part_mesh,
                min_island_triangles=self.min_island_triangles,
                max_propagation_distance=self.max_propagation_distance,
                propagation_step_size=0.5,
                smooth_iterations=0,  # Don't smooth here, do it in step 5
                smooth_factor=0.3
            )
            
            result.propagation_time_ms = (time.time() - propagation_start) * 1000
            result.islands_removed = propagation_result.islands_removed
            result.triangles_removed = propagation_result.triangles_removed
            result.boundary_vertices_extended = propagation_result.boundary_vertices_extended
            result.new_faces_added = propagation_result.new_faces_added
            
            if propagation_result.mesh is None or propagation_result.final_faces == 0:
                self.progress.emit("Propagation failed - no surface remaining")
                result.total_time_ms = (time.time() - total_start) * 1000
                self.complete.emit(result)
                return
            
            self.progress.emit(f"Propagated: {propagation_result.final_faces:,} faces "
                             f"({result.islands_removed} islands removed)")
            
            # === Step 5: Smooth surface with boundary re-projection ===
            if self.smooth_iterations > 0:
                self.progress.emit(f"Smoothing surface ({self.smooth_iterations} iterations, λ={self.damping_factor})...")
                smoothing_start = time.time()
                
                smoothing_result = smooth_membrane_with_boundary_reprojection(
                    propagation_result.mesh,
                    part_mesh=self.part_mesh,
                    hull_mesh=self.hull_mesh,
                    primary_mesh=self.primary_mesh,  # Re-project to primary surface too
                    iterations=self.smooth_iterations,
                    damping_factor=self.damping_factor
                )
                
                result.smoothing_time_ms = (time.time() - smoothing_start) * 1000
                result.boundary_vertices = smoothing_result.boundary_vertices
                result.interior_vertices = smoothing_result.interior_vertices
                
                if smoothing_result.mesh is not None:
                    result.mesh = smoothing_result.mesh
                    result.num_vertices = len(smoothing_result.mesh.vertices)
                    result.num_faces = len(smoothing_result.mesh.faces)
                else:
                    # Smoothing failed, use propagated mesh
                    result.mesh = propagation_result.mesh
                    result.num_vertices = propagation_result.final_vertices
                    result.num_faces = propagation_result.final_faces
            else:
                # No smoothing requested
                result.mesh = propagation_result.mesh
                result.num_vertices = propagation_result.final_vertices
                result.num_faces = propagation_result.final_faces
            
            result.total_time_ms = (time.time() - total_start) * 1000
            
            self.progress.emit(
                f"Complete: {result.num_vertices:,} verts, {result.num_faces:,} faces "
                f"(extract: {result.extraction_time_ms:.0f}ms, propagate: {result.propagation_time_ms:.0f}ms, "
                f"smooth: {result.smoothing_time_ms:.0f}ms)"
            )
            
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error in comprehensive secondary surface processing: {e}")
            self.error.emit(str(e))


class PartingSurfaceWorker(QThread):
    """Background worker for extracting the parting surface mesh using Marching Tetrahedra."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # PartingSurfaceResult
    error = pyqtSignal(str)
    
    def __init__(self, tet_result, use_original_vertices: bool = True, smooth_iterations: int = 0, 
                 repair_surface: bool = True, cut_type: str = 'primary', extend_to_primary: bool = True,
                 part_mesh=None):
        """
        Initialize parting surface worker.
        
        Args:
            tet_result: TetrahedralMeshResult with primary/secondary cut edges
            use_original_vertices: If True, use non-inflated vertex positions
            smooth_iterations: Number of Laplacian smoothing iterations (0 = none)
            repair_surface: If True, clean surface (merge vertices, remove degenerates)
            cut_type: Which cut edges to use: 'primary', 'secondary', or 'both'
            extend_to_primary: If True and cut_type='secondary', extend secondary surface
                              to connect with primary surface in shared tetrahedra
            part_mesh: Original part mesh (reserved for future use)
        """
        super().__init__()
        self.tet_result = tet_result
        self.use_original_vertices = use_original_vertices
        self.smooth_iterations = smooth_iterations
        self.repair_surface = repair_surface
        self.cut_type = cut_type
        self.extend_to_primary = extend_to_primary
        self.part_mesh = part_mesh
    
    def run(self):
        try:
            from core.parting_surface import (
                extract_parting_surface_from_tet_result,
                smooth_parting_surface,
                repair_parting_surface,
                repair_parting_surface_with_part
            )
            from core.tetrahedral_mesh import prepare_parting_surface_data
            import time
            
            start_time = time.time()
            
            # Step 1: Prepare data structures if needed
            if self.tet_result.tet_edge_indices is None:
                self.progress.emit("Building edge index maps...")
                self.tet_result = prepare_parting_surface_data(self.tet_result)
            
            # Step 2: Extract parting surface using Marching Tetrahedra
            cut_type_label = self.cut_type.capitalize()
            if self.cut_type == 'secondary' and self.extend_to_primary:
                self.progress.emit(f"Extracting {cut_type_label} surface (extended to primary)...")
            else:
                self.progress.emit(f"Extracting {cut_type_label} surface with Marching Tetrahedra...")
            
            result = extract_parting_surface_from_tet_result(
                self.tet_result,
                use_original_vertices=self.use_original_vertices,
                prepare_data=False,  # Already prepared above
                cut_type=self.cut_type,
                extend_to_primary=self.extend_to_primary
            )
            
            # Step 3: Clean surface (merge vertices, remove degenerates)
            if self.repair_surface and result.mesh is not None:
                self.progress.emit(f"Cleaning {cut_type_label} surface...")
                if self.part_mesh is not None:
                    result = repair_parting_surface_with_part(result, self.part_mesh)
                else:
                    result = repair_parting_surface(result, merge_vertices=True)
            
            # Step 4: Optional smoothing
            if self.smooth_iterations > 0 and result.mesh is not None:
                self.progress.emit(f"Smoothing surface ({self.smooth_iterations} iterations)...")
                result = smooth_parting_surface(result, iterations=self.smooth_iterations)
            
            elapsed = (time.time() - start_time) * 1000
            
            if result.num_faces > 0:
                self.progress.emit(f"{cut_type_label}: {result.num_vertices:,} verts, {result.num_faces:,} faces in {elapsed:.0f}ms")
            else:
                self.progress.emit(f"No {cut_type_label} surface generated in {elapsed:.0f}ms")
            
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error in parting surface extraction: {e}")
            self.error.emit(str(e))


class SurfacePropagationWorker(QThread):
    """Background worker for surface propagation and smoothing."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # PropagationResult
    error = pyqtSignal(str)
    
    def __init__(self, secondary_mesh, primary_mesh=None, part_mesh=None,
                 min_island_triangles: int = 3, max_distance: float = 10.0,
                 step_size: float = 0.5, smooth_iterations: int = 2):
        """
        Initialize surface propagation worker.
        
        Args:
            secondary_mesh: The secondary parting surface to propagate
            primary_mesh: The primary parting surface (target)
            part_mesh: The part mesh (target)
            min_island_triangles: Minimum triangles to keep an island
            max_distance: Maximum propagation distance
            step_size: Propagation step size
            smooth_iterations: Number of smoothing iterations
        """
        super().__init__()
        self.secondary_mesh = secondary_mesh
        self.primary_mesh = primary_mesh
        self.part_mesh = part_mesh
        self.min_island_triangles = min_island_triangles
        self.max_distance = max_distance
        self.step_size = step_size
        self.smooth_iterations = smooth_iterations
    
    def run(self):
        try:
            from core.surface_propagation import propagate_secondary_surface
            import time
            
            start_time = time.time()
            
            self.progress.emit("Removing isolated islands...")
            
            result = propagate_secondary_surface(
                self.secondary_mesh,
                primary_mesh=self.primary_mesh,
                part_mesh=self.part_mesh,
                min_island_triangles=self.min_island_triangles,
                max_propagation_distance=self.max_distance,
                propagation_step_size=self.step_size,
                smooth_iterations=self.smooth_iterations,
                smooth_factor=0.3
            )
            
            elapsed = (time.time() - start_time) * 1000
            
            if result.mesh is not None and result.final_faces > 0:
                self.progress.emit(
                    f"Complete: {result.final_faces:,} faces "
                    f"({result.islands_removed} islands removed, "
                    f"+{result.new_faces_added} new) in {elapsed:.0f}ms"
                )
            else:
                self.progress.emit(f"No surface remaining after propagation")
            
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error in surface propagation: {e}")
            self.error.emit(str(e))


class MembraneSmoothingWorker(QThread):
    """Background worker for boundary-aware membrane smoothing."""
    
    progress = pyqtSignal(str)
    complete = pyqtSignal(object)  # SmoothingResult
    error = pyqtSignal(str)
    
    def __init__(self, membrane_mesh, part_mesh=None, hull_mesh=None, primary_mesh=None,
                 iterations: int = 5, damping_factor: float = 0.5):
        """
        Initialize membrane smoothing worker.
        
        Args:
            membrane_mesh: The membrane surface to smooth (primary or secondary)
            part_mesh: The part/object mesh M for boundary re-projection
            hull_mesh: The hull mesh ∂H for boundary re-projection
            primary_mesh: The primary parting surface for secondary smoothing (re-projection)
            iterations: Number of alternating smooth/re-project iterations
            damping_factor: Smoothing damping factor (0.5 recommended)
        """
        super().__init__()
        self.membrane_mesh = membrane_mesh
        self.part_mesh = part_mesh
        self.hull_mesh = hull_mesh
        self.primary_mesh = primary_mesh
        self.iterations = iterations
        self.damping_factor = damping_factor
    
    def run(self):
        try:
            from core.surface_propagation import smooth_membrane_with_boundary_reprojection
            import time
            
            start_time = time.time()
            
            self.progress.emit(f"Smoothing membrane ({self.iterations} iterations, λ={self.damping_factor})...")
            
            result = smooth_membrane_with_boundary_reprojection(
                self.membrane_mesh,
                part_mesh=self.part_mesh,
                hull_mesh=self.hull_mesh,
                primary_mesh=self.primary_mesh,
                iterations=self.iterations,
                damping_factor=self.damping_factor
            )
            
            elapsed = (time.time() - start_time) * 1000
            
            if result.mesh is not None:
                self.progress.emit(
                    f"Complete: {result.final_vertices:,} vertices "
                    f"({result.boundary_vertices} boundary, {result.interior_vertices} interior) "
                    f"in {elapsed:.0f}ms"
                )
            else:
                self.progress.emit("Smoothing failed - no mesh output")
            
            self.complete.emit(result)
            
        except Exception as e:
            logger.exception(f"Error in membrane smoothing: {e}")
            self.error.emit(str(e))


# ============================================================================
# STYLED WIDGETS
# ============================================================================

class TitleBar(QFrame):
    """Custom title bar with logo, file info, and view controls matching the React frontend."""
    
    # Signals
    reset_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    redo_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    load_clicked = pyqtSignal()
    view_changed = pyqtSignal(str)  # Emits view name: 'front', 'back', 'left', 'right', 'top', 'bottom', 'iso'
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet(f"""
            TitleBar {{
                background-color: {Colors.WHITE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 8, 0)
        layout.setSpacing(0)
        
        # === LEFT SECTION: Logo + Reset/Undo/Redo ===
        left_section = QHBoxLayout()
        left_section.setSpacing(4)
        
        # App logo
        logo_label = QLabel("🔧")
        logo_label.setStyleSheet(f"""
            font-size: 18px;
            padding-right: 8px;
        """)
        left_section.addWidget(logo_label)
        
        # Separator after logo
        sep1 = self._create_separator()
        left_section.addWidget(sep1)
        
        # Reset button with icon and label
        self.reset_btn = self._create_labeled_button("↻", "Reset", "Reset All")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        left_section.addWidget(self.reset_btn)
        
        # Undo button (icon only)
        self.undo_btn = self._create_icon_button("↶", "Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_clicked.emit)
        self.undo_btn.setEnabled(False)
        left_section.addWidget(self.undo_btn)
        
        # Redo button (icon only)
        self.redo_btn = self._create_icon_button("↷", "Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo_clicked.emit)
        self.redo_btn.setEnabled(False)
        left_section.addWidget(self.redo_btn)
        
        # Separator before save/load
        sep2 = self._create_separator()
        left_section.addWidget(sep2)
        
        # Save button
        self.save_btn = self._create_icon_button("💾", "Save Session (Ctrl+S)")
        self.save_btn.clicked.connect(self.save_clicked.emit)
        left_section.addWidget(self.save_btn)
        
        # Load button
        self.load_btn = self._create_icon_button("📂", "Load Session (Ctrl+O)")
        self.load_btn.clicked.connect(self.load_clicked.emit)
        left_section.addWidget(self.load_btn)
        
        layout.addLayout(left_section)
        
        # === CENTER SECTION: File Info ===
        layout.addStretch()
        
        center_section = QHBoxLayout()
        center_section.setSpacing(12)
        
        # File icon and name
        self.file_icon = QLabel("⊙")
        self.file_icon.setStyleSheet(f"""
            font-size: 14px;
            color: {Colors.GRAY};
        """)
        self.file_icon.hide()
        center_section.addWidget(self.file_icon)
        
        self.file_label = QLabel("")
        self.file_label.setStyleSheet(f"""
            font-size: 13px;
            font-weight: 500;
            color: {Colors.DARK};
        """)
        self.file_label.hide()
        center_section.addWidget(self.file_label)
        
        # Triangle count badge
        self.triangle_label = QLabel("")
        self.triangle_label.setStyleSheet(f"""
            font-size: 12px;
            color: {Colors.GRAY};
        """)
        self.triangle_label.hide()
        center_section.addWidget(self.triangle_label)
        
        # File size
        self.size_label = QLabel("")
        self.size_label.setStyleSheet(f"""
            font-size: 12px;
            color: {Colors.GRAY};
        """)
        self.size_label.hide()
        center_section.addWidget(self.size_label)
        
        layout.addLayout(center_section)
        
        layout.addStretch()
        
        # === RIGHT SECTION: View Controls ===
        right_section = QHBoxLayout()
        right_section.setSpacing(2)
        
        # View buttons - using cube face symbols
        # Front, Back, Left, Right, Top, Bottom views (cube faces)
        views = [
            ("▣", "front", "Front View"),      # Front face highlighted
            ("▢", "back", "Back View"),        # Back face  
            ("◧", "left", "Left View"),        # Left face
            ("◨", "right", "Right View"),      # Right face
            ("⬒", "top", "Top View"),          # Top face
            ("⬓", "bottom", "Bottom View"),    # Bottom face
            ("●", "iso", "Reset View"),        # Center/reset dot
        ]
        
        for label, view_name, tooltip in views:
            btn = self._create_view_button(label, tooltip)
            btn.clicked.connect(lambda checked, v=view_name: self.view_changed.emit(v))
            right_section.addWidget(btn)
        
        layout.addLayout(right_section)
    
    def _create_separator(self) -> QFrame:
        """Create a vertical separator line."""
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sep.setStyleSheet(f"background-color: {Colors.BORDER}; margin: 0 8px;")
        return sep
    
    def _create_labeled_button(self, icon: str, label: str, tooltip: str) -> QPushButton:
        """Create a button with icon and label (like Import, Reset)."""
        btn = QPushButton(f" {icon}  {label}")
        btn.setFixedHeight(32)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                color: {Colors.GRAY_DARK};
                padding: 0 12px;
            }}
            QPushButton:hover {{
                background-color: {Colors.LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Colors.GRAY_LIGHT};
            }}
        """)
        return btn
    
    def _create_icon_button(self, icon: str, tooltip: str) -> QPushButton:
        """Create an icon-only button (like Undo, Redo)."""
        btn = QPushButton(icon)
        btn.setFixedSize(32, 32)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                color: {Colors.GRAY_DARK};
            }}
            QPushButton:hover {{
                background-color: {Colors.LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Colors.GRAY_LIGHT};
            }}
            QPushButton:disabled {{
                color: {Colors.GRAY_LIGHT};
            }}
        """)
        return btn
    
    def _create_view_button(self, label: str, tooltip: str) -> QPushButton:
        """Create a view control button (cube face icons)."""
        btn = QPushButton(label)
        btn.setFixedSize(28, 28)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                color: {Colors.GRAY};
            }}
            QPushButton:hover {{
                background-color: {Colors.LIGHT};
                color: {Colors.DARK};
            }}
            QPushButton:pressed {{
                background-color: {Colors.GRAY_LIGHT};
            }}
        """)
        return btn
    
    def set_file_info(self, filename: str, triangle_count: int, file_size_bytes: int = 0):
        """Update the file information display."""
        # Show file icon and name
        self.file_icon.show()
        self.file_label.setText(filename)
        self.file_label.show()
        
        # Format triangle count with comma separators
        tri_text = f"{triangle_count:,} tri"
        self.triangle_label.setText(tri_text)
        self.triangle_label.show()
        
        # Format file size
        if file_size_bytes > 0:
            if file_size_bytes >= 1024 * 1024:
                size_text = f"{file_size_bytes / (1024 * 1024):.2f} MB"
            elif file_size_bytes >= 1024:
                size_text = f"{file_size_bytes / 1024:.1f} KB"
            else:
                size_text = f"{file_size_bytes} B"
            self.size_label.setText(size_text)
            self.size_label.show()
        else:
            self.size_label.hide()
    
    def clear_file_info(self):
        """Clear the file information display."""
        self.file_icon.hide()
        self.file_label.hide()
        self.triangle_label.hide()
        self.size_label.hide()
    
    def set_undo_enabled(self, enabled: bool):
        """Enable/disable undo button."""
        self.undo_btn.setEnabled(enabled)
    
    def set_redo_enabled(self, enabled: bool):
        """Enable/disable redo button."""
        self.redo_btn.setEnabled(enabled)


class StepButton(QPushButton):
    """A step button for the sidebar."""
    
    def __init__(self, icon: str, title: str, parent=None):
        super().__init__(parent)
        self.icon_text = icon
        self.title = title
        self._status = 'available'
        self._is_active = False
        self.setFixedSize(80, 70)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()
    
    def set_active(self, active: bool):
        self._is_active = active
        self._update_style()
    
    def set_status(self, status: str):
        self._status = status
        self._update_style()
    
    def _update_style(self):
        bg_color = 'transparent'
        border_color = 'transparent'
        text_color = Colors.DARK
        
        if self._is_active:
            bg_color = 'rgba(0, 123, 255, 0.1)'
            border_color = Colors.PRIMARY
        elif self._status == 'completed':
            # Completed steps: subtle highlight, no checkmark
            bg_color = 'rgba(23, 198, 113, 0.05)'
            border_color = Colors.SUCCESS
        elif self._status == 'needs-recalc':
            bg_color = 'rgba(255, 180, 0, 0.05)'
            border_color = Colors.WARNING
        elif self._status == 'locked':
            # Locked steps: greyed out
            text_color = Colors.GRAY_LIGHT
        
        # Just show the icon, no status indicators
        self.setText(self.icon_text)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: none;
                border-left: 3px solid {border_color};
                color: {text_color};
                font-size: 20px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: rgba(0, 123, 255, 0.05);
            }}
            QPushButton:disabled {{
                color: {Colors.GRAY_LIGHT};
            }}
        """)
        self.setToolTip(self.title)
        
        # Disable button if locked
        self.setEnabled(self._status != 'locked')


class StatsBox(QFrame):
    """A statistics display box matching the frontend's statsBox style."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.setStyleSheet(f"""
            StatsBox {{
                background-color: {Colors.LIGHT};
                border: 1px solid {Colors.BORDER};
                border-radius: 10px;
            }}
        """)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 14, 14, 14)
        self._layout.setSpacing(4)
    
    def clear(self):
        while self._layout.count():
            child = self._layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def add_row(self, text: str, color: str = None):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"""
            color: {color or Colors.GRAY_DARK};
            font-size: 12px;
            background: transparent;
            border: none;
        """)
        self._layout.addWidget(label)
    
    def add_header(self, text: str, color: str = None):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"""
            color: {color or Colors.DARK};
            font-size: 12px;
            font-weight: 500;
            background: transparent;
            border: none;
            margin-bottom: 4px;
        """)
        self._layout.addWidget(label)


# ============================================================================
# UNIT SELECTION DIALOG
# ============================================================================

class UnitSelectionDialog(QDialog):
    """Dialog to select units for STL file import with clean radio button UI."""
    
    # Map of unit keys to (display name, scale factor)
    UNITS = {
        "mm": ("Millimeters", 1.0),
        "cm": ("Centimeters", 10.0),
        "in": ("Inches", 25.4),
    }
    
    def __init__(self, filename: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import STL")
        self.setModal(True)
        self.setFixedWidth(340)
        self._selected_unit = "mm"  # Default
        self._setup_ui(filename)
    
    def _setup_ui(self, filename: str):
        # Set dialog background
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.WHITE};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title_label = QLabel("Select Units")
        title_label.setStyleSheet(f"""
            color: {Colors.DARK};
            font-size: 18px;
            font-weight: 600;
        """)
        layout.addWidget(title_label)
        
        # Filename
        file_label = QLabel(f"📁 {filename}")
        file_label.setStyleSheet(f"""
            color: {Colors.GRAY_DARK};
            font-size: 12px;
            padding: 8px 12px;
            background-color: {Colors.LIGHT};
            border-radius: 4px;
        """)
        file_label.setWordWrap(True)
        layout.addWidget(file_label)
        
        # Info text
        info_label = QLabel("STL files don't contain unit information.\nSelect the units used when this file was created:")
        info_label.setStyleSheet(f"""
            color: {Colors.GRAY};
            font-size: 12px;
            line-height: 1.4;
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Unit selection buttons (radio-style)
        self.unit_buttons = {}
        self.button_group = QButtonGroup(self)
        
        units_layout = QHBoxLayout()
        units_layout.setSpacing(12)
        
        for unit_key, (unit_name, scale) in self.UNITS.items():
            btn = QPushButton(f"{unit_key}")
            btn.setCheckable(True)
            btn.setFixedSize(80, 50)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("unit_key", unit_key)
            btn.setProperty("scale", scale)
            
            # Style for unselected state
            btn.setStyleSheet(self._get_unit_button_style(False))
            
            btn.clicked.connect(lambda checked, k=unit_key: self._on_unit_selected(k))
            
            self.button_group.addButton(btn)
            self.unit_buttons[unit_key] = btn
            units_layout.addWidget(btn)
        
        units_layout.addStretch()
        layout.addLayout(units_layout)
        
        # Select default (mm)
        self.unit_buttons["mm"].setChecked(True)
        self._update_button_styles()
        
        # Scale info label
        self.scale_label = QLabel("Coordinates will be treated as millimeters")
        self.scale_label.setStyleSheet(f"""
            color: {Colors.GRAY};
            font-size: 11px;
            font-style: italic;
        """)
        layout.addWidget(self.scale_label)
        
        layout.addSpacing(8)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(90)
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.WHITE};
                color: {Colors.GRAY_DARK};
                border: 1px solid {Colors.GRAY_LIGHT};
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.LIGHT};
                border-color: {Colors.GRAY};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        import_btn = QPushButton("Import")
        import_btn.setFixedWidth(90)
        import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: {Colors.WHITE};
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
        """)
        import_btn.clicked.connect(self.accept)
        btn_layout.addWidget(import_btn)
        
        layout.addLayout(btn_layout)
    
    def _get_unit_button_style(self, selected: bool) -> str:
        """Get stylesheet for unit button based on selection state."""
        if selected:
            return f"""
                QPushButton {{
                    background-color: {Colors.PRIMARY};
                    color: {Colors.WHITE};
                    border: 2px solid {Colors.PRIMARY};
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 600;
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: {Colors.WHITE};
                    color: {Colors.DARK};
                    border: 2px solid {Colors.GRAY_LIGHT};
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    border-color: {Colors.PRIMARY};
                    background-color: rgba(0, 123, 255, 0.05);
                }}
            """
    
    def _on_unit_selected(self, unit_key: str):
        """Handle unit selection."""
        self._selected_unit = unit_key
        self._update_button_styles()
        
        # Update scale label
        unit_name, scale = self.UNITS[unit_key]
        if scale == 1.0:
            self.scale_label.setText("Coordinates will be treated as millimeters")
        else:
            self.scale_label.setText(f"Coordinates will be scaled from {unit_name.lower()} to mm (×{scale})")
    
    def _update_button_styles(self):
        """Update button styles based on selection."""
        for key, btn in self.unit_buttons.items():
            is_selected = (key == self._selected_unit)
            btn.setStyleSheet(self._get_unit_button_style(is_selected))
    
    @property
    def scale_factor(self) -> float:
        """Get the selected scale factor."""
        _, scale = self.UNITS[self._selected_unit]
        return scale


class DropZone(QFrame):
    """File drop zone widget matching the frontend's upload style."""
    
    file_dropped = pyqtSignal(str, float)  # file_path, scale_factor
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._is_dragging = False
        self._loaded_file = None
        self._setup_ui()
    
    def _setup_ui(self):
        self.setMinimumHeight(140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)
        
        self.icon_label = QLabel('📁')
        self.icon_label.setStyleSheet('font-size: 36px; background: transparent; border: none;')
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.title_label = QLabel('Upload STL File')
        self.title_label.setStyleSheet(f'''
            font-size: 14px;
            font-weight: 500;
            color: {Colors.DARK};
            background: transparent;
            border: none;
        ''')
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.hint_label = QLabel('Click or drag & drop')
        self.hint_label.setStyleSheet(f'''
            font-size: 12px;
            color: {Colors.GRAY};
            background: transparent;
            border: none;
        ''')
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.hint_label)
        
        self._update_style()
    
    def _update_style(self):
        if self._loaded_file:
            bg = 'rgba(23, 198, 113, 0.05)'
            border_color = Colors.SUCCESS
        elif self._is_dragging:
            bg = 'rgba(0, 123, 255, 0.1)'
            border_color = Colors.PRIMARY
        else:
            bg = Colors.LIGHT
            border_color = Colors.GRAY_LIGHT
        
        self.setStyleSheet(f"""
            DropZone {{
                background-color: {bg};
                border: 2px dashed {border_color};
                border-radius: 10px;
            }}
        """)
    
    def set_loaded(self, filename: str):
        self._loaded_file = filename
        self.icon_label.setText('✅')
        self.icon_label.setStyleSheet('font-size: 28px; background: transparent; border: none;')
        self.title_label.setText('Loaded:')
        self.title_label.setStyleSheet(f'''
            font-size: 12px;
            font-weight: 500;
            color: {Colors.SUCCESS};
            background: transparent;
            border: none;
        ''')
        self.hint_label.setText(filename)
        self.hint_label.setStyleSheet(f'''
            font-size: 12px;
            color: {Colors.GRAY_DARK};
            background: transparent;
            border: none;
        ''')
        self._update_style()
    
    def reset(self):
        self._loaded_file = None
        self.icon_label.setText('📁')
        self.icon_label.setStyleSheet('font-size: 36px; background: transparent; border: none;')
        self.title_label.setText('Upload STL File')
        self.title_label.setStyleSheet(f'''
            font-size: 14px;
            font-weight: 500;
            color: {Colors.DARK};
            background: transparent;
            border: none;
        ''')
        self.hint_label.setText('Click or drag & drop')
        self.hint_label.setStyleSheet(f'''
            font-size: 12px;
            color: {Colors.GRAY};
            background: transparent;
            border: none;
        ''')
        self._update_style()
    
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open STL File", "", "STL Files (*.stl *.STL);;All Files (*)"
        )
        if file_path:
            self._show_unit_dialog_and_emit(file_path)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile() and url.toLocalFile().lower().endswith('.stl'):
                event.acceptProposedAction()
                self._is_dragging = True
                self._update_style()
    
    def dragLeaveEvent(self, event):
        self._is_dragging = False
        self._update_style()
    
    def dropEvent(self, event: QDropEvent):
        self._is_dragging = False
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.stl'):
                    event.acceptProposedAction()
                    self._show_unit_dialog_and_emit(file_path)
        self._update_style()
    
    def _show_unit_dialog_and_emit(self, file_path: str):
        """Show unit selection dialog and emit file_dropped signal if accepted."""
        from pathlib import Path
        filename = Path(file_path).name
        dialog = UnitSelectionDialog(filename, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.file_dropped.emit(file_path, dialog.scale_factor)
        self._update_style()


class DisplayOptionsPanel(QFrame):
    """Floating display options panel for the viewer."""
    
    hide_mesh_changed = pyqtSignal(bool)
    hide_parting_changed = pyqtSignal(bool)  # Toggle visibility paint + arrows
    hide_hull_changed = pyqtSignal(bool)  # Toggle bounding hull visibility
    hide_cavity_changed = pyqtSignal(bool)  # Toggle mold cavity visibility
    hide_tet_mesh_changed = pyqtSignal(bool)  # Toggle tetrahedral mesh visibility
    hide_mold_halves_changed = pyqtSignal(bool)  # Toggle mold halves visibility (both)
    hide_outer_boundary_changed = pyqtSignal(bool)  # Toggle outer boundary (H1/H2/boundary zone)
    hide_inner_boundary_changed = pyqtSignal(bool)  # Toggle inner boundary (seed edges)
    
    # Edge weight visualization signals
    show_edge_weights_changed = pyqtSignal(bool)  # Toggle edge weight visualization
    show_interior_edges_changed = pyqtSignal(bool)  # Toggle interior edges
    show_boundary_edges_changed = pyqtSignal(bool)  # Toggle boundary edges
    show_original_tet_changed = pyqtSignal(bool)  # Toggle original tet mesh
    show_inflated_boundary_changed = pyqtSignal(bool)  # Toggle inflated boundary
    
    # Dijkstra result signal
    show_dijkstra_result_changed = pyqtSignal(bool)  # Toggle Dijkstra result visualization
    show_dijkstra_h1h2_changed = pyqtSignal(bool)  # Toggle H1/H2 edges visibility (hide to show only yellow primary cuts)
    
    # Secondary cuts signal
    show_secondary_cuts_changed = pyqtSignal(bool)  # Toggle secondary cuts visualization
    
    # Parting surface signals
    show_primary_parting_surface_changed = pyqtSignal(bool)  # Toggle primary parting surface (blue)
    show_secondary_parting_surface_changed = pyqtSignal(bool)  # Toggle secondary parting surface (red)
    
    # Must match MeshViewer.BACKGROUND_COLOR for rounded corners to blend
    VIEWER_BG = "#1a1a1a"  # Matches React frontend BACKGROUND_COLOR
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("displayOptionsPanel")
        self.setFixedWidth(180)
        
        # Set the palette background to match the 3D viewer background
        # This way, rounded corner "transparent" areas show the viewer color, not black
        from PyQt6.QtGui import QPalette, QColor
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(self.VIEWER_BG))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
        # Now we can use rounded corners - they'll blend with the viewer background
        self.setStyleSheet("""
            #displayOptionsPanel {
                background-color: #1e2227;
                border: 1px solid #3a3f47;
                border-radius: 8px;
            }
            #displayOptionsPanel QLabel {
                background: transparent;
                color: rgba(255, 255, 255, 0.9);
            }
            #displayOptionsPanel QCheckBox {
                background: transparent;
                font-size: 11px;
                color: rgba(255, 255, 255, 0.85);
                spacing: 6px;
                padding: 4px 0;
            }
            #displayOptionsPanel QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                background-color: rgba(255, 255, 255, 0.1);
            }
            #displayOptionsPanel QCheckBox::indicator:checked {
                background-color: #4a9eff;
                border-color: #4a9eff;
            }
            #displayOptionsPanel QCheckBox::indicator:hover {
                border-color: rgba(255, 255, 255, 0.5);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)
        
        # Header
        header = QLabel('Display Options')
        header.setStyleSheet("font-size: 11px; font-weight: bold; padding-bottom: 4px;")
        layout.addWidget(header)
        
        # Separator line
        separator = QFrame()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #3a3f47;")
        layout.addWidget(separator)
        
        # Hide mesh checkbox
        self.hide_mesh_cb = QCheckBox('Hide Original Mesh')
        self.hide_mesh_cb.stateChanged.connect(lambda s: self.hide_mesh_changed.emit(s == Qt.CheckState.Checked.value))
        layout.addWidget(self.hide_mesh_cb)
        
        # Hide parting analysis checkbox (visibility paint + arrows)
        self.hide_parting_cb = QCheckBox('Hide Parting Analysis')
        self.hide_parting_cb.setChecked(False)  # Not hidden by default (analysis is shown)
        self.hide_parting_cb.stateChanged.connect(lambda s: self.hide_parting_changed.emit(s == Qt.CheckState.Checked.value))
        self.hide_parting_cb.hide()  # Hidden until parting is computed
        layout.addWidget(self.hide_parting_cb)
        
        # Hide bounding hull checkbox
        self.hide_hull_cb = QCheckBox('Hide Bounding Hull')
        self.hide_hull_cb.setChecked(False)  # Not hidden by default (hull is shown)
        self.hide_hull_cb.stateChanged.connect(lambda s: self.hide_hull_changed.emit(s == Qt.CheckState.Checked.value))
        self.hide_hull_cb.hide()  # Hidden until hull is computed
        layout.addWidget(self.hide_hull_cb)
        
        # Hide mold cavity checkbox
        self.hide_cavity_cb = QCheckBox('Hide Mold Cavity')
        self.hide_cavity_cb.setChecked(False)  # Not hidden by default (cavity is shown)
        self.hide_cavity_cb.stateChanged.connect(lambda s: self.hide_cavity_changed.emit(s == Qt.CheckState.Checked.value))
        self.hide_cavity_cb.hide()  # Hidden until cavity is computed
        layout.addWidget(self.hide_cavity_cb)
        
        # Hide tetrahedral mesh checkbox
        self.hide_tet_mesh_cb = QCheckBox('Hide Tet Mesh')
        self.hide_tet_mesh_cb.setChecked(False)  # Not hidden by default (tet mesh is shown)
        self.hide_tet_mesh_cb.stateChanged.connect(lambda s: self.hide_tet_mesh_changed.emit(s == Qt.CheckState.Checked.value))
        self.hide_tet_mesh_cb.hide()  # Hidden until tet mesh is computed
        layout.addWidget(self.hide_tet_mesh_cb)
        
        # Hide mold boundary label (section header)
        self.mold_boundary_label = QLabel('Hide Mold Boundary')
        self.mold_boundary_label.setStyleSheet("font-size: 10px; font-weight: bold; padding-top: 4px; color: rgba(255, 255, 255, 0.7);")
        self.mold_boundary_label.hide()
        layout.addWidget(self.mold_boundary_label)
        
        # Hide outer boundary checkbox (H1/H2/boundary zone edges)
        self.hide_outer_boundary_cb = QCheckBox('  Outer Boundary')
        self.hide_outer_boundary_cb.setChecked(False)  # Not hidden by default
        self.hide_outer_boundary_cb.stateChanged.connect(lambda s: self.hide_outer_boundary_changed.emit(s == Qt.CheckState.Checked.value))
        self.hide_outer_boundary_cb.hide()
        layout.addWidget(self.hide_outer_boundary_cb)
        
        # Hide inner boundary checkbox (seed edges close to part)
        self.hide_inner_boundary_cb = QCheckBox('  Inner Boundary')
        self.hide_inner_boundary_cb.setChecked(False)  # Not hidden by default
        self.hide_inner_boundary_cb.stateChanged.connect(lambda s: self.hide_inner_boundary_changed.emit(s == Qt.CheckState.Checked.value))
        self.hide_inner_boundary_cb.hide()
        layout.addWidget(self.hide_inner_boundary_cb)
        
        # Separator for edge weight options
        self.edge_weight_separator = QFrame()
        self.edge_weight_separator.setFixedHeight(1)
        self.edge_weight_separator.setStyleSheet("background-color: #3a3f47;")
        self.edge_weight_separator.hide()
        layout.addWidget(self.edge_weight_separator)
        
        # Edge weight section label
        self.edge_weight_label = QLabel('Edge Weights')
        self.edge_weight_label.setStyleSheet("font-size: 10px; font-weight: bold; padding-top: 4px; color: rgba(255, 255, 255, 0.7);")
        self.edge_weight_label.hide()
        layout.addWidget(self.edge_weight_label)
        
        # Show edge weight visualization checkbox
        self.show_edge_weights_cb = QCheckBox('Edge Weight Vis.')
        self.show_edge_weights_cb.setChecked(True)
        self.show_edge_weights_cb.stateChanged.connect(lambda s: self.show_edge_weights_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_edge_weights_cb.hide()
        layout.addWidget(self.show_edge_weights_cb)
        
        # Show interior edges checkbox
        self.show_interior_edges_cb = QCheckBox('  Interior Edges')
        self.show_interior_edges_cb.setChecked(True)
        self.show_interior_edges_cb.stateChanged.connect(lambda s: self.show_interior_edges_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_interior_edges_cb.hide()
        layout.addWidget(self.show_interior_edges_cb)
        
        # Show boundary edges checkbox
        self.show_boundary_edges_cb = QCheckBox('  Boundary Edges')
        self.show_boundary_edges_cb.setChecked(True)
        self.show_boundary_edges_cb.stateChanged.connect(lambda s: self.show_boundary_edges_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_boundary_edges_cb.hide()
        layout.addWidget(self.show_boundary_edges_cb)
        
        # Show original tet mesh checkbox
        self.show_original_tet_cb = QCheckBox('Original Tet Mesh')
        self.show_original_tet_cb.setChecked(False)
        self.show_original_tet_cb.stateChanged.connect(lambda s: self.show_original_tet_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_original_tet_cb.hide()
        layout.addWidget(self.show_original_tet_cb)
        
        # Show inflated boundary checkbox
        self.show_inflated_boundary_cb = QCheckBox('Inflated Boundary')
        self.show_inflated_boundary_cb.setChecked(True)
        self.show_inflated_boundary_cb.stateChanged.connect(lambda s: self.show_inflated_boundary_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_inflated_boundary_cb.hide()
        layout.addWidget(self.show_inflated_boundary_cb)
        
        # Dijkstra result checkbox
        self.show_dijkstra_result_cb = QCheckBox('Dijkstra Result')
        self.show_dijkstra_result_cb.setChecked(True)
        self.show_dijkstra_result_cb.stateChanged.connect(lambda s: self.show_dijkstra_result_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_dijkstra_result_cb.hide()
        layout.addWidget(self.show_dijkstra_result_cb)
        
        # Show H1/H2 edges checkbox (uncheck to see only yellow primary cuts)
        self.show_dijkstra_h1h2_cb = QCheckBox('Show H1/H2 Edges')
        self.show_dijkstra_h1h2_cb.setChecked(True)
        self.show_dijkstra_h1h2_cb.stateChanged.connect(lambda s: self.show_dijkstra_h1h2_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_dijkstra_h1h2_cb.hide()
        layout.addWidget(self.show_dijkstra_h1h2_cb)
        
        # Secondary cuts checkbox
        self.show_secondary_cuts_cb = QCheckBox('Secondary Cuts (Red)')
        self.show_secondary_cuts_cb.setChecked(True)
        self.show_secondary_cuts_cb.stateChanged.connect(lambda s: self.show_secondary_cuts_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_secondary_cuts_cb.hide()
        layout.addWidget(self.show_secondary_cuts_cb)
        
        # Parting surface section separator
        self.parting_surface_separator = QFrame()
        self.parting_surface_separator.setFixedHeight(1)
        self.parting_surface_separator.setStyleSheet("background-color: #3a3f47;")
        self.parting_surface_separator.hide()
        layout.addWidget(self.parting_surface_separator)
        
        # Parting surface section label
        self.parting_surface_label = QLabel('Parting Surfaces')
        self.parting_surface_label.setStyleSheet("font-size: 10px; font-weight: bold; padding-top: 4px; color: rgba(255, 255, 255, 0.7);")
        self.parting_surface_label.hide()
        layout.addWidget(self.parting_surface_label)
        
        # Primary parting surface checkbox (blue)
        self.show_primary_parting_surface_cb = QCheckBox('Primary Surface (Blue)')
        self.show_primary_parting_surface_cb.setChecked(True)
        self.show_primary_parting_surface_cb.stateChanged.connect(lambda s: self.show_primary_parting_surface_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_primary_parting_surface_cb.hide()
        layout.addWidget(self.show_primary_parting_surface_cb)
        
        # Secondary parting surface checkbox (red)
        self.show_secondary_parting_surface_cb = QCheckBox('Secondary Surface (Red)')
        self.show_secondary_parting_surface_cb.setChecked(True)
        self.show_secondary_parting_surface_cb.stateChanged.connect(lambda s: self.show_secondary_parting_surface_changed.emit(s == Qt.CheckState.Checked.value))
        self.show_secondary_parting_surface_cb.hide()
        layout.addWidget(self.show_secondary_parting_surface_cb)
        
        self.adjustSize()
        self.hide()
    
    def show_parting_option(self, show: bool = True):
        """Show or hide the parting analysis checkbox."""
        if show:
            self.hide_parting_cb.show()
            self.hide_parting_cb.setChecked(False)  # Reset to showing analysis
        else:
            self.hide_parting_cb.hide()
        self.adjustSize()
    
    def show_hull_option(self, show: bool = True):
        """Show or hide the hull visibility checkbox."""
        if show:
            self.hide_hull_cb.show()
            self.hide_hull_cb.setChecked(False)  # Reset to showing hull
        else:
            self.hide_hull_cb.hide()
        self.adjustSize()
    
    def show_cavity_option(self, show: bool = True):
        """Show or hide the cavity visibility checkbox."""
        if show:
            self.hide_cavity_cb.show()
            self.hide_cavity_cb.setChecked(False)  # Reset to showing cavity
        else:
            self.hide_cavity_cb.hide()
        self.adjustSize()
    
    def show_tet_mesh_option(self, show: bool = True):
        """Show or hide the tetrahedral mesh visibility checkbox."""
        if show:
            self.hide_tet_mesh_cb.show()
            self.hide_tet_mesh_cb.setChecked(False)  # Reset to showing tet mesh
        else:
            self.hide_tet_mesh_cb.hide()
        self.adjustSize()
    
    def show_mold_halves_option(self, show: bool = True):
        """Show or hide the mold boundary visibility options."""
        if show:
            self.mold_boundary_label.show()
            self.hide_outer_boundary_cb.show()
            self.hide_inner_boundary_cb.show()
            # Reset to showing both boundaries
            self.hide_outer_boundary_cb.setChecked(False)
            self.hide_inner_boundary_cb.setChecked(False)
        else:
            self.mold_boundary_label.hide()
            self.hide_outer_boundary_cb.hide()
            self.hide_inner_boundary_cb.hide()
        self.adjustSize()
    
    def show_edge_weight_options(self, show: bool = True):
        """Show or hide the edge weight visibility options."""
        if show:
            self.edge_weight_separator.show()
            self.edge_weight_label.show()
            self.show_edge_weights_cb.show()
            self.show_interior_edges_cb.show()
            self.show_boundary_edges_cb.show()
            self.show_original_tet_cb.show()
            self.show_inflated_boundary_cb.show()
            # Reset to defaults
            self.show_edge_weights_cb.setChecked(True)
            self.show_interior_edges_cb.setChecked(True)
            self.show_boundary_edges_cb.setChecked(True)
            self.show_original_tet_cb.setChecked(False)
            self.show_inflated_boundary_cb.setChecked(True)
        else:
            self.edge_weight_separator.hide()
            self.edge_weight_label.hide()
            self.show_edge_weights_cb.hide()
            self.show_interior_edges_cb.hide()
            self.show_boundary_edges_cb.hide()
            self.show_original_tet_cb.hide()
            self.show_inflated_boundary_cb.hide()
        self.adjustSize()
    
    def show_dijkstra_result_option(self, show: bool = True):
        """Show or hide the Dijkstra result visibility checkboxes."""
        if show:
            self.show_dijkstra_result_cb.show()
            self.show_dijkstra_result_cb.setChecked(True)  # Reset to showing result
            self.show_dijkstra_h1h2_cb.show()
            self.show_dijkstra_h1h2_cb.setChecked(True)  # Reset to showing H1/H2 edges
        else:
            self.show_dijkstra_result_cb.hide()
            self.show_dijkstra_h1h2_cb.hide()
        self.adjustSize()
    
    def show_secondary_cuts_option(self, show: bool = True):
        """Show or hide the secondary cuts visibility checkbox."""
        if show:
            self.show_secondary_cuts_cb.show()
            self.show_secondary_cuts_cb.setChecked(True)  # Reset to showing cuts
        else:
            self.show_secondary_cuts_cb.hide()
        self.adjustSize()
    
    def show_parting_surface_options(self, show_primary: bool = False, show_secondary: bool = False):
        """Show or hide the parting surface visibility checkboxes."""
        show_any = show_primary or show_secondary
        
        if show_any:
            self.parting_surface_separator.show()
            self.parting_surface_label.show()
        else:
            self.parting_surface_separator.hide()
            self.parting_surface_label.hide()
        
        if show_primary:
            self.show_primary_parting_surface_cb.show()
            self.show_primary_parting_surface_cb.setChecked(True)  # Reset to showing
        else:
            self.show_primary_parting_surface_cb.hide()
        
        if show_secondary:
            self.show_secondary_parting_surface_cb.show()
            self.show_secondary_parting_surface_cb.setChecked(True)  # Reset to showing
        else:
            self.show_secondary_parting_surface_cb.hide()
        
        self.adjustSize()


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    """Main application window - matches React frontend layout."""
    
    def __init__(self):
        logger.debug("MainWindow.__init__ starting...")
        super().__init__()
        logger.debug("MainWindow: QMainWindow super().__init__ complete")
        self._current_mesh = None
        self._current_diagnostics: Optional[MeshDiagnostics] = None
        self._repair_result: Optional[MeshRepairResult] = None
        self._worker: Optional[MeshLoadWorker] = None
        self._active_step = Step.IMPORT
        self._loaded_filename: Optional[str] = None
        
        # Parting direction state
        self._parting_worker: Optional[PartingDirectionWorker] = None
        self._visibility_worker: Optional[VisibilityPaintWorker] = None
        self._parting_result: Optional[PartingDirectionResult] = None
        self._visibility_paint_data: Optional[VisibilityPaintData] = None
        
        # Inflated hull state
        self._hull_worker: Optional[HullWorker] = None
        self._hull_result: Optional[InflatedHullResult] = None
        
        # Mold cavity state
        self._cavity_worker: Optional[CavityWorker] = None
        self._cavity_result: Optional[MoldCavityResult] = None
        
        # Mold halves classification state
        self._mold_halves_worker = None
        self._mold_halves_result: Optional[MoldHalfClassificationResult] = None
        self._boundary_zone_threshold: float = 0.15  # Default 15%
        
        # Tetrahedral mesh state
        self._tet_worker = None
        self._tet_result: Optional[TetrahedralMeshResult] = None
        self._tet_edge_length_fac: float = 0.05  # Default: bbox/20
        self._tet_optimize: bool = True
        
        # Edge weights state
        self._edge_weights_worker = None
        
        # Dijkstra state
        self._dijkstra_worker = None
        
        # Secondary cuts state
        self._secondary_cuts_worker = None
        
        # Parting surface state
        self._parting_surface_worker = None
        self._parting_surface_result = None
        
        # Secondary parting surface state
        self._secondary_parting_surface_worker = None
        self._secondary_parting_surface_result = None
        
        # Surface propagation state
        self._surface_propagation_worker = None
        self._propagation_result = None
        
        # Primary smoothing state
        self._primary_smoothing_worker = None
        self._primary_smoothing_result = None
        
        # Secondary smoothing state
        self._secondary_smoothing_worker = None
        self._secondary_smoothing_result = None
        
        # Comprehensive secondary surface state (combined gen + propagate + smooth)
        self._secondary_surface_worker = None
        self._secondary_surface_result = None
        
        logger.debug("MainWindow: Calling _setup_window...")
        self._setup_window()
        logger.debug("MainWindow: _setup_window complete")
        
        logger.debug("MainWindow: Calling _setup_ui...")
        self._setup_ui()
        logger.debug("MainWindow: _setup_ui complete")
        
        # Setup keyboard shortcuts
        self._setup_shortcuts()
        logger.debug("MainWindow.__init__ complete")
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PyQt6.QtGui import QShortcut, QKeySequence
        
        # Ctrl+S - Save session
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._on_save_session)
        
        # Ctrl+O - Load session  
        load_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        load_shortcut.activated.connect(self._on_load_session)
        
        logger.debug("Keyboard shortcuts configured")
    
    def _setup_window(self):
        logger.debug("_setup_window: Setting window title...")
        self.setWindowTitle("VcMoldCreator - Mold Analysis Tool")
        logger.debug("_setup_window: Setting window size (min: 1200x800, initial: 1400x900)...")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        logger.debug("_setup_window: Setting stylesheet...")
        
        # Set light background for the app (matching Shards theme)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Colors.LIGHT};
            }}
        """)
    
    def _setup_ui(self):
        logger.debug("_setup_ui: Creating central widget...")
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        logger.debug("_setup_ui: Central widget set")
        
        # Main vertical layout (title bar on top, content below)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        
        # Title bar at the top
        logger.debug("_setup_ui: Creating title bar...")
        self.title_bar = TitleBar()
        self.title_bar.reset_clicked.connect(self._on_reset_all)
        self.title_bar.undo_clicked.connect(self._on_undo)
        self.title_bar.redo_clicked.connect(self._on_redo)
        self.title_bar.save_clicked.connect(self._on_save_session)
        self.title_bar.load_clicked.connect(self._on_load_session)
        self.title_bar.view_changed.connect(self._on_view_changed)
        outer_layout.addWidget(self.title_bar)
        logger.debug("_setup_ui: Title bar created and connected")
        
        # Content area (sidebar + context + viewer)
        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Column 1: Steps Sidebar (narrow)
        logger.debug("_setup_ui: Creating steps sidebar...")
        self.steps_sidebar = self._create_steps_sidebar()
        main_layout.addWidget(self.steps_sidebar)
        logger.debug("_setup_ui: Steps sidebar created")
        
        # Column 2: Context Panel (options)
        logger.debug("_setup_ui: Creating context panel...")
        self.context_panel = self._create_context_panel()
        main_layout.addWidget(self.context_panel)
        logger.debug("_setup_ui: Context panel created")
        
        # Column 3: 3D Viewer (main area)
        logger.debug("_setup_ui: Creating viewer container (this may take a moment)...")
        self.viewer_container = self._create_viewer_container()
        main_layout.addWidget(self.viewer_container, 1)
        logger.debug("_setup_ui: Viewer container created")
        
        outer_layout.addWidget(content_widget, 1)
        logger.debug("_setup_ui: All UI components created and added to layout")
    
    def _create_steps_sidebar(self) -> QWidget:
        """Create the narrow steps sidebar (left column)."""
        sidebar = QFrame()
        sidebar.setFixedWidth(80)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.WHITE};
                border-right: 1px solid {Colors.BORDER};
            }}
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Logo header
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.PRIMARY};
                border: none;
            }}
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo = QLabel('🔧')
        logo.setStyleSheet('font-size: 24px; color: white; background: transparent;')
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(logo)
        layout.addWidget(header)
        
        # Step buttons
        self.step_buttons: Dict[Step, StepButton] = {}
        for step_info in STEPS:
            btn = StepButton(step_info['icon'], step_info['title'])
            btn.clicked.connect(lambda checked, s=step_info['id']: self._on_step_clicked(s))
            self.step_buttons[step_info['id']] = btn
            layout.addWidget(btn)
        
        # Update initial state - import is active, other steps locked
        self.step_buttons[Step.IMPORT].set_active(True)
        if Step.PARTING in self.step_buttons:
            self.step_buttons[Step.PARTING].set_status('locked')
        if Step.HULL in self.step_buttons:
            self.step_buttons[Step.HULL].set_status('locked')
        
        layout.addStretch()
        
        # Help button at bottom
        help_btn = QPushButton('❓')
        help_btn.setFixedSize(80, 50)
        help_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                font-size: 18px;
                color: {Colors.GRAY_DARK};
            }}
            QPushButton:hover {{
                background-color: {Colors.LIGHT};
            }}
        """)
        help_btn.setToolTip('Controls: Left-click drag = Orbit, Right-click drag = Pan, Scroll = Zoom')
        layout.addWidget(help_btn)
        
        return sidebar
    
    def _create_context_panel(self) -> QWidget:
        """Create the context/options panel (middle column)."""
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.WHITE};
                border-right: 1px solid {Colors.BORDER};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel('Options')
        header.setStyleSheet(f"""
            padding: 20px 24px;
            font-size: 16px;
            font-weight: 500;
            color: {Colors.DARK};
            border-bottom: 1px solid {Colors.BORDER};
            letter-spacing: 0.5px;
        """)
        layout.addWidget(header)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 8px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)
        
        self.context_content = QWidget()
        self.context_content.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum
        )
        self.context_layout = QVBoxLayout(self.context_content)
        self.context_layout.setContentsMargins(16, 16, 16, 16)
        self.context_layout.setSpacing(12)
        self.context_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.context_content)
        
        layout.addWidget(scroll, 1)
        
        # Initialize with import step content
        self._update_context_panel()
        
        return panel
    
    def _create_viewer_container(self) -> QWidget:
        """Create the 3D viewer container (right column)."""
        # Outer wrapper for margin
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 12, 12, 12)
        
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.VIEWER_BG};
                border-radius: 10px;
            }}
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create a splitter to allow resizing between viewer and console
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #333333;
                height: 3px;
            }
            QSplitter::handle:hover {
                background-color: #4a9eff;
            }
        """)
        
        # 3D Viewer
        self.mesh_viewer = MeshViewer()
        splitter.addWidget(self.mesh_viewer)
        
        # Console log panel
        self.console_log = ConsoleLogPanel()
        self.console_log.setMinimumHeight(80)
        self.console_log.setMaximumHeight(300)
        splitter.addWidget(self.console_log)
        
        # Set initial sizes (viewer takes most space)
        splitter.setSizes([600, 120])
        splitter.setCollapsible(0, False)  # Don't collapse viewer
        splitter.setCollapsible(1, True)   # Console can be collapsed
        
        layout.addWidget(splitter)
        
        wrapper_layout.addWidget(container)
        
        # Display options panel (floating overlay) - parent to mesh_viewer so background is the 3D view
        self.display_options = DisplayOptionsPanel(self.mesh_viewer)
        self.display_options.move(16, 16)
        self.display_options.hide_mesh_changed.connect(self._on_hide_mesh_changed)
        self.display_options.hide_parting_changed.connect(self._on_hide_parting_changed)
        self.display_options.hide_hull_changed.connect(self._on_hide_hull_changed)
        self.display_options.hide_cavity_changed.connect(self._on_hide_cavity_changed)
        self.display_options.hide_tet_mesh_changed.connect(self._on_hide_tet_mesh_changed)
        self.display_options.hide_mold_halves_changed.connect(self._on_hide_mold_halves_changed)
        self.display_options.hide_outer_boundary_changed.connect(self._on_hide_outer_boundary_changed)
        self.display_options.hide_inner_boundary_changed.connect(self._on_hide_inner_boundary_changed)
        
        # Edge weight visibility signals
        self.display_options.show_edge_weights_changed.connect(
            lambda show: self.mesh_viewer.set_edge_weights_visible(show)
        )
        self.display_options.show_interior_edges_changed.connect(
            lambda show: self.mesh_viewer.set_tet_interior_visible(show)
        )
        self.display_options.show_boundary_edges_changed.connect(
            lambda show: self.mesh_viewer.set_tet_boundary_visible(show)
        )
        self.display_options.show_original_tet_changed.connect(
            lambda show: self.mesh_viewer.set_original_tet_visible(show)
        )
        self.display_options.show_inflated_boundary_changed.connect(
            lambda show: self.mesh_viewer.set_inflated_boundary_visible(show)
        )
        self.display_options.show_dijkstra_result_changed.connect(
            lambda show: self.mesh_viewer.set_dijkstra_visible(show)
        )
        self.display_options.show_dijkstra_h1h2_changed.connect(
            lambda show: self.mesh_viewer.set_dijkstra_show_h1h2(show)
        )
        self.display_options.show_secondary_cuts_changed.connect(
            lambda show: self.mesh_viewer.set_secondary_cuts_visible(show)
        )
        self.display_options.show_primary_parting_surface_changed.connect(
            lambda show: self.mesh_viewer.set_parting_surface_visible(show)
        )
        self.display_options.show_secondary_parting_surface_changed.connect(
            lambda show: self.mesh_viewer.set_secondary_parting_surface_visible(show)
        )
        
        return wrapper
    
    def _clear_context_layout(self):
        """Clear all widgets from context layout."""
        while self.context_layout.count():
            child = self.context_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _update_context_panel(self):
        """Update context panel based on active step."""
        self._clear_context_layout()
        
        step_info = next((s for s in STEPS if s['id'] == self._active_step), None)
        if not step_info:
            return
        
        # Title
        title = QLabel(f"{step_info['icon']} {step_info['title']}")
        title.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 500;
            color: {Colors.DARK};
        """)
        self.context_layout.addWidget(title)
        
        # Description
        desc = QLabel(step_info['description'])
        desc.setStyleSheet(f"""
            font-size: 13px;
            color: {Colors.GRAY};
            line-height: 1.5;
        """)
        desc.setWordWrap(True)
        self.context_layout.addWidget(desc)
        
        # Step-specific content
        if self._active_step == Step.IMPORT:
            self._setup_import_step()
        elif self._active_step == Step.PARTING:
            self._setup_parting_step()
        elif self._active_step == Step.HULL:
            self._setup_hull_step()
        elif self._active_step == Step.CAVITY:
            self._setup_cavity_step()
        elif self._active_step == Step.MOLD_HALVES:
            self._setup_mold_halves_step()
        elif self._active_step == Step.TETRAHEDRALIZE:
            self._setup_tetrahedralize_step()
        elif self._active_step == Step.EDGE_WEIGHTS:
            self._setup_edge_weights_step()
        elif self._active_step == Step.DIJKSTRA:
            self._setup_dijkstra_step()
        elif self._active_step == Step.SECONDARY_CUTS:
            self._setup_secondary_cuts_step()
        elif self._active_step == Step.PARTING_SURFACE:
            self._setup_parting_surface_step()
        elif self._active_step == Step.SECONDARY_SURFACE:
            self._setup_secondary_surface_step()
        
        self.context_layout.addStretch()
    
    def _setup_import_step(self):
        """Setup the import step UI."""
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        self.context_layout.addWidget(self.drop_zone)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY};
                border-radius: 4px;
            }}
        """)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        self.context_layout.addWidget(self.progress_bar)
        
        # Progress label
        self.progress_label = QLabel('Ready')
        self.progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.context_layout.addWidget(self.progress_label)
        
        # Mesh info
        self.mesh_stats = StatsBox()
        self.mesh_stats.hide()
        self.context_layout.addWidget(self.mesh_stats)
        
        # Restore state if file was already loaded
        if self._loaded_filename:
            self.drop_zone.set_loaded(Path(self._loaded_filename).name)
            self._update_mesh_stats()
    
    def _update_mesh_stats(self):
        """Update mesh statistics display."""
        if not self._current_diagnostics:
            self.mesh_stats.hide()
            return
        
        self.mesh_stats.clear()
        self.mesh_stats.show()
        
        diag = self._current_diagnostics
        
        # Status header
        if diag.is_manifold:
            self.mesh_stats.add_header('✅ Mesh Valid', Colors.SUCCESS)
        else:
            self.mesh_stats.add_header('⚠️ Mesh Issues', Colors.WARNING)
        
        self.mesh_stats.add_row(f'Vertices: {diag.vertex_count:,}')
        self.mesh_stats.add_row(f'Faces: {diag.face_count:,}')
        
        if diag.genus >= 0:
            self.mesh_stats.add_row(f'Genus: {diag.genus}')
        
        if diag.volume > 0:
            self.mesh_stats.add_row(f'Volume: {diag.volume:,.2f}')
        
        self.mesh_stats.add_row(f'Surface Area: {diag.surface_area:,.2f}')
        
        if self._repair_result and self._repair_result.was_repaired:
            # Show repair summary
            repair_info = self._repair_result.repair_method
            if len(repair_info) > 30:
                repair_info = repair_info[:27] + "..."
            self.mesh_stats.add_row(f'Repaired: {repair_info}', Colors.PRIMARY)
            
            # Show vertex/face changes if significant
            vert_change = self._repair_result.original_vertex_count - diag.vertex_count
            face_change = self._repair_result.original_face_count - diag.face_count
            if vert_change > 0 or face_change > 0:
                changes = []
                if vert_change > 0:
                    changes.append(f"-{vert_change} verts")
                if face_change > 0:
                    changes.append(f"-{face_change} faces")
                self.mesh_stats.add_row(f'  Cleaned: {", ".join(changes)}', Colors.SUCCESS)
    
    def _setup_parting_step(self):
        """Setup the parting direction step UI."""
        # Check if mesh is loaded
        if self._current_mesh is None:
            no_mesh_label = QLabel("⚠️ No mesh loaded. Please import an STL file first.")
            no_mesh_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_mesh_label.setWordWrap(True)
            self.context_layout.addWidget(no_mesh_label)
            return
        
        # Calculate button (if not computed yet)
        self.parting_calc_btn = QPushButton("🔀 Calculate Parting Directions")
        self.parting_calc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.parting_calc_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.parting_calc_btn.clicked.connect(self._on_calculate_parting)
        self.context_layout.addWidget(self.parting_calc_btn)
        
        # Progress bar for parting computation
        self.parting_progress = QProgressBar()
        self.parting_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY};
                border-radius: 4px;
            }}
        """)
        self.parting_progress.setTextVisible(False)
        self.parting_progress.hide()
        self.context_layout.addWidget(self.parting_progress)
        
        # Progress label
        self.parting_progress_label = QLabel("")
        self.parting_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.parting_progress_label.hide()
        self.context_layout.addWidget(self.parting_progress_label)
        
        # Parting direction stats (show if computed)
        self.parting_stats = StatsBox()
        self.parting_stats.hide()
        self.context_layout.addWidget(self.parting_stats)
        
        # Color legend for visibility paint (shown after calculation)
        self.color_legend = QLabel("""
            <b>Color Legend:</b><br>
            <span style='color: #00ff00;'>■</span> D1 visible<br>
            <span style='color: #ff6600;'>■</span> D2 visible<br>
            <span style='color: #ffff00;'>■</span> Both visible<br>
            <span style='color: #888888;'>■</span> Neither
        """)
        self.color_legend.setWordWrap(True)
        self.color_legend.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.color_legend.setStyleSheet(f"""
            color: {Colors.GRAY_DARK};
            font-size: 11px;
            padding: 10px 12px;
            background-color: {Colors.LIGHT};
            border-radius: 6px;
        """)
        self.color_legend.hide()
        self.context_layout.addWidget(self.color_legend)
        
        # Update UI with current state
        self._update_parting_step_ui()
    
    def _update_parting_step_ui(self):
        """Update parting step UI based on current state."""
        if not hasattr(self, 'parting_stats'):
            return
        
        if self._parting_result is not None:
            # Show stats
            self.parting_stats.clear()
            self.parting_stats.show()
            
            result = self._parting_result
            
            self.parting_stats.add_header('✅ Parting Directions Found', Colors.SUCCESS)
            
            # D1 direction
            d1_str = f"[{result.d1[0]:.3f}, {result.d1[1]:.3f}, {result.d1[2]:.3f}]"
            self.parting_stats.add_row(f'D1: {d1_str}', Colors.PARTING_D1)
            
            # D2 direction
            d2_str = f"[{result.d2[0]:.3f}, {result.d2[1]:.3f}, {result.d2[2]:.3f}]"
            self.parting_stats.add_row(f'D2: {d2_str}', Colors.PARTING_D2)
            
            # Coverage and angle
            self.parting_stats.add_row(f'Coverage: {result.total_coverage:.1f}%')
            self.parting_stats.add_row(f'Angle: {result.angle_degrees:.1f}°')
            self.parting_stats.add_row(f'Time: {result.computation_time_ms:.0f}ms')
            
            # Show color legend
            self.color_legend.show()
            
            # Update button text
            self.parting_calc_btn.setText("🔄 Recalculate")
        else:
            self.parting_stats.hide()
            self.color_legend.hide()
    
    def _on_calculate_parting(self):
        """Start parting direction computation."""
        if self._current_mesh is None:
            return
        
        # Disable button during computation
        self.parting_calc_btn.setEnabled(False)
        self.parting_progress.setRange(0, 128)  # k=128 directions
        self.parting_progress.setValue(0)
        self.parting_progress.show()
        self.parting_progress_label.setText("Starting computation...")
        self.parting_progress_label.show()
        
        # Clear previous results
        self._parting_result = None
        self._visibility_paint_data = None
        self.mesh_viewer.remove_parting_direction_arrows()
        self.mesh_viewer.remove_visibility_paint()
        
        # Hide display options parting toggle
        self.display_options.show_parting_option(False)
        
        # Start worker
        self._parting_worker = PartingDirectionWorker(self._current_mesh, k=128)
        self._parting_worker.progress.connect(self._on_parting_progress)
        self._parting_worker.progress_value.connect(self._on_parting_progress_value)
        self._parting_worker.complete.connect(self._on_parting_complete)
        self._parting_worker.error.connect(self._on_parting_error)
        self._parting_worker.finished.connect(self._on_parting_worker_finished)
        self._parting_worker.start()
    
    def _on_parting_progress(self, message: str):
        """Handle parting progress updates."""
        self.parting_progress_label.setText(message)
    
    def _on_parting_progress_value(self, current: int, total: int):
        """Handle parting progress value updates."""
        self.parting_progress.setRange(0, total)
        self.parting_progress.setValue(current)
    
    def _on_parting_complete(self, result: PartingDirectionResult):
        """Handle parting direction computation complete."""
        self._parting_result = result
        
        # Add arrows to viewer
        self.mesh_viewer.add_parting_direction_arrows(result.d1, result.d2)
        
        # Update UI
        self._update_parting_step_ui()
        
        # Update step status
        self.step_buttons[Step.PARTING].set_status('completed')
        
        # Unlock Hull step
        if Step.HULL in self.step_buttons:
            self.step_buttons[Step.HULL].set_status('available')
        
        logger.info(f"Parting directions computed: coverage={result.total_coverage:.1f}%, angle={result.angle_degrees:.1f}°")
        
        # Automatically compute and show visibility paint
        self._auto_compute_visibility_paint()
    
    def _on_parting_error(self, message: str):
        """Handle parting computation error."""
        self.parting_progress_label.setText(f"Error: {message}")
        self.parting_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
        QMessageBox.critical(self, "Error", f"Parting direction computation failed:\n{message}")
    
    def _on_parting_worker_finished(self):
        """Handle parting worker finished."""
        self.parting_calc_btn.setEnabled(True)
        self.parting_progress.hide()
        self._parting_worker = None
    
    def _auto_compute_visibility_paint(self):
        """Automatically compute and display visibility paint after parting directions found."""
        if self._parting_result is None or self._current_mesh is None:
            return
        
        # Show progress
        if hasattr(self, 'parting_progress_label'):
            self.parting_progress_label.setText("Computing visibility paint...")
            self.parting_progress_label.show()
        
        # Start worker for visibility paint (show both D1 and D2)
        self._visibility_worker = VisibilityPaintWorker(
            self._current_mesh,
            self._parting_result.d1,
            self._parting_result.d2,
            show_d1=True,
            show_d2=True
        )
        self._visibility_worker.progress.connect(self._on_visibility_progress)
        self._visibility_worker.complete.connect(self._on_visibility_complete)
        self._visibility_worker.error.connect(self._on_visibility_error)
        self._visibility_worker.start()
    
    def _on_visibility_progress(self, message: str):
        """Handle visibility progress."""
        if hasattr(self, 'parting_progress_label'):
            self.parting_progress_label.setText(message)
    
    def _on_visibility_complete(self, paint_data: VisibilityPaintData, face_colors: np.ndarray):
        """Handle visibility paint complete."""
        self._visibility_paint_data = paint_data
        
        # Apply paint to viewer
        self.mesh_viewer.apply_visibility_paint(face_colors)
        
        # Show the parting analysis toggle in display options (checkbox resets to unchecked = showing)
        self.display_options.show_parting_option(True)
        
        if hasattr(self, 'parting_progress_label'):
            self.parting_progress_label.setText("Visibility paint applied")
            self.parting_progress_label.setStyleSheet(f'color: {Colors.SUCCESS}; font-size: 12px;')
    
    def _on_visibility_error(self, message: str):
        """Handle visibility computation error."""
        if hasattr(self, 'parting_progress_label'):
            self.parting_progress_label.setText(f"Error: {message}")
            self.parting_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
    
    # =========================================================================
    # HULL STEP
    # =========================================================================
    
    def _setup_hull_step(self):
        """Setup the bounding hull step UI."""
        # Check if mesh is loaded
        if self._current_mesh is None:
            no_mesh_label = QLabel("⚠️ No mesh loaded. Please import an STL file first.")
            no_mesh_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_mesh_label.setWordWrap(True)
            self.context_layout.addWidget(no_mesh_label)
            return
        
        # Check if parting direction is computed
        if self._parting_result is None:
            no_parting_label = QLabel("⚠️ Please compute parting directions first.")
            no_parting_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_parting_label.setWordWrap(True)
            self.context_layout.addWidget(no_parting_label)
            return
        
        # Compute default offset (20% of bounding box diagonal)
        default_offset = compute_default_offset(self._current_mesh)
        
        # Offset input section
        offset_group = QGroupBox("Inflation Offset")
        offset_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        offset_layout = QVBoxLayout(offset_group)
        
        # Offset input row
        input_row = QHBoxLayout()
        
        offset_label = QLabel("Offset:")
        offset_label.setStyleSheet(f'color: {Colors.DARK}; font-size: 12px;')
        input_row.addWidget(offset_label)
        
        self.hull_offset_spinbox = QDoubleSpinBox()
        self.hull_offset_spinbox.setMinimum(0.01)
        self.hull_offset_spinbox.setMaximum(1000.0)
        self.hull_offset_spinbox.setDecimals(2)
        self.hull_offset_spinbox.setSingleStep(0.1)
        self.hull_offset_spinbox.setValue(default_offset)
        self.hull_offset_spinbox.setSuffix(" units")
        self.hull_offset_spinbox.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {Colors.WHITE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
                color: {Colors.DARK};
                min-width: 100px;
            }}
            QDoubleSpinBox:focus {{
                border-color: {Colors.PRIMARY};
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 20px;
                border: none;
                background: {Colors.LIGHT};
            }}
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background: {Colors.GRAY_LIGHT};
            }}
        """)
        input_row.addWidget(self.hull_offset_spinbox)
        
        input_row.addStretch()
        
        offset_layout.addLayout(input_row)
        
        # Info label showing default calculation
        info_label = QLabel(f"Default: 10% of bounding box diagonal ({default_offset:.2f})")
        info_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 11px; font-style: italic;')
        offset_layout.addWidget(info_label)
        
        self.context_layout.addWidget(offset_group)
        
        # Calculate button
        self.hull_calc_btn = QPushButton("📦 Generate Bounding Hull")
        self.hull_calc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hull_calc_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.hull_calc_btn.clicked.connect(self._on_calculate_hull)
        self.context_layout.addWidget(self.hull_calc_btn)
        
        # Progress bar
        self.hull_progress = QProgressBar()
        self.hull_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY};
                border-radius: 4px;
            }}
        """)
        self.hull_progress.setTextVisible(False)
        self.hull_progress.setRange(0, 0)  # Indeterminate
        self.hull_progress.hide()
        self.context_layout.addWidget(self.hull_progress)
        
        # Progress label
        self.hull_progress_label = QLabel("")
        self.hull_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.hull_progress_label.hide()
        self.context_layout.addWidget(self.hull_progress_label)
        
        # Hull stats (show if computed)
        self.hull_stats = StatsBox()
        self.hull_stats.hide()
        self.context_layout.addWidget(self.hull_stats)
        
        # Update UI with current state
        self._update_hull_step_ui()
    

    
    def _update_hull_step_ui(self):
        """Update hull step UI based on current state."""
        if not hasattr(self, 'hull_stats'):
            return
        
        if self._hull_result is not None:
            # Show stats
            self.hull_stats.clear()
            self.hull_stats.show()
            
            result = self._hull_result
            
            self.hull_stats.add_header('✅ Hull Generated', Colors.SUCCESS)
            self.hull_stats.add_row(f'Vertices: {result.vertex_count:,}')
            self.hull_stats.add_row(f'Faces: {result.face_count:,}')
            self.hull_stats.add_row(f'Offset: {result.offset:.2f} units')
            
            # Manifold validation
            if result.manifold_validation.is_closed:
                self.hull_stats.add_row('Manifold: ✅ Closed', Colors.SUCCESS)
            else:
                self.hull_stats.add_row('Manifold: ⚠️ Open', Colors.WARNING)
            
            # Update button text
            self.hull_calc_btn.setText("🔄 Regenerate Hull")
        else:
            self.hull_stats.hide()
    
    def _on_calculate_hull(self):
        """Start hull computation."""
        if self._current_mesh is None:
            return
        
        # Get offset from spinbox
        offset = self.hull_offset_spinbox.value()
        
        # Disable button during computation
        self.hull_calc_btn.setEnabled(False)
        self.hull_progress.show()
        self.hull_progress_label.setText("Computing inflated hull...")
        self.hull_progress_label.show()
        
        # Clear previous hull
        self.mesh_viewer.clear_hull()
        self._hull_result = None
        
        # Start worker
        self._hull_worker = HullWorker(self._current_mesh, offset=offset, subdivisions=2)
        self._hull_worker.progress.connect(self._on_hull_progress)
        self._hull_worker.complete.connect(self._on_hull_complete)
        self._hull_worker.error.connect(self._on_hull_error)
        self._hull_worker.finished.connect(self._on_hull_worker_finished)
        self._hull_worker.start()
    
    def _on_hull_progress(self, message: str):
        """Handle hull progress updates."""
        # Check if widget still exists (user may have navigated away)
        if hasattr(self, 'hull_progress_label') and self.hull_progress_label is not None:
            try:
                self.hull_progress_label.setText(message)
            except RuntimeError:
                pass  # Widget was deleted
    
    def _on_hull_complete(self, result: InflatedHullResult):
        """Handle hull computation complete."""
        self._hull_result = result
        
        # Add hull to viewer (no original hull wireframe)
        self.mesh_viewer.set_hull_mesh(result.mesh, None)
        
        # Update UI only if widgets still exist
        if hasattr(self, 'hull_stats') and self.hull_stats is not None:
            try:
                self._update_hull_step_ui()
            except RuntimeError:
                pass  # Widget was deleted
        
        # Show hull option in display options panel
        self.display_options.show_hull_option(True)
        
        # Update step status
        self.step_buttons[Step.HULL].set_status('completed')
        
        # Unlock the CAVITY step now that hull is computed
        if Step.CAVITY in self.step_buttons:
            self.step_buttons[Step.CAVITY].set_status('available')
        
        logger.info(f"Hull generated: {result.vertex_count} vertices, {result.face_count} faces")
    
    def _on_hull_error(self, message: str):
        """Handle hull computation error."""
        # Check if widget still exists
        if hasattr(self, 'hull_progress_label') and self.hull_progress_label is not None:
            try:
                self.hull_progress_label.setText(f"Error: {message}")
                self.hull_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            except RuntimeError:
                pass  # Widget was deleted
        QMessageBox.critical(self, "Error", f"Hull generation failed:\n{message}")
    
    def _on_hull_worker_finished(self):
        """Handle hull worker finished."""
        # Check if widgets still exist
        if hasattr(self, 'hull_calc_btn') and self.hull_calc_btn is not None:
            try:
                self.hull_calc_btn.setEnabled(True)
            except RuntimeError:
                pass  # Widget was deleted
        if hasattr(self, 'hull_progress') and self.hull_progress is not None:
            try:
                self.hull_progress.hide()
            except RuntimeError:
                pass  # Widget was deleted
        self._hull_worker = None
    
    def _on_step_clicked(self, step: Step):
        """Handle step button click."""
        for s, btn in self.step_buttons.items():
            btn.set_active(s == step)
        
        self._active_step = step
        self._update_context_panel()
    
    def _on_reset_all(self):
        """Handle reset all button click - reset to initial state."""
        # Clear mesh viewer
        self.mesh_viewer.clear()
        self.mesh_viewer.clear_hull()
        self.mesh_viewer.clear_cavity()
        
        # Clear parting results
        self._parting_result = None
        self._visibility_paint_data = None
        
        # Clear hull results
        self._hull_result = None
        
        # Clear cavity results
        self._cavity_result = None
        
        # Clear mold halves results
        self._mold_halves_result = None
        
        # Clear tetrahedral mesh results
        self._tet_mesh_result = None
        
        # Reset current mesh
        self._current_mesh = None
        self._current_diagnostics = None
        self._repair_result = None
        self._loaded_filename = None
        
        # Reset step buttons - import available, others locked
        self.step_buttons[Step.IMPORT].set_status('available')
        self.step_buttons[Step.IMPORT].set_active(True)
        if Step.PARTING in self.step_buttons:
            self.step_buttons[Step.PARTING].set_status('locked')
        if Step.HULL in self.step_buttons:
            self.step_buttons[Step.HULL].set_status('locked')
        if Step.CAVITY in self.step_buttons:
            self.step_buttons[Step.CAVITY].set_status('locked')
        if Step.MOLD_HALVES in self.step_buttons:
            self.step_buttons[Step.MOLD_HALVES].set_status('locked')
        if Step.TETRAHEDRALIZE in self.step_buttons:
            self.step_buttons[Step.TETRAHEDRALIZE].set_status('locked')
        if Step.EDGE_WEIGHTS in self.step_buttons:
            self.step_buttons[Step.EDGE_WEIGHTS].set_status('locked')
        self._active_step = Step.IMPORT
        
        # Reset title bar
        self.title_bar.clear_file_info()
        
        # Reset display options
        self.display_options.hide()
        self.display_options.show_parting_option(False)
        self.display_options.show_hull_option(False)
        self.display_options.show_cavity_option(False)
        self.display_options.show_tet_mesh_option(False)
        self.display_options.show_mold_halves_option(False)
        
        # Update context panel
        self._update_context_panel()
        
        logger.info("Application reset to initial state")
    
    def _on_undo(self):
        """Handle undo button click."""
        # TODO: Implement undo stack
        logger.info("Undo clicked (not yet implemented)")
    
    def _on_redo(self):
        """Handle redo button click."""
        # TODO: Implement redo stack
        logger.info("Redo clicked (not yet implemented)")
    
    # =========================================================================
    # SESSION SAVE/LOAD
    # =========================================================================
    
    def _on_save_session(self):
        """Handle save session button click - save all computed data."""
        from PyQt6.QtWidgets import QFileDialog
        import pickle
        import gzip
        import os
        
        # Get save file path
        default_name = "mold_session.vcm"
        if self._loaded_filename:
            base_name = os.path.splitext(os.path.basename(self._loaded_filename))[0]
            default_name = f"{base_name}_session.vcm"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            default_name,
            "VcMoldCreator Session (*.vcm);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Collect all state data
            session_data = self._collect_session_data()
            
            # Save with gzip compression
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(session_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Session saved to: {file_path}")
            QMessageBox.information(
                self,
                "Session Saved",
                f"Session saved successfully to:\n{file_path}"
            )
            
        except Exception as e:
            logger.exception(f"Failed to save session: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save session:\n{str(e)}"
            )
    
    def _on_load_session(self):
        """Handle load session button click - restore all computed data."""
        from PyQt6.QtWidgets import QFileDialog
        import pickle
        import gzip
        
        # Get load file path
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            "",
            "VcMoldCreator Session (*.vcm);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load session data
            with gzip.open(file_path, 'rb') as f:
                session_data = pickle.load(f)
            
            # Restore state
            self._restore_session_data(session_data)
            
            logger.info(f"Session loaded from: {file_path}")
            QMessageBox.information(
                self,
                "Session Loaded",
                f"Session loaded successfully from:\n{file_path}"
            )
            
        except Exception as e:
            logger.exception(f"Failed to load session: {e}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load session:\n{str(e)}"
            )
    
    def _collect_session_data(self) -> dict:
        """Collect all session data for saving."""
        import trimesh
        
        def mesh_to_dict(mesh):
            """Convert trimesh to serializable dict."""
            if mesh is None:
                return None
            return {
                'vertices': mesh.vertices.copy(),
                'faces': mesh.faces.copy(),
            }
        
        def result_to_dict(result, mesh_attrs=None):
            """Convert a result dataclass to a serializable dict."""
            if result is None:
                return None
            data = {}
            for key, value in result.__dict__.items():
                if isinstance(value, trimesh.Trimesh):
                    data[key] = mesh_to_dict(value)
                elif isinstance(value, np.ndarray):
                    data[key] = value.copy()
                elif isinstance(value, (int, float, str, bool, type(None), list, tuple)):
                    data[key] = value
                elif isinstance(value, dict):
                    # Try to serialize dict items
                    data[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            data[key][k] = v.copy()
                        elif isinstance(v, (int, float, str, bool, type(None), list, tuple)):
                            data[key][k] = v
                        else:
                            data[key][k] = str(v)  # Fallback
                else:
                    # Skip non-serializable objects
                    pass
            return data
        
        session = {
            'version': 1,
            'filename': self._loaded_filename,
            'active_step': self._active_step.value if self._active_step else None,
            
            # Step completion status
            'step_status': {
                step.value: btn._status for step, btn in self.step_buttons.items()
            },
            
            # Original mesh
            'current_mesh': mesh_to_dict(self._current_mesh),
            
            # Parting direction
            'parting_result': result_to_dict(self._parting_result),
            'visibility_paint_data': result_to_dict(self._visibility_paint_data),
            
            # Hull
            'hull_result': result_to_dict(self._hull_result),
            
            # Cavity
            'cavity_result': result_to_dict(self._cavity_result),
            
            # Mold halves
            'mold_halves_result': result_to_dict(self._mold_halves_result),
            'boundary_zone_threshold': self._boundary_zone_threshold,
            
            # Tetrahedral mesh
            'tet_result': result_to_dict(self._tet_result),
            'tet_edge_length_fac': self._tet_edge_length_fac,
            'tet_optimize': self._tet_optimize,
            
            # Parting surface
            'parting_surface_result': result_to_dict(self._parting_surface_result),
            'primary_smoothing_result': result_to_dict(self._primary_smoothing_result),
            
            # Secondary surface
            'secondary_parting_surface_result': result_to_dict(self._secondary_parting_surface_result),
            'propagation_result': result_to_dict(self._propagation_result),
            'secondary_smoothing_result': result_to_dict(self._secondary_smoothing_result),
            'secondary_surface_result': result_to_dict(self._secondary_surface_result),
        }
        
        return session
    
    def _restore_session_data(self, session: dict):
        """Restore session data from loaded dict."""
        import trimesh
        import os
        from dataclasses import fields, is_dataclass
        
        def dict_to_mesh(data):
            """Convert dict back to trimesh."""
            if data is None:
                return None
            return trimesh.Trimesh(
                vertices=data['vertices'],
                faces=data['faces']
            )
        
        def dict_to_result(data, result_class):
            """Convert dict back to result dataclass."""
            if data is None:
                return None
            
            # Process data - convert mesh dicts back to trimesh objects
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, dict) and 'vertices' in value and 'faces' in value:
                    # This is a mesh
                    processed_data[key] = dict_to_mesh(value)
                else:
                    processed_data[key] = value
            
            # Try to create instance using **kwargs if it's a dataclass
            if is_dataclass(result_class):
                # Get the field names for this dataclass
                field_names = {f.name for f in fields(result_class)}
                # Filter to only include valid fields
                valid_data = {k: v for k, v in processed_data.items() if k in field_names}
                try:
                    return result_class(**valid_data)
                except TypeError:
                    # If that fails, try creating with defaults and setting attrs
                    pass
            
            # Fallback: create empty instance and set attributes
            try:
                result = result_class()
                for key, value in processed_data.items():
                    if hasattr(result, key):
                        setattr(result, key, value)
                return result
            except TypeError:
                # Can't create instance - return a simple object with attrs
                result = type('RestoredResult', (), processed_data)()
                return result
        
        # First reset everything
        self._on_reset_all()
        
        # Restore filename
        self._loaded_filename = session.get('filename')
        
        # Restore meshes and results
        if session.get('current_mesh'):
            self._current_mesh = dict_to_mesh(session['current_mesh'])
            # Update viewer
            if self._current_mesh is not None:
                self.mesh_viewer.set_mesh(self._current_mesh)
                # Update title bar
                if self._loaded_filename:
                    self.title_bar.set_file_info(
                        os.path.basename(self._loaded_filename),
                        len(self._current_mesh.faces),
                        0  # Size not stored
                    )
        
        # Restore parting result
        if session.get('parting_result'):
            from core.parting_direction import PartingDirectionResult
            self._parting_result = dict_to_result(session['parting_result'], PartingDirectionResult)
        
        if session.get('visibility_paint_data'):
            from core.parting_direction import VisibilityPaintData, get_face_colors
            self._visibility_paint_data = dict_to_result(session['visibility_paint_data'], VisibilityPaintData)
            if self._visibility_paint_data and hasattr(self._visibility_paint_data, 'triangle_classes') and self._visibility_paint_data.triangle_classes is not None:
                # Recompute face colors from triangle_classes
                face_colors = get_face_colors(self._visibility_paint_data)
                self.mesh_viewer.apply_visibility_paint(face_colors)
                self.display_options.show_parting_option(True)
        
        # Restore hull result
        if session.get('hull_result'):
            from core.inflated_hull import InflatedHullResult
            self._hull_result = dict_to_result(session['hull_result'], InflatedHullResult)
            if self._hull_result and hasattr(self._hull_result, 'mesh') and self._hull_result.mesh:
                self.mesh_viewer.set_hull_mesh(self._hull_result.mesh)
                self.display_options.show_hull_option(True)
        
        # Restore cavity result
        if session.get('cavity_result'):
            from core.mold_cavity import MoldCavityResult
            self._cavity_result = dict_to_result(session['cavity_result'], MoldCavityResult)
            if self._cavity_result and hasattr(self._cavity_result, 'cavity_mesh') and self._cavity_result.cavity_mesh:
                self.mesh_viewer.set_cavity_mesh(self._cavity_result.cavity_mesh)
                self.display_options.show_cavity_option(True)
        
        # Restore mold halves result
        if session.get('mold_halves_result'):
            from core.mold_half_classification import MoldHalfClassificationResult
            self._mold_halves_result = dict_to_result(session['mold_halves_result'], MoldHalfClassificationResult)
            self._boundary_zone_threshold = session.get('boundary_zone_threshold', 0.15)
            if self._mold_halves_result and hasattr(self._mold_halves_result, 'classification') and self._mold_halves_result.classification is not None:
                # Update viewer with mold halves colors
                self.display_options.show_mold_halves_option(True)
        
        # Restore tet result
        if session.get('tet_result'):
            from core.tetrahedral_mesh import TetrahedralMeshResult
            self._tet_result = dict_to_result(session['tet_result'], TetrahedralMeshResult)
            self._tet_edge_length_fac = session.get('tet_edge_length_fac', 0.05)
            self._tet_optimize = session.get('tet_optimize', True)
            if self._tet_result:
                self.display_options.show_tet_mesh_option(True)
                
                # Restore tetrahedral mesh visualization if edge weights computed
                if hasattr(self._tet_result, 'edge_weights') and self._tet_result.edge_weights is not None:
                    self.mesh_viewer.set_tetrahedral_mesh(
                        self._tet_result.vertices,
                        self._tet_result.edges,
                        edge_weights=self._tet_result.edge_weights,
                        edge_boundary_labels=self._tet_result.edge_boundary_labels,
                        colormap='coolwarm'
                    )
                    # Show original tet mesh if available
                    if hasattr(self._tet_result, 'vertices_original') and self._tet_result.vertices_original is not None:
                        self.mesh_viewer.set_original_tetrahedral_mesh(self._tet_result.vertices_original, self._tet_result.edges)
                    # Show edge weight options
                    self.display_options.show_edge_weight_options(True)
                
                # Restore Dijkstra visualization if results exist
                if hasattr(self._tet_result, 'seed_escape_labels') and self._tet_result.seed_escape_labels is not None:
                    if hasattr(self._tet_result, 'seed_vertex_indices') and self._tet_result.seed_vertex_indices is not None:
                        self.mesh_viewer.set_dijkstra_result(
                            self._tet_result.vertices,
                            self._tet_result.seed_vertex_indices,
                            self._tet_result.seed_escape_labels,
                            boundary_mesh=None,
                            interior_distances=getattr(self._tet_result, 'seed_distances', None),
                            tet_edges=self._tet_result.edges
                        )
                        self.display_options.show_dijkstra_result_option(True)
                    
                    # Show secondary cuts if they exist
                    if hasattr(self._tet_result, 'secondary_cut_edges') and self._tet_result.secondary_cut_edges is not None:
                        self.display_options.show_secondary_cuts_option(True)
        
        # Restore parting surface results
        if session.get('parting_surface_result'):
            from core.parting_surface import PartingSurfaceResult
            self._parting_surface_result = dict_to_result(session['parting_surface_result'], PartingSurfaceResult)
        
        if session.get('primary_smoothing_result'):
            # This could be a ComprehensiveSurfaceResult or SmoothingResult
            self._primary_smoothing_result = type('Result', (), session['primary_smoothing_result'])()
            for k, v in session['primary_smoothing_result'].items():
                if isinstance(v, dict) and 'vertices' in v and 'faces' in v:
                    setattr(self._primary_smoothing_result, k, dict_to_mesh(v))
                else:
                    setattr(self._primary_smoothing_result, k, v)
        
        # Show primary parting surface in viewer
        if self._primary_smoothing_result and hasattr(self._primary_smoothing_result, 'mesh') and self._primary_smoothing_result.mesh:
            self.mesh_viewer.set_parting_surface(self._primary_smoothing_result.mesh)
            self.display_options.show_parting_surface_options(show_primary=True)
        elif self._parting_surface_result and hasattr(self._parting_surface_result, 'mesh') and self._parting_surface_result.mesh:
            self.mesh_viewer.set_parting_surface(self._parting_surface_result.mesh)
            self.display_options.show_parting_surface_options(show_primary=True)
        
        # Restore secondary surface results
        if session.get('secondary_surface_result'):
            self._secondary_surface_result = type('Result', (), session['secondary_surface_result'])()
            for k, v in session['secondary_surface_result'].items():
                if isinstance(v, dict) and 'vertices' in v and 'faces' in v:
                    setattr(self._secondary_surface_result, k, dict_to_mesh(v))
                else:
                    setattr(self._secondary_surface_result, k, v)
            
            if self._secondary_surface_result and hasattr(self._secondary_surface_result, 'mesh') and self._secondary_surface_result.mesh:
                self.mesh_viewer.set_secondary_parting_surface(self._secondary_surface_result.mesh)
                self.display_options.show_parting_surface_options(show_primary=True, show_secondary=True)
        
        # Restore step statuses
        step_status = session.get('step_status', {})
        for step_val, status in step_status.items():
            try:
                step = Step(step_val)
                if step in self.step_buttons:
                    self.step_buttons[step].set_status(status)
            except ValueError:
                pass
        
        # Restore active step
        active_step_val = session.get('active_step')
        if active_step_val:
            try:
                self._active_step = Step(active_step_val)
                for step, btn in self.step_buttons.items():
                    btn.set_active(step == self._active_step)
            except ValueError:
                pass
        
        # Update context panel for current step
        self._update_context_panel()
        
        # Show display options panel if we have any results to display
        if self._current_mesh is not None:
            self.display_options.show()
            self.display_options.raise_()  # Bring to front
        
        logger.info(f"Session restored, active step: {self._active_step}")

    def _on_view_changed(self, view: str):
        """Handle view change from title bar."""
        if self.mesh_viewer:
            self.mesh_viewer.set_view(view)
            logger.debug(f"View changed to: {view}")
    
    def _on_file_dropped(self, file_path: str, scale_factor: float = 1.0):
        """Handle file drop/selection."""
        self._load_mesh(file_path, scale_factor)
    
    def _on_hide_mesh_changed(self, hide: bool):
        """Handle hide mesh checkbox change."""
        if self.mesh_viewer:
            self.mesh_viewer.set_mesh_visible(not hide)
            logger.debug(f"Mesh visibility changed: hide={hide}")
    
    def _on_hide_parting_changed(self, hide: bool):
        """Handle hide parting analysis checkbox change."""
        if self.mesh_viewer:
            # Toggle visibility paint (inverted - hide=True means don't show)
            # Use the efficient toggle method that doesn't clear the scene
            self.mesh_viewer.set_visibility_paint_visible(not hide)
            
            # Toggle parting direction arrows (inverted)
            self.mesh_viewer.set_parting_arrows_visible(not hide)
            logger.debug(f"Parting analysis visibility changed: hide={hide}")
    
    def _on_hide_hull_changed(self, hide: bool):
        """Handle hide bounding hull checkbox change."""
        if self.mesh_viewer:
            self.mesh_viewer.set_hull_visible(not hide)
            logger.debug(f"Hull visibility changed: hide={hide}")
    
    def _on_hide_cavity_changed(self, hide: bool):
        """Handle hide mold cavity checkbox change."""
        if self.mesh_viewer:
            self.mesh_viewer.set_cavity_visible(not hide)
            logger.debug(f"Cavity visibility changed: hide={hide}")
    
    def _on_hide_tet_mesh_changed(self, hide: bool):
        """Handle hide tetrahedral mesh checkbox change."""
        if self.mesh_viewer:
            self.mesh_viewer.set_tetrahedral_mesh_visible(not hide)
            logger.debug(f"Tetrahedral mesh visibility changed: hide={hide}")
    
    def _on_hide_mold_halves_changed(self, hide: bool):
        """Handle hide mold halves checkbox change (both boundaries)."""
        if self.mesh_viewer:
            self.mesh_viewer.set_mold_halves_visible(not hide)
            logger.debug(f"Mold halves visibility changed: hide={hide}")
    
    def _on_hide_outer_boundary_changed(self, hide: bool):
        """Handle hide outer boundary checkbox change (H1/H2/boundary zone)."""
        if self.mesh_viewer:
            self.mesh_viewer.set_outer_boundary_visible(not hide)
            logger.debug(f"Outer boundary visibility changed: hide={hide}")
    
    def _on_hide_inner_boundary_changed(self, hide: bool):
        """Handle hide inner boundary checkbox change (seed edges)."""
        if self.mesh_viewer:
            self.mesh_viewer.set_inner_boundary_visible(not hide)
            logger.debug(f"Inner boundary visibility changed: hide={hide}")
    
    # =========================================================================
    # CAVITY STEP
    # =========================================================================
    
    def _setup_cavity_step(self):
        """Setup the mold cavity step UI."""
        # Check if mesh is loaded
        if self._current_mesh is None:
            no_mesh_label = QLabel("⚠️ No mesh loaded. Please import an STL file first.")
            no_mesh_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_mesh_label.setWordWrap(True)
            self.context_layout.addWidget(no_mesh_label)
            return
        
        # Check if hull is computed
        if self._hull_result is None:
            no_hull_label = QLabel("⚠️ Please generate bounding hull first.")
            no_hull_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_hull_label.setWordWrap(True)
            self.context_layout.addWidget(no_hull_label)
            return
        
        # Description section
        info_group = QGroupBox("Mold Cavity Creation")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Creates the mold cavity by performing CSG (Constructive Solid Geometry) "
            "subtraction: Hull - Original Mesh = Cavity. The cavity represents the "
            "hollow space between the hull and the original mesh."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Calculate button
        self.cavity_calc_btn = QPushButton("🕳️ Create Mold Cavity")
        self.cavity_calc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cavity_calc_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.cavity_calc_btn.clicked.connect(self._on_calculate_cavity)
        self.context_layout.addWidget(self.cavity_calc_btn)
        
        # Progress bar
        self.cavity_progress = QProgressBar()
        self.cavity_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.INFO};
                border-radius: 4px;
            }}
        """)
        self.cavity_progress.setTextVisible(False)
        self.cavity_progress.setRange(0, 0)  # Indeterminate
        self.cavity_progress.hide()
        self.context_layout.addWidget(self.cavity_progress)
        
        # Progress label
        self.cavity_progress_label = QLabel("")
        self.cavity_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.cavity_progress_label.hide()
        self.context_layout.addWidget(self.cavity_progress_label)
        
        # Cavity stats (show if computed)
        self.cavity_stats = StatsBox()
        self.cavity_stats.hide()
        self.context_layout.addWidget(self.cavity_stats)
        
        # Update UI with current state
        self._update_cavity_step_ui()
    
    def _update_cavity_step_ui(self):
        """Update cavity step UI based on current state."""
        if not hasattr(self, 'cavity_stats'):
            return
        
        if self._cavity_result is not None:
            # Show stats
            self.cavity_stats.clear()
            self.cavity_stats.show()
            
            result = self._cavity_result
            
            self.cavity_stats.add_header('✅ Cavity Created', Colors.SUCCESS)
            self.cavity_stats.add_row(f'Vertices: {result.validation.vertex_count:,}')
            self.cavity_stats.add_row(f'Faces: {result.validation.face_count:,}')
            
            # Validation info
            if result.validation.is_closed:
                self.cavity_stats.add_row('Manifold: ✅ Closed', Colors.SUCCESS)
            else:
                self.cavity_stats.add_row('Manifold: ⚠️ Open', Colors.WARNING)
            
            if result.validation.volume is not None:
                self.cavity_stats.add_row(f'Volume: {result.validation.volume:.2f} cubic units')
            
            if result.used_fallback:
                self.cavity_stats.add_row('⚠️ Used fallback engine', Colors.WARNING)
            
            # Update button text
            self.cavity_calc_btn.setText("🔄 Regenerate Cavity")
        else:
            self.cavity_stats.hide()
    
    def _on_calculate_cavity(self):
        """Start cavity computation."""
        if self._current_mesh is None or self._hull_result is None:
            return
        
        # Disable button during computation
        self.cavity_calc_btn.setEnabled(False)
        self.cavity_progress.show()
        self.cavity_progress_label.setText("Computing mold cavity (CSG subtraction)...")
        self.cavity_progress_label.show()
        
        # Clear previous cavity
        self.mesh_viewer.clear_cavity()
        self._cavity_result = None
        
        # Start worker
        self._cavity_worker = CavityWorker(self._hull_result.mesh, self._current_mesh)
        self._cavity_worker.progress.connect(self._on_cavity_progress)
        self._cavity_worker.complete.connect(self._on_cavity_complete)
        self._cavity_worker.error.connect(self._on_cavity_error)
        self._cavity_worker.finished.connect(self._on_cavity_worker_finished)
        self._cavity_worker.start()
    
    def _on_cavity_progress(self, message: str):
        """Handle cavity progress updates."""
        # Check if widget still exists (user may have navigated away)
        if hasattr(self, 'cavity_progress_label') and self.cavity_progress_label is not None:
            try:
                self.cavity_progress_label.setText(message)
            except RuntimeError:
                pass  # Widget was deleted
    
    def _on_cavity_complete(self, result: MoldCavityResult):
        """Handle cavity computation complete."""
        self._cavity_result = result
        
        # Add cavity to viewer
        self.mesh_viewer.set_cavity_mesh(result.mesh)
        
        # Update UI only if widgets still exist
        if hasattr(self, 'cavity_stats') and self.cavity_stats is not None:
            try:
                self._update_cavity_step_ui()
            except RuntimeError:
                pass  # Widget was deleted
        
        # Show cavity option in display options panel
        self.display_options.show_cavity_option(True)
        
        # Update step status
        self.step_buttons[Step.CAVITY].set_status('completed')
        
        # Unlock tetrahedralize step (now comes before mold halves)
        if Step.TETRAHEDRALIZE in self.step_buttons:
            self.step_buttons[Step.TETRAHEDRALIZE].set_status('available')
        
        logger.info(f"Cavity created: {result.validation.vertex_count} vertices, {result.validation.face_count} faces")
    
    def _on_cavity_error(self, message: str):
        """Handle cavity computation error."""
        # Check if widget still exists
        if hasattr(self, 'cavity_progress_label') and self.cavity_progress_label is not None:
            try:
                self.cavity_progress_label.setText(f"Error: {message}")
                self.cavity_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            except RuntimeError:
                pass  # Widget was deleted
        QMessageBox.critical(self, "Error", f"Cavity creation failed:\n{message}")
    
    def _on_cavity_worker_finished(self):
        """Handle cavity worker finished."""
        # Check if widgets still exist
        if hasattr(self, 'cavity_calc_btn') and self.cavity_calc_btn is not None:
            try:
                self.cavity_calc_btn.setEnabled(True)
            except RuntimeError:
                pass  # Widget was deleted
        if hasattr(self, 'cavity_progress') and self.cavity_progress is not None:
            try:
                self.cavity_progress.hide()
            except RuntimeError:
                pass  # Widget was deleted
        self._cavity_worker = None

    # =========================================================================
    # MOLD HALVES STEP
    # =========================================================================
    
    def _setup_mold_halves_step(self):
        """Setup the mold halves classification step UI."""
        # Check prerequisites
        if self._current_mesh is None:
            no_mesh_label = QLabel("⚠️ No mesh loaded. Please import an STL file first.")
            no_mesh_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_mesh_label.setWordWrap(True)
            self.context_layout.addWidget(no_mesh_label)
            return
        
        if self._tet_result is None:
            no_tet_label = QLabel("⚠️ Please generate tetrahedral mesh first.")
            no_tet_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_tet_label.setWordWrap(True)
            self.context_layout.addWidget(no_tet_label)
            return
        
        if self._parting_result is None:
            no_parting_label = QLabel("⚠️ Please compute parting directions first.")
            no_parting_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_parting_label.setWordWrap(True)
            self.context_layout.addWidget(no_parting_label)
            return
        
        # Description section
        info_group = QGroupBox("Mold Half Classification")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Classifies the tetrahedral boundary surface into H₁ and H₂ mold halves based on "
            "parting directions. H₁ (green) is pulled by D1, H₂ (orange) is pulled by D2. "
            "The boundary zone (gray) is the interface between the two halves."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Boundary zone slider
        slider_group = QGroupBox("Boundary Zone Width")
        slider_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        slider_layout = QVBoxLayout(slider_group)
        
        # Slider label
        self.boundary_zone_label = QLabel(f"{int(self._boundary_zone_threshold * 100)}% of interface diagonal")
        self.boundary_zone_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        slider_layout.addWidget(self.boundary_zone_label)
        
        # Slider
        self.boundary_zone_slider = QSlider(Qt.Orientation.Horizontal)
        self.boundary_zone_slider.setRange(0, 30)
        self.boundary_zone_slider.setValue(int(self._boundary_zone_threshold * 100))
        self.boundary_zone_slider.valueChanged.connect(self._on_boundary_zone_changed)
        self.boundary_zone_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {Colors.LIGHT};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.PRIMARY};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.PRIMARY};
                border-radius: 3px;
            }}
        """)
        slider_layout.addWidget(self.boundary_zone_slider)
        
        self.context_layout.addWidget(slider_group)
        
        # Calculate button
        self.mold_halves_calc_btn = QPushButton("🎨 Classify Mold Halves")
        self.mold_halves_calc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mold_halves_calc_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.mold_halves_calc_btn.clicked.connect(self._on_calculate_mold_halves)
        self.context_layout.addWidget(self.mold_halves_calc_btn)
        
        # Progress bar
        self.mold_halves_progress = QProgressBar()
        self.mold_halves_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.INFO};
                border-radius: 4px;
            }}
        """)
        self.mold_halves_progress.setTextVisible(False)
        self.mold_halves_progress.setRange(0, 0)  # Indeterminate
        self.mold_halves_progress.hide()
        self.context_layout.addWidget(self.mold_halves_progress)
        
        # Progress label
        self.mold_halves_progress_label = QLabel("")
        self.mold_halves_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.mold_halves_progress_label.hide()
        self.context_layout.addWidget(self.mold_halves_progress_label)
        
        # Stats (show if computed)
        self.mold_halves_stats = StatsBox()
        self.mold_halves_stats.hide()
        self.context_layout.addWidget(self.mold_halves_stats)
        
        # Update UI with current state
        self._update_mold_halves_step_ui()
    
    def _on_boundary_zone_changed(self, value: int):
        """Handle boundary zone slider change."""
        self._boundary_zone_threshold = value / 100.0
        if hasattr(self, 'boundary_zone_label') and self.boundary_zone_label is not None:
            try:
                self.boundary_zone_label.setText(f"{value}% of interface diagonal")
            except RuntimeError:
                pass
    
    def _update_mold_halves_step_ui(self):
        """Update mold halves step UI based on current state."""
        if not hasattr(self, 'mold_halves_stats'):
            return
        
        if self._mold_halves_result is not None:
            # Show stats
            self.mold_halves_stats.clear()
            self.mold_halves_stats.show()
            
            result = self._mold_halves_result
            
            self.mold_halves_stats.add_header('✅ Classification Complete', Colors.SUCCESS)
            
            # Calculate percentages
            outer_total = len(result.h1_triangles) + len(result.h2_triangles) + len(result.boundary_zone_triangles)
            h1_pct = (len(result.h1_triangles) / outer_total * 100) if outer_total > 0 else 0
            h2_pct = (len(result.h2_triangles) / outer_total * 100) if outer_total > 0 else 0
            
            self.mold_halves_stats.add_row(f'H₁ (Green): {len(result.h1_triangles):,} ({h1_pct:.1f}%)', Colors.SUCCESS)
            self.mold_halves_stats.add_row(f'H₂ (Orange): {len(result.h2_triangles):,} ({h2_pct:.1f}%)', Colors.WARNING)
            self.mold_halves_stats.add_row(f'Boundary Zone: {len(result.boundary_zone_triangles):,}')
            self.mold_halves_stats.add_row(f'Inner (part): {len(result.inner_boundary_triangles):,}')
            self.mold_halves_stats.add_row(f'Total: {result.total_triangles:,}')
            
            # Update button text
            self.mold_halves_calc_btn.setText("🔄 Reclassify")
        else:
            self.mold_halves_stats.hide()
    
    def _on_calculate_mold_halves(self):
        """Start mold halves classification."""
        if self._tet_result is None or self._parting_result is None or self._hull_result is None:
            return
        
        # Check for boundary mesh
        if self._tet_result.boundary_mesh is None:
            QMessageBox.warning(self, "Error", "Tetrahedral boundary mesh not available")
            return
        
        # Disable button during computation
        self.mold_halves_calc_btn.setEnabled(False)
        self.mold_halves_progress.show()
        self.mold_halves_progress_label.setText("Classifying mold halves...")
        self.mold_halves_progress_label.show()
        
        # Clear previous result
        self._mold_halves_result = None
        
        # Get parting directions
        d1 = self._parting_result.d1
        d2 = self._parting_result.d2
        
        # Use tetrahedral boundary mesh instead of cavity mesh
        boundary_mesh = self._tet_result.boundary_mesh
        
        # Log mesh stats (quick)
        logger.info(f"Mold halves using tet boundary mesh: {len(boundary_mesh.vertices)} verts, {len(boundary_mesh.faces)} faces")
        
        # Start worker with tet mesh data for fast label-based detection
        self._mold_halves_worker = MoldHalvesWorker(
            boundary_mesh,  # Use tet boundary mesh
            self._hull_result.mesh,
            d1, d2,
            self._boundary_zone_threshold,
            part_mesh=self._current_mesh,  # Keep for fallback
            use_proximity_method=False,  # Don't use slow proximity method
            tet_vertices=self._tet_result.vertices,  # Pass tet vertices
            tet_boundary_labels=self._tet_result.boundary_labels  # Pass pre-computed labels
        )
        self._mold_halves_worker.progress.connect(self._on_mold_halves_progress)
        self._mold_halves_worker.complete.connect(self._on_mold_halves_complete)
        self._mold_halves_worker.error.connect(self._on_mold_halves_error)
        self._mold_halves_worker.finished.connect(self._on_mold_halves_worker_finished)
        self._mold_halves_worker.start()
    
    def _on_mold_halves_progress(self, message: str):
        """Handle mold halves progress updates."""
        if hasattr(self, 'mold_halves_progress_label') and self.mold_halves_progress_label is not None:
            try:
                self.mold_halves_progress_label.setText(message)
            except RuntimeError:
                pass
    
    def _on_mold_halves_complete(self, result: MoldHalfClassificationResult):
        """Handle mold halves classification complete."""
        self._mold_halves_result = result
        
        # Get the tetrahedral boundary mesh
        boundary_mesh = self._tet_result.boundary_mesh
        
        # Set up cavity mesh for visualization
        self.mesh_viewer.set_cavity_mesh(boundary_mesh)
        
        # Apply classification colors AND seed vertex identification
        # Seeds are identified by distance to part mesh (not boundary mesh classification)
        # This shows H1=green, H2=orange, boundary=gray, seed=blue
        self.mesh_viewer.apply_tet_mesh_classification(
            tet_vertices=self._tet_result.vertices,
            tet_edges=self._tet_result.edges,
            boundary_mesh=boundary_mesh,
            classification_result=result,
            part_mesh=self._current_mesh,
            seed_distance_threshold=None  # Auto-compute (2% of mesh size) - same as inflation
        )
        
        # Show the mold halves display option
        self.display_options.show_mold_halves_option(True)
        
        # Update UI
        if hasattr(self, 'mold_halves_stats') and self.mold_halves_stats is not None:
            try:
                self._update_mold_halves_step_ui()
            except RuntimeError:
                pass
        
        # Update step status and unlock edge weights step
        self.step_buttons[Step.MOLD_HALVES].set_status('completed')
        if Step.EDGE_WEIGHTS in self.step_buttons:
            self.step_buttons[Step.EDGE_WEIGHTS].set_status('available')
        
        logger.info(
            f"Mold halves classified: H1={len(result.h1_triangles)}, "
            f"H2={len(result.h2_triangles)}, boundary={len(result.boundary_zone_triangles)}"
        )
    
    def _on_mold_halves_error(self, message: str):
        """Handle mold halves classification error."""
        if hasattr(self, 'mold_halves_progress_label') and self.mold_halves_progress_label is not None:
            try:
                self.mold_halves_progress_label.setText(f"Error: {message}")
                self.mold_halves_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            except RuntimeError:
                pass
        QMessageBox.critical(self, "Error", f"Mold halves classification failed:\n{message}")
    
    def _on_mold_halves_worker_finished(self):
        """Handle mold halves worker finished."""
        if hasattr(self, 'mold_halves_calc_btn') and self.mold_halves_calc_btn is not None:
            try:
                self.mold_halves_calc_btn.setEnabled(True)
            except RuntimeError:
                pass
        if hasattr(self, 'mold_halves_progress') and self.mold_halves_progress is not None:
            try:
                self.mold_halves_progress.hide()
            except RuntimeError:
                pass
        self._mold_halves_worker = None

    # =========================================================================
    # TETRAHEDRALIZE STEP
    # =========================================================================
    
    def _setup_tetrahedralize_step(self):
        """Setup the tetrahedralize step UI."""
        # Check prerequisites
        if self._current_mesh is None:
            no_mesh_label = QLabel("⚠️ No mesh loaded. Please import an STL file first.")
            no_mesh_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_mesh_label.setWordWrap(True)
            self.context_layout.addWidget(no_mesh_label)
            return
        
        if self._cavity_result is None:
            no_cavity_label = QLabel("⚠️ Please create mold cavity first.")
            no_cavity_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_cavity_label.setWordWrap(True)
            self.context_layout.addWidget(no_cavity_label)
            return
        
        if not PYTETWILD_AVAILABLE:
            no_tetwild_label = QLabel("⚠️ pytetwild is not installed. Install with: pip install pytetwild")
            no_tetwild_label.setStyleSheet(f"""
                color: {Colors.DANGER};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 0, 0, 0.1);
                border: 1px solid {Colors.DANGER};
                border-radius: 6px;
            """)
            no_tetwild_label.setWordWrap(True)
            self.context_layout.addWidget(no_tetwild_label)
            return
        
        # Description section
        info_group = QGroupBox("Tetrahedral Mesh Generation")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Generates a tetrahedral mesh of the mold cavity volume using fTetWild. "
            "This replaces the voxel grid with a more accurate mesh representation. "
            "Edge weights are computed for the parting surface algorithm."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Settings group
        settings_group = QGroupBox("Mesh Resolution")
        settings_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        settings_layout = QVBoxLayout(settings_group)
        
        # Edge length factor slider
        edge_label = QLabel(f"Edge length: {self._tet_edge_length_fac:.2f} × bbox diagonal")
        edge_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.tet_edge_label = edge_label
        settings_layout.addWidget(edge_label)
        
        self.tet_edge_slider = QSlider(Qt.Orientation.Horizontal)
        self.tet_edge_slider.setRange(1, 20)  # 0.01 to 0.20
        self.tet_edge_slider.setValue(int(self._tet_edge_length_fac * 100))
        self.tet_edge_slider.valueChanged.connect(self._on_tet_edge_changed)
        self.tet_edge_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {Colors.LIGHT};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.PRIMARY};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.PRIMARY};
                border-radius: 3px;
            }}
        """)
        settings_layout.addWidget(self.tet_edge_slider)
        
        # Optimization checkbox
        self.tet_optimize_checkbox = QCheckBox("Optimize mesh quality (slower but better)")
        self.tet_optimize_checkbox.setChecked(self._tet_optimize)
        self.tet_optimize_checkbox.stateChanged.connect(self._on_tet_optimize_changed)
        self.tet_optimize_checkbox.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        settings_layout.addWidget(self.tet_optimize_checkbox)
        
        self.context_layout.addWidget(settings_group)
        
        # Complexity estimate group
        estimate_group = QGroupBox("Estimated Complexity")
        estimate_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        estimate_layout = QVBoxLayout(estimate_group)
        
        # Complexity info label
        self.tet_complexity_label = QLabel()
        self.tet_complexity_label.setWordWrap(True)
        self.tet_complexity_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        estimate_layout.addWidget(self.tet_complexity_label)
        
        # Estimated time label
        self.tet_time_estimate_label = QLabel()
        self.tet_time_estimate_label.setStyleSheet(f'color: {Colors.INFO}; font-size: 12px; font-weight: 500;')
        estimate_layout.addWidget(self.tet_time_estimate_label)
        
        self.context_layout.addWidget(estimate_group)
        
        # Update estimate based on current settings
        self._update_tet_estimate()
        
        # Generate button
        self.tet_generate_btn = QPushButton("🔷 Generate Tetrahedral Mesh")
        self.tet_generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tet_generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.tet_generate_btn.clicked.connect(self._on_generate_tet_mesh)
        self.context_layout.addWidget(self.tet_generate_btn)
        
        # Progress bar
        self.tet_progress = QProgressBar()
        self.tet_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.INFO};
                border-radius: 4px;
            }}
        """)
        self.tet_progress.setTextVisible(False)
        self.tet_progress.setRange(0, 0)  # Indeterminate
        self.tet_progress.hide()
        self.context_layout.addWidget(self.tet_progress)
        
        # Progress label
        self.tet_progress_label = QLabel("")
        self.tet_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.tet_progress_label.hide()
        self.context_layout.addWidget(self.tet_progress_label)
        
        # Stats (show if computed)
        self.tet_stats = StatsBox()
        self.tet_stats.hide()
        self.context_layout.addWidget(self.tet_stats)
        
        # Update UI with current state
        self._update_tet_step_ui()
    
    def _on_tet_edge_changed(self, value: int):
        """Handle tet edge length slider change."""
        self._tet_edge_length_fac = value / 100.0
        if hasattr(self, 'tet_edge_label') and self.tet_edge_label is not None:
            try:
                self.tet_edge_label.setText(f"Edge length: {self._tet_edge_length_fac:.2f} × bbox diagonal")
            except RuntimeError:
                pass
        # Update estimate when edge length changes
        self._update_tet_estimate()
    
    def _on_tet_optimize_changed(self, state: int):
        """Handle tet optimize checkbox change."""
        self._tet_optimize = state == Qt.CheckState.Checked.value
        # Update estimate when optimize setting changes
        self._update_tet_estimate()
    
    def _update_tet_estimate(self):
        """Update the estimated time and complexity display for tetrahedralization."""
        if not hasattr(self, 'tet_complexity_label') or self.tet_complexity_label is None:
            return
        if self._cavity_result is None:
            return
        
        try:
            cavity_mesh = self._cavity_result.mesh
            n_verts = len(cavity_mesh.vertices)
            n_faces = len(cavity_mesh.faces)
            
            # Calculate bounding box diagonal
            bounds = cavity_mesh.bounds
            bbox_diag = np.linalg.norm(bounds[1] - bounds[0])
            
            # Estimate target edge length
            target_edge = bbox_diag * self._tet_edge_length_fac
            
            # Estimate mesh volume (approximate)
            volume = cavity_mesh.volume if cavity_mesh.is_watertight else bbox_diag ** 3 * 0.1
            
            # Estimate number of tetrahedra based on volume and edge length
            # Rough formula: n_tets ≈ volume / (edge^3 * 0.1)
            # Each tet has volume ≈ edge^3 / 6
            est_tet_volume = (target_edge ** 3) / 6.0
            est_n_tets = max(100, int(abs(volume) / est_tet_volume))
            
            # Estimate vertices (roughly n_tets^(1/3) * factor)
            est_n_verts = max(100, int(est_n_tets ** 0.4 * 10))
            
            # Cap estimates at reasonable bounds
            est_n_tets = min(est_n_tets, 10_000_000)
            est_n_verts = min(est_n_verts, 1_000_000)
            
            # Estimate time based on complexity
            # Base time roughly proportional to input faces * output tets
            complexity_factor = (n_faces * est_n_tets) / 1_000_000
            
            # Base time in seconds (empirically calibrated)
            # Small meshes: ~5-30 sec, medium: ~2-8 min, large: 15-60+ min
            base_time_sec = 7.5 + complexity_factor * 0.15
            
            # Optimization adds ~50% more time
            if self._tet_optimize:
                base_time_sec *= 1.5
            
            # Format time estimate
            if base_time_sec < 60:
                time_str = f"~{base_time_sec:.0f} seconds"
            elif base_time_sec < 300:
                time_str = f"~{base_time_sec/60:.1f} minutes"
            else:
                time_str = f"~{base_time_sec/60:.0f} minutes"
            
            # Determine complexity level for color coding
            if base_time_sec < 10:
                complexity_level = "Low"
                time_color = Colors.SUCCESS
            elif base_time_sec < 60:
                complexity_level = "Medium"
                time_color = Colors.WARNING
            else:
                complexity_level = "High"
                time_color = Colors.DANGER
            
            # Update labels
            self.tet_complexity_label.setText(
                f"Input: {n_verts:,} vertices, {n_faces:,} faces\n"
                f"Target edge: {target_edge:.3f} units\n"
                f"Est. output: ~{est_n_tets:,} tetrahedra"
            )
            
            self.tet_time_estimate_label.setText(f"⏱️ Estimated time: {time_str} ({complexity_level} complexity)")
            self.tet_time_estimate_label.setStyleSheet(f'color: {time_color}; font-size: 12px; font-weight: 500;')
            
        except Exception as e:
            logger.debug(f"Error updating tet estimate: {e}")
            self.tet_complexity_label.setText("Unable to estimate complexity")
            self.tet_time_estimate_label.setText("")
    
    def _update_tet_step_ui(self):
        """Update tetrahedralize step UI based on current state."""
        if not hasattr(self, 'tet_stats'):
            return
        
        if self._tet_result is not None:
            # Show stats
            self.tet_stats.clear()
            self.tet_stats.show()
            
            result = self._tet_result
            
            self.tet_stats.add_header('✅ Tetrahedral Mesh Generated', Colors.SUCCESS)
            self.tet_stats.add_row(f'Vertices: {result.num_vertices:,}')
            self.tet_stats.add_row(f'Tetrahedra: {result.num_tetrahedra:,}')
            self.tet_stats.add_row(f'Edges: {result.num_edges:,}')
            self.tet_stats.add_row(f'Boundary Faces: {result.num_boundary_faces:,}')
            self.tet_stats.add_row(f'Tet Time: {result.tetrahedralize_time_ms:.0f}ms')
        else:
            self.tet_stats.hide()
    
    def _on_generate_tet_mesh(self):
        """Handle generate tetrahedral mesh button click."""
        if self._tet_worker is not None:
            return
        
        # Show progress with indeterminate mode (pulsing bar)
        self.tet_generate_btn.setEnabled(False)
        self.tet_progress.setRange(0, 0)  # Indeterminate mode - pulsing bar
        self.tet_progress.show()
        self.tet_progress_label.show()
        self.tet_progress_label.setText("Initializing tetrahedralization...")
        self.tet_stats.hide()
        
        # Log cavity mesh stats (quick - no distance computation)
        cavity_mesh = self._cavity_result.mesh
        logger.info(f"Cavity mesh for tetrahedralization: {len(cavity_mesh.vertices)} verts, {len(cavity_mesh.faces)} faces")
        
        # Process events to update UI before blocking operation
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Clear previous result
        self._tet_result = None
        
        # Start worker - tetrahedralize and filter out tetrahedra inside part
        self._tet_worker = TetrahedralizeWorker(
            self._cavity_result.mesh,
            part_mesh=self._current_mesh,  # Pass part mesh for filtering
            edge_length_fac=self._tet_edge_length_fac,
            optimize=self._tet_optimize
        )
        self._tet_worker.progress.connect(self._on_tet_progress)
        self._tet_worker.complete.connect(self._on_tet_complete)
        self._tet_worker.error.connect(self._on_tet_error)
        self._tet_worker.finished.connect(self._on_tet_worker_finished)
        self._tet_worker.start()
    
    def _on_tet_progress(self, message: str):
        """Handle tetrahedralize progress update."""
        if hasattr(self, 'tet_progress_label') and self.tet_progress_label is not None:
            try:
                self.tet_progress_label.setText(message)
            except RuntimeError:
                pass
    
    def _on_tet_complete(self, result: TetrahedralMeshResult):
        """Handle tetrahedralize complete."""
        self._tet_result = result
        
        # Update UI
        if hasattr(self, 'tet_stats') and self.tet_stats is not None:
            try:
                self._update_tet_step_ui()
            except RuntimeError:
                pass
        
        # Visualize tetrahedral mesh edges (no weights yet - just solid color)
        self.mesh_viewer.set_tetrahedral_mesh(
            result.vertices,
            result.edges
        )
        
        # Also set the boundary mesh as the cavity for visualization
        if result.boundary_mesh is not None:
            self.mesh_viewer.set_cavity_mesh(result.boundary_mesh)
        
        # Show tet mesh display option
        self.display_options.show_tet_mesh_option(True)
        
        # Update step status and unlock mold halves step (now comes after tetrahedralize)
        self.step_buttons[Step.TETRAHEDRALIZE].set_status('completed')
        if Step.MOLD_HALVES in self.step_buttons:
            self.step_buttons[Step.MOLD_HALVES].set_status('available')
        
        logger.info(
            f"Tetrahedral mesh complete: {result.num_vertices} verts, "
            f"{result.num_tetrahedra} tets, {result.num_edges} edges, "
            f"{result.num_boundary_faces} boundary faces"
        )
    
    def _on_tet_error(self, message: str):
        """Handle tetrahedralize error."""
        if hasattr(self, 'tet_progress_label') and self.tet_progress_label is not None:
            try:
                self.tet_progress_label.setText(f"Error: {message}")
                self.tet_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            except RuntimeError:
                pass
        QMessageBox.critical(self, "Error", f"Tetrahedral mesh generation failed:\n{message}")
    
    def _on_tet_worker_finished(self):
        """Handle tetrahedralize worker finished."""
        if hasattr(self, 'tet_generate_btn') and self.tet_generate_btn is not None:
            try:
                self.tet_generate_btn.setEnabled(True)
            except RuntimeError:
                pass
        if hasattr(self, 'tet_progress') and self.tet_progress is not None:
            try:
                self.tet_progress.setRange(0, 100)  # Reset to determinate mode
                self.tet_progress.hide()
            except RuntimeError:
                pass
        self._tet_worker = None

    # =========================================================================
    # EDGE WEIGHTS STEP
    # =========================================================================
    
    def _setup_edge_weights_step(self):
        """Setup the edge weights step UI."""
        # Check prerequisites
        if self._tet_result is None:
            no_tet_label = QLabel("⚠️ Please generate tetrahedral mesh first.")
            no_tet_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_tet_label.setWordWrap(True)
            self.context_layout.addWidget(no_tet_label)
            return
        
        # Description section
        info_group = QGroupBox("Edge Weight Computation")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Computes edge weights based on distance to the part surface. "
            "Edges closer to the part get higher weights, guiding the parting "
            "surface to stay near the part boundary.\n\n"
            "Formula: weight = 1 / (distance² + 0.25)"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Mesh info
        tet_info = StatsBox()
        tet_info.add_header('Tetrahedral Mesh', Colors.INFO)
        tet_info.add_row(f'Vertices: {self._tet_result.num_vertices:,}')
        tet_info.add_row(f'Edges: {self._tet_result.num_edges:,}')
        tet_info.add_row(f'Tetrahedra: {self._tet_result.num_tetrahedra:,}')
        self.context_layout.addWidget(tet_info)
        
        # Compute button
        self.edge_weights_btn = QPushButton("⚖️ Compute Edge Weights")
        self.edge_weights_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edge_weights_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.edge_weights_btn.clicked.connect(self._on_compute_edge_weights)
        self.context_layout.addWidget(self.edge_weights_btn)
        
        # Progress bar
        self.edge_weights_progress = QProgressBar()
        self.edge_weights_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.INFO};
                border-radius: 4px;
            }}
        """)
        self.edge_weights_progress.setTextVisible(False)
        self.edge_weights_progress.setRange(0, 0)  # Indeterminate
        self.edge_weights_progress.hide()
        self.context_layout.addWidget(self.edge_weights_progress)
        
        # Progress label
        self.edge_weights_progress_label = QLabel("")
        self.edge_weights_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.edge_weights_progress_label.hide()
        self.context_layout.addWidget(self.edge_weights_progress_label)
        
        # Stats (show if computed)
        self.edge_weights_stats = StatsBox()
        self.edge_weights_stats.hide()
        self.context_layout.addWidget(self.edge_weights_stats)
        
        # Update UI with current state
        self._update_edge_weights_step_ui()
    
    def _update_edge_weights_step_ui(self):
        """Update edge weights step UI based on current state."""
        if not hasattr(self, 'edge_weights_stats'):
            return
        
        if self._tet_result is not None and self._tet_result.edge_weights is not None:
            # Show stats
            self.edge_weights_stats.clear()
            self.edge_weights_stats.show()
            
            result = self._tet_result
            
            self.edge_weights_stats.add_header('✅ Edge Weights Computed', Colors.SUCCESS)
            self.edge_weights_stats.add_row(f'Weight Range: {result.edge_weights.min():.4f} - {result.edge_weights.max():.4f}')
            self.edge_weights_stats.add_row(f'Mean Weight: {result.edge_weights.mean():.4f}')
            
            if result.boundary_labels is not None:
                n_h1 = np.sum(result.boundary_labels == 1)
                n_h2 = np.sum(result.boundary_labels == 2)
                n_inner = np.sum(result.boundary_labels == -1)
                self.edge_weights_stats.add_header('Boundary Labels', Colors.INFO)
                self.edge_weights_stats.add_row(f'H₁ vertices: {n_h1:,}')
                self.edge_weights_stats.add_row(f'H₂ vertices: {n_h2:,}')
                self.edge_weights_stats.add_row(f'Seed vertices: {n_inner:,}')
        else:
            self.edge_weights_stats.hide()
    
    def _on_compute_edge_weights(self):
        """Handle compute edge weights button click."""
        if self._edge_weights_worker is not None:
            return
        
        # Show progress
        self.edge_weights_btn.setEnabled(False)
        self.edge_weights_progress.show()
        self.edge_weights_progress_label.show()
        self.edge_weights_progress_label.setText("Initializing...")
        self.edge_weights_stats.hide()
        
        # Start worker
        # Get parting directions from parting result (if available)
        d1, d2 = None, None
        if self._parting_result is not None:
            d1 = self._parting_result.d1
            d2 = self._parting_result.d2
        
        self._edge_weights_worker = EdgeWeightsWorker(
            self._tet_result,
            self._current_mesh,
            self._cavity_result.mesh,
            self._hull_result.mesh,
            self._mold_halves_result,
            d1=d1,
            d2=d2
        )
        self._edge_weights_worker.progress.connect(self._on_edge_weights_progress)
        self._edge_weights_worker.r_distance_computed.connect(self._on_r_distance_computed)
        self._edge_weights_worker.mesh_inflated.connect(self._on_mesh_inflated)
        self._edge_weights_worker.boundary_classified.connect(self._on_boundary_classified)
        self._edge_weights_worker.complete.connect(self._on_edge_weights_complete)
        self._edge_weights_worker.error.connect(self._on_edge_weights_error)
        self._edge_weights_worker.finished.connect(self._on_edge_weights_worker_finished)
        self._edge_weights_worker.start()
    
    def _on_edge_weights_progress(self, message: str):
        """Handle edge weights progress update."""
        if hasattr(self, 'edge_weights_progress_label') and self.edge_weights_progress_label is not None:
            try:
                self.edge_weights_progress_label.setText(message)
            except RuntimeError:
                pass
    
    def _on_r_distance_computed(self, r_data: dict):
        """Handle R distance computed - visualize the R line."""
        hull_point = r_data['hull_point']
        part_point = r_data['part_point']
        r_value = r_data['r_value']
        
        # Visualize R as a red line in the 3D scene
        self.mesh_viewer.set_r_distance_line(hull_point, part_point, r_value)
        
        # Update stats display
        if hasattr(self, 'edge_weights_stats') and self.edge_weights_stats is not None:
            try:
                self.edge_weights_stats.clear()
                self.edge_weights_stats.show()
                self.edge_weights_stats.add_header('📏 R Distance Computed', Colors.INFO)
                self.edge_weights_stats.add_row(f'R = {r_value:.4f}')
                self.edge_weights_stats.add_row(f'Hull point: ({hull_point[0]:.2f}, {hull_point[1]:.2f}, {hull_point[2]:.2f})')
                self.edge_weights_stats.add_row(f'Part point: ({part_point[0]:.2f}, {part_point[1]:.2f}, {part_point[2]:.2f})')
            except RuntimeError:
                pass
        
        logger.info(f"R distance visualized: R={r_value:.4f}")
    
    def _on_mesh_inflated(self, result: TetrahedralMeshResult):
        """Handle mesh inflation complete - visualize the inflated boundary mesh."""
        logger.info(f"Mesh inflated: {result.num_vertices} vertices, {result.num_boundary_faces} boundary faces")
        
        # Update stored result
        self._tet_result = result
        
        # Display the inflated boundary mesh
        if result.boundary_mesh is not None:
            # Set the inflated boundary mesh as a surface visualization
            self.mesh_viewer.set_inflated_boundary_mesh(result.boundary_mesh)
            
            # Update stats display
            if hasattr(self, 'edge_weights_stats') and self.edge_weights_stats is not None:
                try:
                    self.edge_weights_stats.add_header('🔄 Mesh Inflated', Colors.SUCCESS)
                    self.edge_weights_stats.add_row(f'Boundary vertices: {len(result.boundary_mesh.vertices):,}')
                    self.edge_weights_stats.add_row(f'Boundary faces: {len(result.boundary_mesh.faces):,}')
                except RuntimeError:
                    pass
        
        logger.info("Inflated boundary mesh displayed")
    
    def _on_boundary_classified(self, classification_result):
        """Handle boundary mesh classification result."""
        # Update stats display with classification info
        if hasattr(self, 'edge_weights_stats') and self.edge_weights_stats is not None:
            try:
                self.edge_weights_stats.add_header('🏷️ Boundary Classified', Colors.INFO)
                self.edge_weights_stats.add_row(f'H₁ faces: {len(classification_result.h1_triangles):,}')
                self.edge_weights_stats.add_row(f'H₂ faces: {len(classification_result.h2_triangles):,}')
                self.edge_weights_stats.add_row(f'Inner boundary: {len(classification_result.inner_boundary_triangles):,}')
                self.edge_weights_stats.add_row(f'Boundary zone: {len(classification_result.boundary_zone_triangles):,}')
            except RuntimeError:
                pass
        
        logger.info(
            f"Tetrahedral boundary classified: H1={len(classification_result.h1_triangles)}, "
            f"H2={len(classification_result.h2_triangles)}, inner={len(classification_result.inner_boundary_triangles)}"
        )
    
    def _on_edge_weights_complete(self, result: TetrahedralMeshResult):
        """Handle edge weights complete."""
        self._tet_result = result
        
        # Update UI with boundary labels
        if result.boundary_labels is not None:
            n_h1 = np.sum(result.boundary_labels == 1)
            n_h2 = np.sum(result.boundary_labels == 2)
            n_inner = np.sum(result.boundary_labels == -1)
            
            if hasattr(self, 'edge_weights_stats') and self.edge_weights_stats is not None:
                try:
                    self.edge_weights_stats.add_header('🏷️ Vertex Labels', Colors.SUCCESS)
                    self.edge_weights_stats.add_row(f'H₁ vertices: {n_h1:,}')
                    self.edge_weights_stats.add_row(f'H₂ vertices: {n_h2:,}')
                    self.edge_weights_stats.add_row(f'Seed (inner) vertices: {n_inner:,}')
                except RuntimeError:
                    pass
            
            logger.info(f"Boundary vertex labels: H1={n_h1}, H2={n_h2}, inner(seeds)={n_inner}")
        
        # Update UI with edge stats
        if result.edge_weights is not None:
            if hasattr(self, 'edge_weights_stats') and self.edge_weights_stats is not None:
                try:
                    self.edge_weights_stats.add_header('⚖️ Edge Weights', Colors.INFO)
                    self.edge_weights_stats.add_row(f'Range: [{result.edge_weights.min():.4f}, {result.edge_weights.max():.4f}]')
                    self.edge_weights_stats.add_row(f'Mean: {result.edge_weights.mean():.4f}')
                    
                    if result.edge_boundary_labels is not None:
                        n_interior = np.sum(result.edge_boundary_labels == 0)
                        n_h1_edges = np.sum(result.edge_boundary_labels == 1)
                        n_h2_edges = np.sum(result.edge_boundary_labels == 2)
                        n_inner_edges = np.sum(result.edge_boundary_labels == -1)
                        n_mixed_edges = np.sum(result.edge_boundary_labels == -2)
                        
                        self.edge_weights_stats.add_header('🔗 Edge Labels', Colors.INFO)
                        self.edge_weights_stats.add_row(f'Interior edges: {n_interior:,}')
                        self.edge_weights_stats.add_row(f'H₁ edges: {n_h1_edges:,}')
                        self.edge_weights_stats.add_row(f'H₂ edges: {n_h2_edges:,}')
                        self.edge_weights_stats.add_row(f'Inner edges: {n_inner_edges:,}')
                        self.edge_weights_stats.add_row(f'Mixed edges: {n_mixed_edges:,}')
                except RuntimeError:
                    pass
            
            # Visualize tetrahedral mesh edges:
            # - Interior edges colored by weight (coolwarm)
            # - Boundary edges colored by mold half (green/orange/gray)
            self.mesh_viewer.set_tetrahedral_mesh(
                result.vertices,
                result.edges,
                edge_weights=result.edge_weights,
                edge_boundary_labels=result.edge_boundary_labels,
                colormap='coolwarm'  # Blue (low weight) to Red (high weight)
            )
            
            # Also store original (non-inflated) tetrahedral mesh for visualization
            if result.vertices_original is not None:
                logger.info(f"Setting original tet mesh with {len(result.vertices_original)} vertices")
                self.mesh_viewer.set_original_tetrahedral_mesh(result.vertices_original, result.edges)
            else:
                logger.warning("No original vertices available for original tet mesh visualization")
            
            logger.info(f"Edge weights computed: range [{result.edge_weights.min():.4f}, {result.edge_weights.max():.4f}]")
            
            # Show edge weight options in display panel
            self.display_options.show_edge_weight_options(True)
        else:
            # Only R was computed - just log that
            if result.r_value is not None:
                logger.info(f"R computation complete: R={result.r_value:.4f}")
        
        # Update step status
        self.step_buttons[Step.EDGE_WEIGHTS].set_status('completed')
    
    def _on_edge_weights_error(self, message: str):
        """Handle edge weights error."""
        if hasattr(self, 'edge_weights_progress_label') and self.edge_weights_progress_label is not None:
            try:
                self.edge_weights_progress_label.setText(f"Error: {message}")
                self.edge_weights_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            except RuntimeError:
                pass
        QMessageBox.critical(self, "Error", f"Edge weight computation failed:\n{message}")
    
    def _on_edge_weights_worker_finished(self):
        """Handle edge weights worker finished."""
        if hasattr(self, 'edge_weights_btn') and self.edge_weights_btn is not None:
            try:
                self.edge_weights_btn.setEnabled(True)
            except RuntimeError:
                pass
        if hasattr(self, 'edge_weights_progress') and self.edge_weights_progress is not None:
            try:
                self.edge_weights_progress.hide()
            except RuntimeError:
                pass
        self._edge_weights_worker = None

    # ========================================================================
    # DIJKSTRA STEP
    # ========================================================================
    
    def _setup_dijkstra_step(self):
        """Setup the Dijkstra step UI."""
        # Check prerequisites
        if self._tet_result is None or self._tet_result.edge_weights is None:
            no_weights_label = QLabel("⚠️ Please compute edge weights first.")
            no_weights_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_weights_label.setWordWrap(True)
            self.context_layout.addWidget(no_weights_label)
            return
        
        # Description section
        info_group = QGroupBox("Dijkstra Escape Labeling")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Each seed vertex (blue vertices on part surface) walks through the "
            "tetrahedral mesh to find the shortest weighted path to either H₁ or H₂ boundary.\n\n"
            "Seeds are colored by destination:\n"
            "• Green = walks to H₁\n"
            "• Orange = walks to H₂"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Info about current state
        result = self._tet_result
        state_info = StatsBox()
        state_info.add_header('Current State', Colors.INFO)
        
        if result.boundary_labels is not None:
            n_seeds = np.sum(result.boundary_labels == -1)
            n_h1 = np.sum(result.boundary_labels == 1)
            n_h2 = np.sum(result.boundary_labels == 2)
            state_info.add_row(f'Seed vertices (inner): {n_seeds:,}')
            state_info.add_row(f'H₁ target vertices: {n_h1:,}')
            state_info.add_row(f'H₂ target vertices: {n_h2:,}')
        
        state_info.add_row(f'Total edges: {result.num_edges:,}')
        state_info.add_row(f'Total tetrahedra: {result.num_tetrahedra:,}')
        self.context_layout.addWidget(state_info)
        
        # Compute button
        self.dijkstra_btn = QPushButton("🛤️ Run Dijkstra Walk")
        self.dijkstra_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dijkstra_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.dijkstra_btn.clicked.connect(self._on_run_dijkstra)
        self.context_layout.addWidget(self.dijkstra_btn)
        
        # Progress bar
        self.dijkstra_progress = QProgressBar()
        self.dijkstra_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.INFO};
                border-radius: 4px;
            }}
        """)
        self.dijkstra_progress.setTextVisible(False)
        self.dijkstra_progress.setRange(0, 0)  # Indeterminate
        self.dijkstra_progress.hide()
        self.context_layout.addWidget(self.dijkstra_progress)
        
        # Progress label
        self.dijkstra_progress_label = QLabel("")
        self.dijkstra_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.dijkstra_progress_label.hide()
        self.context_layout.addWidget(self.dijkstra_progress_label)
        
        # Stats (show if computed)
        self.dijkstra_stats = StatsBox()
        self.dijkstra_stats.hide()
        self.context_layout.addWidget(self.dijkstra_stats)
        
        # Update UI with current state
        self._update_dijkstra_step_ui()
    
    def _update_dijkstra_step_ui(self):
        """Update Dijkstra step UI based on current state."""
        if not hasattr(self, 'dijkstra_stats'):
            return
        
        if self._tet_result is not None and self._tet_result.seed_escape_labels is not None:
            # Show stats
            self.dijkstra_stats.clear()
            self.dijkstra_stats.show()
            
            result = self._tet_result
            interior_labels = result.seed_escape_labels
            interior_indices = result.seed_vertex_indices
            interior_distances = result.seed_distances
            boundary_labels = result.boundary_labels
            
            n_interior = len(interior_labels)
            n_h1 = np.sum(interior_labels == 1)
            n_h2 = np.sum(interior_labels == 2)
            n_unreached = np.sum(interior_labels == 0)
            
            # Count parting edges (yellow) - edges between different label vertices
            n_parting_edges = 0
            interior_set = set(interior_indices)
            vertex_labels = np.full(len(result.vertices), -1, dtype=np.int8)
            for i, idx in enumerate(interior_indices):
                vertex_labels[idx] = interior_labels[i]
            
            for v0, v1 in result.edges:
                if v0 in interior_set and v1 in interior_set:
                    l0 = vertex_labels[v0]
                    l1 = vertex_labels[v1]
                    if (l0 == 1 and l1 == 2) or (l0 == 2 and l1 == 1):
                        n_parting_edges += 1
            
            # Count vertices on part surface vs cavity interior
            n_on_part = 0
            n_in_cavity = 0
            if boundary_labels is not None:
                for idx in interior_indices:
                    if boundary_labels[idx] == -1:
                        n_on_part += 1
                    else:
                        n_in_cavity += 1
            
            self.dijkstra_stats.add_header('✅ Dijkstra Complete', Colors.SUCCESS)
            self.dijkstra_stats.add_row(f'Total interior vertices: {n_interior:,}')
            if n_on_part > 0 or n_in_cavity > 0:
                self.dijkstra_stats.add_row(f'  On part surface: {n_on_part:,}')
                self.dijkstra_stats.add_row(f'  In cavity: {n_in_cavity:,}')
            self.dijkstra_stats.add_row(f'Vertices → H₁ (green): {n_h1:,}')
            self.dijkstra_stats.add_row(f'Vertices → H₂ (orange): {n_h2:,}')
            self.dijkstra_stats.add_row(f'Parting edges (yellow): {n_parting_edges:,}')
            if n_unreached > 0:
                self.dijkstra_stats.add_row(f'Unreachable: {n_unreached:,}')
            
            if interior_distances is not None:
                # Show distance stats for reached vertices
                reached_mask = interior_labels != 0
                if np.any(reached_mask):
                    reached_dists = interior_distances[reached_mask]
                    self.dijkstra_stats.add_header('Path Distances', Colors.INFO)
                    self.dijkstra_stats.add_row(f'Min: {reached_dists.min():.4f}')
                    self.dijkstra_stats.add_row(f'Max: {reached_dists.max():.4f}')
                    self.dijkstra_stats.add_row(f'Mean: {reached_dists.mean():.4f}')
        else:
            self.dijkstra_stats.hide()
    
    def _on_run_dijkstra(self):
        """Handle run Dijkstra button click."""
        if self._dijkstra_worker is not None:
            return
        
        # Show progress
        self.dijkstra_btn.setEnabled(False)
        self.dijkstra_progress.show()
        self.dijkstra_progress_label.show()
        self.dijkstra_progress_label.setText("Initializing...")
        self.dijkstra_stats.hide()
        
        # Also set original tet mesh for visualization
        if self._tet_result.vertices_original is not None:
            self.mesh_viewer.set_original_tetrahedral_mesh(
                self._tet_result.vertices_original,
                self._tet_result.edges
            )
        
        # Start worker
        self._dijkstra_worker = DijkstraWorker(self._tet_result)
        self._dijkstra_worker.progress.connect(self._on_dijkstra_progress)
        self._dijkstra_worker.complete.connect(self._on_dijkstra_complete)
        self._dijkstra_worker.error.connect(self._on_dijkstra_error)
        self._dijkstra_worker.finished.connect(self._on_dijkstra_worker_finished)
        self._dijkstra_worker.start()
    
    def _on_dijkstra_progress(self, message: str):
        """Handle Dijkstra progress update."""
        if hasattr(self, 'dijkstra_progress_label') and self.dijkstra_progress_label is not None:
            try:
                self.dijkstra_progress_label.setText(message)
            except RuntimeError:
                pass
    
    def _on_dijkstra_complete(self, result: TetrahedralMeshResult):
        """Handle Dijkstra complete."""
        self._tet_result = result
        
        # Update stats
        self._update_dijkstra_step_ui()
        
        # Visualize result - use ORIGINAL vertices and boundary mesh for proper geometry
        vertices_for_viz = result.vertices_original if result.vertices_original is not None else result.vertices
        boundary_mesh_for_viz = result.boundary_mesh_original if result.boundary_mesh_original is not None else result.boundary_mesh
        
        # Show interior vertices colored by their escape labels as edges
        # Pass tet edges so we can show ALL interior edges (not just surface edges)
        self.mesh_viewer.set_dijkstra_result(
            vertices_for_viz,
            result.seed_vertex_indices,
            result.seed_escape_labels,
            boundary_mesh_for_viz,
            result.seed_distances,
            result.edges  # Pass tet edges to find all interior edges
        )
        
        # Show Dijkstra result option in display panel
        self.display_options.show_dijkstra_result_option(True)
        
        # Update step status
        self.step_buttons[Step.DIJKSTRA].set_status('completed')
        
        # Unlock secondary cuts step
        if Step.SECONDARY_CUTS in self.step_buttons:
            self.step_buttons[Step.SECONDARY_CUTS].set_status('ready')
        
        n_h1 = np.sum(result.seed_escape_labels == 1)
        n_h2 = np.sum(result.seed_escape_labels == 2)
        logger.info(f"Dijkstra complete: {n_h1} seeds→H1, {n_h2} seeds→H2")
    
    def _on_dijkstra_error(self, message: str):
        """Handle Dijkstra error."""
        if hasattr(self, 'dijkstra_progress_label') and self.dijkstra_progress_label is not None:
            try:
                self.dijkstra_progress_label.setText(f"Error: {message}")
                self.dijkstra_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            except RuntimeError:
                pass
        QMessageBox.critical(self, "Error", f"Dijkstra failed:\n{message}")
    
    def _on_dijkstra_worker_finished(self):
        """Handle Dijkstra worker finished."""
        if hasattr(self, 'dijkstra_btn') and self.dijkstra_btn is not None:
            try:
                self.dijkstra_btn.setEnabled(True)
            except RuntimeError:
                pass
        if hasattr(self, 'dijkstra_progress') and self.dijkstra_progress is not None:
            try:
                self.dijkstra_progress.hide()
            except RuntimeError:
                pass
        self._dijkstra_worker = None

    # =========================================================================
    # SECONDARY CUTS STEP
    # =========================================================================
    
    def _setup_secondary_cuts_step(self):
        """Setup the secondary cuts step UI."""
        # Check prerequisites
        if self._tet_result is None or self._tet_result.seed_escape_paths is None:
            no_dijkstra_label = QLabel("⚠️ Please run Dijkstra first.")
            no_dijkstra_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_dijkstra_label.setWordWrap(True)
            self.context_layout.addWidget(no_dijkstra_label)
            return
        
        # Description section
        info_group = QGroupBox("Secondary Cutting Edges")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "For ALL interior edges with the SAME escape label (both→H₁ or both→H₂), "
            "we form a 'membrane' bounded by:\n\n"
            "• The edge (vi, vj)\n"
            "• Escape paths vi→wi and vj→wj\n"
            "• Shortest path wi→wj on boundary\n\n"
            "If this membrane intersects the part mesh representation, the edge is a "
            "secondary cutting edge (shown in red). This identifies where the parting "
            "surface would cut through the part."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Info about current state
        result = self._tet_result
        state_info = StatsBox()
        state_info.add_header('Current State', Colors.INFO)
        
        if result.seed_escape_labels is not None:
            n_interior = len(result.seed_escape_labels)
            n_h1 = np.sum(result.seed_escape_labels == 1)
            n_h2 = np.sum(result.seed_escape_labels == 2)
            state_info.add_row(f'Interior vertices: {n_interior:,}')
            state_info.add_row(f'Vertices → H₁: {n_h1:,}')
            state_info.add_row(f'Vertices → H₂: {n_h2:,}')
        
        self.context_layout.addWidget(state_info)
        
        # Min intersection count section
        threshold_group = QGroupBox("Intersection Threshold")
        threshold_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        threshold_layout = QVBoxLayout(threshold_group)
        
        # Threshold description
        threshold_desc = QLabel(
            "Minimum segment-triangle intersections required.\n"
            "Higher = fewer false positives, may miss some cuts."
        )
        threshold_desc.setWordWrap(True)
        threshold_desc.setStyleSheet(f'color: {Colors.GRAY}; font-size: 11px;')
        threshold_layout.addWidget(threshold_desc)
        
        # Slider row
        slider_row = QHBoxLayout()
        
        slider_label_low = QLabel("1")
        slider_label_low.setStyleSheet(f'color: {Colors.GRAY}; font-size: 11px;')
        slider_row.addWidget(slider_label_low)
        
        self.secondary_cuts_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.secondary_cuts_threshold_slider.setRange(1, 50)
        self.secondary_cuts_threshold_slider.setValue(5)  # Default = 5 intersections required
        self.secondary_cuts_threshold_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {Colors.BORDER};
                height: 6px;
                background: {Colors.LIGHT};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.DANGER};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #c82333;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.DANGER};
                border-radius: 3px;
            }}
        """)
        self.secondary_cuts_threshold_slider.valueChanged.connect(self._on_threshold_changed)
        slider_row.addWidget(self.secondary_cuts_threshold_slider)
        
        slider_label_high = QLabel("50")
        slider_label_high.setStyleSheet(f'color: {Colors.GRAY}; font-size: 11px;')
        slider_row.addWidget(slider_label_high)
        
        threshold_layout.addLayout(slider_row)
        
        # Value display
        self.secondary_cuts_threshold_value = QLabel("Min intersections: 20")
        self.secondary_cuts_threshold_value.setStyleSheet(f'color: {Colors.DARK}; font-size: 12px; font-weight: 500;')
        self.secondary_cuts_threshold_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        threshold_layout.addWidget(self.secondary_cuts_threshold_value)
        
        # GPU acceleration checkbox
        from core.tetrahedral_mesh import CUDA_AVAILABLE
        self.secondary_cuts_gpu_checkbox = QCheckBox("🚀 Use GPU Acceleration (CUDA)")
        self.secondary_cuts_gpu_checkbox.setChecked(CUDA_AVAILABLE)  # Auto-enable if available
        self.secondary_cuts_gpu_checkbox.setEnabled(CUDA_AVAILABLE)  # Disable if not available
        
        if CUDA_AVAILABLE:
            gpu_tooltip = "CUDA GPU acceleration - significantly faster"
            checkbox_style = f'color: {Colors.SUCCESS}; font-size: 12px; font-weight: 500;'
        else:
            gpu_tooltip = "CUDA not available - using CPU"
            checkbox_style = f'color: {Colors.GRAY}; font-size: 12px;'
        
        self.secondary_cuts_gpu_checkbox.setToolTip(gpu_tooltip)
        self.secondary_cuts_gpu_checkbox.setStyleSheet(f"""
            QCheckBox {{
                {checkbox_style}
                padding: 4px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QCheckBox::indicator:unchecked {{
                border: 2px solid {Colors.BORDER};
                background: white;
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                border: 2px solid {Colors.SUCCESS};
                background: {Colors.SUCCESS};
                border-radius: 3px;
            }}
        """)
        threshold_layout.addWidget(self.secondary_cuts_gpu_checkbox)
        
        self.context_layout.addWidget(threshold_group)
        
        # Compute button
        self.secondary_cuts_btn = QPushButton("✂️ Find Secondary Cuts")
        self.secondary_cuts_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.secondary_cuts_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.DANGER};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.secondary_cuts_btn.clicked.connect(self._on_run_secondary_cuts)
        self.context_layout.addWidget(self.secondary_cuts_btn)
        
        # Progress bar
        self.secondary_cuts_progress = QProgressBar()
        self.secondary_cuts_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.DANGER};
                border-radius: 4px;
            }}
        """)
        self.secondary_cuts_progress.setTextVisible(False)
        self.secondary_cuts_progress.setRange(0, 0)  # Indeterminate
        self.secondary_cuts_progress.hide()
        self.context_layout.addWidget(self.secondary_cuts_progress)
        
        # Progress label
        self.secondary_cuts_progress_label = QLabel("")
        self.secondary_cuts_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.secondary_cuts_progress_label.hide()
        self.context_layout.addWidget(self.secondary_cuts_progress_label)
        
        # Stats (show if computed)
        self.secondary_cuts_stats = StatsBox()
        self.secondary_cuts_stats.hide()
        self.context_layout.addWidget(self.secondary_cuts_stats)
        
        # Update UI with current state
        self._update_secondary_cuts_step_ui()
    
    def _on_threshold_changed(self, value: int):
        """Handle threshold slider change."""
        if hasattr(self, 'secondary_cuts_threshold_value'):
            self.secondary_cuts_threshold_value.setText(f"Min intersections: {value}")
    
    def _update_secondary_cuts_step_ui(self):
        """Update secondary cuts step UI based on current state."""
        if not hasattr(self, 'secondary_cuts_stats'):
            return
        
        if self._tet_result is not None and self._tet_result.secondary_cut_edges is not None:
            # Show stats
            self.secondary_cuts_stats.clear()
            self.secondary_cuts_stats.show()
            
            n_secondary = len(self._tet_result.secondary_cut_edges)
            
            self.secondary_cuts_stats.add_header('✅ Secondary Cuts Complete', Colors.SUCCESS)
            self.secondary_cuts_stats.add_row(f'Secondary cut edges: {n_secondary:,}')
            
            if n_secondary > 0:
                self.secondary_cuts_stats.add_row('(Shown in red)')
        else:
            self.secondary_cuts_stats.hide()
    
    def _on_run_secondary_cuts(self):
        """Handle run secondary cuts button click."""
        if self._secondary_cuts_worker is not None:
            return
        
        if self._current_mesh is None:
            QMessageBox.warning(self, "Warning", "No part mesh loaded.")
            return
        
        # Clear previous secondary cuts visualization before re-running
        self.mesh_viewer.clear_secondary_cuts()
        
        # Get min_intersection_count from slider (1-50)
        min_intersection_count = 5
        if hasattr(self, 'secondary_cuts_threshold_slider'):
            min_intersection_count = self.secondary_cuts_threshold_slider.value()
        
        # Get GPU preference
        use_gpu = True
        if hasattr(self, 'secondary_cuts_gpu_checkbox'):
            use_gpu = self.secondary_cuts_gpu_checkbox.isChecked()
        
        # Show progress
        self.secondary_cuts_btn.setEnabled(False)
        self.secondary_cuts_progress.show()
        self.secondary_cuts_progress_label.show()
        self.secondary_cuts_progress_label.setText("Initializing...")
        self.secondary_cuts_stats.hide()
        
        # Start worker with min_intersection_count and GPU preference
        self._secondary_cuts_worker = SecondaryCutsWorker(
            self._tet_result, 
            self._current_mesh,
            min_intersection_count=min_intersection_count,
            use_gpu=use_gpu
        )
        self._secondary_cuts_worker.progress.connect(self._on_secondary_cuts_progress)
        self._secondary_cuts_worker.complete.connect(self._on_secondary_cuts_complete)
        self._secondary_cuts_worker.error.connect(self._on_secondary_cuts_error)
        self._secondary_cuts_worker.finished.connect(self._on_secondary_cuts_worker_finished)
        self._secondary_cuts_worker.start()
    
    def _on_secondary_cuts_progress(self, message: str):
        """Handle secondary cuts progress update."""
        if hasattr(self, 'secondary_cuts_progress_label') and self.secondary_cuts_progress_label is not None:
            try:
                self.secondary_cuts_progress_label.setText(message)
            except RuntimeError:
                pass
    
    def _on_secondary_cuts_complete(self, result: TetrahedralMeshResult):
        """Handle secondary cuts complete."""
        self._tet_result = result
        
        # Update stats
        self._update_secondary_cuts_step_ui()
        
        # Visualize result - show secondary cuts in red
        if result.secondary_cut_edges is not None and len(result.secondary_cut_edges) > 0:
            vertices_for_viz = result.vertices_original if result.vertices_original is not None else result.vertices
            self.mesh_viewer.set_secondary_cuts(
                vertices_for_viz,
                result.secondary_cut_edges
            )
            # Show secondary cuts option in display panel
            self.display_options.show_secondary_cuts_option(True)
        
        # Update step status
        self.step_buttons[Step.SECONDARY_CUTS].set_status('completed')
        
        # Unlock parting surface step
        if Step.PARTING_SURFACE in self.step_buttons:
            self.step_buttons[Step.PARTING_SURFACE].set_status('ready')
        
        n_secondary = len(result.secondary_cut_edges) if result.secondary_cut_edges else 0
        logger.info(f"Secondary cuts complete: {n_secondary} secondary cutting edges found")
    
    def _on_secondary_cuts_error(self, message: str):
        """Handle secondary cuts error."""
        if hasattr(self, 'secondary_cuts_progress_label') and self.secondary_cuts_progress_label is not None:
            try:
                self.secondary_cuts_progress_label.setText(f"Error: {message}")
                self.secondary_cuts_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            except RuntimeError:
                pass
        QMessageBox.critical(self, "Error", f"Secondary cuts failed:\n{message}")
    
    def _on_secondary_cuts_worker_finished(self):
        """Handle secondary cuts worker finished."""
        if hasattr(self, 'secondary_cuts_btn') and self.secondary_cuts_btn is not None:
            try:
                self.secondary_cuts_btn.setEnabled(True)
            except RuntimeError:
                pass
        if hasattr(self, 'secondary_cuts_progress') and self.secondary_cuts_progress is not None:
            try:
                self.secondary_cuts_progress.hide()
            except RuntimeError:
                pass
        self._secondary_cuts_worker = None

    # ========================================================================
    # PARTING SURFACE STEP
    # ========================================================================
    
    def _setup_parting_surface_step(self):
        """Setup the parting surface step UI."""
        # Check prerequisites
        if self._tet_result is None or self._tet_result.secondary_cut_edges is None:
            no_secondary_label = QLabel("⚠️ Please run Secondary Cuts first.")
            no_secondary_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_secondary_label.setWordWrap(True)
            self.context_layout.addWidget(no_secondary_label)
            return
        
        # Description section
        info_group = QGroupBox("Primary Parting Surface (Comprehensive)")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Generate and smooth the primary parting surface mesh.\n\n"
            "This step performs:\n"
            "• Extract surface using Marching Tetrahedra\n"
            "• Clean mesh (merge vertices, remove degenerates)\n"
            "• Smooth with boundary re-projection to part/hull"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Info about current state
        result = self._tet_result
        state_info = StatsBox()
        state_info.add_header('Current State', Colors.INFO)
        
        if result.seed_escape_paths is not None:
            n_paths = len(result.seed_escape_paths)
            state_info.add_row(f'Escape paths computed: {n_paths:,}')
        
        if result.secondary_cut_edges is not None:
            n_secondary = len(result.secondary_cut_edges)
            state_info.add_row(f'Secondary cut edges: {n_secondary:,}')
        
        if result.primary_cut_edges is not None:
            n_primary = len(result.primary_cut_edges)
            state_info.add_row(f'Primary cut edges: {n_primary:,}')
        
        self.context_layout.addWidget(state_info)
        
        # Smoothing parameters group
        smooth_group = QGroupBox("Surface Parameters")
        smooth_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {Colors.DARK};
                background-color: {Colors.LIGHT};
            }}
            QGroupBox QLabel {{
                color: {Colors.DARK};
                font-size: 12px;
            }}
        """)
        smooth_layout = QFormLayout(smooth_group)
        smooth_layout.setContentsMargins(12, 12, 12, 12)
        smooth_layout.setSpacing(10)
        
        # Smoothing iterations
        self.primary_smooth_iterations_spin = QSpinBox()
        self.primary_smooth_iterations_spin.setRange(0, 50)
        self.primary_smooth_iterations_spin.setValue(5)
        self.primary_smooth_iterations_spin.setToolTip("Number of Laplacian smoothing iterations (0 = no smoothing)")
        smooth_layout.addRow("Smooth iterations:", self.primary_smooth_iterations_spin)
        
        # Damping factor
        self.primary_smooth_damping_spin = QDoubleSpinBox()
        self.primary_smooth_damping_spin.setRange(0.01, 1.0)
        self.primary_smooth_damping_spin.setValue(0.5)
        self.primary_smooth_damping_spin.setSingleStep(0.05)
        self.primary_smooth_damping_spin.setToolTip("Damping factor (λ) - higher = more smoothing per iteration")
        smooth_layout.addRow("Damping (λ):", self.primary_smooth_damping_spin)
        
        self.context_layout.addWidget(smooth_group)
        
        # Compute button
        self.parting_surface_btn = QPushButton("🔲 Generate Parting Surface")
        self.parting_surface_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.parting_surface_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #0069d9;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.parting_surface_btn.clicked.connect(self._on_run_parting_surface)
        self.context_layout.addWidget(self.parting_surface_btn)
        
        # Progress bar
        self.parting_surface_progress = QProgressBar()
        self.parting_surface_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY};
                border-radius: 4px;
            }}
        """)
        self.parting_surface_progress.setTextVisible(False)
        self.parting_surface_progress.setRange(0, 0)  # Indeterminate
        self.parting_surface_progress.hide()
        self.context_layout.addWidget(self.parting_surface_progress)
        
        # Progress label
        self.parting_surface_progress_label = QLabel("")
        self.parting_surface_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.parting_surface_progress_label.hide()
        self.context_layout.addWidget(self.parting_surface_progress_label)
        
        # Stats (show if computed)
        self.parting_surface_stats = StatsBox()
        self.parting_surface_stats.hide()
        self.context_layout.addWidget(self.parting_surface_stats)
        
        # Update UI with current state
        self._update_parting_surface_step_ui()
    
    def _update_parting_surface_step_ui(self):
        """Update parting surface step UI based on current state."""
        if not hasattr(self, 'parting_surface_stats'):
            return
        
        # Enable/disable secondary parting surface button based on secondary cuts availability
        if hasattr(self, 'secondary_parting_surface_btn'):
            has_secondary = (self._tet_result is not None and 
                            self._tet_result.secondary_cut_edges is not None and
                            len(self._tet_result.secondary_cut_edges) > 0)
            self.secondary_parting_surface_btn.setEnabled(has_secondary)
            
            if has_secondary:
                n_secondary = len(self._tet_result.secondary_cut_edges)
                self.secondary_parting_surface_btn.setText(f"🔴 Generate Secondary Surface ({n_secondary} edges)")
            else:
                self.secondary_parting_surface_btn.setText("🔴 No Secondary Cuts Available")
    
    def _on_run_parting_surface(self):
        """Start parting surface generation (extract + clean + smooth)."""
        if self._tet_result is None:
            QMessageBox.warning(self, "Warning", "No tetrahedral mesh data available")
            return
        
        # Get parameters from UI
        smooth_iterations = getattr(self, 'primary_smooth_iterations_spin', None)
        smooth_iterations = smooth_iterations.value() if smooth_iterations else 5
        
        damping_factor = getattr(self, 'primary_smooth_damping_spin', None)
        damping_factor = damping_factor.value() if damping_factor else 0.5
        
        # Show progress
        self.parting_surface_btn.setEnabled(False)
        self.parting_surface_progress.show()
        self.parting_surface_progress_label.show()
        self.parting_surface_progress_label.setText("Preparing data structures...")
        self.parting_surface_stats.hide()
        
        # Start worker thread for PRIMARY parting surface
        # This does: extraction + cleanup + smoothing
        hull_mesh = self._hull_result.mesh if self._hull_result is not None else None
        
        self._parting_surface_worker = ComprehensivePrimarySurfaceWorker(
            self._tet_result,
            part_mesh=self._current_mesh,
            hull_mesh=hull_mesh,
            smooth_iterations=smooth_iterations,
            damping_factor=damping_factor
        )
        self._parting_surface_worker.progress.connect(self._on_parting_surface_progress)
        self._parting_surface_worker.complete.connect(self._on_parting_surface_complete)
        self._parting_surface_worker.error.connect(self._on_parting_surface_error)
        self._parting_surface_worker.start()
    
    def _on_parting_surface_progress(self, message: str):
        """Handle parting surface progress updates."""
        if hasattr(self, 'parting_surface_progress_label'):
            self.parting_surface_progress_label.setText(message)
    
    def _on_parting_surface_complete(self, result):
        """Handle comprehensive parting surface complete (extract + repair + smooth)."""
        # Store result - this is a ComprehensiveSurfaceResult
        self._parting_surface_result = result
        
        # Also store in _primary_smoothing_result for propagation step to use
        # (propagation uses _primary_smoothing_result.mesh for reference)
        self._primary_smoothing_result = result
        
        # Reset UI
        self.parting_surface_btn.setEnabled(True)
        self.parting_surface_progress.hide()
        self.parting_surface_progress_label.hide()
        
        if result.mesh is None or result.num_faces == 0:
            self.parting_surface_progress_label.setText("No surface generated")
            self.parting_surface_progress_label.show()
            self.parting_surface_progress_label.setStyleSheet(f'color: {Colors.WARNING}; font-size: 12px;')
            return
        
        # Update stats - show comprehensive info
        self.parting_surface_stats.clear()
        self.parting_surface_stats.add_header('Primary Parting Surface', Colors.SUCCESS)
        self.parting_surface_stats.add_row(f'Vertices: {result.num_vertices:,}')
        self.parting_surface_stats.add_row(f'Faces: {result.num_faces:,}')
        self.parting_surface_stats.add_row(f'Tets contributing: {result.num_tets_contributing:,}/{result.num_tets_processed:,}')
        
        # Show timing breakdown
        self.parting_surface_stats.add_row('')
        self.parting_surface_stats.add_row(f'⏱ Extraction: {result.extraction_time_ms:.0f}ms')
        self.parting_surface_stats.add_row(f'⏱ Cleanup: {result.repair_time_ms:.0f}ms')
        
        # Show gap filling stats if gaps were detected
        if result.gaps_detected > 0:
            self.parting_surface_stats.add_row(f'⏱ Gap filling: {result.gap_fill_time_ms:.0f}ms')
            self.parting_surface_stats.add_row(f'   Gaps detected: {result.gaps_detected:,}')
            self.parting_surface_stats.add_row(f'   Gaps filled: {result.gaps_filled:,}')
            self.parting_surface_stats.add_row(f'   Faces added: {result.fill_faces_added:,}')
        elif result.gap_fill_time_ms > 0:
            self.parting_surface_stats.add_row(f'⏱ Gap filling: {result.gap_fill_time_ms:.0f}ms (no gaps)')
        
        if result.smooth_iterations > 0:
            self.parting_surface_stats.add_row(f'⏱ Smoothing ({result.smooth_iterations} iter): {result.smoothing_time_ms:.0f}ms')
            self.parting_surface_stats.add_row(f'   Boundary verts: {result.boundary_vertices:,}')
            self.parting_surface_stats.add_row(f'   Interior verts: {result.interior_vertices:,}')
        
        self.parting_surface_stats.add_row('')
        self.parting_surface_stats.add_row(f'Total time: {result.total_time_ms:.0f}ms')
        self.parting_surface_stats.show()
        
        # Visualize the parting surface
        self._visualize_parting_surface(result)
        
        # Mark step complete and unlock secondary surface step
        if Step.PARTING_SURFACE in self.step_buttons:
            self.step_buttons[Step.PARTING_SURFACE].set_status('completed')
        if Step.SECONDARY_SURFACE in self.step_buttons:
            self.step_buttons[Step.SECONDARY_SURFACE].set_status('available')
    
    def _on_parting_surface_error(self, error_msg: str):
        """Handle parting surface generation error."""
        self.parting_surface_btn.setEnabled(True)
        self.parting_surface_progress.hide()
        self.parting_surface_progress_label.setText(f"Error: {error_msg}")
        self.parting_surface_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
        QMessageBox.critical(self, "Error", f"Parting surface generation failed:\n{error_msg}")
    
    def _visualize_parting_surface(self, result):
        """Visualize the parting surface mesh in the viewer."""
        if result.mesh is None:
            logger.warning("No parting surface mesh to visualize")
            return
        
        try:
            # Use the mesh viewer's set_parting_surface method
            # Pass fill_face_indices if available (to show in yellow)
            fill_face_indices = getattr(result, 'fill_face_indices', None)
            self.mesh_viewer.set_parting_surface(result.mesh, fill_face_indices=fill_face_indices)
            
            # Show display option for primary parting surface
            self.display_options.show_parting_surface_options(show_primary=True)
            
            logger.info(f"Parting surface visualized: {result.num_vertices} vertices, {result.num_faces} faces"
                       f"{f', {len(fill_face_indices)} fill faces in yellow' if fill_face_indices is not None else ''}")
                    
        except Exception as e:
            logger.exception(f"Error visualizing parting surface: {e}")
            QMessageBox.warning(
                self,
                "Visualization Warning",
                f"Parting surface generated but visualization failed:\n{str(e)}\n\n"
                f"Surface has {result.num_vertices:,} vertices and {result.num_faces:,} faces."
            )

    # =========================================================================
    # SECONDARY PARTING SURFACE METHODS
    # =========================================================================

    def _on_run_secondary_parting_surface(self):
        """Start secondary parting surface generation."""
        if self._tet_result is None:
            QMessageBox.warning(self, "Warning", "No tetrahedral mesh data available")
            return
        
        if (self._tet_result.secondary_cut_edges is None or 
            len(self._tet_result.secondary_cut_edges) == 0):
            QMessageBox.warning(self, "Warning", "No secondary cut edges available.\nThe mold may not have any secondary features.")
            return
        
        # Show progress
        self.secondary_parting_surface_btn.setEnabled(False)
        self.secondary_parting_surface_progress.show()
        self.secondary_parting_surface_progress_label.show()
        self.secondary_parting_surface_progress_label.setText("Preparing data structures...")
        self.secondary_parting_surface_stats.hide()
        
        # Start worker thread for SECONDARY parting surface
        self._secondary_parting_surface_worker = PartingSurfaceWorker(
            self._tet_result, 
            cut_type='secondary',
            part_mesh=self._current_mesh
        )
        self._secondary_parting_surface_worker.progress.connect(self._on_secondary_parting_surface_progress)
        self._secondary_parting_surface_worker.complete.connect(self._on_secondary_parting_surface_complete)
        self._secondary_parting_surface_worker.error.connect(self._on_secondary_parting_surface_error)
        self._secondary_parting_surface_worker.start()
    
    def _on_secondary_parting_surface_progress(self, message: str):
        """Handle secondary parting surface progress updates."""
        if hasattr(self, 'secondary_parting_surface_progress_label'):
            self.secondary_parting_surface_progress_label.setText(message)
    
    def _on_secondary_parting_surface_complete(self, result):
        """Handle secondary parting surface extraction complete."""
        from core.parting_surface import PartingSurfaceResult
        
        self._secondary_parting_surface_result = result
        
        # Reset UI
        self.secondary_parting_surface_btn.setEnabled(True)
        self.secondary_parting_surface_progress.hide()
        self.secondary_parting_surface_progress_label.hide()
        
        if result.mesh is None or result.num_faces == 0:
            self.secondary_parting_surface_progress_label.setText("No secondary surface generated")
            self.secondary_parting_surface_progress_label.show()
            self.secondary_parting_surface_progress_label.setStyleSheet(f'color: {Colors.WARNING}; font-size: 12px;')
            return
        
        # Update stats
        self.secondary_parting_surface_stats.clear()
        self.secondary_parting_surface_stats.add_header('Secondary Surface', '#dc3545')  # Red header
        self.secondary_parting_surface_stats.add_row(f'Vertices: {result.num_vertices:,}')
        self.secondary_parting_surface_stats.add_row(f'Faces: {result.num_faces:,}')
        self.secondary_parting_surface_stats.add_row(f'Tets contributing: {result.num_tets_contributing:,}/{result.num_tets_processed:,}')
        self.secondary_parting_surface_stats.add_row(f'Time: {result.extraction_time_ms:.1f} ms')
        self.secondary_parting_surface_stats.show()
        
        # Visualize the secondary parting surface
        self._visualize_secondary_parting_surface(result)
    
    def _on_secondary_parting_surface_error(self, error_msg: str):
        """Handle secondary parting surface generation error."""
        self.secondary_parting_surface_btn.setEnabled(True)
        self.secondary_parting_surface_progress.hide()
        self.secondary_parting_surface_progress_label.setText(f"Error: {error_msg}")
        self.secondary_parting_surface_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
        QMessageBox.critical(self, "Error", f"Secondary parting surface generation failed:\n{error_msg}")
    
    def _visualize_secondary_parting_surface(self, result):
        """Visualize the secondary parting surface mesh in the viewer."""
        if result.mesh is None:
            logger.warning("No secondary parting surface mesh to visualize")
            return
        
        try:
            # Use the mesh viewer's set_secondary_parting_surface method
            self.mesh_viewer.set_secondary_parting_surface(result.mesh)
            
            # Update display options to show secondary surface checkbox
            # Check if primary is also visible
            has_primary = self.mesh_viewer.has_parting_surface
            self.display_options.show_parting_surface_options(show_primary=has_primary, show_secondary=True)
            
            logger.info(f"Secondary parting surface visualized: {result.num_vertices} vertices, {result.num_faces} faces")
                    
        except Exception as e:
            logger.exception(f"Error visualizing secondary parting surface: {e}")
            QMessageBox.warning(
                self,
                "Visualization Warning",
                f"Secondary parting surface generated but visualization failed:\n{str(e)}\n\n"
                f"Surface has {result.num_vertices:,} vertices and {result.num_faces:,} faces."
            )

    # =========================================================================
    # COMPREHENSIVE SECONDARY SURFACE STEP
    # =========================================================================
    
    def _setup_secondary_surface_step(self):
        """Setup the comprehensive secondary surface step UI (generation + propagation + smoothing)."""
        # Check prerequisites - need primary surface
        has_primary = (self._primary_smoothing_result is not None and self._primary_smoothing_result.mesh is not None) or \
                      (self._parting_surface_result is not None and self._parting_surface_result.mesh is not None)
        has_secondary_cuts = (self._tet_result is not None and 
                             self._tet_result.secondary_cut_edges is not None and
                             len(self._tet_result.secondary_cut_edges) > 0)
        
        if not has_primary:
            no_primary_label = QLabel("⚠️ Please complete Primary Surface generation first.")
            no_primary_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(255, 180, 0, 0.1);
                border: 1px solid {Colors.WARNING};
                border-radius: 6px;
            """)
            no_primary_label.setWordWrap(True)
            self.context_layout.addWidget(no_primary_label)
            return
        
        if not has_secondary_cuts:
            no_cuts_label = QLabel("ℹ️ No secondary cut edges detected.\n\nThis part may not require secondary parting surfaces.")
            no_cuts_label.setStyleSheet(f"""
                color: {Colors.INFO};
                font-size: 13px;
                padding: 12px;
                background-color: rgba(0, 123, 255, 0.1);
                border: 1px solid {Colors.INFO};
                border-radius: 6px;
            """)
            no_cuts_label.setWordWrap(True)
            self.context_layout.addWidget(no_cuts_label)
            
            # Mark as complete since no secondary surface needed
            if Step.SECONDARY_SURFACE in self.step_buttons:
                self.step_buttons[Step.SECONDARY_SURFACE].set_status('completed')
            return
        
        # Description section
        info_group = QGroupBox("Secondary Parting Surface (Comprehensive)")
        info_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Generate, propagate, and smooth the secondary parting surface.\n\n"
            "This comprehensive step performs:\n"
            "• Extract secondary surface using Marching Tetrahedra\n"
            "• Remove isolated islands and extend boundaries\n"
            "• Smooth with boundary re-projection to primary/part/hull"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px; line-height: 1.4;')
        info_layout.addWidget(info_text)
        
        self.context_layout.addWidget(info_group)
        
        # Current state info
        state_info = StatsBox()
        state_info.add_header('Current State', Colors.INFO)
        
        n_secondary = len(self._tet_result.secondary_cut_edges)
        state_info.add_row(f'Secondary cut edges: {n_secondary:,}')
        
        if self._primary_smoothing_result is not None and self._primary_smoothing_result.mesh is not None:
            pri_mesh = self._primary_smoothing_result.mesh
            state_info.add_row(f'Primary surface: {len(pri_mesh.faces):,} faces (smoothed)')
        elif self._parting_surface_result is not None and self._parting_surface_result.mesh is not None:
            pri_mesh = self._parting_surface_result.mesh
            state_info.add_row(f'Primary surface: {len(pri_mesh.faces):,} faces')
        
        self.context_layout.addWidget(state_info)
        
        # Parameters group
        params_group = QGroupBox("Processing Parameters")
        params_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 500;
                color: {Colors.DARK};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {Colors.DARK};
                background-color: {Colors.LIGHT};
            }}
            QGroupBox QLabel {{
                color: {Colors.DARK};
                font-size: 12px;
            }}
        """)
        params_layout = QFormLayout(params_group)
        params_layout.setContentsMargins(12, 12, 12, 12)
        params_layout.setSpacing(10)
        
        # Min island size
        self.secondary_min_island_spin = QSpinBox()
        self.secondary_min_island_spin.setRange(1, 100)
        self.secondary_min_island_spin.setValue(3)
        self.secondary_min_island_spin.setToolTip("Minimum triangles in island to keep")
        params_layout.addRow("Min island size:", self.secondary_min_island_spin)
        
        # Max propagation distance
        self.secondary_max_dist_spin = QDoubleSpinBox()
        self.secondary_max_dist_spin.setRange(0.1, 100.0)
        self.secondary_max_dist_spin.setValue(10.0)
        self.secondary_max_dist_spin.setSingleStep(1.0)
        self.secondary_max_dist_spin.setToolTip("Maximum distance to extend boundaries")
        params_layout.addRow("Max distance:", self.secondary_max_dist_spin)
        
        # Smoothing iterations
        self.secondary_smooth_iterations_spin = QSpinBox()
        self.secondary_smooth_iterations_spin.setRange(0, 50)
        self.secondary_smooth_iterations_spin.setValue(5)
        self.secondary_smooth_iterations_spin.setToolTip("Number of Laplacian smoothing iterations (0 = no smoothing)")
        params_layout.addRow("Smooth iterations:", self.secondary_smooth_iterations_spin)
        
        # Damping factor
        self.secondary_smooth_damping_spin = QDoubleSpinBox()
        self.secondary_smooth_damping_spin.setRange(0.01, 1.0)
        self.secondary_smooth_damping_spin.setValue(0.5)
        self.secondary_smooth_damping_spin.setSingleStep(0.05)
        self.secondary_smooth_damping_spin.setToolTip("Damping factor (λ) - higher = more smoothing per iteration")
        params_layout.addRow("Damping (λ):", self.secondary_smooth_damping_spin)
        
        self.context_layout.addWidget(params_group)
        
        # Run button
        self.secondary_surface_btn = QPushButton("🔴 Generate Secondary Surface")
        self.secondary_surface_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.secondary_surface_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
            QPushButton:disabled {{
                background-color: {Colors.GRAY_LIGHT};
                color: {Colors.GRAY};
            }}
        """)
        self.secondary_surface_btn.clicked.connect(self._on_run_secondary_surface)
        self.context_layout.addWidget(self.secondary_surface_btn)
        
        # Progress bar
        self.secondary_surface_progress = QProgressBar()
        self.secondary_surface_progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: #dc3545;
                border-radius: 4px;
            }}
        """)
        self.secondary_surface_progress.setTextVisible(False)
        self.secondary_surface_progress.setRange(0, 0)  # Indeterminate
        self.secondary_surface_progress.hide()
        self.context_layout.addWidget(self.secondary_surface_progress)
        
        # Progress label
        self.secondary_surface_progress_label = QLabel("")
        self.secondary_surface_progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.secondary_surface_progress_label.hide()
        self.context_layout.addWidget(self.secondary_surface_progress_label)
        
        # Stats (show if computed)
        self.secondary_surface_stats = StatsBox()
        self.secondary_surface_stats.hide()
        self.context_layout.addWidget(self.secondary_surface_stats)
    
    def _on_run_secondary_surface(self):
        """Start comprehensive secondary surface generation."""
        if self._tet_result is None:
            QMessageBox.warning(self, "Warning", "No tetrahedral mesh data available")
            return
        
        if (self._tet_result.secondary_cut_edges is None or 
            len(self._tet_result.secondary_cut_edges) == 0):
            QMessageBox.warning(self, "Warning", "No secondary cut edges available")
            return
        
        # Get parameters from UI
        min_island = self.secondary_min_island_spin.value()
        max_dist = self.secondary_max_dist_spin.value()
        smooth_iterations = self.secondary_smooth_iterations_spin.value()
        damping_factor = self.secondary_smooth_damping_spin.value()
        
        # Get target surfaces
        primary_mesh = None
        if self._primary_smoothing_result is not None and self._primary_smoothing_result.mesh is not None:
            primary_mesh = self._primary_smoothing_result.mesh
        elif self._parting_surface_result is not None and self._parting_surface_result.mesh is not None:
            primary_mesh = self._parting_surface_result.mesh
        
        part_mesh = self._current_mesh
        hull_mesh = self._hull_result.mesh if self._hull_result is not None else None
        
        # Show progress
        self.secondary_surface_btn.setEnabled(False)
        self.secondary_surface_progress.show()
        self.secondary_surface_progress_label.show()
        self.secondary_surface_progress_label.setText("Starting secondary surface generation...")
        self.secondary_surface_stats.hide()
        
        # Start comprehensive worker
        self._secondary_surface_worker = ComprehensiveSecondarySurfaceWorker(
            self._tet_result,
            primary_mesh=primary_mesh,
            part_mesh=part_mesh,
            hull_mesh=hull_mesh,
            min_island_triangles=min_island,
            max_propagation_distance=max_dist,
            smooth_iterations=smooth_iterations,
            damping_factor=damping_factor
        )
        self._secondary_surface_worker.progress.connect(self._on_secondary_surface_progress)
        self._secondary_surface_worker.complete.connect(self._on_secondary_surface_complete)
        self._secondary_surface_worker.error.connect(self._on_secondary_surface_error)
        self._secondary_surface_worker.start()
    
    def _on_secondary_surface_progress(self, message: str):
        """Handle secondary surface progress updates."""
        try:
            if hasattr(self, 'secondary_surface_progress_label') and self.secondary_surface_progress_label is not None:
                self.secondary_surface_progress_label.setText(message)
        except RuntimeError:
            pass
    
    def _on_secondary_surface_complete(self, result):
        """Handle comprehensive secondary surface complete."""
        self._secondary_surface_result = result
        # Also store in old result variables for compatibility
        self._secondary_parting_surface_result = result
        self._propagation_result = result
        self._secondary_smoothing_result = result
        
        # Reset UI
        try:
            if hasattr(self, 'secondary_surface_btn') and self.secondary_surface_btn is not None:
                self.secondary_surface_btn.setEnabled(True)
            if hasattr(self, 'secondary_surface_progress') and self.secondary_surface_progress is not None:
                self.secondary_surface_progress.hide()
            if hasattr(self, 'secondary_surface_progress_label') and self.secondary_surface_progress_label is not None:
                self.secondary_surface_progress_label.hide()
        except RuntimeError:
            pass
        
        if result.mesh is None or result.num_faces == 0:
            try:
                if hasattr(self, 'secondary_surface_progress_label') and self.secondary_surface_progress_label is not None:
                    self.secondary_surface_progress_label.setText("No secondary surface generated")
                    self.secondary_surface_progress_label.show()
                    self.secondary_surface_progress_label.setStyleSheet(f'color: {Colors.WARNING}; font-size: 12px;')
            except RuntimeError:
                pass
            return
        
        # Update stats - show comprehensive info
        try:
            if hasattr(self, 'secondary_surface_stats') and self.secondary_surface_stats is not None:
                self.secondary_surface_stats.clear()
                self.secondary_surface_stats.add_header('Secondary Parting Surface', '#dc3545')
                self.secondary_surface_stats.add_row(f'Vertices: {result.num_vertices:,}')
                self.secondary_surface_stats.add_row(f'Faces: {result.num_faces:,}')
                self.secondary_surface_stats.add_row('')
                self.secondary_surface_stats.add_row(f'⏱ Extraction: {result.extraction_time_ms:.0f}ms')
                self.secondary_surface_stats.add_row(f'   Initial: {result.extraction_faces:,} faces')
                self.secondary_surface_stats.add_row(f'⏱ Propagation: {result.propagation_time_ms:.0f}ms')
                self.secondary_surface_stats.add_row(f'   Islands removed: {result.islands_removed}')
                self.secondary_surface_stats.add_row(f'   Vertices extended: {result.boundary_vertices_extended}')
                
                if result.smooth_iterations > 0:
                    self.secondary_surface_stats.add_row(f'⏱ Smoothing ({result.smooth_iterations} iter): {result.smoothing_time_ms:.0f}ms')
                    self.secondary_surface_stats.add_row(f'   Boundary verts: {result.boundary_vertices:,}')
                    self.secondary_surface_stats.add_row(f'   Interior verts: {result.interior_vertices:,}')
                
                self.secondary_surface_stats.add_row('')
                self.secondary_surface_stats.add_row(f'Total time: {result.total_time_ms:.0f}ms')
                self.secondary_surface_stats.show()
        except RuntimeError:
            pass
        
        # Visualize the secondary surface
        try:
            self.mesh_viewer.set_secondary_parting_surface(result.mesh)
            
            # Ensure display options show secondary surface
            has_primary = self.mesh_viewer.has_parting_surface
            self.display_options.show_parting_surface_options(show_primary=has_primary, show_secondary=True)
            
            logger.info(f"Secondary surface visualized: {result.num_vertices} vertices, {result.num_faces} faces")
        except Exception as e:
            logger.exception(f"Error visualizing secondary surface: {e}")
        
        # Mark step complete
        if Step.SECONDARY_SURFACE in self.step_buttons:
            self.step_buttons[Step.SECONDARY_SURFACE].set_status('completed')
    
    def _on_secondary_surface_error(self, error_msg: str):
        """Handle secondary surface generation error."""
        try:
            if hasattr(self, 'secondary_surface_btn') and self.secondary_surface_btn is not None:
                self.secondary_surface_btn.setEnabled(True)
            if hasattr(self, 'secondary_surface_progress') and self.secondary_surface_progress is not None:
                self.secondary_surface_progress.hide()
            if hasattr(self, 'secondary_surface_progress_label') and self.secondary_surface_progress_label is not None:
                self.secondary_surface_progress_label.setText(f"Error: {error_msg}")
                self.secondary_surface_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
        except RuntimeError:
            pass
        QMessageBox.critical(self, "Error", f"Secondary surface generation failed:\n{error_msg}")

    def _load_mesh(self, file_path: str, scale_factor: float = 1.0):
        """Load and process a mesh file."""
        self._loaded_filename = file_path
        self._current_scale_factor = scale_factor
        
        # Update UI
        self.drop_zone.set_loaded(Path(file_path).name)
        self.progress_bar.show()
        self.progress_label.setText('Loading...')
        self.progress_label.setStyleSheet(f'color: {Colors.GRAY}; font-size: 12px;')
        self.mesh_stats.hide()
        
        # Clear previous mesh
        self.mesh_viewer.clear()
        self.display_options.hide()
        
        # Start worker thread
        self._worker = MeshLoadWorker(file_path, scale_factor)
        self._worker.progress.connect(self._on_progress)
        self._worker.load_complete.connect(self._on_load_complete)
        self._worker.analysis_complete.connect(self._on_analysis_complete)
        self._worker.repair_complete.connect(self._on_repair_complete)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
    
    def _on_progress(self, message: str):
        self.progress_label.setText(message)
    
    def _on_load_complete(self, result: LoadResult):
        self._current_mesh = result.mesh
        self.mesh_viewer.set_mesh(result.mesh)
        
        # Update title bar with file info
        if self._loaded_filename:
            filename = Path(self._loaded_filename).name
            triangle_count = len(result.mesh.faces)
            
            # Get file size
            try:
                file_size = Path(self._loaded_filename).stat().st_size
            except:
                file_size = 0
            
            self.title_bar.set_file_info(filename, triangle_count, file_size)
        
        # Reset display options and show panel
        self.display_options.hide_mesh_cb.setChecked(False)
        self.display_options.show_parting_option(False)  # Hide parting option until computed
        self._visibility_paint_data = None  # Clear previous paint data
        self.display_options.show()
        self.display_options.raise_()  # Bring to front
    
    def _on_analysis_complete(self, diagnostics: MeshDiagnostics):
        self._current_diagnostics = diagnostics
        self._update_mesh_stats()
        
        # Update step status - import complete, unlock parting
        self.step_buttons[Step.IMPORT].set_status('completed')
        if Step.PARTING in self.step_buttons:
            self.step_buttons[Step.PARTING].set_status('available')
    
    def _on_repair_complete(self, result: MeshRepairResult):
        self._repair_result = result
        self._current_mesh = result.mesh
        self._current_diagnostics = result.diagnostics
        
        # Only change mesh color if actual repairs were made
        if result.was_repaired:
            # Update viewer with repaired mesh (green tint to indicate changes)
            self.mesh_viewer.set_mesh(result.mesh, color='#66ff99')
            
            # Update title bar with new triangle count
            if self._loaded_filename:
                filename = Path(self._loaded_filename).name
                triangle_count = len(result.mesh.faces)
                try:
                    file_size = Path(self._loaded_filename).stat().st_size
                except:
                    file_size = 0
                self.title_bar.set_file_info(filename, triangle_count, file_size)
        
        self._update_mesh_stats()
    
    def _on_error(self, message: str):
        self.progress_label.setText(f'Error: {message}')
        self.progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
        QMessageBox.critical(self, 'Error', message)
    
    def _on_worker_finished(self):
        self.progress_bar.hide()
        self._worker = None
        
        # Check if mesh needs decimation
        self._check_and_offer_decimation()
    
    def _check_and_offer_decimation(self):
        """Check if the mesh has too many triangles and offer decimation."""
        if self._current_mesh is None:
            return
        
        triangle_count = len(self._current_mesh.faces)
        recommendation = get_decimation_recommendation(triangle_count)
        
        logger.info(f"Triangle count: {triangle_count:,}, recommendation: {recommendation['severity']}")
        
        # Only show dialog if decimation is recommended
        if not recommendation['needs_decimation']:
            logger.info("Mesh triangle count is optimal, no decimation needed")
            return
        
        # Show decimation dialog
        dialog = MeshDecimationDialog(triangle_count, recommendation, self)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted and dialog.should_decimate:
            # User chose to decimate
            self._perform_decimation(dialog.target_triangles)
        else:
            # User chose to keep original mesh
            logger.info("User chose to keep original mesh without decimation")
    
    def _perform_decimation(self, target_triangles: int):
        """Perform mesh decimation with the specified target."""
        if self._current_mesh is None:
            return
        
        logger.info(f"Starting mesh decimation to {target_triangles:,} triangles...")
        self.progress_bar.show()
        self.progress_label.setText("Decimating mesh...")
        
        try:
            decimator = MeshDecimator(self._current_mesh)
            result = decimator.decimate(target_triangles=target_triangles)
            
            if result.was_decimated:
                # Update current mesh with decimated version
                self._current_mesh = result.mesh
                
                # Re-analyze the decimated mesh
                analyzer = MeshAnalyzer(result.mesh)
                self._current_diagnostics = analyzer.analyze()
                
                # Update viewer with decimated mesh (blue color for decimated)
                self.mesh_viewer.set_mesh(result.mesh, color='#00aaff')
                
                # Update stats display
                self._update_mesh_stats()
                
                logger.info(f"Decimation complete: {result.original_triangle_count:,} -> {result.final_triangle_count:,} triangles ({result.reduction_percentage:.1f}% reduction)")
                self.progress_label.setText(f"Decimated: {result.reduction_percentage:.1f}% reduction")
                self.progress_label.setStyleSheet(f'color: {Colors.SUCCESS}; font-size: 12px;')
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Mesh Optimized",
                    f"Mesh successfully decimated!\n\n"
                    f"Original: {result.original_triangle_count:,} triangles\n"
                    f"Optimized: {result.final_triangle_count:,} triangles\n"
                    f"Reduction: {result.reduction_percentage:.1f}%"
                )
            else:
                error_msg = result.error_message or "Unknown error during decimation"
                logger.error(f"Decimation failed: {error_msg}")
                self.progress_label.setText(f"Decimation failed")
                self.progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
                QMessageBox.warning(self, "Decimation Failed", error_msg)
                
        except Exception as e:
            logger.exception(f"Decimation error: {e}")
            self.progress_label.setText(f"Decimation error")
            self.progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
            QMessageBox.critical(self, "Error", f"Decimation failed: {str(e)}")
        finally:
            self.progress_bar.hide()
