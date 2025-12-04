"""
Main Window

Main application window for the VcMoldCreator desktop app.
Designed to match the React frontend's UI/UX with Shards Dashboard theme.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum
from datetime import datetime

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
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
    PartingColors,
)
from core.inflated_hull import (
    generate_inflated_hull,
    compute_default_offset,
    InflatedHullResult,
    ManifoldValidation,
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
    # Future steps will be added here as we implement more features
    # CAVITY = 'cavity'
    # MOLD_HALVES = 'mold-halves'
    # VOXEL = 'voxel'
    # PARTING_SURFACE = 'parting-surface'


STEPS = [
    {'id': Step.IMPORT, 'icon': '📁', 'title': 'Import STL', 'description': 'Load a 3D model file in STL format for mold analysis'},
    {'id': Step.PARTING, 'icon': '🔀', 'title': 'Parting Direction', 'description': 'Compute optimal parting directions for mold separation'},
    {'id': Step.HULL, 'icon': '📦', 'title': 'Bounding Hull', 'description': 'Generate inflated convex hull for mold cavity creation'},
    # Future steps will be added here
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
            
            logger.info("Analyzing mesh...")
            self.progress.emit("Analyzing mesh...")
            analyzer = MeshAnalyzer(result.mesh)
            diagnostics = analyzer.analyze()
            logger.info(f"Analysis complete: manifold={diagnostics.is_manifold}, watertight={diagnostics.is_watertight}")
            self.analysis_complete.emit(diagnostics)
            
            if self._should_repair and not diagnostics.is_watertight:
                logger.info("Repairing mesh...")
                self.progress.emit("Repairing mesh...")
                repairer = MeshRepairer(result.mesh)
                # IMPORTANT: Disable convex hull fallback to preserve original geometry
                # Convex hull would destroy the original shape of non-watertight meshes
                repair_result = repairer.repair(use_convex_hull_fallback=False)
                logger.info(f"Repair complete: was_repaired={repair_result.was_repaired}")
                self.repair_complete.emit(repair_result)
                self.progress.emit("Mesh processing complete")
            else:
                logger.info("Mesh is valid, no repair needed")
                self.progress.emit("Mesh is valid, no repair needed")
                
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


# ============================================================================
# STYLED WIDGETS
# ============================================================================

class TitleBar(QFrame):
    """Custom title bar with logo, file info, and view controls matching the React frontend."""
    
    # Signals
    reset_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    redo_clicked = pyqtSignal()
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
    
    def _create_tool_button(self, icon: str, tooltip: str) -> QPushButton:
        """Create a tool button (legacy, for compatibility)."""
        return self._create_icon_button(icon, tooltip)
    
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


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    """Main application window - matches React frontend layout."""
    
    def __init__(self):
        super().__init__()
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
        
        self._setup_window()
        self._setup_ui()
    
    def _setup_window(self):
        self.setWindowTitle("VcMoldCreator - Mold Analysis Tool")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set light background for the app (matching Shards theme)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Colors.LIGHT};
            }}
        """)
    
    def _setup_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main vertical layout (title bar on top, content below)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        
        # Title bar at the top
        self.title_bar = TitleBar()
        self.title_bar.reset_clicked.connect(self._on_reset_all)
        self.title_bar.undo_clicked.connect(self._on_undo)
        self.title_bar.redo_clicked.connect(self._on_redo)
        self.title_bar.view_changed.connect(self._on_view_changed)
        outer_layout.addWidget(self.title_bar)
        
        # Content area (sidebar + context + viewer)
        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Column 1: Steps Sidebar (narrow)
        self.steps_sidebar = self._create_steps_sidebar()
        main_layout.addWidget(self.steps_sidebar)
        
        # Column 2: Context Panel (options)
        self.context_panel = self._create_context_panel()
        main_layout.addWidget(self.context_panel)
        
        # Column 3: 3D Viewer (main area)
        self.viewer_container = self._create_viewer_container()
        main_layout.addWidget(self.viewer_container, 1)
        
        outer_layout.addWidget(content_widget, 1)
    
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
            self.mesh_stats.add_row(f'Repaired: {self._repair_result.repair_method}', Colors.PRIMARY)
    
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
        info_label = QLabel(f"Default: 20% of bounding box diagonal ({default_offset:.2f})")
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
        
        # Visibility toggles (show after calculation)
        self.hull_visibility_group = QGroupBox("Display Options")
        self.hull_visibility_group.setStyleSheet(f"""
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
        vis_layout = QVBoxLayout(self.hull_visibility_group)
        
        self.hull_visible_check = QCheckBox("Show Inflated Hull")
        self.hull_visible_check.setChecked(True)
        self.hull_visible_check.setStyleSheet(f'color: {Colors.DARK}; font-size: 12px;')
        self.hull_visible_check.stateChanged.connect(self._on_hull_visibility_changed)
        vis_layout.addWidget(self.hull_visible_check)
        
        self.original_hull_visible_check = QCheckBox("Show Original Hull (wireframe)")
        self.original_hull_visible_check.setChecked(False)
        self.original_hull_visible_check.setStyleSheet(f'color: {Colors.DARK}; font-size: 12px;')
        self.original_hull_visible_check.stateChanged.connect(self._on_original_hull_visibility_changed)
        vis_layout.addWidget(self.original_hull_visible_check)
        
        self.hull_visibility_group.hide()
        self.context_layout.addWidget(self.hull_visibility_group)
        
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
            
            # Show visibility toggles
            self.hull_visibility_group.show()
            
            # Update button text
            self.hull_calc_btn.setText("🔄 Regenerate Hull")
        else:
            self.hull_stats.hide()
            self.hull_visibility_group.hide()
    
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
        self.hull_progress_label.setText(message)
    
    def _on_hull_complete(self, result: InflatedHullResult):
        """Handle hull computation complete."""
        self._hull_result = result
        
        # Add hull to viewer
        self.mesh_viewer.set_hull_mesh(result.mesh, result.original_hull)
        
        # Update UI
        self._update_hull_step_ui()
        
        # Update step status
        self.step_buttons[Step.HULL].set_status('completed')
        
        logger.info(f"Hull generated: {result.vertex_count} vertices, {result.face_count} faces")
    
    def _on_hull_error(self, message: str):
        """Handle hull computation error."""
        self.hull_progress_label.setText(f"Error: {message}")
        self.hull_progress_label.setStyleSheet(f'color: {Colors.DANGER}; font-size: 12px;')
        QMessageBox.critical(self, "Error", f"Hull generation failed:\n{message}")
    
    def _on_hull_worker_finished(self):
        """Handle hull worker finished."""
        self.hull_calc_btn.setEnabled(True)
        self.hull_progress.hide()
        self._hull_worker = None
    
    def _on_hull_visibility_changed(self, state: int):
        """Handle hull visibility checkbox change."""
        visible = state == Qt.CheckState.Checked.value
        self.mesh_viewer.set_hull_visible(visible)
    
    def _on_original_hull_visibility_changed(self, state: int):
        """Handle original hull visibility checkbox change."""
        visible = state == Qt.CheckState.Checked.value
        self.mesh_viewer.set_original_hull_visible(visible)

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
        
        # Clear parting results
        self._parting_result = None
        self._visibility_paint_data = None
        
        # Clear hull results
        self._hull_result = None
        
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
        self._active_step = Step.IMPORT
        
        # Reset title bar
        self.title_bar.clear_file_info()
        
        # Reset display options
        self.display_options.hide()
        self.display_options.show_parting_option(False)
        
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
            if hide:
                self.mesh_viewer.remove_visibility_paint()
            else:
                # Re-apply visibility paint if we have the data
                if self._visibility_paint_data is not None:
                    face_colors = get_face_colors(self._visibility_paint_data)
                    self.mesh_viewer.apply_visibility_paint(face_colors)
            
            # Toggle parting direction arrows (inverted)
            self.mesh_viewer.set_parting_arrows_visible(not hide)
            logger.debug(f"Parting analysis visibility changed: hide={hide}")
    
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
        
        # Update viewer with repaired mesh (green tint)
        self.mesh_viewer.set_mesh(result.mesh, color='#66ff99')
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
