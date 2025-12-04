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

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QLabel, QProgressBar, QPushButton, 
    QMessageBox, QFileDialog, QScrollArea, QCheckBox,
    QPlainTextEdit, QSplitter, QDialog, QSlider, QSpinBox,
    QRadioButton, QButtonGroup, QGroupBox
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
from viewer import MeshViewer

logger = logging.getLogger(__name__)


# ============================================================================
# STEP DEFINITIONS (matching React frontend)
# ============================================================================

class Step(Enum):
    IMPORT = 'import'
    # Future steps will be added here as we implement more features
    # PARTING = 'parting'
    # HULL = 'hull'
    # CAVITY = 'cavity'
    # MOLD_HALVES = 'mold-halves'
    # VOXEL = 'voxel'
    # PARTING_SURFACE = 'parting-surface'


STEPS = [
    {'id': Step.IMPORT, 'icon': '📁', 'title': 'Import STL', 'description': 'Load a 3D model file in STL format for mold analysis'},
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
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
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
                repair_result = repairer.repair()
                logger.info(f"Repair complete: was_repaired={repair_result.was_repaired}")
                self.repair_complete.emit(repair_result)
                self.progress.emit("Mesh processing complete")
            else:
                logger.info("Mesh is valid, no repair needed")
                self.progress.emit("Mesh is valid, no repair needed")
                
        except Exception as e:
            logger.exception(f"Error in mesh processing: {e}")
            self.error.emit(str(e))


# ============================================================================
# STYLED WIDGETS
# ============================================================================

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
        
        if self._is_active:
            bg_color = 'rgba(0, 123, 255, 0.1)'
            border_color = Colors.PRIMARY
        elif self._status == 'completed':
            bg_color = 'rgba(23, 198, 113, 0.08)'
            border_color = Colors.SUCCESS
        elif self._status == 'needs-recalc':
            bg_color = 'rgba(255, 180, 0, 0.08)'
            border_color = Colors.WARNING
        
        opacity = '0.4' if self._status == 'locked' else '1'
        
        # Build the display text
        display_text = self.icon_text
        if self._status == 'completed':
            display_text += '\n✓'
        elif self._status == 'needs-recalc':
            display_text += '\n⟳'
        
        self.setText(display_text)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: none;
                border-left: 3px solid {border_color};
                color: {Colors.DARK};
                font-size: 20px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: rgba(0, 123, 255, 0.05);
            }}
        """)
        self.setToolTip(self.title)


class StatsBox(QFrame):
    """A statistics display box matching the frontend's statsBox style."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
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
        label.setStyleSheet(f"""
            color: {color or Colors.GRAY_DARK};
            font-size: 12px;
            background: transparent;
            border: none;
            line-height: 1.7;
        """)
        self._layout.addWidget(label)
    
    def add_header(self, text: str, color: str = None):
        label = QLabel(text)
        label.setStyleSheet(f"""
            color: {color or Colors.DARK};
            font-size: 12px;
            font-weight: 500;
            background: transparent;
            border: none;
            margin-bottom: 6px;
        """)
        self._layout.addWidget(label)


class DropZone(QFrame):
    """File drop zone widget matching the frontend's upload style."""
    
    file_dropped = pyqtSignal(str)
    
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
            self.file_dropped.emit(file_path)
    
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
                    self.file_dropped.emit(file_path)
        self._update_style()


class DisplayOptionsPanel(QFrame):
    """Floating display options panel for the viewer."""
    
    hide_mesh_changed = pyqtSignal(bool)
    
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
        
        # Add default checkbox
        self.hide_mesh_cb = QCheckBox('Hide Original Mesh')
        self.hide_mesh_cb.stateChanged.connect(lambda s: self.hide_mesh_changed.emit(s == Qt.CheckState.Checked.value))
        layout.addWidget(self.hide_mesh_cb)
        
        self.adjustSize()
        self.hide()


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
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central)
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
        
        # Update initial state
        self.step_buttons[Step.IMPORT].set_active(True)
        
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
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)
        
        self.context_content = QWidget()
        self.context_layout = QVBoxLayout(self.context_content)
        self.context_layout.setContentsMargins(20, 20, 20, 20)
        self.context_layout.setSpacing(16)
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
    
    def _on_step_clicked(self, step: Step):
        """Handle step button click."""
        for s, btn in self.step_buttons.items():
            btn.set_active(s == step)
        
        self._active_step = step
        self._update_context_panel()
    
    def _on_file_dropped(self, file_path: str):
        """Handle file drop/selection."""
        self._load_mesh(file_path)
    
    def _on_hide_mesh_changed(self, hide: bool):
        """Handle hide mesh checkbox change."""
        if self.mesh_viewer:
            self.mesh_viewer.set_mesh_visible(not hide)
            logger.debug(f"Mesh visibility changed: hide={hide}")
    
    def _load_mesh(self, file_path: str):
        """Load and process a mesh file."""
        self._loaded_filename = file_path
        
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
        self._worker = MeshLoadWorker(file_path)
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
        
        # Reset display options and show panel
        self.display_options.hide_mesh_cb.setChecked(False)
        self.display_options.show()
        self.display_options.raise_()  # Bring to front
    
    def _on_analysis_complete(self, diagnostics: MeshDiagnostics):
        self._current_diagnostics = diagnostics
        self._update_mesh_stats()
        
        # Update step status
        self.step_buttons[Step.IMPORT].set_status('completed')
    
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
