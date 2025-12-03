"""
Main Window

Main application window for the VcMoldCreator desktop app.
"""

import logging
from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QTextEdit, QLabel,
    QProgressBar, QPushButton, QMessageBox, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ui.file_upload import FileUploadWidget
from viewer import MeshViewer
from core import (
    STLLoader, LoadResult,
    MeshAnalyzer, MeshDiagnostics,
    MeshRepairer, MeshRepairResult
)

logger = logging.getLogger(__name__)


class MeshLoadWorker(QThread):
    """Background worker for loading and processing meshes."""
    
    # Signals
    progress = pyqtSignal(str)  # Progress message
    load_complete = pyqtSignal(object)  # LoadResult
    analysis_complete = pyqtSignal(object)  # MeshDiagnostics
    repair_complete = pyqtSignal(object)  # MeshRepairResult
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._should_repair = True
    
    def set_repair(self, should_repair: bool):
        """Set whether to attempt mesh repair."""
        self._should_repair = should_repair
    
    def run(self):
        """Run the mesh loading and processing."""
        try:
            # Load STL
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
            
            # Analyze mesh
            logger.info("Analyzing mesh...")
            self.progress.emit("Analyzing mesh...")
            analyzer = MeshAnalyzer(result.mesh)
            diagnostics = analyzer.analyze()
            logger.info(f"Analysis complete: manifold={diagnostics.is_manifold}, watertight={diagnostics.is_watertight}")
            self.analysis_complete.emit(diagnostics)
            
            # Repair if needed
            if self._should_repair and not diagnostics.is_watertight:
                logger.info("Repairing mesh...")
                self.progress.emit("Repairing mesh...")
                repairer = MeshRepairer(result.mesh)
                repair_result = repairer.repair()
                logger.info(f"Repair complete: was_repaired={repair_result.was_repaired}, method={repair_result.repair_method}")
                self.repair_complete.emit(repair_result)
                self.progress.emit("Mesh processing complete")
            else:
                logger.info("Mesh is valid, no repair needed")
                self.progress.emit("Mesh is valid, no repair needed")
                
        except Exception as e:
            logger.exception(f"Error in mesh processing: {e}")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Layout:
    - Left panel: File upload, mesh info, controls
    - Right panel: 3D mesh viewer
    """
    
    def __init__(self):
        super().__init__()
        self._current_mesh = None
        self._current_diagnostics: Optional[MeshDiagnostics] = None
        self._worker: Optional[MeshLoadWorker] = None
        
        self._setup_window()
        self._setup_ui()
        self._setup_statusbar()
    
    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle("VcMoldCreator - Mold Analysis Tool")
        self.setMinimumSize(1200, 800)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0066cc;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0077ee;
            }
            QPushButton:pressed {
                background-color: #0055aa;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
            QProgressBar {
                border: 1px solid #333;
                border-radius: 4px;
                background-color: #2a2a2a;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
                border-radius: 3px;
            }
        """)
    
    def _setup_ui(self):
        """Set up the main UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout with splitter
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel (controls)
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (3D viewer)
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([350, 850])
    
    def _create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # File upload section
        upload_group = QGroupBox("File")
        upload_layout = QVBoxLayout(upload_group)
        
        self.file_upload = FileUploadWidget()
        self.file_upload.file_selected.connect(self._on_file_selected)
        upload_layout.addWidget(self.file_upload)
        
        layout.addWidget(upload_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color: #888;")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        # Mesh info section
        info_group = QGroupBox("Mesh Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("Load an STL file to see mesh information...")
        self.info_text.setMinimumHeight(250)
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # Actions section
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.repair_btn = QPushButton("Repair Mesh")
        self.repair_btn.setEnabled(False)
        self.repair_btn.clicked.connect(self._on_repair_clicked)
        actions_layout.addWidget(self.repair_btn)
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._on_reset_view)
        actions_layout.addWidget(self.reset_btn)
        
        layout.addWidget(actions_group)
        
        # Spacer
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right 3D viewer panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D Viewer
        self.mesh_viewer = MeshViewer()
        layout.addWidget(self.mesh_viewer)
        
        return panel
    
    def _setup_statusbar(self):
        """Set up the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")
    
    def _on_file_selected(self, file_path: str):
        """Handle file selection."""
        self._load_mesh(file_path)
    
    def _load_mesh(self, file_path: str):
        """Load and process a mesh file."""
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setText("Loading...")
        self.repair_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        # Clear previous mesh
        self.mesh_viewer.clear()
        self.info_text.clear()
        
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
        """Handle progress updates."""
        self.progress_label.setText(message)
        self.statusbar.showMessage(message)
    
    def _on_load_complete(self, result: LoadResult):
        """Handle successful mesh load."""
        self._current_mesh = result.mesh
        
        # Display mesh
        self.mesh_viewer.set_mesh(result.mesh)
        self.reset_btn.setEnabled(True)
        
        # Update info
        info = f"File: {result.file_name}\n"
        info += f"Size: {result.file_size_bytes / 1024:.1f} KB\n"
        info += f"Load time: {result.load_time_ms:.1f} ms\n"
        info += "-" * 30 + "\n"
        self.info_text.setText(info)
    
    def _on_analysis_complete(self, diagnostics: MeshDiagnostics):
        """Handle mesh analysis completion."""
        self._current_diagnostics = diagnostics
        
        # Append diagnostics to info
        current_text = self.info_text.toPlainText()
        self.info_text.setText(current_text + diagnostics.format())
        
        # Enable repair button if mesh needs repair
        if not diagnostics.is_watertight:
            self.repair_btn.setEnabled(True)
            self.repair_btn.setText("Repair Mesh (Recommended)")
            self.repair_btn.setStyleSheet("""
                QPushButton {
                    background-color: #cc6600;
                }
                QPushButton:hover {
                    background-color: #ee7700;
                }
            """)
        else:
            self.repair_btn.setEnabled(False)
            self.repair_btn.setText("Mesh is Valid ✓")
            self.repair_btn.setStyleSheet("""
                QPushButton {
                    background-color: #006633;
                }
            """)
    
    def _on_repair_complete(self, result: MeshRepairResult):
        """Handle mesh repair completion."""
        self._current_mesh = result.mesh
        self._current_diagnostics = result.diagnostics
        
        # Update viewer with repaired mesh
        self.mesh_viewer.set_mesh(result.mesh, color="#66ff99")  # Green tint for repaired
        
        # Update info
        repair_info = "\n" + "=" * 30 + "\n"
        repair_info += "REPAIR RESULTS\n"
        repair_info += "=" * 30 + "\n"
        repair_info += f"Was repaired: {result.was_repaired}\n"
        repair_info += f"Method: {result.repair_method}\n"
        repair_info += f"Original vertices: {result.original_vertex_count}\n"
        repair_info += f"Original faces: {result.original_face_count}\n"
        repair_info += "\nRepair steps:\n"
        for step in result.repair_steps:
            repair_info += f"  • {step}\n"
        repair_info += "\n" + "-" * 30 + "\n"
        repair_info += result.diagnostics.format()
        
        self.info_text.setText(repair_info)
        
        # Update repair button
        if result.diagnostics.is_watertight:
            self.repair_btn.setEnabled(False)
            self.repair_btn.setText("Mesh Repaired ✓")
            self.repair_btn.setStyleSheet("""
                QPushButton {
                    background-color: #006633;
                }
            """)
    
    def _on_error(self, message: str):
        """Handle errors."""
        self.progress_label.setText(f"Error: {message}")
        self.statusbar.showMessage(f"Error: {message}")
        
        QMessageBox.critical(self, "Error", message)
    
    def _on_worker_finished(self):
        """Handle worker completion."""
        self.progress_bar.setVisible(False)
        self._worker = None
    
    def _on_repair_clicked(self):
        """Handle repair button click."""
        if self._current_mesh is None:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_label.setText("Repairing mesh...")
        self.repair_btn.setEnabled(False)
        
        # Run repair in background
        # For simplicity, doing it in main thread here
        # In production, use a worker thread
        try:
            repairer = MeshRepairer(self._current_mesh)
            result = repairer.repair()
            self._on_repair_complete(result)
        except Exception as e:
            self._on_error(str(e))
        finally:
            self.progress_bar.setVisible(False)
    
    def _on_reset_view(self):
        """Reset the 3D view."""
        self.mesh_viewer.reset_camera()
