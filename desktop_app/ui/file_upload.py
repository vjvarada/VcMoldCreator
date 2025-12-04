"""
File Upload Widget

PyQt6 widget for uploading STL files via drag-and-drop or file browser.
Similar to FileUpload.tsx from the frontend.
"""

from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, 
    QFileDialog, QFrame, QDialog, QHBoxLayout,
    QComboBox, QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


class UnitSelectionDialog(QDialog):
    """
    Dialog for selecting the unit system of the imported STL file.
    
    STL files don't contain unit information, so we need to ask the user.
    """
    
    # Unit options with their scale factors to convert to mm
    UNITS = {
        "Millimeters (mm)": 1.0,
        "Centimeters (cm)": 10.0,
        "Meters (m)": 1000.0,
        "Inches (in)": 25.4,
        "Feet (ft)": 304.8,
    }
    
    def __init__(self, filename: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Select Units")
        self.setModal(True)
        self.setMinimumWidth(350)
        
        self._scale_factor = 1.0  # Default: mm (no scaling)
        self._setup_ui(filename)
    
    def _setup_ui(self, filename: str):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Info label
        info_label = QLabel(
            f"<b>File:</b> {filename}<br><br>"
            "STL files don't contain unit information.<br>"
            "Please select the units used when the model was created:"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Unit selection combo box
        unit_layout = QHBoxLayout()
        unit_label = QLabel("Units:")
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(self.UNITS.keys())
        self.unit_combo.setCurrentText("Millimeters (mm)")  # Default
        self.unit_combo.currentTextChanged.connect(self._on_unit_changed)
        
        unit_layout.addWidget(unit_label)
        unit_layout.addWidget(self.unit_combo, 1)
        layout.addLayout(unit_layout)
        
        # Preview label showing conversion
        self.preview_label = QLabel("Model will be imported as-is (1 unit = 1 mm)")
        self.preview_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.preview_label)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _on_unit_changed(self, unit_text: str):
        """Handle unit selection change."""
        self._scale_factor = self.UNITS.get(unit_text, 1.0)
        
        if self._scale_factor == 1.0:
            self.preview_label.setText("Model will be imported as-is (1 unit = 1 mm)")
        else:
            self.preview_label.setText(
                f"Model will be scaled by {self._scale_factor}x "
                f"(1 {unit_text.split('(')[1].split(')')[0]} = {self._scale_factor} mm)"
            )
    
    @property
    def scale_factor(self) -> float:
        """Get the scale factor to convert to mm."""
        return self._scale_factor


class FileUploadWidget(QWidget):
    """
    File upload widget with drag-and-drop support.
    
    Signals:
        file_selected: Emitted when a valid STL file is selected (path: str, scale_factor: float)
    """
    
    file_selected = pyqtSignal(str, float)  # Emits file path and scale factor
    
    SUPPORTED_EXTENSIONS = {'.stl', '.STL'}
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_file: Optional[str] = None
        self._current_scale: float = 1.0
        self._setup_ui()
        self._setup_drag_drop()
    
    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Drop zone frame
        self.drop_frame = QFrame()
        self.drop_frame.setObjectName("dropFrame")
        self.drop_frame.setMinimumHeight(100)
        self.drop_frame.setStyleSheet("""
            QFrame#dropFrame {
                background-color: rgba(255, 255, 255, 20);
                border: 2px dashed rgba(255, 255, 255, 80);
                border-radius: 8px;
            }
            QFrame#dropFrame:hover {
                background-color: rgba(0, 170, 255, 50);
                border-color: #00aaff;
            }
        """)
        
        frame_layout = QVBoxLayout(self.drop_frame)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon label
        self.icon_label = QLabel("📁")
        self.icon_label.setStyleSheet("font-size: 32px; border: none;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.icon_label)
        
        # Status label
        self.status_label = QLabel("Upload STL File")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: white;
            border: none;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.status_label)
        
        # Hint label
        self.hint_label = QLabel("Click or drag & drop")
        self.hint_label.setStyleSheet("""
            font-size: 11px;
            color: rgba(255, 255, 255, 150);
            border: none;
        """)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.hint_label)
        
        layout.addWidget(self.drop_frame)
        
        # Make frame clickable
        self.drop_frame.mousePressEvent = self._on_click
    
    def _setup_drag_drop(self):
        """Enable drag and drop."""
        self.setAcceptDrops(True)
    
    def _on_click(self, event):
        """Handle click to open file browser."""
        self.browse_file()
    
    def browse_file(self):
        """Open file browser dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open STL File",
            "",
            "STL Files (*.stl *.STL);;All Files (*)"
        )
        
        if file_path:
            self._handle_file(file_path)
    
    def _handle_file(self, file_path: str):
        """Process the selected file."""
        path = Path(file_path)
        
        if path.suffix.lower() not in {ext.lower() for ext in self.SUPPORTED_EXTENSIONS}:
            self._show_error("Please select an STL file")
            return
        
        if not path.exists():
            self._show_error("File does not exist")
            return
        
        # Show unit selection dialog
        dialog = UnitSelectionDialog(path.name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return  # User cancelled
        
        self._current_file = str(path.absolute())
        self._current_scale = dialog.scale_factor
        self._show_loaded(path.name)
        self.file_selected.emit(self._current_file, self._current_scale)
    
    def _show_loaded(self, filename: str):
        """Update UI to show loaded state."""
        self.icon_label.setText("✓")
        self.icon_label.setStyleSheet("font-size: 32px; color: #00ff88; border: none;")
        self.status_label.setText("Loaded:")
        self.hint_label.setText(filename)
        self.hint_label.setStyleSheet("""
            font-size: 12px;
            color: rgba(255, 255, 255, 200);
            border: none;
        """)
        
        self.drop_frame.setStyleSheet("""
            QFrame#dropFrame {
                background-color: rgba(0, 255, 136, 30);
                border: 2px solid rgba(0, 255, 136, 150);
                border-radius: 8px;
            }
            QFrame#dropFrame:hover {
                background-color: rgba(0, 170, 255, 50);
                border-color: #00aaff;
            }
        """)
    
    def _show_error(self, message: str):
        """Update UI to show error state."""
        self.icon_label.setText("✗")
        self.icon_label.setStyleSheet("font-size: 32px; color: #ff4444; border: none;")
        self.status_label.setText("Error")
        self.hint_label.setText(message)
        
        self.drop_frame.setStyleSheet("""
            QFrame#dropFrame {
                background-color: rgba(255, 68, 68, 30);
                border: 2px solid rgba(255, 68, 68, 150);
                border-radius: 8px;
            }
        """)
    
    def reset(self):
        """Reset widget to initial state."""
        self._current_file = None
        self.icon_label.setText("📁")
        self.icon_label.setStyleSheet("font-size: 32px; border: none;")
        self.status_label.setText("Upload STL File")
        self.hint_label.setText("Click or drag & drop")
        self.hint_label.setStyleSheet("""
            font-size: 11px;
            color: rgba(255, 255, 255, 150);
            border: none;
        """)
        
        self.drop_frame.setStyleSheet("""
            QFrame#dropFrame {
                background-color: rgba(255, 255, 255, 20);
                border: 2px dashed rgba(255, 255, 255, 80);
                border-radius: 8px;
            }
            QFrame#dropFrame:hover {
                background-color: rgba(0, 170, 255, 50);
                border-color: #00aaff;
            }
        """)
    
    @property
    def current_file(self) -> Optional[str]:
        """Get the currently loaded file path."""
        return self._current_file
    
    # Drag and drop event handlers
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if Path(file_path).suffix.lower() in {ext.lower() for ext in self.SUPPORTED_EXTENSIONS}:
                    event.acceptProposedAction()
                    self.drop_frame.setStyleSheet("""
                        QFrame#dropFrame {
                            background-color: rgba(0, 170, 255, 80);
                            border: 2px dashed #00aaff;
                            border-radius: 8px;
                        }
                    """)
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        # Restore normal style
        if self._current_file:
            self._show_loaded(Path(self._current_file).name)
        else:
            self.reset()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                self._handle_file(file_path)
                event.acceptProposedAction()
                return
        event.ignore()
