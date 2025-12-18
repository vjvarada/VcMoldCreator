#!/usr/bin/env python3
"""
VcMoldCreator Desktop Application

Main entry point for the desktop application.
A Python/PyQt6 implementation of the mold analysis tool.
"""

import sys
import os
import logging
import traceback
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger('trimesh').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('pyvista').setLevel(logging.WARNING)
logging.getLogger('vtkmodules').setLevel(logging.WARNING)

# Global exception handler
def exception_hook(exc_type, exc_value, exc_tb):
    """Global exception handler to log uncaught exceptions."""
    logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))
    traceback.print_exception(exc_type, exc_value, exc_tb)

sys.excepthook = exception_hook

# Ensure the app directory is in the path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)


def main():
    """Main entry point."""
    logger.info("Starting VcMoldCreator Desktop App...")
    
    # Import PyQt after path setup
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPalette, QColor
    
    # High DPI support
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("VcMoldCreator")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("VcMoldCreator")
    
    logger.info("Qt application created")
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(26, 26, 26))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 122, 204))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 122, 204))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # Import and show main window
    logger.info("Loading main window...")
    from ui import MainWindow
    from PyQt6.QtGui import QScreen
    
    window = MainWindow()
    logger.debug(f"MainWindow created, geometry: {window.geometry()}")
    
    # Center window on primary screen to ensure visibility
    primary_screen = app.primaryScreen()
    if primary_screen:
        screen_geometry = primary_screen.availableGeometry()
        logger.debug(f"Primary screen: {primary_screen.name()}, geometry: {screen_geometry}")
        
        # Use 80% of screen size or max 1400x900, whichever is smaller
        target_width = min(1400, int(screen_geometry.width() * 0.8))
        target_height = min(900, int(screen_geometry.height() * 0.8))
        window.resize(target_width, target_height)
        
        # Center the window
        x = screen_geometry.x() + (screen_geometry.width() - target_width) // 2
        y = screen_geometry.y() + (screen_geometry.height() - target_height) // 2
        window.move(x, y)
        logger.debug(f"Window resized to: {target_width}x{target_height}, moved to: ({x}, {y})")
    
    logger.debug(f"MainWindow isVisible before show(): {window.isVisible()}")
    
    window.show()
    window.raise_()  # Bring to front
    window.activateWindow()  # Make it the active window
    logger.info("Main window displayed")
    
    logger.debug(f"MainWindow isVisible after show(): {window.isVisible()}")
    logger.debug(f"MainWindow geometry after show(): {window.geometry()}")
    logger.debug(f"MainWindow screen: {window.screen().name() if window.screen() else 'None'}")
    logger.debug(f"MainWindow windowState: {window.windowState()}")
    
    # Process events to ensure window is fully rendered
    app.processEvents()
    logger.debug("Processed pending events")
    logger.debug(f"MainWindow isVisible after processEvents: {window.isVisible()}")
    logger.info("Starting Qt event loop...")
    
    # Run event loop
    return app.exec()


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
