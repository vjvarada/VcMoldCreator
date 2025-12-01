<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# FastAPI server for Mold Creator with tetrahedralization support

from fastapi import FastAPI, UploadFile, File, HTTPException
=======
# FastAPI server for mold creation with tetrahedralization support

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
>>>>>>> Stashed changes
=======
# FastAPI server for mold creation with tetrahedralization support

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
>>>>>>> Stashed changes
=======
# FastAPI server for mold creation with tetrahedralization support

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
>>>>>>> Stashed changes
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
import trimesh
import tempfile
import os
import io
import json
import asyncio

# Try to import TetGen
try:
    import tetgen
    import pyvista as pv
    TETGEN_AVAILABLE = True
except ImportError:
    TETGEN_AVAILABLE = False
    print("Warning: tetgen not installed. Tetrahedralization will not be available.")
=======
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
import tempfile
import os
import threading
import trimesh
import pytetwild
import meshio
import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
<<<<<<< Updated upstream
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

app = FastAPI(title="Mold Creator Backend")

# Thread pool for running blocking tetrahedralization
executor = ThreadPoolExecutor(max_workers=2)

# Store for tracking job progress
jobs: dict = {}

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
=======
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Vite default ports
>>>>>>> Stashed changes
=======
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Vite default ports
>>>>>>> Stashed changes
=======
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Vite default ports
>>>>>>> Stashed changes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TetrahedralizeParams(BaseModel):
    """Parameters for tetrahedralization"""
    edge_length_fac: float = 0.0065  # Tetrahedral edge length as fraction of bounding box diagonal
    optimize: bool = True  # Improve mesh quality (slower but better results)


class JobStatus:
    """Track status of a tetrahedralization job"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"  # pending, loading, preprocessing, tetrahedralizing, postprocessing, complete, error
        self.progress = 0  # 0-100
        self.message = "Initializing..."
        self.substep = ""  # Current substep description
        self.result = None
        self.error = None
        self.start_time = time.time()
        self.input_stats = None
        self.estimated_duration = None  # Estimated total duration based on mesh size
        self._stop_progress_thread = False
        self.logs: list = []  # Log messages for streaming to frontend
    
    def log(self, message: str):
        """Add a log message with timestamp"""
        elapsed = time.time() - self.start_time
        self.logs.append({"time": elapsed, "message": message})
        # Keep only last 100 logs to avoid memory issues
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]


def estimate_duration(num_vertices: int, num_faces: int, edge_length_fac: float) -> float:
    """Estimate tetrahedralization duration based on mesh complexity"""
    # Empirical formula based on observed timings
    # Smaller edge_length_fac = denser mesh = MUCH longer time (cubic relationship)
    # For edge_length_fac=0.0065, we expect very long processing times
    edge_factor = (0.05 / max(edge_length_fac, 0.001)) ** 2.0  # Quadratic scaling
    complexity = num_faces * edge_factor
    
    # Base time + complexity factor
    # Calibrated for: 50k faces at 0.05 edge_length = ~5s, at 0.0065 = ~300s
    estimated = 3.0 + complexity / 5000.0
    
    # Log the estimate for debugging
    print(f"Duration estimate: faces={num_faces}, edge_fac={edge_length_fac}, complexity={complexity:.0f}, estimated={estimated:.1f}s")
    
    return estimated

def progress_updater_thread(job: JobStatus):
    """Background thread to update progress based on elapsed time with more granular updates"""
    # More detailed stages with finer granularity
    stages = [
        (0.00, "Initializing fTetWild engine..."),
        (0.03, "Loading input mesh into fTetWild..."),
        (0.06, "Building edge data structures..."),
        (0.10, "Removing duplicate vertices (pass 1)..."),
        (0.14, "Collapsing short edges..."),
        (0.18, "Edge collapse iteration 1..."),
        (0.22, "Edge collapse iteration 2..."),
        (0.26, "Swapping edges for quality..."),
        (0.30, "Edge swap optimization pass 1..."),
        (0.34, "Edge swap optimization pass 2..."),
        (0.38, "Removing duplicate vertices (pass 2)..."),
        (0.42, "Smoothing vertex positions..."),
        (0.46, "Preparing for tetrahedralization..."),
        (0.50, "Building Delaunay tetrahedralization..."),
        (0.54, "Triangle insertion starting..."),
        (0.58, "Inserting boundary triangles (batch 1)..."),
        (0.62, "Inserting boundary triangles (batch 2)..."),
        (0.66, "Matching triangles to tetrahedra..."),
        (0.70, "Triangle matching pass 2..."),
        (0.74, "Inserting remaining triangles..."),
        (0.78, "Refining mesh near boundaries..."),
        (0.82, "Tracking surface faces..."),
        (0.86, "Finding boundary edges..."),
        (0.90, "Marking surface faces..."),
        (0.94, "Quality optimization..."),
        (0.97, "Finalizing mesh..."),
    ]
    
    last_stage_idx = -1
    heartbeat_counter = 0
    
    while not job._stop_progress_thread and job.status == "tetrahedralizing":
        elapsed = time.time() - job.start_time
        heartbeat_counter += 1
        
        if job.estimated_duration:
            # Calculate progress based on elapsed time vs estimated duration
            # Use a slower curve to avoid getting stuck at high percentages
            raw_progress = elapsed / job.estimated_duration
            # Apply a curve that slows down as we approach 1.0
            time_progress = min(0.97, raw_progress * 0.85 / (1 + raw_progress * 0.3))
            
            # Find the current stage based on progress
            current_stage_idx = 0
            current_stage_msg = "Processing..."
            for i, (threshold, msg) in enumerate(stages):
                if time_progress >= threshold:
                    current_stage_idx = i
                    current_stage_msg = msg
                else:
                    break
            
            # Log when we enter a new stage
            if current_stage_idx > last_stage_idx:
                job.log(current_stage_msg)
                last_stage_idx = current_stage_idx
            
            # Add heartbeat log every 10 updates (~3 seconds) to show it's still working
            if heartbeat_counter % 10 == 0:
                job.log(f"Still processing... ({elapsed:.1f}s elapsed)")
            
            # Convert to percentage (30-90 range for tetrahedralization phase)
            job.progress = int(30 + time_progress * 60)
            job.substep = current_stage_msg
            job.message = f"Tetrahedralizing: {current_stage_msg}"
        else:
            # No estimate - just show elapsed time
            if heartbeat_counter % 10 == 0:
                job.log(f"Processing... ({elapsed:.1f}s elapsed)")
        
        time.sleep(0.3)  # Update every 300ms


def run_tetrahedralization_sync(tmp_path: str, file_ext: str, edge_length_fac: float, optimize: bool, job: JobStatus):
    """Run tetrahedralization synchronously (called from thread pool)"""
    progress_thread = None
    try:
        # Loading mesh
        job.status = "loading"
        job.progress = 5
        job.message = "Loading mesh file..."
        job.substep = "Reading file from disk"
        job.log("Starting mesh load...")
        
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Could not load mesh as a valid triangle mesh")
        
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        job.progress = 10
        job.message = f"Loaded mesh: {len(vertices):,} vertices, {len(faces):,} faces"
        job.substep = "Mesh loaded successfully"
        job.input_stats = {"num_vertices": len(vertices), "num_faces": len(faces)}
        job.log(f"Mesh loaded: {len(vertices):,} vertices, {len(faces):,} faces")
        
        # Estimate duration for progress calculation
        job.estimated_duration = estimate_duration(len(vertices), len(faces), edge_length_fac)
        job.log(f"Estimated duration: {job.estimated_duration:.1f}s (edge_length_fac={edge_length_fac})")
        
        # Preprocessing
        job.status = "preprocessing"
        job.progress = 15
        job.message = "Preprocessing mesh..."
        job.substep = "Validating mesh topology"
        job.log("Validating mesh topology...")
        time.sleep(0.1)  # Small delay to allow UI to update
        
        job.progress = 20
        job.substep = "Computing bounding box"
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_diag = np.linalg.norm(bbox_size)
        target_edge_length = edge_length_fac * bbox_diag
        job.log(f"Bounding box: {bbox_size[0]:.3f} x {bbox_size[1]:.3f} x {bbox_size[2]:.3f}")
        job.log(f"Diagonal: {bbox_diag:.3f}, Target edge length: {target_edge_length:.4f}")
        time.sleep(0.1)
        
        job.progress = 25
        job.substep = "Preparing vertex and face arrays"
        job.log("Preparing data for fTetWild...")
        time.sleep(0.1)
        
        # Tetrahedralizing - start progress updater thread
        job.status = "tetrahedralizing"
        job.progress = 30
        job.message = "Starting fTetWild tetrahedralization..."
        job.substep = "Initializing fTetWild"
        job.start_time = time.time()  # Reset start time for progress estimation
        job.log("Starting fTetWild tetrahedralization (this may take a while)...")
        
        # Start background progress updater
        progress_thread = threading.Thread(target=progress_updater_thread, args=(job,), daemon=True)
        progress_thread.start()
        
        # Run tetrahedralization (blocking)
        # Note: fTetWild outputs to C stdout which we can't easily capture on Windows
        # The progress_updater_thread provides simulated progress updates
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        # Stop progress thread
        job._stop_progress_thread = True
        if progress_thread:
            progress_thread.join(timeout=0.5)
        
        tet_time = time.time() - job.start_time
        job.log(f"fTetWild completed in {tet_time:.1f}s")
        job.log(f"Generated {len(tet_vertices):,} vertices, {len(tetrahedra):,} tetrahedra")
        
        # Post-processing
        job.status = "postprocessing"
        job.progress = 92
        job.message = "Post-processing results..."
        job.substep = "Converting vertex data"
        job.log("Converting vertex data to list format...")
        time.sleep(0.05)
        
        job.progress = 95
        job.substep = "Converting tetrahedra data"
        job.log("Converting tetrahedra data to list format...")
        
        result_vertices = tet_vertices.tolist()
        
        job.progress = 97
        job.substep = "Preparing JSON response"
        job.log(f"Preparing response ({len(result_vertices)} vertices)...")
        
        result_tetrahedra = tetrahedra.tolist()
        
        job.progress = 99
        job.substep = "Finalizing"
        job.log("Finalizing result...")
        
        # Complete - but keep progress at 99 so frontend knows there's more to do
        job.status = "complete"
        job.progress = 99  # Stay at 99, frontend will set 100 when visualization is ready
        job.message = f"Backend complete: {len(tet_vertices):,} vertices, {len(tetrahedra):,} tetrahedra"
        job.substep = "Sending to frontend..."
        job.log("Backend processing complete, sending data to frontend...")
        job.result = {
            "success": True,
            "message": "Tetrahedralization completed successfully",
            "input_stats": {
                "num_vertices": len(vertices),
                "num_faces": len(faces)
            },
            "output_stats": {
                "num_vertices": len(tet_vertices),
                "num_tetrahedra": len(tetrahedra)
            },
            "vertices": result_vertices,
            "tetrahedra": result_tetrahedra
        }
        
    except Exception as e:
        job._stop_progress_thread = True
        if progress_thread:
            progress_thread.join(timeout=0.5)
        job.status = "error"
        job.progress = 0
        job.message = f"Error: {str(e)}"
        job.substep = "Failed"
        job.error = str(e)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "tetgen_available": TETGEN_AVAILABLE
    }


# TetWild-style default: target edge length = 0.0065 * bbox diagonal
TETWILD_EDGE_LENGTH_RATIO = 0.0065


class TetrahedralizeParams(BaseModel):
    """Parameters for tetrahedralization"""
    order: Optional[int] = 1  # 1 for linear, 2 for quadratic tetrahedra
    mindihedral: Optional[float] = 20.0  # Minimum dihedral angle (quality)
    minratio: Optional[float] = 1.5  # Maximum radius-edge ratio
    edge_length_ratio: Optional[float] = TETWILD_EDGE_LENGTH_RATIO  # Target edge length as ratio of bbox diagonal
    max_volume: Optional[float] = None  # Maximum tetrahedron volume (auto-computed from edge_length if None)
    quality: Optional[bool] = True  # Enable quality mesh generation


def send_sse_message(event: str, data: dict) -> str:
    """Format a Server-Sent Event message"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/tetrahedralize/stream")
async def tetrahedralize_mesh_stream(
    file: UploadFile = File(...),
    order: int = 1,
    mindihedral: float = 20.0,
    minratio: float = 1.5,
    edge_length_ratio: float = TETWILD_EDGE_LENGTH_RATIO,
    max_volume: Optional[float] = None,
    quality: bool = True,
):
    """
    Tetrahedralize a surface mesh using TetGen with Server-Sent Events for progress updates.
    
    Returns: SSE stream with progress updates and final result
    """
    if not TETGEN_AVAILABLE:
        async def error_stream():
            yield send_sse_message("error", {"message": "TetGen is not installed on the server"})
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    # Read the file content
    content = await file.read()
    filename = file.filename or "mesh.stl"
    ext = os.path.splitext(filename)[1].lower()
    
    async def generate():
        try:
            # Step 1: Loading mesh
            yield send_sse_message("progress", {"step": 1, "total": 5, "message": "Loading mesh..."})
            await asyncio.sleep(0.05)  # Allow UI to update
            
            mesh = trimesh.load(
                io.BytesIO(content),
                file_type=ext[1:] if ext else 'stl'
            )
            
            if not isinstance(mesh, trimesh.Trimesh):
                yield send_sse_message("error", {"message": "Uploaded file is not a valid triangle mesh"})
                return
            
            # Step 2: Analyzing mesh
            yield send_sse_message("progress", {"step": 2, "total": 5, "message": f"Analyzing mesh ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)..."})
            await asyncio.sleep(0.05)
            
            bounds = mesh.bounds
            bbox_diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
            target_edge_length = edge_length_ratio * bbox_diagonal
            
            computed_max_volume = max_volume
            if computed_max_volume is None:
                computed_max_volume = 0.1178 * (target_edge_length ** 3)
            
            # Step 3: Preparing for TetGen
            yield send_sse_message("progress", {"step": 3, "total": 5, "message": f"Preparing mesh (target edge: {target_edge_length:.4f})..."})
            await asyncio.sleep(0.05)
            
            faces_pv = np.hstack([
                np.full((len(mesh.faces), 1), 3, dtype=np.int64),
                mesh.faces
            ]).flatten()
            
            pv_mesh = pv.PolyData(mesh.vertices, faces_pv)
            
            # Step 4: Running TetGen
            yield send_sse_message("progress", {"step": 4, "total": 5, "message": "Running TetGen tetrahedralization..."})
            await asyncio.sleep(0.05)
            
            tet = tetgen.TetGen(pv_mesh)
            tet.tetrahedralize(
                order=order,
                mindihedral=mindihedral,
                minratio=minratio,
                maxvolume=computed_max_volume
            )
            
            grid = tet.grid
            
            # Step 5: Extracting results
            yield send_sse_message("progress", {"step": 5, "total": 5, "message": "Extracting tetrahedral mesh..."})
            await asyncio.sleep(0.05)
            
            tet_vertices = np.array(grid.points)
            
            cells = grid.cells
            cell_types = grid.celltypes
            
            tetrahedra = []
            idx = 0
            for cell_type in cell_types:
                n_points = cells[idx]
                if cell_type == 10:  # VTK_TETRA
                    tetrahedra.append(cells[idx + 1:idx + 1 + n_points].tolist())
                idx += n_points + 1
            
            tet_tets = np.array(tetrahedra)
            
            # Final result - use compact format to reduce payload size
            # Round vertices to 6 decimal places to reduce JSON size
            vertices_rounded = np.round(tet_vertices.flatten(), 6).tolist()
            tetrahedra_flat = tet_tets.flatten().tolist()
            
            result = {
                "vertices": vertices_rounded,
                "tetrahedra": tetrahedra_flat,
                "num_vertices": len(tet_vertices),
                "num_tetrahedra": len(tet_tets),
                "input_vertices": len(mesh.vertices),
                "input_faces": len(mesh.faces),
                "bbox_diagonal": round(bbox_diagonal, 6),
                "target_edge_length": round(float(target_edge_length), 6),
                "max_volume_used": round(float(computed_max_volume), 9)
            }
            
            # Log result size for debugging
            result_json = json.dumps(result)
            print(f"Tetrahedralization complete: {len(tet_vertices)} vertices, {len(tet_tets)} tets, JSON size: {len(result_json)} bytes")
            
            yield send_sse_message("complete", result)
            await asyncio.sleep(0.1)  # Ensure message is flushed
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Tetrahedralization error: {error_msg}")
            print(traceback.format_exc())
            yield send_sse_message("error", {"message": error_msg})
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/tetrahedralize")
async def tetrahedralize_mesh(
    file: UploadFile = File(...),
    order: int = 1,
    mindihedral: float = 20.0,
    minratio: float = 1.5,
    edge_length_ratio: float = TETWILD_EDGE_LENGTH_RATIO,
    max_volume: Optional[float] = None,
    quality: bool = True,
    output_format: str = "json"
):
    """
    Tetrahedralize a surface mesh using TetGen.
    
    Parameters:
    - file: Input mesh file (STL, OBJ, PLY, OFF)
    - order: 1 for linear tetrahedra, 2 for quadratic (default: 1)
    - mindihedral: Minimum dihedral angle in degrees (default: 20.0)
    - minratio: Maximum radius-edge ratio for quality (default: 1.5)
    - edge_length_ratio: Target edge length as ratio of bbox diagonal (default: 0.0065, TetWild-style)
    - max_volume: Maximum tetrahedron volume constraint (auto-computed from edge_length if not provided)
    - quality: Enable quality mesh generation (default: True)
    - output_format: Output format - "msh", "vtk", or "json" (default: "json")
    
    Returns:
    - Tetrahedral mesh in requested format
    """
    if not TETGEN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="TetGen is not installed on the server. Install with: pip install tetgen"
        )
    
    # Read uploaded file
    try:
        content = await file.read()
        
        # Determine file type from filename
        filename = file.filename or "mesh.stl"
        ext = os.path.splitext(filename)[1].lower()
        
        # Load mesh using trimesh
        mesh = trimesh.load(
            io.BytesIO(content),
            file_type=ext[1:] if ext else 'stl'
        )
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Uploaded file is not a valid triangle mesh")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load mesh: {str(e)}")
    
    # Calculate bounding box diagonal and target edge length (TetWild-style)
    bounds = mesh.bounds
    bbox_diagonal = np.linalg.norm(bounds[1] - bounds[0])
    target_edge_length = edge_length_ratio * bbox_diagonal
    
    # Compute max_volume from target edge length if not provided
    # For a regular tetrahedron with edge length a, volume = a³ / (6√2) ≈ 0.1178 * a³
    if max_volume is None:
        max_volume = 0.1178 * (target_edge_length ** 3)
    
    # Convert to PyVista mesh for TetGen
    try:
        # Create PyVista surface mesh
        faces_pv = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces
        ]).flatten()
        
        pv_mesh = pv.PolyData(mesh.vertices, faces_pv)
        
        # Create TetGen instance
        tet = tetgen.TetGen(pv_mesh)
        
        # Run tetrahedralization with volume constraint for edge length control
        tet.tetrahedralize(
            order=order,
            mindihedral=mindihedral,
            minratio=minratio,
            maxvolume=max_volume  # Controls tetrahedron size (derived from target edge length)
        )
        
        # Get result
        grid = tet.grid
        
        # Extract vertices and tetrahedra
        tet_vertices = np.array(grid.points)
        
        # Extract cell connectivity (tetrahedra)
        cells = grid.cells
        cell_types = grid.celltypes
        
        # Parse tetrahedra from VTK cell array
        tetrahedra = []
        idx = 0
        for cell_type in cell_types:
            n_points = cells[idx]
            if cell_type == 10:  # VTK_TETRA
                tetrahedra.append(cells[idx + 1:idx + 1 + n_points].tolist())
            idx += n_points + 1
        
        tet_tets = np.array(tetrahedra)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )
    
    # Return result in requested format
    if output_format == "json":
        # Flatten vertices and tetrahedra for efficient transfer
        vertices_flat = tet_vertices.flatten().tolist()
        tetrahedra_flat = tet_tets.flatten().tolist()
        
        return {
            "vertices": vertices_flat,
            "tetrahedra": tetrahedra_flat,
            "num_vertices": len(tet_vertices),
            "num_tetrahedra": len(tet_tets),
            "input_vertices": len(mesh.vertices),
            "input_faces": len(mesh.faces),
            "bbox_diagonal": float(bbox_diagonal),
            "target_edge_length": float(target_edge_length),
            "max_volume_used": float(max_volume)
        }
    
    elif output_format == "vtk":
        # Write VTK format
        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            grid.save(tmp_path)
            
            with open(tmp_path, "rb") as f:
                result_data = f.read()
            
            return Response(
                content=result_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": "attachment; filename=tetrahedralized.vtk"
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    elif output_format == "msh":
        # Use meshio to write Gmsh format
        import meshio
        
        cells = [("tetra", tet_tets)]
        meshio_mesh = meshio.Mesh(points=tet_vertices, cells=cells)
        
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            meshio.write(tmp_path, meshio_mesh, file_format="gmsh")
            
            with open(tmp_path, "rb") as f:
                result_data = f.read()
            
            return Response(
                content=result_data,
                media_type="application/x-gmsh",
                headers={
                    "Content-Disposition": "attachment; filename=tetrahedralized.msh"
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format: {output_format}. Use 'json', 'vtk', or 'msh'"
        )


@app.post("/tetrahedralize/info")
async def tetrahedralize_info(file: UploadFile = File(...)):
    """
    Get information about a mesh without tetrahedralizing it.
    Useful for estimating parameters before running the full process.
    """
    try:
        content = await file.read()
        filename = file.filename or "mesh.stl"
        ext = os.path.splitext(filename)[1].lower()
        
        mesh = trimesh.load(
            io.BytesIO(content),
            file_type=ext[1:] if ext else 'stl'
        )
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Uploaded file is not a valid triangle mesh")
        
        # Calculate mesh statistics
        bounds = mesh.bounds
        bbox_diagonal = np.linalg.norm(bounds[1] - bounds[0])
        volume = float(mesh.volume) if mesh.is_watertight else None
        
        # TetWild-style target edge length (0.0065 * bbox diagonal)
        tetwild_edge_length = TETWILD_EDGE_LENGTH_RATIO * bbox_diagonal
        tetwild_max_volume = 0.1178 * (tetwild_edge_length ** 3)
        
        return {
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "bounds": bounds.tolist(),
            "bbox_diagonal": float(bbox_diagonal),
            "volume": volume,
            "is_watertight": bool(mesh.is_watertight),
            "euler_number": int(mesh.euler_number),
            # TetWild-style defaults
            "tetwild_edge_length_ratio": TETWILD_EDGE_LENGTH_RATIO,
            "suggested_edge_length": float(tetwild_edge_length),
            "suggested_max_volume": float(tetwild_max_volume),
            "suggested_mindihedral": 20.0,
            "suggested_minratio": 1.5
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze mesh: {str(e)}")


@app.post("/tetrahedralize/start")
async def start_tetrahedralization(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Start a tetrahedralization job and return a job ID for tracking progress.
    Use GET /tetrahedralize/progress/{job_id} to check status.
    Use GET /tetrahedralize/result/{job_id} to get the result when complete.
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Create job
    job_id = str(uuid.uuid4())
    job = JobStatus(job_id)
    jobs[job_id] = job
    
    # Start tetrahedralization in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_tetrahedralization_sync,
        tmp_path, file_ext, edge_length_fac, optimize, job
    )
    
    return {"job_id": job_id, "message": "Tetrahedralization started"}


@app.get("/tetrahedralize/progress/{job_id}")
async def get_tetrahedralization_progress(job_id: str, last_log_index: int = 0):
    """Get the progress of a tetrahedralization job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    elapsed = time.time() - job.start_time
    
    # Get new logs since last_log_index
    new_logs = job.logs[last_log_index:] if hasattr(job, 'logs') else []
    
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "substep": getattr(job, 'substep', ''),
        "elapsed_seconds": round(elapsed, 1),
        "estimated_remaining": round(max(0, (job.estimated_duration or 0) - elapsed), 1) if job.estimated_duration else None,
        "input_stats": job.input_stats,
        "complete": job.status in ("complete", "error"),
        "error": job.error,
        "logs": new_logs,
        "log_index": len(job.logs) if hasattr(job, 'logs') else 0
    }


@app.get("/tetrahedralize/result/{job_id}")
async def get_tetrahedralization_result(job_id: str):
    """Get the result of a completed tetrahedralization job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error)
    
    if job.status != "complete":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    
    # Clean up job after returning result
    result = job.result
    del jobs[job_id]
    
    return result


@app.post("/tetrahedralize/stats")
async def tetrahedralize_mesh_stats(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh and return only statistics (no mesh data).
    Use this endpoint for testing or when you only need to know the output size.
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        Statistics about input and output meshes
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "message": "Tetrahedralization completed successfully",
            "input_stats": {
                "num_vertices": int(len(vertices)),
                "num_faces": int(len(faces)),
                "bounding_box": {
                    "min": mesh.bounds[0].tolist(),
                    "max": mesh.bounds[1].tolist()
                }
            },
            "output_stats": {
                "num_vertices": int(len(tet_vertices)),
                "num_tetrahedra": int(len(tetrahedra))
            },
            "parameters": {
                "edge_length_fac": edge_length_fac,
                "optimize": optimize
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize")
async def tetrahedralize_mesh(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh using fTetWild (via pytetwild).
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal (default: 0.0065)
        optimize: Whether to optimize mesh quality (default: True)
    
    Returns:
        Tetrahedral mesh data as JSON with vertices and tetrahedra
    """
    # Validate file extension
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load mesh using trimesh
        mesh = trimesh.load(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        # Get vertices and faces
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        # Tetrahedralize using pytetwild
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "message": "Tetrahedralization completed successfully",
            "input_stats": {
                "num_vertices": len(vertices),
                "num_faces": len(faces)
            },
            "output_stats": {
                "num_vertices": len(tet_vertices),
                "num_tetrahedra": len(tetrahedra)
            },
            "vertices": tet_vertices.tolist(),
            "tetrahedra": tetrahedra.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize/msh")
async def tetrahedralize_mesh_to_msh(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh and return as MSH file format.
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        MSH file as downloadable response
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load mesh using trimesh
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        # Get vertices and faces
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        # Tetrahedralize
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        # Create meshio mesh and save to MSH
        cells = [("tetra", tetrahedra)]
        msh_mesh = meshio.Mesh(points=tet_vertices, cells=cells)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.msh') as tmp:
            msh_path = tmp.name
        
        meshio.write(msh_path, msh_mesh, file_format="gmsh")
        
        # Read file content
        with open(msh_path, 'rb') as f:
            msh_content = f.read()
        
        os.unlink(msh_path)
        
        # Return as downloadable file
        output_filename = os.path.splitext(file.filename)[0] + '_tet.msh'
        return Response(
            content=msh_content,
            media_type='application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename="{output_filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize/arrays")
async def tetrahedralize_from_arrays(
    vertices: list,
    faces: list,
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize from raw vertex and face arrays.
    
    Args:
        vertices: List of [x, y, z] vertex coordinates
        faces: List of face vertex indices (triangles or quads)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        Tetrahedral mesh data with vertices and tetrahedra
    """
    try:
        vertices_np = np.array(vertices, dtype=np.float64)
        faces_np = np.array(faces, dtype=np.int32)
        
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices_np,
            faces_np,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "vertices": tet_vertices.tolist(),
            "tetrahedra": tetrahedra.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )



@app.post("/tetrahedralize/start")
async def start_tetrahedralization(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Start a tetrahedralization job and return a job ID for tracking progress.
    Use GET /tetrahedralize/progress/{job_id} to check status.
    Use GET /tetrahedralize/result/{job_id} to get the result when complete.
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Create job
    job_id = str(uuid.uuid4())
    job = JobStatus(job_id)
    jobs[job_id] = job
    
    # Start tetrahedralization in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_tetrahedralization_sync,
        tmp_path, file_ext, edge_length_fac, optimize, job
    )
    
    return {"job_id": job_id, "message": "Tetrahedralization started"}


@app.get("/tetrahedralize/progress/{job_id}")
async def get_tetrahedralization_progress(job_id: str, last_log_index: int = 0):
    """Get the progress of a tetrahedralization job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    elapsed = time.time() - job.start_time
    
    # Get new logs since last_log_index
    new_logs = job.logs[last_log_index:] if hasattr(job, 'logs') else []
    
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "substep": getattr(job, 'substep', ''),
        "elapsed_seconds": round(elapsed, 1),
        "estimated_remaining": round(max(0, (job.estimated_duration or 0) - elapsed), 1) if job.estimated_duration else None,
        "input_stats": job.input_stats,
        "complete": job.status in ("complete", "error"),
        "error": job.error,
        "logs": new_logs,
        "log_index": len(job.logs) if hasattr(job, 'logs') else 0
    }


@app.get("/tetrahedralize/result/{job_id}")
async def get_tetrahedralization_result(job_id: str):
    """Get the result of a completed tetrahedralization job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error)
    
    if job.status != "complete":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    
    # Clean up job after returning result
    result = job.result
    del jobs[job_id]
    
    return result


@app.post("/tetrahedralize/stats")
async def tetrahedralize_mesh_stats(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh and return only statistics (no mesh data).
    Use this endpoint for testing or when you only need to know the output size.
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        Statistics about input and output meshes
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "message": "Tetrahedralization completed successfully",
            "input_stats": {
                "num_vertices": int(len(vertices)),
                "num_faces": int(len(faces)),
                "bounding_box": {
                    "min": mesh.bounds[0].tolist(),
                    "max": mesh.bounds[1].tolist()
                }
            },
            "output_stats": {
                "num_vertices": int(len(tet_vertices)),
                "num_tetrahedra": int(len(tetrahedra))
            },
            "parameters": {
                "edge_length_fac": edge_length_fac,
                "optimize": optimize
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize")
async def tetrahedralize_mesh(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh using fTetWild (via pytetwild).
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal (default: 0.0065)
        optimize: Whether to optimize mesh quality (default: True)
    
    Returns:
        Tetrahedral mesh data as JSON with vertices and tetrahedra
    """
    # Validate file extension
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load mesh using trimesh
        mesh = trimesh.load(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        # Get vertices and faces
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        # Tetrahedralize using pytetwild
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "message": "Tetrahedralization completed successfully",
            "input_stats": {
                "num_vertices": len(vertices),
                "num_faces": len(faces)
            },
            "output_stats": {
                "num_vertices": len(tet_vertices),
                "num_tetrahedra": len(tetrahedra)
            },
            "vertices": tet_vertices.tolist(),
            "tetrahedra": tetrahedra.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize/msh")
async def tetrahedralize_mesh_to_msh(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh and return as MSH file format.
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        MSH file as downloadable response
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load mesh using trimesh
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        # Get vertices and faces
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        # Tetrahedralize
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        # Create meshio mesh and save to MSH
        cells = [("tetra", tetrahedra)]
        msh_mesh = meshio.Mesh(points=tet_vertices, cells=cells)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.msh') as tmp:
            msh_path = tmp.name
        
        meshio.write(msh_path, msh_mesh, file_format="gmsh")
        
        # Read file content
        with open(msh_path, 'rb') as f:
            msh_content = f.read()
        
        os.unlink(msh_path)
        
        # Return as downloadable file
        output_filename = os.path.splitext(file.filename)[0] + '_tet.msh'
        return Response(
            content=msh_content,
            media_type='application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename="{output_filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize/arrays")
async def tetrahedralize_from_arrays(
    vertices: list,
    faces: list,
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize from raw vertex and face arrays.
    
    Args:
        vertices: List of [x, y, z] vertex coordinates
        faces: List of face vertex indices (triangles or quads)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        Tetrahedral mesh data with vertices and tetrahedra
    """
    try:
        vertices_np = np.array(vertices, dtype=np.float64)
        faces_np = np.array(faces, dtype=np.int32)
        
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices_np,
            faces_np,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "vertices": tet_vertices.tolist(),
            "tetrahedra": tetrahedra.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )



@app.post("/tetrahedralize/start")
async def start_tetrahedralization(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Start a tetrahedralization job and return a job ID for tracking progress.
    Use GET /tetrahedralize/progress/{job_id} to check status.
    Use GET /tetrahedralize/result/{job_id} to get the result when complete.
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Create job
    job_id = str(uuid.uuid4())
    job = JobStatus(job_id)
    jobs[job_id] = job
    
    # Start tetrahedralization in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_tetrahedralization_sync,
        tmp_path, file_ext, edge_length_fac, optimize, job
    )
    
    return {"job_id": job_id, "message": "Tetrahedralization started"}


@app.get("/tetrahedralize/progress/{job_id}")
async def get_tetrahedralization_progress(job_id: str, last_log_index: int = 0):
    """Get the progress of a tetrahedralization job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    elapsed = time.time() - job.start_time
    
    # Get new logs since last_log_index
    new_logs = job.logs[last_log_index:] if hasattr(job, 'logs') else []
    
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "substep": getattr(job, 'substep', ''),
        "elapsed_seconds": round(elapsed, 1),
        "estimated_remaining": round(max(0, (job.estimated_duration or 0) - elapsed), 1) if job.estimated_duration else None,
        "input_stats": job.input_stats,
        "complete": job.status in ("complete", "error"),
        "error": job.error,
        "logs": new_logs,
        "log_index": len(job.logs) if hasattr(job, 'logs') else 0
    }


@app.get("/tetrahedralize/result/{job_id}")
async def get_tetrahedralization_result(job_id: str):
    """Get the result of a completed tetrahedralization job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error)
    
    if job.status != "complete":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    
    # Clean up job after returning result
    result = job.result
    del jobs[job_id]
    
    return result


@app.post("/tetrahedralize/stats")
async def tetrahedralize_mesh_stats(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh and return only statistics (no mesh data).
    Use this endpoint for testing or when you only need to know the output size.
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        Statistics about input and output meshes
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "message": "Tetrahedralization completed successfully",
            "input_stats": {
                "num_vertices": int(len(vertices)),
                "num_faces": int(len(faces)),
                "bounding_box": {
                    "min": mesh.bounds[0].tolist(),
                    "max": mesh.bounds[1].tolist()
                }
            },
            "output_stats": {
                "num_vertices": int(len(tet_vertices)),
                "num_tetrahedra": int(len(tetrahedra))
            },
            "parameters": {
                "edge_length_fac": edge_length_fac,
                "optimize": optimize
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize")
async def tetrahedralize_mesh(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh using fTetWild (via pytetwild).
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal (default: 0.0065)
        optimize: Whether to optimize mesh quality (default: True)
    
    Returns:
        Tetrahedral mesh data as JSON with vertices and tetrahedra
    """
    # Validate file extension
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load mesh using trimesh
        mesh = trimesh.load(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        # Get vertices and faces
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        # Tetrahedralize using pytetwild
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "message": "Tetrahedralization completed successfully",
            "input_stats": {
                "num_vertices": len(vertices),
                "num_faces": len(faces)
            },
            "output_stats": {
                "num_vertices": len(tet_vertices),
                "num_tetrahedra": len(tetrahedra)
            },
            "vertices": tet_vertices.tolist(),
            "tetrahedra": tetrahedra.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize/msh")
async def tetrahedralize_mesh_to_msh(
    file: UploadFile = File(...),
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize a surface mesh and return as MSH file format.
    
    Args:
        file: Surface mesh file (.stl, .obj, .ply, .off)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        MSH file as downloadable response
    """
    allowed_extensions = {'.stl', '.obj', '.ply', '.off'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load mesh using trimesh
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise HTTPException(
                status_code=400,
                detail="Could not load mesh as a valid triangle mesh"
            )
        
        # Get vertices and faces
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        # Tetrahedralize
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices,
            faces,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        # Create meshio mesh and save to MSH
        cells = [("tetra", tetrahedra)]
        msh_mesh = meshio.Mesh(points=tet_vertices, cells=cells)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.msh') as tmp:
            msh_path = tmp.name
        
        meshio.write(msh_path, msh_mesh, file_format="gmsh")
        
        # Read file content
        with open(msh_path, 'rb') as f:
            msh_content = f.read()
        
        os.unlink(msh_path)
        
        # Return as downloadable file
        output_filename = os.path.splitext(file.filename)[0] + '_tet.msh'
        return Response(
            content=msh_content,
            media_type='application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename="{output_filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


@app.post("/tetrahedralize/arrays")
async def tetrahedralize_from_arrays(
    vertices: list,
    faces: list,
    edge_length_fac: float = 0.0065,
    optimize: bool = True
):
    """
    Tetrahedralize from raw vertex and face arrays.
    
    Args:
        vertices: List of [x, y, z] vertex coordinates
        faces: List of face vertex indices (triangles or quads)
        edge_length_fac: Tetrahedral edge length as fraction of bounding box diagonal
        optimize: Whether to optimize mesh quality
    
    Returns:
        Tetrahedral mesh data with vertices and tetrahedra
    """
    try:
        vertices_np = np.array(vertices, dtype=np.float64)
        faces_np = np.array(faces, dtype=np.int32)
        
        tet_vertices, tetrahedra = pytetwild.tetrahedralize(
            vertices_np,
            faces_np,
            edge_length_fac=edge_length_fac,
            optimize=optimize
        )
        
        return {
            "success": True,
            "vertices": tet_vertices.tolist(),
            "tetrahedra": tetrahedra.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tetrahedralization failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
