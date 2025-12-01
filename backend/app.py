# FastAPI server for mold creation with tetrahedralization support

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
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

app = FastAPI(title="Mold Creator Backend")

# Thread pool for running blocking tetrahedralization
executor = ThreadPoolExecutor(max_workers=2)

# Store for tracking job progress
jobs: dict = {}

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Vite default ports
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
    return {"status": "ok"}


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
