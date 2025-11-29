# FastAPI server for Mold Creator with tetrahedralization support

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
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

app = FastAPI(title="Mold Creator Backend")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
