/**
 * Tetrahedralization API Client
 * 
 * Calls the backend fTetWild service to convert surface meshes to tetrahedral meshes.
 * Used for mold volume discretization.
 */

import * as THREE from 'three';

// ============================================================================
// TYPES
// ============================================================================

export interface TetrahedralizationResult {
  /** Tetrahedral mesh vertices */
  vertices: Float32Array;
  /** Tetrahedra indices (4 vertices per tetrahedron) */
  tetrahedra: Uint32Array;
  /** Number of vertices */
  numVertices: number;
  /** Number of tetrahedra */
  numTetrahedra: number;
  /** Computation time in milliseconds */
  computeTimeMs: number;
  /** Input mesh statistics */
  inputStats: {
    numVertices: number;
    numFaces: number;
  };
}

export interface TetrahedralizationOptions {
  /** Edge length factor relative to bounding box diagonal (default: 0.0065, smaller = denser mesh) */
  edgeLengthFac?: number;
  /** Whether to optimize mesh quality (default: true, slower but better) */
  optimize?: boolean;
}

export interface TetrahedralizationStats {
  numVertices: number;
  numTetrahedra: number;
  inputVertices: number;
  inputFaces: number;
  computeTimeMs: number;
}

export interface TetrahedralizationProgress {
  job_id: string;
  status: 'pending' | 'loading' | 'preprocessing' | 'tetrahedralizing' | 'postprocessing' | 'complete' | 'downloading' | 'visualizing' | 'error';
  progress: number;  // 0-100
  message: string;
  substep: string;  // Current substep description
  elapsed_seconds: number;
  estimated_remaining: number | null;  // Estimated seconds remaining
  input_stats: { num_vertices: number; num_faces: number } | null;
  complete: boolean;
  error: string | null;
  logs?: Array<{ time: number; message: string }>;  // Streaming log messages
  log_index?: number;  // Index for fetching new logs
}

export type ProgressCallback = (progress: TetrahedralizationProgress) => void;

// ============================================================================
// API CONFIGURATION
// ============================================================================

const API_BASE_URL = 'http://localhost:8000';

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Convert a THREE.js BufferGeometry to an STL binary buffer
 */
export function geometryToSTLBuffer(geometry: THREE.BufferGeometry): ArrayBuffer {
  const positions = geometry.getAttribute('position');
  const indices = geometry.getIndex();
  
  let triangleCount: number;
  
  if (indices) {
    triangleCount = indices.count / 3;
  } else {
    triangleCount = positions.count / 3;
  }
  
  // STL binary format:
  // - 80 bytes header
  // - 4 bytes triangle count (uint32)
  // - For each triangle:
  //   - 12 bytes normal (3 x float32)
  //   - 36 bytes vertices (9 x float32)
  //   - 2 bytes attribute byte count (uint16)
  
  const headerSize = 80;
  const triangleSize = 50; // 12 + 36 + 2
  const bufferSize = headerSize + 4 + triangleCount * triangleSize;
  
  const buffer = new ArrayBuffer(bufferSize);
  const dataView = new DataView(buffer);
  
  // Write header (80 bytes of zeros)
  // Header is already zeros from ArrayBuffer initialization
  
  // Write triangle count
  dataView.setUint32(80, triangleCount, true);
  
  let offset = 84;
  
  const vertex = new THREE.Vector3();
  const normal = new THREE.Vector3();
  const cb = new THREE.Vector3();
  const ab = new THREE.Vector3();
  
  for (let i = 0; i < triangleCount; i++) {
    let i0: number, i1: number, i2: number;
    
    if (indices) {
      i0 = indices.getX(i * 3);
      i1 = indices.getX(i * 3 + 1);
      i2 = indices.getX(i * 3 + 2);
    } else {
      i0 = i * 3;
      i1 = i * 3 + 1;
      i2 = i * 3 + 2;
    }
    
    // Get vertices
    const v0 = new THREE.Vector3(
      positions.getX(i0),
      positions.getY(i0),
      positions.getZ(i0)
    );
    const v1 = new THREE.Vector3(
      positions.getX(i1),
      positions.getY(i1),
      positions.getZ(i1)
    );
    const v2 = new THREE.Vector3(
      positions.getX(i2),
      positions.getY(i2),
      positions.getZ(i2)
    );
    
    // Calculate normal
    cb.subVectors(v2, v1);
    ab.subVectors(v0, v1);
    cb.cross(ab);
    cb.normalize();
    
    // Write normal
    dataView.setFloat32(offset, cb.x, true); offset += 4;
    dataView.setFloat32(offset, cb.y, true); offset += 4;
    dataView.setFloat32(offset, cb.z, true); offset += 4;
    
    // Write vertices
    dataView.setFloat32(offset, v0.x, true); offset += 4;
    dataView.setFloat32(offset, v0.y, true); offset += 4;
    dataView.setFloat32(offset, v0.z, true); offset += 4;
    
    dataView.setFloat32(offset, v1.x, true); offset += 4;
    dataView.setFloat32(offset, v1.y, true); offset += 4;
    dataView.setFloat32(offset, v1.z, true); offset += 4;
    
    dataView.setFloat32(offset, v2.x, true); offset += 4;
    dataView.setFloat32(offset, v2.y, true); offset += 4;
    dataView.setFloat32(offset, v2.z, true); offset += 4;
    
    // Write attribute byte count (0)
    dataView.setUint16(offset, 0, true); offset += 2;
  }
  
  return buffer;
}

/**
 * Create a File object from geometry for upload
 */
export function geometryToSTLFile(geometry: THREE.BufferGeometry, filename: string = 'mesh.stl'): File {
  const buffer = geometryToSTLBuffer(geometry);
  return new File([buffer], filename, { type: 'application/sla' });
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * Check if the backend server is available
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

/**
 * Start a tetrahedralization job and return the job ID
 */
async function startTetrahedralizationJob(
  geometry: THREE.BufferGeometry,
  options: TetrahedralizationOptions = {}
): Promise<string> {
  const { edgeLengthFac = 0.05, optimize = false } = options;
  
  // Convert geometry to STL file
  const stlFile = geometryToSTLFile(geometry);
  
  // Create form data
  const formData = new FormData();
  formData.append('file', stlFile);
  
  // Build URL with query parameters
  const url = new URL(`${API_BASE_URL}/tetrahedralize/start`);
  url.searchParams.set('edge_length_fac', edgeLengthFac.toString());
  url.searchParams.set('optimize', optimize.toString());
  
  const response = await fetch(url.toString(), {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to start tetrahedralization');
  }
  
  const data = await response.json();
  return data.job_id;
}

/**
 * Poll for job progress
 */
async function getJobProgress(jobId: string, lastLogIndex: number = 0): Promise<TetrahedralizationProgress> {
  const response = await fetch(`${API_BASE_URL}/tetrahedralize/progress/${jobId}?last_log_index=${lastLogIndex}`);
  if (!response.ok) {
    throw new Error('Failed to get job progress');
  }
  return response.json();
}

/**
 * Get job result with progress callbacks for download and processing
 */
async function getJobResult(
  jobId: string, 
  onProgress?: (stage: string, detail: string) => void
): Promise<TetrahedralizationResult> {
  onProgress?.('downloading', 'Fetching result from server...');
  
  const response = await fetch(`${API_BASE_URL}/tetrahedralize/result/${jobId}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get result');
  }
  
  onProgress?.('downloading', 'Parsing JSON response...');
  const data = await response.json();
  
  onProgress?.('processing', `Converting ${data.output_stats.num_vertices.toLocaleString()} vertices...`);
  // Convert arrays to typed arrays - this can be slow for large meshes
  // Use setTimeout to allow UI to update
  await new Promise(resolve => setTimeout(resolve, 10));
  
  const vertices = new Float32Array(data.vertices.flat());
  
  onProgress?.('processing', `Converting ${data.output_stats.num_tetrahedra.toLocaleString()} tetrahedra...`);
  await new Promise(resolve => setTimeout(resolve, 10));
  
  const tetrahedra = new Uint32Array(data.tetrahedra.flat());
  
  onProgress?.('processing', 'Data ready');
  
  return {
    vertices,
    tetrahedra,
    numVertices: data.output_stats.num_vertices,
    numTetrahedra: data.output_stats.num_tetrahedra,
    computeTimeMs: 0, // Will be calculated by caller
    inputStats: {
      numVertices: data.input_stats.num_vertices,
      numFaces: data.input_stats.num_faces,
    },
  };
}

/**
 * Tetrahedralize a THREE.js geometry with progress tracking
 * Progress continues until result is returned (includes download time)
 */
export async function tetrahedralizeGeometryWithProgress(
  geometry: THREE.BufferGeometry,
  options: TetrahedralizationOptions = {},
  onProgress?: ProgressCallback
): Promise<TetrahedralizationResult> {
  const startTime = performance.now();
  
  // Start job
  const jobId = await startTetrahedralizationJob(geometry, options);
  
  // Track logs for streaming
  let lastLogIndex = 0;
  let allLogs: Array<{ time: number; message: string }> = [];
  
  // Poll for progress more frequently for smoother updates
  let progress: TetrahedralizationProgress;
  do {
    await new Promise(resolve => setTimeout(resolve, 100)); // Poll every 100ms for smooth updates
    progress = await getJobProgress(jobId, lastLogIndex);
    
    // Accumulate logs
    if (progress.logs && progress.logs.length > 0) {
      allLogs = [...allLogs, ...progress.logs];
      lastLogIndex = progress.log_index || lastLogIndex;
    }
    
    // Include all logs in progress callback
    onProgress?.({ ...progress, logs: allLogs });
  } while (!progress.complete);
  
  if (progress.status === 'error') {
    throw new Error(progress.error || 'Tetrahedralization failed');
  }
  
  // Get result with progress updates for download/processing
  const result = await getJobResult(jobId, (stage, detail) => {
    const downloadProgress: TetrahedralizationProgress = {
      ...progress,
      status: stage === 'downloading' ? 'downloading' : 'postprocessing',
      progress: 99,
      message: stage === 'downloading' ? 'Downloading mesh data...' : 'Processing mesh data...',
      substep: detail,
      logs: allLogs,
    };
    onProgress?.(downloadProgress);
  });
  
  result.computeTimeMs = performance.now() - startTime;
  
  return result;
}

/**
 * Tetrahedralize a THREE.js geometry by sending it to the backend (simple version without progress)
 */
export async function tetrahedralizeGeometry(
  geometry: THREE.BufferGeometry,
  options: TetrahedralizationOptions = {}
): Promise<TetrahedralizationResult> {
  const startTime = performance.now();
  
  const { edgeLengthFac = 0.05, optimize = false } = options;
  
  // Convert geometry to STL file
  const stlFile = geometryToSTLFile(geometry);
  
  // Create form data
  const formData = new FormData();
  formData.append('file', stlFile);
  
  // Build URL with query parameters
  const url = new URL(`${API_BASE_URL}/tetrahedralize`);
  url.searchParams.set('edge_length_fac', edgeLengthFac.toString());
  url.searchParams.set('optimize', optimize.toString());
  
  // Make request
  const response = await fetch(url.toString(), {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Tetrahedralization failed');
  }
  
  const data = await response.json();
  
  // Convert arrays to typed arrays
  const vertices = new Float32Array(data.vertices.flat());
  const tetrahedra = new Uint32Array(data.tetrahedra.flat());
  
  const endTime = performance.now();
  
  return {
    vertices,
    tetrahedra,
    numVertices: data.output_stats.num_vertices,
    numTetrahedra: data.output_stats.num_tetrahedra,
    computeTimeMs: endTime - startTime,
    inputStats: {
      numVertices: data.input_stats.num_vertices,
      numFaces: data.input_stats.num_faces,
    },
  };
}

/**
 * Tetrahedralize a geometry and get only statistics (faster for testing)
 */
export async function tetrahedralizeGeometryStats(
  geometry: THREE.BufferGeometry,
  options: TetrahedralizationOptions = {}
): Promise<TetrahedralizationStats> {
  const startTime = performance.now();
  
  const { edgeLengthFac = 0.05, optimize = false } = options;
  
  // Convert geometry to STL file
  const stlFile = geometryToSTLFile(geometry);
  
  // Create form data
  const formData = new FormData();
  formData.append('file', stlFile);
  
  // Build URL with query parameters
  const url = new URL(`${API_BASE_URL}/tetrahedralize/stats`);
  url.searchParams.set('edge_length_fac', edgeLengthFac.toString());
  url.searchParams.set('optimize', optimize.toString());
  
  // Make request
  const response = await fetch(url.toString(), {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Tetrahedralization failed');
  }
  
  const data = await response.json();
  const endTime = performance.now();
  
  return {
    numVertices: data.output_stats.num_vertices,
    numTetrahedra: data.output_stats.num_tetrahedra,
    inputVertices: data.input_stats.num_vertices,
    inputFaces: data.input_stats.num_faces,
    computeTimeMs: endTime - startTime,
  };
}

// ============================================================================
// VISUALIZATION HELPERS
// ============================================================================

/**
 * Create a point cloud visualization from tetrahedra centers
 */
export function createTetrahedraPointCloud(
  result: TetrahedralizationResult,
  color: number = 0x00ffff,
  pointSize: number = 3
): THREE.Points {
  const { vertices, tetrahedra, numTetrahedra } = result;
  
  // Calculate tetrahedra centers
  const centers = new Float32Array(numTetrahedra * 3);
  
  for (let i = 0; i < numTetrahedra; i++) {
    const i0 = tetrahedra[i * 4] * 3;
    const i1 = tetrahedra[i * 4 + 1] * 3;
    const i2 = tetrahedra[i * 4 + 2] * 3;
    const i3 = tetrahedra[i * 4 + 3] * 3;
    
    centers[i * 3] = (vertices[i0] + vertices[i1] + vertices[i2] + vertices[i3]) / 4;
    centers[i * 3 + 1] = (vertices[i0 + 1] + vertices[i1 + 1] + vertices[i2 + 1] + vertices[i3 + 1]) / 4;
    centers[i * 3 + 2] = (vertices[i0 + 2] + vertices[i1 + 2] + vertices[i2 + 2] + vertices[i3 + 2]) / 4;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(centers, 3));
  
  const material = new THREE.PointsMaterial({
    color,
    size: pointSize,
    sizeAttenuation: false,
  });
  
  const points = new THREE.Points(geometry, material);
  points.name = 'tetrahedra-points';
  
  return points;
}

/**
 * Create a wireframe visualization of tetrahedra edges
 */
export function createTetrahedraWireframe(
  result: TetrahedralizationResult,
  color: number = 0x00ffff,
  opacity: number = 0.3
): THREE.LineSegments {
  const { vertices, tetrahedra, numTetrahedra } = result;
  
  // Each tetrahedron has 6 edges
  const edgeIndices: number[] = [];
  
  // Edge pairs for a tetrahedron (0-1, 0-2, 0-3, 1-2, 1-3, 2-3)
  const edgePairs = [
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
  ];
  
  for (let i = 0; i < numTetrahedra; i++) {
    const v0 = tetrahedra[i * 4];
    const v1 = tetrahedra[i * 4 + 1];
    const v2 = tetrahedra[i * 4 + 2];
    const v3 = tetrahedra[i * 4 + 3];
    const tetVerts = [v0, v1, v2, v3];
    
    for (const [a, b] of edgePairs) {
      edgeIndices.push(tetVerts[a], tetVerts[b]);
    }
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(edgeIndices);
  
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
  });
  
  const wireframe = new THREE.LineSegments(geometry, material);
  wireframe.name = 'tetrahedra-wireframe';
  
  return wireframe;
}

/**
 * Create a bounding box helper for the tetrahedralized mesh
 */
export function createTetrahedraBoundingBox(
  result: TetrahedralizationResult,
  color: number = 0xffff00
): THREE.LineSegments {
  const { vertices, numVertices } = result;
  
  // Find bounding box
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  
  for (let i = 0; i < numVertices; i++) {
    const x = vertices[i * 3];
    const y = vertices[i * 3 + 1];
    const z = vertices[i * 3 + 2];
    
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  
  const box = new THREE.Box3(
    new THREE.Vector3(minX, minY, minZ),
    new THREE.Vector3(maxX, maxY, maxZ)
  );
  
  const helper = new THREE.Box3Helper(box, new THREE.Color(color));
  helper.name = 'tetrahedra-bounds';
  
  return helper;
}

/**
 * Remove tetrahedralization visualization from scene
 */
export function removeTetrahedralizationVisualization(
  scene: THREE.Scene,
  visualization: THREE.Object3D
): void {
  scene.remove(visualization);
  
  if (visualization instanceof THREE.Points || visualization instanceof THREE.LineSegments) {
    visualization.geometry?.dispose();
    if (Array.isArray(visualization.material)) {
      visualization.material.forEach(m => m.dispose());
    } else {
      (visualization.material as THREE.Material)?.dispose();
    }
  }
}

// ============================================================================
// SAVE/LOAD TETRAHEDRAL MESH
// ============================================================================

/**
 * File format for saved tetrahedral mesh (.tetra.json)
 */
interface TetraMeshFileFormat {
  version: 1;
  numVertices: number;
  numTetrahedra: number;
  vertices: number[];  // Flat array of x,y,z coordinates
  tetrahedra: number[];  // Flat array of 4 vertex indices per tetrahedron
  inputStats: {
    numVertices: number;
    numFaces: number;
  };
  computeTimeMs: number;
  savedAt: string;  // ISO timestamp
}

/**
 * Save a tetrahedral mesh result to a downloadable JSON file
 */
export function saveTetrahedralMesh(result: TetrahedralizationResult, filename?: string): void {
  const data: TetraMeshFileFormat = {
    version: 1,
    numVertices: result.numVertices,
    numTetrahedra: result.numTetrahedra,
    vertices: Array.from(result.vertices),
    tetrahedra: Array.from(result.tetrahedra),
    inputStats: result.inputStats,
    computeTimeMs: result.computeTimeMs,
    savedAt: new Date().toISOString(),
  };
  
  const json = JSON.stringify(data);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const defaultFilename = filename || `tetra_mesh_${result.numVertices}v_${result.numTetrahedra}t.tetra.json`;
  
  const a = document.createElement('a');
  a.href = url;
  a.download = defaultFilename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  console.log(`Saved tetrahedral mesh to ${defaultFilename}`);
  console.log(`  Vertices: ${result.numVertices}, Tetrahedra: ${result.numTetrahedra}`);
}

/**
 * Load a tetrahedral mesh from a JSON file
 * Returns a Promise that resolves to the TetrahedralizationResult
 */
export function loadTetrahedralMesh(file: File): Promise<TetrahedralizationResult> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (event) => {
      try {
        const json = event.target?.result as string;
        const data = JSON.parse(json) as TetraMeshFileFormat;
        
        // Validate file format
        if (data.version !== 1) {
          throw new Error(`Unsupported file version: ${data.version}`);
        }
        
        if (!data.vertices || !data.tetrahedra || !data.numVertices || !data.numTetrahedra) {
          throw new Error('Invalid tetrahedral mesh file: missing required fields');
        }
        
        // Convert arrays back to typed arrays
        const result: TetrahedralizationResult = {
          vertices: new Float32Array(data.vertices),
          tetrahedra: new Uint32Array(data.tetrahedra),
          numVertices: data.numVertices,
          numTetrahedra: data.numTetrahedra,
          inputStats: data.inputStats || { numVertices: 0, numFaces: 0 },
          computeTimeMs: data.computeTimeMs || 0,
        };
        
        // Validate array lengths
        if (result.vertices.length !== data.numVertices * 3) {
          throw new Error(`Invalid vertices array length: expected ${data.numVertices * 3}, got ${result.vertices.length}`);
        }
        if (result.tetrahedra.length !== data.numTetrahedra * 4) {
          throw new Error(`Invalid tetrahedra array length: expected ${data.numTetrahedra * 4}, got ${result.tetrahedra.length}`);
        }
        
        console.log(`Loaded tetrahedral mesh from ${file.name}`);
        console.log(`  Vertices: ${result.numVertices}, Tetrahedra: ${result.numTetrahedra}`);
        console.log(`  Original compute time: ${result.computeTimeMs.toFixed(0)} ms`);
        if (data.savedAt) {
          console.log(`  Saved at: ${data.savedAt}`);
        }
        
        resolve(result);
      } catch (error) {
        reject(error);
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsText(file);
  });
}

/**
 * Open a file picker dialog and load a tetrahedral mesh
 * Returns a Promise that resolves to the TetrahedralizationResult
 */
export function openTetrahedralMeshFile(): Promise<TetrahedralizationResult> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.tetra.json';
    
    input.onchange = async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) {
        reject(new Error('No file selected'));
        return;
      }
      
      try {
        const result = await loadTetrahedralMesh(file);
        resolve(result);
      } catch (error) {
        reject(error);
      }
    };
    
    input.click();
  });
}
