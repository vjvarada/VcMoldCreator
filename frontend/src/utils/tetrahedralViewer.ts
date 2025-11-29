/**
 * Tetrahedral Mesh Visualization
 * 
 * Provides utilities for visualizing tetrahedral meshes in Three.js:
 * - Wireframe view (shows all tet edges)
 * - Surface view (outer boundary faces)
 */

import * as THREE from 'three';

// ============================================================================
// TYPES
// ============================================================================

export interface TetMeshData {
  /** Flat array of vertex positions [x0,y0,z0, x1,y1,z1, ...] */
  vertices: number[];
  /** Flat array of tetrahedra indices [v0,v1,v2,v3, ...] (4 indices per tet) */
  tetrahedra: number[];
  /** Number of vertices */
  num_vertices: number;
  /** Number of tetrahedra */
  num_tetrahedra: number;
  /** Bounding box diagonal */
  bbox_diagonal?: number;
  /** Target edge length used */
  target_edge_length?: number;
  /** Max volume used */
  max_volume_used?: number;
}

export interface TetMeshVisualization {
  /** The wireframe mesh showing tet edges */
  wireframe: THREE.LineSegments;
  /** The surface mesh showing outer boundary */
  surface: THREE.Mesh;
  /** Group containing all visualizations */
  group: THREE.Group;
  /** Statistics about the visualization */
  stats: {
    numVertices: number;
    numTetrahedra: number;
    numEdges: number;
    numSurfaceFaces: number;
  };
}

export interface TetrahedralizeProgress {
  step: number;
  total: number;
  message: string;
}

// ============================================================================
// CONSTANTS
// ============================================================================

export const TET_COLORS = {
  WIREFRAME: 0x00ff88,
  SURFACE: 0xff8844,
};

const API_BASE_URL = 'http://localhost:8000';

// Flag to use simple endpoint instead of SSE streaming
const USE_SIMPLE_ENDPOINT = true;

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * Export a Three.js mesh geometry to STL blob (binary format)
 */
export function geometryToSTLBlob(geometry: THREE.BufferGeometry): Blob {
  const positions = geometry.getAttribute('position');
  const indices = geometry.getIndex();
  
  let triangleCount: number;
  if (indices) {
    triangleCount = indices.count / 3;
  } else {
    triangleCount = positions.count / 3;
  }
  
  // Binary STL format
  const headerSize = 80;
  const triangleCountSize = 4;
  const triangleSize = 50; // 12 bytes normal + 36 bytes vertices + 2 bytes attribute
  const bufferSize = headerSize + triangleCountSize + triangleSize * triangleCount;
  
  const buffer = new ArrayBuffer(bufferSize);
  const dataView = new DataView(buffer);
  
  // Header (80 bytes) - fill with zeros
  for (let i = 0; i < 80; i++) {
    dataView.setUint8(i, 0);
  }
  
  // Triangle count
  dataView.setUint32(80, triangleCount, true);
  
  let offset = 84;
  
  const getVertex = (index: number): [number, number, number] => {
    return [
      positions.getX(index),
      positions.getY(index),
      positions.getZ(index),
    ];
  };
  
  const computeNormal = (v0: number[], v1: number[], v2: number[]): [number, number, number] => {
    const ax = v1[0] - v0[0];
    const ay = v1[1] - v0[1];
    const az = v1[2] - v0[2];
    const bx = v2[0] - v0[0];
    const by = v2[1] - v0[1];
    const bz = v2[2] - v0[2];
    
    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;
    
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    return len > 0 ? [nx / len, ny / len, nz / len] : [0, 0, 1];
  };
  
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
    
    const v0 = getVertex(i0);
    const v1 = getVertex(i1);
    const v2 = getVertex(i2);
    const normal = computeNormal(v0, v1, v2);
    
    // Normal
    dataView.setFloat32(offset, normal[0], true); offset += 4;
    dataView.setFloat32(offset, normal[1], true); offset += 4;
    dataView.setFloat32(offset, normal[2], true); offset += 4;
    
    // Vertices
    dataView.setFloat32(offset, v0[0], true); offset += 4;
    dataView.setFloat32(offset, v0[1], true); offset += 4;
    dataView.setFloat32(offset, v0[2], true); offset += 4;
    
    dataView.setFloat32(offset, v1[0], true); offset += 4;
    dataView.setFloat32(offset, v1[1], true); offset += 4;
    dataView.setFloat32(offset, v1[2], true); offset += 4;
    
    dataView.setFloat32(offset, v2[0], true); offset += 4;
    dataView.setFloat32(offset, v2[1], true); offset += 4;
    dataView.setFloat32(offset, v2[2], true); offset += 4;
    
    // Attribute byte count
    dataView.setUint16(offset, 0, true); offset += 2;
  }
  
  return new Blob([buffer], { type: 'application/octet-stream' });
}

/**
 * Tetrahedralize a mesh with progress updates via SSE
 */
export async function tetrahedralizeMeshWithProgress(
  mesh: THREE.Mesh,
  callbacks: {
    onProgress?: (progress: TetrahedralizeProgress) => void;
    onComplete?: (data: TetMeshData) => void;
    onError?: (error: string) => void;
  },
  options: {
    edgeLengthRatio?: number;
    mindihedral?: number;
    minratio?: number;
  } = {}
): Promise<TetMeshData | null> {
  const { onProgress, onComplete, onError } = callbacks;
  const { edgeLengthRatio = 0.0065, mindihedral = 20.0, minratio = 1.5 } = options;

  // Get geometry in world coordinates
  const geometry = mesh.geometry.clone();
  geometry.applyMatrix4(mesh.matrixWorld);
  
  // Convert to STL blob
  const meshBlob = geometryToSTLBlob(geometry);
  geometry.dispose();

  const formData = new FormData();
  formData.append('file', meshBlob, 'mesh.stl');

  // Use simple endpoint instead of SSE streaming (more reliable)
  if (USE_SIMPLE_ENDPOINT) {
    const url = new URL(`${API_BASE_URL}/tetrahedralize`);
    url.searchParams.set('edge_length_ratio', edgeLengthRatio.toString());
    url.searchParams.set('mindihedral', mindihedral.toString());
    url.searchParams.set('minratio', minratio.toString());
    url.searchParams.set('output_format', 'json');

    try {
      console.log('Starting tetrahedralization request (simple) to:', url.toString());
      onProgress?.({ step: 1, total: 2, message: 'Sending mesh to server...' });
      
      const response = await fetch(url.toString(), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      onProgress?.({ step: 2, total: 2, message: 'Processing response...' });
      
      const data = await response.json() as TetMeshData;
      console.log('Tetrahedralization complete:', data.num_vertices, 'vertices,', data.num_tetrahedra, 'tets');
      onComplete?.(data);
      return data;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      console.error('Tetrahedralization request failed:', message, error);
      onError?.(message);
      return null;
    }
  }

  // SSE streaming endpoint (fallback)
  const url = new URL(`${API_BASE_URL}/tetrahedralize/stream`);
  url.searchParams.set('edge_length_ratio', edgeLengthRatio.toString());
  url.searchParams.set('mindihedral', mindihedral.toString());
  url.searchParams.set('minratio', minratio.toString());

  try {
    console.log('Starting tetrahedralization request to:', url.toString());
    
    const response = await fetch(url.toString(), {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let result: TetMeshData | null = null;
    let currentEvent = ''; // Move outside the loop to persist across chunks

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        console.log('SSE stream ended');
        // Process any remaining buffer content
        if (buffer.trim()) {
          console.log('Remaining buffer length:', buffer.length);
          console.log('Remaining buffer start:', buffer.substring(0, 200));
          console.log('Remaining buffer end:', buffer.substring(buffer.length - 200));
          
          // Try to parse remaining buffer as a complete message
          const lines = buffer.split('\n');
          let eventType = currentEvent;
          let eventData = '';
          
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6);
            }
          }
          
          if (eventData && eventType === 'complete') {
            try {
              const data = JSON.parse(eventData);
              console.log('Parsed remaining complete event:', data.num_vertices, 'vertices,', data.num_tetrahedra, 'tets');
              result = data as TetMeshData;
              onComplete?.(result);
            } catch (e) {
              console.warn('Failed to parse remaining buffer as complete event:', e);
            }
          }
        }
        break;
      }
      
      buffer += decoder.decode(value, { stream: true });
      
      // Parse SSE messages - look for complete messages (ending with double newline)
      let messageEnd: number;
      while ((messageEnd = buffer.indexOf('\n\n')) !== -1) {
        const message = buffer.substring(0, messageEnd);
        buffer = buffer.substring(messageEnd + 2);
        
        // Parse the message
        const lines = message.split('\n');
        let eventType = currentEvent;
        let eventData = '';
        
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
            currentEvent = eventType;
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6);
          }
        }
        
        if (eventData) {
          try {
            const data = JSON.parse(eventData);
            
            if (eventType === 'progress') {
              console.log('Progress:', data);
              onProgress?.(data as TetrahedralizeProgress);
            } else if (eventType === 'complete') {
              console.log('Tetrahedralization complete:', data.num_vertices, 'vertices,', data.num_tetrahedra, 'tets');
              result = data as TetMeshData;
              onComplete?.(result);
            } else if (eventType === 'error') {
              console.error('Backend error:', data.message);
              onError?.(data.message);
              return null;
            }
          } catch (e) {
            // Only warn if it looks like it should be JSON
            if (eventData.trim() && eventData.startsWith('{')) {
              console.warn('Failed to parse SSE data (length:', eventData.length, '):', e);
            }
          }
        }
      }
    }

    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    console.error('Tetrahedralization request failed:', message, error);
    onError?.(message);
    return null;
  }
}

// ============================================================================
// VISUALIZATION FUNCTIONS
// ============================================================================

/**
 * Extract surface faces (faces that are only used by one tetrahedron)
 */
function extractSurfaceFaces(data: TetMeshData): number[] {
  const { tetrahedra, num_tetrahedra } = data;
  
  // Count face occurrences
  const faceCount = new Map<string, { indices: number[]; count: number }>();
  
  // Each tet has 4 faces
  const faceIndices = [
    [0, 2, 1], // face opposite to vertex 3
    [0, 1, 3], // face opposite to vertex 2
    [0, 3, 2], // face opposite to vertex 1
    [1, 2, 3], // face opposite to vertex 0
  ];
  
  for (let i = 0; i < num_tetrahedra; i++) {
    const t = [
      tetrahedra[i * 4],
      tetrahedra[i * 4 + 1],
      tetrahedra[i * 4 + 2],
      tetrahedra[i * 4 + 3],
    ];
    
    for (const [a, b, c] of faceIndices) {
      const faceVerts = [t[a], t[b], t[c]].sort((x, y) => x - y);
      const key = faceVerts.join(',');
      
      if (faceCount.has(key)) {
        faceCount.get(key)!.count++;
      } else {
        faceCount.set(key, { indices: [t[a], t[b], t[c]], count: 1 });
      }
    }
  }
  
  // Surface faces are those that appear only once
  const surfaceIndices: number[] = [];
  for (const { indices, count } of faceCount.values()) {
    if (count === 1) {
      surfaceIndices.push(...indices);
    }
  }
  
  return surfaceIndices;
}

/**
 * Create a wireframe visualization showing all tetrahedra edges
 */
export function createTetWireframe(data: TetMeshData, opacity = 0.3): THREE.LineSegments {
  const { vertices, tetrahedra, num_tetrahedra } = data;
  
  const positions: number[] = [];
  
  for (let i = 0; i < num_tetrahedra; i++) {
    const t0 = tetrahedra[i * 4];
    const t1 = tetrahedra[i * 4 + 1];
    const t2 = tetrahedra[i * 4 + 2];
    const t3 = tetrahedra[i * 4 + 3];
    
    // Get vertex positions
    const v0 = [vertices[t0 * 3], vertices[t0 * 3 + 1], vertices[t0 * 3 + 2]];
    const v1 = [vertices[t1 * 3], vertices[t1 * 3 + 1], vertices[t1 * 3 + 2]];
    const v2 = [vertices[t2 * 3], vertices[t2 * 3 + 1], vertices[t2 * 3 + 2]];
    const v3 = [vertices[t3 * 3], vertices[t3 * 3 + 1], vertices[t3 * 3 + 2]];
    
    // 6 edges per tetrahedron
    const edges: [number[], number[]][] = [
      [v0, v1], [v0, v2], [v0, v3],
      [v1, v2], [v1, v3], [v2, v3]
    ];
    
    for (const [a, b] of edges) {
      positions.push(...a, ...b);
    }
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  
  const material = new THREE.LineBasicMaterial({ 
    color: TET_COLORS.WIREFRAME,
    transparent: true,
    opacity,
    depthTest: true,
  });
  
  const wireframe = new THREE.LineSegments(geometry, material);
  wireframe.name = 'TetWireframe';
  
  return wireframe;
}

/**
 * Create a surface mesh showing the outer boundary of the tetrahedral mesh
 */
export function createTetSurface(
  data: TetMeshData, 
  opacity = 0.85,
  color = TET_COLORS.SURFACE
): THREE.Mesh {
  const { vertices } = data;
  
  // Extract only surface faces
  const surfaceIndices = extractSurfaceFaces(data);
  
  // Build positions array
  const positions: number[] = [];
  for (let i = 0; i < surfaceIndices.length; i++) {
    const idx = surfaceIndices[i];
    positions.push(
      vertices[idx * 3],
      vertices[idx * 3 + 1],
      vertices[idx * 3 + 2]
    );
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  
  const material = new THREE.MeshPhongMaterial({
    color,
    side: THREE.DoubleSide,
    transparent: true,
    opacity,
    specular: 0x333333,
    shininess: 30,
  });
  
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'TetSurface';
  
  return mesh;
}

/**
 * Create a complete visualization of a tetrahedral mesh
 */
export function createTetVisualization(
  data: TetMeshData,
  options: {
    showWireframe?: boolean;
    showSurface?: boolean;
    wireframeOpacity?: number;
    surfaceOpacity?: number;
    surfaceColor?: number;
  } = {}
): TetMeshVisualization {
  const {
    showWireframe = true,
    showSurface = true,
    wireframeOpacity = 0.3,
    surfaceOpacity = 0.85,
    surfaceColor = TET_COLORS.SURFACE,
  } = options;

  // Validate data
  console.log('Creating tet visualization with data:', {
    num_vertices: data.num_vertices,
    num_tetrahedra: data.num_tetrahedra,
    vertices_length: data.vertices?.length,
    tetrahedra_length: data.tetrahedra?.length,
    sample_vertices: data.vertices?.slice(0, 12),
    sample_tetrahedra: data.tetrahedra?.slice(0, 8),
  });

  // Check for NaN values
  const hasNaN = data.vertices?.some(v => isNaN(v) || !isFinite(v));
  if (hasNaN) {
    console.error('Vertex data contains NaN or Infinity values!');
    const nanIndices = data.vertices?.map((v, i) => isNaN(v) || !isFinite(v) ? i : -1).filter(i => i >= 0);
    console.error('NaN indices:', nanIndices?.slice(0, 20));
  }

  const group = new THREE.Group();
  group.name = 'TetMeshVisualization';

  // Create wireframe
  const wireframe = createTetWireframe(data, wireframeOpacity);
  wireframe.visible = showWireframe;
  group.add(wireframe);

  // Create surface mesh
  const surface = createTetSurface(data, surfaceOpacity, surfaceColor);
  surface.visible = showSurface;
  group.add(surface);

  // Calculate statistics
  const numEdges = data.num_tetrahedra * 6;
  const surfaceFaces = extractSurfaceFaces(data);

  return {
    wireframe,
    surface,
    group,
    stats: {
      numVertices: data.num_vertices,
      numTetrahedra: data.num_tetrahedra,
      numEdges,
      numSurfaceFaces: surfaceFaces.length / 3,
    },
  };
}

/**
 * Remove tetrahedral visualization from scene
 */
export function removeTetVisualization(
  scene: THREE.Scene,
  visualization: TetMeshVisualization
): void {
  scene.remove(visualization.group);
  
  // Dispose geometries and materials
  visualization.wireframe.geometry.dispose();
  (visualization.wireframe.material as THREE.Material).dispose();
  
  visualization.surface.geometry.dispose();
  (visualization.surface.material as THREE.Material).dispose();
}
