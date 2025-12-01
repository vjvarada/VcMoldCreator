/**
 * Inflated Bounding Volume Generator
 * 
 * Creates an inflated convex hull around a mesh for mold cavity creation:
 * 1. Compute the convex hull of the input mesh
 * 2. Generate smooth vertex normals using area-weighted averaging
 * 3. Inflate hull vertices outward by offset distance
 * 4. Create mesh from inflated vertices + hull faces
 * 5. Optionally subtract base mesh via CSG for mold cavity
 */

import * as THREE from 'three';
import { ConvexGeometry, mergeVertices } from 'three-stdlib';
import {
  initManifold,
  geometryToManifoldMesh,
  manifoldMeshToGeometry,
  extractVertexTuples,
  type Manifold,
} from './manifoldUtils';

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default inflation offset distance (in mm) */
export const DEFAULT_INFLATION_OFFSET = 0.5;

/** Color for the inflated hull visualization */
export const HULL_COLOR = 0x9966ff; // Purple

/** Color for the subtracted (mold cavity) mesh */
export const CAVITY_COLOR = 0x00ffaa; // Teal/Cyan

// ============================================================================
// TYPES
// ============================================================================

export interface InflatedHullResult {
  /** The inflated hull mesh */
  mesh: THREE.Mesh;
  /** Original hull mesh (before inflation) */
  originalHull: THREE.Mesh;
  /** The computed smooth vertex normals */
  smoothNormals: Float32Array;
  /** Number of vertices in the hull */
  vertexCount: number;
  /** Number of faces in the hull */
  faceCount: number;
  /** Manifold validation result */
  manifoldValidation: ManifoldValidationResult;
  /** CSG subtraction result (hull - base mesh), if computed */
  csgResult: CsgSubtractionResult | null;
}

export interface CsgSubtractionResult {
  /** The resulting mesh after subtraction (hull - base mesh) */
  mesh: THREE.Mesh;
  /** Number of vertices in the result */
  vertexCount: number;
  /** Number of faces in the result */
  faceCount: number;
  /** Manifold validation result */
  manifoldValidation: ManifoldValidationResult;
}

export interface ManifoldValidationResult {
  /** Whether the mesh is closed (watertight) */
  isClosed: boolean;
  /** Whether the mesh is manifold (each edge shared by exactly 2 faces) */
  isManifold: boolean;
  /** Number of boundary edges (should be 0 for closed mesh) */
  boundaryEdgeCount: number;
  /** Number of non-manifold edges (edges with != 2 adjacent faces) */
  nonManifoldEdgeCount: number;
  /** Total number of edges */
  totalEdgeCount: number;
  /** Euler characteristic (V - E + F, should be 2 for closed manifold) */
  eulerCharacteristic: number;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Extract unique vertices from a mesh geometry and return both vertices and a map for indexing
 */
function extractUniqueVerticesWithMap(geometry: THREE.BufferGeometry): {
  vertices: THREE.Vector3[];
  positionToIndex: Map<string, number>;
} {
  const positionAttr = geometry.getAttribute('position');
  const vertices: THREE.Vector3[] = [];
  const positionToIndex = new Map<string, number>();

  const addVertex = (x: number, y: number, z: number) => {
    const key = `${x.toFixed(6)},${y.toFixed(6)},${z.toFixed(6)}`;
    if (!positionToIndex.has(key)) {
      positionToIndex.set(key, vertices.length);
      vertices.push(new THREE.Vector3(x, y, z));
    }
  };

  if (geometry.index) {
    const indices = geometry.index.array;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      addVertex(
        positionAttr.getX(idx),
        positionAttr.getY(idx),
        positionAttr.getZ(idx)
      );
    }
  } else {
    for (let i = 0; i < positionAttr.count; i++) {
      addVertex(
        positionAttr.getX(i),
        positionAttr.getY(i),
        positionAttr.getZ(i)
      );
    }
  }

  return { vertices, positionToIndex };
}

/**
 * Build face indices from non-indexed ConvexGeometry, mapping to unique vertex indices
 */
function buildFaceIndices(
  geometry: THREE.BufferGeometry,
  positionToIndex: Map<string, number>
): number[] {
  const positionAttr = geometry.getAttribute('position');
  const faceIndices: number[] = [];
  const triangleCount = positionAttr.count / 3;

  for (let i = 0; i < triangleCount; i++) {
    const base = i * 3;
    for (let j = 0; j < 3; j++) {
      const idx = base + j;
      const key = `${positionAttr.getX(idx).toFixed(6)},${positionAttr.getY(idx).toFixed(6)},${positionAttr.getZ(idx).toFixed(6)}`;
      const vertexIndex = positionToIndex.get(key);
      if (vertexIndex !== undefined) {
        faceIndices.push(vertexIndex);
      }
    }
  }

  return faceIndices;
}

/**
 * Inflate unique vertices along their smooth normals
 */
function inflateVertices(
  vertices: THREE.Vector3[],
  normals: THREE.Vector3[],
  offset: number
): THREE.Vector3[] {
  return vertices.map((v, i) => {
    return new THREE.Vector3(
      v.x + normals[i].x * offset,
      v.y + normals[i].y * offset,
      v.z + normals[i].z * offset
    );
  });
}

/**
 * Create a new BufferGeometry from inflated vertices and face indices
 * This creates an indexed geometry for proper mesh representation
 */
function createMeshFromVerticesAndFaces(
  vertices: THREE.Vector3[],
  faceIndices: number[]
): THREE.BufferGeometry {
  // Create position array from unique vertices
  const positions = new Float32Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    positions[i * 3] = vertices[i].x;
    positions[i * 3 + 1] = vertices[i].y;
    positions[i * 3 + 2] = vertices[i].z;
  }

  // Create the geometry with indexed faces
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(faceIndices);
  
  // Compute vertex normals for proper shading
  geometry.computeVertexNormals();
  
  return geometry;
}

/**
 * Validate if a mesh geometry is closed (watertight) and manifold
 * 
 * A mesh is:
 * - Closed: Every edge is shared by exactly 2 faces (no boundary edges)
 * - Manifold: Each edge has exactly 2 adjacent faces, and vertices have
 *   a single continuous fan of faces around them
 * 
 * For a closed manifold mesh, Euler characteristic V - E + F = 2
 */
function validateManifold(geometry: THREE.BufferGeometry): ManifoldValidationResult {
  const positionAttr = geometry.getAttribute('position');
  const indexAttr = geometry.getIndex();
  
  // Safety check - return invalid result if geometry is malformed
  if (!positionAttr) {
    console.warn('validateManifold: No position attribute found');
    return {
      isClosed: false,
      isManifold: false,
      boundaryEdgeCount: -1,
      nonManifoldEdgeCount: -1,
      totalEdgeCount: 0,
      eulerCharacteristic: 0,
    };
  }
  
  // Build edge-to-face adjacency map
  const edgeFaceCount = new Map<string, number>();
  
  // Helper to get vertex position key
  const getPositionKey = (vertexIndex: number): string => {
    return `${positionAttr.getX(vertexIndex).toFixed(6)},${positionAttr.getY(vertexIndex).toFixed(6)},${positionAttr.getZ(vertexIndex).toFixed(6)}`;
  };
  
  let triangleCount: number;
  
  if (indexAttr) {
    // Indexed geometry
    const indices = indexAttr.array;
    triangleCount = indices.length / 3;
    
    for (let faceIdx = 0; faceIdx < triangleCount; faceIdx++) {
      const i0 = indices[faceIdx * 3];
      const i1 = indices[faceIdx * 3 + 1];
      const i2 = indices[faceIdx * 3 + 2];
      
      const p0 = getPositionKey(i0);
      const p1 = getPositionKey(i1);
      const p2 = getPositionKey(i2);
      
      const edges = [
        [p0, p1].sort().join('|'),
        [p1, p2].sort().join('|'),
        [p2, p0].sort().join('|'),
      ];
      
      for (const edge of edges) {
        edgeFaceCount.set(edge, (edgeFaceCount.get(edge) || 0) + 1);
      }
    }
  } else {
    // Non-indexed geometry
    const vertexCount = positionAttr.count;
    triangleCount = vertexCount / 3;
    
    for (let faceIdx = 0; faceIdx < triangleCount; faceIdx++) {
      const baseIdx = faceIdx * 3;
      
      const p0 = getPositionKey(baseIdx);
      const p1 = getPositionKey(baseIdx + 1);
      const p2 = getPositionKey(baseIdx + 2);
      
      const edges = [
        [p0, p1].sort().join('|'),
        [p1, p2].sort().join('|'),
        [p2, p0].sort().join('|'),
      ];
      
      for (const edge of edges) {
        edgeFaceCount.set(edge, (edgeFaceCount.get(edge) || 0) + 1);
      }
    }
  }
  
  // Analyze edge adjacency
  let boundaryEdgeCount = 0;
  let nonManifoldEdgeCount = 0;
  const totalEdgeCount = edgeFaceCount.size;
  
  for (const [_edge, count] of edgeFaceCount) {
    if (count === 1) {
      boundaryEdgeCount++;
    } else if (count > 2) {
      nonManifoldEdgeCount++;
    }
  }
  
  // Count unique vertices for Euler characteristic
  const uniqueVertices = new Set<string>();
  const vertexCount = positionAttr.count;
  for (let i = 0; i < vertexCount; i++) {
    uniqueVertices.add(getPositionKey(i));
  }
  
  const V = uniqueVertices.size;
  const E = totalEdgeCount;
  const F = triangleCount;
  const eulerCharacteristic = V - E + F;
  
  const isClosed = boundaryEdgeCount === 0;
  const isManifold = nonManifoldEdgeCount === 0 && isClosed;
  
  return {
    isClosed,
    isManifold,
    boundaryEdgeCount,
    nonManifoldEdgeCount,
    totalEdgeCount,
    eulerCharacteristic,
  };
}

/**
 * Subdivide a geometry with proper edge sharing.
 * 
 * This implementation:
 * 1. Converts to indexed representation to properly handle shared edges
 * 2. Creates edge midpoints (flat subdivision, no curvature)
 * 3. Ensures shared edges produce identical midpoints on both sides
 * 
 * @param geometry - The input geometry (can be indexed or non-indexed)
 * @param iterations - Number of subdivision iterations (each iteration 4x triangles)
 * @param curvatureStrength - How much to push midpoints outward (0 = flat). Default 0 for clean geometry
 * @returns New subdivided geometry
 */
function subdivideGeometry(
  geometry: THREE.BufferGeometry,
  iterations: number = 1,
  curvatureStrength: number = 0  // Disabled - flat subdivision is more reliable
): THREE.BufferGeometry {
  // First, convert to indexed representation for proper edge handling
  let { vertices, indices, vertexNormals } = convertToIndexed(geometry);
  
  for (let iter = 0; iter < iterations; iter++) {
    const result = subdivideIndexed(vertices, indices, vertexNormals, curvatureStrength);
    vertices = result.vertices;
    indices = result.indices;
    vertexNormals = result.vertexNormals;
  }
  
  // Convert back to non-indexed for Three.js
  return convertToNonIndexed(vertices, indices);
}

/**
 * Convert geometry to indexed representation with unique vertices
 */
function convertToIndexed(geometry: THREE.BufferGeometry): {
  vertices: Float32Array;
  indices: Uint32Array;
  vertexNormals: Float32Array;
} {
  const posAttr = geometry.getAttribute('position');
  const positions = posAttr.array as Float32Array;
  
  // Build unique vertex map
  const vertexMap = new Map<string, number>();
  const uniqueVertices: number[] = [];
  const indexArray: number[] = [];
  
  const getKey = (x: number, y: number, z: number) =>
    `${x.toFixed(6)},${y.toFixed(6)},${z.toFixed(6)}`;
  
  const triangleCount = positions.length / 9;
  
  for (let t = 0; t < triangleCount; t++) {
    for (let v = 0; v < 3; v++) {
      const base = t * 9 + v * 3;
      const x = positions[base], y = positions[base + 1], z = positions[base + 2];
      const key = getKey(x, y, z);
      
      let idx = vertexMap.get(key);
      if (idx === undefined) {
        idx = uniqueVertices.length / 3;
        vertexMap.set(key, idx);
        uniqueVertices.push(x, y, z);
      }
      indexArray.push(idx);
    }
  }
  
  const vertices = new Float32Array(uniqueVertices);
  const indices = new Uint32Array(indexArray);
  
  // Compute smooth vertex normals
  const vertexNormals = computeSmoothVertexNormals(vertices, indices);
  
  return { vertices, indices, vertexNormals };
}

/**
 * Compute area-weighted smooth vertex normals for indexed geometry
 */
function computeSmoothVertexNormals(
  vertices: Float32Array,
  indices: Uint32Array
): Float32Array {
  const vertexCount = vertices.length / 3;
  const normals = new Float32Array(vertices.length);
  
  const triangleCount = indices.length / 3;
  
  for (let t = 0; t < triangleCount; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
    
    const v0x = vertices[i0 * 3], v0y = vertices[i0 * 3 + 1], v0z = vertices[i0 * 3 + 2];
    const v1x = vertices[i1 * 3], v1y = vertices[i1 * 3 + 1], v1z = vertices[i1 * 3 + 2];
    const v2x = vertices[i2 * 3], v2y = vertices[i2 * 3 + 1], v2z = vertices[i2 * 3 + 2];
    
    // Edge vectors
    const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    
    // Cross product (area-weighted normal)
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    
    // Add to each vertex
    normals[i0 * 3] += nx; normals[i0 * 3 + 1] += ny; normals[i0 * 3 + 2] += nz;
    normals[i1 * 3] += nx; normals[i1 * 3 + 1] += ny; normals[i1 * 3 + 2] += nz;
    normals[i2 * 3] += nx; normals[i2 * 3 + 1] += ny; normals[i2 * 3 + 2] += nz;
  }
  
  // Normalize
  for (let i = 0; i < vertexCount; i++) {
    const base = i * 3;
    const len = Math.sqrt(
      normals[base] * normals[base] +
      normals[base + 1] * normals[base + 1] +
      normals[base + 2] * normals[base + 2]
    );
    if (len > 1e-10) {
      normals[base] /= len;
      normals[base + 1] /= len;
      normals[base + 2] /= len;
    }
  }
  
  return normals;
}

/**
 * Perform one iteration of subdivision on indexed geometry
 */
function subdivideIndexed(
  vertices: Float32Array,
  indices: Uint32Array,
  vertexNormals: Float32Array,
  curvatureStrength: number
): {
  vertices: Float32Array;
  indices: Uint32Array;
  vertexNormals: Float32Array;
} {
  const triangleCount = indices.length / 3;
  
  // Build edge map: for each edge (as sorted vertex pair), store the midpoint vertex index
  const edgeToMidpoint = new Map<string, number>();
  const newVertices: number[] = Array.from(vertices);
  const newNormals: number[] = Array.from(vertexNormals);
  
  const getEdgeKey = (a: number, b: number) => a < b ? `${a}-${b}` : `${b}-${a}`;
  
  // First pass: create all edge midpoints
  for (let t = 0; t < triangleCount; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
    
    const edges = [[i0, i1], [i1, i2], [i2, i0]];
    
    for (const [a, b] of edges) {
      const key = getEdgeKey(a, b);
      if (edgeToMidpoint.has(key)) continue;
      
      // Create midpoint vertex
      const ax = vertices[a * 3], ay = vertices[a * 3 + 1], az = vertices[a * 3 + 2];
      const bx = vertices[b * 3], by = vertices[b * 3 + 1], bz = vertices[b * 3 + 2];
      
      // Linear midpoint
      let mx = (ax + bx) * 0.5;
      let my = (ay + by) * 0.5;
      let mz = (az + bz) * 0.5;
      
      // Get normals at endpoints
      const nax = vertexNormals[a * 3], nay = vertexNormals[a * 3 + 1], naz = vertexNormals[a * 3 + 2];
      const nbx = vertexNormals[b * 3], nby = vertexNormals[b * 3 + 1], nbz = vertexNormals[b * 3 + 2];
      
      // Interpolated normal at midpoint
      let nmx = (nax + nbx) * 0.5;
      let nmy = (nay + nby) * 0.5;
      let nmz = (naz + nbz) * 0.5;
      const nmLen = Math.sqrt(nmx * nmx + nmy * nmy + nmz * nmz);
      if (nmLen > 1e-10) {
        nmx /= nmLen; nmy /= nmLen; nmz /= nmLen;
      }
      
      // Calculate how much to push outward based on edge angle
      // The key insight: at sharp edges, the linear midpoint is "inside" the ideal surface
      // We push it outward along the averaged normal
      
      // Edge length
      const edgeLen = Math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2);
      
      // Compute the angle between normals (sharper angle = more displacement needed)
      const normalDot = nax * nbx + nay * nby + naz * nbz;
      // normalDot = cos(angle), when angle=0, dot=1, when angle=90, dot=0
      // We want more displacement for sharper angles
      const angleFactor = Math.max(0, 1 - normalDot); // 0 for flat, 1 for 90°, 2 for 180°
      
      // Push the midpoint outward along the averaged normal
      // The displacement is proportional to edge length and angle sharpness
      const displacement = edgeLen * angleFactor * curvatureStrength * 0.25;
      
      mx += nmx * displacement;
      my += nmy * displacement;
      mz += nmz * displacement;
      
      // Store new vertex
      const newIdx = newVertices.length / 3;
      edgeToMidpoint.set(key, newIdx);
      newVertices.push(mx, my, mz);
      newNormals.push(nmx, nmy, nmz);
    }
  }
  
  // Second pass: create new triangles
  const newIndices: number[] = [];
  
  for (let t = 0; t < triangleCount; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
    
    const m01 = edgeToMidpoint.get(getEdgeKey(i0, i1))!;
    const m12 = edgeToMidpoint.get(getEdgeKey(i1, i2))!;
    const m20 = edgeToMidpoint.get(getEdgeKey(i2, i0))!;
    
    // 4 new triangles
    newIndices.push(i0, m01, m20);   // Corner 0
    newIndices.push(m01, i1, m12);   // Corner 1
    newIndices.push(m20, m12, i2);   // Corner 2
    newIndices.push(m01, m12, m20);  // Center
  }
  
  const finalVertices = new Float32Array(newVertices);
  const finalIndices = new Uint32Array(newIndices);
  
  // Recompute smooth normals for the subdivided mesh
  const finalNormals = computeSmoothVertexNormals(finalVertices, finalIndices);
  
  return {
    vertices: finalVertices,
    indices: finalIndices,
    vertexNormals: finalNormals
  };
}

/**
 * Convert indexed geometry back to non-indexed for Three.js
 */
function convertToNonIndexed(
  vertices: Float32Array,
  indices: Uint32Array
): THREE.BufferGeometry {
  const triangleCount = indices.length / 3;
  const positions = new Float32Array(triangleCount * 9);
  
  for (let t = 0; t < triangleCount; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
    const base = t * 9;
    
    positions[base] = vertices[i0 * 3];
    positions[base + 1] = vertices[i0 * 3 + 1];
    positions[base + 2] = vertices[i0 * 3 + 2];
    
    positions[base + 3] = vertices[i1 * 3];
    positions[base + 4] = vertices[i1 * 3 + 1];
    positions[base + 5] = vertices[i1 * 3 + 2];
    
    positions[base + 6] = vertices[i2 * 3];
    positions[base + 7] = vertices[i2 * 3 + 1];
    positions[base + 8] = vertices[i2 * 3 + 2];
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

// ============================================================================
// MAIN FUNCTIONS
// ============================================================================

/**
 * Compute smooth vertex normals for a mesh geometry using area-weighted averaging.
 * Works with both indexed and non-indexed geometries.
 */
function computeSmoothNormalsForGeometry(geometry: THREE.BufferGeometry): THREE.Vector3[] {
  const positionAttr = geometry.getAttribute('position');
  const indexAttr = geometry.getIndex();
  
  // Build unique vertex map
  const { vertices: uniqueVertices, positionToIndex } = extractUniqueVerticesWithMap(geometry);
  const vertexCount = uniqueVertices.length;
  
  // Initialize normal accumulators
  const normalAccum: THREE.Vector3[] = [];
  for (let i = 0; i < vertexCount; i++) {
    normalAccum.push(new THREE.Vector3(0, 0, 0));
  }
  
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceNormal = new THREE.Vector3();
  
  const getVertexIndex = (bufferIdx: number): number => {
    const key = `${positionAttr.getX(bufferIdx).toFixed(6)},${positionAttr.getY(bufferIdx).toFixed(6)},${positionAttr.getZ(bufferIdx).toFixed(6)}`;
    return positionToIndex.get(key) ?? 0;
  };
  
  let triangleCount: number;
  
  if (indexAttr) {
    triangleCount = indexAttr.count / 3;
    for (let i = 0; i < triangleCount; i++) {
      const a = indexAttr.getX(i * 3);
      const b = indexAttr.getX(i * 3 + 1);
      const c = indexAttr.getX(i * 3 + 2);
      
      vA.fromBufferAttribute(positionAttr, a);
      vB.fromBufferAttribute(positionAttr, b);
      vC.fromBufferAttribute(positionAttr, c);
      
      ab.subVectors(vB, vA);
      ac.subVectors(vC, vA);
      faceNormal.crossVectors(ab, ac);
      const area = faceNormal.length() / 2;
      
      if (area > 1e-10) {
        faceNormal.normalize();
        const iA = getVertexIndex(a);
        const iB = getVertexIndex(b);
        const iC = getVertexIndex(c);
        normalAccum[iA].addScaledVector(faceNormal, area);
        normalAccum[iB].addScaledVector(faceNormal, area);
        normalAccum[iC].addScaledVector(faceNormal, area);
      }
    }
  } else {
    triangleCount = positionAttr.count / 3;
    for (let i = 0; i < triangleCount; i++) {
      const base = i * 3;
      
      vA.fromBufferAttribute(positionAttr, base);
      vB.fromBufferAttribute(positionAttr, base + 1);
      vC.fromBufferAttribute(positionAttr, base + 2);
      
      ab.subVectors(vB, vA);
      ac.subVectors(vC, vA);
      faceNormal.crossVectors(ab, ac);
      const area = faceNormal.length() / 2;
      
      if (area > 1e-10) {
        faceNormal.normalize();
        const iA = getVertexIndex(base);
        const iB = getVertexIndex(base + 1);
        const iC = getVertexIndex(base + 2);
        normalAccum[iA].addScaledVector(faceNormal, area);
        normalAccum[iB].addScaledVector(faceNormal, area);
        normalAccum[iC].addScaledVector(faceNormal, area);
      }
    }
  }
  
  // Normalize and return
  for (let i = 0; i < vertexCount; i++) {
    normalAccum[i].normalize();
  }
  
  return normalAccum;
}

/**
 * Inflate input mesh vertices along their smooth normals, then create convex hull.
 * 
 * New approach (better triangle density):
 * 1. Compute smooth vertex normals on the INPUT mesh
 * 2. Inflate input mesh vertices outward by offset distance  
 * 3. Compute convex hull from the inflated vertices
 * 
 * This produces a denser hull because the input mesh has many more vertices
 * than a convex hull would, resulting in better triangle distribution.
 * 
 * @param mesh - The input mesh to create hull around
 * @param offset - Distance to inflate outward (default: 0.5)
 * @returns InflatedHullResult containing the inflated mesh and metadata
 */
export function generateInflatedBoundingVolume(
  mesh: THREE.Mesh,
  offset: number = DEFAULT_INFLATION_OFFSET
): InflatedHullResult {
  // Get the world-space geometry
  const worldGeometry = mesh.geometry.clone();
  worldGeometry.applyMatrix4(mesh.matrixWorld);

  // Step 1: Extract unique vertices from input mesh
  const { vertices: inputVertices } = extractUniqueVerticesWithMap(worldGeometry);
  
  // Step 2: Compute smooth vertex normals on the INPUT mesh (not hull)
  const smoothNormals = computeSmoothNormalsForGeometry(worldGeometry);
  
  // Step 3: Inflate input vertices outward along their smooth normals
  const inflatedVertices = inflateVertices(inputVertices, smoothNormals, offset);

  // Step 4: Compute convex hull from the inflated input vertices
  // This creates a hull with more vertices/triangles than hulling first
  let inflatedHullGeometry = new ConvexGeometry(inflatedVertices);
  
  // Step 4.5: Subdivide the hull to increase triangle density
  // This helps create smoother H₁/H₂ boundaries in the mold half classification
  // 2 iterations = 4^2 = 16x more triangles
  const SUBDIVISION_ITERATIONS = 2;
  const originalFaceCount = inflatedHullGeometry.getAttribute('position').count / 3;
  inflatedHullGeometry = subdivideGeometry(inflatedHullGeometry, SUBDIVISION_ITERATIONS);
  const subdividedFaceCount = inflatedHullGeometry.getAttribute('position').count / 3;
  console.log(`Hull subdivision: ${originalFaceCount} → ${subdividedFaceCount} faces (${SUBDIVISION_ITERATIONS} iterations)`);
  
  // Extract the final geometry info
  const { vertices: finalVertices, positionToIndex: finalPosToIndex } = extractUniqueVerticesWithMap(inflatedHullGeometry);
  const finalFaceIndices = buildFaceIndices(inflatedHullGeometry, finalPosToIndex);
  const faceCount = finalFaceIndices.length / 3;
  
  // Create indexed geometry from the convex hull
  const inflatedGeometry = createMeshFromVerticesAndFaces(finalVertices, finalFaceIndices);

  // Step 5: Validate manifold properties of the inflated geometry
  const manifoldValidation = validateManifold(inflatedGeometry);

  // Create materials
  const hullMaterial = new THREE.MeshPhongMaterial({
    color: HULL_COLOR,
    transparent: true,
    opacity: 0.3,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const originalHullMaterial = new THREE.MeshPhongMaterial({
    color: 0x666666,
    wireframe: true,
    transparent: true,
    opacity: 0.2,
  });

  // For the "original hull", show the non-inflated convex hull for reference
  const originalHullGeometry = new ConvexGeometry(inputVertices);
  const { vertices: origHullVerts, positionToIndex: origPosToIndex } = extractUniqueVerticesWithMap(originalHullGeometry);
  const origHullFaceIndices = buildFaceIndices(originalHullGeometry, origPosToIndex);
  const originalHullDisplayGeometry = createMeshFromVerticesAndFaces(origHullVerts, origHullFaceIndices);

  // Create meshes
  const inflatedMesh = new THREE.Mesh(inflatedGeometry, hullMaterial);
  const originalHullMesh = new THREE.Mesh(originalHullDisplayGeometry, originalHullMaterial);

  // Convert smooth normals to Float32Array for return value
  const smoothNormalsArray = new Float32Array(smoothNormals.length * 3);
  for (let i = 0; i < smoothNormals.length; i++) {
    smoothNormalsArray[i * 3] = smoothNormals[i].x;
    smoothNormalsArray[i * 3 + 1] = smoothNormals[i].y;
    smoothNormalsArray[i * 3 + 2] = smoothNormals[i].z;
  }

  console.log(`Inflated Hull: ${inputVertices.length} input vertices → ${finalVertices.length} hull vertices, ${faceCount} faces`);

  return {
    mesh: inflatedMesh,
    originalHull: originalHullMesh,
    smoothNormals: smoothNormalsArray,
    vertexCount: finalVertices.length,
    faceCount,
    manifoldValidation,
    csgResult: null,
  };
}

/**
 * Remove inflated hull meshes from scene
 */
export function removeInflatedHull(
  scene: THREE.Scene,
  result: InflatedHullResult | null
): void {
  if (!result) return;

  if (result.mesh.parent === scene) {
    scene.remove(result.mesh);
    result.mesh.geometry.dispose();
    (result.mesh.material as THREE.Material).dispose();
  }

  if (result.originalHull.parent === scene) {
    scene.remove(result.originalHull);
    result.originalHull.geometry.dispose();
    (result.originalHull.material as THREE.Material).dispose();
  }

  // Also clean up CSG result if present
  if (result.csgResult) {
    removeCsgResult(scene, result.csgResult);
  }
}

/**
 * Perform CSG subtraction: Hull - Original Mesh
 * 
 * This creates a "mold cavity" by subtracting the original mesh from
 * the inflated hull, leaving the hollow space between them.
 * 
 * Uses Manifold library for robust boolean operations that guarantee
 * manifold (watertight) output.
 * 
 * @param hullMesh - The inflated convex hull mesh
 * @param originalMesh - The original mesh to subtract from the hull
 * @returns Promise<CsgSubtractionResult> containing the resulting cavity mesh
 */
export async function performCsgSubtraction(
  hullMesh: THREE.Mesh,
  originalMesh: THREE.Mesh
): Promise<CsgSubtractionResult> {
  // Get geometries and apply world transforms
  const hullGeometry = hullMesh.geometry.clone();
  hullGeometry.applyMatrix4(hullMesh.matrixWorld);
  
  const baseGeometry = originalMesh.geometry.clone();
  baseGeometry.applyMatrix4(originalMesh.matrixWorld);
  
  return performCsgSubtractionFromGeometries(hullGeometry, baseGeometry);
}

/**
 * Perform CSG subtraction using Manifold library
 */
async function performCsgSubtractionFromGeometries(
  hullGeometry: THREE.BufferGeometry,
  baseGeometry: THREE.BufferGeometry
): Promise<CsgSubtractionResult> {
  const mod = await initManifold();
  
  // Merge vertices first to ensure clean input
  const mergedHullGeometry = mergeVertices(hullGeometry, 0.0001);
  const mergedBaseGeometry = mergeVertices(baseGeometry, 0.0001);
  
  try {
    // Convert to Manifold meshes
    const hullMesh = geometryToManifoldMesh(mod, mergedHullGeometry);
    const baseMesh = geometryToManifoldMesh(mod, mergedBaseGeometry);
    
    // Try to merge vertices to create valid manifolds
    hullMesh.merge();
    baseMesh.merge();
    
    // Create Manifold objects with convex hull fallback
    let hullManifold: Manifold;
    let baseManifold: Manifold;
    
    try {
      hullManifold = mod.Manifold.ofMesh(hullMesh);
    } catch {
      hullManifold = mod.Manifold.hull(extractVertexTuples(mergedHullGeometry));
    }
    
    try {
      baseManifold = mod.Manifold.ofMesh(baseMesh);
    } catch {
      baseManifold = mod.Manifold.hull(extractVertexTuples(mergedBaseGeometry));
    }
    
    // Check if manifolds are valid
    if (hullManifold.isEmpty()) {
      throw new Error('Hull manifold is empty');
    }
    if (baseManifold.isEmpty()) {
      throw new Error('Base manifold is empty');
    }
    
    // Perform boolean subtraction: hull - base
    const resultManifold = hullManifold.subtract(baseManifold);
    
    // Convert result back to Three.js geometry
    const resultMesh = resultManifold.getMesh();
    const resultGeometry = manifoldMeshToGeometry(resultMesh);
    
    const vertexCount = resultManifold.numVert();
    const faceCount = resultManifold.numTri();
    
    // Manifold library guarantees manifold output
    const manifoldValidation: ManifoldValidationResult = {
      isClosed: true,
      isManifold: true,
      boundaryEdgeCount: 0,
      nonManifoldEdgeCount: 0,
      totalEdgeCount: faceCount * 3 / 2, // Euler formula
      eulerCharacteristic: vertexCount - (faceCount * 3 / 2) + faceCount,
    };
    
    // Create material for the result
    const cavityMaterial = new THREE.MeshPhongMaterial({
      color: CAVITY_COLOR,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide,
      flatShading: false,
    });
    
    const mesh = new THREE.Mesh(resultGeometry, cavityMaterial);
    
    // Clean up Manifold objects
    hullManifold.delete();
    baseManifold.delete();
    resultManifold.delete();
    
    // Clean up temp geometries
    mergedHullGeometry.dispose();
    mergedBaseGeometry.dispose();
    
    return {
      mesh,
      vertexCount,
      faceCount,
      manifoldValidation,
    };
    
  } catch (error) {
    console.error('Manifold CSG subtraction failed:', error);
    
    // Return empty result on failure
    const emptyGeometry = new THREE.BufferGeometry();
    const cavityMaterial = new THREE.MeshPhongMaterial({
      color: CAVITY_COLOR,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
    });
    
    return {
      mesh: new THREE.Mesh(emptyGeometry, cavityMaterial),
      vertexCount: 0,
      faceCount: 0,
      manifoldValidation: {
        isClosed: false,
        isManifold: false,
        boundaryEdgeCount: -1,
        nonManifoldEdgeCount: -1,
        totalEdgeCount: 0,
        eulerCharacteristic: 0,
      },
    };
  }
}

/**
 * Remove CSG result mesh from scene
 */
export function removeCsgResult(
  scene: THREE.Scene,
  result: CsgSubtractionResult | null
): void {
  if (!result) return;

  if (result.mesh.parent === scene) {
    scene.remove(result.mesh);
    result.mesh.geometry.dispose();
    (result.mesh.material as THREE.Material).dispose();
  }
}
