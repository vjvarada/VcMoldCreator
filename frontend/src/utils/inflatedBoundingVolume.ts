/**
 * Inflated Bounding Volume Generator
 * 
 * Creates an inflated convex hull around a mesh:
 * 1. Compute the convex hull of the input mesh
 * 2. Generate smooth vertex normals using area-weighted averaging
 * 3. Inflate hull vertices outward by a small offset distance
 * 4. Create a new mesh from inflated vertices + original hull faces
 */

import * as THREE from 'three';
import { ConvexGeometry } from 'three-stdlib';

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default inflation offset distance (in mm) */
export const DEFAULT_INFLATION_OFFSET = 0.5;

/** Color for the inflated hull visualization */
export const HULL_COLOR = 0x9966ff; // Purple

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
 * Extract unique vertices from a mesh geometry (handles indexed and non-indexed)
 */
function extractUniqueVertices(geometry: THREE.BufferGeometry): THREE.Vector3[] {
  return extractUniqueVerticesWithMap(geometry).vertices;
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
 * Compute area-weighted smooth vertex normals for unique hull vertices
 * 
 * For each unique vertex, accumulates face normals (weighted by face area)
 * from all faces sharing that vertex, then normalizes.
 */
function computeAreaWeightedNormalsForUniqueVertices(
  uniqueVertices: THREE.Vector3[],
  faceIndices: number[]
): THREE.Vector3[] {
  const vertexCount = uniqueVertices.length;
  
  // Initialize normal accumulators
  const normalAccum: THREE.Vector3[] = [];
  for (let i = 0; i < vertexCount; i++) {
    normalAccum.push(new THREE.Vector3(0, 0, 0));
  }
  
  // Temporary vectors for calculations
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceNormal = new THREE.Vector3();

  // Process triangles
  const triangleCount = faceIndices.length / 3;
  
  for (let i = 0; i < triangleCount; i++) {
    const iA = faceIndices[i * 3];
    const iB = faceIndices[i * 3 + 1];
    const iC = faceIndices[i * 3 + 2];

    const vA = uniqueVertices[iA];
    const vB = uniqueVertices[iB];
    const vC = uniqueVertices[iC];

    // Compute edge vectors
    ab.subVectors(vB, vA);
    ac.subVectors(vC, vA);

    // Cross product gives normal * 2 * area
    faceNormal.crossVectors(ab, ac);
    const area = faceNormal.length() / 2;
    
    if (area > 1e-10) {
      faceNormal.normalize();
      
      // Weight by area and accumulate to each vertex
      normalAccum[iA].addScaledVector(faceNormal, area);
      normalAccum[iB].addScaledVector(faceNormal, area);
      normalAccum[iC].addScaledVector(faceNormal, area);
    }
  }

  // Normalize accumulated normals
  for (let i = 0; i < vertexCount; i++) {
    normalAccum[i].normalize();
  }

  return normalAccum;
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
  
  console.log(`Manifold validation: V=${V}, E=${E}, F=${F}, χ=${eulerCharacteristic}`);
  console.log(`  Boundary edges: ${boundaryEdgeCount}, Non-manifold edges: ${nonManifoldEdgeCount}`);
  console.log(`  Is closed: ${isClosed}, Is manifold: ${isManifold}`);
  
  return {
    isClosed,
    isManifold,
    boundaryEdgeCount,
    nonManifoldEdgeCount,
    totalEdgeCount,
    eulerCharacteristic,
  };
}

// ============================================================================
// MAIN FUNCTIONS
// ============================================================================

/**
 * Generate an inflated bounding volume (convex hull) for a mesh
 * 
 * Steps:
 * 1. Compute convex hull of the input mesh
 * 2. Extract unique vertices and face indices from the hull
 * 3. Generate smooth vertex normals using area-weighted normal averaging
 * 4. Inflate hull vertices outward by the offset distance
 * 5. Create a new mesh from the inflated vertices + same hull faces
 * 6. Validate if the mesh is closed/manifold
 * 
 * @param mesh - The input mesh to create hull around
 * @param offset - Distance to inflate outward (default: 1.0)
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
  console.log('Extracting unique vertices from input mesh...');
  const inputVertices = extractUniqueVertices(worldGeometry);
  console.log(`Found ${inputVertices.length} unique input vertices`);

  // Step 2: Generate convex hull
  console.log('Computing convex hull...');
  const hullGeometry = new ConvexGeometry(inputVertices);
  
  // Step 3: Extract unique hull vertices and build face indices
  // ConvexGeometry creates non-indexed geometry with duplicated vertices per face
  // We need to extract unique vertices and build proper face connectivity
  const { vertices: hullVertices, positionToIndex } = extractUniqueVerticesWithMap(hullGeometry);
  const hullFaceIndices = buildFaceIndices(hullGeometry, positionToIndex);
  
  console.log(`Hull has ${hullVertices.length} unique vertices and ${hullFaceIndices.length / 3} faces`);

  // Step 4: Compute area-weighted smooth vertex normals for unique hull vertices
  console.log('Computing area-weighted vertex normals...');
  const smoothNormals = computeAreaWeightedNormalsForUniqueVertices(hullVertices, hullFaceIndices);

  // Step 5: Inflate hull vertices outward along their smooth normals
  console.log(`Inflating hull vertices by ${offset} units...`);
  const inflatedVertices = inflateVertices(hullVertices, smoothNormals, offset);

  // Step 6: Recompute convex hull from inflated vertices to ensure valid convex geometry
  // This is crucial - simply reusing the old faces can create concave regions
  console.log('Recomputing convex hull from inflated vertices...');
  const inflatedHullGeometry = new ConvexGeometry(inflatedVertices);
  
  // Extract the final geometry info
  const { vertices: finalVertices, positionToIndex: finalPosToIndex } = extractUniqueVerticesWithMap(inflatedHullGeometry);
  const finalFaceIndices = buildFaceIndices(inflatedHullGeometry, finalPosToIndex);
  const faceCount = finalFaceIndices.length / 3;
  
  // Create indexed geometry from the new convex hull
  const inflatedGeometry = createMeshFromVerticesAndFaces(finalVertices, finalFaceIndices);

  // Step 7: Validate manifold properties of the inflated geometry
  console.log('Validating manifold properties...');
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

  // Create original hull geometry from unique vertices for display
  const originalHullDisplayGeometry = createMeshFromVerticesAndFaces(hullVertices, hullFaceIndices);

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
  
  console.log('Inflated bounding volume generated successfully');
  console.log(`Final inflated hull: ${finalVertices.length} vertices, ${faceCount} faces`);

  return {
    mesh: inflatedMesh,
    originalHull: originalHullMesh,
    smoothNormals: smoothNormalsArray,
    vertexCount: finalVertices.length,
    faceCount,
    manifoldValidation,
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
}
