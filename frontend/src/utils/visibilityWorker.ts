/**
 * Web Worker for parallel visibility computation
 * 
 * This worker builds its own BVH from serialized geometry data
 * and computes visibility scores for batches of directions.
 */

import { MeshBVH } from 'three-mesh-bvh';
import { Ray, Vector3, BufferGeometry, BufferAttribute } from 'three';

// Message types
interface InitMessage {
  type: 'init';
  positionArray: Float32Array;
  indexArray: Uint32Array | null;
  workerId: number;
}

interface ComputeMessage {
  type: 'compute';
  directions: Float32Array; // Flat array: [dx1, dy1, dz1, dx2, dy2, dz2, ...]
  directionIndices: number[]; // Original indices for these directions
}

interface ResultMessage {
  type: 'result';
  workerId: number;
  results: {
    directionIndex: number;
    visibleTriangles: number[];
    visibleArea: number;
  }[];
}

interface ReadyMessage {
  type: 'ready';
  workerId: number;
  triangleCount: number;
  totalArea: number;
}

// Worker state
let workerId = -1;
let bvh: MeshBVH | null = null;
let geometry: BufferGeometry | null = null;

// Optimized triangle data
let normals: Float32Array | null = null;
let centroids: Float32Array | null = null;
let areas: Float32Array | null = null;
let triangleCount = 0;
let totalSurfaceArea = 0;
let rayOffset = 0;

// Reusable objects
const ray = new Ray();
const rayDirection = new Vector3();

/**
 * Initialize worker with geometry data
 */
function initWorker(positionArray: Float32Array, indexArray: Uint32Array | null): void {
  console.log(`[Worker ${workerId}] Initializing with ${positionArray.length / 3} vertices`);
  const initStart = performance.now();
  
  // Reconstruct geometry
  geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positionArray, 3));
  
  if (indexArray) {
    geometry.setIndex(new BufferAttribute(indexArray, 1));
  } else {
    // Create index for non-indexed geometry (each 3 vertices = 1 triangle)
    const vertexCount = positionArray.length / 3;
    const indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) {
      indices[i] = i;
    }
    geometry.setIndex(new BufferAttribute(indices, 1));
  }
  
  // Build BVH
  bvh = new MeshBVH(geometry);
  (geometry as any).boundsTree = bvh;
  
  // Extract triangle data into typed arrays
  extractTriangleData();
  
  // Compute ray offset from bounding box
  geometry.computeBoundingBox();
  const box = geometry.boundingBox!;
  const sizeX = box.max.x - box.min.x;
  const sizeY = box.max.y - box.min.y;
  const sizeZ = box.max.z - box.min.z;
  rayOffset = Math.max(sizeX, sizeY, sizeZ) * 2;
  
  const initTime = performance.now() - initStart;
  console.log(`[Worker ${workerId}] BVH built for ${triangleCount} triangles in ${initTime.toFixed(0)}ms`);
  
  // Send ready message
  const msg: ReadyMessage = {
    type: 'ready',
    workerId,
    triangleCount,
    totalArea: totalSurfaceArea
  };
  self.postMessage(msg);
}

/**
 * Extract triangle data from geometry into typed arrays
 */
function extractTriangleData(): void {
  if (!geometry) return;
  
  const position = geometry.getAttribute('position') as BufferAttribute;
  const index = geometry.getIndex();
  
  triangleCount = index ? index.count / 3 : position.count / 3;
  
  normals = new Float32Array(triangleCount * 3);
  centroids = new Float32Array(triangleCount * 3);
  areas = new Float32Array(triangleCount);
  
  const vA = new Vector3();
  const vB = new Vector3();
  const vC = new Vector3();
  const edge1 = new Vector3();
  const edge2 = new Vector3();
  const normal = new Vector3();
  
  totalSurfaceArea = 0;
  
  for (let i = 0; i < triangleCount; i++) {
    let a: number, b: number, c: number;
    
    if (index) {
      a = index.getX(i * 3);
      b = index.getX(i * 3 + 1);
      c = index.getX(i * 3 + 2);
    } else {
      a = i * 3;
      b = i * 3 + 1;
      c = i * 3 + 2;
    }
    
    vA.fromBufferAttribute(position, a);
    vB.fromBufferAttribute(position, b);
    vC.fromBufferAttribute(position, c);
    
    // Compute normal
    edge1.subVectors(vB, vA);
    edge2.subVectors(vC, vA);
    normal.crossVectors(edge1, edge2);
    const area = normal.length() * 0.5;
    normal.normalize();
    
    const i3 = i * 3;
    normals[i3] = normal.x;
    normals[i3 + 1] = normal.y;
    normals[i3 + 2] = normal.z;
    
    // Compute centroid
    centroids[i3] = (vA.x + vB.x + vC.x) / 3;
    centroids[i3 + 1] = (vA.y + vB.y + vC.y) / 3;
    centroids[i3 + 2] = (vA.z + vB.z + vC.z) / 3;
    
    areas[i] = area;
    totalSurfaceArea += area;
  }
}

/**
 * Compute visible triangles for a single direction
 */
function computeVisibleTriangles(dx: number, dy: number, dz: number): { visible: number[], area: number } {
  if (!bvh || !normals || !centroids || !areas) {
    return { visible: [], area: 0 };
  }
  
  const visible: number[] = [];
  let visibleArea = 0;
  
  // Set up ray direction (negated)
  rayDirection.set(-dx, -dy, -dz);
  ray.direction.copy(rayDirection);
  
  for (let i = 0; i < triangleCount; i++) {
    const i3 = i * 3;
    
    // Inlined dot product: normal · direction
    const dot = normals[i3] * dx + normals[i3 + 1] * dy + normals[i3 + 2] * dz;
    
    // Skip back-facing triangles
    if (dot <= 0) continue;
    
    // Set ray origin: centroid + direction * offset
    ray.origin.set(
      centroids[i3] + dx * rayOffset,
      centroids[i3 + 1] + dy * rayOffset,
      centroids[i3 + 2] + dz * rayOffset
    );
    
    // Find first intersection using BVH
    const hit = bvh.raycastFirst(ray);
    
    if (hit && hit.faceIndex === i) {
      visible.push(i);
      visibleArea += areas[i];
    }
  }
  
  return { visible, area: visibleArea };
}

/**
 * Process a batch of directions
 */
function processDirections(directions: Float32Array, directionIndices: number[]): void {
  const results: ResultMessage['results'] = [];
  
  const directionCount = directionIndices.length;
  
  for (let i = 0; i < directionCount; i++) {
    const i3 = i * 3;
    const dx = directions[i3];
    const dy = directions[i3 + 1];
    const dz = directions[i3 + 2];
    
    const { visible, area } = computeVisibleTriangles(dx, dy, dz);
    
    results.push({
      directionIndex: directionIndices[i],
      visibleTriangles: visible,
      visibleArea: area
    });
  }
  
  const msg: ResultMessage = {
    type: 'result',
    workerId,
    results
  };
  
  // Transfer the results back
  self.postMessage(msg);
}

// Message handler
self.onmessage = (event: MessageEvent<InitMessage | ComputeMessage>) => {
  const data = event.data;
  
  if (data.type === 'init') {
    workerId = data.workerId;
    initWorker(data.positionArray, data.indexArray);
  } else if (data.type === 'compute') {
    processDirections(data.directions, data.directionIndices);
  }
};

export {}; // Make this a module
