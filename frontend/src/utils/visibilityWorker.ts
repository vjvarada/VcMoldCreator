/**
 * Web Worker for parallel visibility computation
 * 
 * Builds its own BVH from serialized geometry data and computes 
 * visibility scores for batches of directions.
 */

import { MeshBVH } from 'three-mesh-bvh';
import { Ray, Vector3, BufferGeometry, BufferAttribute } from 'three';

// ============================================================================
// TYPES
// ============================================================================

interface InitMessage {
  type: 'init';
  positionArray: Float32Array;
  indexArray: Uint32Array | null;
  workerId: number;
}

interface ComputeMessage {
  type: 'compute';
  directions: Float32Array;
  directionIndices: number[];
}

interface ResultMessage {
  type: 'result';
  workerId: number;
  results: Array<{
    directionIndex: number;
    visibleTriangles: number[];
    visibleArea: number;
  }>;
}

interface ReadyMessage {
  type: 'ready';
  workerId: number;
  triangleCount: number;
  totalArea: number;
}

// ============================================================================
// WORKER STATE
// ============================================================================

let workerId = -1;
let bvh: MeshBVH | null = null;

// Typed arrays for triangle data
let normals: Float32Array | null = null;
let centroids: Float32Array | null = null;
let areas: Float32Array | null = null;
let triangleCount = 0;
let totalSurfaceArea = 0;
let rayOffset = 0;

// Reusable objects to avoid allocations
const ray = new Ray();
const rayDirection = new Vector3();

// ============================================================================
// INITIALIZATION
// ============================================================================

function initWorker(positionArray: Float32Array, indexArray: Uint32Array | null): void {
  console.log(`[Worker ${workerId}] Initializing...`);
  const startTime = performance.now();
  
  // Reconstruct geometry
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positionArray, 3));
  
  if (indexArray) {
    geometry.setIndex(new BufferAttribute(indexArray, 1));
  } else {
    const vertexCount = positionArray.length / 3;
    const indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
    geometry.setIndex(new BufferAttribute(indices, 1));
  }
  
  // Build BVH
  bvh = new MeshBVH(geometry);
  
  // Extract triangle data
  extractTriangleData(geometry);
  
  // Compute ray offset from bounding box
  geometry.computeBoundingBox();
  const box = geometry.boundingBox!;
  rayOffset = Math.max(
    box.max.x - box.min.x,
    box.max.y - box.min.y,
    box.max.z - box.min.z
  ) * 2;
  
  const elapsed = performance.now() - startTime;
  console.log(`[Worker ${workerId}] Ready: ${triangleCount} triangles in ${elapsed.toFixed(0)}ms`);
  
  self.postMessage({
    type: 'ready',
    workerId,
    triangleCount,
    totalArea: totalSurfaceArea
  } as ReadyMessage);
}

function extractTriangleData(geometry: BufferGeometry): void {
  const position = geometry.getAttribute('position') as BufferAttribute;
  const index = geometry.getIndex();
  
  triangleCount = index ? index.count / 3 : position.count / 3;
  
  normals = new Float32Array(triangleCount * 3);
  centroids = new Float32Array(triangleCount * 3);
  areas = new Float32Array(triangleCount);
  totalSurfaceArea = 0;
  
  const vA = new Vector3(), vB = new Vector3(), vC = new Vector3();
  const edge1 = new Vector3(), edge2 = new Vector3(), normal = new Vector3();
  
  for (let i = 0; i < triangleCount; i++) {
    const [a, b, c] = index
      ? [index.getX(i * 3), index.getX(i * 3 + 1), index.getX(i * 3 + 2)]
      : [i * 3, i * 3 + 1, i * 3 + 2];
    
    vA.fromBufferAttribute(position, a);
    vB.fromBufferAttribute(position, b);
    vC.fromBufferAttribute(position, c);
    
    edge1.subVectors(vB, vA);
    edge2.subVectors(vC, vA);
    normal.crossVectors(edge1, edge2);
    
    const area = normal.length() * 0.5;
    normal.normalize();
    
    const i3 = i * 3;
    normals[i3] = normal.x;
    normals[i3 + 1] = normal.y;
    normals[i3 + 2] = normal.z;
    
    centroids[i3] = (vA.x + vB.x + vC.x) / 3;
    centroids[i3 + 1] = (vA.y + vB.y + vC.y) / 3;
    centroids[i3 + 2] = (vA.z + vB.z + vC.z) / 3;
    
    areas[i] = area;
    totalSurfaceArea += area;
  }
}

// ============================================================================
// VISIBILITY COMPUTATION
// ============================================================================

function computeVisibleTriangles(dx: number, dy: number, dz: number): { visible: number[]; area: number } {
  if (!bvh || !normals || !centroids || !areas) {
    return { visible: [], area: 0 };
  }
  
  const visible: number[] = [];
  let visibleArea = 0;
  
  rayDirection.set(-dx, -dy, -dz);
  ray.direction.copy(rayDirection);
  
  for (let i = 0; i < triangleCount; i++) {
    const i3 = i * 3;
    
    // Dot product: normal · direction
    const dot = normals[i3] * dx + normals[i3 + 1] * dy + normals[i3 + 2] * dz;
    if (dot <= 0) continue; // Back-facing
    
    // Ray origin: centroid + direction * offset
    ray.origin.set(
      centroids[i3] + dx * rayOffset,
      centroids[i3 + 1] + dy * rayOffset,
      centroids[i3 + 2] + dz * rayOffset
    );
    
    const hit = bvh.raycastFirst(ray);
    if (hit && hit.faceIndex === i) {
      visible.push(i);
      visibleArea += areas[i];
    }
  }
  
  return { visible, area: visibleArea };
}

function processDirections(directions: Float32Array, directionIndices: number[]): void {
  const results: ResultMessage['results'] = [];
  
  for (let i = 0; i < directionIndices.length; i++) {
    const i3 = i * 3;
    const { visible, area } = computeVisibleTriangles(
      directions[i3],
      directions[i3 + 1],
      directions[i3 + 2]
    );
    
    results.push({
      directionIndex: directionIndices[i],
      visibleTriangles: visible,
      visibleArea: area
    });
  }
  
  self.postMessage({ type: 'result', workerId, results } as ResultMessage);
}

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

self.onmessage = (event: MessageEvent<InitMessage | ComputeMessage>) => {
  const { data } = event;
  
  if (data.type === 'init') {
    workerId = data.workerId;
    initWorker(data.positionArray, data.indexArray);
  } else if (data.type === 'compute') {
    processDirections(data.directions, data.directionIndices);
  }
};

export {};
