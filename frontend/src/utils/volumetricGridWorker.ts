/**
 * Web Worker for parallel volumetric grid computation
 * 
 * Builds BVHs from serialized geometry data and computes 
 * inside/outside tests for batches of grid cells.
 */

import { MeshBVH } from 'three-mesh-bvh';
import { Ray, Vector3, BufferGeometry, BufferAttribute, Box3 } from 'three';

// ============================================================================
// TYPES
// ============================================================================

interface InitMessage {
  type: 'init';
  workerId: number;
  shellPositionArray: Float32Array;
  shellIndexArray: Uint32Array | null;
  partPositionArray: Float32Array;
  partIndexArray: Uint32Array | null;
}

interface ComputeMessage {
  type: 'compute';
  cellCenters: Float32Array; // Flattened [x, y, z, x, y, z, ...] for each cell
  cellIndices: Uint32Array;  // Original grid indices [i*resY*resZ + j*resZ + k, ...]
}

interface ResultMessage {
  type: 'result';
  workerId: number;
  moldVolumeCellIndices: Uint32Array; // Indices of cells that are mold volume
}

interface ReadyMessage {
  type: 'ready';
  workerId: number;
}

// ============================================================================
// WORKER STATE
// ============================================================================

let workerId = -1;
let shellBvh: MeshBVH | null = null;
let partBvh: MeshBVH | null = null;
let shellGeometry: BufferGeometry | null = null;
let partGeometry: BufferGeometry | null = null;
let shellBoundingBox: Box3 | null = null;
let partBoundingBox: Box3 | null = null;

// Ray casting setup - use 3 directions for faster computation
const RAY_DIRECTIONS = [
  new Vector3(1, 0, 0),
  new Vector3(0, 1, 0),
  new Vector3(0, 0, 1),
];

const ray = new Ray();
const tempVec = new Vector3();

// ============================================================================
// INITIALIZATION
// ============================================================================

function initWorker(
  shellPositionArray: Float32Array,
  shellIndexArray: Uint32Array | null,
  partPositionArray: Float32Array,
  partIndexArray: Uint32Array | null
): void {
  const startTime = performance.now();
  
  // Build shell geometry and BVH
  shellGeometry = new BufferGeometry();
  shellGeometry.setAttribute('position', new BufferAttribute(shellPositionArray, 3));
  
  if (shellIndexArray) {
    shellGeometry.setIndex(new BufferAttribute(shellIndexArray, 1));
  } else {
    const vertexCount = shellPositionArray.length / 3;
    const indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
    shellGeometry.setIndex(new BufferAttribute(indices, 1));
  }
  
  shellGeometry.computeBoundingBox();
  shellBoundingBox = shellGeometry.boundingBox!.clone();
  shellBvh = new MeshBVH(shellGeometry, { maxLeafTris: 10 });
  
  // Build part geometry and BVH
  partGeometry = new BufferGeometry();
  partGeometry.setAttribute('position', new BufferAttribute(partPositionArray, 3));
  
  if (partIndexArray) {
    partGeometry.setIndex(new BufferAttribute(partIndexArray, 1));
  } else {
    const vertexCount = partPositionArray.length / 3;
    const indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
    partGeometry.setIndex(new BufferAttribute(indices, 1));
  }
  
  partGeometry.computeBoundingBox();
  partBoundingBox = partGeometry.boundingBox!.clone();
  partBvh = new MeshBVH(partGeometry, { maxLeafTris: 10 });
  
  const elapsed = performance.now() - startTime;
  console.log(`[GridWorker ${workerId}] Ready in ${elapsed.toFixed(0)}ms`);
  
  self.postMessage({
    type: 'ready',
    workerId,
  } as ReadyMessage);
}

// ============================================================================
// INSIDE/OUTSIDE TESTING
// ============================================================================

/**
 * Test if a point is inside a mesh using ray casting with BVH
 * Uses majority voting across 3 ray directions
 */
function isInsideMesh(bvh: MeshBVH, boundingBox: Box3, point: Vector3): boolean {
  // Quick bounding box check
  if (!boundingBox.containsPoint(point)) {
    return false;
  }
  
  let insideVotes = 0;
  
  for (const direction of RAY_DIRECTIONS) {
    ray.origin.copy(point);
    ray.direction.copy(direction);
    
    // Get all intersections using shapecast
    let intersectionCount = 0;
    bvh.shapecast({
      intersectsBounds: () => true,
      intersectsTriangle: (tri) => {
        const intersection = ray.intersectTriangle(tri.a, tri.b, tri.c, false, tempVec);
        if (intersection) {
          intersectionCount++;
        }
        return false; // Continue checking
      }
    });
    
    // Odd number of intersections = inside
    if (intersectionCount % 2 === 1) {
      insideVotes++;
    }
  }
  
  // Majority voting (2 or more out of 3)
  return insideVotes >= 2;
}

// ============================================================================
// GRID CELL PROCESSING
// ============================================================================

function processCells(cellCenters: Float32Array, cellIndices: Uint32Array): void {
  if (!shellBvh || !partBvh || !shellBoundingBox || !partBoundingBox) {
    self.postMessage({
      type: 'result',
      workerId,
      moldVolumeCellIndices: new Uint32Array(0),
    } as ResultMessage);
    return;
  }
  
  const cellCount = cellIndices.length;
  const moldVolumeIndices: number[] = [];
  
  for (let i = 0; i < cellCount; i++) {
    const i3 = i * 3;
    tempVec.set(cellCenters[i3], cellCenters[i3 + 1], cellCenters[i3 + 2]);
    
    // Test if inside shell
    const isInsideShell = isInsideMesh(shellBvh, shellBoundingBox, tempVec);
    
    if (isInsideShell) {
      // Only test part if inside shell (optimization)
      const isInsidePart = isInsideMesh(partBvh, partBoundingBox, tempVec);
      
      // Mold volume = inside shell AND outside part
      if (!isInsidePart) {
        moldVolumeIndices.push(cellIndices[i]);
      }
    }
  }
  
  self.postMessage({
    type: 'result',
    workerId,
    moldVolumeCellIndices: new Uint32Array(moldVolumeIndices),
  } as ResultMessage);
}

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

self.onmessage = (event: MessageEvent<InitMessage | ComputeMessage>) => {
  const { data } = event;
  
  if (data.type === 'init') {
    workerId = data.workerId;
    initWorker(
      data.shellPositionArray,
      data.shellIndexArray,
      data.partPositionArray,
      data.partIndexArray
    );
  } else if (data.type === 'compute') {
    processCells(data.cellCenters, data.cellIndices);
  }
};

export {};
