/**
 * Web Worker for parallel distance field computation
 * 
 * Computes distances from voxel centers to mesh surfaces using BVH.
 */

import { MeshBVH } from 'three-mesh-bvh';
import { Vector3, BufferGeometry, BufferAttribute } from 'three';

// ============================================================================
// TYPES
// ============================================================================

interface InitMessage {
  type: 'init';
  workerId: number;
  partPositionArray: Float32Array;
  partIndexArray: Uint32Array | null;
  shellPositionArray: Float32Array;
  shellIndexArray: Uint32Array | null;
}

interface ComputeMessage {
  type: 'compute';
  voxelCenters: Float32Array; // Flattened [x, y, z, x, y, z, ...]
  startIndex: number; // Starting index in the global voxel array
}

interface ResultMessage {
  type: 'result';
  workerId: number;
  startIndex: number;
  partDistances: Float32Array;
  shellDistances: Float32Array;
}

interface ReadyMessage {
  type: 'ready';
  workerId: number;
}

// ============================================================================
// WORKER STATE
// ============================================================================

let workerId = -1;
let partBvh: MeshBVH | null = null;
let shellBvh: MeshBVH | null = null;

// ============================================================================
// INITIALIZATION
// ============================================================================

function initWorker(
  partPositionArray: Float32Array,
  partIndexArray: Uint32Array | null,
  shellPositionArray: Float32Array,
  shellIndexArray: Uint32Array | null
): void {
  const startTime = performance.now();
  
  // Build part geometry and BVH
  const partGeometry = new BufferGeometry();
  partGeometry.setAttribute('position', new BufferAttribute(partPositionArray, 3));
  
  if (partIndexArray) {
    partGeometry.setIndex(new BufferAttribute(partIndexArray, 1));
  } else {
    const vertexCount = partPositionArray.length / 3;
    const indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
    partGeometry.setIndex(new BufferAttribute(indices, 1));
  }
  
  partBvh = new MeshBVH(partGeometry, { maxLeafTris: 10 });
  
  // Build shell geometry and BVH
  const shellGeometry = new BufferGeometry();
  shellGeometry.setAttribute('position', new BufferAttribute(shellPositionArray, 3));
  
  if (shellIndexArray) {
    shellGeometry.setIndex(new BufferAttribute(shellIndexArray, 1));
  } else {
    const vertexCount = shellPositionArray.length / 3;
    const indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
    shellGeometry.setIndex(new BufferAttribute(indices, 1));
  }
  
  shellBvh = new MeshBVH(shellGeometry, { maxLeafTris: 10 });
  
  const elapsed = performance.now() - startTime;
  console.log(`[DistanceWorker ${workerId}] Ready in ${elapsed.toFixed(0)}ms`);
  
  self.postMessage({
    type: 'ready',
    workerId,
  } as ReadyMessage);
}

// ============================================================================
// DISTANCE COMPUTATION
// ============================================================================

function computeDistances(voxelCenters: Float32Array, startIndex: number): void {
  if (!partBvh || !shellBvh) {
    self.postMessage({
      type: 'result',
      workerId,
      startIndex,
      partDistances: new Float32Array(0),
      shellDistances: new Float32Array(0),
    } as ResultMessage);
    return;
  }
  
  const voxelCount = voxelCenters.length / 3;
  const partDistances = new Float32Array(voxelCount);
  const shellDistances = new Float32Array(voxelCount);
  
  const point = new Vector3();
  const target = {
    point: new Vector3(),
    distance: Infinity,
    faceIndex: 0
  };
  
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    point.set(voxelCenters[i3], voxelCenters[i3 + 1], voxelCenters[i3 + 2]);
    
    // Distance to part
    target.distance = Infinity;
    partBvh.closestPointToPoint(point, target);
    partDistances[i] = target.distance;
    
    // Distance to shell
    target.distance = Infinity;
    shellBvh.closestPointToPoint(point, target);
    shellDistances[i] = target.distance;
  }
  
  const message: ResultMessage = {
    type: 'result',
    workerId,
    startIndex,
    partDistances,
    shellDistances,
  };
  
  (self as unknown as Worker).postMessage(message, [partDistances.buffer, shellDistances.buffer]);
}

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

self.onmessage = (event: MessageEvent<InitMessage | ComputeMessage>) => {
  const { data } = event;
  
  if (data.type === 'init') {
    workerId = data.workerId;
    initWorker(
      data.partPositionArray,
      data.partIndexArray,
      data.shellPositionArray,
      data.shellIndexArray
    );
  } else if (data.type === 'compute') {
    computeDistances(data.voxelCenters, data.startIndex);
  }
};

export {};
