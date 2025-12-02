/**
 * Web Worker for parallel volume intersection computation
 * Computes the ratio of sample points inside a part mesh for each voxel
 */

import * as THREE from 'three';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';

// Extend THREE.Mesh to use accelerated raycast
THREE.Mesh.prototype.raycast = acceleratedRaycast;

interface WorkerInitMessage {
  type: 'init';
  workerId: number;
  partPositionArray: Float32Array;
  partIndexArray: Uint32Array | null;
}

interface WorkerComputeMessage {
  type: 'compute';
  voxelCenters: Float32Array;  // [x,y,z, x,y,z, ...]
  voxelSizeX: number;
  voxelSizeY: number;
  voxelSizeZ: number;
  samplesPerAxis: number;
  threshold: number;
  startIndex: number;
}

interface WorkerReadyMessage {
  type: 'ready';
  workerId: number;
}

interface WorkerResultMessage {
  type: 'result';
  workerId: number;
  startIndex: number;
  seedMask: Uint8Array;  // 1 if voxel is a seed (volume >= threshold), 0 otherwise
}

type WorkerMessage = WorkerInitMessage | WorkerComputeMessage;

let workerId = -1;
let partMesh: THREE.Mesh | null = null;
let raycaster: THREE.Raycaster | null = null;
const rayDirection = new THREE.Vector3(1, 0, 0);

/**
 * Build a THREE.Mesh with BVH from geometry data
 */
function buildMeshWithBVH(
  positionArray: Float32Array,
  indexArray: Uint32Array | null
): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));
  
  if (indexArray) {
    geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
  }
  
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  
  // Build BVH for fast raycasting
  const bvh = new MeshBVH(geometry);
  geometry.boundsTree = bvh;
  
  const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  return new THREE.Mesh(geometry, material);
}

/**
 * Test if a point is inside the part mesh using ray casting
 */
function isInsidePart(point: THREE.Vector3): boolean {
  if (!partMesh || !raycaster) return false;
  
  raycaster.ray.origin.copy(point);
  raycaster.ray.direction.copy(rayDirection);
  raycaster.firstHitOnly = false;
  
  const intersects = raycaster.intersectObject(partMesh);
  
  // Count only forward intersections
  let forwardCount = 0;
  for (const hit of intersects) {
    if (hit.distance > 0) {
      forwardCount++;
    }
  }
  
  return forwardCount % 2 === 1;
}

/**
 * Compute volume intersection ratio for a single voxel
 */
function computeVolumeIntersection(
  cx: number, cy: number, cz: number,
  sizeX: number, sizeY: number, sizeZ: number,
  samplesPerAxis: number
): number {
  const halfX = sizeX / 2;
  const halfY = sizeY / 2;
  const halfZ = sizeZ / 2;
  const samplePoint = new THREE.Vector3();
  
  let insideCount = 0;
  const totalSamples = samplesPerAxis * samplesPerAxis * samplesPerAxis;
  
  for (let i = 0; i < samplesPerAxis; i++) {
    const tx = (i + 0.5) / samplesPerAxis;
    const x = cx - halfX + tx * sizeX;
    
    for (let j = 0; j < samplesPerAxis; j++) {
      const ty = (j + 0.5) / samplesPerAxis;
      const y = cy - halfY + ty * sizeY;
      
      for (let k = 0; k < samplesPerAxis; k++) {
        const tz = (k + 0.5) / samplesPerAxis;
        const z = cz - halfZ + tz * sizeZ;
        
        samplePoint.set(x, y, z);
        
        if (isInsidePart(samplePoint)) {
          insideCount++;
        }
      }
    }
  }
  
  return insideCount / totalSamples;
}

/**
 * Handle incoming messages
 */
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  
  if (message.type === 'init') {
    workerId = message.workerId;
    
    // Build part mesh with BVH
    partMesh = buildMeshWithBVH(message.partPositionArray, message.partIndexArray);
    raycaster = new THREE.Raycaster();
    
    // Signal ready
    const readyMsg: WorkerReadyMessage = { type: 'ready', workerId };
    self.postMessage(readyMsg);
    
  } else if (message.type === 'compute') {
    const { voxelCenters, voxelSizeX, voxelSizeY, voxelSizeZ, samplesPerAxis, threshold, startIndex } = message;
    const voxelCount = voxelCenters.length / 3;
    const seedMask = new Uint8Array(voxelCount);
    
    for (let i = 0; i < voxelCount; i++) {
      const i3 = i * 3;
      const cx = voxelCenters[i3];
      const cy = voxelCenters[i3 + 1];
      const cz = voxelCenters[i3 + 2];
      
      const volumeRatio = computeVolumeIntersection(
        cx, cy, cz,
        voxelSizeX, voxelSizeY, voxelSizeZ,
        samplesPerAxis
      );
      
      if (volumeRatio >= threshold) {
        seedMask[i] = 1;
      }
    }
    
    const resultMsg: WorkerResultMessage = {
      type: 'result',
      workerId,
      startIndex,
      seedMask,
    };
    
    (self as unknown as Worker).postMessage(resultMsg, [seedMask.buffer]);
  }
};
