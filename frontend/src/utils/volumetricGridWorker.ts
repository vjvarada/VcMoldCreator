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
  /** Surface tolerance for including boundary voxels (0 = strict containment) */
  surfaceTolerance?: number;
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
let surfaceTolerance = 0; // 0 = strict containment, > 0 = include surface voxels

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
  partIndexArray: Uint32Array | null,
  tolerance: number = 0
): void {
  const startTime = performance.now();
  surfaceTolerance = tolerance;
  
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

// Temporary vector for intersection calculations (separate from tempVec used for cell centers)
const intersectPoint = new Vector3();

/**
 * Test if a point is inside a mesh using ray casting with BVH
 * Uses majority voting across 3 ray directions for robustness
 */
function isInsideMesh(bvh: MeshBVH, boundingBox: Box3, point: Vector3): boolean {
  // Quick bounding box check
  if (!boundingBox.containsPoint(point)) {
    return false;
  }
  
  let insideVotes = 0;
  
  // Test with 3 ray directions for robustness against edge cases
  for (const direction of RAY_DIRECTIONS) {
    ray.origin.copy(point);
    ray.direction.copy(direction);
    
    // Count intersections along this ray (only in positive direction)
    let intersectionCount = 0;
    
    bvh.shapecast({
      intersectsBounds: (box) => ray.intersectsBox(box),
      intersectsTriangle: (tri) => {
        const intersection = ray.intersectTriangle(tri.a, tri.b, tri.c, false, intersectPoint);
        if (intersection) {
          // Check if intersection is in positive ray direction (t > 0)
          // t = distance along ray = (intersectPoint - origin) · direction
          const t = intersectPoint.clone().sub(point).dot(direction);
          if (t > 0.00001) {
            intersectionCount++;
          }
        }
        return false; // Continue to find all intersections
      }
    });
    
    // Odd number of intersections = inside for this ray
    if (intersectionCount % 2 === 1) {
      insideVotes++;
    }
  }
  
  // Majority voting: 2 or more rays say inside = inside
  return insideVotes >= 2;
}

/**
 * Get the distance from a point to the nearest surface of a mesh
 * Uses BVH closestPointToPoint for efficient queries
 */
function getDistanceToSurface(bvh: MeshBVH, point: Vector3): number {
  const target = {
    point: new Vector3(),
    distance: Infinity,
    faceIndex: 0
  };
  bvh.closestPointToPoint(point, target);
  return target.distance;
}

/**
 * Test if a point is inside the mesh OR within a tolerance of the surface
 */
function isInsideOrOnSurface(bvh: MeshBVH, boundingBox: Box3, point: Vector3, tolerance: number): boolean {
  if (isInsideMesh(bvh, boundingBox, point)) {
    return true;
  }
  // If outside, check if within tolerance of surface
  const dist = getDistanceToSurface(bvh, point);
  return dist <= tolerance;
}

/**
 * Test if a point is outside the mesh OR within a tolerance of the surface
 */
function isOutsideOrOnSurface(bvh: MeshBVH, boundingBox: Box3, point: Vector3, tolerance: number): boolean {
  if (!isInsideMesh(bvh, boundingBox, point)) {
    return true;
  }
  // If inside, check if within tolerance of surface (near boundary)
  const dist = getDistanceToSurface(bvh, point);
  return dist <= tolerance;
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
  const useSurfaceTolerance = surfaceTolerance > 0;
  
  for (let i = 0; i < cellCount; i++) {
    const i3 = i * 3;
    tempVec.set(cellCenters[i3], cellCenters[i3 + 1], cellCenters[i3 + 2]);
    
    let isMoldVolume: boolean;
    
    if (useSurfaceTolerance) {
      // Include voxels that are inside OR on/near the shell surface
      const isInsideOrOnShell = isInsideOrOnSurface(shellBvh, shellBoundingBox, tempVec, surfaceTolerance);
      
      if (isInsideOrOnShell) {
        // For part: we want voxels that are outside the part OR touching its surface
        const isOutsideOrOnPart = isOutsideOrOnSurface(partBvh, partBoundingBox, tempVec, surfaceTolerance);
        isMoldVolume = isOutsideOrOnPart;
      } else {
        isMoldVolume = false;
      }
    } else {
      // Original strict containment logic
      // Test if inside shell
      const isInsideShell = isInsideMesh(shellBvh, shellBoundingBox, tempVec);
      
      if (isInsideShell) {
        // Only test part if inside shell (optimization)
        const isInsidePart = isInsideMesh(partBvh, partBoundingBox, tempVec);
        // Mold volume = inside shell AND outside part
        isMoldVolume = !isInsidePart;
      } else {
        isMoldVolume = false;
      }
    }
    
    if (isMoldVolume) {
      moldVolumeIndices.push(cellIndices[i]);
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
      data.partIndexArray,
      data.surfaceTolerance ?? 0
    );
  } else if (data.type === 'compute') {
    processCells(data.cellCenters, data.cellIndices);
  }
};

export {};
