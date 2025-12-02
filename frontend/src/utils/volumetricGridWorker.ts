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
  /** Maximum overlap ratio with part mesh for boundary voxels (default: 0.10 = 10%) */
  maxPartOverlapRatio?: number;
  /** Minimum overlap ratio with part mesh for surface voxels (default: 0.01 = 1%) */
  minPartOverlapRatio?: number;
  /** Distance threshold as fraction of voxel size for surface voxels (default: 0.5 = 50%) */
  surfaceDistanceThreshold?: number;
  /** Cell size for volume intersection calculations [x, y, z] */
  cellSize?: [number, number, number];
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
let maxPartOverlapRatio = 0.10; // Maximum overlap ratio with part for boundary voxels (10%)
let minPartOverlapRatio = 0.01; // Minimum overlap ratio for surface voxels (1%)
let surfaceDistanceThreshold = 0.5; // Distance threshold as fraction of voxel size (50%)
let cellSizeVec = new Vector3(1, 1, 1); // Cell size for volume intersection

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
  tolerance: number = 0,
  overlapRatio: number = 0.10,
  minOverlapRatio: number = 0.01,
  distanceThreshold: number = 0.5,
  cellSize: [number, number, number] = [1, 1, 1]
): void {
  const startTime = performance.now();
  surfaceTolerance = tolerance;
  maxPartOverlapRatio = overlapRatio;
  minPartOverlapRatio = minOverlapRatio;
  surfaceDistanceThreshold = distanceThreshold;
  cellSizeVec.set(cellSize[0], cellSize[1], cellSize[2]);
  
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
  console.log(`[GridWorker ${workerId}] Ready in ${elapsed.toFixed(0)}ms (overlapRange=${minPartOverlapRatio*100}%-${maxPartOverlapRatio*100}%, distThresh=${surfaceDistanceThreshold*100}%)`);
  
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
 * Compute volume intersection ratio of a box with the mesh
 * Uses uniform sampling within the box
 * 
 * @param bvh - BVH of the mesh to test against
 * @param boundingBox - Bounding box of the mesh
 * @param center - Box center position
 * @param size - Box size vector
 * @param samplesPerAxis - Number of samples per axis (e.g., 3 = 27 total samples)
 * @returns Ratio of sample points inside the mesh (0 to 1)
 */
function computeVolumeIntersection(
  bvh: MeshBVH,
  boundingBox: Box3,
  center: Vector3,
  size: Vector3,
  samplesPerAxis: number = 3
): number {
  const halfSize = size.clone().multiplyScalar(0.5);
  const samplePoint = new Vector3();
  
  let insideCount = 0;
  const totalSamples = samplesPerAxis * samplesPerAxis * samplesPerAxis;
  
  for (let i = 0; i < samplesPerAxis; i++) {
    const tx = (i + 0.5) / samplesPerAxis;
    const x = center.x - halfSize.x + tx * size.x;
    
    for (let j = 0; j < samplesPerAxis; j++) {
      const ty = (j + 0.5) / samplesPerAxis;
      const y = center.y - halfSize.y + ty * size.y;
      
      for (let k = 0; k < samplesPerAxis; k++) {
        const tz = (k + 0.5) / samplesPerAxis;
        const z = center.z - halfSize.z + tz * size.z;
        
        samplePoint.set(x, y, z);
        
        if (isInsideMesh(bvh, boundingBox, samplePoint)) {
          insideCount++;
        }
      }
    }
  }
  
  return insideCount / totalSamples;
}

/**
 * Check if a voxel qualifies as a surface voxel
 * Surface voxels are identified by:
 * 1. Having 1-10% overlap with the part mesh, OR
 * 2. Being within 50% of a voxel distance from the part boundary
 */
function isSurfaceVoxel(
  partBvh: MeshBVH,
  partBoundingBox: Box3,
  center: Vector3,
  cellSize: Vector3
): boolean {
  // Calculate average cell size for distance threshold
  const avgCellSize = (cellSize.x + cellSize.y + cellSize.z) / 3;
  const distanceThreshold = avgCellSize * surfaceDistanceThreshold;
  
  // Check distance to part surface
  const distToPart = getDistanceToSurface(partBvh, center);
  if (distToPart <= distanceThreshold) {
    return true;
  }
  
  // Check volume overlap ratio (1-10%)
  const overlapRatio = computeVolumeIntersection(partBvh, partBoundingBox, center, cellSize, 3);
  if (overlapRatio >= minPartOverlapRatio && overlapRatio <= maxPartOverlapRatio) {
    return true;
  }
  
  return false;
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
        // Check if voxel center is outside the part
        const isOutsidePart = !isInsideMesh(partBvh, partBoundingBox, tempVec);
        
        if (isOutsidePart) {
          // Check if this is a surface voxel (1-10% overlap OR within 50% voxel distance of part)
          // Surface voxels are near the part boundary
          if (isSurfaceVoxel(partBvh, partBoundingBox, tempVec, cellSizeVec)) {
            isMoldVolume = true;
          } else {
            // Regular mold volume voxel - include it
            isMoldVolume = true;
          }
        } else {
          // Voxel center is inside part - check if it qualifies as a surface voxel
          // Surface voxels have 1-10% overlap OR are within 50% voxel distance of part surface
          isMoldVolume = isSurfaceVoxel(partBvh, partBoundingBox, tempVec, cellSizeVec);
        }
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
      data.surfaceTolerance ?? 0,
      data.maxPartOverlapRatio ?? 0.10,
      data.minPartOverlapRatio ?? 0.01,
      data.surfaceDistanceThreshold ?? 0.5,
      data.cellSize ?? [1, 1, 1]
    );
  } else if (data.type === 'compute') {
    processCells(data.cellCenters, data.cellIndices);
  }
};

export {};
