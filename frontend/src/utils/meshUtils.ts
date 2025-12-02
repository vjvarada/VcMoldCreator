/**
 * Mesh Utilities
 * 
 * Shared utilities for mesh operations including:
 * - Inside/outside testing using BVH-accelerated raycasting
 * - Distance field computation
 * - Volume intersection testing
 * - Debug logging utilities
 * 
 * These utilities are used by both volumetricGrid.ts and partingSurface.ts
 */

import * as THREE from 'three';
import { MeshBVH, acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh';

// Extend Three.js with BVH acceleration
THREE.Mesh.prototype.raycast = acceleratedRaycast;
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

// ============================================================================
// LOGGING SYSTEM
// ============================================================================

/**
 * Log levels for filtering output
 * - NONE: No logging
 * - ERROR: Only errors
 * - WARN: Errors and warnings
 * - INFO: Core timing and results (recommended for production)
 * - DEBUG: Detailed intermediate steps
 * - TRACE: Very verbose, all internal operations
 */
export const LogLevel = {
  NONE: 0,
  ERROR: 1,
  WARN: 2,
  INFO: 3,
  DEBUG: 4,
  TRACE: 5,
} as const;

export type LogLevel = typeof LogLevel[keyof typeof LogLevel];

/** Current log level - set to INFO for production */
let currentLogLevel: LogLevel = LogLevel.INFO;

/** Get current log level */
export function getLogLevel(): LogLevel {
  return currentLogLevel;
}

/** Set log level */
export function setLogLevel(level: LogLevel): void {
  currentLogLevel = level;
}

/** Log an error message */
export function logError(...args: unknown[]): void {
  if (currentLogLevel >= LogLevel.ERROR) {
    console.error('[MOLD ERROR]', ...args);
  }
}

/** Log a warning message */
export function logWarn(...args: unknown[]): void {
  if (currentLogLevel >= LogLevel.WARN) {
    console.warn('[MOLD WARN]', ...args);
  }
}

/** Log an info message (timing, results, key steps) */
export function logInfo(...args: unknown[]): void {
  if (currentLogLevel >= LogLevel.INFO) {
    console.log('[MOLD]', ...args);
  }
}

/** Log a debug message (intermediate details) */
export function logDebug(...args: unknown[]): void {
  if (currentLogLevel >= LogLevel.DEBUG) {
    console.log('[MOLD DEBUG]', ...args);
  }
}

/** Log a trace message (very verbose) */
export function logTrace(...args: unknown[]): void {
  if (currentLogLevel >= LogLevel.TRACE) {
    console.log('[MOLD TRACE]', ...args);
  }
}

/** 
 * Log timing information for a step
 * @param step - Name of the step
 * @param timeMs - Time in milliseconds
 * @param extra - Optional extra info to include
 */
export function logTiming(step: string, timeMs: number, extra?: string): void {
  if (currentLogLevel >= LogLevel.INFO) {
    const msg = extra ? `${step}: ${timeMs.toFixed(1)}ms (${extra})` : `${step}: ${timeMs.toFixed(1)}ms`;
    console.log('[MOLD]', msg);
  }
}

/** 
 * Log step results (counts, statistics)
 */
export function logResult(step: string, data: Record<string, unknown>): void {
  if (currentLogLevel >= LogLevel.INFO) {
    const parts = Object.entries(data).map(([k, v]) => {
      if (typeof v === 'number') {
        return `${k}=${Number.isInteger(v) ? v : v.toFixed(4)}`;
      }
      return `${k}=${v}`;
    });
    console.log('[MOLD]', `${step}:`, parts.join(', '));
  }
}

// Legacy compatibility
/** @deprecated Use setLogLevel(LogLevel.DEBUG) instead */
export let DEBUG_LOGGING = true;

/** @deprecated Use setLogLevel instead */
export function setDebugLogging(enabled: boolean): void {
  currentLogLevel = enabled ? LogLevel.DEBUG : LogLevel.INFO;
  DEBUG_LOGGING = enabled;
}

/** @deprecated Use logDebug instead */
export function debugLog(...args: unknown[]): void {
  logDebug(...args);
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default ray direction for inside/outside tests (+X axis) */
const RAY_DIRECTION = new THREE.Vector3(1, 0, 0);

// ============================================================================
// NEIGHBOR OFFSETS FOR VOXEL ADJACENCY
// ============================================================================

/** 6-connected neighbors (face adjacency) */
export const NEIGHBORS_6: readonly [number, number, number][] = [
  [-1, 0, 0], [1, 0, 0],
  [0, -1, 0], [0, 1, 0],
  [0, 0, -1], [0, 0, 1],
];

/** 18-connected neighbors (face + edge adjacency) */
export const NEIGHBORS_18: readonly [number, number, number][] = [
  ...NEIGHBORS_6,
  [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],
  [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],
  [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1],
];

/** 26-connected neighbors (face + edge + corner adjacency) */
export const NEIGHBORS_26: readonly [number, number, number][] = [
  ...NEIGHBORS_18,
  [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
  [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
];

export type AdjacencyType = 6 | 18 | 26;

/**
 * Get neighbor offsets for a given adjacency type
 */
export function getNeighborOffsets(adjacency: AdjacencyType): readonly [number, number, number][] {
  switch (adjacency) {
    case 6: return NEIGHBORS_6;
    case 18: return NEIGHBORS_18;
    case 26: return NEIGHBORS_26;
  }
}

// ============================================================================
// MESH INSIDE/OUTSIDE TESTER
// ============================================================================

/**
 * Helper class for fast inside/outside tests on a mesh
 * Uses BVH for acceleration and ray casting for inside/outside determination
 */
export class MeshTester {
  private mesh: THREE.Mesh;
  private bvh: MeshBVH;
  private raycaster: THREE.Raycaster;
  private boundingBox: THREE.Box3;
  public readonly name: string;
  
  constructor(geometry: THREE.BufferGeometry, name: string = 'mesh') {
    this.name = name;
    
    // Ensure geometry is indexed for BVH
    let indexedGeometry = geometry;
    if (!geometry.index) {
      const posAttr = geometry.getAttribute('position');
      const indices: number[] = [];
      for (let i = 0; i < posAttr.count; i++) {
        indices.push(i);
      }
      indexedGeometry = geometry.clone();
      indexedGeometry.setIndex(indices);
    }
    
    // Compute normals and bounds
    indexedGeometry.computeVertexNormals();
    indexedGeometry.computeBoundingBox();
    this.boundingBox = indexedGeometry.boundingBox!.clone();
    
    // Create mesh with BVH
    const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(indexedGeometry, material);
    this.mesh.updateMatrixWorld(true);
    
    this.bvh = new MeshBVH(indexedGeometry, { maxLeafTris: 10 });
    indexedGeometry.boundsTree = this.bvh;
    
    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = false;
  }
  
  /**
   * Quick check if point is outside the bounding box
   */
  isOutsideBoundingBox(point: THREE.Vector3): boolean {
    return !this.boundingBox.containsPoint(point);
  }
  
  /**
   * Test if a point is inside the mesh using ray casting
   * Uses the crossing number algorithm: odd intersections = inside
   */
  isInside(point: THREE.Vector3): boolean {
    if (this.isOutsideBoundingBox(point)) {
      return false;
    }
    
    this.raycaster.set(point, RAY_DIRECTION);
    this.raycaster.far = Infinity;
    const intersects = this.raycaster.intersectObject(this.mesh, false);
    
    return intersects.length % 2 === 1;
  }
  
  /**
   * Test if a point is outside the mesh
   */
  isOutside(point: THREE.Vector3): boolean {
    return !this.isInside(point);
  }
  
  /**
   * Get approximate signed distance to mesh surface
   * Positive = outside, Negative = inside
   */
  getSignedDistance(point: THREE.Vector3): number {
    const target = { 
      point: new THREE.Vector3(), 
      distance: Infinity,
      faceIndex: 0
    };
    
    this.bvh.closestPointToPoint(point, target);
    const distance = target.distance;
    const isInside = this.isInside(point);
    
    return isInside ? -distance : distance;
  }
  
  /**
   * Get unsigned distance to mesh surface
   */
  getDistanceToSurface(point: THREE.Vector3): number {
    const target = { 
      point: new THREE.Vector3(), 
      distance: Infinity,
      faceIndex: 0
    };
    this.bvh.closestPointToPoint(point, target);
    return target.distance;
  }
  
  /**
   * Find closest point on mesh surface to given point
   */
  closestPointToPoint(
    point: THREE.Vector3,
    target: { point: THREE.Vector3; distance: number; faceIndex: number }
  ): void {
    this.bvh.closestPointToPoint(point, target);
  }
  
  /**
   * Test if a point is inside or within tolerance of the surface
   */
  isInsideOrOnSurface(point: THREE.Vector3, tolerance: number): boolean {
    if (this.isInside(point)) {
      return true;
    }
    return this.getDistanceToSurface(point) <= tolerance;
  }
  
  /**
   * Test if a point is outside or within tolerance of the surface
   */
  isOutsideOrOnSurface(point: THREE.Vector3, tolerance: number): boolean {
    if (this.isOutside(point)) {
      return true;
    }
    return this.getDistanceToSurface(point) <= tolerance;
  }
  
  /**
   * Compute volume intersection ratio of a box with the mesh
   * Uses uniform sampling within the box
   * 
   * @param center - Box center position
   * @param size - Box size in each dimension
   * @param samplesPerAxis - Number of samples per axis (e.g., 4 = 64 total samples)
   * @returns Ratio of sample points inside the mesh (0 to 1)
   */
  computeVolumeIntersection(
    center: THREE.Vector3,
    size: THREE.Vector3,
    samplesPerAxis: number = 4
  ): number {
    const halfSize = size.clone().multiplyScalar(0.5);
    const samplePoint = new THREE.Vector3();
    
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
          
          if (this.isInside(samplePoint)) {
            insideCount++;
          }
        }
      }
    }
    
    return insideCount / totalSamples;
  }
  
  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.mesh.geometry.boundsTree) {
      this.mesh.geometry.disposeBoundsTree();
    }
    (this.mesh.material as THREE.Material).dispose();
  }
}

// ============================================================================
// GEOMETRY UTILITIES
// ============================================================================

/**
 * Extract geometry data for worker transfer
 */
export function extractGeometryData(geometry: THREE.BufferGeometry): {
  positionArray: Float32Array;
  indexArray: Uint32Array | null;
} {
  const position = geometry.getAttribute('position');
  const positionArray = new Float32Array(position.array);
  
  const index = geometry.getIndex();
  const indexArray = index ? new Uint32Array(index.array) : null;
  
  return { positionArray, indexArray };
}

/**
 * Create a BufferGeometry from raw arrays
 */
export function createGeometryFromArrays(
  positionArray: Float32Array,
  indexArray: Uint32Array | null
): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));
  
  if (indexArray) {
    geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
  }
  
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  
  return geometry;
}
