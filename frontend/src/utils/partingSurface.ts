/**
 * Parting Surface / Escape Labeling
 * 
 * This module computes the parting surface for mold design using multi-source
 * Dijkstra from inner seed boundaries (near part mesh) outward to the mold boundary.
 * 
 * Algorithm (STEP 3.5):
 * - Identify inner surface voxels (near part mesh M) as seeds
 * - Identify outer surface voxels (near shell ∂H) and label them as H₁ or H₂
 * - Run Dijkstra from each seed to find lowest cost path to boundary:
 *     edgeCost(i,j) = L * wt_j
 *   where L = Euclidean distance, wt_j = weighting factor of destination voxel
 * 
 * Weighting factor computation:
 * - For boundary-adjacent voxels (outer surface + one level deep):
 *     wt = 1/[(biasedDist² + 0.25)]
 *     biasedDist = δ_i + (R - δ_w)  (distance to part + bias from shell distance)
 * - For interior voxels (more than one level from boundary):
 *     wt = 1/[(δ_i² + 0.25)]  (unbiased, only distance to part)
 * 
 * This ensures the shell-based bias only affects paths near the boundary,
 * not throughout the entire volume.
 * 
 * - Each seed inherits the label of the boundary it reaches via lowest cost path
 * - The parting surface is where label transitions occur (interface voxels)
 */

import * as THREE from 'three';
import { MeshBVH, acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh';
import type { VolumetricGridResult } from './volumetricGrid';
import type { MoldHalfClassificationResult } from './moldHalfClassification';
import { runParallelDijkstra, isWebWorkersAvailable } from './parallelDijkstra';
import { MeshTester, getNeighborOffsets, type AdjacencyType } from './meshUtils';

// Extend Three.js with BVH acceleration
THREE.Mesh.prototype.raycast = acceleratedRaycast;
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

// ============================================================================
// TYPES
// ============================================================================

// Re-export AdjacencyType for backwards compatibility
export type { AdjacencyType } from './meshUtils';

export interface EscapeLabelingResult {
  /** Labels for each mold volume voxel: -1 = unlabeled, 1 = escapes via H₁, 2 = escapes via H₂ */
  labels: Int8Array;
  /** Escape cost for each voxel (thickness-weighted path cost to boundary) */
  escapeCost: Float32Array;
  /** Number of voxels labeled as escaping via H₁ (side 1) */
  d1Count: number;
  /** Number of voxels labeled as escaping via H₂ (side 2) */
  d2Count: number;
  /** Number of interface voxels (where labels differ between neighbors) */
  interfaceCount: number;
  /** Number of unlabeled voxels (should be 0 for connected volumes) */
  unlabeledCount: number;
  /** Computation time in ms */
  computeTimeMs: number;
  /** DEBUG: Boundary labels (0 = not boundary, 1 = H₁ boundary, 2 = H₂ boundary) */
  boundaryLabels?: Int8Array;
  /** DEBUG: Seed voxels (true = seed near part) */
  seedMask?: Uint8Array;
}

export interface EscapeLabelingOptions {
  /** Adjacency type: 6-connected, 18-connected, or 26-connected (default: 6) */
  adjacency?: AdjacencyType;
  /** Seed radius: how far from boundary faces to seed voxels (in voxel units, default: 1.5) */
  seedRadius?: number;
  /** 
   * Minimum volume intersection ratio with part mesh for a voxel to be considered a seed.
   * Value between 0 and 1. Default: 0.75 (75% of voxel volume must intersect part).
   * Set to 0 to use the legacy surface distance approach.
   */
  seedVolumeThreshold?: number;
  /** Number of sample points per axis for volume intersection test (default: 4 = 64 samples) */
  seedVolumeSamples?: number;
  /** 
   * Enable parallel processing using Web Workers.
   * Set to true to automatically use available CPU cores, or a number to specify worker count.
   * Default: false (single-threaded)
   */
  parallel?: boolean | number;
  /** 
   * Callback for progress updates during parallel processing.
   * Only called when parallel=true.
   */
  onProgress?: (progress: { completed: number; total: number }) => void;
}

// ============================================================================
// MIN-HEAP PRIORITY QUEUE
// ============================================================================

/**
 * Simple min-heap priority queue for Dijkstra
 */
class MinHeap {
  private heap: { index: number; cost: number }[] = [];
  private positions: Map<number, number> = new Map(); // voxel index -> heap position

  get size(): number {
    return this.heap.length;
  }

  push(index: number, cost: number): void {
    const node = { index, cost };
    this.heap.push(node);
    this.positions.set(index, this.heap.length - 1);
    this.bubbleUp(this.heap.length - 1);
  }

  pop(): { index: number; cost: number } | undefined {
    if (this.heap.length === 0) return undefined;
    
    const min = this.heap[0];
    const last = this.heap.pop()!;
    this.positions.delete(min.index);
    
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.positions.set(last.index, 0);
      this.bubbleDown(0);
    }
    
    return min;
  }

  decreaseKey(index: number, newCost: number): void {
    const pos = this.positions.get(index);
    if (pos === undefined) {
      // Not in heap, add it
      this.push(index, newCost);
      return;
    }
    
    if (newCost < this.heap[pos].cost) {
      this.heap[pos].cost = newCost;
      this.bubbleUp(pos);
    }
  }

  has(index: number): boolean {
    return this.positions.has(index);
  }

  private bubbleUp(pos: number): void {
    while (pos > 0) {
      const parent = Math.floor((pos - 1) / 2);
      if (this.heap[pos].cost >= this.heap[parent].cost) break;
      
      this.swap(pos, parent);
      pos = parent;
    }
  }

  private bubbleDown(pos: number): void {
    const n = this.heap.length;
    while (true) {
      const left = 2 * pos + 1;
      const right = 2 * pos + 2;
      let smallest = pos;
      
      if (left < n && this.heap[left].cost < this.heap[smallest].cost) {
        smallest = left;
      }
      if (right < n && this.heap[right].cost < this.heap[smallest].cost) {
        smallest = right;
      }
      
      if (smallest === pos) break;
      
      this.swap(pos, smallest);
      pos = smallest;
    }
  }

  private swap(i: number, j: number): void {
    const temp = this.heap[i];
    this.heap[i] = this.heap[j];
    this.heap[j] = temp;
    this.positions.set(this.heap[i].index, i);
    this.positions.set(this.heap[j].index, j);
  }
}

// ============================================================================
// SPATIAL INDEX FOR VOXELS
// ============================================================================

/**
 * Build a spatial hash map from grid indices to voxel array index
 */
function buildVoxelSpatialIndex(
  gridResult: VolumetricGridResult
): Map<string, number> {
  const indexMap = new Map<string, number>();
  const voxelCount = gridResult.moldVolumeCellCount;
  
  // Use flat arrays if available (more memory efficient)
  const useFlat = gridResult.voxelIndices !== null;
  
  for (let i = 0; i < voxelCount; i++) {
    let gi: number, gj: number, gk: number;
    
    if (useFlat) {
      const i3 = i * 3;
      gi = gridResult.voxelIndices![i3];
      gj = gridResult.voxelIndices![i3 + 1];
      gk = gridResult.voxelIndices![i3 + 2];
    } else {
      const cell = gridResult.moldVolumeCells[i];
      gi = Math.round(cell.index.x);
      gj = Math.round(cell.index.y);
      gk = Math.round(cell.index.z);
    }
    
    const key = `${gi},${gj},${gk}`;
    indexMap.set(key, i);
  }
  
  return indexMap;
}

/**
 * Get voxel index from grid coordinates, or -1 if not found
 */
function getVoxelIndex(
  i: number, j: number, k: number,
  spatialIndex: Map<string, number>
): number {
  const key = `${i},${j},${k}`;
  return spatialIndex.get(key) ?? -1;
}

// ============================================================================
// BOUNDARY VOXEL DETECTION
// ============================================================================

// Default number of workers for parallel processing
const DEFAULT_NUM_WORKERS = typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency || 4) : 4;

/**
 * Extract geometry data for worker transfer
 */
function extractGeometryDataForWorker(geometry: THREE.BufferGeometry): {
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
 * Compute volume intersection for all voxels in parallel using Web Workers
 */
async function computeVolumeIntersectionParallel(
  gridResult: VolumetricGridResult,
  partGeometry: THREE.BufferGeometry,
  seedVolumeThreshold: number,
  seedVolumeSamples: number,
  numWorkers: number = DEFAULT_NUM_WORKERS
): Promise<{ innerSurfaceMask: Uint8Array; innerSurfaceCount: number }> {
  const voxelCount = gridResult.moldVolumeCellCount;
  const innerSurfaceMask = new Uint8Array(voxelCount);
  
  // Get voxel centers - use flat array if available
  let voxelCentersArray: Float32Array;
  if (gridResult.voxelCenters !== null) {
    voxelCentersArray = gridResult.voxelCenters;
  } else {
    // Fall back to building from moldVolumeCells
    voxelCentersArray = new Float32Array(voxelCount * 3);
    for (let i = 0; i < voxelCount; i++) {
      const center = gridResult.moldVolumeCells[i].center;
      voxelCentersArray[i * 3] = center.x;
      voxelCentersArray[i * 3 + 1] = center.y;
      voxelCentersArray[i * 3 + 2] = center.z;
    }
  }
  
  const voxelSizeX = gridResult.cellSize.x;
  const voxelSizeY = gridResult.cellSize.y;
  const voxelSizeZ = gridResult.cellSize.z;
  
  // Extract part geometry data
  const partData = extractGeometryDataForWorker(partGeometry);
  
  // Create workers
  const workers: Worker[] = [];
  const workerReadyPromises: Promise<void>[] = [];
  
  console.log(`    Initializing ${numWorkers} volume intersection workers...`);
  
  for (let i = 0; i < numWorkers; i++) {
    const worker = new Worker(
      new URL('./volumeIntersectionWorker.ts', import.meta.url),
      { type: 'module' }
    );
    workers.push(worker);
    
    workerReadyPromises.push(new Promise<void>((resolve) => {
      const handler = (event: MessageEvent) => {
        if (event.data.type === 'ready') {
          worker.removeEventListener('message', handler);
          resolve();
        }
      };
      worker.addEventListener('message', handler);
    }));
    
    // Send init message with geometry
    const initMsg = {
      type: 'init',
      workerId: i,
      partPositionArray: partData.positionArray.slice(),
      partIndexArray: partData.indexArray?.slice() ?? null,
    };
    
    const transfers: Transferable[] = [initMsg.partPositionArray.buffer];
    if (initMsg.partIndexArray) transfers.push(initMsg.partIndexArray.buffer);
    
    worker.postMessage(initMsg, transfers);
  }
  
  await Promise.all(workerReadyPromises);
  console.log(`    Workers initialized`);
  
  // Distribute voxels across workers
  const voxelsPerWorker = Math.ceil(voxelCount / numWorkers);
  const workerPromises: Promise<{ startIndex: number; seedMask: Uint8Array }>[] = [];
  
  for (let w = 0; w < numWorkers; w++) {
    const startIdx = w * voxelsPerWorker;
    const endIdx = Math.min(startIdx + voxelsPerWorker, voxelCount);
    if (endIdx <= startIdx) continue;
    
    const voxelCenters = voxelCentersArray.slice(startIdx * 3, endIdx * 3);
    
    workerPromises.push(new Promise((resolve) => {
      const handler = (event: MessageEvent) => {
        if (event.data.type === 'result' && event.data.workerId === w) {
          workers[w].removeEventListener('message', handler);
          resolve({
            startIndex: event.data.startIndex,
            seedMask: event.data.seedMask,
          });
        }
      };
      workers[w].addEventListener('message', handler);
    }));
    
    workers[w].postMessage({
      type: 'compute',
      voxelCenters,
      voxelSizeX,
      voxelSizeY,
      voxelSizeZ,
      samplesPerAxis: seedVolumeSamples,
      threshold: seedVolumeThreshold,
      startIndex: startIdx,
    }, [voxelCenters.buffer]);
  }
  
  // Wait for all workers
  const results = await Promise.all(workerPromises);
  
  // Terminate workers
  workers.forEach(w => w.terminate());
  
  // Merge results
  let innerSurfaceCount = 0;
  for (const result of results) {
    const { startIndex, seedMask } = result;
    for (let i = 0; i < seedMask.length; i++) {
      if (seedMask[i]) {
        innerSurfaceMask[startIndex + i] = 1;
        innerSurfaceCount++;
      }
    }
  }
  
  return { innerSurfaceMask, innerSurfaceCount };
}

/**
 * Identify which voxels are on the OUTER SURFACE of the silicone volume
 * (near the mold shell ∂H, NOT near the part mesh M).
 * 
 * Also identifies SEED voxels using one of two approaches:
 * 1. Volume intersection (if partMeshTester provided): voxels with ≥ seedVolumeThreshold 
 *    of their volume inside the part mesh
 * 2. Surface distance (legacy): surface voxels with small distance to part
 * 
 * A voxel is a surface voxel if it has at least one 6-connected neighbor 
 * that is NOT a silicone voxel.
 * 
 * - Outer surface = surface voxels with large distance to part
 * - Inner surface/Seeds = determined by volume intersection or surface distance
 */
function identifyBoundaryAdjacentVoxels(
  gridResult: VolumetricGridResult,
  shellGeometry: THREE.BufferGeometry,
  classification: MoldHalfClassificationResult,
  _boundaryRadius: number,
  partMeshTester?: MeshTester,
  seedVolumeThreshold: number = 0.75,
  seedVolumeSamples: number = 4
): { 
  boundaryLabel: Int8Array; 
  innerSurfaceMask: Uint8Array;
  /** Mask for voxels that are boundary or one level deep from boundary (for biased weighting) */
  boundaryAdjacentMask: Uint8Array;
  h1Adjacent: number; 
  h2Adjacent: number;
  innerSurfaceCount: number;
} {
  const voxelCount = gridResult.moldVolumeCellCount;
  const cells = gridResult.moldVolumeCells;
  const voxelDist = gridResult.voxelDist;
  
  if (!voxelDist) {
    console.error('ERROR: voxelDist is required for boundary detection');
    return { 
      boundaryLabel: new Int8Array(voxelCount), 
      innerSurfaceMask: new Uint8Array(voxelCount),
      boundaryAdjacentMask: new Uint8Array(voxelCount),
      h1Adjacent: 0, 
      h2Adjacent: 0,
      innerSurfaceCount: 0
    };
  }
  
  // Build spatial index for neighbor lookups
  const spatialIndex = buildVoxelSpatialIndex(gridResult);
  
  // 6-connected neighbor offsets (face-adjacent only for surface detection)
  const faceNeighborOffsets: [number, number, number][] = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
  ];
  
  // Step 1: Find ALL surface voxels (those with at least one missing 6-neighbor)
  const isSurfaceVoxel = new Uint8Array(voxelCount);
  let totalSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const cell = cells[i];
    const gi = Math.round(cell.index.x);
    const gj = Math.round(cell.index.y);
    const gk = Math.round(cell.index.z);
    
    // Check if any 6-connected neighbor is missing (not a silicone voxel)
    for (const [di, dj, dk] of faceNeighborOffsets) {
      const neighborIdx = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
      if (neighborIdx < 0) {
        // Missing neighbor = this is a surface voxel
        isSurfaceVoxel[i] = 1;
        totalSurfaceCount++;
        break;
      }
    }
  }
  
  console.log(`  Total surface voxels (both inner & outer): ${totalSurfaceCount}`);
  
  // Step 2: Compute threshold to distinguish inner vs outer surface
  // Inner surface = close to part (small voxelDist)
  // Outer surface = far from part (large voxelDist)
  // 
  // Find the distance distribution among surface voxels
  const surfaceDistances: number[] = [];
  const surfaceIndices: number[] = [];
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i]) {
      surfaceDistances.push(voxelDist[i]);
      surfaceIndices.push(i);
    }
  }
  
  // Sort by distance
  const sortedPairs = surfaceIndices.map((idx, i) => ({ idx, dist: surfaceDistances[i] }));
  sortedPairs.sort((a, b) => a.dist - b.dist);
  
  const minDist = sortedPairs[0]?.dist || 0;
  const maxDist = sortedPairs[sortedPairs.length - 1]?.dist || 0;
  
  console.log(`  Surface distance range: [${minDist.toFixed(4)}, ${maxDist.toFixed(4)}]`);
  
  // Find the largest gap in the sorted distances
  // This gap should separate inner surface (low distances) from outer surface (high distances)
  let maxGap = 0;
  let gapIndex = Math.floor(sortedPairs.length / 2); // Default to median
  
  for (let i = 1; i < sortedPairs.length; i++) {
    const gap = sortedPairs[i].dist - sortedPairs[i - 1].dist;
    if (gap > maxGap) {
      maxGap = gap;
      gapIndex = i;
    }
  }
  
  // The threshold is at the gap - everything before is inner, everything after is outer
  const distanceThreshold = sortedPairs[gapIndex]?.dist || (minDist + maxDist) / 2;
  
  console.log(`  Largest gap at index ${gapIndex}/${sortedPairs.length}, gap size: ${maxGap.toFixed(4)}`);
  console.log(`  Distance threshold: ${distanceThreshold.toFixed(4)}`);
  console.log(`  Inner candidates: ${gapIndex}, Outer candidates: ${sortedPairs.length - gapIndex}`);
  
  // Step 3: Mark OUTER surface voxels (using distance threshold)
  // and identify SEED voxels using either volume intersection or surface distance
  const isOuterSurface = new Uint8Array(voxelCount);
  const innerSurfaceMask = new Uint8Array(voxelCount);
  let outerSurfaceCount = 0;
  let innerSurfaceCount = 0;
  
  // First pass: identify outer surface voxels
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i] && voxelDist[i] >= distanceThreshold) {
      isOuterSurface[i] = 1;
      outerSurfaceCount++;
    }
  }
  
  // Second pass: identify seed voxels
  if (partMeshTester && seedVolumeThreshold > 0) {
    // Volume intersection approach: find voxels with ≥ threshold volume inside part
    console.log(`  Using volume intersection approach for seeds (threshold: ${(seedVolumeThreshold * 100).toFixed(0)}%, samples: ${seedVolumeSamples}³)`);
    const volumeStartTime = performance.now();
    
    for (let i = 0; i < voxelCount; i++) {
      const cell = cells[i];
      const volumeRatio = partMeshTester.computeVolumeIntersection(
        cell.center,
        cell.size,
        seedVolumeSamples
      );
      
      if (volumeRatio >= seedVolumeThreshold) {
        innerSurfaceMask[i] = 1;
        innerSurfaceCount++;
      }
    }
    
    console.log(`  Volume intersection computed in ${(performance.now() - volumeStartTime).toFixed(1)}ms`);
    console.log(`  Seed voxels (≥${(seedVolumeThreshold * 100).toFixed(0)}% volume intersection): ${innerSurfaceCount}`);
  } else {
    // Legacy surface distance approach: surface voxels close to part
    console.log(`  Using legacy surface distance approach for seeds`);
    for (let i = 0; i < voxelCount; i++) {
      if (isSurfaceVoxel[i] && voxelDist[i] < distanceThreshold) {
        innerSurfaceMask[i] = 1;
        innerSurfaceCount++;
      }
    }
    console.log(`  Inner surface (near part): ${innerSurfaceCount}`);
  }
  
  console.log(`  Outer surface (near shell): ${outerSurfaceCount}`);
  
  // Step 4: For each OUTER surface voxel, determine if it's closer to H₁, H₂, or boundary zone
  // Use BVH for fast closest-point queries
  // Voxels closest to boundary zone triangles will not receive a label (stay 0)
  
  console.log('  Building BVH for H₁, H₂, and boundary zone triangles...');
  const bvhStartTime = performance.now();
  
  const position = shellGeometry.getAttribute('position');
  const shellIndex = shellGeometry.getIndex();
  
  // Create separate geometries for H1, H2, and boundary zone with their own BVHs
  const createSubGeometry = (triangleSet: Set<number>): THREE.BufferGeometry => {
    const triCount = triangleSet.size;
    if (triCount === 0) {
      // Return empty geometry
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      return geom;
    }
    const positions = new Float32Array(triCount * 9); // 3 vertices × 3 components
    
    let writeIdx = 0;
    for (const triIdx of triangleSet) {
      let a: number, b: number, c: number;
      if (shellIndex) {
        a = shellIndex.getX(triIdx * 3);
        b = shellIndex.getX(triIdx * 3 + 1);
        c = shellIndex.getX(triIdx * 3 + 2);
      } else {
        a = triIdx * 3;
        b = triIdx * 3 + 1;
        c = triIdx * 3 + 2;
      }
      
      positions[writeIdx++] = position.getX(a);
      positions[writeIdx++] = position.getY(a);
      positions[writeIdx++] = position.getZ(a);
      positions[writeIdx++] = position.getX(b);
      positions[writeIdx++] = position.getY(b);
      positions[writeIdx++] = position.getZ(b);
      positions[writeIdx++] = position.getX(c);
      positions[writeIdx++] = position.getY(c);
      positions[writeIdx++] = position.getZ(c);
    }
    
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geom;
  };
  
  // Create sub-geometries for H1, H2, and boundary zone
  const h1Geometry = createSubGeometry(classification.h1Triangles);
  const h2Geometry = createSubGeometry(classification.h2Triangles);
  const boundaryZoneGeometry = createSubGeometry(classification.boundaryZoneTriangles);
  
  // Build BVH for each (only if they have triangles)
  const h1BVH = classification.h1Triangles.size > 0 ? new MeshBVH(h1Geometry) : null;
  const h2BVH = classification.h2Triangles.size > 0 ? new MeshBVH(h2Geometry) : null;
  const boundaryZoneBVH = classification.boundaryZoneTriangles.size > 0 ? new MeshBVH(boundaryZoneGeometry) : null;
  
  console.log(`  BVH build time: ${(performance.now() - bvhStartTime).toFixed(1)}ms`);
  console.log(`  H₁ triangles: ${classification.h1Triangles.size}, H₂ triangles: ${classification.h2Triangles.size}, Boundary zone: ${classification.boundaryZoneTriangles.size}`);
  
  // Label each OUTER surface voxel based on closest region using BVH
  // Voxels closest to boundary zone triangles stay unlabeled (0)
  const boundaryLabel = new Int8Array(voxelCount);
  let h1Adjacent = 0;
  let h2Adjacent = 0;
  let boundaryZoneAdjacent = 0;
  
  const labelStartTime = performance.now();
  const hitInfoH1 = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoH2 = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoBZ = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  
  for (let i = 0; i < voxelCount; i++) {
    if (!isOuterSurface[i]) continue;
    
    const center = cells[i].center;
    
    // Use BVH closestPointToPoint for fast queries
    // Reset hit info
    hitInfoH1.distance = Infinity;
    hitInfoH2.distance = Infinity;
    hitInfoBZ.distance = Infinity;
    
    const distH1 = h1BVH ? (h1BVH.closestPointToPoint(center, hitInfoH1), hitInfoH1.distance) : Infinity;
    const distH2 = h2BVH ? (h2BVH.closestPointToPoint(center, hitInfoH2), hitInfoH2.distance) : Infinity;
    const distBZ = boundaryZoneBVH ? (boundaryZoneBVH.closestPointToPoint(center, hitInfoBZ), hitInfoBZ.distance) : Infinity;
    
    // Assign based on which region is closest
    // If closest to boundary zone, leave unlabeled (0)
    if (distBZ <= distH1 && distBZ <= distH2) {
      // Closest to boundary zone - no label
      boundaryLabel[i] = 0;
      boundaryZoneAdjacent++;
    } else if (distH1 <= distH2) {
      boundaryLabel[i] = 1;
      h1Adjacent++;
    } else {
      boundaryLabel[i] = 2;
      h2Adjacent++;
    }
  }
  
  console.log(`  Labeling time: ${(performance.now() - labelStartTime).toFixed(1)}ms`);
  console.log(`  Outer boundary labeled: H₁=${h1Adjacent}, H₂=${h2Adjacent}, BoundaryZone(unlabeled)=${boundaryZoneAdjacent}`);
  
  // Clean up
  h1Geometry.dispose();
  h2Geometry.dispose();
  boundaryZoneGeometry.dispose();
  
  // Step 5: Build boundaryAdjacentMask - outer surface + one level deep
  // This mask identifies which voxels should use the biased weighting factor
  const boundaryAdjacentMask = new Uint8Array(voxelCount);
  let boundaryAdjacentCount = 0;
  
  // First, mark all outer surface voxels
  for (let i = 0; i < voxelCount; i++) {
    if (isOuterSurface[i]) {
      boundaryAdjacentMask[i] = 1;
      boundaryAdjacentCount++;
    }
  }
  
  // Mark level 1: immediate neighbors of outer surface voxels
  for (let i = 0; i < voxelCount; i++) {
    if (isOuterSurface[i]) {
      const cell = cells[i];
      const gi = Math.round(cell.index.x);
      const gj = Math.round(cell.index.y);
      const gk = Math.round(cell.index.z);
      
      for (const [di, dj, dk] of faceNeighborOffsets) {
        const neighborIdx = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
        if (neighborIdx >= 0 && !boundaryAdjacentMask[neighborIdx]) {
          boundaryAdjacentMask[neighborIdx] = 1;
          boundaryAdjacentCount++;
        }
      }
    }
  }
  
  console.log(`  Boundary voxels (outer + 1 level): ${boundaryAdjacentCount}`);
  
  return { boundaryLabel, innerSurfaceMask, boundaryAdjacentMask, h1Adjacent, h2Adjacent, innerSurfaceCount };
}

/**
 * Async version of identifyBoundaryAdjacentVoxels that uses parallel workers
 * for volume intersection computation (much faster for large voxel counts)
 */
async function identifyBoundaryAdjacentVoxelsAsync(
  gridResult: VolumetricGridResult,
  shellGeometry: THREE.BufferGeometry,
  classification: MoldHalfClassificationResult,
  _boundaryRadius: number,
  partGeometry?: THREE.BufferGeometry,
  seedVolumeThreshold: number = 0.75,
  seedVolumeSamples: number = 4,
  numWorkers: number = DEFAULT_NUM_WORKERS
): Promise<{ 
  boundaryLabel: Int8Array; 
  innerSurfaceMask: Uint8Array;
  boundaryAdjacentMask: Uint8Array;
  h1Adjacent: number; 
  h2Adjacent: number;
  innerSurfaceCount: number;
}> {
  const voxelCount = gridResult.moldVolumeCellCount;
  const cells = gridResult.moldVolumeCells;
  const voxelDist = gridResult.voxelDist;
  
  // Use flat arrays if available
  const useFlat = gridResult.voxelIndices !== null;
  
  if (!voxelDist) {
    console.error('ERROR: voxelDist is required for boundary detection');
    return { 
      boundaryLabel: new Int8Array(voxelCount), 
      innerSurfaceMask: new Uint8Array(voxelCount),
      boundaryAdjacentMask: new Uint8Array(voxelCount),
      h1Adjacent: 0, 
      h2Adjacent: 0,
      innerSurfaceCount: 0
    };
  }
  
  // Build spatial index for neighbor lookups
  const spatialIndex = buildVoxelSpatialIndex(gridResult);
  
  // 6-connected neighbor offsets (face-adjacent only for surface detection)
  const faceNeighborOffsets: [number, number, number][] = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
  ];
  
  // Step 1: Find ALL surface voxels
  const isSurfaceVoxel = new Uint8Array(voxelCount);
  let totalSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    let gi: number, gj: number, gk: number;
    
    if (useFlat) {
      const i3 = i * 3;
      gi = gridResult.voxelIndices![i3];
      gj = gridResult.voxelIndices![i3 + 1];
      gk = gridResult.voxelIndices![i3 + 2];
    } else {
      const cell = cells[i];
      gi = Math.round(cell.index.x);
      gj = Math.round(cell.index.y);
      gk = Math.round(cell.index.z);
    }
    
    for (const [di, dj, dk] of faceNeighborOffsets) {
      const neighborIdx = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
      if (neighborIdx < 0) {
        isSurfaceVoxel[i] = 1;
        totalSurfaceCount++;
        break;
      }
    }
  }
  
  console.log(`  Total surface voxels (both inner & outer): ${totalSurfaceCount}`);
  
  // Step 2: Compute threshold to distinguish inner vs outer surface
  const surfaceDistances: number[] = [];
  const surfaceIndices: number[] = [];
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i]) {
      surfaceDistances.push(voxelDist[i]);
      surfaceIndices.push(i);
    }
  }
  
  const sortedPairs = surfaceIndices.map((idx, i) => ({ idx, dist: surfaceDistances[i] }));
  sortedPairs.sort((a, b) => a.dist - b.dist);
  
  const minDist = sortedPairs[0]?.dist || 0;
  const maxDist = sortedPairs[sortedPairs.length - 1]?.dist || 0;
  
  console.log(`  Surface distance range: [${minDist.toFixed(4)}, ${maxDist.toFixed(4)}]`);
  
  let maxGap = 0;
  let gapIndex = Math.floor(sortedPairs.length / 2);
  
  for (let i = 1; i < sortedPairs.length; i++) {
    const gap = sortedPairs[i].dist - sortedPairs[i - 1].dist;
    if (gap > maxGap) {
      maxGap = gap;
      gapIndex = i;
    }
  }
  
  const distanceThreshold = sortedPairs[gapIndex]?.dist || (minDist + maxDist) / 2;
  
  console.log(`  Largest gap at index ${gapIndex}/${sortedPairs.length}, gap size: ${maxGap.toFixed(4)}`);
  console.log(`  Distance threshold: ${distanceThreshold.toFixed(4)}`);
  console.log(`  Inner candidates: ${gapIndex}, Outer candidates: ${sortedPairs.length - gapIndex}`);
  
  // Step 3: Mark outer surface voxels and identify seeds
  const isOuterSurface = new Uint8Array(voxelCount);
  let innerSurfaceMask = new Uint8Array(voxelCount);
  let outerSurfaceCount = 0;
  let innerSurfaceCount = 0;
  
  // First pass: identify outer surface voxels
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i] && voxelDist[i] >= distanceThreshold) {
      isOuterSurface[i] = 1;
      outerSurfaceCount++;
    }
  }
  
  // Second pass: identify seed voxels using PARALLEL workers
  if (partGeometry && seedVolumeThreshold > 0) {
    console.log(`  Using PARALLEL volume intersection approach for seeds (threshold: ${(seedVolumeThreshold * 100).toFixed(0)}%, samples: ${seedVolumeSamples}³)`);
    const volumeStartTime = performance.now();
    
    const result = await computeVolumeIntersectionParallel(
      gridResult,
      partGeometry,
      seedVolumeThreshold,
      seedVolumeSamples,
      numWorkers
    );
    
    innerSurfaceMask = new Uint8Array(result.innerSurfaceMask);
    innerSurfaceCount = result.innerSurfaceCount;
    
    console.log(`  Volume intersection computed in ${(performance.now() - volumeStartTime).toFixed(1)}ms`);
    console.log(`  Seed voxels (≥${(seedVolumeThreshold * 100).toFixed(0)}% volume intersection): ${innerSurfaceCount}`);
  } else {
    // Legacy surface distance approach
    console.log(`  Using legacy surface distance approach for seeds`);
    for (let i = 0; i < voxelCount; i++) {
      if (isSurfaceVoxel[i] && voxelDist[i] < distanceThreshold) {
        innerSurfaceMask[i] = 1;
        innerSurfaceCount++;
      }
    }
    console.log(`  Inner surface (near part): ${innerSurfaceCount}`);
  }
  
  console.log(`  Outer surface (near shell): ${outerSurfaceCount}`);
  
  // Step 4: For each OUTER surface voxel, determine if it's closer to H₁, H₂, or boundary zone
  console.log('  Building BVH for H₁, H₂, and boundary zone triangles...');
  const bvhStartTime = performance.now();
  
  const position = shellGeometry.getAttribute('position');
  const shellIndex = shellGeometry.getIndex();
  
  const createSubGeometry = (triangleSet: Set<number>): THREE.BufferGeometry => {
    const triCount = triangleSet.size;
    if (triCount === 0) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      return geom;
    }
    const positions = new Float32Array(triCount * 9);
    
    let writeIdx = 0;
    for (const triIdx of triangleSet) {
      let a: number, b: number, c: number;
      if (shellIndex) {
        a = shellIndex.getX(triIdx * 3);
        b = shellIndex.getX(triIdx * 3 + 1);
        c = shellIndex.getX(triIdx * 3 + 2);
      } else {
        a = triIdx * 3;
        b = triIdx * 3 + 1;
        c = triIdx * 3 + 2;
      }
      
      positions[writeIdx++] = position.getX(a);
      positions[writeIdx++] = position.getY(a);
      positions[writeIdx++] = position.getZ(a);
      positions[writeIdx++] = position.getX(b);
      positions[writeIdx++] = position.getY(b);
      positions[writeIdx++] = position.getZ(b);
      positions[writeIdx++] = position.getX(c);
      positions[writeIdx++] = position.getY(c);
      positions[writeIdx++] = position.getZ(c);
    }
    
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geom;
  };
  
  const h1Geometry = createSubGeometry(classification.h1Triangles);
  const h2Geometry = createSubGeometry(classification.h2Triangles);
  const boundaryZoneGeometry = createSubGeometry(classification.boundaryZoneTriangles);
  
  const h1BVH = classification.h1Triangles.size > 0 ? new MeshBVH(h1Geometry) : null;
  const h2BVH = classification.h2Triangles.size > 0 ? new MeshBVH(h2Geometry) : null;
  const boundaryZoneBVH = classification.boundaryZoneTriangles.size > 0 ? new MeshBVH(boundaryZoneGeometry) : null;
  
  console.log(`  BVH build time: ${(performance.now() - bvhStartTime).toFixed(1)}ms`);
  console.log(`  H₁ triangles: ${classification.h1Triangles.size}, H₂ triangles: ${classification.h2Triangles.size}, Boundary zone: ${classification.boundaryZoneTriangles.size}`);
  
  // Label each OUTER surface voxel based on closest region using BVH
  const boundaryLabel = new Int8Array(voxelCount);
  let h1Adjacent = 0;
  let h2Adjacent = 0;
  let boundaryZoneAdjacent = 0;
  
  const labelStartTime = performance.now();
  const hitInfoH1 = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoH2 = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoBZ = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const tempCenter = new THREE.Vector3();
  
  for (let i = 0; i < voxelCount; i++) {
    if (!isOuterSurface[i]) continue;
    
    // Get center from flat array or legacy
    if (useFlat && gridResult.voxelCenters) {
      const i3 = i * 3;
      tempCenter.set(
        gridResult.voxelCenters[i3],
        gridResult.voxelCenters[i3 + 1],
        gridResult.voxelCenters[i3 + 2]
      );
    } else {
      tempCenter.copy(cells[i].center);
    }
    
    hitInfoH1.distance = Infinity;
    hitInfoH2.distance = Infinity;
    hitInfoBZ.distance = Infinity;
    
    const distH1 = h1BVH ? (h1BVH.closestPointToPoint(tempCenter, hitInfoH1), hitInfoH1.distance) : Infinity;
    const distH2 = h2BVH ? (h2BVH.closestPointToPoint(tempCenter, hitInfoH2), hitInfoH2.distance) : Infinity;
    const distBZ = boundaryZoneBVH ? (boundaryZoneBVH.closestPointToPoint(tempCenter, hitInfoBZ), hitInfoBZ.distance) : Infinity;
    
    if (distBZ <= distH1 && distBZ <= distH2) {
      boundaryLabel[i] = 0;
      boundaryZoneAdjacent++;
    } else if (distH1 <= distH2) {
      boundaryLabel[i] = 1;
      h1Adjacent++;
    } else {
      boundaryLabel[i] = 2;
      h2Adjacent++;
    }
  }
  
  console.log(`  Labeling time: ${(performance.now() - labelStartTime).toFixed(1)}ms`);
  console.log(`  Outer boundary labeled: H₁=${h1Adjacent}, H₂=${h2Adjacent}, BoundaryZone(unlabeled)=${boundaryZoneAdjacent}`);
  
  h1Geometry.dispose();
  h2Geometry.dispose();
  boundaryZoneGeometry.dispose();
  
  // Step 5: Build boundaryAdjacentMask - outer surface + one level deep
  const boundaryAdjacentMask = new Uint8Array(voxelCount);
  let boundaryAdjacentCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    if (isOuterSurface[i]) {
      boundaryAdjacentMask[i] = 1;
      boundaryAdjacentCount++;
    }
  }
  
  for (let i = 0; i < voxelCount; i++) {
    if (isOuterSurface[i]) {
      let gi: number, gj: number, gk: number;
      
      if (useFlat) {
        const i3 = i * 3;
        gi = gridResult.voxelIndices![i3];
        gj = gridResult.voxelIndices![i3 + 1];
        gk = gridResult.voxelIndices![i3 + 2];
      } else {
        const cell = cells[i];
        gi = Math.round(cell.index.x);
        gj = Math.round(cell.index.y);
        gk = Math.round(cell.index.z);
      }
      
      for (const [di, dj, dk] of faceNeighborOffsets) {
        const neighborIdx = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
        if (neighborIdx >= 0 && !boundaryAdjacentMask[neighborIdx]) {
          boundaryAdjacentMask[neighborIdx] = 1;
          boundaryAdjacentCount++;
        }
      }
    }
  }
  
  console.log(`  Boundary voxels (outer + 1 level): ${boundaryAdjacentCount}`);
  
  return { boundaryLabel, innerSurfaceMask, boundaryAdjacentMask, h1Adjacent, h2Adjacent, innerSurfaceCount };
}

// ============================================================================
// ESCAPE LABELING COMPUTATION (OUTWARD DIJKSTRA)
// ============================================================================

/**
 * Compute escape labeling for mold volume voxels using outward Dijkstra
 * 
 * Algorithm:
 * 1. Seed from voxels with high volume intersection with part mesh (≥75% default)
 * 2. Flood outward using thickness-weighted edge costs
 * 3. When reaching a boundary-adjacent voxel, assign the label (H₁ or H₂)
 * 4. Propagate labels so each voxel knows which mold half it escapes to
 * 
 * @param gridResult - The volumetric grid result with mold volume cells and voxelDist
 * @param shellGeometry - The outer shell geometry (∂H) in world space
 * @param classification - The mold half classification (H₁/H₂ triangle sets)
 * @param options - Labeling options
 * @param partGeometry - Optional part geometry for volume intersection seed detection
 * @returns Escape labeling result
 */
export function computeEscapeLabelingDijkstra(
  gridResult: VolumetricGridResult,
  shellGeometry: THREE.BufferGeometry,
  classification: MoldHalfClassificationResult,
  options: EscapeLabelingOptions = {},
  partGeometry?: THREE.BufferGeometry
): EscapeLabelingResult {
  const startTime = performance.now();
  
  const adjacency = options.adjacency ?? 6;
  const seedRadius = options.seedRadius ?? 1.5;
  const seedVolumeThreshold = options.seedVolumeThreshold ?? 0.75;
  const seedVolumeSamples = options.seedVolumeSamples ?? 4;
  
  const voxelCount = gridResult.moldVolumeCellCount;
  const cells = gridResult.moldVolumeCells;
  const voxelDist = gridResult.voxelDist;
  const voxelSize = gridResult.cellSize.x;
  
  console.log('═══════════════════════════════════════════════════════');
  console.log('ESCAPE LABELING (Outward Dijkstra from Part)');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Voxel count: ${voxelCount}`);
  console.log(`  Adjacency: ${adjacency}-connected`);
  console.log(`  Voxel size: ${voxelSize.toFixed(4)}`);
  console.log(`  Seed volume threshold: ${(seedVolumeThreshold * 100).toFixed(0)}%`);
  console.log(`  Part geometry provided: ${partGeometry ? 'yes' : 'no'}`);
  
  if (!voxelDist) {
    console.error('ERROR: voxelDist is null - distance field not computed!');
    throw new Error('Distance field (voxelDist) is required for escape labeling');
  }
  
  const weightingFactor = gridResult.weightingFactor;
  if (!weightingFactor) {
    console.error('ERROR: weightingFactor is null - weighting factor not computed!');
    throw new Error('Weighting factor is required for escape labeling');
  }
  
  console.log(`  Weighting factor range: [${gridResult.weightingFactorStats?.min.toFixed(6)}, ${gridResult.weightingFactorStats?.max.toFixed(6)}]`);
  
  // Build spatial index for neighbor lookups
  const spatialIndex = buildVoxelSpatialIndex(gridResult);
  
  // Get neighbor offsets (used for connectivity checks)
  const neighborOffsets = getNeighborOffsets(adjacency);
  
  // Create part mesh tester if geometry is provided and threshold > 0
  let partMeshTester: MeshTester | undefined;
  if (partGeometry && seedVolumeThreshold > 0) {
    console.log('  Building part mesh BVH for volume intersection...');
    partMeshTester = new MeshTester(partGeometry, 'part');
  }
  
  // ========================================================================
  // STEP 1: Identify boundary-adjacent voxels (which touch H₁ or H₂)
  // ========================================================================
  
  console.log('Identifying boundary-adjacent voxels...');
  const { boundaryLabel, innerSurfaceMask, h1Adjacent, h2Adjacent, innerSurfaceCount } = identifyBoundaryAdjacentVoxels(
    gridResult,
    shellGeometry,
    classification,
    seedRadius,
    partMeshTester,
    seedVolumeThreshold,
    seedVolumeSamples
  );
  
  // Clean up part mesh tester if created
  if (partMeshTester) {
    partMeshTester.dispose();
  }
  
  console.log(`  H₁ adjacent: ${h1Adjacent}`);
  console.log(`  H₂ adjacent: ${h2Adjacent}`);
  console.log(`  Seeds: ${innerSurfaceCount}`);
  
  // Diagnostic: Check if there are any boundary voxels at all
  const totalBoundaryVoxels = h1Adjacent + h2Adjacent;
  if (totalBoundaryVoxels === 0) {
    console.error('ERROR: No boundary voxels found! All seeds will be unlabeled.');
  }
  
  // Diagnostic: Quick connectivity check - count how many voxels have at least one neighbor
  let connectedVoxels = 0;
  let isolatedVoxels = 0;
  for (let i = 0; i < voxelCount; i++) {
    const cell = cells[i];
    const gi = Math.round(cell.index.x);
    const gj = Math.round(cell.index.y);
    const gk = Math.round(cell.index.z);
    
    let hasNeighbor = false;
    for (const [di, dj, dk] of neighborOffsets) {
      const j = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
      if (j >= 0) {
        hasNeighbor = true;
        break;
      }
    }
    
    if (hasNeighbor) connectedVoxels++;
    else isolatedVoxels++;
  }
  console.log(`  Connectivity check: ${connectedVoxels} connected, ${isolatedVoxels} isolated voxels`);
  
  // ========================================================================
  // STEP 2: Independent Dijkstra from each seed to find nearest boundary
  // 
  // The multi-source approach fails because seeds compete for voxels - inner
  // seeds get blocked by outer seeds' paths. Instead, we run independent
  // Dijkstra for each seed, but optimize by:
  // 1. Reusing boundary voxel indices
  // 2. Early termination when any boundary is reached
  // 3. Batching for efficiency
  //
  // Gap jumping: If a direct neighbor doesn't exist, we try progressively
  // larger hops (up to 5 voxels) in the same direction.
  // ========================================================================
  
  console.log('Running independent Dijkstra from each seed to find nearest boundary...');
  
  // Use 26-connectivity for better coverage (diagonal neighbors)
  const fullNeighborOffsets = getNeighborOffsets(26);
  const fullNeighborDistances = fullNeighborOffsets.map(([di, dj, dk]) => 
    Math.sqrt(di * di + dj * dj + dk * dk) * voxelSize
  );
  
  // Collect seed indices
  const seedIndices: number[] = [];
  for (let i = 0; i < voxelCount; i++) {
    if (innerSurfaceMask[i]) {
      seedIndices.push(i);
    }
  }
  console.log(`  Total seeds: ${seedIndices.length}`);
  
  // Count how many boundary voxels exist
  let boundaryVoxelCount = 0;
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabel[i] !== 0) {
      boundaryVoxelCount++;
    }
  }
  console.log(`  Total boundary voxels (H₁ + H₂): ${boundaryVoxelCount}`);
  
  // Seed labels array
  const seedLabel = new Int8Array(voxelCount).fill(-1);
  
  // For each seed, run Dijkstra until we hit a boundary
  // Use a reusable cost array and visited set per seed
  const MAX_HOP_DISTANCE = 5;
  
  let labeledCount = 0;
  let totalVisited = 0;
  let totalExpanded = 0;
  const dijkstraStartTime = performance.now();
  
  // Process seeds in batches for better memory locality
  // We'll use a shared cost array but reset relevant parts between seeds
  const costArray = new Float32Array(voxelCount);
  const visitedSet = new Uint8Array(voxelCount);
  
  // Version counter to avoid clearing arrays (much faster)
  // Each seed gets a unique version number
  const versionArray = new Uint32Array(voxelCount);
  let currentVersion = 1;
  
  const progressInterval = Math.max(1, Math.floor(seedIndices.length / 10));
  
  for (let seedNum = 0; seedNum < seedIndices.length; seedNum++) {
    const seedIdx = seedIndices[seedNum];
    
    // Progress logging
    if (seedNum % progressInterval === 0) {
      const pct = Math.round(100 * seedNum / seedIndices.length);
      console.log(`  Processing seed ${seedNum}/${seedIndices.length} (${pct}%)...`);
    }
    
    // Increment version to "clear" arrays without actually clearing them
    currentVersion++;
    if (currentVersion === 0) {
      // Handle overflow - actually clear and reset
      versionArray.fill(0);
      currentVersion = 1;
    }
    
    // Initialize Dijkstra for this seed
    const pq = new MinHeap();
    costArray[seedIdx] = 0;
    versionArray[seedIdx] = currentVersion;
    pq.push(seedIdx, 0);
    
    let foundBoundary = false;
    let boundaryLabelFound = 0;
    
    while (pq.size > 0 && !foundBoundary) {
      const current = pq.pop()!;
      const i = current.index;
      const currentCost = current.cost;
      
      // Skip if already visited in this version or cost is stale
      if (visitedSet[i] === currentVersion) continue;
      if (versionArray[i] === currentVersion && currentCost > costArray[i]) continue;
      
      visitedSet[i] = currentVersion;
      totalVisited++;
      
      // Check if we've reached a boundary
      if (boundaryLabel[i] !== 0) {
        boundaryLabelFound = boundaryLabel[i];
        foundBoundary = true;
        break;
      }
      
      totalExpanded++;
      
      // Get current voxel's grid coordinates
      const cell = cells[i];
      const gi = Math.round(cell.index.x);
      const gj = Math.round(cell.index.y);
      const gk = Math.round(cell.index.z);
      
      // Expand to neighbors
      for (let n = 0; n < fullNeighborOffsets.length; n++) {
        const [di, dj, dk] = fullNeighborOffsets[n];
        
        // Try hop distances from 1 to MAX_HOP_DISTANCE
        for (let hopDistance = 1; hopDistance <= MAX_HOP_DISTANCE; hopDistance++) {
          const j = getVoxelIndex(gi + hopDistance*di, gj + hopDistance*dj, gk + hopDistance*dk, spatialIndex);
          
          if (j < 0) continue;  // No voxel at this hop distance, try next
          
          // Skip if already visited in this version
          if (visitedSet[j] === currentVersion) {
            break;  // Found a neighbor, don't try larger hops
          }
          
          const L = fullNeighborDistances[n] * hopDistance;
          const wt_j = weightingFactor[j];
          const edgeCost = L * wt_j;
          const candidateCost = currentCost + edgeCost;
          
          // Check if this is a better path (or first path in this version)
          if (versionArray[j] !== currentVersion || candidateCost < costArray[j]) {
            costArray[j] = candidateCost;
            versionArray[j] = currentVersion;
            pq.push(j, candidateCost);
          }
          
          break;  // Found a neighbor at this hop distance, don't try larger hops
        }
      }
    }
    
    // Label this seed
    if (foundBoundary) {
      seedLabel[seedIdx] = boundaryLabelFound;
      labeledCount++;
    }
  }
  
  const dijkstraElapsed = performance.now() - dijkstraStartTime;
  console.log(`  Independent Dijkstra completed in ${dijkstraElapsed.toFixed(1)}ms`);
  console.log(`  Total visited: ${totalVisited}, Total expanded: ${totalExpanded}`);
  console.log(`  Average voxels visited per seed: ${(totalVisited / seedIndices.length).toFixed(1)}`);
  console.log(`  Seeds labeled: ${labeledCount} / ${seedIndices.length}`);
  
  // Count results
  let h1Seeds = 0, h2Seeds = 0, unlabeledSeeds = 0;
  for (const seedIdx of seedIndices) {
    if (seedLabel[seedIdx] === 1) h1Seeds++;
    else if (seedLabel[seedIdx] === 2) h2Seeds++;
    else unlabeledSeeds++;
  }
  console.log(`  Seeds after Dijkstra: H₁=${h1Seeds}, H₂=${h2Seeds}, orphaned=${unlabeledSeeds}`);
  
  // ========================================================================
  // STEP 3: Build final label array
  // ========================================================================
  
  const voxelLabel = new Int8Array(voxelCount).fill(-1);
  
  // Label seeds
  for (let i = 0; i < voxelCount; i++) {
    if (innerSurfaceMask[i]) {
      voxelLabel[i] = seedLabel[i];
    }
  }
  
  // Also label boundary voxels for visualization
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabel[i] !== 0) {
      voxelLabel[i] = boundaryLabel[i];
    }
  }
  
  // Count how many interior voxels there are (for logging)
  let interiorCount = 0;
  for (let i = 0; i < voxelCount; i++) {
    if (voxelLabel[i] === -1) {
      interiorCount++;
    }
  }
  console.log(`  Interior voxels (intentionally unlabeled): ${interiorCount}`);
  
  // ========================================================================
  // COUNT RESULTS
  // ========================================================================
  
  // Create a simple escape cost array (not used for algorithm, just for API)
  const escapeCost = new Float32Array(voxelCount).fill(0);
  
  let d1Count = 0;
  let d2Count = 0;
  let unlabeledCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const label = voxelLabel[i];
    if (label === 1) d1Count++;
    else if (label === 2) d2Count++;
    else unlabeledCount++;
  }
  
  const computeTimeMs = performance.now() - startTime;
  
  console.log('───────────────────────────────────────────────────────');
  console.log('ESCAPE LABELING RESULTS:');
  console.log(`  H₁ (side 1): ${d1Count} (${(d1Count / voxelCount * 100).toFixed(1)}%)`);
  console.log(`  H₂ (side 2): ${d2Count} (${(d2Count / voxelCount * 100).toFixed(1)}%)`);
  console.log(`  Unlabeled:   ${unlabeledCount} (${(unlabeledCount / voxelCount * 100).toFixed(1)}%)`);
  console.log(`  Time: ${computeTimeMs.toFixed(1)}ms`);
  console.log('═══════════════════════════════════════════════════════');
  
  // DEBUG: Log boundary label distribution
  let boundaryH1 = 0, boundaryH2 = 0, boundaryNone = 0;
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabel[i] === 1) boundaryH1++;
    else if (boundaryLabel[i] === 2) boundaryH2++;
    else boundaryNone++;
  }
  console.log('DEBUG - Boundary voxel labels:');
  console.log(`  Boundary H₁: ${boundaryH1}`);
  console.log(`  Boundary H₂: ${boundaryH2}`);
  console.log(`  Not boundary: ${boundaryNone}`);
  
  // DEBUG: Analyze seed label neighbor consistency (detect mixing/orphans)
  let consistentSeeds = 0;
  let mixedSeeds = 0;
  let isolatedSeeds = 0;
  const mixedExamples: string[] = [];
  
  for (let i = 0; i < voxelCount; i++) {
    if (!innerSurfaceMask[i]) continue;
    
    const myLabel = seedLabel[i];
    if (myLabel === -1) continue;
    
    const cell = cells[i];
    const gi = Math.round(cell.index.x);
    const gj = Math.round(cell.index.y);
    const gk = Math.round(cell.index.z);
    
    let sameCount = 0;
    let diffCount = 0;
    let neighborLabels: number[] = [];
    
    for (const [di, dj, dk] of neighborOffsets) {
      const j = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
      if (j >= 0 && innerSurfaceMask[j] && seedLabel[j] !== -1) {
        neighborLabels.push(seedLabel[j]);
        if (seedLabel[j] === myLabel) sameCount++;
        else diffCount++;
      }
    }
    
    if (neighborLabels.length === 0) {
      isolatedSeeds++;
    } else if (diffCount === 0) {
      consistentSeeds++;
    } else {
      mixedSeeds++;
      if (mixedExamples.length < 5) {
        mixedExamples.push(`Seed ${i}: label=${myLabel}, neighbors=[${neighborLabels.join(',')}]`);
      }
    }
  }
  
  console.log('DEBUG - Seed neighbor consistency:');
  console.log(`  Consistent (all neighbors same label): ${consistentSeeds}`);
  console.log(`  Mixed (has neighbors with different label): ${mixedSeeds}`);
  console.log(`  Isolated (no seed neighbors): ${isolatedSeeds}`);
  if (mixedExamples.length > 0) {
    console.log('  Examples of mixed seeds:');
    mixedExamples.forEach(ex => console.log(`    ${ex}`));
  }
  
  // DEBUG: Analyze boundary voxel neighbor consistency
  let consistentBoundary = 0;
  let mixedBoundary = 0;
  let isolatedBoundary = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabel[i] === 0) continue;
    
    const myLabel = boundaryLabel[i];
    const cell = cells[i];
    const gi = Math.round(cell.index.x);
    const gj = Math.round(cell.index.y);
    const gk = Math.round(cell.index.z);
    
    let sameCount = 0;
    let diffCount = 0;
    let neighborCount = 0;
    
    for (const [di, dj, dk] of neighborOffsets) {
      const j = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
      if (j >= 0 && boundaryLabel[j] !== 0) {
        neighborCount++;
        if (boundaryLabel[j] === myLabel) sameCount++;
        else diffCount++;
      }
    }
    
    if (neighborCount === 0) {
      isolatedBoundary++;
    } else if (diffCount === 0) {
      consistentBoundary++;
    } else {
      mixedBoundary++;
    }
  }
  
  console.log('DEBUG - Boundary voxel neighbor consistency:');
  console.log(`  Consistent: ${consistentBoundary}`);
  console.log(`  At H₁/H₂ interface: ${mixedBoundary}`);
  console.log(`  Isolated: ${isolatedBoundary}`);
  
  return {
    labels: voxelLabel,
    escapeCost,
    d1Count,
    d2Count,
    interfaceCount: 0,
    unlabeledCount,
    computeTimeMs,
    // DEBUG fields
    boundaryLabels: boundaryLabel,
    seedMask: innerSurfaceMask,
  };
}

// ============================================================================
// ASYNC PARALLEL VERSION (Uses Web Workers)
// ============================================================================

/**
 * Async version of computeEscapeLabelingDijkstra that supports parallel processing
 * via Web Workers. When parallel=true, the Dijkstra searches are distributed across
 * multiple workers for significant speedup on multi-core systems.
 * 
 * For single-threaded execution (parallel=false), this delegates to the synchronous version.
 */
export async function computeEscapeLabelingDijkstraAsync(
  gridResult: VolumetricGridResult,
  shellGeometry: THREE.BufferGeometry,
  classification: MoldHalfClassificationResult,
  options: EscapeLabelingOptions = {},
  partGeometry?: THREE.BufferGeometry
): Promise<EscapeLabelingResult> {
  const parallel = options.parallel ?? false;
  
  // If not parallel or Web Workers unavailable, use synchronous version
  if (!parallel || !isWebWorkersAvailable()) {
    return computeEscapeLabelingDijkstra(gridResult, shellGeometry, classification, options, partGeometry);
  }
  
  const startTime = performance.now();
  
  const adjacency = options.adjacency ?? 6;
  const seedRadius = options.seedRadius ?? 1.5;
  const seedVolumeThreshold = options.seedVolumeThreshold ?? 0.75;
  const seedVolumeSamples = options.seedVolumeSamples ?? 4;
  const numWorkers = typeof parallel === 'number' ? parallel : undefined;
  
  const voxelCount = gridResult.moldVolumeCellCount;
  const voxelSize = gridResult.cellSize.x;
  
  console.log('═══════════════════════════════════════════════════════');
  console.log('ESCAPE LABELING (Parallel Dijkstra with Web Workers)');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Voxel count: ${voxelCount}`);
  console.log(`  Workers: ${numWorkers ?? 'auto (' + (navigator.hardwareConcurrency || 4) + ')'}`);
  console.log(`  Adjacency: ${adjacency}-connected`);
  console.log(`  Voxel size: ${voxelSize.toFixed(4)}`);
  
  if (!gridResult.voxelDist) {
    throw new Error('Distance field (voxelDist) is required for escape labeling');
  }
  
  if (!gridResult.weightingFactor) {
    throw new Error('Weighting factor is required for escape labeling');
  }
  
  // Step 1: Identify boundary-adjacent voxels using PARALLEL volume intersection
  console.log('Identifying boundary-adjacent voxels (parallel)...');
  const { boundaryLabel, innerSurfaceMask, h1Adjacent, h2Adjacent, innerSurfaceCount } = await identifyBoundaryAdjacentVoxelsAsync(
    gridResult,
    shellGeometry,
    classification,
    seedRadius,
    partGeometry,
    seedVolumeThreshold,
    seedVolumeSamples,
    numWorkers ?? DEFAULT_NUM_WORKERS
  );
  
  console.log(`  H₁ adjacent: ${h1Adjacent}`);
  console.log(`  H₂ adjacent: ${h2Adjacent}`);
  console.log(`  Seeds: ${innerSurfaceCount}`);
  
  // Collect seed indices
  const seedIndices: number[] = [];
  for (let i = 0; i < voxelCount; i++) {
    if (innerSurfaceMask[i]) {
      seedIndices.push(i);
    }
  }
  console.log(`  Total seeds: ${seedIndices.length}`);
  
  // Step 2: Run parallel Dijkstra
  console.log('Running parallel Dijkstra from each seed to find nearest boundary...');
  
  const parallelResult = await runParallelDijkstra(
    gridResult,
    seedIndices,
    boundaryLabel,
    {
      numWorkers,
      maxHopDistance: 5,
      onProgress: options.onProgress ? (p) => {
        options.onProgress?.({ completed: p.completed, total: p.total });
      } : undefined,
    }
  );
  
  const seedLabel = parallelResult.seedLabel;
  
  // Count seed labels
  let h1Seeds = 0, h2Seeds = 0, unlabeledSeeds = 0;
  for (const seedIdx of seedIndices) {
    if (seedLabel[seedIdx] === 1) h1Seeds++;
    else if (seedLabel[seedIdx] === 2) h2Seeds++;
    else unlabeledSeeds++;
  }
  console.log(`  Seeds after Dijkstra: H₁=${h1Seeds}, H₂=${h2Seeds}, orphaned=${unlabeledSeeds}`);
  
  // Step 3: Build final label array
  const voxelLabel = new Int8Array(voxelCount).fill(-1);
  
  // Label seeds
  for (let i = 0; i < voxelCount; i++) {
    if (innerSurfaceMask[i]) {
      voxelLabel[i] = seedLabel[i];
    }
  }
  
  // Also label boundary voxels
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabel[i] !== 0) {
      voxelLabel[i] = boundaryLabel[i];
    }
  }
  
  // Count results
  const escapeCost = new Float32Array(voxelCount).fill(0);
  let d1Count = 0;
  let d2Count = 0;
  let unlabeledCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const label = voxelLabel[i];
    if (label === 1) d1Count++;
    else if (label === 2) d2Count++;
    else unlabeledCount++;
  }
  
  const computeTimeMs = performance.now() - startTime;
  
  console.log('───────────────────────────────────────────────────────');
  console.log('PARALLEL ESCAPE LABELING RESULTS:');
  console.log(`  H₁ (side 1): ${d1Count} (${(d1Count / voxelCount * 100).toFixed(1)}%)`);
  console.log(`  H₂ (side 2): ${d2Count} (${(d2Count / voxelCount * 100).toFixed(1)}%)`);
  console.log(`  Unlabeled:   ${unlabeledCount} (${(unlabeledCount / voxelCount * 100).toFixed(1)}%)`);
  console.log(`  Total time: ${computeTimeMs.toFixed(1)}ms`);
  console.log(`  Dijkstra time: ${parallelResult.totalTimeMs.toFixed(1)}ms`);
  console.log('═══════════════════════════════════════════════════════');
  
  return {
    labels: voxelLabel,
    escapeCost,
    d1Count,
    d2Count,
    interfaceCount: 0,
    unlabeledCount,
    computeTimeMs,
    boundaryLabels: boundaryLabel,
    seedMask: innerSurfaceMask,
  };
}

// ============================================================================
// LEGACY API (for backwards compatibility with ThreeViewer)
// ============================================================================

/**
 * Legacy wrapper that accepts partGeometry and d1/d2 directions
 * This is a fallback if mold half classification is not available
 */
export function computeEscapeLabeling(
  gridResult: VolumetricGridResult,
  _partGeometry: THREE.BufferGeometry,
  d1: THREE.Vector3,
  d2: THREE.Vector3,
  _options: EscapeLabelingOptions = {}
): EscapeLabelingResult {
  console.warn('Using legacy escape labeling (simple directional heuristic)');
  console.warn('For proper Dijkstra-based labeling, use computeEscapeLabelingDijkstra with mold half classification');
  
  const startTime = performance.now();
  const voxelCount = gridResult.moldVolumeCellCount;
  
  const labels = new Int8Array(voxelCount);
  const escapeCost = new Float32Array(voxelCount);
  
  let d1Count = 0;
  let d2Count = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const cell = gridResult.moldVolumeCells[i];
    const pos = cell.center;
    
    const dot1 = pos.dot(d1);
    const dot2 = pos.dot(d2);
    
    if (dot1 >= dot2) {
      labels[i] = 1;
      escapeCost[i] = Math.abs(dot1);
      d1Count++;
    } else {
      labels[i] = 2;
      escapeCost[i] = Math.abs(dot2);
      d2Count++;
    }
  }
  
  const computeTimeMs = performance.now() - startTime;
  
  return {
    labels,
    escapeCost,
    d1Count,
    d2Count,
    interfaceCount: 0,
    unlabeledCount: 0,
    computeTimeMs,
  };
}

// ============================================================================
// VISUALIZATION
// ============================================================================

/**
 * Create a point cloud visualization of escape labeling
 * Colors: Green (H₁/side 1), Orange (H₂/side 2), Gray (unlabeled)
 */
export function createEscapeLabelingPointCloud(
  gridResult: VolumetricGridResult,
  labeling: EscapeLabelingResult,
  pointSize: number = 3
): THREE.Points {
  const voxelCount = gridResult.moldVolumeCellCount;
  const positions = new Float32Array(voxelCount * 3);
  const colors = new Float32Array(voxelCount * 3);
  
  const colorD1 = new THREE.Color(0x00ff00);    // Green (H₁)
  const colorD2 = new THREE.Color(0xff6600);    // Orange (H₂)
  const colorUnlabeled = new THREE.Color(0x888888); // Gray
  
  for (let i = 0; i < voxelCount; i++) {
    const cell = gridResult.moldVolumeCells[i];
    positions[i * 3] = cell.center.x;
    positions[i * 3 + 1] = cell.center.y;
    positions[i * 3 + 2] = cell.center.z;
    
    const label = labeling.labels[i];
    let color: THREE.Color;
    if (label === 1) {
      color = colorD1;
    } else if (label === 2) {
      color = colorD2;
    } else {
      color = colorUnlabeled;
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: pointSize,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.7,
    vertexColors: true,
  });
  
  return new THREE.Points(geometry, material);
}

/**
 * Create a point cloud showing seed voxels (inner surface near part).
 * 
 * When showLabeled=false (default): Shows boundary voxels colored by H₁/H₂, 
 * and seed voxels in cyan (original behavior).
 * 
 * When showLabeled=true: Shows ONLY seed voxels colored by their escape labels
 * (which boundary they reach via shortest path).
 * 
 * Colors:
 * - Bright Green: H₁ (boundary or seed escaping to H₁)
 * - Bright Orange: H₂ (boundary or seed escaping to H₂)
 * - Cyan: Seeds (only when showLabeled=false)
 * - Gray: Unlabeled seeds (only when showLabeled=true)
 */
export function createPartingSurfaceInterfaceCloud(
  gridResult: VolumetricGridResult,
  labeling: EscapeLabelingResult,
  _adjacency: AdjacencyType = 6,
  pointSize: number = 4,
  showLabeled: boolean = false
): THREE.Points {
  const voxelCount = gridResult.moldVolumeCellCount;
  const cells = gridResult.moldVolumeCells;
  const seedMask = labeling.seedMask;
  const boundaryLabels = labeling.boundaryLabels;
  const labels = labeling.labels;
  
  if (showLabeled) {
    // Show ONLY seed voxels, colored by their escape labels
    let count = 0;
    for (let i = 0; i < voxelCount; i++) {
      if (seedMask && seedMask[i]) count++;
    }
    
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    const colorH1 = new THREE.Color(0x00ff00);       // Green for H₁
    const colorH2 = new THREE.Color(0xff6600);       // Orange for H₂
    const colorUnlabeled = new THREE.Color(0x888888); // Gray for unlabeled
    
    let idx = 0;
    let h1Count = 0, h2Count = 0, unlabeledCount = 0;
    
    for (let i = 0; i < voxelCount; i++) {
      if (seedMask && seedMask[i]) {
        const cell = cells[i];
        positions[idx * 3] = cell.center.x;
        positions[idx * 3 + 1] = cell.center.y;
        positions[idx * 3 + 2] = cell.center.z;
        
        let color: THREE.Color;
        if (labels[i] === 1) {
          color = colorH1;
          h1Count++;
        } else if (labels[i] === 2) {
          color = colorH2;
          h2Count++;
        } else {
          color = colorUnlabeled;
          unlabeledCount++;
        }
        
        colors[idx * 3] = color.r;
        colors[idx * 3 + 1] = color.g;
        colors[idx * 3 + 2] = color.b;
        idx++;
      }
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
      vertexColors: true,
    });
    
    console.log(`Labeled seed cloud: ${count} voxels (H₁: ${h1Count}, H₂: ${h2Count}, unlabeled: ${unlabeledCount})`);
    
    return new THREE.Points(geometry, material);
  } else {
    // Original behavior: boundary voxels + cyan seeds
    let count = 0;
    for (let i = 0; i < voxelCount; i++) {
      if (boundaryLabels && boundaryLabels[i] !== 0) count++;
      else if (seedMask && seedMask[i]) count++;
    }
    
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    const colorH1 = new THREE.Color(0x00ff00);    // Green for H₁ boundary
    const colorH2 = new THREE.Color(0xff6600);    // Orange for H₂ boundary
    const colorSeed = new THREE.Color(0x00ffff);  // Cyan for seeds
    
    let idx = 0;
    for (let i = 0; i < voxelCount; i++) {
      let color: THREE.Color | null = null;
      
      if (boundaryLabels && boundaryLabels[i] === 1) {
        color = colorH1;
      } else if (boundaryLabels && boundaryLabels[i] === 2) {
        color = colorH2;
      } else if (seedMask && seedMask[i]) {
        color = colorSeed;
      }
      
      if (color) {
        const cell = cells[i];
        positions[idx * 3] = cell.center.x;
        positions[idx * 3 + 1] = cell.center.y;
        positions[idx * 3 + 2] = cell.center.z;
        colors[idx * 3] = color.r;
        colors[idx * 3 + 1] = color.g;
        colors[idx * 3 + 2] = color.b;
        idx++;
      }
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
      vertexColors: true,
    });
    
    console.log(`Boundary/Seed cloud: ${count} voxels`);
    
    return new THREE.Points(geometry, material);
  }
}

/**
 * DEBUG: Create a point cloud showing ONLY boundary voxels with their fixed labels
 * This helps verify that boundary detection is working correctly.
 * 
 * Colors:
 * - Bright Green: Boundary adjacent to H₁
 * - Bright Orange: Boundary adjacent to H₂
 * - Also shows seeds in a different opacity
 */
export function createBoundaryDebugPointCloud(
  gridResult: VolumetricGridResult,
  labeling: EscapeLabelingResult,
  pointSize: number = 6
): THREE.Points {
  const voxelCount = gridResult.moldVolumeCellCount;
  const boundaryLabels = labeling.boundaryLabels;
  const seedMask = labeling.seedMask;
  
  if (!boundaryLabels) {
    console.error('DEBUG: boundaryLabels not available in labeling result');
    // Fallback to regular visualization
    return createEscapeLabelingPointCloud(gridResult, labeling, pointSize);
  }
  
  // Count boundary voxels
  let h1BoundaryCount = 0;
  let h2BoundaryCount = 0;
  let seedCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabels[i] === 1) h1BoundaryCount++;
    else if (boundaryLabels[i] === 2) h2BoundaryCount++;
    if (seedMask && seedMask[i]) seedCount++;
  }
  
  console.log('═══════════════════════════════════════════════════════');
  console.log('DEBUG: Boundary Voxel Visualization');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Total voxels: ${voxelCount}`);
  console.log(`  H₁ boundary voxels: ${h1BoundaryCount}`);
  console.log(`  H₂ boundary voxels: ${h2BoundaryCount}`);
  console.log(`  Seed voxels: ${seedCount}`);
  console.log(`  Non-boundary: ${voxelCount - h1BoundaryCount - h2BoundaryCount}`);
  
  // Create arrays for all voxels
  const positions = new Float32Array(voxelCount * 3);
  const colors = new Float32Array(voxelCount * 3);
  
  // Colors for boundary voxels (bright, saturated)
  const colorBoundaryH1 = new THREE.Color(0x00ff00);    // Bright Green
  const colorBoundaryH2 = new THREE.Color(0xff6600);    // Bright Orange
  // Colors for non-boundary voxels (dim)
  const colorNonBoundary = new THREE.Color(0x333333);   // Dark gray
  const colorSeed = new THREE.Color(0x0066ff);          // Blue for seeds
  
  for (let i = 0; i < voxelCount; i++) {
    const cell = gridResult.moldVolumeCells[i];
    positions[i * 3] = cell.center.x;
    positions[i * 3 + 1] = cell.center.y;
    positions[i * 3 + 2] = cell.center.z;
    
    let color: THREE.Color;
    const bLabel = boundaryLabels[i];
    const isSeed = seedMask && seedMask[i];
    
    if (bLabel === 1) {
      color = colorBoundaryH1;  // H₁ boundary - green
    } else if (bLabel === 2) {
      color = colorBoundaryH2;  // H₂ boundary - orange
    } else if (isSeed) {
      color = colorSeed;        // Seed - blue
    } else {
      color = colorNonBoundary; // Not boundary - dark
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: pointSize,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.9,
    vertexColors: true,
  });
  
  return new THREE.Points(geometry, material);
}

/**
 * DEBUG: Create visualization showing ONLY boundary voxels (hide everything else)
 */
export function createBoundaryOnlyPointCloud(
  gridResult: VolumetricGridResult,
  labeling: EscapeLabelingResult,
  pointSize: number = 8
): THREE.Points | null {
  const voxelCount = gridResult.moldVolumeCellCount;
  const boundaryLabels = labeling.boundaryLabels;
  
  if (!boundaryLabels) {
    console.error('DEBUG: boundaryLabels not available');
    return null;
  }
  
  // Collect only boundary voxels
  const boundaryIndices: number[] = [];
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabels[i] !== 0) {
      boundaryIndices.push(i);
    }
  }
  
  const count = boundaryIndices.length;
  console.log(`DEBUG: Creating boundary-only cloud with ${count} voxels`);
  
  if (count === 0) {
    console.warn('DEBUG: No boundary voxels found!');
    return null;
  }
  
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  
  const colorH1 = new THREE.Color(0x00ff00);  // Green
  const colorH2 = new THREE.Color(0xff6600);  // Orange
  
  for (let i = 0; i < count; i++) {
    const voxelIdx = boundaryIndices[i];
    const cell = gridResult.moldVolumeCells[voxelIdx];
    
    positions[i * 3] = cell.center.x;
    positions[i * 3 + 1] = cell.center.y;
    positions[i * 3 + 2] = cell.center.z;
    
    const color = boundaryLabels[voxelIdx] === 1 ? colorH1 : colorH2;
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: pointSize,
    sizeAttenuation: true,
    transparent: true,
    opacity: 1.0,
    vertexColors: true,
  });
  
  return new THREE.Points(geometry, material);
}

// ============================================================================
// DEBUG VISUALIZATION FUNCTIONS
// ============================================================================

export type PartingSurfaceDebugMode = 'none' | 'surface-detection' | 'boundary-labels' | 'seed-labels' | 'seed-labels-only';

/**
 * Create debug visualization based on the selected mode.
 * 
 * Modes:
 * - 'surface-detection': Shows inner surface (cyan) vs outer surface (yellow)
 *   Only shows surface voxels, not bulk volume.
 * - 'boundary-labels': Shows outer surface with H₁ (green) vs H₂ (orange) labels
 * - 'seed-labels': Shows inner surface (seeds) with their Dijkstra-assigned labels
 */
export function createDebugVisualization(
  gridResult: VolumetricGridResult,
  labeling: EscapeLabelingResult,
  mode: PartingSurfaceDebugMode,
  pointSize: number = 6
): THREE.Object3D | null {
  if (mode === 'none') {
    return null;
  }
  
  const voxelCount = gridResult.moldVolumeCellCount;
  const cells = gridResult.moldVolumeCells;
  const seedMask = labeling.seedMask;
  const boundaryLabels = labeling.boundaryLabels;
  const labels = labeling.labels;
  
  if (!seedMask || !boundaryLabels) {
    console.error('DEBUG: seedMask or boundaryLabels not available');
    console.error('  seedMask:', seedMask);
    console.error('  boundaryLabels:', boundaryLabels);
    return null;
  }
  
  // Log seed mask stats
  let seedMaskCount = 0;
  for (let i = 0; i < voxelCount; i++) {
    if (seedMask[i]) seedMaskCount++;
  }
  console.log(`DEBUG: seedMask has ${seedMaskCount} seeds out of ${voxelCount} voxels`);
  
  // Colors
  const colorInnerSurface = new THREE.Color(0x00ffff);   // Cyan - inner surface (seeds)
  const colorOuterSurface = new THREE.Color(0xffff00);   // Yellow - outer surface (boundary)
  const colorH1 = new THREE.Color(0x00ff00);             // Green - H₁
  const colorH2 = new THREE.Color(0xff6600);             // Orange - H₂
  const colorUnlabeled = new THREE.Color(0x888888);      // Grey - unlabeled (boundary zone)
  const unlabeledOpacity = 0.15;                          // Very low opacity for unlabeled
  
  // Collect voxels to display based on mode
  // Separate labeled and unlabeled voxels for different opacity
  const displayVoxels: { idx: number; color: THREE.Color; isUnlabeled: boolean }[] = [];
  
  if (mode === 'surface-detection') {
    // Show ONLY surface voxels: inner (seeds) and outer (boundary)
    // Inner surface = cyan, Outer surface = yellow
    console.log('DEBUG MODE: Surface Detection (Inner=Cyan, Outer=Yellow)');
    
    let innerCount = 0;
    let outerCount = 0;
    
    for (let i = 0; i < voxelCount; i++) {
      if (seedMask[i]) {
        displayVoxels.push({ idx: i, color: colorInnerSurface, isUnlabeled: false });
        innerCount++;
      } else if (boundaryLabels[i] !== 0) {
        displayVoxels.push({ idx: i, color: colorOuterSurface, isUnlabeled: false });
        outerCount++;
      }
      // Bulk volume voxels are NOT added
    }
    
    console.log(`  Inner surface (seeds): ${innerCount}`);
    console.log(`  Outer surface (boundary): ${outerCount}`);
    console.log(`  Total displayed: ${displayVoxels.length} / ${voxelCount}`);
    
  } else if (mode === 'boundary-labels') {
    // Show outer surface (boundary) voxels with H₁/H₂ colors AND seed voxels in cyan
    // This shows the state BEFORE Dijkstra runs
    console.log('DEBUG MODE: Boundary Labels + Seeds (H₁=Green, H₂=Orange, Seeds=Cyan)');
    
    let h1Count = 0;
    let h2Count = 0;
    let seedCount = 0;
    
    for (let i = 0; i < voxelCount; i++) {
      if (boundaryLabels[i] === 1) {
        displayVoxels.push({ idx: i, color: colorH1, isUnlabeled: false });
        h1Count++;
      } else if (boundaryLabels[i] === 2) {
        displayVoxels.push({ idx: i, color: colorH2, isUnlabeled: false });
        h2Count++;
      } else if (seedMask[i]) {
        // Seeds shown in cyan (before they get labeled)
        displayVoxels.push({ idx: i, color: colorInnerSurface, isUnlabeled: false });
        seedCount++;
      }
    }
    
    console.log(`  H₁ boundary: ${h1Count}`);
    console.log(`  H₂ boundary: ${h2Count}`);
    console.log(`  Seeds (unlabeled): ${seedCount}`);
    console.log(`  Total displayed: ${displayVoxels.length}`);
    
  } else if (mode === 'seed-labels') {
    // Show ALL voxels with their Dijkstra-assigned labels
    console.log('DEBUG MODE: All Voxel Labels (H₁=Green, H₂=Orange, Unlabeled=Grey)');
    
    let h1Count = 0;
    let h2Count = 0;
    let unlabeledCount = 0;
    
    for (let i = 0; i < voxelCount; i++) {
      const label = labels[i];
      if (label === 1) {
        displayVoxels.push({ idx: i, color: colorH1, isUnlabeled: false });
        h1Count++;
      } else if (label === 2) {
        displayVoxels.push({ idx: i, color: colorH2, isUnlabeled: false });
        h2Count++;
      } else {
        displayVoxels.push({ idx: i, color: colorUnlabeled, isUnlabeled: true });
        unlabeledCount++;
      }
    }
    
    console.log(`  H₁ voxels: ${h1Count}`);
    console.log(`  H₂ voxels: ${h2Count}`);
    console.log(`  Unlabeled: ${unlabeledCount}`);
    console.log(`  Total displayed: ${displayVoxels.length}`);
    
  } else if (mode === 'seed-labels-only') {
    // Show ONLY seed voxels with their Dijkstra-assigned labels
    console.log('DEBUG MODE: Seed Labels Only (H₁=Green, H₂=Orange, Unlabeled=Grey)');
    
    let h1Count = 0;
    let h2Count = 0;
    let unlabeledCount = 0;
    
    for (let i = 0; i < voxelCount; i++) {
      if (seedMask[i]) {
        const label = labels[i];
        if (label === 1) {
          displayVoxels.push({ idx: i, color: colorH1, isUnlabeled: false });
          h1Count++;
        } else if (label === 2) {
          displayVoxels.push({ idx: i, color: colorH2, isUnlabeled: false });
          h2Count++;
        } else {
          displayVoxels.push({ idx: i, color: colorUnlabeled, isUnlabeled: true });
          unlabeledCount++;
        }
      }
    }
    
    console.log(`  Seeds → H₁: ${h1Count}`);
    console.log(`  Seeds → H₂: ${h2Count}`);
    console.log(`  Seeds unlabeled: ${unlabeledCount}`);
    console.log(`  Total displayed: ${displayVoxels.length}`);
  }
  
  if (displayVoxels.length === 0) {
    console.warn('DEBUG: No voxels to display!');
    return null;
  }
  
  // Separate labeled and unlabeled voxels for different opacity rendering
  const labeledVoxels = displayVoxels.filter(v => !v.isUnlabeled);
  const unlabeledVoxels = displayVoxels.filter(v => v.isUnlabeled);
  
  // Create a group to hold both point clouds
  const group = new THREE.Group();
  
  // Create labeled voxels point cloud (full opacity)
  if (labeledVoxels.length > 0) {
    const count = labeledVoxels.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const { idx, color } = labeledVoxels[i];
      const cell = cells[idx];
      
      positions[i * 3] = cell.center.x;
      positions[i * 3 + 1] = cell.center.y;
      positions[i * 3 + 2] = cell.center.z;
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity: 1.0,
      vertexColors: true,
    });
    
    group.add(new THREE.Points(geometry, material));
  }
  
  // Create unlabeled voxels point cloud (very low opacity)
  if (unlabeledVoxels.length > 0) {
    const count = unlabeledVoxels.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const { idx, color } = unlabeledVoxels[i];
      const cell = cells[idx];
      
      positions[i * 3] = cell.center.x;
      positions[i * 3 + 1] = cell.center.y;
      positions[i * 3 + 2] = cell.center.z;
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity: unlabeledOpacity,
      vertexColors: true,
      depthWrite: false,  // Prevent z-fighting with labeled voxels
    });
    
    group.add(new THREE.Points(geometry, material));
  }
  
  return group;
}