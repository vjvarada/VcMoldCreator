/**
 * Parting Surface / Escape Labeling
 * 
 * This module computes the parting surface for mold design using multi-source
 * Dijkstra from outer mold halves (H₁, H₂) inward through the silicone volume.
 * 
 * Algorithm (STEP 3.5):
 * - Seed voxels near H₁ faces with label=1, cost=0
 * - Seed voxels near H₂ faces with label=2, cost=0
 * - Run Dijkstra with thickness-weighted edge costs:
 *     edgeCost(i,j) = L / (d_mid² + eps)
 *   where L = Euclidean distance, d_mid = average distance to part mesh
 * - Each voxel inherits the label of its cheapest escape path to the boundary
 * - The parting surface is where label transitions occur (interface voxels)
 */

import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';
import type { VolumetricGridResult } from './volumetricGrid';
import type { MoldHalfClassificationResult } from './moldHalfClassification';

// ============================================================================
// TYPES
// ============================================================================

export type AdjacencyType = 6 | 18 | 26;

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
// NEIGHBOR OFFSETS
// ============================================================================

/** 6-connected neighbors (face adjacency) */
const NEIGHBORS_6 = [
  [-1, 0, 0], [1, 0, 0],
  [0, -1, 0], [0, 1, 0],
  [0, 0, -1], [0, 0, 1],
];

/** 18-connected neighbors (face + edge adjacency) */
const NEIGHBORS_18 = [
  ...NEIGHBORS_6,
  [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],
  [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],
  [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1],
];

/** 26-connected neighbors (face + edge + corner adjacency) */
const NEIGHBORS_26 = [
  ...NEIGHBORS_18,
  [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
  [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
];

function getNeighborOffsets(adjacency: AdjacencyType): number[][] {
  switch (adjacency) {
    case 6: return NEIGHBORS_6;
    case 18: return NEIGHBORS_18;
    case 26: return NEIGHBORS_26;
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
  
  for (let i = 0; i < gridResult.moldVolumeCells.length; i++) {
    const cell = gridResult.moldVolumeCells[i];
    const key = `${Math.round(cell.index.x)},${Math.round(cell.index.y)},${Math.round(cell.index.z)}`;
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

/**
 * Identify which voxels are on the OUTER SURFACE of the silicone volume
 * (near the mold shell ∂H, NOT near the part mesh M).
 * 
 * Also identifies INNER SURFACE voxels (near the part mesh) for seeding.
 * 
 * A voxel is a surface voxel if it has at least one 6-connected neighbor 
 * that is NOT a silicone voxel.
 * 
 * - Outer surface = surface voxels with large distance to part
 * - Inner surface = surface voxels with small distance to part
 */
function identifyBoundaryAdjacentVoxels(
  gridResult: VolumetricGridResult,
  shellGeometry: THREE.BufferGeometry,
  classification: MoldHalfClassificationResult,
  _boundaryRadius: number
): { 
  boundaryLabel: Int8Array; 
  innerSurfaceMask: Uint8Array;
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
  
  // Step 3: Mark OUTER and INNER surface voxels
  const isOuterSurface = new Uint8Array(voxelCount);
  const innerSurfaceMask = new Uint8Array(voxelCount);
  let outerSurfaceCount = 0;
  let innerSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i]) {
      if (voxelDist[i] >= distanceThreshold) {
        isOuterSurface[i] = 1;
        outerSurfaceCount++;
      } else {
        innerSurfaceMask[i] = 1;
        innerSurfaceCount++;
      }
    }
  }
  
  console.log(`  Inner surface (near part): ${innerSurfaceCount}`);
  console.log(`  Outer surface (near shell): ${outerSurfaceCount}`);
  
  // Step 4: For each OUTER surface voxel, determine if it's closer to H₁ or H₂
  // Use BVH for fast closest-point queries
  
  console.log('  Building BVH for H₁ and H₂ triangles...');
  const bvhStartTime = performance.now();
  
  const position = shellGeometry.getAttribute('position');
  const shellIndex = shellGeometry.getIndex();
  
  // Create separate geometries for H1 and H2 with their own BVHs
  const createSubGeometry = (triangleSet: Set<number>): THREE.BufferGeometry => {
    const triCount = triangleSet.size;
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
  
  // Create sub-geometries for H1 and H2
  const h1Geometry = createSubGeometry(classification.h1Triangles);
  const h2Geometry = createSubGeometry(classification.h2Triangles);
  
  // Build BVH for each
  const h1BVH = new MeshBVH(h1Geometry);
  const h2BVH = new MeshBVH(h2Geometry);
  
  console.log(`  BVH build time: ${(performance.now() - bvhStartTime).toFixed(1)}ms`);
  console.log(`  H₁ triangles: ${classification.h1Triangles.size}, H₂ triangles: ${classification.h2Triangles.size}`);
  
  // Label each OUTER surface voxel based on closest mold half using BVH
  const boundaryLabel = new Int8Array(voxelCount);
  let h1Adjacent = 0;
  let h2Adjacent = 0;
  
  const labelStartTime = performance.now();
  const hitInfoH1 = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoH2 = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  
  for (let i = 0; i < voxelCount; i++) {
    if (!isOuterSurface[i]) continue;
    
    const center = cells[i].center;
    
    // Use BVH closestPointToPoint for fast queries
    // Reset hit info
    hitInfoH1.distance = Infinity;
    hitInfoH2.distance = Infinity;
    
    h1BVH.closestPointToPoint(center, hitInfoH1);
    h2BVH.closestPointToPoint(center, hitInfoH2);
    
    const distH1 = hitInfoH1.distance;
    const distH2 = hitInfoH2.distance;
    
    // Assign to closer mold half
    if (distH1 <= distH2) {
      boundaryLabel[i] = 1;
      h1Adjacent++;
    } else {
      boundaryLabel[i] = 2;
      h2Adjacent++;
    }
  }
  
  console.log(`  Labeling time: ${(performance.now() - labelStartTime).toFixed(1)}ms`);
  console.log(`  Outer boundary labeled: H₁=${h1Adjacent}, H₂=${h2Adjacent}`);
  
  // Clean up
  h1Geometry.dispose();
  h2Geometry.dispose();
  
  return { boundaryLabel, innerSurfaceMask, h1Adjacent, h2Adjacent, innerSurfaceCount };
}

// ============================================================================
// ESCAPE LABELING COMPUTATION (OUTWARD DIJKSTRA)
// ============================================================================

/**
 * Compute escape labeling for mold volume voxels using outward Dijkstra
 * 
 * Algorithm:
 * 1. Seed from voxels closest to the part mesh (small voxelDist values)
 * 2. Flood outward using thickness-weighted edge costs
 * 3. When reaching a boundary-adjacent voxel, assign the label (H₁ or H₂)
 * 4. Propagate labels so each voxel knows which mold half it escapes to
 * 
 * @param gridResult - The volumetric grid result with mold volume cells and voxelDist
 * @param shellGeometry - The outer shell geometry (∂H) in world space
 * @param classification - The mold half classification (H₁/H₂ triangle sets)
 * @param options - Labeling options
 * @returns Escape labeling result
 */
export function computeEscapeLabelingDijkstra(
  gridResult: VolumetricGridResult,
  shellGeometry: THREE.BufferGeometry,
  classification: MoldHalfClassificationResult,
  options: EscapeLabelingOptions = {}
): EscapeLabelingResult {
  const startTime = performance.now();
  
  const adjacency = options.adjacency ?? 6;
  const seedRadius = options.seedRadius ?? 1.5;
  
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
  
  if (!voxelDist) {
    console.error('ERROR: voxelDist is null - distance field not computed!');
    throw new Error('Distance field (voxelDist) is required for escape labeling');
  }
  
  // Build spatial index for neighbor lookups
  const spatialIndex = buildVoxelSpatialIndex(gridResult);
  
  // Get neighbor offsets
  const neighborOffsets = getNeighborOffsets(adjacency);
  
  // Pre-compute Euclidean distances for each neighbor offset
  const neighborDistances = neighborOffsets.map(([di, dj, dk]) => 
    Math.sqrt(di * di + dj * dj + dk * dk) * voxelSize
  );
  
  // Small epsilon to avoid division by zero
  const eps = 1e-6 * voxelSize * voxelSize;
  
  // ========================================================================
  // STEP 1: Identify boundary-adjacent voxels (which touch H₁ or H₂)
  // ========================================================================
  
  console.log('Identifying boundary-adjacent voxels...');
  const { boundaryLabel, innerSurfaceMask, h1Adjacent, h2Adjacent, innerSurfaceCount } = identifyBoundaryAdjacentVoxels(
    gridResult,
    shellGeometry,
    classification,
    seedRadius
  );
  console.log(`  H₁ adjacent: ${h1Adjacent}`);
  console.log(`  H₂ adjacent: ${h2Adjacent}`);
  console.log(`  Inner surface (seeds): ${innerSurfaceCount}`);
  
  // ========================================================================
  // TEMPORARY: Skip Dijkstra and return boundary-only results for debugging
  // ========================================================================
  
  const SKIP_DIJKSTRA = false;  // Set to true to disable Dijkstra for debugging
  
  if (SKIP_DIJKSTRA) {
    console.log('DIJKSTRA DISABLED - returning boundary labels only');
    
    const computeTimeMs = performance.now() - startTime;
    
    // Use boundary labels directly as the final labels
    // Interior voxels remain unlabeled (-1)
    const voxelLabel = new Int8Array(voxelCount).fill(-1);
    for (let i = 0; i < voxelCount; i++) {
      if (boundaryLabel[i] !== 0) {
        voxelLabel[i] = boundaryLabel[i];
      }
    }
    
    // Count results
    let d1Count = 0;
    let d2Count = 0;
    let unlabeledCount = 0;
    for (let i = 0; i < voxelCount; i++) {
      if (voxelLabel[i] === 1) d1Count++;
      else if (voxelLabel[i] === 2) d2Count++;
      else unlabeledCount++;
    }
    
    console.log('───────────────────────────────────────────────────────');
    console.log('BOUNDARY-ONLY RESULTS (Dijkstra disabled):');
    console.log(`  H₁ boundary: ${d1Count}`);
    console.log(`  H₂ boundary: ${d2Count}`);
    console.log(`  Unlabeled (interior): ${unlabeledCount}`);
    console.log(`  Time: ${computeTimeMs.toFixed(1)}ms`);
    console.log('═══════════════════════════════════════════════════════');
    
    return {
      labels: voxelLabel,
      escapeCost: new Float32Array(voxelCount),
      d1Count,
      d2Count,
      interfaceCount: 0,
      unlabeledCount,
      computeTimeMs,
      boundaryLabels: boundaryLabel,
      seedMask: innerSurfaceMask,
    };
  }
  
  // ========================================================================
  // STEP 2: Run Dijkstra FROM BOUNDARY VOXELS inward
  // 
  // Instead of flooding outward from seeds, we flood INWARD from boundaries.
  // This way, each voxel directly knows which boundary it's closest to.
  // 
  // Multi-source Dijkstra:
  // - Seed H₁ boundary voxels with label=1, cost=0
  // - Seed H₂ boundary voxels with label=2, cost=0
  // - Each voxel inherits the label of its nearest boundary
  // ========================================================================
  
  // Initialize data structures
  const voxelCost = new Float32Array(voxelCount).fill(Infinity);
  const voxelLabel = new Int8Array(voxelCount).fill(-1);
  const pq = new MinHeap();
  
  // Seed from boundary voxels
  let h1Seeds = 0, h2Seeds = 0;
  for (let i = 0; i < voxelCount; i++) {
    if (boundaryLabel[i] === 1) {
      voxelCost[i] = 0;
      voxelLabel[i] = 1;
      pq.push(i, 0);
      h1Seeds++;
    } else if (boundaryLabel[i] === 2) {
      voxelCost[i] = 0;
      voxelLabel[i] = 2;
      pq.push(i, 0);
      h2Seeds++;
    }
  }
  console.log(`  Boundary seeds: H₁=${h1Seeds}, H₂=${h2Seeds}`);
  
  // ========================================================================
  // STEP 3: Dijkstra flood INWARD from boundaries
  // 
  // Edge cost: thickness-weighted to prefer paths through thick regions
  // ========================================================================
  
  console.log('Running inward Dijkstra flood from boundaries...');
  
  let iterations = 0;
  
  while (pq.size > 0) {
    const current = pq.pop()!;
    const i = current.index;
    const currentCost = current.cost;
    
    // Skip if we already found a better path
    if (currentCost > voxelCost[i]) continue;
    
    // Get current voxel's grid coordinates
    const cell = cells[i];
    const gi = Math.round(cell.index.x);
    const gj = Math.round(cell.index.y);
    const gk = Math.round(cell.index.z);
    
    const di_current = voxelDist[i];  // Current voxel's distance to part
    
    // Visit all neighbors
    for (let n = 0; n < neighborOffsets.length; n++) {
      const [di, dj, dk] = neighborOffsets[n];
      const ni = gi + di;
      const nj = gj + dj;
      const nk = gk + dk;
      
      // Find neighbor voxel index
      const j = getVoxelIndex(ni, nj, nk, spatialIndex);
      if (j < 0) continue; // Not a silicone voxel
      
      const dj_neighbor = voxelDist[j];
      const L = neighborDistances[n];
      
      // Edge cost: thickness-weighted
      // Lower cost in thick regions (high voxelDist) 
      // Higher cost in thin regions (low voxelDist)
      const d_mid = 0.5 * (di_current + dj_neighbor);
      const edgeCost = L / (d_mid * d_mid + eps);
      
      // Candidate cost to reach j through i
      const candidateCost = voxelCost[i] + edgeCost;
      
      // If this is a better path, update
      if (candidateCost < voxelCost[j]) {
        voxelCost[j] = candidateCost;
        voxelLabel[j] = voxelLabel[i];  // Inherit label from source
        pq.decreaseKey(j, candidateCost);
      }
    }
    
    iterations++;
  }
  
  console.log(`  Dijkstra iterations: ${iterations}`);
  
  // ========================================================================
  // STEP 4: Handle any voxels that weren't reached by Dijkstra
  // ========================================================================
  
  // Handle any remaining unlabeled voxels (disconnected regions)
  let stillUnlabeled = 0;
  for (let i = 0; i < voxelCount; i++) {
    if (voxelLabel[i] === -1) {
      stillUnlabeled++;
      
      // Find nearest labeled neighbor
      const cell = cells[i];
      const gi = Math.round(cell.index.x);
      const gj = Math.round(cell.index.y);
      const gk = Math.round(cell.index.z);
      
      let bestLabel = -1;
      let bestCost = Infinity;
      
      for (const [di, dj, dk] of neighborOffsets) {
        const j = getVoxelIndex(gi + di, gj + dj, gk + dk, spatialIndex);
        if (j >= 0 && voxelLabel[j] !== -1 && voxelCost[j] < bestCost) {
          bestCost = voxelCost[j];
          bestLabel = voxelLabel[j];
        }
      }
      
      if (bestLabel !== -1) {
        voxelLabel[i] = bestLabel;
      }
    }
  }
  
  if (stillUnlabeled > 0) {
    console.log(`  Unreached voxels: ${stillUnlabeled} (fixed via neighbor lookup)`);
  }
  
  // ========================================================================
  // COUNT RESULTS
  // ========================================================================
  
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
  
  return {
    labels: voxelLabel,
    escapeCost: voxelCost,
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
 * Create a point cloud showing boundary voxels and seed voxels
 * This is useful for debugging the escape labeling algorithm.
 * 
 * Colors:
 * - Bright Green: Boundary voxels adjacent to H₁ (outer surface)
 * - Bright Orange: Boundary voxels adjacent to H₂ (outer surface)
 * - Cyan: Seed voxels (inner surface near part mesh)
 */
export function createPartingSurfaceInterfaceCloud(
  gridResult: VolumetricGridResult,
  labeling: EscapeLabelingResult,
  _adjacency: AdjacencyType = 6,
  pointSize: number = 4
): THREE.Points {
  const voxelCount = gridResult.moldVolumeCellCount;
  const cells = gridResult.moldVolumeCells;
  const boundaryLabels = labeling.boundaryLabels;
  const seedMask = labeling.seedMask;
  
  // Count how many voxels we'll show
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
  
  console.log(`Boundary/Seed cloud: ${count} voxels (H₁: boundary, H₂: boundary, Cyan: seeds)`);
  
  return new THREE.Points(geometry, material);
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
