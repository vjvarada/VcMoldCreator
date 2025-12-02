/**
 * Web Worker for parallelized Dijkstra search
 * 
 * OPTIMIZED VERSION:
 * - Direct 3D array index lookup (no string keys)
 * - 6-connected neighbors only (faster, usually sufficient)
 * - Inlined hot paths
 */

// ============================================================================
// OPTIMIZED MIN-HEAP (Flat arrays, no object allocation)
// ============================================================================

class FastMinHeap {
  private indices: Uint32Array;
  private costs: Float32Array;
  private size_: number = 0;
  private capacity: number;

  constructor(initialCapacity: number = 4096) {
    this.capacity = initialCapacity;
    this.indices = new Uint32Array(initialCapacity);
    this.costs = new Float32Array(initialCapacity);
  }

  get size(): number {
    return this.size_;
  }

  clear(): void {
    this.size_ = 0;
  }

  push(index: number, cost: number): void {
    if (this.size_ >= this.capacity) {
      this.grow();
    }
    
    let pos = this.size_++;
    this.indices[pos] = index;
    this.costs[pos] = cost;
    
    // Bubble up
    while (pos > 0) {
      const parent = (pos - 1) >> 1;
      if (this.costs[pos] >= this.costs[parent]) break;
      
      // Swap
      const tmpIdx = this.indices[pos];
      const tmpCost = this.costs[pos];
      this.indices[pos] = this.indices[parent];
      this.costs[pos] = this.costs[parent];
      this.indices[parent] = tmpIdx;
      this.costs[parent] = tmpCost;
      pos = parent;
    }
  }

  pop(): { index: number; cost: number } | null {
    if (this.size_ === 0) return null;
    
    const minIdx = this.indices[0];
    const minCost = this.costs[0];
    
    this.size_--;
    if (this.size_ > 0) {
      this.indices[0] = this.indices[this.size_];
      this.costs[0] = this.costs[this.size_];
      this.bubbleDown();
    }
    
    return { index: minIdx, cost: minCost };
  }

  private bubbleDown(): void {
    let pos = 0;
    const n = this.size_;
    
    while (true) {
      const left = (pos << 1) + 1;
      const right = left + 1;
      let smallest = pos;
      
      if (left < n && this.costs[left] < this.costs[smallest]) {
        smallest = left;
      }
      if (right < n && this.costs[right] < this.costs[smallest]) {
        smallest = right;
      }
      
      if (smallest === pos) break;
      
      // Swap
      const tmpIdx = this.indices[pos];
      const tmpCost = this.costs[pos];
      this.indices[pos] = this.indices[smallest];
      this.costs[pos] = this.costs[smallest];
      this.indices[smallest] = tmpIdx;
      this.costs[smallest] = tmpCost;
      pos = smallest;
    }
  }

  private grow(): void {
    this.capacity *= 2;
    const newIndices = new Uint32Array(this.capacity);
    const newCosts = new Float32Array(this.capacity);
    newIndices.set(this.indices);
    newCosts.set(this.costs);
    this.indices = newIndices;
    this.costs = newCosts;
  }
}

// ============================================================================
// 6-CONNECTED NEIGHBORS (face-adjacent only - much faster)
// ============================================================================

// Pre-computed offsets for 6-connected neighbors
const NEIGHBOR_DI = new Int8Array([-1, 1, 0, 0, 0, 0]);
const NEIGHBOR_DJ = new Int8Array([0, 0, -1, 1, 0, 0]);
const NEIGHBOR_DK = new Int8Array([0, 0, 0, 0, -1, 1]);
const NEIGHBOR_DIST = 1.0; // All face neighbors are distance 1

// ============================================================================
// TYPES
// ============================================================================

export interface DijkstraWorkerInput {
  type: 'process';
  workerId: number;
  seedIndices: number[];
  voxelCount: number;
  voxelSize: number;
  maxHopDistance: number;
  gridIndices: Int32Array;
  boundaryLabel: Int8Array;
  weightingFactor: Float32Array;
  spatialIndexKeys: string[];
  spatialIndexValues: Int32Array;
  // Grid dimensions for direct index computation
  gridResX?: number;
  gridResY?: number;
  gridResZ?: number;
}

export interface DijkstraWorkerOutput {
  type: 'result';
  workerId: number;
  seedLabels: Int8Array;
  visitedCount: number;
  expandedCount: number;
  labeledCount: number;
  processingTimeMs: number;
}

export interface DijkstraWorkerProgress {
  type: 'progress';
  workerId: number;
  processedSeeds: number;
  totalSeeds: number;
}

// ============================================================================
// FAST SPATIAL INDEX (Int32Array-based 3D lookup)
// ============================================================================

/**
 * Build a fast 3D lookup table from grid indices
 * Returns -1 for empty cells
 */
function buildFastSpatialIndex(
  gridIndices: Int32Array,
  voxelCount: number,
  resX: number,
  resY: number,
  resZ: number
): Int32Array {
  // Allocate 3D grid as flat array, initialized to -1
  const gridSize = resX * resY * resZ;
  const lookup = new Int32Array(gridSize).fill(-1);
  
  for (let idx = 0; idx < voxelCount; idx++) {
    const i3 = idx * 3;
    const gi = gridIndices[i3];
    const gj = gridIndices[i3 + 1];
    const gk = gridIndices[i3 + 2];
    
    // Flat index into 3D grid
    const flatIdx = gi + gj * resX + gk * resX * resY;
    lookup[flatIdx] = idx;
  }
  
  return lookup;
}

/**
 * Process a batch of seeds using optimized Dijkstra
 */
function processSeedBatch(input: DijkstraWorkerInput): DijkstraWorkerOutput {
  const startTime = performance.now();
  
  const {
    workerId,
    seedIndices,
    voxelCount,
    voxelSize,
    gridIndices,
    boundaryLabel,
    weightingFactor,
    gridResX,
    gridResY,
    gridResZ,
  } = input;
  
  // If grid dimensions not provided, fall back to Map-based lookup
  const useDirectLookup = gridResX !== undefined && gridResY !== undefined && gridResZ !== undefined;
  
  let spatialLookup: Int32Array | null = null;
  let spatialMap: Map<string, number> | null = null;
  
  if (useDirectLookup) {
    spatialLookup = buildFastSpatialIndex(gridIndices, voxelCount, gridResX!, gridResY!, gridResZ!);
  } else {
    // Fallback: rebuild Map-based spatial index
    spatialMap = new Map<string, number>();
    for (let i = 0; i < input.spatialIndexKeys.length; i++) {
      spatialMap.set(input.spatialIndexKeys[i], input.spatialIndexValues[i]);
    }
  }
  
  const resX = gridResX ?? 0;
  const resY = gridResY ?? 0;
  const resZ = gridResZ ?? 0;
  const resXY = resX * resY;
  
  // Pre-compute neighbor distance with voxel size
  const neighborDist = NEIGHBOR_DIST * voxelSize;
  
  // Result arrays
  const seedLabels = new Int8Array(seedIndices.length).fill(-1);
  
  // Reusable arrays for Dijkstra (version-based clearing)
  const costArray = new Float32Array(voxelCount);
  const versionArray = new Uint32Array(voxelCount);
  let currentVersion = 1;
  
  // Reusable priority queue
  const pq = new FastMinHeap(Math.min(voxelCount, 65536));
  
  let totalVisited = 0;
  let totalExpanded = 0;
  let labeledCount = 0;
  
  // Progress reporting interval
  const progressInterval = Math.max(1, Math.floor(seedIndices.length / 10));
  
  for (let seedNum = 0; seedNum < seedIndices.length; seedNum++) {
    const seedIdx = seedIndices[seedNum];
    
    // Report progress periodically
    if (seedNum % progressInterval === 0) {
      const progressMsg: DijkstraWorkerProgress = {
        type: 'progress',
        workerId,
        processedSeeds: seedNum,
        totalSeeds: seedIndices.length,
      };
      self.postMessage(progressMsg);
    }
    
    // Increment version to "clear" arrays
    currentVersion++;
    if (currentVersion === 0) {
      versionArray.fill(0);
      currentVersion = 1;
    }
    
    // Clear and initialize priority queue
    pq.clear();
    costArray[seedIdx] = 0;
    versionArray[seedIdx] = currentVersion;
    pq.push(seedIdx, 0);
    
    let foundBoundary = false;
    let boundaryLabelFound = 0;
    
    while (pq.size > 0) {
      const current = pq.pop()!;
      const idx = current.index;
      const currentCost = current.cost;
      
      // Skip if cost is stale (already found better path)
      if (versionArray[idx] === currentVersion && currentCost > costArray[idx]) {
        continue;
      }
      
      totalVisited++;
      
      // Check if we've reached a boundary
      const bl = boundaryLabel[idx];
      if (bl !== 0) {
        boundaryLabelFound = bl;
        foundBoundary = true;
        break;
      }
      
      totalExpanded++;
      
      // Get current voxel's grid coordinates
      const i3 = idx * 3;
      const gi = gridIndices[i3];
      const gj = gridIndices[i3 + 1];
      const gk = gridIndices[i3 + 2];
      
      // Expand to 6-connected neighbors (inlined for speed)
      for (let n = 0; n < 6; n++) {
        const ni = gi + NEIGHBOR_DI[n];
        const nj = gj + NEIGHBOR_DJ[n];
        const nk = gk + NEIGHBOR_DK[n];
        
        // Get neighbor voxel index
        let neighborIdx: number;
        if (useDirectLookup) {
          // Bounds check
          if (ni < 0 || ni >= resX || nj < 0 || nj >= resY || nk < 0 || nk >= resZ) {
            continue;
          }
          neighborIdx = spatialLookup![ni + nj * resX + nk * resXY];
        } else {
          const key = `${ni},${nj},${nk}`;
          neighborIdx = spatialMap!.get(key) ?? -1;
        }
        
        if (neighborIdx < 0) continue;
        
        // Compute edge cost
        const wt = weightingFactor[neighborIdx];
        const edgeCost = neighborDist * wt;
        const candidateCost = currentCost + edgeCost;
        
        // Check if this is a better path
        if (versionArray[neighborIdx] !== currentVersion || candidateCost < costArray[neighborIdx]) {
          costArray[neighborIdx] = candidateCost;
          versionArray[neighborIdx] = currentVersion;
          pq.push(neighborIdx, candidateCost);
        }
      }
    }
    
    // Store the label for this seed
    if (foundBoundary) {
      seedLabels[seedNum] = boundaryLabelFound;
      labeledCount++;
    }
  }
  
  const processingTimeMs = performance.now() - startTime;
  
  return {
    type: 'result',
    workerId,
    seedLabels,
    visitedCount: totalVisited,
    expandedCount: totalExpanded,
    labeledCount,
    processingTimeMs,
  };
}

// ============================================================================
// WORKER ENTRY POINT
// ============================================================================

self.onmessage = (event: MessageEvent<DijkstraWorkerInput>) => {
  const input = event.data;
  
  if (input.type === 'process') {
    const result = processSeedBatch(input);
    (self as unknown as Worker).postMessage(result, [result.seedLabels.buffer]);
  }
};

export {};
