/**
 * Web Worker for parallelized Dijkstra search
 * 
 * This worker processes batches of seed voxels, running independent Dijkstra
 * searches from each seed to find the nearest boundary voxel.
 */

// ============================================================================
// MIN-HEAP PRIORITY QUEUE (Worker-local copy)
// ============================================================================

class MinHeap {
  private heap: { index: number; cost: number }[] = [];

  get size(): number {
    return this.heap.length;
  }

  push(index: number, cost: number): void {
    const node = { index, cost };
    this.heap.push(node);
    this.bubbleUp(this.heap.length - 1);
  }

  pop(): { index: number; cost: number } | undefined {
    if (this.heap.length === 0) return undefined;
    
    const min = this.heap[0];
    const last = this.heap.pop()!;
    
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.bubbleDown(0);
    }
    
    return min;
  }

  private bubbleUp(pos: number): void {
    while (pos > 0) {
      const parent = Math.floor((pos - 1) / 2);
      if (this.heap[pos].cost >= this.heap[parent].cost) break;
      
      // Swap
      const temp = this.heap[pos];
      this.heap[pos] = this.heap[parent];
      this.heap[parent] = temp;
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
      
      // Swap
      const temp = this.heap[pos];
      this.heap[pos] = this.heap[smallest];
      this.heap[smallest] = temp;
      pos = smallest;
    }
  }
}

// ============================================================================
// NEIGHBOR OFFSETS (26-connected)
// ============================================================================

const NEIGHBORS_26: [number, number, number][] = [
  // Face neighbors (6)
  [-1, 0, 0], [1, 0, 0],
  [0, -1, 0], [0, 1, 0],
  [0, 0, -1], [0, 0, 1],
  // Edge neighbors (12)
  [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],
  [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],
  [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1],
  // Corner neighbors (8)
  [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
  [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
];

const NEIGHBOR_DISTANCES = NEIGHBORS_26.map(([di, dj, dk]) => 
  Math.sqrt(di * di + dj * dj + dk * dk)
);

// ============================================================================
// TYPES
// ============================================================================

export interface DijkstraWorkerInput {
  type: 'process';
  workerId: number;
  seedIndices: number[];           // Indices of seeds to process
  voxelCount: number;
  voxelSize: number;
  maxHopDistance: number;
  
  // Transferred arrays (SharedArrayBuffer or regular ArrayBuffer)
  gridIndices: Int32Array;         // [i0, j0, k0, i1, j1, k1, ...] for all voxels
  boundaryLabel: Int8Array;        // Boundary labels for all voxels
  weightingFactor: Float32Array;   // Weighting factor for all voxels
  spatialIndexKeys: string[];      // Serialized spatial index keys
  spatialIndexValues: Int32Array;  // Corresponding voxel indices
}

export interface DijkstraWorkerOutput {
  type: 'result';
  workerId: number;
  seedLabels: Int8Array;           // Labels for the processed seeds (same order as input)
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
// WORKER MESSAGE HANDLER
// ============================================================================

// Rebuild spatial index from serialized data
function rebuildSpatialIndex(keys: string[], values: Int32Array): Map<string, number> {
  const map = new Map<string, number>();
  for (let i = 0; i < keys.length; i++) {
    map.set(keys[i], values[i]);
  }
  return map;
}

function getVoxelIndex(
  i: number, j: number, k: number,
  spatialIndex: Map<string, number>
): number {
  const key = `${i},${j},${k}`;
  return spatialIndex.get(key) ?? -1;
}

/**
 * Process a batch of seeds using Dijkstra
 */
function processSeedBatch(input: DijkstraWorkerInput): DijkstraWorkerOutput {
  const startTime = performance.now();
  
  const {
    workerId,
    seedIndices,
    voxelCount,
    voxelSize,
    maxHopDistance,
    gridIndices,
    boundaryLabel,
    weightingFactor,
    spatialIndexKeys,
    spatialIndexValues,
  } = input;
  
  // Rebuild spatial index
  const spatialIndex = rebuildSpatialIndex(spatialIndexKeys, spatialIndexValues);
  
  // Pre-compute neighbor distances
  const fullNeighborDistances = NEIGHBOR_DISTANCES.map(d => d * voxelSize);
  
  // Result arrays
  const seedLabels = new Int8Array(seedIndices.length).fill(-1);
  
  // Reusable arrays for Dijkstra (version-based clearing)
  const costArray = new Float32Array(voxelCount);
  const visitedSet = new Uint8Array(voxelCount);
  const versionArray = new Uint32Array(voxelCount);
  let currentVersion = 1;
  
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
    
    // Seed's grid coordinates are retrieved via gridIndices array during traversal
    
    // Initialize Dijkstra
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
      
      // Skip if already visited or cost is stale
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
      const gi = gridIndices[i * 3];
      const gj = gridIndices[i * 3 + 1];
      const gk = gridIndices[i * 3 + 2];
      
      // Expand to neighbors
      for (let n = 0; n < NEIGHBORS_26.length; n++) {
        const [di, dj, dk] = NEIGHBORS_26[n];
        
        // Try hop distances from 1 to maxHopDistance
        for (let hopDistance = 1; hopDistance <= maxHopDistance; hopDistance++) {
          const j = getVoxelIndex(
            gi + hopDistance * di,
            gj + hopDistance * dj,
            gk + hopDistance * dk,
            spatialIndex
          );
          
          if (j < 0) continue;
          
          // Skip if already visited
          if (visitedSet[j] === currentVersion) {
            break;
          }
          
          const L = fullNeighborDistances[n] * hopDistance;
          const wt_j = weightingFactor[j];
          const edgeCost = L * wt_j;
          const candidateCost = currentCost + edgeCost;
          
          // Check if this is a better path
          if (versionArray[j] !== currentVersion || candidateCost < costArray[j]) {
            costArray[j] = candidateCost;
            versionArray[j] = currentVersion;
            pq.push(j, candidateCost);
          }
          
          break;
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
    // Transfer the result array back to main thread
    (self as unknown as Worker).postMessage(result, [result.seedLabels.buffer]);
  }
};

export {};
