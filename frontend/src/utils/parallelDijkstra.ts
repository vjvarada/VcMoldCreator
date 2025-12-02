/**
 * Parallel Dijkstra Coordinator
 * 
 * Manages multiple Web Workers to parallelize independent Dijkstra searches
 * from seed voxels to boundary voxels.
 */

import type { VolumetricGridResult } from './volumetricGrid';
import { logDebug, logResult } from './meshUtils';

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
  // Grid dimensions for fast 3D lookup (optional for backward compatibility)
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

export interface ParallelDijkstraOptions {
  /** Number of workers to use. Defaults to navigator.hardwareConcurrency or 4 */
  numWorkers?: number;
  /** Maximum hop distance for gap jumping. Default: 5 */
  maxHopDistance?: number;
  /** Progress callback */
  onProgress?: (progress: { completed: number; total: number; workerId: number }) => void;
}

export interface ParallelDijkstraResult {
  seedLabel: Int8Array;
  totalVisited: number;
  totalExpanded: number;
  labeledCount: number;
  totalTimeMs: number;
  workerTimesMs: number[];
}

/**
 * Serialize spatial index for transfer to workers
 */
function serializeSpatialIndex(
  spatialIndex: Map<string, number>
): { keys: string[]; values: Int32Array } {
  const keys: string[] = [];
  const values: number[] = [];
  
  for (const [key, value] of spatialIndex) {
    keys.push(key);
    values.push(value);
  }
  
  return {
    keys,
    values: new Int32Array(values),
  };
}

/**
 * Build grid indices array for worker transfer
 * Format: [i0, j0, k0, i1, j1, k1, ...] for all voxels
 */
function buildGridIndicesArray(gridResult: VolumetricGridResult): Int32Array {
  // Use flat arrays if available
  if (gridResult.voxelIndices) {
    // Already in the right format, just need to convert to Int32Array
    return new Int32Array(gridResult.voxelIndices);
  }
  
  // Fallback to building from moldVolumeCells
  const cells = gridResult.moldVolumeCells;
  const indices = new Int32Array(cells.length * 3);
  
  for (let i = 0; i < cells.length; i++) {
    indices[i * 3] = Math.round(cells[i].index.x);
    indices[i * 3 + 1] = Math.round(cells[i].index.y);
    indices[i * 3 + 2] = Math.round(cells[i].index.z);
  }
  
  return indices;
}

/**
 * Build spatial index from grid result
 */
function buildVoxelSpatialIndex(gridResult: VolumetricGridResult): Map<string, number> {
  const indexMap = new Map<string, number>();
  const voxelCount = gridResult.voxelIndices ? gridResult.voxelIndices.length / 3 : gridResult.moldVolumeCells.length;
  
  if (gridResult.voxelIndices) {
    // Use flat array
    for (let i = 0; i < voxelCount; i++) {
      const i3 = i * 3;
      const gi = gridResult.voxelIndices[i3];
      const gj = gridResult.voxelIndices[i3 + 1];
      const gk = gridResult.voxelIndices[i3 + 2];
      const key = `${gi},${gj},${gk}`;
      indexMap.set(key, i);
    }
  } else {
    // Fallback to moldVolumeCells
    for (let i = 0; i < voxelCount; i++) {
      const cell = gridResult.moldVolumeCells[i];
      const key = `${Math.round(cell.index.x)},${Math.round(cell.index.y)},${Math.round(cell.index.z)}`;
      indexMap.set(key, i);
    }
  }
  
  return indexMap;
}

/**
 * Run parallelized Dijkstra searches using Web Workers
 */
export async function runParallelDijkstra(
  gridResult: VolumetricGridResult,
  seedIndices: number[],
  boundaryLabel: Int8Array,
  options: ParallelDijkstraOptions = {}
): Promise<ParallelDijkstraResult> {
  const startTime = performance.now();
  
  const numWorkers = options.numWorkers ?? (navigator.hardwareConcurrency || 4);
  const maxHopDistance = options.maxHopDistance ?? 5;
  const voxelCount = gridResult.moldVolumeCellCount;
  const voxelSize = gridResult.cellSize.x;
  
  logDebug(`[ParallelDijkstra] Starting: ${numWorkers} workers, ${seedIndices.length} seeds`);
  
  // Prepare shared data
  const spatialIndex = buildVoxelSpatialIndex(gridResult);
  const { keys: spatialIndexKeys, values: spatialIndexValues } = serializeSpatialIndex(spatialIndex);
  const gridIndices = buildGridIndicesArray(gridResult);
  const weightingFactor = gridResult.weightingFactor!;
  
  // Extract grid dimensions for fast 3D lookup in workers
  const gridResX = gridResult.resolution?.x ?? 0;
  const gridResY = gridResult.resolution?.y ?? 0;
  const gridResZ = gridResult.resolution?.z ?? 0;
  
  // ========================================================================
  // SMART SEED ORDERING: Sort seeds by weighting factor (distance to part)
  // Seeds with higher weighting factor (closer to part) will have smaller
  // propagation areas, so we interleave them to balance load
  // ========================================================================
  const sortedSeeds = seedIndices.slice().sort((a, b) => {
    return (weightingFactor[b] || 0) - (weightingFactor[a] || 0); // High weight first
  });
  
  // Interleave seeds across workers for better load balancing
  // Round-robin distribution ensures each worker gets a mix of easy/hard seeds
  const workerSeedBatches: number[][] = Array.from({ length: numWorkers }, () => []);
  for (let i = 0; i < sortedSeeds.length; i++) {
    workerSeedBatches[i % numWorkers].push(sortedSeeds[i]);
  }
  
  const actualWorkerCount = workerSeedBatches.filter(b => b.length > 0).length;
  const avgSeeds = Math.round(seedIndices.length / actualWorkerCount);
  logDebug(`[ParallelDijkstra] Using ${actualWorkerCount} workers, ~${avgSeeds} seeds each (interleaved by weight)`);
  
  // Create workers and process batches
  const workers: Worker[] = [];
  const workerPromises: Promise<DijkstraWorkerOutput>[] = [];
  
  for (let w = 0; w < actualWorkerCount; w++) {
    // Create worker using URL constructor for Vite compatibility
    const worker = new Worker(
      new URL('./dijkstraWorker.ts', import.meta.url),
      { type: 'module' }
    );
    workers.push(worker);
    
    const promise = new Promise<DijkstraWorkerOutput>((resolve, reject) => {
      worker.onmessage = (event) => {
        const data = event.data;
        if (data.type === 'result') {
          resolve(data as DijkstraWorkerOutput);
        } else if (data.type === 'progress' && options.onProgress) {
          options.onProgress({
            completed: data.processedSeeds,
            total: data.totalSeeds,
            workerId: data.workerId,
          });
        }
      };
      
      worker.onerror = (error) => {
        reject(new Error(`Worker ${w} error: ${error.message}`));
      };
    });
    
    workerPromises.push(promise);
    
    // Send work to worker
    // Clone arrays for transfer (each worker gets its own copy)
    const workerInput: DijkstraWorkerInput = {
      type: 'process',
      workerId: w,
      seedIndices: workerSeedBatches[w],
      voxelCount,
      voxelSize,
      maxHopDistance,
      gridIndices: gridIndices.slice(),
      boundaryLabel: boundaryLabel.slice(),
      weightingFactor: weightingFactor.slice(),
      spatialIndexKeys,
      spatialIndexValues: spatialIndexValues.slice(),
      // Pass grid dimensions for fast 3D lookup
      gridResX,
      gridResY,
      gridResZ,
    };
    
    // Transfer arrays to worker
    worker.postMessage(workerInput, [
      workerInput.gridIndices.buffer,
      workerInput.boundaryLabel.buffer,
      workerInput.weightingFactor.buffer,
      workerInput.spatialIndexValues.buffer,
    ]);
  }
  
  // Wait for all workers to complete
  const results = await Promise.all(workerPromises);
  
  // Terminate workers
  workers.forEach(w => w.terminate());
  
  // Combine results
  const seedLabel = new Int8Array(voxelCount).fill(-1);
  let totalVisited = 0;
  let totalExpanded = 0;
  let labeledCount = 0;
  const workerTimesMs: number[] = [];
  
  for (let w = 0; w < results.length; w++) {
    const result = results[w];
    const batch = workerSeedBatches[w];
    
    // Copy seed labels to final array
    for (let i = 0; i < batch.length; i++) {
      const seedIdx = batch[i];
      seedLabel[seedIdx] = result.seedLabels[i];
    }
    
    totalVisited += result.visitedCount;
    totalExpanded += result.expandedCount;
    labeledCount += result.labeledCount;
    workerTimesMs.push(result.processingTimeMs);
  }
  
  const totalTimeMs = performance.now() - startTime;
  
  logResult('ParallelDijkstra', {
    timeMs: totalTimeMs.toFixed(1),
    visited: totalVisited,
    expanded: totalExpanded,
    labeled: `${labeledCount}/${seedIndices.length}`,
    workerTimes: `[${workerTimesMs.map(t => t.toFixed(0)).join(', ')}]ms`,
    speedup: `${(Math.max(...workerTimesMs) / (totalTimeMs || 1) * actualWorkerCount).toFixed(2)}x`
  });
  
  return {
    seedLabel,
    totalVisited,
    totalExpanded,
    labeledCount,
    totalTimeMs,
    workerTimesMs,
  };
}

/**
 * Check if Web Workers are available
 */
export function isWebWorkersAvailable(): boolean {
  return typeof Worker !== 'undefined';
}
