/**
 * Parallel Dijkstra Coordinator
 * 
 * Manages multiple Web Workers to parallelize independent Dijkstra searches
 * from seed voxels to boundary voxels.
 */

import type { VolumetricGridResult } from './volumetricGrid';

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
  
  for (let i = 0; i < gridResult.moldVolumeCells.length; i++) {
    const cell = gridResult.moldVolumeCells[i];
    const key = `${Math.round(cell.index.x)},${Math.round(cell.index.y)},${Math.round(cell.index.z)}`;
    indexMap.set(key, i);
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
  
  console.log(`[ParallelDijkstra] Starting with ${numWorkers} workers for ${seedIndices.length} seeds`);
  
  // Prepare shared data
  const spatialIndex = buildVoxelSpatialIndex(gridResult);
  const { keys: spatialIndexKeys, values: spatialIndexValues } = serializeSpatialIndex(spatialIndex);
  const gridIndices = buildGridIndicesArray(gridResult);
  const weightingFactor = gridResult.weightingFactor!;
  
  // Split seeds among workers
  const seedsPerWorker = Math.ceil(seedIndices.length / numWorkers);
  const workerSeedBatches: number[][] = [];
  
  for (let w = 0; w < numWorkers; w++) {
    const start = w * seedsPerWorker;
    const end = Math.min(start + seedsPerWorker, seedIndices.length);
    if (start < end) {
      workerSeedBatches.push(seedIndices.slice(start, end));
    }
  }
  
  const actualWorkerCount = workerSeedBatches.length;
  console.log(`[ParallelDijkstra] Using ${actualWorkerCount} workers, ~${seedsPerWorker} seeds each`);
  
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
  
  console.log(`[ParallelDijkstra] Completed in ${totalTimeMs.toFixed(1)}ms`);
  console.log(`  Total visited: ${totalVisited}, Total expanded: ${totalExpanded}`);
  console.log(`  Seeds labeled: ${labeledCount} / ${seedIndices.length}`);
  console.log(`  Worker times: [${workerTimesMs.map(t => t.toFixed(0)).join(', ')}]ms`);
  console.log(`  Speedup estimate: ${(Math.max(...workerTimesMs) / (totalTimeMs || 1) * actualWorkerCount).toFixed(2)}x`);
  
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
