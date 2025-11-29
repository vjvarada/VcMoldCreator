/**
 * Volumetric Grid for Mold Volume Sampling
 * 
 * Creates a 3D grid discretization of the mold volume (silicone region)
 * between the outer shell (∂H) and part mesh (M).
 * 
 * Algorithm:
 * 1. Define bounding box covering the mold volume (outer shell bounds)
 * 2. Discretize into a regular 3D grid (configurable resolution)
 * 3. For each grid cell center x:
 *    - Use BVH/ray casting to check if x is inside the outer shell
 *    - Use BVH/ray casting to check if x is outside the part mesh
 * 4. Keep cells that are: inside ∂H AND outside M → silicone volume O
 * 
 * These grid cells serve as "volume nodes" for mold analysis.
 */

import * as THREE from 'three';
import { MeshBVH, acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh';

// Extend Three.js with BVH acceleration
THREE.Mesh.prototype.raycast = acceleratedRaycast;
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

// ============================================================================
// TYPES
// ============================================================================

export interface GridCell {
  /** Grid indices (i, j, k) */
  index: THREE.Vector3;
  /** World-space center position */
  center: THREE.Vector3;
  /** Cell size in each dimension */
  size: THREE.Vector3;
  /** Whether this cell is part of the mold volume */
  isMoldVolume: boolean;
  /** Signed distance to outer shell (negative = inside) */
  distanceToShell?: number;
  /** Signed distance to part mesh (positive = outside) */
  distanceToPart?: number;
}

export interface VolumetricGridResult {
  /** All grid cells (including non-mold volume for reference) */
  allCells: GridCell[];
  /** Only cells that are in the mold volume (inside shell, outside part) */
  moldVolumeCells: GridCell[];
  /** Grid resolution in each dimension */
  resolution: THREE.Vector3;
  /** Total cell count in grid */
  totalCellCount: number;
  /** Number of cells in mold volume */
  moldVolumeCellCount: number;
  /** Grid bounding box (from outer shell) */
  boundingBox: THREE.Box3;
  /** Cell size */
  cellSize: THREE.Vector3;
  /** Volume statistics */
  stats: VolumetricGridStats;
}

export interface VolumetricGridStats {
  /** Approximate mold volume (sum of cell volumes) */
  moldVolume: number;
  /** Total grid volume */
  totalVolume: number;
  /** Ratio of mold volume to total grid volume */
  fillRatio: number;
  /** Time taken to compute (ms) */
  computeTimeMs: number;
}

export interface VolumetricGridOptions {
  /** Grid resolution (cells per dimension, or specify per-axis) */
  resolution?: number | THREE.Vector3;
  /** Whether to store all cells or only mold volume cells */
  storeAllCells?: boolean;
  /** Margin to add around bounding box (percentage, default 0.05 = 5%) */
  marginPercent?: number;
  /** Whether to compute signed distances (more expensive but useful for visualization) */
  computeDistances?: boolean;
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default grid resolution (cells per dimension) */
export const DEFAULT_GRID_RESOLUTION = 64;

/** Ray direction for inside/outside testing (single ray is sufficient for watertight meshes) */
const RAY_DIRECTION = new THREE.Vector3(1, 0, 0);

// ============================================================================
// BVH HELPER CLASS
// ============================================================================

/**
 * Helper class that wraps a mesh with BVH acceleration for fast inside/outside queries
 */
class MeshInsideOutsideTester {
  private mesh: THREE.Mesh;
  private bvh: MeshBVH;
  private raycaster: THREE.Raycaster;
  private boundingBox: THREE.Box3;
  public readonly name: string;
  
  constructor(geometry: THREE.BufferGeometry, debugName: string = 'mesh') {
    this.name = debugName;
    
    // Ensure geometry is indexed for BVH
    let indexedGeometry = geometry;
    if (!geometry.index) {
      // Create indexed version
      const posAttr = geometry.getAttribute('position');
      const indices: number[] = [];
      for (let i = 0; i < posAttr.count; i++) {
        indices.push(i);
      }
      indexedGeometry = geometry.clone();
      indexedGeometry.setIndex(indices);
    }
    
    // Ensure normals and bounds are computed
    indexedGeometry.computeVertexNormals();
    indexedGeometry.computeBoundingBox();
    this.boundingBox = indexedGeometry.boundingBox!.clone();
    
    // Create a temporary mesh for raycasting
    const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(indexedGeometry, material);
    this.mesh.updateMatrixWorld(true); // Ensure matrix is updated
    
    // Build BVH for the geometry
    this.bvh = new MeshBVH(indexedGeometry, { maxLeafTris: 10 });
    indexedGeometry.boundsTree = this.bvh;
    
    // Set up raycaster with proper settings
    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = false; // We need all hits for inside/outside test
  }
  
  /**
   * Quick check if point is outside the bounding box
   */
  isOutsideBoundingBox(point: THREE.Vector3): boolean {
    return !this.boundingBox.containsPoint(point);
  }
  
  /**
   * Test if a point is inside the mesh using ray casting
   * 
   * Uses the crossing number algorithm: cast a ray and count intersections.
   * Odd number of crossings → inside.
   */
  isInside(point: THREE.Vector3): boolean {
    // Quick bounding box check
    if (this.isOutsideBoundingBox(point)) {
      return false;
    }
    
    // Cast single ray in +X direction
    this.raycaster.set(point, RAY_DIRECTION);
    this.raycaster.far = Infinity;
    const intersects = this.raycaster.intersectObject(this.mesh, false);
    
    // Odd number of intersections → inside
    return intersects.length % 2 === 1;
  }
  
  /**
   * Test if a point is outside the mesh (inverse of isInside)
   */
  isOutside(point: THREE.Vector3): boolean {
    return !this.isInside(point);
  }
  
  /**
   * Get approximate signed distance to mesh surface
   * Positive = outside, Negative = inside
   * 
   * Uses closest point query with BVH for efficiency
   */
  getSignedDistance(point: THREE.Vector3): number {
    const closestPoint = new THREE.Vector3();
    const target = { 
      point: closestPoint, 
      distance: Infinity,
      faceIndex: 0
    };
    
    // Use BVH to find closest point on surface
    this.bvh.closestPointToPoint(point, target);
    
    const distance = target.distance;
    const isInside = this.isInside(point);
    
    return isInside ? -distance : distance;
  }
  
  /**
   * Debug: test a single point and log results
   */
  debugPoint(point: THREE.Vector3): { insideVotes: number; intersectionCounts: number[] } {
    const intersectionCounts: number[] = [];
    let insideVotes = 0;
    
    this.raycaster.set(point, RAY_DIRECTION);
    this.raycaster.far = Infinity;
    const intersects = this.raycaster.intersectObject(this.mesh, false);
    intersectionCounts.push(intersects.length);
    
    if (intersects.length % 2 === 1) {
      insideVotes++;
    }
    
    return { insideVotes, intersectionCounts };
  }
  
  /**
   * Clean up BVH and resources
   */
  dispose(): void {
    if (this.mesh.geometry.boundsTree) {
      this.mesh.geometry.disposeBoundsTree();
    }
    (this.mesh.material as THREE.Material).dispose();
  }
}

// ============================================================================
// GRID GENERATION
// ============================================================================

/**
 * Generate a volumetric grid sampling the mold volume
 * 
 * The mold volume is defined as the region that is:
 * - INSIDE the outer shell (∂H)
 * - OUTSIDE the part mesh (M)
 * 
 * @param outerShellGeometry - Geometry of the inflated outer shell (∂H)
 * @param partGeometry - Geometry of the original part mesh (M)
 * @param options - Grid generation options
 * @returns VolumetricGridResult containing the grid cells
 */
export function generateVolumetricGrid(
  outerShellGeometry: THREE.BufferGeometry,
  partGeometry: THREE.BufferGeometry,
  options: VolumetricGridOptions = {}
): VolumetricGridResult {
  const startTime = performance.now();
  
  // Parse options
  const resolution = options.resolution ?? DEFAULT_GRID_RESOLUTION;
  const storeAllCells = options.storeAllCells ?? false;
  const marginPercent = options.marginPercent ?? 0.05;
  const computeDistances = options.computeDistances ?? false;
  
  // Determine resolution per axis
  let resX: number, resY: number, resZ: number;
  if (typeof resolution === 'number') {
    resX = resY = resZ = resolution;
  } else {
    resX = resolution.x;
    resY = resolution.y;
    resZ = resolution.z;
  }
  
  // Clone geometries to avoid modifying originals
  const shellGeom = outerShellGeometry.clone();
  const partGeom = partGeometry.clone();
  
  // Ensure geometries have computed bounds
  shellGeom.computeBoundingBox();
  partGeom.computeBoundingBox();
  
  // Get bounding box from outer shell (this defines the grid extent)
  const boundingBox = shellGeom.boundingBox!.clone();
  
  // Add margin to bounding box
  const boxSize = new THREE.Vector3();
  boundingBox.getSize(boxSize);
  const margin = boxSize.clone().multiplyScalar(marginPercent);
  boundingBox.min.sub(margin);
  boundingBox.max.add(margin);
  
  // Recompute size after margin
  boundingBox.getSize(boxSize);
  
  // Calculate cell size
  const cellSize = new THREE.Vector3(
    boxSize.x / resX,
    boxSize.y / resY,
    boxSize.z / resZ
  );
  
  // Create BVH testers for both meshes
  const shellTester = new MeshInsideOutsideTester(shellGeom, 'shell');
  const partTester = new MeshInsideOutsideTester(partGeom, 'part');
  
  // Initialize result arrays
  const allCells: GridCell[] = [];
  const moldVolumeCells: GridCell[] = [];
  
  const totalCellCount = resX * resY * resZ;
  const cellCenter = new THREE.Vector3();
  const cellIndex = new THREE.Vector3();

  // Iterate through all grid cells
  for (let k = 0; k < resZ; k++) {
    for (let j = 0; j < resY; j++) {
      for (let i = 0; i < resX; i++) {
        // Calculate cell center position
        cellCenter.set(
          boundingBox.min.x + (i + 0.5) * cellSize.x,
          boundingBox.min.y + (j + 0.5) * cellSize.y,
          boundingBox.min.z + (k + 0.5) * cellSize.z
        );
        
        // Test if point is inside shell first (early exit if outside)
        const isInsideShell = shellTester.isInside(cellCenter);
        
        // Only test part if inside shell (optimization)
        const isOutsidePart = isInsideShell ? partTester.isOutside(cellCenter) : true;
        const isMoldVolume = isInsideShell && isOutsidePart;
        
        // Only create cell objects when needed
        if (isMoldVolume || storeAllCells) {
          cellIndex.set(i, j, k);
          
          const cell: GridCell = {
            index: cellIndex.clone(),
            center: cellCenter.clone(),
            size: cellSize.clone(),
            isMoldVolume,
          };
          
          // Optionally compute signed distances
          if (computeDistances) {
            cell.distanceToShell = shellTester.getSignedDistance(cellCenter);
            cell.distanceToPart = partTester.getSignedDistance(cellCenter);
          }
          
          if (isMoldVolume) {
            moldVolumeCells.push(cell);
          }
          
          if (storeAllCells) {
            allCells.push(cell);
          }
        }
      }
    }
  }
  
  // Clean up BVH resources
  shellTester.dispose();
  partTester.dispose();
  shellGeom.dispose();
  partGeom.dispose();
  
  const endTime = performance.now();
  
  // Calculate statistics
  const cellVolume = cellSize.x * cellSize.y * cellSize.z;
  const moldVolume = moldVolumeCells.length * cellVolume;
  const totalVolume = totalCellCount * cellVolume;
  
  const stats: VolumetricGridStats = {
    moldVolume,
    totalVolume,
    fillRatio: moldVolume / totalVolume,
    computeTimeMs: endTime - startTime,
  };
  
  return {
    allCells: storeAllCells ? allCells : [],
    moldVolumeCells,
    resolution: new THREE.Vector3(resX, resY, resZ),
    totalCellCount,
    moldVolumeCellCount: moldVolumeCells.length,
    boundingBox,
    cellSize,
    stats,
  };
}

// ============================================================================
// VISUALIZATION HELPERS
// ============================================================================

/**
 * Create a point cloud visualization of the mold volume cells
 * 
 * @param gridResult - The volumetric grid result
 * @param color - Color for the points (default: cyan)
 * @param pointSize - Size of each point (default: 2)
 * @returns THREE.Points object for adding to scene
 */
export function createMoldVolumePointCloud(
  gridResult: VolumetricGridResult,
  color: THREE.ColorRepresentation = 0x00ffff,
  pointSize: number = 2
): THREE.Points {
  const positions = new Float32Array(gridResult.moldVolumeCells.length * 3);
  
  for (let i = 0; i < gridResult.moldVolumeCells.length; i++) {
    const cell = gridResult.moldVolumeCells[i];
    positions[i * 3] = cell.center.x;
    positions[i * 3 + 1] = cell.center.y;
    positions[i * 3 + 2] = cell.center.z;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  
  const material = new THREE.PointsMaterial({
    color,
    size: pointSize,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.6,
  });
  
  return new THREE.Points(geometry, material);
}

/**
 * Create a voxel box visualization of the mold volume cells
 * Uses instanced mesh for efficient rendering of many boxes
 * 
 * @param gridResult - The volumetric grid result
 * @param color - Color for the boxes (default: cyan)
 * @param opacity - Opacity of boxes (default: 0.3)
 * @returns THREE.InstancedMesh for adding to scene
 */
export function createMoldVolumeVoxels(
  gridResult: VolumetricGridResult,
  color: THREE.ColorRepresentation = 0x00ffff,
  opacity: number = 0.3
): THREE.InstancedMesh {
  const cellCount = gridResult.moldVolumeCells.length;
  const { cellSize } = gridResult;
  
  // Create box geometry for a single cell
  const boxGeometry = new THREE.BoxGeometry(cellSize.x, cellSize.y, cellSize.z);
  
  const material = new THREE.MeshPhongMaterial({
    color,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
  });
  
  // Create instanced mesh
  const instancedMesh = new THREE.InstancedMesh(boxGeometry, material, cellCount);
  
  // Set transforms for each instance
  const matrix = new THREE.Matrix4();
  
  for (let i = 0; i < cellCount; i++) {
    const cell = gridResult.moldVolumeCells[i];
    matrix.setPosition(cell.center);
    instancedMesh.setMatrixAt(i, matrix);
  }
  
  instancedMesh.instanceMatrix.needsUpdate = true;
  
  return instancedMesh;
}

/**
 * Create a wireframe box showing the grid bounding box
 * 
 * @param gridResult - The volumetric grid result
 * @param color - Color for the wireframe (default: white)
 * @returns THREE.LineSegments for adding to scene
 */
export function createGridBoundingBoxHelper(
  gridResult: VolumetricGridResult,
  color: THREE.ColorRepresentation = 0xffffff
): THREE.LineSegments {
  const { boundingBox } = gridResult;
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  
  boundingBox.getSize(size);
  boundingBox.getCenter(center);
  
  const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
  const edges = new THREE.EdgesGeometry(geometry);
  const material = new THREE.LineBasicMaterial({ color });
  
  const wireframe = new THREE.LineSegments(edges, material);
  wireframe.position.copy(center);
  
  return wireframe;
}

/**
 * Create a helper to visualize the grid structure (wireframe grid lines)
 * Note: Only creates lines on the boundary for performance
 * 
 * @param gridResult - The volumetric grid result
 * @param color - Color for the grid lines (default: gray)
 * @returns THREE.LineSegments for adding to scene
 */
export function createGridLinesHelper(
  gridResult: VolumetricGridResult,
  color: THREE.ColorRepresentation = 0x666666
): THREE.LineSegments {
  const { boundingBox, resolution, cellSize } = gridResult;
  const points: THREE.Vector3[] = [];
  
  // X-aligned lines (on YZ boundaries)
  for (let j = 0; j <= resolution.y; j++) {
    for (let k = 0; k <= resolution.z; k++) {
      const y = boundingBox.min.y + j * cellSize.y;
      const z = boundingBox.min.z + k * cellSize.z;
      points.push(new THREE.Vector3(boundingBox.min.x, y, z));
      points.push(new THREE.Vector3(boundingBox.max.x, y, z));
    }
  }
  
  // Y-aligned lines (on XZ boundaries)
  for (let i = 0; i <= resolution.x; i++) {
    for (let k = 0; k <= resolution.z; k++) {
      const x = boundingBox.min.x + i * cellSize.x;
      const z = boundingBox.min.z + k * cellSize.z;
      points.push(new THREE.Vector3(x, boundingBox.min.y, z));
      points.push(new THREE.Vector3(x, boundingBox.max.y, z));
    }
  }
  
  // Z-aligned lines (on XY boundaries)
  for (let i = 0; i <= resolution.x; i++) {
    for (let j = 0; j <= resolution.y; j++) {
      const x = boundingBox.min.x + i * cellSize.x;
      const y = boundingBox.min.y + j * cellSize.y;
      points.push(new THREE.Vector3(x, y, boundingBox.min.z));
      points.push(new THREE.Vector3(x, y, boundingBox.max.z));
    }
  }
  
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.3 });
  
  return new THREE.LineSegments(geometry, material);
}

// ============================================================================
// CLEANUP UTILITIES
// ============================================================================

/**
 * Dispose of grid visualization objects
 */
export function disposeGridVisualization(
  object: THREE.Points | THREE.InstancedMesh | THREE.LineSegments
): void {
  if (object.geometry) {
    object.geometry.dispose();
  }
  if (object.material) {
    if (Array.isArray(object.material)) {
      object.material.forEach(m => m.dispose());
    } else {
      object.material.dispose();
    }
  }
}

/**
 * Remove grid visualization from scene and dispose resources
 */
export function removeGridVisualization(
  scene: THREE.Scene,
  object: THREE.Points | THREE.InstancedMesh | THREE.LineSegments | null
): void {
  if (!object) return;
  
  if (object.parent === scene) {
    scene.remove(object);
  }
  disposeGridVisualization(object);
}

// ============================================================================
// PARALLEL GRID GENERATION (WEB WORKERS)
// ============================================================================

/** Number of Web Workers to use for parallel computation */
const DEFAULT_NUM_WORKERS = navigator.hardwareConcurrency || 4;

/** Worker message types */
interface WorkerInitMessage {
  type: 'init';
  workerId: number;
  shellPositionArray: Float32Array;
  shellIndexArray: Uint32Array | null;
  partPositionArray: Float32Array;
  partIndexArray: Uint32Array | null;
}

interface WorkerComputeMessage {
  type: 'compute';
  cellCenters: Float32Array;
  cellIndices: Uint32Array;
}

interface WorkerResultMessage {
  type: 'result';
  workerId: number;
  moldVolumeCellIndices: Uint32Array;
}

interface WorkerReadyMessage {
  type: 'ready';
  workerId: number;
}

type WorkerMessage = WorkerInitMessage | WorkerComputeMessage | WorkerResultMessage | WorkerReadyMessage;

/**
 * Extract geometry data for transfer to workers
 */
function extractGeometryData(geometry: THREE.BufferGeometry): {
  positionArray: Float32Array;
  indexArray: Uint32Array | null;
} {
  const posAttr = geometry.getAttribute('position');
  const positionArray = new Float32Array(posAttr.array);
  
  let indexArray: Uint32Array | null = null;
  if (geometry.index) {
    indexArray = new Uint32Array(geometry.index.array);
  }
  
  return { positionArray, indexArray };
}

/**
 * Generate a volumetric grid using parallel Web Workers
 * 
 * This is significantly faster than the single-threaded version for large grids.
 * 
 * @param outerShellGeometry - Geometry of the inflated outer shell (∂H)
 * @param partGeometry - Geometry of the original part mesh (M)
 * @param options - Grid generation options
 * @param numWorkers - Number of Web Workers to use (default: CPU core count)
 * @returns Promise<VolumetricGridResult> containing the grid cells
 */
export async function generateVolumetricGridParallel(
  outerShellGeometry: THREE.BufferGeometry,
  partGeometry: THREE.BufferGeometry,
  options: VolumetricGridOptions = {},
  numWorkers: number = DEFAULT_NUM_WORKERS
): Promise<VolumetricGridResult> {
  const startTime = performance.now();
  
  // Parse options
  const resolution = options.resolution ?? DEFAULT_GRID_RESOLUTION;
  const marginPercent = options.marginPercent ?? 0.05;
  
  // Determine resolution per axis
  let resX: number, resY: number, resZ: number;
  if (typeof resolution === 'number') {
    resX = resY = resZ = resolution;
  } else {
    resX = resolution.x;
    resY = resolution.y;
    resZ = resolution.z;
  }
  
  // Clone geometries to avoid modifying originals
  const shellGeom = outerShellGeometry.clone();
  const partGeom = partGeometry.clone();
  
  // Ensure geometries have computed bounds
  shellGeom.computeBoundingBox();
  partGeom.computeBoundingBox();
  
  // Get bounding box from outer shell
  const boundingBox = shellGeom.boundingBox!.clone();
  
  // Add margin to bounding box
  const boxSize = new THREE.Vector3();
  boundingBox.getSize(boxSize);
  const margin = boxSize.clone().multiplyScalar(marginPercent);
  boundingBox.min.sub(margin);
  boundingBox.max.add(margin);
  
  // Recompute size after margin
  boundingBox.getSize(boxSize);
  
  // Calculate cell size
  const cellSize = new THREE.Vector3(
    boxSize.x / resX,
    boxSize.y / resY,
    boxSize.z / resZ
  );
  
  const totalCellCount = resX * resY * resZ;
  
  // Extract geometry data for workers
  const shellData = extractGeometryData(shellGeom);
  const partData = extractGeometryData(partGeom);
  
  // Create and initialize workers
  const workers: Worker[] = [];
  const workerReadyPromises: Promise<void>[] = [];
  
  console.log(`Initializing ${numWorkers} grid workers...`);
  
  for (let i = 0; i < numWorkers; i++) {
    const worker = new Worker(
      new URL('./volumetricGridWorker.ts', import.meta.url),
      { type: 'module' }
    );
    workers.push(worker);
    
    workerReadyPromises.push(new Promise<void>((resolve) => {
      const handler = (event: MessageEvent<WorkerMessage>) => {
        if (event.data.type === 'ready') {
          worker.removeEventListener('message', handler);
          resolve();
        }
      };
      worker.addEventListener('message', handler);
    }));
    
    // Send init message with geometry data (use slices to allow transfer)
    const initMsg: WorkerInitMessage = {
      type: 'init',
      workerId: i,
      shellPositionArray: shellData.positionArray.slice(),
      shellIndexArray: shellData.indexArray?.slice() ?? null,
      partPositionArray: partData.positionArray.slice(),
      partIndexArray: partData.indexArray?.slice() ?? null,
    };
    
    const transfers: Transferable[] = [initMsg.shellPositionArray.buffer, initMsg.partPositionArray.buffer];
    if (initMsg.shellIndexArray) transfers.push(initMsg.shellIndexArray.buffer);
    if (initMsg.partIndexArray) transfers.push(initMsg.partIndexArray.buffer);
    
    worker.postMessage(initMsg, transfers);
  }
  
  await Promise.all(workerReadyPromises);
  console.log('Grid workers initialized');
  
  // Generate all cell centers and indices
  const allCellCenters = new Float32Array(totalCellCount * 3);
  const allCellIndices = new Uint32Array(totalCellCount);
  
  let idx = 0;
  for (let k = 0; k < resZ; k++) {
    for (let j = 0; j < resY; j++) {
      for (let i = 0; i < resX; i++) {
        const cellIdx = i * resY * resZ + j * resZ + k;
        allCellIndices[idx] = cellIdx;
        
        const i3 = idx * 3;
        allCellCenters[i3] = boundingBox.min.x + (i + 0.5) * cellSize.x;
        allCellCenters[i3 + 1] = boundingBox.min.y + (j + 0.5) * cellSize.y;
        allCellCenters[i3 + 2] = boundingBox.min.z + (k + 0.5) * cellSize.z;
        
        idx++;
      }
    }
  }
  
  // Distribute work across workers
  const cellsPerWorker = Math.ceil(totalCellCount / numWorkers);
  const workerPromises: Promise<Uint32Array>[] = [];
  
  for (let w = 0; w < numWorkers; w++) {
    const startIdx = w * cellsPerWorker;
    const endIdx = Math.min(startIdx + cellsPerWorker, totalCellCount);
    if (endIdx <= startIdx) continue;
    
    const cellCenters = allCellCenters.slice(startIdx * 3, endIdx * 3);
    const cellIndices = allCellIndices.slice(startIdx, endIdx);
    
    workerPromises.push(new Promise<Uint32Array>((resolve) => {
      const handler = (event: MessageEvent<WorkerMessage>) => {
        if (event.data.type === 'result' && event.data.workerId === w) {
          workers[w].removeEventListener('message', handler);
          resolve((event.data as WorkerResultMessage).moldVolumeCellIndices);
        }
      };
      workers[w].addEventListener('message', handler);
    }));
    
    const computeMsg: WorkerComputeMessage = {
      type: 'compute',
      cellCenters,
      cellIndices,
    };
    
    workers[w].postMessage(computeMsg, [cellCenters.buffer, cellIndices.buffer]);
  }
  
  // Wait for all workers to complete
  const allResults = await Promise.all(workerPromises);
  
  // Terminate workers
  workers.forEach(w => w.terminate());
  
  // Collect mold volume cell indices
  const moldVolumeCellIndexSet = new Set<number>();
  for (const result of allResults) {
    for (let i = 0; i < result.length; i++) {
      moldVolumeCellIndexSet.add(result[i]);
    }
  }
  
  // Create GridCell objects for mold volume cells
  const moldVolumeCells: GridCell[] = [];
  
  for (const cellIdx of moldVolumeCellIndexSet) {
    // Decode cell index back to i, j, k
    const i = Math.floor(cellIdx / (resY * resZ));
    const remainder = cellIdx % (resY * resZ);
    const j = Math.floor(remainder / resZ);
    const k = remainder % resZ;
    
    const center = new THREE.Vector3(
      boundingBox.min.x + (i + 0.5) * cellSize.x,
      boundingBox.min.y + (j + 0.5) * cellSize.y,
      boundingBox.min.z + (k + 0.5) * cellSize.z
    );
    
    moldVolumeCells.push({
      index: new THREE.Vector3(i, j, k),
      center,
      size: cellSize.clone(),
      isMoldVolume: true,
    });
  }
  
  // Clean up
  shellGeom.dispose();
  partGeom.dispose();
  
  const endTime = performance.now();
  
  // Calculate statistics
  const cellVolume = cellSize.x * cellSize.y * cellSize.z;
  const moldVolume = moldVolumeCells.length * cellVolume;
  const totalVolume = totalCellCount * cellVolume;
  
  const stats: VolumetricGridStats = {
    moldVolume,
    totalVolume,
    fillRatio: moldVolume / totalVolume,
    computeTimeMs: endTime - startTime,
  };
  
  return {
    allCells: [],
    moldVolumeCells,
    resolution: new THREE.Vector3(resX, resY, resZ),
    totalCellCount,
    moldVolumeCellCount: moldVolumeCells.length,
    boundingBox,
    cellSize,
    stats,
  };
}
