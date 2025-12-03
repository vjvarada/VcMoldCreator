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
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh';
import { MeshTester, logInfo, logDebug, logResult, logTiming } from './meshUtils';

// Extend Three.js with BVH acceleration
THREE.Mesh.prototype.raycast = acceleratedRaycast;
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

// ============================================================================
// ASYNC YIELD UTILITIES
// ============================================================================

/**
 * Yield to the event loop to prevent UI blocking during heavy computation.
 * Uses setTimeout(0) which is more reliable than requestAnimationFrame for background work.
 */
const yieldToEventLoop = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0));

/**
 * Batch size for yielding during heavy computation loops.
 * Yield every N iterations to keep UI responsive.
 */
const YIELD_BATCH_SIZE = 500;

// ============================================================================
// TYPES
// ============================================================================

/** @deprecated Use flat typed arrays instead (voxelCenters, voxelIndices) */
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
  /** @deprecated Use voxelCenters instead - kept for backward compatibility */
  allCells: GridCell[];
  /** @deprecated Use voxelCenters and voxelIndices instead - kept for backward compatibility */
  moldVolumeCells: GridCell[];
  
  // ========== NEW FLAT TYPED ARRAYS (memory efficient) ==========
  /** 
   * Flat array of voxel center positions [x0,y0,z0, x1,y1,z1, ...]
   * Length = moldVolumeCellCount * 3
   */
  voxelCenters: Float32Array | null;
  /** 
   * Flat array of voxel grid indices [i0,j0,k0, i1,j1,k1, ...]
   * Length = moldVolumeCellCount * 3
   */
  voxelIndices: Uint32Array | null;
  
  /** Grid resolution in each dimension */
  resolution: THREE.Vector3;
  /** Total cell count in grid */
  totalCellCount: number;
  /** Number of cells in mold volume */
  moldVolumeCellCount: number;
  /** Grid bounding box (from outer shell) */
  boundingBox: THREE.Box3;
  /** Cell size (uniform for all voxels) */
  cellSize: THREE.Vector3;
  /** Volume statistics */
  stats: VolumetricGridStats;
  /** Distance field δ_i: distance from each mold volume voxel center to part mesh M */
  voxelDist: Float32Array | null;
  /** Distance field δ_w: distance from each mold volume voxel center to shell boundary ∂H */
  voxelDistToShell: Float32Array | null;
  /** Biased distance field: δ_i + λ_w where λ_w = R - δ_w */
  biasedDist: Float32Array | null;
  /** Weighting factor: wt = 1/[(biasedDist^2) + 0.25] */
  weightingFactor: Float32Array | null;
  /** Min/max distance values for δ_i (part distance) */
  distanceStats: { min: number; max: number } | null;
  /** Min/max distance values for δ_w (shell distance) */
  shellDistanceStats: { min: number; max: number } | null;
  /** Min/max distance values for biased distance field */
  biasedDistanceStats: { min: number; max: number } | null;
  /** Min/max values for weighting factor */
  weightingFactorStats: { min: number; max: number } | null;
  /** R: Maximum distance from any voxel to part mesh M */
  R: number;
  /** Visualization line for R: start point (on shell boundary) */
  rLineStart: THREE.Vector3 | null;
  /** Visualization line for R: end point (on part mesh) */
  rLineEnd: THREE.Vector3 | null;
  /** Mask indicating which voxels have biased weighting applied (boundary + one level deep) */
  boundaryAdjacentMask: Uint8Array | null;
  /** 
   * Mask indicating which voxels are seed voxels (at part mesh vertices).
   * Only populated when using generateVolumetricGridWithVertexSeeds.
   * Length = moldVolumeCellCount
   */
  seedVoxelMask: Uint8Array | null;
}

/**
 * Helper to get voxel center at index from flat array
 */
export function getVoxelCenter(gridResult: VolumetricGridResult, voxelIdx: number, target?: THREE.Vector3): THREE.Vector3 {
  const result = target ?? new THREE.Vector3();
  if (gridResult.voxelCenters) {
    const i3 = voxelIdx * 3;
    result.set(
      gridResult.voxelCenters[i3],
      gridResult.voxelCenters[i3 + 1],
      gridResult.voxelCenters[i3 + 2]
    );
  } else if (gridResult.moldVolumeCells[voxelIdx]) {
    result.copy(gridResult.moldVolumeCells[voxelIdx].center);
  }
  return result;
}

/**
 * Helper to get voxel grid index at index from flat array
 */
export function getVoxelIndex(gridResult: VolumetricGridResult, voxelIdx: number, target?: THREE.Vector3): THREE.Vector3 {
  const result = target ?? new THREE.Vector3();
  if (gridResult.voxelIndices) {
    const i3 = voxelIdx * 3;
    result.set(
      gridResult.voxelIndices[i3],
      gridResult.voxelIndices[i3 + 1],
      gridResult.voxelIndices[i3 + 2]
    );
  } else if (gridResult.moldVolumeCells[voxelIdx]) {
    result.copy(gridResult.moldVolumeCells[voxelIdx].index);
  }
  return result;
}

/**
 * Get voxel center coordinates directly from flat array (no Vector3 allocation)
 * @returns [cx, cy, cz] tuple
 */
export function getVoxelCenterXYZ(gridResult: VolumetricGridResult, voxelIdx: number): [number, number, number] {
  if (gridResult.voxelCenters) {
    const i3 = voxelIdx * 3;
    return [
      gridResult.voxelCenters[i3],
      gridResult.voxelCenters[i3 + 1],
      gridResult.voxelCenters[i3 + 2]
    ];
  }
  const cell = gridResult.moldVolumeCells[voxelIdx];
  return [cell.center.x, cell.center.y, cell.center.z];
}

/**
 * Get voxel grid indices directly from flat array (no Vector3 allocation)
 * @returns [i, j, k] tuple
 */
export function getVoxelGridIndices(gridResult: VolumetricGridResult, voxelIdx: number): [number, number, number] {
  if (gridResult.voxelIndices) {
    const i3 = voxelIdx * 3;
    return [
      gridResult.voxelIndices[i3],
      gridResult.voxelIndices[i3 + 1],
      gridResult.voxelIndices[i3 + 2]
    ];
  }
  const cell = gridResult.moldVolumeCells[voxelIdx];
  return [Math.round(cell.index.x), Math.round(cell.index.y), Math.round(cell.index.z)];
}

/**
 * Build a spatial index map for voxel grid indices -> voxel array index
 * Uses flat arrays directly when available
 */
export function buildVoxelSpatialIndex(gridResult: VolumetricGridResult): Map<string, number> {
  const voxelCount = gridResult.voxelCenters ? gridResult.voxelCenters.length / 3 : gridResult.moldVolumeCells.length;
  const spatialIndex = new Map<string, number>();
  
  if (gridResult.voxelIndices) {
    for (let idx = 0; idx < voxelCount; idx++) {
      const i3 = idx * 3;
      const gi = gridResult.voxelIndices[i3];
      const gj = gridResult.voxelIndices[i3 + 1];
      const gk = gridResult.voxelIndices[i3 + 2];
      spatialIndex.set(`${gi},${gj},${gk}`, idx);
    }
  } else {
    for (let idx = 0; idx < voxelCount; idx++) {
      const cell = gridResult.moldVolumeCells[idx];
      const gi = Math.round(cell.index.x);
      const gj = Math.round(cell.index.y);
      const gk = Math.round(cell.index.z);
      spatialIndex.set(`${gi},${gj},${gk}`, idx);
    }
  }
  
  return spatialIndex;
}

/**
 * Get voxel count from grid result (works with flat arrays or legacy)
 */
export function getVoxelCount(gridResult: VolumetricGridResult): number {
  return gridResult.voxelCenters ? gridResult.voxelCenters.length / 3 : gridResult.moldVolumeCells.length;
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
  /** Grid resolution (cells per dimension, or specify per-axis). If not provided, auto-calculated from voxelsPerUnit or targetVoxelSizePercent. */
  resolution?: number | THREE.Vector3;
  /**
   * Voxel density in voxels per unit length (assuming STL is in mm, this is voxels/mm).
   * Default: 1.0 (1 voxel per mm).
   * Higher values = finer grid, better thin wall detection but slower.
   * This takes precedence over targetVoxelSizePercent when both are specified.
   */
  voxelsPerUnit?: number;
  /** 
   * Target voxel size as percentage of bounding box diagonal (default: 2% = 0.02).
   * Used to auto-calculate resolution when resolution and voxelsPerUnit are not provided.
   * Smaller values = finer grid, better thin wall detection but slower.
   * @deprecated Use voxelsPerUnit instead for more intuitive control.
   */
  targetVoxelSizePercent?: number;
  /** Whether to store all cells or only mold volume cells */
  storeAllCells?: boolean;
  /** Margin to add around bounding box (percentage, default 0.05 = 5%) */
  marginPercent?: number;
  /** Whether to compute signed distances (more expensive but useful for visualization) */
  computeDistances?: boolean;
  /** Whether to use GPU acceleration (WebGPU if available, otherwise falls back to CPU) */
  useGPU?: boolean;
  /** 
   * Whether to include voxels that touch/intersect the boundary surfaces.
   * When true, includes voxels whose centers are within half a cell size of:
   * - The outer shell boundary (∂H) - expands outward
   * - The inner part surface (M) - expands inward toward the part
   * Default: false (only strictly contained voxels)
   */
  includeSurfaceVoxels?: boolean;
  /**
   * Maximum volume overlap ratio with the part mesh for a voxel to be considered a boundary voxel.
   * If a voxel overlaps more than this ratio with the part, it's excluded (inside the part).
   * Default: 0.10 (10%) - voxels with >10% overlap are considered inside the part.
   * This helps prevent thin wall bridging issues.
   */
  maxPartOverlapRatio?: number;
  /**
   * Minimum volume overlap ratio with the part mesh for a voxel to be considered a surface voxel.
   * Voxels with overlap between minPartOverlapRatio and maxPartOverlapRatio are surface voxels.
   * Default: 0.01 (1%)
   */
  minPartOverlapRatio?: number;
  /**
   * Distance threshold as fraction of voxel size for surface voxel detection.
   * Voxels within this distance of the part surface are considered surface voxels.
   * Default: 0.5 (50% of voxel size)
   */
  surfaceDistanceThreshold?: number;
}

export interface BiasedDistanceWeights {
  /** Weight for δ_i (distance to part) term. Default: 1.0 */
  partDistanceWeight: number;
  /** Weight for (R - δ_w) (shell bias) term. Default: 1.0 */
  shellBiasWeight: number;
}

/**
 * Options for adaptive voxelization with distance-based LOD
 */
export interface AdaptiveVoxelizationOptions extends VolumetricGridOptions {
  /**
   * Base voxel density at the part surface (voxels per mm).
   * This is the finest resolution, used for seed voxels at vertices/edges.
   * Default: Uses voxelsPerUnit from parent options or DEFAULT_VOXELS_PER_UNIT
   */
  baseVoxelsPerUnit?: number;
  
  /**
   * Number of levels of detail (LOD).
   * Each level multiplies the voxel size by sizeFactor.
   * Default: 3 (with sizeFactor=1.5, gives ~2.25x voxel size at coarsest level)
   * Use 4 for finer gradation, 2 for faster processing
   */
  lodLevels?: number;
  
  /**
   * Distance thresholds for each LOD level (in mm from part surface).
   * The array length should be lodLevels - 1.
   * If not provided, thresholds are calculated based on the bounding box diagonal.
   * Example: [2, 5, 10] means:
   *   - LOD 0 (finest): 0-2mm from part
   *   - LOD 1: 2-5mm from part
   *   - LOD 2: 5-10mm from part
   *   - LOD 3 (coarsest): >10mm from part
   */
  lodDistanceThresholds?: number[];
  
  /**
   * Size growth factor per LOD level.
   * Each successive LOD multiplies voxel size by this factor.
   * - Default: 1.5 (gentler scaling, better for Dijkstra continuity)
   * - Use 2.0 for aggressive octree-style halving (more memory savings)
   */
  sizeFactor?: number;
  
  /**
   * Distance scaling mode for automatic threshold calculation.
   * - 'linear': threshold[n] = baseThreshold * (n+1) × distanceScale
   *   Better for Dijkstra - more uniform voxel size transitions
   * - 'quadratic': threshold[n] = baseThreshold * (n+1)² × distanceScale  
   *   More aggressive memory savings, but larger jumps in voxel size
   * Default: 'linear'
   */
  distanceScalingMode?: 'linear' | 'quadratic';
  
  /**
   * Distance scale factor for automatic threshold calculation.
   * Multiplied by the base threshold (2% of bbox diagonal) to set LOD boundaries.
   * Default: 1.0
   */
  distanceScale?: number;
  
  /**
   * Minimum distance from part surface for a mold volume voxel to be included.
   * Default: 0 (include voxels that touch the surface)
   */
  minDistanceFromPart?: number;
}

/**
 * Result for adaptive voxelization, extends the base result with LOD info
 */
export interface AdaptiveVoxelGridResult extends VolumetricGridResult {
  /**
   * LOD level for each voxel (0 = finest, higher = coarser).
   * Length = moldVolumeCellCount
   */
  voxelLOD: Uint8Array | null;
  
  /**
   * Size (in mm) for each voxel. Since voxels have different sizes in adaptive mode,
   * this stores the actual size for each voxel.
   * Length = moldVolumeCellCount
   */
  voxelSizes: Float32Array | null;
  
  /**
   * LOD level statistics
   */
  lodStats: {
    /** Number of voxels at each LOD level */
    voxelCountPerLevel: number[];
    /** Voxel size at each LOD level */
    voxelSizePerLevel: number[];
    /** Distance threshold for each LOD transition */
    distanceThresholds: number[];
  } | null;
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default grid resolution (cells per dimension) - used as fallback */
export const DEFAULT_GRID_RESOLUTION = 64;

/** Default voxel density in voxels per unit (assuming mm units, 10 voxels/mm = 0.1mm voxel size) */
export const DEFAULT_VOXELS_PER_UNIT = 10.0;

/** Default target voxel size as percentage of bounding box diagonal (2%) */
export const DEFAULT_TARGET_VOXEL_SIZE_PERCENT = 0.02;

/** Default maximum overlap ratio with part mesh for boundary voxels (10%) */
export const DEFAULT_MAX_PART_OVERLAP_RATIO = 0.10;

/** Default minimum overlap ratio with part mesh for surface voxels (1%) */
export const DEFAULT_MIN_PART_OVERLAP_RATIO = 0.01;

/** Default distance threshold as fraction of voxel size for surface voxels (50%) */
export const DEFAULT_SURFACE_DISTANCE_THRESHOLD = 0.5;

/** 
 * Number of voxel levels deep from shell boundary to apply biased weighting.
 * Level 0 = outer surface voxels (touching shell)
 * Level 1 = immediate neighbors of outer surface
 * Default: 1 (outer surface + 1 level deep = 2 total layers)
 */
export const BOUNDARY_DEPTH_LEVELS = 1;

/** Minimum resolution to use per axis (prevents too coarse grids) */
export const MIN_GRID_RESOLUTION = 16;

/** Maximum resolution to use per axis (prevents memory issues - 256³ = 16M max voxels) */
export const MAX_GRID_RESOLUTION = 256;

// ============================================================================
// AUTO RESOLUTION CALCULATION
// ============================================================================

/**
 * Calculate grid resolution from voxel density (voxels per unit length).
 * 
 * @param boundingBox - The bounding box of the grid domain
 * @param voxelsPerUnit - Voxel density (e.g., 1.0 = 1 voxel/mm if STL is in mm)
 * @returns Resolution per axis as Vector3, clamped between MIN and MAX values
 */
export function calculateResolutionFromDensity(
  boundingBox: THREE.Box3,
  voxelsPerUnit: number = DEFAULT_VOXELS_PER_UNIT
): THREE.Vector3 {
  const size = new THREE.Vector3();
  boundingBox.getSize(size);
  
  // Calculate resolution for each axis based on voxel density
  // voxelsPerUnit = 1 means 1 voxel per mm (if units are mm)
  // voxel size = 1 / voxelsPerUnit
  const voxelSize = 1.0 / voxelsPerUnit;
  
  const resX = Math.ceil(size.x / voxelSize);
  const resY = Math.ceil(size.y / voxelSize);
  const resZ = Math.ceil(size.z / voxelSize);
  
  // Clamp each dimension to valid range
  const clampedX = Math.max(MIN_GRID_RESOLUTION, Math.min(MAX_GRID_RESOLUTION, resX));
  const clampedY = Math.max(MIN_GRID_RESOLUTION, Math.min(MAX_GRID_RESOLUTION, resY));
  const clampedZ = Math.max(MIN_GRID_RESOLUTION, Math.min(MAX_GRID_RESOLUTION, resZ));
  
  logDebug(`Resolution from density: voxelsPerUnit=${voxelsPerUnit}, voxelSize=${voxelSize.toFixed(4)}mm, size=(${size.x.toFixed(2)}, ${size.y.toFixed(2)}, ${size.z.toFixed(2)})`);
  logDebug(`  Calculated: (${resX}, ${resY}, ${resZ}) -> clamped: (${clampedX}, ${clampedY}, ${clampedZ})`);
  
  return new THREE.Vector3(clampedX, clampedY, clampedZ);
}

/**
 * Calculate adaptive grid resolution based on bounding box diagonal
 * 
 * @param boundingBox - The bounding box of the grid domain
 * @param targetVoxelSizePercent - Target voxel size as percentage of diagonal (default: 2%)
 * @returns Resolution clamped between MIN and MAX values
 * @deprecated Use calculateResolutionFromDensity for more intuitive control
 */
export function calculateAdaptiveResolution(
  boundingBox: THREE.Box3,
  targetVoxelSizePercent: number = DEFAULT_TARGET_VOXEL_SIZE_PERCENT
): number {
  const size = new THREE.Vector3();
  boundingBox.getSize(size);
  const diagonal = size.length();
  
  // Target voxel size based on diagonal
  const targetVoxelSize = diagonal * targetVoxelSizePercent;
  
  // Calculate resolution based on largest dimension
  const maxDim = Math.max(size.x, size.y, size.z);
  const calculatedResolution = Math.ceil(maxDim / targetVoxelSize);
  
  // Clamp to valid range
  const resolution = Math.max(MIN_GRID_RESOLUTION, Math.min(MAX_GRID_RESOLUTION, calculatedResolution));
  
  logDebug(`Adaptive resolution: diagonal=${diagonal.toFixed(3)}, targetVoxelSize=${targetVoxelSize.toFixed(4)}, resolution=${resolution}`);
  
  return resolution;
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
  
  // Clone geometries to avoid modifying originals
  const shellGeom = outerShellGeometry.clone();
  const partGeom = partGeometry.clone();
  
  // Ensure geometries have computed bounds
  shellGeom.computeBoundingBox();
  partGeom.computeBoundingBox();
  
  // Get bounding box from outer shell (this defines the grid extent)
  const boundingBox = shellGeom.boundingBox!.clone();
  
  // Parse options
  const marginPercent = options.marginPercent ?? 0.05;
  const storeAllCells = options.storeAllCells ?? false;
  const computeDistances = options.computeDistances ?? false;
  const includeSurfaceVoxels = options.includeSurfaceVoxels ?? false;
  const targetVoxelSizePercent = options.targetVoxelSizePercent ?? DEFAULT_TARGET_VOXEL_SIZE_PERCENT;
  
  // Auto-calculate resolution if not provided
  let resolution: number | THREE.Vector3;
  if (options.resolution !== undefined) {
    resolution = options.resolution;
  } else {
    // Calculate adaptive resolution based on bounding box diagonal
    resolution = calculateAdaptiveResolution(boundingBox, targetVoxelSizePercent);
  }
  
  // Determine resolution per axis
  let resX: number, resY: number, resZ: number;
  if (typeof resolution === 'number') {
    resX = resY = resZ = resolution;
  } else {
    resX = resolution.x;
    resY = resolution.y;
    resZ = resolution.z;
  }
  
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
  const shellTester = new MeshTester(shellGeom, 'shell');
  const partTester = new MeshTester(partGeom, 'part');
  
  // Initialize result arrays (moldVolumeCells deprecated - using flat arrays only)
  const allCells: GridCell[] = [];
  
  // For building flat arrays: collect indices and centers in temporary arrays first
  // (we don't know the final count until we finish iterating)
  const tempCenters: number[] = [];
  const tempIndices: number[] = [];
  
  const totalCellCount = resX * resY * resZ;
  const cellCenter = new THREE.Vector3();
  const cellIndex = new THREE.Vector3();
  
  // Calculate surface tolerance: half the diagonal of a cell
  // This ensures voxels touching/intersecting surfaces are included
  const surfaceTolerance = includeSurfaceVoxels 
    ? Math.sqrt(cellSize.x * cellSize.x + cellSize.y * cellSize.y + cellSize.z * cellSize.z) / 2
    : 0;
  
  if (includeSurfaceVoxels) {
    logDebug(`Surface voxel inclusion enabled, tolerance: ${surfaceTolerance.toFixed(6)} (centers must be outside part)`);
  }

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
        
        let isMoldVolume: boolean;
        
        if (includeSurfaceVoxels) {
          // Include voxels that are inside OR on/near the shell surface
          const isInsideOrOnShell = shellTester.isInsideOrOnSurface(cellCenter, surfaceTolerance);
          
          if (isInsideOrOnShell) {
            // Check if voxel center is outside the part - CRITICAL for thin wall prevention
            const isOutsidePart = partTester.isOutside(cellCenter);
            
            if (isOutsidePart) {
              // Voxel center is outside part - include it as mold volume
              // This includes both regular mold voxels and surface voxels
              // (surface voxels have 1-50% overlap OR within 50% voxel distance of part)
              isMoldVolume = true;
            } else {
              // Voxel center is INSIDE the part - EXCLUDE to prevent thin wall bridging
              // We never include voxels whose centers are inside the part mesh
              isMoldVolume = false;
            }
          } else {
            isMoldVolume = false;
          }
        } else {
          // Original strict containment logic
          // Test if point is inside shell first (early exit if outside)
          const isInsideShell = shellTester.isInside(cellCenter);
          
          // Only test part if inside shell (optimization)
          const isOutsidePart = isInsideShell ? partTester.isOutside(cellCenter) : true;
          isMoldVolume = isInsideShell && isOutsidePart;
        }
        
        // Only create cell objects when needed for allCells
        if (isMoldVolume) {
          // Store in flat arrays (primary storage)
          tempCenters.push(cellCenter.x, cellCenter.y, cellCenter.z);
          tempIndices.push(i, j, k);
        }
        
        if (storeAllCells) {
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
          
          allCells.push(cell);
        }
      }
    }
  }
  
  // ========================================================================
  // DISTANCE FIELD COMPUTATION
  // For each silicone voxel node position x[i], compute d[i] = shortest distance to part surface M
  // Using BVH for efficient closest point queries
  // R = maximum distance (the farthest voxel from the part mesh)
  // ========================================================================
  
  // Convert temp arrays to typed arrays
  const voxelCenters = new Float32Array(tempCenters);
  const voxelIndices = new Uint32Array(tempIndices);
  const voxelCount = voxelCenters.length / 3;
  const voxelDist = new Float32Array(voxelCount);
  
  // Track min/max and the voxel with max distance for R visualization
  let minDist = Infinity;
  let maxDist = -Infinity;
  let rLineStart: THREE.Vector3 | null = null;
  let rLineEnd: THREE.Vector3 | null = null;
  
  const closestPoint = new THREE.Vector3();
  const target = { 
    point: closestPoint, 
    distance: Infinity,
    faceIndex: 0
  };
  
  // Reusable vector for center lookups
  const tempCenter = new THREE.Vector3();
  
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    tempCenter.set(voxelCenters[i3], voxelCenters[i3 + 1], voxelCenters[i3 + 2]);
    
    // Reset target for each query
    target.distance = Infinity;
    
    // Use BVH closestPointToPoint for efficient nearest-point query
    partTester.closestPointToPoint(tempCenter, target);
    
    const dist = target.distance;
    voxelDist[i] = dist;
    
    // Track min/max for validation and normalization
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) {
      maxDist = dist;
      // Store the line endpoints for R visualization
      rLineStart = tempCenter.clone();
      rLineEnd = target.point.clone();
    }
  }
  
  // R = maximum distance from any voxel to the part mesh
  const R = maxDist;
  
  // Log distance field and R values
  logResult('Distance field δ_i', { voxels: voxelCount, minDist, maxDist, R });
  if (rLineStart && rLineEnd) {
    logDebug(`R line: voxel (${rLineStart.x.toFixed(4)}, ${rLineStart.y.toFixed(4)}, ${rLineStart.z.toFixed(4)}) → part (${rLineEnd.x.toFixed(4)}, ${rLineEnd.y.toFixed(4)}, ${rLineEnd.z.toFixed(4)})`);
  }
  
  const distanceStats = { min: minDist, max: maxDist };
  
  // ========================================================================
  // SHELL DISTANCE FIELD COMPUTATION (δ_w)
  // For each voxel, compute distance to the outer shell boundary ∂H
  // ========================================================================
  
  const voxelDistToShell = new Float32Array(voxelCount);
  let minShellDist = Infinity;
  let maxShellDist = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    tempCenter.set(voxelCenters[i3], voxelCenters[i3 + 1], voxelCenters[i3 + 2]);
    
    // Reset target for each query
    target.distance = Infinity;
    
    // Use shellTester's BVH for closest point to shell boundary
    shellTester.closestPointToPoint(tempCenter, target);
    
    const dist = target.distance;
    voxelDistToShell[i] = dist;
    
    if (dist < minShellDist) minShellDist = dist;
    if (dist > maxShellDist) maxShellDist = dist;
  }
  
  const shellDistanceStats = { min: minShellDist, max: maxShellDist };
  
  logResult('Distance field δ_w (to shell)', { minShellDist, maxShellDist });
  
  // ========================================================================
  // IDENTIFY OUTER SURFACE VOXELS (NEAR SHELL) AND TWO LEVELS DEEP
  // Using same approach as partingSurface.ts:
  // 1. Find ALL surface voxels (those with at least one missing 6-neighbor)
  // 2. Use distance-to-part to distinguish inner (near part) vs outer (near shell)
  // 3. Mark outer surface + two levels deep for biased weighting
  // ========================================================================
  
  logDebug('Identifying boundary voxels for biased weighting...');
  
  // Build spatial index for neighbor lookups using flat arrays
  const voxelSpatialIndex = new Map<string, number>();
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    const gi = voxelIndices[i3];
    const gj = voxelIndices[i3 + 1];
    const gk = voxelIndices[i3 + 2];
    const key = `${gi},${gj},${gk}`;
    voxelSpatialIndex.set(key, i);
  }
  
  const getVoxelIdx = (gi: number, gj: number, gk: number): number => {
    const key = `${gi},${gj},${gk}`;
    return voxelSpatialIndex.get(key) ?? -1;
  };
  
  // 6-connected neighbor offsets (face-adjacent)
  const faceNeighborOffsets: [number, number, number][] = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
  ];
  
  // Step 1: Find ALL surface voxels (those with at least one missing 6-neighbor)
  const isSurfaceVoxel = new Uint8Array(voxelCount);
  let totalSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    const gi = voxelIndices[i3];
    const gj = voxelIndices[i3 + 1];
    const gk = voxelIndices[i3 + 2];
    
    // Check if any 6-connected neighbor is missing
    for (const [di, dj, dk] of faceNeighborOffsets) {
      const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
      if (neighborIdx < 0) {
        // Missing neighbor = this is a surface voxel
        isSurfaceVoxel[i] = 1;
        totalSurfaceCount++;
        break;
      }
    }
  }
  
  logDebug(`Total surface voxels: ${totalSurfaceCount}`);
  
  // Step 2: Use distance-to-part to distinguish inner vs outer surface
  // Inner surface = close to part (small voxelDist)
  // Outer surface = far from part (large voxelDist)
  const surfaceDistances: number[] = [];
  const surfaceIndices: number[] = [];
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i]) {
      surfaceDistances.push(voxelDist[i]);
      surfaceIndices.push(i);
    }
  }
  
  // Sort by distance to find the gap
  const sortedPairs = surfaceIndices.map((idx, i) => ({ idx, dist: surfaceDistances[i] }));
  sortedPairs.sort((a, b) => a.dist - b.dist);
  
  const surfaceMinDist = sortedPairs[0]?.dist || 0;
  const surfaceMaxDist = sortedPairs[sortedPairs.length - 1]?.dist || 0;
  
  logDebug(`Surface distance range: [${surfaceMinDist.toFixed(4)}, ${surfaceMaxDist.toFixed(4)}]`);
  
  // Find the largest gap in the sorted distances
  // This gap separates inner surface (low distances) from outer surface (high distances)
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
  const distanceThreshold = sortedPairs[gapIndex]?.dist || (surfaceMinDist + surfaceMaxDist) / 2;
  
  logDebug(`Gap detection: index ${gapIndex}/${sortedPairs.length}, gap=${maxGap.toFixed(4)}, threshold=${distanceThreshold.toFixed(4)}`);
  
  // Step 3: Mark OUTER surface voxels only (far from part = near shell)
  const isOuterSurface = new Uint8Array(voxelCount);
  let outerSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i] && voxelDist[i] >= distanceThreshold) {
      isOuterSurface[i] = 1;
      outerSurfaceCount++;
    }
  }
  
  logDebug(`Surface voxels: outer=${outerSurfaceCount}, inner=${totalSurfaceCount - outerSurfaceCount}`);
  
  // Step 4: Build mask for boundary voxels (outer surface + BOUNDARY_DEPTH_LEVELS levels deep)
  // This limits the biased weighting to voxels near the shell boundary
  const isBoundaryAdjacent = new Uint8Array(voxelCount);
  let boundaryAdjacentCount = 0;
  
  // First, mark all outer surface voxels (level 0)
  for (let i = 0; i < voxelCount; i++) {
    if (isOuterSurface[i]) {
      isBoundaryAdjacent[i] = 1;
      boundaryAdjacentCount++;
    }
  }
  
  // Expand outward for each level (1 to BOUNDARY_DEPTH_LEVELS)
  for (let level = 1; level <= BOUNDARY_DEPTH_LEVELS; level++) {
    // Track which voxels were marked in previous iteration
    const previouslyMarked = new Uint8Array(isBoundaryAdjacent);
    
    for (let i = 0; i < voxelCount; i++) {
      // Only expand from voxels marked in previous level
      if (previouslyMarked[i] && (level === 1 ? isOuterSurface[i] : true)) {
        const i3 = i * 3;
        const gi = voxelIndices[i3];
        const gj = voxelIndices[i3 + 1];
        const gk = voxelIndices[i3 + 2];
        
        for (const [di, dj, dk] of faceNeighborOffsets) {
          const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
          if (neighborIdx >= 0 && !isBoundaryAdjacent[neighborIdx]) {
            isBoundaryAdjacent[neighborIdx] = 1;
            boundaryAdjacentCount++;
          }
        }
      }
    }
  }
  
  logResult('Boundary detection', { boundaryVoxels: boundaryAdjacentCount, interiorVoxels: voxelCount - boundaryAdjacentCount, depthLevels: BOUNDARY_DEPTH_LEVELS });
  
  // ========================================================================
  // BIASED DISTANCE FIELD COMPUTATION
  // biasedDist = w₁ * δ_i + w₂ * λ_w where λ_w = R - δ_w
  // Default weights: w₁ = 1.0, w₂ = 1.0
  // BUT: Only apply bias to boundary-adjacent voxels (outer surface + one level deep)
  // Interior voxels use w₁ * δ_i (no shell bias)
  // ========================================================================
  
  // Default weights (can be recalculated later with custom weights)
  const w1 = DEFAULT_BIASED_DISTANCE_WEIGHTS.partDistanceWeight;
  const w2 = DEFAULT_BIASED_DISTANCE_WEIGHTS.shellBiasWeight;
  
  const biasedDist = new Float32Array(voxelCount);
  let minBiasedDist = Infinity;
  let maxBiasedDist = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const delta_i = voxelDist[i];           // Distance to part
    
    let biased: number;
    if (isBoundaryAdjacent[i]) {
      // Boundary-adjacent: apply full bias with weights
      const delta_w = voxelDistToShell[i];    // Distance to shell
      const lambda_w = R - delta_w;           // Bias penalty
      biased = w1 * delta_i + w2 * lambda_w;  // Weighted biased distance
    } else {
      // Interior: only part distance term with weight
      biased = w1 * delta_i;
    }
    
    biasedDist[i] = biased;
    
    if (biased < minBiasedDist) minBiasedDist = biased;
    if (biased > maxBiasedDist) maxBiasedDist = biased;
  }
  
  const biasedDistanceStats = { min: minBiasedDist, max: maxBiasedDist };
  
  logResult('Biased distance field', { minBiasedDist, maxBiasedDist });
  
  // ========================================================================
  // WEIGHTING FACTOR COMPUTATION
  // wt = 1/[(biasedDist^2) + 0.25]
  // Higher weight near part (low biased distance), lower weight far from part
  // ========================================================================
  
  const weightingFactor = new Float32Array(voxelCount);
  let minWeight = Infinity;
  let maxWeight = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const bd = biasedDist[i];
    const wt = 1.0 / (bd * bd + 0.25);
    
    weightingFactor[i] = wt;
    
    if (wt < minWeight) minWeight = wt;
    if (wt > maxWeight) maxWeight = wt;
  }
  
  const weightingFactorStats = { min: minWeight, max: maxWeight };
  
  logResult('Weighting factor', { minWeight, maxWeight });
  
  // Clean up BVH resources
  shellTester.dispose();
  partTester.dispose();
  shellGeom.dispose();
  partGeom.dispose();
  
  const endTime = performance.now();
  
  // Calculate statistics
  const cellVolume = cellSize.x * cellSize.y * cellSize.z;
  const moldVolume = voxelCount * cellVolume;
  const totalVolume = totalCellCount * cellVolume;
  
  const stats: VolumetricGridStats = {
    moldVolume,
    totalVolume,
    fillRatio: moldVolume / totalVolume,
    computeTimeMs: endTime - startTime,
  };
  
  return {
    allCells: storeAllCells ? allCells : [],
    moldVolumeCells: [], // Deprecated - use voxelCenters/voxelIndices instead
    voxelCenters,
    voxelIndices,
    resolution: new THREE.Vector3(resX, resY, resZ),
    totalCellCount,
    moldVolumeCellCount: voxelCount,
    boundingBox,
    cellSize,
    stats,
    voxelDist,
    voxelDistToShell,
    biasedDist,
    weightingFactor,
    distanceStats,
    shellDistanceStats,
    biasedDistanceStats,
    weightingFactorStats,
    R,
    rLineStart,
    rLineEnd,
    boundaryAdjacentMask: isBoundaryAdjacent,
    seedVoxelMask: null,
  };
}

// ============================================================================
// GPU-ACCELERATED GENERATION
// ============================================================================

/**
 * Check if WebGPU is available in the browser
 */
export function isWebGPUAvailable(): boolean {
  return 'gpu' in navigator;
}

/**
 * GPU-accelerated volumetric grid generator using WebGPU compute shaders
 * Falls back to CPU if WebGPU is not available or if includeSurfaceVoxels is enabled
 */
export async function generateVolumetricGridGPU(
  outerShellGeometry: THREE.BufferGeometry,
  partGeometry: THREE.BufferGeometry,
  options: VolumetricGridOptions = {}
): Promise<VolumetricGridResult> {
  // Fall back to CPU if surface voxel inclusion is enabled (requires distance queries)
  if (options.includeSurfaceVoxels) {
    logDebug('Surface voxel inclusion enabled, using CPU implementation');
    return generateVolumetricGrid(outerShellGeometry, partGeometry, options);
  }
  
  // Check for WebGPU support
  if (!isWebGPUAvailable()) {
    logInfo('WebGPU not available, falling back to CPU');
    return generateVolumetricGrid(outerShellGeometry, partGeometry, options);
  }
  
  const startTime = performance.now();
  
  // Parse options
  const resolution = typeof options.resolution === 'number' 
    ? options.resolution 
    : options.resolution?.x ?? DEFAULT_GRID_RESOLUTION;
  const marginPercent = options.marginPercent ?? 0.05;
  const storeAllCells = options.storeAllCells ?? false;
  
  logInfo(`GPU Grid Generation: resolution=${resolution}³`);
  
  try {
    // Initialize WebGPU
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.warn('No WebGPU adapter found, falling back to CPU');
      return generateVolumetricGrid(outerShellGeometry, partGeometry, options);
    }
    
    const device = await adapter.requestDevice();
    
    // Clone and prepare geometries
    const shellGeom = outerShellGeometry.clone();
    const partGeom = partGeometry.clone();
    shellGeom.computeBoundingBox();
    
    // Get bounding box with margin
    const boundingBox = shellGeom.boundingBox!.clone();
    const boxSize = new THREE.Vector3();
    boundingBox.getSize(boxSize);
    const margin = boxSize.clone().multiplyScalar(marginPercent);
    boundingBox.min.sub(margin);
    boundingBox.max.add(margin);
    boundingBox.getSize(boxSize);
    
    const cellSize = new THREE.Vector3(
      boxSize.x / resolution,
      boxSize.y / resolution,
      boxSize.z / resolution
    );
    
    // Extract triangles
    const shellTriangles = extractTrianglesForGPU(shellGeom);
    const partTriangles = extractTrianglesForGPU(partGeom);
    
    logDebug(`Triangles: shell=${shellTriangles.length / 9}, part=${partTriangles.length / 9}`);
    
    const totalCells = resolution ** 3;
    
    // Create GPU buffers
    const paramsData = new Float32Array([
      boundingBox.min.x, boundingBox.min.y, boundingBox.min.z, resolution,
      cellSize.x, cellSize.y, cellSize.z, 0,
    ]);
    
    const paramsBuffer = device.createBuffer({
      size: paramsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    
    // Triangle count buffer
    const triCountData = new Uint32Array([shellTriangles.length / 9, partTriangles.length / 9]);
    const triCountBuffer = device.createBuffer({
      size: triCountData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(triCountBuffer, 0, triCountData);
    
    // Shell triangles buffer
    const shellBuffer = device.createBuffer({
      size: shellTriangles.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(shellBuffer, 0, shellTriangles.buffer);
    
    // Part triangles buffer
    const partBuffer = device.createBuffer({
      size: partTriangles.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(partBuffer, 0, partTriangles.buffer);
    
    // Results buffer
    const resultBuffer = device.createBuffer({
      size: totalCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    // Staging buffer for reading results
    const stagingBuffer = device.createBuffer({
      size: totalCells * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    
    // Create compute shader
    const shaderCode = getWebGPUComputeShader();
    const shaderModule = device.createShaderModule({ code: shaderCode });
    
    // Create compute pipeline
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
    
    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: triCountBuffer } },
        { binding: 2, resource: { buffer: shellBuffer } },
        { binding: 3, resource: { buffer: partBuffer } },
        { binding: 4, resource: { buffer: resultBuffer } },
      ],
    });
    
    // Dispatch compute shader
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    const workgroupSize = 64;
    const numWorkgroups = Math.ceil(totalCells / workgroupSize);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();
    
    // Copy results to staging buffer
    commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, totalCells * 4);
    
    // Submit and wait
    device.queue.submit([commandEncoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    
    // Extract mold volume cells into flat arrays only (moldVolumeCells deprecated)
    const allCells: GridCell[] = [];
    
    // Temporary arrays for flat data (we don't know the count yet)
    const tempCenters: number[] = [];
    const tempIndices: number[] = [];
    
    for (let idx = 0; idx < totalCells; idx++) {
      const i = idx % resolution;
      const j = Math.floor(idx / resolution) % resolution;
      const k = Math.floor(idx / (resolution * resolution));
      
      const cx = boundingBox.min.x + (i + 0.5) * cellSize.x;
      const cy = boundingBox.min.y + (j + 0.5) * cellSize.y;
      const cz = boundingBox.min.z + (k + 0.5) * cellSize.z;
      
      const isMoldVolume = resultData[idx] === 1;
      
      if (isMoldVolume) {
        // Store in flat arrays (primary storage)
        tempCenters.push(cx, cy, cz);
        tempIndices.push(i, j, k);
      }
      
      if (storeAllCells) {
        allCells.push({
          index: new THREE.Vector3(i, j, k),
          center: new THREE.Vector3(cx, cy, cz),
          size: cellSize.clone(),
          isMoldVolume,
        });
      }
    }
    
    // Convert to typed arrays
    const voxelCenters = new Float32Array(tempCenters);
    const voxelIndices = new Uint32Array(tempIndices);
    
    // Clean up GPU resources
    paramsBuffer.destroy();
    triCountBuffer.destroy();
    shellBuffer.destroy();
    partBuffer.destroy();
    resultBuffer.destroy();
    stagingBuffer.destroy();
    
    // ========================================================================
    // DISTANCE FIELD COMPUTATION (CPU-based for GPU grid)
    // For each mold volume voxel, compute distance to part surface M
    // R = maximum distance (the farthest voxel from the part mesh)
    // ========================================================================
    
    const voxelCount = voxelCenters.length / 3;
    const voxelDist = new Float32Array(voxelCount);
    
    // Build BVH on part geometry for distance queries
    const partTester = new MeshTester(partGeom, 'part');
    // Build BVH on shell geometry for shell distance queries
    const shellTester = new MeshTester(shellGeom, 'shell');
    
    // Track min/max and the voxel with max distance for R visualization
    let minDist = Infinity;
    let maxDist = -Infinity;
    let rLineStart: THREE.Vector3 | null = null;
    let rLineEnd: THREE.Vector3 | null = null;
    
    const closestPoint = new THREE.Vector3();
    const target = { 
      point: closestPoint, 
      distance: Infinity,
      faceIndex: 0
    };
    
    // Reusable vector for center lookups
    const tempCenter = new THREE.Vector3();
    
    for (let i = 0; i < voxelCount; i++) {
      const i3 = i * 3;
      tempCenter.set(voxelCenters[i3], voxelCenters[i3 + 1], voxelCenters[i3 + 2]);
      target.distance = Infinity;
      partTester.closestPointToPoint(tempCenter, target);
      const dist = target.distance;
      voxelDist[i] = dist;
      
      if (dist < minDist) minDist = dist;
      if (dist > maxDist) {
        maxDist = dist;
        // Store the line endpoints for R visualization
        rLineStart = tempCenter.clone();
        rLineEnd = target.point.clone();
      }
    }
    
    // R = maximum distance from any voxel to the part mesh
    const R = maxDist;
    
    logResult('Distance field δ_i (GPU)', { voxels: voxelCount, minDist, maxDist, R });
    if (rLineStart && rLineEnd) {
      logDebug(`R line: voxel (${rLineStart.x.toFixed(4)}, ${rLineStart.y.toFixed(4)}, ${rLineStart.z.toFixed(4)}) → part (${rLineEnd.x.toFixed(4)}, ${rLineEnd.y.toFixed(4)}, ${rLineEnd.z.toFixed(4)})`);
    }
    
    const distanceStats = { min: minDist, max: maxDist };
    
    // ========================================================================
    // SHELL DISTANCE FIELD COMPUTATION (δ_w)
    // For each voxel, compute distance to the outer shell boundary ∂H
    // ========================================================================
    
    const voxelDistToShell = new Float32Array(voxelCount);
    let minShellDist = Infinity;
    let maxShellDist = -Infinity;
    
    for (let i = 0; i < voxelCount; i++) {
      const i3 = i * 3;
      tempCenter.set(voxelCenters[i3], voxelCenters[i3 + 1], voxelCenters[i3 + 2]);
      target.distance = Infinity;
      shellTester.closestPointToPoint(tempCenter, target);
      const dist = target.distance;
      voxelDistToShell[i] = dist;
      
      if (dist < minShellDist) minShellDist = dist;
      if (dist > maxShellDist) maxShellDist = dist;
    }
    
    const shellDistanceStats = { min: minShellDist, max: maxShellDist };
    
    logResult('Distance field δ_w (to shell, GPU)', { minShellDist, maxShellDist });
    
    // ========================================================================
    // IDENTIFY OUTER SURFACE VOXELS (NEAR SHELL) AND TWO LEVELS DEEP
    // Same logic as CPU version
    // ========================================================================
    
    logDebug('Identifying boundary voxels (GPU)...');
    
    // Build spatial index for neighbor lookups using flat arrays
    const voxelSpatialIndex = new Map<string, number>();
    for (let i = 0; i < voxelCount; i++) {
      const i3 = i * 3;
      const gi = voxelIndices[i3];
      const gj = voxelIndices[i3 + 1];
      const gk = voxelIndices[i3 + 2];
      const key = `${gi},${gj},${gk}`;
      voxelSpatialIndex.set(key, i);
    }
    
    const getVoxelIdx = (gi: number, gj: number, gk: number): number => {
      const key = `${gi},${gj},${gk}`;
      return voxelSpatialIndex.get(key) ?? -1;
    };
    
    // 6-connected neighbor offsets
    const faceNeighborOffsets: [number, number, number][] = [
      [1, 0, 0], [-1, 0, 0],
      [0, 1, 0], [0, -1, 0],
      [0, 0, 1], [0, 0, -1],
    ];
    
    // Step 1: Find ALL surface voxels
    const isSurfaceVoxel = new Uint8Array(voxelCount);
    let totalSurfaceCount = 0;
    
    for (let i = 0; i < voxelCount; i++) {
      const i3 = i * 3;
      const gi = voxelIndices[i3];
      const gj = voxelIndices[i3 + 1];
      const gk = voxelIndices[i3 + 2];
      
      for (const [di, dj, dk] of faceNeighborOffsets) {
        const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
        if (neighborIdx < 0) {
          isSurfaceVoxel[i] = 1;
          totalSurfaceCount++;
          break;
        }
      }
    }
    
    logDebug(`Total surface voxels (GPU): ${totalSurfaceCount}`);
    
    // Step 2: Use distance-to-part to distinguish inner vs outer surface
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
    
    const surfaceMinDist = sortedPairs[0]?.dist || 0;
    const surfaceMaxDist = sortedPairs[sortedPairs.length - 1]?.dist || 0;
    
    // Find the largest gap
    let maxGap = 0;
    let gapIndex = Math.floor(sortedPairs.length / 2);
    
    for (let i = 1; i < sortedPairs.length; i++) {
      const gap = sortedPairs[i].dist - sortedPairs[i - 1].dist;
      if (gap > maxGap) {
        maxGap = gap;
        gapIndex = i;
      }
    }
    
    const distanceThreshold = sortedPairs[gapIndex]?.dist || (surfaceMinDist + surfaceMaxDist) / 2;
    logDebug(`Distance threshold (GPU): ${distanceThreshold.toFixed(4)}`);
    
    // Step 3: Mark OUTER surface voxels only
    const isOuterSurface = new Uint8Array(voxelCount);
    let outerSurfaceCount = 0;
    
    for (let i = 0; i < voxelCount; i++) {
      if (isSurfaceVoxel[i] && voxelDist[i] >= distanceThreshold) {
        isOuterSurface[i] = 1;
        outerSurfaceCount++;
      }
    }
    
    logDebug(`Outer surface voxels (GPU): ${outerSurfaceCount}`);
    
    // Step 4: Build mask for boundary voxels (outer surface + BOUNDARY_DEPTH_LEVELS levels deep)
    const isBoundaryAdjacent = new Uint8Array(voxelCount);
    let boundaryAdjacentCount = 0;
    
    // First, mark all outer surface voxels (level 0)
    for (let i = 0; i < voxelCount; i++) {
      if (isOuterSurface[i]) {
        isBoundaryAdjacent[i] = 1;
        boundaryAdjacentCount++;
      }
    }
    
    // Expand outward for each level (1 to BOUNDARY_DEPTH_LEVELS)
    for (let level = 1; level <= BOUNDARY_DEPTH_LEVELS; level++) {
      const previouslyMarked = new Uint8Array(isBoundaryAdjacent);
      
      for (let i = 0; i < voxelCount; i++) {
        if (previouslyMarked[i] && (level === 1 ? isOuterSurface[i] : true)) {
          const i3 = i * 3;
          const gi = voxelIndices[i3];
          const gj = voxelIndices[i3 + 1];
          const gk = voxelIndices[i3 + 2];
          
          for (const [di, dj, dk] of faceNeighborOffsets) {
            const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
            if (neighborIdx >= 0 && !isBoundaryAdjacent[neighborIdx]) {
              isBoundaryAdjacent[neighborIdx] = 1;
              boundaryAdjacentCount++;
            }
          }
        }
      }
    }
    
    logResult('Boundary detection (GPU)', { boundaryVoxels: boundaryAdjacentCount, interiorVoxels: voxelCount - boundaryAdjacentCount, depthLevels: BOUNDARY_DEPTH_LEVELS });
    
    // ========================================================================
    // BIASED DISTANCE FIELD COMPUTATION
    // biasedDist = w₁ * δ_i + w₂ * λ_w where λ_w = R - δ_w
    // Default weights: w₁ = 1.0, w₂ = 1.0
    // BUT: Only apply bias to boundary-adjacent voxels
    // ========================================================================
    
    // Default weights (can be recalculated later with custom weights)
    const w1 = DEFAULT_BIASED_DISTANCE_WEIGHTS.partDistanceWeight;
    const w2 = DEFAULT_BIASED_DISTANCE_WEIGHTS.shellBiasWeight;
    
    const biasedDist = new Float32Array(voxelCount);
    let minBiasedDist = Infinity;
    let maxBiasedDist = -Infinity;
    
    for (let i = 0; i < voxelCount; i++) {
      const delta_i = voxelDist[i];
      
      let biased: number;
      if (isBoundaryAdjacent[i]) {
        const delta_w = voxelDistToShell[i];
        const lambda_w = R - delta_w;
        biased = w1 * delta_i + w2 * lambda_w;
      } else {
        biased = w1 * delta_i;
      }
      
      biasedDist[i] = biased;
      
      if (biased < minBiasedDist) minBiasedDist = biased;
      if (biased > maxBiasedDist) maxBiasedDist = biased;
    }
    
    const biasedDistanceStats = { min: minBiasedDist, max: maxBiasedDist };
    
    logResult('Biased distance (GPU)', { minBiasedDist, maxBiasedDist });
    
    // ========================================================================
    // WEIGHTING FACTOR COMPUTATION
    // wt = 1/[(biasedDist^2) + 0.25]
    // ========================================================================
    
    const weightingFactor = new Float32Array(voxelCount);
    let minWeight = Infinity;
    let maxWeight = -Infinity;
    
    for (let i = 0; i < voxelCount; i++) {
      const bd = biasedDist[i];
      const wt = 1.0 / (bd * bd + 0.25);
      
      weightingFactor[i] = wt;
      
      if (wt < minWeight) minWeight = wt;
      if (wt > maxWeight) maxWeight = wt;
    }
    
    const weightingFactorStats = { min: minWeight, max: maxWeight };
    
    logResult('Weighting factor (GPU)', { minWeight, maxWeight });
    
    // Clean up
    partTester.dispose();
    shellTester.dispose();
    shellGeom.dispose();
    partGeom.dispose();
    
    const endTime = performance.now();
    
    const cellVolume = cellSize.x * cellSize.y * cellSize.z;
    const moldVolume = voxelCount * cellVolume;
    const totalVolume = totalCells * cellVolume;
    
    const stats: VolumetricGridStats = {
      moldVolume,
      totalVolume,
      fillRatio: moldVolume / totalVolume,
      computeTimeMs: endTime - startTime,
    };
    
    logResult('GPU Grid Complete', { moldCells: voxelCount, totalCells, fillRatio: (stats.fillRatio * 100).toFixed(1) + '%', timeMs: stats.computeTimeMs.toFixed(1) });
    
    return {
      allCells: storeAllCells ? allCells : [],
      moldVolumeCells: [], // Deprecated - use voxelCenters/voxelIndices instead
      voxelCenters,
      voxelIndices,
      resolution: new THREE.Vector3(resolution, resolution, resolution),
      totalCellCount: totalCells,
      moldVolumeCellCount: voxelCount,
      boundingBox,
      cellSize,
      stats,
      voxelDist,
      voxelDistToShell,
      biasedDist,
      weightingFactor,
      distanceStats,
      shellDistanceStats,
      biasedDistanceStats,
      weightingFactorStats,
      R,
      rLineStart,
      rLineEnd,
      boundaryAdjacentMask: isBoundaryAdjacent,
      seedVoxelMask: null,
    };
    
  } catch (error) {
    console.error('GPU generation failed, falling back to CPU:', error);
    return generateVolumetricGrid(outerShellGeometry, partGeometry, options);
  }
}

/**
 * Extract triangles from geometry as flat Float32Array for GPU
 */
function extractTrianglesForGPU(geometry: THREE.BufferGeometry): Float32Array {
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  
  const triangleCount = index ? index.count / 3 : position.count / 3;
  const triangles = new Float32Array(triangleCount * 9);
  
  for (let i = 0; i < triangleCount; i++) {
    const [a, b, c] = index
      ? [index.getX(i * 3), index.getX(i * 3 + 1), index.getX(i * 3 + 2)]
      : [i * 3, i * 3 + 1, i * 3 + 2];
    
    triangles[i * 9 + 0] = position.getX(a);
    triangles[i * 9 + 1] = position.getY(a);
    triangles[i * 9 + 2] = position.getZ(a);
    triangles[i * 9 + 3] = position.getX(b);
    triangles[i * 9 + 4] = position.getY(b);
    triangles[i * 9 + 5] = position.getZ(b);
    triangles[i * 9 + 6] = position.getX(c);
    triangles[i * 9 + 7] = position.getY(c);
    triangles[i * 9 + 8] = position.getZ(c);
  }
  
  return triangles;
}

/**
 * Generate WebGPU compute shader for volumetric grid
 */
function getWebGPUComputeShader(): string {
  return /* wgsl */`
    struct Params {
      boundsMin: vec3f,
      resolution: f32,
      cellSize: vec3f,
      _padding: f32,
    }
    
    struct TriCounts {
      shellCount: u32,
      partCount: u32,
    }
    
    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<uniform> triCounts: TriCounts;
    @group(0) @binding(2) var<storage, read> shellTriangles: array<f32>;
    @group(0) @binding(3) var<storage, read> partTriangles: array<f32>;
    @group(0) @binding(4) var<storage, read_write> results: array<u32>;
    
    // Möller–Trumbore ray-triangle intersection
    fn rayTriangleIntersect(origin: vec3f, dir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> bool {
      let edge1 = v1 - v0;
      let edge2 = v2 - v0;
      let h = cross(dir, edge2);
      let a = dot(edge1, h);
      
      if (abs(a) < 0.000001) {
        return false;
      }
      
      let f = 1.0 / a;
      let s = origin - v0;
      let u = f * dot(s, h);
      
      if (u < 0.0 || u > 1.0) {
        return false;
      }
      
      let q = cross(s, edge1);
      let v = f * dot(dir, q);
      
      if (v < 0.0 || u + v > 1.0) {
        return false;
      }
      
      let t = f * dot(edge2, q);
      return t > 0.000001;
    }
    
    // Count intersections with shell mesh
    fn countShellIntersections(point: vec3f) -> u32 {
      let rayDir = vec3f(1.0, 0.0, 0.0);
      var count: u32 = 0u;
      
      for (var i: u32 = 0u; i < triCounts.shellCount; i = i + 1u) {
        let idx = i * 9u;
        let v0 = vec3f(shellTriangles[idx], shellTriangles[idx + 1u], shellTriangles[idx + 2u]);
        let v1 = vec3f(shellTriangles[idx + 3u], shellTriangles[idx + 4u], shellTriangles[idx + 5u]);
        let v2 = vec3f(shellTriangles[idx + 6u], shellTriangles[idx + 7u], shellTriangles[idx + 8u]);
        
        if (rayTriangleIntersect(point, rayDir, v0, v1, v2)) {
          count = count + 1u;
        }
      }
      
      return count;
    }
    
    // Count intersections with part mesh
    fn countPartIntersections(point: vec3f) -> u32 {
      let rayDir = vec3f(1.0, 0.0, 0.0);
      var count: u32 = 0u;
      
      for (var i: u32 = 0u; i < triCounts.partCount; i = i + 1u) {
        let idx = i * 9u;
        let v0 = vec3f(partTriangles[idx], partTriangles[idx + 1u], partTriangles[idx + 2u]);
        let v1 = vec3f(partTriangles[idx + 3u], partTriangles[idx + 4u], partTriangles[idx + 5u]);
        let v2 = vec3f(partTriangles[idx + 6u], partTriangles[idx + 7u], partTriangles[idx + 8u]);
        
        if (rayTriangleIntersect(point, rayDir, v0, v1, v2)) {
          count = count + 1u;
        }
      }
      
      return count;
    }
    
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      let resolution = u32(params.resolution);
      let totalCells = resolution * resolution * resolution;
      let cellIdx = id.x;
      
      if (cellIdx >= totalCells) {
        return;
      }
      
      // Convert linear index to 3D grid coordinates
      let i = cellIdx % resolution;
      let j = (cellIdx / resolution) % resolution;
      let k = cellIdx / (resolution * resolution);
      
      // Calculate cell center
      let center = params.boundsMin + params.cellSize * (vec3f(f32(i), f32(j), f32(k)) + 0.5);
      
      // Check inside shell (odd intersections = inside)
      let shellHits = countShellIntersections(center);
      let insideShell = (shellHits % 2u) == 1u;
      
      if (!insideShell) {
        results[cellIdx] = 0u;
        return;
      }
      
      // Check outside part (even intersections = outside)
      let partHits = countPartIntersections(center);
      let insidePart = (partHits % 2u) == 1u;
      
      // Mold volume = inside shell AND outside part
      results[cellIdx] = select(0u, 1u, !insidePart);
    }
  `;
}

// ============================================================================
// VISUALIZATION HELPERS
// ============================================================================

/** Type of field to use for visualization coloring */
export type DistanceFieldType = 'part' | 'biased' | 'weight' | 'boundary';

/**
 * Get the distance data and stats for a given field type
 */
function getDistanceFieldData(
  gridResult: VolumetricGridResult,
  fieldType: DistanceFieldType
): { data: Float32Array | Uint8Array | null; stats: { min: number; max: number } | null; isBinaryMask?: boolean } {
  if (fieldType === 'biased') {
    return {
      data: gridResult.biasedDist,
      stats: gridResult.biasedDistanceStats
    };
  }
  if (fieldType === 'weight') {
    return {
      data: gridResult.weightingFactor,
      stats: gridResult.weightingFactorStats
    };
  }
  if (fieldType === 'boundary') {
    // Return boundary mask as binary data (0 = interior/unbiased, 1 = boundary-adjacent/biased)
    return {
      data: gridResult.boundaryAdjacentMask,
      stats: { min: 0, max: 1 },
      isBinaryMask: true
    };
  }
  // Default to part distance
  return {
    data: gridResult.voxelDist,
    stats: gridResult.distanceStats
  };
}

/**
 * Create a point cloud visualization of the mold volume cells
 * Colors points based on distance field: red (close/low) to teal (far/high)
 * 
 * @param gridResult - The volumetric grid result
 * @param color - Fallback color for points if no distance data (default: cyan)
 * @param pointSize - Size of each point (default: 2)
 * @param distanceFieldType - Which distance field to use for coloring: 'part' or 'biased' (default: 'part')
 * @param voxelSizes - Optional array of per-voxel sizes for adaptive grids (scales point size)
 * @returns THREE.Points object for adding to scene
 */
export function createMoldVolumePointCloud(
  gridResult: VolumetricGridResult,
  color: THREE.ColorRepresentation = 0x00ffff,
  pointSize: number = 2,
  distanceFieldType: DistanceFieldType = 'part',
  voxelSizes: Float32Array | null = null
): THREE.Points {
  const cellCount = gridResult.moldVolumeCellCount;
  const positions = new Float32Array(cellCount * 3);
  const colors = new Float32Array(cellCount * 3);
  const sizes = new Float32Array(cellCount); // Per-point sizes for adaptive grids
  
  // Use flat arrays if available (more memory efficient), fall back to deprecated moldVolumeCells
  const useFlat = gridResult.voxelCenters !== null;
  
  // Get the appropriate distance field data
  const fieldData = getDistanceFieldData(gridResult, distanceFieldType);
  const { data: distData, stats: distStats, isBinaryMask } = fieldData;
  const hasDistanceData = distData !== null && distStats !== null;
  
  // Debug logging for boundary mask
  if (distanceFieldType === 'boundary') {
    logDebug('Boundary mask visualization:', {
      hasData: distData !== null,
      dataLength: distData?.length,
      isBinaryMask,
      hasDistanceData
    });
  }
  
  // Color gradient: red (close to part) -> teal (far from part)
  // For weighting factor: red (high cost) -> blue (low cost) [inverted]
  // For boundary mask: orange (biased/boundary) vs blue (unbiased/interior)
  const colorClose = new THREE.Color(0xff0000); // Red
  const colorFar = new THREE.Color(0x00ffff);   // Teal (cyan)
  const colorWeightHigh = new THREE.Color(0xff0000); // Red - high cost (costly to walk)
  const colorWeightLow = new THREE.Color(0x0088ff);  // Blue - low cost (cheap to walk)
  const colorBiased = new THREE.Color(0xff8800);   // Orange - biased (boundary-adjacent)
  const colorUnbiased = new THREE.Color(0x0088ff); // Blue - unbiased (interior)
  const tempColor = new THREE.Color();
  
  // Base size for point scaling (from uniform grid cell size)
  const baseCellSize = gridResult.cellSize.x;
  
  for (let i = 0; i < cellCount; i++) {
    // Get position from flat array or legacy object
    if (useFlat) {
      const i3 = i * 3;
      positions[i3] = gridResult.voxelCenters![i3];
      positions[i3 + 1] = gridResult.voxelCenters![i3 + 1];
      positions[i3 + 2] = gridResult.voxelCenters![i3 + 2];
    } else {
      const cell = gridResult.moldVolumeCells[i];
      positions[i * 3] = cell.center.x;
      positions[i * 3 + 1] = cell.center.y;
      positions[i * 3 + 2] = cell.center.z;
    }
    
    // Set point size based on voxel size (for adaptive grids)
    if (voxelSizes) {
      // Scale point size proportionally to voxel size
      sizes[i] = pointSize * (voxelSizes[i] / baseCellSize);
    } else {
      sizes[i] = pointSize;
    }
    
    if (hasDistanceData) {
      if (isBinaryMask === true) {
        // Binary mask: use distinct colors for biased vs unbiased
        const maskValue = distData![i];
        const isBiased = maskValue === 1;
        tempColor.copy(isBiased ? colorBiased : colorUnbiased);
      } else if (distanceFieldType === 'weight') {
        // Weighting factor: red (high cost) -> blue (low cost)
        // High weight = close to boundary = costly to traverse
        const dist = distData![i];
        const { min, max } = distStats!;
        const range = max - min;
        const t = range > 0 ? (dist - min) / range : 0;
        
        // Interpolate: low weight (t=0) -> blue, high weight (t=1) -> red
        tempColor.lerpColors(colorWeightLow, colorWeightHigh, t);
      } else {
        // Continuous field: interpolate based on normalized value
        const dist = distData![i];
        const { min, max } = distStats!;
        const range = max - min;
        const t = range > 0 ? (dist - min) / range : 0;
        
        // Interpolate between red (close) and teal (far)
        tempColor.lerpColors(colorClose, colorFar, t);
      }
      colors[i * 3] = tempColor.r;
      colors[i * 3 + 1] = tempColor.g;
      colors[i * 3 + 2] = tempColor.b;
    } else {
      // Fallback to uniform color
      const fallbackColor = new THREE.Color(color);
      colors[i * 3] = fallbackColor.r;
      colors[i * 3 + 1] = fallbackColor.g;
      colors[i * 3 + 2] = fallbackColor.b;
    }
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  
  // Use shader material for variable point sizes
  const hasVariableSizes = voxelSizes !== null;
  
  let material: THREE.PointsMaterial | THREE.ShaderMaterial;
  
  if (hasVariableSizes) {
    // Custom shader material for per-point sizes
    material = new THREE.ShaderMaterial({
      uniforms: {
        opacity: { value: 0.6 }
      },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform float opacity;
        varying vec3 vColor;
        void main() {
          float r = distance(gl_PointCoord, vec2(0.5, 0.5));
          if (r > 0.5) discard;
          gl_FragColor = vec4(vColor, opacity);
        }
      `,
      transparent: true,
      vertexColors: true,
      depthWrite: false
    });
  } else {
    material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.6,
      vertexColors: true, // Enable per-vertex coloring
    });
  }
  
  return new THREE.Points(geometry, material);
}

/**
 * Create a voxel box visualization of the mold volume cells
 * Uses instanced mesh for efficient rendering of many boxes
 * Colors boxes based on distance to part mesh: red (close) to teal (far)
 * 
 * @param gridResult - The volumetric grid result
 * @param color - Fallback color for the boxes if no distance data (default: cyan)
 * @param opacity - Opacity of boxes (default: 0.3)
 * @param distanceFieldType - Which distance field to use for coloring: 'part' or 'biased' (default: 'part')
 * @param voxelSizes - Optional array of per-voxel sizes for adaptive grids
 * @returns THREE.InstancedMesh for adding to scene
 */
export function createMoldVolumeVoxels(
  gridResult: VolumetricGridResult,
  color: THREE.ColorRepresentation = 0x00ffff,
  opacity: number = 0.3,
  distanceFieldType: DistanceFieldType = 'part',
  voxelSizes: Float32Array | null = null
): THREE.InstancedMesh {
  const cellCount = gridResult.moldVolumeCellCount;
  const { cellSize } = gridResult;
  
  // Use flat arrays if available (more memory efficient), fall back to deprecated moldVolumeCells
  const useFlat = gridResult.voxelCenters !== null;
  
  // For adaptive grids with variable voxel sizes, use a unit box and scale per instance
  // For uniform grids, use the cell size directly for better performance
  const hasVariableSizes = voxelSizes !== null;
  const boxGeometry = hasVariableSizes 
    ? new THREE.BoxGeometry(1, 1, 1) // Unit box, will be scaled per instance
    : new THREE.BoxGeometry(cellSize.x, cellSize.y, cellSize.z);
  
  const material = new THREE.MeshPhongMaterial({
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    vertexColors: false, // We'll use instance colors instead
  });
  
  // Create instanced mesh
  const instancedMesh = new THREE.InstancedMesh(boxGeometry, material, cellCount);
  
  // Get the appropriate distance field data
  const fieldData = getDistanceFieldData(gridResult, distanceFieldType);
  const { data: distData, stats: distStats, isBinaryMask } = fieldData;
  const hasDistanceData = distData !== null && distStats !== null;
  
  // Debug logging for boundary mask
  if (distanceFieldType === 'boundary') {
    logDebug('Boundary mask visualization (voxels):', {
      hasData: distData !== null,
      dataLength: distData?.length,
      isBinaryMask,
      hasDistanceData,
      boundaryAdjacentMask: gridResult.boundaryAdjacentMask !== null ? 'exists' : 'null'
    });
    if (distData) {
      let biasedCount = 0, unbiasedCount = 0;
      for (let i = 0; i < distData.length; i++) {
        if (distData[i] === 1) biasedCount++;
        else unbiasedCount++;
      }
      logDebug(`Biased voxels: ${biasedCount}, Unbiased voxels: ${unbiasedCount}`);
    }
  }
  
  // Color gradient: red (close/low) -> teal (far/high)
  // For weighting factor: red (high cost) -> blue (low cost) [inverted]
  // For boundary mask: orange (biased/boundary) vs blue (unbiased/interior)
  const colorClose = new THREE.Color(0xff0000); // Red
  const colorFar = new THREE.Color(0x00ffff);   // Teal (cyan)
  const colorWeightHigh = new THREE.Color(0xff0000); // Red - high cost (costly to walk)
  const colorWeightLow = new THREE.Color(0x0088ff);  // Blue - low cost (cheap to walk)
  const colorBiased = new THREE.Color(0xff8800);   // Orange - biased (boundary-adjacent)
  const colorUnbiased = new THREE.Color(0x0088ff); // Blue - unbiased (interior)
  const fallbackColor = new THREE.Color(color);
  const tempColor = new THREE.Color();
  
  // Set transforms and colors for each instance
  const matrix = new THREE.Matrix4();
  const tempVec = new THREE.Vector3();
  const tempScale = new THREE.Vector3();
  const quaternion = new THREE.Quaternion(); // Identity quaternion (no rotation)
  
  for (let i = 0; i < cellCount; i++) {
    // Get position from flat array or legacy object
    if (useFlat) {
      const i3 = i * 3;
      tempVec.set(
        gridResult.voxelCenters![i3],
        gridResult.voxelCenters![i3 + 1],
        gridResult.voxelCenters![i3 + 2]
      );
    } else {
      const cell = gridResult.moldVolumeCells[i];
      tempVec.copy(cell.center);
    }
    
    // For adaptive grids, set scale based on voxel size
    if (hasVariableSizes && voxelSizes) {
      const voxelSize = voxelSizes[i];
      tempScale.set(voxelSize, voxelSize, voxelSize);
      matrix.compose(tempVec, quaternion, tempScale);
    } else {
      matrix.setPosition(tempVec);
    }
    instancedMesh.setMatrixAt(i, matrix);
    
    if (hasDistanceData) {
      if (isBinaryMask === true) {
        // Binary mask: use distinct colors for biased vs unbiased
        const maskValue = distData![i];
        const isBiased = maskValue === 1;
        tempColor.copy(isBiased ? colorBiased : colorUnbiased);
      } else if (distanceFieldType === 'weight') {
        // Weighting factor: red (high cost) -> blue (low cost)
        // High weight = close to boundary = costly to traverse
        const dist = distData![i];
        const { min, max } = distStats!;
        const range = max - min;
        const t = range > 0 ? (dist - min) / range : 0;
        
        // Interpolate: low weight (t=0) -> blue, high weight (t=1) -> red
        tempColor.lerpColors(colorWeightLow, colorWeightHigh, t);
      } else {
        // Continuous field: interpolate based on normalized value
        const dist = distData![i];
        const { min, max } = distStats!;
        const range = max - min;
        const t = range > 0 ? (dist - min) / range : 0;
        
        // Interpolate between red (close) and teal (far)
        tempColor.lerpColors(colorClose, colorFar, t);
      }
      instancedMesh.setColorAt(i, tempColor);
    } else {
      instancedMesh.setColorAt(i, fallbackColor);
    }
  }
  
  instancedMesh.instanceMatrix.needsUpdate = true;
  if (instancedMesh.instanceColor) {
    instancedMesh.instanceColor.needsUpdate = true;
  }
  
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

// ============================================================================
// DIMENSION LABELS AND SCALE BAR
// ============================================================================

/** Subtle gray color for scale elements */
const SCALE_COLOR = 0x888888;
const SCALE_COLOR_HEX = '#888888';

/**
 * Create a flat text mesh on the XY plane (non-billboard)
 * 
 * @param text - The text to display
 * @param fontSize - Font size in pixels (default: 32)
 * @param color - Text color (default: gray)
 * @param worldScale - Size multiplier for the plane in world units (default: 1.0 for ~1mm height)
 * @returns THREE.Mesh with the text texture on a plane
 */
function createFlatTextLabel(
  text: string,
  fontSize: number = 32,
  color: string = SCALE_COLOR_HEX,
  worldScale: number = 1.0
): THREE.Mesh {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d')!;
  
  // Set font to measure text
  const font = `${fontSize}px Arial, sans-serif`;
  context.font = font;
  const textMetrics = context.measureText(text);
  
  // Add padding
  const padding = fontSize * 0.2;
  const width = Math.ceil(textMetrics.width + padding * 2);
  const height = Math.ceil(fontSize * 1.2 + padding * 2);
  
  // Resize canvas (power of 2 for better texture quality)
  canvas.width = Math.pow(2, Math.ceil(Math.log2(width)));
  canvas.height = Math.pow(2, Math.ceil(Math.log2(height)));
  
  // Clear with transparency
  context.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw text
  context.font = font;
  context.fillStyle = color;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  
  // Create texture
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  
  // Create plane geometry - base size scaled by worldScale
  const aspectRatio = width / height;
  const planeHeight = 5.0 * worldScale; // Base height of 5 units, scaled
  const planeWidth = planeHeight * aspectRatio;
  const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
  
  const material = new THREE.MeshBasicMaterial({ 
    map: texture,
    transparent: true,
    side: THREE.DoubleSide,
    depthTest: true,  // Allow occlusion
    depthWrite: false,
    opacity: 0.9
  });
  
  const mesh = new THREE.Mesh(geometry, material);
  
  return mesh;
}

/**
 * Create a scale ruler with 1mm tick marks
 * 
 * @param lengthMM - Total length of the ruler in mm
 * @param scaleFactor - Scale factor applied to the mesh (scene units per mm)
 * @param position - Position of the ruler start in scene coordinates
 * @param direction - Direction vector for the ruler (normalized)
 * @param label - Axis label (e.g., "X", "Y")
 * @returns THREE.Group containing the ruler
 */
function createScaleRuler(
  lengthMM: number,
  scaleFactor: number,
  position: THREE.Vector3,
  direction: THREE.Vector3,
  label: string
): THREE.Group {
  const group = new THREE.Group();
  group.name = `ScaleRuler_${label}`;
  
  const sceneLength = lengthMM * scaleFactor;
  const mmInScene = scaleFactor; // 1mm in scene units
  
  // For 1:1 scale (scaleFactor=1), lengthMM is also the scene length
  // Scale tick marks and labels based on the ruler length for visibility
  const rulerScale = Math.max(1, lengthMM / 50); // Minimum scale of 1, grows for larger parts
  
  const lineMaterial = new THREE.LineBasicMaterial({ 
    color: SCALE_COLOR, 
    transparent: true,
    opacity: 0.6
  });
  
  // Main ruler line
  const start = position.clone();
  const end = position.clone().add(direction.clone().multiplyScalar(sceneLength));
  const mainLineGeometry = new THREE.BufferGeometry().setFromPoints([start, end]);
  const mainLine = new THREE.Line(mainLineGeometry, lineMaterial);
  group.add(mainLine);
  
  // Perpendicular direction for ticks (on the ground plane)
  const up = new THREE.Vector3(0, 1, 0);
  const perpendicular = new THREE.Vector3().crossVectors(direction, up).normalize();
  if (perpendicular.length() < 0.1) {
    // If direction is vertical, use X as perpendicular
    perpendicular.set(1, 0, 0);
  }
  
  // Tick marks - adaptive spacing based on length
  // For small parts (<20mm): every 1mm, major every 5mm
  // For medium parts (20-100mm): every 5mm, major every 10mm  
  // For large parts (>100mm): every 10mm, major every 50mm
  let tickInterval = 1;
  let majorTickInterval = 5;
  if (lengthMM > 100) {
    tickInterval = 10;
    majorTickInterval = 50;
  } else if (lengthMM > 20) {
    tickInterval = 5;
    majorTickInterval = 10;
  }
  
  const tickPoints: THREE.Vector3[] = [];
  
  for (let mm = 0; mm <= lengthMM; mm += tickInterval) {
    const tickPos = position.clone().add(direction.clone().multiplyScalar(mm * mmInScene));
    const isMajor = mm % majorTickInterval === 0;
    const tickLength = isMajor ? rulerScale * 2 : rulerScale * 1;
    
    const tickStart = tickPos.clone().add(perpendicular.clone().multiplyScalar(-tickLength / 2));
    const tickEnd = tickPos.clone().add(perpendicular.clone().multiplyScalar(tickLength / 2));
    
    tickPoints.push(tickStart, tickEnd);
  }
  
  if (tickPoints.length > 0) {
    const tickGeometry = new THREE.BufferGeometry().setFromPoints(tickPoints);
    const tickLines = new THREE.LineSegments(tickGeometry, lineMaterial);
    group.add(tickLines);
  }
  
  // End caps (vertical lines at start and end)
  const capHeight = rulerScale * 3;
  const startCap1 = start.clone().add(perpendicular.clone().multiplyScalar(-capHeight / 2));
  const startCap2 = start.clone().add(perpendicular.clone().multiplyScalar(capHeight / 2));
  const endCap1 = end.clone().add(perpendicular.clone().multiplyScalar(-capHeight / 2));
  const endCap2 = end.clone().add(perpendicular.clone().multiplyScalar(capHeight / 2));
  
  const capGeometry = new THREE.BufferGeometry().setFromPoints([startCap1, startCap2, endCap1, endCap2]);
  const capLines = new THREE.LineSegments(capGeometry, lineMaterial);
  group.add(capLines);
  
  // Dimension label at the end - scale based on ruler length
  const dimensionText = `${lengthMM.toFixed(1)} mm`;
  const dimensionLabel = createFlatTextLabel(dimensionText, 32, SCALE_COLOR_HEX, rulerScale);
  const labelOffset = perpendicular.clone().multiplyScalar(rulerScale * 4);
  dimensionLabel.position.copy(end).add(labelOffset);
  // Rotate to lie flat on XZ plane (ground)
  dimensionLabel.rotation.x = -Math.PI / 2;
  // Align with direction
  if (Math.abs(direction.z) > 0.5) {
    dimensionLabel.rotation.z = Math.PI / 2;
  }
  group.add(dimensionLabel);
  
  // Axis label at start
  const axisLabel = createFlatTextLabel(label, 36, SCALE_COLOR_HEX, rulerScale);
  const axisLabelOffset = perpendicular.clone().multiplyScalar(-rulerScale * 4);
  axisLabel.position.copy(start).add(axisLabelOffset);
  axisLabel.rotation.x = -Math.PI / 2;
  if (Math.abs(direction.z) > 0.5) {
    axisLabel.rotation.z = Math.PI / 2;
  }
  group.add(axisLabel);
  
  return group;
}

/**
 * Create an axis indicator (like CAD software origin indicator)
 * 
 * @param meshSize - Maximum dimension of the mesh in mm (for scaling the indicator)
 * @param position - Position of the origin
 * @returns THREE.Group containing axis arrows
 */
function createAxisIndicator(
  meshSize: number,
  position: THREE.Vector3
): THREE.Group {
  const group = new THREE.Group();
  group.name = 'AxisIndicator';
  
  // Scale axis indicator based on mesh size (5% of max dimension, min 3mm)
  const axisLength = Math.max(3, meshSize * 0.05);
  const arrowHeadLength = axisLength * 0.2;
  const arrowHeadWidth = axisLength * 0.1;
  
  // X axis (red) - subtle
  const xDir = new THREE.Vector3(1, 0, 0);
  const xArrow = new THREE.ArrowHelper(xDir, position, axisLength, 0xff4444, arrowHeadLength, arrowHeadWidth);
  (xArrow.line.material as THREE.LineBasicMaterial).transparent = true;
  (xArrow.line.material as THREE.LineBasicMaterial).opacity = 0.5;
  (xArrow.cone.material as THREE.MeshBasicMaterial).transparent = true;
  (xArrow.cone.material as THREE.MeshBasicMaterial).opacity = 0.5;
  group.add(xArrow);
  
  // Y axis (green) - pointing up in scene (STL Z)
  const yDir = new THREE.Vector3(0, 1, 0);
  const yArrow = new THREE.ArrowHelper(yDir, position, axisLength, 0x44ff44, arrowHeadLength, arrowHeadWidth);
  (yArrow.line.material as THREE.LineBasicMaterial).transparent = true;
  (yArrow.line.material as THREE.LineBasicMaterial).opacity = 0.5;
  (yArrow.cone.material as THREE.MeshBasicMaterial).transparent = true;
  (yArrow.cone.material as THREE.MeshBasicMaterial).opacity = 0.5;
  group.add(yArrow);
  
  // Z axis (blue) - STL Y direction
  const zDir = new THREE.Vector3(0, 0, 1);
  const zArrow = new THREE.ArrowHelper(zDir, position, axisLength, 0x4444ff, arrowHeadLength, arrowHeadWidth);
  (zArrow.line.material as THREE.LineBasicMaterial).transparent = true;
  (zArrow.line.material as THREE.LineBasicMaterial).opacity = 0.5;
  (zArrow.cone.material as THREE.MeshBasicMaterial).transparent = true;
  (zArrow.cone.material as THREE.MeshBasicMaterial).opacity = 0.5;
  group.add(zArrow);
  
  return group;
}

/**
 * Create dimension labels showing the bounding box size in mm
 * 
 * This creates a subtle industry-standard visualization with:
 * - Scale rulers on X and Y axes (on ground plane) with adaptive tick marks
 * - Small axis indicator at origin
 * - All elements can be occluded by scene objects
 * 
 * @param boundingBox - Original bounding box in mm (before scaling)
 * @param scaleFactor - Scale factor applied to the mesh (1.0 for 1:1 scale)
 * @returns THREE.Group containing all dimension labels and scale bar
 */
export function createDimensionLabels(
  boundingBox: THREE.Box3,
  scaleFactor: number
): THREE.Group {
  const group = new THREE.Group();
  group.name = 'DimensionLabels';
  
  // Get original size in mm
  const sizeInMM = new THREE.Vector3();
  boundingBox.getSize(sizeInMM);
  const maxDimension = Math.max(sizeInMM.x, sizeInMM.y, sizeInMM.z);
  
  // The mesh is rotated -90° around X, so:
  // STL X -> Scene X
  // STL Y -> Scene -Z (but we treat it as +Z for simplicity)
  // STL Z -> Scene Y
  // And scaled uniformly
  
  const scaledSize = new THREE.Vector3(
    sizeInMM.x * scaleFactor,
    sizeInMM.z * scaleFactor, // STL Z becomes scene Y (height)
    sizeInMM.y * scaleFactor  // STL Y becomes scene Z (depth)
  );
  
  // Position at the base of the mesh (assuming centered at origin, bottom at Y=0)
  const baseY = 0.01; // Slightly above ground to avoid z-fighting (larger for 1:1 scale)
  const halfX = scaledSize.x / 2;
  const halfZ = scaledSize.z / 2;
  
  // Offset for scale rulers (outside the bounding box) - scale with mesh size
  const rulerOffset = Math.max(2, maxDimension * 0.05) * scaleFactor; // 5% of max dimension, min 2mm
  
  // X scale ruler (along X axis, at -Z edge)
  const xRulerStart = new THREE.Vector3(-halfX, baseY, -halfZ - rulerOffset);
  const xRuler = createScaleRuler(
    sizeInMM.x,
    scaleFactor,
    xRulerStart,
    new THREE.Vector3(1, 0, 0),
    'X'
  );
  group.add(xRuler);
  
  // Y scale ruler (STL Y = scene Z, along Z axis at -X edge)
  // Note: In STL, Y is depth, which becomes scene Z after rotation
  const yRulerStart = new THREE.Vector3(-halfX - rulerOffset, baseY, -halfZ);
  const yRuler = createScaleRuler(
    sizeInMM.y, // STL Y dimension
    scaleFactor,
    yRulerStart,
    new THREE.Vector3(0, 0, 1),
    'Y'
  );
  group.add(yRuler);
  
  // Axis indicator at corner (origin reference)
  const axisOrigin = new THREE.Vector3(-halfX - rulerOffset * 0.5, baseY, -halfZ - rulerOffset * 0.5);
  const axisIndicator = createAxisIndicator(maxDimension, axisOrigin);
  group.add(axisIndicator);
  
  return group;
}

/**
 * Remove dimension labels from the scene
 * 
 * @param scene - The THREE.js scene
 * @param labelsGroup - The dimension labels group to remove
 */
export function removeDimensionLabels(
  scene: THREE.Scene,
  labelsGroup: THREE.Group
): void {
  scene.remove(labelsGroup);
  
  // Dispose of all geometries, materials, and textures
  labelsGroup.traverse((child) => {
    if (child instanceof THREE.Line || child instanceof THREE.LineSegments) {
      child.geometry.dispose();
      if (child.material instanceof THREE.Material) {
        child.material.dispose();
      }
    }
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose();
      if (child.material instanceof THREE.Material) {
        if ((child.material as THREE.MeshBasicMaterial).map) {
          (child.material as THREE.MeshBasicMaterial).map!.dispose();
        }
        child.material.dispose();
      }
    }
    if (child instanceof THREE.Sprite) {
      if (child.material.map) {
        child.material.map.dispose();
      }
      child.material.dispose();
    }
    // ArrowHelper cleanup
    if (child instanceof THREE.ArrowHelper) {
      if (child.line) {
        child.line.geometry.dispose();
        (child.line.material as THREE.Material).dispose();
      }
      if (child.cone) {
        child.cone.geometry.dispose();
        (child.cone.material as THREE.Material).dispose();
      }
    }
  });
}

/**
 * Create a visualization line showing R: the maximum distance from any voxel to the part mesh
 * 
 * This creates a thick line from the voxel with the maximum distance (rLineStart) to 
 * the closest point on the part mesh (rLineEnd), with spheres at both endpoints for visibility.
 * 
 * @param gridResult - The volumetric grid result containing R and line endpoints
 * @param lineColor - Color for the line (default: magenta)
 * @param sphereColor - Color for the endpoint spheres (default: same as line)
 * @returns THREE.Group containing the line and spheres, or null if R line not available
 */
export function createRLineVisualization(
  gridResult: VolumetricGridResult,
  lineColor: THREE.ColorRepresentation = 0xff00ff,
  sphereColor?: THREE.ColorRepresentation
): THREE.Group | null {
  const { rLineStart, rLineEnd, R } = gridResult;
  
  if (!rLineStart || !rLineEnd || R <= 0) {
    console.warn('R line visualization not available: missing endpoints or invalid R');
    return null;
  }
  
  const group = new THREE.Group();
  group.name = 'RLineVisualization';
  
  // Create the line geometry
  const lineGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array([
    rLineStart.x, rLineStart.y, rLineStart.z,
    rLineEnd.x, rLineEnd.y, rLineEnd.z
  ]);
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  
  // Create a thick line using LineBasicMaterial
  const lineMaterial = new THREE.LineBasicMaterial({ 
    color: lineColor, 
    linewidth: 3  // Note: linewidth > 1 only works on some systems
  });
  const line = new THREE.Line(lineGeometry, lineMaterial);
  line.name = 'RLine';
  group.add(line);
  
  // Calculate sphere size based on R (very small, subtle visualization)
  const sphereRadius = Math.max(R * 0.0025, 0.03);
  
  // Create spheres at endpoints
  const sphereGeometry = new THREE.SphereGeometry(sphereRadius, 8, 8);
  const sphereMaterial = new THREE.MeshBasicMaterial({ 
    color: sphereColor ?? lineColor 
  });
  
  // Start sphere (voxel with max distance) - magenta
  const startSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
  startSphere.position.copy(rLineStart);
  startSphere.name = 'RLineStart_Voxel';
  group.add(startSphere);
  
  // End sphere (closest point on part mesh) - green
  const endSphere = new THREE.Mesh(sphereGeometry, sphereMaterial.clone());
  (endSphere.material as THREE.MeshBasicMaterial).color.setHex(0x00ff00); // Green for part mesh point
  endSphere.position.copy(rLineEnd);
  endSphere.name = 'RLineEnd_PartMesh';
  group.add(endSphere);
  
  // Add a cylinder for better visibility (tube connecting the points)
  const direction = new THREE.Vector3().subVectors(rLineEnd, rLineStart);
  const length = direction.length();
  direction.normalize();
  
  const tubeRadius = sphereRadius * 0.5;
  const tubeGeometry = new THREE.CylinderGeometry(tubeRadius, tubeRadius, length, 6);
  const tubeMaterial = new THREE.MeshBasicMaterial({ 
    color: lineColor,
    transparent: true,
    opacity: 0.7
  });
  const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
  
  // Position and orient the tube
  const midpoint = new THREE.Vector3().addVectors(rLineStart, rLineEnd).multiplyScalar(0.5);
  tube.position.copy(midpoint);
  
  // Orient tube along the direction
  const up = new THREE.Vector3(0, 1, 0);
  const quaternion = new THREE.Quaternion();
  quaternion.setFromUnitVectors(up, direction);
  tube.setRotationFromQuaternion(quaternion);
  tube.name = 'RLineTube';
  group.add(tube);
  
  logDebug(`R Line visualization created: R = ${R.toFixed(4)}`);
  
  return group;
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
  /** Surface tolerance for including boundary voxels (0 = strict containment) */
  surfaceTolerance?: number;
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
  
  // Clone geometries to avoid modifying originals
  const shellGeom = outerShellGeometry.clone();
  const partGeom = partGeometry.clone();
  
  // Ensure geometries have computed bounds
  shellGeom.computeBoundingBox();
  partGeom.computeBoundingBox();
  
  // Get bounding box from outer shell
  const boundingBox = shellGeom.boundingBox!.clone();
  
  // Parse options
  const marginPercent = options.marginPercent ?? 0.05;
  const includeSurfaceVoxels = options.includeSurfaceVoxels ?? false;
  const targetVoxelSizePercent = options.targetVoxelSizePercent ?? DEFAULT_TARGET_VOXEL_SIZE_PERCENT;
  
  // Auto-calculate resolution if not provided
  let resolution: number | THREE.Vector3;
  if (options.resolution !== undefined) {
    resolution = options.resolution;
  } else {
    // Calculate adaptive resolution based on bounding box diagonal
    resolution = calculateAdaptiveResolution(boundingBox, targetVoxelSizePercent);
  }
  
  // Determine resolution per axis
  let resX: number, resY: number, resZ: number;
  if (typeof resolution === 'number') {
    resX = resY = resZ = resolution;
  } else {
    resX = resolution.x;
    resY = resolution.y;
    resZ = resolution.z;
  }
  
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
  
  // Calculate surface tolerance: half the diagonal of a cell (when surface voxels are enabled)
  const surfaceTolerance = includeSurfaceVoxels 
    ? Math.sqrt(cellSize.x * cellSize.x + cellSize.y * cellSize.y + cellSize.z * cellSize.z) / 2
    : 0;
  
  if (includeSurfaceVoxels) {
    logDebug(`[Parallel] Surface voxel inclusion enabled, tolerance: ${surfaceTolerance.toFixed(6)} (centers must be outside part)`);
  }
  
  const totalCellCount = resX * resY * resZ;
  
  // Extract geometry data for workers
  const shellData = extractGeometryData(shellGeom);
  const partData = extractGeometryData(partGeom);
  
  // Create and initialize workers
  const workers: Worker[] = [];
  const workerReadyPromises: Promise<void>[] = [];
  
  logDebug(`Initializing ${numWorkers} grid workers...`);
  
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
      surfaceTolerance,
    };
    
    const transfers: Transferable[] = [initMsg.shellPositionArray.buffer, initMsg.partPositionArray.buffer];
    if (initMsg.shellIndexArray) transfers.push(initMsg.shellIndexArray.buffer);
    if (initMsg.partIndexArray) transfers.push(initMsg.partIndexArray.buffer);
    
    worker.postMessage(initMsg, transfers);
  }
  
  await Promise.all(workerReadyPromises);
  logDebug('Grid workers initialized');
  
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
  
  // Create flat arrays only (moldVolumeCells deprecated)
  const voxelCount = moldVolumeCellIndexSet.size;
  const voxelCenters = new Float32Array(voxelCount * 3);
  const voxelIndices = new Uint32Array(voxelCount * 3);
  
  let moldIdx = 0;
  for (const cellIdx of moldVolumeCellIndexSet) {
    // Decode cell index back to i, j, k
    const i = Math.floor(cellIdx / (resY * resZ));
    const remainder = cellIdx % (resY * resZ);
    const j = Math.floor(remainder / resZ);
    const k = remainder % resZ;
    
    const cx = boundingBox.min.x + (i + 0.5) * cellSize.x;
    const cy = boundingBox.min.y + (j + 0.5) * cellSize.y;
    const cz = boundingBox.min.z + (k + 0.5) * cellSize.z;
    
    // Store in flat arrays (primary storage)
    const i3 = moldIdx * 3;
    voxelCenters[i3] = cx;
    voxelCenters[i3 + 1] = cy;
    voxelCenters[i3 + 2] = cz;
    voxelIndices[i3] = i;
    voxelIndices[i3 + 1] = j;
    voxelIndices[i3 + 2] = k;
    
    moldIdx++;
  }
  
  const voxelizationEndTime = performance.now();
  logTiming('Parallel voxelization', voxelizationEndTime - startTime, `${voxelCount} mold volume cells`);
  
  // ========================================================================
  // PARALLEL DISTANCE FIELD COMPUTATION
  // Use workers to compute distances in parallel
  // ========================================================================
  
  const distanceStartTime = performance.now();
  // voxelCount is already defined above when creating flat arrays
  const voxelDist = new Float32Array(voxelCount);
  const voxelDistToShell = new Float32Array(voxelCount);
  
  // Create voxel centers array for workers (reuse from flat arrays)
  const voxelCentersArray = voxelCenters;
  
  // Initialize distance workers
  const distanceWorkers: Worker[] = [];
  const distanceWorkerReadyPromises: Promise<void>[] = [];
  
  logDebug(`[Parallel] Initializing ${numWorkers} distance workers...`);
  
  for (let i = 0; i < numWorkers; i++) {
    const worker = new Worker(
      new URL('./distanceFieldWorker.ts', import.meta.url),
      { type: 'module' }
    );
    distanceWorkers.push(worker);
    
    distanceWorkerReadyPromises.push(new Promise<void>((resolve) => {
      const handler = (event: MessageEvent) => {
        if (event.data.type === 'ready') {
          worker.removeEventListener('message', handler);
          resolve();
        }
      };
      worker.addEventListener('message', handler);
    }));
    
    // Send init message with geometry data
    const initMsg = {
      type: 'init',
      workerId: i,
      partPositionArray: partData.positionArray.slice(),
      partIndexArray: partData.indexArray?.slice() ?? null,
      shellPositionArray: shellData.positionArray.slice(),
      shellIndexArray: shellData.indexArray?.slice() ?? null,
    };
    
    const transfers: Transferable[] = [initMsg.partPositionArray.buffer, initMsg.shellPositionArray.buffer];
    if (initMsg.partIndexArray) transfers.push(initMsg.partIndexArray.buffer);
    if (initMsg.shellIndexArray) transfers.push(initMsg.shellIndexArray.buffer);
    
    worker.postMessage(initMsg, transfers);
  }
  
  await Promise.all(distanceWorkerReadyPromises);
  logDebug('[Parallel] Distance workers initialized');
  
  // Distribute voxels across workers
  const voxelsPerWorker = Math.ceil(voxelCount / numWorkers);
  const distancePromises: Promise<{ startIndex: number; partDistances: Float32Array; shellDistances: Float32Array }>[] = [];
  
  for (let w = 0; w < numWorkers; w++) {
    const startIdx = w * voxelsPerWorker;
    const endIdx = Math.min(startIdx + voxelsPerWorker, voxelCount);
    if (endIdx <= startIdx) continue;
    
    const voxelCenters = voxelCentersArray.slice(startIdx * 3, endIdx * 3);
    
    distancePromises.push(new Promise((resolve) => {
      const handler = (event: MessageEvent) => {
        if (event.data.type === 'result' && event.data.workerId === w) {
          distanceWorkers[w].removeEventListener('message', handler);
          resolve({
            startIndex: event.data.startIndex,
            partDistances: event.data.partDistances,
            shellDistances: event.data.shellDistances,
          });
        }
      };
      distanceWorkers[w].addEventListener('message', handler);
    }));
    
    distanceWorkers[w].postMessage({
      type: 'compute',
      voxelCenters,
      startIndex: startIdx,
    }, [voxelCenters.buffer]);
  }
  
  // Wait for all distance computations
  const distanceResults = await Promise.all(distancePromises);
  
  // Terminate distance workers
  distanceWorkers.forEach(w => w.terminate());
  
  // Merge results into the distance arrays
  for (const result of distanceResults) {
    const { startIndex, partDistances, shellDistances } = result;
    for (let i = 0; i < partDistances.length; i++) {
      voxelDist[startIndex + i] = partDistances[i];
      voxelDistToShell[startIndex + i] = shellDistances[i];
    }
  }
  
  // Find min/max and R line visualization data
  let minDist = Infinity;
  let maxDist = -Infinity;
  let maxDistIdx = -1;
  let rLineStart: THREE.Vector3 | null = null;
  let rLineEnd: THREE.Vector3 | null = null;
  
  for (let i = 0; i < voxelCount; i++) {
    const dist = voxelDist[i];
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) {
      maxDist = dist;
      maxDistIdx = i;
    }
  }
  
  // Get R line endpoints (need to recompute closest point for the max distance voxel)
  if (maxDistIdx >= 0) {
    const partTester = new MeshTester(partGeom, 'part');
    const i3 = maxDistIdx * 3;
    const center = new THREE.Vector3(
      voxelCenters[i3],
      voxelCenters[i3 + 1],
      voxelCenters[i3 + 2]
    );
    const target = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
    partTester.closestPointToPoint(center, target);
    rLineStart = center;
    rLineEnd = target.point.clone();
    partTester.dispose();
  }
  
  const R = maxDist;
  const distanceStats = { min: minDist, max: maxDist };
  
  // Shell distance stats
  let minShellDist = Infinity;
  let maxShellDist = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const dist = voxelDistToShell[i];
    if (dist < minShellDist) minShellDist = dist;
    if (dist > maxShellDist) maxShellDist = dist;
  }
  
  const shellDistanceStats = { min: minShellDist, max: maxShellDist };
  
  const distanceEndTime = performance.now();
  logTiming('Parallel distance fields', distanceEndTime - distanceStartTime, `R=${R.toFixed(6)}`);
  
  // ========================================================================
  // BOUNDARY DETECTION AND BIASED DISTANCE COMPUTATION
  // ========================================================================
  
  // Build spatial index for neighbor lookups using flat arrays
  const voxelSpatialIndex = new Map<string, number>();
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    const gi = voxelIndices[i3];
    const gj = voxelIndices[i3 + 1];
    const gk = voxelIndices[i3 + 2];
    const key = `${gi},${gj},${gk}`;
    voxelSpatialIndex.set(key, i);
  }
  
  const getVoxelIdx = (gi: number, gj: number, gk: number): number => {
    const key = `${gi},${gj},${gk}`;
    return voxelSpatialIndex.get(key) ?? -1;
  };
  
  // 6-connected neighbor offsets
  const faceNeighborOffsets: [number, number, number][] = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
  ];
  
  // Find ALL surface voxels
  const isSurfaceVoxel = new Uint8Array(voxelCount);
  let totalSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    const gi = voxelIndices[i3];
    const gj = voxelIndices[i3 + 1];
    const gk = voxelIndices[i3 + 2];
    
    for (const [di, dj, dk] of faceNeighborOffsets) {
      const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
      if (neighborIdx < 0) {
        isSurfaceVoxel[i] = 1;
        totalSurfaceCount++;
        break;
      }
    }
  }
  
  // Distinguish inner vs outer surface using distance-to-part
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
  
  const surfaceMinDist = sortedPairs[0]?.dist || 0;
  const surfaceMaxDist = sortedPairs[sortedPairs.length - 1]?.dist || 0;
  
  let maxGap = 0;
  let gapIndex = Math.floor(sortedPairs.length / 2);
  
  for (let i = 1; i < sortedPairs.length; i++) {
    const gap = sortedPairs[i].dist - sortedPairs[i - 1].dist;
    if (gap > maxGap) {
      maxGap = gap;
      gapIndex = i;
    }
  }
  
  const distanceThreshold = sortedPairs[gapIndex]?.dist || (surfaceMinDist + surfaceMaxDist) / 2;
  
  // Mark OUTER surface voxels
  const isOuterSurface = new Uint8Array(voxelCount);
  
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i] && voxelDist[i] >= distanceThreshold) {
      isOuterSurface[i] = 1;
    }
  }
  
  // Build mask for boundary voxels (outer surface + BOUNDARY_DEPTH_LEVELS levels deep)
  const isBoundaryAdjacent = new Uint8Array(voxelCount);
  let boundaryAdjacentCount = 0;
  
  // Mark outer surface (level 0)
  for (let i = 0; i < voxelCount; i++) {
    if (isOuterSurface[i]) {
      isBoundaryAdjacent[i] = 1;
      boundaryAdjacentCount++;
    }
  }
  
  // Expand outward for each level
  for (let level = 1; level <= BOUNDARY_DEPTH_LEVELS; level++) {
    const previouslyMarked = new Uint8Array(isBoundaryAdjacent);
    
    for (let i = 0; i < voxelCount; i++) {
      if (previouslyMarked[i] && (level === 1 ? isOuterSurface[i] : true)) {
        const i3 = i * 3;
        const gi = voxelIndices[i3];
        const gj = voxelIndices[i3 + 1];
        const gk = voxelIndices[i3 + 2];
        
        for (const [di, dj, dk] of faceNeighborOffsets) {
          const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
          if (neighborIdx >= 0 && !isBoundaryAdjacent[neighborIdx]) {
            isBoundaryAdjacent[neighborIdx] = 1;
            boundaryAdjacentCount++;
          }
        }
      }
    }
  }
  
  // Compute biased distances
  const w1 = DEFAULT_BIASED_DISTANCE_WEIGHTS.partDistanceWeight;
  const w2 = DEFAULT_BIASED_DISTANCE_WEIGHTS.shellBiasWeight;
  
  const biasedDist = new Float32Array(voxelCount);
  let minBiasedDist = Infinity;
  let maxBiasedDist = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const delta_i = voxelDist[i];
    
    let biased: number;
    if (isBoundaryAdjacent[i]) {
      const delta_w = voxelDistToShell[i];
      const lambda_w = R - delta_w;
      biased = w1 * delta_i + w2 * lambda_w;
    } else {
      biased = w1 * delta_i;
    }
    
    biasedDist[i] = biased;
    
    if (biased < minBiasedDist) minBiasedDist = biased;
    if (biased > maxBiasedDist) maxBiasedDist = biased;
  }
  
  const biasedDistanceStats = { min: minBiasedDist, max: maxBiasedDist };
  
  // Compute weighting factors
  const weightingFactor = new Float32Array(voxelCount);
  let minWeight = Infinity;
  let maxWeight = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const bd = biasedDist[i];
    const wt = 1.0 / (bd * bd + 0.25);
    
    weightingFactor[i] = wt;
    
    if (wt < minWeight) minWeight = wt;
    if (wt > maxWeight) maxWeight = wt;
  }
  
  const weightingFactorStats = { min: minWeight, max: maxWeight };
  
  // Clean up geometries
  shellGeom.dispose();
  partGeom.dispose();
  
  const endTime = performance.now();
  
  // Calculate statistics
  const cellVolume = cellSize.x * cellSize.y * cellSize.z;
  const moldVolume = voxelCount * cellVolume;
  const totalVolume = totalCellCount * cellVolume;
  
  const stats: VolumetricGridStats = {
    moldVolume,
    totalVolume,
    fillRatio: moldVolume / totalVolume,
    computeTimeMs: endTime - startTime,
  };
  
  logResult('Parallel Grid Complete', { moldCells: voxelCount, totalCells: totalCellCount, timeMs: stats.computeTimeMs.toFixed(1) });
  
  return {
    allCells: [],
    moldVolumeCells: [], // Deprecated - use voxelCenters/voxelIndices instead
    voxelCenters,
    voxelIndices,
    resolution: new THREE.Vector3(resX, resY, resZ),
    totalCellCount,
    moldVolumeCellCount: voxelCount,
    boundingBox,
    cellSize,
    stats,
    voxelDist,
    voxelDistToShell,
    biasedDist,
    weightingFactor,
    distanceStats,
    shellDistanceStats,
    biasedDistanceStats,
    weightingFactorStats,
    R,
    rLineStart,
    rLineEnd,
    boundaryAdjacentMask: isBoundaryAdjacent,
    seedVoxelMask: null,
  };
}

// ============================================================================
// BIASED DISTANCE RECALCULATION WITH WEIGHTS
// ============================================================================

/** Default weights for biased distance calculation */
export const DEFAULT_BIASED_DISTANCE_WEIGHTS: BiasedDistanceWeights = {
  partDistanceWeight: 1.0,
  shellBiasWeight: 1.0,
};

/**
 * Recalculate biased distances and weighting factors with custom weights.
 * This is much faster than regenerating the entire grid since it only
 * recomputes the biased distance field using the already-computed distance fields.
 * 
 * Formula: biasedDist = w₁ * δ_i + w₂ * (R - δ_w)
 * Where:
 *   - w₁ = partDistanceWeight (weight for distance to part)
 *   - w₂ = shellBiasWeight (weight for shell bias term)
 *   - δ_i = distance to part mesh
 *   - δ_w = distance to shell
 *   - R = maximum distance from any voxel to part mesh
 * 
 * Note: The bias is only applied to boundary-adjacent voxels (outer surface + one level deep).
 * Interior voxels use just w₁ * δ_i (no shell bias).
 * 
 * @param gridResult - The existing volumetric grid result
 * @param weights - Custom weights for the biased distance calculation
 * @returns Updated VolumetricGridResult with recalculated biasedDist and weightingFactor
 */
export function recalculateBiasedDistances(
  gridResult: VolumetricGridResult,
  weights: BiasedDistanceWeights = DEFAULT_BIASED_DISTANCE_WEIGHTS
): VolumetricGridResult {
  const { voxelDist, voxelDistToShell, boundaryAdjacentMask, R } = gridResult;
  
  if (!voxelDist || !voxelDistToShell || !boundaryAdjacentMask) {
    logInfo('Cannot recalculate biased distances: missing distance field data');
    return gridResult;
  }
  
  const voxelCount = voxelDist.length;
  const { partDistanceWeight: w1, shellBiasWeight: w2 } = weights;
  
  logInfo(`Recalculating biased distances: w₁=${w1.toFixed(2)}, w₂=${w2.toFixed(2)}, R=${R.toFixed(6)}, voxels=${voxelCount}`);
  
  // Recalculate biased distances
  const biasedDist = new Float32Array(voxelCount);
  let minBiasedDist = Infinity;
  let maxBiasedDist = -Infinity;
  
  let boundaryCount = 0;
  let interiorCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const delta_i = voxelDist[i];
    
    let biased: number;
    if (boundaryAdjacentMask[i]) {
      // Boundary-adjacent: apply weighted bias
      const delta_w = voxelDistToShell[i];
      const lambda_w = R - delta_w;
      biased = w1 * delta_i + w2 * lambda_w;
      boundaryCount++;
    } else {
      // Interior: only part distance term (no shell bias)
      biased = w1 * delta_i;
      interiorCount++;
    }
    
    biasedDist[i] = biased;
    
    if (biased < minBiasedDist) minBiasedDist = biased;
    if (biased > maxBiasedDist) maxBiasedDist = biased;
  }
  
  const biasedDistanceStats = { min: minBiasedDist, max: maxBiasedDist };
  
  logResult('Biased distance recalculated', { boundary: boundaryCount, interior: interiorCount, minBiasedDist, maxBiasedDist });
  
  // Recalculate weighting factors
  const weightingFactor = new Float32Array(voxelCount);
  let minWeight = Infinity;
  let maxWeight = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const bd = biasedDist[i];
    const wt = 1.0 / (bd * bd + 0.25);
    
    weightingFactor[i] = wt;
    
    if (wt < minWeight) minWeight = wt;
    if (wt > maxWeight) maxWeight = wt;
  }
  
  const weightingFactorStats = { min: minWeight, max: maxWeight };
  
  logResult('Weighting factor recalculated', { minWeight, maxWeight });
  
  // Return updated result (preserving all other fields)
  return {
    ...gridResult,
    biasedDist,
    weightingFactor,
    biasedDistanceStats,
    weightingFactorStats,
  };
}

// ============================================================================
// VERTEX AND EDGE SEED VOXELIZATION
// ============================================================================

export interface VertexSeedVoxelizationOptions extends VolumetricGridOptions {
  /**
   * Minimum distance from part surface for a mold volume voxel to be included.
   * Voxels closer than this distance are excluded from the mold volume (but seeds are always included).
   * Default: 0 (include voxels that touch the surface)
   */
  minDistanceFromPart?: number;
}

/**
 * Generate a volumetric grid using vertex and edge-based seed placement.
 * 
 * This approach places seed voxels at:
 * 1. Every vertex of the part mesh
 * 2. Along every edge of the part mesh (sampled at intervals smaller than voxel size)
 * 
 * The rest of the mold volume is voxelized normally, ensuring all non-seed voxels
 * are strictly outside the part surface.
 * 
 * Benefits:
 * - Guaranteed seed coverage at all part surface features
 * - No volume intersection computation needed for seed detection
 * - Edge sampling ensures continuous seed coverage along mesh edges
 * - More robust for complex geometry with thin features
 * 
 * @param outerShellGeometry - Geometry of the inflated outer shell (∂H)
 * @param partGeometry - Geometry of the original part mesh (M)
 * @param options - Grid generation options
 * @returns Promise<VolumetricGridResult> with seedVoxelMask populated
 */
export async function generateVolumetricGridWithVertexSeeds(
  outerShellGeometry: THREE.BufferGeometry,
  partGeometry: THREE.BufferGeometry,
  options: VertexSeedVoxelizationOptions = {}
): Promise<VolumetricGridResult> {
  const startTime = performance.now();
  
  // Clone geometries to avoid modifying originals
  const shellGeom = outerShellGeometry.clone();
  const partGeom = partGeometry.clone();
  
  // Ensure geometries have computed bounds
  shellGeom.computeBoundingBox();
  partGeom.computeBoundingBox();
  
  // Get bounding box from outer shell (this defines the grid extent)
  const boundingBox = shellGeom.boundingBox!.clone();
  
  // Parse options
  const marginPercent = options.marginPercent ?? 0.05;
  const voxelsPerUnit = options.voxelsPerUnit ?? DEFAULT_VOXELS_PER_UNIT;
  const minDistanceFromPart = options.minDistanceFromPart ?? 0;
  
  // Add margin to bounding box FIRST (before calculating resolution)
  const boxSize = new THREE.Vector3();
  boundingBox.getSize(boxSize);
  const margin = boxSize.clone().multiplyScalar(marginPercent);
  boundingBox.min.sub(margin);
  boundingBox.max.add(margin);
  
  // Recompute size after margin
  boundingBox.getSize(boxSize);
  
  // Auto-calculate resolution if not provided
  // Priority: resolution > voxelsPerUnit > targetVoxelSizePercent
  let resX: number, resY: number, resZ: number;
  if (options.resolution !== undefined) {
    // Use explicit resolution
    if (typeof options.resolution === 'number') {
      resX = resY = resZ = options.resolution;
    } else {
      resX = options.resolution.x;
      resY = options.resolution.y;
      resZ = options.resolution.z;
    }
  } else {
    // Use voxelsPerUnit to calculate resolution (default: 1 voxel/mm)
    const resolution = calculateResolutionFromDensity(boundingBox, voxelsPerUnit);
    resX = resolution.x;
    resY = resolution.y;
    resZ = resolution.z;
  }
  
  // Calculate cell size
  const cellSize = new THREE.Vector3(
    boxSize.x / resX,
    boxSize.y / resY,
    boxSize.z / resZ
  );
  
  logInfo(`Vertex/Edge Seed Grid: resolution=${resX}x${resY}x${resZ}, cellSize=(${cellSize.x.toFixed(4)}, ${cellSize.y.toFixed(4)}, ${cellSize.z.toFixed(4)})`);
  
  // Create BVH testers for both meshes
  const shellTester = new MeshTester(shellGeom, 'shell');
  const partTester = new MeshTester(partGeom, 'part');
  
  const totalCellCount = resX * resY * resZ;
  
  // ========================================================================
  // STEP 1: Create seed voxels at part mesh vertices AND along edges
  // ========================================================================
  
  logDebug('Creating seed voxels at part mesh vertices and edges...');
  
  const seedPositions = partGeom.getAttribute('position');
  const seedIndex = partGeom.getIndex();
  const numVertices = seedPositions.count;
  
  // Map from grid index key to seed voxel data
  // Using a Map to deduplicate voxels when multiple vertices/edge points map to same grid cell
  const seedVoxelMap = new Map<string, { center: THREE.Vector3; gridIndex: THREE.Vector3 }>();
  
  // Helper to add a seed voxel at a world position
  const addSeedVoxel = (worldX: number, worldY: number, worldZ: number): void => {
    // Convert world position to grid indices
    const gi = Math.floor((worldX - boundingBox.min.x) / cellSize.x);
    const gj = Math.floor((worldY - boundingBox.min.y) / cellSize.y);
    const gk = Math.floor((worldZ - boundingBox.min.z) / cellSize.z);
    
    // Clamp to valid range
    const clampedI = Math.max(0, Math.min(resX - 1, gi));
    const clampedJ = Math.max(0, Math.min(resY - 1, gj));
    const clampedK = Math.max(0, Math.min(resZ - 1, gk));
    
    const key = `${clampedI},${clampedJ},${clampedK}`;
    
    // Only add if not already present
    if (!seedVoxelMap.has(key)) {
      // Calculate voxel center position
      const centerX = boundingBox.min.x + (clampedI + 0.5) * cellSize.x;
      const centerY = boundingBox.min.y + (clampedJ + 0.5) * cellSize.y;
      const centerZ = boundingBox.min.z + (clampedK + 0.5) * cellSize.z;
      
      seedVoxelMap.set(key, {
        center: new THREE.Vector3(centerX, centerY, centerZ),
        gridIndex: new THREE.Vector3(clampedI, clampedJ, clampedK)
      });
    }
  };
  
  // Add seed voxels at all vertices
  for (let v = 0; v < numVertices; v++) {
    addSeedVoxel(
      seedPositions.getX(v),
      seedPositions.getY(v),
      seedPositions.getZ(v)
    );
  }
  
  const vertexSeedCount = seedVoxelMap.size;
  logDebug(`Vertex seeds: ${vertexSeedCount} from ${numVertices} vertices`);
  
  // Add seed voxels along all edges
  // Calculate step size based on voxel diagonal to ensure no gaps
  const voxelDiagonal = Math.sqrt(cellSize.x * cellSize.x + cellSize.y * cellSize.y + cellSize.z * cellSize.z);
  const edgeStepSize = voxelDiagonal * 0.5; // Half diagonal ensures overlap
  
  // Track unique edges to avoid processing duplicates
  const processedEdges = new Set<string>();
  
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const edgeDir = new THREE.Vector3();
  const samplePoint = new THREE.Vector3();
  
  if (seedIndex) {
    // Indexed geometry - iterate through triangles
    const numTriangles = seedIndex.count / 3;
    
    for (let t = 0; t < numTriangles; t++) {
      const i0 = seedIndex.getX(t * 3);
      const i1 = seedIndex.getX(t * 3 + 1);
      const i2 = seedIndex.getX(t * 3 + 2);
      
      // Process each edge of the triangle
      const edges: [number, number][] = [[i0, i1], [i1, i2], [i2, i0]];
      
      for (const [a, b] of edges) {
        // Create canonical edge key (smaller index first)
        const edgeKey = a < b ? `${a}-${b}` : `${b}-${a}`;
        
        if (processedEdges.has(edgeKey)) continue;
        processedEdges.add(edgeKey);
        
        // Get edge vertices
        v0.set(seedPositions.getX(a), seedPositions.getY(a), seedPositions.getZ(a));
        v1.set(seedPositions.getX(b), seedPositions.getY(b), seedPositions.getZ(b));
        
        // Calculate edge length and direction
        edgeDir.subVectors(v1, v0);
        const edgeLength = edgeDir.length();
        
        if (edgeLength < edgeStepSize) continue; // Skip very short edges
        
        edgeDir.normalize();
        
        // Sample points along the edge (excluding endpoints - they're already added as vertices)
        const numSamples = Math.floor(edgeLength / edgeStepSize);
        for (let s = 1; s < numSamples; s++) {
          const t = s / numSamples;
          samplePoint.copy(v0).addScaledVector(edgeDir, t * edgeLength);
          addSeedVoxel(samplePoint.x, samplePoint.y, samplePoint.z);
        }
      }
    }
  } else {
    // Non-indexed geometry - every 3 vertices form a triangle
    const numTriangles = numVertices / 3;
    
    for (let t = 0; t < numTriangles; t++) {
      const i0 = t * 3;
      const i1 = t * 3 + 1;
      const i2 = t * 3 + 2;
      
      // Process each edge of the triangle
      const edges: [number, number][] = [[i0, i1], [i1, i2], [i2, i0]];
      
      for (const [a, b] of edges) {
        // Get edge vertices
        v0.set(seedPositions.getX(a), seedPositions.getY(a), seedPositions.getZ(a));
        v1.set(seedPositions.getX(b), seedPositions.getY(b), seedPositions.getZ(b));
        
        // Calculate edge length and direction
        edgeDir.subVectors(v1, v0);
        const edgeLength = edgeDir.length();
        
        if (edgeLength < edgeStepSize) continue;
        
        edgeDir.normalize();
        
        // Sample points along the edge
        const numSamples = Math.floor(edgeLength / edgeStepSize);
        for (let s = 1; s < numSamples; s++) {
          const t = s / numSamples;
          samplePoint.copy(v0).addScaledVector(edgeDir, t * edgeLength);
          addSeedVoxel(samplePoint.x, samplePoint.y, samplePoint.z);
        }
      }
    }
  }
  
  const totalSeedCount = seedVoxelMap.size;
  const edgeSeedCount = totalSeedCount - vertexSeedCount;
  logDebug(`Edge seeds: ${edgeSeedCount} from ${processedEdges.size} edges`);
  logDebug(`Total seed voxels: ${totalSeedCount}`);
  
  // ========================================================================
  // STEP 2: Create mold volume voxels (outside part surface)
  // ========================================================================
  
  logDebug('Creating mold volume voxels...');
  
  // Temporary arrays for collecting voxels
  const tempCenters: number[] = [];
  const tempIndices: number[] = [];
  const tempIsSeed: number[] = []; // 1 if seed, 0 if mold volume
  
  const cellCenter = new THREE.Vector3();
  
  // First, add all seed voxels
  for (const [_key, seedData] of seedVoxelMap) {
    tempCenters.push(seedData.center.x, seedData.center.y, seedData.center.z);
    tempIndices.push(seedData.gridIndex.x, seedData.gridIndex.y, seedData.gridIndex.z);
    tempIsSeed.push(1);
  }
  
  const seedCount = seedVoxelMap.size;
  
  // Iterate through all grid cells for mold volume
  let moldVolumeCount = 0;
  let skippedInsidePart = 0;
  let skippedOutsideShell = 0;
  let skippedTooCloseToPart = 0;
  let skippedDuplicateSeed = 0;
  let processedCells = 0;
  
  for (let k = 0; k < resZ; k++) {
    // Yield to event loop once per Z-slice to keep UI responsive
    if (k > 0) {
      await yieldToEventLoop();
    }
    
    for (let j = 0; j < resY; j++) {
      for (let i = 0; i < resX; i++) {
        processedCells++;
        const key = `${i},${j},${k}`;
        
        // Skip if this is already a seed voxel
        if (seedVoxelMap.has(key)) {
          skippedDuplicateSeed++;
          continue;
        }
        
        // Calculate cell center position
        cellCenter.set(
          boundingBox.min.x + (i + 0.5) * cellSize.x,
          boundingBox.min.y + (j + 0.5) * cellSize.y,
          boundingBox.min.z + (k + 0.5) * cellSize.z
        );
        
        // Check if inside shell
        const isInsideShell = shellTester.isInside(cellCenter);
        if (!isInsideShell) {
          skippedOutsideShell++;
          continue;
        }
        
        // Check if outside part (must be strictly outside)
        const isOutsidePart = partTester.isOutside(cellCenter);
        if (!isOutsidePart) {
          skippedInsidePart++;
          continue;
        }
        
        // Optional: Check minimum distance from part surface
        if (minDistanceFromPart > 0) {
          const distToPart = partTester.getDistanceToSurface(cellCenter);
          if (distToPart < minDistanceFromPart) {
            skippedTooCloseToPart++;
            continue;
          }
        }
        
        // This is a valid mold volume voxel
        tempCenters.push(cellCenter.x, cellCenter.y, cellCenter.z);
        tempIndices.push(i, j, k);
        tempIsSeed.push(0);
        moldVolumeCount++;
      }
    }
  }
  
  logDebug(`Mold volume voxels: ${moldVolumeCount}`);
  logDebug(`Skipped: ${skippedInsidePart} inside part, ${skippedOutsideShell} outside shell, ${skippedTooCloseToPart} too close to part, ${skippedDuplicateSeed} duplicate seeds`);
  
  // Convert to typed arrays
  const voxelCenters = new Float32Array(tempCenters);
  const voxelIndices = new Uint32Array(tempIndices);
  const seedVoxelMask = new Uint8Array(tempIsSeed);
  const voxelCount = voxelCenters.length / 3;
  
  logResult('Vertex seed grid', { totalVoxels: voxelCount, seeds: seedCount, moldVolume: moldVolumeCount });
  
  // ========================================================================
  // STEP 3: Compute distance fields
  // ========================================================================
  
  logDebug('Computing distance fields...');
  
  const voxelDist = new Float32Array(voxelCount);
  const voxelDistToShell = new Float32Array(voxelCount);
  
  let minDist = Infinity;
  let maxDist = -Infinity;
  let minShellDist = Infinity;
  let maxShellDist = -Infinity;
  let rLineStart: THREE.Vector3 | null = null;
  let rLineEnd: THREE.Vector3 | null = null;
  
  const closestPoint = new THREE.Vector3();
  const target = { 
    point: closestPoint, 
    distance: Infinity,
    faceIndex: 0
  };
  const tempCenter = new THREE.Vector3();
  
  for (let i = 0; i < voxelCount; i++) {
    // Yield periodically to keep UI responsive
    if (i % YIELD_BATCH_SIZE === 0 && i > 0) {
      await yieldToEventLoop();
    }
    
    const i3 = i * 3;
    tempCenter.set(voxelCenters[i3], voxelCenters[i3 + 1], voxelCenters[i3 + 2]);
    
    // Distance to part
    target.distance = Infinity;
    partTester.closestPointToPoint(tempCenter, target);
    const distPart = target.distance;
    voxelDist[i] = distPart;
    
    if (distPart < minDist) minDist = distPart;
    if (distPart > maxDist) {
      maxDist = distPart;
      rLineStart = tempCenter.clone();
      rLineEnd = target.point.clone();
    }
    
    // Distance to shell
    target.distance = Infinity;
    shellTester.closestPointToPoint(tempCenter, target);
    const distShell = target.distance;
    voxelDistToShell[i] = distShell;
    
    if (distShell < minShellDist) minShellDist = distShell;
    if (distShell > maxShellDist) maxShellDist = distShell;
  }
  
  const R = maxDist;
  const distanceStats = { min: minDist, max: maxDist };
  const shellDistanceStats = { min: minShellDist, max: maxShellDist };
  
  logResult('Distance field δ_i', { minDist, maxDist, R });
  logResult('Distance field δ_w (to shell)', { minShellDist, maxShellDist });
  
  // ========================================================================
  // STEP 4: Identify boundary voxels (outer surface + one level deep)
  // ========================================================================
  
  logDebug('Identifying boundary voxels for biased weighting...');
  
  // Build spatial index
  const voxelSpatialIndex = new Map<string, number>();
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    const gi = voxelIndices[i3];
    const gj = voxelIndices[i3 + 1];
    const gk = voxelIndices[i3 + 2];
    voxelSpatialIndex.set(`${gi},${gj},${gk}`, i);
  }
  
  const getVoxelIdx = (gi: number, gj: number, gk: number): number => {
    return voxelSpatialIndex.get(`${gi},${gj},${gk}`) ?? -1;
  };
  
  // 6-connected neighbor offsets
  const faceNeighborOffsets: [number, number, number][] = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
  ];
  
  // Find all surface voxels (those with at least one missing 6-neighbor)
  const isSurfaceVoxel = new Uint8Array(voxelCount);
  let totalSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    const i3 = i * 3;
    const gi = voxelIndices[i3];
    const gj = voxelIndices[i3 + 1];
    const gk = voxelIndices[i3 + 2];
    
    for (const [di, dj, dk] of faceNeighborOffsets) {
      const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
      if (neighborIdx < 0) {
        isSurfaceVoxel[i] = 1;
        totalSurfaceCount++;
        break;
      }
    }
  }
  
  // Distinguish inner vs outer surface using distance to part
  // Seed voxels (at part surface) should have low distance
  // Outer surface voxels (near shell) should have high distance
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
  
  // Find largest gap to separate inner from outer
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
  
  // Mark outer surface voxels
  const isOuterSurface = new Uint8Array(voxelCount);
  let outerSurfaceCount = 0;
  
  for (let i = 0; i < voxelCount; i++) {
    if (isSurfaceVoxel[i] && voxelDist[i] >= distanceThreshold) {
      isOuterSurface[i] = 1;
      outerSurfaceCount++;
    }
  }
  
  logDebug(`Surface voxels: ${totalSurfaceCount} total, ${outerSurfaceCount} outer (near shell)`);
  
  // Build boundary adjacent mask (outer surface + BOUNDARY_DEPTH_LEVELS levels deep)
  const isBoundaryAdjacent = new Uint8Array(voxelCount);
  let boundaryAdjacentCount = 0;
  
  // Mark outer surface voxels (level 0)
  for (let i = 0; i < voxelCount; i++) {
    if (isOuterSurface[i]) {
      isBoundaryAdjacent[i] = 1;
      boundaryAdjacentCount++;
    }
  }
  
  // Expand outward for each level (1 to BOUNDARY_DEPTH_LEVELS)
  for (let level = 1; level <= BOUNDARY_DEPTH_LEVELS; level++) {
    const previouslyMarked = new Uint8Array(isBoundaryAdjacent);
    
    for (let i = 0; i < voxelCount; i++) {
      if (previouslyMarked[i] && (level === 1 ? isOuterSurface[i] : true)) {
        const i3 = i * 3;
        const gi = voxelIndices[i3];
        const gj = voxelIndices[i3 + 1];
        const gk = voxelIndices[i3 + 2];
        
        for (const [di, dj, dk] of faceNeighborOffsets) {
          const neighborIdx = getVoxelIdx(gi + di, gj + dj, gk + dk);
          if (neighborIdx >= 0 && !isBoundaryAdjacent[neighborIdx]) {
            isBoundaryAdjacent[neighborIdx] = 1;
            boundaryAdjacentCount++;
          }
        }
      }
    }
  }
  
  logResult('Boundary detection', { boundaryVoxels: boundaryAdjacentCount, interiorVoxels: voxelCount - boundaryAdjacentCount, depthLevels: BOUNDARY_DEPTH_LEVELS });
  
  // ========================================================================
  // STEP 5: Compute biased distance and weighting factors
  // ========================================================================
  
  const w1 = DEFAULT_BIASED_DISTANCE_WEIGHTS.partDistanceWeight;
  const w2 = DEFAULT_BIASED_DISTANCE_WEIGHTS.shellBiasWeight;
  
  const biasedDist = new Float32Array(voxelCount);
  let minBiasedDist = Infinity;
  let maxBiasedDist = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const delta_i = voxelDist[i];
    
    let biased: number;
    if (isBoundaryAdjacent[i]) {
      const delta_w = voxelDistToShell[i];
      const lambda_w = R - delta_w;
      biased = w1 * delta_i + w2 * lambda_w;
    } else {
      biased = w1 * delta_i;
    }
    
    biasedDist[i] = biased;
    
    if (biased < minBiasedDist) minBiasedDist = biased;
    if (biased > maxBiasedDist) maxBiasedDist = biased;
  }
  
  const biasedDistanceStats = { min: minBiasedDist, max: maxBiasedDist };
  
  // Weighting factors
  const weightingFactor = new Float32Array(voxelCount);
  let minWeight = Infinity;
  let maxWeight = -Infinity;
  
  for (let i = 0; i < voxelCount; i++) {
    const bd = biasedDist[i];
    const wt = 1.0 / (bd * bd + 0.25);
    
    weightingFactor[i] = wt;
    
    if (wt < minWeight) minWeight = wt;
    if (wt > maxWeight) maxWeight = wt;
  }
  
  const weightingFactorStats = { min: minWeight, max: maxWeight };
  
  logResult('Biased distance field', { minBiasedDist, maxBiasedDist });
  logResult('Weighting factor', { minWeight, maxWeight });
  
  // Clean up
  shellTester.dispose();
  partTester.dispose();
  shellGeom.dispose();
  partGeom.dispose();
  
  const endTime = performance.now();
  
  // Calculate statistics
  const cellVolume = cellSize.x * cellSize.y * cellSize.z;
  const moldVolume = voxelCount * cellVolume;
  const totalVolume = totalCellCount * cellVolume;
  
  const stats: VolumetricGridStats = {
    moldVolume,
    totalVolume,
    fillRatio: moldVolume / totalVolume,
    computeTimeMs: endTime - startTime,
  };
  
  logTiming('Vertex seed grid generation', endTime - startTime);
  
  return {
    allCells: [],
    moldVolumeCells: [],
    voxelCenters,
    voxelIndices,
    resolution: new THREE.Vector3(resX, resY, resZ),
    totalCellCount,
    moldVolumeCellCount: voxelCount,
    boundingBox,
    cellSize,
    stats,
    voxelDist,
    voxelDistToShell,
    biasedDist,
    weightingFactor,
    distanceStats,
    shellDistanceStats,
    biasedDistanceStats,
    weightingFactorStats,
    R,
    rLineStart,
    rLineEnd,
    boundaryAdjacentMask: isBoundaryAdjacent,
    seedVoxelMask,
  };
}

// ============================================================================
// ADAPTIVE LOD VOXELIZATION
// ============================================================================

/**
 * Generate an adaptive volumetric grid with distance-based Level of Detail (LOD).
 * 
 * This approach uses fine voxels near the part surface and progressively coarser
 * voxels as distance from the part increases. This dramatically reduces memory
 * usage while maintaining precision where it matters most.
 * 
 * Algorithm:
 * 1. Place seed voxels at part vertices/edges at the finest LOD (base voxel size)
 * 2. For the rest of the volume, determine LOD level based on distance to part
 * 3. Voxel size at each LOD level doubles: size_n = baseSize * 2^n
 * 4. Distance thresholds grow exponentially: threshold_n = baseThreshold * growthFactor^n
 * 
 * Benefits:
 * - Dramatically reduced voxel count (typically 80-90% reduction)
 * - Maintains fine detail near part surface
 * - Coarse voxels far from part where precision is less critical
 * - Still computes proper distance fields for all voxels
 * 
 * @param outerShellGeometry - Geometry of the inflated outer shell (∂H)
 * @param partGeometry - Geometry of the original part mesh (M)
 * @param options - Adaptive grid generation options
 * @returns Promise<AdaptiveVoxelGridResult> with LOD information
 */
export async function generateAdaptiveVoxelGrid(
  outerShellGeometry: THREE.BufferGeometry,
  partGeometry: THREE.BufferGeometry,
  options: AdaptiveVoxelizationOptions = {}
): Promise<AdaptiveVoxelGridResult> {
  const startTime = performance.now();
  
  // Clone geometries to avoid modifying originals
  const shellGeom = outerShellGeometry.clone();
  const partGeom = partGeometry.clone();
  
  // Ensure geometries have computed bounds
  shellGeom.computeBoundingBox();
  partGeom.computeBoundingBox();
  
  // Get bounding box from outer shell (this defines the grid extent)
  const boundingBox = shellGeom.boundingBox!.clone();
  
  // Parse options with new linear-friendly defaults
  const marginPercent = options.marginPercent ?? 0.02;
  const baseVoxelsPerUnit = options.baseVoxelsPerUnit ?? options.voxelsPerUnit ?? DEFAULT_VOXELS_PER_UNIT;
  const lodLevels = options.lodLevels ?? 3; // Default to 3 for faster processing
  const sizeFactor = options.sizeFactor ?? 1.5; // Gentler than 2.0 for better Dijkstra continuity
  const distanceScalingMode = options.distanceScalingMode ?? 'linear'; // Linear is better for walk algorithms
  const distanceScale = options.distanceScale ?? 1.0;
  const minDistanceFromPart = options.minDistanceFromPart ?? 0;
  
  // Add margin to bounding box
  const boxSize = new THREE.Vector3();
  boundingBox.getSize(boxSize);
  const margin = boxSize.clone().multiplyScalar(marginPercent);
  boundingBox.min.sub(margin);
  boundingBox.max.add(margin);
  
  // Recompute size after margin
  boundingBox.getSize(boxSize);
  const bboxDiagonal = boxSize.length();
  
  // Calculate base cell size (finest level)
  const baseCellSize = 1.0 / baseVoxelsPerUnit;
  
  // Calculate LOD cell sizes using sizeFactor (gentler scaling for better continuity)
  const lodCellSizes: number[] = [];
  for (let level = 0; level < lodLevels; level++) {
    lodCellSizes.push(baseCellSize * Math.pow(sizeFactor, level));
  }
  
  // Calculate or use provided distance thresholds
  let lodDistanceThresholds: number[];
  if (options.lodDistanceThresholds && options.lodDistanceThresholds.length === lodLevels - 1) {
    lodDistanceThresholds = options.lodDistanceThresholds;
  } else {
    // Auto-calculate thresholds based on bounding box diagonal and scaling mode
    // Base threshold is roughly 2% of diagonal
    const baseThreshold = bboxDiagonal * 0.02 * distanceScale;
    lodDistanceThresholds = [];
    
    for (let level = 0; level < lodLevels - 1; level++) {
      const levelNum = level + 1; // 1, 2, 3, ... for the threshold boundaries
      
      if (distanceScalingMode === 'linear') {
        // Linear: thresholds grow linearly with level
        // This gives more uniform voxel density transitions, better for Dijkstra
        lodDistanceThresholds.push(baseThreshold * levelNum);
      } else {
        // Quadratic: thresholds grow quadratically (more aggressive far from part)
        // Saves more memory but creates larger voxel size jumps
        lodDistanceThresholds.push(baseThreshold * levelNum * levelNum);
      }
    }
  }
  
  logInfo(`Adaptive Grid: ${lodLevels} LOD levels, sizeFactor=${sizeFactor}, mode=${distanceScalingMode}`);
  logInfo(`Base cell size=${baseCellSize.toFixed(4)}mm, coarsest=${lodCellSizes[lodLevels-1].toFixed(4)}mm`);
  logInfo(`LOD thresholds (mm from part): ${lodDistanceThresholds.map(t => t.toFixed(2)).join(', ')}`);
  logInfo(`LOD cell sizes (mm): ${lodCellSizes.map(s => s.toFixed(4)).join(', ')}`);
  
  // Create BVH testers for both meshes
  const shellTester = new MeshTester(shellGeom, 'shell');
  const partTester = new MeshTester(partGeom, 'part');
  
  // ========================================================================
  // STEP 1: Create seed voxels at part mesh vertices AND along edges (LOD 0)
  // ========================================================================
  
  logDebug('Creating seed voxels at part mesh vertices and edges (LOD 0)...');
  
  const seedPositions = partGeom.getAttribute('position');
  const seedIndex = partGeom.getIndex();
  const numVertices = seedPositions.count;
  
  // Storage for all voxels across all LOD levels
  // Using arrays initially, will convert to typed arrays at the end
  interface VoxelData {
    center: THREE.Vector3;
    lodLevel: number;
    cellSize: number;
    distToPart: number;
    distToShell: number;
    isSeed: boolean;
  }
  
  const allVoxels: VoxelData[] = [];
  
  // Map from position key to voxel index (for deduplication)
  // Using spatial hashing at each LOD level
  const voxelMaps: Map<string, number>[] = [];
  for (let level = 0; level < lodLevels; level++) {
    voxelMaps.push(new Map<string, number>());
  }
  
  // Helper to get position key for a given LOD level
  const getPosKey = (x: number, y: number, z: number, cellSize: number): string => {
    const gi = Math.floor((x - boundingBox.min.x) / cellSize);
    const gj = Math.floor((y - boundingBox.min.y) / cellSize);
    const gk = Math.floor((z - boundingBox.min.z) / cellSize);
    return `${gi},${gj},${gk}`;
  };
  
  // Helper to get voxel center for a grid position at a given LOD level
  const getVoxelCenterForGrid = (gi: number, gj: number, gk: number, cellSize: number): THREE.Vector3 => {
    return new THREE.Vector3(
      boundingBox.min.x + (gi + 0.5) * cellSize,
      boundingBox.min.y + (gj + 0.5) * cellSize,
      boundingBox.min.z + (gk + 0.5) * cellSize
    );
  };
  
  // Helper to add a seed voxel at a world position (always LOD 0)
  const addSeedVoxel = (worldX: number, worldY: number, worldZ: number): void => {
    const cellSize = lodCellSizes[0];
    const key = getPosKey(worldX, worldY, worldZ, cellSize);
    
    // Only add if not already present at LOD 0
    if (!voxelMaps[0].has(key)) {
      const gi = Math.floor((worldX - boundingBox.min.x) / cellSize);
      const gj = Math.floor((worldY - boundingBox.min.y) / cellSize);
      const gk = Math.floor((worldZ - boundingBox.min.z) / cellSize);
      
      const center = getVoxelCenterForGrid(gi, gj, gk, cellSize);
      
      // Get distances (needed for later calculations)
      const target = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
      partTester.closestPointToPoint(center, target);
      const distToPart = target.distance;
      
      target.distance = Infinity;
      shellTester.closestPointToPoint(center, target);
      const distToShell = target.distance;
      
      const voxelIdx = allVoxels.length;
      voxelMaps[0].set(key, voxelIdx);
      
      allVoxels.push({
        center,
        lodLevel: 0,
        cellSize,
        distToPart,
        distToShell,
        isSeed: true
      });
    }
  };
  
  // Add seed voxels at all vertices
  for (let v = 0; v < numVertices; v++) {
    addSeedVoxel(
      seedPositions.getX(v),
      seedPositions.getY(v),
      seedPositions.getZ(v)
    );
  }
  
  const vertexSeedCount = allVoxels.length;
  logDebug(`Vertex seeds: ${vertexSeedCount} from ${numVertices} vertices`);
  
  // Add seed voxels along all edges
  const voxelDiagonal = Math.sqrt(3) * baseCellSize;
  const edgeStepSize = voxelDiagonal * 0.5;
  
  const processedEdges = new Set<string>();
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const edgeDir = new THREE.Vector3();
  const samplePoint = new THREE.Vector3();
  
  if (seedIndex) {
    const numTriangles = seedIndex.count / 3;
    
    for (let t = 0; t < numTriangles; t++) {
      const i0 = seedIndex.getX(t * 3);
      const i1 = seedIndex.getX(t * 3 + 1);
      const i2 = seedIndex.getX(t * 3 + 2);
      
      const edges: [number, number][] = [[i0, i1], [i1, i2], [i2, i0]];
      
      for (const [a, b] of edges) {
        const edgeKey = a < b ? `${a}-${b}` : `${b}-${a}`;
        
        if (processedEdges.has(edgeKey)) continue;
        processedEdges.add(edgeKey);
        
        v0.set(seedPositions.getX(a), seedPositions.getY(a), seedPositions.getZ(a));
        v1.set(seedPositions.getX(b), seedPositions.getY(b), seedPositions.getZ(b));
        
        edgeDir.subVectors(v1, v0);
        const edgeLength = edgeDir.length();
        
        if (edgeLength < edgeStepSize) continue;
        
        edgeDir.normalize();
        
        const numSamples = Math.floor(edgeLength / edgeStepSize);
        for (let s = 1; s < numSamples; s++) {
          const t = s / numSamples;
          samplePoint.copy(v0).addScaledVector(edgeDir, t * edgeLength);
          addSeedVoxel(samplePoint.x, samplePoint.y, samplePoint.z);
        }
      }
    }
  } else {
    const numTriangles = numVertices / 3;
    
    for (let t = 0; t < numTriangles; t++) {
      const i0 = t * 3;
      const i1 = t * 3 + 1;
      const i2 = t * 3 + 2;
      
      const edges: [number, number][] = [[i0, i1], [i1, i2], [i2, i0]];
      
      for (const [a, b] of edges) {
        v0.set(seedPositions.getX(a), seedPositions.getY(a), seedPositions.getZ(a));
        v1.set(seedPositions.getX(b), seedPositions.getY(b), seedPositions.getZ(b));
        
        edgeDir.subVectors(v1, v0);
        const edgeLength = edgeDir.length();
        
        if (edgeLength < edgeStepSize) continue;
        
        edgeDir.normalize();
        
        const numSamples = Math.floor(edgeLength / edgeStepSize);
        for (let s = 1; s < numSamples; s++) {
          const t = s / numSamples;
          samplePoint.copy(v0).addScaledVector(edgeDir, t * edgeLength);
          addSeedVoxel(samplePoint.x, samplePoint.y, samplePoint.z);
        }
      }
    }
  }
  
  const totalSeedCount = allVoxels.length;
  const edgeSeedCount = totalSeedCount - vertexSeedCount;
  logDebug(`Edge seeds: ${edgeSeedCount} from ${processedEdges.size} edges`);
  logDebug(`Total seed voxels (LOD 0): ${totalSeedCount}`);
  
  // ========================================================================
  // STEP 2: Fill mold volume with adaptive LOD voxels
  // ========================================================================
  
  logDebug('Creating adaptive LOD voxels for mold volume...');
  
  // We'll process from coarsest LOD to finest, skipping cells that are already
  // Determine LOD level based on distance to part
  const getLODLevel = (distToPart: number): number => {
    for (let level = 0; level < lodLevels - 1; level++) {
      if (distToPart < lodDistanceThresholds[level]) {
        return level;
      }
    }
    return lodLevels - 1; // Coarsest level
  };
  
  const target = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const cellCenter = new THREE.Vector3();
  
  // ========================================================================
  // STEP 2: Flood-fill outward from seed voxels to fill the mold volume
  // This is MUCH faster than iterating all cells because we only visit
  // neighbors of existing voxels, and most of the grid is empty space.
  // ========================================================================
  
  logDebug('Flood-filling mold volume from seed voxels...');
  
  const voxelCountPerLevel: number[] = new Array(lodLevels).fill(0);
  voxelCountPerLevel[0] = totalSeedCount;
  
  // Use a queue for BFS flood-fill, starting from all seed voxels
  // Each queue entry: [worldX, worldY, worldZ, sourceVoxelIdx]
  const fillQueue: Array<[number, number, number, number]> = [];
  
  // 6-connected neighbor offsets (face neighbors only for efficiency)
  const neighborOffsets: [number, number, number][] = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
  ];
  
  // Initialize queue with all seed voxel neighbors
  const baseCellSizeLOD0 = lodCellSizes[0];
  for (let seedIdx = 0; seedIdx < totalSeedCount; seedIdx++) {
    const seedVoxel = allVoxels[seedIdx];
    const cx = seedVoxel.center.x;
    const cy = seedVoxel.center.y;
    const cz = seedVoxel.center.z;
    
    for (const [dx, dy, dz] of neighborOffsets) {
      fillQueue.push([
        cx + dx * baseCellSizeLOD0,
        cy + dy * baseCellSizeLOD0,
        cz + dz * baseCellSizeLOD0,
        seedIdx
      ]);
    }
  }
  
  // Track visited positions at each LOD level to avoid reprocessing
  const visitedAtLevel: Set<string>[] = [];
  for (let level = 0; level < lodLevels; level++) {
    visitedAtLevel.push(new Set<string>());
  }
  // Mark all seed positions as visited at LOD 0
  for (const key of voxelMaps[0].keys()) {
    visitedAtLevel[0].add(key);
  }
  
  let addedCount = 0;
  let skippedInsidePart = 0;
  let skippedOutsideShell = 0;
  let skippedVisited = 0;
  let processedCount = 0;
  
  // Pre-compute bounds
  const minX = boundingBox.min.x;
  const minY = boundingBox.min.y;
  const minZ = boundingBox.min.z;
  const maxX = boundingBox.max.x;
  const maxY = boundingBox.max.y;
  const maxZ = boundingBox.max.z;
  
  while (fillQueue.length > 0) {
    // Yield to event loop periodically to keep UI responsive
    processedCount++;
    if (processedCount % YIELD_BATCH_SIZE === 0) {
      await yieldToEventLoop();
    }
    
    const [wx, wy, wz, _sourceIdx] = fillQueue.shift()!;
    
    // Skip if outside bounding box
    if (wx < minX || wx > maxX || wy < minY || wy > maxY || wz < minZ || wz > maxZ) {
      continue;
    }
    
    // First, get distance to part to determine LOD level
    cellCenter.set(wx, wy, wz);
    target.distance = Infinity;
    partTester.closestPointToPoint(cellCenter, target);
    const distToPart = target.distance;
    
    // Determine appropriate LOD level for this distance
    const level = getLODLevel(distToPart);
    const cellSize = lodCellSizes[level];
    
    // Get grid key at this LOD level
    const gi = Math.floor((wx - minX) / cellSize);
    const gj = Math.floor((wy - minY) / cellSize);
    const gk = Math.floor((wz - minZ) / cellSize);
    const key = `${gi},${gj},${gk}`;
    
    // Skip if already visited at this LOD level
    if (visitedAtLevel[level].has(key)) {
      skippedVisited++;
      continue;
    }
    visitedAtLevel[level].add(key);
    
    // Calculate actual voxel center for this grid cell
    const voxelCx = minX + (gi + 0.5) * cellSize;
    const voxelCy = minY + (gj + 0.5) * cellSize;
    const voxelCz = minZ + (gk + 0.5) * cellSize;
    cellCenter.set(voxelCx, voxelCy, voxelCz);
    
    // Check if inside shell (BVH accelerated)
    const isInsideShell = shellTester.isInside(cellCenter);
    if (!isInsideShell) {
      skippedOutsideShell++;
      continue;
    }
    
    // Check if outside part (BVH accelerated)
    const isOutsidePart = partTester.isOutside(cellCenter);
    if (!isOutsidePart) {
      skippedInsidePart++;
      continue;
    }
    
    // Re-compute distance at the actual voxel center
    target.distance = Infinity;
    partTester.closestPointToPoint(cellCenter, target);
    const actualDistToPart = target.distance;
    
    // Check minimum distance requirement
    if (actualDistToPart < minDistanceFromPart) {
      continue;
    }
    
    // Get distance to shell
    target.distance = Infinity;
    shellTester.closestPointToPoint(cellCenter, target);
    const distToShell = target.distance;
    
    // Add the voxel
    const voxelIdx = allVoxels.length;
    voxelMaps[level].set(key, voxelIdx);
    
    allVoxels.push({
      center: cellCenter.clone(),
      lodLevel: level,
      cellSize,
      distToPart: actualDistToPart,
      distToShell,
      isSeed: false
    });
    
    addedCount++;
    voxelCountPerLevel[level]++;
    
    // Add neighbors to queue (at current LOD level's cell size)
    for (const [dx, dy, dz] of neighborOffsets) {
      fillQueue.push([
        voxelCx + dx * cellSize,
        voxelCy + dy * cellSize,
        voxelCz + dz * cellSize,
        voxelIdx
      ]);
    }
  }
  
  logDebug(`Flood-fill complete: added ${addedCount}, skipped (inside part: ${skippedInsidePart}, outside shell: ${skippedOutsideShell}, visited: ${skippedVisited})`);
  
  const totalVoxels = allVoxels.length;
  logResult('Adaptive voxel grid', { 
    totalVoxels, 
    lodLevels,
    voxelCountPerLevel: voxelCountPerLevel.join(', ')
  });
  
  // ========================================================================
  // STEP 3: Convert to typed arrays
  // ========================================================================
  
  logDebug('Converting to typed arrays...');
  
  const voxelCenters = new Float32Array(totalVoxels * 3);
  const voxelIndices = new Uint32Array(totalVoxels * 3);
  const seedVoxelMask = new Uint8Array(totalVoxels);
  const voxelLOD = new Uint8Array(totalVoxels);
  const voxelSizes = new Float32Array(totalVoxels);
  const voxelDist = new Float32Array(totalVoxels);
  const voxelDistToShell = new Float32Array(totalVoxels);
  
  let minDist = Infinity;
  let maxDist = -Infinity;
  let minShellDist = Infinity;
  let maxShellDist = -Infinity;
  let rLineStart: THREE.Vector3 | null = null;
  let rLineEnd: THREE.Vector3 | null = null;
  
  for (let i = 0; i < totalVoxels; i++) {
    // Yield periodically to keep UI responsive
    if (i % YIELD_BATCH_SIZE === 0 && i > 0) {
      await yieldToEventLoop();
    }
    
    const voxel = allVoxels[i];
    const i3 = i * 3;
    
    voxelCenters[i3] = voxel.center.x;
    voxelCenters[i3 + 1] = voxel.center.y;
    voxelCenters[i3 + 2] = voxel.center.z;
    
    // Calculate grid indices at the base (finest) resolution for compatibility
    const baseGi = Math.floor((voxel.center.x - boundingBox.min.x) / baseCellSize);
    const baseGj = Math.floor((voxel.center.y - boundingBox.min.y) / baseCellSize);
    const baseGk = Math.floor((voxel.center.z - boundingBox.min.z) / baseCellSize);
    voxelIndices[i3] = baseGi;
    voxelIndices[i3 + 1] = baseGj;
    voxelIndices[i3 + 2] = baseGk;
    
    seedVoxelMask[i] = voxel.isSeed ? 1 : 0;
    voxelLOD[i] = voxel.lodLevel;
    voxelSizes[i] = voxel.cellSize;
    
    voxelDist[i] = voxel.distToPart;
    voxelDistToShell[i] = voxel.distToShell;
    
    if (voxel.distToPart < minDist) minDist = voxel.distToPart;
    if (voxel.distToPart > maxDist) {
      maxDist = voxel.distToPart;
      rLineStart = voxel.center.clone();
      // Get the closest point on part for R line visualization
      target.distance = Infinity;
      partTester.closestPointToPoint(voxel.center, target);
      rLineEnd = target.point.clone();
    }
    
    if (voxel.distToShell < minShellDist) minShellDist = voxel.distToShell;
    if (voxel.distToShell > maxShellDist) maxShellDist = voxel.distToShell;
  }
  
  const R = maxDist;
  const distanceStats = { min: minDist, max: maxDist };
  const shellDistanceStats = { min: minShellDist, max: maxShellDist };
  
  logResult('Distance field δ_i', { minDist, maxDist, R });
  logResult('Distance field δ_w (to shell)', { minShellDist, maxShellDist });
  
  // ========================================================================
  // STEP 4: Compute biased distance and weighting factors
  // ========================================================================
  
  logDebug('Computing biased distances and weighting factors...');
  
  // For adaptive grid, boundary voxels are those close to the shell (outer surface)
  // We use a distance threshold based on BOUNDARY_DEPTH_LEVELS × voxel size
  const boundaryAdjacentMask = new Uint8Array(totalVoxels);
  
  // Boundary threshold: voxels within BOUNDARY_DEPTH_LEVELS voxel-widths from shell
  // Use baseCellSize as reference since outer boundary voxels are typically LOD 0
  const shellBoundaryThreshold = baseCellSize * (BOUNDARY_DEPTH_LEVELS + 1);
  
  let boundaryCount = 0;
  for (let i = 0; i < totalVoxels; i++) {
    // Mark as boundary if close to shell (within threshold distance)
    if (voxelDistToShell[i] < shellBoundaryThreshold) {
      boundaryAdjacentMask[i] = 1;
      boundaryCount++;
    }
  }
  
  logResult('Boundary detection (adaptive)', { boundaryVoxels: boundaryCount, interiorVoxels: totalVoxels - boundaryCount, depthLevels: BOUNDARY_DEPTH_LEVELS });
  
  const w1 = DEFAULT_BIASED_DISTANCE_WEIGHTS.partDistanceWeight;
  const w2 = DEFAULT_BIASED_DISTANCE_WEIGHTS.shellBiasWeight;
  
  const biasedDist = new Float32Array(totalVoxels);
  let minBiasedDist = Infinity;
  let maxBiasedDist = -Infinity;
  
  for (let i = 0; i < totalVoxels; i++) {
    const delta_i = voxelDist[i];
    
    let biased: number;
    if (boundaryAdjacentMask[i]) {
      const delta_w = voxelDistToShell[i];
      const lambda_w = R - delta_w;
      biased = w1 * delta_i + w2 * lambda_w;
    } else {
      biased = w1 * delta_i;
    }
    
    biasedDist[i] = biased;
    
    if (biased < minBiasedDist) minBiasedDist = biased;
    if (biased > maxBiasedDist) maxBiasedDist = biased;
  }
  
  const biasedDistanceStats = { min: minBiasedDist, max: maxBiasedDist };
  
  // Weighting factors
  const weightingFactor = new Float32Array(totalVoxels);
  let minWeight = Infinity;
  let maxWeight = -Infinity;
  
  for (let i = 0; i < totalVoxels; i++) {
    const bd = biasedDist[i];
    const wt = 1.0 / (bd * bd + 0.25);
    
    weightingFactor[i] = wt;
    
    if (wt < minWeight) minWeight = wt;
    if (wt > maxWeight) maxWeight = wt;
  }
  
  const weightingFactorStats = { min: minWeight, max: maxWeight };
  
  logResult('Biased distance field', { minBiasedDist, maxBiasedDist });
  logResult('Weighting factor', { minWeight, maxWeight });
  
  // Clean up
  shellTester.dispose();
  partTester.dispose();
  shellGeom.dispose();
  partGeom.dispose();
  
  const endTime = performance.now();
  
  // Calculate statistics
  // For adaptive grid, volume is the sum of individual voxel volumes
  let moldVolume = 0;
  for (let i = 0; i < totalVoxels; i++) {
    const size = voxelSizes[i];
    moldVolume += size * size * size;
  }
  
  // Calculate base resolution (for compatibility)
  const baseResX = Math.ceil(boxSize.x / baseCellSize);
  const baseResY = Math.ceil(boxSize.y / baseCellSize);
  const baseResZ = Math.ceil(boxSize.z / baseCellSize);
  const totalCellCount = baseResX * baseResY * baseResZ;
  const totalVolume = boxSize.x * boxSize.y * boxSize.z;
  
  const stats: VolumetricGridStats = {
    moldVolume,
    totalVolume,
    fillRatio: moldVolume / totalVolume,
    computeTimeMs: endTime - startTime,
  };
  
  const lodStats = {
    voxelCountPerLevel,
    voxelSizePerLevel: lodCellSizes,
    distanceThresholds: lodDistanceThresholds
  };
  
  logTiming('Adaptive voxel grid generation', endTime - startTime);
  
  // Calculate reduction compared to uniform grid
  const uniformVoxelCount = totalCellCount;
  const reductionPercent = ((uniformVoxelCount - totalVoxels) / uniformVoxelCount * 100).toFixed(1);
  logResult('Voxel reduction', { 
    uniformWouldBe: uniformVoxelCount, 
    adaptive: totalVoxels, 
    reduction: `${reductionPercent}%` 
  });
  
  return {
    allCells: [],
    moldVolumeCells: [],
    voxelCenters,
    voxelIndices,
    resolution: new THREE.Vector3(baseResX, baseResY, baseResZ),
    totalCellCount,
    moldVolumeCellCount: totalVoxels,
    boundingBox,
    cellSize: new THREE.Vector3(baseCellSize, baseCellSize, baseCellSize), // Base cell size
    stats,
    voxelDist,
    voxelDistToShell,
    biasedDist,
    weightingFactor,
    distanceStats,
    shellDistanceStats,
    biasedDistanceStats,
    weightingFactorStats,
    R,
    rLineStart,
    rLineEnd,
    boundaryAdjacentMask,
    seedVoxelMask,
    // Adaptive-specific fields
    voxelLOD,
    voxelSizes,
    lodStats
  };
}