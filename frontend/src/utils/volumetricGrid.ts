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
  /** 
   * Distance field: distance from each mold volume voxel center to part mesh M.
   * Indexed by voxel ID (same order as moldVolumeCells array).
   * Uses BVH-accelerated nearest-point queries for efficiency.
   */
  voxelDist: Float32Array;
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
  /** Minimum distance from any voxel to part mesh */
  minDist: number;
  /** Maximum distance from any voxel to part mesh */
  maxDist: number;
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
  /** Whether to use GPU acceleration (WebGPU if available, otherwise falls back to CPU) */
  useGPU?: boolean;
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
   * Get unsigned (absolute) distance to mesh surface
   * Always positive - measures distance to nearest point on surface
   * 
   * Uses closest point query with BVH for efficiency
   */
  getDistance(point: THREE.Vector3): number {
    const closestPoint = new THREE.Vector3();
    const target = { 
      point: closestPoint, 
      distance: Infinity,
      faceIndex: 0
    };
    
    // Use BVH to find closest point on surface
    this.bvh.closestPointToPoint(point, target);
    
    return target.distance;
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
  
  // Temporary storage for distances (will be converted to Float32Array after we know the count)
  const distanceList: number[] = [];
  
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
            // Compute distance to part mesh for distance field using BVH
            const dist = partTester.getDistance(cellCenter);
            distanceList.push(dist);
          }
          
          if (storeAllCells) {
            allCells.push(cell);
          }
        }
      }
    }
  }
  
  // Convert distance list to Float32Array (indexed by voxel ID)
  const voxelDist = new Float32Array(distanceList);
  
  // Compute min/max distances for validation
  let minDist = Infinity;
  let maxDist = -Infinity;
  for (let i = 0; i < voxelDist.length; i++) {
    const d = voxelDist[i];
    if (d < minDist) minDist = d;
    if (d > maxDist) maxDist = d;
  }
  
  // Handle edge case of no mold volume cells
  if (voxelDist.length === 0) {
    minDist = 0;
    maxDist = 0;
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
    minDist,
    maxDist,
  };
  
  // Log distance field validation
  console.log(`Distance field computed: min=${minDist.toFixed(6)}, max=${maxDist.toFixed(6)}`);
  
  return {
    allCells: storeAllCells ? allCells : [],
    moldVolumeCells,
    resolution: new THREE.Vector3(resX, resY, resZ),
    totalCellCount,
    moldVolumeCellCount: moldVolumeCells.length,
    boundingBox,
    cellSize,
    stats,
    voxelDist,
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
 * Falls back to CPU if WebGPU is not available
 */
export async function generateVolumetricGridGPU(
  outerShellGeometry: THREE.BufferGeometry,
  partGeometry: THREE.BufferGeometry,
  options: VolumetricGridOptions = {}
): Promise<VolumetricGridResult> {
  // Check for WebGPU support
  if (!isWebGPUAvailable()) {
    console.log('WebGPU not available, falling back to CPU');
    return generateVolumetricGrid(outerShellGeometry, partGeometry, options);
  }
  
  const startTime = performance.now();
  
  // Parse options
  const resolution = typeof options.resolution === 'number' 
    ? options.resolution 
    : options.resolution?.x ?? DEFAULT_GRID_RESOLUTION;
  const marginPercent = options.marginPercent ?? 0.05;
  const storeAllCells = options.storeAllCells ?? false;
  
  console.log('═══════════════════════════════════════════════════════');
  console.log('GPU VOLUMETRIC GRID GENERATION (WebGPU)');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Resolution: ${resolution}³`);
  
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
    
    console.log(`  Shell triangles: ${shellTriangles.length / 9}`);
    console.log(`  Part triangles: ${partTriangles.length / 9}`);
    
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
    
    // Extract mold volume cells
    const moldVolumeCells: GridCell[] = [];
    const allCells: GridCell[] = [];
    
    for (let idx = 0; idx < totalCells; idx++) {
      const i = idx % resolution;
      const j = Math.floor(idx / resolution) % resolution;
      const k = Math.floor(idx / (resolution * resolution));
      
      const center = new THREE.Vector3(
        boundingBox.min.x + (i + 0.5) * cellSize.x,
        boundingBox.min.y + (j + 0.5) * cellSize.y,
        boundingBox.min.z + (k + 0.5) * cellSize.z
      );
      
      const isMoldVolume = resultData[idx] === 1;
      
      if (isMoldVolume) {
        moldVolumeCells.push({
          index: new THREE.Vector3(i, j, k),
          center,
          size: cellSize.clone(),
          isMoldVolume: true,
        });
      }
      
      if (storeAllCells) {
        allCells.push({
          index: new THREE.Vector3(i, j, k),
          center: isMoldVolume ? center : center.clone(),
          size: cellSize.clone(),
          isMoldVolume,
        });
      }
    }
    
    // Compute distance field using BVH on CPU (GPU path determines which cells, CPU computes distances)
    // Create a BVH tester for the part mesh to compute distances
    const partTester = new MeshInsideOutsideTester(partGeom, 'part');
    const voxelDist = new Float32Array(moldVolumeCells.length);
    
    let minDist = Infinity;
    let maxDist = -Infinity;
    
    for (let i = 0; i < moldVolumeCells.length; i++) {
      const dist = partTester.getDistance(moldVolumeCells[i].center);
      voxelDist[i] = dist;
      if (dist < minDist) minDist = dist;
      if (dist > maxDist) maxDist = dist;
    }
    
    // Handle edge case of no mold volume cells
    if (voxelDist.length === 0) {
      minDist = 0;
      maxDist = 0;
    }
    
    // Clean up distance tester
    partTester.dispose();
    
    // Clean up GPU resources
    paramsBuffer.destroy();
    triCountBuffer.destroy();
    shellBuffer.destroy();
    partBuffer.destroy();
    resultBuffer.destroy();
    stagingBuffer.destroy();
    shellGeom.dispose();
    partGeom.dispose();
    
    const endTime = performance.now();
    
    const cellVolume = cellSize.x * cellSize.y * cellSize.z;
    const moldVolume = moldVolumeCells.length * cellVolume;
    const totalVolume = totalCells * cellVolume;
    
    const stats: VolumetricGridStats = {
      moldVolume,
      totalVolume,
      fillRatio: moldVolume / totalVolume,
      computeTimeMs: endTime - startTime,
      minDist,
      maxDist,
    };
    
    // Log distance field validation
    console.log(`Distance field computed: min=${minDist.toFixed(6)}, max=${maxDist.toFixed(6)}`);
    
    console.log(`GPU Grid Generation Complete:`);
    console.log(`  Mold volume cells: ${moldVolumeCells.length} / ${totalCells}`);
    console.log(`  Fill ratio: ${(stats.fillRatio * 100).toFixed(1)}%`);
    console.log(`  Compute time: ${stats.computeTimeMs.toFixed(1)} ms`);
    
    return {
      allCells: storeAllCells ? allCells : [],
      moldVolumeCells,
      resolution: new THREE.Vector3(resolution, resolution, resolution),
      totalCellCount: totalCells,
      moldVolumeCellCount: moldVolumeCells.length,
      boundingBox,
      cellSize,
      stats,
      voxelDist,
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

/**
 * Interpolate between two colors based on a normalized value (0-1)
 * Red (close to part) → Teal (far from part)
 */
function interpolateDistanceColor(t: number): { r: number; g: number; b: number } {
  // Clamp t to [0, 1]
  t = Math.max(0, Math.min(1, t));
  
  // Red: RGB(255, 0, 0) → Teal: RGB(0, 255, 255)
  const r = Math.round(255 * (1 - t));
  const g = Math.round(255 * t);
  const b = Math.round(255 * t);
  
  return { r, g, b };
}

/**
 * Create a point cloud visualization of the mold volume cells
 * Colors points based on distance to part mesh: red (close) → teal (far)
 * 
 * @param gridResult - The volumetric grid result (must include voxelDist)
 * @param _color - Deprecated, colors are now computed from distance field
 * @param pointSize - Size of each point (default: 2)
 * @returns THREE.Points object for adding to scene
 */
export function createMoldVolumePointCloud(
  gridResult: VolumetricGridResult,
  _color: THREE.ColorRepresentation = 0x00ffff,
  pointSize: number = 2
): THREE.Points {
  const cellCount = gridResult.moldVolumeCells.length;
  const positions = new Float32Array(cellCount * 3);
  const colors = new Float32Array(cellCount * 3);
  
  const { minDist, maxDist } = gridResult.stats;
  const distRange = maxDist - minDist;
  
  for (let i = 0; i < cellCount; i++) {
    const cell = gridResult.moldVolumeCells[i];
    positions[i * 3] = cell.center.x;
    positions[i * 3 + 1] = cell.center.y;
    positions[i * 3 + 2] = cell.center.z;
    
    // Normalize distance to [0, 1] range
    const dist = gridResult.voxelDist[i];
    const t = distRange > 0 ? (dist - minDist) / distRange : 0.5;
    
    // Get interpolated color (red → teal)
    const color = interpolateDistanceColor(t);
    colors[i * 3] = color.r / 255;
    colors[i * 3 + 1] = color.g / 255;
    colors[i * 3 + 2] = color.b / 255;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: pointSize,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.6,
    vertexColors: true,
  });
  
  return new THREE.Points(geometry, material);
}

/**
 * Create a voxel box visualization of the mold volume cells
 * Uses instanced mesh for efficient rendering of many boxes
 * Colors voxels based on distance to part mesh: red (close) → teal (far)
 * 
 * @param gridResult - The volumetric grid result (must include voxelDist)
 * @param _color - Deprecated, colors are now computed from distance field
 * @param opacity - Opacity of boxes (default: 0.3)
 * @returns THREE.InstancedMesh for adding to scene
 */
export function createMoldVolumeVoxels(
  gridResult: VolumetricGridResult,
  _color: THREE.ColorRepresentation = 0x00ffff,
  opacity: number = 0.3
): THREE.InstancedMesh {
  const cellCount = gridResult.moldVolumeCells.length;
  const { cellSize } = gridResult;
  const { minDist, maxDist } = gridResult.stats;
  const distRange = maxDist - minDist;
  
  // Create box geometry for a single cell
  const boxGeometry = new THREE.BoxGeometry(cellSize.x, cellSize.y, cellSize.z);
  
  const material = new THREE.MeshPhongMaterial({
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    vertexColors: false,
  });
  
  // Create instanced mesh with per-instance colors
  const instancedMesh = new THREE.InstancedMesh(boxGeometry, material, cellCount);
  
  // Set transforms and colors for each instance
  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();
  
  for (let i = 0; i < cellCount; i++) {
    const cell = gridResult.moldVolumeCells[i];
    matrix.setPosition(cell.center);
    instancedMesh.setMatrixAt(i, matrix);
    
    // Normalize distance to [0, 1] range
    const dist = gridResult.voxelDist[i];
    const t = distRange > 0 ? (dist - minDist) / distRange : 0.5;
    
    // Get interpolated color (red → teal)
    const colorRGB = interpolateDistanceColor(t);
    color.setRGB(colorRGB.r / 255, colorRGB.g / 255, colorRGB.b / 255);
    instancedMesh.setColorAt(i, color);
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
  
  // Compute distance field using BVH on CPU (workers determine which cells, main thread computes distances)
  const partTester = new MeshInsideOutsideTester(partGeom, 'part');
  const voxelDist = new Float32Array(moldVolumeCells.length);
  
  let minDist = Infinity;
  let maxDist = -Infinity;
  
  for (let i = 0; i < moldVolumeCells.length; i++) {
    const dist = partTester.getDistance(moldVolumeCells[i].center);
    voxelDist[i] = dist;
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;
  }
  
  // Handle edge case of no mold volume cells
  if (voxelDist.length === 0) {
    minDist = 0;
    maxDist = 0;
  }
  
  // Clean up
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
    minDist,
    maxDist,
  };
  
  // Log distance field validation
  console.log(`Distance field computed: min=${minDist.toFixed(6)}, max=${maxDist.toFixed(6)}`);
  
  return {
    allCells: [],
    moldVolumeCells,
    resolution: new THREE.Vector3(resX, resY, resZ),
    totalCellCount,
    moldVolumeCellCount: moldVolumeCells.length,
    boundingBox,
    cellSize,
    stats,
    voxelDist,
  };
}
