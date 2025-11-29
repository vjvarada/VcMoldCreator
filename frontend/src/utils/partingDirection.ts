/**
 * Parting Direction Estimation for Two-Piece Mold Analysis
 * 
 * Uses BVH-accelerated raycasting with parallel Web Workers to find optimal
 * mold parting directions that maximize surface visibility while ensuring
 * mutually exclusive surface ownership.
 * 
 * Key constraints:
 * - Directions must be > 135° apart (suitable for two-piece molds)
 * - Each triangle is owned by at most one direction (no double-counting)
 * - Self-occlusion is handled via BVH raycasting
 */

import * as THREE from 'three';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';

// Extend Three.js Mesh to use accelerated raycasting
THREE.Mesh.prototype.raycast = acceleratedRaycast;

// ============================================================================
// CONSTANTS
// ============================================================================

/** Colors for visualization (hex values for arrows) */
export const COLORS = {
  D1: 0x00ff00,      // Green
  D2: 0xff6600,      // Orange
  NEUTRAL: 0x888888, // Gray
  OVERLAP: 0xffff00, // Yellow
  MESH_DEFAULT: 0x00aaff, // Light blue (original mesh color)
} as const;

/** Colors for vertex painting (RGB 0-255) */
const PAINT_COLORS = {
  D1: { r: 0, g: 255, b: 0 },
  D2: { r: 255, g: 102, b: 0 },
  NEUTRAL: { r: 136, g: 136, b: 136 },
  OVERLAP: { r: 255, g: 255, b: 0 },
} as const;

/** Minimum angle between parting directions (degrees) */
const MIN_ANGLE_DEG = 135;
const MIN_ANGLE_COS = Math.cos(MIN_ANGLE_DEG * Math.PI / 180);

// ============================================================================
// TYPES
// ============================================================================

export interface VisibilityPaintData {
  d1: THREE.Vector3;
  d2: THREE.Vector3;
}

interface DirectionScore {
  direction: THREE.Vector3;
  visibleArea: number;
  visibleTriangles: Set<number>;
}

interface TriangleData {
  normal: THREE.Vector3;
  area: number;
  centroid: THREE.Vector3;
}

interface WorkerResult {
  directionIndex: number;
  visibleTriangles: number[];
  visibleArea: number;
}

interface WorkerMessage {
  type: 'init' | 'compute' | 'result' | 'ready';
  workerId?: number;
  positionArray?: Float32Array;
  indexArray?: Uint32Array | null;
  directions?: Float32Array;
  directionIndices?: number[];
  results?: WorkerResult[];
  triangleCount?: number;
  totalArea?: number;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Generate k uniformly distributed points on a unit sphere using Fibonacci sampling
 */
function fibonacciSphere(k: number): THREE.Vector3[] {
  const directions: THREE.Vector3[] = [];
  const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle

  for (let i = 0; i < k; i++) {
    const y = 1 - (i / (k - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = phi * i;
    directions.push(new THREE.Vector3(
      Math.cos(theta) * radius,
      y,
      Math.sin(theta) * radius
    ).normalize());
  }

  return directions;
}

/**
 * Extract triangle data from geometry
 */
function extractTriangleData(geometry: THREE.BufferGeometry): TriangleData[] {
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  const triangleCount = index ? index.count / 3 : position.count / 3;
  
  const triangles: TriangleData[] = [];
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const edge1 = new THREE.Vector3();
  const edge2 = new THREE.Vector3();
  const normal = new THREE.Vector3();
  
  for (let i = 0; i < triangleCount; i++) {
    const [a, b, c] = index 
      ? [index.getX(i * 3), index.getX(i * 3 + 1), index.getX(i * 3 + 2)]
      : [i * 3, i * 3 + 1, i * 3 + 2];
    
    vA.fromBufferAttribute(position, a);
    vB.fromBufferAttribute(position, b);
    vC.fromBufferAttribute(position, c);
    
    edge1.subVectors(vB, vA);
    edge2.subVectors(vC, vA);
    normal.crossVectors(edge1, edge2);
    const area = normal.length() * 0.5;
    normal.normalize();
    
    triangles.push({
      normal: normal.clone(),
      area,
      centroid: new THREE.Vector3().add(vA).add(vB).add(vC).divideScalar(3),
    });
  }
  
  return triangles;
}

// ============================================================================
// PARALLEL VISIBILITY COMPUTATION
// ============================================================================

/**
 * Find optimal parting directions using parallel Web Workers
 */
export async function findPartingDirectionsParallel(
  geometry: THREE.BufferGeometry,
  k: number = 64,
  numWorkers: number = navigator.hardwareConcurrency || 4
): Promise<{ d1: THREE.Vector3; d2: THREE.Vector3 }> {
  console.log('═══════════════════════════════════════════════════════');
  console.log('PARALLEL VISIBILITY ALGORITHM');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Direction samples: ${k}`);
  console.log(`  Worker threads: ${numWorkers}`);
  
  const startTime = performance.now();
  
  // Prepare geometry for workers
  const geomClone = geometry.clone();
  if (!geomClone.index) {
    const count = geomClone.getAttribute('position').count;
    geomClone.setIndex(Array.from({ length: count }, (_, i) => i));
  }
  
  const positionArray = new Float32Array(geomClone.getAttribute('position').array);
  const indexAttr = geomClone.getIndex();
  const indexArray = indexAttr ? new Uint32Array(indexAttr.array) : null;
  
  // Extract triangle data for pair evaluation
  const triangles = extractTriangleData(geomClone);
  const areas = new Float32Array(triangles.map(t => t.area));
  const totalSurfaceArea = areas.reduce((sum, a) => sum + a, 0);
  
  console.log(`  Triangles: ${triangles.length}`);
  console.log(`  Total surface area: ${totalSurfaceArea.toFixed(4)}`);
  
  const directions = fibonacciSphere(k);
  
  // Initialize workers
  const workers: Worker[] = [];
  const workerReadyPromises: Promise<void>[] = [];
  
  for (let i = 0; i < numWorkers; i++) {
    const worker = new Worker(
      new URL('./visibilityWorker.ts', import.meta.url),
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
    
    const initMsg: WorkerMessage = {
      type: 'init',
      workerId: i,
      positionArray: positionArray.slice(),
      indexArray: indexArray?.slice() ?? null
    };
    
    const transfers: Transferable[] = [initMsg.positionArray!.buffer];
    if (initMsg.indexArray) transfers.push(initMsg.indexArray.buffer);
    worker.postMessage(initMsg, transfers);
  }
  
  await Promise.all(workerReadyPromises);
  console.log('  Workers initialized');
  
  // Distribute work
  const directionsPerWorker = Math.ceil(k / numWorkers);
  const workerPromises: Promise<WorkerResult[]>[] = [];
  
  for (let w = 0; w < numWorkers; w++) {
    const startIdx = w * directionsPerWorker;
    const endIdx = Math.min(startIdx + directionsPerWorker, k);
    if (endIdx <= startIdx) continue;
    
    const count = endIdx - startIdx;
    const dirArray = new Float32Array(count * 3);
    const dirIndices: number[] = [];
    
    for (let i = 0; i < count; i++) {
      const d = directions[startIdx + i];
      dirArray[i * 3] = d.x;
      dirArray[i * 3 + 1] = d.y;
      dirArray[i * 3 + 2] = d.z;
      dirIndices.push(startIdx + i);
    }
    
    workerPromises.push(new Promise<WorkerResult[]>((resolve) => {
      const handler = (event: MessageEvent<WorkerMessage>) => {
        if (event.data.type === 'result' && event.data.workerId === w) {
          workers[w].removeEventListener('message', handler);
          resolve(event.data.results!);
        }
      };
      workers[w].addEventListener('message', handler);
    }));
    
    workers[w].postMessage({ type: 'compute', directions: dirArray, directionIndices: dirIndices }, [dirArray.buffer]);
  }
  
  const allResults = await Promise.all(workerPromises);
  workers.forEach(w => w.terminate());
  
  // Collect results
  const directionScores: Map<number, DirectionScore> = new Map();
  
  for (const workerResults of allResults) {
    for (const result of workerResults) {
      const visibleTriangles = new Set(result.visibleTriangles);
      let visibleArea = 0;
      for (const tri of visibleTriangles) {
        visibleArea += areas[tri];
      }
      
      directionScores.set(result.directionIndex, {
        direction: directions[result.directionIndex].clone(),
        visibleArea,
        visibleTriangles
      });
    }
  }
  
  // Find best pair
  const { bestD1, bestD2 } = findBestPair(directionScores, areas, totalSurfaceArea);
  
  // Report results
  const d1 = bestD1.direction;
  const d2 = bestD2.direction;
  const angleDeg = Math.acos(Math.max(-1, Math.min(1, d1.dot(d2)))) * (180 / Math.PI);
  
  let d2UniqueArea = 0;
  for (const tri of bestD2.visibleTriangles) {
    if (!bestD1.visibleTriangles.has(tri)) d2UniqueArea += areas[tri];
  }
  const totalCoverage = (bestD1.visibleArea + d2UniqueArea) / totalSurfaceArea * 100;
  
  const elapsed = performance.now() - startTime;
  
  console.log('\n═══════════════════════════════════════════════════════');
  console.log('RESULTS');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  d1: [${d1.toArray().map(v => v.toFixed(3)).join(', ')}] - ${(bestD1.visibleArea / totalSurfaceArea * 100).toFixed(1)}%`);
  console.log(`  d2: [${d2.toArray().map(v => v.toFixed(3)).join(', ')}] - ${(d2UniqueArea / totalSurfaceArea * 100).toFixed(1)}%`);
  console.log(`  Total coverage: ${totalCoverage.toFixed(1)}%`);
  console.log(`  Angle: ${angleDeg.toFixed(1)}°`);
  console.log(`  Time: ${(elapsed / 1000).toFixed(2)}s`);
  console.log('═══════════════════════════════════════════════════════');
  
  geomClone.dispose();
  
  return { d1, d2 };
}

/**
 * Find the best direction pair from computed scores
 */
function findBestPair(
  scores: Map<number, DirectionScore>,
  areas: Float32Array,
  _totalArea: number
): { bestD1: DirectionScore; bestD2: DirectionScore } {
  const sortedScores = Array.from(scores.values()).sort((a, b) => b.visibleArea - a.visibleArea);
  
  let bestD1 = sortedScores[0];
  let bestD2 = sortedScores[1] || sortedScores[0];
  let bestCoverage = 0;
  let bestDot = 1;
  
  for (let i = 0; i < sortedScores.length; i++) {
    const s1 = sortedScores[i];
    
    for (let j = i + 1; j < sortedScores.length; j++) {
      const s2 = sortedScores[j];
      const dot = s1.direction.dot(s2.direction);
      
      // Skip if angle too narrow
      if (dot > MIN_ANGLE_COS) continue;
      
      // Quick upper bound check
      if (s1.visibleArea + s2.visibleArea <= bestCoverage) continue;
      
      // Try s1 as primary
      let s2Unique = 0;
      for (const tri of s2.visibleTriangles) {
        if (!s1.visibleTriangles.has(tri)) s2Unique += areas[tri];
      }
      let coverage = s1.visibleArea + s2Unique;
      
      if (coverage > bestCoverage + 0.0001 || (Math.abs(coverage - bestCoverage) < 0.0001 && dot < bestDot)) {
        bestCoverage = coverage;
        bestD1 = s1;
        bestD2 = s2;
        bestDot = dot;
      }
      
      // Try s2 as primary
      let s1Unique = 0;
      for (const tri of s1.visibleTriangles) {
        if (!s2.visibleTriangles.has(tri)) s1Unique += areas[tri];
      }
      coverage = s2.visibleArea + s1Unique;
      
      if (coverage > bestCoverage + 0.0001 || (Math.abs(coverage - bestCoverage) < 0.0001 && dot < bestDot)) {
        bestCoverage = coverage;
        bestD1 = s2;
        bestD2 = s1;
        bestDot = dot;
      }
    }
  }
  
  return { bestD1, bestD2 };
}

// ============================================================================
// ARROW VISUALIZATION
// ============================================================================

/**
 * Create an arrow helper for a direction
 */
function createDirectionArrow(
  direction: THREE.Vector3,
  origin: THREE.Vector3,
  color: number,
  length: number
): THREE.ArrowHelper {
  return new THREE.ArrowHelper(
    direction.clone().normalize(),
    origin.clone(),
    length,
    color,
    length * 0.2,
    length * 0.1
  );
}

/**
 * Remove arrows from scene and dispose resources
 */
export function removePartingDirectionArrows(arrows: THREE.ArrowHelper[]): void {
  for (const arrow of arrows) {
    arrow.parent?.remove(arrow);
    arrow.dispose();
  }
}

/**
 * Main entry point: compute and visualize parting directions
 */
export async function computeAndShowPartingDirectionsParallel(
  mesh: THREE.Mesh,
  k: number = 64,
  numWorkers?: number
): Promise<{ d1: THREE.Vector3; d2: THREE.Vector3; arrows: THREE.ArrowHelper[]; visibilityData: VisibilityPaintData }> {
  const geometry = mesh.geometry as THREE.BufferGeometry;
  if (!geometry) throw new Error('Mesh must have a BufferGeometry');
  
  // Apply world transform for analysis
  const worldGeometry = geometry.clone();
  mesh.updateMatrixWorld(true);
  worldGeometry.applyMatrix4(mesh.matrixWorld);
  
  const { d1, d2 } = await findPartingDirectionsParallel(worldGeometry, k, numWorkers);
  
  // Create arrows at mesh center
  worldGeometry.computeBoundingBox();
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  worldGeometry.boundingBox!.getCenter(center);
  worldGeometry.boundingBox!.getSize(size);
  const arrowLength = Math.max(size.x, size.y, size.z) * 1.2;
  
  const arrow1 = createDirectionArrow(d1, center, COLORS.D1, arrowLength);
  const arrow2 = createDirectionArrow(d2, center, COLORS.D2, arrowLength);
  
  mesh.parent?.add(arrow1);
  mesh.parent?.add(arrow2);
  
  console.log('\nComputing visibility for triangle painting...');
  
  worldGeometry.dispose();
  
  return {
    d1,
    d2,
    arrows: [arrow1, arrow2],
    visibilityData: { d1: d1.clone(), d2: d2.clone() }
  };
}

// ============================================================================
// VISIBILITY PAINTING
// ============================================================================

/**
 * Apply vertex colors to visualize visibility from parting directions
 */
export function applyVisibilityPaint(
  mesh: THREE.Mesh,
  visibilityData: VisibilityPaintData,
  showD1: boolean,
  showD2: boolean
): void {
  const geometry = mesh.geometry as THREE.BufferGeometry;
  if (!geometry) return;
  
  console.log('Applying visibility paint...');
  const startTime = performance.now();
  
  // Apply world transform for raycasting
  const worldGeometry = geometry.clone();
  mesh.updateMatrixWorld(true);
  worldGeometry.applyMatrix4(mesh.matrixWorld);
  
  // Convert to non-indexed for per-triangle coloring
  let paintGeometry = geometry;
  let worldPaintGeometry = worldGeometry;
  
  if (geometry.index) {
    paintGeometry = geometry.toNonIndexed();
    worldPaintGeometry = worldGeometry.toNonIndexed();
    mesh.geometry = paintGeometry;
  }
  
  const bvh = new MeshBVH(worldPaintGeometry);
  const position = paintGeometry.getAttribute('position');
  const worldPosition = worldPaintGeometry.getAttribute('position');
  const triangleCount = position.count / 3;
  
  // Create color buffer
  const colorArray = new Uint8Array(position.count * 3);
  colorArray.fill(PAINT_COLORS.NEUTRAL.r);
  const colorAttr = new THREE.BufferAttribute(colorArray, 3, true);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  paintGeometry.setAttribute('color', colorAttr);
  
  // Compute ray offset
  worldPaintGeometry.computeBoundingBox();
  const size = new THREE.Vector3();
  worldPaintGeometry.boundingBox!.getSize(size);
  const rayOffset = Math.max(size.x, size.y, size.z) * 2;
  
  // Reusable objects
  const ray = new THREE.Ray();
  const centroid = new THREE.Vector3();
  const hitPoint = new THREE.Vector3();
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const normal = new THREE.Vector3();
  const edge1 = new THREE.Vector3();
  const edge2 = new THREE.Vector3();
  
  let d1Count = 0, d2Count = 0, overlapCount = 0, neutralCount = 0;
  
  for (let triIdx = 0; triIdx < triangleCount; triIdx++) {
    const i3 = triIdx * 3;
    
    vA.fromBufferAttribute(worldPosition, i3);
    vB.fromBufferAttribute(worldPosition, i3 + 1);
    vC.fromBufferAttribute(worldPosition, i3 + 2);
    
    centroid.set(
      (vA.x + vB.x + vC.x) / 3,
      (vA.y + vB.y + vC.y) / 3,
      (vA.z + vB.z + vC.z) / 3
    );
    
    edge1.subVectors(vB, vA);
    edge2.subVectors(vC, vA);
    normal.crossVectors(edge1, edge2).normalize();
    
    let isD1Visible = false;
    let isD2Visible = false;
    
    // Check D1 visibility
    if (showD1) {
      isD1Visible = checkTriangleVisibility(
        visibilityData.d1, normal, centroid, rayOffset,
        ray, hitPoint, bvh, vA, vB, vC
      );
    }
    
    // Check D2 visibility
    if (showD2) {
      isD2Visible = checkTriangleVisibility(
        visibilityData.d2, normal, centroid, rayOffset,
        ray, hitPoint, bvh, vA, vB, vC
      );
    }
    
    // Determine color
    let color: { readonly r: number; readonly g: number; readonly b: number } = PAINT_COLORS.NEUTRAL;
    if (isD1Visible && isD2Visible) {
      color = PAINT_COLORS.OVERLAP;
      overlapCount++;
    } else if (isD1Visible) {
      color = PAINT_COLORS.D1;
      d1Count++;
    } else if (isD2Visible) {
      color = PAINT_COLORS.D2;
      d2Count++;
    } else {
      neutralCount++;
    }
    
    // Paint triangle vertices
    for (let v = 0; v < 3; v++) {
      const idx = (i3 + v) * 3;
      colorArray[idx] = color.r;
      colorArray[idx + 1] = color.g;
      colorArray[idx + 2] = color.b;
    }
  }
  
  const elapsed = performance.now() - startTime;
  console.log(`  Painted: D1=${d1Count}, D2=${d2Count}, Overlap=${overlapCount}, Neutral=${neutralCount}`);
  console.log(`  Coverage: ${((d1Count + d2Count + overlapCount) / triangleCount * 100).toFixed(1)}%`);
  console.log(`  Time: ${elapsed.toFixed(0)}ms`);
  
  colorAttr.needsUpdate = true;
  
  // Configure material for vertex colors
  const material = mesh.material as THREE.Material & { vertexColors?: boolean; color?: THREE.Color };
  if (material) {
    material.vertexColors = true;
    material.color?.setRGB(1, 1, 1);
    material.needsUpdate = true;
  }
  
  worldGeometry.dispose();
  worldPaintGeometry.dispose();
}

/**
 * Check if a triangle is visible from a direction
 */
function checkTriangleVisibility(
  direction: THREE.Vector3,
  normal: THREE.Vector3,
  centroid: THREE.Vector3,
  rayOffset: number,
  ray: THREE.Ray,
  hitPoint: THREE.Vector3,
  bvh: MeshBVH,
  vA: THREE.Vector3,
  vB: THREE.Vector3,
  vC: THREE.Vector3
): boolean {
  const dot = normal.dot(direction);
  if (dot <= 0) return false;
  
  ray.origin.set(
    centroid.x + direction.x * rayOffset,
    centroid.y + direction.y * rayOffset,
    centroid.z + direction.z * rayOffset
  );
  ray.direction.set(-direction.x, -direction.y, -direction.z);
  
  const hit = bvh.raycastFirst(ray);
  if (!hit) return false;
  
  hitPoint.copy(ray.origin).addScaledVector(ray.direction, hit.distance);
  return pointInTriangle(hitPoint, vA, vB, vC, 0.01);
}

/**
 * Check if a point lies within a triangle using barycentric coordinates
 */
function pointInTriangle(
  p: THREE.Vector3,
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3,
  tolerance: number
): boolean {
  const v0 = new THREE.Vector3().subVectors(c, a);
  const v1 = new THREE.Vector3().subVectors(b, a);
  const v2 = new THREE.Vector3().subVectors(p, a);
  
  const dot00 = v0.dot(v0);
  const dot01 = v0.dot(v1);
  const dot02 = v0.dot(v2);
  const dot11 = v1.dot(v1);
  const dot12 = v1.dot(v2);
  
  const invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
  const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const v = (dot00 * dot12 - dot01 * dot02) * invDenom;
  
  return u >= -tolerance && v >= -tolerance && u + v <= 1 + tolerance;
}

/**
 * Remove visibility paint and restore original appearance
 */
export function removeVisibilityPaint(mesh: THREE.Mesh): void {
  const geometry = mesh.geometry as THREE.BufferGeometry;
  if (!geometry) return;
  
  geometry.deleteAttribute('color');
  
  const material = mesh.material as THREE.Material & { vertexColors?: boolean; color?: THREE.Color };
  if (material) {
    material.vertexColors = false;
    material.color?.setHex(COLORS.MESH_DEFAULT);
    material.needsUpdate = true;
  }
}
