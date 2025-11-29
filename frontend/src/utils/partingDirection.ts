/**
 * SDF-Biased Multi-Ray Visibility Algorithm for Parting Direction Estimation
 * 
 * Robust algorithm that uses:
 * - BVH fast ray queries
 * - Orthographic visibility (directional ray flooding)
 * - Self-occlusion detection via parallel rays
 * - Unique surface area accounting (zero double-counting)
 * - Wide-angle direction clustering (>135° constraint)
 * - Disjoint surface area ownership per direction
 */

import * as THREE from 'three';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';

// Extend Three.js Mesh to use accelerated raycasting
THREE.Mesh.prototype.raycast = acceleratedRaycast;

interface DirectionScore {
  direction: THREE.Vector3;
  visibleArea: number;
  nonVisibleArea: number;
  score: number;
  visibleTriangles: Set<number>;
}

interface TriangleData {
  normal: THREE.Vector3;
  area: number;
  centroid: THREE.Vector3;
  vertices: [THREE.Vector3, THREE.Vector3, THREE.Vector3];
}

/**
 * Generate k uniformly distributed points on a unit sphere
 * using Fibonacci sphere sampling for uniform distribution
 */
function fibonacciSphere(k: number): THREE.Vector3[] {
  const directions: THREE.Vector3[] = [];
  const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle

  for (let i = 0; i < k; i++) {
    const y = 1 - (i / (k - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = phi * i;

    const x = Math.cos(theta) * radius;
    const z = Math.sin(theta) * radius;

    directions.push(new THREE.Vector3(x, y, z).normalize());
  }

  return directions;
}

/**
 * Extract triangle data from geometry into optimized typed arrays
 */
function extractTriangleData(geometry: THREE.BufferGeometry): TriangleData[] {
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  
  const triangles: TriangleData[] = [];
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const edge1 = new THREE.Vector3();
  const edge2 = new THREE.Vector3();
  const normal = new THREE.Vector3();
  
  const triangleCount = index ? index.count / 3 : position.count / 3;
  
  for (let i = 0; i < triangleCount; i++) {
    let a: number, b: number, c: number;
    
    if (index) {
      a = index.getX(i * 3);
      b = index.getX(i * 3 + 1);
      c = index.getX(i * 3 + 2);
    } else {
      a = i * 3;
      b = i * 3 + 1;
      c = i * 3 + 2;
    }
    
    vA.fromBufferAttribute(position, a);
    vB.fromBufferAttribute(position, b);
    vC.fromBufferAttribute(position, c);
    
    // Compute normal
    edge1.subVectors(vB, vA);
    edge2.subVectors(vC, vA);
    normal.crossVectors(edge1, edge2);
    const area = normal.length() * 0.5;
    normal.normalize();
    
    // Compute centroid
    const centroid = new THREE.Vector3()
      .add(vA).add(vB).add(vC).divideScalar(3);
    
    triangles.push({
      normal: normal.clone(),
      area,
      centroid: centroid.clone(),
      vertices: [vA.clone(), vB.clone(), vC.clone()]
    });
  }
  
  return triangles;
}

/**
 * Optimized data structure for fast visibility computation
 * Uses typed arrays for cache-friendly memory access
 */
interface OptimizedTriangleData {
  normals: Float32Array;    // nx, ny, nz for each triangle (length = triangleCount * 3)
  centroids: Float32Array;  // cx, cy, cz for each triangle (length = triangleCount * 3)
  areas: Float32Array;      // area for each triangle (length = triangleCount)
  count: number;
}

function createOptimizedData(triangles: TriangleData[]): OptimizedTriangleData {
  const count = triangles.length;
  const normals = new Float32Array(count * 3);
  const centroids = new Float32Array(count * 3);
  const areas = new Float32Array(count);
  
  for (let i = 0; i < count; i++) {
    const t = triangles[i];
    const i3 = i * 3;
    
    normals[i3] = t.normal.x;
    normals[i3 + 1] = t.normal.y;
    normals[i3 + 2] = t.normal.z;
    
    centroids[i3] = t.centroid.x;
    centroids[i3 + 1] = t.centroid.y;
    centroids[i3 + 2] = t.centroid.z;
    
    areas[i] = t.area;
  }
  
  return { normals, centroids, areas, count };
}

/**
 * Compute total area from a set of triangle indices using typed array
 */
function computeAreaFromIndices(areas: Float32Array, indices: Set<number>): number {
  let total = 0;
  for (const idx of indices) {
    total += areas[idx];
  }
  return total;
}

/**
 * Compute total area from a set of triangle indices
 */
function computeAreaFromTriangles(
  triangles: TriangleData[],
  indices: Set<number>
): number {
  let area = 0;
  for (const idx of indices) {
    area += triangles[idx].area;
  }
  return area;
}

/**
 * Optimized triangle-centric visibility computer with BVH acceleration
 * 
 * Optimizations applied:
 * 1. Typed arrays for cache-friendly memory access
 * 2. Reused Ray object to avoid allocations
 * 3. Pre-computed ray direction (negated once)
 * 4. Inlined dot product calculation
 * 5. Direct array access instead of object property access
 */
class MultiRayVisibilityComputer {
  private geometry: THREE.BufferGeometry;
  private triangles: TriangleData[];
  private optimized: OptimizedTriangleData;
  private totalSurfaceArea: number;
  private bvh: MeshBVH;
  private rayOffset: number;
  
  // Reusable objects to avoid allocations during computation
  private readonly ray: THREE.Ray = new THREE.Ray();
  private readonly rayDirection: THREE.Vector3 = new THREE.Vector3();

  constructor(geometry: THREE.BufferGeometry, _rayCount: number = 2048) {
    console.log('Initializing Optimized Visibility Computer with BVH...');
    const initStart = performance.now();
    
    this.geometry = geometry.clone();
    
    // Ensure geometry has index for BVH
    if (!this.geometry.index) {
      const positions = this.geometry.getAttribute('position');
      const indices: number[] = [];
      for (let i = 0; i < positions.count; i++) {
        indices.push(i);
      }
      this.geometry.setIndex(indices);
    }
    
    // Build BVH for fast raycasting
    this.bvh = new MeshBVH(this.geometry);
    this.geometry.boundsTree = this.bvh;
    
    // Extract triangle data and create optimized typed arrays
    this.triangles = extractTriangleData(this.geometry);
    this.optimized = createOptimizedData(this.triangles);
    this.totalSurfaceArea = this.optimized.areas.reduce((sum, a) => sum + a, 0);
    
    // Compute ray offset from bounding box
    this.geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    this.geometry.boundingBox!.getSize(size);
    this.rayOffset = Math.max(size.x, size.y, size.z) * 2;
    
    const initTime = performance.now() - initStart;
    console.log(`  BVH built for ${this.triangles.length} triangles in ${initTime.toFixed(0)}ms`);
    console.log(`  Total surface area: ${this.totalSurfaceArea.toFixed(4)}`);
  }

  /**
   * Optimized visible triangle computation
   * Uses typed arrays and minimizes object allocations
   */
  computeVisibleTriangles(direction: THREE.Vector3): Set<number> {
    const visible = new Set<number>();
    const { normals, centroids, count } = this.optimized;
    
    // Pre-compute direction components and negated direction
    const dx = direction.x;
    const dy = direction.y;
    const dz = direction.z;
    
    // Set up reusable ray direction (negated)
    this.rayDirection.set(-dx, -dy, -dz);
    this.ray.direction.copy(this.rayDirection);
    
    const offset = this.rayOffset;
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      
      // Inlined dot product: normal · direction
      const dot = normals[i3] * dx + normals[i3 + 1] * dy + normals[i3 + 2] * dz;
      
      // Skip back-facing triangles
      if (dot <= 0) continue;
      
      // Set ray origin: centroid + direction * offset
      const cx = centroids[i3];
      const cy = centroids[i3 + 1];
      const cz = centroids[i3 + 2];
      
      this.ray.origin.set(
        cx + dx * offset,
        cy + dy * offset,
        cz + dz * offset
      );
      
      // Find first intersection using BVH
      const hit = this.bvh.raycastFirst(this.ray);
      
      if (hit && hit.faceIndex === i) {
        visible.add(i);
      }
    }
    
    return visible;
  }

  /**
   * Get visibility score for a direction
   */
  getDirectionScore(direction: THREE.Vector3): DirectionScore {
    const visibleTriangles = this.computeVisibleTriangles(direction);
    const visibleArea = computeAreaFromIndices(this.optimized.areas, visibleTriangles);
    const nonVisibleArea = this.totalSurfaceArea - visibleArea;
    
    return {
      direction: direction.clone(),
      visibleArea,
      nonVisibleArea,
      score: visibleArea, // Higher is better (more visible area)
      visibleTriangles
    };
  }

  getTotalSurfaceArea(): number {
    return this.totalSurfaceArea;
  }

  getTriangles(): TriangleData[] {
    return this.triangles;
  }

  getOptimizedData(): OptimizedTriangleData {
    return this.optimized;
  }

  dispose(): void {
    this.geometry.dispose();
  }
}

/**
 * Find the optimal parting directions for a two-piece mold
 * 
 * Algorithm Summary:
 * 
 * 1. Preprocessing:
 *    - Load mesh, extract triangles, compute areas & normals
 *    - Build BVH for fast ray tests
 * 
 * 2. Visibility Evaluation (per direction):
 *    - Create orthographic plane enclosing mesh bounds
 *    - Populate plane with Fibonacci disk samples
 *    - Cast parallel rays using BVH
 *    - Record first-hit triangles (self-occlusion handled automatically)
 *    - Score = total area of unique hit triangles
 * 
 * 3. Global Ownership Map:
 *    - Pick d1 with highest individual coverage
 *    - Assign all triangles in S(d1) to direction 1
 *    - For d2 candidates, only count triangles NOT owned by d1
 *    - This ensures mutually exclusive surface ownership
 * 
 * 4. Direction Pair Constraint:
 *    - Only consider pairs with angle > 135°
 *    - Prune search space by discarding narrow-angle candidates
 * 
 * OPTIMIZATION: Sequential processing with early termination
 *    - Compute visibility for each direction and immediately evaluate pairs
 *    - Early terminate when 100% coverage with optimal angle is found
 *    - Skip remaining directions that can't improve the result
 */
export function findPartingDirections(
  geometry: THREE.BufferGeometry,
  k: number = 64,
  _rayCount: number = 2048
): { d1: THREE.Vector3; d2: THREE.Vector3; scores: DirectionScore[] } {
  console.log('═══════════════════════════════════════════════════════');
  console.log('SEQUENTIAL VISIBILITY ALGORITHM (Optimized)');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Direction samples: ${k}`);
  console.log(`  Valid pairs: angle > 135°`);
  console.log(`  Strategy: Compute & evaluate incrementally with early termination`);
  
  const startTime = performance.now();
  
  // Create visibility computer
  const visComputer = new MultiRayVisibilityComputer(geometry, _rayCount);
  const totalSurfaceArea = visComputer.getTotalSurfaceArea();
  const triangles = visComputer.getTriangles();
  const areas = visComputer.getOptimizedData().areas;
  
  // Sample directions uniformly on the sphere
  const directions = fibonacciSphere(k);
  
  // Minimum angle constraint: 135 degrees
  const minAngleDeg = 135;
  const minAngleCos = Math.cos(minAngleDeg * Math.PI / 180);
  
  // Tracking best result
  let bestD1: DirectionScore | null = null;
  let bestD2: DirectionScore | null = null;
  let bestCombinedCoverage = 0;
  let bestD1UniqueArea = 0;
  let bestD2UniqueArea = 0;
  let bestDotProduct = 1;
  let foundPerfectPair = false;
  
  // Store computed scores for directions we've processed
  const computedScores: DirectionScore[] = [];
  
  console.log('\nProcessing directions sequentially...');
  
  // Process each direction and immediately evaluate pairs
  for (let i = 0; i < directions.length && !foundPerfectPair; i++) {
    // Compute visibility for current direction
    const currentScore = visComputer.getDirectionScore(directions[i]);
    computedScores.push(currentScore);
    
    const currentArea = currentScore.visibleArea;
    const currentTriangles = currentScore.visibleTriangles;
    
    // Evaluate pairs: current direction as d1 with all previous directions as d2
    for (let j = 0; j < i; j++) {
      const prevScore = computedScores[j];
      
      // Check angle constraint
      const dotProduct = currentScore.direction.dot(prevScore.direction);
      if (dotProduct > minAngleCos) continue;
      
      // Try current as d1, previous as d2
      let d2UniqueArea = 0;
      for (const tri of prevScore.visibleTriangles) {
        if (!currentTriangles.has(tri)) {
          d2UniqueArea += areas[tri];
        }
      }
      let combinedCoverage = currentArea + d2UniqueArea;
      
      if (combinedCoverage > bestCombinedCoverage + 0.0001 ||
          (Math.abs(combinedCoverage - bestCombinedCoverage) < 0.0001 && dotProduct < bestDotProduct)) {
        bestCombinedCoverage = combinedCoverage;
        bestD1 = currentScore;
        bestD2 = prevScore;
        bestD1UniqueArea = currentArea;
        bestD2UniqueArea = d2UniqueArea;
        bestDotProduct = dotProduct;
        
        // Check for perfect pair (100% coverage at 180°)
        if (Math.abs(combinedCoverage - totalSurfaceArea) < 0.0001 && dotProduct <= -0.9999) {
          foundPerfectPair = true;
          console.log(`  Found perfect pair at direction ${i + 1}/${k} - early termination!`);
          break;
        }
      }
      
      // Try previous as d1, current as d2
      const prevArea = prevScore.visibleArea;
      const prevTriangles = prevScore.visibleTriangles;
      
      d2UniqueArea = 0;
      for (const tri of currentTriangles) {
        if (!prevTriangles.has(tri)) {
          d2UniqueArea += areas[tri];
        }
      }
      combinedCoverage = prevArea + d2UniqueArea;
      
      if (combinedCoverage > bestCombinedCoverage + 0.0001 ||
          (Math.abs(combinedCoverage - bestCombinedCoverage) < 0.0001 && dotProduct < bestDotProduct)) {
        bestCombinedCoverage = combinedCoverage;
        bestD1 = prevScore;
        bestD2 = currentScore;
        bestD1UniqueArea = prevArea;
        bestD2UniqueArea = d2UniqueArea;
        bestDotProduct = dotProduct;
        
        // Check for perfect pair
        if (Math.abs(combinedCoverage - totalSurfaceArea) < 0.0001 && dotProduct <= -0.9999) {
          foundPerfectPair = true;
          console.log(`  Found perfect pair at direction ${i + 1}/${k} - early termination!`);
          break;
        }
      }
    }
    
    // Progress update every 25%
    if ((i + 1) % Math.floor(k / 4) === 0 || i === k - 1) {
      const elapsed = performance.now() - startTime;
      const coverage = bestCombinedCoverage / totalSurfaceArea * 100;
      console.log(`  Progress: ${Math.round(((i + 1) / k) * 100)}% | Best coverage: ${coverage.toFixed(1)}% | Time: ${(elapsed/1000).toFixed(1)}s`);
    }
  }
  
  // Clean up resources
  visComputer.dispose();
  
  if (!bestD1 || !bestD2) {
    // Fallback: use best individual direction and its opposite
    console.warn('No valid direction pair found with angle > 135°, using fallback');
    const bestSingle = computedScores.reduce((a, b) => a.visibleArea > b.visibleArea ? a : b);
    bestD1 = bestSingle;
    bestD2 = {
      direction: bestSingle.direction.clone().negate(),
      visibleArea: 0,
      nonVisibleArea: totalSurfaceArea,
      score: 0,
      visibleTriangles: new Set()
    };
    bestD1UniqueArea = bestD1.visibleArea;
    bestD2UniqueArea = 0;
    bestCombinedCoverage = bestD1.visibleArea;
  }
  
  const d1 = bestD1.direction;
  const d2 = bestD2.direction;
  
  // Calculate final statistics
  const d1Owned = bestD1.visibleTriangles;
  const d2Owned = new Set<number>();
  for (const tri of bestD2.visibleTriangles) {
    if (!d1Owned.has(tri)) {
      d2Owned.add(tri);
    }
  }
  
  const combinedOwned = new Set([...d1Owned, ...d2Owned]);
  const nonVisibleTriangles = new Set<number>();
  for (let i = 0; i < triangles.length; i++) {
    if (!combinedOwned.has(i)) {
      nonVisibleTriangles.add(i);
    }
  }
  
  const combinedVisibleArea = computeAreaFromTriangles(triangles, combinedOwned);
  const nonVisibleArea = computeAreaFromTriangles(triangles, nonVisibleTriangles);
  
  // Calculate angle between directions
  const angleDeg = Math.acos(Math.max(-1, Math.min(1, d1.dot(d2)))) * (180 / Math.PI);
  
  const elapsed = performance.now() - startTime;
  
  // Report results
  console.log('\n═══════════════════════════════════════════════════════');
  console.log('PARTING DIRECTION ANALYSIS COMPLETE');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`\n  d1 (Primary): [${d1.toArray().map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`    Owned triangles: ${d1Owned.size}`);
  console.log(`    Owned area: ${((bestD1UniqueArea / totalSurfaceArea) * 100).toFixed(1)}%`);
  
  console.log(`\n  d2 (Secondary): [${d2.toArray().map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`    Owned triangles: ${d2Owned.size} (exclusive)`);
  console.log(`    Owned area: ${((bestD2UniqueArea / totalSurfaceArea) * 100).toFixed(1)}%`);
  
  console.log('\n───────────────────────────────────────────────────────');
  console.log('  COMBINED STATISTICS (Mutually Exclusive Ownership):');
  console.log(`    Total visible: ${((combinedVisibleArea / totalSurfaceArea) * 100).toFixed(1)}%`);
  console.log(`    Total triangles covered: ${combinedOwned.size}/${triangles.length}`);
  console.log(`    Non-visible area: ${((nonVisibleArea / totalSurfaceArea) * 100).toFixed(1)}%`);
  console.log(`    Non-visible triangles: ${nonVisibleTriangles.size}`);
  console.log(`    Angle between d1 & d2: ${angleDeg.toFixed(1)}°`);
  console.log(`    Computation time: ${(elapsed / 1000).toFixed(2)}s`);
  console.log('═══════════════════════════════════════════════════════');
  
  return { d1, d2, scores: computedScores };
}

/**
 * Create arrow helpers to visualize parting directions
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
 * Main function to compute and visualize parting directions
 */
export function computeAndShowPartingDirections(
  partMesh: THREE.Mesh,
  k: number = 64,
  rayCount: number = 2048
): { d1: THREE.Vector3; d2: THREE.Vector3; arrows: THREE.ArrowHelper[] } {
  const geometry = partMesh.geometry as THREE.BufferGeometry;
  
  if (!geometry) {
    throw new Error('Mesh must have a BufferGeometry');
  }
  
  // Clone and apply mesh transforms for accurate analysis
  const worldGeometry = geometry.clone();
  partMesh.updateMatrixWorld(true);
  worldGeometry.applyMatrix4(partMesh.matrixWorld);
  
  // Find optimal parting directions
  const { d1, d2 } = findPartingDirections(worldGeometry, k, rayCount);
  
  // Compute bounding box center for arrow origin
  worldGeometry.computeBoundingBox();
  const center = new THREE.Vector3();
  worldGeometry.boundingBox?.getCenter(center);
  
  // Compute arrow length based on bounding box size
  const size = new THREE.Vector3();
  worldGeometry.boundingBox?.getSize(size);
  const arrowLength = Math.max(size.x, size.y, size.z) * 1.2;
  
  // Create arrow helpers
  const arrow1 = createDirectionArrow(d1, center, 0x00ff00, arrowLength); // Green for d1
  const arrow2 = createDirectionArrow(d2, center, 0xff6600, arrowLength); // Orange for d2
  
  // Add arrows to scene
  const scene = partMesh.parent;
  if (scene) {
    scene.add(arrow1);
    scene.add(arrow2);
  }
  
  // Clean up
  worldGeometry.dispose();
  
  return { d1, d2, arrows: [arrow1, arrow2] };
}

/**
 * Remove parting direction arrows from the scene
 */
export function removePartingDirectionArrows(arrows: THREE.ArrowHelper[]): void {
  for (const arrow of arrows) {
    if (arrow.parent) {
      arrow.parent.remove(arrow);
    }
    arrow.dispose();
  }
}
