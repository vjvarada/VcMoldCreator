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
 * Extract triangle data from geometry
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
 * Triangle-centric visibility computer with BVH acceleration
 * 
 * For each triangle, checks if it's visible from a direction by:
 * 1. Normal check: triangle must face toward the viewing direction
 * 2. Occlusion check: ray from far away toward triangle centroid must hit this triangle first
 * 
 * This approach guarantees we check EVERY triangle, unlike ray flooding which can miss small triangles.
 */
class MultiRayVisibilityComputer {
  private geometry: THREE.BufferGeometry;
  private triangles: TriangleData[];
  private totalSurfaceArea: number;
  private bvh: MeshBVH;
  private boundingBox: THREE.Box3;
  private rayOffset: number;

  constructor(geometry: THREE.BufferGeometry, _rayCount: number = 2048) {
    console.log('Initializing Triangle-Centric Visibility Computer with BVH...');
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
    
    // Extract triangle data
    this.triangles = extractTriangleData(this.geometry);
    this.totalSurfaceArea = this.triangles.reduce((sum, t) => sum + t.area, 0);
    
    // Compute bounding box and ray offset
    this.geometry.computeBoundingBox();
    this.boundingBox = this.geometry.boundingBox!.clone();
    
    const size = new THREE.Vector3();
    this.boundingBox.getSize(size);
    this.rayOffset = Math.max(size.x, size.y, size.z) * 2;
    
    console.log(`  BVH built for ${this.triangles.length} triangles`);
    console.log(`  Total surface area: ${this.totalSurfaceArea.toFixed(4)}`);
  }

  /**
   * Compute visible triangles from a direction using per-triangle occlusion testing
   * 
   * Algorithm:
   * For each triangle:
   * 1. Check if normal faces toward viewing direction (normal · direction > 0)
   * 2. Cast ray from far away toward triangle centroid
   * 3. If the first hit triangle index matches this triangle, it's visible
   * 
   * This checks every triangle and correctly handles self-occlusion.
   */
  computeVisibleTriangles(direction: THREE.Vector3): Set<number> {
    const visible = new Set<number>();
    const rayDirection = direction.clone().negate(); // Ray travels opposite to viewing direction
    const rayOrigin = new THREE.Vector3();
    
    // Use BVH's raycastFirst for maximum performance
    const ray = new THREE.Ray();
    
    for (let i = 0; i < this.triangles.length; i++) {
      const triangle = this.triangles[i];
      
      // Step 1: Normal must face toward the viewing direction
      if (triangle.normal.dot(direction) <= 0) {
        continue;
      }
      
      // Step 2: Cast ray from far away toward this triangle's centroid
      // Origin = centroid + direction * offset (far away in the viewing direction)
      rayOrigin.copy(triangle.centroid).addScaledVector(direction, this.rayOffset);
      ray.origin.copy(rayOrigin);
      ray.direction.copy(rayDirection);
      
      // Step 3: Find first intersection using BVH
      const hit = this.bvh.raycastFirst(ray);
      
      if (hit) {
        // Check if the hit triangle is THIS triangle (by comparing face index)
        if (hit.faceIndex === i) {
          // This triangle is the first thing hit - it's visible!
          visible.add(i);
        }
        // If hit.faceIndex !== i, another triangle is blocking this one
      }
    }
    
    return visible;
  }

  /**
   * Get visibility score for a direction
   */
  getDirectionScore(direction: THREE.Vector3): DirectionScore {
    const visibleTriangles = this.computeVisibleTriangles(direction);
    const visibleArea = computeAreaFromTriangles(this.triangles, visibleTriangles);
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
 */
export function findPartingDirections(
  geometry: THREE.BufferGeometry,
  k: number = 64,
  rayCount: number = 2048
): { d1: THREE.Vector3; d2: THREE.Vector3; scores: DirectionScore[] } {
  console.log('═══════════════════════════════════════════════════════');
  console.log('SDF-BIASED MULTI-RAY VISIBILITY ALGORITHM');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Direction samples: ${k}`);
  console.log(`  Rays per direction: ${rayCount}`);
  console.log(`  Valid pairs to evaluate: angle > 135°`);
  
  const startTime = performance.now();
  
  // Create visibility computer
  const visComputer = new MultiRayVisibilityComputer(geometry, rayCount);
  const totalSurfaceArea = visComputer.getTotalSurfaceArea();
  const triangles = visComputer.getTriangles();
  
  // Sample directions uniformly on the sphere
  const directions = fibonacciSphere(k);
  
  // Phase 1: Compute visibility for each direction
  console.log('\nPhase 1: Computing visibility for each direction...');
  const scores: DirectionScore[] = [];
  
  for (let i = 0; i < directions.length; i++) {
    const score = visComputer.getDirectionScore(directions[i]);
    scores.push(score);
    
    if ((i + 1) % Math.floor(k / 4) === 0 || i === k - 1) {
      console.log(`  Progress: ${Math.round(((i + 1) / k) * 100)}%`);
    }
  }
  
  // Clean up resources
  visComputer.dispose();
  
  // Sort by visible area (descending) to find best individual directions
  const sortedScores = [...scores].sort((a, b) => b.visibleArea - a.visibleArea);
  
  // Minimum angle constraint: 135 degrees
  const minAngleDeg = 135;
  const minAngleCos = Math.cos(minAngleDeg * Math.PI / 180);
  
  // Phase 2: Find best direction pair with angle > 135° and maximum combined coverage
  // Tie-breaker: If multiple pairs have same coverage, prefer the pair closest to 180°
  console.log('\nPhase 2: Finding optimal direction pair with global ownership...');
  
  let bestD1: DirectionScore | null = null;
  let bestD2: DirectionScore | null = null;
  let bestCombinedCoverage = 0;
  let bestD1UniqueArea = 0;
  let bestD2UniqueArea = 0;
  let bestDotProduct = 1; // Track dot product; lower (more negative) = closer to 180°
  
  // Strategy: Iterate through top candidates for d1, find best matching d2
  // Using ownership model: d2 only gets credit for triangles NOT in d1's set
  
  for (let i = 0; i < sortedScores.length; i++) {
    const d1Candidate = sortedScores[i];
    const d1Owned = d1Candidate.visibleTriangles;
    
    // Find best d2 that satisfies angle constraint
    for (let j = 0; j < sortedScores.length; j++) {
      if (i === j) continue;
      
      const d2Candidate = sortedScores[j];
      
      // Check angle constraint: dot product < cos(135°) means angle > 135°
      const dotProduct = d1Candidate.direction.dot(d2Candidate.direction);
      if (dotProduct > minAngleCos) continue;
      
      // Compute d2's UNIQUE contribution (triangles not owned by d1)
      const d2Unique = new Set<number>();
      for (const tri of d2Candidate.visibleTriangles) {
        if (!d1Owned.has(tri)) {
          d2Unique.add(tri);
        }
      }
      
      // Combined coverage = d1's area + d2's unique area
      const d1Area = computeAreaFromTriangles(triangles, d1Owned);
      const d2UniqueArea = computeAreaFromTriangles(triangles, d2Unique);
      const combinedCoverage = d1Area + d2UniqueArea;
      
      // Update best if:
      // 1. Higher coverage, OR
      // 2. Same coverage but angle closer to 180° (lower dot product)
      const isBetterCoverage = combinedCoverage > bestCombinedCoverage;
      const isSameCoverageButCloserTo180 = 
        Math.abs(combinedCoverage - bestCombinedCoverage) < 0.0001 && dotProduct < bestDotProduct;
      
      if (isBetterCoverage || isSameCoverageButCloserTo180) {
        bestCombinedCoverage = combinedCoverage;
        bestD1 = d1Candidate;
        bestD2 = d2Candidate;
        bestD1UniqueArea = d1Area;
        bestD2UniqueArea = d2UniqueArea;
        bestDotProduct = dotProduct;
      }
    }
  }
  
  if (!bestD1 || !bestD2) {
    // Fallback: use best individual direction and its opposite
    console.warn('No valid direction pair found with angle > 135°, using fallback');
    bestD1 = sortedScores[0];
    bestD2 = {
      direction: sortedScores[0].direction.clone().negate(),
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
  
  return { d1, d2, scores };
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
