/**
 * Mold Half Classification
 * 
 * Classifies boundary triangles of a mold cavity into two mold halves (H₁ and H₂)
 * based on parting directions.
 * 
 * Algorithm:
 * 1. Identify outer boundary triangles by matching against hull geometry
 * 2. Initial classification using directional scoring (dot product with parting directions)
 * 3. Morphological opening (erosion + dilation) to remove thin peninsulas
 * 4. Laplacian smoothing with 2-ring neighborhood for smooth boundaries
 * 5. Orphan region removal using connected component analysis
 * 6. Re-apply directional constraints for strong alignments
 * 
 * Color convention:
 * - H₁ (Green): Triangles whose normals align with d1 (dot(n, d1) >= 0)
 * - H₂ (Orange): Triangles whose normals align with d2 (dot(n, d2) >= 0)
 * - Inner boundary (Dark gray): Triangles from the original part surface
 */

import * as THREE from 'three';

// ============================================================================
// TYPES
// ============================================================================

export interface MoldHalfClassificationResult {
  /** Map from triangle index to mold half (1 or 2), only for outer boundary triangles */
  sideMap: Map<number, 1 | 2>;
  /** Triangle indices belonging to mold half 1 (H₁) */
  h1Triangles: Set<number>;
  /** Triangle indices belonging to mold half 2 (H₂) */
  h2Triangles: Set<number>;
  /** Triangle indices belonging to inner boundary (part surface) - not classified */
  innerBoundaryTriangles: Set<number>;
  /** Total triangles in boundary mesh */
  totalTriangles: number;
  /** Total outer boundary triangles (H₁ + H₂) */
  outerBoundaryCount: number;
}

interface TriangleInfo {
  normal: THREE.Vector3;
  centroid: THREE.Vector3;
  area: number;
}

// ============================================================================
// COLORS FOR MOLD HALVES (Same as parting directions)
// ============================================================================

/** Paint colors for vertex coloring (RGB 0-255) */
const MOLD_HALF_COLORS: Record<string, { r: number; g: number; b: number }> = {
  H1: { r: 0, g: 255, b: 0 },       // Green (same as D1) - faces extracted by d1
  H2: { r: 255, g: 102, b: 0 },     // Orange (same as D2) - faces extracted by d2
  INNER: { r: 80, g: 80, b: 80 },   // Dark gray for inner boundary (part surface)
  UNCLASSIFIED: { r: 136, g: 136, b: 136 }, // Gray
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Extract triangle information from a BufferGeometry
 * Returns array of normals, centroids, and areas for each triangle
 */
function extractTriangleInfo(geometry: THREE.BufferGeometry): TriangleInfo[] {
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  const triangleCount = index ? index.count / 3 : position.count / 3;
  
  const triangles: TriangleInfo[] = [];
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const edge1 = new THREE.Vector3();
  const edge2 = new THREE.Vector3();
  const normal = new THREE.Vector3();
  
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
    
    edge1.subVectors(vB, vA);
    edge2.subVectors(vC, vA);
    normal.crossVectors(edge1, edge2);
    const area = normal.length() * 0.5;
    normal.normalize();
    
    triangles.push({
      normal: normal.clone(),
      centroid: new THREE.Vector3().add(vA).add(vB).add(vC).divideScalar(3),
      area,
    });
  }
  
  return triangles;
}

/**
 * Build triangle adjacency map using half-edge structure
 * Each triangle index maps to array of neighboring triangle indices
 * (triangles sharing at least one edge)
 * 
 * OPTIMIZED: Uses Set for deduplication instead of includes()
 */
function buildTriangleAdjacency(geometry: THREE.BufferGeometry): Map<number, number[]> {
  const position = geometry.getAttribute('position');
  const index = geometry.getIndex();
  const triangleCount = index ? index.count / 3 : position.count / 3;
  
  // Map from edge key to list of triangles sharing that edge
  const edgeToTriangles = new Map<string, number[]>();
  
  // Pre-compute vertex keys to avoid redundant toFixed calls
  const vertexCount = position.count;
  const vertexKeys: string[] = new Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    const x = position.getX(i).toFixed(5);
    const y = position.getY(i).toFixed(5);
    const z = position.getZ(i).toFixed(5);
    vertexKeys[i] = `${x},${y},${z}`;
  }
  
  const createEdgeKey = (v1Key: string, v2Key: string): string => {
    return v1Key < v2Key ? `${v1Key}|${v2Key}` : `${v2Key}|${v1Key}`;
  };
  
  // Build edge-to-triangles map
  for (let triIdx = 0; triIdx < triangleCount; triIdx++) {
    let a: number, b: number, c: number;
    
    if (index) {
      a = index.getX(triIdx * 3);
      b = index.getX(triIdx * 3 + 1);
      c = index.getX(triIdx * 3 + 2);
    } else {
      a = triIdx * 3;
      b = triIdx * 3 + 1;
      c = triIdx * 3 + 2;
    }
    
    const keyA = vertexKeys[a];
    const keyB = vertexKeys[b];
    const keyC = vertexKeys[c];
    
    // Three edges per triangle
    const edges = [
      createEdgeKey(keyA, keyB),
      createEdgeKey(keyB, keyC),
      createEdgeKey(keyC, keyA),
    ];
    
    for (const edge of edges) {
      let tris = edgeToTriangles.get(edge);
      if (!tris) {
        tris = [];
        edgeToTriangles.set(edge, tris);
      }
      tris.push(triIdx);
    }
  }
  
  // Build adjacency map using Sets for O(1) deduplication
  const adjacencySets = new Map<number, Set<number>>();
  for (let triIdx = 0; triIdx < triangleCount; triIdx++) {
    adjacencySets.set(triIdx, new Set());
  }
  
  for (const triangles of edgeToTriangles.values()) {
    for (let i = 0; i < triangles.length; i++) {
      for (let j = i + 1; j < triangles.length; j++) {
        const triA = triangles[i];
        const triB = triangles[j];
        adjacencySets.get(triA)!.add(triB);
        adjacencySets.get(triB)!.add(triA);
      }
    }
  }
  
  // Convert Sets to Arrays for faster iteration
  const adjacency = new Map<number, number[]>();
  for (const [triIdx, neighbors] of adjacencySets) {
    adjacency.set(triIdx, Array.from(neighbors));
  }
  
  return adjacency;
}

/**
 * Extract hull face planes (normal + distance) for distance-to-surface testing.
 * For a convex hull, each face defines a half-space.
 */
function extractHullFacePlanes(hullGeometry: THREE.BufferGeometry): { normal: THREE.Vector3; d: number }[] {
  const position = hullGeometry.getAttribute('position');
  const index = hullGeometry.getIndex();
  const triangleCount = index ? index.count / 3 : position.count / 3;
  
  const planes: { normal: THREE.Vector3; d: number }[] = [];
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const normal = new THREE.Vector3();
  
  // Use a Map to deduplicate planes by normal direction (convex hull faces)
  const normalKeys = new Set<string>();
  
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
    
    ab.subVectors(vB, vA);
    ac.subVectors(vC, vA);
    normal.crossVectors(ab, ac).normalize();
    
    if (normal.length() < 0.001) continue; // Degenerate triangle
    
    // Check if we already have this plane (same normal direction)
    const normalKey = `${normal.x.toFixed(3)},${normal.y.toFixed(3)},${normal.z.toFixed(3)}`;
    if (normalKeys.has(normalKey)) continue;
    normalKeys.add(normalKey);
    
    // Plane equation: n · p = d, where d = n · pointOnPlane
    const d = normal.dot(vA);
    planes.push({ normal: normal.clone(), d });
  }
  
  return planes;
}

/**
 * Identify outer boundary triangles by checking if their centroids lie on the hull surface.
 * 
 * The cavity mesh is created by CSG: Hull - Part
 * Outer boundary = triangles on the Hull surface (distance to hull surface ≈ 0)
 * Inner boundary = triangles from the Part surface (inside the hull)
 * 
 * For a convex hull, a point is on the surface if it satisfies at least one face
 * plane equation (n · p ≈ d) with near-zero distance.
 * 
 * @param cavityTriangles - Triangle info from the cavity mesh
 * @param hullGeometry - The original hull geometry (in world space)
 * @returns Set of triangle indices that belong to the outer boundary (hull surface)
 */
function identifyOuterBoundaryFromHull(
  cavityTriangles: TriangleInfo[],
  hullGeometry: THREE.BufferGeometry
): Set<number> {
  const planes = extractHullFacePlanes(hullGeometry);
  const outerTriangles = new Set<number>();
  
  // Compute hull bounding box to determine appropriate tolerance
  hullGeometry.computeBoundingBox();
  const bbox = hullGeometry.boundingBox!;
  const hullSize = new THREE.Vector3();
  bbox.getSize(hullSize);
  const maxDim = Math.max(hullSize.x, hullSize.y, hullSize.z);
  
  // Tolerance: small fraction of hull size (handles different model scales)
  const tolerance = maxDim * 0.001; // 0.1% of hull size
  
  for (let i = 0; i < cavityTriangles.length; i++) {
    const centroid = cavityTriangles[i].centroid;
    
    // Check if centroid lies on any hull face plane
    let isOnHullSurface = false;
    for (const plane of planes) {
      const dist = Math.abs(plane.normal.dot(centroid) - plane.d);
      if (dist < tolerance) {
        isOnHullSurface = true;
        break;
      }
    }
    
    if (isOnHullSurface) {
      outerTriangles.add(i);
    }
  }
  
  return outerTriangles;
}

/**
 * Reclassify all outer boundary triangles based on directional alignment
 * with morphological smoothing for clean boundaries.
 * 
 * ALGORITHM:
 * 1. Initial classification by directional preference
 * 2. Morphological opening (erode + dilate) to remove thin peninsulas
 * 3. Laplacian smoothing for final cleanup
 * 4. Orphan removal for isolated regions
 * 5. Re-apply strong directional constraints
 */
function classifyAndSmooth(
  triangles: TriangleInfo[],
  adjacency: Map<number, number[]>,
  outerTriangles: Set<number>,
  d1: THREE.Vector3,
  d2: THREE.Vector3
): Map<number, 1 | 2> {
  const side = new Map<number, 1 | 2>();
  
  // Step 1: Initial classification by directional preference
  for (const i of outerTriangles) {
    const normal = triangles[i].normal;
    const dot1 = normal.dot(d1);
    const dot2 = normal.dot(d2);
    side.set(i, dot1 >= dot2 ? 1 : 2);
  }
  
  // Pre-compute 1-ring neighbors (filtered to outer triangles only)
  const neighbors = new Map<number, number[]>();
  for (const i of outerTriangles) {
    neighbors.set(i, (adjacency.get(i) || []).filter(n => outerTriangles.has(n)));
  }
  
  // Step 2: Morphological opening to remove peninsulas
  // Opening = Erosion followed by Dilation
  // This removes thin protrusions while preserving the overall shape
  
  const erodeIterations = 2;
  const dilateIterations = 2;
  
  // Erode H1 (shrink H1 by converting boundary H1 triangles to H2)
  for (let iter = 0; iter < erodeIterations; iter++) {
    const toFlip: number[] = [];
    for (const i of outerTriangles) {
      if (side.get(i) !== 1) continue;
      
      const nbrs = neighbors.get(i)!;
      let oppCount = 0;
      for (const n of nbrs) {
        if (side.get(n) === 2) oppCount++;
      }
      
      // Erode: if any neighbor is opposite, and we're not deeply embedded
      if (oppCount > 0 && oppCount >= nbrs.length / 2) {
        toFlip.push(i);
      }
    }
    for (const i of toFlip) side.set(i, 2);
    if (toFlip.length === 0) break;
  }
  
  // Dilate H1 (grow H1 back by converting boundary H2 triangles to H1)
  for (let iter = 0; iter < dilateIterations; iter++) {
    const toFlip: number[] = [];
    for (const i of outerTriangles) {
      if (side.get(i) !== 2) continue;
      
      const nbrs = neighbors.get(i)!;
      let h1Count = 0;
      for (const n of nbrs) {
        if (side.get(n) === 1) h1Count++;
      }
      
      // Dilate: if we have H1 neighbors and they're majority
      if (h1Count > 0 && h1Count >= nbrs.length / 2) {
        toFlip.push(i);
      }
    }
    for (const i of toFlip) side.set(i, 1);
    if (toFlip.length === 0) break;
  }
  
  // Now do the same for H2 (erode H2, dilate H2)
  for (let iter = 0; iter < erodeIterations; iter++) {
    const toFlip: number[] = [];
    for (const i of outerTriangles) {
      if (side.get(i) !== 2) continue;
      
      const nbrs = neighbors.get(i)!;
      let oppCount = 0;
      for (const n of nbrs) {
        if (side.get(n) === 1) oppCount++;
      }
      
      if (oppCount > 0 && oppCount >= nbrs.length / 2) {
        toFlip.push(i);
      }
    }
    for (const i of toFlip) side.set(i, 1);
    if (toFlip.length === 0) break;
  }
  
  for (let iter = 0; iter < dilateIterations; iter++) {
    const toFlip: number[] = [];
    for (const i of outerTriangles) {
      if (side.get(i) !== 1) continue;
      
      const nbrs = neighbors.get(i)!;
      let h2Count = 0;
      for (const n of nbrs) {
        if (side.get(n) === 2) h2Count++;
      }
      
      if (h2Count > 0 && h2Count >= nbrs.length / 2) {
        toFlip.push(i);
      }
    }
    for (const i of toFlip) side.set(i, 2);
    if (toFlip.length === 0) break;
  }
  
  // Step 3: Laplacian smoothing - only flip if STRONGLY outnumbered
  for (let pass = 0; pass < 5; pass++) {
    const toFlip: number[] = [];
    
    for (const i of outerTriangles) {
      const currentSide = side.get(i)!;
      const nbrs = neighbors.get(i)!;
      
      let h1Count = 0;
      let h2Count = 0;
      for (const n of nbrs) {
        if (side.get(n) === 1) h1Count++;
        else h2Count++;
      }
      
      // Only flip if ratio is > 2:1 (strong majority)
      if (currentSide === 1 && h2Count > h1Count * 2) {
        toFlip.push(i);
      } else if (currentSide === 2 && h1Count > h2Count * 2) {
        toFlip.push(i);
      }
    }
    
    if (toFlip.length === 0) break;
    for (const i of toFlip) {
      side.set(i, side.get(i) === 1 ? 2 : 1);
    }
  }
  
  // Step 4: Remove orphan regions
  removeOrphans(side, adjacency, outerTriangles, 40);
  
  // Step 5: Re-apply strong directional constraints
  // Triangles with normals strongly aligned to a direction must be that side
  const strongThreshold = 0.6;
  for (const i of outerTriangles) {
    const normal = triangles[i].normal;
    const dot1 = normal.dot(d1);
    const dot2 = normal.dot(d2);
    
    if (dot1 > strongThreshold && dot1 > dot2 + 0.2) {
      side.set(i, 1);
    } else if (dot2 > strongThreshold && dot2 > dot1 + 0.2) {
      side.set(i, 2);
    }
  }
  
  // Final orphan cleanup
  removeOrphans(side, adjacency, outerTriangles, 25);
  
  return side;
}

/**
 * Remove small isolated regions by flipping them to match surroundings
 */
function removeOrphans(
  side: Map<number, 1 | 2>,
  adjacency: Map<number, number[]>,
  outerTriangles: Set<number>,
  maxSize: number
): void {
  const processed = new Set<number>();
  
  for (const startIdx of outerTriangles) {
    if (processed.has(startIdx)) continue;
    
    const startSide = side.get(startIdx)!;
    
    // BFS to find connected component (using array as queue with index pointer)
    const component: number[] = [startIdx];
    processed.add(startIdx);
    let head = 0;
    
    while (head < component.length) {
      const idx = component[head++];
      for (const neighbor of adjacency.get(idx) || []) {
        if (processed.has(neighbor)) continue;
        if (!outerTriangles.has(neighbor)) continue;
        if (side.get(neighbor) !== startSide) continue;
        
        processed.add(neighbor);
        component.push(neighbor);
      }
    }
    
    // Flip small components
    if (component.length <= maxSize) {
      const newSide: 1 | 2 = startSide === 1 ? 2 : 1;
      for (const idx of component) {
        side.set(idx, newSide);
      }
    }
  }
}

// ============================================================================
// MAIN CLASSIFICATION FUNCTION
// ============================================================================

/**
 * Classify boundary triangles of mold cavity into two mold halves
 * Only classifies the OUTER boundary (hull surface), not the inner boundary (part surface)
 * 
 * @param cavityGeometry - The mold cavity mesh geometry (in world space)
 * @param hullGeometry - The original hull geometry (in world space) - used to identify outer boundary
 * @param d1 - First parting direction (pull direction for H₁)
 * @param d2 - Second parting direction (pull direction for H₂)
 * @returns MoldHalfClassificationResult with classification data
 */
export function classifyMoldHalves(
  cavityGeometry: THREE.BufferGeometry,
  hullGeometry: THREE.BufferGeometry,
  d1: THREE.Vector3,
  d2: THREE.Vector3
): MoldHalfClassificationResult {
  const startTime = performance.now();
  
  // Step 1: Extract triangle information from cavity
  const triangles = extractTriangleInfo(cavityGeometry);
  
  // Step 2: Identify outer boundary triangles by matching against hull geometry
  const outerTriangles = identifyOuterBoundaryFromHull(triangles, hullGeometry);
  const innerBoundaryTriangles = new Set<number>();
  for (let i = 0; i < triangles.length; i++) {
    if (!outerTriangles.has(i)) {
      innerBoundaryTriangles.add(i);
    }
  }
  
  // If hull matching found very few outer triangles, treat ALL as outer
  // This handles cases where CSG has modified triangle geometry
  let effectiveOuterTriangles = outerTriangles;
  if (outerTriangles.size < triangles.length * 0.1) {
    console.warn('Mold Half Classification: Hull matching found too few triangles, using all triangles');
    effectiveOuterTriangles = new Set<number>();
    for (let i = 0; i < triangles.length; i++) {
      effectiveOuterTriangles.add(i);
    }
  }
  
  // Step 3: Build triangle adjacency on the FULL cavity mesh
  const adjacency = buildTriangleAdjacency(cavityGeometry);
  
  // Early exit if no outer triangles
  if (effectiveOuterTriangles.size === 0) {
    console.warn('Mold Half Classification: No outer triangles to classify');
    return {
      sideMap: new Map(),
      h1Triangles: new Set(),
      h2Triangles: new Set(),
      innerBoundaryTriangles,
      totalTriangles: triangles.length,
      outerBoundaryCount: 0,
    };
  }
  
  // Step 4: Classify and smooth
  const side = classifyAndSmooth(triangles, adjacency, effectiveOuterTriangles, d1, d2);
  
  // Build result sets
  const h1Triangles = new Set<number>();
  const h2Triangles = new Set<number>();
  
  for (const [triIdx, moldSide] of side) {
    if (moldSide === 1) {
      h1Triangles.add(triIdx);
    } else {
      h2Triangles.add(triIdx);
    }
  }
  
  const elapsed = performance.now() - startTime;
  
  // Summary logging
  console.log(`Mold Half Classification: H₁=${h1Triangles.size} (${(h1Triangles.size / effectiveOuterTriangles.size * 100).toFixed(1)}%), H₂=${h2Triangles.size} (${(h2Triangles.size / effectiveOuterTriangles.size * 100).toFixed(1)}%), Inner=${innerBoundaryTriangles.size} [${elapsed.toFixed(0)}ms]`);
  
  return {
    sideMap: side,
    h1Triangles,
    h2Triangles,
    innerBoundaryTriangles,
    totalTriangles: triangles.length,
    outerBoundaryCount: effectiveOuterTriangles.size,
  };
}

// ============================================================================
// VISUALIZATION
// ============================================================================

/**
 * Apply vertex colors to visualize mold half classification
 * Paints H₁ triangles with D1 color (green), H₂ with D2 color (orange),
 * and inner boundary triangles with dark gray
 * 
 * NOTE: The mesh geometry MUST already be non-indexed before calling this function.
 * This is done in ThreeViewer before classification to ensure triangle indices match.
 */
export function applyMoldHalfPaint(
  mesh: THREE.Mesh,
  classification: MoldHalfClassificationResult
): void {
  const geometry = mesh.geometry as THREE.BufferGeometry;
  if (!geometry) return;
  
  // Geometry should already be non-indexed (converted before classification)
  if (geometry.index) {
    console.warn('applyMoldHalfPaint: Geometry is still indexed, triangle indices may not match');
  }
  
  const position = geometry.getAttribute('position');
  const triangleCount = position.count / 3;
  
  // Create color buffer
  const colorArray = new Uint8Array(position.count * 3);
  const colorAttr = new THREE.BufferAttribute(colorArray, 3, true);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('color', colorAttr);
  
  for (let triIdx = 0; triIdx < triangleCount; triIdx++) {
    let color: { r: number; g: number; b: number };
    
    // Check if this is an inner boundary triangle
    if (classification.innerBoundaryTriangles.has(triIdx)) {
      color = MOLD_HALF_COLORS.INNER;
    } else {
      // Outer boundary - check classification
      const side = classification.sideMap.get(triIdx);
      if (side === 1) {
        color = MOLD_HALF_COLORS.H1;
      } else if (side === 2) {
        color = MOLD_HALF_COLORS.H2;
      } else {
        color = MOLD_HALF_COLORS.UNCLASSIFIED;
      }
    }
    
    // Paint all three vertices of the triangle
    const baseIdx = triIdx * 3;
    for (let v = 0; v < 3; v++) {
      const idx = (baseIdx + v) * 3;
      colorArray[idx] = color.r;
      colorArray[idx + 1] = color.g;
      colorArray[idx + 2] = color.b;
    }
  }
  
  colorAttr.needsUpdate = true;
  
  // Configure material for vertex colors
  const material = mesh.material as THREE.Material & { vertexColors?: boolean; color?: THREE.Color; opacity?: number };
  if (material) {
    material.vertexColors = true;
    material.color?.setRGB(1, 1, 1);
    // Increase opacity to better see the colors
    if (material.opacity !== undefined && material.opacity < 0.8) {
      material.opacity = 0.85;
    }
    material.needsUpdate = true;
  }
}

/**
 * Remove mold half paint and restore original appearance
 */
export function removeMoldHalfPaint(mesh: THREE.Mesh): void {
  const geometry = mesh.geometry as THREE.BufferGeometry;
  if (!geometry) return;
  
  geometry.deleteAttribute('color');
  
  const material = mesh.material as THREE.Material & { vertexColors?: boolean; color?: THREE.Color; opacity?: number };
  if (material) {
    material.vertexColors = false;
    // Restore original cavity color
    material.color?.setHex(0x00ffaa); // CAVITY_COLOR
    material.opacity = 0.7;
    material.needsUpdate = true;
  }
}
