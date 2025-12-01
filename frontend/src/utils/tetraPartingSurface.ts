/**
 * Tetrahedral Parting Surface Calculation
 * 
 * This module computes the parting surface for mold design using tetrahedral mesh
 * edges with Dijkstra's algorithm, similar to the voxel-based approach but with
 * better geometric fidelity.
 * 
 * Algorithm:
 * 1. Build edge graph from tetrahedral mesh
 * 2. Compute geodesic distance from each edge midpoint to part surface M
 * 3. For shell boundary edges only: compute biased distance with shell penalty
 * 4. Compute edge weighting factors:
 *    - Internal edges: w(e) = 1/[(d_e)² + ε] where d_e = distance to part
 *    - Shell boundary edges: w(e) = 1/[(d_biased)² + ε] 
 *      where d_biased = d_e + (R - δ_w), δ_w = distance to shell
 * 5. Edge cost = l(e) * w(e) where l(e) is edge Euclidean length
 * 6. Run multi-source Dijkstra from seed vertices (near part) to boundary
 * 7. Label vertices based on which boundary (H₁ or H₂) they reach with lowest cost
 * 8. Parting surface = edges connecting differently labeled vertices
 */

import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';
import type { TetrahedralizationResult } from './tetrahedralization';
import type { MoldHalfClassificationResult } from './moldHalfClassification';

// ============================================================================
// TYPES
// ============================================================================

/**
 * Edge in the tetrahedral mesh graph
 */
export interface TetraEdge {
  /** First vertex index */
  v0: number;
  /** Second vertex index */
  v1: number;
  /** Euclidean length of edge */
  length: number;
  /** Edge midpoint */
  midpoint: THREE.Vector3;
  /** Geodesic distance from midpoint to part surface */
  distToPart: number;
  /** Geodesic distance from midpoint to shell surface */
  distToShell: number;
  /** Whether this edge is on the shell boundary (distToShell < threshold) */
  isOnShellBoundary: boolean;
  /** 
   * Effective distance used for weighting:
   * - Internal edges: distToPart
   * - Shell boundary edges: distToPart + (R - distToShell)
   */
  effectiveDist: number;
  /** Weighting factor: 1/(effectiveDist² + ε) */
  weight: number;
  /** Edge cost for Dijkstra: length * weight */
  cost: number;
}

/**
 * Adjacency list for graph representation
 */
export interface TetraGraph {
  /** Number of vertices */
  numVertices: number;
  /** Number of unique edges */
  numEdges: number;
  /** Number of tetrahedra */
  numTetrahedra: number;
  /** For each vertex, list of (neighborVertex, edgeIndex) pairs */
  adjacency: Array<Array<{ neighbor: number; edgeIndex: number }>>;
  /** All edges */
  edges: TetraEdge[];
  /** Vertex positions */
  vertices: Float32Array;
  /** Tetrahedra indices (4 vertices per tetrahedron) */
  tetrahedra: Uint32Array;
}

/**
 * Options for computing weighting factors
 */
export interface TetraWeightingOptions {
  /** Epsilon to avoid division by zero (default: 0.25) */
  epsilon?: number;
  /** Shell radius R for bias calculation (default: auto-computed from mesh) */
  shellRadius?: number;
  /** 
   * Threshold for classifying edge as "on shell boundary" 
   * Edges with distToShell < threshold are considered boundary edges
   * Default: auto-computed as 0.5 * average edge length
   */
  shellBoundaryThreshold?: number;
}

/**
 * Result of weighting factor computation
 */
export interface TetraWeightingResult {
  /** The tetrahedral graph with edge weights */
  graph: TetraGraph;
  /** Statistics */
  stats: {
    minDistToPart: number;
    maxDistToPart: number;
    minDistToShell: number;
    maxDistToShell: number;
    minEffectiveDist: number;
    maxEffectiveDist: number;
    minWeight: number;
    maxWeight: number;
    minEdgeCost: number;
    maxEdgeCost: number;
    shellRadius: number;
    shellBoundaryThreshold: number;
    numInternalEdges: number;
    numBoundaryEdges: number;
  };
  /** Computation time in ms */
  computeTimeMs: number;
}

// ============================================================================
// GRAPH CONSTRUCTION
// ============================================================================

/**
 * Build a graph from tetrahedral mesh edges
 * Each tetrahedron has 6 edges, but we only store unique edges
 */
export function buildTetraGraph(tetraResult: TetrahedralizationResult): TetraGraph {
  const startTime = performance.now();
  
  const vertices = tetraResult.vertices;
  const tetrahedra = tetraResult.tetrahedra;
  const numVertices = tetraResult.numVertices;
  const numTetrahedra = tetraResult.numTetrahedra;
  
  console.log(`Building tetrahedral graph...`);
  console.log(`  Input: ${numVertices} vertices, ${numTetrahedra} tetrahedra`);
  
  // Use a Set to track unique edges (stored as "min_max" string key)
  const edgeSet = new Map<string, { v0: number; v1: number }>();
  
  // Helper to add an edge
  const addEdge = (a: number, b: number) => {
    const v0 = Math.min(a, b);
    const v1 = Math.max(a, b);
    const key = `${v0}_${v1}`;
    if (!edgeSet.has(key)) {
      edgeSet.set(key, { v0, v1 });
    }
  };
  
  // Iterate through all tetrahedra and collect edges
  // Each tetrahedron has vertices at indices [4*t, 4*t+1, 4*t+2, 4*t+3]
  // The 6 edges are: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
  for (let t = 0; t < numTetrahedra; t++) {
    const i0 = tetrahedra[t * 4];
    const i1 = tetrahedra[t * 4 + 1];
    const i2 = tetrahedra[t * 4 + 2];
    const i3 = tetrahedra[t * 4 + 3];
    
    addEdge(i0, i1);
    addEdge(i0, i2);
    addEdge(i0, i3);
    addEdge(i1, i2);
    addEdge(i1, i3);
    addEdge(i2, i3);
  }
  
  // Build edges array with geometric properties
  const edges: TetraEdge[] = [];
  const adjacency: Array<Array<{ neighbor: number; edgeIndex: number }>> = 
    new Array(numVertices).fill(null).map(() => []);
  
  let edgeIndex = 0;
  for (const [, edge] of edgeSet) {
    const v0 = edge.v0;
    const v1 = edge.v1;
    
    // Get vertex positions
    const p0 = new THREE.Vector3(
      vertices[v0 * 3],
      vertices[v0 * 3 + 1],
      vertices[v0 * 3 + 2]
    );
    const p1 = new THREE.Vector3(
      vertices[v1 * 3],
      vertices[v1 * 3 + 1],
      vertices[v1 * 3 + 2]
    );
    
    // Compute edge properties
    const length = p0.distanceTo(p1);
    const midpoint = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
    
    edges.push({
      v0,
      v1,
      length,
      midpoint,
      distToPart: 0,  // Will be computed later
      distToShell: 0,
      isOnShellBoundary: false,
      effectiveDist: 0,
      weight: 0,
      cost: 0,
    });
    
    // Add to adjacency lists
    adjacency[v0].push({ neighbor: v1, edgeIndex });
    adjacency[v1].push({ neighbor: v0, edgeIndex });
    
    edgeIndex++;
  }
  
  const elapsed = performance.now() - startTime;
  console.log(`  Graph built: ${edges.length} unique edges`);
  console.log(`  Average edges per vertex: ${(edges.length * 2 / numVertices).toFixed(1)}`);
  console.log(`  Time: ${elapsed.toFixed(1)} ms`);
  
  return {
    numVertices,
    numEdges: edges.length,
    numTetrahedra,
    adjacency,
    edges,
    vertices,
    tetrahedra,
  };
}

/**
 * Build a MINIMAL graph structure for validation only.
 * Only includes vertices and tetrahedra - no edge computation.
 * Much faster than buildTetraGraph since we skip edge enumeration.
 */
export function buildMinimalTetraGraph(tetraResult: TetrahedralizationResult): TetraGraph {
  const startTime = performance.now();
  
  const vertices = tetraResult.vertices;
  const tetrahedra = tetraResult.tetrahedra;
  const numVertices = tetraResult.numVertices;
  const numTetrahedra = tetraResult.numTetrahedra;
  
  console.log(`Building minimal tetrahedral graph (validation mode)...`);
  console.log(`  Input: ${numVertices} vertices, ${numTetrahedra} tetrahedra`);
  
  const elapsed = performance.now() - startTime;
  console.log(`  Time: ${elapsed.toFixed(1)} ms`);
  
  // Return minimal structure - edges and adjacency are empty
  // This is sufficient for boundary face detection which only needs tetrahedra
  return {
    numVertices,
    numEdges: 0,
    numTetrahedra,
    adjacency: [],
    edges: [],
    vertices,
    tetrahedra,
  };
}

// ============================================================================
// WEIGHTING FACTOR COMPUTATION
// ============================================================================

/**
 * Compute weighting factors for all edges in the tetrahedral graph
 * 
 * For each edge e:
 * 1. Compute geodesic distance d_e from edge midpoint to part surface M
 * 2. Compute geodesic distance δ_w from edge midpoint to shell surface ∂H
 * 3. Classify edge as internal or shell boundary based on δ_w threshold
 * 4. Compute effective distance:
 *    - Internal edges: d_eff = d_e (distance to part only)
 *    - Shell boundary edges: d_eff = d_e + (R - δ_w) (biased distance)
 * 5. Compute weighting factor: w(e) = 1/[(d_eff)² + ε]
 * 6. Compute edge cost: cost(e) = l(e) * w(e)
 * 
 * @param graph - The tetrahedral graph
 * @param partMesh - The part surface mesh (for distance computation)
 * @param shellMesh - The shell surface mesh (for distance computation)
 * @param options - Weighting options
 */
export function computeTetraEdgeWeights(
  graph: TetraGraph,
  partMesh: THREE.Mesh,
  shellMesh: THREE.Mesh,
  options: TetraWeightingOptions = {}
): TetraWeightingResult {
  const startTime = performance.now();
  
  const epsilon = options.epsilon ?? 0.25;
  
  console.log(`Computing tetrahedral edge weights...`);
  console.log(`  Epsilon (ε): ${epsilon}`);
  console.log(`  Number of edges: ${graph.numEdges}`);
  
  // Build BVH for efficient distance queries
  const partGeom = partMesh.geometry.clone();
  partGeom.applyMatrix4(partMesh.matrixWorld);
  const partBVH = new MeshBVH(partGeom);
  
  const shellGeom = shellMesh.geometry.clone();
  shellGeom.applyMatrix4(shellMesh.matrixWorld);
  const shellBVH = new MeshBVH(shellGeom);
  
  // Target object for distance queries (BVH API requires this structure)
  const hitInfoPart = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoShell = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  
  // ========================================================================
  // STEP 1: Compute distance from each edge midpoint to part and shell
  // ========================================================================
  
  let minDistToPart = Infinity;
  let maxDistToPart = -Infinity;
  let minDistToShell = Infinity;
  let maxDistToShell = -Infinity;
  let totalEdgeLength = 0;
  
  for (const edge of graph.edges) {
    // Distance to part surface (reset hitInfo before each query)
    hitInfoPart.distance = Infinity;
    partBVH.closestPointToPoint(edge.midpoint, hitInfoPart);
    edge.distToPart = hitInfoPart.distance;
    
    // Distance to shell surface
    hitInfoShell.distance = Infinity;
    shellBVH.closestPointToPoint(edge.midpoint, hitInfoShell);
    edge.distToShell = hitInfoShell.distance;
    
    // Track stats
    if (edge.distToPart < minDistToPart) minDistToPart = edge.distToPart;
    if (edge.distToPart > maxDistToPart) maxDistToPart = edge.distToPart;
    if (edge.distToShell < minDistToShell) minDistToShell = edge.distToShell;
    if (edge.distToShell > maxDistToShell) maxDistToShell = edge.distToShell;
    totalEdgeLength += edge.length;
  }
  
  const avgEdgeLength = totalEdgeLength / graph.numEdges;
  
  console.log(`  Distance to part: [${minDistToPart.toFixed(4)}, ${maxDistToPart.toFixed(4)}]`);
  console.log(`  Distance to shell: [${minDistToShell.toFixed(4)}, ${maxDistToShell.toFixed(4)}]`);
  console.log(`  Average edge length: ${avgEdgeLength.toFixed(4)}`);
  
  // ========================================================================
  // STEP 2: Determine shell boundary threshold and shell radius R
  // ========================================================================
  
  // Shell boundary threshold: edges closer than this to shell are "boundary edges"
  // Default: 0.5 * average edge length (edges very close to shell surface)
  const shellBoundaryThreshold = options.shellBoundaryThreshold ?? (0.5 * avgEdgeLength);
  
  // Shell radius R: max distance from any internal point to shell
  const R = options.shellRadius ?? maxDistToShell;
  
  console.log(`  Shell boundary threshold: ${shellBoundaryThreshold.toFixed(4)}`);
  console.log(`  Shell radius R: ${R.toFixed(4)}`);
  
  // ========================================================================
  // STEP 3: Classify edges and compute effective distance + weighting factor
  // 
  // Internal edges (distToShell >= threshold):
  //   effectiveDist = distToPart
  //   weight = 1 / (distToPart² + ε)
  //
  // Shell boundary edges (distToShell < threshold):
  //   effectiveDist = distToPart + (R - distToShell)  [biased distance]
  //   weight = 1 / (effectiveDist² + ε)
  //
  // Edge cost = length * weight
  // ========================================================================
  
  let minEffectiveDist = Infinity;
  let maxEffectiveDist = -Infinity;
  let minWeight = Infinity;
  let maxWeight = -Infinity;
  let minEdgeCost = Infinity;
  let maxEdgeCost = -Infinity;
  let numInternalEdges = 0;
  let numBoundaryEdges = 0;
  
  for (const edge of graph.edges) {
    // Classify edge based on distance to shell
    edge.isOnShellBoundary = edge.distToShell < shellBoundaryThreshold;
    
    if (edge.isOnShellBoundary) {
      // Shell boundary edge: use biased distance
      // d_biased = d_e + (R - δ_w)
      // The bias (R - δ_w) adds penalty for being far from shell center
      const biasPenalty = R - edge.distToShell;
      edge.effectiveDist = edge.distToPart + biasPenalty;
      numBoundaryEdges++;
    } else {
      // Internal edge: use only distance to part
      edge.effectiveDist = edge.distToPart;
      numInternalEdges++;
    }
    
    // Weighting factor: w(e) = 1 / [(d_eff)² + ε]
    // Higher weight (lower cost per unit length) when close to part surface
    edge.weight = 1.0 / (edge.effectiveDist * edge.effectiveDist + epsilon);
    
    // Edge cost for Dijkstra: length * weight
    // Short edges near the part surface have low cost
    edge.cost = edge.length * edge.weight;
    
    // Track stats
    if (edge.effectiveDist < minEffectiveDist) minEffectiveDist = edge.effectiveDist;
    if (edge.effectiveDist > maxEffectiveDist) maxEffectiveDist = edge.effectiveDist;
    if (edge.weight < minWeight) minWeight = edge.weight;
    if (edge.weight > maxWeight) maxWeight = edge.weight;
    if (edge.cost < minEdgeCost) minEdgeCost = edge.cost;
    if (edge.cost > maxEdgeCost) maxEdgeCost = edge.cost;
  }
  
  console.log(`  Edge classification:`);
  console.log(`    Internal edges: ${numInternalEdges} (${(100 * numInternalEdges / graph.numEdges).toFixed(1)}%)`);
  console.log(`    Boundary edges: ${numBoundaryEdges} (${(100 * numBoundaryEdges / graph.numEdges).toFixed(1)}%)`);
  console.log(`  Effective distance: [${minEffectiveDist.toFixed(4)}, ${maxEffectiveDist.toFixed(4)}]`);
  console.log(`  Weight factor: [${minWeight.toFixed(6)}, ${maxWeight.toFixed(6)}]`);
  console.log(`  Edge cost: [${minEdgeCost.toFixed(6)}, ${maxEdgeCost.toFixed(6)}]`);
  
  // Clean up
  partGeom.dispose();
  shellGeom.dispose();
  
  const elapsed = performance.now() - startTime;
  console.log(`  Edge weights computed in ${elapsed.toFixed(1)} ms`);
  
  return {
    graph,
    stats: {
      minDistToPart,
      maxDistToPart,
      minDistToShell,
      maxDistToShell,
      minEffectiveDist,
      maxEffectiveDist,
      minWeight,
      maxWeight,
      minEdgeCost,
      maxEdgeCost,
      shellRadius: R,
      shellBoundaryThreshold,
      numInternalEdges,
      numBoundaryEdges,
    },
    computeTimeMs: elapsed,
  };
}

// ============================================================================
// VISUALIZATION HELPERS
// ============================================================================

/**
 * Create a point cloud visualization of edge midpoints colored by weight
 */
export function createEdgeWeightVisualization(
  graph: TetraGraph,
  stats: TetraWeightingResult['stats'],
  colorMode: 'weight' | 'effectiveDist' | 'distToPart' | 'distToShell' | 'boundary' = 'weight'
): THREE.Points {
  const positions = new Float32Array(graph.numEdges * 3);
  const colors = new Float32Array(graph.numEdges * 3);
  
  // Color scale helpers
  const getColor = (value: number, min: number, max: number): THREE.Color => {
    const t = (value - min) / (max - min || 1);
    // Blue (low) -> Green -> Yellow -> Red (high)
    const color = new THREE.Color();
    color.setHSL((1 - t) * 0.7, 1, 0.5);  // Hue from blue to red
    return color;
  };
  
  for (let i = 0; i < graph.edges.length; i++) {
    const edge = graph.edges[i];
    
    positions[i * 3] = edge.midpoint.x;
    positions[i * 3 + 1] = edge.midpoint.y;
    positions[i * 3 + 2] = edge.midpoint.z;
    
    let color: THREE.Color;
    switch (colorMode) {
      case 'weight':
        color = getColor(edge.weight, stats.minWeight, stats.maxWeight);
        break;
      case 'effectiveDist':
        color = getColor(edge.effectiveDist, stats.minEffectiveDist, stats.maxEffectiveDist);
        break;
      case 'distToPart':
        color = getColor(edge.distToPart, stats.minDistToPart, stats.maxDistToPart);
        break;
      case 'distToShell':
        color = getColor(edge.distToShell, stats.minDistToShell, stats.maxDistToShell);
        break;
      case 'boundary':
        // Green for internal, Red for boundary
        color = edge.isOnShellBoundary ? new THREE.Color(1, 0, 0) : new THREE.Color(0, 1, 0);
        break;
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: 3,
    vertexColors: true,
    sizeAttenuation: true,
  });
  
  return new THREE.Points(geometry, material);
}

/**
 * Create a line visualization of edges colored by weight
 */
export function createEdgeLineVisualization(
  graph: TetraGraph,
  stats: TetraWeightingResult['stats'],
  colorMode: 'weight' | 'effectiveDist' | 'cost' | 'boundary' = 'weight'
): THREE.LineSegments {
  const positions = new Float32Array(graph.numEdges * 6);  // 2 vertices per edge
  const colors = new Float32Array(graph.numEdges * 6);
  
  const getColor = (value: number, min: number, max: number): THREE.Color => {
    const t = (value - min) / (max - min || 1);
    const color = new THREE.Color();
    color.setHSL((1 - t) * 0.7, 1, 0.5);
    return color;
  };
  
  for (let i = 0; i < graph.edges.length; i++) {
    const edge = graph.edges[i];
    const v0 = edge.v0;
    const v1 = edge.v1;
    
    // First vertex
    positions[i * 6] = graph.vertices[v0 * 3];
    positions[i * 6 + 1] = graph.vertices[v0 * 3 + 1];
    positions[i * 6 + 2] = graph.vertices[v0 * 3 + 2];
    
    // Second vertex
    positions[i * 6 + 3] = graph.vertices[v1 * 3];
    positions[i * 6 + 4] = graph.vertices[v1 * 3 + 1];
    positions[i * 6 + 5] = graph.vertices[v1 * 3 + 2];
    
    let color: THREE.Color;
    switch (colorMode) {
      case 'weight':
        color = getColor(edge.weight, stats.minWeight, stats.maxWeight);
        break;
      case 'effectiveDist':
        color = getColor(edge.effectiveDist, stats.minEffectiveDist, stats.maxEffectiveDist);
        break;
      case 'cost':
        color = getColor(edge.cost, stats.minEdgeCost, stats.maxEdgeCost);
        break;
      case 'boundary':
        // Green for internal, Red for boundary
        color = edge.isOnShellBoundary ? new THREE.Color(1, 0, 0) : new THREE.Color(0, 1, 0);
        break;
    }
    
    // Same color for both vertices of edge
    colors[i * 6] = colors[i * 6 + 3] = color.r;
    colors[i * 6 + 1] = colors[i * 6 + 4] = color.g;
    colors[i * 6 + 2] = colors[i * 6 + 5] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    linewidth: 1,
  });
  
  return new THREE.LineSegments(geometry, material);
}

// ============================================================================
// MIN-HEAP PRIORITY QUEUE
// ============================================================================

/**
 * Simple min-heap priority queue for Dijkstra
 */
class MinHeap {
  private heap: { index: number; cost: number }[] = [];
  private positions: Map<number, number> = new Map();

  get size(): number {
    return this.heap.length;
  }

  push(index: number, cost: number): void {
    const node = { index, cost };
    this.heap.push(node);
    this.positions.set(index, this.heap.length - 1);
    this.bubbleUp(this.heap.length - 1);
  }

  pop(): { index: number; cost: number } | undefined {
    if (this.heap.length === 0) return undefined;
    
    const min = this.heap[0];
    const last = this.heap.pop()!;
    this.positions.delete(min.index);
    
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.positions.set(last.index, 0);
      this.bubbleDown(0);
    }
    
    return min;
  }

  decreaseKey(index: number, newCost: number): void {
    const pos = this.positions.get(index);
    if (pos === undefined) {
      this.push(index, newCost);
      return;
    }
    
    if (newCost < this.heap[pos].cost) {
      this.heap[pos].cost = newCost;
      this.bubbleUp(pos);
    }
  }

  private bubbleUp(pos: number): void {
    while (pos > 0) {
      const parent = Math.floor((pos - 1) / 2);
      if (this.heap[parent].cost <= this.heap[pos].cost) break;
      
      this.swap(pos, parent);
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
      
      this.swap(pos, smallest);
      pos = smallest;
    }
  }

  private swap(i: number, j: number): void {
    const temp = this.heap[i];
    this.heap[i] = this.heap[j];
    this.heap[j] = temp;
    this.positions.set(this.heap[i].index, i);
    this.positions.set(this.heap[j].index, j);
  }
}

// ============================================================================
// PARTING SURFACE TYPES
// ============================================================================

/**
 * Result of tetrahedral parting surface computation
 */
export interface TetraPartingSurfaceResult {
  /** Vertex labels: -1 = unlabeled, 1 = H₁ (escapes via side 1), 2 = H₂ (escapes via side 2) */
  vertexLabels: Int8Array;
  /** Escape cost for each vertex */
  vertexCost: Float32Array;
  /** Edges that form the parting surface (where labels differ) */
  partingEdgeIndices: number[];
  /** Number of vertices labeled H₁ */
  h1Count: number;
  /** Number of vertices labeled H₂ */
  h2Count: number;
  /** Number of parting surface edges */
  partingEdgeCount: number;
  /** Number of unlabeled vertices */
  unlabeledCount: number;
  /** Computation time in ms */
  computeTimeMs: number;
  
  // Debug/visualization data
  /** Indices of seed vertices (near part surface) */
  seedVertices: number[];
  /** Boundary labels for each vertex: 0 = not boundary, 1 = H₁ boundary, 2 = H₂ boundary */
  boundaryLabels: Int8Array;
  /** Label assigned to each seed vertex after Dijkstra (before propagation) */
  seedLabels: Int8Array;
  /** Number of seeds */
  numSeeds: number;
  /** Number of H₁ boundary vertices */
  numH1Boundary: number;
  /** Number of H₂ boundary vertices */
  numH2Boundary: number;
}

/**
 * Options for parting surface computation
 */
export interface TetraPartingSurfaceOptions {
  /** 
   * Threshold for identifying seed vertices (near part surface)
   * Vertices with distToPart < seedThreshold are seeds
   * Default: auto-computed as 0.5 * average edge length
   */
  seedThreshold?: number;
}

// ============================================================================
// SEED AND BOUNDARY IDENTIFICATION
// ============================================================================

/**
 * Identify seed vertices (on/adjacent to part surface) and boundary vertices (on/adjacent to hull surface)
 * 
 * Approach:
 * 1. Find all SURFACE vertices of the tetrahedral mesh (vertices on boundary faces)
 *    - Boundary faces are faces that belong to only one tetrahedron
 * 2. For each surface vertex, check if it's closer to part mesh or hull mesh:
 *    - Close to PART mesh → seed vertex (inner boundary)
 *    - Close to HULL mesh → boundary vertex, labeled H₁ or H₂ based on mold half classification
 * 
 * This is analogous to how voxel grids identified inner vs outer surface voxels.
 * 
 * @param graph - The tetrahedral graph
 * @param partMesh - The original part mesh (for identifying inner boundary)
 * @param hullMesh - The outer hull mesh (for identifying outer boundary)
 * @param classification - Mold half classification result (provides H₁/H₂ labeling)
 */
export function identifySeedsAndBoundaries(
  graph: TetraGraph,
  cavityMesh: THREE.Mesh,
  classification: MoldHalfClassificationResult,
  _seedThreshold: number,  // Kept for API compatibility, not used in this approach
  _boundaryThreshold?: number,  // Kept for API compatibility, not used in this approach
  partMesh?: THREE.Mesh,
  hullMesh?: THREE.Mesh
): {
  seedVertices: number[];
  boundaryLabels: Int8Array;  // 0 = not boundary, 1 = H₁, 2 = H₂
  numSeeds: number;
  numH1Boundary: number;
  numH2Boundary: number;
} {
  const numVertices = graph.numVertices;
  const startTime = performance.now();
  
  console.log(`  Identifying seeds and boundaries...`);
  console.log(`    Total vertices: ${numVertices}`);
  
  // ========================================================================
  // STEP 1: Find boundary faces (faces belonging to only one tetrahedron)
  // ========================================================================
  
  // Build a map of face -> tetrahedra that share it
  // A face is defined by 3 vertex indices (sorted for consistent key)
  const faceToTetras = new Map<string, number[]>();
  const numTetrahedra = graph.tetrahedra.length / 4;
  
  const getFaceKey = (v0: number, v1: number, v2: number): string => {
    const sorted = [v0, v1, v2].sort((a, b) => a - b);
    return `${sorted[0]}_${sorted[1]}_${sorted[2]}`;
  };
  
  // Each tetrahedron has 4 faces
  for (let t = 0; t < numTetrahedra; t++) {
    const base = t * 4;
    const v0 = graph.tetrahedra[base];
    const v1 = graph.tetrahedra[base + 1];
    const v2 = graph.tetrahedra[base + 2];
    const v3 = graph.tetrahedra[base + 3];
    
    // 4 faces of tetrahedron: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    const faces = [
      [v0, v1, v2],
      [v0, v1, v3],
      [v0, v2, v3],
      [v1, v2, v3]
    ];
    
    for (const [a, b, c] of faces) {
      const key = getFaceKey(a, b, c);
      if (!faceToTetras.has(key)) {
        faceToTetras.set(key, []);
      }
      faceToTetras.get(key)!.push(t);
    }
  }
  
  // Find boundary faces (shared by only 1 tetrahedron) and collect boundary vertices
  const boundaryVertexSet = new Set<number>();
  let boundaryFaceCount = 0;
  
  for (const [faceKey, tetras] of faceToTetras) {
    if (tetras.length === 1) {
      // This is a boundary face
      boundaryFaceCount++;
      const [v0, v1, v2] = faceKey.split('_').map(Number);
      boundaryVertexSet.add(v0);
      boundaryVertexSet.add(v1);
      boundaryVertexSet.add(v2);
    }
  }
  
  console.log(`    Boundary faces: ${boundaryFaceCount}`);
  console.log(`    Boundary vertices: ${boundaryVertexSet.size}`);
  
  // ========================================================================
  // STEP 2: Build BVHs for part mesh and hull mesh
  // ========================================================================
  
  // Part mesh BVH (for identifying inner boundary / seeds)
  let partBVH: MeshBVH | null = null;
  if (partMesh) {
    let partGeom = partMesh.geometry.clone();
    partGeom.applyMatrix4(partMesh.matrixWorld);
    if (partGeom.index) {
      const nonIndexed = partGeom.toNonIndexed();
      partGeom.dispose();
      partGeom = nonIndexed;
    }
    partBVH = new MeshBVH(partGeom);
  }
  
  // Hull mesh BVH (for identifying outer boundary)
  // We use classification sets to determine H₁/H₂, but need hull BVH for distance
  let hullBVH: MeshBVH | null = null;
  if (hullMesh) {
    let hullGeom = hullMesh.geometry.clone();
    hullGeom.applyMatrix4(hullMesh.matrixWorld);
    if (hullGeom.index) {
      const nonIndexed = hullGeom.toNonIndexed();
      hullGeom.dispose();
      hullGeom = nonIndexed;
    }
    hullBVH = new MeshBVH(hullGeom);
  }
  
  // Cavity mesh BVH (for H₁/H₂ classification lookup via closest triangle)
  let cavityGeom = cavityMesh.geometry.clone();
  cavityGeom.applyMatrix4(cavityMesh.matrixWorld);
  if (cavityGeom.index) {
    const nonIndexed = cavityGeom.toNonIndexed();
    cavityGeom.dispose();
    cavityGeom = nonIndexed;
  }
  const cavityBVH = new MeshBVH(cavityGeom);
  
  const hitInfoPart = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoHull = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  const hitInfoCavity = { point: new THREE.Vector3(), distance: Infinity, faceIndex: 0 };
  
  // ========================================================================
  // STEP 3: Classify boundary vertices as seeds or H₁/H₂ boundaries
  // ========================================================================
  
  const seedVertices: number[] = [];
  const boundaryLabels = new Int8Array(numVertices).fill(0);
  let numH1Boundary = 0;
  let numH2Boundary = 0;
  let numBoundaryZone = 0;
  
  const vertexPos = new THREE.Vector3();
  
  // For boundary vertices, we classify based on which surface they are closer to:
  // - Closer to part mesh → SEED (inner boundary)
  // - Closer to hull mesh → BOUNDARY (outer, labeled H₁/H₂)
  // No distance threshold needed - we use relative comparison
  
  // Debug counters
  let nearPartCount = 0;
  let nearHullCount = 0;
  let ambiguousCount = 0;
  
  for (const vertIdx of boundaryVertexSet) {
    vertexPos.set(
      graph.vertices[vertIdx * 3],
      graph.vertices[vertIdx * 3 + 1],
      graph.vertices[vertIdx * 3 + 2]
    );
    
    // Compute distance to part mesh
    let distToPart = Infinity;
    if (partBVH) {
      hitInfoPart.distance = Infinity;
      partBVH.closestPointToPoint(vertexPos, hitInfoPart);
      distToPart = hitInfoPart.distance;
    }
    
    // Compute distance to hull mesh
    let distToHull = Infinity;
    if (hullBVH) {
      hitInfoHull.distance = Infinity;
      hullBVH.closestPointToPoint(vertexPos, hitInfoHull);
      distToHull = hitInfoHull.distance;
    }
    
    // Classify based on which surface the vertex is closer to
    // No distance threshold - pure relative comparison
    if (distToPart < distToHull) {
      // Vertex is closer to part surface → SEED
      seedVertices.push(vertIdx);
      nearPartCount++;
    } else if (distToHull < distToPart) {
      // Vertex is closer to hull surface → BOUNDARY (H₁ or H₂)
      nearHullCount++;
      
      // Find closest triangle on cavity mesh to determine H₁/H₂
      hitInfoCavity.distance = Infinity;
      cavityBVH.closestPointToPoint(vertexPos, hitInfoCavity);
      const closestTriIdx = hitInfoCavity.faceIndex;
      
      if (classification.h1Triangles.has(closestTriIdx)) {
        boundaryLabels[vertIdx] = 1;
        numH1Boundary++;
      } else if (classification.h2Triangles.has(closestTriIdx)) {
        boundaryLabels[vertIdx] = 2;
        numH2Boundary++;
      } else if (classification.boundaryZoneTriangles.has(closestTriIdx)) {
        // In boundary zone - don't label
        numBoundaryZone++;
      } else {
        // Fallback: check which mold half based on position relative to parting direction
        // For now, leave unlabeled
        numBoundaryZone++;
      }
    } else {
      // Ambiguous (equidistant) - rare case
      ambiguousCount++;
      
      // Default to seed (inner boundary)
      seedVertices.push(vertIdx);
    }
  }
  
  const elapsed = performance.now() - startTime;
  
  console.log(`  Vertex classification results:`);
  console.log(`    Seeds (near part): ${seedVertices.length}`);
  console.log(`    H₁ boundary: ${numH1Boundary}`);
  console.log(`    H₂ boundary: ${numH2Boundary}`);
  console.log(`    Boundary zone (unlabeled): ${numBoundaryZone}`);
  console.log(`    Near part surface: ${nearPartCount}`);
  console.log(`    Near hull surface: ${nearHullCount}`);
  console.log(`    Ambiguous: ${ambiguousCount}`);
  console.log(`    Time: ${elapsed.toFixed(1)} ms`);
  
  // Clean up
  cavityGeom.dispose();
  
  return {
    seedVertices,
    boundaryLabels,
    numSeeds: seedVertices.length,
    numH1Boundary,
    numH2Boundary,
  };
}

// ============================================================================
// DIJKSTRA ALGORITHM FOR PARTING SURFACE
// ============================================================================

/**
 * Compute parting surface using multi-source Dijkstra from seed vertices to boundaries
 * 
 * Algorithm:
 * 1. Initialize all seed vertices with cost 0
 * 2. Run Dijkstra, expanding along tetrahedral edges using computed weights
 * 3. When a path reaches a boundary vertex (H₁ or H₂), label the originating seed
 * 4. Propagate labels from seeds to all vertices
 * 5. Parting surface = edges connecting vertices with different labels
 * 
 * @param weightResult - The graph with computed edge weights
 * @param cavityMesh - The cavity mesh (for H₁/H₂ classification lookup)
 * @param classification - Mold half classification for boundary labeling
 * @param options - Parting surface options
 */
export function computeTetraPartingSurface(
  weightResult: TetraWeightingResult,
  cavityMesh: THREE.Mesh,
  classification: MoldHalfClassificationResult,
  options: TetraPartingSurfaceOptions = {}
): TetraPartingSurfaceResult {
  const startTime = performance.now();
  
  const graph = weightResult.graph;
  const numVertices = graph.numVertices;
  const numEdges = graph.numEdges;
  
  console.log(`Computing tetrahedral parting surface...`);
  console.log(`  Vertices: ${numVertices}, Edges: ${numEdges}`);
  
  // Determine seed threshold (only if edges exist)
  let seedThreshold = options.seedThreshold ?? 0;
  if (numEdges > 0) {
    const avgEdgeLength = graph.edges.reduce((sum, e) => sum + e.length, 0) / numEdges;
    seedThreshold = options.seedThreshold ?? (0.5 * avgEdgeLength);
  }
  console.log(`  Seed threshold: ${seedThreshold.toFixed(4)}`);
  
  // ========================================================================
  // STEP 1: Identify seeds and boundaries
  // ========================================================================
  
  // In validation mode, partMesh and hullMesh are passed in weightResult
  const partMesh = (weightResult as any).partMesh as THREE.Mesh | undefined;
  const hullMesh = (weightResult as any).hullMesh as THREE.Mesh | undefined;
  
  const { seedVertices, boundaryLabels, numSeeds, numH1Boundary, numH2Boundary } = 
    identifySeedsAndBoundaries(graph, cavityMesh, classification, seedThreshold, undefined, partMesh, hullMesh);
  
  console.log(`  Seeds (near part): ${numSeeds}`);
  console.log(`  H₁ boundary vertices: ${numH1Boundary}`);
  console.log(`  H₂ boundary vertices: ${numH2Boundary}`);
  
  // ========================================================================
  // VALIDATION MODE: Return early with just seed and boundary info
  // This allows validating the seed/boundary detection before running Dijkstra
  // ========================================================================
  const VALIDATION_MODE = true; // Set to false to run full algorithm
  
  if (VALIDATION_MODE) {
    console.log(`  [VALIDATION MODE] Returning early with seed and boundary info only`);
    const elapsed = performance.now() - startTime;
    
    return {
      vertexLabels: new Int8Array(numVertices).fill(-1),
      vertexCost: new Float32Array(numVertices).fill(Infinity),
      partingEdgeIndices: [],
      h1Count: numH1Boundary,
      h2Count: numH2Boundary,
      partingEdgeCount: 0,
      unlabeledCount: numVertices - numSeeds - numH1Boundary - numH2Boundary,
      computeTimeMs: elapsed,
      seedVertices,
      boundaryLabels,
      seedLabels: new Int8Array(numSeeds).fill(-1),
      numSeeds,
      numH1Boundary,
      numH2Boundary,
    };
  }
  
  if (numSeeds === 0) {
    console.error('ERROR: No seed vertices found! Cannot compute parting surface.');
    return {
      vertexLabels: new Int8Array(numVertices).fill(-1),
      vertexCost: new Float32Array(numVertices).fill(Infinity),
      partingEdgeIndices: [],
      h1Count: 0,
      h2Count: 0,
      partingEdgeCount: 0,
      unlabeledCount: numVertices,
      computeTimeMs: performance.now() - startTime,
      seedVertices: [],
      boundaryLabels: new Int8Array(numVertices).fill(0),
      seedLabels: new Int8Array(0),
      numSeeds: 0,
      numH1Boundary: 0,
      numH2Boundary: 0,
    };
  }
  
  // ========================================================================
  // STEP 2: Multi-source Dijkstra from all seeds
  // ========================================================================
  
  console.log(`  Running multi-source Dijkstra...`);
  
  // Arrays for Dijkstra
  const vertexCost = new Float32Array(numVertices).fill(Infinity);
  const originSeed = new Int32Array(numVertices).fill(-1);   // Which seed this vertex's path came from
  const seedLabel = new Int8Array(numVertices).fill(-1);     // Label assigned to each seed
  const seedLabeled = new Uint8Array(numVertices).fill(0);   // Whether seed has been labeled
  
  // Initialize priority queue with all seeds at cost 0
  const pq = new MinHeap();
  for (const seedIdx of seedVertices) {
    vertexCost[seedIdx] = 0;
    originSeed[seedIdx] = seedIdx;  // Seeds originate from themselves
    pq.push(seedIdx, 0);
  }
  
  let labeledSeedCount = 0;
  let visitedCount = 0;
  
  // Run Dijkstra until all seeds are labeled or queue is exhausted
  while (pq.size > 0 && labeledSeedCount < numSeeds) {
    const current = pq.pop()!;
    const v = current.index;
    const currentCost = current.cost;
    
    // Skip if we've already processed with better cost
    if (currentCost > vertexCost[v]) continue;
    
    visitedCount++;
    
    // Get the seed this path originated from
    const mySeed = originSeed[v];
    
    // If this seed is already labeled, skip
    if (mySeed >= 0 && seedLabeled[mySeed]) continue;
    
    // Check if we've reached a boundary
    if (boundaryLabels[v] !== 0 && mySeed >= 0) {
      // Label the originating seed with this boundary's label
      seedLabel[mySeed] = boundaryLabels[v];
      seedLabeled[mySeed] = 1;
      labeledSeedCount++;
      continue;  // Don't expand from boundary vertices
    }
    
    // Expand to neighbors along edges
    for (const { neighbor, edgeIndex } of graph.adjacency[v]) {
      const edge = graph.edges[edgeIndex];
      const candidateCost = vertexCost[v] + edge.cost;
      
      if (candidateCost < vertexCost[neighbor]) {
        vertexCost[neighbor] = candidateCost;
        originSeed[neighbor] = mySeed;  // Inherit the seed origin
        pq.decreaseKey(neighbor, candidateCost);
      }
    }
  }
  
  console.log(`  Dijkstra visited ${visitedCount} vertices`);
  console.log(`  Seeds labeled: ${labeledSeedCount}/${numSeeds}`);
  
  // ========================================================================
  // STEP 3: Propagate labels from seeds to all vertices
  // ========================================================================
  
  const vertexLabels = new Int8Array(numVertices).fill(-1);
  
  // First, assign labels to seeds
  for (const seedIdx of seedVertices) {
    vertexLabels[seedIdx] = seedLabel[seedIdx];
  }
  
  // Propagate labels to all vertices based on their origin seed
  for (let v = 0; v < numVertices; v++) {
    const seed = originSeed[v];
    if (seed >= 0 && seedLabel[seed] !== -1) {
      vertexLabels[v] = seedLabel[seed];
    }
  }
  
  // Handle unlabeled vertices using nearest labeled neighbor
  let unlabeledCount = 0;
  for (let v = 0; v < numVertices; v++) {
    if (vertexLabels[v] === -1) {
      unlabeledCount++;
      // Try to inherit label from nearest labeled neighbor
      let bestLabel = -1;
      let bestCost = Infinity;
      for (const { neighbor, edgeIndex } of graph.adjacency[v]) {
        if (vertexLabels[neighbor] !== -1) {
          const edge = graph.edges[edgeIndex];
          if (edge.cost < bestCost) {
            bestCost = edge.cost;
            bestLabel = vertexLabels[neighbor];
          }
        }
      }
      if (bestLabel !== -1) {
        vertexLabels[v] = bestLabel;
        unlabeledCount--;
      }
    }
  }
  
  // Count labels
  let h1Count = 0;
  let h2Count = 0;
  for (let v = 0; v < numVertices; v++) {
    if (vertexLabels[v] === 1) h1Count++;
    else if (vertexLabels[v] === 2) h2Count++;
  }
  
  console.log(`  Final labels: H₁=${h1Count}, H₂=${h2Count}, unlabeled=${unlabeledCount}`);
  
  // ========================================================================
  // STEP 4: Find parting surface edges (where labels differ)
  // ========================================================================
  
  const partingEdgeIndices: number[] = [];
  for (let i = 0; i < numEdges; i++) {
    const edge = graph.edges[i];
    const label0 = vertexLabels[edge.v0];
    const label1 = vertexLabels[edge.v1];
    
    // Edge is on parting surface if labels are different and both are valid
    if (label0 !== -1 && label1 !== -1 && label0 !== label1) {
      partingEdgeIndices.push(i);
    }
  }
  
  console.log(`  Parting surface edges: ${partingEdgeIndices.length}`);
  
  const elapsed = performance.now() - startTime;
  console.log(`  Parting surface computed in ${elapsed.toFixed(1)} ms`);
  
  return {
    vertexLabels,
    vertexCost,
    partingEdgeIndices,
    h1Count,
    h2Count,
    partingEdgeCount: partingEdgeIndices.length,
    unlabeledCount,
    computeTimeMs: elapsed,
    // Debug/visualization data
    seedVertices,
    boundaryLabels,
    seedLabels: seedLabel,
    numSeeds,
    numH1Boundary,
    numH2Boundary,
  };
}

// ============================================================================
// PARTING SURFACE VISUALIZATION
// ============================================================================

/**
 * Create a line visualization of the parting surface
 */
export function createPartingSurfaceVisualization(
  graph: TetraGraph,
  partingResult: TetraPartingSurfaceResult,
  color: THREE.ColorRepresentation = 0xff0000
): THREE.LineSegments {
  const numPartingEdges = partingResult.partingEdgeIndices.length;
  const positions = new Float32Array(numPartingEdges * 6);
  
  for (let i = 0; i < numPartingEdges; i++) {
    const edgeIndex = partingResult.partingEdgeIndices[i];
    const edge = graph.edges[edgeIndex];
    const v0 = edge.v0;
    const v1 = edge.v1;
    
    positions[i * 6] = graph.vertices[v0 * 3];
    positions[i * 6 + 1] = graph.vertices[v0 * 3 + 1];
    positions[i * 6 + 2] = graph.vertices[v0 * 3 + 2];
    
    positions[i * 6 + 3] = graph.vertices[v1 * 3];
    positions[i * 6 + 4] = graph.vertices[v1 * 3 + 1];
    positions[i * 6 + 5] = graph.vertices[v1 * 3 + 2];
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  
  const material = new THREE.LineBasicMaterial({
    color,
    linewidth: 2,
  });
  
  return new THREE.LineSegments(geometry, material);
}

/**
 * Create a visualization of vertex labels (H₁ vs H₂)
 */
export function createVertexLabelVisualization(
  graph: TetraGraph,
  partingResult: TetraPartingSurfaceResult
): THREE.Points {
  const positions = new Float32Array(graph.numVertices * 3);
  const colors = new Float32Array(graph.numVertices * 3);
  
  const h1Color = new THREE.Color(0x0088ff);  // Blue for H₁
  const h2Color = new THREE.Color(0xff8800);  // Orange for H₂
  const unlabeledColor = new THREE.Color(0x888888);  // Gray for unlabeled
  
  for (let i = 0; i < graph.numVertices; i++) {
    positions[i * 3] = graph.vertices[i * 3];
    positions[i * 3 + 1] = graph.vertices[i * 3 + 1];
    positions[i * 3 + 2] = graph.vertices[i * 3 + 2];
    
    const label = partingResult.vertexLabels[i];
    let color: THREE.Color;
    if (label === 1) {
      color = h1Color;
    } else if (label === 2) {
      color = h2Color;
    } else {
      color = unlabeledColor;
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: 2,
    vertexColors: true,
    sizeAttenuation: true,
  });
  
  return new THREE.Points(geometry, material);
}

// ============================================================================
// DEBUG VISUALIZATIONS FOR VERIFICATION
// ============================================================================

/**
 * Debug Mode 1: Visualize seed and boundary vertices WITHOUT labeling
 * - Seeds: Yellow (vertices near part surface)
 * - H₁ Boundary: Blue (boundary vertices on H₁ region)
 * - H₂ Boundary: Orange (boundary vertices on H₂ region)
 * - Other vertices: not shown
 */
export function createSeedBoundaryVisualization(
  graph: TetraGraph,
  partingResult: TetraPartingSurfaceResult,
  pointSize: number = 4
): THREE.Points {
  const seedSet = new Set(partingResult.seedVertices);
  
  // Count vertices to show
  let vertexCount = 0;
  for (let i = 0; i < graph.numVertices; i++) {
    if (seedSet.has(i) || partingResult.boundaryLabels[i] !== 0) {
      vertexCount++;
    }
  }
  
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  
  const seedColor = new THREE.Color(0xffff00);      // Yellow for seeds (inner boundary - part surface)
  const h1BoundaryColor = new THREE.Color(0x00ff00); // Green for H₁ boundary (matches mold half)
  const h2BoundaryColor = new THREE.Color(0xff8800); // Orange for H₂ boundary (matches mold half)
  
  let idx = 0;
  for (let i = 0; i < graph.numVertices; i++) {
    const isSeed = seedSet.has(i);
    const boundaryLabel = partingResult.boundaryLabels[i];
    
    if (!isSeed && boundaryLabel === 0) continue;
    
    positions[idx * 3] = graph.vertices[i * 3];
    positions[idx * 3 + 1] = graph.vertices[i * 3 + 1];
    positions[idx * 3 + 2] = graph.vertices[i * 3 + 2];
    
    let color: THREE.Color;
    if (isSeed) {
      color = seedColor;
    } else if (boundaryLabel === 1) {
      color = h1BoundaryColor;
    } else {
      color = h2BoundaryColor;
    }
    
    colors[idx * 3] = color.r;
    colors[idx * 3 + 1] = color.g;
    colors[idx * 3 + 2] = color.b;
    idx++;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: true,
    sizeAttenuation: true,
  });
  
  console.log(`SeedBoundary viz: ${partingResult.numSeeds} seeds, ${partingResult.numH1Boundary} H₁ boundary, ${partingResult.numH2Boundary} H₂ boundary`);
  
  return new THREE.Points(geometry, material);
}

/**
 * Debug Mode 2: Visualize only SEED vertices after they are labeled
 * - Seeds labeled H₁: Blue
 * - Seeds labeled H₂: Orange
 * - Seeds unlabeled: Gray
 */
export function createLabeledSeedsOnlyVisualization(
  graph: TetraGraph,
  partingResult: TetraPartingSurfaceResult,
  pointSize: number = 5
): THREE.Points {
  const numSeeds = partingResult.seedVertices.length;
  const positions = new Float32Array(numSeeds * 3);
  const colors = new Float32Array(numSeeds * 3);
  
  const h1Color = new THREE.Color(0x0088ff);  // Blue for H₁
  const h2Color = new THREE.Color(0xff8800);  // Orange for H₂
  const unlabeledColor = new THREE.Color(0x888888);  // Gray for unlabeled
  
  for (let i = 0; i < numSeeds; i++) {
    const vertIdx = partingResult.seedVertices[i];
    positions[i * 3] = graph.vertices[vertIdx * 3];
    positions[i * 3 + 1] = graph.vertices[vertIdx * 3 + 1];
    positions[i * 3 + 2] = graph.vertices[vertIdx * 3 + 2];
    
    // Use seedLabels which has the label assigned directly to the seed
    const label = partingResult.seedLabels[vertIdx];
    let color: THREE.Color;
    if (label === 1) {
      color = h1Color;
    } else if (label === 2) {
      color = h2Color;
    } else {
      color = unlabeledColor;
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: true,
    sizeAttenuation: true,
  });
  
  // Count labeled seeds
  let h1Seeds = 0, h2Seeds = 0, unlabeledSeeds = 0;
  for (const seedIdx of partingResult.seedVertices) {
    const label = partingResult.seedLabels[seedIdx];
    if (label === 1) h1Seeds++;
    else if (label === 2) h2Seeds++;
    else unlabeledSeeds++;
  }
  console.log(`Labeled seeds: H₁=${h1Seeds}, H₂=${h2Seeds}, unlabeled=${unlabeledSeeds}`);
  
  return new THREE.Points(geometry, material);
}

/**
 * Debug Mode 3: Visualize labeled seed AND boundary vertices together
 * - Seeds: Use their assigned label color (blue/orange) or gray if unlabeled
 * - Boundary H₁: Blue (larger size)
 * - Boundary H₂: Orange (larger size)
 */
export function createLabeledSeedAndBoundaryVisualization(
  graph: TetraGraph,
  partingResult: TetraPartingSurfaceResult,
  seedPointSize: number = 4,
  boundaryPointSize: number = 6
): THREE.Group {
  const group = new THREE.Group();
  
  // Colors
  const h1Color = new THREE.Color(0x0088ff);  // Blue for H₁
  const h2Color = new THREE.Color(0xff8800);  // Orange for H₂
  const unlabeledColor = new THREE.Color(0x888888);  // Gray
  
  // Create seed points
  const numSeeds = partingResult.seedVertices.length;
  const seedPositions = new Float32Array(numSeeds * 3);
  const seedColors = new Float32Array(numSeeds * 3);
  
  for (let i = 0; i < numSeeds; i++) {
    const vertIdx = partingResult.seedVertices[i];
    seedPositions[i * 3] = graph.vertices[vertIdx * 3];
    seedPositions[i * 3 + 1] = graph.vertices[vertIdx * 3 + 1];
    seedPositions[i * 3 + 2] = graph.vertices[vertIdx * 3 + 2];
    
    const label = partingResult.seedLabels[vertIdx];
    let color: THREE.Color;
    if (label === 1) color = h1Color;
    else if (label === 2) color = h2Color;
    else color = unlabeledColor;
    
    seedColors[i * 3] = color.r;
    seedColors[i * 3 + 1] = color.g;
    seedColors[i * 3 + 2] = color.b;
  }
  
  const seedGeom = new THREE.BufferGeometry();
  seedGeom.setAttribute('position', new THREE.BufferAttribute(seedPositions, 3));
  seedGeom.setAttribute('color', new THREE.BufferAttribute(seedColors, 3));
  const seedMat = new THREE.PointsMaterial({ size: seedPointSize, vertexColors: true, sizeAttenuation: true });
  group.add(new THREE.Points(seedGeom, seedMat));
  
  // Create boundary points (larger)
  let boundaryCount = 0;
  for (let i = 0; i < graph.numVertices; i++) {
    if (partingResult.boundaryLabels[i] !== 0) boundaryCount++;
  }
  
  const boundaryPositions = new Float32Array(boundaryCount * 3);
  const boundaryColors = new Float32Array(boundaryCount * 3);
  
  let idx = 0;
  for (let i = 0; i < graph.numVertices; i++) {
    const boundaryLabel = partingResult.boundaryLabels[i];
    if (boundaryLabel === 0) continue;
    
    boundaryPositions[idx * 3] = graph.vertices[i * 3];
    boundaryPositions[idx * 3 + 1] = graph.vertices[i * 3 + 1];
    boundaryPositions[idx * 3 + 2] = graph.vertices[i * 3 + 2];
    
    const color = boundaryLabel === 1 ? h1Color : h2Color;
    boundaryColors[idx * 3] = color.r;
    boundaryColors[idx * 3 + 1] = color.g;
    boundaryColors[idx * 3 + 2] = color.b;
    idx++;
  }
  
  const boundaryGeom = new THREE.BufferGeometry();
  boundaryGeom.setAttribute('position', new THREE.BufferAttribute(boundaryPositions, 3));
  boundaryGeom.setAttribute('color', new THREE.BufferAttribute(boundaryColors, 3));
  const boundaryMat = new THREE.PointsMaterial({ size: boundaryPointSize, vertexColors: true, sizeAttenuation: true });
  group.add(new THREE.Points(boundaryGeom, boundaryMat));
  
  console.log(`LabeledSeedBoundary viz: ${numSeeds} seeds, ${boundaryCount} boundary vertices`);
  
  return group;
}

/**
 * Debug Mode 4: Visualize ALL vertices after labeling completion
 * - H₁ vertices: Blue
 * - H₂ vertices: Orange
 * - Unlabeled: Gray
 */
export function createAllVerticesLabeledVisualization(
  graph: TetraGraph,
  partingResult: TetraPartingSurfaceResult,
  pointSize: number = 2
): THREE.Points {
  const positions = new Float32Array(graph.numVertices * 3);
  const colors = new Float32Array(graph.numVertices * 3);
  
  const h1Color = new THREE.Color(0x0088ff);  // Blue for H₁
  const h2Color = new THREE.Color(0xff8800);  // Orange for H₂
  const unlabeledColor = new THREE.Color(0x888888);  // Gray for unlabeled
  
  for (let i = 0; i < graph.numVertices; i++) {
    positions[i * 3] = graph.vertices[i * 3];
    positions[i * 3 + 1] = graph.vertices[i * 3 + 1];
    positions[i * 3 + 2] = graph.vertices[i * 3 + 2];
    
    const label = partingResult.vertexLabels[i];
    let color: THREE.Color;
    if (label === 1) color = h1Color;
    else if (label === 2) color = h2Color;
    else color = unlabeledColor;
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  
  const material = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: true,
    sizeAttenuation: true,
  });
  
  console.log(`AllVerticesLabeled viz: H₁=${partingResult.h1Count}, H₂=${partingResult.h2Count}, unlabeled=${partingResult.unlabeledCount}`);
  
  return new THREE.Points(geometry, material);
}
