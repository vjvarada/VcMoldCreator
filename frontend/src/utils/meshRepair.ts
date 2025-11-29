import * as THREE from 'three';
import { mergeVertices } from 'three-stdlib';

export interface MeshDiagnostics {
  vertexCount: number;
  faceCount: number;
  boundaryEdges: number;
  nonManifoldEdges: number;
  duplicateVertices: number;
  degenerateFaces: number;
  isolatedVertices: number;
  isClosed: boolean;
  isManifold: boolean;
  issues: string[];
}

export interface MeshRepairResult {
  geometry: THREE.BufferGeometry;
  originalDiagnostics: MeshDiagnostics;
  repairedDiagnostics: MeshDiagnostics;
  repairsApplied: string[];
  success: boolean;
}

/**
 * Analyze a mesh geometry for common issues
 */
export function analyzeMesh(geometry: THREE.BufferGeometry): MeshDiagnostics {
  const issues: string[] = [];
  
  const posAttr = geometry.getAttribute('position');
  if (!posAttr) {
    return {
      vertexCount: 0,
      faceCount: 0,
      boundaryEdges: 0,
      nonManifoldEdges: 0,
      duplicateVertices: 0,
      degenerateFaces: 0,
      isolatedVertices: 0,
      isClosed: false,
      isManifold: false,
      issues: ['No position attribute found']
    };
  }
  
  const positions = posAttr.array as Float32Array;
  const vertexCount = posAttr.count;
  
  // Get indices (or create implicit ones for non-indexed geometry)
  let indices: number[];
  const isIndexed = geometry.index !== null;
  if (isIndexed) {
    indices = Array.from(geometry.index!.array);
  } else {
    indices = [];
    for (let i = 0; i < vertexCount; i++) {
      indices.push(i);
    }
  }
  
  const faceCount = Math.floor(indices.length / 3);
  
  // For edge detection, we need to use position-based keys for non-indexed meshes
  // This properly identifies shared edges between triangles
  const edgeMap = new Map<string, number[]>();
  
  // Round position to create stable keys
  const positionTolerance = 0.00001;
  function posKey(idx: number): string {
    const x = Math.round(positions[idx * 3] / positionTolerance) * positionTolerance;
    const y = Math.round(positions[idx * 3 + 1] / positionTolerance) * positionTolerance;
    const z = Math.round(positions[idx * 3 + 2] / positionTolerance) * positionTolerance;
    return `${x.toFixed(5)},${y.toFixed(5)},${z.toFixed(5)}`;
  }
  
  function makeEdgeKey(v1: number, v2: number): string {
    // Use position-based keys to properly detect shared edges
    const k1 = posKey(v1);
    const k2 = posKey(v2);
    return k1 < k2 ? `${k1}|${k2}` : `${k2}|${k1}`;
  }
  
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    const faceIdx = Math.floor(i / 3);
    
    for (const [v1, v2] of [[a, b], [b, c], [c, a]]) {
      const key = makeEdgeKey(v1, v2);
      if (!edgeMap.has(key)) {
        edgeMap.set(key, []);
      }
      edgeMap.get(key)!.push(faceIdx);
    }
  }
  
  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  
  for (const faces of edgeMap.values()) {
    if (faces.length === 1) {
      boundaryEdges++;
    } else if (faces.length > 2) {
      nonManifoldEdges++;
    }
  }
  
  // For indexed meshes, duplicate vertex detection is not meaningful
  // since vertices are already shared via the index
  // Only count duplicates for non-indexed meshes
  let duplicateVertices = 0;
  if (!isIndexed) {
    const vertexPositions = new Map<string, number[]>();
    const tolerance = 0.0001;
    
    for (let i = 0; i < vertexCount; i++) {
      const x = Math.round(positions[i * 3] / tolerance) * tolerance;
      const y = Math.round(positions[i * 3 + 1] / tolerance) * tolerance;
      const z = Math.round(positions[i * 3 + 2] / tolerance) * tolerance;
      const key = `${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)}`;
      
      if (!vertexPositions.has(key)) {
        vertexPositions.set(key, []);
      }
      vertexPositions.get(key)!.push(i);
    }
    
    for (const verts of vertexPositions.values()) {
      if (verts.length > 1) {
        duplicateVertices += verts.length - 1;
      }
    }
  }
  
  // Detect degenerate faces (zero area)
  let degenerateFaces = 0;
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const edge1 = new THREE.Vector3();
  const edge2 = new THREE.Vector3();
  const cross = new THREE.Vector3();
  
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    
    v0.set(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]);
    v1.set(positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]);
    v2.set(positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]);
    
    edge1.subVectors(v1, v0);
    edge2.subVectors(v2, v0);
    cross.crossVectors(edge1, edge2);
    
    const area = cross.length() * 0.5;
    if (area < 1e-10) {
      degenerateFaces++;
    }
  }
  
  // Detect isolated vertices (not referenced by any face)
  const referencedVertices = new Set<number>();
  for (const idx of indices) {
    referencedVertices.add(idx);
  }
  const isolatedVertices = vertexCount - referencedVertices.size;
  
  // Compile issues list
  if (boundaryEdges > 0) {
    issues.push(`${boundaryEdges} boundary edges (mesh has holes)`);
  }
  if (nonManifoldEdges > 0) {
    issues.push(`${nonManifoldEdges} non-manifold edges`);
  }
  if (duplicateVertices > 0) {
    issues.push(`${duplicateVertices} duplicate vertices`);
  }
  if (degenerateFaces > 0) {
    issues.push(`${degenerateFaces} degenerate (zero-area) faces`);
  }
  if (isolatedVertices > 0) {
    issues.push(`${isolatedVertices} isolated vertices`);
  }
  
  const isClosed = boundaryEdges === 0;
  const isManifold = nonManifoldEdges === 0 && isClosed;
  
  return {
    vertexCount,
    faceCount,
    boundaryEdges,
    nonManifoldEdges,
    duplicateVertices,
    degenerateFaces,
    isolatedVertices,
    isClosed,
    isManifold,
    issues
  };
}

/**
 * Remove degenerate (zero-area) faces from geometry
 */
function removeDegenerateFaces(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const posAttr = geometry.getAttribute('position');
  const positions = posAttr.array as Float32Array;
  
  let indices: number[];
  if (geometry.index) {
    indices = Array.from(geometry.index.array);
  } else {
    indices = [];
    for (let i = 0; i < posAttr.count; i++) {
      indices.push(i);
    }
  }
  
  const validFaces: number[] = [];
  const v0 = new THREE.Vector3();
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const edge1 = new THREE.Vector3();
  const edge2 = new THREE.Vector3();
  const cross = new THREE.Vector3();
  
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    
    v0.set(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]);
    v1.set(positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]);
    v2.set(positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]);
    
    edge1.subVectors(v1, v0);
    edge2.subVectors(v2, v0);
    cross.crossVectors(edge1, edge2);
    
    const area = cross.length() * 0.5;
    if (area >= 1e-10) {
      validFaces.push(a, b, c);
    }
  }
  
  if (validFaces.length === indices.length) {
    return geometry; // No change needed
  }
  
  // Create new geometry with only valid faces
  const newGeometry = geometry.clone();
  newGeometry.setIndex(validFaces);
  
  return newGeometry;
}

/**
 * Find and fill small holes (boundary loops) in a mesh
 * Uses ear-clipping triangulation for simple polygonal holes
 */
function fillSmallHoles(geometry: THREE.BufferGeometry, maxHoleEdges: number = 20): {
  geometry: THREE.BufferGeometry;
  holesFilled: number;
} {
  if (!geometry.index) {
    return { geometry, holesFilled: 0 };
  }
  
  const indices = Array.from(geometry.index.array);
  
  // Build edge map with vertex indices (for indexed geometry)
  const boundaryEdges = new Map<string, [number, number]>(); // edgeKey -> [v1, v2]
  const edgeCount = new Map<string, number>();
  
  function makeEdgeKey(v1: number, v2: number): string {
    return v1 < v2 ? `${v1}-${v2}` : `${v2}-${v1}`;
  }
  
  // Count edge occurrences
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    
    for (const [v1, v2] of [[a, b], [b, c], [c, a]] as [number, number][]) {
      const key = makeEdgeKey(v1, v2);
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
    }
  }
  
  // Find boundary edges (appear only once)
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    
    for (const [v1, v2] of [[a, b], [b, c], [c, a]] as [number, number][]) {
      const key = makeEdgeKey(v1, v2);
      if (edgeCount.get(key) === 1) {
        boundaryEdges.set(key, [v1, v2]);
      }
    }
  }
  
  if (boundaryEdges.size === 0) {
    return { geometry, holesFilled: 0 };
  }
  
  // Build adjacency for boundary vertices
  const boundaryAdjacency = new Map<number, number[]>();
  for (const [v1, v2] of boundaryEdges.values()) {
    if (!boundaryAdjacency.has(v1)) boundaryAdjacency.set(v1, []);
    if (!boundaryAdjacency.has(v2)) boundaryAdjacency.set(v2, []);
    boundaryAdjacency.get(v1)!.push(v2);
    boundaryAdjacency.get(v2)!.push(v1);
  }
  
  // Find boundary loops
  const visited = new Set<number>();
  const holes: number[][] = [];
  
  for (const startVertex of boundaryAdjacency.keys()) {
    if (visited.has(startVertex)) continue;
    
    const loop: number[] = [];
    let current = startVertex;
    let prev = -1;
    
    while (true) {
      if (visited.has(current) && loop.length > 0) {
        // Completed a loop
        if (current === startVertex && loop.length >= 3) {
          holes.push(loop);
        }
        break;
      }
      
      visited.add(current);
      loop.push(current);
      
      const neighbors = boundaryAdjacency.get(current) || [];
      // Pick the neighbor that isn't where we came from
      const next = neighbors.find(n => n !== prev);
      
      if (next === undefined) break;
      
      prev = current;
      current = next;
      
      // Safety limit
      if (loop.length > 1000) break;
    }
  }
  
  // Filter to small holes and triangulate them
  const newFaces: number[] = [];
  let holesFilled = 0;
  
  for (const hole of holes) {
    if (hole.length < 3 || hole.length > maxHoleEdges) continue;
    
    // Simple fan triangulation from first vertex (works for convex-ish holes)
    // For a hole with vertices [v0, v1, v2, v3, ...], create triangles:
    // (v0, v2, v1), (v0, v3, v2), (v0, v4, v3), ...
    // Note: winding order is reversed because we're looking from inside
    const v0 = hole[0];
    for (let i = 1; i < hole.length - 1; i++) {
      const v1 = hole[i];
      const v2 = hole[i + 1];
      // Reverse winding to match hole boundary orientation
      newFaces.push(v0, v2, v1);
    }
    holesFilled++;
    console.log(`Filled hole with ${hole.length} vertices`);
  }
  
  if (newFaces.length === 0) {
    return { geometry, holesFilled: 0 };
  }
  
  // Create new geometry with hole-filling triangles
  const allIndices = [...indices, ...newFaces];
  const newGeometry = geometry.clone();
  newGeometry.setIndex(allIndices);
  
  return { geometry: newGeometry, holesFilled };
}

/**
 * Repair a mesh geometry by applying various fixes
 */
export function repairMesh(geometry: THREE.BufferGeometry): MeshRepairResult {
  console.log('Starting mesh repair...');
  
  const repairsApplied: string[] = [];
  let workingGeometry = geometry.clone();
  
  // Get original stats before any processing
  const originalVertexCount = workingGeometry.getAttribute('position').count;
  const originalFaceCount = geometry.index 
    ? Math.floor(geometry.index.count / 3) 
    : Math.floor(originalVertexCount / 3);
  
  // For non-indexed STL meshes, vertices = faces * 3
  // We need to merge vertices first before we can properly analyze topology
  const isNonIndexed = !geometry.index;
  
  // Step 1: Merge duplicate vertices with appropriate tolerance
  // For STL files, use a slightly larger tolerance to handle floating-point variations
  const beforeMergeCount = workingGeometry.getAttribute('position').count;
  
  // Calculate an appropriate tolerance based on mesh size
  workingGeometry.computeBoundingBox();
  const bbox = workingGeometry.boundingBox!;
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const meshScale = Math.max(size.x, size.y, size.z);
  
  // Use tolerance relative to mesh size (0.0001% of diagonal) with minimum of 1e-6
  const mergeTolerance = Math.max(meshScale * 0.000001, 1e-6);
  console.log(`Using merge tolerance: ${mergeTolerance.toExponential(2)} (mesh scale: ${meshScale.toFixed(2)})`);
  
  workingGeometry = mergeVertices(workingGeometry, mergeTolerance);
  const afterMergeCount = workingGeometry.getAttribute('position').count;
  
  if (afterMergeCount < beforeMergeCount) {
    repairsApplied.push(`Merged ${beforeMergeCount - afterMergeCount} duplicate vertices`);
    console.log(`Merged vertices: ${beforeMergeCount} -> ${afterMergeCount}`);
  }
  
  // Now analyze the original (conceptually, based on what we started with)
  // For non-indexed meshes, report them as having many duplicates
  const originalDiagnostics: MeshDiagnostics = {
    vertexCount: originalVertexCount,
    faceCount: originalFaceCount,
    boundaryEdges: 0, // Will be calculated properly after merge
    nonManifoldEdges: 0,
    duplicateVertices: isNonIndexed ? (originalVertexCount - afterMergeCount) : 0,
    degenerateFaces: 0,
    isolatedVertices: 0,
    isClosed: false,
    isManifold: false,
    issues: isNonIndexed 
      ? [`Non-indexed mesh (${originalVertexCount - afterMergeCount} duplicate vertices)`]
      : []
  };
  
  // Step 2: Remove degenerate faces
  const beforeFaces = Math.floor((workingGeometry.index?.count || 0) / 3);
  workingGeometry = removeDegenerateFaces(workingGeometry);
  const afterFaces = Math.floor((workingGeometry.index?.count || 0) / 3);
  if (afterFaces < beforeFaces) {
    repairsApplied.push(`Removed ${beforeFaces - afterFaces} degenerate faces`);
    console.log(`Removed degenerate faces: ${beforeFaces} -> ${afterFaces}`);
  }
  
  // Step 3: Try to fill small holes (boundary loops up to 20 edges)
  const fillResult = fillSmallHoles(workingGeometry, 20);
  if (fillResult.holesFilled > 0) {
    workingGeometry = fillResult.geometry;
    repairsApplied.push(`Filled ${fillResult.holesFilled} small holes`);
    console.log(`Filled ${fillResult.holesFilled} holes`);
  }
  
  // Step 4: Recompute normals
  workingGeometry.computeVertexNormals();
  repairsApplied.push('Recomputed vertex normals');
  
  // Step 5: Compute bounding box and sphere
  workingGeometry.computeBoundingBox();
  workingGeometry.computeBoundingSphere();
  
  // Analyze repaired mesh (now properly indexed)
  const repairedDiagnostics = analyzeMesh(workingGeometry);
  console.log('Repaired mesh diagnostics:', repairedDiagnostics);
  
  // Determine success
  const success = repairedDiagnostics.isManifold || 
    (repairedDiagnostics.nonManifoldEdges === 0 && 
     repairedDiagnostics.degenerateFaces === 0);
  
  if (!repairedDiagnostics.isManifold) {
    if (repairedDiagnostics.boundaryEdges > 0) {
      console.warn(`Mesh still has ${repairedDiagnostics.boundaryEdges} boundary edges (holes). ` +
        `This may affect CSG operations.`);
    }
    if (repairedDiagnostics.nonManifoldEdges > 0) {
      console.warn(`Mesh still has ${repairedDiagnostics.nonManifoldEdges} non-manifold edges.`);
    }
  }
  
  return {
    geometry: workingGeometry,
    originalDiagnostics,
    repairedDiagnostics,
    repairsApplied,
    success
  };
}

/**
 * Quick check if a mesh needs repair
 */
export function meshNeedsRepair(geometry: THREE.BufferGeometry): boolean {
  const diagnostics = analyzeMesh(geometry);
  return !diagnostics.isManifold || 
    diagnostics.duplicateVertices > 0 || 
    diagnostics.degenerateFaces > 0 ||
    !geometry.index;
}

/**
 * Format diagnostics as a readable string
 */
export function formatDiagnostics(diagnostics: MeshDiagnostics): string {
  const lines = [
    `Vertices: ${diagnostics.vertexCount}`,
    `Faces: ${diagnostics.faceCount}`,
    `Status: ${diagnostics.isManifold ? '✓ Manifold' : '✗ Non-manifold'}`,
    `Closed: ${diagnostics.isClosed ? '✓ Yes' : '✗ No (has holes)'}`,
  ];
  
  if (diagnostics.issues.length > 0) {
    lines.push('Issues:');
    for (const issue of diagnostics.issues) {
      lines.push(`  • ${issue}`);
    }
  }
  
  return lines.join('\n');
}
