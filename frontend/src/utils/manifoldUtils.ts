/**
 * Manifold Utilities
 * 
 * Shared utilities for working with the Manifold library for robust mesh operations.
 * Provides initialization, geometry conversion, and common operations.
 */

import * as THREE from 'three';
import Module, { type Manifold, type Mesh as ManifoldMesh } from 'manifold-3d';

// ============================================================================
// TYPES
// ============================================================================

export type ManifoldModule = Awaited<ReturnType<typeof Module>>;

export interface ManifoldDiagnostics {
  vertexCount: number;
  faceCount: number;
  isManifold: boolean;
  genus: number;
  volume: number;
  surfaceArea: number;
  boundingBox: { min: THREE.Vector3; max: THREE.Vector3 };
}

// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

let wasmModule: ManifoldModule | null = null;

/**
 * Initialize the Manifold WASM module (singleton pattern)
 */
export async function initManifold(): Promise<ManifoldModule> {
  if (wasmModule) return wasmModule;
  
  wasmModule = await Module();
  wasmModule.setup();
  return wasmModule;
}

// ============================================================================
// GEOMETRY CONVERSION
// ============================================================================

/**
 * Convert Three.js BufferGeometry to Manifold Mesh
 */
export function geometryToManifoldMesh(
  mod: ManifoldModule,
  geometry: THREE.BufferGeometry
): ManifoldMesh {
  const posAttr = geometry.getAttribute('position');
  const positions = posAttr.array as Float32Array;
  const vertCount = posAttr.count;
  
  // Create vertex properties (x, y, z for each vertex)
  const vertProperties = new Float32Array(vertCount * 3);
  for (let i = 0; i < vertCount; i++) {
    vertProperties[i * 3] = positions[i * 3];
    vertProperties[i * 3 + 1] = positions[i * 3 + 1];
    vertProperties[i * 3 + 2] = positions[i * 3 + 2];
  }
  
  // Get or create indices
  let triVerts: Uint32Array;
  if (geometry.index) {
    triVerts = new Uint32Array(geometry.index.array);
  } else {
    triVerts = new Uint32Array(vertCount);
    for (let i = 0; i < vertCount; i++) {
      triVerts[i] = i;
    }
  }
  
  return new mod.Mesh({
    numProp: 3,
    vertProperties,
    triVerts
  });
}

/**
 * Convert Manifold mesh result to Three.js BufferGeometry
 */
export function manifoldMeshToGeometry(mesh: ManifoldMesh): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  
  const numVerts = mesh.numVert;
  const positions = new Float32Array(numVerts * 3);
  
  for (let i = 0; i < numVerts; i++) {
    const pos = mesh.position(i);
    positions[i * 3] = pos[0];
    positions[i * 3 + 1] = pos[1];
    positions[i * 3 + 2] = pos[2];
  }
  
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(mesh.triVerts), 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  
  return geometry;
}

/**
 * Extract vertices from geometry as array of coordinate tuples
 */
export function extractVertexTuples(
  geometry: THREE.BufferGeometry
): [number, number, number][] {
  const posAttr = geometry.getAttribute('position');
  const vertices: [number, number, number][] = [];
  
  for (let i = 0; i < posAttr.count; i++) {
    vertices.push([posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)]);
  }
  
  return vertices;
}

// ============================================================================
// DIAGNOSTICS
// ============================================================================

/**
 * Get diagnostics from a Manifold object
 */
export function getManifoldDiagnostics(manifold: Manifold): ManifoldDiagnostics {
  const bbox = manifold.boundingBox();
  
  return {
    vertexCount: manifold.numVert(),
    faceCount: manifold.numTri(),
    isManifold: true, // If we have a Manifold object, it's valid
    genus: manifold.genus(),
    volume: manifold.volume(),
    surfaceArea: manifold.surfaceArea(),
    boundingBox: {
      min: new THREE.Vector3(bbox.min[0], bbox.min[1], bbox.min[2]),
      max: new THREE.Vector3(bbox.max[0], bbox.max[1], bbox.max[2])
    }
  };
}

/**
 * Try to create a Manifold from geometry, with fallback to convex hull
 */
export async function createManifoldFromGeometry(
  geometry: THREE.BufferGeometry,
  useHullFallback: boolean = true
): Promise<{ manifold: Manifold; usedHull: boolean }> {
  const mod = await initManifold();
  const mesh = geometryToManifoldMesh(mod, geometry);
  
  // Try to merge vertices first
  mesh.merge();
  
  try {
    const manifold = mod.Manifold.ofMesh(mesh);
    if (!manifold.isEmpty()) {
      return { manifold, usedHull: false };
    }
    throw new Error('Created manifold is empty');
  } catch (e) {
    if (!useHullFallback) {
      throw e;
    }
    
    const vertices = extractVertexTuples(geometry);
    const manifold = mod.Manifold.hull(vertices);
    
    if (manifold.isEmpty()) {
      throw new Error('Convex hull creation also failed');
    }
    
    return { manifold, usedHull: true };
  }
}

// Re-export Manifold types for convenience
export type { Manifold, ManifoldMesh };
