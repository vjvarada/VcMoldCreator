/**
 * Mesh Repair using Manifold library
 * 
 * Uses the Manifold library for robust mesh validation and repair.
 * Manifold is specifically designed for watertight mesh operations.
 */

import * as THREE from 'three';
import Module, { type Manifold, type Mesh as ManifoldMesh } from 'manifold-3d';

// ============================================================================
// TYPES
// ============================================================================

export interface MeshDiagnostics {
  vertexCount: number;
  faceCount: number;
  isManifold: boolean;
  genus: number;
  volume: number;
  surfaceArea: number;
  boundingBox: { min: THREE.Vector3; max: THREE.Vector3 };
  issues: string[];
}

export interface MeshRepairResult {
  geometry: THREE.BufferGeometry;
  diagnostics: MeshDiagnostics;
  wasRepaired: boolean;
  repairMethod: string;
}

// ============================================================================
// MODULE TYPE
// ============================================================================

type ManifoldModule = Awaited<ReturnType<typeof Module>>;

let wasm: ManifoldModule | null = null;

/**
 * Initialize the Manifold WASM module
 */
async function initManifold(): Promise<ManifoldModule> {
  if (wasm) return wasm;
  
  console.log('Initializing Manifold WASM module...');
  wasm = await Module();
  wasm.setup();
  console.log('Manifold module initialized');
  return wasm;
}

// ============================================================================
// CONVERSION UTILITIES
// ============================================================================

/**
 * Convert Three.js BufferGeometry to Manifold Mesh options
 */
function geometryToMeshOptions(geometry: THREE.BufferGeometry): {
  numProp: number;
  vertProperties: Float32Array;
  triVerts: Uint32Array;
} {
  const posAttr = geometry.getAttribute('position');
  const positions = posAttr.array as Float32Array;
  const vertCount = posAttr.count;
  
  // Manifold needs at least 3 properties per vertex (x, y, z)
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
    // Non-indexed: create sequential indices
    triVerts = new Uint32Array(vertCount);
    for (let i = 0; i < vertCount; i++) {
      triVerts[i] = i;
    }
  }
  
  return {
    numProp: 3,
    vertProperties,
    triVerts
  };
}

/**
 * Convert Manifold getMesh() result to Three.js BufferGeometry
 */
function manifoldMeshToGeometry(mesh: ManifoldMesh): THREE.BufferGeometry {
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
 * Get diagnostics from a Manifold object
 */
function getManifoldDiagnostics(manifold: Manifold): Omit<MeshDiagnostics, 'issues'> {
  const bbox = manifold.boundingBox();
  const volume = manifold.volume();
  const surfaceArea = manifold.surfaceArea();
  
  return {
    vertexCount: manifold.numVert(),
    faceCount: manifold.numTri(),
    isManifold: true,
    genus: manifold.genus(),
    volume,
    surfaceArea,
    boundingBox: {
      min: new THREE.Vector3(bbox.min[0], bbox.min[1], bbox.min[2]),
      max: new THREE.Vector3(bbox.max[0], bbox.max[1], bbox.max[2])
    }
  };
}

// ============================================================================
// MAIN REPAIR FUNCTION
// ============================================================================

/**
 * Repair a mesh using Manifold library
 * 
 * Manifold automatically handles:
 * - Merging duplicate vertices
 * - Removing degenerate triangles  
 * - Ensuring consistent winding
 * - Creating a valid 2-manifold mesh
 */
export async function repairMeshWithManifold(
  geometry: THREE.BufferGeometry
): Promise<MeshRepairResult> {
  console.log('Starting Manifold-based mesh repair...');
  
  const mod = await initManifold();
  
  const posAttr = geometry.getAttribute('position');
  const originalVertexCount = posAttr.count;
  const originalFaceCount = geometry.index 
    ? Math.floor(geometry.index.count / 3)
    : Math.floor(posAttr.count / 3);
  
  console.log(`Original mesh: ${originalVertexCount} vertices, ${originalFaceCount} faces`);
  
  try {
    // Convert to Manifold mesh format
    const meshOptions = geometryToMeshOptions(geometry);
    const inputMesh = new mod.Mesh(meshOptions);
    
    // Try to merge vertices to create a manifold
    // This is the key repair step for non-manifold meshes
    const merged = inputMesh.merge();
    if (merged) {
      console.log('Mesh vertices were merged to create manifold');
    }
    
    let manifold: Manifold;
    let wasRepaired = false;
    let repairMethod = 'none';
    
    try {
      // Try to create a Manifold directly
      manifold = mod.Manifold.ofMesh(inputMesh);
      
      if (manifold.isEmpty()) {
        throw new Error('Created manifold is empty');
      }
      
      repairMethod = merged ? 'vertex merge' : 'already valid';
      wasRepaired = merged;
      console.log('Successfully created manifold');
      
    } catch (directError) {
      console.log('Direct manifold creation failed, trying hull approach...', directError);
      
      // If direct creation fails, use convex hull as fallback
      // This loses concave details but guarantees a valid manifold
      try {
        // Extract vertices for hull
        const vertices: [number, number, number][] = [];
        for (let i = 0; i < meshOptions.vertProperties.length; i += 3) {
          vertices.push([
            meshOptions.vertProperties[i],
            meshOptions.vertProperties[i + 1],
            meshOptions.vertProperties[i + 2]
          ]);
        }
        
        manifold = mod.Manifold.hull(vertices);
        
        if (manifold.isEmpty()) {
          throw new Error('Hull creation returned empty manifold');
        }
        
        wasRepaired = true;
        repairMethod = 'convex hull (mesh was not repairable)';
        console.log('Used convex hull as fallback');
        
      } catch (hullError) {
        console.error('Hull creation also failed:', hullError);
        
        // Final fallback: bounding box
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox!;
        const size = new THREE.Vector3();
        const center = new THREE.Vector3();
        bbox.getSize(size);
        bbox.getCenter(center);
        
        manifold = mod.Manifold.cube([size.x, size.y, size.z], true)
          .translate([center.x, center.y, center.z]);
        
        wasRepaired = true;
        repairMethod = 'bounding box (severe mesh issues)';
        console.log('Used bounding box as final fallback');
      }
    }
    
    // Get manifold properties for diagnostics
    const diagBase = getManifoldDiagnostics(manifold);
    
    console.log(`Result: ${diagBase.vertexCount} vertices, ${diagBase.faceCount} faces, genus=${diagBase.genus}`);
    console.log(`Volume: ${diagBase.volume.toFixed(2)}, Surface area: ${diagBase.surfaceArea.toFixed(2)}`);
    
    // Convert back to geometry
    const resultMesh = manifold.getMesh();
    const resultGeometry = manifoldMeshToGeometry(resultMesh);
    
    // Build issues list
    const issues: string[] = [];
    if (wasRepaired) {
      issues.push(`Repaired using: ${repairMethod}`);
    }
    if (diagBase.genus > 0) {
      issues.push(`Genus ${diagBase.genus} (has ${diagBase.genus} hole(s)/handle(s))`);
    }
    
    const diagnostics: MeshDiagnostics = {
      ...diagBase,
      issues
    };
    
    // Clean up Manifold objects
    manifold.delete();
    
    return {
      geometry: resultGeometry,
      diagnostics,
      wasRepaired,
      repairMethod
    };
    
  } catch (error) {
    console.error('Manifold repair failed completely:', error);
    
    // Return original geometry with error diagnostics
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox || new THREE.Box3();
    
    return {
      geometry: geometry.clone(),
      diagnostics: {
        vertexCount: originalVertexCount,
        faceCount: originalFaceCount,
        isManifold: false,
        genus: -1,
        volume: 0,
        surfaceArea: 0,
        boundingBox: {
          min: bbox.min.clone(),
          max: bbox.max.clone()
        },
        issues: [`Repair failed: ${error}`]
      },
      wasRepaired: false,
      repairMethod: 'failed'
    };
  }
}

/**
 * Format diagnostics for display
 */
export function formatDiagnostics(diagnostics: MeshDiagnostics): string {
  const lines = [
    `Vertices: ${diagnostics.vertexCount}`,
    `Faces: ${diagnostics.faceCount}`,
    `Status: ${diagnostics.isManifold ? '✓ Valid Manifold' : '✗ Invalid'}`,
  ];
  
  if (diagnostics.genus >= 0) {
    const genusDesc = diagnostics.genus === 0 ? '(sphere-like)' : 
                      diagnostics.genus === 1 ? '(torus-like)' : '';
    lines.push(`Genus: ${diagnostics.genus} ${genusDesc}`);
  }
  
  if (diagnostics.volume > 0) {
    lines.push(`Volume: ${diagnostics.volume.toFixed(2)}`);
    lines.push(`Surface Area: ${diagnostics.surfaceArea.toFixed(2)}`);
  }
  
  if (diagnostics.issues.length > 0) {
    lines.push('Notes:');
    for (const issue of diagnostics.issues) {
      lines.push(`  • ${issue}`);
    }
  }
  
  return lines.join('\n');
}
