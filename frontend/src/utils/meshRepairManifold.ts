/**
 * Mesh Repair using Manifold library
 * 
 * Uses the Manifold library for robust mesh validation and repair.
 * Manifold is specifically designed for watertight mesh operations.
 */

import * as THREE from 'three';
import {
  initManifold,
  geometryToManifoldMesh,
  manifoldMeshToGeometry,
  getManifoldDiagnostics,
  extractVertexTuples,
  type Manifold,
} from './manifoldUtils';

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
  const mod = await initManifold();
  
  const posAttr = geometry.getAttribute('position');
  const originalVertexCount = posAttr.count;
  const originalFaceCount = geometry.index 
    ? Math.floor(geometry.index.count / 3)
    : Math.floor(posAttr.count / 3);
  
  try {
    // Convert to Manifold mesh format
    const inputMesh = geometryToManifoldMesh(mod, geometry);
    
    // Try to merge vertices to create a manifold
    const merged = inputMesh.merge();
    
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
      
    } catch {
      // If direct creation fails, use convex hull as fallback
      try {
        const vertices = extractVertexTuples(geometry);
        manifold = mod.Manifold.hull(vertices);
        
        if (manifold.isEmpty()) {
          throw new Error('Hull creation returned empty manifold');
        }
        
        wasRepaired = true;
        repairMethod = 'convex hull (mesh was not repairable)';
        
      } catch {
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
      }
    }
    
    // Get manifold properties for diagnostics
    const diagBase = getManifoldDiagnostics(manifold);
    
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
    console.error('Manifold repair failed:', error);
    
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
