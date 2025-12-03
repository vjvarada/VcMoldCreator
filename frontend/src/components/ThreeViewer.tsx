/**
 * ThreeViewer - 3D STL viewer with mold analysis visualization
 * 
 * Features:
 * - STL file loading and display
 * - Orthographic camera with orbit controls
 * - Parting direction computation and visualization
 * - Visibility painting for mold analysis
 */

import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import * as THREE from 'three';
import { OrbitControls, STLLoader } from 'three-stdlib';
import {
  computeAndShowPartingDirectionsParallel,
  removePartingDirectionArrows,
  applyVisibilityPaint,
  removeVisibilityPaint,
  COLORS,
  type VisibilityPaintData
} from '../utils/partingDirection';
import {
  generateInflatedBoundingVolume,
  removeInflatedHull,
  performCsgSubtraction,
  removeCsgResult,
  type InflatedHullResult,
  type CsgSubtractionResult
} from '../utils/inflatedBoundingVolume';
import {
  repairMeshWithManifold,
  formatDiagnostics,
  type MeshRepairResult
} from '../utils/meshRepairManifold';
import {
  generateAdaptiveVoxelGrid,
  createMoldVolumePointCloud,
  createMoldVolumeVoxels,
  createGridBoundingBoxHelper,
  createRLineVisualization,
  createDimensionLabels,
  removeDimensionLabels,
  removeGridVisualization,
  recalculateBiasedDistances,
  type VolumetricGridResult,
  type AdaptiveVoxelGridResult,
  type DistanceFieldType,
  type BiasedDistanceWeights
} from '../utils/volumetricGrid';
import {
  classifyMoldHalves,
  applyMoldHalfPaint,
  removeMoldHalfPaint,
  type MoldHalfClassificationResult
} from '../utils/moldHalfClassification';
import {
  computeEscapeLabelingDijkstraAsync,
  createEscapeLabelingPointCloud,
  createDebugVisualization,
  type EscapeLabelingResult,
  type AdjacencyType,
} from '../utils/partingSurface';

// ============================================================================
// TYPES
// ============================================================================

/** Visualization mode for volumetric grid */
export type GridVisualizationMode = 'points' | 'voxels' | 'none';

interface ThreeViewerProps {
  stlUrl?: string;
  showPartingDirections?: boolean;
  showD1Paint?: boolean;
  showD2Paint?: boolean;
  showInflatedHull?: boolean;
  inflationOffset?: number;
  showCsgResult?: boolean;
  hideOriginalMesh?: boolean;
  hideHull?: boolean;
  /** Hide the mold cavity visualization */
  hideCavity?: boolean;
  /** Show volumetric grid visualization */
  showVolumetricGrid?: boolean;
  /** Hide the volumetric grid visualization (without recomputing) */
  hideVoxelGrid?: boolean;
  /** Show R line visualization (max distance from voxel to part) */
  showRLine?: boolean;
  /** Voxel density in voxels per unit (default: 5.0 = 5 voxels/mm assuming STL is in mm) */
  voxelsPerUnit?: number;
  /** Number of LOD levels for adaptive voxelization (default: 3) */
  adaptiveLodLevels?: number;
  /** Grid visualization mode: 'points', 'voxels', or 'none' */
  gridVisualizationMode?: GridVisualizationMode;
  /** Which distance field to use for voxel coloring: 'part' (distance to part) or 'biased' (biased distance) */
  distanceFieldType?: DistanceFieldType;
  /** Show mold half classification (H₁/H₂ coloring on cavity) */
  showMoldHalfClassification?: boolean;
  /** Boundary zone threshold as fraction of bounding box diagonal (default: 0.15 = 15%) */
  boundaryZoneThreshold?: number;
  /** Show parting surface escape labeling */
  showPartingSurface?: boolean;
  /** Adjacency type for parting surface computation (6 or 26) */
  partingSurfaceAdjacency?: AdjacencyType;
  /** Debug visualization mode for parting surface */
  partingSurfaceDebugMode?: 'none' | 'surface-detection' | 'boundary-labels' | 'seed-labels' | 'seed-labels-only';
  onMeshLoaded?: (mesh: THREE.Mesh) => void;
  onMeshRepaired?: (result: MeshRepairResult) => void;
  onVisibilityDataReady?: (data: VisibilityPaintData | null) => void;
  onInflatedHullReady?: (result: InflatedHullResult | null) => void;
  onCsgResultReady?: (result: CsgSubtractionResult | null) => void;
  onVolumetricGridReady?: (result: VolumetricGridResult | null) => void;
  onMoldHalfClassificationReady?: (result: MoldHalfClassificationResult | null) => void;
  onEscapeLabelingReady?: (result: EscapeLabelingResult | null) => void;
}

/** Handle exposed by ThreeViewer via useImperativeHandle */
export interface ThreeViewerHandle {
  recalculateBiasedDistances: (weights: BiasedDistanceWeights) => void;
}

// ============================================================================
// CONSTANTS
// ============================================================================

// Initial frustum size - will be adjusted based on loaded mesh
let currentFrustumSize = 200; // Default for typical mechanical parts (mm)
const BACKGROUND_COLOR = 0x1a1a1a;
const GRID_COLOR_CENTER = 0x444444;
const GRID_COLOR_LINES = 0x333333;

// ============================================================================
// COMPONENT
// ============================================================================

const ThreeViewer = forwardRef<ThreeViewerHandle, ThreeViewerProps>(({
  stlUrl,
  showPartingDirections = false,
  showD1Paint = false,
  showD2Paint = false,
  showInflatedHull = false,
  inflationOffset = 0.05,
  showCsgResult = false,
  hideOriginalMesh = false,
  hideHull = false,
  hideCavity = false,
  showVolumetricGrid = false,
  hideVoxelGrid = false,
  showRLine = true,
  voxelsPerUnit = 0.5,
  adaptiveLodLevels = 3,
  gridVisualizationMode = 'points',
  distanceFieldType = 'weight',
  showMoldHalfClassification = false,
  boundaryZoneThreshold = 0.15,
  showPartingSurface = false,
  partingSurfaceAdjacency = 6,
  partingSurfaceDebugMode = 'none',
  onMeshLoaded,
  onMeshRepaired,
  onVisibilityDataReady,
  onInflatedHullReady,
  onCsgResultReady,
  onVolumetricGridReady,
  onMoldHalfClassificationReady,
  onEscapeLabelingReady
}, ref) => {
  // Refs for Three.js objects
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const animationIdRef = useRef<number | null>(null);
  const partingArrowsRef = useRef<THREE.ArrowHelper[]>([]);
  const visibilityDataRef = useRef<VisibilityPaintData | null>(null);
  const inflatedHullRef = useRef<InflatedHullResult | null>(null);
  const csgResultRef = useRef<CsgSubtractionResult | null>(null);
  const moldHalfClassificationRef = useRef<MoldHalfClassificationResult | null>(null);
  const volumetricGridRef = useRef<VolumetricGridResult | null>(null);
  const gridVisualizationRef = useRef<THREE.Points | THREE.InstancedMesh | null>(null);
  const gridBoundingBoxRef = useRef<THREE.LineSegments | null>(null);
  const rLineVisualizationRef = useRef<THREE.Group | null>(null);
  const escapeLabelingRef = useRef<EscapeLabelingResult | null>(null);
  const escapeLabelingVisualizationRef = useRef<THREE.Object3D | null>(null);
  const dimensionLabelsRef = useRef<THREE.Group | null>(null);
  const gridHelperRef = useRef<THREE.GridHelper | null>(null);
  const lightsRef = useRef<THREE.DirectionalLight[]>([]);
  // Store original bounding box (in mm) - no scaling applied (1:1 scale)
  const originalBoundingBoxRef = useRef<THREE.Box3 | null>(null);

  // ============================================================================
  // IMPERATIVE HANDLE - Expose methods to parent component
  // ============================================================================

  useImperativeHandle(ref, () => ({
    recalculateBiasedDistances: (weights: BiasedDistanceWeights) => {
      const gridResult = volumetricGridRef.current;
      const scene = sceneRef.current;
      const renderer = rendererRef.current;
      const camera = cameraRef.current;
      
      if (!gridResult || !scene) {
        console.warn('Cannot recalculate: no volumetric grid or scene available');
        return;
      }
      
      console.log('Recalculating biased distances with weights:', weights);
      
      // Recalculate biased distances
      const updatedResult = recalculateBiasedDistances(gridResult, weights);
      volumetricGridRef.current = updatedResult;
      
      // Always update visualization if it exists (regardless of hideVoxelGrid)
      // This ensures the colors update when viewing the grid with different field types
      if (gridVisualizationRef.current) {
        // Remove old visualization
        scene.remove(gridVisualizationRef.current);
        if (gridVisualizationRef.current instanceof THREE.Points) {
          gridVisualizationRef.current.geometry.dispose();
          (gridVisualizationRef.current.material as THREE.PointsMaterial).dispose();
        } else if (gridVisualizationRef.current instanceof THREE.InstancedMesh) {
          gridVisualizationRef.current.geometry.dispose();
          (gridVisualizationRef.current.material as THREE.Material).dispose();
        }
        
        // Create new visualization with updated data
        if (gridVisualizationMode === 'points') {
          gridVisualizationRef.current = createMoldVolumePointCloud(updatedResult, 0x00ffff, 3, distanceFieldType);
        } else {
          gridVisualizationRef.current = createMoldVolumeVoxels(updatedResult, 0x00ffff, 0.2, distanceFieldType);
        }
        scene.add(gridVisualizationRef.current);
        // Respect current visibility setting
        gridVisualizationRef.current.visible = !hideVoxelGrid;
        
        // Force a render to show the updated visualization
        if (renderer && camera) {
          renderer.render(scene, camera);
        }
        
        console.log('Visualization updated with new weights, visible:', !hideVoxelGrid);
      } else {
        console.log('No visualization to update (gridVisualizationRef.current is null)');
      }
      
      // Notify parent of updated grid result
      onVolumetricGridReady?.(updatedResult);
    }
  }), [hideVoxelGrid, gridVisualizationMode, distanceFieldType, onVolumetricGridReady]);

  // ============================================================================
  // SCENE SETUP
  // ============================================================================

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BACKGROUND_COLOR);
    sceneRef.current = scene;

    // Add grid (will be updated when mesh is loaded)
    const gridHelper = new THREE.GridHelper(200, 20, GRID_COLOR_CENTER, GRID_COLOR_LINES);
    scene.add(gridHelper);
    gridHelperRef.current = gridHelper;

    // Create orthographic camera (will be adjusted when mesh is loaded)
    const aspect = container.clientWidth / container.clientHeight;
    const camera = new THREE.OrthographicCamera(
      -currentFrustumSize * aspect / 2,
      currentFrustumSize * aspect / 2,
      currentFrustumSize / 2,
      -currentFrustumSize / 2,
      0.1,
      10000  // Far plane extended for larger scenes
    );
    camera.position.set(150, 100, 150);
    cameraRef.current = camera;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Add lights (positions will be adjusted when mesh is loaded)
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    
    const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
    light1.position.set(200, 200, 200);
    scene.add(light1);
    
    const light2 = new THREE.DirectionalLight(0xffffff, 0.4);
    light2.position.set(-200, -200, -200);
    scene.add(light2);
    
    lightsRef.current = [light1, light2];

    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1;
    controls.maxDistance = 5000;  // Extended for larger models
    // Configure mouse buttons: LEFT=rotate, MIDDLE=pan, RIGHT=dolly (zoom)
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.DOLLY
    };
    controls.enablePan = true;
    controlsRef.current = controls;

    // Handle resize
    const handleResize = () => {
      if (!cameraRef.current || !rendererRef.current) return;
      
      const aspect = container.clientWidth / container.clientHeight;
      cameraRef.current.left = -currentFrustumSize * aspect / 2;
      cameraRef.current.right = currentFrustumSize * aspect / 2;
      cameraRef.current.top = currentFrustumSize / 2;
      cameraRef.current.bottom = -currentFrustumSize / 2;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controlsRef.current?.update();
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      controls.dispose();
    };
  }, []);

  // ============================================================================
  // STL LOADING
  // ============================================================================

  useEffect(() => {
    if (!stlUrl || !sceneRef.current || !cameraRef.current || !controlsRef.current) return;

    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;

    // Clean up previous mesh
    if (meshRef.current) {
      scene.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      const material = meshRef.current.material;
      if (Array.isArray(material)) {
        material.forEach(m => m.dispose());
      } else {
        material.dispose();
      }
      meshRef.current = null;
    }

    console.log('Loading STL:', stlUrl);

    fetch(stlUrl)
      .then(response => response.arrayBuffer())
      .then(async buffer => {
        let geometry = new STLLoader().parse(buffer);
        
        // Repair mesh using Manifold library
        console.log('Repairing mesh with Manifold...');
        const repairResult = await repairMeshWithManifold(geometry);
        geometry = repairResult.geometry;
        
        console.log('Mesh repair result:', repairResult.repairMethod);
        console.log('Mesh diagnostics:\n' + formatDiagnostics(repairResult.diagnostics));
        
        // Notify parent of repair result
        onMeshRepaired?.(repairResult);
        
        geometry.computeBoundingBox();
        
        if (!geometry.boundingBox) return;

        // Store original bounding box (in mm) before any transformations
        originalBoundingBoxRef.current = geometry.boundingBox.clone();

        // Center geometry with bottom at Z=0 (1:1 scale, no scaling applied)
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -geometry.boundingBox.min.z);
        geometry.computeBoundingBox();
        
        // Get size in mm (1:1 scale)
        const size = new THREE.Vector3();
        geometry.boundingBox!.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        // Create mesh at 1:1 scale (1 unit = 1 mm)
        const material = new THREE.MeshPhongMaterial({
          color: COLORS.MESH_DEFAULT,
          specular: 0x333333,
          shininess: 60
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        // No scaling - mesh.scale stays at (1, 1, 1)
        mesh.rotation.x = -Math.PI / 2; // Z-up to Y-up
        
        scene.add(mesh);
        meshRef.current = mesh;

        console.log('STL loaded:', geometry.attributes.position.count, 'vertices');
        console.log(`Size (mm): ${size.x.toFixed(1)} x ${size.y.toFixed(1)} x ${size.z.toFixed(1)}`);
        console.log('Scale: 1:1 (1 unit = 1 mm)');
        onMeshLoaded?.(mesh);

        // Update camera frustum based on mesh size
        const frustumPadding = 1.5; // Add 50% padding around the mesh
        currentFrustumSize = maxDim * frustumPadding;
        
        if (cameraRef.current && containerRef.current) {
          const aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
          cameraRef.current.left = -currentFrustumSize * aspect / 2;
          cameraRef.current.right = currentFrustumSize * aspect / 2;
          cameraRef.current.top = currentFrustumSize / 2;
          cameraRef.current.bottom = -currentFrustumSize / 2;
          cameraRef.current.updateProjectionMatrix();
        }

        // Update grid to match mesh size (grid lines every 10mm, with major lines)
        if (gridHelperRef.current) {
          scene.remove(gridHelperRef.current);
        }
        const gridSize = Math.ceil(maxDim * 1.5 / 10) * 10; // Round up to nearest 10mm
        const gridDivisions = gridSize / 10; // 10mm per division
        const newGrid = new THREE.GridHelper(gridSize, gridDivisions, GRID_COLOR_CENTER, GRID_COLOR_LINES);
        scene.add(newGrid);
        gridHelperRef.current = newGrid;

        // Update light positions based on mesh size
        const lightDistance = maxDim * 2;
        if (lightsRef.current.length >= 2) {
          lightsRef.current[0].position.set(lightDistance, lightDistance, lightDistance);
          lightsRef.current[1].position.set(-lightDistance, -lightDistance, -lightDistance);
        }

        // Remove existing dimension labels
        if (dimensionLabelsRef.current) {
          removeDimensionLabels(scene, dimensionLabelsRef.current);
          dimensionLabelsRef.current = null;
        }

        // Create dimension labels at 1:1 scale
        const dimensionLabels = createDimensionLabels(
          originalBoundingBoxRef.current,
          1.0  // No scaling
        );
        scene.add(dimensionLabels);
        dimensionLabelsRef.current = dimensionLabels;

        // Position camera based on mesh size
        const cameraDistance = maxDim * 1.2;
        const meshHeight = size.z; // Height in scene Y after rotation
        camera.position.set(cameraDistance, cameraDistance * 0.7, cameraDistance);
        controls.target.set(0, meshHeight / 2, 0);
        controls.update();
      })
      .catch(error => console.error('Error loading STL:', error));
  }, [stlUrl, onMeshLoaded, onMeshRepaired]);

  // ============================================================================
  // PARTING DIRECTIONS
  // ============================================================================

  useEffect(() => {
    if (!meshRef.current || !sceneRef.current) return;

    // Clear existing arrows
    if (partingArrowsRef.current.length > 0) {
      removePartingDirectionArrows(partingArrowsRef.current);
      partingArrowsRef.current = [];
    }
    
    visibilityDataRef.current = null;
    onVisibilityDataReady?.(null);

    if (showPartingDirections && meshRef.current) {
      computeAndShowPartingDirectionsParallel(meshRef.current, 128)
        .then((result: { arrows: THREE.ArrowHelper[]; visibilityData: VisibilityPaintData }) => {
          partingArrowsRef.current = result.arrows;
          visibilityDataRef.current = result.visibilityData;
          onVisibilityDataReady?.(result.visibilityData);
        })
        .catch((error: unknown) => console.error('Error computing parting directions:', error));
    }
  }, [showPartingDirections, stlUrl, onVisibilityDataReady]);

  // ============================================================================
  // INFLATED BOUNDING VOLUME
  // ============================================================================

  useEffect(() => {
    if (!meshRef.current || !sceneRef.current) return;

    const scene = sceneRef.current;

    // Remove existing inflated hull
    if (inflatedHullRef.current) {
      removeInflatedHull(scene, inflatedHullRef.current);
      inflatedHullRef.current = null;
      onInflatedHullReady?.(null);
    }
    
    // Remove existing CSG result when hull changes
    if (csgResultRef.current) {
      removeCsgResult(scene, csgResultRef.current);
      csgResultRef.current = null;
      onCsgResultReady?.(null);
    }

    if (showInflatedHull && meshRef.current) {
      try {
        const result = generateInflatedBoundingVolume(meshRef.current, inflationOffset);
        scene.add(result.mesh);
        inflatedHullRef.current = result;
        onInflatedHullReady?.(result);
      } catch (error) {
        console.error('Error generating inflated hull:', error);
      }
    }
  }, [showInflatedHull, inflationOffset, stlUrl, onInflatedHullReady, onCsgResultReady]);

  // ============================================================================
  // CSG SUBTRACTION
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;

    // Remove existing CSG result
    if (csgResultRef.current) {
      removeCsgResult(scene, csgResultRef.current);
      csgResultRef.current = null;
      onCsgResultReady?.(null);
    }

    // Perform CSG if requested and hull exists
    if (showCsgResult && inflatedHullRef.current && meshRef.current) {
      const performCsg = async () => {
        try {
          console.log('Performing CSG: Hull - Original Mesh');
          const result = await performCsgSubtraction(
            inflatedHullRef.current!.mesh,
            meshRef.current!
          );
          scene.add(result.mesh);
          csgResultRef.current = result;
          onCsgResultReady?.(result);
          console.log('CSG result added to scene');
        } catch (error) {
          console.error('Error performing CSG subtraction:', error);
        }
      };
      performCsg();
    }
  }, [showCsgResult, onCsgResultReady]);

  // ============================================================================
  // MOLD HALF CLASSIFICATION
  // ============================================================================

  useEffect(() => {
    // Clear existing classification when dependencies change
    moldHalfClassificationRef.current = null;
    onMoldHalfClassificationReady?.(null);
    
    // Only classify if we have CSG result (cavity), hull, and visibility data (parting directions)
    if (!showMoldHalfClassification || !csgResultRef.current || !inflatedHullRef.current || !visibilityDataRef.current) {
      // Remove paint if classification is disabled
      if (csgResultRef.current) {
        removeMoldHalfPaint(csgResultRef.current.mesh);
      }
      return;
    }
    
    const cavityMesh = csgResultRef.current.mesh;
    const hullMesh = inflatedHullRef.current.mesh;
    const { d1, d2 } = visibilityDataRef.current;
    
    try {
      // IMPORTANT: Convert cavity mesh to non-indexed FIRST (before classification)
      // This ensures triangle indices match between classification and painting
      const cavityGeom = cavityMesh.geometry as THREE.BufferGeometry;
      if (cavityGeom.index) {
        const nonIndexedGeom = cavityGeom.toNonIndexed();
        cavityGeom.dispose();
        cavityMesh.geometry = nonIndexedGeom;
      }
      
      // Get cavity geometry in world space (already non-indexed)
      const cavityGeometry = cavityMesh.geometry.clone();
      cavityGeometry.applyMatrix4(cavityMesh.matrixWorld);
      
      // Get hull geometry in world space (needed to identify outer boundary)
      const hullGeometry = hullMesh.geometry.clone();
      hullGeometry.applyMatrix4(hullMesh.matrixWorld);
      
      // Classify triangles into H₁ and H₂
      const classification = classifyMoldHalves(cavityGeometry, hullGeometry, d1, d2, boundaryZoneThreshold);
      moldHalfClassificationRef.current = classification;
      onMoldHalfClassificationReady?.(classification);
      
      // Apply visualization (paint the cavity mesh)
      applyMoldHalfPaint(cavityMesh, classification);
      
      // Clean up temp geometry
      cavityGeometry.dispose();
      hullGeometry.dispose();
      
    } catch (error) {
      console.error('Error classifying mold halves:', error);
    }
  // Note: boundaryZoneThreshold is intentionally NOT in dependency array
  // so that moving the slider doesn't trigger recalculation - only clicking "Calculate" does
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showMoldHalfClassification, showCsgResult, showPartingDirections, onMoldHalfClassificationReady]);

  // ============================================================================
  // VOLUMETRIC GRID COMPUTATION
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;

    // Clean up existing grid visualization
    if (gridVisualizationRef.current) {
      removeGridVisualization(scene, gridVisualizationRef.current);
      gridVisualizationRef.current = null;
    }
    if (gridBoundingBoxRef.current) {
      removeGridVisualization(scene, gridBoundingBoxRef.current);
      gridBoundingBoxRef.current = null;
    }
    // Clean up R line visualization
    if (rLineVisualizationRef.current) {
      scene.remove(rLineVisualizationRef.current);
      rLineVisualizationRef.current.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          obj.geometry?.dispose();
          if (Array.isArray(obj.material)) {
            obj.material.forEach(m => m.dispose());
          } else {
            obj.material?.dispose();
          }
        }
      });
      rLineVisualizationRef.current = null;
    }
    volumetricGridRef.current = null;
    onVolumetricGridReady?.(null);

    // Generate volumetric grid if requested and we have both hull and mesh
    if (showVolumetricGrid && inflatedHullRef.current && meshRef.current) {
      // Use async IIFE for potential future async operations
      (async () => {
        try {
          console.log('Generating volumetric grid...');
          console.log(`  Method: Adaptive LOD voxelization`);
          console.log(`  Base voxel density: ${voxelsPerUnit} voxels/mm`);
          console.log(`  LOD levels: ${adaptiveLodLevels}`);
          
          // Get geometries in world space
          const shellGeometry = inflatedHullRef.current!.mesh.geometry.clone();
          shellGeometry.applyMatrix4(inflatedHullRef.current!.mesh.matrixWorld);
          
          const partGeometry = meshRef.current!.geometry.clone();
          partGeometry.applyMatrix4(meshRef.current!.matrixWorld);
          
          // Use adaptive LOD voxelization with linear scaling for better Dijkstra traversal
          const adaptiveOptions = {
            baseVoxelsPerUnit: voxelsPerUnit,
            lodLevels: adaptiveLodLevels,
            marginPercent: 0.02,
            // Linear scaling is better for Dijkstra walk algorithms:
            // - More uniform voxel size transitions
            // - Fewer discretization artifacts at LOD boundaries
            // - Better path continuity in escape labeling
            distanceScalingMode: 'linear' as const,
            sizeFactor: 1.5, // Gentler than 2.0 for smoother transitions
          };
          const gridResult = await generateAdaptiveVoxelGrid(shellGeometry, partGeometry, adaptiveOptions);
          
          // Log adaptive-specific info
          const adaptiveResult = gridResult as AdaptiveVoxelGridResult;
          if (adaptiveResult.lodStats) {
            console.log(`  LOD Statistics:`);
            for (let l = 0; l < adaptiveResult.lodStats.voxelCountPerLevel.length; l++) {
              console.log(`    LOD ${l}: ${adaptiveResult.lodStats.voxelCountPerLevel[l]} voxels, size=${adaptiveResult.lodStats.voxelSizePerLevel[l].toFixed(4)}mm`);
            }
          }
          
          console.log(`Volumetric grid generated:`);
          console.log(`  Resolution: ${gridResult.resolution.x}×${gridResult.resolution.y}×${gridResult.resolution.z}`);
          console.log(`  Mold volume cells: ${gridResult.moldVolumeCellCount} / ${gridResult.totalCellCount}`);
          console.log(`  Fill ratio: ${(gridResult.stats.fillRatio * 100).toFixed(1)}%`);
          console.log(`  Approx mold volume: ${gridResult.stats.moldVolume.toFixed(4)} mm³`);
          console.log(`  Base cell size: ${gridResult.cellSize.x.toFixed(4)} × ${gridResult.cellSize.y.toFixed(4)} × ${gridResult.cellSize.z.toFixed(4)} mm`);
          console.log(`  Compute time: ${gridResult.stats.computeTimeMs.toFixed(1)} ms`);
          if (gridResult.seedVoxelMask) {
            let seedCount = 0;
            for (let i = 0; i < gridResult.seedVoxelMask.length; i++) {
              if (gridResult.seedVoxelMask[i]) seedCount++;
            }
            console.log(`  Seed voxels (vertices + edges): ${seedCount}`);
          }
          
          // Log memory size of voxel grid data
          const voxelCount = gridResult.moldVolumeCellCount;
          
          // New flat arrays (memory efficient)
          const voxelCentersMemory = gridResult.voxelCenters?.byteLength ?? 0;
          const voxelIndicesMemory = gridResult.voxelIndices?.byteLength ?? 0;
          
          const voxelDistMemory = gridResult.voxelDist?.byteLength ?? 0;
          const voxelDistToShellMemory = gridResult.voxelDistToShell?.byteLength ?? 0;
          const biasedDistMemory = gridResult.biasedDist?.byteLength ?? 0;
          const weightingFactorMemory = gridResult.weightingFactor?.byteLength ?? 0;
          const boundaryMaskMemory = gridResult.boundaryAdjacentMask?.byteLength ?? 0;
          
          // Total memory for flat arrays (legacy moldVolumeCells no longer populated)
          const flatArraysTotal = voxelCentersMemory + voxelIndicesMemory + voxelDistMemory + voxelDistToShellMemory + biasedDistMemory + weightingFactorMemory + boundaryMaskMemory;
          
          console.log(`  📊 Memory footprint:`);
          console.log(`    voxelCenters (flat Float32): ${(voxelCentersMemory / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    voxelIndices (flat Uint32): ${(voxelIndicesMemory / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    voxelDist: ${(voxelDistMemory / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    voxelDistToShell: ${(voxelDistToShellMemory / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    biasedDist: ${(biasedDistMemory / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    weightingFactor: ${(weightingFactorMemory / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    boundaryAdjacentMask: ${(boundaryMaskMemory / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    --- `);
          console.log(`    ✅ Total (flat arrays only): ${(flatArraysTotal / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    💾 Memory saved vs legacy: ~${(voxelCount * 120 / 1024 / 1024).toFixed(2)} MB (moldVolumeCells no longer populated)`);
          
          volumetricGridRef.current = gridResult;
          onVolumetricGridReady?.(gridResult);
          
          // Create initial visualization based on mode
          // For adaptive grids, pass the voxelSizes array for variable-size rendering
          const voxelSizes = adaptiveResult.voxelSizes || null;
          
          if (gridVisualizationMode !== 'none' && gridResult.moldVolumeCellCount > 0) {
            if (gridVisualizationMode === 'points') {
              gridVisualizationRef.current = createMoldVolumePointCloud(gridResult, 0x00ffff, 3, distanceFieldType, voxelSizes);
            } else if (gridVisualizationMode === 'voxels') {
              gridVisualizationRef.current = createMoldVolumeVoxels(gridResult, 0x00ffff, 0.2, distanceFieldType, voxelSizes);
            }
            
            if (gridVisualizationRef.current) {
              scene.add(gridVisualizationRef.current);
            }
            
            // Add bounding box helper
            gridBoundingBoxRef.current = createGridBoundingBoxHelper(gridResult, 0xffff00);
            scene.add(gridBoundingBoxRef.current);
            
            // Add R line visualization (max distance from voxel to part) - visibility controlled by showRLine
            rLineVisualizationRef.current = createRLineVisualization(gridResult, 0xff00ff);
            if (rLineVisualizationRef.current) {
              rLineVisualizationRef.current.visible = showRLine;
              scene.add(rLineVisualizationRef.current);
            }
          }
          
          // Clean up temporary geometries
          shellGeometry.dispose();
          partGeometry.dispose();
          
        } catch (error) {
          console.error('Error generating volumetric grid:', error);
        }
      })();
    }
  }, [showVolumetricGrid, voxelsPerUnit, adaptiveLodLevels, showRLine, onVolumetricGridReady]);

  // ============================================================================
  // VOLUMETRIC GRID VISUALIZATION MODE UPDATE
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current || !volumetricGridRef.current) return;
    
    const scene = sceneRef.current;
    const gridResult = volumetricGridRef.current;

    // Remove existing visualization (but keep the computed data)
    if (gridVisualizationRef.current) {
      removeGridVisualization(scene, gridVisualizationRef.current);
      gridVisualizationRef.current = null;
    }
    if (gridBoundingBoxRef.current) {
      removeGridVisualization(scene, gridBoundingBoxRef.current);
      gridBoundingBoxRef.current = null;
    }

    // Recreate visualization with new mode
    if (gridVisualizationMode !== 'none' && gridResult.moldVolumeCellCount > 0) {
      if (gridVisualizationMode === 'points') {
        gridVisualizationRef.current = createMoldVolumePointCloud(gridResult, 0x00ffff, 3, distanceFieldType);
      } else if (gridVisualizationMode === 'voxels') {
        gridVisualizationRef.current = createMoldVolumeVoxels(gridResult, 0x00ffff, 0.2, distanceFieldType);
      }
      
      if (gridVisualizationRef.current) {
        scene.add(gridVisualizationRef.current);
      }
      
      // Add bounding box helper
      gridBoundingBoxRef.current = createGridBoundingBoxHelper(gridResult, 0xffff00);
      scene.add(gridBoundingBoxRef.current);
    }
  }, [gridVisualizationMode, distanceFieldType]);

  // ============================================================================
  // R LINE VISIBILITY TOGGLE
  // ============================================================================

  useEffect(() => {
    if (rLineVisualizationRef.current) {
      rLineVisualizationRef.current.visible = showRLine;
    }
  }, [showRLine]);

  // ============================================================================
  // PARTING SURFACE COMPUTATION
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;

    // Clean up existing escape labeling visualization
    if (escapeLabelingVisualizationRef.current) {
      scene.remove(escapeLabelingVisualizationRef.current);
      // Handle both Group and Points cleanup
      escapeLabelingVisualizationRef.current.traverse((child) => {
        if (child instanceof THREE.Points) {
          child.geometry.dispose();
          (child.material as THREE.Material).dispose();
        }
      });
      escapeLabelingVisualizationRef.current = null;
    }

    // Only recompute labeling when showPartingSurface changes or adjacency changes
    // Don't recompute just because debug mode or interface mode changed
    if (showPartingSurface && volumetricGridRef.current && visibilityDataRef.current && meshRef.current) {
      // Check if we need to recompute or can use cached result
      const needsRecompute = !escapeLabelingRef.current;
      
      if (needsRecompute) {
        // Use async IIFE to handle the async computeEscapeLabelingDijkstraAsync call
        (async () => {
          try {
            console.log('Computing escape labeling (parting surface)...');
            
            // Ensure we have required dependencies for Dijkstra-based labeling
            if (!moldHalfClassificationRef.current || !csgResultRef.current) {
              console.warn('Skipping parting surface: mold half classification or CSG result not available');
              return;
            }
            
            console.log('Using Dijkstra-based escape labeling with mold half classification');
            
            // Get CSG cavity geometry in world space (this is the outer shell ∂H)
            const shellGeometry = csgResultRef.current.mesh.geometry.clone();
            shellGeometry.applyMatrix4(csgResultRef.current.mesh.matrixWorld);
            
            // Get part geometry in world space for volume intersection seed detection
            const partGeometry = meshRef.current!.geometry.clone();
            partGeometry.applyMatrix4(meshRef.current!.matrixWorld);
            
            // Compute escape labeling using multi-source Dijkstra
            // Seeds are voxels with ≥10% volume intersection with part mesh
            // Uses parallel Web Workers for faster computation on multi-core systems
            const labeling = await computeEscapeLabelingDijkstraAsync(
              volumetricGridRef.current!,
              shellGeometry,
              moldHalfClassificationRef.current,
              { 
                adjacency: partingSurfaceAdjacency, 
                seedRadius: 1.5,
                seedVolumeThreshold: 0.10,  // 10% volume intersection threshold
                seedVolumeSamples: 4,       // 4³ = 64 samples per voxel
                parallel: true,             // Enable parallel processing with Web Workers
              },
              partGeometry  // Pass part geometry for volume intersection
            );
            
            shellGeometry.dispose();
            partGeometry.dispose();
            
            escapeLabelingRef.current = labeling;
            onEscapeLabelingReady?.(labeling);
            
            // Log memory size of parting surface data
            const labelsMemory = labeling.labels?.byteLength ?? 0;
            const escapeCostMemory = labeling.escapeCost?.byteLength ?? 0;
            const boundaryLabelsMemory = labeling.boundaryLabels?.byteLength ?? 0;
            const seedMaskMemory = labeling.seedMask?.byteLength ?? 0;
            const partingSurfaceMemory = labelsMemory + escapeCostMemory + boundaryLabelsMemory + seedMaskMemory;
            
            console.log(`  📊 Parting Surface Memory footprint:`);
            console.log(`    labels: ${(labelsMemory / 1024 / 1024).toFixed(2)} MB`);
            console.log(`    escapeCost: ${(escapeCostMemory / 1024 / 1024).toFixed(2)} MB`);
            console.log(`    boundaryLabels: ${(boundaryLabelsMemory / 1024 / 1024).toFixed(2)} MB`);
            console.log(`    seedMask: ${(seedMaskMemory / 1024 / 1024).toFixed(2)} MB`);
            console.log(`    TOTAL (parting surface only): ${(partingSurfaceMemory / 1024 / 1024).toFixed(2)} MB`);
            
            // Create visualization after labeling is computed
            if (escapeLabelingRef.current && volumetricGridRef.current) {
              const labeling = escapeLabelingRef.current;
              
              if (partingSurfaceDebugMode !== 'none') {
                // Debug visualization mode
                console.log(`Using debug visualization mode: ${partingSurfaceDebugMode}`);
                escapeLabelingVisualizationRef.current = createDebugVisualization(
                  volumetricGridRef.current,
                  labeling,
                  partingSurfaceDebugMode,
                  6  // larger point size for visibility
                );
              } else {
                escapeLabelingVisualizationRef.current = createEscapeLabelingPointCloud(
                  volumetricGridRef.current,
                  labeling,
                  3
                );
              }
              
              if (escapeLabelingVisualizationRef.current) {
                scene.add(escapeLabelingVisualizationRef.current);
              }
            }
          } catch (error) {
            console.error('Error computing escape labeling:', error);
          }
        })();
      } else {
        // Use cached result - create visualization synchronously
        if (escapeLabelingRef.current && volumetricGridRef.current) {
          const labeling = escapeLabelingRef.current;
          
          if (partingSurfaceDebugMode !== 'none') {
            // Debug visualization mode
            console.log(`Using debug visualization mode: ${partingSurfaceDebugMode}`);
            escapeLabelingVisualizationRef.current = createDebugVisualization(
              volumetricGridRef.current,
              labeling,
              partingSurfaceDebugMode,
              6  // larger point size for visibility
            );
          } else {
            escapeLabelingVisualizationRef.current = createEscapeLabelingPointCloud(
              volumetricGridRef.current,
              labeling,
              3
            );
          }
          
          if (escapeLabelingVisualizationRef.current) {
            scene.add(escapeLabelingVisualizationRef.current);
          }
        }
      }
      
      // Hide the regular voxel grid visualization when showing escape labeling
      if (gridVisualizationRef.current) {
        gridVisualizationRef.current.visible = false;
      }
      
    } else {
      // Clear labeling when not showing parting surface
      if (!showPartingSurface) {
        escapeLabelingRef.current = null;
        onEscapeLabelingReady?.(null);
      }
      // Show regular voxel grid if escape labeling is not shown
      if (gridVisualizationRef.current) {
        gridVisualizationRef.current.visible = !hideVoxelGrid;
      }
    }
  }, [showPartingSurface, partingSurfaceAdjacency, partingSurfaceDebugMode, onEscapeLabelingReady, hideVoxelGrid, showMoldHalfClassification, showCsgResult]);

  // ============================================================================
  // VISIBILITY PAINTING
  // ============================================================================

  useEffect(() => {
    if (!meshRef.current) return;
    
    const mesh = meshRef.current;
    const visData = visibilityDataRef.current;
    
    if (visData && (showD1Paint || showD2Paint)) {
      applyVisibilityPaint(mesh, visData, showD1Paint, showD2Paint);
    } else {
      removeVisibilityPaint(mesh);
    }
  }, [showD1Paint, showD2Paint]);

  // ============================================================================
  // ORIGINAL MESH VISIBILITY
  // ============================================================================

  useEffect(() => {
    if (!meshRef.current) return;
    meshRef.current.visible = !hideOriginalMesh;
  }, [hideOriginalMesh]);

  // ============================================================================
  // HULL VISIBILITY
  // ============================================================================

  useEffect(() => {
    if (!inflatedHullRef.current) return;
    inflatedHullRef.current.mesh.visible = !hideHull;
  }, [hideHull]);

  // ============================================================================
  // CAVITY VISIBILITY
  // ============================================================================

  useEffect(() => {
    if (!csgResultRef.current) return;
    csgResultRef.current.mesh.visible = !hideCavity;
  }, [hideCavity]);

  // ============================================================================
  // VOXEL GRID VISIBILITY
  // ============================================================================

  useEffect(() => {
    if (gridVisualizationRef.current) {
      gridVisualizationRef.current.visible = !hideVoxelGrid;
    }
    if (gridBoundingBoxRef.current) {
      gridBoundingBoxRef.current.visible = !hideVoxelGrid;
    }
  }, [hideVoxelGrid]);

  // Cleanup arrows, hull, CSG result, grid, dimension labels, and escape labeling on unmount
  useEffect(() => {
    return () => {
      if (partingArrowsRef.current.length > 0) {
        removePartingDirectionArrows(partingArrowsRef.current);
      }
      if (inflatedHullRef.current && sceneRef.current) {
        removeInflatedHull(sceneRef.current, inflatedHullRef.current);
      }
      if (csgResultRef.current && sceneRef.current) {
        removeCsgResult(sceneRef.current, csgResultRef.current);
      }
      if (gridVisualizationRef.current && sceneRef.current) {
        removeGridVisualization(sceneRef.current, gridVisualizationRef.current);
      }
      if (gridBoundingBoxRef.current && sceneRef.current) {
        removeGridVisualization(sceneRef.current, gridBoundingBoxRef.current);
      }
      if (dimensionLabelsRef.current && sceneRef.current) {
        removeDimensionLabels(sceneRef.current, dimensionLabelsRef.current);
      }
      if (escapeLabelingVisualizationRef.current && sceneRef.current) {
        sceneRef.current.remove(escapeLabelingVisualizationRef.current);
        // Handle both Group and Points cleanup
        escapeLabelingVisualizationRef.current.traverse((child) => {
          if (child instanceof THREE.Points) {
            child.geometry.dispose();
            (child.material as THREE.Material).dispose();
          }
        });
      }
    };
  }, []);

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100vh',
        margin: 0,
        padding: 0,
        overflow: 'hidden'
      }}
    />
  );
});

export default ThreeViewer;
