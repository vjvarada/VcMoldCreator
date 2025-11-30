/**
 * ThreeViewer - 3D STL viewer with mold analysis visualization
 * 
 * Features:
 * - STL file loading and display
 * - Orthographic camera with orbit controls
 * - Parting direction computation and visualization
 * - Visibility painting for mold analysis
 */

import { useEffect, useRef } from 'react';
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
  generateVolumetricGrid,
  generateVolumetricGridGPU,
  isWebGPUAvailable,
  createMoldVolumePointCloud,
  createMoldVolumeVoxels,
  createGridBoundingBoxHelper,
  removeGridVisualization,
  type VolumetricGridResult,
  type VolumetricGridOptions
} from '../utils/volumetricGrid';
import {
  classifyMoldHalves,
  applyMoldHalfPaint,
  removeMoldHalfPaint,
  type MoldHalfClassificationResult
} from '../utils/moldHalfClassification';
import {
  computeEscapeLabeling,
  computeEscapeLabelingDijkstra,
  createEscapeLabelingPointCloud,
  createPartingSurfaceInterfaceCloud,
  createBoundaryDebugPointCloud,
  type EscapeLabelingResult,
  type AdjacencyType
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
  /** Grid resolution (cells per dimension) */
  gridResolution?: number;
  /** Grid visualization mode: 'points', 'voxels', or 'none' */
  gridVisualizationMode?: GridVisualizationMode;
  /** Use GPU acceleration for grid generation (WebGPU if available) */
  useGPUGrid?: boolean;
  /** Show mold half classification (H₁/H₂ coloring on cavity) */
  showMoldHalfClassification?: boolean;
  /** Show parting surface escape labeling */
  showPartingSurface?: boolean;
  /** Adjacency type for parting surface computation (6 or 26) */
  partingSurfaceAdjacency?: AdjacencyType;
  /** Show only the parting surface interface (boundary between H1 and H2) */
  showPartingSurfaceInterface?: boolean;
  onMeshLoaded?: (mesh: THREE.Mesh) => void;
  onMeshRepaired?: (result: MeshRepairResult) => void;
  onVisibilityDataReady?: (data: VisibilityPaintData | null) => void;
  onInflatedHullReady?: (result: InflatedHullResult | null) => void;
  onCsgResultReady?: (result: CsgSubtractionResult | null) => void;
  onVolumetricGridReady?: (result: VolumetricGridResult | null) => void;
  onMoldHalfClassificationReady?: (result: MoldHalfClassificationResult | null) => void;
  onEscapeLabelingReady?: (result: EscapeLabelingResult | null) => void;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const FRUSTUM_SIZE = 10;
const BACKGROUND_COLOR = 0x1a1a1a;
const GRID_COLOR_CENTER = 0x444444;
const GRID_COLOR_LINES = 0x333333;

// ============================================================================
// COMPONENT
// ============================================================================

const ThreeViewer: React.FC<ThreeViewerProps> = ({
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
  gridResolution = 64,
  gridVisualizationMode = 'points',
  useGPUGrid = true,
  showMoldHalfClassification = false,
  showPartingSurface = false,
  partingSurfaceAdjacency = 6,
  showPartingSurfaceInterface = false,
  onMeshLoaded,
  onMeshRepaired,
  onVisibilityDataReady,
  onInflatedHullReady,
  onCsgResultReady,
  onVolumetricGridReady,
  onMoldHalfClassificationReady,
  onEscapeLabelingReady
}) => {
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
  const escapeLabelingRef = useRef<EscapeLabelingResult | null>(null);
  const escapeLabelingVisualizationRef = useRef<THREE.Points | null>(null);

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

    // Add grid
    scene.add(new THREE.GridHelper(10, 10, GRID_COLOR_CENTER, GRID_COLOR_LINES));

    // Create orthographic camera
    const aspect = container.clientWidth / container.clientHeight;
    const camera = new THREE.OrthographicCamera(
      -FRUSTUM_SIZE * aspect / 2,
      FRUSTUM_SIZE * aspect / 2,
      FRUSTUM_SIZE / 2,
      -FRUSTUM_SIZE / 2,
      0.1,
      1000
    );
    camera.position.set(5, 5, 5);
    cameraRef.current = camera;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Add lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    
    const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
    light1.position.set(10, 10, 10);
    scene.add(light1);
    
    const light2 = new THREE.DirectionalLight(0xffffff, 0.4);
    light2.position.set(-10, -10, -10);
    scene.add(light2);

    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 0.5;
    controls.maxDistance = 100;
    controlsRef.current = controls;

    // Handle resize
    const handleResize = () => {
      if (!cameraRef.current || !rendererRef.current) return;
      
      const aspect = container.clientWidth / container.clientHeight;
      cameraRef.current.left = -FRUSTUM_SIZE * aspect / 2;
      cameraRef.current.right = FRUSTUM_SIZE * aspect / 2;
      cameraRef.current.top = FRUSTUM_SIZE / 2;
      cameraRef.current.bottom = -FRUSTUM_SIZE / 2;
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

        // Center geometry with bottom at Z=0
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -geometry.boundingBox.min.z);
        geometry.computeBoundingBox();
        
        // Scale to fit
        const size = new THREE.Vector3();
        geometry.boundingBox!.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = maxDim > 0 ? 4 / maxDim : 1;

        // Create mesh
        const material = new THREE.MeshPhongMaterial({
          color: COLORS.MESH_DEFAULT,
          specular: 0x333333,
          shininess: 60
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.scale.setScalar(scale);
        mesh.rotation.x = -Math.PI / 2; // Z-up to Y-up
        
        scene.add(mesh);
        meshRef.current = mesh;

        console.log('STL loaded:', geometry.attributes.position.count, 'vertices');
        onMeshLoaded?.(mesh);

        // Position camera
        const scaledHeight = size.z * scale;
        camera.position.set(8, 5.6, 8);
        controls.target.set(0, scaledHeight / 2, 0);
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
      const classification = classifyMoldHalves(cavityGeometry, hullGeometry, d1, d2);
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
    volumetricGridRef.current = null;
    onVolumetricGridReady?.(null);

    // Generate volumetric grid if requested and we have both hull and mesh
    if (showVolumetricGrid && inflatedHullRef.current && meshRef.current) {
      // Use async IIFE for GPU version
      (async () => {
        try {
          // Check WebGPU availability
          const gpuAvailable = isWebGPUAvailable();
          const useGPU = useGPUGrid && gpuAvailable;
          
          console.log('Generating volumetric grid...');
          console.log(`  Using: ${useGPU ? 'WebGPU (GPU)' : 'CPU'}`);
          if (useGPUGrid && !gpuAvailable) {
            console.log('  Note: WebGPU not available, falling back to CPU');
          }
          
          // Get geometries in world space
          const shellGeometry = inflatedHullRef.current!.mesh.geometry.clone();
          shellGeometry.applyMatrix4(inflatedHullRef.current!.mesh.matrixWorld);
          
          const partGeometry = meshRef.current!.geometry.clone();
          partGeometry.applyMatrix4(meshRef.current!.matrixWorld);
          
          const options: VolumetricGridOptions = {
            resolution: gridResolution,
            storeAllCells: false,
            marginPercent: 0.02,
            computeDistances: false,
          };
          
          // Use GPU or CPU version based on availability and preference
          const gridResult = useGPU
            ? await generateVolumetricGridGPU(shellGeometry, partGeometry, options)
            : generateVolumetricGrid(shellGeometry, partGeometry, options);
          
          console.log(`Volumetric grid generated:`);
          console.log(`  Resolution: ${gridResult.resolution.x}×${gridResult.resolution.y}×${gridResult.resolution.z}`);
          console.log(`  Mold volume cells: ${gridResult.moldVolumeCellCount} / ${gridResult.totalCellCount}`);
          console.log(`  Fill ratio: ${(gridResult.stats.fillRatio * 100).toFixed(1)}%`);
          console.log(`  Approx mold volume: ${gridResult.stats.moldVolume.toFixed(4)}`);
          console.log(`  Compute time: ${gridResult.stats.computeTimeMs.toFixed(1)} ms`);
          
          volumetricGridRef.current = gridResult;
          onVolumetricGridReady?.(gridResult);
          
          // Create initial visualization based on mode
          if (gridVisualizationMode !== 'none' && gridResult.moldVolumeCellCount > 0) {
            if (gridVisualizationMode === 'points') {
              gridVisualizationRef.current = createMoldVolumePointCloud(gridResult, 0x00ffff, 3);
            } else if (gridVisualizationMode === 'voxels') {
              gridVisualizationRef.current = createMoldVolumeVoxels(gridResult, 0x00ffff, 0.2);
            }
            
            if (gridVisualizationRef.current) {
              scene.add(gridVisualizationRef.current);
            }
            
            // Add bounding box helper
            gridBoundingBoxRef.current = createGridBoundingBoxHelper(gridResult, 0xffff00);
            scene.add(gridBoundingBoxRef.current);
          }
          
          // Clean up temporary geometries
          shellGeometry.dispose();
          partGeometry.dispose();
          
        } catch (error) {
          console.error('Error generating volumetric grid:', error);
        }
      })();
    }
  }, [showVolumetricGrid, gridResolution, useGPUGrid, onVolumetricGridReady]);

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
        gridVisualizationRef.current = createMoldVolumePointCloud(gridResult, 0x00ffff, 3);
      } else if (gridVisualizationMode === 'voxels') {
        gridVisualizationRef.current = createMoldVolumeVoxels(gridResult, 0x00ffff, 0.2);
      }
      
      if (gridVisualizationRef.current) {
        scene.add(gridVisualizationRef.current);
      }
      
      // Add bounding box helper
      gridBoundingBoxRef.current = createGridBoundingBoxHelper(gridResult, 0xffff00);
      scene.add(gridBoundingBoxRef.current);
    }
  }, [gridVisualizationMode]);

  // ============================================================================
  // PARTING SURFACE COMPUTATION
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;

    // Clean up existing escape labeling visualization
    if (escapeLabelingVisualizationRef.current) {
      scene.remove(escapeLabelingVisualizationRef.current);
      escapeLabelingVisualizationRef.current.geometry.dispose();
      (escapeLabelingVisualizationRef.current.material as THREE.Material).dispose();
      escapeLabelingVisualizationRef.current = null;
    }
    escapeLabelingRef.current = null;
    onEscapeLabelingReady?.(null);

    // Compute escape labeling if requested and we have the required data
    if (showPartingSurface && volumetricGridRef.current && visibilityDataRef.current && meshRef.current) {
      try {
        console.log('Computing escape labeling (parting surface)...');
        
        let labeling: EscapeLabelingResult;
        
        // Check if we have mold half classification and CSG result for Dijkstra-based labeling
        if (moldHalfClassificationRef.current && csgResultRef.current) {
          console.log('Using Dijkstra-based escape labeling with mold half classification');
          
          // Get CSG cavity geometry in world space (this is the outer shell ∂H)
          const shellGeometry = csgResultRef.current.mesh.geometry.clone();
          shellGeometry.applyMatrix4(csgResultRef.current.mesh.matrixWorld);
          
          // Compute escape labeling using multi-source Dijkstra
          labeling = computeEscapeLabelingDijkstra(
            volumetricGridRef.current,
            shellGeometry,
            moldHalfClassificationRef.current,
            { adjacency: partingSurfaceAdjacency, seedRadius: 1.5 }
          );
          
          shellGeometry.dispose();
        } else {
          console.log('Falling back to legacy escape labeling (no mold half classification)');
          
          // Get part geometry in world space
          const partGeometry = meshRef.current.geometry.clone();
          partGeometry.applyMatrix4(meshRef.current.matrixWorld);
          
          // Get d1/d2 directions from visibility data
          const { d1, d2 } = visibilityDataRef.current;
          
          // Compute escape labeling using legacy method
          labeling = computeEscapeLabeling(
            volumetricGridRef.current,
            partGeometry,
            d1,
            d2,
            { adjacency: partingSurfaceAdjacency }
          );
          
          partGeometry.dispose();
        }
        
        escapeLabelingRef.current = labeling;
        onEscapeLabelingReady?.(labeling);
        
        // DEBUG: Use boundary debug visualization to verify boundary labeling
        // This shows:
        // - Green: Boundary voxels adjacent to H₁
        // - Orange: Boundary voxels adjacent to H₂
        // - Blue: Seed voxels (near part mesh)
        // - Dark gray: Everything else
        const DEBUG_BOUNDARY_VISUALIZATION = false;  // Set to true to debug boundary detection
        
        // Create visualization
        if (DEBUG_BOUNDARY_VISUALIZATION) {
          console.log('DEBUG MODE: Showing boundary voxel labels only');
          escapeLabelingVisualizationRef.current = createBoundaryDebugPointCloud(
            volumetricGridRef.current,
            labeling,
            6  // larger point size for visibility
          );
        } else if (showPartingSurfaceInterface) {
          escapeLabelingVisualizationRef.current = createPartingSurfaceInterfaceCloud(
            volumetricGridRef.current,
            labeling,
            partingSurfaceAdjacency,
            4
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
        
        // Hide the regular voxel grid visualization when showing escape labeling
        if (gridVisualizationRef.current) {
          gridVisualizationRef.current.visible = false;
        }
        
      } catch (error) {
        console.error('Error computing escape labeling:', error);
      }
    } else {
      // Show regular voxel grid if escape labeling is not shown
      if (gridVisualizationRef.current) {
        gridVisualizationRef.current.visible = !hideVoxelGrid;
      }
    }
  }, [showPartingSurface, partingSurfaceAdjacency, showPartingSurfaceInterface, onEscapeLabelingReady, hideVoxelGrid, showMoldHalfClassification, showCsgResult]);

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

  // Cleanup arrows, hull, CSG result, grid, and escape labeling on unmount
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
      if (escapeLabelingVisualizationRef.current && sceneRef.current) {
        sceneRef.current.remove(escapeLabelingVisualizationRef.current);
        escapeLabelingVisualizationRef.current.geometry.dispose();
        (escapeLabelingVisualizationRef.current.material as THREE.Material).dispose();
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
};

export default ThreeViewer;
