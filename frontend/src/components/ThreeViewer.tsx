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
  createRLineVisualization,
  removeGridVisualization,
  type VolumetricGridResult,
  type VolumetricGridOptions,
  type DistanceFieldType
} from '../utils/volumetricGrid';
import {
  classifyMoldHalves,
  applyMoldHalfPaint,
  removeMoldHalfPaint,
  type MoldHalfClassificationResult
} from '../utils/moldHalfClassification';
import {
  buildTetraGraph,
  buildMinimalTetraGraph,
  computeTetraEdgeWeights,
  computeTetraPartingSurface,
  createPartingSurfaceVisualization,
  createVertexLabelVisualization,
  createSeedBoundaryVisualization,
  createLabeledSeedsOnlyVisualization,
  createLabeledSeedAndBoundaryVisualization,
  createAllVerticesLabeledVisualization,
  type TetraPartingSurfaceResult
} from '../utils/tetraPartingSurface';
import {
  tetrahedralizeGeometryWithProgress,
  createTetrahedraPointCloud,
  createTetrahedraWireframe,
  createTetrahedraBoundingBox,
  removeTetrahedralizationVisualization,
  checkBackendHealth,
  type TetrahedralizationResult,
  type TetrahedralizationProgress,
  type TetrahedralizationOptions
} from '../utils/tetrahedralization';

// ============================================================================
// TYPES
// ============================================================================

/** Visualization mode for volumetric grid */
export type GridVisualizationMode = 'points' | 'voxels' | 'none';

/** Visualization mode for tetrahedralization */
export type TetraVisualizationMode = 'points' | 'wireframe' | 'none';

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
  /** Grid resolution (cells per dimension) */
  gridResolution?: number;
  /** Grid visualization mode: 'points', 'voxels', or 'none' */
  gridVisualizationMode?: GridVisualizationMode;
  /** Use GPU acceleration for grid generation (WebGPU if available) */
  useGPUGrid?: boolean;
  /** Which distance field to use for voxel coloring: 'part' (distance to part) or 'biased' (biased distance) */
  distanceFieldType?: DistanceFieldType;
  /** Show mold half classification (H₁/H₂ coloring on cavity) */
  showMoldHalfClassification?: boolean;
  /** Boundary zone threshold as fraction of bounding box diagonal (default: 0.15 = 15%) */
  boundaryZoneThreshold?: number;
  /** Show parting surface escape labeling */
  showPartingSurface?: boolean;
  /** Debug visualization mode for parting surface */
  partingSurfaceDebugMode?: 'none' | 'seed-boundary' | 'labeled-seeds' | 'seed-boundary-labeled' | 'all-vertices';
  /** Whether CSG result is ready (used to trigger tetrahedralization) */
  csgReady?: boolean;
  /** Show tetrahedralization (replaces voxelization) */
  showTetrahedralization?: boolean;
  /** Hide tetrahedralization visualization (without recomputing) */
  hideTetrahedralization?: boolean;
  /** Edge length factor for tetrahedralization (smaller = denser mesh, default: 0.0065) */
  tetraEdgeLengthFac?: number;
  /** Visualization mode for tetrahedralization */
  tetraVisualizationMode?: TetraVisualizationMode;
  /** Pre-loaded tetrahedral mesh (from file) to use instead of computing */
  preloadedTetraResult?: TetrahedralizationResult | null;
  onMeshLoaded?: (mesh: THREE.Mesh, meshInfo: { diagonal: number; size: { x: number; y: number; z: number }; scale: number }) => void;
  onMeshRepaired?: (result: MeshRepairResult) => void;
  onVisibilityDataReady?: (data: VisibilityPaintData | null) => void;
  onInflatedHullReady?: (result: InflatedHullResult | null) => void;
  onCsgResultReady?: (result: CsgSubtractionResult | null) => void;
  onVolumetricGridReady?: (result: VolumetricGridResult | null) => void;
  onMoldHalfClassificationReady?: (result: MoldHalfClassificationResult | null) => void;
  onTetraPartingSurfaceReady?: (result: TetraPartingSurfaceResult | null) => void;
  onTetrahedralizationReady?: (result: TetrahedralizationResult | null) => void;
  onTetrahedralizationProgress?: (progress: TetrahedralizationProgress) => void;
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
  showRLine = true,
  gridResolution = 64,
  gridVisualizationMode = 'points',
  useGPUGrid = true,
  distanceFieldType = 'part',
  showMoldHalfClassification = false,
  boundaryZoneThreshold = 0.15,
  showPartingSurface = false,
  partingSurfaceDebugMode = 'none',
  csgReady = false,
  showTetrahedralization = false,
  hideTetrahedralization = false,
  tetraEdgeLengthFac = 0.0065,
  tetraVisualizationMode = 'points',
  preloadedTetraResult = null,
  onMeshLoaded,
  onMeshRepaired,
  onVisibilityDataReady,
  onInflatedHullReady,
  onCsgResultReady,
  onVolumetricGridReady,
  onMoldHalfClassificationReady,
  onTetraPartingSurfaceReady,
  onTetrahedralizationReady,
  onTetrahedralizationProgress
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
  const rLineVisualizationRef = useRef<THREE.Group | null>(null);
  // Tetrahedral parting surface refs
  const tetraPartingSurfaceRef = useRef<TetraPartingSurfaceResult | null>(null);
  const partingSurfaceVisualizationRef = useRef<THREE.Object3D | null>(null);
  // Tetrahedralization refs
  const tetrahedralizationRef = useRef<TetrahedralizationResult | null>(null);
  const tetraVisualizationRef = useRef<THREE.Object3D | null>(null);
  const tetraBoundingBoxRef = useRef<THREE.LineSegments | null>(null);

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

        // Calculate bounding box diagonal for mesh info
        const diagonal = Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z);
        
        console.log('STL loaded:', geometry.attributes.position.count, 'vertices');
        console.log('Bounding box size:', size.x.toFixed(3), 'x', size.y.toFixed(3), 'x', size.z.toFixed(3));
        console.log('Bounding box diagonal:', diagonal.toFixed(3));
        onMeshLoaded?.(mesh, {
          diagonal,
          size: { x: size.x, y: size.y, z: size.z },
          scale
        });

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
  }, [showVolumetricGrid, gridResolution, useGPUGrid, showRLine, onVolumetricGridReady]);

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
  // TETRAHEDRALIZATION (Backend fTetWild)
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;
    
    // Create abort flag for this effect instance
    let aborted = false;

    // Clean up existing tetrahedralization visualization
    if (tetraVisualizationRef.current) {
      scene.remove(tetraVisualizationRef.current);
      if (tetraVisualizationRef.current instanceof THREE.Points || 
          tetraVisualizationRef.current instanceof THREE.LineSegments) {
        tetraVisualizationRef.current.geometry?.dispose();
        if (Array.isArray(tetraVisualizationRef.current.material)) {
          tetraVisualizationRef.current.material.forEach(m => m.dispose());
        } else {
          (tetraVisualizationRef.current.material as THREE.Material)?.dispose();
        }
      }
      tetraVisualizationRef.current = null;
    }
    if (tetraBoundingBoxRef.current) {
      scene.remove(tetraBoundingBoxRef.current);
      tetraBoundingBoxRef.current.geometry?.dispose();
      (tetraBoundingBoxRef.current.material as THREE.Material)?.dispose();
      tetraBoundingBoxRef.current = null;
    }
    tetrahedralizationRef.current = null;
    onTetrahedralizationReady?.(null);

    // Use preloaded result if provided (from loaded file)
    if (preloadedTetraResult && showTetrahedralization) {
      console.log('Using preloaded tetrahedral mesh:');
      console.log(`  Input: ${preloadedTetraResult.inputStats.numVertices} vertices, ${preloadedTetraResult.inputStats.numFaces} faces`);
      console.log(`  Output: ${preloadedTetraResult.numVertices} vertices, ${preloadedTetraResult.numTetrahedra} tetrahedra`);
      
      tetrahedralizationRef.current = preloadedTetraResult;
      
      // Create visualization based on mode
      if (tetraVisualizationMode !== 'none' && preloadedTetraResult.numTetrahedra > 0) {
        console.log(`  Creating ${tetraVisualizationMode} visualization...`);
        
        if (tetraVisualizationMode === 'points') {
          tetraVisualizationRef.current = createTetrahedraPointCloud(preloadedTetraResult, 0x00ff88, 3);
        } else if (tetraVisualizationMode === 'wireframe') {
          tetraVisualizationRef.current = createTetrahedraWireframe(preloadedTetraResult, 0x00ff88, 0.5);
        }
        
        if (tetraVisualizationRef.current) {
          scene.add(tetraVisualizationRef.current);
        }
        
        // Add bounding box helper
        tetraBoundingBoxRef.current = createTetrahedraBoundingBox(preloadedTetraResult, 0x88ff00);
        scene.add(tetraBoundingBoxRef.current);
        console.log(`  Visualization added to scene`);
      }
      
      onTetrahedralizationReady?.(preloadedTetraResult);
      return; // Don't run the backend computation
    }

    // Generate tetrahedralization if requested and we have the CSG result (mold cavity)
    // Note: csgReady prop signals when csgResultRef.current is set
    if (showTetrahedralization && csgReady && csgResultRef.current) {
      (async () => {
        try {
          // Check backend health first
          const isHealthy = await checkBackendHealth();
          if (!isHealthy) {
            console.error('Backend server is not available. Start the backend with: python backend/app.py');
            return;
          }
          
          // Check if aborted before starting
          if (aborted) return;

          console.log('Generating tetrahedralization via backend...');
          console.log(`  Edge length factor: ${tetraEdgeLengthFac}`);
          
          // Get CSG cavity geometry in world space
          const cavityGeometry = csgResultRef.current!.mesh.geometry.clone();
          cavityGeometry.applyMatrix4(csgResultRef.current!.mesh.matrixWorld);
          
          // Track last progress for visualization update
          let lastProgress: TetrahedralizationProgress | null = null;
          
          // Helper to get last progress with safe type
          const getLastProgress = () => lastProgress;
          
          // Send to backend for tetrahedralization with progress tracking
          const result = await tetrahedralizeGeometryWithProgress(
            cavityGeometry,
            {
              edgeLengthFac: tetraEdgeLengthFac,
              optimize: false, // Faster, good enough for visualization
            },
            (progress) => {
              // Don't update progress if aborted
              if (!aborted) {
                // Log new messages from backend
                if (progress.logs && progress.logs.length > 0) {
                  const newLogs = lastProgress?.logs 
                    ? progress.logs.slice(lastProgress.logs.length)
                    : progress.logs;
                  newLogs.forEach(log => {
                    console.log(`  [${log.time.toFixed(1)}s] ${log.message}`);
                  });
                }
                lastProgress = progress;
                onTetrahedralizationProgress?.(progress);
              }
            }
          );
          
          // Check if aborted before applying results
          if (aborted) {
            console.log('Tetrahedralization completed but was aborted, discarding result');
            cavityGeometry.dispose();
            return;
          }
          
          // Get last progress safely
          const savedProgress = getLastProgress();
          
          // Signal visualization phase
          const visualizingProgress: TetrahedralizationProgress = {
            job_id: savedProgress?.job_id || '',
            status: 'visualizing',
            progress: 99,
            message: 'Creating 3D visualization...',
            substep: `Building ${tetraVisualizationMode} view for ${result.numTetrahedra.toLocaleString()} tetrahedra`,
            elapsed_seconds: savedProgress?.elapsed_seconds || 0,
            estimated_remaining: null,
            input_stats: savedProgress?.input_stats || null,
            complete: false,
            error: null,
            logs: savedProgress?.logs,
          };
          onTetrahedralizationProgress?.(visualizingProgress);
          
          // Small delay to allow UI to update before heavy visualization work
          await new Promise(resolve => setTimeout(resolve, 50));
          
          console.log(`Tetrahedralization completed:`);
          console.log(`  Input: ${result.inputStats.numVertices} vertices, ${result.inputStats.numFaces} faces`);
          console.log(`  Output: ${result.numVertices} vertices, ${result.numTetrahedra} tetrahedra`);
          console.log(`  Time: ${result.computeTimeMs.toFixed(1)} ms`);
          
          tetrahedralizationRef.current = result;
          
          // Create visualization based on mode
          if (tetraVisualizationMode !== 'none' && result.numTetrahedra > 0) {
            console.log(`  Creating ${tetraVisualizationMode} visualization...`);
            
            // Update progress before heavy computation
            onTetrahedralizationProgress?.({
              ...visualizingProgress,
              substep: `Generating ${tetraVisualizationMode} geometry...`,
            });
            await new Promise(resolve => setTimeout(resolve, 10));
            
            if (tetraVisualizationMode === 'points') {
              tetraVisualizationRef.current = createTetrahedraPointCloud(result, 0x00ff88, 3);
            } else if (tetraVisualizationMode === 'wireframe') {
              tetraVisualizationRef.current = createTetrahedraWireframe(result, 0x00ff88, 0.5);
            }
            
            if (tetraVisualizationRef.current) {
              scene.add(tetraVisualizationRef.current);
            }
            
            // Add bounding box helper
            tetraBoundingBoxRef.current = createTetrahedraBoundingBox(result, 0x88ff00);
            scene.add(tetraBoundingBoxRef.current);
            console.log(`  Visualization added to scene`);
          }
          
          // Signal complete - now truly done with visualization
          const completeProgress: TetrahedralizationProgress = {
            job_id: savedProgress?.job_id || '',
            status: 'complete',
            progress: 100,
            message: `Complete: ${result.numVertices.toLocaleString()} vertices, ${result.numTetrahedra.toLocaleString()} tetrahedra`,
            substep: 'Visualization ready',
            elapsed_seconds: result.computeTimeMs / 1000,
            estimated_remaining: 0,
            input_stats: savedProgress?.input_stats || null,
            complete: true,
            error: null,
            logs: savedProgress?.logs,
          };
          onTetrahedralizationProgress?.(completeProgress);
          onTetrahedralizationReady?.(result);
          
          // Clean up temporary geometry
          cavityGeometry.dispose();
          
        } catch (error) {
          if (!aborted) {
            console.error('Error generating tetrahedralization:', error);
          }
        }
      })();
    }
    
    // Cleanup function - abort any in-progress tetrahedralization when effect re-runs
    return () => {
      aborted = true;
    };
  }, [showTetrahedralization, csgReady, tetraEdgeLengthFac, tetraVisualizationMode, onTetrahedralizationReady, onTetrahedralizationProgress, preloadedTetraResult]);

  // ============================================================================
  // TETRAHEDRALIZATION VISUALIZATION MODE UPDATE
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current || !tetrahedralizationRef.current) return;
    
    const scene = sceneRef.current;
    const result = tetrahedralizationRef.current;

    // Remove existing visualization (but keep the computed data)
    if (tetraVisualizationRef.current) {
      removeTetrahedralizationVisualization(scene, tetraVisualizationRef.current);
      tetraVisualizationRef.current = null;
    }
    if (tetraBoundingBoxRef.current) {
      scene.remove(tetraBoundingBoxRef.current);
      tetraBoundingBoxRef.current.geometry?.dispose();
      (tetraBoundingBoxRef.current.material as THREE.Material)?.dispose();
      tetraBoundingBoxRef.current = null;
    }

    // Recreate visualization with new mode
    if (tetraVisualizationMode !== 'none' && result.numTetrahedra > 0) {
      if (tetraVisualizationMode === 'points') {
        tetraVisualizationRef.current = createTetrahedraPointCloud(result, 0x00ff88, 3);
      } else if (tetraVisualizationMode === 'wireframe') {
        tetraVisualizationRef.current = createTetrahedraWireframe(result, 0x00ff88, 0.5);
      }
      
      if (tetraVisualizationRef.current) {
        scene.add(tetraVisualizationRef.current);
      }
      
      // Add bounding box helper
      tetraBoundingBoxRef.current = createTetrahedraBoundingBox(result, 0x88ff00);
      scene.add(tetraBoundingBoxRef.current);
    }
  }, [tetraVisualizationMode]);

  // ============================================================================
  // TETRAHEDRALIZATION VISIBILITY TOGGLE
  // ============================================================================

  useEffect(() => {
    if (tetraVisualizationRef.current) {
      tetraVisualizationRef.current.visible = !hideTetrahedralization;
    }
    if (tetraBoundingBoxRef.current) {
      tetraBoundingBoxRef.current.visible = !hideTetrahedralization;
    }
  }, [hideTetrahedralization]);

  // ============================================================================
  // TETRAHEDRAL PARTING SURFACE COMPUTATION
  // ============================================================================

  useEffect(() => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;

    // Clean up existing parting surface visualization
    if (partingSurfaceVisualizationRef.current) {
      scene.remove(partingSurfaceVisualizationRef.current);
      // Handle Group or Points cleanup
      partingSurfaceVisualizationRef.current.traverse((child) => {
        if (child instanceof THREE.Points || child instanceof THREE.LineSegments) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          } else {
            (child.material as THREE.Material).dispose();
          }
        }
      });
      partingSurfaceVisualizationRef.current = null;
    }

    // Only compute parting surface when conditions are met
    if (showPartingSurface && 
        tetrahedralizationRef.current && 
        moldHalfClassificationRef.current && 
        csgResultRef.current && 
        meshRef.current) {
      
      // Check if we need to recompute or can use cached result
      const needsRecompute = !tetraPartingSurfaceRef.current;
      
      if (needsRecompute) {
        try {
          console.log('Computing tetrahedral parting surface...');
          
          // VALIDATION MODE: Use minimal graph (skips expensive edge computation)
          const graph = buildMinimalTetraGraph(tetrahedralizationRef.current);
          
          // Get meshes for distance queries
          const partMesh = meshRef.current;
          const cavityMesh = csgResultRef.current.mesh;
          const hullMesh = inflatedHullRef.current?.mesh;
          
          // VALIDATION MODE: Skip edge weight computation for now
          // Just pass graph directly to validate seed/boundary detection
          const weightResult = { graph, partMesh, cavityMesh, hullMesh } as any;
          
          // Compute parting surface via multi-source Dijkstra
          // Uses cavity mesh for both boundary detection and H1/H2 classification
          const partingResult = computeTetraPartingSurface(
            weightResult,
            cavityMesh,
            moldHalfClassificationRef.current
          );
          
          tetraPartingSurfaceRef.current = partingResult;
          onTetraPartingSurfaceReady?.(partingResult);
          
          console.log(`Parting surface computed:`);
          console.log(`  H₁ vertices: ${partingResult.h1Count}`);
          console.log(`  H₂ vertices: ${partingResult.h2Count}`);
          console.log(`  Parting edges: ${partingResult.partingEdgeCount}`);
          console.log(`  Time: ${partingResult.computeTimeMs.toFixed(1)} ms`);
          
        } catch (error) {
          console.error('Error computing tetrahedral parting surface:', error);
          onTetraPartingSurfaceReady?.(null);
        }
      }
      
      // Create visualization from result (cached or newly computed)
      if (tetraPartingSurfaceRef.current && tetrahedralizationRef.current) {
        const result = tetraPartingSurfaceRef.current;
        const tetraResult = tetrahedralizationRef.current;
        
        // Use minimal graph for visualization (only needs vertex positions)
        const graph = buildMinimalTetraGraph(tetraResult);
        
        switch (partingSurfaceDebugMode) {
          case 'seed-boundary':
            // Step 1: Show seeds (yellow) and boundary vertices (blue/orange) WITHOUT labeling
            console.log('Creating seed + boundary visualization (unlabeled)');
            partingSurfaceVisualizationRef.current = createSeedBoundaryVisualization(graph, result, 5);
            break;
            
          case 'labeled-seeds':
            // Step 2: Show only seed vertices after they are labeled
            console.log('Creating labeled seeds only visualization');
            partingSurfaceVisualizationRef.current = createLabeledSeedsOnlyVisualization(graph, result, 6);
            break;
            
          case 'seed-boundary-labeled':
            // Step 3: Show seeds and boundary vertices with labels
            console.log('Creating labeled seed + boundary visualization');
            partingSurfaceVisualizationRef.current = createLabeledSeedAndBoundaryVisualization(graph, result, 4, 6);
            break;
            
          case 'all-vertices':
            // Step 4: Show all vertices after labeling completion
            console.log('Creating all vertices labeled visualization');
            partingSurfaceVisualizationRef.current = createAllVerticesLabeledVisualization(graph, result, 3);
            break;
            
          case 'none':
          default:
            // Normal view: show parting surface edges
            console.log('Creating parting surface edge visualization');
            partingSurfaceVisualizationRef.current = createPartingSurfaceVisualization(graph, result);
            break;
        }
        
        if (partingSurfaceVisualizationRef.current) {
          scene.add(partingSurfaceVisualizationRef.current);
        }
      }
      
      // Hide voxel grid and tetra visualization when showing parting surface
      if (gridVisualizationRef.current) {
        gridVisualizationRef.current.visible = false;
      }
      if (tetraVisualizationRef.current) {
        tetraVisualizationRef.current.visible = false;
      }
      
    } else {
      // Clear parting surface when not showing
      if (!showPartingSurface) {
        tetraPartingSurfaceRef.current = null;
        onTetraPartingSurfaceReady?.(null);
      }
      // Show voxel/tetra visualization when parting surface is not shown
      if (gridVisualizationRef.current) {
        gridVisualizationRef.current.visible = !hideVoxelGrid;
      }
      if (tetraVisualizationRef.current) {
        tetraVisualizationRef.current.visible = !hideTetrahedralization;
      }
    }
  }, [showPartingSurface, partingSurfaceDebugMode, onTetraPartingSurfaceReady, hideVoxelGrid, hideTetrahedralization]);

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

  // Cleanup arrows, hull, CSG result, grid, tetrahedralization, and escape labeling on unmount
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
      if (tetraVisualizationRef.current && sceneRef.current) {
        removeTetrahedralizationVisualization(sceneRef.current, tetraVisualizationRef.current);
      }
      if (tetraBoundingBoxRef.current && sceneRef.current) {
        sceneRef.current.remove(tetraBoundingBoxRef.current);
        tetraBoundingBoxRef.current.geometry?.dispose();
        (tetraBoundingBoxRef.current.material as THREE.Material)?.dispose();
      }
      if (partingSurfaceVisualizationRef.current && sceneRef.current) {
        sceneRef.current.remove(partingSurfaceVisualizationRef.current);
        // Handle Group, Points, and LineSegments cleanup
        partingSurfaceVisualizationRef.current.traverse((child) => {
          if (child instanceof THREE.Points || child instanceof THREE.LineSegments) {
            child.geometry.dispose();
            if (Array.isArray(child.material)) {
              child.material.forEach(m => m.dispose());
            } else {
              (child.material as THREE.Material).dispose();
            }
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
};

export default ThreeViewer;
