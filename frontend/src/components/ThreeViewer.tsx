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

// ============================================================================
// TYPES
// ============================================================================

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
  onMeshLoaded?: (mesh: THREE.Mesh) => void;
  onMeshRepaired?: (result: MeshRepairResult) => void;
  onVisibilityDataReady?: (data: VisibilityPaintData | null) => void;
  onInflatedHullReady?: (result: InflatedHullResult | null) => void;
  onCsgResultReady?: (result: CsgSubtractionResult | null) => void;
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
  onMeshLoaded,
  onMeshRepaired,
  onVisibilityDataReady,
  onInflatedHullReady,
  onCsgResultReady
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

  // Cleanup arrows, hull, and CSG result on unmount
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
