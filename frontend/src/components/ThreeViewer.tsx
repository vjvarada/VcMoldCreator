// The component should:
// - create a WebGLRenderer
// - add ambient and directional light
// - load an STL from a given URL prop
// - center and scale the mesh
// - add OrbitControls (pan, tilt, orbit, zoom)
// - handle window resize

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three-stdlib';
import { STLLoader } from 'three-stdlib';

interface ThreeViewerProps {
  stlUrl?: string;
}

const ThreeViewer: React.FC<ThreeViewerProps> = ({ stlUrl }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const animationIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Add grid helper for reference
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x333333);
    scene.add(gridHelper);

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-10, -10, -10);
    scene.add(directionalLight2);

    // Add OrbitControls with pan, tilt, orbit, and zoom
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.enableRotate = true;
    controls.panSpeed = 1.0;
    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.minDistance = 0.5;
    controls.maxDistance = 100;
    controlsRef.current = controls;

    // Handle window resize
    const handleResize = () => {
      if (!container || !cameraRef.current || !rendererRef.current) return;
      
      cameraRef.current.aspect = container.clientWidth / container.clientHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      if (controlsRef.current) controlsRef.current.update();
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
        if (container && rendererRef.current.domElement) {
          container.removeChild(rendererRef.current.domElement);
        }
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
    };
  }, []);

  // Load STL when URL changes
  useEffect(() => {
    if (!stlUrl || !sceneRef.current || !cameraRef.current || !controlsRef.current) return;

    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;

    // Remove previous mesh if exists
    if (meshRef.current) {
      scene.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      if (Array.isArray(meshRef.current.material)) {
        meshRef.current.material.forEach((m: THREE.Material) => m.dispose());
      } else {
        (meshRef.current.material as THREE.Material).dispose();
      }
      meshRef.current = null;
    }

    console.log('Loading STL from:', stlUrl);

    // Load STL using fetch for blob URLs
    fetch(stlUrl)
      .then(response => response.arrayBuffer())
      .then(buffer => {
        const loader = new STLLoader();
        const geometry = loader.parse(buffer);
        
        // Compute normals for proper lighting
        geometry.computeVertexNormals();
        
        // Get original bounding box before any transforms
        geometry.computeBoundingBox();
        if (!geometry.boundingBox) return;
        
        // Center geometry horizontally (X and Y in original coords), but keep Z base at 0
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        const minZ = geometry.boundingBox.min.z;
        geometry.translate(-center.x, -center.y, -minZ); // Move so bottom is at Z=0

        // Recalculate bounding box after centering
        geometry.computeBoundingBox();
        
        // Scale to fit in view
        const size = new THREE.Vector3();
        geometry.boundingBox!.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = maxDim > 0 ? 4 / maxDim : 1;

        // Create material and mesh
        const material = new THREE.MeshPhongMaterial({ 
          color: 0x00aaff,
          specular: 0x333333,
          shininess: 60,
          flatShading: false
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.scale.setScalar(scale);
        
        // Rotate from Z-up (common in CAD) to Y-up (Three.js convention)
        mesh.rotation.x = -Math.PI / 2;
        
        scene.add(mesh);
        meshRef.current = mesh;

        console.log('STL loaded successfully, vertices:', geometry.attributes.position.count);

        // Adjust camera to see the whole object
        const scaledHeight = size.z * scale; // Z becomes Y after rotation
        const distance = 8;
        camera.position.set(distance, distance * 0.7, distance);
        controls.target.set(0, scaledHeight / 2, 0);
        controls.update();
      })
      .catch(error => {
        console.error('Error loading STL:', error);
      });
  }, [stlUrl]);

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
