/**
 * VcMoldCreator - Main Application
 * 
 * A tool for analyzing 3D models for two-piece mold manufacturing.
 * Computes optimal parting directions and visualizes surface visibility.
 */

<<<<<<< Updated upstream
import { useState, useCallback } from 'react';
import './App.css';
import ThreeViewer from './components/ThreeViewer';
import FileUpload from './components/FileUpload';
import type { VisibilityPaintData } from './utils/partingDirection';
import type { InflatedHullResult, ManifoldValidationResult, CsgSubtractionResult } from './utils/inflatedBoundingVolume';
import type { MeshRepairResult, MeshDiagnostics } from './utils/meshRepairManifold';
import type { TetMeshData, TetrahedralizeProgress, TetMeshVisualization } from './utils/tetrahedralViewer';
=======
import React, { useState, useCallback, useRef } from 'react';
import * as THREE from 'three';
import './App.css';
import ThreeViewer, { type GridVisualizationMode, type TetraVisualizationMode } from './components/ThreeViewer';
import type { VisibilityPaintData } from './utils/partingDirection';
import type { InflatedHullResult, ManifoldValidationResult, CsgSubtractionResult } from './utils/inflatedBoundingVolume';
import type { MeshRepairResult, MeshDiagnostics } from './utils/meshRepairManifold';
import type { VolumetricGridResult, DistanceFieldType } from './utils/volumetricGrid';
import type { MoldHalfClassificationResult } from './utils/moldHalfClassification';
import type { TetrahedralizationProgress } from './utils/tetrahedralization';
import type { TetraPartingSurfaceResult } from './utils/tetraPartingSurface';
import type { TetrahedralizationResult } from './utils/tetrahedralization';
import { loadTetrahedralMesh } from './utils/tetrahedralization';
>>>>>>> Stashed changes

// ============================================================================
// COMPONENT
// ============================================================================

<<<<<<< Updated upstream
=======
type Step = 'import' | 'parting' | 'hull' | 'cavity' | 'mold-halves' | 'tetra' | 'parting-surface';

interface StepInfo {
  id: Step;
  icon: string;
  title: string;
  description: string;
}

const STEPS: StepInfo[] = [
  { id: 'import', icon: '📁', title: 'Import STL', description: 'Load a 3D model file in STL format for mold analysis' },
  { id: 'parting', icon: '🔀', title: 'Parting Direction', description: 'Compute optimal parting directions for mold separation' },
  { id: 'hull', icon: '📦', title: 'Bounding Hull', description: 'Generate inflated convex hull around the mesh' },
  { id: 'cavity', icon: '✂️', title: 'Mold Cavity', description: 'Subtract base mesh from hull to create cavity' },
  { id: 'mold-halves', icon: '🎨', title: 'Mold Halves', description: 'Classify cavity surface into H₁ and H₂ mold halves' },
  { id: 'tetra', icon: '🔷', title: 'Tetrahedralize', description: 'Generate tetrahedral mesh of mold cavity via fTetWild' },
  { id: 'parting-surface', icon: '✂️', title: 'Parting Surface', description: 'Compute escape labeling via multi-source Dijkstra to define parting surface' },
];

>>>>>>> Stashed changes
interface HullStats {
  vertexCount: number;
  faceCount: number;
  manifoldValidation: ManifoldValidationResult;
}

interface CsgStats {
  vertexCount: number;
  faceCount: number;
  manifoldValidation: ManifoldValidationResult;
}

interface TetStats {
  vertexCount: number;
  tetrahedraCount: number;
}

interface TetrahedralizationStats {
  numVertices: number;
  numTetrahedra: number;
  inputVertices: number;
  inputFaces: number;
  computeTimeMs: number;
}

function App() {
  // State
  const [stlUrl, setStlUrl] = useState<string>();
  const [showPartingDirections, setShowPartingDirections] = useState(false);
  const [showD1Paint, setShowD1Paint] = useState(false);
  const [showD2Paint, setShowD2Paint] = useState(false);
  const [meshLoaded, setMeshLoaded] = useState(false);
  const [visibilityDataReady, setVisibilityDataReady] = useState(false);
  const [showInflatedHull, setShowInflatedHull] = useState(false);
  const [inflationOffset, setInflationOffset] = useState(0.5);
  const [hullStats, setHullStats] = useState<HullStats | null>(null);
  const [showCsgResult, setShowCsgResult] = useState(false);
  const [csgStats, setCsgStats] = useState<CsgStats | null>(null);
  const [hideOriginalMesh, setHideOriginalMesh] = useState(false);
  const [hideHull, setHideHull] = useState(false);
  const [meshRepairResult, setMeshRepairResult] = useState<{
    diagnostics: MeshDiagnostics;
    wasRepaired: boolean;
    repairMethod: string;
  } | null>(null);
<<<<<<< Updated upstream
  
  // Tetrahedralization state
  const [showTetMesh, setShowTetMesh] = useState(false);
  const [tetStats, setTetStats] = useState<TetStats | null>(null);
  const [tetProgress, setTetProgress] = useState<TetrahedralizeProgress | null>(null);
  const [tetError, setTetError] = useState<string | null>(null);
  const [isTetrahedralizing, setIsTetrahedralizing] = useState(false);
  const [hideCsgMesh, setHideCsgMesh] = useState(false);
  const [tetVisualization, setTetVisualization] = useState<TetMeshVisualization | null>(null);
  
  // Section plane state
  const [sectionPlaneEnabled, setSectionPlaneEnabled] = useState(false);
  const [sectionPlanePosition, setSectionPlanePosition] = useState(0.5);
=======
  const [showVolumetricGrid, setShowVolumetricGrid] = useState(false);
  const [gridResolution, setGridResolution] = useState(64);
  const [gridVisualizationMode, setGridVisualizationMode] = useState<GridVisualizationMode>('points');
  const [volumetricGridStats, setVolumetricGridStats] = useState<VolumetricGridStats | null>(null);
  const [useGPUGrid, setUseGPUGrid] = useState(true);
  const [hideVoxelGrid, setHideVoxelGrid] = useState(false);
  const [showRLine, setShowRLine] = useState(true);
  const [distanceFieldType, setDistanceFieldType] = useState<DistanceFieldType>('part');
  
  // Tetrahedralization state (replaces voxel for main workflow)
  const [showTetrahedralization, setShowTetrahedralization] = useState(false);
  const [tetraEdgeLengthFac, setTetraEdgeLengthFac] = useState(0.0065);
  const [tetraVisualizationMode, setTetraVisualizationMode] = useState<TetraVisualizationMode>('points');
  const [tetraStats, setTetraStats] = useState<TetrahedralizationStats | null>(null);
  const [hideTetrahedralization, setHideTetrahedralization] = useState(false);
  const [lastComputedTetraEdgeLengthFac, setLastComputedTetraEdgeLengthFac] = useState<number | null>(null);
  const [tetraProgress, setTetraProgress] = useState<TetrahedralizationProgress | null>(null);
  const [preloadedTetraResult, setPreloadedTetraResult] = useState<TetrahedralizationResult | null>(null);
  
  // Mesh info from loaded model (bounding box diagonal, size, scale)
  const [meshInfo, setMeshInfo] = useState<{ diagonal: number; size: { x: number; y: number; z: number }; scale: number } | null>(null);
  
  const [showMoldHalfClassification, setShowMoldHalfClassification] = useState(false);
  const [boundaryZoneThreshold, setBoundaryZoneThreshold] = useState(0.15); // 15% of bbox diagonal
  const [moldHalfStats, setMoldHalfStats] = useState<{
    h1Count: number;
    h2Count: number;
    h1Percentage: number;
    h2Percentage: number;
    totalTriangles: number;
    outerBoundaryCount: number;
    innerBoundaryCount: number;
  } | null>(null);

  const [showPartingSurface, setShowPartingSurface] = useState(false);
  const [tetraPartingSurfaceStats, setTetraPartingSurfaceStats] = useState<{
    h1VertexCount: number;
    h2VertexCount: number;
    unlabeledCount: number;
    partingEdgeCount: number;
    computeTimeMs: number;
    h1Percentage: number;
    h2Percentage: number;
  } | null>(null);
  // Debug visualization modes for tetrahedral parting surface
  const [partingSurfaceDebugMode, setPartingSurfaceDebugMode] = useState<
    'none' | 'seed-boundary' | 'labeled-seeds' | 'seed-boundary-labeled' | 'all-vertices'
  >('seed-boundary'); // Using seed-boundary for validation

  // Track parameters used for last computation (to detect changes)
  const [lastComputedInflationOffset, setLastComputedInflationOffset] = useState<number | null>(null);
  const [lastComputedGridResolution, setLastComputedGridResolution] = useState<number | null>(null);

  // Helper to clear downstream steps (clears steps AFTER the given step, not the step itself)
  const clearFromStep = useCallback((step: Step) => {
    switch (step) {
      case 'parting':
        // Clear hull and beyond (downstream from parting)
        setShowInflatedHull(false);
        setHullStats(null);
        setLastComputedInflationOffset(null);
        setShowCsgResult(false);
        setCsgStats(null);
        setShowMoldHalfClassification(false);
        setMoldHalfStats(null);
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        setShowTetrahedralization(false);
        setTetraStats(null);
        setLastComputedTetraEdgeLengthFac(null);
        setPreloadedTetraResult(null);
        setShowPartingSurface(false);
        setTetraPartingSurfaceStats(null);
        break;
      case 'hull':
        // Clear cavity and beyond (downstream from hull)
        setShowCsgResult(false);
        setCsgStats(null);
        setShowMoldHalfClassification(false);
        setMoldHalfStats(null);
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        setShowTetrahedralization(false);
        setTetraStats(null);
        setLastComputedTetraEdgeLengthFac(null);
        setPreloadedTetraResult(null);
        setShowPartingSurface(false);
        setTetraPartingSurfaceStats(null);
        break;
      case 'cavity':
        // Clear mold-halves and beyond (downstream from cavity)
        setShowMoldHalfClassification(false);
        setMoldHalfStats(null);
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        setShowTetrahedralization(false);
        setTetraStats(null);
        setLastComputedTetraEdgeLengthFac(null);
        setPreloadedTetraResult(null);
        setShowPartingSurface(false);
        setTetraPartingSurfaceStats(null);
        break;
      case 'mold-halves':
        // Clear tetra and parting-surface (downstream from mold-halves)
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        setShowTetrahedralization(false);
        setTetraStats(null);
        setLastComputedTetraEdgeLengthFac(null);
        setPreloadedTetraResult(null);
        setShowPartingSurface(false);
        setTetraPartingSurfaceStats(null);
        break;
      case 'tetra':
        // Clear parting-surface (downstream from tetra)
        setShowPartingSurface(false);
        setTetraPartingSurfaceStats(null);
        break;
      case 'parting-surface':
        // No downstream steps
        break;
    }
  }, []);

  // Handle parameter changes - clear downstream when parameters change
  const handleInflationOffsetChange = useCallback((value: number) => {
    setInflationOffset(value);
    // If hull was computed with different offset, clear hull and downstream
    if (hullStats && lastComputedInflationOffset !== null && value !== lastComputedInflationOffset) {
      clearFromStep('hull');
    }
  }, [hullStats, lastComputedInflationOffset, clearFromStep]);

  const handleGridResolutionChange = useCallback((value: number) => {
    setGridResolution(value);
    // If grid was computed with different resolution, clear grid
    if (volumetricGridStats && lastComputedGridResolution !== null && value !== lastComputedGridResolution) {
      clearFromStep('tetra');
    }
  }, [volumetricGridStats, lastComputedGridResolution, clearFromStep]);
>>>>>>> Stashed changes

  const handleTetraEdgeLengthFacChange = useCallback((value: number) => {
    setTetraEdgeLengthFac(value);
    // If tetra was computed with different edge length, clear tetra and downstream
    if (tetraStats && lastComputedTetraEdgeLengthFac !== null && value !== lastComputedTetraEdgeLengthFac) {
      clearFromStep('tetra');
    }
  }, [tetraStats, lastComputedTetraEdgeLengthFac, clearFromStep]);

  // Handlers
  const handleFileLoad = useCallback((url: string, _fileName: string) => {
    if (stlUrl) URL.revokeObjectURL(stlUrl);
    setStlUrl(url);
    setShowPartingDirections(false);
    setShowD1Paint(false);
    setShowD2Paint(false);
    setMeshLoaded(false);
    setVisibilityDataReady(false);
    setShowInflatedHull(false);
    setHullStats(null);
    setShowCsgResult(false);
    setCsgStats(null);
    setHideOriginalMesh(false);
    setHideHull(false);
    setMeshRepairResult(null);
<<<<<<< Updated upstream
    // Reset tetrahedralization state
    setShowTetMesh(false);
    setTetStats(null);
    setTetProgress(null);
    setTetError(null);
    setIsTetrahedralizing(false);
    setHideCsgMesh(false);
    setTetVisualization(null);
    // Reset section plane
    setSectionPlaneEnabled(false);
    setSectionPlanePosition(0.5);
=======
    setShowVolumetricGrid(false);
    setVolumetricGridStats(null);
    setShowMoldHalfClassification(false);
    setMoldHalfStats(null);
    setShowPartingSurface(false);
    setTetraPartingSurfaceStats(null);
    // Reset tetrahedralization state
    setShowTetrahedralization(false);
    setTetraStats(null);
    setPreloadedTetraResult(null);
    setLastComputedTetraEdgeLengthFac(null);
    setTetraProgress(null);
    tetraResultRef.current = null;
    // Reset mesh info when loading new file
    setMeshInfo(null);
    // Move to parting step after loading
    setActiveStep('parting');
>>>>>>> Stashed changes
  }, [stlUrl]);

  const handleMeshLoaded = useCallback((_mesh: THREE.Mesh, info: { diagonal: number; size: { x: number; y: number; z: number }; scale: number }) => {
    setMeshLoaded(true);
    setMeshInfo(info);
    // Auto-calculate default edge length factor based on bounding box diagonal
    // Target: 0.0065 * diagonal gives the actual edge length
    const autoEdgeLengthFac = 0.0065;
    setTetraEdgeLengthFac(autoEdgeLengthFac);
    console.log(`Mesh loaded - Diagonal: ${info.diagonal.toFixed(3)}, Auto edge length factor: ${autoEdgeLengthFac}`);
  }, []);

  const handleMeshRepaired = useCallback(
    (result: MeshRepairResult) => {
      setMeshRepairResult({
        diagnostics: result.diagnostics,
        wasRepaired: result.wasRepaired,
        repairMethod: result.repairMethod,
      });
    },
    []
  );

  const handleVisibilityDataReady = useCallback(
    (data: VisibilityPaintData | null) => setVisibilityDataReady(data !== null),
    []
  );

  const handleInflatedHullReady = useCallback(
    (result: InflatedHullResult | null) => {
      setHullStats(result ? {
        vertexCount: result.vertexCount,
        faceCount: result.faceCount,
        manifoldValidation: result.manifoldValidation,
      } : null);
      // Reset CSG when hull changes
      if (!result) {
        setShowCsgResult(false);
        setCsgStats(null);
      }
    },
    []
  );

  const handleCsgResultReady = useCallback(
    (result: CsgSubtractionResult | null) => {
      setCsgStats(result ? {
        vertexCount: result.vertexCount,
        faceCount: result.faceCount,
        manifoldValidation: result.manifoldValidation,
      } : null);
    },
    []
  );

<<<<<<< Updated upstream
  const togglePartingDirections = useCallback(() => {
    setShowPartingDirections(prev => {
      if (prev) {
        setShowD1Paint(false);
        setShowD2Paint(false);
        setVisibilityDataReady(false);
=======
  const handleVolumetricGridReady = useCallback(
    (result: VolumetricGridResult | null) => {
      setVolumetricGridStats(result ? {
        resolution: { x: result.resolution.x, y: result.resolution.y, z: result.resolution.z },
        totalCellCount: result.totalCellCount,
        moldVolumeCellCount: result.moldVolumeCellCount,
        moldVolume: result.stats.moldVolume,
        fillRatio: result.stats.fillRatio,
        computeTimeMs: result.stats.computeTimeMs,
      } : null);
      // Track the parameters used for this computation
      if (result) {
        setLastComputedGridResolution(gridResolution);
      }
    },
    [gridResolution]
  );

  const handleTetrahedralizationReady = useCallback(
    (result: TetrahedralizationResult | null) => {
      setTetraProgress(null); // Clear progress when done
      tetraResultRef.current = result; // Store for save functionality
      setTetraStats(result ? {
        numVertices: result.numVertices,
        numTetrahedra: result.numTetrahedra,
        inputVertices: result.inputStats.numVertices,
        inputFaces: result.inputStats.numFaces,
        computeTimeMs: result.computeTimeMs,
      } : null);
      // Track the parameters used for this computation
      if (result) {
        setLastComputedTetraEdgeLengthFac(tetraEdgeLengthFac);
      }
    },
    [tetraEdgeLengthFac]
  );

  const handleTetrahedralizationProgress = useCallback(
    (progress: TetrahedralizationProgress) => {
      setTetraProgress(progress);
    },
    []
  );

  const handleMoldHalfClassificationReady = useCallback(
    (result: MoldHalfClassificationResult | null) => {
      setMoldHalfStats(result ? {
        h1Count: result.h1Triangles.size,
        h2Count: result.h2Triangles.size,
        h1Percentage: result.outerBoundaryCount > 0 ? result.h1Triangles.size / result.outerBoundaryCount * 100 : 0,
        h2Percentage: result.outerBoundaryCount > 0 ? result.h2Triangles.size / result.outerBoundaryCount * 100 : 0,
        totalTriangles: result.totalTriangles,
        outerBoundaryCount: result.outerBoundaryCount,
        innerBoundaryCount: result.innerBoundaryTriangles.size,
      } : null);
    },
    []
  );

  const handleTetraPartingSurfaceReady = useCallback(
    (result: TetraPartingSurfaceResult | null) => {
      if (result) {
        const total = result.h1Count + result.h2Count + result.unlabeledCount;
        setTetraPartingSurfaceStats({
          h1VertexCount: result.h1Count,
          h2VertexCount: result.h2Count,
          unlabeledCount: result.unlabeledCount,
          partingEdgeCount: result.partingEdgeCount,
          computeTimeMs: result.computeTimeMs,
          h1Percentage: total > 0 ? result.h1Count / total * 100 : 0,
          h2Percentage: total > 0 ? result.h2Count / total * 100 : 0,
        });
      } else {
        setTetraPartingSurfaceStats(null);
      }
    },
    []
  );

  // Active step for context menu
  const [activeStep, setActiveStep] = useState<Step>('import');
  const [isCalculating, setIsCalculating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loadedFileName, setLoadedFileName] = useState<string | null>(null);

  // Save tetrahedral mesh to file (JSON format)
  const handleSaveTetraMesh = useCallback(() => {
    if (!tetraResultRef.current) {
      console.warn('No tetrahedral mesh to save');
      return;
    }
    const result = tetraResultRef.current;
    const data = {
      version: 1,
      numVertices: result.numVertices,
      numTetrahedra: result.numTetrahedra,
      vertices: Array.from(result.vertices),
      tetrahedra: Array.from(result.tetrahedra),
      inputStats: result.inputStats,
      computeTimeMs: result.computeTimeMs,
      savedAt: new Date().toISOString(),
    };
    const json = JSON.stringify(data);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const baseName = loadedFileName ? loadedFileName.replace(/\.[^/.]+$/, '') : 'mesh';
    const a = document.createElement('a');
    a.href = url;
    a.download = `${baseName}.tetra.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log(`Saved tetrahedral mesh: ${baseName}.tetra.json`);
  }, [loadedFileName]);

  // Load tetrahedral mesh from file
  const handleLoadTetraMesh = useCallback(async (file: File) => {
    try {
      console.log(`Loading tetrahedral mesh: ${file.name}`);
      const result = await loadTetrahedralMesh(file);
      console.log(`Loaded: ${result.numVertices} vertices, ${result.numTetrahedra} tetrahedra`);
      
      // Set the preloaded result - ThreeViewer will pick it up
      setPreloadedTetraResult(result);
      setShowTetrahedralization(true);
      
      // Update stats
      setTetraStats({
        numVertices: result.numVertices,
        numTetrahedra: result.numTetrahedra,
        inputVertices: result.inputStats.numVertices,
        inputFaces: result.inputStats.numFaces,
        computeTimeMs: result.computeTimeMs,
      });
      setLastComputedTetraEdgeLengthFac(tetraEdgeLengthFac);
    } catch (error) {
      console.error('Failed to load tetrahedral mesh:', error);
      alert(`Failed to load tetrahedral mesh: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [tetraEdgeLengthFac, loadedFileName]);

  // Determine which steps are available/completed
  // Returns 'needs-recalc' if parameters changed since last computation
  const getStepStatus = (step: Step): 'locked' | 'available' | 'completed' | 'needs-recalc' => {
    switch (step) {
      case 'import':
        if (meshLoaded) return 'completed';
        return 'available';
      case 'parting':
        if (visibilityDataReady) return 'completed';
        return meshLoaded ? 'available' : 'locked';
      case 'hull':
        if (hullStats) {
          // Check if parameters changed
          if (lastComputedInflationOffset !== null && inflationOffset !== lastComputedInflationOffset) {
            return 'needs-recalc';
          }
          return 'completed';
        }
        return visibilityDataReady ? 'available' : 'locked';
      case 'cavity':
        // Cavity depends on hull - if hull needs recalc, cavity is locked
        if (getStepStatus('hull') === 'needs-recalc') return 'locked';
        if (csgStats) return 'completed';
        return hullStats ? 'available' : 'locked';
      case 'mold-halves':
        // Mold halves depends on cavity
        if (getStepStatus('cavity') === 'locked') return 'locked';
        if (moldHalfStats) return 'completed';
        return csgStats ? 'available' : 'locked';
      case 'tetra':
        // Tetra depends on cavity
        if (getStepStatus('cavity') === 'locked') return 'locked';
        if (tetraStats) {
          // Check if parameters changed
          if (lastComputedTetraEdgeLengthFac !== null && tetraEdgeLengthFac !== lastComputedTetraEdgeLengthFac) {
            return 'needs-recalc';
          }
          return 'completed';
        }
        return csgStats ? 'available' : 'locked';
      case 'parting-surface':
        // Parting surface depends on tetra AND mold-halves
        if (getStepStatus('tetra') === 'locked' || getStepStatus('tetra') === 'needs-recalc') return 'locked';
        if (getStepStatus('mold-halves') === 'locked') return 'locked';
        if (tetraPartingSurfaceStats) {
          return 'completed';
        }
        return (tetraStats && moldHalfStats) ? 'available' : 'locked';
    }
  };

  // Handle calculate button for each step
  const handleCalculate = useCallback(() => {
    setIsCalculating(true);
    setProgress(0);
    
    // Simulate progress (actual progress comes from callbacks)
    const interval = setInterval(() => {
      setProgress(p => Math.min(p + 10, 90));
    }, 100);
    
    // Clear current step's results and downstream when recalculating
    switch (activeStep) {
      case 'parting':
        clearFromStep('parting');
        // Reset current step's state to trigger recalculation
        setVisibilityDataReady(false);
        setShowPartingDirections(false);
        // Use setTimeout to ensure state updates before triggering
        setTimeout(() => setShowPartingDirections(true), 0);
        break;
      case 'hull':
        clearFromStep('hull');
        // Reset current step's state to trigger recalculation
        setHullStats(null);
        setShowInflatedHull(false);
        setTimeout(() => setShowInflatedHull(true), 0);
        break;
      case 'cavity':
        clearFromStep('cavity');
        // Reset current step's state to trigger recalculation
        setCsgStats(null);
        setShowCsgResult(false);
        setTimeout(() => setShowCsgResult(true), 0);
        break;
      case 'mold-halves':
        clearFromStep('mold-halves');
        // Reset current step's state to trigger recalculation
        setMoldHalfStats(null);
        setShowMoldHalfClassification(false);
        setTimeout(() => setShowMoldHalfClassification(true), 0);
        break;
      case 'tetra':
        clearFromStep('tetra');
        // Reset current step's state to trigger recalculation
        setTetraStats(null);
        setShowTetrahedralization(false);
        setTimeout(() => setShowTetrahedralization(true), 0);
        break;
      case 'parting-surface':
        clearFromStep('parting-surface');
        // Reset current step's state to trigger recalculation
        setTetraPartingSurfaceStats(null);
        setShowPartingSurface(false);
        setTimeout(() => setShowPartingSurface(true), 0);
        break;
    }
    
    // Complete after a delay (will be set properly when data is ready)
    setTimeout(() => {
      clearInterval(interval);
      setProgress(100);
      setTimeout(() => {
        setIsCalculating(false);
        setProgress(0);
      }, 300);
    }, 1500);
  }, [activeStep, clearFromStep]);

  // ============================================================================
  // RENDER CONTEXT MENU CONTENT
  // ============================================================================

  // File input ref for import step
  const fileInputRef = useRef<HTMLInputElement>(null);
  // File input ref for loading tetrahedral mesh
  const tetraFileInputRef = useRef<HTMLInputElement>(null);
  // Ref to store current tetrahedral result for saving
  const tetraResultRef = useRef<TetrahedralizationResult | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.stl')) {
        alert('Please upload an STL file');
        return;
>>>>>>> Stashed changes
      }
      return !prev;
    });
  }, []);

  // Tetrahedralization handlers
  const handleTetrahedralize = useCallback(() => {
    if (isTetrahedralizing) return;
    setShowTetMesh(prev => !prev);
    if (!showTetMesh) {
      // Starting tetrahedralization
      setTetError(null);
      setTetProgress(null);
      setTetStats(null);
      setTetVisualization(null);
      setIsTetrahedralizing(true);
    }
  }, [isTetrahedralizing, showTetMesh]);

  const handleTetProgress = useCallback((progress: TetrahedralizeProgress) => {
    setTetProgress(progress);
  }, []);

  const handleTetComplete = useCallback((data: TetMeshData) => {
    setIsTetrahedralizing(false);
    setTetStats({
      vertexCount: data.vertices.length / 3,
      tetrahedraCount: data.tetrahedra.length / 4,
    });
    setTetProgress(null);
  }, []);

  const handleTetError = useCallback((error: string) => {
    setIsTetrahedralizing(false);
    setTetError(error);
    setShowTetMesh(false);
    setTetProgress(null);
  }, []);

<<<<<<< Updated upstream
  const handleTetVisualizationReady = useCallback((visualization: TetMeshVisualization | null) => {
    setTetVisualization(visualization);
  }, []);
=======
        {status === 'locked' && (
          <div style={styles.lockedMessage}>
            ⚠️ Complete previous steps first
          </div>
        )}

        {/* Import step - file upload UI */}
        {activeStep === 'import' && (
          <div style={styles.optionsSection}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".stl"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
            
            <div
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleFileDrop}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
              style={{
                padding: '24px 16px',
                backgroundColor: isDragging ? 'rgba(0, 170, 255, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                border: `2px dashed ${isDragging ? '#00aaff' : '#444'}`,
                borderRadius: '8px',
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'all 0.2s',
              }}
            >
              {loadedFileName ? (
                <>
                  <div style={{ fontSize: '24px', marginBottom: '8px' }}>✅</div>
                  <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '4px' }}>Loaded:</div>
                  <div style={{ fontSize: '11px', opacity: 0.8, wordBreak: 'break-all' }}>{loadedFileName}</div>
                  <div style={{ fontSize: '10px', opacity: 0.5, marginTop: '8px' }}>Click to replace</div>
                </>
              ) : (
                <>
                  <div style={{ fontSize: '32px', marginBottom: '8px' }}>📁</div>
                  <div style={{ fontSize: '13px', fontWeight: 'bold' }}>Upload STL File</div>
                  <div style={{ fontSize: '11px', opacity: 0.6, marginTop: '4px' }}>Click or drag & drop</div>
                </>
              )}
            </div>

            {/* Mesh info after loading */}
            {meshLoaded && meshRepairResult && (
              <div style={{ ...styles.statsBox, marginTop: '16px' }}>
                <div style={{ fontWeight: 'bold', marginBottom: '6px' }}>
                  {meshRepairResult.diagnostics.isManifold ? '✅ Mesh Valid' : '⚠️ Mesh Issues'}
                </div>
                <div>Vertices: {meshRepairResult.diagnostics.vertexCount.toLocaleString()}</div>
                <div>Faces: {meshRepairResult.diagnostics.faceCount.toLocaleString()}</div>
                {meshRepairResult.diagnostics.genus >= 0 && (
                  <div>Genus: {meshRepairResult.diagnostics.genus}</div>
                )}
                {meshRepairResult.wasRepaired && (
                  <div style={{ color: '#aaf', marginTop: '4px' }}>Repaired: {meshRepairResult.repairMethod}</div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Step-specific options */}
        {activeStep === 'parting' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            {visibilityDataReady && (
              <div style={styles.statsBox}>
                <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>✅ Parting Directions Computed</div>
                <div style={styles.legend}>
                  <div>🟢 Green arrow: Primary direction (D1)</div>
                  <div>🟠 Orange arrow: Secondary direction (D2)</div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeStep === 'hull' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            <div style={styles.optionLabel}>Inflation Offset (mm):</div>
            <input
              type="number"
              min="0"
              step="0.1"
              defaultValue={inflationOffset}
              key={`inflation-${lastComputedInflationOffset}`}
              onBlur={(e) => {
                const val = e.target.value;
                const num = parseFloat(val) || 0;
                const clamped = Math.max(0, num);
                e.target.value = String(clamped);
                handleInflationOffsetChange(clamped);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  (e.target as HTMLInputElement).blur();
                }
              }}
              style={styles.input}
            />

            {hullStats && (
              <div style={styles.statsBox}>
                <div>Vertices: {hullStats.vertexCount}</div>
                <div>Faces: {hullStats.faceCount}</div>
                <div style={{ color: hullStats.manifoldValidation.isManifold ? '#0f0' : '#f90' }}>
                  {hullStats.manifoldValidation.isManifold ? '✅ Valid Manifold' : '⚠️ Not Manifold'}
                </div>
              </div>
            )}
          </div>
        )}

        {activeStep === 'cavity' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            {csgStats && (
              <div style={styles.statsBox}>
                <div>Cavity vertices: {csgStats.vertexCount}</div>
                <div>Cavity faces: {csgStats.faceCount}</div>
                <div style={{ color: csgStats.manifoldValidation.isManifold ? '#0f0' : '#f90' }}>
                  {csgStats.manifoldValidation.isManifold ? '✅ Valid Manifold' : '⚠️ Not Manifold'}
                </div>
              </div>
            )}
          </div>
        )}

        {activeStep === 'mold-halves' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            <div style={styles.optionLabel}>Boundary Zone Width: {(boundaryZoneThreshold * 100).toFixed(0)}% of diagonal</div>
            <input
              type="range"
              min="0"
              max="30"
              step="1"
              value={boundaryZoneThreshold * 100}
              onChange={(e) => setBoundaryZoneThreshold(parseFloat(e.target.value) / 100)}
              style={{ width: '100%', marginBottom: '12px' }}
            />
            {/* Mold Half Stats */}
            {moldHalfStats && (
              <div style={styles.statsBox}>
                <div style={{ marginBottom: '4px', fontSize: '0.85em', color: '#aaa' }}>
                  Outer boundary (hull):
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#0f0' }}>H₁ (Green):</span>
                  <span>{moldHalfStats.h1Count} ({moldHalfStats.h1Percentage.toFixed(1)}%)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#f60' }}>H₂ (Orange):</span>
                  <span>{moldHalfStats.h2Count} ({moldHalfStats.h2Percentage.toFixed(1)}%)</span>
                </div>
                <div style={{ marginTop: '6px', paddingTop: '4px', borderTop: '1px solid #444', fontSize: '0.85em', color: '#888' }}>
                  <div>Outer: {moldHalfStats.outerBoundaryCount} tris</div>
                  <div>Inner (part): {moldHalfStats.innerBoundaryCount} tris</div>
                </div>
              </div>
            )}
            
            {!moldHalfStats && csgStats && (
              <div style={{ color: '#888', fontSize: '0.9em' }}>
                Click "Calculate" to classify the mold cavity into H₁ and H₂ halves based on parting directions.
              </div>
            )}
          </div>
        )}

        {activeStep === 'tetra' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            <div style={styles.optionLabel}>Edge Length Factor:</div>
            <input
              type="number"
              min="0.001"
              max="0.5"
              step="0.001"
              value={tetraEdgeLengthFac}
              onChange={(e) => {
                const val = e.target.value;
                const num = parseFloat(val);
                if (!isNaN(num)) {
                  const clamped = Math.max(0.001, Math.min(0.5, num));
                  handleTetraEdgeLengthFacChange(clamped);
                }
              }}
              style={styles.input}
            />
            {meshInfo && (
              <div style={{ fontSize: '0.8em', color: '#00ff88', marginTop: '4px' }}>
                Target edge length: {(tetraEdgeLengthFac * meshInfo.diagonal).toFixed(4)} units
                <br />
                <span style={{ color: '#888' }}>
                  (= {tetraEdgeLengthFac.toFixed(4)} × {meshInfo.diagonal.toFixed(3)} diagonal)
                </span>
              </div>
            )}
            <div style={{ fontSize: '0.8em', color: '#888', marginTop: '4px' }}>
              Smaller = denser mesh (0.001-0.5)
            </div>

            <div style={{ ...styles.optionLabel, marginTop: '12px' }}>Visualization:</div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <button
                onClick={() => setTetraVisualizationMode('points')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: tetraVisualizationMode === 'points' ? '#00ff88' : '#333',
                  borderColor: tetraVisualizationMode === 'points' ? '#00ff88' : '#555',
                  color: tetraVisualizationMode === 'points' ? '#000' : '#fff',
                }}
              >
                Points
              </button>
              <button
                onClick={() => setTetraVisualizationMode('wireframe')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: tetraVisualizationMode === 'wireframe' ? '#00ff88' : '#333',
                  borderColor: tetraVisualizationMode === 'wireframe' ? '#00ff88' : '#555',
                  color: tetraVisualizationMode === 'wireframe' ? '#000' : '#fff',
                }}
              >
                Wireframe
              </button>
            </div>

            <label style={styles.checkbox}>
              <input
                type="checkbox"
                checked={!hideTetrahedralization}
                onChange={(e) => setHideTetrahedralization(!e.target.checked)}
              />
              Show Tetrahedralization
            </label>

            {/* Progress indicator during computation */}
            {tetraProgress && !tetraProgress.complete && (
              <div style={{ 
                marginTop: '12px', 
                padding: '12px', 
                backgroundColor: '#1a2a1a', 
                borderRadius: '6px',
                border: '1px solid #00ff88'
              }}>
                {/* Header with status and timing */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ color: '#00ff88', fontWeight: 'bold' }}>
                    {tetraProgress.status === 'loading' && '📂'}
                    {tetraProgress.status === 'preprocessing' && '⚙️'}
                    {tetraProgress.status === 'tetrahedralizing' && '🔷'}
                    {tetraProgress.status === 'postprocessing' && '📦'}
                    {tetraProgress.status === 'downloading' && '⬇️'}
                    {tetraProgress.status === 'visualizing' && '🎨'}
                    {' '}{tetraProgress.status.charAt(0).toUpperCase() + tetraProgress.status.slice(1)}
                  </span>
                  <span style={{ color: '#888', fontSize: '0.9em' }}>
                    {tetraProgress.elapsed_seconds.toFixed(1)}s
                    {tetraProgress.estimated_remaining !== null && tetraProgress.estimated_remaining > 0 && (
                      <span style={{ color: '#666' }}> (~{tetraProgress.estimated_remaining.toFixed(0)}s left)</span>
                    )}
                  </span>
                </div>
                
                {/* Progress bar with animation for long-running stages */}
                <div style={{ 
                  height: '6px', 
                  backgroundColor: '#333', 
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginBottom: '8px',
                  position: 'relative'
                }}>
                  <div style={{ 
                    height: '100%', 
                    width: `${tetraProgress.progress}%`,
                    backgroundColor: tetraProgress.status === 'tetrahedralizing' ? '#00ff88' : 
                                     tetraProgress.status === 'downloading' ? '#88aaff' :
                                     tetraProgress.status === 'visualizing' ? '#ffaa00' : '#88ff00',
                    transition: 'width 0.15s ease-out',
                    boxShadow: '0 0 8px rgba(0, 255, 136, 0.5)'
                  }} />
                  {/* Animated pulse overlay for tetrahedralizing to show activity */}
                  {tetraProgress.status === 'tetrahedralizing' && (
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
                      animation: 'shimmer 1.5s infinite',
                    }} />
                  )}
                </div>
                
                {/* Percentage and substep */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.8em', color: '#00ff88', fontFamily: 'monospace' }}>
                    {tetraProgress.progress}%
                  </span>
                  <span style={{ fontSize: '0.8em', color: '#aaa', textAlign: 'right', flex: 1, marginLeft: '12px' }}>
                    {tetraProgress.substep || tetraProgress.message}
                  </span>
                </div>
                
                {/* Log messages - show last 5 for better visibility */}
                {tetraProgress.logs && tetraProgress.logs.length > 0 && (
                  <div style={{ 
                    marginTop: '8px', 
                    paddingTop: '8px', 
                    borderTop: '1px solid #333',
                    maxHeight: '80px',
                    overflowY: 'auto',
                    fontSize: '0.7em',
                    fontFamily: 'monospace',
                    color: '#666',
                    lineHeight: '1.4'
                  }}>
                    {tetraProgress.logs.slice(-5).map((log, i) => (
                      <div key={i} style={{ color: i === tetraProgress.logs!.slice(-5).length - 1 ? '#88ff88' : '#555' }}>
                        [{log.time.toFixed(1)}s] {log.message}
                      </div>
                    ))}
                  </div>
                )}
                
                {/* Input stats if available */}
                {tetraProgress.input_stats && (
                  <div style={{ 
                    marginTop: '8px', 
                    paddingTop: '8px', 
                    borderTop: '1px solid #333',
                    fontSize: '0.75em', 
                    color: '#666' 
                  }}>
                    Input: {tetraProgress.input_stats.num_vertices.toLocaleString()} vertices, {tetraProgress.input_stats.num_faces.toLocaleString()} faces
                  </div>
                )}
              </div>
            )}

            {tetraStats && (
              <div style={styles.statsBox}>
                <div>Input: {tetraStats.inputVertices.toLocaleString()} verts, {tetraStats.inputFaces.toLocaleString()} faces</div>
                <div>Output: {tetraStats.numVertices.toLocaleString()} verts</div>
                <div>Tetrahedra: {tetraStats.numTetrahedra.toLocaleString()}</div>
                <div>Compute: {tetraStats.computeTimeMs.toFixed(0)} ms</div>
              </div>
            )}

            {/* Save/Load Tetrahedral Mesh Buttons */}
            <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                onClick={handleSaveTetraMesh}
                disabled={!tetraStats}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: tetraStats ? '#226622' : '#333',
                  borderColor: tetraStats ? '#44aa44' : '#555',
                  color: tetraStats ? '#88ff88' : '#666',
                  cursor: tetraStats ? 'pointer' : 'not-allowed',
                  flex: 1,
                  minWidth: '80px',
                }}
              >
                💾 Save Mesh
              </button>
              <button
                onClick={() => tetraFileInputRef.current?.click()}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: '#222266',
                  borderColor: '#4444aa',
                  color: '#8888ff',
                  flex: 1,
                  minWidth: '80px',
                }}
              >
                📂 Load Mesh
              </button>
              <input
                type="file"
                ref={tetraFileInputRef}
                accept=".json,.tetra.json"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    handleLoadTetraMesh(file);
                    e.target.value = ''; // Reset to allow loading same file again
                  }
                }}
              />
            </div>
            {tetraStats && (
              <div style={{ fontSize: '0.75em', color: '#666', marginTop: '4px' }}>
                Save to skip recomputation later
              </div>
            )}
            
            {!tetraStats && !tetraProgress && csgStats && (
              <div style={{ color: '#888', fontSize: '0.9em' }}>
                Click "Calculate" to tetrahedralize the mold cavity via fTetWild backend.
                <div style={{ marginTop: '8px', padding: '8px', backgroundColor: '#332200', borderRadius: '4px', fontSize: '0.85em' }}>
                  ⚠️ Backend server must be running:<br/>
                  <code style={{ color: '#ffaa00' }}>python backend/app.py</code>
                </div>
              </div>
            )}
          </div>
        )}

        {activeStep === 'parting-surface' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            {tetraPartingSurfaceStats && (
              <>
                {/* Debug Visualization Mode */}
                <div style={{ marginTop: '8px' }}>
                  <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '4px' }}>
                    Debug View:
                  </label>
                  <select
                    value={partingSurfaceDebugMode}
                    onChange={(e) => setPartingSurfaceDebugMode(e.target.value as 'none' | 'seed-boundary' | 'labeled-seeds' | 'seed-boundary-labeled' | 'all-vertices')}
                    style={{
                      width: '100%',
                      padding: '4px 8px',
                      backgroundColor: '#333',
                      color: '#fff',
                      border: '1px solid #555',
                      borderRadius: '4px',
                      fontSize: '0.9em',
                    }}
                  >
                    <option value="none">Normal View (Parting Edges)</option>
                    <option value="seed-boundary">Step 1: Seeds + Boundary (Unlabeled)</option>
                    <option value="labeled-seeds">Step 2: Seeds After Labeling</option>
                    <option value="seed-boundary-labeled">Step 3: Seeds + Boundary (Labeled)</option>
                    <option value="all-vertices">Step 4: All Vertices (Final)</option>
                  </select>
                </div>

                <div style={styles.statsBox}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#0f0' }}>H₁ vertices:</span>
                    <span>{tetraPartingSurfaceStats.h1VertexCount.toLocaleString()} ({tetraPartingSurfaceStats.h1Percentage.toFixed(1)}%)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#f60' }}>H₂ vertices:</span>
                    <span>{tetraPartingSurfaceStats.h2VertexCount.toLocaleString()} ({tetraPartingSurfaceStats.h2Percentage.toFixed(1)}%)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#0ff' }}>Parting edges:</span>
                    <span>{tetraPartingSurfaceStats.partingEdgeCount.toLocaleString()}</span>
                  </div>
                  {tetraPartingSurfaceStats.unlabeledCount > 0 && (
                    <div style={{ color: '#f00' }}>
                      Unlabeled: {tetraPartingSurfaceStats.unlabeledCount}
                    </div>
                  )}
                  <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px solid #444', fontSize: '0.85em', color: '#888' }}>
                    Compute: {tetraPartingSurfaceStats.computeTimeMs.toFixed(0)} ms
                  </div>
                </div>
              </>
            )}

            {!tetraPartingSurfaceStats && tetraStats && moldHalfStats && (
              <div style={{ color: '#888', fontSize: '0.9em' }}>
                Click "Calculate" to compute parting surface via multi-source Dijkstra on tetrahedral edges.
              </div>
            )}
          </div>
        )}

        {/* Calculate Button and Progress Bar - not shown for import step */}
        {activeStep !== 'import' && status !== 'locked' && (
          <div style={styles.calculateSection}>
            <button
              onClick={handleCalculate}
              disabled={isCalculating}
              style={{
                ...styles.calculateButton,
                backgroundColor: status === 'completed' ? '#00aa00' : status === 'needs-recalc' ? '#ff8800' : isCalculating ? '#666' : '#00aaff',
                cursor: isCalculating ? 'default' : 'pointer',
              }}
            >
              {status === 'completed' ? '⟳ Recalculate' : status === 'needs-recalc' ? '⟳ Recalculate' : isCalculating ? 'Calculating...' : 'Calculate'}
            </button>
            
            {isCalculating && (
              <div style={styles.progressContainer}>
                <div style={{ ...styles.progressBar, width: `${progress}%` }} />
              </div>
            )}
          </div>
        )}

        {/* Display Options - shows toggles for all completed steps */}
        {meshLoaded && activeStep !== 'import' && (
          <div style={{ marginTop: '16px', borderTop: '1px solid #444', paddingTop: '12px' }}>
            <div style={styles.optionLabel}>Display Options:</div>
            
            {/* Original Mesh - always available when loaded */}
            <label style={styles.checkbox}>
              <input
                type="checkbox"
                checked={hideOriginalMesh}
                onChange={(e) => setHideOriginalMesh(e.target.checked)}
              />
              Hide Original Mesh
            </label>

            {/* Parting Direction visibility options */}
            {visibilityDataReady && (
              <>
                <label style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={showD1Paint}
                    onChange={(e) => setShowD1Paint(e.target.checked)}
                  />
                  Show D1 Visibility 🟢
                </label>
                <label style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={showD2Paint}
                    onChange={(e) => setShowD2Paint(e.target.checked)}
                  />
                  Show D2 Visibility 🟠
                </label>
              </>
            )}

            {/* Hull visibility - available after hull is computed */}
            {hullStats && (
              <label style={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={hideHull}
                  onChange={(e) => setHideHull(e.target.checked)}
                />
                Hide Hull 🟣
              </label>
            )}

            {/* Cavity visibility - available after cavity is computed */}
            {csgStats && (
              <label style={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={hideCavity}
                  onChange={(e) => setHideCavity(e.target.checked)}
                />
                Hide Cavity 🩵
              </label>
            )}

            {/* Voxel grid visibility - available after voxel is computed */}
            {volumetricGridStats && (
              <label style={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={hideVoxelGrid}
                  onChange={(e) => setHideVoxelGrid(e.target.checked)}
                />
                Hide Voxel Grid 🧊
              </label>
            )}

            {/* R Line visibility - available after voxel is computed */}
            {volumetricGridStats && (
              <label style={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={showRLine}
                  onChange={(e) => setShowRLine(e.target.checked)}
                />
                Show R Line 📏
              </label>
            )}

            {/* Voxel Grid Coloring - available after voxel is computed */}
            {volumetricGridStats && (
              <div style={{ marginTop: '8px' }}>
                <label style={{ fontSize: '12px', color: '#aaa', display: 'block', marginBottom: '4px' }}>
                  Voxel Grid Coloring:
                </label>
                <select
                  value={distanceFieldType}
                  onChange={(e) => setDistanceFieldType(e.target.value as DistanceFieldType)}
                  style={{
                    width: '100%',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    border: '1px solid #555',
                    backgroundColor: '#2a2a2a',
                    color: '#fff',
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  <option value="part">Part Distance (δᵢ)</option>
                  <option value="biased">Biased Distance (δᵢ + λw)</option>
                  <option value="weight">Weighting Factor (wt)</option>
                </select>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };
>>>>>>> Stashed changes

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0, position: 'relative' }}>
      {/* 3D Viewer */}
      <ThreeViewer
        stlUrl={stlUrl}
        showPartingDirections={showPartingDirections}
        showD1Paint={showD1Paint}
        showD2Paint={showD2Paint}
        showInflatedHull={showInflatedHull}
        inflationOffset={inflationOffset}
        showCsgResult={showCsgResult}
        hideOriginalMesh={hideOriginalMesh}
        hideHull={hideHull}
        hideCsgMesh={hideCsgMesh}
        showTetMesh={showTetMesh}
        sectionPlaneEnabled={sectionPlaneEnabled}
        sectionPlanePosition={sectionPlanePosition}
        onMeshLoaded={handleMeshLoaded}
        onMeshRepaired={handleMeshRepaired}
        onVisibilityDataReady={handleVisibilityDataReady}
        onInflatedHullReady={handleInflatedHullReady}
        onCsgResultReady={handleCsgResultReady}
        onTetProgress={handleTetProgress}
        onTetComplete={handleTetComplete}
        onTetError={handleTetError}
        onTetVisualizationReady={handleTetVisualizationReady}
      />

      {/* File Upload */}
      <FileUpload onFileLoad={handleFileLoad} />

      {/* Analysis Controls */}
      {meshLoaded && (
        <div style={styles.controlPanel}>
          <div style={styles.title}>🔧 Mold Analysis</div>
          
          {/* Mesh Health Status */}
          {meshRepairResult && (
            <div style={{ 
              marginBottom: '12px', 
              padding: '8px', 
              backgroundColor: meshRepairResult.diagnostics.isManifold 
                ? 'rgba(0, 255, 0, 0.15)' 
                : 'rgba(255, 165, 0, 0.15)',
              borderRadius: '4px',
              fontSize: '10px'
            }}>
              <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                {meshRepairResult.diagnostics.isManifold 
                  ? '✅ Mesh: Valid Manifold' 
                  : '⚠️ Mesh: Invalid'}
              </div>
              <div>Vertices: {meshRepairResult.diagnostics.vertexCount}</div>
              <div>Faces: {meshRepairResult.diagnostics.faceCount}</div>
              {meshRepairResult.diagnostics.genus >= 0 && (
                <div>Genus: {meshRepairResult.diagnostics.genus}</div>
              )}
              {meshRepairResult.diagnostics.volume > 0 && (
                <div>Volume: {meshRepairResult.diagnostics.volume.toFixed(1)}</div>
              )}
              {meshRepairResult.wasRepaired && (
                <div style={{ marginTop: '4px', color: '#aaf' }}>
                  Repaired: {meshRepairResult.repairMethod}
                </div>
              )}
              {meshRepairResult.diagnostics.issues.length > 0 && (
                <div style={{ marginTop: '4px', color: '#ffa' }}>
                  {meshRepairResult.diagnostics.issues.map((issue, i) => (
                    <div key={i}>• {issue}</div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Hide Original Mesh Checkbox */}
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', cursor: 'pointer', fontSize: '12px' }}>
            <input
              type="checkbox"
              checked={hideOriginalMesh}
              onChange={(e) => setHideOriginalMesh(e.target.checked)}
              style={{ cursor: 'pointer' }}
            />
            Hide Original Mesh
          </label>
          
          {/* Parting Direction Toggle */}
          <button
            onClick={togglePartingDirections}
            style={{
              ...styles.button,
              backgroundColor: showPartingDirections ? '#00aa00' : '#00aaff',
            }}
          >
            {showPartingDirections ? '✓ Parting Directions ON' : 'Show Parting Directions'}
          </button>

          {/* Visibility Paint Toggles */}
          {showPartingDirections && visibilityDataReady && (
            <div style={{ marginTop: '12px' }}>
              <div style={styles.sectionLabel}>Visibility Painting:</div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={() => setShowD1Paint(prev => !prev)}
                  style={{
                    ...styles.toggleButton,
                    backgroundColor: showD1Paint ? '#00ff00' : '#333',
                    borderColor: showD1Paint ? '#00ff00' : '#555',
                    color: showD1Paint ? '#000' : '#fff',
                    fontWeight: showD1Paint ? 'bold' : 'normal',
                  }}
                >
                  🟢 D1
                </button>
                <button
                  onClick={() => setShowD2Paint(prev => !prev)}
                  style={{
                    ...styles.toggleButton,
                    backgroundColor: showD2Paint ? '#ff6600' : '#333',
                    borderColor: showD2Paint ? '#ff6600' : '#555',
                    color: showD2Paint ? '#000' : '#fff',
                    fontWeight: showD2Paint ? 'bold' : 'normal',
                  }}
                >
                  🟠 D2
                </button>
              </div>
              
              {/* Legend */}
              {(showD1Paint || showD2Paint) && (
                <div style={styles.legend}>
                  {showD1Paint && showD2Paint && <div>🟡 Yellow = visible from both</div>}
                  <div>⬜ Gray = not visible</div>
                </div>
              )}
            </div>
          )}

          {/* Loading indicator */}
          {showPartingDirections && !visibilityDataReady && (
            <div style={styles.loading}>Computing visibility...</div>
          )}

          {/* Arrow legend */}
          {showPartingDirections && visibilityDataReady && !showD1Paint && !showD2Paint && (
            <div style={{ marginTop: '10px', fontSize: '11px', opacity: 0.8 }}>
              <div>🟢 Green arrow: Primary direction</div>
              <div>🟠 Orange arrow: Secondary direction</div>
            </div>
          )}

          {/* Inflated Hull Section - only available after parting directions are calculated */}
          {visibilityDataReady && (
            <>
              {/* Separator */}
              <div style={{ borderTop: '1px solid #444', margin: '16px 0 12px 0' }} />

              {/* Inflated Hull Section */}
              <div style={styles.title}>📦 Bounding Volume</div>
              
              <button
                onClick={() => setShowInflatedHull(prev => !prev)}
                style={{
                  ...styles.button,
                  backgroundColor: showInflatedHull ? '#9966ff' : '#00aaff',
                }}
              >
                {showInflatedHull ? '✓ Inflated Hull ON' : 'Show Inflated Hull'}
              </button>
              
              {/* Hide Hull Checkbox */}
              {showInflatedHull && (
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', cursor: 'pointer', fontSize: '12px' }}>
                  <input
                    type="checkbox"
                    checked={hideHull}
                    onChange={(e) => setHideHull(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  Hide Hull (show only cavity)
                </label>
              )}
            </>
          )}

          {/* Inflation Offset Input */}
          {showInflatedHull && (
            <div style={{ marginTop: '12px' }}>
              <div style={styles.sectionLabel}>
                Inflation Offset (mm):
              </div>
              <input
                type="number"
                min="0"
                step="0.1"
                value={inflationOffset}
                onChange={(e) => setInflationOffset(Math.max(0, parseFloat(e.target.value) || 0))}
                style={{
                  width: '100%',
                  padding: '6px 10px',
                  backgroundColor: '#222',
                  border: '1px solid #555',
                  borderRadius: '4px',
                  color: '#fff',
                  fontSize: '13px',
                }}
              />
              
              {/* Hull Stats */}
              {hullStats && (
                <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
                  <div>Hull vertices: {hullStats.vertexCount}</div>
                  <div>Hull faces: {hullStats.faceCount}</div>
                  <div>Edges: {hullStats.manifoldValidation.totalEdgeCount}</div>
                  <div>Euler (V-E+F): {hullStats.manifoldValidation.eulerCharacteristic}</div>
                </div>
              )}

              {/* Manifold Validation */}
              {hullStats && (
                <div style={{ 
                  marginTop: '10px', 
                  padding: '8px', 
                  backgroundColor: hullStats.manifoldValidation.isManifold ? 'rgba(0, 255, 0, 0.15)' : 'rgba(255, 100, 0, 0.15)',
                  borderRadius: '4px',
                  fontSize: '10px'
                }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                    {hullStats.manifoldValidation.isManifold ? '✅ Valid Manifold' : '⚠️ Not Manifold'}
                  </div>
                  <div>Closed: {hullStats.manifoldValidation.isClosed ? '✓ Yes' : `✗ No (${hullStats.manifoldValidation.boundaryEdgeCount} boundary edges)`}</div>
                  {hullStats.manifoldValidation.nonManifoldEdgeCount > 0 && (
                    <div>Non-manifold edges: {hullStats.manifoldValidation.nonManifoldEdgeCount}</div>
                  )}
                </div>
              )}

              {/* Legend */}
              <div style={styles.legend}>
                <div>🟣 Purple = Inflated hull</div>
              </div>
            </div>
          )}

          {/* CSG Subtraction Section - only available after hull is created */}
          {hullStats && (
            <>
              {/* Separator */}
              <div style={{ borderTop: '1px solid #444', margin: '16px 0 12px 0' }} />

              {/* CSG Section */}
              <div style={styles.title}>✂️ Mold Cavity</div>
              
              <button
                onClick={() => setShowCsgResult(prev => !prev)}
                style={{
                  ...styles.button,
                  backgroundColor: showCsgResult ? '#00ffaa' : '#00aaff',
                }}
              >
                {showCsgResult ? '✓ Cavity ON' : 'Subtract Base Mesh'}
              </button>

              {/* CSG Stats */}
              {showCsgResult && csgStats && (
                <div style={{ marginTop: '12px' }}>
                  <div style={{ fontSize: '10px', opacity: 0.7 }}>
                    <div>Cavity vertices: {csgStats.vertexCount}</div>
                    <div>Cavity faces: {csgStats.faceCount}</div>
                  </div>

                  {/* CSG Manifold Validation */}
                  <div style={{ 
                    marginTop: '10px', 
                    padding: '8px', 
                    backgroundColor: csgStats.manifoldValidation.isManifold ? 'rgba(0, 255, 0, 0.15)' : 'rgba(255, 100, 0, 0.15)',
                    borderRadius: '4px',
                    fontSize: '10px'
                  }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                      {csgStats.manifoldValidation.isManifold ? '✅ Valid Manifold' : '⚠️ Not Manifold'}
                    </div>
                    <div>Closed: {csgStats.manifoldValidation.isClosed ? '✓ Yes' : `✗ No (${csgStats.manifoldValidation.boundaryEdgeCount} boundary edges)`}</div>
                  </div>

                  {/* Legend */}
                  <div style={styles.legend}>
                    <div>🩵 Teal = Mold cavity (Hull - Base)</div>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Tetrahedralization Section - only available after CSG is created */}
          {csgStats && (
            <>
              {/* Separator */}
              <div style={{ borderTop: '1px solid #444', margin: '16px 0 12px 0' }} />

              {/* Tetrahedralization Section */}
              <div style={styles.title}>🔺 Tetrahedralize Mold Cavity</div>
              
              <button
                onClick={handleTetrahedralize}
                disabled={isTetrahedralizing}
                style={{
                  ...styles.button,
                  backgroundColor: isTetrahedralizing 
                    ? '#666' 
                    : (showTetMesh && tetVisualization) 
                      ? '#ff6600' 
                      : '#00aaff',
                  cursor: isTetrahedralizing ? 'wait' : 'pointer',
                }}
              >
                {isTetrahedralizing 
                  ? '⏳ Processing...' 
                  : (showTetMesh && tetVisualization) 
                    ? '✓ Tetrahedral Mesh ON' 
                    : 'Tetrahedralize Cavity'}
              </button>

              {/* Hide CSG Mesh Checkbox - show when tet process started */}
              {showTetMesh && (
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', cursor: 'pointer', fontSize: '12px' }}>
                  <input
                    type="checkbox"
                    checked={hideCsgMesh}
                    onChange={(e) => setHideCsgMesh(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  Hide Mold Cavity (show only tet mesh)
                </label>
              )}

              {/* Progress Bar */}
              {isTetrahedralizing && tetProgress && (
                <div style={{ marginTop: '12px' }}>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: '10px', 
                    marginBottom: '4px' 
                  }}>
                    <span>Step {tetProgress.step}/{tetProgress.total}</span>
                    <span>{Math.round((tetProgress.step / tetProgress.total) * 100)}%</span>
                  </div>
                  <div style={{ 
                    width: '100%', 
                    height: '6px', 
                    backgroundColor: '#333', 
                    borderRadius: '3px',
                    overflow: 'hidden'
                  }}>
                    <div style={{ 
                      width: `${(tetProgress.step / tetProgress.total) * 100}%`, 
                      height: '100%', 
                      backgroundColor: '#ff6600',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  {tetProgress.message && (
                    <div style={{ fontSize: '9px', opacity: 0.7, marginTop: '4px' }}>
                      {tetProgress.message}
                    </div>
                  )}
                </div>
              )}

              {/* Error Display */}
              {tetError && (
                <div style={{ 
                  marginTop: '10px', 
                  padding: '8px', 
                  backgroundColor: 'rgba(255, 50, 50, 0.2)',
                  borderRadius: '4px',
                  fontSize: '10px',
                  color: '#ff6666'
                }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>❌ Error</div>
                  <div>{tetError}</div>
                </div>
              )}

              {/* Tet Stats */}
              {tetStats && tetVisualization && (
                <div style={{ marginTop: '12px' }}>
                  <div style={{ fontSize: '10px', opacity: 0.7 }}>
                    <div>Vertices: {tetStats.vertexCount.toLocaleString()}</div>
                    <div>Tetrahedra: {tetStats.tetrahedraCount.toLocaleString()}</div>
                  </div>

                  {/* Success indicator */}
                  <div style={{ 
                    marginTop: '10px', 
                    padding: '8px', 
                    backgroundColor: 'rgba(255, 102, 0, 0.15)',
                    borderRadius: '4px',
                    fontSize: '10px'
                  }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                      ✅ Tetrahedralization Complete
                    </div>
                    <div>Edge length ratio: 0.0065 (TetWild default)</div>
                  </div>

                  {/* Section View Controls */}
                  <div style={{ marginTop: '12px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '12px' }}>
                      <input
                        type="checkbox"
                        checked={sectionPlaneEnabled}
                        onChange={(e) => setSectionPlaneEnabled(e.target.checked)}
                        style={{ cursor: 'pointer' }}
                      />
                      ✂️ Section View (slice model)
                    </label>
                    
                    {sectionPlaneEnabled && (
                      <div style={{ marginTop: '8px' }}>
                        <div style={{ fontSize: '10px', opacity: 0.7, marginBottom: '4px' }}>
                          Slice Height: {Math.round(sectionPlanePosition * 100)}%
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={sectionPlanePosition * 100}
                          onChange={(e) => setSectionPlanePosition(Number(e.target.value) / 100)}
                          style={{
                            width: '100%',
                            cursor: 'pointer',
                          }}
                        />
                      </div>
                    )}
                  </div>

                  {/* Legend */}
                  <div style={styles.legend}>
                    <div>🟠 Orange wireframe = Tet mesh edges</div>
                    <div>🟧 Semi-transparent = Tet surface</div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

<<<<<<< Updated upstream
      {/* Controls Help */}
      <div style={styles.helpPanel}>
        <div><strong>Controls:</strong></div>
        <div>🖱️ Left-click + drag → Orbit</div>
        <div>🖱️ Right-click + drag → Pan</div>
        <div>🖱️ Scroll → Zoom</div>
=======
      {/* Column 2: Context Menu (20%) */}
      <div style={styles.contextPanel}>
        <div style={styles.contextHeader}>Options</div>
        {renderContextMenu()}
      </div>

      {/* Column 3: 3D Viewer (70%) */}
      <div style={styles.viewerContainer}>
        <ThreeViewer
          stlUrl={stlUrl}
          showPartingDirections={showPartingDirections}
          showD1Paint={showD1Paint}
          showD2Paint={showD2Paint}
          showInflatedHull={showInflatedHull}
          inflationOffset={inflationOffset}
          showCsgResult={showCsgResult}
          hideOriginalMesh={hideOriginalMesh}
          hideHull={hideHull}
          hideCavity={hideCavity}
          showVolumetricGrid={showVolumetricGrid}
          hideVoxelGrid={hideVoxelGrid}
          showRLine={showRLine}
          gridResolution={gridResolution}
          gridVisualizationMode={gridVisualizationMode}
          useGPUGrid={useGPUGrid}
          distanceFieldType={distanceFieldType}
          showMoldHalfClassification={showMoldHalfClassification}
          boundaryZoneThreshold={boundaryZoneThreshold}
          showPartingSurface={showPartingSurface}
          partingSurfaceDebugMode={partingSurfaceDebugMode}
          csgReady={csgStats !== null}
          showTetrahedralization={showTetrahedralization}
          hideTetrahedralization={hideTetrahedralization}
          tetraEdgeLengthFac={tetraEdgeLengthFac}
          tetraVisualizationMode={tetraVisualizationMode}
          preloadedTetraResult={preloadedTetraResult}
          onMeshLoaded={handleMeshLoaded}
          onMeshRepaired={handleMeshRepaired}
          onVisibilityDataReady={handleVisibilityDataReady}
          onInflatedHullReady={handleInflatedHullReady}
          onCsgResultReady={handleCsgResultReady}
          onVolumetricGridReady={handleVolumetricGridReady}
          onMoldHalfClassificationReady={handleMoldHalfClassificationReady}
          onTetraPartingSurfaceReady={handleTetraPartingSurfaceReady}
          onTetrahedralizationReady={handleTetrahedralizationReady}
          onTetrahedralizationProgress={handleTetrahedralizationProgress}
        />
>>>>>>> Stashed changes
      </div>
    </div>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
  controlPanel: {
    position: 'absolute',
    top: '20px',
    right: '20px',
    padding: '15px 20px',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: '8px',
    color: '#fff',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    fontSize: '13px',
    zIndex: 1000,
    minWidth: '200px',
  },
  title: {
    marginBottom: '12px',
    fontWeight: 'bold',
  },
  button: {
    padding: '8px 16px',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '12px',
    width: '100%',
    transition: 'background-color 0.2s',
  },
  sectionLabel: {
    marginBottom: '8px',
    fontSize: '11px',
    opacity: 0.8,
  },
  toggleButton: {
    flex: 1,
    padding: '6px 12px',
    border: '2px solid',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    transition: 'all 0.2s',
  },
  legend: {
    marginTop: '8px',
    fontSize: '10px',
    opacity: 0.7,
  },
  loading: {
    marginTop: '10px',
    fontSize: '11px',
    opacity: 0.6,
  },
  helpPanel: {
    position: 'absolute',
    bottom: '20px',
    left: '20px',
    padding: '12px 16px',
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: '8px',
    color: '#fff',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    fontSize: '12px',
    lineHeight: '1.6',
    zIndex: 1000,
  },
};

export default App;
