/**
 * VcMoldCreator - Main Application
 * 
 * A tool for analyzing 3D models for two-piece mold manufacturing.
 * Computes optimal parting directions and visualizes surface visibility.
 */

import React, { useState, useCallback, useRef } from 'react';
import * as THREE from 'three';
import './App.css';
import ThreeViewer, { type GridVisualizationMode, type ThreeViewerHandle } from './components/ThreeViewer';
import type { VisibilityPaintData } from './utils/partingDirection';
import type { InflatedHullResult, ManifoldValidationResult, CsgSubtractionResult } from './utils/inflatedBoundingVolume';
import type { MeshRepairResult, MeshDiagnostics } from './utils/meshRepairManifold';
import type { VolumetricGridResult, DistanceFieldType, BiasedDistanceWeights } from './utils/volumetricGrid';
import type { MoldHalfClassificationResult } from './utils/moldHalfClassification';
import type { EscapeLabelingResult, AdjacencyType } from './utils/partingSurface';

// ============================================================================
// TYPES
// ============================================================================

type Step = 'import' | 'parting' | 'hull' | 'cavity' | 'mold-halves' | 'voxel' | 'parting-surface';

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
  { id: 'voxel', icon: '🧊', title: 'Voxel Grid', description: 'Generate volumetric grid for mold analysis' },
  { id: 'parting-surface', icon: '✂️', title: 'Parting Surface', description: 'Compute escape labeling via multi-source Dijkstra to define parting surface' },
];

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

interface VolumetricGridStats {
  resolution: { x: number; y: number; z: number };
  cellSize: { x: number; y: number; z: number };
  totalCellCount: number;
  moldVolumeCellCount: number;
  moldVolume: number;
  fillRatio: number;
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
  const [hideCavity, setHideCavity] = useState(false);
  const [meshRepairResult, setMeshRepairResult] = useState<{
    diagnostics: MeshDiagnostics;
    wasRepaired: boolean;
    repairMethod: string;
  } | null>(null);
  const [showVolumetricGrid, setShowVolumetricGrid] = useState(false);
  // undefined = auto-calculate based on bounding box diagonal
  const [gridResolution, setGridResolution] = useState<number | undefined>(undefined);
  const [gridVisualizationMode, setGridVisualizationMode] = useState<GridVisualizationMode>('points');
  const [volumetricGridStats, setVolumetricGridStats] = useState<VolumetricGridStats | null>(null);
  const [useGPUGrid, setUseGPUGrid] = useState(true);
  const [hideVoxelGrid, setHideVoxelGrid] = useState(false);
  const [showRLine, setShowRLine] = useState(true);
  const [distanceFieldType, setDistanceFieldType] = useState<DistanceFieldType>('part');
  // Biased distance weights
  const [biasedDistanceWeights, setBiasedDistanceWeights] = useState<BiasedDistanceWeights>({
    partDistanceWeight: 1.0,
    shellBiasWeight: 1.0,
  });
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
  const [partingSurfaceAdjacency, setPartingSurfaceAdjacency] = useState<AdjacencyType>(6);
  const [escapeLabelingStats, setEscapeLabelingStats] = useState<{
    h1VoxelCount: number;
    h2VoxelCount: number;
    unassignedCount: number;
    computeTimeMs: number;
    h1Percentage: number;
    h2Percentage: number;
  } | null>(null);
  // Debug visualization modes: 'none' | 'surface-detection' | 'boundary-labels' | 'seed-labels' | 'seed-labels-only'
  const [partingSurfaceDebugMode, setPartingSurfaceDebugMode] = useState<'none' | 'surface-detection' | 'boundary-labels' | 'seed-labels' | 'seed-labels-only'>('none');

  // Track parameters used for last computation (to detect changes)
  const [lastComputedInflationOffset, setLastComputedInflationOffset] = useState<number | null>(null);
  const [lastComputedGridResolution, setLastComputedGridResolution] = useState<number | null>(null);
  const [lastComputedAdjacency, setLastComputedAdjacency] = useState<AdjacencyType | null>(null);

  // Ref for ThreeViewer component to call methods like recalculateBiasedDistances
  const threeViewerRef = useRef<ThreeViewerHandle | null>(null);

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
        setShowPartingSurface(false);
        setEscapeLabelingStats(null);
        setLastComputedAdjacency(null);
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
        setShowPartingSurface(false);
        setEscapeLabelingStats(null);
        setLastComputedAdjacency(null);
        break;
      case 'cavity':
        // Clear mold-halves and beyond (downstream from cavity)
        setShowMoldHalfClassification(false);
        setMoldHalfStats(null);
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        setShowPartingSurface(false);
        setEscapeLabelingStats(null);
        setLastComputedAdjacency(null);
        break;
      case 'mold-halves':
        // Clear voxel and parting-surface (downstream from mold-halves)
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        setShowPartingSurface(false);
        setEscapeLabelingStats(null);
        setLastComputedAdjacency(null);
        break;
      case 'voxel':
        // Clear parting-surface (downstream from voxel)
        setShowPartingSurface(false);
        setEscapeLabelingStats(null);
        setLastComputedAdjacency(null);
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
      clearFromStep('voxel');
    }
  }, [volumetricGridStats, lastComputedGridResolution, clearFromStep]);

  // Handlers
  const handleFileLoad = useCallback((url: string, fileName: string) => {
    if (stlUrl) URL.revokeObjectURL(stlUrl);
    setStlUrl(url);
    setLoadedFileName(fileName);
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
    setHideCavity(false);
    setMeshRepairResult(null);
    setShowVolumetricGrid(false);
    setVolumetricGridStats(null);
    setShowMoldHalfClassification(false);
    setMoldHalfStats(null);
    setShowPartingSurface(false);
    setEscapeLabelingStats(null);
    // Move to parting step after loading
    setActiveStep('parting');
  }, [stlUrl]);

  const handleMeshLoaded = useCallback((mesh: THREE.Mesh) => {
    setMeshLoaded(true);
    
    // Compute bounding box diagonal and set default inflation offset to 15%
    if (mesh.geometry.boundingBox) {
      const size = new THREE.Vector3();
      mesh.geometry.boundingBox.getSize(size);
      // Apply the mesh scale to get actual world-space size
      size.multiply(mesh.scale);
      const diagonal = size.length();
      const defaultOffset = diagonal * 0.15; // 15% of bounding box diagonal
      setInflationOffset(defaultOffset);
    }
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
      // Track the parameters used for this computation
      if (result) {
        setLastComputedInflationOffset(inflationOffset);
      }
      // Reset CSG when hull changes
      if (!result) {
        setShowCsgResult(false);
        setCsgStats(null);
      }
    },
    [inflationOffset]
  );

  const handleCsgResultReady = useCallback(
    (result: CsgSubtractionResult | null) => {
      setCsgStats(result ? {
        vertexCount: result.vertexCount,
        faceCount: result.faceCount,
        manifoldValidation: result.manifoldValidation,
      } : null);
      // Reset volumetric grid when CSG changes
      if (!result) {
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
      }
    },
    []
  );

  const handleVolumetricGridReady = useCallback(
    (result: VolumetricGridResult | null) => {
      setVolumetricGridStats(result ? {
        resolution: { x: result.resolution.x, y: result.resolution.y, z: result.resolution.z },
        cellSize: { x: result.cellSize.x, y: result.cellSize.y, z: result.cellSize.z },
        totalCellCount: result.totalCellCount,
        moldVolumeCellCount: result.moldVolumeCellCount,
        moldVolume: result.stats.moldVolume,
        fillRatio: result.stats.fillRatio,
        computeTimeMs: result.stats.computeTimeMs,
      } : null);
      // Track the actual computed resolution (may be auto-calculated)
      if (result) {
        // Store the actual resolution from the result (handles auto-calculation case)
        setLastComputedGridResolution(result.resolution.x);
      }
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

  const handleEscapeLabelingReady = useCallback(
    (result: EscapeLabelingResult | null) => {
      if (result) {
        const total = result.d1Count + result.d2Count + result.unlabeledCount;
        setEscapeLabelingStats({
          h1VoxelCount: result.d1Count,
          h2VoxelCount: result.d2Count,
          unassignedCount: result.unlabeledCount,
          computeTimeMs: result.computeTimeMs,
          h1Percentage: total > 0 ? result.d1Count / total * 100 : 0,
          h2Percentage: total > 0 ? result.d2Count / total * 100 : 0,
        });
        setLastComputedAdjacency(partingSurfaceAdjacency);
      } else {
        setEscapeLabelingStats(null);
      }
    },
    [partingSurfaceAdjacency]
  );

  // Active step for context menu
  const [activeStep, setActiveStep] = useState<Step>('import');
  const [isCalculating, setIsCalculating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loadedFileName, setLoadedFileName] = useState<string | null>(null);

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
      case 'voxel':
        // Voxel depends on mold-halves (or at least cavity)
        if (getStepStatus('cavity') === 'locked') return 'locked';
        if (volumetricGridStats) {
          // Check if parameters changed - only if user explicitly set a resolution
          // (gridResolution undefined means auto-calculate, so no recalc needed)
          if (gridResolution !== undefined && lastComputedGridResolution !== null && gridResolution !== lastComputedGridResolution) {
            return 'needs-recalc';
          }
          return 'completed';
        }
        return csgStats ? 'available' : 'locked';
      case 'parting-surface':
        // Parting surface depends on voxel grid AND mold-halves
        if (getStepStatus('voxel') === 'locked' || getStepStatus('voxel') === 'needs-recalc') return 'locked';
        if (getStepStatus('mold-halves') === 'locked') return 'locked';
        if (escapeLabelingStats) {
          // Check if parameters changed
          if (lastComputedAdjacency !== null && partingSurfaceAdjacency !== lastComputedAdjacency) {
            return 'needs-recalc';
          }
          return 'completed';
        }
        return (volumetricGridStats && moldHalfStats) ? 'available' : 'locked';
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
      case 'voxel':
        clearFromStep('voxel');
        // Reset current step's state to trigger recalculation
        setVolumetricGridStats(null);
        setShowVolumetricGrid(false);
        setTimeout(() => setShowVolumetricGrid(true), 0);
        break;
      case 'parting-surface':
        clearFromStep('parting-surface');
        // Reset current step's state to trigger recalculation
        setEscapeLabelingStats(null);
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
  const [isDragging, setIsDragging] = useState(false);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.stl')) {
        alert('Please upload an STL file');
        return;
      }
      const url = URL.createObjectURL(file);
      handleFileLoad(url, file.name);
    }
  }, [handleFileLoad]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.stl')) {
        alert('Please upload an STL file');
        return;
      }
      const url = URL.createObjectURL(file);
      handleFileLoad(url, file.name);
    }
  }, [handleFileLoad]);

  const renderContextMenu = () => {
    const status = getStepStatus(activeStep);
    const stepInfo = STEPS.find(s => s.id === activeStep)!;

    return (
      <div style={styles.contextContent}>
        <div style={styles.contextTitle}>{stepInfo.icon} {stepInfo.title}</div>
        <div style={styles.contextDescription}>{stepInfo.description}</div>

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
                padding: '28px 20px',
                backgroundColor: isDragging ? 'rgba(0, 123, 255, 0.1)' : '#f9fafb',
                border: `2px dashed ${isDragging ? '#007bff' : '#becad6'}`,
                borderRadius: '0.625rem',
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'all 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06)',
              }}
            >
              {loadedFileName ? (
                <>
                  <div style={{ fontSize: '28px', marginBottom: '10px' }}>✅</div>
                  <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: '#17c671' }}>Loaded:</div>
                  <div style={{ fontSize: '12px', color: '#5A6169', wordBreak: 'break-all' }}>{loadedFileName}</div>
                  <div style={{ fontSize: '11px', color: '#868e96', marginTop: '10px' }}>Click to replace</div>
                </>
              ) : (
                <>
                  <div style={{ fontSize: '36px', marginBottom: '10px' }}>📁</div>
                  <div style={{ fontSize: '14px', fontWeight: 500, color: '#212529' }}>Upload STL File</div>
                  <div style={{ fontSize: '12px', color: '#868e96', marginTop: '6px' }}>Click or drag & drop</div>
                </>
              )}
            </div>

            {/* Mesh info after loading */}
            {meshLoaded && meshRepairResult && (
              <div style={{ ...styles.statsBox, marginTop: '16px' }}>
                <div style={{ fontWeight: 500, marginBottom: '8px', color: meshRepairResult.diagnostics.isManifold ? '#17c671' : '#ffb400' }}>
                  {meshRepairResult.diagnostics.isManifold ? '✅ Mesh Valid' : '⚠️ Mesh Issues'}
                </div>
                <div>Vertices: {meshRepairResult.diagnostics.vertexCount.toLocaleString()}</div>
                <div>Faces: {meshRepairResult.diagnostics.faceCount.toLocaleString()}</div>
                {meshRepairResult.diagnostics.genus >= 0 && (
                  <div>Genus: {meshRepairResult.diagnostics.genus}</div>
                )}
                {meshRepairResult.wasRepaired && (
                  <div style={{ color: '#007bff', marginTop: '6px' }}>Repaired: {meshRepairResult.repairMethod}</div>
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
                <div style={{ fontWeight: 500, marginBottom: '6px', color: '#17c671' }}>✅ Parting Directions Computed</div>
                <div style={styles.legend}>
                  <div style={{ color: '#17c671' }}>🟢 Green arrow: Primary direction (D1)</div>
                  <div style={{ color: '#ffb400' }}>🟠 Orange arrow: Secondary direction (D2)</div>
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
                <div style={{ color: hullStats.manifoldValidation.isManifold ? '#17c671' : '#ffb400', fontWeight: 500, marginTop: '4px' }}>
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
                <div style={{ color: csgStats.manifoldValidation.isManifold ? '#17c671' : '#ffb400', fontWeight: 500, marginTop: '4px' }}>
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
              style={{ width: '100%', marginBottom: '16px' }}
            />
            {/* Mold Half Stats */}
            {moldHalfStats && (
              <div style={styles.statsBox}>
                <div style={{ marginBottom: '6px', fontSize: '11px', color: '#868e96', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Outer boundary (hull):
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: '#17c671', fontWeight: 500 }}>H₁ (Green):</span>
                  <span>{moldHalfStats.h1Count} ({moldHalfStats.h1Percentage.toFixed(1)}%)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#ffb400', fontWeight: 500 }}>H₂ (Orange):</span>
                  <span>{moldHalfStats.h2Count} ({moldHalfStats.h2Percentage.toFixed(1)}%)</span>
                </div>
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #e1e5eb', fontSize: '11px', color: '#868e96' }}>
                  <div>Outer: {moldHalfStats.outerBoundaryCount} tris</div>
                  <div>Inner (part): {moldHalfStats.innerBoundaryCount} tris</div>
                </div>
              </div>
            )}
            
            {!moldHalfStats && csgStats && (
              <div style={{ color: '#868e96', fontSize: '13px', lineHeight: 1.5 }}>
                Click "Calculate" to classify the mold cavity into H₁ and H₂ halves based on parting directions.
              </div>
            )}
          </div>
        )}

        {activeStep === 'voxel' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            <div style={styles.optionLabel}>Grid Resolution: {gridResolution === undefined ? '(Auto)' : ''}</div>
            <input
              type="text"
              placeholder="Auto"
              defaultValue={gridResolution !== undefined ? String(gridResolution) : ''}
              key={`grid-res-${lastComputedGridResolution}`}
              onBlur={(e) => {
                const val = e.target.value.trim();
                if (val === '' || val.toLowerCase() === 'auto') {
                  e.target.value = '';
                  setGridResolution(undefined);
                  if (volumetricGridStats && lastComputedGridResolution !== null) {
                    clearFromStep('voxel');
                  }
                } else {
                  const num = parseInt(val) || 64;
                  const clamped = Math.max(16, Math.min(128, num));
                  e.target.value = String(clamped);
                  handleGridResolutionChange(clamped);
                }
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  (e.target as HTMLInputElement).blur();
                }
              }}
              style={styles.input}
            />
            <div style={{ fontSize: '11px', color: '#868e96', marginTop: '4px' }}>
              Leave empty for auto (2% of diagonal). Range: 16-128
            </div>

            <div style={{ ...styles.optionLabel, marginTop: '16px' }}>Visualization:</div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <button
                onClick={() => setGridVisualizationMode('points')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: gridVisualizationMode === 'points' ? '#007bff' : '#ffffff',
                  borderColor: gridVisualizationMode === 'points' ? '#007bff' : '#becad6',
                  color: gridVisualizationMode === 'points' ? '#ffffff' : '#5A6169',
                  boxShadow: gridVisualizationMode === 'points' ? '0 2px 8px rgba(0,123,255,0.3)' : 'none',
                }}
              >
                Points
              </button>
              <button
                onClick={() => setGridVisualizationMode('voxels')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: gridVisualizationMode === 'voxels' ? '#007bff' : '#ffffff',
                  borderColor: gridVisualizationMode === 'voxels' ? '#007bff' : '#becad6',
                  color: gridVisualizationMode === 'voxels' ? '#ffffff' : '#5A6169',
                  boxShadow: gridVisualizationMode === 'voxels' ? '0 2px 8px rgba(0,123,255,0.3)' : 'none',
                }}
              >
                Voxels
              </button>
            </div>

            <label style={styles.checkbox}>
              <input
                type="checkbox"
                checked={useGPUGrid}
                onChange={(e) => setUseGPUGrid(e.target.checked)}
              />
              Use GPU (WebGPU)
              <span style={{ fontSize: '11px', marginLeft: '4px', color: typeof navigator !== 'undefined' && 'gpu' in navigator ? '#17c671' : '#c4183c' }}>
                {typeof navigator !== 'undefined' && 'gpu' in navigator ? '✅' : '❌'}
              </span>
            </label>

            {volumetricGridStats && (
              <div style={styles.statsBox}>
                <div>Resolution: {volumetricGridStats.resolution.x}×{volumetricGridStats.resolution.y}×{volumetricGridStats.resolution.z}</div>
                <div>Voxel size: {volumetricGridStats.cellSize.x.toFixed(3)} × {volumetricGridStats.cellSize.y.toFixed(3)} × {volumetricGridStats.cellSize.z.toFixed(3)}</div>
                <div>Mold cells: {volumetricGridStats.moldVolumeCellCount.toLocaleString()}</div>
                <div>Fill ratio: {(volumetricGridStats.fillRatio * 100).toFixed(1)}%</div>
                <div>Compute: {volumetricGridStats.computeTimeMs.toFixed(0)} ms</div>
              </div>
            )}

            {/* Biased Distance Weights - only available after grid is computed */}
            {volumetricGridStats && (
              <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid #e1e5eb' }}>
                <div style={{ ...styles.optionLabel, marginBottom: '12px' }}>Biased Distance Weights:</div>
                
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ fontSize: '12px', color: '#5A6169', display: 'block', marginBottom: '4px' }}>
                    Part Distance (δᵢ) weight: {biasedDistanceWeights.partDistanceWeight.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.1"
                    value={biasedDistanceWeights.partDistanceWeight}
                    onChange={(e) => setBiasedDistanceWeights(prev => ({
                      ...prev,
                      partDistanceWeight: parseFloat(e.target.value)
                    }))}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                </div>
                
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ fontSize: '12px', color: '#5A6169', display: 'block', marginBottom: '4px' }}>
                    Shell Bias (R-δw) weight: {biasedDistanceWeights.shellBiasWeight.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.1"
                    value={biasedDistanceWeights.shellBiasWeight}
                    onChange={(e) => setBiasedDistanceWeights(prev => ({
                      ...prev,
                      shellBiasWeight: parseFloat(e.target.value)
                    }))}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                </div>
                
                <button
                  onClick={() => {
                    // Trigger recalculation via ThreeViewer ref
                    if (threeViewerRef.current) {
                      threeViewerRef.current.recalculateBiasedDistances(biasedDistanceWeights);
                    }
                  }}
                  style={{
                    ...styles.primaryButton,
                    marginTop: '8px',
                    width: '100%',
                    padding: '10px 14px',
                    fontSize: '13px',
                  }}
                >
                  🔄 Recalculate Biased Distances
                </button>
                
                <button
                  onClick={() => {
                    setBiasedDistanceWeights({ partDistanceWeight: 1.0, shellBiasWeight: 1.0 });
                  }}
                  style={{
                    ...styles.secondaryButton,
                    marginTop: '8px',
                    width: '100%',
                    padding: '8px 12px',
                    fontSize: '12px',
                  }}
                >
                  Reset to Defaults
                </button>
              </div>
            )}
          </div>
        )}

        {activeStep === 'parting-surface' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            <div style={styles.optionLabel}>Adjacency Type:</div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <button
                onClick={() => setPartingSurfaceAdjacency(6)}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: partingSurfaceAdjacency === 6 ? '#007bff' : '#ffffff',
                  borderColor: partingSurfaceAdjacency === 6 ? '#007bff' : '#becad6',
                  color: partingSurfaceAdjacency === 6 ? '#ffffff' : '#5A6169',
                  boxShadow: partingSurfaceAdjacency === 6 ? '0 2px 8px rgba(0,123,255,0.3)' : 'none',
                }}
              >
                6 (Faces)
              </button>
              <button
                onClick={() => setPartingSurfaceAdjacency(26)}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: partingSurfaceAdjacency === 26 ? '#007bff' : '#ffffff',
                  borderColor: partingSurfaceAdjacency === 26 ? '#007bff' : '#becad6',
                  color: partingSurfaceAdjacency === 26 ? '#ffffff' : '#5A6169',
                  boxShadow: partingSurfaceAdjacency === 26 ? '0 2px 8px rgba(0,123,255,0.3)' : 'none',
                }}
              >
                26 (Full)
              </button>
            </div>

            {escapeLabelingStats && (
              <>
                {/* Debug Visualization Mode */}
                <div style={{ marginTop: '12px' }}>
                  <label style={{ color: '#5A6169', fontSize: '12px', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Debug View:
                  </label>
                  <select
                    value={partingSurfaceDebugMode}
                    onChange={(e) => setPartingSurfaceDebugMode(e.target.value as 'none' | 'surface-detection' | 'boundary-labels' | 'seed-labels' | 'seed-labels-only')}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      backgroundColor: '#ffffff',
                      color: '#495057',
                      border: '1px solid #becad6',
                      borderRadius: '0.375rem',
                      fontSize: '13px',
                      cursor: 'pointer',
                    }}
                  >
                    <option value="none">Normal View</option>
                    <option value="surface-detection">Step 1: Surface Detection (Inner/Outer)</option>
                    <option value="boundary-labels">Step 2: Boundary Labels (H₁/H₂)</option>
                    <option value="seed-labels">Step 3: Dijkstra Result (All Voxels)</option>
                    <option value="seed-labels-only">Step 3.1: Dijkstra Result (Seeds Only)</option>
                  </select>
                </div>

                <div style={styles.statsBox}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ color: '#17c671', fontWeight: 500 }}>H₁ voxels:</span>
                    <span>{escapeLabelingStats.h1VoxelCount.toLocaleString()} ({escapeLabelingStats.h1Percentage.toFixed(1)}%)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#ffb400', fontWeight: 500 }}>H₂ voxels:</span>
                    <span>{escapeLabelingStats.h2VoxelCount.toLocaleString()} ({escapeLabelingStats.h2Percentage.toFixed(1)}%)</span>
                  </div>
                  {escapeLabelingStats.unassignedCount > 0 && (
                    <div style={{ color: '#c4183c', marginTop: '4px' }}>
                      Unassigned: {escapeLabelingStats.unassignedCount}
                    </div>
                  )}
                  <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #e1e5eb', fontSize: '11px', color: '#868e96' }}>
                    Compute: {escapeLabelingStats.computeTimeMs.toFixed(0)} ms
                  </div>
                </div>
              </>
            )}

            {!escapeLabelingStats && volumetricGridStats && moldHalfStats && (
              <div style={{ color: '#868e96', fontSize: '13px', lineHeight: 1.5 }}>
                Click "Calculate" to compute escape labeling via multi-source Dijkstra flood.
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
                backgroundColor: status === 'completed' ? '#17c671' : status === 'needs-recalc' ? '#ffb400' : isCalculating ? '#868e96' : '#007bff',
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

      </div>
    );
  };

  // Render Display Options floating panel (top right of viewer)
  const renderDisplayOptions = () => {
    if (!meshLoaded) return null;
    
    return (
      <div style={styles.displayOptionsPanel}>
        <div style={styles.displayOptionsHeader}>Display Options</div>
        <div style={styles.displayOptionsContent}>
          {/* Original Mesh - always available when loaded */}
          <label style={styles.displayCheckbox}>
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
              <label style={styles.displayCheckbox}>
                <input
                  type="checkbox"
                  checked={showD1Paint}
                  onChange={(e) => setShowD1Paint(e.target.checked)}
                />
                <span>Show D1 Visibility</span>
                <span style={{ color: '#17c671', marginLeft: '4px' }}>🟢</span>
              </label>
              <label style={styles.displayCheckbox}>
                <input
                  type="checkbox"
                  checked={showD2Paint}
                  onChange={(e) => setShowD2Paint(e.target.checked)}
                />
                <span>Show D2 Visibility</span>
                <span style={{ color: '#ffb400', marginLeft: '4px' }}>🟠</span>
              </label>
            </>
          )}

          {/* Hull visibility - available after hull is computed */}
          {hullStats && (
            <label style={styles.displayCheckbox}>
              <input
                type="checkbox"
                checked={hideHull}
                onChange={(e) => setHideHull(e.target.checked)}
              />
              <span>Hide Hull</span>
              <span style={{ color: '#8445f7', marginLeft: '4px' }}>🟣</span>
            </label>
          )}

          {/* Cavity visibility - available after cavity is computed */}
          {csgStats && (
            <label style={styles.displayCheckbox}>
              <input
                type="checkbox"
                checked={hideCavity}
                onChange={(e) => setHideCavity(e.target.checked)}
              />
              <span>Hide Cavity</span>
              <span style={{ color: '#00b8d8', marginLeft: '4px' }}>🩵</span>
            </label>
          )}

          {/* Voxel grid visibility - available after voxel is computed */}
          {volumetricGridStats && (
            <label style={styles.displayCheckbox}>
              <input
                type="checkbox"
                checked={hideVoxelGrid}
                onChange={(e) => setHideVoxelGrid(e.target.checked)}
              />
              <span>Hide Voxel Grid</span>
              <span style={{ color: '#00b8d8', marginLeft: '4px' }}>🧊</span>
            </label>
          )}

          {/* R Line visibility - available after voxel is computed */}
          {volumetricGridStats && (
            <label style={styles.displayCheckbox}>
              <input
                type="checkbox"
                checked={showRLine}
                onChange={(e) => setShowRLine(e.target.checked)}
              />
              <span>Show R Line</span>
              <span style={{ marginLeft: '4px' }}>📏</span>
            </label>
          )}

          {/* Voxel Grid Coloring - available after voxel is computed */}
          {volumetricGridStats && (
            <div style={{ marginTop: '8px' }}>
              <label style={{ fontSize: '11px', color: 'rgba(255,255,255,0.7)', display: 'block', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Voxel Grid Coloring:
              </label>
              <select
                value={distanceFieldType}
                onChange={(e) => setDistanceFieldType(e.target.value as DistanceFieldType)}
                style={{
                  width: '100%',
                  padding: '6px 10px',
                  borderRadius: '0.375rem',
                  border: '1px solid rgba(255,255,255,0.2)',
                  backgroundColor: 'rgba(0,0,0,0.3)',
                  color: '#ffffff',
                  fontSize: '12px',
                  cursor: 'pointer'
                }}
              >
                <option value="part">Part Distance (δᵢ)</option>
                <option value="biased">Biased Distance (δᵢ + λw)</option>
                <option value="weight">Weighting Factor (wt)</option>
                <option value="boundary">Boundary Mask (Biased vs Unbiased)</option>
              </select>
            </div>
          )}
        </div>
      </div>
    );
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div style={styles.container}>
      {/* Column 1: Steps Sidebar (10%) */}
      <div style={styles.stepsSidebar}>
        <div style={styles.sidebarHeader}>
          <span style={{ fontSize: '20px' }}>🔧</span>
        </div>
        {STEPS.map((step) => {
          const status = getStepStatus(step.id);
          const isActive = activeStep === step.id;
          return (
            <div
              key={step.id}
              onClick={() => status !== 'locked' && setActiveStep(step.id)}
              style={{
                ...styles.stepIcon,
                backgroundColor: isActive ? 'rgba(0, 123, 255, 0.1)' : status === 'completed' ? 'rgba(23, 198, 113, 0.08)' : status === 'needs-recalc' ? 'rgba(255, 180, 0, 0.08)' : 'transparent',
                opacity: status === 'locked' ? 0.4 : 1,
                cursor: status === 'locked' ? 'not-allowed' : 'pointer',
                borderLeft: isActive ? '3px solid #007bff' : status === 'completed' ? '3px solid #17c671' : status === 'needs-recalc' ? '3px solid #ffb400' : '3px solid transparent',
              }}
              title={step.title + '\n' + step.description}
            >
              <span style={{ fontSize: '20px' }}>{step.icon}</span>
              {status === 'completed' && (
                <span style={styles.checkMark}>✓</span>
              )}
              {status === 'needs-recalc' && (
                <span style={{ ...styles.checkMark, color: '#ffb400' }}>⟳</span>
              )}
            </div>
          );
        })}
        
        {/* Help at bottom */}
        <div style={styles.helpIcon} title="Controls: Left-click drag = Orbit, Right-click drag = Pan, Scroll = Zoom">
          ❓
        </div>
      </div>

      {/* Column 2: Context Menu (20%) */}
      <div style={styles.contextPanel}>
        <div style={styles.contextHeader}>Options</div>
        {renderContextMenu()}
      </div>

      {/* Column 3: 3D Viewer (70%) */}
      <div style={styles.viewerContainer}>
        {renderDisplayOptions()}
        <ThreeViewer
          ref={threeViewerRef}
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
          partingSurfaceAdjacency={partingSurfaceAdjacency}
          partingSurfaceDebugMode={partingSurfaceDebugMode}
          onMeshLoaded={handleMeshLoaded}
          onMeshRepaired={handleMeshRepaired}
          onVisibilityDataReady={handleVisibilityDataReady}
          onInflatedHullReady={handleInflatedHullReady}
          onCsgResultReady={handleCsgResultReady}
          onVolumetricGridReady={handleVolumetricGridReady}
          onMoldHalfClassificationReady={handleMoldHalfClassificationReady}
          onEscapeLabelingReady={handleEscapeLabelingReady}
        />
      </div>
    </div>
  );
}

// ============================================================================
// STYLES - Shards Dashboard Theme
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    width: '100vw',
    height: '100vh',
    margin: 0,
    padding: 0,
    overflow: 'hidden',
    fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    backgroundColor: '#f5f6f8',
  },
  stepsSidebar: {
    width: '80px',
    minWidth: '80px',
    maxWidth: '80px',
    height: '100%',
    backgroundColor: '#ffffff',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    paddingTop: '0',
    borderRight: '1px solid #e1e5eb',
    boxShadow: '2px 0 10px rgba(90,97,105,0.08)',
  },
  sidebarHeader: {
    padding: '20px 12px',
    marginBottom: '8px',
    borderBottom: '1px solid #e1e5eb',
    width: '100%',
    textAlign: 'center',
    backgroundColor: '#007bff',
    color: '#ffffff',
  },
  stepIcon: {
    width: '100%',
    padding: '16px 0',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    transition: 'all 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06)',
    borderLeft: '3px solid transparent',
  },
  checkMark: {
    position: 'absolute',
    bottom: '4px',
    right: '12px',
    fontSize: '10px',
    color: '#17c671',
    fontWeight: 'bold',
  },
  helpIcon: {
    marginTop: 'auto',
    padding: '16px',
    fontSize: '18px',
    cursor: 'help',
    opacity: 0.6,
    color: '#5A6169',
  },
  contextPanel: {
    width: '280px',
    minWidth: '260px',
    maxWidth: '320px',
    height: '100%',
    backgroundColor: '#ffffff',
    borderRight: '1px solid #e1e5eb',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    boxShadow: '2px 0 10px rgba(90,97,105,0.08)',
  },
  contextHeader: {
    padding: '20px 24px',
    fontSize: '16px',
    fontWeight: 500,
    color: '#212529',
    borderBottom: '1px solid #e1e5eb',
    backgroundColor: '#ffffff',
    letterSpacing: '0.5px',
  },
  contextContent: {
    flex: 1,
    padding: '20px',
    overflowY: 'auto',
    color: '#5A6169',
  },
  contextTitle: {
    fontSize: '15px',
    fontWeight: 500,
    marginBottom: '8px',
    color: '#212529',
  },
  contextDescription: {
    fontSize: '13px',
    color: '#868e96',
    marginBottom: '20px',
    lineHeight: 1.5,
  },
  optionsSection: {
    marginBottom: '20px',
  },
  optionLabel: {
    fontSize: '12px',
    color: '#5A6169',
    marginBottom: '8px',
    fontWeight: 400,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  input: {
    width: '100%',
    padding: '10px 14px',
    backgroundColor: '#ffffff',
    border: '1px solid #becad6',
    borderRadius: '0.375rem',
    color: '#495057',
    fontSize: '14px',
    fontWeight: 300,
    boxSizing: 'border-box',
    transition: 'box-shadow 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06), border 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06)',
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '13px',
    cursor: 'pointer',
    marginTop: '10px',
    color: '#5A6169',
    fontWeight: 300,
  },
  toggleButton: {
    flex: 1,
    padding: '10px 14px',
    border: '1px solid #becad6',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: 400,
    transition: 'all 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06)',
    backgroundColor: '#ffffff',
    color: '#5A6169',
  },
  statsBox: {
    marginTop: '16px',
    padding: '14px',
    backgroundColor: '#f5f6f8',
    borderRadius: '0.625rem',
    fontSize: '12px',
    lineHeight: 1.7,
    color: '#5A6169',
    border: '1px solid #e1e5eb',
  },
  primaryButton: {
    backgroundColor: '#007bff',
    border: 'none',
    borderRadius: '0.375rem',
    color: '#ffffff',
    fontWeight: 400,
    cursor: 'pointer',
    transition: 'all 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06)',
    boxShadow: '0 2px 8px rgba(0,123,255,0.2)',
  },
  secondaryButton: {
    backgroundColor: '#ffffff',
    border: '1px solid #becad6',
    borderRadius: '0.375rem',
    color: '#5A6169',
    cursor: 'pointer',
    transition: 'all 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06)',
  },
  legend: {
    marginTop: '10px',
    fontSize: '11px',
    color: '#868e96',
    lineHeight: 1.6,
  },
  calculateSection: {
    marginTop: '24px',
    paddingTop: '20px',
    borderTop: '1px solid #e1e5eb',
  },
  calculateButton: {
    width: '100%',
    padding: '14px 20px',
    border: 'none',
    borderRadius: '0.375rem',
    color: '#ffffff',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 250ms cubic-bezier(0.27, 0.01, 0.38, 1.06)',
    boxShadow: '0 4px 12px rgba(0,123,255,0.3)',
  },
  progressContainer: {
    marginTop: '12px',
    height: '8px',
    backgroundColor: '#e9ecef',
    borderRadius: '1rem',
    overflow: 'hidden',
    boxShadow: 'inset 0 1px 2px rgba(90,97,105,0.15)',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#007bff',
    borderRadius: '1rem',
    transition: 'width 0.3s ease',
  },
  lockedMessage: {
    padding: '16px',
    backgroundColor: 'rgba(255, 180, 0, 0.1)',
    borderRadius: '0.625rem',
    fontSize: '13px',
    textAlign: 'center',
    color: '#ffb400',
    border: '1px solid rgba(255, 180, 0, 0.3)',
  },
  viewerContainer: {
    flex: 1,
    height: '100%',
    position: 'relative',
    backgroundColor: '#1a1d21',
    borderRadius: '0.625rem 0 0 0.625rem',
    overflow: 'hidden',
    margin: '12px 12px 12px 0',
    boxShadow: '0 0.46875rem 2.1875rem rgba(90,97,105,0.1), 0 0.9375rem 1.40625rem rgba(90,97,105,0.1)',
  },
  displayOptionsPanel: {
    position: 'absolute',
    top: '16px',
    right: '16px',
    zIndex: 100,
    backgroundColor: 'rgba(26, 29, 33, 0.9)',
    backdropFilter: 'blur(10px)',
    borderRadius: '0.625rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
    minWidth: '200px',
    maxWidth: '250px',
  },
  displayOptionsHeader: {
    padding: '12px 16px',
    fontSize: '12px',
    fontWeight: 500,
    color: 'rgba(255, 255, 255, 0.9)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  displayOptionsContent: {
    padding: '12px 16px',
  },
  displayCheckbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    cursor: 'pointer',
    marginBottom: '8px',
    color: 'rgba(255, 255, 255, 0.8)',
    fontWeight: 300,
  },
};

export default App;
