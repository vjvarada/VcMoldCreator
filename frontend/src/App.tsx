/**
 * VcMoldCreator - Main Application
 * 
 * A tool for analyzing 3D models for two-piece mold manufacturing.
 * Computes optimal parting directions and visualizes surface visibility.
 */

import React, { useState, useCallback, useRef } from 'react';
import './App.css';
import ThreeViewer, { type GridVisualizationMode } from './components/ThreeViewer';
import type { VisibilityPaintData } from './utils/partingDirection';
import type { InflatedHullResult, ManifoldValidationResult, CsgSubtractionResult } from './utils/inflatedBoundingVolume';
import type { MeshRepairResult, MeshDiagnostics } from './utils/meshRepairManifold';
import type { VolumetricGridResult } from './utils/volumetricGrid';
import type { MoldHalfClassificationResult } from './utils/moldHalfClassification';

// ============================================================================
// TYPES
// ============================================================================

type Step = 'import' | 'parting' | 'hull' | 'cavity' | 'mold-halves' | 'voxel';

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
  const [gridResolution, setGridResolution] = useState(64);
  const [gridVisualizationMode, setGridVisualizationMode] = useState<GridVisualizationMode>('points');
  const [volumetricGridStats, setVolumetricGridStats] = useState<VolumetricGridStats | null>(null);
  const [useGPUGrid, setUseGPUGrid] = useState(true);
  const [hideVoxelGrid, setHideVoxelGrid] = useState(false);
  const [showMoldHalfClassification, setShowMoldHalfClassification] = useState(false);
  const [moldHalfStats, setMoldHalfStats] = useState<{
    h1Count: number;
    h2Count: number;
    h1Percentage: number;
    h2Percentage: number;
    totalTriangles: number;
    outerBoundaryCount: number;
    innerBoundaryCount: number;
  } | null>(null);

  // Track parameters used for last computation (to detect changes)
  const [lastComputedInflationOffset, setLastComputedInflationOffset] = useState<number | null>(null);
  const [lastComputedGridResolution, setLastComputedGridResolution] = useState<number | null>(null);

  // Helper to clear downstream steps
  const clearFromStep = useCallback((step: Step) => {
    switch (step) {
      case 'parting':
        setShowPartingDirections(false);
        setVisibilityDataReady(false);
        setShowD1Paint(false);
        setShowD2Paint(false);
        // Fall through to clear hull and beyond
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
        break;
      case 'hull':
        setShowInflatedHull(false);
        setHullStats(null);
        setLastComputedInflationOffset(null);
        // Fall through to clear cavity and beyond
        setShowCsgResult(false);
        setCsgStats(null);
        setShowMoldHalfClassification(false);
        setMoldHalfStats(null);
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        break;
      case 'cavity':
        setShowCsgResult(false);
        setCsgStats(null);
        // Fall through to clear mold-halves and beyond
        setShowMoldHalfClassification(false);
        setMoldHalfStats(null);
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        break;
      case 'mold-halves':
        setShowMoldHalfClassification(false);
        setMoldHalfStats(null);
        // Fall through to clear voxel
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
        break;
      case 'voxel':
        setShowVolumetricGrid(false);
        setVolumetricGridStats(null);
        setLastComputedGridResolution(null);
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
    // Move to parting step after loading
    setActiveStep('parting');
  }, [stlUrl]);

  const handleMeshLoaded = useCallback(() => setMeshLoaded(true), []);

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
          // Check if parameters changed
          if (lastComputedGridResolution !== null && gridResolution !== lastComputedGridResolution) {
            return 'needs-recalc';
          }
          return 'completed';
        }
        return csgStats ? 'available' : 'locked';
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
        setShowPartingDirections(true);
        break;
      case 'hull':
        clearFromStep('hull');
        setShowInflatedHull(true);
        break;
      case 'cavity':
        clearFromStep('cavity');
        setShowCsgResult(true);
        break;
      case 'mold-halves':
        clearFromStep('mold-halves');
        setShowMoldHalfClassification(true);
        break;
      case 'voxel':
        clearFromStep('voxel');
        setShowVolumetricGrid(true);
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

        {activeStep === 'voxel' && status !== 'locked' && (
          <div style={styles.optionsSection}>
            <div style={styles.optionLabel}>Grid Resolution:</div>
            <input
              type="number"
              min="8"
              max="128"
              step="8"
              defaultValue={gridResolution}
              key={`grid-res-${lastComputedGridResolution}`}
              onBlur={(e) => {
                const val = e.target.value;
                const num = parseInt(val) || 64;
                const clamped = Math.max(8, Math.min(128, num));
                e.target.value = String(clamped);
                handleGridResolutionChange(clamped);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  (e.target as HTMLInputElement).blur();
                }
              }}
              style={styles.input}
            />

            <div style={{ ...styles.optionLabel, marginTop: '12px' }}>Visualization:</div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <button
                onClick={() => setGridVisualizationMode('points')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: gridVisualizationMode === 'points' ? '#00ffff' : '#333',
                  borderColor: gridVisualizationMode === 'points' ? '#00ffff' : '#555',
                  color: gridVisualizationMode === 'points' ? '#000' : '#fff',
                }}
              >
                Points
              </button>
              <button
                onClick={() => setGridVisualizationMode('voxels')}
                style={{
                  ...styles.toggleButton,
                  backgroundColor: gridVisualizationMode === 'voxels' ? '#00ffff' : '#333',
                  borderColor: gridVisualizationMode === 'voxels' ? '#00ffff' : '#555',
                  color: gridVisualizationMode === 'voxels' ? '#000' : '#fff',
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
              <span style={{ opacity: 0.6, fontSize: '10px', marginLeft: '4px' }}>
                {typeof navigator !== 'undefined' && 'gpu' in navigator ? '✅' : '❌'}
              </span>
            </label>

            {volumetricGridStats && (
              <div style={styles.statsBox}>
                <div>Resolution: {volumetricGridStats.resolution.x}×{volumetricGridStats.resolution.y}×{volumetricGridStats.resolution.z}</div>
                <div>Mold cells: {volumetricGridStats.moldVolumeCellCount.toLocaleString()}</div>
                <div>Fill ratio: {(volumetricGridStats.fillRatio * 100).toFixed(1)}%</div>
                <div>Compute: {volumetricGridStats.computeTimeMs.toFixed(0)} ms</div>
              </div>
            )}
          </div>
        )}

        {/* Calculate Button and Progress Bar - not shown for import step */}
        {activeStep !== 'import' && status !== 'locked' && (
          <div style={styles.calculateSection}>
            <button
              onClick={handleCalculate}
              disabled={isCalculating || status === 'completed'}
              style={{
                ...styles.calculateButton,
                backgroundColor: status === 'completed' ? '#00aa00' : status === 'needs-recalc' ? '#ff8800' : isCalculating ? '#666' : '#00aaff',
                cursor: isCalculating || status === 'completed' ? 'default' : 'pointer',
              }}
            >
              {status === 'completed' ? '✓ Completed' : status === 'needs-recalc' ? '⟳ Recalculate' : isCalculating ? 'Calculating...' : 'Calculate'}
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
          </div>
        )}
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
          <span style={{ fontSize: '16px' }}>🔧</span>
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
                backgroundColor: isActive ? '#00aaff' : status === 'completed' ? '#00aa00' : status === 'needs-recalc' ? '#ff8800' : 'transparent',
                opacity: status === 'locked' ? 0.3 : 1,
                cursor: status === 'locked' ? 'not-allowed' : 'pointer',
                borderLeft: isActive ? '3px solid #fff' : '3px solid transparent',
              }}
              title={step.title + '\n' + step.description}
            >
              <span style={{ fontSize: '20px' }}>{step.icon}</span>
              {status === 'completed' && (
                <span style={styles.checkMark}>✓</span>
              )}
              {status === 'needs-recalc' && (
                <span style={styles.checkMark}>⟳</span>
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
          gridResolution={gridResolution}
          gridVisualizationMode={gridVisualizationMode}
          useGPUGrid={useGPUGrid}
          showMoldHalfClassification={showMoldHalfClassification}
          onMeshLoaded={handleMeshLoaded}
          onMeshRepaired={handleMeshRepaired}
          onVisibilityDataReady={handleVisibilityDataReady}
          onInflatedHullReady={handleInflatedHullReady}
          onCsgResultReady={handleCsgResultReady}
          onVolumetricGridReady={handleVolumetricGridReady}
          onMoldHalfClassificationReady={handleMoldHalfClassificationReady}
        />
      </div>
    </div>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    width: '100vw',
    height: '100vh',
    margin: 0,
    padding: 0,
    overflow: 'hidden',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  stepsSidebar: {
    width: '10%',
    minWidth: '60px',
    maxWidth: '80px',
    height: '100%',
    backgroundColor: '#1a1a2e',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    paddingTop: '8px',
    borderRight: '1px solid #333',
  },
  sidebarHeader: {
    padding: '12px',
    marginBottom: '8px',
    borderBottom: '1px solid #333',
    width: '100%',
    textAlign: 'center',
  },
  stepIcon: {
    width: '100%',
    padding: '16px 0',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    transition: 'all 0.2s',
  },
  checkMark: {
    position: 'absolute',
    bottom: '4px',
    right: '8px',
    fontSize: '10px',
    color: '#fff',
  },
  helpIcon: {
    marginTop: 'auto',
    padding: '16px',
    fontSize: '18px',
    cursor: 'help',
    opacity: 0.6,
  },
  contextPanel: {
    width: '20%',
    minWidth: '200px',
    maxWidth: '300px',
    height: '100%',
    backgroundColor: '#16213e',
    borderRight: '1px solid #333',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  contextHeader: {
    padding: '16px',
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#fff',
    borderBottom: '1px solid #333',
    backgroundColor: '#1a1a2e',
  },
  contextContent: {
    flex: 1,
    padding: '16px',
    overflowY: 'auto',
    color: '#fff',
  },
  contextTitle: {
    fontSize: '16px',
    fontWeight: 'bold',
    marginBottom: '8px',
  },
  contextDescription: {
    fontSize: '12px',
    opacity: 0.7,
    marginBottom: '16px',
    lineHeight: 1.4,
  },
  optionsSection: {
    marginBottom: '16px',
  },
  optionLabel: {
    fontSize: '11px',
    opacity: 0.8,
    marginBottom: '6px',
  },
  input: {
    width: '100%',
    padding: '8px 12px',
    backgroundColor: '#222',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '13px',
    boxSizing: 'border-box',
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    cursor: 'pointer',
    marginTop: '8px',
  },
  toggleButton: {
    flex: 1,
    padding: '8px 12px',
    border: '2px solid',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    transition: 'all 0.2s',
  },
  statsBox: {
    marginTop: '12px',
    padding: '10px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '4px',
    fontSize: '11px',
    lineHeight: 1.6,
  },
  legend: {
    marginTop: '8px',
    fontSize: '10px',
    opacity: 0.7,
  },
  calculateSection: {
    marginTop: '20px',
    paddingTop: '16px',
    borderTop: '1px solid #333',
  },
  calculateButton: {
    width: '100%',
    padding: '12px 16px',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '14px',
    fontWeight: 'bold',
    transition: 'background-color 0.2s',
  },
  progressContainer: {
    marginTop: '10px',
    height: '6px',
    backgroundColor: '#333',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#00aaff',
    transition: 'width 0.1s',
  },
  lockedMessage: {
    padding: '16px',
    backgroundColor: 'rgba(255, 165, 0, 0.1)',
    borderRadius: '4px',
    fontSize: '12px',
    textAlign: 'center',
    color: '#f90',
  },
  viewerContainer: {
    flex: 1,
    height: '100%',
    position: 'relative',
    backgroundColor: '#000',
  },
};

export default App;
