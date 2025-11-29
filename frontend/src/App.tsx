/**
 * VcMoldCreator - Main Application
 * 
 * A tool for analyzing 3D models for two-piece mold manufacturing.
 * Computes optimal parting directions and visualizes surface visibility.
 */

import { useState, useCallback } from 'react';
import './App.css';
import ThreeViewer from './components/ThreeViewer';
import FileUpload from './components/FileUpload';
import type { VisibilityPaintData } from './utils/partingDirection';
import type { InflatedHullResult, ManifoldValidationResult, CsgSubtractionResult } from './utils/inflatedBoundingVolume';
import type { MeshRepairResult, MeshDiagnostics } from './utils/meshRepairManifold';

// ============================================================================
// COMPONENT
// ============================================================================

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

  const togglePartingDirections = useCallback(() => {
    setShowPartingDirections(prev => {
      if (prev) {
        setShowD1Paint(false);
        setShowD2Paint(false);
        setVisibilityDataReady(false);
      }
      return !prev;
    });
  }, []);

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
        onMeshLoaded={handleMeshLoaded}
        onMeshRepaired={handleMeshRepaired}
        onVisibilityDataReady={handleVisibilityDataReady}
        onInflatedHullReady={handleInflatedHullReady}
        onCsgResultReady={handleCsgResultReady}
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
        </div>
      )}

      {/* Controls Help */}
      <div style={styles.helpPanel}>
        <div><strong>Controls:</strong></div>
        <div>🖱️ Left-click + drag → Orbit</div>
        <div>🖱️ Right-click + drag → Pan</div>
        <div>🖱️ Scroll → Zoom</div>
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
