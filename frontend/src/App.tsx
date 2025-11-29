import { useState, useCallback } from 'react'
import './App.css'
import ThreeViewer from './components/ThreeViewer'
import FileUpload from './components/FileUpload'

function App() {
  const [stlUrl, setStlUrl] = useState<string | undefined>(undefined);
  const [showPartingDirections, setShowPartingDirections] = useState(false);
  const [meshLoaded, setMeshLoaded] = useState(false);

  const handleFileLoad = useCallback((url: string, _fileName: string) => {
    // Revoke previous URL to prevent memory leaks
    if (stlUrl) {
      URL.revokeObjectURL(stlUrl);
    }
    setStlUrl(url);
    setShowPartingDirections(false); // Reset when new file loaded
    setMeshLoaded(false);
  }, [stlUrl]);

  const handleMeshLoaded = useCallback(() => {
    setMeshLoaded(true);
  }, []);

  const togglePartingDirections = useCallback(() => {
    setShowPartingDirections(prev => !prev);
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0, position: 'relative' }}>
      <ThreeViewer 
        stlUrl={stlUrl} 
        showPartingDirections={showPartingDirections}
        onMeshLoaded={handleMeshLoaded}
      />
      <FileUpload onFileLoad={handleFileLoad} />
      
      {/* Analysis controls */}
      {meshLoaded && (
        <div style={{
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
        }}>
          <div style={{ marginBottom: '12px', fontWeight: 'bold' }}>🔧 Mold Analysis</div>
          <button
            onClick={togglePartingDirections}
            style={{
              padding: '8px 16px',
              backgroundColor: showPartingDirections ? '#00aa00' : '#00aaff',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '12px',
              width: '100%',
              transition: 'background-color 0.2s',
            }}
          >
            {showPartingDirections ? '✓ Parting Directions ON' : 'Show Parting Directions'}
          </button>
          {showPartingDirections && (
            <div style={{ marginTop: '10px', fontSize: '11px', opacity: 0.8 }}>
              <div>🟢 Green: Best direction</div>
              <div>🟠 Orange: Second best</div>
            </div>
          )}
        </div>
      )}
      
      {/* Controls help overlay */}
      <div style={{
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
      }}>
        <div><strong>Controls:</strong></div>
        <div>🖱️ Left-click + drag → Orbit</div>
        <div>🖱️ Right-click + drag → Pan</div>
        <div>🖱️ Scroll → Zoom</div>
      </div>
    </div>
  )
}

export default App
