import { useState, useCallback } from 'react'
import './App.css'
import ThreeViewer from './components/ThreeViewer'
import FileUpload from './components/FileUpload'

function App() {
  const [stlUrl, setStlUrl] = useState<string | undefined>(undefined);

  const handleFileLoad = useCallback((url: string, _fileName: string) => {
    // Revoke previous URL to prevent memory leaks
    if (stlUrl) {
      URL.revokeObjectURL(stlUrl);
    }
    setStlUrl(url);
  }, [stlUrl]);

  return (
    <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0, position: 'relative' }}>
      <ThreeViewer stlUrl={stlUrl} />
      <FileUpload onFileLoad={handleFileLoad} />
      
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
