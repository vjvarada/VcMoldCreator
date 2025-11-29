// File upload component for STL files
// Allows users to drag-and-drop or click to select STL files

import React, { useCallback, useRef, useState } from 'react';

interface FileUploadProps {
  onFileLoad: (fileUrl: string, fileName: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileLoad }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.name.toLowerCase().endsWith('.stl')) {
      alert('Please upload an STL file');
      return;
    }

    setFileName(file.name);
    const url = URL.createObjectURL(file);
    onFileLoad(url, file.name);
  }, [onFileLoad]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  return (
    <div
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        padding: '15px 25px',
        backgroundColor: isDragging ? 'rgba(0, 170, 255, 0.3)' : 'rgba(255, 255, 255, 0.1)',
        border: `2px dashed ${isDragging ? '#00aaff' : 'rgba(255, 255, 255, 0.3)'}`,
        borderRadius: '8px',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        zIndex: 1000,
        color: '#fff',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".stl"
        onChange={handleInputChange}
        style={{ display: 'none' }}
      />
      <div style={{ textAlign: 'center' }}>
        {fileName ? (
          <>
            <div style={{ fontSize: '14px', marginBottom: '4px' }}>✓ Loaded:</div>
            <div style={{ fontSize: '12px', opacity: 0.8 }}>{fileName}</div>
          </>
        ) : (
          <>
            <div style={{ fontSize: '14px', marginBottom: '4px' }}>📁 Upload STL File</div>
            <div style={{ fontSize: '11px', opacity: 0.6 }}>Click or drag & drop</div>
          </>
        )}
      </div>
    </div>
  );
};

export default FileUpload;
