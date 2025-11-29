import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/manifold-3d/manifold.wasm',
          dest: 'assets'
        }
      ]
    })
  ],
  resolve: {
    alias: {
      'react': path.resolve(__dirname, 'node_modules/react'),
      'react-dom': path.resolve(__dirname, 'node_modules/react-dom'),
    }
  },
  optimizeDeps: {
    exclude: ['manifold-3d'],
  },
  build: {
    rollupOptions: {
      // Ensure WASM files are handled properly
      output: {
        assetFileNames: 'assets/[name].[ext]'
      }
    }
  }
})
