import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    watch: {
      ignored: [
        '**/.venv/**',
        '**/backend/data/**',
        '**/backend/models/**',
        '**/backend/**/*.pyc',
        '**/backend/**/__pycache__/**',
      ],
    },
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
