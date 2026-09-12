import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { suggestionMiddleware } from './server/suggestions.mjs'

export default defineConfig({
  plugins: [react(), {
    name: 'clearspeak-suggestions',
    configureServer(server) {
      server.middlewares.use(suggestionMiddleware(server.config.root))
    },
    configurePreviewServer(server) {
      server.middlewares.use(suggestionMiddleware(server.config.root))
    },
  }],
  server: { host: '127.0.0.1' },
  preview: { host: '127.0.0.1' },
})
