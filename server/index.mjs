import { createServer } from 'node:http'
import { fileURLToPath } from 'node:url'
import { suggestionMiddleware } from './suggestions.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const allowedOrigins = (process.env.ALLOWED_ORIGINS || 'https://raghavk612.github.io').split(',').map(value => value.trim()).filter(Boolean)
const api = suggestionMiddleware(root, undefined, { allowedOrigins })
const server = createServer((req, res) => {
  res.setHeader('X-Content-Type-Options', 'nosniff')
  if (req.url === '/health' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    return res.end('{"ok":true}')
  }
  void api(req, res, () => { res.writeHead(404); res.end('Not found') })
})
server.requestTimeout = 15000
server.headersTimeout = 10000
server.listen(Number(process.env.PORT || 3001), '0.0.0.0', () => {
  console.log('ClearSpeak replies backend is listening')
})
process.on('SIGTERM', () => server.close())
