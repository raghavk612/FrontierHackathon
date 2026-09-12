import { execFileSync, spawn } from 'node:child_process'

let activeSpeech = null

function installedVoice() {
  if (process.platform !== 'darwin') return null
  try {
    const voices = execFileSync('/usr/bin/say', ['-v', '?'], { encoding: 'utf8' })
    for (const preferred of ['Samantha', 'Alex', 'Ava']) {
      if (voices.split('\n').some((line) => line.startsWith(`${preferred} `))) return preferred
    }
  } catch {
    // Using no -v argument below falls back to the Mac's configured system voice.
  }
  return null
}

const voice = installedVoice()

export function speechMiddleware(req, res, next) {
  if (req.method !== 'POST' || req.url?.split('?')[0] !== '/api/speak') {
    next()
    return
  }

  let body = ''
  req.setEncoding('utf8')
  req.on('data', (chunk) => {
    body += chunk
    if (body.length > 10_000) req.destroy()
  })
  req.on('end', () => {
    try {
      const text = JSON.parse(body).text?.trim()
      if (typeof text !== 'string' || text.length === 0 || text.length > 500) {
        res.statusCode = 400
        res.end(JSON.stringify({ error: 'Speech text must contain 1–500 characters.' }))
        return
      }
      if (process.platform !== 'darwin') {
        res.statusCode = 501
        res.end(JSON.stringify({ error: 'Native speaker output is available on macOS.' }))
        return
      }

      activeSpeech?.kill()
      const args = voice ? ['-v', voice, text] : [text]
      activeSpeech = spawn('/usr/bin/say', args, {
        stdio: 'ignore',
        detached: false,
      })
      activeSpeech.once('exit', () => { activeSpeech = null })
      activeSpeech.once('error', () => { activeSpeech = null })

      res.setHeader('Content-Type', 'application/json')
      res.end(JSON.stringify({ ok: true, voice: voice ?? 'system default' }))
    } catch {
      res.statusCode = 400
      res.end(JSON.stringify({ error: 'Invalid speech request.' }))
    }
  })
}
