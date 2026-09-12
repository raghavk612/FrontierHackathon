import test from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { generateSuggestions, suggestionMiddleware, validateInput } from './suggestions.mjs'

const options = { words: ['Tell me more', 'Maybe later', 'Sounds good', 'Something else', 'Please repeat'].map(label => ({ label, speech: label })), because: 'about your plans' }
test('structured replies use server credentials and disable response storage', async () => {
  const result = await generateSuggestions({ text: 'We could go outside.', context: [] }, { key: 'test-only', model: 'gpt-4.1-mini' }, async (url, init) => {
    assert.equal(url, 'https://api.openai.com/v1/responses')
    assert.equal(init.headers.Authorization, 'Bearer test-only')
    const body = JSON.parse(init.body)
    assert.equal(body.store, false)
    assert.equal(body.text.format.strict, true)
    assert.match(body.instructions, /whether a question or a statement/)
    return { ok: true, json: async () => ({ output: [{ content: [{ type: 'output_text', text: JSON.stringify(options) }] }] }) }
  })
  assert.deepEqual(result, options)
})
test('refusals and duplicate or fixed-button replies fail safely', async () => {
  for (const value of [{}, { ...options, words: options.words.map(() => ({ label: 'Yes', speech: 'Yes' })) }]) {
    await assert.rejects(generateSuggestions({ text: 'Hello' }, { key: 'test', model: 'test' }, async () => ({ ok: true, json: async () => ({ output: [{ content: [{ type: 'output_text', text: JSON.stringify(value) }] }] }) })))
  }
})
test('bounds context and rejects empty or oversized utterances', () => {
  assert.equal(validateInput({ text: '' }), null)
  assert.equal(validateInput({ text: 'a'.repeat(1501) }), null)
  const input = validateInput({ text: 'Hello', context: Array.from({ length: 20 }, () => ({ speaker: 'them', text: 'x'.repeat(1000) })) })
  assert.equal(input.context.length, 6)
  assert.equal(input.context[0].text.length, 500)
})
test('HTTP endpoint rejects cross-origin and invalid requests, hides key, and returns replies', async () => {
  const root = mkdtempSync(join(tmpdir(), 'clearspeak-test-'))
  writeFileSync(join(root, '.env.local'), 'OPENAI_API_KEY=test-only\n')
  let calls = 0
  const middleware = suggestionMiddleware(root, async () => { calls++; return options })
  const server = createServer((req, res) => middleware(req, res, () => { res.writeHead(404); res.end() }))
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${server.address().port}`
  try {
    assert.deepEqual(await (await fetch(base + '/api/suggestions/status')).json(), { configured: true })
    const post = (body, headers = {}) => fetch(base + '/api/suggestions', { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body })
    assert.equal((await post('{}', { Origin: 'https://unrelated.example' })).status, 403)
    assert.equal((await post('{')).status, 400)
    assert.equal((await post(JSON.stringify({ text: '' }))).status, 400)
    assert.equal((await post('a'.repeat(13000))).status, 413)
    const response = await post(JSON.stringify({ text: 'I had a nice morning.' }))
    assert.deepEqual(await response.json(), options)
    assert.equal(calls, 1)
  } finally { await new Promise(resolve => server.close(resolve)); rmSync(root, { recursive: true }) }
})

test('quota failures report an actionable message without leaking upstream details', async () => {
  await assert.rejects(
    generateSuggestions({ text: 'Hello' }, { key: 'test-only', model: 'test' }, async () => ({
      ok: false, status: 429,
      json: async () => ({ error: { type: 'insufficient_quota', code: 'credit_balance_exhausted', message: 'private upstream details' } }),
    })),
    error => error.safeForClient && error.message === 'OpenAI credits exhausted · using local replies',
  )
})
