// Headless Chromium ships no speech voices at all, so a real audio test is impossible
// here. Instead this installs a fake synthesiser with the exact shape that goes silent on
// a bad network: a remote "Google" voice that accepts speak() and then never starts.
import { chromium } from 'playwright'

const URL = process.env.URL ?? 'http://127.0.0.1:5184/'
const browser = await chromium.launch()
const page = await browser.newPage()

await page.addInitScript(() => {
  const voices = [
    { name: 'Google US English', lang: 'en-US', localService: false, default: true },
    { name: 'Samantha', lang: 'en-US', localService: true, default: false },
    { name: 'Anna', lang: 'de-DE', localService: true, default: false },
  ]
  window.__heard = []
  window.__attempts = []
  const synth = {
    speaking: false,
    pending: false,
    paused: false,
    getVoices: () => voices,
    addEventListener() {},
    removeEventListener() {},
    cancel() { this.speaking = false },
    resume() { this.paused = false },
    speak(utterance) {
      const voice = utterance.voice ? utterance.voice.name : '(browser default)'
      if (!utterance.text.trim()) return          // the silent primer
      window.__attempts.push(voice)
      // The remote voice behaves the way it does offline: accepted, then nothing.
      if (voice === 'Google US English') return
      this.speaking = true
      window.__heard.push(`${voice}: ${utterance.text}`)
      setTimeout(() => { this.speaking = false; utterance.onstart?.(); utterance.onend?.() }, 10)
    },
  }
  Object.defineProperty(window, 'speechSynthesis', { value: synth, configurable: true })
  window.SpeechSynthesisUtterance = class {
    constructor(text) { this.text = text; this.voice = null; this.rate = 1; this.volume = 1; this.lang = '' }
  }
})

const errors = []
page.on('pageerror', (error) => errors.push(error.message))

await page.goto(URL, { waitUntil: 'networkidle' })
await page.locator('.welcome-skip').click()
await page.waitForTimeout(300)
const skip = page.locator('.guide-skip')
if (await skip.count()) { await skip.first().click(); await page.waitForTimeout(300) }

await page.locator('[data-dwell-target="Yes"]').click()
await page.waitForTimeout(200)
await page.locator('[data-dwell-target="Speak"]').click()
await page.waitForTimeout(1600)

console.log('voices offered   :', (await page.evaluate(() => window.__attempts)).length ? 'yes' : 'none')
console.log('tried in order   :', (await page.evaluate(() => window.__attempts)).join(' then '))
console.log('ACTUALLY HEARD   :', JSON.stringify(await page.evaluate(() => window.__heard)))
console.log('bar after 3.5s   : waiting…')
await page.waitForTimeout(2600)
const placeholder = page.locator('.sentence-placeholder')
console.log('bar cleared      :', (await placeholder.innerText()).trim() === 'Choose words, then Speak' ? 'yes, back to prompt' : `NO -> ${await placeholder.innerText()}`)
console.log('errors           :', errors.length ? errors.join(' | ') : 'none')

// Now the backstop: force the remote voice as a saved preference and confirm the board
// notices the silence and re-speaks the same sentence on the on-device voice.
console.log('\n-- with the network voice deliberately chosen --')
await page.evaluate(() => {
  localStorage.setItem('clearspeak.voice', 'Google US English')
  window.__heard = []
  window.__attempts = []
})
await page.reload({ waitUntil: 'networkidle' })
await page.locator('.welcome-skip').click()
await page.waitForTimeout(300)
if (await page.locator('.guide-skip').count()) { await page.locator('.guide-skip').first().click(); await page.waitForTimeout(300) }
await page.locator('[data-dwell-target="Help"]').click()
await page.waitForTimeout(200)
await page.locator('[data-dwell-target="Speak"]').click()
await page.waitForTimeout(2000)
console.log('tried in order   :', (await page.evaluate(() => window.__attempts)).join(' then '))
console.log('ACTUALLY HEARD   :', JSON.stringify(await page.evaluate(() => window.__heard)))
const note = page.locator('.panel-copy.warn')
console.log('told the user    :', (await note.count()) ? (await note.last().innerText()).trim() : '(no note)')

await browser.close()
