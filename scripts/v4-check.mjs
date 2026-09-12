// Loads version 4, confirms the beacon panel is present and nothing throws.
import { chromium } from 'playwright'

const browser = await chromium.launch({
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
})
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text())
})

await page.goto('http://localhost:5184/', { waitUntil: 'networkidle' })
await page.waitForTimeout(900)

const heading = await page.locator('h1').first().innerText()
const kickers = await page.locator('.section-kicker').allInnerTexts()
const tiles = await page.locator('.dwell-target').count()
const beaconText = await page.locator('.panel-section', { hasText: 'caregiver beacon' }).innerText()

console.log('heading     :', heading)
console.log('tiles       :', tiles)
console.log('beacon panel:', kickers.includes('caregiver beacon'))
console.log('---- beacon panel text ----')
console.log(beaconText.trim())
console.log('---------------------------')
// Reveal the board, then confirm the tiles carry the accent the beacon routes on.
await page.getByRole('button', { name: 'Skip' }).click()
await page.waitForTimeout(400)

const accents = await page.$$eval('.board-grid .dwell-target, .urgent-row .dwell-target', (nodes) =>
  nodes.map((n) => `${n.textContent.trim()}=${n.dataset.dwellAccent}`),
)
console.log('tile accents:', accents.join('  '))

await page.locator('[data-dwell-target="Help"]').click()
await page.waitForTimeout(400)
const said = await page.locator('.sentence-bar').innerText()
console.log('after clicking Help:', said.replace(/\s+/g, ' ').trim())
console.log('errors after click :', errors.length ? errors : 'none')

await page.screenshot({ path: '/tmp/v4.png', fullPage: false })
await browser.close()
