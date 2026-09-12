// Verifies the welcome overlay and polished UI render correctly.

import { chromium } from 'playwright'

const URL = process.env.URL ?? 'http://127.0.0.1:5184'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })

const errors = []
page.on('pageerror', (error) => errors.push(error.message))
page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })

await page.goto(URL, { waitUntil: 'networkidle' })

// Welcome overlay should be visible on first load
const overlay = page.locator('.welcome-overlay')
console.log('welcome overlay:', (await overlay.count()) ? 'visible' : 'MISSING')

const title = await page.locator('.welcome-title').first().innerText().catch(() => '(missing)')
console.log('welcome title  :', title)

const steps = await page.locator('.welcome-step').count()
console.log('welcome steps  :', steps)

// Click "Begin" to dismiss overlay and open the calibration guide
await page.locator('.welcome-start').click()
await page.waitForTimeout(800)
console.log('overlay after  :', (await overlay.count()) ? 'still visible' : 'dismissed')

// Guide should now be open
const guide = page.locator('.calibration-guide')
console.log('guide visible  :', (await guide.count()) ? 'yes' : 'NO')

// Skip to the board
await page.locator('.guide-skip').click()
await page.waitForTimeout(600)

const kickers = (await page.locator('.section-kicker').allInnerTexts()).map((k) => k.toLowerCase())
console.log('panels         :', kickers.join(' | '))
console.log('has beacon     :', kickers.includes('caregiver beacon') ? 'yes' : 'NO')

const tiles = await page.locator('[data-dwell-target]').count()
console.log('tiles rendered :', tiles)

// Test the beacon wiring still works
await page.locator('[data-dwell-target="Help"]').click()
await page.waitForTimeout(300)
const spoken = await page.locator('.sentence-bar span').first().innerText()
console.log('after Help     :', spoken)

// Take a screenshot for visual reference
await page.screenshot({ path: '/tmp/clearspeak-polished.png', fullPage: false })
console.log('screenshot     : /tmp/clearspeak-polished.png')

console.log('errors         :', errors.length ? errors.join(' // ') : 'none')

await browser.close()
