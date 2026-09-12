// Renders wiring-diagram.html to a PNG so the layout can be read on a phone at the table.
import { chromium } from 'playwright'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const source = path.join(here, '..', 'wiring-diagram.html')
const target = path.join(here, '..', 'wiring.png')

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1240, height: 900 }, deviceScaleFactor: 2 })
await page.goto(`file://${source}`)
await page.waitForTimeout(250)
await page.locator('#sheet').screenshot({ path: target })
await browser.close()
console.log('wrote', target)
