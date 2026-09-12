// Verifies the merge: the newer app code and the beacon layer both survived.
//
// The two halves came from different places and neither side's tests would have caught
// the other going missing, so this checks one representative thing from each.

import { chromium } from 'playwright'

const URL = process.env.URL ?? 'http://127.0.0.1:5184'

const browser = await chromium.launch()
const page = await browser.newPage()

const errors = []
page.on('pageerror', (error) => errors.push(error.message))
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text())
})

await page.goto(URL, { waitUntil: 'networkidle' })

const text = (selector) => page.locator(selector).first().innerText().catch(() => '(missing)')

console.log('eyebrow      :', await text('.eyebrow'))
console.log('heading      :', await text('h1'))

// Panel names are uppercased by CSS, so compare on a folded copy rather than what the
// accessibility tree hands back.
const kickers = (await page.locator('.section-kicker').allInnerTexts()).map((entry) => entry.toLowerCase())
console.log('panels       :', kickers.join(' | '))

for (const needed of ['caregiver beacon', 'say something', 'suggested answers']) {
  console.log(`has "${needed}"`.padEnd(26), ':', kickers.includes(needed) ? 'yes' : 'NO  <-- MISSING')
}

const pairButton = page.getByRole('button', { name: 'Pair over Bluetooth' })
console.log('pair button  :', (await pairButton.count()) ? 'present' : 'MISSING')
console.log('usb button   :', (await page.getByRole('button', { name: 'USB', exact: true }).count()) ? 'present' : 'MISSING')

// The board is deliberately hidden behind the calibration guide on first load, so get
// past it the way a user without a working camera would.
await page.locator('.guide-skip').click()
await page.waitForTimeout(600)
console.log('board shown  :', (await page.locator('.board-grid').count()) ? 'yes' : 'NO  <-- still gated')

// The beacon decides urgent vs ordinary purely from this attribute, so if the merge had
// dropped it every alert would silently downgrade to the quiet pulse.
const accents = await page.locator('[data-dwell-target]').evaluateAll((nodes) =>
  nodes.map((node) => `${node.dataset.dwellTarget}=${node.dataset.dwellAccent}`),
)
console.log('tile accents :', accents.join('  '))

const urgentCount = accents.filter((entry) => entry.endsWith('=urgent')).length
console.log('urgent tiles :', urgentCount, urgentCount > 0 ? '' : ' <-- MISSING, beacon would never alarm')

await page.locator('[data-dwell-target="Help"]').click()
await page.waitForTimeout(400)
console.log('after Help   :', await text('.sentence-bar'))

console.log('errors       :', errors.length ? errors.join(' // ') : 'none')

await browser.close()
