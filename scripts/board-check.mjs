// Verifies the sidebar-free board: sentence composition, head-operable Speak and
// Delete, word-level deletion, and that the transcription rail survived.

import { chromium } from 'playwright'

const URL = process.env.URL ?? 'http://127.0.0.1:5184'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })

const errors = []
page.on('pageerror', (error) => errors.push(error.message))
page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })

// Headless Chromium has no voices, so asserting on audio is impossible. Recording the
// calls instead is what actually catches the bug: speak() living inside a setState
// updater looked fine in every DOM assertion while never reliably firing.
await page.addInitScript(() => {
  window.__spoken = []
  const real = window.speechSynthesis.speak.bind(window.speechSynthesis)
  window.speechSynthesis.speak = (utterance) => {
    window.__spoken.push(utterance.text)
    return real(utterance)
  }
})

await page.goto(URL, { waitUntil: 'networkidle' })
await page.locator('.welcome-start').click()
await page.waitForTimeout(400)

// Beacon pairing has to have survived the move into the guide, or the demo loses its
// hardware beat with no warning.
console.log('guide has beacon :', (await page.getByRole('button', { name: 'Pair over Bluetooth' }).count()) ? 'yes' : 'NO')

await page.locator('.guide-skip').click()
await page.waitForTimeout(500)

console.log('sidebar gone     :', (await page.locator('.control-panel').count()) === 0 ? 'yes' : 'NO  <-- still there')
console.log('transcript rail  :', (await page.locator('.transcript-rail').count()) ? 'present' : 'MISSING')

// Everything that can be selected must be reachable by head, which means it must carry
// data-dwell-target. Anything missing from this list is mouse-only.
const targets = await page.locator('[data-dwell-target]').evaluateAll((nodes) =>
  nodes.map((node) => node.dataset.dwellTarget),
)
console.log('dwell targets    :', targets.join(', '))
for (const needed of ['Speak', 'Delete']) {
  console.log(`  "${needed}" aimable`.padEnd(19), ':', targets.includes(needed) ? 'yes' : 'NO  <-- mouse only')
}

const words = () => page.locator('.sentence-word').allInnerTexts()

// Plain word tiles, not folders. A folder tile navigates instead of composing, which is
// correct but makes for a useless assertion.
await page.locator('[data-dwell-target="Yes"]').click()
await page.waitForTimeout(250)
await page.locator('[data-dwell-target="Help"]').click()
await page.waitForTimeout(250)
await page.locator('[data-dwell-target="No"]').click()
await page.waitForTimeout(250)
console.log('after 3 picks    :', (await words()).join(' + ') || '(empty)')

// A folder tile should navigate, never land in the sentence.
await page.locator('[data-dwell-target="I want"]').click()
await page.waitForTimeout(300)
console.log('folder navigates :', (await words()).length === 3 ? 'yes, sentence untouched' : 'NO  <-- folder added a word')

// Delete mode should hand the whole board grid over to the sentence words, at full tile
// size, because chips in the sentence bar were far below the accuracy budget.
await page.locator('[data-dwell-target="Delete"]').click()
await page.waitForTimeout(350)
const removable = await page.locator('.tile-remove').count()
console.log('removable tiles  :', removable, removable === 3 ? '(one per word)' : ' <-- expected 3')
console.log('done tile        :', (await page.locator('[data-dwell-target="Done deleting"]').count()) ? 'present' : 'MISSING')

const tileBox = await page.locator('.tile-remove').first().boundingBox()
console.log('tile size        :', `${Math.round(tileBox.width)}x${Math.round(tileBox.height)} px`)

// Delete the middle word specifically; index-based removal is the whole reason the
// sentence carries positions rather than labels.
await page.locator('.tile-remove').nth(1).click()
await page.waitForTimeout(350)
console.log('after deleting #2:', (await words()).join(' + ') || '(empty)')

await page.locator('[data-dwell-target="Done deleting"]').click()
await page.waitForTimeout(300)
console.log('board restored   :', (await page.locator('[data-dwell-target="Help"]').count()) ? 'yes' : 'NO')

await page.locator('[data-dwell-target="Speak"]').click()
await page.waitForTimeout(400)
console.log('after Speak      :', (await words()).join(' + ') || '(sentence cleared)')
console.log('SPOKE ALOUD      :', JSON.stringify(await page.evaluate(() => window.__spoken)))

// Speaking must remove the composed words immediately. The transcript preserves what was
// said, so the composer should go straight back to its empty prompt.
console.log('bar now reads    :', (await page.locator('.sentence-placeholder').first().innerText()).replace(/\n/g, ' '))
console.log('cleared at once  :', (await page.locator('.sentence-placeholder').first().innerText()).trim() === 'Choose words, then Speak' ? 'yes' : 'NO')
await page.locator('[data-dwell-target="Delete"]').click()
await page.waitForTimeout(300)
console.log('delete on empty  :', await page.locator('.stage-label span').innerText())

// And a fresh word must still be deletable afterwards.
await page.locator('[data-dwell-target="Pain"]').click()
await page.waitForTimeout(250)
await page.locator('[data-dwell-target="Delete"]').click()
await page.waitForTimeout(350)
console.log('delete on 1 word :', (await page.locator('.tile-remove').count()) === 1 ? 'one tile offered' : 'NO tiles')
await page.locator('.tile-remove').first().click()
await page.waitForTimeout(350)
console.log('removed last one :', (await words()).length === 0 ? 'empty, mode exited' : 'still has words')
console.log('back to board    :', (await page.locator('[data-dwell-target="Help"]').count()) ? 'yes' : 'NO')

await page.screenshot({ path: '/tmp/clearspeak-board.png' })
console.log('errors           :', errors.length ? errors.join(' // ') : 'none')
await browser.close()
