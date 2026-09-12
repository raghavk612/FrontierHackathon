// Renders the app with a synthetic webcam and reports the stage geometry, so layout
// regressions show up as numbers rather than needing a human to eyeball them.
// Run with: node scripts/layout-check.mjs

import { chromium } from 'playwright'

const VIEWPORTS = [
  { name: 'laptop 1440x900', width: 1440, height: 900 },
  { name: 'desktop 1920x1080', width: 1920, height: 1080 },
  { name: 'small 1280x720', width: 1280, height: 720 },
]

const browser = await chromium.launch({
  args: [
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    '--allow-file-access-from-files',
  ],
})

let failures = 0

for (const viewport of VIEWPORTS) {
  const context = await browser.newContext({
    viewport: { width: viewport.width, height: viewport.height },
    permissions: ['camera'],
  })
  const page = await context.newPage()
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
  await page.click('.primary-button')
  await page.waitForTimeout(2500)

  const geometry = await page.evaluate(() => {
    const stage = document.querySelector('.stage')
    const video = document.querySelector('.camera-feed')
    const panel = document.querySelector('.control-panel')
    const stageBox = stage.getBoundingClientRect()
    const panelBox = panel.getBoundingClientRect()
    return {
      stageWidth: Math.round(stageBox.width),
      stageHeight: Math.round(stageBox.height),
      panelHeight: Math.round(panelBox.height),
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      documentScrolls: document.documentElement.scrollHeight > window.innerHeight + 2,
    }
  })

  const ratio = geometry.stageWidth / geometry.stageHeight
  const landscape = ratio > 1.4
  if (!landscape) failures += 1

  console.log(
    `${viewport.name.padEnd(20)} stage ${String(geometry.stageWidth).padStart(4)}x${String(geometry.stageHeight).padEnd(4)} ` +
      `ratio ${ratio.toFixed(2)} ${landscape ? 'landscape OK' : 'NOT LANDSCAPE'} | ` +
      `camera ${geometry.videoWidth}x${geometry.videoHeight} | page scrolls: ${geometry.documentScrolls}`,
  )

  // Reveal the board and confirm the bands stack without colliding.
  await page.click('.text-button')
  await page.waitForTimeout(500)

  const bands = await page.evaluate(() => {
    const stage = document.querySelector('.stage').getBoundingClientRect()
    const read = (selector) => {
      const element = document.querySelector(selector)
      if (!element) return null
      const box = element.getBoundingClientRect()
      return { top: ((box.top - stage.top) / stage.height) * 100, bottom: ((box.bottom - stage.top) / stage.height) * 100 }
    }
    const sides = (selector) => {
      const element = document.querySelector(selector)
      if (!element) return null
      const box = element.getBoundingClientRect()
      return { left: ((box.left - stage.left) / stage.width) * 100, right: ((box.right - stage.left) / stage.width) * 100 }
    }
    return {
      sentence: read('.sentence-bar'),
      grid: read('.board-grid'),
      gridSides: sides('.board-grid'),
      restLeft: sides('.rest-zone.left'),
      restRight: sides('.rest-zone.right'),
      tileHeight: Math.round(document.querySelector('.board-grid .dwell-target').getBoundingClientRect().height),
    }
  })

  const overlaps = []
  const order = [
    ['sentence', bands.sentence],
    ['grid', bands.grid],
  ]
  for (let i = 0; i < order.length - 1; i += 1) {
    if (order[i][1].bottom > order[i + 1][1].top + 0.1) {
      overlaps.push(`${order[i][0]} overruns ${order[i + 1][0]}`)
      failures += 1
    }
  }
  // Rest zones now flank the board rather than sitting under it, so the collision that
  // matters is horizontal: a tile reaching into a zone makes resting select something.
  if (!bands.restLeft || !bands.restRight) {
    overlaps.push('rest zones missing')
    failures += 1
  } else {
    if (bands.gridSides.left < bands.restLeft.right - 0.1) {
      overlaps.push('grid overruns left rest zone')
      failures += 1
    }
    if (bands.gridSides.right > bands.restRight.left + 0.1) {
      overlaps.push('grid overruns right rest zone')
      failures += 1
    }
    // Too thin and the user cannot reliably park in it; the zones are edge-anchored so
    // this only has to exceed the horizontal pointing error, not double it.
    if (bands.restLeft.right < 4 || 100 - bands.restRight.left < 4) {
      overlaps.push('rest zone too narrow')
      failures += 1
    }
  }
  console.log(
    `${' '.repeat(20)} bands ${order.map(([name, b]) => `${name} ${b.top.toFixed(0)}-${b.bottom.toFixed(0)}%`).join('  ')}` +
      `  rest sides 0-${bands.restLeft ? bands.restLeft.right.toFixed(1) : '?'}% / ` +
      `${bands.restRight ? bands.restRight.left.toFixed(1) : '?'}-100%` +
      ` | tile ${bands.tileHeight}px ${overlaps.length ? '| ' + overlaps.join(', ') : '| no overlap'}`,
  )

  // The number that decides whether the board is usable: with nearest-word targeting
  // you only have to land closer to the right word than to any other, so the accuracy
  // budget is half the gap between neighbouring word centres, not half a word.
  const tolerance = await page.evaluate(() => {
    const stage = document.querySelector('.stage').getBoundingClientRect()
    const targets = [...document.querySelectorAll('[data-dwell-target]')].map((element) => {
      const box = element.getBoundingClientRect()
      return {
        label: element.dataset.dwellTarget,
        x: ((box.left + box.width / 2 - stage.left) / stage.width) * 100,
        y: ((box.top + box.height / 2 - stage.top) / stage.height) * 100,
        halfBox: Math.min((box.width / stage.width) * 100, (box.height / stage.height) * 100) / 2,
      }
    })

    // Reported per axis, because vertical gaze is about twice as noisy as horizontal
    // and a single combined figure hides which of the two the board is short on.
    let across = Infinity
    let down = Infinity
    for (const target of targets) {
      for (const other of targets) {
        if (other === target) continue
        const dx = Math.abs(target.x - other.x)
        const dy = Math.abs(target.y - other.y)
        // Neighbours in a row share a y; neighbours in a column share an x.
        if (dy < 1) across = Math.min(across, dx / 2)
        if (dx < 1) down = Math.min(down, dy / 2)
      }
    }
    const worstContainment = Math.min(...targets.map((target) => target.halfBox))
    return { count: targets.length, across, down, worstContainment }
  })

  console.log(
    `${' '.repeat(20)} ${tolerance.count} dwell targets | budget ${tolerance.across.toFixed(1)}% across, ` +
      `${tolerance.down.toFixed(1)}% down | ${tolerance.worstContainment.toFixed(1)}% if it had to land inside the word`,
  )
  // Vertical gaze runs about 1.9x the horizontal error, so the down budget has to be
  // proportionally larger for the board to be usable on both axes at once.
  if (tolerance.across < 6 || tolerance.down < tolerance.across * 1.9) failures += 1

  await page.screenshot({ path: `/tmp/layout-${viewport.width}.png` })
  await context.close()
}

await browser.close()
console.log(failures === 0 ? '\nall viewports landscape' : `\n${failures} viewport(s) not landscape`)
process.exit(failures === 0 ? 0 : 1)
