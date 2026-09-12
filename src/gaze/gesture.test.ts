// The question this test answers is not "does it see a nod" but "does it stay quiet
// while someone uses the board normally". A gesture detector that fires during sentence
// composition is worse than no gesture detector.

import { createGestureDetector } from './gesture'
import type { Gesture } from './gesture'

declare const process: { exit(code: number): never }

const FPS = 30
const FRAME_MS = 1000 / FPS
const DEG = Math.PI / 180

type Frame = { yaw: number; pitch: number }

function run(frames: Frame[], startTime = 0) {
  const detector = createGestureDetector()
  const fired: { gesture: Gesture; frame: number }[] = []
  frames.forEach((frame, index) => {
    const gesture = detector.push(startTime + index * FRAME_MS, frame.yaw, frame.pitch)
    if (gesture) fired.push({ gesture, frame: index })
  })
  return fired
}

function still(ms: number, yaw = 0, pitch = 0): Frame[] {
  return Array.from({ length: Math.round(ms / FRAME_MS) }, () => ({ yaw, pitch }))
}

/** A sinusoidal oscillation about the current pose: `cycles` full swings over `ms`. */
function oscillate(axis: 'yaw' | 'pitch', amplitudeDeg: number, cycles: number, ms: number, base: Frame = { yaw: 0, pitch: 0 }): Frame[] {
  const count = Math.round(ms / FRAME_MS)
  return Array.from({ length: count }, (_, index) => {
    const phase = (index / count) * cycles * 2 * Math.PI
    const offset = Math.sin(phase) * amplitudeDeg * DEG
    return axis === 'yaw' ? { yaw: base.yaw + offset, pitch: base.pitch } : { yaw: base.yaw, pitch: base.pitch + offset }
  })
}

/** Move from one pose to another over `ms`, the way the head travels between tiles. */
function glide(from: Frame, to: Frame, ms: number): Frame[] {
  const count = Math.max(1, Math.round(ms / FRAME_MS))
  return Array.from({ length: count }, (_, index) => {
    const t = (index + 1) / count
    const ease = t * t * (3 - 2 * t)
    return { yaw: from.yaw + (to.yaw - from.yaw) * ease, pitch: from.pitch + (to.pitch - from.pitch) * ease }
  })
}

// Aiming at the far columns of the board is a big yaw move; dwell holds it there.
const LEFT: Frame = { yaw: -14 * DEG, pitch: 0 }
const RIGHT: Frame = { yaw: 14 * DEG, pitch: 0 }
const TOP: Frame = { yaw: 0, pitch: -7 * DEG }
const BOTTOM: Frame = { yaw: 0, pitch: 7 * DEG }
const DWELL_HOLD_MS = 850

type Case = { name: string; frames: Frame[]; expect: Gesture | 'nothing' }

const cases: Case[] = [
  // --- should fire -------------------------------------------------------------
  { name: 'deliberate shake, 3 swings', frames: [...still(300), ...oscillate('yaw', 9, 1.5, 750), ...still(300)], expect: 'shake' },
  { name: 'brisk shake, 2 swings', frames: [...still(300), ...oscillate('yaw', 11, 1, 600), ...still(400)], expect: 'shake' },
  { name: 'deliberate nod, 3 swings', frames: [...still(300), ...oscillate('pitch', 8, 1.5, 750), ...still(300)], expect: 'nod' },
  { name: 'small quick nod', frames: [...still(300), ...oscillate('pitch', 6, 1.5, 620), ...still(300)], expect: 'nod' },
  {
    name: 'shake while parked at the left rest zone',
    frames: [...still(300, LEFT.yaw), ...oscillate('yaw', 9, 1.5, 750, LEFT), ...still(300, LEFT.yaw)],
    expect: 'shake',
  },

  // --- should stay quiet -------------------------------------------------------
  { name: 'sitting still', frames: still(2500), expect: 'nothing' },
  {
    name: 'scanning left word then right word then left again',
    frames: [
      ...still(400), ...glide({ yaw: 0, pitch: 0 }, LEFT, 350), ...still(DWELL_HOLD_MS, LEFT.yaw),
      ...glide(LEFT, RIGHT, 450), ...still(DWELL_HOLD_MS, RIGHT.yaw),
      ...glide(RIGHT, LEFT, 450), ...still(DWELL_HOLD_MS, LEFT.yaw),
    ],
    expect: 'nothing',
  },
  {
    name: 'scanning top row then bottom row then top again',
    frames: [
      ...still(400), ...glide({ yaw: 0, pitch: 0 }, TOP, 300), ...still(DWELL_HOLD_MS, 0, TOP.pitch),
      ...glide(TOP, BOTTOM, 400), ...still(DWELL_HOLD_MS, 0, BOTTOM.pitch),
      ...glide(BOTTOM, TOP, 400), ...still(DWELL_HOLD_MS, 0, TOP.pitch),
    ],
    expect: 'nothing',
  },
  {
    name: 'moving to a word and staying there',
    frames: [...still(400), ...glide({ yaw: 0, pitch: 0 }, RIGHT, 400), ...still(1500, RIGHT.yaw)],
    expect: 'nothing',
  },
  { name: 'postural sway', frames: oscillate('yaw', 2, 3, 3000), expect: 'nothing' },
  { name: 'slow drift across the board', frames: glide(LEFT, RIGHT, 2500), expect: 'nothing' },
  {
    name: 'leaning in and back',
    frames: [...still(300), ...glide({ yaw: 0, pitch: 0 }, { yaw: 0, pitch: 5 * DEG }, 700), ...still(900, 0, 5 * DEG)],
    expect: 'nothing',
  },
]

console.log('gesture detection')
let failures = 0
for (const testCase of cases) {
  const fired = run(testCase.frames)
  const got = fired.length === 0 ? 'nothing' : fired.map((entry) => entry.gesture).join('+')
  const ok = testCase.expect === 'nothing' ? fired.length === 0 : fired.length === 1 && fired[0].gesture === testCase.expect
  if (!ok) failures += 1
  const seconds = ((testCase.frames.length * FRAME_MS) / 1000).toFixed(1)
  console.log(
    `  ${ok ? 'ok  ' : 'FAIL'} ${testCase.name.padEnd(46)} ${seconds}s  expected ${String(testCase.expect).padEnd(8)} got ${got}`,
  )
}

// A gesture must not fire twice off one movement, or a nod says "Yes" two or three times.
const longShake = [...still(300), ...oscillate('yaw', 10, 3, 1500), ...still(500)]
const repeats = run(longShake)
const repeatOk = repeats.length === 1
if (!repeatOk) failures += 1
console.log(`  ${repeatOk ? 'ok  ' : 'FAIL'} ${'a long shake fires once, not per swing'.padEnd(46)}       fired ${repeats.length}x`)

console.log(failures === 0 ? `${cases.length + 1}/${cases.length + 1} gesture cases pass` : `${failures} gesture cases FAILED`)
process.exit(failures === 0 ? 0 : 1)
