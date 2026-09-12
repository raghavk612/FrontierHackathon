// Nod for yes, shake for no, read off the same head pose the pointer already uses.
//
// The hard part is not spotting an oscillation, it is not spotting one when the user was
// only reading the board. Looking from a word on the left to a word on the right and back
// is a yaw reversal too, and firing "No" in the middle of someone composing a sentence is
// worse than missing a shake. Three things separate a gesture from browsing:
//
//   speed      a shake completes several swings inside a second; scanning cannot, because
//              selecting anything requires holding still for the dwell time
//   return     a gesture ends where it started, browsing ends somewhere new
//   dominance  a nod is pitch and barely any yaw, and the reverse for a shake
//
// All three have to hold, which costs a little sensitivity and buys a lot of quiet.

export type Gesture = 'nod' | 'shake'

type Sample = { time: number; yaw: number; pitch: number }

export type GestureConfig = {
  windowMs: number
  minSwingRad: number
  minReversals: number
  dominance: number
  returnFraction: number
  cooldownMs: number
  minStepRad: number
}

export const DEFAULT_GESTURE_CONFIG: GestureConfig = {
  // Short enough that a dwell-paced scan of the board cannot fit two reversals into it.
  windowMs: 900,
  // ~6 degrees peak to peak. Below this it is postural sway, not an answer.
  minSwingRad: 0.105,
  // Two reversals is there-and-back-and-there: a deliberate double movement.
  minReversals: 2,
  dominance: 1.7,
  // How close to the starting angle the head has to come back, as a fraction of the swing.
  returnFraction: 0.45,
  cooldownMs: 1400,
  // Hysteresis for counting turning points, so tremor does not read as reversals.
  minStepRad: 0.014,
}

/** Direction changes in a series, ignoring wobble smaller than `minStep`. */
export function countReversals(values: number[], minStep: number) {
  let reversals = 0
  let direction = 0
  let anchor = values[0] ?? 0
  for (const value of values) {
    const delta = value - anchor
    if (Math.abs(delta) < minStep) continue
    const heading = Math.sign(delta)
    if (direction !== 0 && heading !== direction) reversals += 1
    direction = heading
    anchor = value
  }
  return reversals
}

function swingOf(values: number[]) {
  let low = Infinity
  let high = -Infinity
  for (const value of values) {
    if (value < low) low = value
    if (value > high) high = value
  }
  return high - low
}

export type GestureDetector = {
  /** Feed one frame of head pose in radians. Returns a gesture on the frame it completes. */
  push(time: number, yaw: number, pitch: number): Gesture | null
  reset(): void
}

export function createGestureDetector(overrides: Partial<GestureConfig> = {}): GestureDetector {
  const config = { ...DEFAULT_GESTURE_CONFIG, ...overrides }
  let samples: Sample[] = []
  let quietUntil = 0

  const reset = () => {
    samples = []
  }

  return {
    reset,
    push(time, yaw, pitch) {
      // Drop everything seen during the cooldown. Keeping it would let the tail of a long
      // shake still sitting in the window fire a second time the moment the cooldown ends.
      if (time < quietUntil) {
        samples.length = 0
        return null
      }
      samples.push({ time, yaw, pitch })
      while (samples.length > 1 && time - samples[0].time > config.windowMs) samples.shift()
      // Two reversals need a handful of frames either side of each turning point.
      if (samples.length < 8) return null

      const yaws = samples.map((sample) => sample.yaw)
      const pitches = samples.map((sample) => sample.pitch)
      const yawSwing = swingOf(yaws)
      const pitchSwing = swingOf(pitches)

      const shaking = yawSwing >= pitchSwing * config.dominance
      const values = shaking ? yaws : pitches
      const swing = shaking ? yawSwing : pitchSwing
      if (swing < config.minSwingRad) return null
      if (!shaking && pitchSwing < yawSwing * config.dominance) return null

      if (countReversals(values, config.minStepRad) < config.minReversals) return null
      // A gesture comes back to where it started; a glance across the board does not.
      if (Math.abs(values[values.length - 1] - values[0]) > swing * config.returnFraction) return null

      quietUntil = time + config.cooldownMs
      reset()
      return shaking ? 'shake' : 'nod'
    },
  }
}
