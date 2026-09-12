// Measures whether the gaze signal recorded from a real camera is usable at all.
//
// A regression will always return something. If the underlying signal barely moves
// when the user looks from one side of the screen to the other, that something is
// noise dressed up as a prediction, and no amount of model tuning fixes it. These
// numbers separate "the model is badly fitted" from "the signal was never there".

export type SignalReport = {
  // How strongly the raw projected gaze tracks the target, per axis, from 0 to 1.
  correlationX: number
  correlationY: number
  // Signal range divided by within-fixation noise. Below about 3 the axis is unusable.
  separationX: number
  separationY: number
  verdict: 'good' | 'weak' | 'dead'
}

// How far the head actually travelled during calibration.
//
// This is the number that decides head pointing, and it is entirely under the user's
// control. Slow postural drift of roughly half a degree is always present and cannot be
// averaged away, because samples within a fixation are correlated. So accuracy is set by
// the ratio between that drift and how far the head swung across the board. Someone who
// turns 20 degrees gets a clean map; someone who turns 3 degrees and moves their eyes
// for the rest gets noise, and no fitting can tell the difference after the fact.
export type MovementReport = { yawDeg: number; pitchDeg: number; enough: boolean }

export function reportMovement(head: number[][], groups: number[], yawIndex: number, pitchIndex: number): MovementReport {
  const sums = new Map<number, { yaw: number; pitch: number; count: number }>()
  for (let i = 0; i < head.length; i += 1) {
    const bucket = sums.get(groups[i]) ?? { yaw: 0, pitch: 0, count: 0 }
    bucket.yaw += head[i][yawIndex]
    bucket.pitch += head[i][pitchIndex]
    bucket.count += 1
    sums.set(groups[i], bucket)
  }

  const yaws: number[] = []
  const pitches: number[] = []
  for (const bucket of sums.values()) {
    yaws.push(bucket.yaw / bucket.count)
    pitches.push(bucket.pitch / bucket.count)
  }
  if (yaws.length < 2) return { yawDeg: 0, pitchDeg: 0, enough: false }

  const toDegrees = (values: number[]) => ((Math.max(...values) - Math.min(...values)) * 180) / Math.PI
  const yawDeg = toDegrees(yaws)
  const pitchDeg = toDegrees(pitches)
  return { yawDeg, pitchDeg, enough: yawDeg > 10 && pitchDeg > 6 }
}

function correlation(a: number[], b: number[]) {
  if (a.length < 3) return 0
  const meanA = a.reduce((sum, value) => sum + value, 0) / a.length
  const meanB = b.reduce((sum, value) => sum + value, 0) / b.length
  let covariance = 0
  let varianceA = 0
  let varianceB = 0
  for (let i = 0; i < a.length; i += 1) {
    const da = a[i] - meanA
    const db = b[i] - meanB
    covariance += da * db
    varianceA += da * da
    varianceB += db * db
  }
  const denominator = Math.sqrt(varianceA * varianceB)
  return denominator > 1e-12 ? covariance / denominator : 0
}

function standardDeviation(values: number[]) {
  if (values.length < 2) return 0
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
}

// signal: raw projected gaze per frame. target: where the dot actually was.
// groups: which calibration point each frame belongs to.
export function reportSignal(
  signalX: number[],
  signalY: number[],
  targetX: number[],
  targetY: number[],
  groups: number[],
): SignalReport {
  const correlationX = Math.abs(correlation(signalX, targetX))
  const correlationY = Math.abs(correlation(signalY, targetY))

  // Within-group spread is noise, because the target was not moving. Between-group
  // spread of the group means is signal.
  const byGroup = new Map<number, { x: number[]; y: number[] }>()
  for (let i = 0; i < groups.length; i += 1) {
    const bucket = byGroup.get(groups[i]) ?? { x: [], y: [] }
    bucket.x.push(signalX[i])
    bucket.y.push(signalY[i])
    byGroup.set(groups[i], bucket)
  }

  const noiseX: number[] = []
  const noiseY: number[] = []
  const meansX: number[] = []
  const meansY: number[] = []
  for (const bucket of byGroup.values()) {
    if (bucket.x.length < 3) continue
    noiseX.push(standardDeviation(bucket.x))
    noiseY.push(standardDeviation(bucket.y))
    meansX.push(bucket.x.reduce((sum, value) => sum + value, 0) / bucket.x.length)
    meansY.push(bucket.y.reduce((sum, value) => sum + value, 0) / bucket.y.length)
  }

  const median = (values: number[]) => {
    if (values.length === 0) return 0
    const sorted = [...values].sort((a, b) => a - b)
    return sorted[Math.floor(sorted.length / 2)]
  }

  const separationX = standardDeviation(meansX) / Math.max(median(noiseX), 1e-6)
  const separationY = standardDeviation(meansY) / Math.max(median(noiseY), 1e-6)

  // Thresholds calibrated against the simulation in model.test.ts rather than guessed.
  // A separation of 1.4 on the vertical axis corresponds to about 3 px of iris jitter,
  // which still lands on the right word once a dwell has averaged it - so anything
  // stricter would report failure on a setup that works. "Dead" is reserved for signal
  // that averaging cannot rescue.
  const worstCorrelation = Math.min(correlationX, correlationY)
  const worstSeparation = Math.min(separationX, separationY)
  const verdict =
    worstCorrelation > 0.9 && worstSeparation > 2
      ? 'good'
      : worstCorrelation > 0.7 && worstSeparation > 0.8
        ? 'weak'
        : 'dead'

  return { correlationX, correlationY, separationX, separationY, verdict }
}
