import { buildNormalEquations, solveNormalEquations } from './solver'

// projX and projY are already where the gaze ray crosses the camera plane in cm, so
// screen position is close to a straight line in them and most of the model's job is
// just fixing the screen's offset and scale. The rest are second-order corrections:
// residual curvature, and the coupling that remains when the head turns.
export function expandGeometric(base: number[]) {
  const [projX, projY, tanX, tanY, eyeX, eyeY, distance, yaw, pitch, roll, blendH, blendV, aperture, irisRise] = base
  return [
    projX, projY, tanX, tanY, eyeX, eyeY, distance, yaw, pitch, roll, blendH, blendV, aperture, irisRise,
    projX * projX, projY * projY, projX * projY,
    tanX * yaw, tanY * pitch,
    projX * distance, projY * distance,
    // Vertical-only terms. Both eyelid cues need scale correction, because a lid opening
    // measured against interocular distance still shifts with how far away the head is,
    // and where the lid sits depends on head pitch as much as on gaze.
    irisRise * distance, aperture * pitch, tanY * tanY,
  ]
}

export const GEOMETRIC_WIDTH = 24

// Head pointing. Position in frame and rotation both aim independently, and the squared
// and scaled terms cover the perspective curvature plus the fact that the same head turn
// covers more screen when you sit closer.
export function expandHead(head: number[]) {
  const [posX, posY, noseX, noseY, foreheadX, foreheadY, scale, yaw, pitch, roll] = head
  return [
    posX, posY, noseX, noseY, foreheadX, foreheadY, scale, yaw, pitch, roll,
    yaw * yaw, pitch * pitch, yaw * pitch,
    posX * scale, posY * scale,
    noseX * scale, noseY * scale,
    yaw * scale, pitch * scale,
  ]
}

export const HEAD_WIDTH = 19

// Which signals a model was fitted on. Stored so predictions are always assembled the
// same way they were trained.
export type InputMix = 'head' | 'head+eye' | 'eye' | 'eye+pixels'

// Appearance pixels enter linearly. They already carry the nonlinearity, and
// expanding a 120-dimensional block would swamp the sample count.
export function buildFeatures(head: number[], geometric: number[], appearance: number[], mix: InputMix) {
  if (mix === 'head') return expandHead(head)
  if (mix === 'head+eye') return [...expandHead(head), ...expandGeometric(geometric)]
  if (mix === 'eye') return expandGeometric(geometric)
  return [...expandGeometric(geometric), ...appearance]
}

// Ridge shrinks large weights, so a column in radians and a column of pixel
// intensities cannot be compared until both are standardised. Without this the
// penalty is effectively arbitrary per column.
const LAMBDA_GRID = [0.01, 0.1, 1, 5, 20, 100, 400, 1500, 6000, 25000]
const CV_FOLDS = 4
const MIN_ROWS = 60

export type FittedModel = {
  mean: number[]
  scale: number[]
  weightsX: number[]
  weightsY: number[]
  interceptX: number
  interceptY: number
  lambda: number
  cvErrorPercent: number
  rows: number
}

function standardise(rows: number[][]) {
  const width = rows[0].length
  const mean = new Array<number>(width).fill(0)
  const scale = new Array<number>(width).fill(0)

  for (const row of rows) {
    for (let i = 0; i < width; i += 1) mean[i] += row[i]
  }
  for (let i = 0; i < width; i += 1) mean[i] /= rows.length

  for (const row of rows) {
    for (let i = 0; i < width; i += 1) scale[i] += (row[i] - mean[i]) ** 2
  }
  for (let i = 0; i < width; i += 1) {
    const deviation = Math.sqrt(scale[i] / rows.length)
    // A constant column (a signal this camera never produced) gets scale 1, so it
    // centres to zeros and the solver assigns it no weight.
    scale[i] = deviation > 1e-9 ? deviation : 1
  }

  return { mean, scale }
}

function applyStandardisation(row: number[], mean: number[], scale: number[]) {
  const out = new Array<number>(row.length)
  for (let i = 0; i < row.length; i += 1) out[i] = (row[i] - mean[i]) / scale[i]
  return out
}

function centre(values: number[]) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  return { mean, centred: values.map((value) => value - mean) }
}

function dot(weights: number[], row: number[]) {
  let total = 0
  for (let i = 0; i < weights.length; i += 1) total += weights[i] * row[i]
  return total
}

// Folds split by calibration group, never by sample. Frames from one fixation, or
// from one stretch of the pursuit path, are near-duplicates; a random split would
// put copies of the same moment on both sides and report a fake-low error.
function crossValidate(
  rows: number[][],
  targetsX: number[],
  targetsY: number[],
  groups: number[],
  lambdas: number[],
) {
  const uniqueGroups = Array.from(new Set(groups))
  const folds = Math.min(CV_FOLDS, uniqueGroups.length)
  if (folds < 2) return null

  const totals = new Array<number>(lambdas.length).fill(0)
  const counts = new Array<number>(lambdas.length).fill(0)

  for (let fold = 0; fold < folds; fold += 1) {
    const heldOut = new Set(uniqueGroups.filter((_, index) => index % folds === fold))
    const trainRows: number[][] = []
    const trainX: number[] = []
    const trainY: number[] = []
    const testRows: number[][] = []
    const testX: number[] = []
    const testY: number[] = []

    for (let i = 0; i < rows.length; i += 1) {
      if (heldOut.has(groups[i])) {
        testRows.push(rows[i])
        testX.push(targetsX[i])
        testY.push(targetsY[i])
      } else {
        trainRows.push(rows[i])
        trainX.push(targetsX[i])
        trainY.push(targetsY[i])
      }
    }
    if (trainRows.length < 2 || testRows.length === 0) continue

    const { mean, scale } = standardise(trainRows)
    const scaled = trainRows.map((row) => applyStandardisation(row, mean, scale))
    const cx = centre(trainX)
    const cy = centre(trainY)
    const equations = buildNormalEquations(scaled, cx.centred, cy.centred)
    const scaledTest = testRows.map((row) => applyStandardisation(row, mean, scale))

    lambdas.forEach((lambda, index) => {
      const solution = solveNormalEquations(equations, lambda)
      if (!solution) return
      let total = 0
      for (let i = 0; i < scaledTest.length; i += 1) {
        const errorX = dot(solution.a, scaledTest[i]) + cx.mean - testX[i]
        const errorY = dot(solution.b, scaledTest[i]) + cy.mean - testY[i]
        total += Math.hypot(errorX, errorY)
      }
      totals[index] += total
      counts[index] += scaledTest.length
    })
  }

  let bestLambda = lambdas[0]
  let bestError = Number.POSITIVE_INFINITY
  lambdas.forEach((lambda, index) => {
    if (counts[index] === 0) return
    const error = totals[index] / counts[index]
    if (error < bestError) {
      bestError = error
      bestLambda = lambda
    }
  })

  return Number.isFinite(bestError) ? { lambda: bestLambda, error: bestError } : null
}

export function fitModel(
  rows: number[][],
  targetsX: number[],
  targetsY: number[],
  groups: number[],
): FittedModel | null {
  if (rows.length < MIN_ROWS) return null

  const selected = crossValidate(rows, targetsX, targetsY, groups, LAMBDA_GRID)
  if (!selected) return null

  const { mean, scale } = standardise(rows)
  const scaled = rows.map((row) => applyStandardisation(row, mean, scale))
  const cx = centre(targetsX)
  const cy = centre(targetsY)
  const solution = solveNormalEquations(buildNormalEquations(scaled, cx.centred, cy.centred), selected.lambda)
  if (!solution) return null

  return {
    mean,
    scale,
    weightsX: solution.a,
    weightsY: solution.b,
    interceptX: cx.mean,
    interceptY: cy.mean,
    lambda: selected.lambda,
    cvErrorPercent: selected.error,
    rows: rows.length,
  }
}

// Which signals to use is an empirical question, not one to settle by argument. Fit
// every combination and keep whichever has the lowest cross-validated error, so adding
// the eye signals to the head signals can never make the result worse than head alone.
export type ModelChoice = { model: FittedModel; mix: InputMix }

export function fitBest(
  head: number[][],
  geometric: number[][],
  appearance: number[][],
  targetsX: number[],
  targetsY: number[],
  groups: number[],
): ModelChoice | null {
  const hasHead = head.length === geometric.length && head.length > 0
  const hasPixels = appearance.length === geometric.length && appearance.length > 0

  const candidates: InputMix[] = ['eye']
  if (hasHead) candidates.unshift('head', 'head+eye')
  if (hasPixels) candidates.push('eye+pixels')

  let best: ModelChoice | null = null
  for (const mix of candidates) {
    const rows = geometric.map((row, index) =>
      buildFeatures(hasHead ? head[index] : [], row, hasPixels ? appearance[index] : [], mix),
    )
    const model = fitModel(rows, targetsX, targetsY, groups)
    if (!model) continue
    if (!best || model.cvErrorPercent < best.model.cvErrorPercent) best = { model, mix }
  }
  return best
}

export function predict(model: FittedModel, features: number[]) {
  const row = applyStandardisation(features, model.mean, model.scale)
  return {
    x: dot(model.weightsX, row) + model.interceptX,
    y: dot(model.weightsY, row) + model.interceptY,
  }
}

export function evaluate(model: FittedModel, rows: number[][], targetsX: number[], targetsY: number[]) {
  if (rows.length === 0) return null
  let total = 0
  for (let i = 0; i < rows.length; i += 1) {
    const point = predict(model, rows[i])
    total += Math.hypot(point.x - targetsX[i], point.y - targetsY[i])
  }
  return total / rows.length
}

// Per-axis error, because the two axes fail for different reasons and by different
// amounts. A combined number hides the case that actually happens in practice: the
// column is right and the row is wrong. The board geometry and the nearest-word
// distance metric are both derived from these.
export function evaluateAxes(model: FittedModel, rows: number[][], targetsX: number[], targetsY: number[]) {
  if (rows.length === 0) return null
  let totalX = 0
  let totalY = 0
  for (let i = 0; i < rows.length; i += 1) {
    const point = predict(model, rows[i])
    totalX += Math.abs(point.x - targetsX[i])
    totalY += Math.abs(point.y - targetsY[i])
  }
  return { x: totalX / rows.length, y: totalY / rows.length }
}

// The error a dwell actually experiences, which is not the per-frame error.
//
// The eyeball radius is a 12 mm lever arm: a pixel of iris jitter at 55 cm becomes about
// 1.5 degrees of gaze, so per-frame error is dominated by landmark noise and is far
// worse than what someone holding a fixation sees. Holding still for 800 ms averages
// roughly 24 frames, and independent noise falls as the square root of that. Scoring the
// median prediction over each fixation is therefore the honest number to compare against
// the board's tolerance.
export function evaluateFixations(
  model: FittedModel,
  rows: number[][],
  targetsX: number[],
  targetsY: number[],
  groups: number[],
) {
  const byGroup = new Map<number, { x: number[]; y: number[]; targetX: number; targetY: number }>()
  for (let i = 0; i < rows.length; i += 1) {
    const bucket = byGroup.get(groups[i]) ?? { x: [], y: [], targetX: targetsX[i], targetY: targetsY[i] }
    const point = predict(model, rows[i])
    bucket.x.push(point.x)
    bucket.y.push(point.y)
    byGroup.set(groups[i], bucket)
  }
  if (byGroup.size === 0) return null

  const median = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b)
    return sorted[Math.floor(sorted.length / 2)]
  }

  let total = 0
  let totalX = 0
  let totalY = 0
  for (const bucket of byGroup.values()) {
    const dx = median(bucket.x) - bucket.targetX
    const dy = median(bucket.y) - bucket.targetY
    total += Math.hypot(dx, dy)
    totalX += Math.abs(dx)
    totalY += Math.abs(dy)
  }
  return { total: total / byGroup.size, x: totalX / byGroup.size, y: totalY / byGroup.size }
}

// A single bad frame can throw the cursor across the screen. Rejecting samples far
// from the fixation median stops those frames poisoning the fit.
export function rejectOutliers(samples: number[][]) {
  if (samples.length < 6) return samples
  const width = samples[0].length
  const median = new Array<number>(width).fill(0)
  for (let i = 0; i < width; i += 1) {
    const column = samples.map((sample) => sample[i]).sort((a, b) => a - b)
    median[i] = column[Math.floor(column.length / 2)]
  }

  const deviations = samples.map((sample) => {
    let total = 0
    for (let i = 0; i < width; i += 1) total += Math.abs(sample[i] - median[i])
    return total
  })
  const sorted = [...deviations].sort((a, b) => a - b)
  const medianDeviation = sorted[Math.floor(sorted.length / 2)] || 1e-9
  const kept = samples.filter((_, index) => deviations[index] <= medianDeviation * 2.5)
  return kept.length >= 5 ? kept : samples
}
