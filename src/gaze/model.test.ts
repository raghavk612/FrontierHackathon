// End-to-end check of the gaze pipeline.
//
// Builds a synthetic 3D scene - screen, head, eyeballs, gaze rays - projects it into
// MediaPipe-shaped landmarks, then runs the real extractSignals and model code over
// those landmarks. Nothing is stubbed between the landmarks and the prediction, so a
// wrong sign or a swapped axis in the eye model shows up as a number here.
//
// Run with: npm run test:gaze

import { extractSignals } from './features'
import { buildFeatures, evaluate, evaluateAxes, evaluateFixations, fitBest } from './model'
import { reportMovement, reportSignal } from './quality'

declare const process: { exit(code: number): never }

let VIDEO_WIDTH = 1280
let VIDEO_HEIGHT = 720
let FOCAL = VIDEO_WIDTH * 0.9

function useResolution(width: number, height: number) {
  VIDEO_WIDTH = width
  VIDEO_HEIGHT = height
  FOCAL = width * 0.9
}

// Webcam sits at the top centre of a 34 x 19 cm screen; camera is the origin, x right,
// y down, z into the scene.
const SCREEN_WIDTH_MM = 340
const SCREEN_HEIGHT_MM = 190
const SCREEN_TOP_MM = 10

type Vec3 = { x: number; y: number; z: number }

const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z })
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z })
const mul = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s })
const norm = (a: Vec3): Vec3 => {
  const length = Math.hypot(a.x, a.y, a.z) || 1
  return mul(a, 1 / length)
}
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
})

let seed = 20260910
function random() {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff
  return seed / 0x7fffffff
}
function jitter(scale: number) {
  return (random() - 0.5) * 2 * scale
}

function screenPointMm(sx: number, sy: number): Vec3 {
  return { x: (sx - 0.5) * SCREEN_WIDTH_MM, y: SCREEN_TOP_MM + sy * SCREEN_HEIGHT_MM, z: 0 }
}

function project(point: Vec3) {
  return { u: (FOCAL * point.x) / point.z + VIDEO_WIDTH / 2, v: (FOCAL * point.y) / point.z + VIDEO_HEIGHT / 2 }
}

type Head = { distanceMm: number; yaw: number; pitch: number; offsetX: number; offsetY: number }

function headBasis(head: Head) {
  const forward = {
    x: Math.sin(head.yaw) * Math.cos(head.pitch),
    y: -Math.sin(head.pitch),
    z: -Math.cos(head.yaw) * Math.cos(head.pitch),
  }
  const up = {
    x: Math.sin(head.yaw) * Math.sin(head.pitch),
    y: -Math.cos(head.pitch),
    z: -Math.cos(head.yaw) * Math.sin(head.pitch),
  }
  return { forward, up, right: norm(cross(up, forward)) }
}

// Row-major 4x4 whose columns are [right, up, -forward], which is what
// decodeHeadBasis reads back out.
function poseMatrix(head: Head, centre: Vec3) {
  const { forward, up, right } = headBasis(head)
  return [
    right.x, up.x, -forward.x, centre.x,
    right.y, up.y, -forward.y, centre.y,
    right.z, up.z, -forward.z, centre.z,
    0, 0, 0, 1,
  ]
}

const EYEBALL_RADIUS_MM = 12
const IRIS_RADIUS_MM = 11.7 / 2
const OFFSET_UP_MM = 3.5
const OFFSET_BACK_MM = 7
const HALF_IPD_MM = 31.5

// Produces a full 478-entry landmark array from the scene, filling the indices the
// extractor actually reads.
// Head pointing: the user turns their head so the face aims at the target. Returns the
// pose that does that, plus the postural tremor nobody can suppress.
// commitment is how much of the required turn the person actually makes. 1.0 is aiming
// the nose right at the dot; 0.15 is barely moving and letting the eyes do the work,
// which is what people do by default when told to "look at" something.
function aimHead(base: Head, sx: number, sy: number, tremorRad: number, commitment: number): Head {
  const target = screenPointMm(sx, sy)
  const dx = target.x - base.offsetX
  const dy = target.y - base.offsetY
  const dz = -base.distanceMm
  const length = Math.hypot(dx, dy, dz) || 1
  return {
    ...base,
    yaw: Math.atan2(dx, -dz) * commitment + jitter(tremorRad),
    pitch: -Math.asin(dy / length) * commitment + jitter(tremorRad),
  }
}

// Postural drift, as opposed to tremor. It wanders slowly, so unlike landmark noise it
// is correlated across the frames of a single fixation and averaging inside that
// fixation does not remove it. Visiting each point twice, far apart in time, does.
let driftPhase = 0
function drift(amplitudeRad: number) {
  driftPhase += 0.05
  return {
    yaw: Math.sin(driftPhase * 0.7) * amplitudeRad,
    pitch: Math.cos(driftPhase * 0.4) * amplitudeRad * 0.6,
  }
}
const DRIFT_RAD = (1.2 * Math.PI) / 180

function renderLandmarks(head: Head, target: Vec3, landmarkNoisePx: number, poseNoiseRad: number) {
  const { forward, up, right } = headBasis(head)
  const back = mul(forward, -1)
  const centre: Vec3 = { x: head.offsetX, y: head.offsetY, z: head.distanceMm }

  const landmarks = new Array(478).fill(null).map(() => ({ x: 0.5, y: 0.5 }))
  const put = (index: number, point: Vec3) => {
    const projected = project(point)
    landmarks[index] = {
      x: (projected.u + jitter(landmarkNoisePx)) / VIDEO_WIDTH,
      y: (projected.v + jitter(landmarkNoisePx)) / VIDEO_HEIGHT,
    }
  }

  const eyes = [
    { sign: -1, iris: 468, outer: 33, inner: 133, lidTop: 159, lidBottom: 145 },
    { sign: 1, iris: 473, outer: 263, inner: 362, lidTop: 386, lidBottom: 374 },
  ]

  for (const eye of eyes) {
    const eyeball = add(centre, mul(right, eye.sign * HALF_IPD_MM))
    const gaze = norm(sub(target, eyeball))
    const pupil = add(eyeball, mul(gaze, EYEBALL_RADIUS_MM))

    // Invert the model's own offset so the corners imply the eyeball we started from.
    const cornerMid = sub(sub(eyeball, mul(up, OFFSET_UP_MM)), mul(back, OFFSET_BACK_MM))
    put(eye.outer, add(cornerMid, mul(right, eye.sign * 15)))
    put(eye.inner, sub(cornerMid, mul(right, eye.sign * 15)))
    put(eye.lidTop, add(cornerMid, mul(up, 5)))
    put(eye.lidBottom, sub(cornerMid, mul(up, 5)))

    // Iris rim, drawn perpendicular to the viewing ray so it projects as a circle of
    // the radius the depth estimate expects.
    put(eye.iris, pupil)
    const view = norm(pupil)
    const axisA = norm(cross(view, { x: 0, y: 1, z: 0 }))
    const axisB = norm(cross(view, axisA))
    put(eye.iris + 1, add(pupil, mul(axisA, IRIS_RADIUS_MM)))
    put(eye.iris + 2, add(pupil, mul(axisB, IRIS_RADIUS_MM)))
    put(eye.iris + 3, sub(pupil, mul(axisA, IRIS_RADIUS_MM)))
    put(eye.iris + 4, sub(pupil, mul(axisB, IRIS_RADIUS_MM)))
  }

  put(1, add(centre, mul(forward, 60)))
  put(234, sub(centre, mul(right, 70)))
  put(454, add(centre, mul(right, 70)))

  // The pose matrix is MediaPipe's own estimate, not ground truth, so it carries its own
  // error. Solving over 478 landmarks averages pixel noise down a long way, which is
  // precisely why head pose is a steadier signal than an iris centre.
  const estimated: Head = { ...head, yaw: head.yaw + jitter(poseNoiseRad), pitch: head.pitch + jitter(poseNoiseRad) }
  return { landmarks, matrix: poseMatrix(estimated, centre) }
}

// 0.2 deg of pose error per pixel of landmark noise, deliberately pessimistic: the
// theoretical figure for 478 points is closer to 0.04.
const POSE_NOISE_PER_PX = (0.2 * Math.PI) / 180

function signalsFor(head: Head, sx: number, sy: number, landmarkNoisePx: number) {
  const target = screenPointMm(sx, sy)
  const scene = renderLandmarks(head, target, landmarkNoisePx, landmarkNoisePx * POSE_NOISE_PER_PX)
  return extractSignals(scene.landmarks, undefined, scene.matrix, VIDEO_WIDTH, VIDEO_HEIGHT)
}

// ---------------------------------------------------------------- geometry accuracy

console.log('PART 1 - does the eye model recover the true gaze point?')
console.log('projX/projY are where the gaze ray crosses the camera plane, in cm.')
console.log('With correct geometry they should equal the real screen point.\n')

const geometryCases: [string, Head][] = [
  ['head centred, 55 cm', { distanceMm: 550, yaw: 0, pitch: 0, offsetX: 0, offsetY: 0 }],
  ['head turned 15 deg', { distanceMm: 550, yaw: 0.26, pitch: 0, offsetX: 0, offsetY: 0 }],
  ['head pitched 10 deg', { distanceMm: 550, yaw: 0, pitch: 0.17, offsetX: 0, offsetY: 0 }],
  ['head shifted 8 cm left', { distanceMm: 550, yaw: 0, pitch: 0, offsetX: -80, offsetY: 0 }],
  ['head close, 40 cm', { distanceMm: 400, yaw: 0, pitch: 0, offsetX: 0, offsetY: 0 }],
  ['head far, 75 cm', { distanceMm: 750, yaw: 0.1, pitch: -0.08, offsetX: 40, offsetY: -20 }],
]

let worstGeometry = 0
for (const [label, head] of geometryCases) {
  let total = 0
  let count = 0
  for (const sx of [0.05, 0.5, 0.95]) {
    for (const sy of [0.05, 0.5, 0.95]) {
      const signals = signalsFor(head, sx, sy, 0)
      if (!signals) continue
      const truth = screenPointMm(sx, sy)
      total += Math.hypot(signals.base[0] * 10 - truth.x, signals.base[1] * 10 - truth.y)
      count += 1
    }
  }
  const error = total / Math.max(count, 1)
  worstGeometry = Math.max(worstGeometry, error)
  const sample = signalsFor(head, 0.5, 0.5, 0)
  console.log(
    `  ${label.padEnd(24)} mean miss ${error.toFixed(1).padStart(5)} mm | ` +
      `centre gaze ${sample ? `${sample.gazeAngleXDeg.toFixed(1)}/${sample.gazeAngleYDeg.toFixed(1)} deg` : 'n/a'} | ` +
      `distance ${sample ? sample.distanceCm.toFixed(0) : '?'} cm`,
  )
}
const geometryOk = worstGeometry < 15
console.log(`  => worst ${worstGeometry.toFixed(1)} mm on a 340 mm screen ${geometryOk ? '(OK)' : '(BROKEN)'}\n`)

// ------------------------------------------------------------- end-to-end accuracy

console.log('PART 2 - full calibration, scored on unseen fixation points')

const CHECK_POINTS = [
  { x: 0.18, y: 0.16 }, { x: 0.82, y: 0.16 },
  { x: 0.5, y: 0.45 },
  { x: 0.18, y: 0.72 }, { x: 0.82, y: 0.72 },
]

function wobble(base: Head, amount: number): Head {
  return {
    distanceMm: base.distanceMm + jitter(amount * 300),
    yaw: base.yaw + jitter(amount),
    pitch: base.pitch + jitter(amount * 0.6),
    offsetX: base.offsetX + jitter(amount * 200),
    offsetY: base.offsetY + jitter(amount * 120),
  }
}

// Mirrors the app: sixteen fixations, ~18 usable frames of recording at each. Fixations
// rather than a moving dot, because smooth pursuit above ~30 deg/sec breaks into
// catch-up saccades and every frame after one of those is mislabelled.
type Point = { x: number; y: number; group: number }

function buildPoints(ys: number[], xs: number[], passes: number): Point[] {
  const single = ys.flatMap((y) => xs.map((x) => ({ x, y }))).map((point, group) => ({ ...point, group }))
  // A second pass revisits the same positions in reverse, so each group is sampled twice
  // at moments far enough apart that the postural drift between them is uncorrelated.
  return passes >= 2 ? [...single, ...[...single].reverse()] : single
}

const GRID_Y = [0.1, 0.44, 0.78]
const GRID_X = [0.06, 0.35, 0.65, 0.94]
const GRID_34 = buildPoints(GRID_Y, GRID_X, 1)
// Sixteen dots: every position once, then four spread positions again. Part 5 below is
// what picked this over the twenty-four it used to be.
const CALIBRATION_POINTS = [...GRID_34, ...[...GRID_34.filter((_, i) => i % 3 === 0)].reverse()]
// The app records 800 ms per dot, which is twenty-four frames at 30 fps.
const FRAMES_PER_POINT = 24

// Postural tremor while holding the head aimed at something. Half a degree is a
// realistic figure for someone sitting unsupported.
const TREMOR_RAD = (0.5 * Math.PI) / 180

function collect(
  base: Head,
  headDrift: number,
  landmarkNoisePx: number,
  commitment: number,
  points: Point[] = CALIBRATION_POINTS,
  framesPerPoint = FRAMES_PER_POINT,
  // Fraction of each dot's frames captured before the head got there, and so labelled
  // with a position the user was not yet looking at.
  stalePortion = 0,
) {
  const head: number[][] = []
  const rows: number[][] = []
  const targetsX: number[] = []
  const targetsY: number[] = []
  const groups: number[] = []

  let previous: Point | null = null
  for (const point of points) {
    const wander = drift(DRIFT_RAD)
    const staleFrames = Math.round(framesPerPoint * stalePortion)
    for (let frame = 0; frame < framesPerPoint; frame += 1) {
      // Where the head actually is. The label pushed below is the new dot either way,
      // which is precisely the damage: the frame is not noisy, it is wrong.
      const aimed = previous && frame < staleFrames ? previous : point
      const lookX = aimed.x + jitter(0.008)
      const lookY = aimed.y + jitter(0.008)
      const posture = aimHead(wobble(base, headDrift), aimed.x, aimed.y, TREMOR_RAD, commitment)
      posture.yaw += wander.yaw
      posture.pitch += wander.pitch
      const signals = signalsFor(posture, lookX, lookY, landmarkNoisePx)
      if (!signals) continue
      head.push(signals.head)
      rows.push(signals.base)
      targetsX.push(point.x * 100)
      targetsY.push(point.y * 100)
      groups.push(point.group)
    }
    previous = point
  }

  return { head, rows, targetsX, targetsY, groups }
}

function calibrate(
  base: Head,
  headDrift: number,
  landmarkNoisePx: number,
  commitment: number,
  points: Point[] = CALIBRATION_POINTS,
  framesPerPoint = FRAMES_PER_POINT,
  stalePortion = 0,
) {
  const data = collect(base, headDrift, landmarkNoisePx, commitment, points, framesPerPoint, stalePortion)
  return fitBest(data.head, data.rows, [], data.targetsX, data.targetsY, data.groups)
}

// 24 frames is one 800 ms dwell at 30 fps, the window the interface actually averages
// over before it commits to a word.
const DWELL_FRAMES = 24

function measure(choice: ReturnType<typeof fitBest>, base: Head, headDrift: number, landmarkNoisePx: number, commitment: number) {
  if (!choice) return { perFrame: Infinity, total: Infinity, x: Infinity, y: Infinity }
  const rows: number[][] = []
  const targetsX: number[] = []
  const targetsY: number[] = []
  const groups: number[] = []
  CHECK_POINTS.forEach((point, index) => {
    const wander = drift(DRIFT_RAD)
    for (let frame = 0; frame < DWELL_FRAMES; frame += 1) {
      const posture = aimHead(wobble(base, headDrift), point.x, point.y, TREMOR_RAD, commitment)
      posture.yaw += wander.yaw
      posture.pitch += wander.pitch
      const signals = signalsFor(posture, point.x, point.y, landmarkNoisePx)
      if (!signals) continue
      rows.push(buildFeatures(signals.head, signals.base, [], choice.mix))
      targetsX.push(point.x * 100)
      targetsY.push(point.y * 100)
      groups.push(index)
    }
  })
  const dwell = evaluateFixations(choice.model, rows, targetsX, targetsY, groups)
  const axes = evaluateAxes(choice.model, rows, targetsX, targetsY)
  return {
    perFrame: evaluate(choice.model, rows, targetsX, targetsY) ?? Infinity,
    total: dwell?.total ?? Infinity,
    x: dwell?.x ?? axes?.x ?? Infinity,
    y: dwell?.y ?? axes?.y ?? Infinity,
  }
}

// The board is six columns by two rows, so the two axes have very different budgets:
// half the gap between neighbouring tile centres, per axis.
const BUDGET_ACROSS = 7.2
const BUDGET_DOWN = 18.3

const baseHead: Head = { distanceMm: 550, yaw: 0, pitch: 0, offsetX: 0, offsetY: 0 }

// Landmark noise was the flaw in every earlier version of this test. Sub-pixel figures
// are what a detector achieves on a still, well-lit, high-resolution face; MediaPipe's
// iris landmarks on a live webcam move by two to four pixels frame to frame. Because the
// 12 mm eyeball radius levers that into roughly 1.5 degrees of gaze per pixel, the two
// regimes differ by an order of magnitude on screen, so the realistic ones are the only
// rows that mean anything.
// The third column is head commitment: how much of the required turn the person makes.
// This is the variable that decides head pointing, and it is the one the interface has
// to coach, because someone told to "look at the dot" defaults to about 0.15.
const conditions: [string, number, number, number][] = [
  ['committed turn, 3 px noise', 0.10, 3, 1.0],
  ['partial turn, 3 px noise', 0.10, 3, 0.6],
  ['barely turning, 3 px noise', 0.10, 3, 0.25],
  ['eyes only, 3 px noise', 0.10, 3, 0.1],
]

const results: boolean[] = []
for (const [label, headDrift, landmarkNoisePx, commitment] of conditions) {
  seed = 424242
  driftPhase = 0
  const choice = calibrate(baseHead, headDrift, landmarkNoisePx, commitment)
  const error = measure(choice, baseHead, headDrift, landmarkNoisePx, commitment)
  const fits = error.x < BUDGET_ACROSS && error.y < BUDGET_DOWN
  results.push(fits)
  console.log(
    `  ${label.padEnd(28)} per-frame ${error.perFrame.toFixed(1).padStart(5)}% -> ` +
      `holding still ${error.total.toFixed(1).padStart(4)}% | ` +
      `across ${error.x.toFixed(1)}% / down ${error.y.toFixed(1)}% | ` +
      `${(choice?.mix ?? 'none').padEnd(8)} | ${fits ? 'hits the right word' : 'MISSES'}`,
  )
}

// A model trained still but used while moving is the realistic failure case.
seed = 424242
const stillModel = calibrate(baseHead, 0.02, 2.5, 1.0)
const moved = measure(stillModel, baseHead, 0.14, 2.5, 1.0)
console.log(
  `  ${'calibrated still, used moving'.padEnd(28)} per-frame ${moved.perFrame.toFixed(1).padStart(5)}% -> ` +
    `holding still ${moved.total.toFixed(1).padStart(4)}% | across ${moved.x.toFixed(1)}% / down ${moved.y.toFixed(1)}%`,
)

// ------------------------------------------------------------- signal quality

// These are the numbers the app now shows the user after calibrating. Printing the
// synthetic baseline gives something to compare a real reading against: if the app
// reports far lower correlation or separation than a noisy simulation does, the problem
// is the camera or the lighting, not the mapping.
console.log('\nPART 4 - signal quality baseline (what the app reports after calibrating)')
for (const [label, headDrift, landmarkNoisePx, commitment] of conditions) {
  seed = 424242
  driftPhase = 0
  const data = collect(baseHead, headDrift, landmarkNoisePx, commitment)
  const movement = reportMovement(data.head, data.groups, 7, 8)
  const report = reportSignal(
    data.rows.map((row) => row[0]),
    data.rows.map((row) => row[1]),
    data.targetsX,
    data.targetsY,
    data.groups,
  )
  console.log(
    `  ${label.padEnd(28)} turned ${movement.yawDeg.toFixed(0).padStart(2)}deg/${movement.pitchDeg.toFixed(0).padStart(2)}deg ` +
      `${(movement.enough ? 'enough' : 'TOO SMALL').padEnd(9)} | r ${report.correlationX.toFixed(2)}/${report.correlationY.toFixed(2)} | ${report.verdict}`,
  )
}

// ------------------------------------------------------- how short can it be?

// Calibration is the worst part of the experience, so the question is which of its three
// costs actually buys accuracy: the number of positions, the number of passes over them,
// and how long each dot records for. Fitting 19 head features needs far fewer samples
// than the current settings collect, so the expectation is that recording time is the
// cheap thing to cut and spatial coverage is not.
console.log('\nPART 5 - what does calibration length actually buy?')

// The question this answers is how few dots we can get away with. Rushing each dot is
// the wrong lever: the recording is already far longer than the fit needs, and hurrying
// someone who has to physically turn their head to reach the far dots just means they
// arrive late and the sample is taken mid-turn. So the pace here is a comfortable one -
// settle until the head goes quiet, then half a second of recording - and the sweep
// varies the number of dots instead.
const SETTLE_MS = 1000
const duration = (dots: number, frames: number) => (dots * (SETTLE_MS + (frames / 30) * 1000)) / 1000

const half = (points: Point[]) => points.filter((_, index) => index % 2 === 0)
const grid34 = GRID_34
const grid33 = buildPoints(GRID_Y, [0.06, 0.5, 0.94], 1)
const grid24 = buildPoints([0.14, 0.74], GRID_X, 1)
const passAndHalf = (points: Point[]) => [...points, ...[...half(points)].reverse()]

const layouts: [string, Point[]][] = [
  ['3x4, two passes', buildPoints(GRID_Y, GRID_X, 2)],
  // Revisiting only half the positions still gives the fit drift-separated pairs to work
  // with, at half the cost of a full second pass.
  ['3x4, pass + half', passAndHalf(grid34)],
  ['3x3, two passes', buildPoints(GRID_Y, [0.06, 0.5, 0.94], 2)],
  ['2x4, two passes', buildPoints([0.14, 0.74], GRID_X, 2)],
  ['3x4, pass + third', [...grid34, ...[...grid34.filter((_, i) => i % 3 === 0)].reverse()]],
  ['3x4, pass + corners', [...grid34, ...[grid34[11], grid34[8], grid34[3], grid34[0]]]],
  ['3x3, pass + half', passAndHalf(grid33)],
  ['2x4, pass + half', passAndHalf(grid24)],
  ['3x4, one pass', grid34],
  ['2x3, two passes', buildPoints([0.14, 0.74], [0.06, 0.5, 0.94], 2)],
  ['3x3, one pass', grid33],
  ['2x4, one pass', grid24],
]

// Every configuration has to be scored against the same posture, or a layout simply
// lands on a kinder stretch of the drift curve and looks better than it is. Averaging
// over several seeds stops one lucky run deciding the settings.
const TRIAL_SEEDS = [424242, 991, 5150, 60607, 7331, 20250912]

function trial(points: Point[], frames: number) {
  let across = 0
  let down = 0
  for (const trialSeed of TRIAL_SEEDS) {
    seed = trialSeed
    driftPhase = 0
    // Scored at the commitment a coached user actually reaches, not the ideal.
    const choice = calibrate(baseHead, 0.1, 3, 0.6, points, frames)
    driftPhase = 11
    const error = measure(choice, baseHead, 0.1, 3, 0.6)
    across += error.x
    down += error.y
  }
  return { across: across / TRIAL_SEEDS.length, down: down / TRIAL_SEEDS.length }
}

for (const [label, points] of layouts) {
  const line: string[] = []
  for (const frames of [24, 15]) {
    const error = trial(points, frames)
    const fits = error.across < BUDGET_ACROSS && error.down < BUDGET_DOWN
    line.push(
      `${frames}f ${duration(points.length, frames).toFixed(0).padStart(2)}s ` +
        `${error.across.toFixed(1)}/${error.down.toFixed(1)}${fits ? ' ' : '!'}`,
    )
  }
  console.log(`  ${label.padEnd(18)} ${String(points.length).padStart(2)} dots  ${line.join(' | ')}`)
}
console.log('  (frames per dot, total seconds, mean across/down error over 6 runs; ! means it misses)')

console.log('\nPART 6 - what does recording before the user arrives cost?')

// The settle phase used to end as soon as the head went quiet for 220 ms with a floor of
// 250 ms. Nobody reacts to a dot in 250 ms, so at the start of every dot the head sat
// perfectly still - aimed at the previous dot - and passed the stillness test. This is
// what that costs, as the fraction of each recording taken before the head arrived.
for (const stale of [0, 0.15, 0.3, 0.5]) {
  let across = 0
  let down = 0
  for (const trialSeed of TRIAL_SEEDS) {
    seed = trialSeed
    driftPhase = 0
    const choice = calibrate(baseHead, 0.1, 3, 0.6, CALIBRATION_POINTS, FRAMES_PER_POINT, stale)
    driftPhase = 11
    const error = measure(choice, baseHead, 0.1, 3, 0.6)
    across += error.x
    down += error.y
  }
  across /= TRIAL_SEEDS.length
  down /= TRIAL_SEEDS.length
  const fits = across < BUDGET_ACROSS && down < BUDGET_DOWN
  console.log(
    `  ${`${(stale * 100).toFixed(0)}% of frames stale`.padEnd(24)} across ${across.toFixed(1).padStart(5)}% / ` +
      `down ${down.toFixed(1).padStart(5)}%  ${fits ? 'within budget' : 'MISSES THE WORD'}`,
  )
}

// ------------------------------------------------------------------ resolution

// Landmark detectors are accurate to roughly a fixed fraction of a pixel, so the same
// noise costs less angular error when more real pixels land on the iris.
console.log('\nPART 3 - does camera resolution matter?')
for (const [width, height] of [[640, 480], [1280, 720], [1920, 1080]]) {
  useResolution(width, height)
  seed = 424242
  const choice = calibrate(baseHead, 0.08, 3, 1.0)
  const error = measure(choice, baseHead, 0.08, 3, 1.0)
  console.log(
    `  ${`${width}x${height}`.padEnd(28)} per-frame ${error.perFrame.toFixed(1).padStart(5)}% -> ` +
      `holding still ${error.total.toFixed(1).padStart(4)}% | across ${error.x.toFixed(1)}% / down ${error.y.toFixed(1)}%`,
  )
}
useResolution(1280, 720)

const passed = results.filter(Boolean).length
console.log(
  `\n${passed}/${results.length} conditions land on the intended word ` +
    `(budget ${BUDGET_ACROSS}% across, ${BUDGET_DOWN}% down) | geometry ${geometryOk ? 'OK' : 'BROKEN'}`,
)
process.exit(geometryOk && passed >= 3 ? 0 : 1)
