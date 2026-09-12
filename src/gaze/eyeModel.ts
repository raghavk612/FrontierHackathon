// 3D eye model, following OpenFace 2.0 (Baltrusaitis et al.).
//
//   pupil3d       = ray from the camera through the iris centre, intersected with
//                   a 12 mm eyeball sphere
//   eyeballCentre = eye-corner midpoint, pushed 7 mm back and 3.5 mm up along the
//                   head's own axes
//   gaze          = normalize(pupil3d - eyeballCentre)
//
// This matters because it produces a real direction in space rather than a 2D iris
// offset. An offset in pixels confounds "the eye rotated" with "the head moved" and
// with "the face got closer"; a 3D ray separates all three, so the regression that
// follows is close to plain geometry.
//
// Depth comes from the iris. The human iris is 11.7 mm across with very little
// variation between people, which is the trick MediaPipe Iris uses to recover metric
// distance from one camera to within about 10%.

export type Vec3 = { x: number; y: number; z: number }
export type Point2 = { u: number; v: number }

export const IRIS_DIAMETER_MM = 11.7
export const EYEBALL_RADIUS_MM = 12
const OFFSET_UP_MM = 3.5
const OFFSET_BACK_MM = 7

export type Intrinsics = { fx: number; fy: number; cx: number; cy: number }

// Webcams cluster near a 60 degree horizontal field of view. The absolute value is
// absorbed by calibration; what must be right is that fx equals fy, so the two image
// axes share one scale.
export function intrinsicsFor(width: number, height: number): Intrinsics {
  const focal = width * 0.9
  return { fx: focal, fy: focal, cx: width / 2, cy: height / 2 }
}

function dot(a: Vec3, b: Vec3) {
  return a.x * b.x + a.y * b.y + a.z * b.z
}
function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}
function subtract(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
}
function scale(a: Vec3, factor: number): Vec3 {
  return { x: a.x * factor, y: a.y * factor, z: a.z * factor }
}
function normalize(a: Vec3): Vec3 {
  const length = Math.hypot(a.x, a.y, a.z) || 1
  return scale(a, 1 / length)
}

export function backProject(point: Point2, depth: number, k: Intrinsics): Vec3 {
  return { x: ((point.u - k.cx) / k.fx) * depth, y: ((point.v - k.cy) / k.fy) * depth, z: depth }
}

// Ray starts at the camera origin, so the quadratic simplifies.
export function raySphereIntersect(direction: Vec3, centre: Vec3, radius: number): Vec3 {
  const b = -2 * dot(direction, centre)
  const c = dot(centre, centre) - radius * radius
  const discriminant = b * b - 4 * c

  if (discriminant >= 0) {
    const t = (-b - Math.sqrt(discriminant)) / 2
    if (t > 0) return scale(direction, t)
  }

  // Noisy landmarks can place the ray just outside the eyeball. Rather than drop the
  // frame, take the closest approach and push it onto the sphere surface.
  const closest = scale(direction, Math.max(dot(direction, centre), 1))
  return add(centre, scale(normalize(subtract(closest, centre)), radius))
}

export type EyeEstimate = {
  centre: Vec3
  gaze: Vec3
  depthMm: number
}

export function estimateEye(
  irisCentre: Point2,
  irisRadiusPx: number,
  cornerA: Point2,
  cornerB: Point2,
  headBack: Vec3,
  headUp: Vec3,
  k: Intrinsics,
): EyeEstimate | null {
  if (!(irisRadiusPx > 0.5)) return null

  const depthMm = (k.fx * IRIS_DIAMETER_MM) / (irisRadiusPx * 2)
  if (!Number.isFinite(depthMm) || depthMm < 50 || depthMm > 3000) return null

  const cornerMid = backProject({ u: (cornerA.u + cornerB.u) / 2, v: (cornerA.v + cornerB.v) / 2 }, depthMm, k)
  const centre = add(cornerMid, add(scale(headUp, OFFSET_UP_MM), scale(headBack, OFFSET_BACK_MM)))
  const rayDirection = normalize(backProject(irisCentre, 1, k))
  const pupil = raySphereIntersect(rayDirection, centre, EYEBALL_RADIUS_MM)

  return { centre, gaze: normalize(subtract(pupil, centre)), depthMm }
}

export type GazeGeometry = {
  // Where the gaze ray crosses the plane of the camera, in cm. The screen sits in
  // roughly that plane, so screen position is very nearly linear in these two.
  projX: number
  projY: number
  tanX: number
  tanY: number
  eye: Vec3
  distanceCm: number
  angleXDeg: number
  angleYDeg: number
}

export function combineEyes(left: EyeEstimate, right: EyeEstimate): GazeGeometry {
  const gaze = normalize(add(left.gaze, right.gaze))
  const eye = scale(add(left.centre, right.centre), 0.5)

  // The eye sits in front of the camera at z > 0 and looks back toward the screen,
  // so the forward component is negative; guard the divide near the singularity.
  const forward = Math.max(-gaze.z, 0.15)
  const tanX = gaze.x / forward
  const tanY = gaze.y / forward

  return {
    projX: (eye.x + eye.z * tanX) / 10,
    projY: (eye.y + eye.z * tanY) / 10,
    tanX,
    tanY,
    eye: scale(eye, 0.1),
    distanceCm: eye.z / 10,
    angleXDeg: (Math.atan(tanX) * 180) / Math.PI,
    angleYDeg: (Math.atan(tanY) * 180) / Math.PI,
  }
}
