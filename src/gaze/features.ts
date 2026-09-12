// Turns one MediaPipe frame into the features a calibration model maps to the screen.
//
// Everything works in PIXELS. MediaPipe normalises x by frame width and y by frame
// height, so on a 16:9 camera those two axes are on scales that differ by 1.78x.
// Any hypot or atan2 taken on the normalised values is therefore wrong - distances
// are skewed and the head-roll angle comes out at the wrong angle entirely.
//
// The gaze signal itself is a real 3D ray from the OpenFace eyeball model, not an
// iris offset in the image. See eyeModel.ts.

import { combineEyes, estimateEye, intrinsicsFor } from './eyeModel'
import type { Point2, Vec3 } from './eyeModel'

type Landmark = { x: number; y: number }
type Category = { categoryName?: string; score: number }

const LEFT_EYE_OUTER = 33
const LEFT_EYE_INNER = 133
const RIGHT_EYE_INNER = 362
const RIGHT_EYE_OUTER = 263
const LEFT_LID_TOP = 159
const LEFT_LID_BOTTOM = 145
const RIGHT_LID_TOP = 386
const RIGHT_LID_BOTTOM = 374
const LEFT_IRIS = 468
const RIGHT_IRIS = 473
const FOREHEAD = 10
const NOSE_TIP = 1
const FACE_LEFT = 234
const FACE_RIGHT = 454

// Iris landmarks (468-477) only exist on the refined 478-point mesh.
export const REQUIRED_LANDMARKS = 478
export const BASE_SIGNAL_COUNT = 14
export const HEAD_SIGNAL_COUNT = 10

export type Signals = {
  base: number[]
  // Where the head is aimed. Kept separate from the eye signals because the two have
  // very different noise: an iris landmark error is levered through a 12 mm eyeball
  // radius into about 1.5 degrees of gaze, whereas the forehead sits on a face spanning
  // hundreds of pixels and the pose matrix is solved from all 478 points at once.
  head: number[]
  blinking: boolean
  hasBlendshapes: boolean
  hasHeadPose: boolean
  hasEyeModel: boolean
  fallbackX: number
  fallbackY: number
  leftCorners: { outerX: number; outerY: number; innerX: number; innerY: number }
  rightCorners: { outerX: number; outerY: number; innerX: number; innerY: number }
  // Surfaced for the diagnostics panel.
  gazeAngleXDeg: number
  gazeAngleYDeg: number
  distanceCm: number
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function lookup(categories: Category[] | undefined) {
  const map = new Map<string, number>()
  for (const category of categories ?? []) {
    if (category.categoryName) map.set(category.categoryName, category.score)
  }
  return map
}

// MediaPipe's matrix layout is not guaranteed, but the homogeneous row [0,0,0,1]
// tells us which one we got: it sits at 12..14 when row-major, 3/7/11 when column-major.
function decodeHeadBasis(data: number[] | undefined) {
  if (!data || data.length < 16) return null

  const rowMajorTail = Math.abs(data[12]) + Math.abs(data[13]) + Math.abs(data[14])
  const columnMajorTail = Math.abs(data[3]) + Math.abs(data[7]) + Math.abs(data[11])
  const rowMajor = rowMajorTail <= columnMajorTail
  const at = (row: number, column: number) => (rowMajor ? data[row * 4 + column] : data[column * 4 + row])

  const forward = { x: -at(0, 2), y: -at(1, 2), z: -at(2, 2) }
  const up = { x: at(0, 1), y: at(1, 1), z: at(2, 1) }

  return {
    forward,
    up,
    yaw: Math.atan2(forward.x, -forward.z),
    pitch: Math.asin(clamp(forward.y, -1, 1)),
    roll: Math.atan2(up.x, up.y),
  }
}

export function extractSignals(
  landmarks: Landmark[] | undefined,
  blendshapes: Category[] | undefined,
  matrix: number[] | undefined,
  videoWidth: number,
  videoHeight: number,
): Signals | null {
  if (!landmarks || landmarks.length < REQUIRED_LANDMARKS || !videoWidth || !videoHeight) return null

  const at = (index: number): Point2 => ({ u: landmarks[index].x * videoWidth, v: landmarks[index].y * videoHeight })
  const span = (a: Point2, b: Point2) => Math.hypot(a.u - b.u, a.v - b.v)

  // Mean centre-to-rim distance over the four rim landmarks is steadier than any
  // single diameter, and this radius sets the depth estimate for the whole model.
  const irisRadius = (centre: number) => {
    const middle = at(centre)
    let total = 0
    for (let i = 1; i <= 4; i += 1) total += span(at(centre + i), middle)
    return total / 4
  }

  const shapes = lookup(blendshapes)
  const hasBlendshapes = shapes.size > 0
  const blendH =
    ((shapes.get('eyeLookOutLeft') ?? 0) - (shapes.get('eyeLookInLeft') ?? 0) +
      (shapes.get('eyeLookInRight') ?? 0) - (shapes.get('eyeLookOutRight') ?? 0)) / 2
  const blendV =
    ((shapes.get('eyeLookUpLeft') ?? 0) - (shapes.get('eyeLookDownLeft') ?? 0) +
      (shapes.get('eyeLookUpRight') ?? 0) - (shapes.get('eyeLookDownRight') ?? 0)) / 2

  const basis = decodeHeadBasis(matrix)
  // Image y points down, so world up is -y when no head pose is available.
  let headUp: Vec3 = basis?.up ?? { x: 0, y: -1, z: 0 }
  let headBack: Vec3 = basis ? { x: -basis.forward.x, y: -basis.forward.y, z: -basis.forward.z } : { x: 0, y: 0, z: 1 }
  // The eyeball sits further from the camera than the eye corners. If the decoded
  // basis says otherwise we have the handedness backwards, so flip it.
  if (headBack.z < 0) headBack = { x: -headBack.x, y: -headBack.y, z: -headBack.z }
  if (headUp.y > 0) headUp = { x: -headUp.x, y: -headUp.y, z: -headUp.z }

  const k = intrinsicsFor(videoWidth, videoHeight)
  const left = estimateEye(at(LEFT_IRIS), irisRadius(LEFT_IRIS), at(LEFT_EYE_OUTER), at(LEFT_EYE_INNER), headBack, headUp, k)
  const right = estimateEye(at(RIGHT_IRIS), irisRadius(RIGHT_IRIS), at(RIGHT_EYE_OUTER), at(RIGHT_EYE_INNER), headBack, headUp, k)

  const interocular = span(at(LEFT_EYE_OUTER), at(RIGHT_EYE_OUTER))
  const openness =
    (span(at(LEFT_LID_TOP), at(LEFT_LID_BOTTOM)) + span(at(RIGHT_LID_TOP), at(RIGHT_LID_BOTTOM))) /
    (2 * Math.max(interocular, 1e-3))

  // Vertical gaze is the hard axis. A 16:9 screen subtends roughly half the vertical
  // angle it does horizontal, so the same landmark noise costs about twice as much
  // vertical error - and looking down drags the eyelid over the iris, biasing the iris
  // centre upward exactly when the answer should be "down". These two signals are the
  // direct cues for that: how far the lid is open, and how high the iris sits relative
  // to the line between the eye corners. Both were already being computed and only used
  // for blink detection.
  const aperture = openness
  const irisRise =
    ((at(LEFT_EYE_OUTER).v + at(LEFT_EYE_INNER).v) / 2 - at(LEFT_IRIS).v +
      (at(RIGHT_EYE_OUTER).v + at(RIGHT_EYE_INNER).v) / 2 - at(RIGHT_IRIS).v) /
    (2 * Math.max(interocular, 1e-3))

  const blinkScore = hasBlendshapes
    ? ((shapes.get('eyeBlinkLeft') ?? 0) + (shapes.get('eyeBlinkRight') ?? 0)) / 2
    : 0

  const geometry = left && right ? combineEyes(left, right) : null

  // Head pointing. Two independent ways to aim, both of which people do naturally and
  // the regression is free to mix: moving the head (the face translates in frame) and
  // turning it (the nose and forehead swing out from the face centre in perspective,
  // and the pose matrix reports the rotation outright). Everything is divided by face
  // width so it stays the same whether you sit close or far.
  const faceLeft = at(FACE_LEFT)
  const faceRight = at(FACE_RIGHT)
  const faceWidth = Math.max(span(faceLeft, faceRight), 1e-3)
  const centreU = (faceLeft.u + faceRight.u) / 2
  const centreV = (faceLeft.v + faceRight.v) / 2
  const nose = at(NOSE_TIP)
  const forehead = at(FOREHEAD)

  const posX = (centreU - videoWidth / 2) / faceWidth
  const posY = (centreV - videoHeight / 2) / faceWidth
  const head = [
    posX,
    posY,
    (nose.u - centreU) / faceWidth,
    (nose.v - centreV) / faceWidth,
    (forehead.u - centreU) / faceWidth,
    (forehead.v - centreV) / faceWidth,
    faceWidth / videoWidth,
    basis?.yaw ?? 0,
    basis?.pitch ?? 0,
    basis?.roll ?? 0,
  ]

  const headOffsetX = (nose.u - videoWidth / 2) / faceWidth
  const headOffsetY = (nose.v - videoHeight / 2) / faceWidth

  const base = [
    geometry?.projX ?? 0,
    geometry?.projY ?? 0,
    geometry?.tanX ?? 0,
    geometry?.tanY ?? 0,
    geometry?.eye.x ?? 0,
    geometry?.eye.y ?? 0,
    geometry?.distanceCm ?? 0,
    basis?.yaw ?? 0,
    basis?.pitch ?? 0,
    basis?.roll ?? 0,
    blendH,
    blendV,
    aperture,
    irisRise,
  ]

  return {
    base,
    head,
    blinking: hasBlendshapes ? blinkScore > 0.5 : openness < 0.045,
    hasBlendshapes,
    hasHeadPose: basis !== null,
    hasEyeModel: geometry !== null,
    // Preview is mirrored, so turning left must move the cursor left.
    fallbackX: clamp(50 - headOffsetX * 130, 2, 98),
    fallbackY: clamp(50 + headOffsetY * 150, 2, 98),
    leftCorners: {
      outerX: landmarks[LEFT_EYE_OUTER].x,
      outerY: landmarks[LEFT_EYE_OUTER].y,
      innerX: landmarks[LEFT_EYE_INNER].x,
      innerY: landmarks[LEFT_EYE_INNER].y,
    },
    rightCorners: {
      outerX: landmarks[RIGHT_EYE_OUTER].x,
      outerY: landmarks[RIGHT_EYE_OUTER].y,
      innerX: landmarks[RIGHT_EYE_INNER].x,
      innerY: landmarks[RIGHT_EYE_INNER].y,
    },
    gazeAngleXDeg: geometry?.angleXDeg ?? 0,
    gazeAngleYDeg: geometry?.angleYDeg ?? 0,
    distanceCm: geometry?.distanceCm ?? 0,
  }
}
