// Draws the detection over the camera preview: face outline, eye contours, irises,
// and the exact crop rectangles handed to the regression.

import type { EyeBox } from './eyePatch'

type Landmark = { x: number; y: number }
type Connection = { start: number; end: number }

// The preview uses object-fit: cover and is mirrored, so landmark coordinates need
// the same crop and flip applied before they line up with what is on screen.
function projector(canvasWidth: number, canvasHeight: number, videoWidth: number, videoHeight: number) {
  const scale = Math.max(canvasWidth / videoWidth, canvasHeight / videoHeight)
  const drawnWidth = videoWidth * scale
  const drawnHeight = videoHeight * scale
  const offsetX = (canvasWidth - drawnWidth) / 2
  const offsetY = (canvasHeight - drawnHeight) / 2
  return {
    normalised: (point: Landmark) => ({
      x: canvasWidth - (offsetX + point.x * drawnWidth),
      y: offsetY + point.y * drawnHeight,
    }),
    pixels: (x: number, y: number) => ({
      x: canvasWidth - (offsetX + (x / videoWidth) * drawnWidth),
      y: offsetY + (y / videoHeight) * drawnHeight,
    }),
    scale,
  }
}

function strokeConnections(
  context: CanvasRenderingContext2D,
  landmarks: Landmark[],
  connections: Connection[],
  project: (point: Landmark) => Landmark,
) {
  context.beginPath()
  for (const connection of connections) {
    const from = landmarks[connection.start]
    const to = landmarks[connection.end]
    if (!from || !to) continue
    const a = project(from)
    const b = project(to)
    context.moveTo(a.x, a.y)
    context.lineTo(b.x, b.y)
  }
  context.stroke()
}

export function drawDetection(
  context: CanvasRenderingContext2D,
  landmarks: Landmark[],
  videoWidth: number,
  videoHeight: number,
  boxes: { left: EyeBox; right: EyeBox },
  outlines: { face: Connection[]; leftEye: Connection[]; rightEye: Connection[]; irises: Connection[] },
) {
  const canvas = context.canvas
  context.clearRect(0, 0, canvas.width, canvas.height)
  if (!videoWidth || !videoHeight) return
  const project = projector(canvas.width, canvas.height, videoWidth, videoHeight)

  context.lineWidth = 1.5
  context.strokeStyle = 'rgba(126, 170, 163, .55)'
  strokeConnections(context, landmarks, outlines.face, project.normalised)

  context.lineWidth = 2
  context.strokeStyle = 'rgba(164, 224, 195, .95)'
  strokeConnections(context, landmarks, outlines.leftEye, project.normalised)
  strokeConnections(context, landmarks, outlines.rightEye, project.normalised)

  context.strokeStyle = 'rgba(245, 196, 109, 1)'
  strokeConnections(context, landmarks, outlines.irises, project.normalised)

  // The rotated rectangles are the literal pixels the appearance model receives.
  context.strokeStyle = 'rgba(245, 196, 109, .8)'
  context.lineWidth = 1.5
  context.setLineDash([4, 3])
  for (const box of [boxes.left, boxes.right]) {
    const cos = Math.cos(box.angle)
    const sin = Math.sin(box.angle)
    const halfWidth = box.width / 2
    const halfHeight = box.height / 2
    context.beginPath()
    ;[
      [-halfWidth, -halfHeight],
      [halfWidth, -halfHeight],
      [halfWidth, halfHeight],
      [-halfWidth, halfHeight],
    ].forEach(([dx, dy], index) => {
      const point = project.pixels(box.centerX + dx * cos - dy * sin, box.centerY + dx * sin + dy * cos)
      if (index === 0) context.moveTo(point.x, point.y)
      else context.lineTo(point.x, point.y)
    })
    context.closePath()
    context.stroke()
  }
  context.setLineDash([])
}
