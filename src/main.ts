// Version 1. The camera finds a face and reports which way the head is pointing.
//
// This is the honest starting point for anything webcam-controlled: before you can use a
// head as a pointer you have to prove you can see one at all. Everything here comes
// straight out of MediaPipe with no processing of any kind - no smoothing, no
// calibration, no mapping onto the screen. Watch the numbers jitter while you hold still.
// That jitter is the entire problem the later versions exist to solve.

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const video = document.getElementById('video') as HTMLVideoElement
const overlay = document.getElementById('overlay') as HTMLCanvasElement
const status = document.getElementById('status') as HTMLParagraphElement
const startButton = document.getElementById('start') as HTMLButtonElement
const foundOut = document.getElementById('found') as HTMLElement
const yawOut = document.getElementById('yaw') as HTMLElement
const pitchOut = document.getElementById('pitch') as HTMLElement
const countOut = document.getElementById('count') as HTMLElement

const context = overlay.getContext('2d')!
let landmarker: FaceLandmarker | null = null
let lastVideoTime = -1

const degrees = (radians: number) => (radians * 180) / Math.PI

async function start() {
  startButton.disabled = true
  status.textContent = 'Starting the camera…'

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    })
    video.srcObject = stream
    await video.play()

    status.textContent = 'Loading the face model…'
    const vision = await FilesetResolver.forVisionTasks('/mediapipe')
    landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/face_landmarker.task', delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
    })

    status.textContent = ''
    requestAnimationFrame(loop)
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : 'Could not start the camera.'
    startButton.disabled = false
  }
}

function loop() {
  requestAnimationFrame(loop)
  if (!landmarker || video.readyState < 2) return
  if (video.currentTime === lastVideoTime) return
  lastVideoTime = video.currentTime

  overlay.width = video.videoWidth
  overlay.height = video.videoHeight
  context.clearRect(0, 0, overlay.width, overlay.height)

  const result = landmarker.detectForVideo(video, performance.now())
  const points = result.faceLandmarks?.[0]

  if (!points) {
    foundOut.textContent = 'no'
    yawOut.textContent = '—'
    pitchOut.textContent = '—'
    countOut.textContent = '0'
    return
  }

  foundOut.textContent = 'yes'
  countOut.textContent = String(points.length)

  // Every landmark, drawn as a dot. Seeing the whole mesh is the point of this version.
  context.fillStyle = 'rgba(164, 224, 195, 0.75)'
  for (const point of points) {
    context.fillRect(point.x * overlay.width - 1, point.y * overlay.height - 1, 2, 2)
  }

  // MediaPipe also hands back a 4x4 transform describing how the head is rotated. Pulling
  // yaw and pitch out of it is much steadier than guessing from individual landmarks,
  // because it is solved from all 478 points at once rather than a handful.
  const matrix = result.facialTransformationMatrixes?.[0]?.data
  if (matrix) {
    const yaw = Math.asin(-Math.max(-1, Math.min(1, matrix[8])))
    const pitch = Math.atan2(matrix[9], matrix[10])
    yawOut.textContent = `${degrees(yaw).toFixed(1)}°`
    pitchOut.textContent = `${degrees(pitch).toFixed(1)}°`
  }
}

startButton.addEventListener('click', start)
