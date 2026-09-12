// Version 2. The head now moves a pointer, and holding the pointer on a word speaks it.
//
// This is the obvious way to build it, and it is worth playing with before reading the
// third version, because everything it gets wrong is something you would only discover by
// trying. Three shortcuts in particular:
//
//   1. The head angle is mapped to the screen with a fixed guess at how far a person
//      turns. No calibration. It is wrong for every individual and every seating position.
//   2. The pointer is drawn from the raw angle with no smoothing, so it carries every bit
//      of frame-to-frame jitter straight onto the board.
//   3. Dwell requires an unbroken hold on one tile. A single jittery frame that lands on a
//      neighbour resets the whole thing to zero.
//
// It works well enough to demo for ten seconds and badly enough to be unusable.

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const WORDS = [
  { label: 'Yes', speech: 'Yes' },
  { label: 'No', speech: 'No' },
  { label: 'Help', speech: 'I need help' },
  { label: 'Water', speech: 'I want water' },
  { label: 'Pain', speech: 'I am in pain' },
  { label: 'Thanks', speech: 'Thank you' },
]

// How far we assume someone turns their head to reach the edge of the screen. This single
// guess is what calibration replaces in version 3.
const ASSUMED_YAW = (20 * Math.PI) / 180
const ASSUMED_PITCH = (12 * Math.PI) / 180
const DWELL_MS = 1000

const video = document.getElementById('video') as HTMLVideoElement
const status = document.getElementById('status') as HTMLElement
const startButton = document.getElementById('start') as HTMLButtonElement
const cursor = document.getElementById('cursor') as HTMLElement
const board = document.getElementById('board') as HTMLElement
const saidOut = document.getElementById('said') as HTMLElement

const tiles: HTMLButtonElement[] = WORDS.map((word) => {
  const tile = document.createElement('button')
  tile.className = 'tile'
  tile.textContent = word.label
  tile.dataset.speech = word.speech
  board.appendChild(tile)
  return tile
})

let landmarker: FaceLandmarker | null = null
let lastVideoTime = -1
let holding: HTMLButtonElement | null = null
let holdStart = 0

function speak(text: string) {
  if (!('speechSynthesis' in window)) return
  window.speechSynthesis.cancel()
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))
}

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

function select(tile: HTMLButtonElement) {
  const phrase = tile.dataset.speech ?? tile.textContent ?? ''
  speak(phrase)
  saidOut.textContent = phrase
  tile.classList.add('hit')
  window.setTimeout(() => tile.classList.remove('hit'), 400)
}

function loop() {
  requestAnimationFrame(loop)
  if (!landmarker || video.readyState < 2) return
  if (video.currentTime === lastVideoTime) return
  lastVideoTime = video.currentTime

  const result = landmarker.detectForVideo(video, performance.now())
  const matrix = result.facialTransformationMatrixes?.[0]?.data
  if (!matrix) return

  const yaw = Math.asin(-Math.max(-1, Math.min(1, matrix[8])))
  const pitch = Math.atan2(matrix[9], matrix[10])

  // Straight line from head angle to screen position, using the fixed guess above. No
  // per-person fit, and no filtering on the way through.
  const x = Math.min(98, Math.max(2, 50 - (yaw / ASSUMED_YAW) * 50))
  const y = Math.min(98, Math.max(2, 50 + (pitch / ASSUMED_PITCH) * 50))
  cursor.style.left = `${x}%`
  cursor.style.top = `${y}%`

  const point = document.elementFromPoint(
    (x / 100) * window.innerWidth,
    (y / 100) * window.innerHeight,
  )
  const over = tiles.find((tile) => tile === point || tile.contains(point as Node)) ?? null

  // Any frame that leaves the tile throws away all the progress so far.
  if (over !== holding) {
    holding?.style.setProperty('--fill', '0%')
    holding = over
    holdStart = performance.now()
  }

  if (holding) {
    const held = performance.now() - holdStart
    holding.style.setProperty('--fill', `${Math.min(100, (held / DWELL_MS) * 100)}%`)
    if (held >= DWELL_MS) {
      select(holding)
      holding.style.setProperty('--fill', '0%')
      holding = null
    }
  }
}

startButton.addEventListener('click', start)
