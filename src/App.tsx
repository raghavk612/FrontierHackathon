import { useCallback, useEffect, useRef, useState } from 'react'
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { OneEuroFilter } from './gaze/filter'
import { extractSignals } from './gaze/features'
import { EyePatchExtractor, eyeBoxFrom } from './gaze/eyePatch'
import { drawDetection } from './gaze/overlay'
import { buildFeatures, evaluate, evaluateAxes, evaluateFixations, fitBest, predict, rejectOutliers } from './gaze/model'
import type { InputMix } from './gaze/model'
import type { FittedModel } from './gaze/model'
import { reportMovement, reportSignal } from './gaze/quality'
import type { MovementReport, SignalReport } from './gaze/quality'
import { createGestureDetector } from './gaze/gesture'
import type { Gesture } from './gaze/gesture'
import { createListener, recognitionAvailable } from './listen'
import type { Listener } from './listen'
import { suggestFor } from './suggest'
import { relativePointer, recenterOffset } from './gaze/position'
import { openFrontCamera, watchCamera } from './camera'
import type { SuggestionSet } from './suggest'
import { BOARDS, TOP_ROW } from './vocabulary'
import type { Tile } from './vocabulary'

const DWELL_MS = 800
const COOLDOWN_MS = 350
const BLINK_SELECT_MS = 600

// A pixel of iris jitter is about 1.5 degrees of gaze once it is levered through the
// 12 mm eyeball radius, which on a 34 x 19 cm screen is 4% across and 8% down. Real
// webcam landmarks jitter by two to four pixels, so the per-frame cursor is unusable no
// matter how good the model is. The only thing that removes independent noise is time,
// and a dwell interface has time to spend: 800 ms is roughly 24 frames, worth about a
// five-fold reduction. These presets trade lag for exactly that.
const STABILITY_PRESETS = {
  responsive: { median: 5, minCutoff: 0.9, beta: 0.012, label: 'responsive' },
  balanced: { median: 9, minCutoff: 0.5, beta: 0.008, label: 'balanced' },
  stable: { median: 15, minCutoff: 0.28, beta: 0.004, label: 'stable' },
} as const
type StabilityKey = keyof typeof STABILITY_PRESETS

// Noise makes the nearest word flicker between neighbours. Requiring the cursor to stay
// on one word for 800 ms unbroken means a single stray frame restarts the timer and the
// dwell may never complete at all. Instead each word charges while it is nearest and
// leaks back while it is not, so a word that wins most frames still gets selected - it
// just takes longer the noisier the signal is.
const DWELL_LEAK = 0.6

const MIX_LABELS: Record<InputMix, string> = {
  head: 'head pointing',
  'head+eye': 'head pointing + gaze',
  eye: 'gaze only',
  'eye+pixels': 'gaze + eye pixels',
}

// Waiting for the head to settle beats guessing a fixed delay, so the dot advances when
// the signal goes quiet and only falls back to the timeout if it never does.
//
// These are all floors on human movement, not arbitrary pacing, and every one of them
// was previously set too low.
//
// Nobody reacts to a dot appearing in under about 250 ms, and turning the head to it
// takes several hundred more. A settle floor shorter than that lets recording begin
// before the user has set off, and a settle ceiling shorter than a full turn cuts them
// off in the middle of one. Either way the frames get labelled with a position the user
// is not looking at, which is worse than collecting nothing - a wrong label does not
// average out, it drags the fit.
const MIN_SETTLE_MS = 700
const MAX_SETTLE_MS = 2600
const RECORD_MS = 800
// The check dots are the reported accuracy, and there are only five of them, so they get
// longer to keep that number from being mostly noise.
const CHECK_RECORD_MS = 900
// Long enough that a pause partway through a turn does not read as having arrived.
const STABLE_WINDOW_MS = 300
// How far the head has to travel before stillness is allowed to mean "arrived". About a
// degree and a half: well above tremor, far below the smallest gap between two dots.
const SETTLE_TRAVEL = 0.03
// In head units: radians for rotation, face-widths for position. 0.012 rad is 0.7 deg,
// just above postural tremor, so it clears once the head has stopped swinging but does
// not wait for a stillness nobody can hold.
const STABLE_THRESHOLD = 0.012

// Fixations rather than a moving dot, because a dot crossing the board in three seconds
// is near the 30 deg/sec limit of smooth pursuit and anyone who cannot keep up starts
// making catch-up movements that mislabel every frame after them.
//
// Twelve positions covering the whole board, then four of them again on the way back.
//
// The dominant error in head pointing is slow postural drift, which averaging inside a
// single fixation cannot remove because the samples are correlated; two visits a dozen
// seconds apart are uncorrelated, so the drift cancels between them. Revisiting every
// position is the best of the options swept in model.test.ts but costs twice the dots for
// a third of a percent, and revisiting four spread positions keeps most of that benefit.
// Cutting positions instead of revisits is worse - dropping to three columns costs more
// horizontal accuracy than dropping eight revisits does, and horizontal is the axis with
// the tight budget.
const CALIBRATION_GRID = [0.1, 0.44, 0.78].flatMap((y) =>
  [0.06, 0.35, 0.65, 0.94].map((x) => ({ x, y })),
)
const CALIBRATION_POINTS = [
  ...CALIBRATION_GRID.map((point, group) => ({ ...point, group })),
  ...CALIBRATION_GRID.map((point, group) => ({ ...point, group }))
    .filter((_, index) => index % 3 === 0)
    .reverse(),
]

// Never fitted on, so the error they report is the error you will feel.
const CHECK_POINTS = [
  { x: 0.18, y: 0.16, group: 0 }, { x: 0.82, y: 0.16, group: 1 },
  { x: 0.5, y: 0.45, group: 2 },
  { x: 0.18, y: 0.72, group: 3 }, { x: 0.82, y: 0.72, group: 4 },
]

// Indices into the head signal vector, used for the stability gate and the movement
// report. Must match the order assembled in features.ts.
const HEAD_POS_X = 0
const HEAD_POS_Y = 1
const HEAD_YAW = 7
const HEAD_PITCH = 8

// Where the board stops and the pointer is free to sit. Must match .rest-zone width and
// .board-grid bottom in style.css.
const REST_EDGE = 6.5
const BOARD_FLOOR = 87

// The bottom row is six tiles. Five of them carry suggested answers and the sixth always
// restores the normal board, so a wrong guess never traps anyone in a set of words that
// does not contain what they wanted to say.
const SUGGESTION_SLOTS = 5
const RESTORE_TILE: Tile = { label: 'All words', restore: true, accent: 'folder' }
// Long enough to answer at dwell speed, short enough that the board is not still showing
// answers to a question from two topics ago.
const SUGGESTION_HOLD_MS = 30000

type Turn = { speaker: 'them' | 'you'; text: string; time: string }

// Shown until the first real turn arrives, so the strip reads as a conversation rather
// than an empty box. The first thing actually heard or said clears it.

const EXAMPLE_TURNS: Turn[] = [
  { speaker: 'them', text: 'Morning. Did you sleep alright?', time: '9:02' },
  { speaker: 'you', text: 'No', time: '9:02' },
  { speaker: 'them', text: 'Are you in pain, or was it just noisy?', time: '9:03' },
  { speaker: 'you', text: 'I am in pain', time: '9:03' },
  { speaker: 'them', text: "I'll get the nurse now.", time: '9:03' },
]

type CameraState = 'off' | 'starting' | 'ready' | 'error'
type Stage = 'idle' | 'calibrating' | 'fitting' | 'checking' | 'complete' | 'skipped'
// The clock only reads to the minute, and two picks a second apart are routine, so the
// display time cannot double as an identity.
type SelectionEvent = { id: number; name: string; time: string }
type Observation = { head: number[]; geometric: number[]; appearance: number[] }

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function medianOf(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

function spread(values: number[]) {
  if (values.length < 2) return Infinity
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
}

// The room mic hears the board's own voice too, and would caption it as the other person
// talking. Track when we are the one making noise so those results can be dropped. The
// grace period covers audio already buffered when the utterance ends.
let speakingUntil = 0
let speaking = false
const ECHO_GRACE_MS = 700

function boardIsSpeaking() {
  return speaking || performance.now() < speakingUntil
}

function speak(text: string) {
  if (!text || typeof window === 'undefined' || !('speechSynthesis' in window)) return
  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = 0.95
  speaking = true
  const release = () => {
    speaking = false
    speakingUntil = performance.now() + ECHO_GRACE_MS
  }
  utterance.onend = release
  utterance.onerror = release
  window.speechSynthesis.speak(utterance)
}

function speechAvailable() {
  return typeof window !== 'undefined' && 'speechSynthesis' in window
}

function clockTime() {
  return new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const cursorRef = useRef<HTMLDivElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const eyePreviewRef = useRef<HTMLCanvasElement>(null)

  const landmarkerRef = useRef<FaceLandmarker | null>(null)
  const patchRef = useRef<EyePatchExtractor | null>(null)
  const rafRef = useRef<number | null>(null)
  const loopRunningRef = useRef(false)
  const frameCountRef = useRef(0)
  const cameraSessionRef = useRef(0)
  const cameraWatchRef = useRef<(() => void) | null>(null)
  const suspendedRef = useRef(false)
  const lastVideoTimeRef = useRef(-1)
  const cameraFailureRef = useRef<(message: string) => void>(() => {})
  const [cameraPaused, setCameraPaused] = useState(false)
  const streamRef = useRef<MediaStream | null>(null)
  const neutralRef = useRef<number[] | null>(null)
  const neutralSamplesRef = useRef<number[][]>([])
  const pointerOffsetRef = useRef({ x: 0, y: 0 })
  const recenterAtRef = useRef(0)
  const lastFaceRef = useRef(0)
  const [positioning, setPositioning] = useState(false)
  const [guideOpen, setGuideOpen] = useState(true)
  const guideOpenRef = useRef(true)
  guideOpenRef.current = guideOpen
  const [panelTab, setPanelTab] = useState<'talk' | 'tracking' | 'details'>('talk')
  const [aiEnabled, setAiEnabled] = useState(true)
  const [aiStatus, setAiStatus] = useState('Checking OpenAI…')
  const aiEnabledRef = useRef(true)
  const suggestionRequestRef = useRef<AbortController | null>(null)
  const suggestionVersionRef = useRef(0)
  const aiTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const conversationRef = useRef<{ speaker: 'you' | 'them'; text: string }[]>([])


  const modelRef = useRef<FittedModel | null>(null)
  const mixRef = useRef<InputMix>('head')
  const filterXRef = useRef(new OneEuroFilter(0.9, 0.012))
  const filterYRef = useRef(new OneEuroFilter(0.9, 0.012))
  const historyRef = useRef<{ x: number[]; y: number[] }>({ x: [], y: [] })
  const cursorPosRef = useRef({ x: 50, y: 45 })

  const dwellScoresRef = useRef(new Map<string, number>())
  const dwellRef = useRef({ target: '', progress: 0 })
  const lastFrameRef = useRef(0)
  const stabilityRef2 = useRef<StabilityKey>('balanced')
  const cooldownUntilRef = useRef(0)
  // Centres are kept in percent-of-stage, the same space the model predicts in, so the
  // distance used to pick a word is directly comparable to the error the app reports.
  const rectsRef = useRef<{ element: HTMLElement; centerX: number; centerY: number; radius: number }[]>([])
  const highlightRef = useRef<HTMLElement | null>(null)
  // Shrinks the axis that measured worse, so the word cells stretch along whichever
  // direction the tracker is unreliable in instead of pretending both are equal.
  const axisWeightRef = useRef({ x: 1, y: 1 })
  const blinkSelectRef = useRef(false)
  const blinkFiredRef = useRef(false)

  const modeRef = useRef<'idle' | 'collect' | 'live'>('idle')
  const phaseRef = useRef<'settle' | 'record'>('settle')
  const phaseStartRef = useRef(0)
  const stabilityRef = useRef<{ yaw: number; pitch: number; x: number; y: number; time: number }[]>([])
  // Running extent of head rotation during a calibration run, so the coaching readout
  // can tell the user they are not moving while there is still time to fix it.
  const turnRangeRef = useRef({ minYaw: Infinity, maxYaw: -Infinity, minPitch: Infinity, maxPitch: -Infinity })
  const bufferRef = useRef<Observation[]>([])
  const pointIndexRef = useRef(0)
  const collectingChecksRef = useRef(false)
  const trainRef = useRef<{
    head: number[][]
    geometric: number[][]
    appearance: number[][]
    x: number[]
    y: number[]
    groups: number[]
  }>({ head: [], geometric: [], appearance: [], x: [], y: [], groups: [] })
  const checkRef = useRef<{ observations: Observation[]; x: number[]; y: number[]; groups: number[] }>({
    observations: [],
    x: [],
    y: [],
    groups: [],
  })
  const advanceRef = useRef<() => void>(() => {})

  const [cameraState, setCameraState] = useState<CameraState>('off')
  const [cameraError, setCameraError] = useState('')
  const [resolution, setResolution] = useState('')
  const [stage, setStage] = useState<Stage>('idle')
  const [pointIndex, setPointIndex] = useState(0)
  const [phase, setPhase] = useState<'settle' | 'record'>('settle')
  const [quality, setQuality] = useState<{
    measured: number
    perFrame: number
    errorX: number
    errorY: number
    cv: number
    rows: number
    mix: InputMix
  } | null>(null)
  const [signal, setSignal] = useState<SignalReport | null>(null)
  const [movement, setMovement] = useState<MovementReport | null>(null)
  const [liveTurn, setLiveTurn] = useState({ yaw: 0, pitch: 0 })
  const [listening, setListening] = useState(false)
  const [turns, setTurns] = useState<Turn[]>(EXAMPLE_TURNS)
  const [caption, setCaption] = useState('')
  const [lastHeard, setLastHeard] = useState('')
  const [suggested, setSuggested] = useState<SuggestionSet | null>(null)
  // Held back rather than applied on arrival: moving a tile out from under someone who is
  // already charging it would select a word they never chose. Flushed on the first frame
  // with no dwell in progress.
  const pendingSuggestionRef = useRef<SuggestionSet | null>(null)
  const suggestedAtRef = useRef(0)
  // Picks and seconds from the question being heard to the answer being spoken. This is
  // the number the whole feature exists to move.
  const [answerStat, setAnswerStat] = useState<{ seconds: number; picks: number } | null>(null)
  const askedAtRef = useRef(0)
  const picksRef = useRef(0)
  const [listenError, setListenError] = useState('')
  const showDetails = panelTab === 'details'
  const [gesturesOn, setGesturesOn] = useState(true)
  const gesturesOnRef = useRef(true)
  const gestureRef = useRef(createGestureDetector())
  const restLeftRef = useRef<HTMLDivElement>(null)
  const restRightRef = useRef<HTMLDivElement>(null)
  const listenerRef = useRef<Listener | null>(null)
  const turnsRef = useRef<HTMLDivElement>(null)
  // The strip ships with an example exchange so the layout reads at a glance; the first
  // real turn clears it rather than appending to fiction.
  const liveRef = useRef(false)
  const [boardId, setBoardId] = useState('home')
  // The last thing the board said out loud. Kept on screen because rooms are noisy and
  // synthetic speech is easy to mishear, so the person listening can read it back.
  const [spoken, setSpoken] = useState('')
  const [typed, setTyped] = useState('')
  const [events, setEvents] = useState<SelectionEvent[]>([])
  const eventIdRef = useRef(0)
  // Head pose when the current dot appeared, and whether the head has since travelled
  // far enough for stillness to mean it arrived rather than never left.
  const settleOriginRef = useRef<{ yaw: number; pitch: number } | null>(null)
  const movedRef = useRef(false)
  const [message, setMessage] = useState('Enable the camera to begin.')
  const [blinkSelect, setBlinkSelect] = useState(false)
  const [stability, setStability] = useState<StabilityKey>('balanced')
  const [capabilities, setCapabilities] = useState({
    face: false,
    eyes: false,
    eyeModel: false,
    headPose: false,
    blendshapes: false,
  })
  const [debug, setDebug] = useState({ target: 'none', tracking: 'waiting', blinking: false, gaze: '—', distance: 0 })
  const [stats, setStats] = useState({ selections: 0 })

  // Only counts once per question: the interesting figure is how much work the first
  // answer took, not the total picks in a conversation.
  const recordAnswer = useCallback(() => {
    if (!askedAtRef.current) return
    setAnswerStat({ seconds: (performance.now() - askedAtRef.current) / 1000, picks: picksRef.current })
    askedAtRef.current = 0
  }, [])

  const recordTurn = useCallback((speaker: Turn['speaker'], text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return
    const fresh = !liveRef.current
    liveRef.current = true
    conversationRef.current = [...conversationRef.current, { speaker, text: trimmed }].slice(-6)
    setTurns((current) => [...(fresh ? [] : current), { speaker, text: trimmed, time: clockTime() }].slice(-40))
  }, [])

  // Everything a selection does when it speaks, minus the selection. Shared so typed
  // speech lands in the transcript, the sentence bar and the answer timing the same way a
  // tile does - a sentence the board said is a sentence the board said, however it got there.
  const sayAloud = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      speak(trimmed)
      setSpoken(trimmed)
      recordTurn('you', trimmed)
      picksRef.current += 1
      recordAnswer()
      setEvents((current) => [{ id: eventIdRef.current++, name: trimmed, time: clockTime() }, ...current].slice(0, 6))
      setStats((current) => ({ ...current, selections: current.selections + 1 }))
      setMessage(`Said "${trimmed}"`)
    },
    [recordTurn, recordAnswer],
  )

  const speakTyped = useCallback(() => {
    const text = typed.trim() || spoken
    if (!text) return
    sayAloud(text)
    setTyped('')
  }, [typed, spoken, sayAloud])

  useEffect(() => {
    if (!guideOpen) return
    dwellRef.current = { target: '', progress: 0 }
    dwellScoresRef.current.clear()
    highlightRef.current?.classList.remove('dwell-near')
    gestureRef.current = createGestureDetector()
  }, [guideOpen])

  const cancelSuggestions = useCallback(() => {
    suggestionVersionRef.current++
    suggestionRequestRef.current?.abort()
    if (aiTimerRef.current) clearTimeout(aiTimerRef.current)
    pendingSuggestionRef.current = null
  }, [])

  useEffect(() => {
    aiEnabledRef.current = aiEnabled
    cancelSuggestions()
    const abort = new AbortController()
    if (!aiEnabled) { setAiStatus('Local replies'); return }
    const check = () => fetch('/api/suggestions/status', { signal: abort.signal })
      .then(r => r.json()).then(data => {
        if (!abort.signal.aborted) setAiStatus(data.message || (data.configured ? 'OpenAI configured' : 'Add your key to enable OpenAI'))
      }).catch(() => { if (!abort.signal.aborted) setAiStatus('Local replies · server unavailable') })
    void check()
    const timer = window.setInterval(check, 15000)
    return () => { abort.abort(); window.clearInterval(timer) }
  }, [aiEnabled, cancelSuggestions])

  useEffect(() => {
    if (!listening) return
    const listener = createListener({
      onInterim: setCaption,
      onFinal: (text) => {
        setCaption('')
        if (boardIsSpeaking()) return
        cancelSuggestions()
        setLastHeard(text)
        recordTurn('them', text)
        const next = suggestFor(text, SUGGESTION_SLOTS)
        pendingSuggestionRef.current = next
        askedAtRef.current = performance.now()
        picksRef.current = 0
        setAnswerStat(null)
        if (!aiEnabledRef.current) return
        const version = suggestionVersionRef.current
        setAiStatus('Finding replies…')
        // Combine nearby final transcription fragments and discard stale results.
        aiTimerRef.current = setTimeout(async () => {
          const abort = new AbortController()
          suggestionRequestRef.current = abort
          const timeout = setTimeout(() => abort.abort(), 12000)
          try {
            const response = await fetch('/api/suggestions', {
              method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: abort.signal,
              body: JSON.stringify({ text: text.slice(0, 1500), context: conversationRef.current }),
            })
            if (!response.ok) {
              const problem = await response.json().catch(() => ({}))
              throw new Error(problem.error || 'OpenAI unavailable · local replies')
            }
            const result = await response.json() as SuggestionSet
            if (version !== suggestionVersionRef.current) return
            pendingSuggestionRef.current = result
            setAiStatus('OpenAI replies ready')
          } catch (error) {
            if (version === suggestionVersionRef.current) setAiStatus(error instanceof Error && error.name !== 'AbortError' ? error.message : 'OpenAI timed out · local replies')
          } finally { clearTimeout(timeout) }
        }, 650)
      },
      onError: (problem) => {
        setListenError(problem)
        if (problem.startsWith('Microphone blocked')) setListening(false)
      },
    })
    if (!listener) {
      setListenError('This browser has no speech recognition. Chrome or Edge will work.')
      setListening(false)
      return
    }
    listenerRef.current = listener
    setListenError('')
    listener.start()
    return () => {
      listener.stop()
      listenerRef.current = null
      cancelSuggestions()
      setCaption('')
    }
  }, [listening, recordTurn, cancelSuggestions])

  useEffect(() => {
    const strip = turnsRef.current
    if (strip) strip.scrollLeft = strip.scrollWidth
  }, [turns])

  // Swapping the bottom row is only safe while nothing is charging: anyone using this has
  // aimed at a tile and is holding on it, and pulling that tile away mid-dwell would
  // commit whatever slid into its place. This deliberately does not live in the tracking
  // loop — tying it to face detection meant the board froze whenever the camera lost the
  // face, which is exactly when someone is likely to be turning to listen to a question.
  useEffect(() => {
    const timer = window.setInterval(() => {
      if (dwellRef.current.progress > 0) return
      const now = performance.now()
      if (pendingSuggestionRef.current) {
        const next = pendingSuggestionRef.current
        pendingSuggestionRef.current = null
        suggestedAtRef.current = now
        // Scores are keyed by label and the labels are about to change. They are already
        // empty at zero progress; this is insurance against a stale label outliving its
        // tile and never decaying.
        dwellScoresRef.current.clear()
        setSuggested(next)
        setMessage(`Reply options: ${next.because}. Choose what you want to say.`)
      } else if (suggestedAtRef.current && now - suggestedAtRef.current > SUGGESTION_HOLD_MS) {
        suggestedAtRef.current = 0
        setSuggested(null)
      }
    }, 120)
    return () => window.clearInterval(timer)
  }, [])

  const refreshRects = useCallback(() => {
    const stageElement = stageRef.current
    if (!stageElement) return
    const bounds = stageElement.getBoundingClientRect()
    rectsRef.current = Array.from(stageElement.querySelectorAll<HTMLElement>('[data-dwell-target]')).map((element) => {
      const rect = element.getBoundingClientRect()
      const halfWidth = (rect.width / bounds.width) * 50
      const halfHeight = (rect.height / bounds.height) * 50
      return {
        element,
        centerX: ((rect.left + rect.width / 2 - bounds.left) / bounds.width) * 100,
        centerY: ((rect.top + rect.height / 2 - bounds.top) / bounds.height) * 100,
        // Generous catch radius; the nearest-target rule below is what really decides.
        radius: Math.max(halfWidth, halfHeight) * 1.9,
      }
    })
    const overlay = overlayRef.current
    if (overlay) {
      overlay.width = Math.round(bounds.width)
      overlay.height = Math.round(bounds.height)
    }
  }, [])

  const commitSelection = useCallback((element: HTMLElement) => {
    const label = element.dataset.dwellTarget ?? ''
    const action = element.dataset.dwellAction ?? 'word'
    const phrase = element.dataset.dwellSpeech || label
    cooldownUntilRef.current = performance.now() + COOLDOWN_MS
    dwellRef.current = { target: '', progress: 0 }
    dwellScoresRef.current.clear()
    element.style.setProperty('--dwell', '0%')
    setEvents((current) => [{ id: eventIdRef.current++, name: label, time: clockTime() }, ...current].slice(0, 6))
    picksRef.current += 1

    if (action === 'restore' || action === 'folder') cancelSuggestions()
    if (action === 'restore') {
      pendingSuggestionRef.current = null
      suggestedAtRef.current = 0
      setSuggested(null)
      setMessage('Back to the usual words')
      return
    }
    if (action === 'folder') {
      pendingSuggestionRef.current = null
      suggestedAtRef.current = 0
      setSuggested(null)
      setBoardId(element.dataset.dwellBoard ?? 'home')
      setMessage(`Opened ${label}`)
      return
    }

    // Everything that is not navigation is speech, and it goes out immediately.
    speak(phrase)
    setSpoken(phrase)
    recordTurn('you', phrase)
    recordAnswer()
    setStats((current) => ({ ...current, selections: current.selections + 1 }))
    setMessage(`Said "${phrase}"`)
  }, [recordTurn, recordAnswer, cancelSuggestions])

  // Nod and shake answer without having to aim at anything, which matters most for the two
  // words people need fastest. The head sweeps across the board during the movement, so
  // whatever dwell built up on the tiles it passed over has to be thrown away.
  const fireGesture = useCallback(
    (gesture: Gesture) => {
      const label = gesture === 'nod' ? 'Yes' : 'No'
      const phrase = TOP_ROW.find((tile) => tile.label === label)?.speech ?? label

      dwellRef.current = { target: '', progress: 0 }
      dwellScoresRef.current.clear()
      highlightRef.current?.classList.remove('dwell-near')
      highlightRef.current?.style.setProperty('--dwell', '0%')
      highlightRef.current = null
      cooldownUntilRef.current = performance.now() + COOLDOWN_MS

      const hit = rectsRef.current.find((entry) => entry.element.dataset.dwellTarget === label)?.element
      if (hit) {
        hit.classList.add('gesture-hit')
        window.setTimeout(() => hit.classList.remove('gesture-hit'), 650)
      }

      speak(phrase)
      recordTurn('you', phrase)
      picksRef.current += 1
      recordAnswer()
      setStats((current) => ({ ...current, selections: current.selections + 1 }))
      setEvents((current) => [{ id: eventIdRef.current++, name: `${label} · ${gesture}`, time: clockTime() }, ...current].slice(0, 6))
      setMessage(gesture === 'nod' ? `Nodded — said "${phrase}"` : `Shook head — said "${phrase}"`)
    },
    [recordTurn, recordAnswer],
  )
  const fireGestureRef = useRef(fireGesture)
  useEffect(() => {
    fireGestureRef.current = fireGesture
  }, [fireGesture])

  useEffect(() => {
    gesturesOnRef.current = gesturesOn
    if (!gesturesOn) gestureRef.current.reset()
  }, [gesturesOn])

  const updateDwell = useCallback(
    (x: number, y: number, now: number, blinking: boolean) => {
      const cursor = cursorRef.current
      if (!cursor) return

      const dt = clamp(now - lastFrameRef.current, 0, 100)
      lastFrameRef.current = now
      const scores = dwellScoresRef.current

      // Nearest target rather than the one under the cursor. Requiring the cursor to
      // land inside a word means the tolerance is half a tile; picking the closest
      // word means it is half the gap between words, which is far more forgiving.
      // Both side edges park the pointer, as does the label strip under the board.
      const restingLeft = x <= REST_EDGE
      const restingRight = x >= 100 - REST_EDGE
      restLeftRef.current?.classList.toggle('active', restingLeft)
      restRightRef.current?.classList.toggle('active', restingRight)
      const resting = restingLeft || restingRight || y >= BOARD_FLOOR

      let nearest: HTMLElement | null = null
      let bestDistance = Infinity
      if (!resting && !blinking && now >= cooldownUntilRef.current) {
        const weight = axisWeightRef.current
        const charging = dwellRef.current.target
        for (const entry of rectsRef.current) {
          let distance = Math.hypot((x - entry.centerX) * weight.x, (y - entry.centerY) * weight.y)
          // The word already charging holds on unless another is clearly closer, so the
          // cursor sitting on a boundary does not thrash between two neighbours.
          if (entry.element.dataset.dwellTarget === charging) distance *= 0.85
          if (distance < bestDistance && distance < entry.radius) {
            bestDistance = distance
            nearest = entry.element
          }
        }
      }

      if (now < cooldownUntilRef.current) scores.clear()
      const nearestLabel = nearest?.dataset.dwellTarget ?? ''
      for (const entry of rectsRef.current) {
        const label = entry.element.dataset.dwellTarget ?? ''
        if (!label) continue
        const current = scores.get(label) ?? 0
        const next = label === nearestLabel ? current + dt : current - dt * DWELL_LEAK
        if (next <= 0) scores.delete(label)
        else scores.set(label, Math.min(next, DWELL_MS))
      }

      let leaderLabel = ''
      let leaderScore = 0
      for (const [label, score] of scores) {
        if (score > leaderScore) {
          leaderScore = score
          leaderLabel = label
        }
      }

      const leader = leaderLabel
        ? (rectsRef.current.find((entry) => entry.element.dataset.dwellTarget === leaderLabel)?.element ?? null)
        : null
      const progress = clamp(leaderScore / DWELL_MS, 0, 1)
      dwellRef.current = { target: leaderLabel, progress }

      if (highlightRef.current !== leader) {
        highlightRef.current?.classList.remove('dwell-near')
        highlightRef.current?.style.setProperty('--dwell', '0%')
        leader?.classList.add('dwell-near')
        highlightRef.current = leader
      }

      cursor.style.setProperty('--progress', `${progress * 100}%`)
      leader?.style.setProperty('--dwell', `${progress * 100}%`)
      if (progress >= 1 && leader) commitSelection(leader)
    },
    [commitSelection],
  )

  const detectFrame = useCallback(() => {
    if (!loopRunningRef.current) return
    rafRef.current = requestAnimationFrame(detectFrame)
    if (document.hidden || suspendedRef.current) return
    const video = videoRef.current
    const landmarker = landmarkerRef.current
    const cursor = cursorRef.current
    const patcher = patchRef.current
    const now = performance.now()
    frameCountRef.current += 1

    if (video && landmarker && cursor && patcher && video.readyState >= 2 && video.videoWidth > 0) {
      if (video.paused || lastVideoTimeRef.current === video.currentTime) return
      lastVideoTimeRef.current = video.currentTime
      let result
      try { result = landmarker.detectForVideo(video, now) }
      catch {
        cameraFailureRef.current('Tracking was interrupted. Tap Enable camera to restart it.')
        return
      }
      const landmarks = result.faceLandmarks?.[0]
      const signals = extractSignals(
        landmarks,
        result.faceBlendshapes?.[0]?.categories,
        result.facialTransformationMatrixes?.[0]?.data,
        video.videoWidth,
        video.videoHeight,
      )

      if (signals && landmarks) {
        lastFaceRef.current = now
        if (!neutralRef.current && !signals.blinking) {
          neutralSamplesRef.current.push([...signals.head])
          if (neutralSamplesRef.current.length >= 20) {
            neutralRef.current = signals.head.map((_, index) => medianOf(neutralSamplesRef.current.map(row => row[index])))
            neutralSamplesRef.current = []
            setMessage('Ready from your current position. Calibrate for better accuracy, or skip to use the board.')
          }
        }
        const leftBox = eyeBoxFrom(signals.leftCorners, video.videoWidth, video.videoHeight)
        const rightBox = eyeBoxFrom(signals.rightCorners, video.videoWidth, video.videoHeight)
        const appearance = patcher.extract(video, leftBox, rightBox) ?? []

        const overlay = overlayRef.current?.getContext('2d')
        if (overlay) {
          drawDetection(overlay, landmarks, video.videoWidth, video.videoHeight, { left: leftBox, right: rightBox }, {
            face: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
            leftEye: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            rightEye: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            irises: [...FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, ...FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS],
          })
        }

        if (frameCountRef.current % 3 === 0) {
          const preview = eyePreviewRef.current?.getContext('2d')
          if (preview) patcher.drawPreview(preview, video, leftBox, rightBox)
        }

        if (modeRef.current === 'collect' && signals.blinking) { phaseRef.current = 'settle'; phaseStartRef.current = now; bufferRef.current = []; stabilityRef.current = []; setPhase('settle') }
        if (modeRef.current === 'collect' && !signals.blinking) {
          const elapsed = now - phaseStartRef.current
          const extent = turnRangeRef.current
          extent.minYaw = Math.min(extent.minYaw, signals.head[HEAD_YAW])
          extent.maxYaw = Math.max(extent.maxYaw, signals.head[HEAD_YAW])
          extent.minPitch = Math.min(extent.minPitch, signals.head[HEAD_PITCH])
          extent.maxPitch = Math.max(extent.maxPitch, signals.head[HEAD_PITCH])
          if (frameCountRef.current % 6 === 0) {
            setLiveTurn({
              yaw: ((extent.maxYaw - extent.minYaw) * 180) / Math.PI,
              pitch: ((extent.maxPitch - extent.minPitch) * 180) / Math.PI,
            })
          }
          if (phaseRef.current === 'settle') {
            // Gate on the head, which is what the model is being fitted on. Gating on the
            // eye projection instead let recording begin while the head was still
            // swinging toward the dot, and every one of those frames got labelled with a
            // position the user was not yet aiming at.
            const window = stabilityRef.current
            window.push({
              yaw: signals.head[HEAD_YAW],
              pitch: signals.head[HEAD_PITCH],
              x: signals.head[HEAD_POS_X],
              y: signals.head[HEAD_POS_Y],
              time: now,
            })
            while (window.length > 1 && now - window[0].time > STABLE_WINDOW_MS) window.shift()

            const settled =
              window.length >= 4 &&
              now - window[0].time >= STABLE_WINDOW_MS - 20 &&
              spread(window.map((entry) => entry.yaw)) < STABLE_THRESHOLD &&
              spread(window.map((entry) => entry.pitch)) < STABLE_THRESHOLD &&
              spread(window.map((entry) => entry.x)) < STABLE_THRESHOLD &&
              spread(window.map((entry) => entry.y)) < STABLE_THRESHOLD

            // Stillness on its own cannot tell "arrived and holding" from "has not set
            // off yet", and at the start of a dot those are the same picture: a head
            // perfectly steady, aimed at the previous dot. Waiting for real travel first
            // is what separates them.
            const origin =
              settleOriginRef.current ??
              (settleOriginRef.current = { yaw: signals.head[HEAD_YAW], pitch: signals.head[HEAD_PITCH] })
            if (
              Math.abs(signals.head[HEAD_YAW] - origin.yaw) > SETTLE_TRAVEL ||
              Math.abs(signals.head[HEAD_PITCH] - origin.pitch) > SETTLE_TRAVEL
            ) {
              movedRef.current = true
            }

            if ((elapsed > MIN_SETTLE_MS && settled && movedRef.current) || elapsed > MAX_SETTLE_MS) {
              phaseRef.current = 'record'
              phaseStartRef.current = now
              stabilityRef.current = []
              bufferRef.current = []
              setPhase('record')
            }
          } else {
            bufferRef.current.push({ head: signals.head, geometric: signals.base, appearance })
            if (elapsed > (collectingChecksRef.current ? CHECK_RECORD_MS : RECORD_MS)) advanceRef.current()
          }
        }

        const model = modelRef.current
        const raw = model
          ? predict(model, buildFeatures(signals.head, signals.base, appearance, mixRef.current))
          : relativePointer(signals.head, neutralRef.current ?? signals.head)
        if (recenterAtRef.current && now >= recenterAtRef.current && !signals.blinking) {
          if (model) pointerOffsetRef.current = recenterOffset(raw)
          else { neutralRef.current = [...signals.head]; pointerOffsetRef.current = { x: 0, y: 0 } }
          recenterAtRef.current = 0
          setPositioning(false)
          historyRef.current = { x: [], y: [] }
          filterXRef.current.reset(); filterYRef.current.reset()
          cooldownUntilRef.current = now + 1000
          setMessage('Pointing position reset. You can stay where you are comfortable.')
        }
        raw.x += pointerOffsetRef.current.x
        raw.y += pointerOffsetRef.current.y

        if (!signals.blinking && Number.isFinite(raw.x) && Number.isFinite(raw.y)) {
          const history = historyRef.current
          const window = STABILITY_PRESETS[stabilityRef2.current].median
          history.x.push(raw.x)
          history.y.push(raw.y)
          while (history.x.length > window) history.x.shift()
          while (history.y.length > window) history.y.shift()
          cursorPosRef.current = {
            x: clamp(filterXRef.current.filter(medianOf(history.x), now), 2, 98),
            y: clamp(filterYRef.current.filter(medianOf(history.y), now), 2, 98),
          }
        }

        const { x, y } = cursorPosRef.current
        cursor.style.left = `${x}%`
        cursor.style.top = `${y}%`
        if (modeRef.current === 'live' && neutralRef.current && !recenterAtRef.current && !guideOpenRef.current) updateDwell(x, y, now, signals.blinking)

        // Read gestures off the head pose directly rather than the smoothed pointer: the
        // median window and One Euro filter exist to damp exactly the motion a nod is
        // made of, so a shake barely shows up by the time it reaches the cursor.
        if (modeRef.current === 'live' && neutralRef.current && !recenterAtRef.current && !guideOpenRef.current && gesturesOnRef.current && !signals.blinking) {
          const gesture = gestureRef.current.push(now, signals.head[HEAD_YAW], signals.head[HEAD_PITCH])
          if (gesture) fireGestureRef.current(gesture)
        } else if (modeRef.current !== 'live') {
          gestureRef.current.reset()
        }

        if (blinkSelectRef.current && modeRef.current === 'live') {
          if (signals.blinking && !blinkFiredRef.current && dwellRef.current.progress > BLINK_SELECT_MS / DWELL_MS) {
            const target = rectsRef.current.find((entry) => entry.element.dataset.dwellTarget === dwellRef.current.target)
            if (target && now >= cooldownUntilRef.current) commitSelection(target.element)
            blinkFiredRef.current = true
          }
          if (!signals.blinking) blinkFiredRef.current = false
        }

        if (frameCountRef.current % 10 === 0) {
          setDebug({
            target: dwellRef.current.target || 'none',
            tracking: model ? MIX_LABELS[mixRef.current] : 'head pose, uncalibrated',
            blinking: signals.blinking,
            gaze: `${signals.gazeAngleXDeg.toFixed(1)}° / ${signals.gazeAngleYDeg.toFixed(1)}°`,
            distance: signals.distanceCm,
          })
          setCapabilities({
            face: true,
            eyes: appearance.length > 0,
            eyeModel: signals.hasEyeModel,
            headPose: signals.hasHeadPose,
            // Carries MediaPipe's own eyeLookUp/Down estimate, the one signal here that
            // was trained on real gaze rather than derived from geometry.
            blendshapes: signals.hasBlendshapes,
          })
        }
      } else {
        if (modeRef.current === 'collect') { phaseRef.current = 'settle'; phaseStartRef.current = now; bufferRef.current = []; stabilityRef.current = []; setPhase('settle') }
        dwellRef.current = { target: '', progress: 0 }
        dwellScoresRef.current.clear()
        highlightRef.current?.classList.remove('dwell-near')
        cursor.style.setProperty('--progress', '0%')
        const overlay = overlayRef.current?.getContext('2d')
        if (overlay) overlay.clearRect(0, 0, overlay.canvas.width, overlay.canvas.height)
        if (frameCountRef.current % 10 === 0) {
          setDebug((current) => ({ ...current, target: 'none', tracking: 'no face detected' }))
          setCapabilities((current) => ({ ...current, face: false }))
        }
      }
    }

  }, [commitSelection, updateDwell])

  const beginPoint = useCallback((index: number, checks: boolean) => {
    pointIndexRef.current = index
    collectingChecksRef.current = checks
    phaseRef.current = 'settle'
    phaseStartRef.current = performance.now()
    stabilityRef.current = []
    settleOriginRef.current = null
    movedRef.current = false
    bufferRef.current = []
    modeRef.current = 'collect'
    setPointIndex(index)
    setPhase('settle')
  }, [])

  const finish = useCallback(() => {
    modeRef.current = 'live'
    const model = modelRef.current
    const data = checkRef.current
    const rows = data.observations.map((observation) =>
      buildFeatures(observation.head, observation.geometric, observation.appearance, mixRef.current),
    )
    // Per-frame error is what the raw model produces; the dwell figure is what someone
    // holding a fixation actually gets, once 800 ms of averaging has cancelled most of
    // the independent landmark noise. Only the second one is comparable to the board's
    // tolerance, so it is the number shown as "measured error".
    const perFrame = model ? evaluate(model, rows, data.x, data.y) : null
    const dwellError = model ? evaluateFixations(model, rows, data.x, data.y, data.groups) : null
    const axes = dwellError ?? (model ? evaluateAxes(model, rows, data.x, data.y) : null)

    if (axes) {
      // Down-weight only the worse axis, floored so one bad axis cannot collapse the
      // metric into a single dimension.
      const ratio = clamp(Math.min(axes.x, axes.y) / Math.max(axes.x, axes.y, 1e-6), 0.35, 1)
      axisWeightRef.current = axes.y > axes.x ? { x: 1, y: ratio } : { x: ratio, y: 1 }
    }

    setStage('complete')
    setQuality({
      measured: dwellError?.total ?? perFrame ?? 0,
      perFrame: perFrame ?? 0,
      errorX: axes?.x ?? 0,
      errorY: axes?.y ?? 0,
      cv: model?.cvErrorPercent ?? 0,
      rows: model?.rows ?? 0,
      mix: mixRef.current,
    })
    setMessage('Ready. Point your nose at a word and hold still to say it.')
  }, [])

  const fitAndCheck = useCallback(() => {
    modeRef.current = 'idle'
    setStage('fitting')
    setMessage('Working out how your head maps to the board…')

    // Yield a frame so the message paints before the solve blocks the thread.
    const session = cameraSessionRef.current
    window.setTimeout(() => {
      if (session !== cameraSessionRef.current || modeRef.current !== 'idle') return
      const train = trainRef.current
      setMovement(reportMovement(train.head, train.groups, HEAD_YAW, HEAD_PITCH))

      const choice = fitBest(train.head, train.geometric, train.appearance, train.x, train.y, train.groups)
      if (!choice) {
        modeRef.current = 'live'
        setStage('skipped')
        setMessage('Could not calibrate — not enough steady frames. The board still works, just less accurately.')
        return
      }

      modelRef.current = choice.model
      mixRef.current = choice.mix

      // Scored on what the model outputs rather than on one hand-picked input column,
      // so the reading stays meaningful whichever signals cross-validation settled on.
      const fitted = train.geometric.map((row, index) =>
        predict(choice.model, buildFeatures(train.head[index], row, train.appearance[index] ?? [], choice.mix)),
      )
      setSignal(
        reportSignal(
          fitted.map((point) => point.x),
          fitted.map((point) => point.y),
          train.x,
          train.y,
          train.groups,
        ),
      )

      filterXRef.current.reset()
      filterYRef.current.reset()
      historyRef.current = { x: [], y: [] }
      checkRef.current = { observations: [], x: [], y: [], groups: [] }
      setStage('checking')
      setMessage('Nearly done — five more dots to check how accurate it is.')
      beginPoint(0, true)
    }, 60)
  }, [beginPoint])

  const advance = useCallback(() => {
    const index = pointIndexRef.current
    const checks = collectingChecksRef.current
    const points = checks ? CHECK_POINTS : CALIBRATION_POINTS
    const point = points[index]
    // Reject on the head rows, since those are what the model is fitted on. Rejecting on
    // the eye rows threw away frames where the head was perfectly steady and kept frames
    // where it was not.
    const kept = new Set(rejectOutliers(bufferRef.current.map((observation) => observation.head)))

    for (const observation of bufferRef.current) {
      if (!kept.has(observation.head)) continue
      if (checks) {
        checkRef.current.observations.push(observation)
        checkRef.current.x.push(point.x * 100)
        checkRef.current.y.push(point.y * 100)
        checkRef.current.groups.push(point.group)
      } else {
        trainRef.current.head.push(observation.head)
        trainRef.current.geometric.push(observation.geometric)
        if (observation.appearance.length > 0) trainRef.current.appearance.push(observation.appearance)
        trainRef.current.x.push(point.x * 100)
        trainRef.current.y.push(point.y * 100)
        trainRef.current.groups.push(point.group)
      }
    }
    bufferRef.current = []

    const next = index + 1
    if (next < points.length) {
      beginPoint(next, checks)
      return
    }
    modeRef.current = 'idle'
    if (checks) finish()
    else fitAndCheck()
  }, [beginPoint, finish, fitAndCheck])

  useEffect(() => {
    advanceRef.current = advance
  }, [advance])

  const startCalibration = useCallback(() => {
    if (cameraState !== 'ready') {
      setMessage('Enable the camera before calibrating.')
      return
    }
    setGuideOpen(false)
    recenterAtRef.current = 0
    setPositioning(false)
    pointerOffsetRef.current = { x: 0, y: 0 }
    modelRef.current = null
    mixRef.current = 'head'
    axisWeightRef.current = { x: 1, y: 1 }
    trainRef.current = { head: [], geometric: [], appearance: [], x: [], y: [], groups: [] }
    checkRef.current = { observations: [], x: [], y: [], groups: [] }
    setQuality(null)
    setSignal(null)
    setMovement(null)
    setLiveTurn({ yaw: 0, pitch: 0 })
    turnRangeRef.current = { minYaw: Infinity, maxYaw: -Infinity, minPitch: Infinity, maxPitch: -Infinity }
    setStage('calibrating')
    setMessage('Stay in your comfortable position. Turn toward each dot as far as comfortable, then pause.')
    beginPoint(0, false)
  }, [cameraState, beginPoint])

  const skipCalibration = useCallback(() => {
    setGuideOpen(false)
    modeRef.current = 'live'
    modelRef.current = null
    pointerOffsetRef.current = { x: 0, y: 0 }
    setStage('skipped')
    setMessage('Board ready. Click a word, or enable the camera to point from your comfortable position.')
  }, [])

  const releaseCamera = useCallback(() => {
    cameraSessionRef.current++
    cameraWatchRef.current?.()
    cameraWatchRef.current = null
    suspendedRef.current = false
    lastVideoTimeRef.current = -1
    loopRunningRef.current = false
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    const video = videoRef.current
    if (video) { video.pause(); video.srcObject = null }
    landmarkerRef.current?.close()
    landmarkerRef.current = null
    patchRef.current = null
    modeRef.current = 'idle'
    modelRef.current = null
    neutralRef.current = null
    neutralSamplesRef.current = []
    pointerOffsetRef.current = { x: 0, y: 0 }
    recenterAtRef.current = 0
    lastFaceRef.current = 0
    dwellRef.current = { target: '', progress: 0 }
    dwellScoresRef.current.clear()
    highlightRef.current?.classList.remove('dwell-near')
    gestureRef.current = createGestureDetector()
    historyRef.current = { x: [], y: [] }
    filterXRef.current.reset(); filterYRef.current.reset()
    const overlay = overlayRef.current?.getContext('2d')
    if (overlay) overlay.clearRect(0, 0, overlay.canvas.width, overlay.canvas.height)
    cursorRef.current?.style.setProperty('--progress', '0%')
  }, [])

  const stopCamera = useCallback(() => {
    releaseCamera()
    setCameraState('off'); setCameraPaused(false); setPositioning(false); setQuality(null)
    setSignal(null); setMovement(null)
    setStage('skipped')
    setCapabilities(current => ({ ...current, face: false }))
    setMessage('Camera stopped. You can still click words and use captions.')
  }, [releaseCamera])

  cameraFailureRef.current = (problem) => {
    releaseCamera()
    setCameraState('error'); setCameraPaused(false); setStage('skipped')
    setPositioning(false); setQuality(null)
    setCameraError(problem); setMessage(problem)
  }

  const resumeCamera = useCallback(async () => {
    const video = videoRef.current
    const track = streamRef.current?.getVideoTracks()[0]
    const session = cameraSessionRef.current
    if (!video || !track) return
    if (track.readyState === 'ended') {
      cameraFailureRef.current('The phone stopped the camera. Tap Enable camera to reconnect.')
      return
    }
    if (document.hidden || track.muted) return
    try {
      await video.play()
      if (session !== cameraSessionRef.current) return
      lastVideoTimeRef.current = -1
      historyRef.current = { x: [], y: [] }
      filterXRef.current.reset(); filterYRef.current.reset()
      if (modeRef.current === 'collect') {
        phaseRef.current = 'settle'; phaseStartRef.current = performance.now()
        stabilityRef.current = []; bufferRef.current = []; setPhase('settle')
      }
      suspendedRef.current = false
      setCameraPaused(false)
      cooldownUntilRef.current = performance.now() + 1000
      setMessage('Camera resumed. Check your pointing position before choosing a word.')
    } catch {
      if (session === cameraSessionRef.current) setCameraPaused(true)
    }
  }, [])

  const resetPosition = useCallback(() => {
    if (performance.now() - lastFaceRef.current > 750 || !lastFaceRef.current) {
      setMessage('Keep your face visible to reset pointing. You do not need to sit in the centre.')
      return
    }
    dwellRef.current = { target: '', progress: 0 }; dwellScoresRef.current.clear()
    recenterAtRef.current = performance.now() + 2500
    setPositioning(true)
    setMessage('Look at the centre of the board from your comfortable position. Resetting in 3 seconds…')
  }, [])

  const startCamera = useCallback(async () => {
    releaseCamera()
    const session = cameraSessionRef.current
    setCameraState('starting'); setCameraPaused(false); setCameraError(''); setQuality(null)
    setStage('idle'); setPositioning(false)
    let stream: MediaStream | null = null
    try {
      if (!window.isSecureContext) throw new Error('Camera access needs HTTPS. Open the secure site link on your phone.')
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot access the camera. Open the site in Safari or Chrome.')
      stream = await openFrontCamera(constraints => navigator.mediaDevices.getUserMedia(constraints), matchMedia('(pointer: coarse)').matches)
      if (session !== cameraSessionRef.current) { stream.getTracks().forEach(track => track.stop()); return }
      streamRef.current = stream
      const video = videoRef.current
      if (!video) throw new Error('Camera preview is unavailable.')
      video.muted = true
      video.playsInline = true
      video.srcObject = stream
      await video.play()
      const vision = await FilesetResolver.forVisionTasks('/mediapipe')
      if (session !== cameraSessionRef.current) return
      const options = {
        baseOptions: { modelAssetPath: '/face_landmarker.task', delegate: 'GPU' as const },
        runningMode: 'VIDEO' as const, numFaces: 1,
        outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
      }
      const detector = await FaceLandmarker.createFromOptions(vision, options).catch(() =>
        FaceLandmarker.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } }),
      )
      if (session !== cameraSessionRef.current) { detector.close(); return }
      landmarkerRef.current = detector
      patchRef.current = new EyePatchExtractor()
      const track = stream.getVideoTracks()[0]
      if (!track || track.readyState === 'ended') throw new Error('Camera disconnected. Tap Enable camera to try again.')
      cameraWatchRef.current = watchCamera(track, document, {
        pause: () => {
          if (session !== cameraSessionRef.current) return
          suspendedRef.current = true
          setCameraPaused(true)
          dwellRef.current = { target: '', progress: 0 }; dwellScoresRef.current.clear()
          gestureRef.current = createGestureDetector()
          highlightRef.current?.classList.remove('dwell-near')
          if (modeRef.current === 'collect') {
            phaseRef.current = 'settle'; phaseStartRef.current = performance.now()
            stabilityRef.current = []; bufferRef.current = []; setPhase('settle')
          }
          setMessage('Camera paused by your device. Return to this page to resume.')
        },
        resume: () => { if (session === cameraSessionRef.current) void resumeCamera() },
        ended: () => { if (session === cameraSessionRef.current) cameraFailureRef.current('Camera disconnected. Tap Enable camera to reconnect.') },
      })
      suspendedRef.current = document.hidden || track.muted
      setCameraPaused(suspendedRef.current)
      const settings = track.getSettings()
      setResolution(`${settings?.width ?? video.videoWidth}×${settings?.height ?? video.videoHeight}`)
      setCameraState('ready')
      modeRef.current = 'live'
      setMessage('Sit comfortably and look toward the board. Learning your starting position…')
      refreshRects()
      loopRunningRef.current = true
      rafRef.current = requestAnimationFrame(detectFrame)
    } catch (error) {
      stream?.getTracks().forEach(track => track.stop())
      if (session !== cameraSessionRef.current) return
      releaseCamera()
      setCameraState('error')
      setCameraError(error instanceof Error ? error.message : 'Camera or model startup failed.')
      setMessage('Camera unavailable. You can still use the board by clicking.')
      setStage('skipped')
    }
  }, [detectFrame, refreshRects, releaseCamera, resumeCamera])

  useEffect(() => {
    blinkSelectRef.current = blinkSelect
  }, [blinkSelect])

  useEffect(() => {
    stabilityRef2.current = stability
    const preset = STABILITY_PRESETS[stability]
    filterXRef.current = new OneEuroFilter(preset.minCutoff, preset.beta)
    filterYRef.current = new OneEuroFilter(preset.minCutoff, preset.beta)
    historyRef.current = { x: [], y: [] }
  }, [stability])

  useEffect(() => {
    refreshRects()
    const onResize = () => {
      refreshRects()
      dwellRef.current = { target: '', progress: 0 }; dwellScoresRef.current.clear()
      cooldownUntilRef.current = performance.now() + 1000
      if (modeRef.current === 'collect') {
        phaseRef.current = 'settle'; phaseStartRef.current = performance.now()
        stabilityRef.current = []; bufferRef.current = []; setPhase('settle')
      }
    }
    const observer = new ResizeObserver(onResize)
    if (stageRef.current) observer.observe(stageRef.current)
    videoRef.current?.addEventListener('resize', onResize)
    window.addEventListener('resize', onResize)
    window.addEventListener('scroll', onResize, true)
    return () => {
      observer.disconnect()
      videoRef.current?.removeEventListener('resize', onResize)
      window.removeEventListener('resize', onResize)
      window.removeEventListener('scroll', onResize, true)
    }
  }, [refreshRects, boardId, stage, suggested, guideOpen])

  useEffect(() => {
    const onRotate = () => {
      if (!streamRef.current) return
      // A different screen orientation changes the head-to-screen mapping, not camera ownership.
      modelRef.current = null; neutralRef.current = null; neutralSamplesRef.current = []
      pointerOffsetRef.current = { x: 0, y: 0 }
      recenterAtRef.current = 0; setPositioning(false)
      modeRef.current = 'live'; setStage('skipped'); setQuality(null)
      dwellRef.current = { target: '', progress: 0 }; dwellScoresRef.current.clear()
      historyRef.current = { x: [], y: [] }; filterXRef.current.reset(); filterYRef.current.reset()
      setMessage('Screen rotated. Camera is still on; calibrate again for this orientation.')
      refreshRects()
    }
    const orientation = window.screen.orientation
    if (orientation?.addEventListener) orientation.addEventListener('change', onRotate)
    else window.addEventListener('orientationchange', onRotate)
    return () => {
      if (orientation?.removeEventListener) orientation.removeEventListener('change', onRotate)
      else window.removeEventListener('orientationchange', onRotate)
    }
  }, [refreshRects])

  useEffect(() => () => { releaseCamera(); cancelSuggestions() }, [releaseCamera, cancelSuggestions])

  const board = BOARDS[boardId] ?? BOARDS.home
  const collecting = stage === 'calibrating' || stage === 'checking'
  const boardReady = stage === 'complete' || stage === 'skipped'
  const points = stage === 'checking' ? CHECK_POINTS : CALIBRATION_POINTS
  const activePoint = points[pointIndex]
  // Graded against the board's own tolerance: 7.2% between neighbouring columns.
  const grade = !quality ? '' : quality.measured < 6 ? 'good' : quality.measured < 12 ? 'usable' : 'poor'
  // Below 720p the iris is too few pixels across for the geometry to survive, and no
  // amount of averaging recovers it. This is the single biggest lever the user controls.
  const lowResolution = cameraState === 'ready' && /^(\d+)/.test(resolution) && Number(resolution.split('×')[0]) < 1280
  // One line of advice rather than three panels of numbers. Order matters: too little
  // head movement is the cause of a weak signal, so reporting both would be saying the
  // same thing twice, and the movement version is the one the user can act on.
  const advice =
    !quality || collecting
      ? ''
      : movement && !movement.enough
        ? 'Your head barely moved. Calibrate again and turn to face each dot instead of glancing at it.'
        : signal?.verdict === 'dead'
          ? 'The board is barely tracking you. Calibrate again and turn your head, not just your eyes.'
          : lowResolution
            ? 'Your camera is below 720p, which limits how accurate this can get.'
            : grade === 'poor'
              ? 'Usable, but calibrating again with bigger head turns should help.'
              : ''
  const progress =
    stage === 'calibrating'
      ? (pointIndex / CALIBRATION_POINTS.length) * 0.75
      : stage === 'fitting'
        ? 0.78
        : stage === 'checking'
          ? 0.8 + (pointIndex / CHECK_POINTS.length) * 0.2
          : quality
            ? 1
            : 0

  // Suggestions replace the bottom row only. The top row - yes, no, help, pain, undo,
  // speak - never moves, because those are the tiles someone builds muscle memory for and
  // they are the ones you reach for when you cannot wait.
  const boardTiles: Tile[] = suggested
    ? [
        ...suggested.words.map((word) => ({ label: word.label, speech: word.speech, accent: 'suggested' })),
        RESTORE_TILE,
      ]
    : board.tiles

  const renderTile = (tile: Tile, key: string) => (
    <button
      key={key}
      data-dwell-target={tile.label}
      data-dwell-action={tile.restore ? 'restore' : tile.goTo ? 'folder' : 'speak'}
      data-dwell-board={tile.goTo ?? ''}
      data-dwell-speech={tile.speech ?? tile.label}
      className={`dwell-target tile-${tile.accent ?? 'word'}`}
      // Head pointing is the point, but a tile that cannot also be pressed is a tile
      // nobody can help you with. A carer leaning over the bed, a visitor being shown how
      // this works, or anyone whose camera has just dropped out gets the same board.
      onClick={(event) => commitSelection(event.currentTarget)}
    >
      {tile.label}
    </button>
  )

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CLEARSPEAK / HEAD POINTING</p>
          <h1>Head-pointing speech board</h1>
        </div>
        <div className="header-actions">
          <button className="guide-help" onClick={() => setGuideOpen(value => !value)} aria-expanded={guideOpen} disabled={collecting || stage === 'fitting'}>How to calibrate</button>
          <div className={`status-pill ${cameraState}`}>
            <span />
            {cameraState === 'ready' ? (cameraPaused ? 'camera paused' : 'tracking live') : cameraState === 'starting' ? 'starting' : 'camera off'}
          </div>
        </div>
      </header>

      <section className="workspace">
        <div ref={stageRef} className={`stage ${guideOpen && !collecting ? 'with-guide' : ''} ${collecting ? 'is-calibrating' : ''}`} aria-label="Speech board stage">
          <video ref={videoRef} className="camera-feed" muted playsInline aria-label="Webcam preview" />
          <canvas ref={overlayRef} className="detection-overlay" aria-hidden="true" />
          <div className="stage-shade" />

          {guideOpen && !collecting && stage !== 'fitting' && (
            <section className="calibration-guide" aria-label="How to calibrate">
              <div className="guide-heading"><span className="guide-kicker">A quick start</span><button className="guide-close" aria-label="Close calibration instructions" onClick={() => setGuideOpen(false)}>×</button></div>
              <h2>Point with your head.</h2>
              <p className="guide-intro">Sit comfortably with your face visible. You do not need to sit perfectly in the middle.</p>
              <svg className="guide-demo" viewBox="0 0 320 78" role="img" aria-label="Turn your head to move the white pointer toward the gold target">
                <defs><marker id="guide-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="currentColor" strokeWidth="1.5" /></marker></defs>
                <g className="demo-head"><ellipse cx="50" cy="32" rx="20" ry="25" /><path d="M42 26h1m14 0h1M49 30l4 6h-6m-4 8q8 5 14-1M27 73q2-20 23-20t24 20" /></g>
                <path className="demo-direction" d="M94 34H152" markerEnd="url(#guide-arrowhead)" />
                <circle className="demo-target" cx="242" cy="32" r="12" />
                <circle className="demo-pointer" cx="190" cy="32" r="7" />
                <text x="50" y="76" textAnchor="middle">your head</text><text x="238" y="76" textAnchor="middle">follow the gold dot</text>
              </svg>
              <ol className="guide-steps">
                <li><strong>Enable camera.</strong> Allow camera access when asked.</li>
                <li><strong>Start calibration.</strong> Turn your head toward each gold dot to guide the white pointer toward it. No perfect overlap needed yet.</li>
                <li><strong>Hold when it turns green.</strong> Keep still until the next dot appears. There are 16 dots, then 5 checks.</li>
              </ol>
              <div className="guide-next" aria-live="polite">
                <span className="guide-arrow" aria-hidden="true">↓</span>
                <span>{cameraState === 'ready' ? 'Camera ready. Start here.' : cameraState === 'starting' ? 'Allow the camera, then we’ll begin.' : 'First, turn on your camera.'}</span>
              </div>
              <button className="primary-button guide-start" onClick={cameraPaused ? resumeCamera : cameraState === 'ready' ? startCalibration : startCamera} disabled={cameraState === 'starting'}>
                {cameraPaused ? 'Resume camera' : cameraState === 'ready' ? 'Start calibration' : cameraState === 'starting' ? 'Starting camera…' : 'Enable camera'}
              </button>
              <button className="guide-skip" onClick={skipCalibration}>Use the word board without calibrating</button>
            </section>
          )}

          {collecting && activePoint && (
            <>
              <div
                key={`${stage}-${pointIndex}`}
                className={`cal-dot ${phase}`}
                style={{
                  left: `${activePoint.x * 100}%`,
                  top: `${activePoint.y * 100}%`,
                  ['--record' as string]: `${stage === 'checking' ? CHECK_RECORD_MS : RECORD_MS}ms`,
                }}
              >
                <i className="cal-fill" />
              </div>
              <span className="cal-target-arrow" aria-hidden="true" style={{ left: `${activePoint.x * 100}%`, top: `${activePoint.y * 100}%` }}>↓</span>
              <div className={`calibration-coach ${phase}`} role="status" aria-live="polite" aria-atomic="true">
                <span className="coach-count">{stage === 'checking' ? 'Check' : 'Dot'} {pointIndex + 1} of {points.length}</span>
                <strong>{cameraPaused ? 'Resume the camera to continue' : !capabilities.face ? 'Keep your face visible to the camera' : phase === 'record' ? 'Hold still — recording your position' : 'Turn your head toward the gold dot'}</strong>
                <span>{phase === 'record' ? 'Wait here until the dot moves.' : 'Move your head, not just your eyes. The white pointer is still learning.'}</span>
              </div>
              {stage === 'calibrating' && showDetails && (
                <div className={`turn-meter ${liveTurn.yaw > 10 && liveTurn.pitch > 6 ? 'ok' : 'low'}`}>
                  <strong>
                    turned {liveTurn.yaw.toFixed(0)}° across · {liveTurn.pitch.toFixed(0)}° down
                  </strong>
                  <span>
                    {liveTurn.yaw > 10 && liveTurn.pitch > 6
                      ? 'Good range — keep aiming your nose at each dot.'
                      : 'Turn your head further. Aim for 20° across, 12° down.'}
                  </span>
                </div>
              )}
            </>
          )}

          <div ref={cursorRef} className="cursor" aria-hidden="true" style={{ visibility: guideOpen ? 'hidden' : 'visible' }}>
            <span />
          </div>

          {boardReady && !guideOpen && (
            <>
              {/* Display only. Controls used to live here as small buttons a few pixels
                  apart, which quietly set the accuracy budget for the whole board to under
                  4% of the screen. They are full-size tiles in the grid now. */}
              <div className={`sentence-bar${spoken ? ' said' : ''}`}>
                <span>{spoken || 'What you say will appear here'}</span>
              </div>
              {/* Two rows of six, not three rows of four. Vertical gaze is roughly half
                  as accurate as horizontal, so rows are the expensive axis to add and
                  columns are the cheap one. Dropping to two rows takes the vertical
                  tolerance from 11.7% of the stage to 17.5% at no cost in vocabulary. */}
              <div className="board-grid">
                {TOP_ROW.map((tile) => renderTile(tile, `top-${tile.label}`))}
                {boardTiles.map((tile) => renderTile(tile, `${board.id}-${tile.label}`))}
              </div>
              <div className="rest-zone left" ref={restLeftRef}>
                REST ZONE
              </div>
              <div className="rest-zone right" ref={restRightRef}>
                REST ZONE
              </div>
            </>
          )}

          {!boardReady && !collecting && !guideOpen && (
            <button className="precalibration-note" onClick={skipCalibration}>Show word board</button>
          )}

          <div className="stage-label">
            <strong>
              {stage === 'calibrating'
                ? `CALIBRATING · DOT ${pointIndex + 1} OF ${CALIBRATION_POINTS.length}`
                : stage === 'fitting'
                  ? 'WORKING IT OUT'
                  : stage === 'checking'
                    ? `CHECKING · DOT ${pointIndex + 1} OF ${CHECK_POINTS.length}`
                    : board.title.toUpperCase()}
            </strong>
            <span role="status">{message}</span>
          </div>
        </div>

        <aside className="control-panel" aria-label="Controls">
          <nav className="control-tabs" aria-label="Control pages">
            {(['talk', 'tracking', 'details'] as const).map(tab => <button key={tab} aria-pressed={panelTab === tab} onClick={() => setPanelTab(tab)}>{tab === 'talk' ? 'Talk' : tab === 'tracking' ? 'Tracking' : 'Details'}</button>)}
          </nav>
          <div className="camera-controls panel-section">
            <div className="button-row">
              <button className="primary-button" onClick={cameraPaused ? resumeCamera : startCamera} disabled={cameraState === 'starting' || (cameraState === 'ready' && !cameraPaused)}>{cameraPaused ? 'Resume camera' : cameraState === 'starting' ? 'Starting…' : cameraState === 'ready' ? 'Camera on' : 'Enable camera'}</button>
              <button className="secondary-button stop-button" onClick={stopCamera} disabled={cameraState !== 'ready' && cameraState !== 'starting'}>Stop camera</button>
            </div>
            {cameraError && <p className="error-text" role="alert">{cameraError}</p>}
          </div>
          <div className={`control-page page-${panelTab}`}>


          <div className="panel-section setup-section">
            <div className="section-heading">
              <p className="section-kicker">your pointing position</p>
              <span>{collecting ? `${pointIndex + 1} / ${points.length}` : stage}</span>
            </div>
            <p className="panel-copy">
              Sit where you are comfortable; keep your face visible. Turn toward each dot, then pause. You do not need to be centred in the camera.
            </p>
            <div className="progress-track">
              <span style={{ width: `${progress * 100}%` }} />
            </div>
            <div className="button-row">
              <button className="secondary-button" onClick={startCalibration} disabled={collecting || cameraState !== 'ready' || cameraPaused}>
                {quality ? 'Calibrate again' : `Calibrate · ${CALIBRATION_POINTS.length} dots`}
              </button>
              <button className="text-button" onClick={skipCalibration}>
                Skip
              </button>
            </div>
            <button className="secondary-button reset-position" onClick={resetPosition} disabled={cameraState !== 'ready' || cameraPaused || collecting || positioning}>
              {positioning ? 'Look at the board centre…' : 'Reset pointing position'}
            </button>
            {quality && !collecting && panelTab === 'tracking' && (
              <div className={`result ${grade}`}>
                <strong>{grade === 'good' ? 'Good' : grade === 'usable' ? 'Usable' : 'Rough'}</strong>
                <span>lands within {quality.measured.toFixed(1)}% of the word you aim at</span>
              </div>
            )}
            {advice && panelTab === 'tracking' && <p className="panel-copy tight warn">{advice}</p>}
          </div>

          <div className="panel-section subtitle-panel talk-section">
            <div className="section-heading">
              <p className="section-kicker">what they said</p>
              <span className={listening ? 'ok' : ''}>{listening ? 'listening' : 'off'}</span>
            </div>
            <div className="subtitle-feed" aria-live="polite">
              {caption ? (
                <p className="caption-live interim">{caption}</p>
              ) : lastHeard ? (
                <p className="caption-live">{lastHeard}</p>
              ) : (
                <p className="caption-idle">
                  {listening ? 'Listening.' : 'Captions anyone talking near the computer, so you can read what you missed.'}
                </p>
              )}
            </div>
            <button
              className="secondary-button"
              onClick={() => setListening((current) => !current)}
              disabled={!recognitionAvailable()}
            >
              {listening ? 'Stop listening' : 'Start listening'}
            </button>
            {listenError ? (
              <p className="panel-copy tight warn">{listenError}</p>
            ) : !recognitionAvailable() ? (
              <p className="panel-copy tight warn">This browser cannot do captions. Chrome or Edge will work.</p>
            ) : null}
          </div>

          <div className="panel-section talk-section">
            <div className="section-heading">
              <p className="section-kicker">say something</p>
              <span className={speechAvailable() ? 'ok' : ''}>{speechAvailable() ? 'voice ready' : 'no voice'}</span>
            </div>
            <p className="panel-copy">
              Type a phrase to speak, or repeat the last thing said.
            </p>
            <input
              className="say-input" aria-label="Phrase to speak"
              value={typed}
              placeholder="Type anything to say out loud"
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') speakTyped()
              }}
            />
            <button
              className="secondary-button"
              onClick={speakTyped}
              disabled={!speechAvailable() || (!typed.trim() && !spoken)}
            >
              {typed.trim() ? 'Speak' : 'Say it again'}
            </button>
            {!speechAvailable() && (
              <p className="panel-copy tight warn">This browser has no speech voice. Chrome or Edge will work.</p>
            )}
          </div>

          <div className="panel-section talk-section">
            <div className="section-heading">
              <p className="section-kicker">suggested answers</p>
              <span className={suggested ? 'ok' : ''}>{suggested ? 'showing' : 'waiting'}</span>
            </div>
            <p className="panel-copy">
              Replies to questions and statements. Choose a tile to speak; nothing is said automatically.
            </p>
            <label className="toggle-row"><input type="checkbox" checked={aiEnabled} onChange={event => setAiEnabled(event.target.checked)} />Use OpenAI suggestions</label>
            <p className="panel-copy tight" role="status">{aiStatus}</p>
            <p className="privacy-note">When enabled, recent transcript text is sent to OpenAI for reply options.</p>
            {suggested && <p className="panel-copy tight">{suggested.because}</p>}
            {answerStat && panelTab === 'details' && (
              <div className="result good">
                <strong>
                  {answerStat.picks} {answerStat.picks === 1 ? 'pick' : 'picks'}
                </strong>
                <span>to answer, {answerStat.seconds.toFixed(1)} s after the question</span>
              </div>
            )}
          </div>

          <div className="panel-section tracking-section">
            <p className="section-kicker">shortcuts</p>
            <label className="toggle-row">
              <input type="checkbox" checked={gesturesOn} onChange={(event) => setGesturesOn(event.target.checked)} />
              Nod for yes, shake for no
            </label>
            <label className="toggle-row">
              <input type="checkbox" checked={blinkSelect} onChange={(event) => setBlinkSelect(event.target.checked)} />
              Long blink picks the word you are on
            </label>
          </div>

          <div className="panel-section tracking-section">
            <div className="section-heading">
              <p className="section-kicker">pointer steadiness</p>
              <span>{STABILITY_PRESETS[stability].label}</span>
            </div>
            <p className="panel-copy">Steadier is easier to aim with but slower to follow you.</p>
            <div className="segmented">
              {(Object.keys(STABILITY_PRESETS) as StabilityKey[]).map((key) => (
                <button key={key} className={key === stability ? 'active' : ''} onClick={() => setStability(key)}>
                  {STABILITY_PRESETS[key].label}
                </button>
              ))}
            </div>
          </div>



          {showDetails && cameraState === 'ready' && (
            <div className="panel-section">
              <p className="section-kicker">what the tracker sees</p>
              <canvas ref={eyePreviewRef} className="eye-preview" width={200} height={60} />
              <div className="signal-grid">
                <span className={capabilities.face ? 'ok' : 'warn'}>{capabilities.face ? '✓' : '✕'} face outline</span>
                <span className={capabilities.eyes ? 'ok' : 'warn'}>{capabilities.eyes ? '✓' : '✕'} eye crop</span>
                <span className={capabilities.eyeModel ? 'ok' : 'warn'}>{capabilities.eyeModel ? '✓' : '✕'} 3D eyeball</span>
                <span className={capabilities.headPose ? 'ok' : 'warn'}>{capabilities.headPose ? '✓' : '✕'} 3D head pose</span>
                <span className={capabilities.blendshapes ? 'ok' : 'warn'}>
                  {capabilities.blendshapes ? '✓' : '✕'} up/down estimate
                </span>
              </div>
            </div>
          )}

          {showDetails && movement && (
            <div className={`panel-section signal-report ${movement.enough ? 'good' : 'dead'}`}>
              <div className="section-heading">
                <p className="section-kicker">head movement</p>
                <span>{movement.enough ? 'enough' : 'too small'}</span>
              </div>
              <dl className="telemetry">
                <div>
                  <dt>turned across / down</dt>
                  <dd>{movement.yawDeg.toFixed(0)}° / {movement.pitchDeg.toFixed(0)}°</dd>
                </div>
              </dl>
              <p className="panel-copy tight">
                {movement.enough
                  ? 'Plenty of range to map the board onto.'
                  : `Your head barely moved between dots, so the drift your neck makes anyway is as large as the signal. Aim for about 20° across and 12° down — recalibrate and physically turn to face each dot instead of glancing at it.`}
              </p>
            </div>
          )}

          {showDetails && signal && (
            <div className={`panel-section signal-report ${signal.verdict}`}>
              <div className="section-heading">
                <p className="section-kicker">signal quality</p>
                <span>{signal.verdict}</span>
              </div>
              <dl className="telemetry">
                <div>
                  <dt>tracks target (x / y)</dt>
                  <dd>{signal.correlationX.toFixed(2)} / {signal.correlationY.toFixed(2)}</dd>
                </div>
                <div>
                  <dt>signal vs noise</dt>
                  <dd>{signal.separationX.toFixed(1)}× / {signal.separationY.toFixed(1)}×</dd>
                </div>
              </dl>
              <p className="panel-copy tight">
                {signal.verdict === 'good'
                  ? 'Tracking cleanly across the whole board.'
                  : signal.verdict === 'weak'
                    ? 'Tracking, but noisier than it should be. Turning your head further for each dot gives the model more to work with.'
                    : 'The cursor barely tracks the dots. The most common cause is moving only your eyes during calibration instead of turning your head. Recalibrate and aim your nose at each dot.'}
              </p>
            </div>
          )}

          {showDetails && (
          <div className="panel-section">
            <div className="section-heading">
              <p className="section-kicker">telemetry</p>
              <span className="live-dot" />
            </div>
            <dl className="telemetry">
              <div>
                <dt>mode</dt>
                <dd>{debug.tracking}</dd>
              </div>
              <div>
                <dt>holding still</dt>
                <dd className={grade}>{quality ? `${quality.measured.toFixed(1)}% · ${grade}` : 'uncalibrated'}</dd>
              </div>
              <div>
                <dt>across / down</dt>
                <dd>{quality ? `${quality.errorX.toFixed(1)}% / ${quality.errorY.toFixed(1)}%` : '—'}</dd>
              </div>
              <div>
                <dt>single frame</dt>
                <dd>{quality ? `${quality.perFrame.toFixed(1)}%` : '—'}</dd>
              </div>
              <div>
                <dt>camera</dt>
                <dd className={lowResolution ? 'poor' : ''}>{resolution || '—'}</dd>
              </div>
              <div>
                <dt>gaze angle</dt>
                <dd>{debug.gaze}</dd>
              </div>
              <div>
                <dt>distance</dt>
                <dd>{debug.distance ? `${debug.distance.toFixed(0)} cm` : '—'}</dd>
              </div>
              <div>
                <dt>training frames</dt>
                <dd>{quality ? quality.rows : '—'}</dd>
              </div>
              <div>
                <dt>nearest word</dt>
                <dd>{debug.target}</dd>
              </div>
              <div>
                <dt>things said</dt>
                <dd>{stats.selections}</dd>
              </div>
            </dl>
          </div>
          )}

          <div className="panel-section details-section">
            <p className="section-kicker">recent picks</p>
            {events.length === 0 ? (
              <p className="muted">Nothing said yet.</p>
            ) : (
              events.map((event) => (
                <div className="event-row" key={event.id}>
                  <span>{event.name}</span>
                  <time>{event.time}</time>
                </div>
              ))
            )}
          </div>
          </div>
        </aside>
      </section>

      <section className="conversation">
        <div className="conversation-label">
          <p className="section-kicker">conversation</p>
          <span className={`placeholder-tag${liveRef.current ? ' live' : ''}`}>
            {liveRef.current ? (listening ? 'live' : 'mic off') : 'example'}
          </span>
        </div>
        <div className="turns" ref={turnsRef}>
          {turns.map((turn, index) => (
            <div className={`turn ${turn.speaker}`} key={`${index}-${turn.time}-${turn.text}`} title={turn.text}>
              <span className="who">{turn.speaker === 'you' ? 'You' : 'Them'}</span>
              <p>{turn.text}</p>
            </div>
          ))}
        </div>
      </section>

      <footer>
        <span>dwell {DWELL_MS} ms · nearest-word targeting · head pointing</span>
        <span>head tracking stays on-device · captions use your browser’s speech service</span>
      </footer>
    </main>
  )
}

export default App
