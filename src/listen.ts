// Room microphone -> live captions, via the browser's built-in speech recogniser.
//
// Two things make this different from the usual "click, say one thing, get a string"
// example. The mic is pointed at a room rather than at a person taking a turn, so it has
// to stay open indefinitely; and Chrome does not actually honour that, so the session has
// to be rebuilt every time it quietly ends.
//
// Note for the offline claim: Chrome implements this by streaming audio to Google's
// servers. It is not on-device.

// Not in lib.dom, and outside Chrome only the webkit-prefixed constructor exists.
type RecognitionAlternative = { transcript: string; confidence: number }
type RecognitionResult = { readonly length: number; isFinal: boolean; [index: number]: RecognitionAlternative }
type RecognitionResultList = { readonly length: number; [index: number]: RecognitionResult }
type RecognitionEvent = { resultIndex: number; results: RecognitionResultList }
type RecognitionErrorEvent = { error: string; message: string }

type Recognition = {
  continuous: boolean
  interimResults: boolean
  lang: string
  maxAlternatives: number
  start(): void
  abort(): void
  onstart: (() => void) | null
  onresult: ((event: RecognitionEvent) => void) | null
  onerror: ((event: RecognitionErrorEvent) => void) | null
  onend: (() => void) | null
}

type RecognitionConstructor = new () => Recognition

function constructorFor(): RecognitionConstructor | null {
  if (typeof window === 'undefined') return null
  const scope = window as unknown as {
    SpeechRecognition?: RecognitionConstructor
    webkitSpeechRecognition?: RecognitionConstructor
  }
  return scope.SpeechRecognition ?? scope.webkitSpeechRecognition ?? null
}

export function recognitionAvailable() {
  return constructorFor() !== null
}

export type ListenerHandlers = {
  onInterim: (text: string) => void
  onFinal: (text: string) => void
  onError: (message: string) => void
}

export type Listener = { start(): void; stop(): void }

const RESTART_MS = 250
const MAX_BACKOFF_MS = 4000

export function createListener(handlers: ListenerHandlers): Listener | null {
  const Recognition = constructorFor()
  if (!Recognition) return null

  let active: Recognition | null = null
  let wanted = false
  let backoff = RESTART_MS
  let timer: number | undefined

  const build = () => {
    const recognition = new Recognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'
    recognition.maxAlternatives = 1

    // Reaching onstart means the device is healthy, so forget any accumulated backoff.
    recognition.onstart = () => {
      backoff = RESTART_MS
    }

    recognition.onresult = (event) => {
      let interim = ''
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i]
        const text = result[0]?.transcript.trim() ?? ''
        if (!text) continue
        if (result.isFinal) handlers.onFinal(text)
        else interim += `${interim ? ' ' : ''}${text}`
      }
      handlers.onInterim(interim)
    }

    recognition.onerror = (event) => {
      // A mic left open on a quiet room produces these constantly; they are not faults.
      if (event.error === 'no-speech' || event.error === 'aborted') return
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        wanted = false
        handlers.onError('Microphone blocked. Allow access and try again.')
        return
      }
      handlers.onError(event.error === 'network' ? 'Speech service unreachable.' : event.error)
    }

    recognition.onend = () => {
      if (!wanted) return
      timer = window.setTimeout(() => {
        if (!wanted) return
        try {
          recognition.start()
        } catch {
          // start() throws if the session is already coming back up; onend will retry.
        }
      }, backoff)
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS)
    }

    return recognition
  }

  return {
    start() {
      if (wanted) return
      wanted = true
      active = build()
      try {
        active.start()
      } catch {
        // Same as above: a start() racing a previous session is not fatal.
      }
    },
    stop() {
      wanted = false
      window.clearTimeout(timer)
      active?.abort()
      active = null
    },
  }
}
