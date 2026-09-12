import { openFrontCamera, watchCamera } from './camera'

function check(value: unknown, message: string) { if (!value) throw new Error(message) }
let attempts = 0
const fakeStream = {} as MediaStream
await openFrontCamera(async constraints => {
  attempts++
  check(constraints.audio === false, 'Camera should not request the microphone')
  if (attempts === 1) throw Object.assign(new Error('Unsupported dimensions'), { name: 'OverconstrainedError' })
  return fakeStream
}, true)
check(attempts === 2, 'Unsupported constraints must get one fallback')
attempts = 0
try {
  await openFrontCamera(async () => { attempts++; throw Object.assign(new Error('Denied'), { name: 'NotAllowedError' }) }, false)
} catch { /* Expected denial. */ }
check(attempts === 1, 'Permission denial must not cause repeated permission prompts')

const track = Object.assign(new EventTarget(), { readyState: 'live', muted: false })
const page = Object.assign(new EventTarget(), { hidden: false })
const events: string[] = []
const cleanup = watchCamera(track as unknown as MediaStreamTrack, page as unknown as Document, {
  pause: () => events.push('pause'), resume: () => events.push('resume'), ended: () => events.push('ended'),
})
page.hidden = true; page.dispatchEvent(new Event('visibilitychange'))
page.hidden = false; page.dispatchEvent(new Event('visibilitychange'))
track.muted = true; track.dispatchEvent(new Event('mute'))
track.muted = false; track.dispatchEvent(new Event('unmute'))
track.readyState = 'ended'; track.dispatchEvent(new Event('ended'))
check(events.join(',') === 'pause,resume,pause,resume,ended', 'Camera interruption lifecycle is incorrect')
cleanup()
page.dispatchEvent(new Event('visibilitychange')); track.dispatchEvent(new Event('ended'))
check(events.length === 5, 'Disposed camera listeners must not restart a stopped session')
console.log('Mobile camera fallback, denial, background/resume, mute/end and cleanup tests pass')
