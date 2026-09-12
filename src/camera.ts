// Keep device handling independent from React so interruptions can be tested without a camera.
export async function openFrontCamera(
  acquire: (constraints: MediaStreamConstraints) => Promise<MediaStream>,
  compact: boolean,
) {
  try {
    return await acquire({ audio: false, video: {
      facingMode: { ideal: 'user' },
      width: { ideal: compact ? 960 : 1280 }, height: { ideal: compact ? 720 : 720 },
      frameRate: { ideal: 24, max: 30 },
    } })
  } catch (error) {
    // Retry an unsupported camera configuration, never repeat a denied permission request.
    if ((error as Error).name !== 'OverconstrainedError') throw error
    return acquire({ audio: false, video: { facingMode: { ideal: 'user' } } })
  }
}

export function watchCamera(
  track: MediaStreamTrack,
  page: Pick<Document, 'hidden' | 'addEventListener' | 'removeEventListener'>,
  callbacks: { pause: () => void; resume: () => void; ended: () => void },
) {
  const update = () => {
    if (track.readyState === 'ended') callbacks.ended()
    else if (page.hidden || track.muted) callbacks.pause()
    else callbacks.resume()
  }
  const end = () => callbacks.ended()
  track.addEventListener('mute', update)
  track.addEventListener('unmute', update)
  track.addEventListener('ended', end)
  page.addEventListener('visibilitychange', update)
  return () => {
    track.removeEventListener('mute', update)
    track.removeEventListener('unmute', update)
    track.removeEventListener('ended', end)
    page.removeEventListener('visibilitychange', update)
  }
}
