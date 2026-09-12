// One Euro filter (Casiez, Roussel & Vogel, CHI 2012).
// A fixed smoothing factor has to choose between jitter when the eyes are still
// and lag when they move. This widens the cutoff as speed rises, so it can do both.

function smoothingFactor(cutoffHz: number, dtSeconds: number) {
  const tau = 1 / (2 * Math.PI * cutoffHz)
  return 1 / (1 + tau / dtSeconds)
}

export class OneEuroFilter {
  private minCutoff: number
  private beta: number
  private derivativeCutoff: number
  private lastValue = 0
  private lastDerivative = 0
  private lastTimeMs = 0
  private initialised = false

  constructor(minCutoff = 1.1, beta = 0.02, derivativeCutoff = 1) {
    this.minCutoff = minCutoff
    this.beta = beta
    this.derivativeCutoff = derivativeCutoff
  }

  reset() {
    this.initialised = false
    this.lastDerivative = 0
  }

  filter(value: number, timeMs: number) {
    if (!this.initialised) {
      this.initialised = true
      this.lastValue = value
      this.lastTimeMs = timeMs
      return value
    }

    // Clamp dt so a dropped frame or a backgrounded tab cannot spike the derivative.
    const dt = Math.min(Math.max((timeMs - this.lastTimeMs) / 1000, 1 / 240), 0.1)
    const rawDerivative = (value - this.lastValue) / dt
    const derivativeAlpha = smoothingFactor(this.derivativeCutoff, dt)
    const derivative = this.lastDerivative + derivativeAlpha * (rawDerivative - this.lastDerivative)

    const cutoff = this.minCutoff + this.beta * Math.abs(derivative)
    const alpha = smoothingFactor(cutoff, dt)
    const filtered = this.lastValue + alpha * (value - this.lastValue)

    this.lastValue = filtered
    this.lastDerivative = derivative
    this.lastTimeMs = timeMs
    return filtered
  }
}
