// Appearance-based eye features: find the face, cut out each eye, warp it to a
// canonical pose, and hand the raw pixels to the regression.
//
// Following WebGazer (Papoutsaki et al., IJCAI 2016), each eye becomes a small
// grayscale histogram-equalised patch and both are concatenated into one vector.
// Following MPIIGaze (Zhang et al., PAMI 2019), the crop is warped rather than
// merely cropped: the eye region is close to planar, so rotating and rescaling it
// to a fixed reference cancels head roll and camera distance before the model ever
// sees it. Their ablation puts that normalisation at 16-33% of total error.

export const PATCH_WIDTH = 10
export const PATCH_HEIGHT = 6
export const APPEARANCE_LENGTH = PATCH_WIDTH * PATCH_HEIGHT * 2

// How much wider than the corner-to-corner span to take, so the crop keeps the
// lids and some sclera on both sides rather than clipping at the corners.
const REGION_SCALE = 1.5

export type EyeCorners = {
  outerX: number
  outerY: number
  innerX: number
  innerY: number
}

export type EyeBox = {
  centerX: number
  centerY: number
  width: number
  height: number
  angle: number
}

export function eyeBoxFrom(corners: EyeCorners, videoWidth: number, videoHeight: number): EyeBox {
  const outerX = corners.outerX * videoWidth
  const outerY = corners.outerY * videoHeight
  const innerX = corners.innerX * videoWidth
  const innerY = corners.innerY * videoHeight
  const span = Math.hypot(innerX - outerX, innerY - outerY)
  const width = span * REGION_SCALE
  return {
    centerX: (outerX + innerX) / 2,
    centerY: (outerY + innerY) / 2,
    width,
    // Uniform scale, so the patch aspect ratio decides the region aspect ratio.
    height: (width * PATCH_HEIGHT) / PATCH_WIDTH,
    angle: Math.atan2(innerY - outerY, innerX - outerX),
  }
}

// Flatten the CDF of the patch so a bright window or a dim room stops changing
// the pixel values the regression is fitted on.
function equalise(pixels: Uint8ClampedArray, start: number, count: number) {
  const histogram = new Uint32Array(256)
  for (let i = 0; i < count; i += 1) histogram[pixels[start + i]] += 1

  let cumulative = 0
  let minimum = 0
  const map = new Uint8Array(256)
  for (let value = 0; value < 256; value += 1) {
    cumulative += histogram[value]
    if (minimum === 0 && cumulative > 0) minimum = cumulative
    map[value] = cumulative > 0 ? Math.round(((cumulative - minimum) / Math.max(count - minimum, 1)) * 255) : 0
  }
  for (let i = 0; i < count; i += 1) pixels[start + i] = map[pixels[start + i]]
}

export class EyePatchExtractor {
  // Both eyes share one canvas so each frame costs a single GPU readback.
  private canvas = document.createElement('canvas')
  private context: CanvasRenderingContext2D | null
  private gray = new Uint8ClampedArray(APPEARANCE_LENGTH)

  constructor() {
    this.canvas.width = PATCH_WIDTH * 2
    this.canvas.height = PATCH_HEIGHT
    // Keeps the canvas CPU-backed; without it every getImageData stalls on a GPU sync.
    this.context = this.canvas.getContext('2d', { willReadFrequently: true })
    if (this.context) {
      this.context.imageSmoothingEnabled = true
      this.context.imageSmoothingQuality = 'high'
    }
  }

  private warp(source: CanvasImageSource, box: EyeBox, destinationX: number) {
    const context = this.context
    if (!context) return
    const scale = PATCH_WIDTH / box.width
    const cos = Math.cos(box.angle)
    const sin = Math.sin(box.angle)
    const originX = destinationX + PATCH_WIDTH / 2
    const originY = PATCH_HEIGHT / 2

    // Rotate by -angle about the eye centre, scale to patch size, land on the slot.
    context.setTransform(
      scale * cos,
      -scale * sin,
      scale * sin,
      scale * cos,
      originX - scale * (cos * box.centerX + sin * box.centerY),
      originY - scale * (cos * box.centerY - sin * box.centerX),
    )
    context.drawImage(source, 0, 0)
  }

  extract(source: CanvasImageSource, left: EyeBox, right: EyeBox) {
    const context = this.context
    if (!context) return null

    context.setTransform(1, 0, 0, 1, 0, 0)
    context.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.warp(source, left, 0)
    this.warp(source, right, PATCH_WIDTH)
    context.setTransform(1, 0, 0, 1, 0, 0)

    const image = context.getImageData(0, 0, this.canvas.width, this.canvas.height).data
    const perEye = PATCH_WIDTH * PATCH_HEIGHT

    // The canvas interleaves the two eyes row by row, so unpack column ranges into
    // two contiguous blocks before equalising each eye against its own lighting.
    for (let row = 0; row < PATCH_HEIGHT; row += 1) {
      for (let column = 0; column < PATCH_WIDTH * 2; column += 1) {
        const offset = (row * PATCH_WIDTH * 2 + column) * 4
        const luma = (image[offset] * 299 + image[offset + 1] * 587 + image[offset + 2] * 114) / 1000
        const eye = column < PATCH_WIDTH ? 0 : 1
        const localColumn = column - eye * PATCH_WIDTH
        this.gray[eye * perEye + row * PATCH_WIDTH + localColumn] = luma
      }
    }

    equalise(this.gray, 0, perEye)
    equalise(this.gray, perEye, perEye)

    const features = new Array<number>(APPEARANCE_LENGTH)
    for (let i = 0; i < APPEARANCE_LENGTH; i += 1) features[i] = this.gray[i] / 255
    return features
  }

  // Same warp at display size, for the on-screen "this is what the model sees" panel.
  drawPreview(target: CanvasRenderingContext2D, source: CanvasImageSource, left: EyeBox, right: EyeBox) {
    const width = target.canvas.width
    const height = target.canvas.height
    const halfWidth = width / 2

    const paint = (box: EyeBox, destinationX: number) => {
      const scale = halfWidth / box.width
      const cos = Math.cos(box.angle)
      const sin = Math.sin(box.angle)
      const originX = destinationX + halfWidth / 2
      const originY = height / 2
      target.setTransform(
        scale * cos,
        -scale * sin,
        scale * sin,
        scale * cos,
        originX - scale * (cos * box.centerX + sin * box.centerY),
        originY - scale * (cos * box.centerY - sin * box.centerX),
      )
      target.drawImage(source, 0, 0)
    }

    target.setTransform(1, 0, 0, 1, 0, 0)
    target.clearRect(0, 0, width, height)
    paint(left, 0)
    paint(right, halfWidth)
    target.setTransform(1, 0, 0, 1, 0, 0)
  }
}
