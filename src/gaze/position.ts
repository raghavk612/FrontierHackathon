// Camera-space translation should not force a person into the centre of the image.
// Use nose position within the face, relative to their own resting pose.
export function relativePointer(head: number[], neutral: number[]) {
  return { x: 50 - (head[2] - neutral[2]) * 180, y: 45 + (head[3] - neutral[3]) * 220 }
}
export function recenterOffset(raw: { x: number; y: number }) {
  return { x: 50 - raw.x, y: 45 - raw.y }
}
