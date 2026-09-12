import { projector } from './overlay'
import { relativePointer, recenterOffset } from './position'
const neutral = [0, 0, .12, .08, 0, 0, .3, .1, -.1, 0]
const shifted = [1.3, -.7, .12, .08, 0, 0, .3, .1, -.1, 0]
const a = relativePointer(shifted, neutral)
if (a.x !== 50 || a.y !== 45) throw new Error('Off-centre translation moved the pointer')
const turned = [...shifted]; turned[2] += .1
if (!(relativePointer(turned, neutral).x < 50)) throw new Error('Turning must still move pointer')
const raw = { x: 80, y: 12 }; const offset = recenterOffset(raw)
if (raw.x + offset.x !== 50 || raw.y + offset.y !== 45) throw new Error('Recentring failed')
console.log('Off-centre position, head turning, and calibrated recenter tests pass')

// A 4:3 camera on a 16:9 board must remain uncropped, including off-centre faces.
const projection = projector(1600, 900, 640, 480)
const leftEdge = projection.normalised({ x: 0, y: .5 })
const rightEdge = projection.normalised({ x: 1, y: .5 })
if (leftEdge.x !== 1400 || rightEdge.x !== 200 || leftEdge.y !== 450) throw new Error('Camera overlay does not match contained preview')
console.log('Letterboxed camera overlay remains aligned at both frame edges')
