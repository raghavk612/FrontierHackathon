// Ridge-regularised least squares through the normal equations.
//
// Expects columns that are already standardised and targets that are already
// centred, so the penalty lands evenly and no intercept is shrunk.
//
// XtX does not depend on lambda, and both screen axes share it, so it is built once
// and then solved repeatedly with different diagonal loadings. Rebuilding it per
// lambda per fold is what makes a cross-validated fit too slow to run inline.

export type NormalEquations = {
  width: number
  xtx: Float64Array
  rhs: Float64Array
}

export function buildNormalEquations(rows: number[][], targetsA: number[], targetsB: number[]): NormalEquations {
  const width = rows[0]?.length ?? 0
  const xtx = new Float64Array(width * width)
  const rhs = new Float64Array(width * 2)

  for (let k = 0; k < rows.length; k += 1) {
    const row = rows[k]
    const a = targetsA[k]
    const b = targetsB[k]
    for (let i = 0; i < width; i += 1) {
      const value = row[i]
      if (value === 0) continue
      rhs[i] += value * a
      rhs[width + i] += value * b
      const base = i * width
      // Only the upper triangle; XtX is symmetric.
      for (let j = i; j < width; j += 1) xtx[base + j] += value * row[j]
    }
  }

  for (let i = 0; i < width; i += 1) {
    for (let j = 0; j < i; j += 1) xtx[i * width + j] = xtx[j * width + i]
  }

  return { width, xtx, rhs }
}

// Solves both screen axes in one elimination pass by carrying two right-hand sides.
export function solveNormalEquations(equations: NormalEquations, lambda: number) {
  const width = equations.width
  if (width === 0) return null
  const stride = width + 2
  const matrix = new Float64Array(width * stride)

  for (let i = 0; i < width; i += 1) {
    const source = i * width
    const target = i * stride
    for (let j = 0; j < width; j += 1) matrix[target + j] = equations.xtx[source + j]
    matrix[target + i] += lambda
    matrix[target + width] = equations.rhs[i]
    matrix[target + width + 1] = equations.rhs[width + i]
  }

  for (let column = 0; column < width; column += 1) {
    let pivot = column
    let best = Math.abs(matrix[column * stride + column])
    for (let row = column + 1; row < width; row += 1) {
      const candidate = Math.abs(matrix[row * stride + column])
      if (candidate > best) {
        best = candidate
        pivot = row
      }
    }
    if (best < 1e-12) return null

    if (pivot !== column) {
      const a = pivot * stride
      const b = column * stride
      for (let j = column; j < stride; j += 1) {
        const swap = matrix[a + j]
        matrix[a + j] = matrix[b + j]
        matrix[b + j] = swap
      }
    }

    const diagonalRow = column * stride
    const diagonal = matrix[diagonalRow + column]
    for (let row = column + 1; row < width; row += 1) {
      const target = row * stride
      const factor = matrix[target + column] / diagonal
      if (factor === 0) continue
      for (let j = column; j < stride; j += 1) matrix[target + j] -= factor * matrix[diagonalRow + j]
    }
  }

  const a = new Array<number>(width).fill(0)
  const b = new Array<number>(width).fill(0)
  for (let row = width - 1; row >= 0; row -= 1) {
    const offset = row * stride
    let sumA = matrix[offset + width]
    let sumB = matrix[offset + width + 1]
    for (let column = row + 1; column < width; column += 1) {
      sumA -= matrix[offset + column] * a[column]
      sumB -= matrix[offset + column] * b[column]
    }
    a[row] = sumA / matrix[offset + row]
    b[row] = sumB / matrix[offset + row]
  }

  return Number.isFinite(a[0]) && Number.isFinite(b[0]) ? { a, b } : null
}
