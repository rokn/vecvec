// Deterministic 2D PCA via power iteration.
//
// We never materialize the dim×dim covariance (dim can be 768+). Instead we use the
// identity  cov·w = (1/n)·Xᵀ(X·w)  — each iteration is O(n·dim), and we extract the
// top two eigenvectors with deflation (Gram–Schmidt against the first).

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function norm(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

function orthogonalize(v: Float64Array, against: Float64Array): void {
  let dot = 0;
  for (let i = 0; i < v.length; i++) dot += v[i] * against[i];
  for (let i = 0; i < v.length; i++) v[i] -= dot * against[i];
}

/** One leading eigenvector of the centered data's covariance, optionally
 *  orthogonal to `against`. `X` is row-major centered data (n×dim). */
function topEigenvector(
  X: Float64Array,
  n: number,
  dim: number,
  rng: () => number,
  against?: Float64Array,
): Float64Array {
  let w = new Float64Array(dim);
  for (let j = 0; j < dim; j++) w[j] = rng() - 0.5;
  if (against) orthogonalize(w, against);
  let wn = norm(w) || 1;
  for (let j = 0; j < dim; j++) w[j] /= wn;

  const u = new Float64Array(n);
  for (let iter = 0; iter < 96; iter++) {
    // u = X · w
    for (let i = 0; i < n; i++) {
      let acc = 0;
      const base = i * dim;
      for (let j = 0; j < dim; j++) acc += X[base + j] * w[j];
      u[i] = acc;
    }
    // z = Xᵀ · u  (reuse w as accumulator target)
    const z = new Float64Array(dim);
    for (let i = 0; i < n; i++) {
      const base = i * dim;
      const ui = u[i];
      for (let j = 0; j < dim; j++) z[j] += X[base + j] * ui;
    }
    if (against) orthogonalize(z, against);
    const zn = norm(z);
    if (zn < 1e-12) break;
    let delta = 0;
    for (let j = 0; j < dim; j++) {
      const nv = z[j] / zn;
      delta += Math.abs(nv - w[j]);
      w[j] = nv;
    }
    if (delta < 1e-6) break;
  }
  return w;
}

export function pca2d(vectors: number[][]): [number, number][] {
  const n = vectors.length;
  if (n === 0) return [];
  const dim = vectors[0].length;
  if (dim === 1) return vectors.map((v) => [v[0], 0]);

  // Mean-center into a flat row-major buffer.
  const mean = new Float64Array(dim);
  for (const v of vectors) for (let j = 0; j < dim; j++) mean[j] += v[j];
  for (let j = 0; j < dim; j++) mean[j] /= n;

  const X = new Float64Array(n * dim);
  for (let i = 0; i < n; i++) {
    const v = vectors[i];
    const base = i * dim;
    for (let j = 0; j < dim; j++) X[base + j] = v[j] - mean[j];
  }

  const rng = lcg(0x5eed_1234);
  const w1 = topEigenvector(X, n, dim, rng, undefined);
  const w2 = topEigenvector(X, n, dim, rng, w1);

  const out: [number, number][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    let a = 0;
    let b = 0;
    for (let j = 0; j < dim; j++) {
      a += X[base + j] * w1[j];
      b += X[base + j] * w2[j];
    }
    out[i] = [a, b];
  }
  return out;
}
