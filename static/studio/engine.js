/** Original Dense / LeakyReLU decoder and a bounded inverse search. */
export const deltaE = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));

export function validateTarget(lab, fabric, metadata) {
  if (!Array.isArray(lab) || lab.length !== 3 || !lab.every(Number.isFinite)
      || lab[0] < 0 || lab[0] > 100 || Math.abs(lab[1]) > 128 || Math.abs(lab[2]) > 128)
    throw new Error('INVALID_LAB');
  if (!Number.isInteger(fabric) || !metadata.fabrics[fabric]) throw new Error('INVALID_FABRIC');
}

export class Decoder {
  constructor(metadata, buffer) {
    this.meta = metadata;
    this.weights = new Float32Array(buffer);
    if (this.weights.length !== metadata.parameters) throw new Error('MODEL_SIZE_MISMATCH');
  }
  vector(dyes, concentrations, fabric) {
    if (!Number.isInteger(fabric) || !this.meta.fabrics[fabric]) throw new Error('INVALID_FABRIC');
    if (!Array.isArray(dyes) || !Array.isArray(concentrations) || !dyes.length || dyes.length !== concentrations.length
        || new Set(dyes).size !== dyes.length) throw new Error('INVALID_RECIPE');
    const x = new Float64Array(this.meta.columns.length);
    dyes.forEach((d, i) => {
      if (!Number.isInteger(d) || d < 0 || d >= this.meta.dyes.length || !Number.isFinite(concentrations[i])
          || concentrations[i] < 0 || concentrations[i] > 5) throw new Error('INVALID_RECIPE');
      x[d] = concentrations[i];
    });
    x[this.meta.dyes.length + fabric] = 1;
    return x;
  }
  forward(x, trace = false) {
    let a = x;
    const activations = [a];
    for (const layer of this.meta.layers) {
      const {inputs: ni, outputs: no, offset: p, alpha} = layer;
      const z = Float64Array.from(this.weights.subarray(p + ni * no, p + ni * no + no));
      for (let i = 0; i < ni; i++) {
        const v = a[i], start = p + i * no;
        if (!v) continue;
        for (let j = 0; j < no; j++) z[j] += v * this.weights[start + j];
      }
      if (alpha !== null) for (let j = 0; j < no; j++) if (z[j] < 0) z[j] *= alpha;
      a = z;
      if (trace) activations.push(a);
    }
    const lab = Array.from(a, (v, i) => v * this.meta.std[i] + this.meta.mean[i]);
    return trace ? {lab, activations} : lab;
  }
  predict(dyes, concentrations, fabric) { return this.forward(this.vector(dyes, concentrations, fabric)); }
  jacobian(activations, dyes) {
    return [0, 1, 2].map(axis => {
      let g = new Float64Array(3); g[axis] = this.meta.std[axis];
      for (let k = this.meta.layers.length - 1; k >= 0; k--) {
        const {inputs: ni, outputs: no, offset: p, alpha} = this.meta.layers[k];
        if (alpha !== null) for (let j = 0; j < no; j++) if (activations[k + 1][j] < 0) g[j] *= alpha;
        const previous = new Float64Array(ni);
        for (let i = 0; i < ni; i++) for (let j = 0; j < no; j++) previous[i] += this.weights[p + i * no + j] * g[j];
        g = previous;
      }
      return dyes.map(d => g[d]);
    });
  }
}

function solveLinear(matrix, rhs) {
  const n = rhs.length, a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(a[j][i]) > Math.abs(a[pivot][i])) pivot = j;
    [a[i], a[pivot]] = [a[pivot], a[i]];
    if (Math.abs(a[i][i]) < 1e-12) return Array(n).fill(0);
    const d = a[i][i]; for (let j = i; j <= n; j++) a[i][j] /= d;
    for (let k = 0; k < n; k++) if (k !== i) {
      const f = a[k][i]; for (let j = i; j <= n; j++) a[k][j] -= f * a[i][j];
    }
  }
  return a.map(row => row[n]);
}

export function refine(decoder, target, fabric, dyes, initial, maxIterations = 45) {
  let concentrations = initial.map(v => Math.max(0, Math.min(5, v)));
  let state = decoder.forward(decoder.vector(dyes, concentrations, fabric), true);
  let error = deltaE(state.lab, target), damping = 1, trace = [error];
  for (let step = 0; step < maxIterations && error > 0.005; step++) {
    const jac = decoder.jacobian(state.activations, dyes), n = dyes.length;
    const residual = target.map((v, i) => v - state.lab[i]);
    const matrix = dyes.map((_, i) => dyes.map((__, j) => jac.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? damping : 0)));
    const rhs = dyes.map((_, i) => jac.reduce((s, row, k) => s + row[i] * residual[k], 0));
    const update = solveLinear(matrix, rhs);
    const proposal = concentrations.map((v, i) => Math.max(0, Math.min(5, v + Math.max(-0.5, Math.min(0.5, update[i])))));
    const next = decoder.forward(decoder.vector(dyes, proposal, fabric), true);
    const nextError = deltaE(next.lab, target);
    if (nextError < error) {
      concentrations = proposal; state = next; error = nextError; damping = Math.max(0.0001, damping * 0.4);
    } else damping = Math.min(1e8, damping * 6);
    trace.push(error);
    if (damping >= 1e8) break;
  }
  return {dyes, concentrations, lab: state.lab, delta: error, trace};
}

export function search(decoder, history, target, fabric, onProgress = () => {}) {
  validateTarget(target, fabric, decoder.meta);
  const nearest = history.filter(r => r.fabric === fabric && r.dyes.length >= 2)
    .map(r => ({...r, distance: deltaE(r.lab, target)})).sort((a, b) => a.distance - b.distance);
  const unique = new Map();
  for (const row of nearest) {
    const key = row.dyes.join(',');
    if (!unique.has(key)) unique.set(key, row);
    if (unique.size === 8) break;
  }
  if (!unique.size) throw new Error('NO_CANDIDATES');
  const results = [];
  for (const row of unique.values()) {
    const seeds = [row.concentrations, row.concentrations.map(v => Math.max(0.1, v * 0.55))];
    const candidates = seeds.map(seed => refine(decoder, target, fabric, row.dyes, seed));
    const best = candidates.sort((a, b) => a.delta - b.delta)[0];
    results.push({...best, sourceDistance: row.distance});
    onProgress(results.length, unique.size);
  }
  return results.sort((a, b) => a.delta - b.delta);
}

/** CIELAB (D65) to a clipped sRGB screen preview, not physical pigment mixing. */
export function labToRgb(lab) {
  const f = (v) => v ** 3 > 216 / 24389 ? v ** 3 : (116 * v - 16) / (24389 / 27);
  const fy = (lab[0] + 16) / 116;
  const x = 0.95047 * f(fy + lab[1] / 500), y = f(fy), z = 1.08883 * f(fy - lab[2] / 200);
  const linear = [3.2404542*x - 1.5371385*y - 0.4985314*z, -0.969266*x + 1.8760108*y + 0.041556*z, 0.0556434*x - 0.2040259*y + 1.0572252*z];
  return linear.map(v => Math.round(255 * Math.max(0, Math.min(1, v <= 0.0031308 ? 12.92*v : 1.055*v**(1/2.4)-0.055))));
}
export const rgbCss = lab => `rgb(${labToRgb(lab).join(',')})`;
