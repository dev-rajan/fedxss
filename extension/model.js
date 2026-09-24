/**
 * model.js — forward pass for the exported detector.
 *
 * Two matrix products and two activations. No ONNX Runtime, no WebAssembly,
 * no external dependency: shipping a multi-megabyte WASM runtime to evaluate
 * a 320x64x1 network would contradict the lightweight-client argument in
 * Section 3.2.
 *
 * Weights are kept as Float32Array so the arithmetic matches the Python and
 * the parity check stays meaningful.
 */

class Detector {
  constructor(payload) {
    const a = payload.architecture;
    this.dim = a.dim;
    this.hidden = a.hidden;
    this.threshold = payload.threshold ?? 0.5;
    this.provenance = payload.provenance || {};

    const w = payload.weights;
    this.W1 = Float32Array.from(w.W1);   // dim x hidden, row-major
    this.b1 = Float32Array.from(w.b1);
    this.W2 = Float32Array.from(w.W2);   // hidden x 1
    this.b2 = Float32Array.from(w.b2);

    if (this.W1.length !== this.dim * this.hidden) {
      throw new Error(`W1 size ${this.W1.length} != ${this.dim}x${this.hidden}`);
    }
    this._h = new Float32Array(this.hidden);   // reused across calls
  }

  /** Returns P(XSS) for one feature vector. */
  score(x) {
    const { dim, hidden, W1, b1, W2, b2 } = this;
    const h = this._h;
    for (let j = 0; j < hidden; j++) h[j] = b1[j];
    // skip zero inputs: the hashed-n-gram block is sparse in practice
    for (let i = 0; i < dim; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const base = i * hidden;
      for (let j = 0; j < hidden; j++) h[j] += xi * W1[base + j];
    }
    let z = b2[0];
    for (let j = 0; j < hidden; j++) {
      if (h[j] > 0) z += h[j] * W2[j];   // ReLU folded into the sum
    }
    if (z > 30) z = 30; else if (z < -30) z = -30;
    return 1 / (1 + Math.exp(-z));
  }

  predict(x) {
    const p = this.score(x);
    return { probability: p, flagged: p >= this.threshold };
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { Detector };
}
