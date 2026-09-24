/**
 * Verify features.js matches features.py exactly.
 *
 * Run after ANY change to either extractor. A silent drift here means the
 * deployed detector is not the detector reported in Chapter 4, and every
 * number in that chapter becomes a claim about something that is not what
 * ships. Exits non-zero on failure so it can gate a commit.
 */
const fs = require("fs");
const path = require("path");
const { featurise, DIM } = require("../extension/features.js");

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "parity_vectors.json")));
let worst = 0, worstIdx = -1, worstDim = -1, failures = 0;
const TOL = 1e-5;

for (let c = 0; c < cases.length; c++) {
  const got = featurise(cases[c].record);
  const want = cases[c].vector;
  if (got.length !== want.length || got.length !== DIM) {
    console.error(`case ${c}: length mismatch ${got.length} vs ${want.length}`);
    failures++; continue;
  }
  let caseWorst = 0, caseDim = -1;
  for (let i = 0; i < DIM; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > caseWorst) { caseWorst = d; caseDim = i; }
  }
  if (caseWorst > worst) { worst = caseWorst; worstIdx = c; worstDim = caseDim; }
  if (caseWorst > TOL) failures++;
}

console.log(`cases: ${cases.length}`);
console.log(`max abs difference: ${worst.toExponential(3)} (case ${worstIdx}, dim ${worstDim})`);
console.log(`cases exceeding ${TOL}: ${failures}`);
if (failures > 0) {
  const c = cases[worstIdx];
  console.error("\nworst-case record:", JSON.stringify(c.record).slice(0, 300));
  console.error(`python=${c.vector[worstDim]}  js=${featurise(c.record)[worstDim]}`);
  process.exit(1);
}
console.log("PARITY OK");
