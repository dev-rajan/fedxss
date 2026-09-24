# fedxss browser extension (research prototype)

Inference-only client for the federated XSS detector. Loads the model
exported by `src/export_model.py` and scores navigation, inline-script and
DOM-mutation events locally.

## What it does and does not do

- Scores observations **entirely on-device**. There is no network code in
  this extension; you can verify that by reading `background.js`.
- Does **not** perform local training or send updates to a server. The
  federated training results in Chapter 4 come from simulation; the
  in-browser training client is future work (Section 5.4).
- Does **not** block or modify pages. It observes and reports.

Be explicit about the second point in Chapter 4. The extension evidences
the *inference* half of Objective 2 — that detection is feasible in-browser
at interactive speed — and not the training half.

## Build and install

```bash
cd ../src && python export_model.py      # writes extension/model.json
```

Then in Chrome: `chrome://extensions` → enable Developer mode → **Load
unpacked** → select this `extension/` directory. Browse normally; the
toolbar badge shows flagged events for the active tab, and the popup shows
session statistics.

## Measuring Table 4.3

Restart Chrome with precise memory reporting, or the heap figures are
rounded to the point of being meaningless:

```bash
# macOS
open -a "Google Chrome" --args --enable-precise-memory-info
```

Then open `bench.html` (File → Open) and press **Run benchmark**. Record
the browser version and machine specification alongside the numbers — a
latency figure without the hardware it was measured on is not reportable.

## Files

| File | Purpose |
|---|---|
| `manifest.json` | MV3 manifest |
| `features.js` | feature extractor, byte-parity with `src/features.py` |
| `model.js` | forward pass (no ONNX, no WASM) |
| `model.json` | exported weights + architecture + provenance (generated) |
| `content.js` | collects observation records from the page |
| `background.js` | owns the model, scores records |
| `popup.html/js` | session statistics and recent detections |
| `bench.html` | measurement harness for Table 4.3 |

## Parity

`features.js` must match `src/features.py` exactly. After changing either:

```bash
cd .. && python tools/dump_vectors.py && node tools/parity_check.js
```

Verified status: 300 test records, maximum absolute difference 2.98e-8
(float32 rounding only).
