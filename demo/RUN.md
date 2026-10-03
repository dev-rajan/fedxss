# Running the demonstration

```bash
cd ~/Desktop/Thesis/fed-xss/demo
python3 -m http.server 8080
```

Open **http://localhost:8080/index.html**

Load the extension first: `chrome://extensions` → Developer mode →
Load unpacked → select `../extension`. Do this before the presentation
begins, not during it.

## Sequence (about 60 seconds)

| Step | Action                | What to say                                                             |
| ---- | --------------------- | ----------------------------------------------------------------------- |
| 1    | Benign catalogue page | "Ordinary markup — no badge. The detector stays silent."                |
| 2    | Reflected XSS page    | "Payload in the URL, written straight to innerHTML. Badge appears."     |
| 3    | Open the popup        | "Probability, the snippet, and mean scoring time per record."           |
| 4    | Show `background.js`  | "No network code anywhere in the extension. Nothing leaves the device." |

Say unprompted, before anyone asks: **"This is inference only. Federated
training is simulated; in-browser training remains future work."**

## IMPORTANT — what NOT to demonstrate

**Do not visit a benign page containing a hand-written inline script.**

Testing the exported model against ordinary page scripts returned:

| Script pattern                                  | Score | Outcome |
| ----------------------------------------------- | ----- | ------- |
| `createElement` + `textContent` + `appendChild` | 74.9% | flagged |
| `DOMContentLoaded` wrapper calling a function   | 89.4% | flagged |
| An `escapeHtml` helper function                 | 52.8% | flagged |
| `URLSearchParams` → `textContent`               | 87.6% | flagged |

The benign demonstration page therefore contains markup only. Its records
score 2.4%–44.5% and none are flagged.

This is a genuine limitation, not a presentational convenience. The benign
training class is drawn from library source, published page templates and
encyclopaedia markup; hand-written page scripts are absent from it, so the
reported false-positive rate of 0.0003 reflects benign content of similar
provenance rather than arbitrary page scripts.

**Record this in the thesis (Section 4.9 or 5.3) before submission.** If an
examiner loads the extension and visits any ordinary site, they will see it.

## Before the day

- Record the sequence as a video and put it on a slide as a fallback.
- Re-test on the exact machine and browser you will present with.
- Confirm the extension is still loaded; a browser update can unload it.
- Check nothing else is using port 8080 (`lsof -ti:8080`).
